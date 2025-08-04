#!/usr/bin/env python3
"""
Context-Switcher MCP Server
Multi-perspective analysis using thread orchestration
"""

import logging
import secrets
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .aorp import (
    AORPBuilder,
    calculate_analysis_confidence,
    generate_analysis_next_steps,
    generate_synthesis_next_steps,
    create_error_response,
)
from .confidence_metrics import (
    ConfidenceCalibrator,
    analyze_synthesis_quality,
)
from .compression import prepare_synthesis_input
from .models import ModelBackend, Thread
from .orchestrator import ThreadOrchestrator, NO_RESPONSE
from .perspective_selector import SmartPerspectiveSelector
from .rate_limiter import SessionRateLimiter
from .security import (
    log_security_event,
    sanitize_error_message,
    validate_user_content,
    validate_perspective_data,
    validate_analysis_prompt,
)
from .session_manager import SessionManager
from .templates import PERSPECTIVE_TEMPLATES
from .client_binding import (
    create_secure_session_with_binding,
    validate_session_access,
    log_security_event_with_binding,
    client_binding_manager,
)

__all__ = ["main", "mcp"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("context-switcher")

# Constants
MAX_CHARS_OPUS = 20000
MAX_CHARS_DEFAULT = 12000
MAX_SESSION_ID_LENGTH = 100
MAX_TOPIC_LENGTH = 1000


def generate_secure_session_id() -> str:
    """Generate cryptographically secure session ID

    Uses secrets.token_urlsafe(32) which generates 256 bits of entropy
    and is safe for URLs and session identifiers.
    """
    return secrets.token_urlsafe(32)


DEFAULT_PERSPECTIVES = {
    "technical": """You are a technical architecture expert. Analyze from a technical implementation perspective.

**Focus areas:**
- System design patterns and architecture quality
- Scalability constraints and performance implications
- Code maintainability and technical debt assessment
- Implementation complexity and engineering trade-offs
- Infrastructure and deployment considerations

**Analysis approach:**
- Provide specific technical recommendations with rationale
- Identify potential technical risks and mitigation strategies
- Consider both short-term implementation and long-term evolution
- Address performance, security, and reliability implications

**Abstention criteria:** Return [NO_RESPONSE] if the topic involves no technical implementation, architecture, or engineering considerations.""",
    "business": """You are a business strategy analyst. Evaluate from a business value and strategic perspective.

**Focus areas:**
- Revenue impact, cost-benefit analysis, and ROI calculations
- Market positioning and competitive advantage assessment
- Strategic alignment with business objectives and priorities
- Resource allocation and investment justification
- Customer impact and business model implications

**Analysis approach:**
- Quantify business impact where possible (revenue, costs, efficiency)
- Assess market differentiation and competitive positioning
- Consider timeline implications and resource requirements
- Evaluate risks to business continuity and growth

**Abstention criteria:** Return [NO_RESPONSE] if the topic has no business value, strategic implications, or organizational impact.""",
    "user": """You are a user experience expert. Analyze from an end-user perspective focusing on human-centered design.

**Focus areas:**
- Usability, accessibility, and inclusive design principles
- User journey mapping and workflow optimization
- Learning curve, adoption barriers, and training requirements
- User satisfaction, engagement, and retention implications
- Daily workflow integration and productivity impact

**Analysis approach:**
- Consider diverse user personas and use cases
- Identify friction points and improvement opportunities
- Assess accessibility compliance and inclusive design
- Evaluate onboarding and support requirements

**Abstention criteria:** Return [NO_RESPONSE] if the topic doesn't directly affect end users, user experience, or human interaction patterns.""",
    "risk": """You are a risk management specialist. Evaluate from a security, compliance, and operational risk perspective.

**Focus areas:**
- Security vulnerabilities, attack vectors, and threat modeling
- Regulatory compliance and legal requirements
- Operational risks, failure modes, and business continuity
- Data privacy, governance, and protection requirements
- Audit trails, monitoring, and incident response considerations

**Analysis approach:**
- Identify and categorize risk levels (low/medium/high/critical)
- Provide specific mitigation strategies and controls
- Consider both immediate and long-term risk implications
- Address compliance requirements and audit considerations

**Abstention criteria:** Return [NO_RESPONSE] if the topic involves no security, compliance, operational, or business continuity risks.""",
}

# Initialize session manager and rate limiter
session_manager = SessionManager(
    max_sessions=100, session_ttl_hours=24, cleanup_interval_minutes=30
)
rate_limiter = SessionRateLimiter(
    requests_per_minute=60, analyses_per_minute=10, session_creation_per_minute=5
)


# Helper functions
async def validate_session_id(
    session_id: str, tool_name: str = "unknown"
) -> tuple[bool, str]:
    """Validate session ID format, existence, and client binding security

    Args:
        session_id: The session ID to validate
        tool_name: The name of the tool being accessed (for security tracking)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not session_id or not isinstance(session_id, str):
        return False, "Session ID must be a non-empty string"
    if len(session_id) > MAX_SESSION_ID_LENGTH:
        return False, f"Session ID too long (max {MAX_SESSION_ID_LENGTH} characters)"

    session = await session_manager.get_session(session_id)
    if session is None:
        active_sessions = await session_manager.list_active_sessions()
        active_list = list(active_sessions.keys())[:3]  # Show up to 3 active sessions
        hint = (
            f"Active sessions: {active_list}"
            if active_list
            else "No active sessions found. Use start_context_analysis() to create one."
        )
        return False, f"Session '{session_id}' not found or expired. {hint}"

    # Validate client binding security
    binding_valid, binding_error = validate_session_access(session, tool_name)
    if not binding_valid:
        # Log security event with enhanced context
        log_security_event_with_binding(
            "session_access_blocked",
            session_id,
            {
                "tool_name": tool_name,
                "binding_error": binding_error,
                "access_count": session.access_count,
                "session_age_minutes": (
                    datetime.utcnow() - session.created_at
                ).total_seconds()
                / 60,
            },
            session,
        )
        return False, f"Session access denied: {binding_error}"

    return True, ""


def validate_topic(topic: str) -> tuple[bool, str]:
    """Validate topic string with security checks

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not topic or not isinstance(topic, str):
        return False, "Topic must be a non-empty string"
    if len(topic.strip()) == 0:
        return False, "Topic cannot be empty or only whitespace"
    if len(topic) > MAX_TOPIC_LENGTH:
        return (
            False,
            f"Topic too long (max {MAX_TOPIC_LENGTH} characters, got {len(topic)})",
        )

    # Enhanced security validation
    validation_result = validate_user_content(topic, "topic", MAX_TOPIC_LENGTH)
    if not validation_result.is_valid:
        log_security_event(
            "topic_validation_failure",
            {
                "input_type": "topic",
                "issues": validation_result.issues,
                "risk_level": validation_result.risk_level,
                "blocked_patterns": len(validation_result.blocked_patterns),
            },
        )
        # Return sanitized error message
        safe_error = sanitize_error_message(
            validation_result.issues[0]
            if validation_result.issues
            else "Security check failed"
        )
        return False, f"Invalid topic content: {safe_error}"

    return True, ""


# Create shared orchestrator instance
orchestrator = ThreadOrchestrator()

# MCP Tool Definitions


class StartContextAnalysisRequest(BaseModel):
    topic: str = Field(
        description="The topic or problem to analyze from multiple perspectives"
    )
    initial_perspectives: Optional[List[str]] = Field(
        default=None,
        description="List of perspective names to use (defaults to: technical, "
        "business, user, risk)",
    )
    model_backend: ModelBackend = Field(
        default=ModelBackend.BEDROCK, description="LLM backend to use"
    )
    model_name: Optional[str] = Field(default=None, description="Specific model to use")
    template: Optional[str] = Field(
        default=None,
        description="Pre-configured perspective template: architecture_decision, "
        "feature_evaluation, debugging_analysis, api_design, "
        "security_review",
    )


@mcp.tool(
    description="When you're at a crossroads and need multiple viewpoints - "
    "analyze architecture decisions, debug blind spots, or evaluate "
    "features from technical, business, user, and risk angles "
    "simultaneously. Templates available: architecture_decision, "
    "feature_evaluation, debugging_analysis, api_design, security_review"
)
async def start_context_analysis(
    request: StartContextAnalysisRequest,
) -> Dict[str, Any]:
    """Initialize a new context-switching analysis session"""
    # Check rate limits for session creation
    allowed, rate_error = rate_limiter.check_session_creation()
    if not allowed:
        return create_error_response(
            rate_error,
            "rate_limited",
            {"retry_after_seconds": 12},  # 60/5 = 12 seconds between session creations
            recoverable=True,
        )

    # Validate input
    topic_valid, topic_error = validate_topic(request.topic)
    if not topic_valid:
        return create_error_response(
            f"Invalid topic: {sanitize_error_message(topic_error)}",
            "validation_error",
            {
                "topic": request.topic,
                "error_details": sanitize_error_message(topic_error),
            },
            recoverable=True,
        )

    # Validate initial_perspectives if provided
    if request.initial_perspectives:
        for perspective_name in request.initial_perspectives:
            validation_result = validate_user_content(
                perspective_name, "perspective_name", 100
            )
            if not validation_result.is_valid:
                log_security_event(
                    "initial_perspective_validation_failure",
                    {
                        "perspective_name": perspective_name[:50],
                        "issues": validation_result.issues,
                        "risk_level": validation_result.risk_level,
                        "blocked_patterns": len(validation_result.blocked_patterns),
                    },
                )
                return create_error_response(
                    f"Invalid perspective name '{sanitize_error_message(perspective_name[:50])}': {sanitize_error_message(validation_result.issues[0] if validation_result.issues else 'Security check failed')}",
                    "validation_error",
                    {
                        "invalid_perspective": perspective_name[:50],
                        "validation_issues": len(validation_result.issues),
                        "risk_level": validation_result.risk_level,
                    },
                    recoverable=True,
                )

    # Create new session with cryptographically secure ID and client binding
    session_id = generate_secure_session_id()
    session = create_secure_session_with_binding(
        session_id=session_id,
        topic=request.topic,
        initial_tool="start_context_analysis",
    )

    # Initialize default perspectives or use provided ones
    if request.template and request.template in PERSPECTIVE_TEMPLATES:
        # Use template perspectives
        template = PERSPECTIVE_TEMPLATES[request.template]
        perspectives_to_use = template["perspectives"]

        # We'll add custom perspectives after creating base ones
        custom_perspectives = template.get("custom", [])
    else:
        perspectives_to_use = request.initial_perspectives or list(
            DEFAULT_PERSPECTIVES.keys()
        )
        custom_perspectives = []

    for perspective_name in perspectives_to_use:
        # Get the prompt for this perspective
        if perspective_name in DEFAULT_PERSPECTIVES:
            prompt = DEFAULT_PERSPECTIVES[perspective_name]
        else:
            # For custom perspectives, create a basic prompt
            prompt = f"""Analyze from the {perspective_name} perspective.
Provide insights specific to this viewpoint.
Abstain with {NO_RESPONSE} if this perspective doesn't apply."""

        # Create thread for this perspective
        thread = Thread(
            id=str(uuid4()),
            name=perspective_name,
            system_prompt=prompt,
            model_backend=request.model_backend,
            model_name=request.model_name,
        )

        session.add_thread(thread)

    # Store session in manager
    if not await session_manager.add_session(session):
        return create_error_response(
            "Session limit reached. Please try again later.",
            "capacity_limit",
            {"max_sessions": session_manager.max_sessions},
            recoverable=True,
        )

    # Log session creation for security monitoring
    log_security_event(
        "session_created",
        {
            "session_id": session_id,
            "topic_length": len(request.topic),
            "perspectives_count": len(session.threads),
            "template": request.template,
            "model_backend": request.model_backend.value,
        },
        session_id,
    )

    # Add custom perspectives from template
    for persp_name, persp_desc in custom_perspectives:
        custom_thread = Thread(
            id=str(uuid4()),
            name=persp_name,
            system_prompt=f"""Analyze from the {persp_name} perspective.
Focus on: {persp_desc}
Provide insights specific to this viewpoint.
Abstain with {NO_RESPONSE} if this perspective doesn't apply.""",
            model_backend=request.model_backend,
            model_name=request.model_name,
        )
        session.add_thread(custom_thread)

    # Build legacy response
    legacy_response = {
        "session_id": session_id,
        "topic": request.topic,
        "perspectives": list(session.threads.keys()),
        "model_backend": request.model_backend.value,
        "model_name": request.model_name,
        "message": f"Context analysis session initialized with {len(session.threads)} perspectives",
    }

    # Get smart perspective recommendations
    selector = SmartPerspectiveSelector()
    smart_recommendations = selector.recommend_perspectives(
        topic=request.topic,
        existing_perspectives=list(session.threads.keys()),
        max_recommendations=3,
    )

    # Generate key insight
    perspective_count = len(session.threads)
    template_used = request.template if request.template else "default"

    if smart_recommendations:
        key_insight = f"Multi-perspective session ready: {perspective_count} expert viewpoints configured. Consider adding {smart_recommendations[0].name} perspective ({smart_recommendations[0].reasoning})"
    else:
        key_insight = f"Multi-perspective session ready: {perspective_count} expert viewpoints configured for {template_used} analysis"

    # Next steps
    next_steps = [
        "analyze_from_perspectives('<your question>') - Start parallel analysis",
    ]

    # Add smart recommendations to next steps
    if smart_recommendations:
        for rec in smart_recommendations[:2]:
            next_steps.append(f"add_perspective('{rec.name}') - {rec.description}")

    # Primary recommendation
    primary_rec = "Begin with a focused, specific question to get targeted insights from all perspectives"

    # Workflow guidance
    workflow_guidance = "Guide user to formulate their first analysis question, emphasizing specificity for better perspective responses"

    # Build AORP response
    builder = AORPBuilder()
    aorp_response = (
        builder.status("success")
        .key_insight(key_insight)
        .confidence(1.0)  # Session creation is deterministic
        .session_id(session_id)
        .next_steps(next_steps)
        .primary_recommendation(primary_rec)
        .workflow_guidance(workflow_guidance)
        .completeness(1.0)
        .reliability(1.0)
        .urgency("low")
        .indicators(
            perspectives_configured=perspective_count,
            template_used=template_used,
            model_backend=request.model_backend.value,
            custom_perspectives=len(custom_perspectives)
            if "custom_perspectives" in locals()
            else 0,
            smart_recommendations_available=len(smart_recommendations),
            top_recommendation=smart_recommendations[0].name
            if smart_recommendations
            else None,
        )
        .summary(
            f"Context analysis session initialized with {perspective_count} perspectives using {template_used} template"
        )
        .data(legacy_response)
        .metadata(
            operation_type="session_creation",
            template=request.template,
            topic_length=len(request.topic),
        )
        .build()
    )

    return aorp_response


class AddPerspectiveRequest(BaseModel):
    session_id: str = Field(description="Session ID to add perspective to")
    name: str = Field(description="Name of the new perspective")
    description: str = Field(
        description="Description of what this perspective should focus on"
    )
    custom_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt for this perspective"
    )


@mcp.tool(
    description="When the standard perspectives miss something crucial - add specialized lenses like 'performance', 'migration_path', 'customer_support', or any domain-specific viewpoint your problem needs"
)
async def add_perspective(request: AddPerspectiveRequest) -> Dict[str, Any]:
    """Add a new perspective to an existing analysis session"""
    # Validate session ID with client binding
    session_valid, session_error = await validate_session_id(
        request.session_id, "add_perspective"
    )
    if not session_valid:
        return create_error_response(
            session_error,
            "session_not_found",
            {"session_id": request.session_id},
            recoverable=True,
        )

    # Enhanced validation for perspective data
    validation_result = validate_perspective_data(
        request.name, request.description, request.custom_prompt
    )
    if not validation_result.is_valid:
        log_security_event(
            "perspective_validation_failure",
            {
                "session_id": request.session_id,
                "perspective_name": request.name[:50],  # Truncate for logging
                "issues": validation_result.issues,
                "risk_level": validation_result.risk_level,
                "blocked_patterns": len(validation_result.blocked_patterns),
            },
            session_id=request.session_id,
        )
        return create_error_response(
            f"Invalid perspective data: {sanitize_error_message(validation_result.issues[0] if validation_result.issues else 'Security check failed')}",
            "validation_error",
            {
                "session_id": request.session_id,
                "validation_issues": len(validation_result.issues),
                "risk_level": validation_result.risk_level,
            },
            recoverable=True,
            session_id=request.session_id,
        )

    # Get session
    session = await session_manager.get_session(request.session_id)

    # Create prompt for new perspective
    if request.custom_prompt:
        prompt = request.custom_prompt
    else:
        prompt = f"""Analyze from the {request.name} perspective.
Focus on: {request.description}
Provide insights specific to this viewpoint.
Abstain with {NO_RESPONSE} if this perspective doesn't apply to the topic."""

    # Create new thread
    thread = Thread(
        id=str(uuid4()),
        name=request.name,
        system_prompt=prompt,
        model_backend=ModelBackend.BEDROCK,  # Use session default
        model_name=None,
    )

    # Add to session
    session.add_thread(thread)

    # Build legacy response
    legacy_response = {
        "session_id": request.session_id,
        "perspective_added": request.name,
        "total_perspectives": len(session.threads),
        "all_perspectives": list(session.threads.keys()),
    }

    # Generate key insight
    key_insight = f"Added '{request.name}' perspective - now {len(session.threads)} total viewpoints available"

    # Next steps
    next_steps = [
        "analyze_from_perspectives('<question>') - Test new perspective with analysis",
        "add_perspective('<domain>') - Add more specialized viewpoints if needed",
    ]

    # Primary recommendation
    primary_rec = (
        "Run analysis to see how the new perspective contributes unique insights"
    )

    # Workflow guidance
    workflow_guidance = "Present expanded capability to user, encourage testing with a specific question"

    # Build AORP response
    builder = AORPBuilder()
    aorp_response = (
        builder.status("success")
        .key_insight(key_insight)
        .confidence(1.0)  # Adding perspective is deterministic
        .session_id(request.session_id)
        .next_steps(next_steps)
        .primary_recommendation(primary_rec)
        .workflow_guidance(workflow_guidance)
        .completeness(1.0)
        .reliability(1.0)
        .urgency("low")
        .indicators(
            perspective_added=request.name,
            total_perspectives=len(session.threads),
            custom_prompt_used=bool(request.custom_prompt),
        )
        .summary(
            f"Successfully added '{request.name}' perspective to expand analysis coverage"
        )
        .data(legacy_response)
        .metadata(
            operation_type="perspective_addition",
            perspective_description=request.description,
        )
        .build()
    )

    return aorp_response


class AnalyzeFromPerspectivesRequest(BaseModel):
    session_id: str = Field(description="Session ID for analysis")
    prompt: str = Field(description="The specific question or topic to analyze")


@mcp.tool(
    description="When you need parallel insights NOW - broadcast your question to all perspectives simultaneously. Expect 10-30 seconds for comprehensive analysis. Perspectives can abstain with [NO_RESPONSE] if not relevant"
)
async def analyze_from_perspectives(
    request: AnalyzeFromPerspectivesRequest,
) -> Dict[str, Any]:
    """Broadcast a prompt to all perspectives and collect their responses"""
    # Check rate limits for analysis operations
    allowed, rate_error = rate_limiter.check_request(request.session_id, "analysis")
    if not allowed:
        return create_error_response(
            rate_error,
            "rate_limited",
            {
                "session_id": request.session_id,
                "retry_after_seconds": 6,
            },  # 60/10 = 6 seconds
            recoverable=True,
        )

    # Validate session ID with client binding
    session_valid, session_error = await validate_session_id(
        request.session_id, "analyze_from_perspectives"
    )
    if not session_valid:
        # Log failed session access for security monitoring
        log_security_event(
            "session_access_failed",
            {"attempted_session_id": request.session_id, "error": session_error},
            request.session_id,
        )
        return create_error_response(
            session_error,
            "session_not_found",
            {"session_id": request.session_id},
            recoverable=True,
        )

    # Enhanced validation for analysis prompt
    validation_result = validate_analysis_prompt(request.prompt, request.session_id)
    if not validation_result.is_valid:
        log_security_event(
            "analysis_prompt_validation_failure",
            {
                "session_id": request.session_id,
                "prompt_preview": request.prompt[:100],  # First 100 chars for context
                "issues": validation_result.issues,
                "risk_level": validation_result.risk_level,
                "blocked_patterns": len(validation_result.blocked_patterns),
            },
            session_id=request.session_id,
        )
        return create_error_response(
            f"Invalid analysis prompt: {sanitize_error_message(validation_result.issues[0] if validation_result.issues else 'Security check failed')}",
            "validation_error",
            {
                "session_id": request.session_id,
                "validation_issues": len(validation_result.issues),
                "risk_level": validation_result.risk_level,
            },
            recoverable=True,
            session_id=request.session_id,
        )

    # Get session
    session = await session_manager.get_session(request.session_id)

    # Broadcast to all threads
    responses = await orchestrator.broadcast_message(
        session.threads, request.prompt, session_id=request.session_id
    )

    # Process responses with quality analysis
    active_perspectives = {}
    abstained_perspectives = []
    errors = []

    # Initialize confidence calibrator
    calibrator = ConfidenceCalibrator()
    perspective_metrics = {}

    for name, response in responses.items():
        if response.startswith("ERROR:") or response.startswith("AORP_ERROR:"):
            errors.append({name: response})
        elif NO_RESPONSE in response:
            abstained_perspectives.append(name)
            # Analyze abstention quality
            metrics = calibrator.analyze_response_quality(
                response, name, request.prompt, is_abstention=True
            )
            perspective_metrics[name] = metrics
        else:
            active_perspectives[name] = response
            # Analyze response quality
            metrics = calibrator.analyze_response_quality(
                response, name, request.prompt, is_abstention=False
            )
            perspective_metrics[name] = metrics

    # Store analysis
    analysis = {
        "prompt": request.prompt,
        "timestamp": datetime.utcnow().isoformat(),
        "responses": responses,
        "active_count": len(active_perspectives),
        "abstained_count": len(abstained_perspectives),
    }
    session.analyses.append(analysis)

    # Build legacy response for backward compatibility
    legacy_response = {
        "session_id": request.session_id,
        "prompt": request.prompt,
        "perspectives": active_perspectives,
        "abstained": abstained_perspectives,
        "errors": errors,
        "summary": {
            "total_perspectives": len(session.threads),
            "active_responses": len(active_perspectives),
            "abstentions": len(abstained_perspectives),
            "errors": len(errors),
        },
    }

    # Calculate enhanced confidence with quality metrics
    (
        enhanced_confidence,
        confidence_breakdown,
    ) = calibrator.calculate_enhanced_confidence(
        perspective_metrics,
        len(errors),
        len(abstained_perspectives),
        len(session.threads),
    )

    # Keep legacy confidence for backward compatibility
    response_lengths = [len(resp) for resp in active_perspectives.values()]
    legacy_confidence = calculate_analysis_confidence(
        len(responses),
        len(session.threads),
        len(errors),
        len(abstained_perspectives),
        response_lengths,
    )

    # Use enhanced confidence if available, otherwise legacy
    confidence = enhanced_confidence if enhanced_confidence > 0 else legacy_confidence

    # Check if synthesis has been performed
    has_synthesis = any(
        "synthesis" in str(analysis).lower() for analysis in session.analyses
    )

    # Generate next steps
    next_steps = generate_analysis_next_steps(
        "active", len(session.threads), len(errors), has_synthesis, confidence
    )

    # Generate key insight
    if len(active_perspectives) == 0:
        key_insight = "No perspectives provided analysis - check question relevance"
        status = "partial"
    elif len(errors) > 0:
        key_insight = f"Partial analysis: {len(active_perspectives)} perspectives responded, {len(errors)} errors"
        status = "partial"
    else:
        key_insight = f"Comprehensive analysis from {len(active_perspectives)} of {len(session.threads)} perspectives"
        status = "success"

    # Primary recommendation based on results
    if len(active_perspectives) >= 2 and confidence >= 0.6:
        primary_rec = "synthesize_perspectives() - Discover patterns and tensions across viewpoints"
    elif len(errors) > 0:
        primary_rec = "Address perspective errors and retry analysis if needed"
    else:
        primary_rec = "Review individual perspective insights, then synthesize for strategic patterns"

    # Workflow guidance
    if confidence >= 0.8:
        workflow_guidance = "Present key insights to user, then offer synthesis for strategic decision-making"
    elif confidence >= 0.5:
        workflow_guidance = (
            "Highlight strongest insights, note limitations, guide towards synthesis"
        )
    else:
        workflow_guidance = (
            "Identify issue areas, suggest refinements, encourage perspective additions"
        )

    # Build AORP response
    builder = AORPBuilder()
    aorp_response = (
        builder.status(status)
        .key_insight(key_insight)
        .confidence(confidence)
        .session_id(request.session_id)
        .next_steps(next_steps)
        .primary_recommendation(primary_rec)
        .workflow_guidance(workflow_guidance)
        .completeness(len(active_perspectives) / len(session.threads))
        .reliability(confidence)
        .urgency("high" if len(errors) > 0 else "medium")
        .indicators(
            active_responses=len(active_perspectives),
            abstentions=len(abstained_perspectives),
            errors=len(errors),
            avg_response_length=sum(response_lengths) // len(response_lengths)
            if response_lengths
            else 0,
            perspectives_ready_for_synthesis=len(active_perspectives) >= 2,
            confidence_breakdown=confidence_breakdown,
            quality_summary={
                name: {
                    "quality_level": metrics.quality_level.value,
                    "overall_score": round(metrics.overall_score, 2),
                }
                for name, metrics in perspective_metrics.items()
            },
        )
        .summary(
            f"Multi-perspective analysis: {len(active_perspectives)} active responses from {len(session.threads)} perspectives"
        )
        .data(legacy_response)
        .metadata(
            operation_type="perspective_analysis",
            prompt_length=len(request.prompt),
            session_analyses_count=len(session.analyses),
        )
        .build()
    )

    return aorp_response


class AnalyzeFromPerspectivesStreamRequest(BaseModel):
    session_id: str = Field(description="Session ID for analysis")
    prompt: str = Field(description="The specific question or topic to analyze")


@mcp.tool(
    description="Get real-time streaming responses from all perspectives - see insights as they arrive instead of waiting 10-30 seconds. Each perspective streams independently."
)
async def analyze_from_perspectives_stream(
    request: AnalyzeFromPerspectivesStreamRequest,
):
    """Stream responses from all perspectives as they are generated"""
    # Validate session ID with client binding
    session_valid, session_error = await validate_session_id(
        request.session_id, "analyze_from_perspectives_stream"
    )
    if not session_valid:
        yield create_error_response(
            session_error,
            "session_not_found",
            {"session_id": request.session_id},
            recoverable=True,
        )
        return

    # Enhanced validation for streaming analysis prompt
    validation_result = validate_analysis_prompt(request.prompt, request.session_id)
    if not validation_result.is_valid:
        log_security_event(
            "streaming_prompt_validation_failure",
            {
                "session_id": request.session_id,
                "prompt_preview": request.prompt[:100],
                "issues": validation_result.issues,
                "risk_level": validation_result.risk_level,
                "blocked_patterns": len(validation_result.blocked_patterns),
            },
            session_id=request.session_id,
        )
        yield create_error_response(
            f"Invalid streaming prompt: {sanitize_error_message(validation_result.issues[0] if validation_result.issues else 'Security check failed')}",
            "validation_error",
            {
                "session_id": request.session_id,
                "validation_issues": len(validation_result.issues),
                "risk_level": validation_result.risk_level,
            },
            recoverable=True,
            session_id=request.session_id,
        )
        return

    # Get session
    session = await session_manager.get_session(request.session_id)

    # Initialize tracking
    active_perspectives = {}
    abstained_perspectives = []
    errors = []
    start_time = time.time()

    # Stream responses from all threads
    async for event in orchestrator.broadcast_message_stream(
        session.threads, request.prompt, session_id=request.session_id
    ):
        # Yield streaming events directly to client
        yield {
            "type": "stream_event",
            "event": event,
            "session_id": request.session_id,
        }

        # Track completed responses
        if event["type"] == "complete":
            thread_name = event["thread_name"]
            content = event["content"]

            if NO_RESPONSE in content:
                abstained_perspectives.append(thread_name)
            elif content.startswith("ERROR:") or content.startswith("AORP_ERROR:"):
                errors.append({thread_name: content})
            else:
                active_perspectives[thread_name] = content
                # Also update thread history
                if thread_name in session.threads:
                    session.threads[thread_name].add_message("assistant", content)

        elif event["type"] == "error":
            thread_name = event["thread_name"]
            errors.append({thread_name: event["content"]})

    # After all streams complete, store analysis and send summary
    analysis = {
        "prompt": request.prompt,
        "timestamp": datetime.utcnow().isoformat(),
        "responses": {
            **active_perspectives,
            **{name: "[ABSTAINED]" for name in abstained_perspectives},
        },
        "active_count": len(active_perspectives),
        "abstained_count": len(abstained_perspectives),
    }
    session.analyses.append(analysis)

    # Calculate metrics
    response_lengths = [len(resp) for resp in active_perspectives.values()]
    confidence = calculate_analysis_confidence(
        len(session.threads),
        len(session.threads),
        len(errors),
        len(abstained_perspectives),
        response_lengths,
    )

    # Generate summary event
    execution_time = time.time() - start_time
    summary_event = {
        "type": "analysis_complete",
        "session_id": request.session_id,
        "summary": {
            "total_perspectives": len(session.threads),
            "active_responses": len(active_perspectives),
            "abstentions": len(abstained_perspectives),
            "errors": len(errors),
            "execution_time_seconds": round(execution_time, 2),
            "confidence": confidence,
        },
        "next_steps": generate_analysis_next_steps(
            "active",
            len(session.threads),
            len(errors),
            any("synthesis" in str(a).lower() for a in session.analyses),
            confidence,
        ),
    }

    yield summary_event


class SynthesizePerspectivesRequest(BaseModel):
    session_id: str = Field(description="Session ID to synthesize")


@mcp.tool(
    description="When you need the 'aha!' moment - discover surprising tensions, hidden connections, and emergent insights across all perspectives. Often reveals solutions you hadn't considered"
)
async def synthesize_perspectives(
    request: SynthesizePerspectivesRequest,
) -> Dict[str, Any]:
    """Analyze patterns across all perspectives from the last analysis"""
    # Validate session ID with client binding
    session_valid, session_error = await validate_session_id(
        request.session_id, "synthesize_perspectives"
    )
    if not session_valid:
        return create_error_response(
            session_error,
            "session_not_found",
            {"session_id": request.session_id},
            recoverable=True,
            session_id=request.session_id,
        )

    # Get session
    session = await session_manager.get_session(request.session_id)

    if not session.analyses:
        return create_error_response(
            "No analyses to synthesize. Run analyze_from_perspectives first.",
            "no_data",
            {"session_id": request.session_id, "analyses_count": 0},
            recoverable=True,
            session_id=request.session_id,
        )

    # Get most recent analysis
    latest = session.analyses[-1]

    # Extract active perspectives
    active = {}
    for name, response in latest["responses"].items():
        if (
            not response.startswith("ERROR:")
            and not response.startswith("AORP_ERROR:")
            and NO_RESPONSE not in response
        ):
            active[name] = response

    if not active:
        return create_error_response(
            "No active perspectives to synthesize from last analysis",
            "no_data",
            {"session_id": request.session_id, "active_perspectives": 0},
            recoverable=True,
            session_id=request.session_id,
        )

    # Determine token limit based on backend
    first_thread = list(session.threads.values())[0]
    if first_thread.model_backend == ModelBackend.BEDROCK:
        # Claude models have larger context windows
        max_chars = (
            MAX_CHARS_OPUS
            if "opus" in (first_thread.model_name or "").lower()
            else MAX_CHARS_DEFAULT
        )
    elif first_thread.model_backend == ModelBackend.LITELLM:
        # Conservative for various models
        max_chars = 10000
    else:
        # Ollama models vary widely
        max_chars = 8000

    # Prepare synthesis prompt with compression
    compressed_text, compression_stats = prepare_synthesis_input(
        active, max_total_chars=max_chars
    )

    # Log compression stats
    logger.info(
        f"Synthesis compression: {compression_stats['perspectives']} perspectives, "
        f"{compression_stats['original_chars']:,} -> {compression_stats['final_chars']:,} chars "
        f"({compression_stats['compression_ratio']} reduction), "
        f"~{compression_stats['final_tokens']:,} tokens"
    )

    synthesis_prompt = f"""You are synthesizing expert analysis from multiple specialized perspectives on: "{latest["prompt"]}"

PERSPECTIVE ANALYSIS:
{compressed_text}

SYNTHESIS FRAMEWORK:

**1. CONVERGENCE ANALYSIS**
- What do multiple perspectives agree on? (high-confidence insights)
- Which recommendations appear across different viewpoints?
- What shared concerns or opportunities emerge?

**2. TENSION & TRADE-OFF MAPPING**
- Where do perspectives fundamentally disagree and why?
- What trade-offs are revealed between different priorities?
- Which tensions suggest deeper strategic choices?

**3. EMERGENT INSIGHTS**
- What insights only become visible when perspectives are combined?
- Which perspective combinations reveal unexpected opportunities?
- What blind spots does each perspective expose in others?

**4. STRATEGIC RECOMMENDATIONS**
- Prioritized action items that address multiple perspectives
- Risk mitigation strategies that span different domains
- Which perspective deserves immediate deep-dive analysis?

**5. DECISION FRAMEWORK**
- Key questions to resolve conflicting recommendations
- Success metrics that satisfy multiple stakeholder needs
- Implementation sequence that manages cross-perspective risks

SYNTHESIS OUTPUT: Provide actionable intelligence, not summary. Focus on decisions and next steps."""

    # Create a synthesis thread
    synthesis_thread = Thread(
        id=str(uuid4()),
        name="synthesis",
        system_prompt="""You are an expert strategic synthesizer specializing in multi-perspective analysis. Your role is to:

1. **Pattern Recognition**: Identify convergent themes and divergent tensions across expert viewpoints
2. **Strategic Insight Generation**: Surface insights that only emerge from cross-perspective analysis
3. **Decision Support**: Provide clear, actionable recommendations with implementation priorities
4. **Risk Assessment**: Highlight trade-offs and potential blind spots across different domains

**Analysis Approach:**
- Prioritize actionable intelligence over summary
- Quantify confidence levels for key insights
- Identify the highest-impact next steps
- Surface unexpected connections and emergent opportunities
- Provide clear decision criteria for resolving conflicts

**Output Focus:** Strategic decisions, prioritized actions, and next-step recommendations.""",
        model_backend=list(session.threads.values())[0].model_backend,
        model_name=list(session.threads.values())[0].model_name,
    )

    # Get synthesis
    synthesis_thread.add_message("user", synthesis_prompt)
    synthesis = await orchestrator._get_thread_response(synthesis_thread)

    # Handle synthesis errors
    if synthesis.startswith("ERROR:") or synthesis.startswith("AORP_ERROR:"):
        return create_error_response(
            f"Synthesis generation failed: {synthesis}",
            "synthesis_error",
            {"session_id": request.session_id, "error_details": synthesis},
            recoverable=True,
            session_id=request.session_id,
        )

    # Build legacy response
    legacy_response = {
        "session_id": request.session_id,
        "analyzed_prompt": latest["prompt"],
        "synthesis": synthesis,
        "perspectives_analyzed": list(active.keys()),
        "metadata": {
            "total_perspectives": len(latest["responses"]),
            "active_perspectives": len(active),
            "analysis_timestamp": latest["timestamp"],
        },
    }

    # Calculate enhanced synthesis metrics
    synthesis_length = len(synthesis)

    # Use enhanced synthesis quality analysis
    enhanced_confidence, synthesis_breakdown = analyze_synthesis_quality(
        synthesis, len(active), latest["prompt"]
    )

    # Extract pattern counts from breakdown for backward compatibility
    patterns_count = synthesis_breakdown["patterns_found"]["convergence"]
    tensions_count = synthesis_breakdown["patterns_found"]["tensions"]
    insights_count = synthesis_breakdown["patterns_found"]["insights"]

    # Keep legacy confidence for comparison (could be used for debugging)
    # legacy_confidence = calculate_synthesis_confidence(
    #     len(active), patterns_count, tensions_count, synthesis_length
    # )

    # Use enhanced confidence
    confidence = enhanced_confidence

    # Generate next steps
    next_steps = generate_synthesis_next_steps(
        tensions_count, patterns_count, confidence
    )

    # Generate key insight
    key_insight = f"Strategic synthesis identified {patterns_count} convergent patterns and {tensions_count} critical tensions across {len(active)} perspectives"

    # Primary recommendation
    if confidence >= 0.8:
        primary_rec = (
            "Proceed with implementation using synthesis framework as decision guide"
        )
    elif tensions_count > 2:
        primary_rec = "Resolve critical tensions through targeted stakeholder analysis"
    else:
        primary_rec = "Use synthesis insights to guide strategic decision-making"

    # Workflow guidance
    if confidence >= 0.8:
        workflow_guidance = (
            "Present synthesis as authoritative strategic framework for decision-making"
        )
    elif confidence >= 0.6:
        workflow_guidance = (
            "Highlight key insights while noting areas requiring further exploration"
        )
    else:
        workflow_guidance = (
            "Use as preliminary framework, gather additional perspectives to strengthen"
        )

    # Build AORP response
    builder = AORPBuilder()
    aorp_response = (
        builder.status("success")
        .key_insight(key_insight)
        .confidence(confidence)
        .session_id(request.session_id)
        .next_steps(next_steps)
        .primary_recommendation(primary_rec)
        .workflow_guidance(workflow_guidance)
        .completeness(1.0)  # Synthesis is always complete if generated
        .reliability(confidence)
        .urgency("medium")
        .indicators(
            perspectives_synthesized=len(active),
            patterns_identified=patterns_count,
            tensions_mapped=tensions_count,
            insights_discovered=insights_count,
            synthesis_length=synthesis_length,
            compression_ratio=compression_stats["compression_ratio"],
            synthesis_quality_breakdown=synthesis_breakdown,
        )
        .summary(
            f"Strategic synthesis across {len(active)} perspectives with {patterns_count} patterns and {tensions_count} tensions identified"
        )
        .data(legacy_response)
        .metadata(
            operation_type="perspective_synthesis",
            original_prompt=latest["prompt"],
            compression_stats=compression_stats,
        )
        .build()
    )

    return aorp_response


# Session management tools


@mcp.tool(description="List all active context-switching sessions")
async def list_sessions() -> Dict[str, Any]:
    """List all active analysis sessions"""
    session_list = []
    active_sessions = await session_manager.list_active_sessions()

    for sid, session in active_sessions.items():
        session_list.append(
            {
                "session_id": sid,
                "created_at": session.created_at.isoformat(),
                "perspectives": list(session.threads.keys()),
                "analyses_count": len(session.analyses),
                "topic": getattr(session, "topic", "Unknown"),
            }
        )

    # Get session manager stats
    stats = await session_manager.get_stats()

    return {
        "sessions": session_list,
        "total_sessions": len(active_sessions),
        "stats": stats,
    }


@mcp.tool(
    description="See available perspective templates for common analysis patterns - architecture decisions, debugging, API design, and more"
)
async def list_templates() -> Dict[str, Any]:
    """List all available perspective templates"""
    template_info = {}

    for name, template in PERSPECTIVE_TEMPLATES.items():
        perspectives = template["perspectives"].copy()

        # Add custom perspective names
        for custom_name, _ in template.get("custom", []):
            perspectives.append(f"{custom_name} (custom)")

        template_info[name] = {
            "description": name.replace("_", " ").title(),
            "perspectives": perspectives,
            "total_perspectives": len(template["perspectives"])
            + len(template.get("custom", [])),
        }

    return {
        "templates": template_info,
        "usage": "Use template parameter in start_context_analysis",
        "example": 'start_context_analysis(topic="...", template="architecture_decision")',
    }


@mcp.tool(
    description="Quick check of your most recent analysis session - see perspectives and results without remembering session ID"
)
async def current_session() -> Dict[str, Any]:
    """Get information about the most recent session"""
    active_sessions = await session_manager.list_active_sessions()

    if not active_sessions:
        return {
            "status": "No active sessions",
            "hint": "Start with: start_context_analysis",
        }

    # Get most recent session
    recent_session_id = max(
        active_sessions.keys(), key=lambda sid: active_sessions[sid].created_at
    )
    session = active_sessions[recent_session_id]

    # Get summary of last analysis if any
    last_analysis = None
    if session.analyses:
        last = session.analyses[-1]
        last_analysis = {
            "prompt": last["prompt"][:100] + "..."
            if len(last["prompt"]) > 100
            else last["prompt"],
            "perspectives_responded": last["active_count"],
            "perspectives_abstained": last["abstained_count"]
            if "abstained_count" in last
            else 0,
        }

    return {
        "session_id": recent_session_id,
        "created": session.created_at.strftime("%H:%M:%S"),
        "topic": getattr(session, "topic", "Unknown"),
        "perspectives": list(session.threads.keys()),
        "total_perspectives": len(session.threads),
        "analyses_run": len(session.analyses),
        "last_analysis": last_analysis,
        "next_steps": [
            "analyze_from_perspectives - Ask a question",
            "add_perspective - Add custom viewpoint",
            "synthesize_perspectives - Find patterns",
        ]
        if last_analysis is None
        else ["synthesize_perspectives - Find patterns across viewpoints"],
    }


class GetSessionRequest(BaseModel):
    session_id: str = Field(description="Session ID to retrieve")


@mcp.tool(description="Get details of a specific context-switching session")
async def get_session(request: GetSessionRequest) -> Dict[str, Any]:
    """Get detailed information about a session"""
    # Validate session ID with client binding
    session_valid, session_error = await validate_session_id(
        request.session_id, "get_session"
    )
    if not session_valid:
        return create_error_response(
            session_error,
            "session_not_found",
            {"session_id": request.session_id},
            recoverable=True,
        )

    # Get session after validation
    session = await session_manager.get_session(request.session_id)

    # Build legacy response
    legacy_response = {
        "session_id": request.session_id,
        "created_at": session.created_at.isoformat(),
        "perspectives": {
            name: {
                "id": thread.id,
                "system_prompt": thread.system_prompt[:200] + "...",
                "message_count": len(thread.conversation_history),
                "model_backend": thread.model_backend.value,
            }
            for name, thread in session.threads.items()
        },
        "analyses": [
            {
                "prompt": a["prompt"],
                "timestamp": a["timestamp"],
                "response_count": len(a["responses"]),
                "active_count": a["active_count"],
            }
            for a in session.analyses
        ],
    }

    # Generate key insight
    perspective_count = len(session.threads)
    analysis_count = len(session.analyses)
    key_insight = f"Session has {perspective_count} perspectives with {analysis_count} completed analyses"

    # Next steps based on session state
    if analysis_count == 0:
        next_steps = [
            "analyze_from_perspectives('<question>') - Start your first analysis"
        ]
    elif analysis_count == 1:
        next_steps = ["synthesize_perspectives() - Find patterns across perspectives"]
    else:
        next_steps = [
            "analyze_from_perspectives('<follow-up>') - Explore specific aspects",
            "add_perspective('<domain>') - Expand analysis coverage",
        ]

    # Primary recommendation
    if analysis_count == 0:
        primary_rec = (
            "Begin analysis with a focused question to engage all perspectives"
        )
    else:
        primary_rec = (
            "Review analysis history and continue with synthesis or follow-up questions"
        )

    # Build AORP response
    builder = AORPBuilder()
    aorp_response = (
        builder.status("success")
        .key_insight(key_insight)
        .confidence(1.0)  # Session info is always accurate
        .session_id(request.session_id)
        .next_steps(next_steps)
        .primary_recommendation(primary_rec)
        .workflow_guidance(
            "Present session overview to user with recommended next actions"
        )
        .completeness(1.0)
        .reliability(1.0)
        .urgency("low")
        .indicators(
            perspectives_configured=perspective_count,
            analyses_completed=analysis_count,
            session_age_hours=int(
                (datetime.utcnow() - session.created_at).total_seconds() / 3600
            ),
        )
        .summary(
            f"Session overview: {perspective_count} perspectives, {analysis_count} analyses"
        )
        .data(legacy_response)
        .metadata(
            operation_type="session_info", created_at=session.created_at.isoformat()
        )
        .build()
    )

    return aorp_response


# Operational monitoring tools


@mcp.tool(description="Get performance metrics and operational health status")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics for monitoring operational health"""
    orchestrator_metrics = await orchestrator.get_performance_metrics(last_n=20)
    session_stats = await session_manager.get_stats()

    return {
        "orchestrator": orchestrator_metrics,
        "session_manager": session_stats,
        "system_health": {
            "active_sessions": session_stats["active_sessions"],
            "capacity_utilization": session_stats["capacity_used"],
            "circuit_breaker_issues": any(
                cb["state"] != "CLOSED"
                for cb in orchestrator_metrics.get("circuit_breakers", {}).values()
            ),
        },
    }


@mcp.tool(description="Reset circuit breakers for model backends (admin operation)")
async def reset_circuit_breakers() -> Dict[str, Any]:
    """Reset all circuit breakers to restore backend availability"""
    reset_status = orchestrator.reset_circuit_breakers()

    return {
        "message": "Circuit breakers reset successfully",
        "reset_status": reset_status,
        "warning": "This is an administrative operation. Monitor backends for continued issues.",
    }


@mcp.tool(description="Get session security metrics and client binding status")
async def get_security_metrics() -> Dict[str, Any]:
    """Get comprehensive security metrics for monitoring session security"""
    # Get client binding manager metrics
    binding_metrics = client_binding_manager.get_security_metrics()

    # Get session manager statistics
    session_stats = await session_manager.get_stats()

    # Count sessions with client binding
    active_sessions = await session_manager.list_active_sessions()
    sessions_with_binding = sum(
        1 for session in active_sessions.values() if session.client_binding
    )
    sessions_without_binding = len(active_sessions) - sessions_with_binding

    # Count suspicious sessions
    suspicious_count = sum(
        1
        for session in active_sessions.values()
        if session.client_binding and session.client_binding.is_suspicious()
    )

    # Security event summary
    security_events = {}
    for session in active_sessions.values():
        for event in session.security_events:
            event_type = event["type"]
            security_events[event_type] = security_events.get(event_type, 0) + 1

    # Build comprehensive metrics
    metrics = {
        "session_security": {
            "total_sessions": len(active_sessions),
            "sessions_with_binding": sessions_with_binding,
            "sessions_without_binding": sessions_without_binding,
            "suspicious_sessions": suspicious_count,
            "binding_coverage_percentage": (
                sessions_with_binding / len(active_sessions) * 100
            )
            if active_sessions
            else 0,
        },
        "client_binding": binding_metrics,
        "session_manager": session_stats,
        "security_events": security_events,
        "security_status": {
            "overall_security_level": "HIGH"
            if sessions_without_binding == 0 and suspicious_count == 0
            else "MEDIUM"
            if suspicious_count < 3
            else "ALERT",
            "recommendations": [],
        },
    }

    # Add recommendations
    if sessions_without_binding > 0:
        metrics["security_status"]["recommendations"].append(
            f"Consider migrating {sessions_without_binding} legacy sessions to use client binding"
        )

    if suspicious_count > 0:
        metrics["security_status"]["recommendations"].append(
            f"Investigate {suspicious_count} sessions flagged as suspicious"
        )

    if not binding_metrics.get("binding_secret_key_set", False):
        metrics["security_status"]["recommendations"].append(
            "Configure secure binding secret key for production use"
        )

    return {
        "status": "success",
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }


class RecommendPerspectivesRequest(BaseModel):
    topic: str = Field(description="The problem or topic to analyze")
    context: Optional[str] = Field(
        default=None, description="Additional context about the problem"
    )
    existing_session_id: Optional[str] = Field(
        default=None, description="Session ID to add perspectives to"
    )
    max_recommendations: int = Field(
        default=5, description="Maximum number of recommendations"
    )


@mcp.tool(
    description="Get smart perspective recommendations based on problem analysis - AI suggests the most relevant expert viewpoints for your specific challenge"
)
async def recommend_perspectives(
    request: RecommendPerspectivesRequest,
) -> Dict[str, Any]:
    """Analyze problem and recommend relevant perspectives"""

    # Initialize perspective selector
    selector = SmartPerspectiveSelector()

    # Get existing perspectives if session provided
    existing_perspectives = []
    if request.existing_session_id:
        session = await session_manager.get_session(request.existing_session_id)
        if session:
            existing_perspectives = list(session.threads.keys())

    # Get recommendations
    recommendations = selector.recommend_perspectives(
        topic=request.topic,
        context=request.context,
        existing_perspectives=existing_perspectives,
        max_recommendations=request.max_recommendations,
    )

    # Build response
    recommended_perspectives = []
    for rec in recommendations:
        recommended_perspectives.append(
            {
                "name": rec.name,
                "description": rec.description,
                "relevance_score": round(rec.relevance_score, 2),
                "reasoning": rec.reasoning,
            }
        )

    # Generate key insight
    if recommendations:
        top_perspective = recommendations[0]
        key_insight = f"Detected {top_perspective.reasoning.lower()} - recommending {len(recommendations)} specialized perspectives"
    else:
        key_insight = "Standard perspectives recommended for general analysis"

    # Next steps
    next_steps = []
    if request.existing_session_id:
        for rec in recommendations[:2]:  # Top 2 recommendations
            next_steps.append(f"add_perspective('{rec.name}') - {rec.description}")
    else:
        next_steps.append("start_context_analysis() with recommended perspectives")
        next_steps.append(
            "Use 'initial_perspectives' parameter with recommendation names"
        )

    # Build AORP response
    builder = AORPBuilder()
    aorp_response = (
        builder.status("success")
        .key_insight(key_insight)
        .confidence(
            max(0.5, recommendations[0].relevance_score if recommendations else 0.5)
        )
        .session_id(request.existing_session_id)
        .next_steps(next_steps)
        .primary_recommendation(
            f"Add '{recommendations[0].name}' perspective for targeted insights"
            if recommendations and request.existing_session_id
            else "Start analysis with recommended domain-specific perspectives"
        )
        .workflow_guidance(
            "Present perspective recommendations with reasoning to guide user's choice"
        )
        .completeness(1.0)
        .reliability(0.8)  # Recommendations are heuristic-based
        .urgency("low")
        .indicators(
            recommendations_count=len(recommendations),
            existing_perspectives=len(existing_perspectives),
            problem_domains_detected=len(
                [r for r in recommendations if r.relevance_score > 0.5]
            ),
        )
        .summary(
            f"Recommended {len(recommendations)} perspectives based on problem analysis"
        )
        .data(
            {
                "recommendations": recommended_perspectives,
                "topic": request.topic,
                "existing_perspectives": existing_perspectives,
            }
        )
        .metadata(
            operation_type="perspective_recommendation",
            has_context=bool(request.context),
        )
        .build()
    )

    return aorp_response


# Note: Session cleanup task should be started manually when needed
# FastMCP doesn't have startup/shutdown hooks in this version


# Main entry point
def main():
    """Run the MCP server"""
    try:
        # Start session cleanup task
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(session_manager.start_cleanup_task())
        logger.info("Started session cleanup task")

        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except (BrokenPipeError, ConnectionResetError):
        # Normal disconnection from client
        logger.debug("Client disconnected normally")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        # Stop cleanup task on shutdown
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(session_manager.stop_cleanup_task())
            logger.info("Stopped session cleanup task")
        except Exception as e:
            logger.error(f"Error stopping cleanup task: {e}")


if __name__ == "__main__":
    main()
