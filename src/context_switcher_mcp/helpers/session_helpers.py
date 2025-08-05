"""Helper functions for session creation and management"""

import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4

from ..config import get_config
from ..models import ModelBackend, Thread
from ..client_binding import log_security_event_with_binding
from ..security import validate_user_content
from ..templates import PERSPECTIVE_TEMPLATES

logger = logging.getLogger(__name__)

config = get_config()

# Default perspectives (extracted from session_handler)
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
- Quantify business value and financial implications where possible
- Consider market dynamics and competitive landscape
- Evaluate strategic fit with organizational goals
- Assess resource requirements and opportunity costs
- Focus on measurable business outcomes

**Abstention criteria:** Return [NO_RESPONSE] if the topic has no business strategy, financial, or market implications.""",
    "user": """You are a user experience researcher. Analyze from an end-user perspective and usability standpoint.

**Focus areas:**
- User journey mapping and experience flow analysis
- Accessibility considerations and inclusive design principles
- Usability heuristics and interaction design patterns
- User adoption barriers and engagement factors
- Interface design and information architecture

**Analysis approach:**
- Apply established UX research methodologies and frameworks
- Consider diverse user personas and accessibility needs
- Evaluate cognitive load and user mental models
- Identify friction points and optimization opportunities
- Focus on measurable user outcomes and satisfaction metrics

**Abstention criteria:** Return [NO_RESPONSE] if the topic has no direct user interface, experience, or human interaction components.""",
    "risk": """You are a risk management specialist. Evaluate potential risks, vulnerabilities, and mitigation strategies.

**Focus areas:**
- Security vulnerabilities and threat vector analysis
- Operational risks and failure mode assessment
- Compliance and regulatory risk evaluation
- Financial and reputational risk implications
- Business continuity and disaster recovery considerations

**Analysis approach:**
- Apply systematic risk assessment frameworks (likelihood × impact)
- Identify both obvious and hidden risk factors
- Evaluate cascading effects and systemic vulnerabilities
- Propose specific, actionable mitigation strategies
- Consider both preventive measures and incident response plans

**Abstention criteria:** Return [NO_RESPONSE] if the topic involves no identifiable risks, security concerns, or potential negative outcomes.""",
}


def determine_perspectives_to_use(
    template: Optional[str] = None, initial_perspectives: Optional[List[str]] = None
) -> Dict[str, str]:
    """Determine which perspectives to use based on template and initial_perspectives

    Args:
        template: Template name for pre-configured perspectives
        initial_perspectives: List of perspective names to use

    Returns:
        Dictionary mapping perspective names to their system prompts
    """
    perspectives_to_use = DEFAULT_PERSPECTIVES.copy()

    # Apply template if specified
    if template and template in PERSPECTIVE_TEMPLATES:
        template_config = PERSPECTIVE_TEMPLATES[template]
        perspectives_to_use = {}

        # Add template perspectives
        for perspective_name in template_config["perspectives"]:
            if perspective_name in DEFAULT_PERSPECTIVES:
                perspectives_to_use[perspective_name] = DEFAULT_PERSPECTIVES[
                    perspective_name
                ]

        # Add custom perspectives from template
        for custom_name, custom_prompt in template_config.get("custom", []):
            perspectives_to_use[custom_name] = custom_prompt

    # Override with initial_perspectives if provided
    elif initial_perspectives:
        perspectives_to_use = {}
        for perspective_name in initial_perspectives:
            if perspective_name in DEFAULT_PERSPECTIVES:
                perspectives_to_use[perspective_name] = DEFAULT_PERSPECTIVES[
                    perspective_name
                ]
            else:
                # Create a generic perspective for unknown names
                perspectives_to_use[perspective_name] = f"""You are a {perspective_name} expert. Analyze from a {perspective_name} perspective.

**Analysis approach:**
- Focus on {perspective_name}-specific considerations
- Provide actionable insights and recommendations
- Consider both immediate and long-term implications

**Abstention criteria:** Return [NO_RESPONSE] if the topic has no relevance to {perspective_name} considerations."""

    return perspectives_to_use


async def create_session_threads(
    session: Any,
    perspectives_to_use: Dict[str, str],
    model_backend: ModelBackend,
    model_name: Optional[str],
    session_id: str,
) -> None:
    """Create threads for each perspective in the session

    Args:
        session: Session object to add threads to
        perspectives_to_use: Dictionary of perspective names to system prompts
        model_backend: LLM backend to use
        model_name: Specific model name
        session_id: Session ID for logging
    """
    for perspective_name, system_prompt in perspectives_to_use.items():
        # Validate system prompt content
        validation_result = validate_user_content(
            system_prompt, "system_prompt", config.model.max_chars_opus
        )
        if not validation_result.is_valid:
            log_security_event_with_binding(
                "system_prompt_validation_failure",
                {
                    "perspective_name": perspective_name,
                    "issues": validation_result.issues,
                    "risk_level": validation_result.risk_level,
                    "blocked_patterns": len(validation_result.blocked_patterns),
                },
                session_id,
            )
            continue  # Skip this perspective

        # Create thread
        thread = Thread(
            id=str(uuid4()),
            system_prompt=system_prompt,
            model_backend=model_backend,
            model_name=model_name,
            temperature=0.7,
        )
        session.threads[perspective_name] = thread


def build_session_creation_response(
    session_id: str,
    topic: str,
    session: Any,
    template: Optional[str],
    model_backend: ModelBackend,
) -> Dict[str, Any]:
    """Build successful session creation response

    Args:
        session_id: Created session ID
        topic: Analysis topic
        session: Session object
        template: Template used (if any)
        model_backend: Model backend used

    Returns:
        Success response dictionary
    """
    return {
        "success": True,
        "session_id": session_id,
        "topic": topic,
        "created_at": session.created_at.isoformat(),
        "perspectives": list(session.threads.keys()),
        "total_perspectives": len(session.threads),
        "template_used": template,
        "model_backend": model_backend.value,
        "next_steps": [
            "analyze_from_perspectives - Ask a specific question",
            "add_perspective - Add custom viewpoints",
            "recommend_perspectives - Get AI suggestions for more perspectives",
        ],
    }
