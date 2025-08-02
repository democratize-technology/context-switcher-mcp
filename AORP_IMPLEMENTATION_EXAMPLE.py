"""
AORP Implementation Example: analyze_from_perspectives Tool Refactoring

This demonstrates how to transform a traditional MCP tool response
into an AI-Optimized Response Protocol (AORP) response.
"""

from datetime import datetime
from typing import Dict, Any
from .aorp import (
    AORPBuilder,
    calculate_analysis_confidence,
    generate_analysis_next_steps,
    create_error_response,
)
from .models import ContextSwitcherSession
from .orchestrator import NO_RESPONSE


# BEFORE: Traditional response format
def analyze_from_perspectives_legacy(
    request, session: ContextSwitcherSession, orchestrator
) -> Dict[str, Any]:
    """Legacy implementation with nested data structure"""

    # Broadcast to all threads
    responses = await orchestrator.broadcast_message(
        session.threads, request.prompt, session_id=request.session_id
    )

    # Process responses
    active_perspectives = {}
    abstained_perspectives = []
    errors = []

    for name, response in responses.items():
        if response.startswith("ERROR:"):
            errors.append({name: response})
        elif NO_RESPONSE in response:
            abstained_perspectives.append(name)
        else:
            active_perspectives[name] = response

    # Store analysis
    analysis = {
        "prompt": request.prompt,
        "timestamp": datetime.utcnow().isoformat(),
        "responses": responses,
        "active_count": len(active_perspectives),
        "abstained_count": len(abstained_perspectives),
    }
    session.analyses.append(analysis)

    # Traditional nested response - requires 3-4 levels of traversal
    return {
        "session_id": request.session_id,
        "prompt": request.prompt,
        "perspectives": active_perspectives,  # KEY DATA BURIED HERE
        "abstained": abstained_perspectives,
        "errors": errors,
        "summary": {  # SUMMARY SEPARATE FROM DATA
            "total_perspectives": len(session.threads),
            "active_responses": len(active_perspectives),
            "abstentions": len(abstained_perspectives),
            "errors": len(errors),
        },
    }


# AFTER: AORP implementation
async def analyze_from_perspectives_aorp(
    request, session: ContextSwitcherSession, orchestrator
) -> Dict[str, Any]:
    """AORP implementation with AI-optimized response structure"""

    try:
        start_time = datetime.utcnow()

        # Broadcast to all threads
        responses = await orchestrator.broadcast_message(
            session.threads, request.prompt, session_id=request.session_id
        )

        # Process responses with quality assessment
        active_perspectives = {}
        abstained_perspectives = []
        errors = []
        response_lengths = []

        for name, response in responses.items():
            if response.startswith("ERROR:"):
                errors.append({name: response})
            elif NO_RESPONSE in response:
                abstained_perspectives.append(name)
            else:
                active_perspectives[name] = response
                response_lengths.append(len(response))

        # Calculate quality metrics
        total_perspectives = len(session.threads)
        active_count = len(active_perspectives)
        error_count = len(errors)
        abstention_count = len(abstained_perspectives)

        confidence = calculate_analysis_confidence(
            active_count,
            total_perspectives,
            error_count,
            abstention_count,
            response_lengths,
        )

        # Determine status
        if error_count > 0 and active_count == 0:
            status = "error"
        elif error_count > 0 or abstention_count > total_perspectives // 2:
            status = "partial"
        else:
            status = "success"

        # Generate key insight (front-loaded for immediate understanding)
        if active_count == 0:
            key_insight = (
                "No perspectives provided analysis - question may be outside scope"
            )
        elif confidence >= 0.8:
            key_insight = f"Strong consensus from {active_count} perspectives with high-quality insights"
        elif confidence >= 0.5:
            key_insight = f"Moderate agreement from {active_count} perspectives with mixed quality"
        else:
            key_insight = f"Limited insights from {active_count} perspectives - consider refining question"

        # Determine urgency based on findings
        urgency = "high" if error_count > total_perspectives // 2 else "medium"
        if confidence >= 0.8 and active_count >= 3:
            urgency = "low"  # Good results, no rush

        # Store analysis with enhanced metadata
        analysis = {
            "prompt": request.prompt,
            "timestamp": start_time.isoformat(),
            "responses": responses,
            "active_count": active_count,
            "abstained_count": abstention_count,
            "confidence": confidence,
        }
        session.analyses.append(analysis)

        # Calculate completeness and reliability
        completeness = active_count / total_perspectives
        reliability = confidence

        # Generate next steps based on results
        has_synthesis = len(session.analyses) > 1  # Estimate if synthesis is available
        next_steps = generate_analysis_next_steps(
            "active", total_perspectives, error_count, has_synthesis, confidence
        )

        # Prioritize perspectives by impact/quality for progressive disclosure
        high_impact_perspectives = {}
        medium_impact_perspectives = {}
        supporting_perspectives = {}

        # Sort by response length as a proxy for depth/quality
        sorted_perspectives = sorted(
            active_perspectives.items(), key=lambda x: len(x[1]), reverse=True
        )

        for i, (name, response) in enumerate(sorted_perspectives):
            if i < 2:  # Top 2 are high impact
                high_impact_perspectives[name] = response
            elif i < 4:  # Next 2 are medium impact
                medium_impact_perspectives[name] = response
            else:  # Rest are supporting
                supporting_perspectives[name] = response

        # Generate workflow guidance
        if confidence >= 0.8:
            workflow_guidance = "Present key insights confidently, offer synthesis for strategic patterns"
        elif confidence >= 0.5:
            workflow_guidance = "Present insights with caveats, suggest adding perspectives for completeness"
        else:
            workflow_guidance = "Present limited insights, strongly recommend refining question or adding perspectives"

        # Build AORP response
        builder = AORPBuilder()

        operation_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return (
            builder.status(status)
            .key_insight(key_insight)
            .confidence(confidence)
            .session_id(request.session_id)
            .next_steps(next_steps)
            .primary_recommendation(
                "Synthesize perspectives for strategic insights"
                if confidence >= 0.6
                else "Add more perspectives or refine question"
            )
            .secondary_recommendations(
                [
                    "Explore specific concerns in detail",
                    "Consider domain-specific perspectives",
                ]
            )
            .workflow_guidance(workflow_guidance)
            .completeness(completeness)
            .reliability(reliability)
            .urgency(urgency)
            .indicators(
                total_perspectives=total_perspectives,
                active_responses=active_count,
                abstentions=abstention_count,
                errors=error_count,
                consensus_level=confidence,
                avg_response_length=sum(response_lengths) / len(response_lengths)
                if response_lengths
                else 0,
            )
            .summary(
                f"Analysis of '{request.prompt}' yielded {active_count} perspective responses "
                f"with {confidence:.1%} confidence. "
                f"{'Strong consensus indicates clear direction.' if confidence >= 0.8 else 'Mixed signals suggest need for deeper exploration.'}"
            )
            .data(
                {
                    # Progressive disclosure: most important first
                    "perspectives": {
                        "high_impact": high_impact_perspectives,
                        "medium_impact": medium_impact_perspectives,
                        "supporting": supporting_perspectives,
                    },
                    "abstained": abstained_perspectives,
                    "errors": errors,
                    # Backward compatibility
                    "legacy_format": {
                        "session_id": request.session_id,
                        "prompt": request.prompt,
                        "perspectives": active_perspectives,
                        "abstained": abstained_perspectives,
                        "errors": errors,
                        "summary": {
                            "total_perspectives": total_perspectives,
                            "active_responses": active_count,
                            "abstentions": abstention_count,
                            "errors": error_count,
                        },
                    },
                }
            )
            .metadata(
                prompt=request.prompt,
                operation_time_ms=operation_time,
                analysis_index=len(session.analyses),
            )
            .build()
        )

    except Exception as e:
        # Error handling with AORP
        return create_error_response(
            f"Analysis failed: {str(e)}",
            "analysis_error",
            {"session_id": request.session_id, "prompt": request.prompt},
            recoverable=True,
            session_id=request.session_id,
        )


# COMPARISON: How AI assistants would process these responses


def ai_assistant_processing_comparison():
    """
    Demonstrates the difference in cognitive load for AI assistants
    """

    # LEGACY RESPONSE PROCESSING (Complex traversal required)
    def process_legacy_response(response):
        """AI assistant needs multiple steps to extract value"""

        # Step 1: Check if request succeeded (nested in structure)
        if "error" in response:
            return f"Error: {response['error']}"

        # Step 2: Navigate to summary for counts (separate from data)
        summary = response.get("summary", {})
        active_count = summary.get("active_responses", 0)
        total_count = summary.get("total_perspectives", 0)

        # Step 3: Check if any responses exist (nested again)
        perspectives = response.get("perspectives", {})
        if not perspectives:
            return "No insights available"

        # Step 4: Extract actual insights (3 levels deep)
        insights = []
        for name, insight in perspectives.items():
            insights.append(f"{name}: {insight[:100]}...")

        # Step 5: Determine confidence (manual assessment)
        confidence = "unknown"
        if active_count == total_count:
            confidence = "high"
        elif active_count >= total_count * 0.7:
            confidence = "medium"
        else:
            confidence = "low"

        # Step 6: Decide next actions (manual logic)
        next_action = "Continue analysis" if confidence != "low" else "Improve question"

        return f"Found {active_count} insights with {confidence} confidence. {next_action}."

    # AORP RESPONSE PROCESSING (Immediate value extraction)
    def process_aorp_response(response):
        """AI assistant gets immediate actionable intelligence"""

        # Step 1: Immediate status check (front-loaded)
        immediate = response["immediate"]
        if immediate["status"] == "error":
            return f"Error: {immediate['key_insight']}"

        # Step 2: Get key insight (pre-processed, one sentence)
        key_insight = immediate["key_insight"]
        confidence = immediate["confidence"]

        # Step 3: Get next action (pre-calculated)
        next_steps = response["actionable"]["next_steps"]
        primary_action = next_steps[0] if next_steps else "Continue"

        # Step 4: Get workflow guidance (tells AI how to proceed)
        guidance = response["actionable"]["workflow_guidance"]

        # Done! AI has everything needed in 4 simple steps vs 6 complex ones
        return f"{key_insight} (Confidence: {confidence:.1%}). Action: {primary_action}. Guidance: {guidance}"

    # EXAMPLE RESPONSES
    legacy_response = {
        "session_id": "abc123",
        "prompt": "Should we implement microservices?",
        "perspectives": {
            "technical": "Microservices add complexity but improve scalability...",
            "business": "Cost increase initially but long-term flexibility...",
            "risk": "Increased failure points and monitoring complexity...",
        },
        "abstained": [],
        "errors": [],
        "summary": {
            "total_perspectives": 4,
            "active_responses": 3,
            "abstentions": 0,
            "errors": 0,
        },
    }

    aorp_response = {
        "immediate": {
            "status": "success",
            "key_insight": "Strong consensus from 3 perspectives with high-quality insights on microservices trade-offs",
            "confidence": 0.85,
            "session_id": "abc123",
        },
        "actionable": {
            "next_steps": [
                "synthesize_perspectives() - Discover strategic patterns",
                "add_perspective('operations') - Cover deployment concerns",
                "analyze_from_perspectives('implementation timeline') - Explore execution",
            ],
            "recommendations": {
                "primary": "Synthesize perspectives for strategic decision framework",
                "secondary": [
                    "Consider operational perspective",
                    "Explore phased approach",
                ],
            },
            "workflow_guidance": "Present key insights confidently, offer synthesis for strategic patterns",
        },
        "quality": {
            "completeness": 0.75,
            "reliability": 0.85,
            "urgency": "medium",
            "indicators": {
                "total_perspectives": 4,
                "active_responses": 3,
                "consensus_level": 0.85,
            },
        },
        "details": {
            "summary": "Analysis of microservices adoption yielded 3 perspective responses with 85% confidence...",
            "data": {
                "perspectives": {
                    "high_impact": {},
                    "medium_impact": {},
                    "supporting": {},
                }
            },
            "metadata": {
                "prompt": "Should we implement microservices?",
                "operation_time_ms": 15420,
            },
        },
    }

    print("LEGACY PROCESSING:")
    print(process_legacy_response(legacy_response))
    print("\nAORP PROCESSING:")
    print(process_aorp_response(aorp_response))


# BENEFITS DEMONSTRATED

"""
COGNITIVE LOAD REDUCTION:
- Legacy: 6 complex steps with manual confidence assessment
- AORP: 4 simple steps with pre-calculated intelligence

DECISION SPEED:
- Legacy: AI must analyze data to determine next action
- AORP: Next actions provided immediately with rationale

WORKFLOW INTEGRATION:
- Legacy: No guidance on how to present information
- AORP: Explicit workflow guidance for user interaction

PROGRESSIVE DISCLOSURE:
- Legacy: All data at same level, requires filtering
- AORP: High-impact insights prioritized, details available

QUALITY TRANSPARENCY:
- Legacy: No indication of result quality or reliability
- AORP: Confidence scores and quality indicators throughout

ERROR HANDLING:
- Legacy: Basic error messages without recovery guidance
- AORP: Structured error responses with recovery steps

The transformation from legacy to AORP represents a fundamental shift
from "data provider" to "AI workflow enabler" - optimizing for the
AI assistant's decision-making process rather than just human readability.
"""
