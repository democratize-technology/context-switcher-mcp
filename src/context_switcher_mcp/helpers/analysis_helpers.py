"""Helper functions for analysis operations"""

from ..logging_config import get_logger
from typing import Dict, Any, Tuple
from datetime import datetime

from ..aorp import AORPBuilder, create_error_response, generate_analysis_next_steps
from ..constants import NO_RESPONSE
from ..rate_limiter import SessionRateLimiter
from ..security import (
    log_security_event,
    sanitize_error_message,
    validate_analysis_prompt,
)
from ..validation import validate_session_id

logger = get_logger(__name__)


async def validate_analysis_request(
    session_id: str, prompt: str, rate_limiter: SessionRateLimiter
) -> Tuple[bool, Dict[str, Any]]:
    """Validate analysis request including rate limits, session, and prompt

    Args:
        session_id: Session ID to validate
        prompt: Analysis prompt to validate
        rate_limiter: Rate limiter instance

    Returns:
        Tuple of (is_valid, error_response_or_none)
    """
    allowed, rate_error = rate_limiter.check_request(session_id, "analysis")
    if not allowed:
        return False, create_error_response(
            rate_error,
            "rate_limited",
            {
                "session_id": session_id,
                "retry_after_seconds": 6,
            },  # 60/10 = 6 seconds
            recoverable=True,
        )

    # Validate session ID with client binding
    session_valid, session_error = await validate_session_id(
        session_id, "analyze_from_perspectives"
    )
    if not session_valid:
        # Log failed session access for security monitoring
        log_security_event(
            "session_access_failed",
            {"attempted_session_id": session_id, "error": session_error},
            session_id,
        )
        return False, create_error_response(
            session_error,
            "session_not_found",
            {"session_id": session_id},
            recoverable=True,
        )

    # Enhanced validation for analysis prompt
    validation_result = validate_analysis_prompt(prompt, session_id)
    if not validation_result.is_valid:
        log_security_event(
            "analysis_prompt_validation_failure",
            {
                "session_id": session_id,
                "prompt_preview": prompt[:100],  # First 100 chars for context
                "issues": validation_result.issues,
                "risk_level": validation_result.risk_level,
                "blocked_patterns": len(validation_result.blocked_patterns),
            },
            session_id=session_id,
        )
        return False, create_error_response(
            f"Invalid analysis prompt: {sanitize_error_message(validation_result.issues[0] if validation_result.issues else 'Security check failed')}",
            "validation_error",
            {
                "session_id": session_id,
                "validation_issues": len(validation_result.issues),
                "risk_level": validation_result.risk_level,
            },
            recoverable=True,
            session_id=session_id,
        )

    return True, {}


def build_analysis_aorp_response(
    prompt: str, results: Any, session: Any, confidence: float
) -> Dict[str, Any]:
    """Build AORP response for analysis results

    Args:
        prompt: Original analysis prompt
        results: Analysis results from orchestrator
        session: Session object
        confidence: Confidence score

    Returns:
        AORP formatted response
    """
    # Generate next steps
    next_steps = generate_analysis_next_steps(
        session_state="active",
        perspectives_count=results.active_count + results.abstained_count,
        error_count=len(results.model_errors),
        has_synthesis=False,
        confidence=confidence,
    )

    # Build AORP response
    builder = AORPBuilder()

    # Set immediate values
    builder.status("success" if len(results.model_errors) == 0 else "partial")

    # Build key insight with convergence awareness
    base_insight = f"Analysis complete: {results.active_count} active perspectives, {results.abstained_count} abstained"
    if hasattr(results, "convergence") and results.convergence:
        if results.convergence.get("converged", False):
            alignment_score = results.convergence.get("alignment_score", 0.0)
            base_insight += f" - Contexts converged at {alignment_score:.3f} alignment"
        else:
            alignment_score = results.convergence.get("alignment_score", 0.0)
            base_insight += (
                f" - Contexts remain diverse ({alignment_score:.3f} alignment)"
            )

    builder.key_insight(base_insight)
    builder.confidence(confidence)

    # Add perspective responses as structured data
    perspectives_data = {}
    for response in results.responses:
        if response["content"] and response["content"] != NO_RESPONSE:
            perspectives_data[response["perspective"]] = response["content"]

    builder.summary(f"Multi-perspective analysis of: {prompt[:100]}...")

    # Prepare data with convergence information
    data_dict = {
        "perspectives": perspectives_data,
        "metrics": {
            "active_count": results.active_count,
            "abstained_count": results.abstained_count,
            "error_count": len(results.model_errors),
            "confidence": confidence,
        },
    }

    # Add convergence information if available
    if hasattr(results, "convergence") and results.convergence:
        data_dict["convergence"] = {
            "converged": results.convergence.get("converged", False),
            "alignment_score": results.convergence.get("alignment_score", 0.0),
            "message": results.convergence.get("message", "No convergence data"),
            "threshold": 0.85,  # Our moderate threshold
        }

    builder.data(data_dict)

    # Add abstained perspectives info
    abstained_perspectives = [
        r["perspective"] for r in results.responses if r["content"] == NO_RESPONSE
    ]

    # Set actionable information
    builder.next_steps(next_steps)

    # Make recommendations convergence-aware
    if (
        hasattr(results, "convergence")
        and results.convergence
        and results.convergence.get("converged", False)
    ):
        builder.primary_recommendation(
            "Contexts have converged - consider synthesis or move to decision-making"
        )
        builder.workflow_guidance(
            "Convergence detected: perspectives are aligning. Consider synthesizing insights or proceeding with consolidated approach"
        )
    else:
        builder.primary_recommendation(
            "Review perspective insights and consider synthesis for deeper analysis"
        )
        builder.workflow_guidance(
            "Present perspectives to user, highlight areas of agreement and tension"
        )

    # Set quality metrics
    total_perspectives = results.active_count + results.abstained_count
    builder.completeness(
        results.active_count / total_perspectives if total_perspectives > 0 else 0
    )
    builder.reliability(confidence)
    builder.urgency("medium" if results.active_count > 0 else "low")

    # Add indicators
    builder.indicators(
        active_perspectives=results.active_count,
        abstained_perspectives=results.abstained_count,
        error_count=len(results.model_errors),
        has_abstentions=len(abstained_perspectives) > 0,
    )

    # Add metadata
    builder.metadata(
        execution_time=f"{results.execution_time:.2f}s",
        abstained_list=abstained_perspectives,
        model_errors=results.model_errors,
    )

    return builder.build()


def store_analysis_results(session: Any, prompt: str, results: Any) -> None:
    """Store analysis results in session

    Args:
        session: Session object to update
        prompt: Analysis prompt
        results: Analysis results from orchestrator
    """
    analysis_data = {
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "responses": results.responses,
        "active_count": results.active_count,
        "abstained_count": results.abstained_count,
        "execution_time": results.execution_time,
        "model_errors": results.model_errors,
    }

    session.analyses.append(analysis_data)
