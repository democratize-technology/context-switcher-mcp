"""Helper functions for analysis operations"""

import logging
from typing import Dict, Any, Tuple
from datetime import datetime

from ..aorp import AORPBuilder, create_error_response, generate_analysis_next_steps
from ..orchestrator import NO_RESPONSE
from ..rate_limiter import SessionRateLimiter
from ..security import (
    log_security_event,
    sanitize_error_message,
    validate_analysis_prompt,
)
from ..validation import validate_session_id

logger = logging.getLogger(__name__)


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
        results.active_count, results.abstained_count, len(session.analyses)
    )

    # Build AORP response
    builder = AORPBuilder()
    builder.set_objective(f"Multi-perspective analysis: {prompt}")
    builder.set_confidence(confidence)

    # Add perspective responses as recommendations
    for response in results.responses:
        if response["content"] and response["content"] != NO_RESPONSE:
            builder.add_recommendation(
                f"{response['perspective']} Perspective",
                response["content"],
                confidence=0.8,  # Individual perspective confidence
            )

    # Add abstained perspectives as context
    abstained_perspectives = [
        r["perspective"] for r in results.responses if r["content"] == NO_RESPONSE
    ]
    if abstained_perspectives:
        builder.add_context(
            "abstained_perspectives",
            f"Perspectives that abstained: {', '.join(abstained_perspectives)}",
        )

    # Add execution metrics
    builder.add_context("execution_time", f"{results.execution_time:.2f}s")
    builder.add_context("perspectives_engaged", str(results.active_count))

    # Handle model errors
    if results.model_errors:
        for error in results.model_errors:
            builder.add_risk(
                f"Model Error ({error['perspective']})",
                f"Failed to get response: {sanitize_error_message(error['error'])}",
                impact="medium",
            )

    # Set next steps
    for step in next_steps:
        builder.add_next_step(step)

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
