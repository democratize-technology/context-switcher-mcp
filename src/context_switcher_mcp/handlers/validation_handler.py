"""Validation handler for Context-Switcher MCP Server"""

from typing import Any

from ..error_helpers import rate_limit_error, validation_error
from ..logging_config import get_logger
from ..rate_limiter import SessionRateLimiter
from ..security import (
    log_security_event,
    validate_user_content,
)
from ..validation import validate_topic

logger = get_logger(__name__)

# Initialize rate limiter
rate_limiter = SessionRateLimiter()


class ValidationHandler:
    """Handler for input validation and security checks"""

    @staticmethod
    def validate_session_creation_request(
        topic: str, initial_perspectives=None
    ) -> tuple[bool, dict[str, Any]]:
        """Validate session creation request

        Args:
            topic: The analysis topic
            initial_perspectives: Optional list of perspective names

        Returns:
            Tuple of (is_valid, error_response_or_none)
        """
        allowed, rate_error = rate_limiter.check_session_creation()
        if not allowed:
            return False, rate_limit_error(
                rate_error,
                retry_after_seconds=12,  # 60/5 = 12 seconds between session creations
            )

        # Validate topic
        topic_valid, topic_error = validate_topic(topic)
        if not topic_valid:
            return False, validation_error(
                f"Invalid topic: {topic_error}",
                context={
                    "topic": topic,
                    "error_details": topic_error,
                },
            )

        # Validate initial_perspectives if provided
        if initial_perspectives:
            for perspective_name in initial_perspectives:
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
                    return False, validation_error(
                        f"Invalid perspective name: {validation_result.issues[0] if validation_result.issues else 'Security check failed'}",
                        context={
                            "perspective_name": perspective_name,
                            "validation_issues": len(validation_result.issues),
                            "risk_level": validation_result.risk_level,
                        },
                    )

        return True, None
