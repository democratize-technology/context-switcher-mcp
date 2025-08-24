"""Validation utilities for Context-Switcher MCP Server"""

from .client_binding import validate_session_access
from .logging_base import get_logger
from .security import sanitize_error_message

logger = get_logger(__name__)


def _get_config():
    """Lazy config loading to avoid module-level import issues"""
    try:
        from .config import get_config

        return get_config()
    except ImportError as e:
        logger.error(f"Failed to load config: {e}")

        # Fallback configuration for validation
        class FallbackValidation:
            max_topic_length = 1000
            max_session_id_length = 100

        class FallbackConfig:
            validation = FallbackValidation()

        return FallbackConfig()


def validate_topic(topic: str) -> tuple[bool, str]:
    """Validate analysis topic

    Args:
        topic: The topic string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not topic or not isinstance(topic, str):
        return False, "Topic must be a non-empty string"

    if len(topic.strip()) == 0:
        return False, "Topic cannot be empty or whitespace only"

    try:
        config = _get_config()
        max_length = config.validation.max_topic_length
    except Exception:
        # Fallback to default if config loading fails
        max_length = 1000  # Default maximum length

    if len(topic) > max_length:
        return (
            False,
            f"Topic must be {max_length} characters or less",
        )

    # Check for suspicious patterns
    suspicious_patterns = [
        "<?php",
        "<script",
        "javascript:",
        "data:",
        "vbscript:",
        "onmouseover=",
        "onclick=",
        "onerror=",
        "onload=",
        "eval(",
        "alert(",
        "prompt(",
        "confirm(",
        "document.cookie",
        "localStorage",
        "sessionStorage",
    ]

    topic_lower = topic.lower()
    for pattern in suspicious_patterns:
        if pattern in topic_lower:
            return False, f"Topic contains suspicious pattern: {pattern}"

    return True, ""


async def validate_session_id(session_id: str, operation: str) -> tuple[bool, str]:
    """Validate session ID and access permissions

    Args:
        session_id: The session ID to validate
        operation: The operation being performed (for logging)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Basic validation
    if not session_id or not isinstance(session_id, str):
        return False, "Session ID must be a non-empty string"

    config = _get_config()
    if len(session_id) > config.validation.max_session_id_length:
        return (
            False,
            f"Session ID must be {config.validation.max_session_id_length} characters or less",
        )

    # Get session object for client binding validation
    from . import session_manager as sm_global

    # Handle test contexts where session_manager might not be initialized
    if sm_global is None:
        return False, f"Session '{session_id}' not found or expired"

    try:
        session = await sm_global.get_session(session_id)
        if session is None:
            return False, f"Session '{session_id}' not found or expired"
    except Exception:
        # Handle any session-related errors (including SessionNotFoundError)
        return False, f"Session '{session_id}' not found or expired"

    # Client binding validation
    access_valid, access_error = await validate_session_access(session, operation)
    if not access_valid:
        # Check if this is specifically a rate limiting issue
        if "excessive_access_rate" in access_error:
            # Return a clear rate limiting message instead of misleading "session not found"
            return (
                False,
                "Access temporarily restricted due to rate limiting - please wait before trying again",
            )
        # For other validation failures, use sanitized error
        return False, sanitize_error_message(access_error)

    return True, ""
