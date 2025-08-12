"""Validation utilities for Context-Switcher MCP Server"""

import logging
from typing import Tuple

from .client_binding import validate_session_access
from .config import get_config
from .security import sanitize_error_message

logger = logging.getLogger(__name__)

config = get_config()


def validate_topic(topic: str) -> Tuple[bool, str]:
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

    if len(topic) > config.validation.max_topic_length:
        return (
            False,
            f"Topic must be {config.validation.max_topic_length} characters or less",
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


async def validate_session_id(session_id: str, operation: str) -> Tuple[bool, str]:
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

    if len(session_id) > config.validation.max_session_id_length:
        return (
            False,
            f"Session ID must be {config.validation.max_session_id_length} characters or less",
        )

    # Get session object for client binding validation
    from . import session_manager

    session = await session_manager.get_session(session_id)
    if session is None:
        return False, f"Session '{session_id}' not found or expired"

    # Client binding validation
    access_valid, access_error = await validate_session_access(session, operation)
    if not access_valid:
        return False, sanitize_error_message(access_error)

    return True, ""
