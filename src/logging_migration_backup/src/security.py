"""Security utilities for input sanitization and validation"""

import logging
from typing import Optional

# Import from focused modules
from .input_validators import (
    ValidationResult,
    InjectionAttempt,
    _validation_orchestrator,
    detect_advanced_prompt_injection,
    sanitize_user_input,
)
from .input_sanitizer import sanitize_for_llm, sanitize_error_message
from .model_validators import validate_model_id
from .security_events import log_security_event

logger = logging.getLogger(__name__)


# Re-export key classes and functions for backward compatibility
__all__ = [
    "ValidationResult",
    "InjectionAttempt",
    "validate_user_content",
    "sanitize_user_input",
    "sanitize_for_llm",
    "sanitize_error_message",
    "validate_model_id",
    "log_security_event",
    "detect_advanced_prompt_injection",
    "validate_perspective_data",
    "validate_analysis_prompt",
]


# Legacy function moved to input_validators module


def validate_user_content(
    content: str, content_type: str, max_length: int = 10000, client_id: str = "default"
) -> ValidationResult:
    """
    Comprehensive validation for all user content passed to LLMs

    Args:
        content: Input content to validate
        content_type: Type of content (topic, prompt, description, custom_prompt)
        max_length: Maximum allowed length
        client_id: Client identifier for rate limiting

    Returns:
        ValidationResult with validation details
    """
    return _validation_orchestrator.validate_content(
        content, content_type, max_length, client_id
    )


def validate_perspective_data(
    name: str, description: str, custom_prompt: Optional[str] = None
) -> ValidationResult:
    """
    Validate perspective-specific data

    Args:
        name: Perspective name
        description: Perspective description
        custom_prompt: Optional custom prompt

    Returns:
        ValidationResult for the perspective data
    """
    import re

    all_content = f"{name}\n{description}"
    if custom_prompt:
        all_content += f"\n{custom_prompt}"

    # Use stricter validation for perspective data
    result = validate_user_content(all_content, "perspective", max_length=5000)

    # Additional perspective-specific checks
    additional_issues = []

    # Check name length and format
    if len(name) > 100:
        additional_issues.append("Perspective name too long (max 100 chars)")

    if not re.match(r"^[a-zA-Z0-9\s_-]+$", name):
        additional_issues.append("Perspective name contains invalid characters")

    # Check description length
    if len(description) > 1000:
        additional_issues.append("Perspective description too long (max 1000 chars)")

    # Check custom prompt if provided
    if custom_prompt and len(custom_prompt) > 2000:
        additional_issues.append("Custom prompt too long (max 2000 chars)")

    if additional_issues:
        result.issues.extend(additional_issues)
        result.is_valid = False
        if result.risk_level == "low":
            result.risk_level = "medium"

    return result


def validate_analysis_prompt(
    prompt: str, session_context: Optional[str] = None
) -> ValidationResult:
    """
    Validate analysis prompts with context awareness

    Args:
        prompt: The analysis prompt
        session_context: Optional session context for validation

    Returns:
        ValidationResult for the analysis prompt
    """
    import re

    # Use enhanced validation for analysis prompts
    result = validate_user_content(prompt, "analysis_prompt", max_length=8000)

    # Additional analysis-specific checks
    additional_issues = []

    # Check for context manipulation attempts
    context_patterns = [
        r"switch\s+to\s+session",
        r"use\s+session\s+id",
        r"change\s+session",
        r"session\s*:\s*override",
    ]

    for pattern in context_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            additional_issues.append("Potential session manipulation attempt")
            result.risk_level = "high"
            break

    # Check for meta-analysis attempts (trying to analyze the system itself)
    meta_patterns = [
        r"analyze\s+(this\s+)?(system|server|mcp|the\s+server)",
        r"how\s+does\s+(this\s+)?(system|server|mcp)\s+(work|function)",
        r"show\s+me\s+the\s+(code|implementation)",
        r"tell\s+me\s+how\s+(this|it)\s+works",
        r"mcp\s+server\s+function",
        r"analyze\s+the\s+(server\s+)?architecture",
    ]

    for pattern in meta_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            additional_issues.append("Meta-analysis attempt detected")
            if result.risk_level == "low":
                result.risk_level = "medium"
            break

    if additional_issues:
        result.issues.extend(additional_issues)
        result.is_valid = False

    return result


# All functions are now imported from focused modules
# This file serves as the main security interface
