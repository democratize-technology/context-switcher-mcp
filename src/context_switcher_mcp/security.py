"""Security utilities for input sanitization and validation"""

import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Security patterns to detect and block
SECURITY_PATTERNS = [
    # Script injection patterns
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    # System command injection patterns
    r"\$\([^)]*\)",
    r"`[^`]*`",
    r";\s*(rm|del|format|sudo|su)\s",
    r"\|\s*(curl|wget|nc|netcat)\s",
    # Path traversal patterns
    r"\.\./+",
    r"\.\.\\+",
    # SQL injection patterns (basic)
    r";\s*(drop|delete|insert|update|select)\s+",
    r"union\s+select",
    r"or\s+1\s*=\s*1",
    # Prompt injection patterns
    r"ignore\s+(previous|all|above)\s+(instructions|prompts?)",
    r"system\s*:\s*you\s+are\s+now",
    r"new\s+instructions?\s*:",
    r"act\s+as\s+(if\s+you\s+are\s+)?a\s+different",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in SECURITY_PATTERNS
]

# Allowed model ID patterns
ALLOWED_MODEL_PATTERNS = [
    # AWS Bedrock patterns
    r"^us\.anthropic\.claude-\d+-\d*-?\w*-\d{8}-v\d+:\d+$",
    r"^anthropic\.claude-\d+-\w+-\d{8}-v\d+:\d+$",
    r"^anthropic\.claude-v\d+$",
    # OpenAI patterns
    r"^gpt-[34](\.\d+)?(-turbo)?(-\d{4})?$",
    r"^text-davinci-\d{3}$",
    # Local model patterns
    r"^llama\d*(\.\d+)?$",
    r"^mistral(\d+)?(\.\d+)?$",
    r"^codellama(\d+)?(\.\d+)?$",
    # Generic patterns for development
    r"^[a-zA-Z0-9\-_.]+$",  # Allow basic alphanumeric with common separators
]

COMPILED_MODEL_PATTERNS = [re.compile(pattern) for pattern in ALLOWED_MODEL_PATTERNS]


def sanitize_user_input(
    text: str, max_length: int = 10000
) -> Tuple[bool, str, List[str]]:
    """
    Sanitize user input for security issues

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Tuple of (is_safe, cleaned_text, issues_found)
    """
    if not text or not isinstance(text, str):
        return False, "", ["Invalid input type"]

    # Check length
    if len(text) > max_length:
        return False, text[:max_length], [f"Input too long (max {max_length} chars)"]

    issues = []

    # Check for security patterns
    for i, pattern in enumerate(COMPILED_PATTERNS):
        if pattern.search(text):
            issues.append(f"Security pattern detected: {SECURITY_PATTERNS[i][:30]}...")
            logger.warning(f"Blocked potentially malicious input: pattern {i}")

    # Basic character filtering
    # Remove null bytes and control characters except newlines/tabs
    cleaned_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Limit excessive whitespace
    cleaned_text = re.sub(r"\s{100,}", " " * 50, cleaned_text)

    is_safe = len(issues) == 0

    return is_safe, cleaned_text, issues


def validate_model_id(model_id: str) -> Tuple[bool, str]:
    """
    Validate model ID against allowed patterns

    Args:
        model_id: Model ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_id or not isinstance(model_id, str):
        return False, "Model ID must be a non-empty string"

    if len(model_id) > 200:
        return False, "Model ID too long (max 200 characters)"

    # Check against allowed patterns
    for pattern in COMPILED_MODEL_PATTERNS:
        if pattern.match(model_id):
            return True, ""

    logger.warning(f"Rejected invalid model ID: {model_id}")
    return False, f"Model ID '{model_id}' not in allowed patterns"


def log_security_event(event_type: str, details: dict, session_id: str = None):
    """
    Log security-related events for monitoring

    Args:
        event_type: Type of security event
        details: Event details
        session_id: Associated session ID if any
    """
    log_entry = {
        "event_type": event_type,
        "session_id": session_id,
        "details": details,
    }

    # Use structured logging for security events
    logger.warning(f"SECURITY_EVENT: {log_entry}")


def sanitize_error_message(error_msg: str) -> str:
    """
    Sanitize error messages to remove internal details

    Args:
        error_msg: Original error message

    Returns:
        Sanitized error message safe for users
    """
    # Patterns that indicate internal details to hide
    internal_patterns = [
        (r'File ".*?"', "internal file"),
        (r"line \d+", "internal location"),
        (r"/[a-zA-Z0-9/._-]*\.py", "internal file"),
        (r"Traceback \(most recent call last\).*", "Internal error occurred"),
        (r"boto3\..*", "AWS service error"),
        (r"psycopg2\..*", "Database error"),
        (r"requests\.exceptions\..*", "Network error"),
        (r"ImportError:.*", "Configuration error"),
        (r"ModuleNotFoundError:.*", "Configuration error"),
    ]

    sanitized = error_msg

    for pattern, replacement in internal_patterns:
        sanitized = re.sub(
            pattern, replacement, sanitized, flags=re.MULTILINE | re.DOTALL
        )

    # Limit error message length
    if len(sanitized) > 500:
        sanitized = sanitized[:497] + "..."

    return sanitized
