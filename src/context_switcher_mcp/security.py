"""Security utilities for input sanitization and validation"""

import re
import logging
import time
from typing import Tuple, List
from dataclasses import dataclass
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Enhanced security patterns to detect and block
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
    # Basic prompt injection patterns
    r"ignore\s+(previous|all|above)\s+(instructions|prompts?)",
    r"system\s*:\s*you\s+are\s+now",
    r"new\s+instructions?\s*:",
    r"act\s+as\s+(if\s+you\s+are\s+)?a\s+different",
]

# Advanced prompt injection patterns
ADVANCED_PROMPT_INJECTION_PATTERNS = [
    # Role manipulation
    r"you\s+are\s+now\s+(a|an)?\s*different\s+(ai|assistant|bot|system)",
    r"forget\s+(your|all|previous)\s+(instructions|training|guidelines)",
    r"override\s+(your|all|previous)\s+(instructions|training|guidelines)",
    r"switch\s+to\s+(unrestricted|developer|debug|admin)\s+mode",
    r"activate\s+(developer|debug|admin|unrestricted)\s+mode",
    # System prompt extraction
    r"show\s+me\s+your\s+(system|initial)\s+prompt",
    r"what\s+(are|were)\s+your\s+(initial|system)\s+(instructions|prompts?)",
    r"repeat\s+your\s+(system|initial)\s+(instructions|prompts?)",
    # Jailbreaking attempts
    r"dan\s+mode",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"unrestricted\s+ai",
    r"bypass\s+(safety|restrictions|guidelines)",
    # Context switching
    r"new\s+conversation\s+starts?\s+now",
    r"forget\s+everything\s+(above|before|prior)",
    r"start\s+over\s+with\s+new\s+(instructions|rules)",
    # Command injection in AI context
    r"execute\s+(this|the\s+following)\s+(command|code)",
    r"run\s+(this|the\s+following)\s+(command|code|script)",
    r"perform\s+the\s+following\s+action",
    # Multi-turn injection attempts
    r"in\s+your\s+next\s+response,\s+(ignore|forget)",
    r"from\s+now\s+on,\s+(ignore|bypass|forget)",
    r"for\s+all\s+future\s+responses,\s+(ignore|bypass)",
    # Unicode and encoding bypasses (basic detection)
    r"[\u200b-\u200f\u2060\ufeff]",  # Zero-width characters
    r"&#x?[0-9a-fA-F]+;",  # HTML entities
    r"%[0-9a-fA-F]{2}",  # URL encoding
    # Conversation hijacking
    r"human:\s*",
    r"user:\s*",
    r"assistant:\s*",
    r"ai:\s*",
    r"system:\s*",
    # Template injection
    r"\{\{.*?\}\}",
    r"\$\{.*?\}",
    r"<%.*?%>",
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in SECURITY_PATTERNS
]

COMPILED_ADVANCED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in ADVANCED_PROMPT_INJECTION_PATTERNS
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


@dataclass
class ValidationResult:
    """Result of content validation"""

    is_valid: bool
    cleaned_content: str
    issues: List[str]
    risk_level: str = "low"  # low, medium, high, critical
    blocked_patterns: List[str] = None

    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = []


@dataclass
class InjectionAttempt:
    """Details of a detected injection attempt"""

    pattern: str
    location: int
    severity: str
    description: str


class ContentValidator:
    """Advanced content validation with rate limiting and context awareness"""

    def __init__(self):
        # Rate limiting: track validation attempts per content type
        self.validation_attempts = defaultdict(lambda: deque(maxlen=100))
        self.blocked_attempts = defaultdict(int)

    def _check_rate_limit(
        self, content_type: str, client_id: str = "default"
    ) -> Tuple[bool, str]:
        """Check if validation requests are within rate limits"""
        key = f"{client_id}:{content_type}"
        now = time.time()

        # Clean old entries (older than 1 minute)
        attempts = self.validation_attempts[key]
        while attempts and attempts[0] < now - 60:
            attempts.popleft()

        # Check if too many attempts
        if len(attempts) >= 50:  # 50 validation attempts per minute
            return False, "Too many validation requests. Please slow down."

        attempts.append(now)
        return True, ""

    def validate_content_structure(self, content: str) -> Tuple[bool, List[str]]:
        """Validate content structure for suspicious patterns"""
        issues = []

        # Check for excessive nesting levels
        open_braces = content.count("{")
        close_braces = content.count("}")
        if abs(open_braces - close_braces) > 5:
            issues.append("Unbalanced braces detected")

        # Check for excessive repeated characters
        if re.search(r"(.)\1{50,}", content):
            issues.append("Excessive character repetition detected")

        # Check for suspicious character sequences
        if re.search(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", content):
            issues.append("Control characters detected")

        # Check for potential binary data
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            issues.append("Invalid character encoding")

        return len(issues) == 0, issues


# Global validator instance
_content_validator = ContentValidator()


def sanitize_user_input(
    text: str, max_length: int = 10000
) -> Tuple[bool, str, List[str]]:
    """
    Basic sanitize user input for security issues (legacy function)
    Use validate_user_content() for enhanced validation

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Tuple of (is_safe, cleaned_text, issues_found)
    """
    result = validate_user_content(text, "general", max_length)
    return result.is_valid, result.cleaned_content, result.issues


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
    if not content or not isinstance(content, str):
        return ValidationResult(
            is_valid=False,
            cleaned_content="",
            issues=["Invalid input type"],
            risk_level="high",
        )

    # Check rate limits
    rate_ok, rate_error = _content_validator._check_rate_limit(content_type, client_id)
    if not rate_ok:
        return ValidationResult(
            is_valid=False, cleaned_content="", issues=[rate_error], risk_level="medium"
        )

    # Check length
    if len(content) > max_length:
        return ValidationResult(
            is_valid=False,
            cleaned_content=content[:max_length],
            issues=[f"Input too long (max {max_length} chars)"],
            risk_level="medium",
        )

    issues = []
    blocked_patterns = []
    risk_level = "low"

    # Check for basic security patterns
    for i, pattern in enumerate(COMPILED_PATTERNS):
        if pattern.search(content):
            pattern_desc = SECURITY_PATTERNS[i][:30] + "..."
            issues.append(f"Security pattern detected: {pattern_desc}")
            blocked_patterns.append(SECURITY_PATTERNS[i])
            risk_level = "high"
            logger.warning(f"Blocked malicious input: pattern {i} in {content_type}")

    # Check for advanced prompt injection patterns
    injection_attempts = detect_advanced_prompt_injection(content)
    for attempt in injection_attempts:
        issues.append(f"Prompt injection detected: {attempt.description}")
        blocked_patterns.append(attempt.pattern)
        if attempt.severity == "critical":
            risk_level = "critical"
        elif attempt.severity == "high" and risk_level not in ["critical"]:
            risk_level = "high"
        elif attempt.severity == "medium" and risk_level == "low":
            risk_level = "medium"

    # Validate content structure
    structure_ok, structure_issues = _content_validator.validate_content_structure(
        content
    )
    if not structure_ok:
        issues.extend(structure_issues)
        if risk_level == "low":
            risk_level = "medium"

    # Clean the content
    cleaned_content = sanitize_for_llm(content)[0]

    # Log security events
    if issues:
        log_security_event(
            "content_validation_failure",
            {
                "content_type": content_type,
                "issues_count": len(issues),
                "risk_level": risk_level,
                "content_length": len(content),
                "blocked_patterns": len(blocked_patterns),
            },
        )

    return ValidationResult(
        is_valid=len(issues) == 0,
        cleaned_content=cleaned_content,
        issues=issues,
        risk_level=risk_level,
        blocked_patterns=blocked_patterns,
    )


def detect_advanced_prompt_injection(text: str) -> List[InjectionAttempt]:
    """
    Detect sophisticated prompt injection attempts

    Args:
        text: Text to analyze

    Returns:
        List of detected injection attempts
    """
    attempts = []

    for i, pattern in enumerate(COMPILED_ADVANCED_PATTERNS):
        matches = pattern.finditer(text)
        for match in matches:
            # Determine severity based on pattern type
            severity = "medium"
            description = "Potential prompt injection"

            pattern_str = ADVANCED_PROMPT_INJECTION_PATTERNS[i]

            if any(
                keyword in pattern_str.lower()
                for keyword in ["system", "forget", "override", "jailbreak"]
            ):
                severity = "critical"
                description = "Critical prompt injection attempt"
            elif any(
                keyword in pattern_str.lower()
                for keyword in [
                    "ignore",
                    "bypass",
                    "unrestricted",
                    "dan",
                    "execute",
                    "switch",
                    "human",
                    "user",
                    "assistant",
                ]
            ):
                severity = "high"
                description = "High-risk prompt injection"

            attempts.append(
                InjectionAttempt(
                    pattern=pattern_str,
                    location=match.start(),
                    severity=severity,
                    description=description,
                )
            )

    return attempts


def validate_perspective_data(
    name: str, description: str, custom_prompt: str = None
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
    prompt: str, session_context: str = None
) -> ValidationResult:
    """
    Validate analysis prompts with context awareness

    Args:
        prompt: The analysis prompt
        session_context: Optional session context for validation

    Returns:
        ValidationResult for the analysis prompt
    """
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


def sanitize_for_llm(content: str) -> Tuple[str, List[str]]:
    """
    Sanitize content specifically for LLM consumption

    Args:
        content: Content to sanitize

    Returns:
        Tuple of (sanitized_content, applied_sanitizations)
    """
    sanitizations = []
    sanitized = content

    # Remove null bytes and dangerous control characters
    original_len = len(sanitized)
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", sanitized)
    if len(sanitized) != original_len:
        sanitizations.append("Removed control characters")

    # Limit excessive whitespace but preserve structure
    original_sanitized = sanitized
    sanitized = re.sub(r"\s{100,}", " " * 50, sanitized)
    if sanitized != original_sanitized:
        sanitizations.append("Limited excessive whitespace")

    # Remove zero-width characters that could be used for injection
    original_sanitized = sanitized
    sanitized = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", sanitized)
    if sanitized != original_sanitized:
        sanitizations.append("Removed zero-width characters")

    # Normalize line endings
    sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")

    # Trim excessive leading/trailing whitespace
    original_sanitized = sanitized
    sanitized = sanitized.strip()
    if sanitized != original_sanitized:
        sanitizations.append("Trimmed excessive whitespace")

    return sanitized, sanitizations


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


def log_security_event(
    event_type: str, details: dict, session_id: str = None, client_id: str = None
):
    """
    Enhanced logging for security-related events

    Args:
        event_type: Type of security event
        details: Event details
        session_id: Associated session ID if any
        client_id: Client identifier if available
    """
    log_entry = {
        "timestamp": time.time(),
        "event_type": event_type,
        "session_id": session_id,
        "client_id": client_id,
        "details": details,
    }

    # Use structured logging for security events
    logger.warning(f"SECURITY_EVENT: {log_entry}")

    # Track blocked attempts for monitoring
    if event_type in ["content_validation_failure", "prompt_injection_blocked"]:
        _content_validator.blocked_attempts[client_id or "default"] += 1


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
