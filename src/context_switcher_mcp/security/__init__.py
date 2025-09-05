"""
Security module for client binding and session validation.

This module provides comprehensive security features for the Context Switcher MCP server
including client binding, session validation, suspicious behavior detection, and audit logging.

Components:
    - SecretKeyManager: Secure key generation, rotation, and storage
    - ClientBindingCore: Central binding management and orchestration
    - ClientValidationService: Suspicious behavior detection and validation rules
    - SecurityEventTracker: Security event logging and audit trails

Usage:
    The security module is typically imported through the main client_binding.py facade
    for backward compatibility, but components can be used directly for advanced use cases.
"""

from ..input_sanitizer import sanitize_error_message, sanitize_for_llm
from ..input_validators import (
    InjectionAttempt,
    ValidationResult,
    _validation_orchestrator,
    detect_advanced_prompt_injection as _detect_advanced_prompt_injection,
    sanitize_user_input,
)
from ..logging_config import get_logger
from ..model_validators import validate_model_id
from ..security_events import log_security_event
from .client_binding_core import (
    ClientBindingManager,
    create_secure_session_with_binding,
    validate_session_access,
)
from .client_validation_service import ClientValidationService, is_suspicious_access
from .enhanced_validators import (
    ConfigurationInputValidator,
    EnhancedInputValidator,
    ValidationConfig,
)
from .path_validator import PathValidator, SecureFileHandler
from .secret_key_manager import SecretKeyManager, load_or_generate_secret_key
from .secure_logging import SecureLogger, get_secure_logger, setup_secure_logging
from .security_event_tracker import (
    SecurityEventTracker,
    log_security_event_with_binding,
)
from .security_monitor import (
    SecurityMonitor,
    get_security_health,
    get_security_monitor,
    record_security_event,
)

# Set up module logger for backward compatibility
logger = get_logger(__name__)


def validate_user_content(
    content: str, content_type: str, max_length: int = 10000, client_id: str = "default"
):
    """Use the proper validation orchestrator"""
    return _validation_orchestrator.validate_content(
        content, content_type, max_length, client_id
    )


def detect_advanced_prompt_injection(text: str):
    """Use the proper prompt injection detector"""
    return _detect_advanced_prompt_injection(text)


def validate_analysis_prompt(prompt: str, session_context: str = None):
    """Validate analysis prompts with context awareness"""
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

    # Check for meta-analysis attempts
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


def validate_perspective_data(name: str, description: str, custom_prompt: str = None):
    """Validate perspective-specific data"""
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


__all__ = [
    # Secret key management
    "SecretKeyManager",
    "load_or_generate_secret_key",
    # Security event tracking
    "SecurityEventTracker",
    "log_security_event_with_binding",
    # Client validation
    "ClientValidationService",
    "is_suspicious_access",
    # Core binding management
    "ClientBindingManager",
    "create_secure_session_with_binding",
    "validate_session_access",
    # Path validation and file security
    "PathValidator",
    "SecureFileHandler",
    # Enhanced input validation
    "EnhancedInputValidator",
    "ConfigurationInputValidator",
    "ValidationConfig",
    # Secure logging
    "SecureLogger",
    "get_secure_logger",
    "setup_secure_logging",
    "logger",
    # Security monitoring
    "SecurityMonitor",
    "get_security_monitor",
    "record_security_event",
    "get_security_health",
    # Data types
    "ValidationResult",
    "InjectionAttempt",
    # Legacy functions
    "validate_user_content",
    "sanitize_error_message",
    "sanitize_user_input",
    "log_security_event",
    "validate_analysis_prompt",
    "validate_perspective_data",
    "detect_advanced_prompt_injection",
    "sanitize_for_llm",
    "validate_model_id",
]
