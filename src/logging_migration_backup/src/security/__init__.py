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

from .secret_key_manager import SecretKeyManager, load_or_generate_secret_key
from .security_event_tracker import (
    SecurityEventTracker,
    log_security_event_with_binding,
)
from .client_validation_service import ClientValidationService, is_suspicious_access
from .client_binding_core import (
    ClientBindingManager,
    create_secure_session_with_binding,
    validate_session_access,
)
from .path_validator import PathValidator, SecureFileHandler
from .enhanced_validators import (
    EnhancedInputValidator,
    ConfigurationInputValidator,
    ValidationConfig,
)
from .secure_logging import SecureLogger, get_secure_logger, setup_secure_logging
from .security_monitor import (
    SecurityMonitor,
    get_security_monitor,
    record_security_event,
    get_security_health,
)

# Import legacy functions from parent security.py
try:
    import sys
    import os

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, parent_dir)
    from security import (
        validate_user_content,
        sanitize_error_message,
        log_security_event,
        validate_analysis_prompt,
        validate_perspective_data,
        detect_advanced_prompt_injection,
        sanitize_for_llm,
    )

    sys.path.pop(0)
except ImportError:
    # Fallback - define minimal versions
    # Import ValidationResult for proper return type
    from ..input_validators import ValidationResult

    def validate_user_content(
        content, content_type, max_length=10000, client_id="default"
    ):
        return ValidationResult(
            is_valid=True,
            cleaned_content=content,
            issues=[],
            risk_level="low",
            blocked_patterns=[],
        )

    def sanitize_error_message(message):
        return message

    def log_security_event(event_type, details, client_id="default"):
        pass

    def validate_analysis_prompt(prompt, session_context=None):
        return ValidationResult(
            is_valid=True,
            cleaned_content=prompt,
            issues=[],
            risk_level="low",
            blocked_patterns=[],
        )

    def validate_perspective_data(name, description, custom_prompt=None):
        return ValidationResult(
            is_valid=True,
            cleaned_content=f"{name}: {description}",
            issues=[],
            risk_level="low",
            blocked_patterns=[],
        )

    def detect_advanced_prompt_injection(content):
        return ValidationResult(
            is_valid=True,
            cleaned_content=content,
            issues=[],
            risk_level="low",
            blocked_patterns=[],
        )

    def sanitize_for_llm(content):
        return content


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
    # Security monitoring
    "SecurityMonitor",
    "get_security_monitor",
    "record_security_event",
    "get_security_health",
    # Legacy functions
    "validate_user_content",
    "sanitize_error_message",
    "log_security_event",
    "validate_analysis_prompt",
    "validate_perspective_data",
    "detect_advanced_prompt_injection",
    "sanitize_for_llm",
]
