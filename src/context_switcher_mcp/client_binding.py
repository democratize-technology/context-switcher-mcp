"""
Client binding security utilities for session hijacking prevention.

This module serves as a facade for the decomposed security components while maintaining
full backward compatibility with the original API. All functionality has been moved
to focused, maintainable modules in the security package.

FACADE PATTERN: This file re-exports the public API from the new modular components
to ensure zero breaking changes for existing code.
"""

# Import all functionality from the new modular security components
from .security.secret_key_manager import (
    SecretKeyManager,
    load_or_generate_secret_key as _load_or_generate_secret_key,
)
from .security.security_event_tracker import (
    SecurityEventTracker,
    log_security_event_with_binding,
)
from .security.client_validation_service import (
    ClientValidationService,
    MAX_VALIDATION_FAILURES,
    MAX_SECURITY_FLAGS,
    BINDING_VALIDATION_WINDOW,
    SUSPICIOUS_ACCESS_THRESHOLD,
)
from .security.client_binding_core import (
    ClientBindingManager,
    get_client_binding_manager,
    create_secure_session_with_binding,
    validate_session_access,
)

# Backward compatibility exports - maintain original API surface
# These are the exact same names and signatures as the original file

# Secret key management (backward compatibility)
_BINDING_SECRET_KEY = _load_or_generate_secret_key()
_SECRET_KEY_MANAGER = SecretKeyManager(_BINDING_SECRET_KEY)

# Global client binding manager instance (backward compatibility)
client_binding_manager = get_client_binding_manager()

# Re-export all public functions with their original names and signatures
__all__ = [
    # Core classes
    "SecretKeyManager",
    "ClientBindingManager",
    "SecurityEventTracker",
    "ClientValidationService",
    # Global instances
    "client_binding_manager",
    "_SECRET_KEY_MANAGER",
    "_BINDING_SECRET_KEY",
    # Main public functions (backward compatibility)
    "create_secure_session_with_binding",
    "validate_session_access",
    "log_security_event_with_binding",
    # Configuration constants
    "MAX_VALIDATION_FAILURES",
    "MAX_SECURITY_FLAGS",
    "BINDING_VALIDATION_WINDOW",
    "SUSPICIOUS_ACCESS_THRESHOLD",
]

# BACKWARD COMPATIBILITY NOTES:
# ==============================
# This facade maintains 100% backward compatibility with the original client_binding.py.
#
# All imports that worked before will continue to work:
# - from .client_binding import ClientBindingManager
# - from .client_binding import client_binding_manager
# - from .client_binding import create_secure_session_with_binding
# - from .client_binding import validate_session_access
# - from .client_binding import log_security_event_with_binding
# - from .client_binding import MAX_VALIDATION_FAILURES, etc.
#
# The implementation has been decomposed into focused modules:
# - security.secret_key_manager: Key generation, rotation, storage
# - security.security_event_tracker: Security event logging and audit
# - security.client_validation_service: Suspicious behavior detection
# - security.client_binding_core: Core binding management coordination
#
# Benefits of decomposition:
# - Single Responsibility Principle compliance
# - Better testability and maintainability
# - Clear separation of concerns
# - Easier to extend and modify individual components
# - Improved code organization and readability
