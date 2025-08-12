"""Error classification and handling decision utilities"""

import time
from enum import Enum
from typing import Any, Dict, Type

from .exceptions import (
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    CircuitBreakerError,
    CircuitBreakerOpenError,
    ModelBackendError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelRateLimitError,
    ModelAuthenticationError,
    ModelValidationError,
    ConfigurationError,
    StorageError,
    SerializationError,
    AuthenticationError,
    AuthorizationError,
    InputValidationError,
    SecurityConfigurationError,
    NetworkTimeoutError,
    NetworkConnectivityError,
    NetworkProtocolError,
    DeadlockError,
    RaceConditionError,
    LockTimeoutError,
    SchemaValidationError,
    ParameterValidationError,
    BusinessRuleValidationError,
    PerformanceTimeoutError,
    ResourceExhaustionError,
    PerformanceDegradationError,
    CoTTimeoutError,
    CoTProcessingError,
)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling decisions."""

    CRITICAL = "critical"  # System-breaking, immediate attention required
    HIGH = "high"  # Major functionality impaired, urgent fix needed
    MEDIUM = "medium"  # Moderate impact, should be addressed soon
    LOW = "low"  # Minor issue, can be addressed in normal cycle
    INFO = "info"  # Informational, no immediate action needed


class ErrorCategory(Enum):
    """Error categories for handling strategy decisions."""

    TRANSIENT = "transient"  # Temporary issues that may resolve themselves
    PERMANENT = "permanent"  # Persistent issues requiring intervention
    USER_ERROR = "user_error"  # User input or behavior issues
    SYSTEM_ERROR = "system_error"  # Internal system problems
    EXTERNAL = "external"  # External service/dependency issues
    SECURITY = "security"  # Security-related issues
    CONFIGURATION = "configuration"  # Configuration or setup issues


# Mapping of exception types to their classifications
ERROR_CLASSIFICATIONS: Dict[Type[Exception], Dict[str, Any]] = {
    # Session Errors
    SessionNotFoundError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.USER_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": True,
        "auto_recover": False,
    },
    SessionExpiredError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.USER_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": True,
        "auto_recover": False,
    },
    SessionError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.SYSTEM_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    # Model Backend Errors
    ModelConnectionError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 2.0,
        "max_retries": 3,
    },
    ModelTimeoutError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 1.0,
        "max_retries": 2,
    },
    ModelRateLimitError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": True,
        "auto_recover": True,
        "retry_delay": 5.0,
        "max_retries": 3,
    },
    ModelAuthenticationError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.CONFIGURATION,
        "transient": False,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    ModelValidationError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.USER_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": True,
        "auto_recover": False,
    },
    ModelBackendError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.EXTERNAL,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 1.0,
        "max_retries": 2,
    },
    # Circuit Breaker Errors
    CircuitBreakerOpenError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.SYSTEM_ERROR,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 30.0,
        "max_retries": 1,
    },
    CircuitBreakerError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.SYSTEM_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    # Network Errors
    NetworkTimeoutError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 2.0,
        "max_retries": 3,
    },
    NetworkConnectivityError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 5.0,
        "max_retries": 2,
    },
    NetworkProtocolError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.EXTERNAL,
        "transient": False,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    # Security Errors
    AuthenticationError: {
        "severity": ErrorSeverity.CRITICAL,
        "category": ErrorCategory.SECURITY,
        "transient": False,
        "retriable": False,
        "user_facing": True,
        "auto_recover": False,
    },
    AuthorizationError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.SECURITY,
        "transient": False,
        "retriable": False,
        "user_facing": True,
        "auto_recover": False,
    },
    InputValidationError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.SECURITY,
        "transient": False,
        "retriable": False,
        "user_facing": True,
        "auto_recover": False,
    },
    SecurityConfigurationError: {
        "severity": ErrorSeverity.CRITICAL,
        "category": ErrorCategory.CONFIGURATION,
        "transient": False,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    # Concurrency Errors
    DeadlockError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 0.1,
        "max_retries": 5,
    },
    RaceConditionError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 0.05,
        "max_retries": 3,
    },
    LockTimeoutError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 0.1,
        "max_retries": 3,
    },
    # Validation Errors
    ParameterValidationError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.USER_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": True,
        "auto_recover": False,
    },
    SchemaValidationError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.USER_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": True,
        "auto_recover": False,
    },
    BusinessRuleValidationError: {
        "severity": ErrorSeverity.LOW,
        "category": ErrorCategory.USER_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": True,
        "auto_recover": False,
    },
    # Performance Errors
    PerformanceTimeoutError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 1.0,
        "max_retries": 1,
    },
    ResourceExhaustionError: {
        "severity": ErrorSeverity.CRITICAL,
        "category": ErrorCategory.SYSTEM_ERROR,
        "transient": True,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    PerformanceDegradationError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.SYSTEM_ERROR,
        "transient": True,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    # Storage Errors
    StorageError: {
        "severity": ErrorSeverity.HIGH,
        "category": ErrorCategory.SYSTEM_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    SerializationError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.SYSTEM_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    # Configuration Errors
    ConfigurationError: {
        "severity": ErrorSeverity.CRITICAL,
        "category": ErrorCategory.CONFIGURATION,
        "transient": False,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
    # Chain of Thought Errors
    CoTTimeoutError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.TRANSIENT,
        "transient": True,
        "retriable": True,
        "user_facing": False,
        "auto_recover": True,
        "retry_delay": 1.0,
        "max_retries": 1,
    },
    CoTProcessingError: {
        "severity": ErrorSeverity.MEDIUM,
        "category": ErrorCategory.SYSTEM_ERROR,
        "transient": False,
        "retriable": False,
        "user_facing": False,
        "auto_recover": False,
    },
}


def classify_error(error: Exception) -> Dict[str, Any]:
    """Classify an error and return its handling characteristics.

    Args:
        error: The exception to classify

    Returns:
        Dictionary with error classification information
    """
    error_type = type(error)

    # Check for exact match first
    if error_type in ERROR_CLASSIFICATIONS:
        classification = ERROR_CLASSIFICATIONS[error_type].copy()
    else:
        # Check inheritance chain
        classification = None
        for exc_type, config in ERROR_CLASSIFICATIONS.items():
            if isinstance(error, exc_type):
                classification = config.copy()
                break

        # Default classification for unknown errors
        if classification is None:
            classification = {
                "severity": ErrorSeverity.MEDIUM,
                "category": ErrorCategory.SYSTEM_ERROR,
                "transient": False,
                "retriable": False,
                "user_facing": False,
                "auto_recover": False,
            }

    # Add dynamic information
    classification.update(
        {
            "error_type": error_type.__name__,
            "error_message": str(error),
            "timestamp": time.time(),
        }
    )

    return classification


def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient (temporary).

    Args:
        error: The exception to check

    Returns:
        True if the error is likely to be temporary
    """
    classification = classify_error(error)
    return classification.get("transient", False)


def is_retriable_error(error: Exception) -> bool:
    """Check if an error should be retried.

    Args:
        error: The exception to check

    Returns:
        True if the error is worth retrying
    """
    classification = classify_error(error)
    return classification.get("retriable", False)


def is_user_facing_error(error: Exception) -> bool:
    """Check if an error should be shown to users.

    Args:
        error: The exception to check

    Returns:
        True if the error should be presented to users
    """
    classification = classify_error(error)
    return classification.get("user_facing", False)


def can_auto_recover(error: Exception) -> bool:
    """Check if an error can be automatically recovered from.

    Args:
        error: The exception to check

    Returns:
        True if automatic recovery is possible
    """
    classification = classify_error(error)
    return classification.get("auto_recover", False)


def get_error_severity(error: Exception) -> ErrorSeverity:
    """Get the severity level of an error.

    Args:
        error: The exception to check

    Returns:
        ErrorSeverity enum value
    """
    classification = classify_error(error)
    return classification.get("severity", ErrorSeverity.MEDIUM)


def get_retry_parameters(error: Exception) -> Dict[str, Any]:
    """Get retry parameters for an error.

    Args:
        error: The exception to check

    Returns:
        Dictionary with retry parameters (delay, max_retries)
    """
    classification = classify_error(error)
    return {
        "delay": classification.get("retry_delay", 1.0),
        "max_retries": classification.get("max_retries", 0),
        "should_retry": classification.get("retriable", False),
    }


def get_security_risk_level(error: Exception) -> str:
    """Assess security risk level of an error.

    Args:
        error: The exception to check

    Returns:
        Security risk level string
    """
    classification = classify_error(error)

    if classification["category"] == ErrorCategory.SECURITY:
        severity = classification["severity"]
        if severity == ErrorSeverity.CRITICAL:
            return "high_risk"
        elif severity == ErrorSeverity.HIGH:
            return "medium_risk"
        else:
            return "low_risk"

    # Non-security errors
    if isinstance(
        error, (InputValidationError, AuthenticationError, AuthorizationError)
    ):
        return "medium_risk"

    return "no_risk"


def should_alert_administrators(error: Exception) -> bool:
    """Determine if administrators should be alerted about an error.

    Args:
        error: The exception to check

    Returns:
        True if administrators should be notified
    """
    classification = classify_error(error)

    # Critical errors always require admin attention
    if classification["severity"] == ErrorSeverity.CRITICAL:
        return True

    # Security errors require attention
    if classification["category"] == ErrorCategory.SECURITY:
        return True

    # Configuration errors need admin intervention
    if classification["category"] == ErrorCategory.CONFIGURATION:
        return True

    # System errors that can't auto-recover
    if classification[
        "category"
    ] == ErrorCategory.SYSTEM_ERROR and not classification.get("auto_recover", False):
        return True

    return False


def get_error_handling_strategy(error: Exception) -> Dict[str, Any]:
    """Get comprehensive error handling strategy for an error.

    Args:
        error: The exception to analyze

    Returns:
        Dictionary with complete handling strategy
    """
    classification = classify_error(error)

    return {
        "classification": classification,
        "should_retry": is_retriable_error(error),
        "retry_params": get_retry_parameters(error),
        "user_facing": is_user_facing_error(error),
        "auto_recover": can_auto_recover(error),
        "alert_admins": should_alert_administrators(error),
        "security_risk": get_security_risk_level(error),
        "log_level": _get_recommended_log_level(classification),
    }


def _get_recommended_log_level(classification: Dict[str, Any]) -> str:
    """Get recommended logging level based on error classification."""
    severity = classification["severity"]
    category = classification["category"]

    if severity == ErrorSeverity.CRITICAL:
        return "CRITICAL"
    elif severity == ErrorSeverity.HIGH:
        return "ERROR"
    elif severity == ErrorSeverity.MEDIUM:
        if category == ErrorCategory.SECURITY:
            return "ERROR"
        else:
            return "WARNING"
    elif severity == ErrorSeverity.LOW:
        return "INFO"
    else:
        return "DEBUG"
