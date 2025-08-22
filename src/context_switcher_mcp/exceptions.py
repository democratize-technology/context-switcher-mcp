"""Custom exceptions for the Context Switcher MCP server.

This module defines specific exception types to replace generic Exception
handling throughout the codebase, improving debugging and error recovery.
"""


class ContextSwitcherError(Exception):
    """Base exception for all context switcher errors."""

    pass


# Domain-specific base classes (must be defined before use in multiple inheritance)
class SecurityError(ContextSwitcherError):
    """Base class for security-related errors."""

    def __init__(self, message: str, security_context: dict = None):
        """Initialize security error with context.

        Args:
            message: Error description
            security_context: Security-relevant context (sanitized for logs)
        """
        super().__init__(message)
        self.security_context = security_context or {}


class NetworkError(ContextSwitcherError):
    """Base class for network-related errors."""

    def __init__(self, message: str, network_context: dict = None):
        """Initialize network error with context.

        Args:
            message: Error description
            network_context: Network-relevant context (host, port, etc.)
        """
        super().__init__(message)
        self.network_context = network_context or {}


class ValidationError(ContextSwitcherError):
    """Base class for validation errors."""

    def __init__(self, message: str, validation_context: dict = None):
        """Initialize validation error with context.

        Args:
            message: Error description
            validation_context: Validation-relevant context (field names, values, etc.)
        """
        super().__init__(message)
        self.validation_context = validation_context or {}


class SessionError(ContextSwitcherError):
    """Errors related to session management."""

    pass


class SessionNotFoundError(SessionError):
    """Raised when a requested session does not exist."""

    pass


class SessionExpiredError(SessionError):
    """Raised when a session has expired."""

    pass


class SessionCleanupError(SessionError):
    """Raised when session cleanup fails."""

    pass


class SessionSecurityError(SessionError):
    """Raised when session security violations occur."""

    pass


class SessionConcurrencyError(SessionError):
    """Raised when session concurrency conflicts occur."""

    pass


class OrchestrationError(ContextSwitcherError):
    """Errors related to thread orchestration."""

    pass


class CircuitBreakerError(OrchestrationError):
    """Errors related to circuit breaker functionality."""

    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Raised when circuit breaker is open and requests are blocked."""

    pass


class CircuitBreakerStateError(CircuitBreakerError):
    """Errors related to circuit breaker state persistence."""

    pass


class ModelBackendError(ContextSwitcherError):
    """Errors from LLM backend operations."""

    def __init__(self, message: str, performance_context: dict = None):
        """Initialize model backend error with context.

        Args:
            message: Error description
            performance_context: Performance-relevant context
        """
        super().__init__(message)
        self.performance_context = performance_context or {}


class ModelConnectionError(ModelBackendError):
    """Network or connection errors with model backends."""

    pass


class ModelTimeoutError(ModelBackendError):
    """Timeout errors from model backends."""

    pass


class ModelRateLimitError(ModelBackendError, NetworkError):
    """Rate limiting errors from model backends."""

    def __init__(self, message: str, network_context: dict = None):
        """Initialize model rate limit error with context.

        Args:
            message: Error description
            network_context: Network-relevant context
        """
        ModelBackendError.__init__(self, message)
        NetworkError.__init__(self, message, network_context)


class ModelAuthenticationError(ModelBackendError, SecurityError):
    """Authentication/authorization errors from model backends."""

    def __init__(self, message: str, security_context: dict = None):
        """Initialize model authentication error with context.

        Args:
            message: Error description
            security_context: Security-relevant context
        """
        ModelBackendError.__init__(self, message)
        SecurityError.__init__(self, message, security_context)


class ModelValidationError(ModelBackendError, ValidationError):
    """Validation errors for model inputs/outputs."""

    def __init__(self, message: str, validation_context: dict = None):
        """Initialize model validation error with context.

        Args:
            message: Error description
            validation_context: Validation-relevant context
        """
        ModelBackendError.__init__(self, message)
        ValidationError.__init__(self, message, validation_context)


class AnalysisError(ContextSwitcherError):
    """Errors during analysis operations."""

    pass


class PerspectiveError(ContextSwitcherError):
    """Errors related to perspective management."""

    pass


class ConfigurationError(ContextSwitcherError):
    """Configuration-related errors."""

    pass


class StorageError(ContextSwitcherError):
    """Errors related to file/storage operations."""

    pass


class SerializationError(StorageError):
    """Errors during JSON serialization/deserialization."""

    pass


# Security Domain Exceptions (specific implementations)


class AuthenticationError(SecurityError):
    """Authentication failures (invalid credentials, expired tokens, etc.)."""

    pass


class AuthorizationError(SecurityError):
    """Authorization failures (insufficient permissions, access denied, etc.)."""

    pass


class InputValidationError(SecurityError):
    """Input validation failures that could indicate security issues."""

    pass


class SecurityConfigurationError(SecurityError):
    """Security configuration errors (missing keys, weak settings, etc.)."""

    pass


# Network Domain Exceptions (specific implementations)


class NetworkTimeoutError(NetworkError):
    """Network timeout errors (connection, read, write timeouts)."""

    pass


class NetworkConnectivityError(NetworkError):
    """Network connectivity errors (DNS, unreachable host, etc.)."""

    pass


class NetworkProtocolError(NetworkError):
    """Network protocol errors (HTTP status codes, malformed responses, etc.)."""

    pass


# Concurrency Domain Exceptions
class ConcurrencyError(ContextSwitcherError):
    """Base class for concurrency-related errors."""

    def __init__(self, message: str, concurrency_context: dict = None):
        """Initialize concurrency error with context.

        Args:
            message: Error description
            concurrency_context: Concurrency-relevant context (thread IDs, lock names, etc.)
        """
        super().__init__(message)
        self.concurrency_context = concurrency_context or {}


class DeadlockError(ConcurrencyError):
    """Deadlock detection or prevention errors."""

    pass


class RaceConditionError(ConcurrencyError):
    """Race condition detection errors."""

    pass


class ThreadSafetyError(ConcurrencyError):
    """Thread safety violation errors."""

    pass


class LockTimeoutError(ConcurrencyError):
    """Lock acquisition timeout errors."""

    pass


# Validation Domain Exceptions (specific implementations)


class SchemaValidationError(ValidationError):
    """Schema or data structure validation errors."""

    pass


class ParameterValidationError(ValidationError):
    """Function/method parameter validation errors."""

    pass


class BusinessRuleValidationError(ValidationError):
    """Business rule validation errors."""

    pass


# Performance Domain Exceptions
class PerformanceError(ContextSwitcherError):
    """Base class for performance-related errors."""

    def __init__(self, message: str, performance_context: dict = None):
        """Initialize performance error with context.

        Args:
            message: Error description
            performance_context: Performance-relevant context (timing, memory, etc.)
        """
        super().__init__(message)
        self.performance_context = performance_context or {}


class PerformanceTimeoutError(PerformanceError):
    """Performance timeout errors (operation too slow)."""

    pass


class ResourceExhaustionError(PerformanceError):
    """Resource exhaustion errors (memory, CPU, handles, etc.)."""

    pass


class PerformanceDegradationError(PerformanceError):
    """Performance degradation detection errors."""

    pass


# Chain of Thought Specific Exceptions (for reasoning_orchestrator.py)
class CoTError(AnalysisError):
    """Base class for Chain of Thought reasoning errors."""

    pass


class CoTTimeoutError(CoTError):
    """Chain of Thought processing timeout."""

    def __init__(self, timeout_seconds: float):
        """Initialize with timeout context.

        Args:
            timeout_seconds: The timeout duration that was exceeded
        """
        super().__init__(
            f"Chain of Thought processing timed out after {timeout_seconds}s"
        )
        self.timeout_seconds = timeout_seconds


class CoTProcessingError(CoTError):
    """Chain of Thought processing errors."""

    pass
