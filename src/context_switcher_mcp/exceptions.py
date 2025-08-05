"""Custom exceptions for the Context Switcher MCP server.

This module defines specific exception types to replace generic Exception
handling throughout the codebase, improving debugging and error recovery.
"""


class ContextSwitcherError(Exception):
    """Base exception for all context switcher errors."""

    pass


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

    pass


class ModelConnectionError(ModelBackendError):
    """Network or connection errors with model backends."""

    pass


class ModelTimeoutError(ModelBackendError):
    """Timeout errors from model backends."""

    pass


class ModelRateLimitError(ModelBackendError):
    """Rate limiting errors from model backends."""

    pass


class ModelAuthenticationError(ModelBackendError):
    """Authentication/authorization errors from model backends."""

    pass


class ModelValidationError(ModelBackendError):
    """Validation errors for model inputs/outputs."""

    pass


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
