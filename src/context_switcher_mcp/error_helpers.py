"""Error handling helper functions to reduce code duplication and standardize error patterns"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from .aorp import create_error_response
from .error_classification import classify_error, get_error_handling_strategy
from .error_logging import log_error_with_context
from .exceptions import (
    AnalysisError,
    ContextSwitcherError,
    ModelBackendError,
    NetworkConnectivityError,
    NetworkTimeoutError,
    OrchestrationError,
    PerformanceTimeoutError,
    PerspectiveError,
    SessionCleanupError,
    SessionError,
    SessionExpiredError,
    SessionNotFoundError,
)
from .logging_base import get_logger
from .security import sanitize_error_message

logger = get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def validation_error(
    message: str,
    context: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Create a validation error response

    Args:
        message: Error message
        context: Additional context data
        session_id: Optional session ID for tracking

    Returns:
        Formatted AORP error response
    """
    return create_error_response(
        f"Validation failed: {sanitize_error_message(message)}",
        "validation_error",
        context or {},
        recoverable=True,
        session_id=session_id,
    )


def session_not_found_error(session_id: str, hint: str | None = None) -> dict[str, Any]:
    """Create a session not found error response

    Args:
        session_id: The session ID that wasn't found
        hint: Optional hint for the user

    Returns:
        Formatted AORP error response
    """
    message = f"Session '{session_id}' not found or expired"
    if hint:
        message += f". {hint}"

    return create_error_response(
        message,
        "session_not_found",
        {"session_id": session_id},
        recoverable=True,
    )


def rate_limit_error(
    message: str, retry_after_seconds: int, session_id: str | None = None
) -> dict[str, Any]:
    """Create a rate limit error response

    Args:
        message: Error message
        retry_after_seconds: How long to wait before retrying
        session_id: Optional session ID

    Returns:
        Formatted AORP error response
    """
    context = {"retry_after_seconds": retry_after_seconds}
    if session_id:
        context["session_id"] = session_id

    return create_error_response(
        message,
        "rate_limited",
        context,
        recoverable=True,
    )


def execution_error(
    message: str,
    context: dict[str, Any] | None = None,
    session_id: str | None = None,
    recoverable: bool = True,
) -> dict[str, Any]:
    """Create an execution error response

    Args:
        message: Error message
        context: Additional context data
        session_id: Optional session ID
        recoverable: Whether the error is recoverable

    Returns:
        Formatted AORP error response
    """
    return create_error_response(
        f"Execution failed: {sanitize_error_message(message)}",
        "execution_error",
        context or {},
        recoverable=recoverable,
        session_id=session_id,
    )


def security_error(
    message: str, details: dict[str, Any], session_id: str | None = None
) -> dict[str, Any]:
    """Create a security error response

    Args:
        message: Error message
        details: Security-related details
        session_id: Optional session ID

    Returns:
        Formatted AORP error response
    """
    return create_error_response(
        f"Security check failed: {sanitize_error_message(message)}",
        "security_error",
        details,
        recoverable=False,  # Security errors typically not recoverable
        session_id=session_id,
    )


def resource_limit_error(
    resource_type: str, limit: int, current: int | None = None
) -> dict[str, Any]:
    """Create a resource limit error response

    Args:
        resource_type: Type of resource (sessions, analyses, etc.)
        limit: The limit that was exceeded
        current: Current usage (optional)

    Returns:
        Formatted AORP error response
    """
    message = f"{resource_type.title()} limit reached ({limit})"
    context = {"resource_type": resource_type, "limit": limit}

    if current is not None:
        message += f". Current: {current}"
        context["current"] = current

    return create_error_response(
        message,
        "resource_limit",
        context,
        recoverable=True,
    )


# ERROR STANDARDIZATION AND MIGRATION UTILITIES


def wrap_generic_exception(
    error: Exception,
    operation_name: str,
    default_exception_class: type[ContextSwitcherError] = OrchestrationError,
    context: dict[str, Any] | None = None,
) -> ContextSwitcherError:
    """Convert generic exceptions to specific typed exceptions with context.

    This is a migration helper to standardize generic Exception handling.

    Args:
        error: The original exception
        operation_name: Name of the operation that failed
        default_exception_class: Default exception class if no specific mapping found
        context: Additional context to include

    Returns:
        Specific typed exception with proper context

    Example:
        try:
            some_operation()
        except Exception as e:
            # OLD: raise OrchestrationError(f"Failed: {e}") from e
            # NEW: raise wrap_generic_exception(e, "some_operation")
    """
    if isinstance(error, ContextSwitcherError):
        # Already a specific exception, return as-is
        return error

    # Get error handling strategy
    strategy = get_error_handling_strategy(error)
    error_message = sanitize_error_message(str(error))

    # Build context
    full_context = {
        "operation_name": operation_name,
        "original_error_type": type(error).__name__,
        "classification": strategy["classification"],
    }
    if context:
        full_context.update(context)

    # Map to specific exception types based on error characteristics
    if "timeout" in error_message.lower():
        return PerformanceTimeoutError(
            f"Operation {operation_name} timed out: {error_message}",
            performance_context=full_context,
        )
    elif "session" in error_message.lower():
        if "not found" in error_message.lower():
            return SessionNotFoundError(
                f"Session error in {operation_name}: {error_message}"
            )
        elif "expired" in error_message.lower():
            return SessionExpiredError(
                f"Session error in {operation_name}: {error_message}"
            )
        else:
            return SessionError(f"Session error in {operation_name}: {error_message}")
    elif any(term in error_message.lower() for term in ["model", "api", "backend"]):
        return ModelBackendError(
            f"Model backend error in {operation_name}: {error_message}",
            performance_context=full_context,
        )
    elif "perspective" in error_message.lower():
        return PerspectiveError(
            f"Perspective error in {operation_name}: {error_message}"
        )
    else:
        # Use default exception class with context
        try:
            # Try to pass context if the exception supports it
            if hasattr(default_exception_class, "__init__"):
                import inspect

                sig = inspect.signature(default_exception_class.__init__)
                params = list(sig.parameters.keys())

                kwargs = {}
                for context_key, context_value in full_context.items():
                    if context_key.endswith("_context") and context_key in params:
                        kwargs[context_key] = context_value

                return default_exception_class(
                    f"{operation_name} failed: {error_message}", **kwargs
                )
            else:
                return default_exception_class(
                    f"{operation_name} failed: {error_message}"
                )
        except (TypeError, ValueError):
            # Fallback to simple message
            return default_exception_class(f"{operation_name} failed: {error_message}")


def standardize_exception_handling(
    operation_name: str,
    session_id: str | None = None,
    log_errors: bool = True,
    default_exception_class: type[ContextSwitcherError] = OrchestrationError,
) -> Callable[[F], F]:
    """Decorator to standardize exception handling for any function.

    This decorator helps migrate from inconsistent error handling patterns
    to the standardized approach.

    Args:
        operation_name: Name of the operation for logging and context
        session_id: Optional session ID for correlation
        log_errors: Whether to log errors with structured logging
        default_exception_class: Default exception class for unmapped errors

    Returns:
        Decorated function with standardized error handling

    Example:
        # OLD:
        def some_function():
            try:
                do_work()
            except Exception as e:
                logger.error(f"Error: {e}")
                raise OrchestrationError(str(e)) from e

        # NEW:
        @standardize_exception_handling("some_function")
        def some_function():
            do_work()  # Errors are automatically handled consistently
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ContextSwitcherError:
                # Already a specific exception, let it propagate
                raise
            except Exception as e:
                # Log error if requested
                if log_errors:
                    log_error_with_context(
                        error=e,
                        operation_name=operation_name,
                        session_id=session_id,
                        additional_context={
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                        },
                    )

                # Convert to specific exception
                specific_error = wrap_generic_exception(
                    e,
                    operation_name,
                    default_exception_class,
                    context={"function": func.__name__},
                )
                raise specific_error from e

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ContextSwitcherError:
                # Already a specific exception, let it propagate
                raise
            except Exception as e:
                # Log error if requested
                if log_errors:
                    log_error_with_context(
                        error=e,
                        operation_name=operation_name,
                        session_id=session_id,
                        additional_context={
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()),
                        },
                    )

                # Convert to specific exception
                specific_error = wrap_generic_exception(
                    e,
                    operation_name,
                    default_exception_class,
                    context={"function": func.__name__},
                )
                raise specific_error from e

        # Return appropriate wrapper based on function type
        import asyncio

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def create_error_mapping(
    error_patterns: dict[str, type[ContextSwitcherError]],
) -> Callable[[Exception, str], ContextSwitcherError]:
    """Create a custom error mapping function for specific modules.

    Args:
        error_patterns: Dict mapping error message patterns to exception types

    Returns:
        Function that maps generic exceptions to specific types

    Example:
        orchestrator_mapping = create_error_mapping({
            "timeout": PerformanceTimeoutError,
            "perspective": PerspectiveError,
            "session": SessionError,
        })

        try:
            do_work()
        except Exception as e:
            raise orchestrator_mapping(e, "orchestration_task") from e
    """

    def map_error(error: Exception, operation_name: str) -> ContextSwitcherError:
        if isinstance(error, ContextSwitcherError):
            return error

        error_message = str(error).lower()

        # Find matching pattern
        for pattern, exception_class in error_patterns.items():
            if pattern.lower() in error_message:
                try:
                    return exception_class(
                        f"{operation_name}: {sanitize_error_message(str(error))}"
                    )
                except TypeError:
                    # If constructor doesn't accept message, try without
                    return exception_class()

        # Default to generic orchestration error
        return OrchestrationError(
            f"{operation_name} failed: {sanitize_error_message(str(error))}"
        )

    return map_error


def get_fallback_response(
    error: Exception, operation_name: str, default_response: Any = None
) -> dict[str, Any]:
    """Generate a standardized fallback response for errors.

    Args:
        error: The exception that occurred
        operation_name: Name of the operation
        default_response: Default response value

    Returns:
        Structured fallback response
    """
    classification = classify_error(error)

    response = {
        "status": "error",
        "operation": operation_name,
        "error_type": classification["error_type"],
        "message": sanitize_error_message(str(error)),
        "retriable": classification.get("retriable", False),
        "timestamp": classification["timestamp"],
    }

    if default_response is not None:
        response["fallback_data"] = default_response

    # Add user-friendly message if error is user-facing
    if classification.get("user_facing", False):
        response["user_message"] = _get_user_friendly_message(error, operation_name)

    return response


def _get_user_friendly_message(error: Exception, operation_name: str) -> str:
    """Generate user-friendly error message."""
    error_msg = str(error).lower()

    if "timeout" in error_msg:
        return f"The {operation_name} is taking longer than expected. Please try again."
    elif "rate limit" in error_msg:
        return "Too many requests. Please wait a moment and try again."
    elif "session" in error_msg and "not found" in error_msg:
        return "Your session has expired. Please start a new analysis."
    elif "authentication" in error_msg or "unauthorized" in error_msg:
        return "Authentication failed. Please check your credentials."
    else:
        return f"An error occurred during {operation_name}. Please try again or contact support."


# Pre-configured error mappers for specific modules
PERSPECTIVE_ORCHESTRATOR_MAPPING = create_error_mapping(
    {
        "perspective": PerspectiveError,
        "thread": OrchestrationError,
        "synthesis": AnalysisError,
        "timeout": PerformanceTimeoutError,
    }
)

STREAMING_COORDINATOR_MAPPING = create_error_mapping(
    {
        "stream": OrchestrationError,
        "backend": ModelBackendError,
        "connection": NetworkConnectivityError,
        "timeout": NetworkTimeoutError,
    }
)

SESSION_MANAGER_MAPPING = create_error_mapping(
    {
        "session": SessionError,
        "cleanup": SessionCleanupError,
        "expired": SessionExpiredError,
        "not found": SessionNotFoundError,
    }
)
