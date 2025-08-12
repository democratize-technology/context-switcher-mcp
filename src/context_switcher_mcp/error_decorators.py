"""Error handling decorators for standardizing exception patterns"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, List, Optional, Type, TypeVar

from .exceptions import (
    ModelBackendError,
    SessionError,
    NetworkError,
    ConcurrencyError,
    ValidationError,
)
from .security import sanitize_error_message

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def handle_model_errors(
    fallback_result: Any = None,
    log_level: int = logging.ERROR,
    include_context: bool = True,
) -> Callable[[F], F]:
    """Decorator to handle model backend errors with consistent formatting.

    Args:
        fallback_result: Value to return if error occurs (None by default)
        log_level: Logging level for errors (ERROR by default)
        include_context: Whether to include error context in logs

    Returns:
        Decorated function that handles ModelBackendError gracefully

    Example:
        @handle_model_errors(fallback_result={"error": "Model unavailable"})
        async def call_model(prompt: str) -> Dict[str, Any]:
            # Model call logic
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except ModelBackendError as e:
                context = {"function": func.__name__, "args_count": len(args)}
                if include_context:
                    context.update(getattr(e, "context", {}))

                logger.log(
                    log_level,
                    f"Model error in {func.__name__}: {sanitize_error_message(str(e))}",
                    extra={"context": context},
                    exc_info=True,
                )
                return fallback_result
            except Exception as e:
                # Re-raise non-model errors with context
                raise ModelBackendError(f"Unexpected error in {func.__name__}") from e

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ModelBackendError as e:
                context = {"function": func.__name__, "args_count": len(args)}
                if include_context:
                    context.update(getattr(e, "context", {}))

                logger.log(
                    log_level,
                    f"Model error in {func.__name__}: {sanitize_error_message(str(e))}",
                    extra={"context": context},
                    exc_info=True,
                )
                return fallback_result
            except Exception as e:
                # Re-raise non-model errors with context
                raise ModelBackendError(f"Unexpected error in {func.__name__}") from e

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def handle_session_errors(
    provide_hints: bool = True,
    log_level: int = logging.WARNING,
) -> Callable[[F], F]:
    """Decorator to handle session errors with user-friendly hints.

    Args:
        provide_hints: Whether to include helpful hints in error responses
        log_level: Logging level for session errors

    Returns:
        Decorated function that handles SessionError gracefully
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except SessionError as e:
                context = {"function": func.__name__, "error_type": type(e).__name__}

                # Add helpful hints based on error type
                if provide_hints:
                    if "not found" in str(e).lower():
                        context[
                            "hint"
                        ] = "Try calling start_context_analysis to create a new session"
                    elif "expired" in str(e).lower():
                        context[
                            "hint"
                        ] = "Session has expired, please start a new analysis"

                logger.log(
                    log_level,
                    f"Session error in {func.__name__}: {sanitize_error_message(str(e))}",
                    extra={"context": context},
                )
                raise  # Re-raise with original context

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except SessionError as e:
                context = {"function": func.__name__, "error_type": type(e).__name__}

                # Add helpful hints based on error type
                if provide_hints:
                    if "not found" in str(e).lower():
                        context[
                            "hint"
                        ] = "Try calling start_context_analysis to create a new session"
                    elif "expired" in str(e).lower():
                        context[
                            "hint"
                        ] = "Session has expired, please start a new analysis"

                logger.log(
                    log_level,
                    f"Session error in {func.__name__}: {sanitize_error_message(str(e))}",
                    extra={"context": context},
                )
                raise  # Re-raise with original context

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def retry_on_transient_errors(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    transient_errors: Optional[List[Type[Exception]]] = None,
) -> Callable[[F], F]:
    """Decorator to retry operations on transient errors with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff
        transient_errors: List of exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    if transient_errors is None:
        transient_errors = [
            NetworkError,
            ModelBackendError,  # Some model errors are transient
            ConcurrencyError,  # Lock timeouts, race conditions
        ]

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(transient_errors) as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"Transient error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): "
                            f"{sanitize_error_message(str(e))}. Retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * backoff_multiplier, max_delay)
                    else:
                        logger.error(
                            f"Max retries exceeded in {func.__name__}: "
                            f"{sanitize_error_message(str(e))}"
                        )
                        raise
                except Exception as e:
                    # Non-transient errors are not retried
                    logger.debug(
                        f"Non-transient error in {func.__name__}: {type(e).__name__}"
                    )
                    raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(transient_errors) as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"Transient error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): "
                            f"{sanitize_error_message(str(e))}. Retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_multiplier, max_delay)
                    else:
                        logger.error(
                            f"Max retries exceeded in {func.__name__}: "
                            f"{sanitize_error_message(str(e))}"
                        )
                        raise
                except Exception as e:
                    # Non-transient errors are not retried
                    logger.debug(
                        f"Non-transient error in {func.__name__}: {type(e).__name__}"
                    )
                    raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def log_errors_with_context(
    logger_instance: Optional[logging.Logger] = None,
    log_level: int = logging.ERROR,
    include_performance: bool = False,
) -> Callable[[F], F]:
    """Decorator to log errors with structured context information.

    Args:
        logger_instance: Logger to use (uses module logger if None)
        log_level: Logging level for errors
        include_performance: Whether to include performance timing

    Returns:
        Decorated function with structured error logging
    """
    if logger_instance is None:
        logger_instance = logger

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time() if include_performance else None

            try:
                result = await func(*args, **kwargs)

                # Log successful completion with timing if requested
                if include_performance and start_time:
                    duration = time.time() - start_time
                    if duration > 1.0:  # Only log slow operations
                        logger_instance.info(
                            f"{func.__name__} completed in {duration:.2f}s",
                            extra={
                                "performance": {
                                    "duration": duration,
                                    "function": func.__name__,
                                }
                            },
                        )

                return result

            except Exception as e:
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "error_type": type(e).__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }

                if include_performance and start_time:
                    context["duration"] = time.time() - start_time

                # Add specific error context if available
                if hasattr(e, "security_context"):
                    context["security_context"] = e.security_context
                elif hasattr(e, "network_context"):
                    context["network_context"] = e.network_context
                elif hasattr(e, "performance_context"):
                    context["performance_context"] = e.performance_context

                logger_instance.log(
                    log_level,
                    f"Error in {func.__name__}: {sanitize_error_message(str(e))}",
                    extra={"context": context},
                    exc_info=True,
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time() if include_performance else None

            try:
                result = func(*args, **kwargs)

                # Log successful completion with timing if requested
                if include_performance and start_time:
                    duration = time.time() - start_time
                    if duration > 1.0:  # Only log slow operations
                        logger_instance.info(
                            f"{func.__name__} completed in {duration:.2f}s",
                            extra={
                                "performance": {
                                    "duration": duration,
                                    "function": func.__name__,
                                }
                            },
                        )

                return result

            except Exception as e:
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "error_type": type(e).__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                }

                if include_performance and start_time:
                    context["duration"] = time.time() - start_time

                # Add specific error context if available
                if hasattr(e, "security_context"):
                    context["security_context"] = e.security_context
                elif hasattr(e, "network_context"):
                    context["network_context"] = e.network_context
                elif hasattr(e, "performance_context"):
                    context["performance_context"] = e.performance_context

                logger_instance.log(
                    log_level,
                    f"Error in {func.__name__}: {sanitize_error_message(str(e))}",
                    extra={"context": context},
                    exc_info=True,
                )
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def validate_parameters(**validation_rules) -> Callable[[F], F]:
    """Decorator to validate function parameters with custom rules.

    Args:
        **validation_rules: Parameter validation rules

    Returns:
        Decorated function with parameter validation

    Example:
        @validate_parameters(
            session_id=lambda x: x and len(x) > 10,
            timeout=lambda x: x > 0 and x <= 300
        )
        def some_function(session_id: str, timeout: float):
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature for parameter mapping
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate parameters
            for param_name, validator in validation_rules.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validator(value):
                            raise ValidationError(
                                f"Parameter '{param_name}' validation failed",
                                validation_context={
                                    "parameter": param_name,
                                    "value": str(value),
                                    "function": func.__name__,
                                },
                            )
                    except Exception as e:
                        if isinstance(e, ValidationError):
                            raise
                        raise ValidationError(
                            f"Parameter '{param_name}' validation error: {e}",
                            validation_context={
                                "parameter": param_name,
                                "value": str(value),
                                "function": func.__name__,
                                "validator_error": str(e),
                            },
                        ) from e

            return func(*args, **kwargs)

        return wrapper

    return decorator
