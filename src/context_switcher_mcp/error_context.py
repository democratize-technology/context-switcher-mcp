"""Error handling context managers for structured error management"""

import asyncio
import contextlib
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, Type

from .exceptions import (
    OrchestrationError,
    PerformanceError,
)
from .input_sanitizer import sanitize_error_message

logger = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def error_context(
    operation_name: str,
    session_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    log_success: bool = True,
    performance_threshold: float = 1.0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async context manager for structured error handling and logging.

    Args:
        operation_name: Name of the operation being performed
        session_id: Optional session ID for correlation
        user_context: Additional context information
        log_success: Whether to log successful operations
        performance_threshold: Log performance if operation takes longer than this (seconds)

    Yields:
        Context dictionary for operation tracking

    Example:
        async with error_context("model_call", session_id="abc123") as ctx:
            ctx["model"] = "bedrock"
            result = await some_model_call()
            ctx["tokens"] = result.token_count
    """
    operation_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    context = {
        "operation_id": operation_id,
        "operation_name": operation_name,
        "session_id": session_id,
        "start_time": start_time,
    }

    if user_context:
        context.update(user_context)

    logger.debug(
        f"Starting operation {operation_name} (ID: {operation_id})",
        extra={"context": context},
    )

    try:
        yield context

        # Operation completed successfully
        duration = time.time() - start_time
        context["duration"] = duration
        context["status"] = "success"

        if log_success:
            log_level = (
                logging.INFO if duration > performance_threshold else logging.DEBUG
            )
            logger.log(
                log_level,
                f"Operation {operation_name} completed in {duration:.2f}s (ID: {operation_id})",
                extra={"context": context},
            )

    except Exception as e:
        # Operation failed
        duration = time.time() - start_time
        context.update(
            {
                "duration": duration,
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": sanitize_error_message(str(e)),
            }
        )

        # Add specific error context if available
        if hasattr(e, "security_context"):
            context["security_context"] = e.security_context
        elif hasattr(e, "network_context"):
            context["network_context"] = e.network_context
        elif hasattr(e, "performance_context"):
            context["performance_context"] = e.performance_context

        logger.error(
            f"Operation {operation_name} failed after {duration:.2f}s (ID: {operation_id}): {sanitize_error_message(str(e))}",
            extra={"context": context},
            exc_info=True,
        )

        raise


@contextlib.contextmanager
def error_context_sync(
    operation_name: str,
    session_id: Optional[str] = None,
    user_context: Optional[Dict[str, Any]] = None,
    log_success: bool = True,
    performance_threshold: float = 1.0,
) -> Generator[Dict[str, Any], None, None]:
    """Synchronous context manager for structured error handling and logging.

    Same functionality as error_context but for synchronous operations.
    """
    operation_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    context = {
        "operation_id": operation_id,
        "operation_name": operation_name,
        "session_id": session_id,
        "start_time": start_time,
    }

    if user_context:
        context.update(user_context)

    logger.debug(
        f"Starting operation {operation_name} (ID: {operation_id})",
        extra={"context": context},
    )

    try:
        yield context

        # Operation completed successfully
        duration = time.time() - start_time
        context["duration"] = duration
        context["status"] = "success"

        if log_success:
            log_level = (
                logging.INFO if duration > performance_threshold else logging.DEBUG
            )
            logger.log(
                log_level,
                f"Operation {operation_name} completed in {duration:.2f}s (ID: {operation_id})",
                extra={"context": context},
            )

    except Exception as e:
        # Operation failed
        duration = time.time() - start_time
        context.update(
            {
                "duration": duration,
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": sanitize_error_message(str(e)),
            }
        )

        # Add specific error context if available
        if hasattr(e, "security_context"):
            context["security_context"] = e.security_context
        elif hasattr(e, "network_context"):
            context["network_context"] = e.network_context
        elif hasattr(e, "performance_context"):
            context["performance_context"] = e.performance_context

        logger.error(
            f"Operation {operation_name} failed after {duration:.2f}s (ID: {operation_id}): {sanitize_error_message(str(e))}",
            extra={"context": context},
            exc_info=True,
        )

        raise


@contextlib.asynccontextmanager
async def suppress_and_log(
    *exception_types: Type[Exception],
    fallback_action: Optional[Callable] = None,
    log_level: int = logging.WARNING,
    operation_name: str = "unknown_operation",
) -> AsyncGenerator[None, None]:
    """Async context manager to suppress specific exceptions and log them.

    Args:
        *exception_types: Exception types to suppress
        fallback_action: Optional callable to execute when exception is suppressed
        log_level: Logging level for suppressed exceptions
        operation_name: Name of the operation for logging

    Example:
        async with suppress_and_log(ConnectionError, operation_name="cleanup"):
            await cleanup_resources()
    """
    try:
        yield
    except tuple(exception_types) as e:
        logger.log(
            log_level,
            f"Suppressed {type(e).__name__} in {operation_name}: {sanitize_error_message(str(e))}",
            extra={
                "context": {
                    "operation_name": operation_name,
                    "suppressed_error": type(e).__name__,
                    "error_message": sanitize_error_message(str(e)),
                }
            },
        )

        if fallback_action:
            try:
                if asyncio.iscoroutinefunction(fallback_action):
                    await fallback_action()
                else:
                    fallback_action()
            except Exception as fallback_error:
                logger.error(
                    f"Fallback action failed in {operation_name}: {sanitize_error_message(str(fallback_error))}",
                    exc_info=True,
                )


@contextlib.contextmanager
def suppress_and_log_sync(
    *exception_types: Type[Exception],
    fallback_action: Optional[Callable] = None,
    log_level: int = logging.WARNING,
    operation_name: str = "unknown_operation",
) -> Generator[None, None, None]:
    """Synchronous context manager to suppress specific exceptions and log them."""
    try:
        yield
    except tuple(exception_types) as e:
        logger.log(
            log_level,
            f"Suppressed {type(e).__name__} in {operation_name}: {sanitize_error_message(str(e))}",
            extra={
                "context": {
                    "operation_name": operation_name,
                    "suppressed_error": type(e).__name__,
                    "error_message": sanitize_error_message(str(e)),
                }
            },
        )

        if fallback_action:
            try:
                fallback_action()
            except Exception as fallback_error:
                logger.error(
                    f"Fallback action failed in {operation_name}: {sanitize_error_message(str(fallback_error))}",
                    exc_info=True,
                )


@contextlib.asynccontextmanager
async def resource_cleanup_context(
    cleanup_actions: List[Callable],
    operation_name: str = "resource_operation",
    ignore_cleanup_errors: bool = True,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Context manager that ensures resource cleanup even if operation fails.

    Args:
        cleanup_actions: List of cleanup functions to call
        operation_name: Name of the operation for logging
        ignore_cleanup_errors: Whether to ignore errors during cleanup

    Yields:
        Context dictionary for the operation

    Example:
        cleanup_funcs = [lambda: session.close(), lambda: file.close()]
        async with resource_cleanup_context(cleanup_funcs, "data_processing") as ctx:
            ctx["processed_items"] = 0
            # Do work that might fail
            ctx["processed_items"] += 1
    """
    context = {"operation_name": operation_name, "cleanup_errors": []}

    try:
        yield context
    finally:
        # Always attempt cleanup
        for cleanup_func in cleanup_actions:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func()
                else:
                    cleanup_func()
                logger.debug(f"Cleanup action completed for {operation_name}")
            except Exception as e:
                error_msg = sanitize_error_message(str(e))
                context["cleanup_errors"].append(
                    {
                        "cleanup_function": getattr(
                            cleanup_func, "__name__", str(cleanup_func)
                        ),
                        "error": error_msg,
                    }
                )

                if ignore_cleanup_errors:
                    logger.warning(
                        f"Cleanup error in {operation_name}: {error_msg}",
                        extra={"context": {"operation_name": operation_name}},
                    )
                else:
                    logger.error(
                        f"Cleanup error in {operation_name}: {error_msg}", exc_info=True
                    )
                    raise


@contextlib.contextmanager
def timeout_context(
    timeout_seconds: float,
    operation_name: str = "timed_operation",
    timeout_error_class: Type[Exception] = PerformanceError,
) -> Generator[None, None, None]:
    """Context manager that raises timeout error if operation takes too long.

    Note: This is a simple timeout implementation. For more sophisticated
    async timeout handling, use asyncio.wait_for() instead.

    Args:
        timeout_seconds: Maximum allowed duration
        operation_name: Name of the operation for error messages
        timeout_error_class: Exception class to raise on timeout
    """
    start_time = time.time()

    try:
        yield
    finally:
        duration = time.time() - start_time
        if duration > timeout_seconds:
            raise timeout_error_class(
                f"Operation {operation_name} timed out after {duration:.2f}s (limit: {timeout_seconds}s)",
                performance_context={
                    "operation_name": operation_name,
                    "duration": duration,
                    "timeout_limit": timeout_seconds,
                },
            )


class ErrorAccumulator:
    """Context manager for collecting multiple errors without failing immediately."""

    def __init__(
        self,
        operation_name: str = "batch_operation",
        max_errors: Optional[int] = None,
        log_individual_errors: bool = True,
    ):
        """Initialize error accumulator.

        Args:
            operation_name: Name of the batch operation
            max_errors: Maximum errors before raising (None = unlimited)
            log_individual_errors: Whether to log each error as it occurs
        """
        self.operation_name = operation_name
        self.max_errors = max_errors
        self.log_individual_errors = log_individual_errors
        self.errors: List[Dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.errors:
            logger.error(
                f"Batch operation {self.operation_name} completed with {len(self.errors)} errors",
                extra={
                    "context": {
                        "errors": self.errors,
                        "operation_name": self.operation_name,
                    }
                },
            )

    @contextlib.contextmanager
    def capture_error(self, sub_operation: str = "sub_operation"):
        """Capture and accumulate an error from a sub-operation."""
        try:
            yield
        except Exception as e:
            error_info = {
                "sub_operation": sub_operation,
                "error_type": type(e).__name__,
                "error_message": sanitize_error_message(str(e)),
                "timestamp": time.time(),
            }

            self.errors.append(error_info)

            if self.log_individual_errors:
                logger.warning(
                    f"Error in {sub_operation} (batch: {self.operation_name}): {error_info['error_message']}"
                )

            # Check if we've hit the error limit
            if self.max_errors and len(self.errors) >= self.max_errors:
                raise OrchestrationError(
                    f"Too many errors in {self.operation_name}: {len(self.errors)}/{self.max_errors}"
                ) from e

    def has_errors(self) -> bool:
        """Check if any errors were accumulated."""
        return len(self.errors) > 0

    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of accumulated errors."""
        error_types = {}
        for error in self.errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "operation_name": self.operation_name,
            "errors": self.errors,
        }
