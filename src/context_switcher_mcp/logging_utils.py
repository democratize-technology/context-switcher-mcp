"""
Logging utilities and decorators for Context Switcher MCP

This module provides common logging patterns, decorators, and context managers
to standardize logging across the application.
"""

import functools
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from .error_logging import log_error_with_context
from .logging_base import (
    get_correlation_id,
    get_logger,
    is_performance_logging_enabled,
    set_correlation_id,
)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class OperationMetrics:
    """Metrics for logged operations"""

    operation_name: str
    duration: float
    success: bool
    correlation_id: str
    error: Exception | None = None
    metadata: dict[str, Any] | None = None


class OperationLogger:
    """Enhanced operation logger with metrics and correlation tracking"""

    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time: float | None = None
        self.correlation_id: str | None = None
        self.metadata: dict[str, Any] = {}

    def start(self, correlation_id: str | None = None, **metadata) -> str:
        """Start operation logging"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]

        self.correlation_id = correlation_id
        self.start_time = time.time()
        self.metadata.update(metadata)

        # Set correlation ID in context
        set_correlation_id(correlation_id)

        self.logger.info(
            f"Starting {self.operation_name}",
            extra={
                "operation": self.operation_name,
                "correlation_id": correlation_id,
                "metadata": self.metadata,
            },
        )

        return correlation_id

    def success(
        self, result_metadata: dict[str, Any] | None = None
    ) -> OperationMetrics:
        """Log successful operation completion"""
        duration = time.time() - (self.start_time or 0)

        if result_metadata:
            self.metadata.update(result_metadata)

        metrics = OperationMetrics(
            operation_name=self.operation_name,
            duration=duration,
            success=True,
            correlation_id=self.correlation_id or "unknown",
            metadata=self.metadata,
        )

        self.logger.info(
            f"Completed {self.operation_name} successfully in {duration:.2f}s",
            extra={
                "operation": self.operation_name,
                "duration": duration,
                "success": True,
                "correlation_id": self.correlation_id,
                "metadata": self.metadata,
            },
        )

        # Log performance metrics if enabled
        if is_performance_logging_enabled():
            perf_logger = get_logger("performance")
            perf_logger.debug(
                f"Performance: {self.operation_name} completed in {duration:.2f}s",
                extra={"performance_metrics": metrics.__dict__},
            )

        return metrics

    def error(
        self, error: Exception, additional_context: dict[str, Any] | None = None
    ) -> OperationMetrics:
        """Log operation error"""
        duration = time.time() - (self.start_time or 0)

        if additional_context:
            self.metadata.update(additional_context)

        metrics = OperationMetrics(
            operation_name=self.operation_name,
            duration=duration,
            success=False,
            correlation_id=self.correlation_id or "unknown",
            error=error,
            metadata=self.metadata,
        )

        # Use structured error logging
        log_error_with_context(
            error=error,
            operation_name=self.operation_name,
            session_id=self.correlation_id,
            additional_context=self.metadata,
        )

        self.logger.error(
            f"Failed {self.operation_name} after {duration:.2f}s: {error}",
            extra={
                "operation": self.operation_name,
                "duration": duration,
                "success": False,
                "error_type": type(error).__name__,
                "correlation_id": self.correlation_id,
                "metadata": self.metadata,
            },
            exc_info=True,
        )

        return metrics


@contextmanager
def log_operation(
    operation_name: str,
    logger: logging.Logger | None = None,
    correlation_id: str | None = None,
    **metadata,
):
    """Context manager for logging operations with timing and error handling"""
    if logger is None:
        logger = get_logger(__name__)

    op_logger = OperationLogger(logger, operation_name)
    correlation_id = op_logger.start(correlation_id, **metadata)

    try:
        yield op_logger
        op_logger.success()
    except Exception as e:
        op_logger.error(e)
        raise


def logged_operation(
    operation_name: str | None = None,
    logger: logging.Logger | None = None,
    include_args: bool = False,
    include_result: bool = False,
    correlation_id_arg: str | None = None,
) -> Callable[[F], F]:
    """Decorator for logging function/method operations"""

    def decorator(func: F) -> F:
        op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
        func_logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract correlation ID from arguments if specified
            corr_id = None
            if correlation_id_arg and correlation_id_arg in kwargs:
                corr_id = kwargs[correlation_id_arg]

            # Prepare metadata
            metadata = {}
            if include_args:
                metadata["args"] = str(args)[:200]  # Limit length
                metadata["kwargs"] = {k: str(v)[:100] for k, v in kwargs.items()}

            with log_operation(op_name, func_logger, corr_id, **metadata) as op:
                result = func(*args, **kwargs)

                if include_result and result is not None:
                    op.metadata["result_type"] = type(result).__name__
                    if hasattr(result, "__len__"):
                        op.metadata["result_length"] = len(result)

                return result

        return wrapper

    return decorator


def async_logged_operation(
    operation_name: str | None = None,
    logger: logging.Logger | None = None,
    include_args: bool = False,
    include_result: bool = False,
    correlation_id_arg: str | None = None,
) -> Callable[[F], F]:
    """Async decorator for logging async function/method operations"""

    def decorator(func: F) -> F:
        op_name = operation_name or f"{func.__module__}.{func.__qualname__}"
        func_logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract correlation ID from arguments if specified
            corr_id = None
            if correlation_id_arg and correlation_id_arg in kwargs:
                corr_id = kwargs[correlation_id_arg]

            # Prepare metadata
            metadata = {}
            if include_args:
                metadata["args"] = str(args)[:200]  # Limit length
                metadata["kwargs"] = {k: str(v)[:100] for k, v in kwargs.items()}

            with log_operation(op_name, func_logger, corr_id, **metadata) as op:
                result = await func(*args, **kwargs)

                if include_result and result is not None:
                    op.metadata["result_type"] = type(result).__name__
                    if hasattr(result, "__len__"):
                        op.metadata["result_length"] = len(result)

                return result

        return wrapper

    return decorator


@contextmanager
def correlation_context(correlation_id: str | None = None):
    """Context manager for correlation ID tracking"""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())[:8]

    # Store previous correlation ID
    previous_id = get_correlation_id()

    try:
        set_correlation_id(correlation_id)
        yield correlation_id
    finally:
        # Restore previous correlation ID
        set_correlation_id(previous_id)


def performance_timer(
    operation_name: str,
    logger: logging.Logger | None = None,
    threshold_seconds: float = 1.0,
    warn_threshold_seconds: float = 5.0,
) -> Callable[[F], F]:
    """Decorator for performance timing with configurable thresholds"""

    def decorator(func: F) -> F:
        perf_logger = logger or get_logger("performance")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log based on duration thresholds
                if duration >= warn_threshold_seconds:
                    perf_logger.warning(
                        f"SLOW OPERATION: {operation_name} took {duration:.2f}s (threshold: {warn_threshold_seconds}s)",
                        extra={
                            "performance_metrics": {
                                "operation": operation_name,
                                "duration": duration,
                                "threshold": warn_threshold_seconds,
                                "status": "slow",
                            }
                        },
                    )
                elif duration >= threshold_seconds:
                    perf_logger.info(
                        f"Performance: {operation_name} took {duration:.2f}s",
                        extra={
                            "performance_metrics": {
                                "operation": operation_name,
                                "duration": duration,
                                "threshold": threshold_seconds,
                                "status": "tracked",
                            }
                        },
                    )
                elif is_performance_logging_enabled():
                    perf_logger.debug(
                        f"Fast operation: {operation_name} took {duration:.2f}s",
                        extra={
                            "performance_metrics": {
                                "operation": operation_name,
                                "duration": duration,
                                "status": "fast",
                            }
                        },
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                perf_logger.error(
                    f"Failed operation: {operation_name} failed after {duration:.2f}s",
                    extra={
                        "performance_metrics": {
                            "operation": operation_name,
                            "duration": duration,
                            "status": "error",
                            "error_type": type(e).__name__,
                        }
                    },
                )
                raise

        return wrapper

    return decorator


class RequestLogger:
    """Logger for MCP tool requests and responses"""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or get_logger("mcp.tools")

    def log_request(
        self,
        tool_name: str,
        request_data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> str:
        """Log MCP tool request"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]

        set_correlation_id(correlation_id)

        # Sanitize request data for logging
        sanitized_data = self._sanitize_request_data(request_data)

        self.logger.info(
            f"MCP tool request: {tool_name}",
            extra={
                "mcp_request": {
                    "tool": tool_name,
                    "correlation_id": correlation_id,
                    "request_data": sanitized_data,
                    "timestamp": time.time(),
                }
            },
        )

        return correlation_id

    def log_response(
        self,
        tool_name: str,
        success: bool,
        duration: float,
        result_metadata: dict[str, Any] | None = None,
        error: Exception | None = None,
        correlation_id: str | None = None,
    ):
        """Log MCP tool response"""
        correlation_id = correlation_id or get_correlation_id() or "unknown"

        log_data = {
            "mcp_response": {
                "tool": tool_name,
                "correlation_id": correlation_id,
                "success": success,
                "duration": duration,
                "timestamp": time.time(),
            }
        }

        if result_metadata:
            log_data["mcp_response"]["result_metadata"] = result_metadata

        if success:
            self.logger.info(
                f"MCP tool response: {tool_name} completed in {duration:.2f}s",
                extra=log_data,
            )
        else:
            log_data["mcp_response"]["error_type"] = (
                type(error).__name__ if error else "Unknown"
            )
            self.logger.error(
                f"MCP tool error: {tool_name} failed after {duration:.2f}s",
                extra=log_data,
                exc_info=error is not None,
            )

    def _sanitize_request_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize request data for safe logging"""
        sanitized = {}

        for key, value in data.items():
            if isinstance(value, str):
                # Limit string length and check for sensitive data
                if any(
                    sensitive in key.lower()
                    for sensitive in ["password", "token", "key", "secret"]
                ):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = value[:500] + ("..." if len(value) > 500 else "")
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_request_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    str(item)[:100] for item in value[:10]
                ]  # Limit list size and item length
            else:
                sanitized[key] = str(value)[:100]

        return sanitized


def mcp_tool_logger(
    tool_name: str | None = None,
    include_request_data: bool = True,
    include_response_data: bool = True,
) -> Callable[[F], F]:
    """Decorator for MCP tool logging that handles both sync and async functions"""

    def decorator(func: F) -> F:
        import inspect

        name = tool_name or func.__name__
        request_logger = RequestLogger()

        # Check if the function is async
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Extract request data for logging
                request_data = {}
                if include_request_data and args:
                    # First arg is typically the request object
                    if hasattr(args[0], "__dict__"):
                        request_data = {
                            k: v
                            for k, v in args[0].__dict__.items()
                            if not k.startswith("_")
                        }

                correlation_id = request_logger.log_request(name, request_data)
                start_time = time.time()

                try:
                    result = await func(*args, **kwargs)  # Await async function
                    duration = time.time() - start_time

                    # Prepare response metadata
                    response_metadata = {}
                    if include_response_data and result is not None:
                        response_metadata["result_type"] = type(result).__name__
                        if hasattr(result, "__len__"):
                            response_metadata["result_length"] = len(result)

                    request_logger.log_response(
                        name, True, duration, response_metadata, None, correlation_id
                    )

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    request_logger.log_response(
                        name, False, duration, None, e, correlation_id
                    )
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Extract request data for logging
                request_data = {}
                if include_request_data and args:
                    # First arg is typically the request object
                    if hasattr(args[0], "__dict__"):
                        request_data = {
                            k: v
                            for k, v in args[0].__dict__.items()
                            if not k.startswith("_")
                        }

                correlation_id = request_logger.log_request(name, request_data)
                start_time = time.time()

                try:
                    result = func(*args, **kwargs)  # Call sync function normally
                    duration = time.time() - start_time

                    # Prepare response metadata
                    response_metadata = {}
                    if include_response_data and result is not None:
                        response_metadata["result_type"] = type(result).__name__
                        if hasattr(result, "__len__"):
                            response_metadata["result_length"] = len(result)

                    request_logger.log_response(
                        name, True, duration, response_metadata, None, correlation_id
                    )

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    request_logger.log_response(
                        name, False, duration, None, e, correlation_id
                    )
                    raise

            return sync_wrapper

    return decorator


# Convenience functions
def get_request_logger() -> RequestLogger:
    """Get a request logger instance"""
    return RequestLogger()


def log_session_event(
    event_type: str,
    session_id: str,
    details: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
):
    """Log session lifecycle events"""
    session_logger = logger or get_logger("session")

    event_data = {
        "session_event": {
            "event_type": event_type,
            "session_id": session_id,
            "timestamp": time.time(),
        }
    }

    if details:
        event_data["session_event"]["details"] = details

    session_logger.info(f"Session {event_type}: {session_id}", extra=event_data)


def log_performance_metric(
    metric_name: str,
    value: int | float,
    unit: str = "count",
    metadata: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
):
    """Log performance metrics"""
    perf_logger = logger or get_logger("performance")

    if not is_performance_logging_enabled():
        return

    metric_data = {
        "performance_metric": {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": time.time(),
        }
    }

    if metadata:
        metric_data["performance_metric"]["metadata"] = metadata

    perf_logger.debug(f"Metric: {metric_name} = {value} {unit}", extra=metric_data)


# Export main classes and functions
__all__ = [
    "OperationLogger",
    "OperationMetrics",
    "log_operation",
    "logged_operation",
    "async_logged_operation",
    "correlation_context",
    "performance_timer",
    "RequestLogger",
    "mcp_tool_logger",
    "get_request_logger",
    "log_session_event",
    "log_performance_metric",
]
