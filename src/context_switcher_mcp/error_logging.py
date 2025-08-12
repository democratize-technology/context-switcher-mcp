"""Structured error logging utilities for consistent error reporting"""

import logging
from .logging_base import get_logger
import time
import uuid
from typing import Any, Dict, List, Optional

from .error_classification import classify_error, ErrorSeverity
from .input_sanitizer import sanitize_error_message
from .security_context_sanitizer import (
    sanitize_exception_context,
    sanitize_context_dict,
)


class StructuredErrorLogger:
    """Structured error logger with consistent formatting and context tracking."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        include_stack_trace: bool = True,
        include_performance_metrics: bool = True,
        correlation_id_generator: Optional[callable] = None,
    ):
        """Initialize structured error logger.

        Args:
            logger: Logger instance to use (creates default if None)
            include_stack_trace: Whether to include stack traces in error logs
            include_performance_metrics: Whether to track performance impact of errors
            correlation_id_generator: Function to generate correlation IDs
        """
        self.logger = logger or get_logger(__name__)
        self.include_stack_trace = include_stack_trace
        self.include_performance_metrics = include_performance_metrics
        self.correlation_id_generator = correlation_id_generator or (
            lambda: str(uuid.uuid4())[:8]
        )

        # Performance tracking
        self.error_metrics = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "start_time": time.time(),
        }

    def log_error(
        self,
        error: Exception,
        operation_name: str = "unknown_operation",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> str:
        """Log an error with structured context information.

        Args:
            error: The exception to log
            operation_name: Name of the operation that failed
            session_id: Optional session ID for correlation
            user_id: Optional user ID for correlation
            additional_context: Additional context information
            correlation_id: Optional correlation ID (generated if None)
            duration: Optional operation duration in seconds

        Returns:
            Correlation ID for tracking the error
        """
        if correlation_id is None:
            correlation_id = self.correlation_id_generator()

        # Get error classification
        classification = classify_error(error)

        # Build structured log entry
        log_entry = self._build_log_entry(
            error=error,
            operation_name=operation_name,
            session_id=session_id,
            user_id=user_id,
            additional_context=additional_context or {},
            correlation_id=correlation_id,
            classification=classification,
            duration=duration,
        )

        # Update metrics
        self._update_error_metrics(classification)

        # Log with appropriate level
        log_level = self._get_log_level(classification)
        self.logger.log(
            log_level,
            f"[{correlation_id}] {operation_name} failed: {sanitize_error_message(str(error))}",
            extra={"structured_error": log_entry},
            exc_info=self.include_stack_trace,
        )

        return correlation_id

    def log_error_chain(
        self,
        error: Exception,
        operation_name: str = "unknown_operation",
        session_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log an error and its entire exception chain.

        Args:
            error: The exception to log
            operation_name: Name of the operation that failed
            session_id: Optional session ID for correlation
            additional_context: Additional context information

        Returns:
            Correlation ID for tracking the error chain
        """
        correlation_id = self.correlation_id_generator()

        # Extract exception chain
        error_chain = self._extract_exception_chain(error)

        # Log the main error
        main_log_entry = self._build_log_entry(
            error=error,
            operation_name=operation_name,
            session_id=session_id,
            additional_context=additional_context or {},
            correlation_id=correlation_id,
            classification=classify_error(error),
            error_chain=error_chain,
        )

        # Log main error
        classification = classify_error(error)
        log_level = self._get_log_level(classification)

        self.logger.log(
            log_level,
            f"[{correlation_id}] Error chain in {operation_name}: {len(error_chain)} errors",
            extra={"structured_error": main_log_entry},
            exc_info=self.include_stack_trace,
        )

        # Log each error in the chain with context
        for i, chain_error in enumerate(error_chain):
            chain_classification = classify_error(chain_error["exception"])
            self.logger.log(
                logging.DEBUG,
                f"[{correlation_id}] Chain error {i + 1}/{len(error_chain)}: {chain_error['type']}",
                extra={
                    "structured_error": {
                        "correlation_id": correlation_id,
                        "chain_position": i + 1,
                        "chain_total": len(error_chain),
                        "error_type": chain_error["type"],
                        "error_message": sanitize_error_message(chain_error["message"]),
                        "classification": chain_classification,
                    }
                },
            )

        self._update_error_metrics(classification)
        return correlation_id

    def log_performance_error(
        self,
        error: Exception,
        operation_name: str,
        duration: float,
        performance_threshold: float,
        session_id: Optional[str] = None,
        performance_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a performance-related error with timing context.

        Args:
            error: The exception to log
            operation_name: Name of the slow operation
            duration: Actual duration in seconds
            performance_threshold: Expected threshold in seconds
            session_id: Optional session ID
            performance_context: Additional performance context

        Returns:
            Correlation ID for tracking the error
        """
        correlation_id = self.correlation_id_generator()

        perf_context = {
            "duration": duration,
            "threshold": performance_threshold,
            "threshold_exceeded_by": duration - performance_threshold,
            "performance_ratio": duration / performance_threshold
            if performance_threshold > 0
            else float("inf"),
        }

        if performance_context:
            perf_context.update(performance_context)

        # Log with performance-specific context
        log_entry = self._build_log_entry(
            error=error,
            operation_name=operation_name,
            session_id=session_id,
            additional_context={"performance": perf_context},
            correlation_id=correlation_id,
            classification=classify_error(error),
            duration=duration,
        )

        self.logger.error(
            f"[{correlation_id}] Performance error in {operation_name}: "
            f"{duration:.2f}s (threshold: {performance_threshold:.2f}s)",
            extra={"structured_error": log_entry},
        )

        return correlation_id

    def _build_log_entry(
        self,
        error: Exception,
        operation_name: str,
        session_id: Optional[str],
        user_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        correlation_id: str = None,
        classification: Optional[Dict[str, Any]] = None,
        duration: Optional[float] = None,
        error_chain: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build structured log entry."""
        log_entry = {
            "timestamp": time.time(),
            "correlation_id": correlation_id,
            "operation_name": operation_name,
            "error": {
                "type": type(error).__name__,
                "message": sanitize_error_message(str(error)),
                "module": getattr(error, "__module__", "unknown"),
            },
        }

        # Add session context
        if session_id:
            log_entry["session_id"] = session_id

        if user_id:
            log_entry["user_id"] = user_id

        # Add classification info
        if classification:
            log_entry["classification"] = {
                "severity": classification["severity"].value
                if hasattr(classification["severity"], "value")
                else str(classification["severity"]),
                "category": classification["category"].value
                if hasattr(classification["category"], "value")
                else str(classification["category"]),
                "transient": classification.get("transient", False),
                "retriable": classification.get("retriable", False),
                "user_facing": classification.get("user_facing", False),
            }

        # Add performance metrics
        if self.include_performance_metrics:
            perf_metrics = {
                "total_errors_session": self.error_metrics["total_errors"],
                "uptime_hours": (time.time() - self.error_metrics["start_time"]) / 3600,
            }
            if duration:
                perf_metrics["operation_duration"] = duration
            log_entry["performance_metrics"] = perf_metrics

        # Add error-specific context with sanitization
        sanitized_exception_context = sanitize_exception_context(error)
        if sanitized_exception_context:
            log_entry["sanitized_context"] = sanitized_exception_context

        # Add additional context with sanitization
        if additional_context:
            sanitized_additional = sanitize_context_dict(additional_context, "generic")
            log_entry["additional_context"] = sanitized_additional

        # Add error chain if available
        if error_chain:
            log_entry["error_chain"] = error_chain

        return log_entry

    def _extract_exception_chain(self, error: Exception) -> List[Dict[str, Any]]:
        """Extract the full exception chain."""
        chain = []
        current = error

        while current is not None:
            chain.append(
                {
                    "type": type(current).__name__,
                    "message": str(current),
                    "exception": current,  # Keep reference for classification
                }
            )
            current = current.__cause__ or current.__context__

        return chain

    def _update_error_metrics(self, classification: Dict[str, Any]):
        """Update internal error metrics."""
        self.error_metrics["total_errors"] += 1

        error_type = classification["error_type"]
        self.error_metrics["errors_by_type"][error_type] = (
            self.error_metrics["errors_by_type"].get(error_type, 0) + 1
        )

        severity = str(classification["severity"])
        self.error_metrics["errors_by_severity"][severity] = (
            self.error_metrics["errors_by_severity"].get(severity, 0) + 1
        )

    def _get_log_level(self, classification: Dict[str, Any]) -> int:
        """Get appropriate logging level for error."""
        severity = classification.get("severity", ErrorSeverity.MEDIUM)

        if severity == ErrorSeverity.CRITICAL:
            return logging.CRITICAL
        elif severity == ErrorSeverity.HIGH:
            return logging.ERROR
        elif severity == ErrorSeverity.MEDIUM:
            return logging.WARNING
        elif severity == ErrorSeverity.LOW:
            return logging.INFO
        else:
            return logging.DEBUG

    def get_error_metrics(self) -> Dict[str, Any]:
        """Get current error metrics."""
        return self.error_metrics.copy()

    def reset_metrics(self):
        """Reset error metrics."""
        self.error_metrics = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "start_time": time.time(),
        }

    # Delegate standard logging methods to underlying logger
    def debug(self, msg, *args, **kwargs):
        """Delegate debug logging to underlying logger."""
        return self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Delegate info logging to underlying logger."""
        return self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Delegate warning logging to underlying logger."""
        return self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Delegate error logging to underlying logger."""
        return self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Delegate critical logging to underlying logger."""
        return self.logger.critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        """Delegate generic logging to underlying logger."""
        return self.logger.log(level, msg, *args, **kwargs)


# Global structured logger instance
_global_error_logger = None


def get_structured_logger() -> StructuredErrorLogger:
    """Get global structured error logger instance."""
    global _global_error_logger
    if _global_error_logger is None:
        _global_error_logger = StructuredErrorLogger()
    return _global_error_logger


def log_error_with_context(
    error: Exception,
    operation_name: str = "unknown_operation",
    session_id: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None,
    logger_instance: Optional[StructuredErrorLogger] = None,
) -> str:
    """Convenience function to log an error with context.

    Args:
        error: The exception to log
        operation_name: Name of the operation that failed
        session_id: Optional session ID for correlation
        additional_context: Additional context information
        logger_instance: Optional logger instance (uses global if None)

    Returns:
        Correlation ID for tracking the error
    """
    logger = logger_instance or get_structured_logger()
    return logger.log_error(
        error=error,
        operation_name=operation_name,
        session_id=session_id,
        additional_context=additional_context,
    )


def log_performance_error_with_context(
    error: Exception,
    operation_name: str,
    duration: float,
    threshold: float,
    session_id: Optional[str] = None,
    performance_context: Optional[Dict[str, Any]] = None,
    logger_instance: Optional[StructuredErrorLogger] = None,
) -> str:
    """Convenience function to log a performance error.

    Args:
        error: The exception to log
        operation_name: Name of the slow operation
        duration: Actual duration in seconds
        threshold: Expected threshold in seconds
        session_id: Optional session ID
        performance_context: Additional performance context
        logger_instance: Optional logger instance (uses global if None)

    Returns:
        Correlation ID for tracking the error
    """
    logger = logger_instance or get_structured_logger()
    return logger.log_performance_error(
        error=error,
        operation_name=operation_name,
        duration=duration,
        performance_threshold=threshold,
        session_id=session_id,
        performance_context=performance_context,
    )


class ErrorContextFilter(logging.Filter):
    """Logging filter to add error context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add error context to log records."""
        # Add default error context if not present
        if not hasattr(record, "structured_error"):
            if hasattr(record, "exc_info") and record.exc_info:
                error = record.exc_info[1]
                if error:
                    classification = classify_error(error)
                    record.structured_error = {
                        "error_type": type(error).__name__,
                        "error_message": sanitize_error_message(str(error)),
                        "severity": classification.get(
                            "severity", ErrorSeverity.MEDIUM
                        ).value,
                        "category": classification.get("category", "unknown").value
                        if hasattr(classification.get("category", "unknown"), "value")
                        else str(classification.get("category", "unknown")),
                        "timestamp": time.time(),
                    }

        return True


def setup_structured_error_logging(
    logger: logging.Logger, include_context_filter: bool = True
) -> StructuredErrorLogger:
    """Setup structured error logging for a logger.

    Args:
        logger: Logger to setup
        include_context_filter: Whether to add error context filter

    Returns:
        Configured StructuredErrorLogger instance
    """
    if include_context_filter:
        logger.addFilter(ErrorContextFilter())

    return StructuredErrorLogger(logger=logger)
