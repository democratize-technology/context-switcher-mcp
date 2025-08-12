"""
Centralized logging configuration system for Context Switcher MCP

This module provides unified logging configuration that builds upon the existing
secure logging infrastructure while standardizing patterns across the codebase.
"""

import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextvars import ContextVar
from datetime import datetime

from .security.secure_logging import (
    SecureLogger,
    SecureLogFormatter,
    setup_secure_logging,
)
from .error_logging import StructuredErrorLogger, setup_structured_error_logging


# Context variable for correlation ID tracking
correlation_id_context: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)

# Global configuration
_logging_configured = False
_loggers_cache: Dict[str, logging.Logger] = {}


class ContextSwitcherLogFormatter(SecureLogFormatter):
    """Enhanced log formatter with correlation ID and structured context"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with correlation ID and context"""
        # Get correlation ID from context
        correlation_id = correlation_id_context.get()
        if correlation_id:
            record.correlation_id = correlation_id
        else:
            record.correlation_id = "no-correlation"

        # Add timestamp in ISO format
        record.timestamp_iso = datetime.utcnow().isoformat() + "Z"

        # Call parent formatting (includes security sanitization)
        formatted_message = super().format(record)

        return formatted_message


class JSONLogFormatter(ContextSwitcherLogFormatter):
    """JSON log formatter for structured logging in production"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        # Get correlation ID
        correlation_id = correlation_id_context.get()

        # Build structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": self.sanitize_log_message(record.getMessage()),
            "correlation_id": correlation_id or "no-correlation",
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__
                if record.exc_info[0]
                else "Unknown",
                "message": self.sanitize_log_message(str(record.exc_info[1]))
                if record.exc_info[1]
                else "No message",
                "traceback": self.format_exception(record.exc_info)
                if record.exc_info
                else None,
            }

        # Add structured error context if present
        if hasattr(record, "structured_error"):
            log_entry["structured_error"] = record.structured_error

        # Add performance metrics if present
        if hasattr(record, "performance_metrics"):
            log_entry["performance_metrics"] = record.performance_metrics

        # Add extra context
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
                "correlation_id",
                "timestamp_iso",
                "structured_error",
                "performance_metrics",
            ]:
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class LoggingConfig:
    """Centralized logging configuration"""

    def __init__(self):
        self.config = self._load_config()
        self._structured_error_logger: Optional[StructuredErrorLogger] = None
        self._secure_loggers: Dict[str, SecureLogger] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load logging configuration from environment and defaults"""
        return {
            # Log levels
            "level": os.getenv("LOG_LEVEL", "INFO").upper(),
            "security_level": os.getenv("SECURITY_LOG_LEVEL", "WARNING").upper(),
            "performance_level": os.getenv("PERFORMANCE_LOG_LEVEL", "DEBUG").upper(),
            # Output format
            "format": os.getenv("LOG_FORMAT", "standard"),  # standard, json, detailed
            "include_correlation_ids": os.getenv("LOG_CORRELATION_IDS", "true").lower()
            == "true",
            # Output destination
            "output": os.getenv("LOG_OUTPUT", "console"),  # console, file, both
            "file_path": os.getenv("LOG_FILE_PATH", "/tmp/context_switcher.log"),
            # Features
            "structured_errors": os.getenv("LOG_STRUCTURED_ERRORS", "true").lower()
            == "true",
            "performance_logging": os.getenv("LOG_PERFORMANCE", "true").lower()
            == "true",
            "security_logging": os.getenv("LOG_SECURITY", "true").lower() == "true",
            # Development features
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
            "verbose_errors": os.getenv("LOG_VERBOSE_ERRORS", "false").lower()
            == "true",
            # Production features
            "json_logs": os.getenv("LOG_JSON", "false").lower() == "true",
            "log_sampling": float(os.getenv("LOG_SAMPLING_RATE", "1.0")),
            # Buffer settings
            "buffer_size": int(os.getenv("LOG_BUFFER_SIZE", "1000")),
            "flush_interval": int(os.getenv("LOG_FLUSH_INTERVAL", "5")),
        }

    def setup_logging(self) -> None:
        """Setup centralized logging configuration"""
        global _logging_configured

        if _logging_configured:
            return

        # Clear any existing configuration
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Setup secure logging as base
        if self.config["security_logging"]:
            setup_secure_logging()

        # Configure root logger level
        log_level = getattr(logging, self.config["level"], logging.INFO)
        root_logger.setLevel(log_level)

        # Create formatter based on configuration
        if self.config["json_logs"] or self.config["format"] == "json":
            formatter = JSONLogFormatter()
        elif self.config["format"] == "detailed":
            formatter = ContextSwitcherLogFormatter(
                "%(timestamp_iso)s - [%(correlation_id)s] - %(name)s - %(levelname)s - "
                "%(module)s:%(funcName)s:%(lineno)d - %(message)s"
            )
        else:
            formatter = ContextSwitcherLogFormatter(
                "%(asctime)s - [%(correlation_id)s] - %(name)s - %(levelname)s - %(message)s"
            )

        # Setup console handler
        if self.config["output"] in ["console", "both"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            root_logger.addHandler(console_handler)

        # Setup file handler
        if self.config["output"] in ["file", "both"]:
            file_path = Path(self.config["file_path"])
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)

        # Setup performance logger
        if self.config["performance_logging"]:
            perf_logger = logging.getLogger("performance")
            perf_level = getattr(
                logging, self.config["performance_level"], logging.DEBUG
            )
            perf_logger.setLevel(perf_level)

        # Setup security logger with enhanced formatting
        if self.config["security_logging"]:
            security_logger = logging.getLogger("security")
            security_level = getattr(
                logging, self.config["security_level"], logging.WARNING
            )
            security_logger.setLevel(security_level)

            # Add special security handler if needed
            if self.config["output"] == "file" or self.config["output"] == "both":
                security_file = file_path.parent / "security.log"
                security_handler = logging.FileHandler(security_file)
                security_handler.setFormatter(formatter)
                security_handler.setLevel(security_level)
                security_logger.addHandler(security_handler)

        # Setup structured error logging
        if self.config["structured_errors"]:
            self._structured_error_logger = StructuredErrorLogger(
                logger=logging.getLogger("errors"),
                include_stack_trace=self.config["verbose_errors"],
                include_performance_metrics=self.config["performance_logging"],
            )

        _logging_configured = True

        # Log successful configuration
        logger = logging.getLogger(__name__)
        logger.info(
            "Unified logging configuration complete",
            extra={
                "config": {
                    "level": self.config["level"],
                    "format": self.config["format"],
                    "output": self.config["output"],
                    "features": {
                        "structured_errors": self.config["structured_errors"],
                        "performance_logging": self.config["performance_logging"],
                        "security_logging": self.config["security_logging"],
                        "correlation_ids": self.config["include_correlation_ids"],
                    },
                }
            },
        )

    def get_logger(
        self, name: str, secure: bool = False
    ) -> Union[logging.Logger, SecureLogger]:
        """Get a configured logger instance"""
        if not _logging_configured:
            self.setup_logging()

        if secure:
            if name not in self._secure_loggers:
                self._secure_loggers[name] = SecureLogger(name)
            return self._secure_loggers[name]

        if name not in _loggers_cache:
            logger = logging.getLogger(name)

            # Add structured error logging if enabled
            if self.config["structured_errors"] and self._structured_error_logger:
                logger = setup_structured_error_logging(
                    logger, include_context_filter=True
                )

            _loggers_cache[name] = logger

        return _loggers_cache[name]

    def get_structured_error_logger(self) -> Optional[StructuredErrorLogger]:
        """Get the structured error logger instance"""
        if not _logging_configured:
            self.setup_logging()
        return self._structured_error_logger

    def is_level_enabled(self, level: str) -> bool:
        """Check if a log level is enabled"""
        current_level = getattr(logging, self.config["level"], logging.INFO)
        check_level = getattr(logging, level.upper(), logging.NOTSET)
        return check_level >= current_level

    def update_correlation_id(self, correlation_id: Optional[str]) -> None:
        """Update the correlation ID in context"""
        correlation_id_context.set(correlation_id)

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID from context"""
        return correlation_id_context.get()


# Global configuration instance
_config_instance: Optional[LoggingConfig] = None


def get_logging_config() -> LoggingConfig:
    """Get global logging configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = LoggingConfig()
    return _config_instance


def setup_logging() -> None:
    """Setup unified logging configuration (convenience function)"""
    config = get_logging_config()
    config.setup_logging()


def get_logger(name: str, secure: bool = False) -> Union[logging.Logger, SecureLogger]:
    """Get a configured logger (convenience function)"""
    config = get_logging_config()
    return config.get_logger(name, secure=secure)


def get_structured_error_logger() -> Optional[StructuredErrorLogger]:
    """Get structured error logger (convenience function)"""
    config = get_logging_config()
    return config.get_structured_error_logger()


def set_correlation_id(correlation_id: Optional[str]) -> None:
    """Set correlation ID for current context (convenience function)"""
    correlation_id_context.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID (convenience function)"""
    return correlation_id_context.get()


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled (convenience function)"""
    config = get_logging_config()
    return config.is_level_enabled("DEBUG")


def is_performance_logging_enabled() -> bool:
    """Check if performance logging is enabled"""
    config = get_logging_config()
    return config.config["performance_logging"] and config.is_level_enabled(
        config.config["performance_level"]
    )


# Export main functions
__all__ = [
    "LoggingConfig",
    "setup_logging",
    "get_logger",
    "get_structured_error_logger",
    "set_correlation_id",
    "get_correlation_id",
    "is_debug_enabled",
    "is_performance_logging_enabled",
    "ContextSwitcherLogFormatter",
    "JSONLogFormatter",
]
