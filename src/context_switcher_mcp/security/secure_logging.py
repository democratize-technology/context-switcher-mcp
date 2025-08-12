"""Secure logging utilities to prevent log injection and information leakage"""

import logging
from ..logging_base import get_logger
import re
import json
import hashlib
from typing import Any, Dict, Optional
from datetime import datetime, timezone

# Configure secure logger
logger = get_logger(__name__)


class SecureLogFormatter(logging.Formatter):
    """Custom log formatter that sanitizes log messages"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Patterns to sanitize from log messages
        self.sensitive_patterns = [
            (r"password\s*[=:]\s*[^\s]+", "password=***REDACTED***"),
            (r"token\s*[=:]\s*[^\s]+", "token=***REDACTED***"),
            (r"key\s*[=:]\s*[^\s]+", "key=***REDACTED***"),
            (r"secret\s*[=:]\s*[^\s]+", "secret=***REDACTED***"),
            (r"apikey\s*[=:]\s*[^\s]+", "apikey=***REDACTED***"),
            (r"authorization\s*[=:]\s*[^\s]+", "authorization=***REDACTED***"),
            (r"session_id\s*[=:]\s*[^\s]+", "session_id=***REDACTED***"),
            (r"client_id\s*[=:]\s*[^\s]+", "client_id=***REDACTED***"),
        ]

        # Control characters that could be used for log injection
        self.control_chars = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

        # ANSI escape sequences
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sanitization"""
        # Sanitize the message
        if hasattr(record, "msg") and record.msg:
            record.msg = self.sanitize_log_message(str(record.msg))

        # Sanitize args if present
        if hasattr(record, "args") and record.args:
            sanitized_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    sanitized_args.append(self.sanitize_log_message(arg))
                else:
                    sanitized_args.append(arg)
            record.args = tuple(sanitized_args)

        return super().format(record)

    def sanitize_log_message(self, message: str) -> str:
        """Sanitize a log message for security"""
        if not isinstance(message, str):
            return str(message)

        sanitized = message

        # Remove control characters and ANSI escape sequences
        sanitized = self.control_chars.sub("", sanitized)
        sanitized = self.ansi_escape.sub("", sanitized)

        # Remove or mask sensitive information
        for pattern, replacement in self.sensitive_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        # Limit message length to prevent log overflow
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[TRUNCATED]"

        # Escape newlines to prevent log injection
        sanitized = sanitized.replace("\n", "\\n").replace("\r", "\\r")

        return sanitized


class SecureLogger:
    """Secure logging wrapper with additional security features"""

    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.security_logger = get_logger(f"{name}.security")

        # Set up secure formatter if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = SecureLogFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        risk_level: str = "medium",
        session_id: Optional[str] = None,
    ) -> None:
        """
        Log security event with structured data

        Args:
            event_type: Type of security event
            details: Event details (will be sanitized)
            risk_level: Risk level (low, medium, high, critical)
            session_id: Optional session ID (will be hashed)
        """
        # Sanitize event details
        sanitized_details = self._sanitize_dict(details)

        # Hash session ID for privacy
        hashed_session_id = None
        if session_id:
            hashed_session_id = hashlib.sha256(session_id.encode()).hexdigest()[:8]

        security_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "risk_level": risk_level,
            "session_hash": hashed_session_id,
            "details": sanitized_details,
        }

        # Log as structured JSON
        self.security_logger.warning(
            "SECURITY_EVENT: %s", json.dumps(security_event, default=str)
        )

    def log_validation_failure(
        self,
        content_type: str,
        validation_errors: list,
        content_preview: str,
        risk_level: str = "medium",
    ) -> None:
        """
        Log input validation failure

        Args:
            content_type: Type of content being validated
            validation_errors: List of validation errors
            content_preview: Preview of content (will be sanitized)
            risk_level: Risk level assessment
        """
        # Sanitize content preview
        safe_preview = self._sanitize_content_preview(content_preview)

        validation_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "validation_failure",
            "content_type": content_type,
            "risk_level": risk_level,
            "error_count": len(validation_errors),
            "errors": validation_errors[:5],  # Limit to first 5 errors
            "content_preview": safe_preview,
        }

        self.security_logger.warning(
            "VALIDATION_FAILURE: %s", json.dumps(validation_event, default=str)
        )

    def log_rate_limit_exceeded(
        self, client_identifier: str, operation: str, attempt_count: int
    ) -> None:
        """
        Log rate limiting event

        Args:
            client_identifier: Client identifier (will be hashed)
            operation: Operation being rate limited
            attempt_count: Number of attempts
        """
        # Hash client identifier
        hashed_client = hashlib.sha256(client_identifier.encode()).hexdigest()[:8]

        rate_limit_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "rate_limit_exceeded",
            "client_hash": hashed_client,
            "operation": operation,
            "attempt_count": attempt_count,
            "risk_level": "medium",
        }

        self.security_logger.warning(
            "RATE_LIMIT_EXCEEDED: %s", json.dumps(rate_limit_event, default=str)
        )

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize dictionary data for logging"""
        sanitized = {}

        for key, value in data.items():
            # Sanitize key
            safe_key = self._sanitize_string(str(key))

            # Sanitize value based on type
            if isinstance(value, str):
                safe_value = self._sanitize_string(value)
            elif isinstance(value, dict):
                safe_value = self._sanitize_dict(value)
            elif isinstance(value, list):
                safe_value = [
                    self._sanitize_string(str(item)) for item in value[:10]
                ]  # Limit list size
            else:
                safe_value = str(value)

            sanitized[safe_key] = safe_value

        return sanitized

    def _sanitize_string(self, text: str) -> str:
        """Sanitize string for safe logging"""
        if not isinstance(text, str):
            text = str(text)

        # Remove control characters
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[TRUNCATED]"

        # Escape special characters for JSON
        sanitized = (
            sanitized.replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        )

        return sanitized

    def _sanitize_content_preview(self, content: str, max_length: int = 200) -> str:
        """Create safe preview of content for logging"""
        if not content:
            return ""

        # Take first part of content
        preview = content[:max_length]

        # Remove sensitive patterns
        sensitive_words = [
            "password",
            "token",
            "key",
            "secret",
            "auth",
            "session",
            "cookie",
        ]

        preview_lower = preview.lower()
        for word in sensitive_words:
            if word in preview_lower:
                preview = "[CONTENT_CONTAINS_SENSITIVE_DATA]"
                break

        return self._sanitize_string(preview)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message with sanitization"""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message with sanitization"""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message with sanitization"""
        self.logger.error(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message with sanitization"""
        self.logger.debug(message, *args, **kwargs)


def get_secure_logger(name: str) -> SecureLogger:
    """Get a secure logger instance"""
    return SecureLogger(name)


def setup_secure_logging() -> None:
    """Set up secure logging configuration for the entire application"""
    # Get root logger
    root_logger = logging.getLogger()

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add secure handler
    handler = logging.StreamHandler()
    formatter = SecureLogFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Set appropriate log level
    root_logger.setLevel(logging.INFO)

    # Create separate security logger with higher visibility
    security_logger = get_logger("security")
    security_handler = logging.StreamHandler()
    security_formatter = SecureLogFormatter(
        "SECURITY - %(asctime)s - %(levelname)s - %(message)s"
    )
    security_handler.setFormatter(security_formatter)
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.WARNING)

    logger.info("Secure logging configured successfully")


# Export main classes and functions
__all__ = [
    "SecureLogger",
    "SecureLogFormatter",
    "get_secure_logger",
    "setup_secure_logging",
]
