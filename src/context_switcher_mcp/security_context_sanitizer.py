"""Context sanitization utilities for secure error logging and handling

This module provides comprehensive sanitization of sensitive context data
before logging to prevent information disclosure vulnerabilities.
"""

import re
import hashlib
from .logging_base import get_logger
from typing import Any, Dict
from datetime import datetime, timezone

logger = get_logger(__name__)


class SecurityContextSanitizer:
    """Sanitizes sensitive context data for secure logging"""

    def __init__(self):
        """Initialize the sanitizer with security patterns"""
        # Sensitive key patterns (case-insensitive)
        self.sensitive_key_patterns = {
            r".*password.*",
            r".*secret.*",
            r".*key.*",
            r".*token.*",
            r".*credential.*",
            r".*auth.*",
            r".*api.*key.*",
            r".*access.*key.*",
            r".*private.*key.*",
            r".*session.*key.*",
            r".*bearer.*",
            r".*authorization.*",
            r".*jwt.*",
            r".*cookie.*",
            r".*x-api-key.*",
            r".*signature.*",
            r".*cert.*",
            r".*pem.*",
        }

        # Sensitive value patterns
        self.sensitive_value_patterns = [
            # API keys (various formats)
            (r"sk-[a-zA-Z0-9]{32,}", "***API_KEY***"),  # OpenAI format
            (r"pk-[a-zA-Z0-9]{32,}", "***PUBLIC_KEY***"),
            (r"AKIA[0-9A-Z]{16}", "***AWS_ACCESS_KEY***"),  # AWS access key
            (r"[A-Za-z0-9+/]{40}", "***SECRET_KEY***"),  # Base64-like secrets
            # JWT tokens
            (
                r"eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*",
                "***JWT_TOKEN***",
            ),
            # UUIDs (potential session IDs)
            (
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                "***UUID***",
            ),
            # Hashes and tokens (32+ hex chars)
            (r"[0-9a-f]{32,}", "***HASH_TOKEN***"),
            # URLs with credentials
            (r"://[^:]+:[^@]+@", "://***:***@"),
            # File paths that could reveal system structure
            (r"/home/[^/\s]+", "/home/***"),
            (r"/Users/[^/\s]+", "/Users/***"),
            (r"C:\\Users\\[^\\]+", r"C:\\Users\\***"),
        ]

        # Context-specific sanitization rules
        self.context_sanitization_rules = {
            "security_context": self._sanitize_security_context,
            "network_context": self._sanitize_network_context,
            "performance_context": self._sanitize_performance_context,
            "validation_context": self._sanitize_validation_context,
            "concurrency_context": self._sanitize_concurrency_context,
        }

        # Safe keys that can be logged without sanitization
        self.safe_keys = {
            "operation_name",
            "operation_type",
            "operation_id",
            "timestamp",
            "duration",
            "status",
            "thread_count",
            "retry_count",
            "attempt_number",
            "error_type",
            "error_category",
            "severity",
            "correlation_id",
            "request_id",
            "method",
            "status_code",
            "content_type",
            "schema_version",
            "validation_type",
            "lock_type",
            "thread_id",
        }

    def sanitize_context_dict(
        self, context: Dict[str, Any], context_type: str = "generic"
    ) -> Dict[str, Any]:
        """
        Sanitize a context dictionary for secure logging

        Args:
            context: The context dictionary to sanitize
            context_type: Type of context for specialized sanitization

        Returns:
            Sanitized context dictionary safe for logging
        """
        if not isinstance(context, dict):
            return {"sanitized": True, "original_type": type(context).__name__}

        # Apply context-specific sanitization if available
        if context_type in self.context_sanitization_rules:
            return self.context_sanitization_rules[context_type](context)

        return self._sanitize_generic_context(context)

    def sanitize_exception_context(self, exception: Exception) -> Dict[str, Any]:
        """
        Sanitize context from an exception object

        Args:
            exception: Exception with potential context attributes

        Returns:
            Sanitized context dictionary
        """
        sanitized = {
            "exception_type": type(exception).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Check for various context attributes
        context_attrs = [
            ("security_context", "security_context"),
            ("network_context", "network_context"),
            ("performance_context", "performance_context"),
            ("validation_context", "validation_context"),
            ("concurrency_context", "concurrency_context"),
        ]

        for attr_name, context_type in context_attrs:
            if hasattr(exception, attr_name):
                context_data = getattr(exception, attr_name)
                if context_data:
                    sanitized[attr_name] = self.sanitize_context_dict(
                        context_data, context_type
                    )

        return sanitized

    def _sanitize_generic_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generic context sanitization"""
        sanitized = {}

        for key, value in context.items():
            # Check if key is safe to log as-is
            if key in self.safe_keys:
                sanitized[key] = self._sanitize_value(value, allow_full_value=True)
                continue

            # Check if key matches sensitive patterns
            key_lower = key.lower()
            is_sensitive_key = any(
                re.match(pattern, key_lower, re.IGNORECASE)
                for pattern in self.sensitive_key_patterns
            )

            if is_sensitive_key:
                sanitized[f"{key}_hash"] = self._hash_sensitive_data(str(value))
                sanitized[f"{key}_present"] = value is not None
            else:
                sanitized[key] = self._sanitize_value(value)

        return sanitized

    def _sanitize_security_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized sanitization for security context"""
        sanitized = {
            "context_type": "security",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Always hash potentially sensitive security data
        for key, value in context.items():
            if key in ["auth_type", "method", "status", "error_code"]:
                # These are safe to log
                sanitized[key] = str(value) if value is not None else None
            elif key in ["user_id", "client_id", "session_id"]:
                # Hash identifiers
                sanitized[f"{key}_hash"] = (
                    self._hash_sensitive_data(str(value)) if value else None
                )
            elif key in ["region", "service", "operation"]:
                # Safe operational context
                sanitized[key] = str(value) if value is not None else None
            else:
                # Everything else gets hashed for security
                sanitized[f"{key}_hash"] = (
                    self._hash_sensitive_data(str(value)) if value else None
                )
                sanitized[f"{key}_present"] = value is not None

        return sanitized

    def _sanitize_network_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized sanitization for network context"""
        sanitized = {
            "context_type": "network",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for key, value in context.items():
            if key in ["host", "hostname", "url"]:
                # Sanitize but preserve general structure
                sanitized[key] = (
                    self._sanitize_url_or_host(str(value)) if value else None
                )
            elif key in ["port", "status_code", "method", "timeout"]:
                # Safe to log
                sanitized[key] = value
            elif key in ["ip", "ip_address"]:
                # Hash IP addresses
                sanitized[f"{key}_hash"] = (
                    self._hash_sensitive_data(str(value)) if value else None
                )
            else:
                sanitized[key] = self._sanitize_value(value)

        return sanitized

    def _sanitize_performance_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized sanitization for performance context"""
        sanitized = {
            "context_type": "performance",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Performance context is generally safe but may contain session IDs
        for key, value in context.items():
            if key in [
                "duration",
                "timeout",
                "threshold",
                "retry_count",
                "thread_count",
                "memory_usage",
                "cpu_usage",
                "operation_name",
                "operation_type",
            ]:
                sanitized[key] = value
            elif key in ["session_id", "correlation_id", "request_id"]:
                # Hash identifiers
                sanitized[f"{key}_hash"] = (
                    self._hash_sensitive_data(str(value)) if value else None
                )
            else:
                sanitized[key] = self._sanitize_value(value)

        return sanitized

    def _sanitize_validation_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized sanitization for validation context"""
        sanitized = {
            "context_type": "validation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        for key, value in context.items():
            if key in [
                "field_name",
                "validation_type",
                "error_count",
                "schema_version",
            ]:
                sanitized[key] = value
            elif key in ["field_value", "input_data", "failed_value"]:
                # Sanitize potentially sensitive input data
                sanitized[f"{key}_sanitized"] = (
                    self._sanitize_input_data(str(value)) if value else None
                )
                sanitized[f"{key}_length"] = len(str(value)) if value else 0
            else:
                sanitized[key] = self._sanitize_value(value)

        return sanitized

    def _sanitize_concurrency_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Specialized sanitization for concurrency context"""
        sanitized = {
            "context_type": "concurrency",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Concurrency context is generally safe
        for key, value in context.items():
            if key in ["thread_id", "lock_name", "lock_type", "timeout", "retry_count"]:
                sanitized[key] = value
            elif key in ["session_id", "operation_id"]:
                # Hash identifiers
                sanitized[f"{key}_hash"] = (
                    self._hash_sensitive_data(str(value)) if value else None
                )
            else:
                sanitized[key] = self._sanitize_value(value)

        return sanitized

    def _sanitize_value(self, value: Any, allow_full_value: bool = False) -> Any:
        """Sanitize an individual value"""
        if value is None:
            return None

        if isinstance(value, dict):
            return self._sanitize_generic_context(value)
        elif isinstance(value, list):
            return [
                self._sanitize_value(item) for item in value[:10]
            ]  # Limit list size
        elif isinstance(value, (int, float, bool)):
            return value
        else:
            value_str = str(value)

            if not allow_full_value:
                # Apply sensitive value pattern sanitization
                for pattern, replacement in self.sensitive_value_patterns:
                    value_str = re.sub(
                        pattern, replacement, value_str, flags=re.IGNORECASE
                    )

            # Limit string length
            if len(value_str) > 500:
                value_str = value_str[:497] + "..."

            return value_str

    def _sanitize_url_or_host(self, url: str) -> str:
        """Sanitize URLs while preserving useful information"""
        # Remove credentials from URLs
        url = re.sub(r"://[^:]+:[^@]+@", "://***:***@", url)

        # Mask specific hosts but preserve domain structure
        url = re.sub(r"://([^./]+\.)*([^./]+\.[^./]+)(:\d+)?/", r"://***.\2\3/", url)

        return url

    def _sanitize_input_data(self, data: str) -> str:
        """Sanitize input data for safe logging"""
        # Check for potentially sensitive content
        sensitive_indicators = [
            "password",
            "secret",
            "token",
            "key",
            "auth",
            "credential",
            "login",
            "signin",
            "jwt",
        ]

        data_lower = data.lower()
        if any(indicator in data_lower for indicator in sensitive_indicators):
            return "[POTENTIALLY_SENSITIVE_DATA_REDACTED]"

        # Apply standard value sanitization
        return self._sanitize_value(data, allow_full_value=False)

    def _hash_sensitive_data(self, data: str) -> str:
        """Create a hash of sensitive data for logging correlation"""
        if not data:
            return None

        # Use SHA256 and truncate to 8 chars for logging correlation
        return hashlib.sha256(data.encode()).hexdigest()[:8]

    def get_sanitization_summary(
        self, original_context: Dict[str, Any], sanitized_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of sanitization actions taken"""
        return {
            "original_keys": len(original_context)
            if isinstance(original_context, dict)
            else 0,
            "sanitized_keys": len(sanitized_context)
            if isinstance(sanitized_context, dict)
            else 0,
            "sensitive_keys_found": sum(
                1 for key in sanitized_context.keys() if key.endswith("_hash")
            ),
            "sanitization_timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Global sanitizer instance
_global_sanitizer = None


def get_context_sanitizer() -> SecurityContextSanitizer:
    """Get global context sanitizer instance"""
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = SecurityContextSanitizer()
    return _global_sanitizer


def sanitize_exception_context(exception: Exception) -> Dict[str, Any]:
    """Convenience function to sanitize exception context"""
    sanitizer = get_context_sanitizer()
    return sanitizer.sanitize_exception_context(exception)


def sanitize_context_dict(
    context: Dict[str, Any], context_type: str = "generic"
) -> Dict[str, Any]:
    """Convenience function to sanitize context dictionary"""
    sanitizer = get_context_sanitizer()
    return sanitizer.sanitize_context_dict(context, context_type)


# Export main classes and functions
__all__ = [
    "SecurityContextSanitizer",
    "get_context_sanitizer",
    "sanitize_exception_context",
    "sanitize_context_dict",
]
