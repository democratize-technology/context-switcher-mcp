"""Enhanced input validation for additional security measures"""

import html
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

from ..logging_base import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for enhanced validation rules"""

    max_nesting_depth: int = 10
    max_array_length: int = 1000
    max_object_keys: int = 100
    max_unicode_categories: int = 5
    allow_html_entities: bool = False
    max_url_length: int = 2048
    max_email_length: int = 254


class EnhancedInputValidator:
    """Enhanced input validation with additional security checks"""

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    def validate_json_structure(
        self, json_str: str
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """
        Validate JSON structure for security issues

        Args:
            json_str: JSON string to validate

        Returns:
            Tuple of (is_valid, error_message, parsed_data)
        """
        if not json_str or not isinstance(json_str, str):
            return False, "JSON must be a non-empty string", None

        try:
            # Parse JSON
            data = json.loads(json_str)

            # Check structure depth and complexity
            is_valid, error_msg = self._validate_json_complexity(data)
            if not is_valid:
                return False, error_msg, None

            return True, "", data

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {str(e)}", None
        except RecursionError:
            return False, "JSON structure too deeply nested", None

    def _validate_json_complexity(self, data: Any, depth: int = 0) -> tuple[bool, str]:
        """Validate JSON complexity to prevent DoS attacks"""
        if depth > self.config.max_nesting_depth:
            return (
                False,
                f"JSON nesting too deep (max: {self.config.max_nesting_depth})",
            )

        if isinstance(data, dict):
            if len(data) > self.config.max_object_keys:
                return (
                    False,
                    f"Too many object keys (max: {self.config.max_object_keys})",
                )

            for key, value in data.items():
                # Validate key
                if not isinstance(key, str) or len(key) > 1000:
                    return False, "Invalid or overly long object key"

                # Recursively validate value
                is_valid, error_msg = self._validate_json_complexity(value, depth + 1)
                if not is_valid:
                    return False, error_msg

        elif isinstance(data, list):
            if len(data) > self.config.max_array_length:
                return False, f"Array too long (max: {self.config.max_array_length})"

            for item in data:
                is_valid, error_msg = self._validate_json_complexity(item, depth + 1)
                if not is_valid:
                    return False, error_msg

        elif isinstance(data, str):
            # Check string for suspicious content
            is_valid, error_msg = self._validate_string_content(data)
            if not is_valid:
                return False, error_msg

        return True, ""

    def _validate_string_content(self, text: str) -> tuple[bool, str]:
        """Validate string content for security issues"""
        # Check for excessive length
        if len(text) > 100000:  # 100KB limit for individual strings
            return False, "String content too long"

        # Check for suspicious Unicode categories
        unicode_categories = set()
        for char in text:
            import unicodedata

            category = unicodedata.category(char)
            unicode_categories.add(category[:2])  # Get general category

        if len(unicode_categories) > self.config.max_unicode_categories:
            return False, "String contains too many different Unicode categories"

        # Check for HTML entities if not allowed
        if not self.config.allow_html_entities:
            if "&" in text and ";" in text:
                # Simple check for HTML entities
                if re.search(r"&[a-zA-Z][a-zA-Z0-9]*;", text):
                    return False, "HTML entities not allowed"

        return True, ""

    def validate_email(self, email: str) -> tuple[bool, str]:
        """
        Validate email address format with security considerations

        Args:
            email: Email address to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not email or not isinstance(email, str):
            return False, "Email must be a non-empty string"

        # Length check
        if len(email) > self.config.max_email_length:
            return False, f"Email too long (max: {self.config.max_email_length} chars)"

        # Basic format validation
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email):
            return False, "Invalid email format"

        # Security checks
        email_lower = email.lower()

        # Check for suspicious patterns
        suspicious_patterns = [
            "javascript:",
            "data:",
            "file:",
            "mailto:",
            "<script",
            "onload=",
            "onerror=",
        ]

        for pattern in suspicious_patterns:
            if pattern in email_lower:
                return False, f"Suspicious pattern in email: {pattern}"

        # Check for excessive dots or special characters
        if email.count(".") > 5 or email.count("+") > 1:
            return False, "Email contains excessive special characters"

        return True, ""

    def validate_identifier(
        self, identifier: str, max_length: int = 100
    ) -> tuple[bool, str]:
        """
        Validate identifier (session ID, perspective name, etc.)

        Args:
            identifier: Identifier to validate
            max_length: Maximum allowed length

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not identifier or not isinstance(identifier, str):
            return False, "Identifier must be a non-empty string"

        # Length check
        if len(identifier) > max_length:
            return False, f"Identifier too long (max: {max_length} chars)"

        # Character validation - alphanumeric, hyphens, underscores only
        if not re.match(r"^[a-zA-Z0-9_-]+$", identifier):
            return (
                False,
                "Identifier contains invalid characters (only alphanumeric, _, - allowed)",
            )

        # Check for suspicious patterns
        suspicious_patterns = [
            "admin",
            "root",
            "system",
            "null",
            "undefined",
            "console",
            "eval",
            "function",
            "constructor",
        ]

        identifier_lower = identifier.lower()
        for pattern in suspicious_patterns:
            if pattern in identifier_lower:
                return False, f"Identifier contains reserved word: {pattern}"

        return True, ""

    def sanitize_html_content(self, content: str) -> tuple[str, list[str]]:
        """
        Sanitize HTML content for safe display

        Args:
            content: HTML content to sanitize

        Returns:
            Tuple of (sanitized_content, applied_sanitizations)
        """
        sanitizations = []
        sanitized = content

        # HTML escape dangerous characters
        original_sanitized = sanitized
        sanitized = html.escape(sanitized, quote=True)
        if sanitized != original_sanitized:
            sanitizations.append("HTML escaped dangerous characters")

        # Remove or escape common attack vectors
        attack_patterns = [
            (r"<script[^>]*>.*?</script>", "[SCRIPT_REMOVED]"),
            (r"javascript:", "[JAVASCRIPT_REMOVED]"),
            (r"vbscript:", "[VBSCRIPT_REMOVED]"),
            (r"data:", "[DATA_URL_REMOVED]"),
            (r"on\w+\s*=", "[EVENT_HANDLER_REMOVED]="),
        ]

        for pattern, replacement in attack_patterns:
            original_sanitized = sanitized
            sanitized = re.sub(
                pattern, replacement, sanitized, flags=re.IGNORECASE | re.DOTALL
            )
            if sanitized != original_sanitized:
                sanitizations.append(f"Removed attack pattern: {pattern}")

        return sanitized, sanitizations

    def validate_url_parameters(self, url: str) -> tuple[bool, str, dict[str, str]]:
        """
        Validate URL parameters for security issues

        Args:
            url: URL with parameters to validate

        Returns:
            Tuple of (is_valid, error_message, sanitized_params)
        """
        try:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            params = parse_qs(parsed.query)

            sanitized_params = {}

            for key, values in params.items():
                # Validate parameter name
                if not re.match(r"^[a-zA-Z0-9_-]+$", key):
                    return False, f"Invalid parameter name: {key}", {}

                # Limit number of values per parameter
                if len(values) > 10:
                    return False, f"Too many values for parameter: {key}", {}

                # Validate each value
                for value in values:
                    # URL decode
                    decoded_value = unquote(value)

                    # Check length
                    if len(decoded_value) > 1000:
                        return False, f"Parameter value too long: {key}", {}

                    # Check for injection patterns
                    injection_patterns = [
                        "javascript:",
                        "data:",
                        "<script",
                        "onload=",
                        "onerror=",
                        "eval(",
                        "alert(",
                        "document.",
                    ]

                    value_lower = decoded_value.lower()
                    for pattern in injection_patterns:
                        if pattern in value_lower:
                            return (
                                False,
                                f"Suspicious pattern in parameter {key}: {pattern}",
                                {},
                            )

                    sanitized_params[key] = decoded_value

            return True, "", sanitized_params

        except Exception as e:
            return False, f"Error validating URL parameters: {str(e)}", {}


class ConfigurationInputValidator:
    """Specialized validator for configuration inputs"""

    @staticmethod
    def validate_environment_variable(name: str, value: str) -> tuple[bool, str]:
        """
        Validate environment variable name and value

        Args:
            name: Environment variable name
            value: Environment variable value

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate name
        if not re.match(r"^[A-Z][A-Z0-9_]*$", name):
            return (
                False,
                "Environment variable name must be uppercase alphanumeric with underscores",
            )

        # Check for reserved names
        reserved_names = [
            "PATH",
            "HOME",
            "USER",
            "PWD",
            "SHELL",
            "TERM",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "DYLD_LIBRARY_PATH",
        ]

        if name in reserved_names:
            return False, f"Environment variable name is reserved: {name}"

        # Validate value
        if len(value) > 10000:
            return False, "Environment variable value too long (max 10000 chars)"

        # Check for suspicious patterns in value
        suspicious_patterns = [
            r"rm\s+-rf",
            r"sudo\s+",
            r"chmod\s+",
            r"/etc/passwd",
            r"/etc/shadow",
            r"cat\s+/proc",
            r"wget\s+",
            r"curl\s+.*\|\s*sh",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False, f"Suspicious pattern in environment variable: {pattern}"

        return True, ""

    @staticmethod
    def validate_config_value(
        key: str, value: Any, expected_type: type
    ) -> tuple[bool, str, Any]:
        """
        Validate configuration value with type checking

        Args:
            key: Configuration key name
            value: Configuration value
            expected_type: Expected Python type

        Returns:
            Tuple of (is_valid, error_message, sanitized_value)
        """
        # Type check
        if not isinstance(value, expected_type):
            try:
                # Try to convert
                if expected_type == int:
                    sanitized_value = int(value)
                elif expected_type == float:
                    sanitized_value = float(value)
                elif expected_type == bool:
                    sanitized_value = str(value).lower() in ("true", "1", "yes", "on")
                elif expected_type == str:
                    sanitized_value = str(value)
                else:
                    return (
                        False,
                        f"Cannot convert {type(value)} to {expected_type}",
                        None,
                    )
            except (ValueError, TypeError):
                return (
                    False,
                    f"Invalid type for {key}: expected {expected_type.__name__}",
                    None,
                )
        else:
            sanitized_value = value

        # Value-specific validation
        if expected_type == str and isinstance(sanitized_value, str):
            # Check string length
            if len(sanitized_value) > 10000:
                return False, f"String value too long for {key}", None

            # Check for null bytes
            if "\x00" in sanitized_value:
                return False, f"Null bytes not allowed in {key}", None

        elif expected_type == int:
            # Check reasonable ranges
            if sanitized_value < 0 or sanitized_value > 2**31 - 1:
                return False, f"Integer value out of range for {key}", None

        elif expected_type == float:
            # Check for NaN and infinity
            import math

            if math.isnan(sanitized_value) or math.isinf(sanitized_value):
                return False, f"Invalid float value for {key}", None

        return True, "", sanitized_value


# Export main classes
__all__ = [
    "EnhancedInputValidator",
    "ConfigurationInputValidator",
    "ValidationConfig",
]
