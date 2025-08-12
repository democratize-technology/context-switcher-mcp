"""Content sanitization utilities for LLM input processing"""

import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class ContentSanitizer:
    """Sanitizes content for safe LLM consumption"""

    @staticmethod
    def sanitize_for_llm(content: str) -> Tuple[str, List[str]]:
        """
        Sanitize content specifically for LLM consumption

        Args:
            content: Content to sanitize

        Returns:
            Tuple of (sanitized_content, applied_sanitizations)
        """
        sanitizations = []
        sanitized = content

        # Remove null bytes and dangerous control characters
        original_len = len(sanitized)
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", sanitized)
        if len(sanitized) != original_len:
            sanitizations.append("Removed control characters")

        # Limit excessive whitespace but preserve structure
        original_sanitized = sanitized
        sanitized = re.sub(r"\s{100,}", " " * 50, sanitized)
        if sanitized != original_sanitized:
            sanitizations.append("Limited excessive whitespace")

        # Remove zero-width characters that could be used for injection
        original_sanitized = sanitized
        sanitized = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", sanitized)
        if sanitized != original_sanitized:
            sanitizations.append("Removed zero-width characters")

        # Normalize line endings
        sanitized = sanitized.replace("\r\n", "\n").replace("\r", "\n")

        # Trim excessive leading/trailing whitespace
        original_sanitized = sanitized
        sanitized = sanitized.strip()
        if sanitized != original_sanitized:
            sanitizations.append("Trimmed excessive whitespace")

        return sanitized, sanitizations

    @staticmethod
    def sanitize_error_message(error_msg: str) -> str:
        """
        Sanitize error messages to remove internal details

        Args:
            error_msg: Original error message

        Returns:
            Sanitized error message safe for users
        """
        # Patterns that indicate internal details to hide
        internal_patterns = [
            (r'File ".*?"', "internal file"),
            (r"line \d+", "internal location"),
            (r"/[a-zA-Z0-9/._-]*\.py", "internal file"),
            (r"Traceback \(most recent call last\).*", "Internal error occurred"),
            (r"boto3\..*", "AWS service error"),
            (r"psycopg2\..*", "Database error"),
            (r"requests\.exceptions\..*", "Network error"),
            (r"ImportError:.*", "Configuration error"),
            (r"ModuleNotFoundError:.*", "Configuration error"),
        ]

        sanitized = error_msg

        for pattern, replacement in internal_patterns:
            sanitized = re.sub(
                pattern, replacement, sanitized, flags=re.MULTILINE | re.DOTALL
            )

        # Limit error message length
        if len(sanitized) > 500:
            sanitized = sanitized[:497] + "..."

        return sanitized


class ContentStructureValidator:
    """Validates content structure for security issues"""

    @staticmethod
    def validate_content_structure(content: str) -> Tuple[bool, List[str]]:
        """Validate content structure for suspicious patterns"""
        issues = []

        # Check for excessive nesting levels
        open_braces = content.count("{")
        close_braces = content.count("}")
        if abs(open_braces - close_braces) > 5:
            issues.append("Unbalanced braces detected")

        # Check for excessive repeated characters
        if re.search(r"(.)\1{50,}", content):
            issues.append("Excessive character repetition detected")

        # Check for suspicious character sequences
        if re.search(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", content):
            issues.append("Control characters detected")

        # Check for potential binary data
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            issues.append("Invalid character encoding")

        return len(issues) == 0, issues


# Global instances
content_sanitizer = ContentSanitizer()
structure_validator = ContentStructureValidator()


# Legacy function wrappers for backward compatibility
def sanitize_for_llm(content: str) -> Tuple[str, List[str]]:
    """Legacy wrapper for ContentSanitizer.sanitize_for_llm"""
    return content_sanitizer.sanitize_for_llm(content)


def sanitize_error_message(error_msg: str) -> str:
    """Legacy wrapper for ContentSanitizer.sanitize_error_message"""
    return content_sanitizer.sanitize_error_message(error_msg)
