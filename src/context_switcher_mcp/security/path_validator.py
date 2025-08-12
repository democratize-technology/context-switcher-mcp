"""Path validation utilities for secure file operations"""

import os
import re
import logging
from pathlib import Path
from typing import Tuple, Optional, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class PathValidator:
    """Validates file paths and URLs for security vulnerabilities"""

    # Dangerous path patterns that could indicate path traversal attacks
    DANGEROUS_PATH_PATTERNS = [
        r"\.\./",  # Directory traversal
        r"\.\.\.",  # Extended traversal
        r"~/",  # Home directory access
        r"/etc/",  # System files
        r"/proc/",  # Process files
        r"/sys/",  # System files
        r"/dev/",  # Device files
        r"/root/",  # Root directory
        r"/var/log/",  # Log files
        r"\\\\",  # Windows UNC paths
        r"file://",  # File protocol
        r'[<>"|*?]',  # Windows invalid chars
    ]

    # Allowed file extensions for configuration files
    ALLOWED_CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".env"}

    @staticmethod
    def validate_file_path(
        file_path: str, base_directory: Optional[str] = None, allow_create: bool = False
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validate file path for security issues

        Args:
            file_path: Path to validate
            base_directory: Optional base directory to restrict access to
            allow_create: Whether to allow creation of non-existent files

        Returns:
            Tuple of (is_valid, error_message, normalized_path)
        """
        if not file_path or not isinstance(file_path, str):
            return False, "File path must be a non-empty string", None

        # Remove null bytes and control characters
        cleaned_path = file_path.replace("\x00", "").strip()
        if not cleaned_path:
            return False, "File path cannot be empty after sanitization", None

        # Check for dangerous patterns
        for pattern in PathValidator.DANGEROUS_PATH_PATTERNS:
            if re.search(pattern, cleaned_path, re.IGNORECASE):
                logger.warning(
                    f"Dangerous path pattern detected: {pattern} in {cleaned_path}"
                )
                return False, f"Path contains suspicious pattern: {pattern}", None

        try:
            # Normalize and resolve the path
            path_obj = Path(cleaned_path).resolve()
            normalized_path = str(path_obj)

            # If base directory is specified, ensure path is within it
            if base_directory:
                base_path = Path(base_directory).resolve()
                try:
                    path_obj.relative_to(base_path)
                except ValueError:
                    return False, f"Path must be within {base_directory}", None

            # Check if path exists or creation is allowed
            if not path_obj.exists():
                if not allow_create:
                    return False, "File does not exist and creation not allowed", None

                # Check if parent directory exists and is writable
                parent_dir = path_obj.parent
                if not parent_dir.exists():
                    return False, "Parent directory does not exist", None
                if not os.access(parent_dir, os.W_OK):
                    return False, "Parent directory is not writable", None

            else:
                # File exists, check if it's readable
                if not os.access(path_obj, os.R_OK):
                    return False, "File is not readable", None

                # Check if it's actually a file (not a directory or special file)
                if not path_obj.is_file():
                    return False, "Path is not a regular file", None

            return True, "", normalized_path

        except (OSError, ValueError) as e:
            logger.warning(f"Path validation error for {cleaned_path}: {e}")
            return False, f"Invalid path: {str(e)}", None

    @staticmethod
    def validate_config_file_path(file_path: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate configuration file path with additional restrictions

        Args:
            file_path: Configuration file path to validate

        Returns:
            Tuple of (is_valid, error_message, normalized_path)
        """
        # First do basic path validation
        is_valid, error_msg, normalized_path = PathValidator.validate_file_path(
            file_path, allow_create=False
        )

        if not is_valid:
            return False, error_msg, None

        # Check file extension
        path_obj = Path(file_path)
        if path_obj.suffix.lower() not in PathValidator.ALLOWED_CONFIG_EXTENSIONS:
            return False, f"Invalid config file extension: {path_obj.suffix}", None

        # Additional size check for config files (max 10MB)
        try:
            if path_obj.stat().st_size > 10 * 1024 * 1024:
                return False, "Configuration file too large (max 10MB)", None
        except OSError:
            pass  # File might not exist yet

        return True, "", normalized_path

    @staticmethod
    def validate_url(
        url: str, allowed_schemes: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Validate URL for security issues

        Args:
            url: URL to validate
            allowed_schemes: Optional list of allowed schemes (default: https, http)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not url or not isinstance(url, str):
            return False, "URL must be a non-empty string"

        if allowed_schemes is None:
            allowed_schemes = ["https", "http"]

        # Remove dangerous characters
        cleaned_url = url.strip().replace("\x00", "")

        try:
            parsed = urlparse(cleaned_url)

            # Check scheme
            if parsed.scheme.lower() not in allowed_schemes:
                return False, f"Scheme '{parsed.scheme}' not allowed"

            # Check for localhost/internal IPs to prevent SSRF
            hostname = parsed.hostname
            if hostname:
                hostname_lower = hostname.lower()

                # Block localhost variations
                if hostname_lower in ["localhost", "127.0.0.1", "0.0.0.0", "::1"]:
                    return False, "Localhost URLs not allowed"

                # Block private IP ranges (basic check)
                if (
                    hostname_lower.startswith("192.168.")
                    or hostname_lower.startswith("10.")
                    or hostname_lower.startswith("172.")
                ):
                    return False, "Private IP addresses not allowed"

                # Block file protocol disguised as hostname
                if "file:" in hostname_lower or "localhost" in hostname_lower:
                    return False, "Suspicious hostname detected"

            # Check for suspicious patterns in URL
            suspicious_patterns = [
                "javascript:",
                "data:",
                "vbscript:",
                "file://",
                "ftp://",
                "gopher://",
                "dict://",
                "ldap://",
            ]

            url_lower = cleaned_url.lower()
            for pattern in suspicious_patterns:
                if pattern in url_lower:
                    return False, f"Suspicious URL pattern: {pattern}"

            return True, ""

        except Exception as e:
            logger.warning(f"URL validation error for {cleaned_url}: {e}")
            return False, f"Invalid URL format: {str(e)}"


class SecureFileHandler:
    """Secure file operations with path validation"""

    @staticmethod
    def safe_read_file(
        file_path: str, max_size: int = 10 * 1024 * 1024
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Safely read file content with validation

        Args:
            file_path: Path to file to read
            max_size: Maximum file size in bytes

        Returns:
            Tuple of (success, error_or_content, normalized_path)
        """
        # Validate path first
        is_valid, error_msg, normalized_path = PathValidator.validate_file_path(
            file_path
        )
        if not is_valid:
            return False, error_msg, None

        try:
            path_obj = Path(normalized_path)

            # Check file size before reading
            file_size = path_obj.stat().st_size
            if file_size > max_size:
                return (
                    False,
                    f"File too large: {file_size} bytes (max: {max_size})",
                    normalized_path,
                )

            # Read file content safely
            with open(normalized_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(
                    max_size + 1
                )  # Read one extra byte to detect oversized files

            if len(content) > max_size:
                return False, "File content exceeds maximum size", normalized_path

            return True, content, normalized_path

        except (OSError, IOError, UnicodeDecodeError) as e:
            logger.error(f"Error reading file {normalized_path}: {e}")
            return False, f"Failed to read file: {str(e)}", normalized_path

    @staticmethod
    def safe_write_file(
        file_path: str,
        content: str,
        base_directory: Optional[str] = None,
        max_size: int = 10 * 1024 * 1024,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Safely write file content with validation

        Args:
            file_path: Path to file to write
            content: Content to write
            base_directory: Optional base directory restriction
            max_size: Maximum content size in bytes

        Returns:
            Tuple of (success, error_message, normalized_path)
        """
        # Validate content size
        if len(content.encode("utf-8")) > max_size:
            return False, f"Content too large: max {max_size} bytes", None

        # Validate path
        is_valid, error_msg, normalized_path = PathValidator.validate_file_path(
            file_path, base_directory=base_directory, allow_create=True
        )
        if not is_valid:
            return False, error_msg, None

        try:
            # Write file safely
            with open(normalized_path, "w", encoding="utf-8") as f:
                f.write(content)

            return True, "", normalized_path

        except (OSError, IOError) as e:
            logger.error(f"Error writing file {normalized_path}: {e}")
            return False, f"Failed to write file: {str(e)}", normalized_path


# Export main classes and functions
__all__ = [
    "PathValidator",
    "SecureFileHandler",
]
