"""
Secret key management with rotation support for enhanced security.

This module provides secure key generation, storage, rotation, and validation
functionality with atomic file operations and cross-platform compatibility.
"""

import hashlib
import json
import os
import platform
import secrets
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..logging_base import get_logger

logger = get_logger(__name__)


def load_or_generate_secret_key() -> str:
    """Load secret key from environment or generate a secure one.

    Priority order:
    1. CONTEXT_SWITCHER_SECRET_KEY environment variable
    2. Secret key file in ~/.context_switcher/secret_key.json
    3. Generate new key and save to file

    Returns:
        str: The secret key to use for cryptographic operations

    Security Features:
        - 32-byte URL-safe random key generation
        - Atomic file writes with secure permissions (0o600)
        - Cross-platform compatibility (POSIX and Windows)
        - Directory permissions hardening (0o700)
    """
    # Try environment variable first
    env_key = os.environ.get("CONTEXT_SWITCHER_SECRET_KEY")
    if env_key:
        logger.info("Using secret key from environment variable")
        return env_key

    # Try loading from file
    config_dir = Path.home() / ".context_switcher"
    secret_file = config_dir / "secret_key.json"

    if secret_file.exists():
        try:
            with open(secret_file) as f:
                data = json.load(f)
                if "current_key" in data:
                    logger.info("Loaded secret key from file")
                    return data["current_key"]
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load secret key from file: {e}")

    # Generate new key and save it with atomic operations
    new_key = secrets.token_urlsafe(32)
    config_dir.mkdir(exist_ok=True, parents=True)

    # CRITICAL: Set secure permissions on parent directory first
    try:
        # Ensure config directory has secure permissions
        config_dir.chmod(0o700)  # Owner read/write/execute only
    except (OSError, AttributeError):
        # May fail on non-POSIX systems, continue anyway
        pass

    try:
        # Save with rotation support structure
        data = {
            "current_key": new_key,
            "previous_keys": [],  # For key rotation support
            "created_at": datetime.now(UTC).isoformat(),
            "rotation_count": 0,
        }

        _save_key_data_atomically(data, config_dir, secret_file)
        logger.info("Generated and saved new secret key")

    except OSError as e:
        logger.warning(f"Failed to save secret key to file: {e}")

    return new_key


def _save_key_data_atomically(data: dict, config_dir: Path, secret_file: Path) -> None:
    """Save key data with atomic write operations and secure permissions.

    Args:
        data: The key data dictionary to save
        config_dir: Configuration directory path
        secret_file: Target secret file path

    Raises:
        IOError: If the save operation fails
    """
    # Ensure the config directory exists
    config_dir.mkdir(exist_ok=True, parents=True)

    # CRITICAL: Use atomic write with secure permissions from the start
    if platform.system() != "Windows":
        # On POSIX systems, create with secure permissions from the start
        temp_fd, temp_path = tempfile.mkstemp(
            dir=config_dir, prefix=".secret_key_", suffix=".tmp"
        )
        # Set secure permissions immediately after creation
        os.chmod(temp_path, 0o600)
        try:
            # Write to temp file using file descriptor
            with os.fdopen(temp_fd, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename (on POSIX, rename is atomic)
            Path(temp_path).replace(secret_file)
            logger.debug("Atomic write completed (POSIX)")
        finally:
            # Clean up temp file if it still exists
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
    else:
        # On Windows, do our best with available tools
        temp_path = secret_file.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)

        # Try to set permissions (may not work on Windows)
        try:
            os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        except Exception:
            pass

        # Replace atomically (as atomic as Windows allows)
        temp_path.replace(secret_file)
        logger.debug("Atomic write completed (Windows)")


class SecretKeyManager:
    """Manages secret key rotation for enhanced security.

    This class provides comprehensive secret key management including:
    - Secure key generation and storage
    - Key rotation with grace period support
    - Validation with current and previous keys
    - Cross-platform atomic file operations

    Features:
        - Maintains up to 5 previous keys for rotation grace period
        - Atomic file operations for data integrity
        - Secure file permissions (POSIX: 0o600, 0o700 for directories)
        - Constant-time signature comparison to prevent timing attacks
    """

    def __init__(self, initial_key: str | None = None):
        """Initialize secret key manager.

        Args:
            initial_key: Optional initial key to use. If not provided,
                        uses the global key loading mechanism.
        """
        self.current_key = initial_key or load_or_generate_secret_key()
        self.previous_keys: list[str] = []
        self._load_previous_keys()

    def _load_previous_keys(self) -> None:
        """Load previous keys for validation during rotation period.

        Loads up to 5 previous keys from the secret key file to support
        validation during key rotation grace periods.
        """
        secret_file = Path.home() / ".context_switcher" / "secret_key.json"
        if secret_file.exists():
            try:
                with open(secret_file) as f:
                    data = json.load(f)
                    self.previous_keys = data.get("previous_keys", [])[
                        :5
                    ]  # Keep last 5 keys
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load previous keys: {e}")

    def rotate_key(self) -> str:
        """Rotate to a new secret key, keeping the old one for grace period.

        Returns:
            str: The new secret key

        Security Features:
            - Generates cryptographically secure 32-byte key
            - Maintains previous keys for grace period validation
            - Atomic file operations to prevent corruption
            - Limits previous key storage to prevent unbounded growth
        """
        new_key = secrets.token_urlsafe(32)
        self.previous_keys.insert(0, self.current_key)
        self.previous_keys = self.previous_keys[:5]  # Keep only last 5 keys

        old_key_hash = hashlib.sha256(self.current_key.encode()).hexdigest()[:8]
        self.current_key = new_key

        # Save the rotated keys with atomic write
        config_dir = Path.home() / ".context_switcher"
        secret_file = config_dir / "secret_key.json"

        try:
            data = {
                "current_key": new_key,
                "previous_keys": self.previous_keys,
                "rotated_at": datetime.now(UTC).isoformat(),
                "rotation_count": len(self.previous_keys),
                "previous_key_hash": old_key_hash,  # For audit purposes
            }

            _save_key_data_atomically(data, config_dir, secret_file)
            logger.info(f"Successfully rotated secret key (previous: {old_key_hash})")

        except OSError as e:
            logger.error(f"Failed to save rotated key: {e}")
            # Revert the rotation on failure
            self.current_key = (
                self.previous_keys.pop(0) if self.previous_keys else self.current_key
            )
            raise

        return new_key

    def validate_with_any_key(self, data: str, signature: str) -> bool:
        """Validate signature with current or previous keys (for rotation grace period).

        Args:
            data: The data that was signed
            signature: The signature to validate

        Returns:
            bool: True if signature is valid with any available key

        Security Features:
            - Constant-time comparison to prevent timing attacks
            - Tries current key first (most common case)
            - Falls back to previous keys during rotation grace period
            - Logs successful validations with previous keys for monitoring
        """
        # Try current key first (most common case)
        if self._validate_signature(data, signature, self.current_key):
            return True

        # Try previous keys during grace period
        for i, key in enumerate(self.previous_keys):
            if self._validate_signature(data, signature, key):
                logger.info(
                    f"Validated with previous key #{i + 1} during rotation grace period"
                )
                return True

        return False

    def _validate_signature(self, data: str, signature: str, key: str) -> bool:
        """Validate a signature with a specific key.

        Args:
            data: The data that was signed
            signature: The signature to validate
            key: The secret key to use for validation

        Returns:
            bool: True if signature is valid

        Security Features:
            - Uses SHA-256 for cryptographic hashing
            - Constant-time comparison via secrets.compare_digest
            - Prevents timing attacks on signature validation
        """
        try:
            expected = hashlib.sha256(f"{data}:{key}".encode()).hexdigest()
            return secrets.compare_digest(expected, signature)
        except (TypeError, ValueError) as e:
            logger.warning(f"Signature validation error: {e}")
            return False

    def get_current_key_info(self) -> dict:
        """Get information about the current key (without exposing the key itself).

        Returns:
            dict: Key information for monitoring and debugging
        """
        key_hash = hashlib.sha256(self.current_key.encode()).hexdigest()[:16]
        return {
            "key_hash": key_hash,
            "previous_keys_count": len(self.previous_keys),
            "has_rotated": len(self.previous_keys) > 0,
            "source": "environment"
            if os.environ.get("CONTEXT_SWITCHER_SECRET_KEY")
            else "file",
        }

    def cleanup_old_keys(self, max_age_days: int = 30) -> int:
        """Clean up old previous keys beyond the specified age.

        Args:
            max_age_days: Maximum age in days for previous keys

        Returns:
            int: Number of keys cleaned up

        Note:
            This is a placeholder for future implementation. Currently,
            we limit to 5 keys regardless of age for simplicity.
        """
        # Future implementation could check key ages and clean up accordingly
        # For now, we rely on the 5-key limit in rotate_key()
        logger.debug(f"Key cleanup requested (max_age: {max_age_days} days)")
        return 0
