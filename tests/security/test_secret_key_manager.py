"""
Test suite for SecretKeyManager security module.
"""

import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import json  # noqa: E402
import secrets  # noqa: E402

import pytest  # noqa: E402
from context_switcher_mcp.security.secret_key_manager import (  # noqa: E402
    SecretKeyManager,
    _save_key_data_atomically,
    load_or_generate_secret_key,
)


class TestSecretKeyManager:
    """Test SecretKeyManager functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        # Use a temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.original_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.temp_dir)

        # Clear environment variable if set
        if "CONTEXT_SWITCHER_SECRET_KEY" in os.environ:
            del os.environ["CONTEXT_SWITCHER_SECRET_KEY"]

    def teardown_method(self):
        """Clean up test fixtures"""
        # Restore original HOME
        if self.original_home:
            os.environ["HOME"] = self.original_home
        else:
            del os.environ["HOME"]

        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_secret_key_manager_initialization(self):
        """Test SecretKeyManager initialization"""
        manager = SecretKeyManager()

        assert manager.current_key
        assert len(manager.current_key) > 0
        assert isinstance(manager.previous_keys, list)

    def test_secret_key_manager_with_initial_key(self):
        """Test SecretKeyManager initialization with provided key"""
        test_key = secrets.token_urlsafe(32)
        manager = SecretKeyManager(test_key)

        assert manager.current_key == test_key
        assert isinstance(manager.previous_keys, list)

    def test_key_rotation(self):
        """Test secret key rotation"""
        manager = SecretKeyManager("initial_key")
        original_key = manager.current_key

        # Rotate key
        new_key = manager.rotate_key()

        assert new_key != original_key
        assert manager.current_key == new_key
        assert original_key in manager.previous_keys
        assert len(manager.previous_keys) == 1

    def test_multiple_key_rotations(self):
        """Test multiple key rotations with history limit"""
        manager = SecretKeyManager("initial_key")
        keys_history = [manager.current_key]

        # Rotate 7 times (more than the 5-key limit)
        for _i in range(7):
            new_key = manager.rotate_key()
            keys_history.append(new_key)

        # Should only keep the last 5 previous keys
        assert len(manager.previous_keys) == 5
        assert manager.current_key == keys_history[-1]

        # Previous keys should be the 5 most recent (excluding current)
        expected_previous = keys_history[-6:-1]  # Last 5 excluding current
        assert manager.previous_keys == expected_previous

    def test_signature_validation_current_key(self):
        """Test signature validation with current key"""
        manager = SecretKeyManager()
        test_data = "test_data_for_signing"

        # Create a signature using the current key directly
        import hashlib

        expected_signature = hashlib.sha256(
            f"{test_data}:{manager.current_key}".encode()
        ).hexdigest()

        # Should validate with current key
        assert manager.validate_with_any_key(test_data, expected_signature) is True

        # Should not validate with wrong signature
        assert manager.validate_with_any_key(test_data, "wrong_signature") is False

    def test_signature_validation_with_previous_key(self):
        """Test signature validation with previous key during rotation"""
        manager = SecretKeyManager("initial_key")
        test_data = "test_data_for_signing"

        # Create signature with current key
        import hashlib

        old_signature = hashlib.sha256(
            f"{test_data}:{manager.current_key}".encode()
        ).hexdigest()

        # Rotate key
        manager.rotate_key()

        # Should still validate with previous key
        assert manager.validate_with_any_key(test_data, old_signature) is True

    def test_signature_validation_invalid_data(self):
        """Test signature validation with invalid data"""
        manager = SecretKeyManager()

        # Test with None/empty data
        assert manager.validate_with_any_key("", "signature") is False

        # Test with malformed signature
        assert manager.validate_with_any_key("data", None) is False

    def test_get_current_key_info(self):
        """Test getting current key information"""
        manager = SecretKeyManager("test_key")
        key_info = manager.get_current_key_info()

        assert "key_hash" in key_info
        assert "previous_keys_count" in key_info
        assert "has_rotated" in key_info
        assert "source" in key_info

        assert key_info["previous_keys_count"] == 0
        assert key_info["has_rotated"] is False
        assert len(key_info["key_hash"]) == 16  # SHA256 hash truncated to 16 chars

    def test_get_key_info_after_rotation(self):
        """Test key info after rotation"""
        manager = SecretKeyManager("test_key")
        manager.rotate_key()

        key_info = manager.get_current_key_info()
        assert key_info["previous_keys_count"] == 1
        assert key_info["has_rotated"] is True

    def test_cleanup_old_keys(self):
        """Test cleanup of old keys (placeholder functionality)"""
        manager = SecretKeyManager()

        # Current implementation is a placeholder
        cleaned = manager.cleanup_old_keys(30)
        assert cleaned == 0  # Placeholder returns 0


class TestLoadOrGenerateSecretKey:
    """Test load_or_generate_secret_key function"""

    def setup_method(self):
        """Set up test fixtures"""
        # Use a temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.original_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.temp_dir)

        # Clear environment variable if set
        if "CONTEXT_SWITCHER_SECRET_KEY" in os.environ:
            del os.environ["CONTEXT_SWITCHER_SECRET_KEY"]

    def teardown_method(self):
        """Clean up test fixtures"""
        # Restore original HOME
        if self.original_home:
            os.environ["HOME"] = self.original_home
        else:
            del os.environ["HOME"]

        # Clean up temp directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_from_environment(self):
        """Test loading secret key from environment variable"""
        test_key = "environment_test_key"
        os.environ["CONTEXT_SWITCHER_SECRET_KEY"] = test_key

        key = load_or_generate_secret_key()
        assert key == test_key

    def test_generate_new_key_when_none_exists(self):
        """Test generating new key when none exists"""
        key = load_or_generate_secret_key()

        assert key
        assert len(key) > 0

        # Should create the config directory and file
        config_dir = self.temp_dir / ".context_switcher"
        secret_file = config_dir / "secret_key.json"

        assert config_dir.exists()
        assert secret_file.exists()

        # Verify file permissions (if on POSIX system)
        if hasattr(os, "stat") and hasattr(os.stat(secret_file), "st_mode"):
            # Check that file is readable only by owner
            file_stat = secret_file.stat()
            # On some systems, the exact permissions might vary, just check it's not world-readable
            assert file_stat.st_mode & 0o077 == 0  # No group/other permissions

    def test_load_from_existing_file(self):
        """Test loading key from existing file"""
        # Create config directory and file
        config_dir = self.temp_dir / ".context_switcher"
        config_dir.mkdir(parents=True)
        secret_file = config_dir / "secret_key.json"

        test_key = "file_test_key"
        data = {
            "current_key": test_key,
            "previous_keys": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rotation_count": 0,
        }

        with open(secret_file, "w") as f:
            json.dump(data, f)

        key = load_or_generate_secret_key()
        assert key == test_key

    def test_handle_corrupted_file(self):
        """Test handling of corrupted secret key file"""
        # Create config directory and corrupted file
        config_dir = self.temp_dir / ".context_switcher"
        config_dir.mkdir(parents=True)
        secret_file = config_dir / "secret_key.json"

        # Write invalid JSON
        with open(secret_file, "w") as f:
            f.write("invalid json content")

        # Should generate new key despite corrupted file
        key = load_or_generate_secret_key()
        assert key
        assert len(key) > 0

        # Should have overwritten the corrupted file
        with open(secret_file) as f:
            data = json.load(f)
            assert data["current_key"] == key

    def test_save_key_data_atomically(self):
        """Test atomic key data saving"""
        config_dir = self.temp_dir / ".context_switcher"
        config_dir.mkdir(parents=True)
        secret_file = config_dir / "secret_key.json"

        test_data = {
            "current_key": "test_key",
            "previous_keys": ["old_key"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        _save_key_data_atomically(test_data, config_dir, secret_file)

        # Verify file was created
        assert secret_file.exists()

        # Verify data was saved correctly
        with open(secret_file) as f:
            saved_data = json.load(f)
            assert saved_data == test_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
