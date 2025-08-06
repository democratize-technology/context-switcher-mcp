"""Tests for high-priority security fixes"""

import sys

sys.path.insert(0, "src")  # noqa: E402

import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import secrets  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from context_switcher_mcp.backend_interface import LiteLLMBackend, OllamaBackend  # noqa: E402
from context_switcher_mcp.circuit_breaker_store import CircuitBreakerStore  # noqa: E402
from context_switcher_mcp.client_binding import (  # noqa: E402
    ClientBindingManager,
    SecretKeyManager,
    _load_or_generate_secret_key,
    create_secure_session_with_binding,
)
from context_switcher_mcp.exceptions import ModelConnectionError, ModelValidationError  # noqa: E402
from context_switcher_mcp.models import (  # noqa: E402
    ClientBinding,
    ContextSwitcherSession,
    ModelBackend,
    Thread,
)
from context_switcher_mcp.orchestrator import ThreadOrchestrator  # noqa: E402


class TestCircuitBreakerPathTraversal:
    """Test path traversal protection in circuit breaker store"""

    def test_valid_path_in_home_directory(self):
        """Test that valid paths in home directory are accepted"""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = Path(tmpdir) / "circuit_breakers.json"
            store = CircuitBreakerStore(str(valid_path))
            assert store.storage_path == valid_path.resolve()

    def test_valid_path_in_tmp_directory(self):
        """Test that valid paths in /tmp are accepted"""
        valid_path = "/tmp/test_circuit_breakers.json"
        store = CircuitBreakerStore(valid_path)
        assert store.storage_path == Path(valid_path).resolve()

    def test_path_traversal_attack_blocked(self):
        """Test that path traversal attempts are blocked"""
        malicious_paths = [
            "/etc/passwd.json",  # System file
            "/var/log/system.json",  # System log
            "/root/.ssh/keys.json",  # Root SSH keys
            "~/../../etc/passwd.json",  # Traverse above home
            "/Windows/System32/config/SAM.json",  # Windows system file
        ]

        for path in malicious_paths:
            # Skip relative paths that might resolve within home on some systems
            if path.startswith(".."):
                continue

            with pytest.raises(
                ValueError,
                match="(Storage path must be within allowed directories|Path traversal attempt detected|Symlinks are not allowed)",
            ):
                CircuitBreakerStore(path)

    def test_non_json_file_rejected(self):
        """Test that non-JSON files are rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "circuit_breakers.txt"
            with pytest.raises(ValueError, match="Storage path must be a .json file"):
                CircuitBreakerStore(str(invalid_path))

    def test_default_path_is_safe(self):
        """Test that default path is within home directory"""
        store = CircuitBreakerStore()
        assert str(store.storage_path).startswith(str(Path.home()))
        assert store.storage_path.name == "circuit_breakers.json"


class TestPBKDF2IterationUpdate:
    """Test updated PBKDF2 iterations for client binding"""

    def test_pbkdf2_uses_600000_iterations(self):
        """Test that PBKDF2 now uses 600,000 iterations"""
        binding = ClientBinding(
            session_entropy="test_entropy",
            creation_timestamp=datetime.now(timezone.utc),
            binding_signature="",
            access_pattern_hash="test_hash",
        )

        secret_key = "test_secret"

        # Generate signature and verify it uses 600,000 iterations
        with patch("hashlib.pbkdf2_hmac") as mock_pbkdf2:
            mock_pbkdf2.return_value = b"mocked_hash"
            binding.generate_binding_signature(secret_key)

            # Verify pbkdf2_hmac was called with 600,000 iterations
            mock_pbkdf2.assert_called_once()
            args = mock_pbkdf2.call_args[0]
            assert args[0] == "sha256"  # hash algorithm
            assert args[3] == 600000  # iterations

    def test_binding_validation_with_new_iterations(self):
        """Test that binding validation works with new iteration count"""
        binding = ClientBinding(
            session_entropy="test_entropy",
            creation_timestamp=datetime.now(timezone.utc),
            binding_signature="",
            access_pattern_hash="test_hash",
        )

        secret_key = "test_secret"
        binding.binding_signature = binding.generate_binding_signature(secret_key)

        # Validation should succeed with correct secret
        assert binding.validate_binding(secret_key) is True

        # Validation should fail with wrong secret
        assert binding.validate_binding("wrong_secret") is False


class TestModelNameValidation:
    """Test model name validation for all backends"""

    @pytest.mark.asyncio
    async def test_litellm_backend_validates_model_names(self):
        """Test that LiteLLM backend validates model names"""
        backend = LiteLLMBackend()
        thread = Thread(
            id="test",
            name="test_thread",
            system_prompt="Test prompt",
            model_backend=ModelBackend.LITELLM,
            model_name="../../etc/passwd",  # Malicious model name
        )

        # Mock litellm.acompletion to test validation happens before call
        with patch("litellm.acompletion") as mock_litellm:
            # Even with successful mock, validation should fail
            mock_litellm.return_value = MagicMock()
            with pytest.raises(ModelValidationError, match="Invalid model ID"):
                await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_ollama_backend_validates_model_names(self):
        """Test that Ollama backend validates model names"""
        backend = OllamaBackend()
        thread = Thread(
            id="test",
            name="test_thread",
            system_prompt="Test prompt",
            model_backend=ModelBackend.OLLAMA,
            model_name="'; DROP TABLE models; --",  # SQL injection attempt
        )

        # Mock httpx.AsyncClient to test validation happens before call
        with patch("httpx.AsyncClient") as mock_httpx:
            # Even with successful mock, validation should fail
            mock_client = MagicMock()
            mock_httpx.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value.json.return_value = {
                "message": {"content": "response"}
            }
            with pytest.raises(ModelValidationError, match="Invalid model ID"):
                await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_valid_model_names_accepted(self):
        """Test that valid model names are accepted"""
        valid_models = [
            "gpt-4",
            "gpt-3.5-turbo",
            "llama2",
            "mistral",
            "codellama",
            "us.anthropic.claude-3-sonnet-20240229-v1:0",
        ]

        backend = LiteLLMBackend()

        for model_name in valid_models:
            thread = Thread(
                id="test",
                name="test_thread",
                system_prompt="Test prompt",
                model_backend=ModelBackend.LITELLM,
                model_name=model_name,
            )

            # Mock successful response
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(message=MagicMock(content="Test response"))
            ]

            with patch(
                "litellm.acompletion",
                return_value=mock_response,
            ):
                # Should not raise validation error
                response = await backend.call_model(thread)
                assert response == "Test response"


class TestCircuitBreakerRaceCondition:
    """Test circuit breaker failure recording race condition fix"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_transient_failures(self):
        """Test that circuit breaker correctly records transient failures"""
        orchestrator = ThreadOrchestrator(max_retries=2, retry_delay=0.01)

        # Create a thread that will fail
        thread = Thread(
            id="test",
            name="test_thread",
            system_prompt="Test",
            model_backend=ModelBackend.BEDROCK,
            model_name="test-model",
        )

        # Mock backend to raise connection error
        async def mock_backend_call(t):
            raise ModelConnectionError("Connection failed")

        orchestrator.backends[ModelBackend.BEDROCK] = mock_backend_call

        # Get circuit breaker
        circuit_breaker = orchestrator.circuit_breakers[ModelBackend.BEDROCK]
        initial_failures = circuit_breaker.failure_count

        # Call should fail and record failures
        response = await orchestrator.thread_manager.get_single_thread_response(thread)

        # Verify failure was recorded
        assert circuit_breaker.failure_count > initial_failures
        assert "AORP_ERROR" in response

    @pytest.mark.asyncio
    async def test_circuit_breaker_does_not_record_non_transient_failures(self):
        """Test that circuit breaker doesn't record non-transient failures"""
        # Mock circuit breaker state loading to prevent restoration
        with patch(
            "context_switcher_mcp.circuit_breaker_manager.load_circuit_breaker_state"
        ) as mock_load:
            mock_load.return_value = None  # No saved state
            orchestrator = ThreadOrchestrator(max_retries=2, retry_delay=0.01)

            thread = Thread(
                id="test",
                name="test_thread",
                system_prompt="Test",
                model_backend=ModelBackend.BEDROCK,
                model_name="test-model",
            )

            # Mock backend to raise authentication error (non-transient)
            from context_switcher_mcp.exceptions import ModelAuthenticationError

            async def mock_backend_call(t):
                raise ModelAuthenticationError("Invalid api_key provided")

            orchestrator.backends[ModelBackend.BEDROCK] = mock_backend_call

            # Get circuit breaker and reset it for clean test
            circuit_breaker = orchestrator.circuit_breakers[ModelBackend.BEDROCK]
            circuit_breaker.failure_count = 0
            circuit_breaker.state = "CLOSED"
            circuit_breaker.last_failure_time = None
            initial_failures = circuit_breaker.failure_count

            # Call should fail but NOT record failure
            try:
                await orchestrator.thread_manager.get_single_thread_response(thread)
            except ModelAuthenticationError:
                # Expected - authentication errors are not retried
                pass

            # Verify failure was NOT recorded (non-transient error)
            assert circuit_breaker.failure_count == initial_failures


class TestAsyncLockInitialization:
    """Test async lock initialization race condition fix"""

    @pytest.mark.asyncio
    async def test_concurrent_access_to_uninitialized_session(self):
        """Test that concurrent access to session with uninitialized lock works"""
        session = ContextSwitcherSession(
            session_id="test", created_at=datetime.now(timezone.utc)
        )

        # Simulate __post_init__ not being called (clear the lock)
        session._access_lock = None

        # Create multiple concurrent access attempts
        async def access_session(tool_name: str):
            await session.record_access(tool_name)
            return session.access_count

        # Run concurrent accesses
        tasks = [access_session(f"tool_{i}") for i in range(10)]
        await asyncio.gather(*tasks)

        # All accesses should succeed and be properly counted
        assert session.access_count == 10
        # Verify lock was created during access
        assert session._access_lock is not None

    @pytest.mark.asyncio
    async def test_session_with_initialized_lock(self):
        """Test that session with properly initialized lock works correctly"""
        session = ContextSwitcherSession(
            session_id="test", created_at=datetime.now(timezone.utc)
        )

        # Verify lock is initialized after __post_init__
        assert session._lock_initialized is True
        assert session._access_lock is not None

        # Test concurrent access
        async def access_session(tool_name: str):
            await session.record_access(tool_name)

        tasks = [access_session(f"tool_{i}") for i in range(5)]
        await asyncio.gather(*tasks)

        assert session.access_count == 5


class TestEnhancedPathTraversalFixes:
    """Test enhanced path traversal fixes with symlink detection"""

    def test_path_resolution_order_fix(self):
        """Test that path validation correctly blocks traversal attempts"""
        # The fix ensures ".." is checked BEFORE path resolution to prevent bypasses
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a nested directory structure
            nested_dir = Path(tmpdir) / "level1" / "level2"
            nested_dir.mkdir(parents=True, exist_ok=True)

            # This path has ".." and should be blocked even if it would resolve safely
            unsafe_path = str(nested_dir / ".." / "circuit_breakers.json")

            # Should be blocked because it contains ".."
            with pytest.raises(ValueError, match="Path traversal attempt detected"):
                CircuitBreakerStore(unsafe_path)

            # Instead, use a proper absolute path to the desired location
            safe_path = str(nested_dir.parent / "circuit_breakers.json")

            # This should work - no ".." in the path
            store = CircuitBreakerStore(safe_path)
            assert store.storage_path == Path(safe_path).resolve()

    def test_symlink_attack_detection(self):
        """Test that symlinks pointing outside safe directories are blocked"""
        with tempfile.TemporaryDirectory() as tmpdir:
            symlink_path = Path(tmpdir) / "evil_link.json"

            # Try to create a symlink to /etc/passwd
            try:
                # Create symlink to a file outside safe directories
                target = (
                    Path("/etc/passwd")
                    if Path("/etc/passwd").exists()
                    else Path("/etc/hosts")
                )
                if target.exists():
                    symlink_path.symlink_to(target)

                    # This should be blocked - match both possible error messages
                    with pytest.raises(
                        ValueError,
                        match="(Symlinks are not allowed|Storage path must be within allowed directories)",
                    ):
                        CircuitBreakerStore(str(symlink_path))
            except (OSError, NotImplementedError):
                # Skip on systems where we can't create symlinks
                pytest.skip("Cannot create symlinks on this system")

    def test_hidden_directory_validation(self):
        """Test that suspicious hidden directories are blocked"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # These should be blocked
            suspicious_paths = [
                str(Path(tmpdir) / ".evil" / "circuit_breakers.json"),
                str(Path(tmpdir) / ".malware" / "data.json"),
            ]

            for path in suspicious_paths:
                # Create parent directory
                Path(path).parent.mkdir(exist_ok=True)

                with pytest.raises(ValueError, match="Hidden directories not allowed"):
                    CircuitBreakerStore(path)

            # These should be allowed (whitelisted hidden dirs)
            allowed_paths = [
                str(Path(tmpdir) / ".context_switcher" / "circuit_breakers.json"),
                str(Path(tmpdir) / ".config" / "circuit_breakers.json"),
                str(Path(tmpdir) / ".local" / "circuit_breakers.json"),
            ]

            for path in allowed_paths:
                # Create parent directory
                Path(path).parent.mkdir(exist_ok=True)

                # Should not raise
                store = CircuitBreakerStore(path)
                assert store.storage_path == Path(path).resolve()


class TestSessionTokenRotation:
    """Test secret key rotation and management improvements"""

    def test_secret_key_from_environment(self):
        """Test loading secret key from environment variable"""
        test_key = secrets.token_urlsafe(32)

        with patch.dict(os.environ, {"CONTEXT_SWITCHER_SECRET_KEY": test_key}):
            loaded_key = _load_or_generate_secret_key()
            assert loaded_key == test_key

    def test_secret_key_persistence(self):
        """Test that secret keys are persisted securely"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("pathlib.Path.home", return_value=Path(tmpdir)):
                # Generate initial key
                key1 = _load_or_generate_secret_key()

                # Verify file was created with proper structure
                secret_file = Path(tmpdir) / ".context_switcher" / "secret_key.json"
                assert secret_file.exists()

                # Check file permissions on Unix
                if os.name != "nt":
                    assert oct(secret_file.stat().st_mode)[-3:] == "600"

                # Load again and verify same key
                key2 = _load_or_generate_secret_key()
                assert key1 == key2

                # Verify file structure
                with open(secret_file) as f:
                    data = json.load(f)
                    assert "current_key" in data
                    assert "previous_keys" in data
                    assert "created_at" in data
                    assert "rotation_count" in data

    def test_key_rotation_functionality(self):
        """Test that key rotation maintains previous keys for grace period"""
        manager = SecretKeyManager()
        original_key = manager.current_key

        # Rotate key
        new_key = manager.rotate_key()

        # Verify rotation
        assert new_key != original_key
        assert manager.current_key == new_key
        assert original_key in manager.previous_keys

        # Test multiple rotations
        keys_rotated = [original_key]
        for _ in range(6):
            old_key = manager.current_key
            manager.rotate_key()
            keys_rotated.append(old_key)

        # Verify only last 5 keys are kept
        assert len(manager.previous_keys) == 5
        # Most recent rotated keys should be in previous_keys
        for key in keys_rotated[-5:]:
            assert key in manager.previous_keys

    def test_validation_with_rotated_keys(self):
        """Test that sessions validate with rotated keys during grace period"""
        manager = ClientBindingManager()

        # Create session with current key
        session = create_secure_session_with_binding(
            session_id="test_rotation", topic="test", initial_tool="test_tool"
        )

        # Verify initial validation
        assert manager._validate_binding_with_rotation(session.client_binding)

        # Rotate the key
        manager.rotate_secret_key()

        # Session should still validate with old key
        assert manager._validate_binding_with_rotation(session.client_binding)

        # After validation, binding should be re-signed with new key
        # Verify it now validates with current key
        assert session.client_binding.validate_binding(manager.key_manager.current_key)


class TestConcurrentLockInitialization:
    """Test fixes for race conditions in lock initialization"""

    def test_concurrent_session_creation(self):
        """Test that concurrent session creation doesn't cause race conditions"""
        sessions = []
        errors = []

        def create_session(index):
            try:
                session = ContextSwitcherSession(
                    session_id=f"concurrent_{index}",
                    topic="test",
                    created_at=datetime.now(timezone.utc),
                )
                sessions.append(session)

                # Verify critical attributes are initialized
                assert session._lock_initialized
                # _initialization_lock should always be created for thread safety
                assert session._initialization_lock is not None
                # _access_lock may be None if no event loop (will be created lazily)
            except Exception as e:
                errors.append(e)

        # Create threads for concurrent session creation
        threads = []
        for i in range(20):
            thread = threading.Thread(target=create_session, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(sessions) == 20

        # Verify all sessions have proper initialization
        for session in sessions:
            assert session._lock_initialized
            # The initialization lock should always be present
            assert session._initialization_lock is not None

    @pytest.mark.asyncio
    async def test_concurrent_record_access(self):
        """Test that concurrent record_access calls are thread-safe"""
        session = ContextSwitcherSession(
            session_id="test_concurrent_access",
            topic="test",
            created_at=datetime.now(timezone.utc),
        )

        # Reset access count
        session.access_count = 0
        num_concurrent_accesses = 50

        # Create concurrent access tasks
        async def record_access_task(index):
            await session.record_access(f"tool_{index}")

        # Run concurrently
        tasks = [record_access_task(i) for i in range(num_concurrent_accesses)]
        await asyncio.gather(*tasks)

        # Verify all accesses were recorded
        assert session.access_count == num_concurrent_accesses
        assert session.version == num_concurrent_accesses


class TestIntegrationScenarios:
    """Integration tests for security fixes"""

    @pytest.mark.asyncio
    async def test_secure_circuit_breaker_with_model_validation(self):
        """Test that circuit breaker and model validation work together"""
        # Create circuit breaker with safe path
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_path = Path(tmpdir) / "test_breakers.json"
            store = CircuitBreakerStore(str(safe_path))

            # Test saving state
            await store.save_state(
                "test_backend", {"failure_count": 3, "state": "OPEN"}
            )

            # Test loading state
            state = await store.load_state("test_backend")
            assert state["failure_count"] == 3
            assert state["state"] == "OPEN"

    def test_session_security_with_strong_binding(self):
        """Test session with strong client binding"""
        session = ContextSwitcherSession(
            session_id="secure_session", created_at=datetime.now(timezone.utc)
        )

        # Create secure binding
        binding = ClientBinding(
            session_entropy="strong_entropy_value",
            creation_timestamp=datetime.now(timezone.utc),
            binding_signature="",
            access_pattern_hash="initial_pattern_hash",
        )

        secret = "strong_secret_key"
        binding.binding_signature = binding.generate_binding_signature(secret)
        session.client_binding = binding

        # Test binding validation
        assert session.is_binding_valid(secret) is True
        assert session.is_binding_valid("wrong_secret") is False

        # Test suspicious activity detection
        for i in range(5):
            binding.validation_failures += 1

        assert binding.is_suspicious() is True

    @pytest.mark.asyncio
    async def test_complete_secure_workflow(self):
        """Test complete secure session workflow with all fixes"""
        # Use secure session creation
        session = create_secure_session_with_binding(
            session_id="integration_test",
            topic="security integration test",
            initial_tool="start_analysis",
        )

        # Verify session has proper security setup
        assert session.client_binding is not None
        assert session._lock_initialized
        assert session._access_lock is not None

        # Test concurrent access (tests lock fix)
        tasks = [session.record_access(f"tool_{i}") for i in range(10)]
        await asyncio.gather(*tasks)

        # Initial access + 10 concurrent
        assert session.access_count == 11

        # Test circuit breaker with safe path
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_path = Path(tmpdir) / "integration_breakers.json"
            store = CircuitBreakerStore(str(safe_path))

            # Save and load state
            await store.save_state("test", {"state": "CLOSED"})
            state = await store.load_state("test")
            assert state["state"] == "CLOSED"
