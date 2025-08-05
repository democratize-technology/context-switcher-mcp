"""Tests for high-priority security fixes"""

import sys

sys.path.insert(0, "src")  # noqa: E402

import asyncio  # noqa: E402
import tempfile  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from context_switcher_mcp.backend_interface import LiteLLMBackend, OllamaBackend  # noqa: E402
from context_switcher_mcp.circuit_breaker_store import CircuitBreakerStore  # noqa: E402
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
            "../../../etc/passwd.json",
            "/etc/passwd.json",
            "~/../../etc/passwd.json",
            "/var/log/system.json",
            "../../../Windows/System32/config/SAM.json",
        ]

        for path in malicious_paths:
            with pytest.raises(
                ValueError,
                match="Storage path must be within home directory or temp directory",
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
            creation_timestamp=datetime.utcnow(),
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
            creation_timestamp=datetime.utcnow(),
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
        response = await orchestrator.thread_manager._get_thread_response(thread)

        # Verify failure was recorded
        assert circuit_breaker.failure_count > initial_failures
        assert "AORP_ERROR" in response

    @pytest.mark.asyncio
    async def test_circuit_breaker_does_not_record_non_transient_failures(self):
        """Test that circuit breaker doesn't record non-transient failures"""
        # Mock circuit breaker state loading to prevent restoration
        with patch(
            "context_switcher_mcp.thread_manager.load_circuit_breaker_state"
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
                await orchestrator.thread_manager._get_thread_response(thread)
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
            session_id="test", created_at=datetime.utcnow()
        )

        # Simulate __post_init__ not being called
        session._lock_initialized = False
        session._access_lock = None

        # Create multiple concurrent access attempts
        async def access_session(tool_name: str):
            await session.record_access(tool_name)
            return session.access_count

        # Run concurrent accesses
        tasks = [access_session(f"tool_{i}") for i in range(10)]
        await asyncio.gather(*tasks)

        # All accesses should succeed
        assert session.access_count == 10
        assert session._lock_initialized is True
        assert session._access_lock is not None

    @pytest.mark.asyncio
    async def test_session_with_initialized_lock(self):
        """Test that session with properly initialized lock works correctly"""
        session = ContextSwitcherSession(
            session_id="test", created_at=datetime.utcnow()
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
            session_id="secure_session", created_at=datetime.utcnow()
        )

        # Create secure binding
        binding = ClientBinding(
            session_entropy="strong_entropy_value",
            creation_timestamp=datetime.utcnow(),
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
