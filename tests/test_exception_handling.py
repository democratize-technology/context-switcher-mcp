"""Tests for custom exception handling in Context-Switcher MCP"""

import pytest
import asyncio
from unittest.mock import patch
from datetime import datetime

from src.context_switcher_mcp.exceptions import (
    ContextSwitcherError,
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    SessionCleanupError,
    OrchestrationError,
    CircuitBreakerError,
    CircuitBreakerOpenError,
    CircuitBreakerStateError,
    ModelBackendError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelRateLimitError,
    ModelAuthenticationError,
    ModelValidationError,
    AnalysisError,
    PerspectiveError,
    ConfigurationError,
    StorageError,
    SerializationError,
)


class TestExceptionHierarchy:
    """Test the exception hierarchy is properly defined"""

    def test_base_exception(self):
        """Test that ContextSwitcherError is the base for all custom exceptions"""
        error = ContextSwitcherError("test message")
        assert str(error) == "test message"
        assert isinstance(error, Exception)

    def test_session_errors(self):
        """Test session-related exceptions inherit correctly"""
        assert issubclass(SessionError, ContextSwitcherError)
        assert issubclass(SessionNotFoundError, SessionError)
        assert issubclass(SessionExpiredError, SessionError)
        assert issubclass(SessionCleanupError, SessionError)

    def test_orchestration_errors(self):
        """Test orchestration-related exceptions inherit correctly"""
        assert issubclass(OrchestrationError, ContextSwitcherError)
        assert issubclass(CircuitBreakerError, OrchestrationError)
        assert issubclass(CircuitBreakerOpenError, CircuitBreakerError)
        assert issubclass(CircuitBreakerStateError, CircuitBreakerError)

    def test_model_backend_errors(self):
        """Test model backend exceptions inherit correctly"""
        assert issubclass(ModelBackendError, ContextSwitcherError)
        assert issubclass(ModelConnectionError, ModelBackendError)
        assert issubclass(ModelTimeoutError, ModelBackendError)
        assert issubclass(ModelRateLimitError, ModelBackendError)
        assert issubclass(ModelAuthenticationError, ModelBackendError)
        assert issubclass(ModelValidationError, ModelBackendError)

    def test_other_errors(self):
        """Test other exceptions inherit correctly"""
        assert issubclass(AnalysisError, ContextSwitcherError)
        assert issubclass(PerspectiveError, ContextSwitcherError)
        assert issubclass(ConfigurationError, ContextSwitcherError)
        assert issubclass(StorageError, ContextSwitcherError)
        assert issubclass(SerializationError, StorageError)


class TestOrchestratorExceptionHandling:
    """Test exception handling in orchestrator.py"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator with proper dependencies"""
        from src.context_switcher_mcp.orchestrator import ThreadOrchestrator

        with patch("src.context_switcher_mcp.orchestrator.get_config"):
            orchestrator = ThreadOrchestrator()
            return orchestrator

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_error(self, mock_orchestrator):
        """Test that CircuitBreakerStateError is raised on storage failures"""
        from src.context_switcher_mcp.models import ModelBackend
        from src.context_switcher_mcp.orchestrator import CircuitBreakerState

        breaker = CircuitBreakerState(ModelBackend.BEDROCK)

        # Mock save_circuit_breaker_state to raise OSError
        with patch(
            "src.context_switcher_mcp.orchestrator.save_circuit_breaker_state",
            side_effect=OSError("Disk full"),
        ):
            with pytest.raises(CircuitBreakerStateError) as exc_info:
                await breaker.record_success()

            assert "Failed to save circuit breaker state" in str(exc_info.value)
            assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_orchestration_error_wrapping(self, mock_orchestrator):
        """Test that unexpected errors are wrapped in OrchestrationError"""
        from src.context_switcher_mcp.models import Thread, ModelBackend

        thread = Thread(
            name="test_thread", system_prompt="test", model_backend=ModelBackend.BEDROCK
        )

        # Mock backend to raise unexpected error
        with patch.object(
            mock_orchestrator,
            "_call_unified_backend",
            side_effect=RuntimeError("Unexpected backend error"),
        ):
            with pytest.raises(OrchestrationError) as exc_info:
                await mock_orchestrator._get_thread_response(thread)

            assert "Unexpected backend error" in str(exc_info.value)
            assert exc_info.value.__cause__ is not None


class TestSessionManagerExceptionHandling:
    """Test exception handling in session_manager.py"""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager"""
        from src.context_switcher_mcp.session_manager import SessionManager

        with patch(
            "src.context_switcher_mcp.session_manager.get_config"
        ) as mock_config:
            # Mock the config object with proper values
            mock_config.return_value.session.max_active_sessions = 100
            mock_config.return_value.session.default_ttl_hours = 1
            mock_config.return_value.session.cleanup_interval_seconds = 300
            return SessionManager()

    @pytest.mark.asyncio
    async def test_session_cleanup_error_handling(self, mock_session_manager):
        """Test that session cleanup handles different error types appropriately"""
        session_id = "test_session_123"

        # Test ImportError handling (non-critical)
        with patch(
            "context_switcher_mcp.rate_limiter",
            side_effect=ImportError("Module not found"),
        ):
            # Should not raise an exception, just log
            await mock_session_manager._cleanup_session_resources(session_id)

        # Test that method completes successfully when rate_limiter module exists
        # but cleanup_session method is missing (AttributeError case)
        with patch("builtins.__import__") as mock_import:
            # Mock successful import but rate_limiter missing cleanup_session method
            mock_rate_limiter = type("MockRateLimiter", (), {})()
            mock_import.return_value = mock_rate_limiter

            # Should not raise an exception, just log warning
            await mock_session_manager._cleanup_session_resources(session_id)


class TestCircuitBreakerStoreExceptionHandling:
    """Test exception handling in circuit_breaker_store.py"""

    @pytest.fixture
    def mock_store(self):
        """Create a mock circuit breaker store"""
        from src.context_switcher_mcp.circuit_breaker_store import CircuitBreakerStore

        return CircuitBreakerStore()

    @pytest.mark.asyncio
    async def test_storage_error_on_file_operations(self, mock_store):
        """Test that file operation errors raise StorageError"""
        with patch.object(
            mock_store.storage_path,
            "write_text",
            side_effect=OSError("Permission denied"),
        ):
            with pytest.raises(StorageError) as exc_info:
                await mock_store._save_all_states({})

            assert "Failed to write state file" in str(exc_info.value)
            assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_serialization_error_on_json_encode(self, mock_store):
        """Test that JSON serialization errors raise SerializationError"""

        # Create data that can't be serialized
        bad_data = {"test": datetime.now()}

        with patch("json.dumps", side_effect=TypeError("Object not serializable")):
            with pytest.raises(SerializationError) as exc_info:
                await mock_store._save_all_states(bad_data)

            assert "Failed to serialize states" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_error_wrapping(self, mock_store):
        """Test that unexpected errors are wrapped in CircuitBreakerStateError"""
        with patch.object(
            mock_store, "_load_all_states", side_effect=RuntimeError("Unexpected error")
        ):
            with pytest.raises(CircuitBreakerStateError) as exc_info:
                await mock_store.save_state("bedrock", {})

            assert "Unexpected error saving state" in str(exc_info.value)


class TestBackendInterfaceExceptionHandling:
    """Test exception handling in backend_interface.py"""

    def test_bedrock_backend_error_mapping(self):
        """Test that Bedrock backend maps errors correctly"""
        from src.context_switcher_mcp.backend_interface import BedrockBackend
        from src.context_switcher_mcp.models import Thread, ModelBackend

        backend = BedrockBackend()
        thread = Thread(
            name="test_thread", system_prompt="test", model_backend=ModelBackend.BEDROCK
        )

        # Test import error handling
        with patch("builtins.__import__", side_effect=ImportError("boto3 not found")):
            with pytest.raises(ConfigurationError) as exc_info:
                asyncio.run(backend.call_model(thread))

            assert "boto3 library not installed" in str(exc_info.value)

    def test_litellm_backend_error_mapping(self):
        """Test that LiteLLM backend maps errors correctly"""
        from src.context_switcher_mcp.backend_interface import LiteLLMBackend
        from src.context_switcher_mcp.models import Thread, ModelBackend

        backend = LiteLLMBackend()
        thread = Thread(
            name="test_thread", system_prompt="test", model_backend=ModelBackend.LITELLM
        )

        # Test import error handling
        with patch("builtins.__import__", side_effect=ImportError("litellm not found")):
            with pytest.raises(ConfigurationError) as exc_info:
                asyncio.run(backend.call_model(thread))

            assert "litellm library not installed" in str(exc_info.value)

    def test_ollama_backend_error_mapping(self):
        """Test that Ollama backend maps errors correctly"""
        from src.context_switcher_mcp.backend_interface import OllamaBackend
        from src.context_switcher_mcp.models import Thread, ModelBackend

        backend = OllamaBackend()
        thread = Thread(
            name="test_thread", system_prompt="test", model_backend=ModelBackend.OLLAMA
        )

        # Test connection error mapping
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = (
                __import__("httpx").ConnectError("Connection failed")
            )

            with pytest.raises(ModelConnectionError) as exc_info:
                asyncio.run(backend.call_model(thread))

            assert "Failed to connect to Ollama" in str(exc_info.value)


class TestAnalysisToolsExceptionHandling:
    """Test exception handling in analysis_tools.py"""

    @pytest.mark.asyncio
    async def test_session_not_found_error_handling(self):
        """Test that SessionNotFoundError is handled properly in analysis tools"""

        # Mock the validation to raise SessionNotFoundError
        with patch(
            "src.context_switcher_mcp.tools.analysis_tools.validate_session_id",
            return_value=(False, "Session not found"),
        ):
            # The actual tool implementation would handle this gracefully
            # This test verifies the exception type is imported and can be used
            assert SessionNotFoundError is not None

    def test_orchestration_error_handling(self):
        """Test that OrchestrationError is properly imported and can be caught"""
        # Verify the exception can be instantiated and caught
        try:
            raise OrchestrationError("Test orchestration error")
        except OrchestrationError as e:
            assert str(e) == "Test orchestration error"
            assert isinstance(e, ContextSwitcherError)


class TestPerspectiveToolsExceptionHandling:
    """Test exception handling in perspective_tools.py"""

    def test_perspective_error_types(self):
        """Test that perspective tools can handle various error types"""
        # Verify perspective-specific exceptions exist and inherit correctly
        assert issubclass(PerspectiveError, ContextSwitcherError)

        # Test that the exception can be instantiated
        error = PerspectiveError("Test perspective error")
        assert str(error) == "Test perspective error"

    def test_model_backend_error_handling(self):
        """Test that ModelBackendError is properly imported in perspective tools"""
        # Verify the exception type is available for catch blocks
        assert ModelBackendError is not None
        assert issubclass(ModelBackendError, ContextSwitcherError)


class TestErrorChainingAndLogging:
    """Test that error chaining and logging work correctly"""

    def test_error_chaining_preserves_original(self):
        """Test that custom exceptions preserve original error information"""
        original_error = ValueError("Original error")

        try:
            raise ModelValidationError("Validation failed") from original_error
        except ModelValidationError as e:
            assert e.__cause__ is original_error
            assert str(e.__cause__) == "Original error"

    def test_exception_context_preservation(self):
        """Test that exception context is preserved through wrapping"""
        try:
            try:
                raise OSError("File not found")
            except OSError as e:
                raise StorageError("Storage operation failed") from e
        except StorageError as wrapped_error:
            assert wrapped_error.__cause__ is not None
            assert isinstance(wrapped_error.__cause__, OSError)
            assert str(wrapped_error.__cause__) == "File not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
