"""Tests for security enhancements in Context-Switcher MCP"""

import asyncio
from unittest.mock import patch

import pytest
from src.context_switcher_mcp.circuit_breaker_manager import CircuitBreakerState
from src.context_switcher_mcp.handlers.session_handler import generate_secure_session_id
from src.context_switcher_mcp.models import ModelBackend


class TestSecureSessionIDGeneration:
    """Test secure session ID generation functionality"""

    def test_session_id_is_string(self):
        """Test that session ID is a string"""
        session_id = generate_secure_session_id()
        assert isinstance(session_id, str)

    def test_session_id_length(self):
        """Test that session ID has adequate length"""
        session_id = generate_secure_session_id()
        # secrets.token_urlsafe(32) generates 43 characters (base64 encoded 32 bytes)
        assert len(session_id) >= 40
        assert len(session_id) <= 50  # Upper bound for base64 encoding variations

    def test_session_id_uniqueness(self):
        """Test that generated session IDs are unique"""
        session_ids = [generate_secure_session_id() for _ in range(100)]
        assert len(set(session_ids)) == 100, "All session IDs should be unique"

    def test_session_id_url_safe(self):
        """Test that session ID contains only URL-safe characters"""
        session_id = generate_secure_session_id()
        # URL-safe base64 uses A-Z, a-z, 0-9, -, _
        allowed_chars = set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        )
        assert all(
            c in allowed_chars for c in session_id
        ), f"Session ID contains invalid characters: {session_id}"

    def test_session_id_entropy(self):
        """Test that session IDs have high entropy"""
        # Generate many session IDs and check character distribution
        session_ids = [generate_secure_session_id() for _ in range(50)]
        all_chars = "".join(session_ids)

        # Count unique characters - should have good distribution
        unique_chars = set(all_chars)
        assert (
            len(unique_chars) >= 20
        ), f"Low character diversity: {len(unique_chars)} unique chars"

    def test_session_id_not_predictable(self):
        """Test that session IDs are not predictable"""
        # Generate pairs of session IDs and check they don't follow patterns
        pairs = [
            (generate_secure_session_id(), generate_secure_session_id())
            for _ in range(10)
        ]

        for id1, id2 in pairs:
            # Check they don't have identical prefixes/suffixes
            common_prefix = 0
            for i in range(min(len(id1), len(id2))):
                if id1[i] == id2[i]:
                    common_prefix += 1
                else:
                    break
            assert common_prefix < 5, f"Too similar session IDs: {id1}, {id2}"

    @patch("src.context_switcher_mcp.handlers.session_handler.secrets.token_urlsafe")
    def test_uses_secrets_module(self, mock_token_urlsafe):
        """Test that the function uses the secrets module"""
        mock_token_urlsafe.return_value = "mock_secure_token_123"

        result = generate_secure_session_id()

        mock_token_urlsafe.assert_called_once_with(32)
        assert result == "mock_secure_token_123"


class TestCircuitBreakerErrorHandling:
    """Test circuit breaker error handling enhancements"""

    @pytest.mark.asyncio
    async def test_record_success_is_async(self):
        """Test that record_success is now an async method"""
        cb = CircuitBreakerState(backend=ModelBackend.BEDROCK)
        # Should be able to await it
        with patch(
            "src.context_switcher_mcp.circuit_breaker_manager.save_circuit_breaker_state"
        ):
            await cb.record_success()
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_record_failure_is_async(self):
        """Test that record_failure is now an async method"""
        cb = CircuitBreakerState(backend=ModelBackend.BEDROCK)
        # Should be able to await it
        with patch(
            "src.context_switcher_mcp.circuit_breaker_manager.save_circuit_breaker_state"
        ):
            await cb.record_failure()
        assert cb.failure_count == 1

    @pytest.mark.asyncio
    async def test_record_success_handles_save_errors(self):
        """Test that record_success handles save errors gracefully"""
        cb = CircuitBreakerState(backend=ModelBackend.BEDROCK)

        with patch(
            "src.context_switcher_mcp.circuit_breaker_manager.save_circuit_breaker_state"
        ) as mock_save:
            # Make save fail
            mock_save.side_effect = Exception("Save failed")

            with patch(
                "src.context_switcher_mcp.circuit_breaker_manager.logger"
            ) as mock_logger:
                # Call should not raise even if save fails
                await cb.record_success()

                # Verify error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args[0][0]
                assert "Failed to save circuit breaker state" in call_args

    @pytest.mark.asyncio
    async def test_record_failure_handles_save_errors(self):
        """Test that record_failure handles save errors gracefully"""
        cb = CircuitBreakerState(backend=ModelBackend.BEDROCK)

        with patch(
            "src.context_switcher_mcp.circuit_breaker_manager.save_circuit_breaker_state"
        ) as mock_save:
            # Make save fail
            mock_save.side_effect = Exception("Save failed")

            with patch(
                "src.context_switcher_mcp.circuit_breaker_manager.logger"
            ) as mock_logger:
                # Call should not raise even if save fails
                await cb.record_failure()

                # Verify error was logged
                mock_logger.error.assert_called_once()
                call_args = mock_logger.error.call_args[0][0]
                assert "Failed to save circuit breaker state" in call_args


class TestAsyncErrorHandlingIntegration:
    """Test integration of async error handling in real scenarios"""

    @pytest.mark.asyncio
    async def test_fire_and_forget_error_captured(self):
        """Test that fire-and-forget async errors are captured"""
        errors_captured = []

        def capture_error(task):
            """Custom error handler that captures errors"""
            try:
                task.result()
            except Exception as e:
                errors_captured.append(str(e))

        # Create a failing async function
        async def failing_operation():
            await asyncio.sleep(0.01)  # Small delay
            raise RuntimeError("Async operation failed")

        # Use the fire-and-forget pattern with error handling
        task = asyncio.create_task(failing_operation())
        task.add_done_callback(capture_error)

        # Wait for task to complete
        await asyncio.sleep(0.1)

        # Verify error was captured
        assert len(errors_captured) == 1
        assert "Async operation failed" in errors_captured[0]

    @pytest.mark.asyncio
    async def test_multiple_concurrent_errors_handled(self):
        """Test that multiple concurrent async errors are all handled"""
        errors_captured = []

        def capture_error(task):
            """Custom error handler that captures errors"""
            try:
                task.result()
            except Exception as e:
                errors_captured.append(str(e))

        # Create multiple failing async functions
        async def failing_operation(msg):
            await asyncio.sleep(0.01)
            raise ValueError(f"Error: {msg}")

        # Start multiple fire-and-forget tasks
        tasks = []
        for i in range(5):
            task = asyncio.create_task(failing_operation(f"task_{i}"))
            task.add_done_callback(capture_error)
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.sleep(0.2)

        # Verify all errors were captured
        assert len(errors_captured) == 5
        for i in range(5):
            assert any(f"task_{i}" in error for error in errors_captured)


class TestSecurityRegression:
    """Test that security fixes don't introduce regressions"""

    def test_session_id_generation_performance(self):
        """Test that secure session ID generation is performant"""
        import time

        start_time = time.time()
        session_ids = [generate_secure_session_id() for _ in range(1000)]
        end_time = time.time()

        # Should generate 1000 session IDs in under 1 second
        assert end_time - start_time < 1.0
        assert len(set(session_ids)) == 1000  # All unique

    def test_backward_compatibility_session_id_format(self):
        """Test that new session IDs are compatible with existing validation"""
        session_id = generate_secure_session_id()

        # Test against existing MAX_SESSION_ID_LENGTH constant (100)
        MAX_SESSION_ID_LENGTH = 100
        assert len(session_id) <= MAX_SESSION_ID_LENGTH

        # Test that it's a valid string that can be used in URLs and databases
        assert isinstance(session_id, str)
        assert session_id.isascii()
        assert " " not in session_id  # No spaces
        assert "\n" not in session_id  # No newlines

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_persistence_still_works(self):
        """Test that circuit breaker state changes are still persisted"""
        cb = CircuitBreakerState(backend=ModelBackend.BEDROCK)

        # Test that state changes work normally
        initial_state = cb.state
        assert initial_state == "CLOSED"

        # Record failures to trigger state change
        with patch(
            "src.context_switcher_mcp.circuit_breaker_manager.save_circuit_breaker_state"
        ):
            for _ in range(cb.failure_threshold):
                await cb.record_failure()

        # Verify state changed
        assert cb.state == "OPEN"
        assert cb.failure_count == cb.failure_threshold

    @pytest.mark.asyncio
    async def test_error_handler_doesnt_affect_circuit_breaker_logic(self):
        """Test that adding error handlers doesn't affect circuit breaker logic"""
        cb = CircuitBreakerState(backend=ModelBackend.BEDROCK)

        # Test normal circuit breaker behavior
        assert cb.should_allow_request() is True

        # Mock the save operations
        with patch(
            "src.context_switcher_mcp.circuit_breaker_manager.save_circuit_breaker_state"
        ):
            # Trigger failures
            for _ in range(cb.failure_threshold):
                await cb.record_failure()

            # Should be open now
            assert cb.state == "OPEN"
            assert cb.should_allow_request() is False

            # Test success recovery
            await cb.record_success()
            assert cb.failure_count == 0
