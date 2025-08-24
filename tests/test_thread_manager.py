"""Tests for ThreadManager component"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from context_switcher_mcp.circuit_breaker_manager import (
    CircuitBreakerState,  # noqa: E402
)
from context_switcher_mcp.metrics_manager import (  # noqa: E402
    ThreadMetrics,
    ThreadOrchestrationMetrics,
)
from context_switcher_mcp.models import ModelBackend, Thread  # noqa: E402
from context_switcher_mcp.thread_manager import ThreadManager  # noqa: E402


@pytest.fixture
def thread_manager():
    """Create ThreadManager instance for testing"""
    return ThreadManager(max_retries=2, retry_delay=0.1)


@pytest.fixture
def mock_thread():
    """Create mock thread for testing"""
    thread = MagicMock(spec=Thread)
    thread.name = "test_thread"
    thread.model_backend = ModelBackend.BEDROCK
    thread.conversation_history = []
    thread.add_message = MagicMock()
    return thread


@pytest.fixture
def mock_threads():
    """Create dictionary of mock threads"""
    threads = {}
    for _i, name in enumerate(["thread1", "thread2", "thread3"]):
        thread = MagicMock(spec=Thread)
        thread.name = name
        thread.model_backend = ModelBackend.BEDROCK
        thread.conversation_history = []
        thread.add_message = MagicMock()
        threads[name] = thread
    return threads


class TestThreadManager:
    """Test ThreadManager functionality"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test ThreadManager initialization"""
        manager = ThreadManager(max_retries=5, retry_delay=0.5)

        # Properties are now in composed components
        assert manager.thread_lifecycle_manager.max_retries == 5
        assert manager.thread_lifecycle_manager.retry_delay == 0.5
        assert len(manager.circuit_breaker_manager.circuit_breakers) == len(
            ModelBackend
        )
        assert not manager.circuit_breaker_manager._states_loaded

    @pytest.mark.asyncio
    async def test_broadcast_message_success(self, thread_manager, mock_threads):
        """Test successful broadcast to multiple threads"""
        # Mock the backend factory in the lifecycle manager
        with patch(
            "context_switcher_mcp.thread_lifecycle_manager.BackendFactory.get_backend"
        ) as mock_get_backend:
            mock_backend = AsyncMock()
            mock_backend.call_model.side_effect = [
                "Response 1",
                "Response 2",
                "Response 3",
            ]
            mock_get_backend.return_value = mock_backend

            # Mock circuit breaker state loading
            with patch.object(
                thread_manager.circuit_breaker_manager, "ensure_states_loaded"
            ):
                result = await thread_manager.broadcast_message(
                    mock_threads, "test message"
                )

                assert len(result) == 3
                assert result["thread1"] == "Response 1"
                assert result["thread2"] == "Response 2"
                assert result["thread3"] == "Response 3"

                # Verify threads received the message
                for thread in mock_threads.values():
                    # Check that add_message was called with user message
                    thread.add_message.assert_any_call("user", "test message")

    @pytest.mark.asyncio
    async def test_broadcast_message_with_errors(self, thread_manager, mock_threads):
        """Test broadcast handling thread errors"""
        # Mock the lifecycle manager execute_threads_parallel method
        with patch.object(
            thread_manager.thread_lifecycle_manager, "execute_threads_parallel"
        ) as mock_response:
            # Return results dictionary with error
            mock_response.return_value = {
                "thread1": "Success response",
                "thread2": "ERROR: Connection failed",
                "thread3": "Another success",
            }

            result = await thread_manager.broadcast_message(
                mock_threads, "test message"
            )

            assert len(result) == 3
            assert result["thread1"] == "Success response"
            assert "ERROR:" in result["thread2"]
            assert result["thread3"] == "Another success"

    @pytest.mark.asyncio
    async def test_broadcast_message_stream(self, thread_manager, mock_threads):
        """Test streaming broadcast functionality"""

        # Mock the streaming coordinator
        async def mock_stream_events(threads, message, session_id="unknown"):
            for name in ["thread1", "thread2", "thread3"]:
                yield {
                    "type": "start",
                    "thread_name": name,
                    "content": "",
                    "timestamp": 1000,
                }
                yield {
                    "type": "chunk",
                    "thread_name": name,
                    "content": f"Response from {name}",
                    "timestamp": 1001,
                }
                yield {
                    "type": "complete",
                    "thread_name": name,
                    "content": "",
                    "timestamp": 1002,
                }

        with patch.object(
            thread_manager.streaming_coordinator,
            "broadcast_stream",
            side_effect=mock_stream_events,
        ):
            events = []
            async for event in thread_manager.broadcast_message_stream(
                mock_threads, "test message"
            ):
                events.append(event)

            # Should have start events plus streaming events for each thread
            assert len(events) >= 6  # At least 2 events per thread

            # Check that start events were generated
            start_events = [e for e in events if e["type"] == "start"]
            assert len(start_events) == 3

    @pytest.mark.asyncio
    async def test_get_thread_metrics_empty(self, thread_manager):
        """Test getting metrics when history is empty"""
        result = await thread_manager.get_thread_metrics()
        assert result["message"] == "No thread metrics available"

    @pytest.mark.asyncio
    async def test_get_thread_metrics_with_data(self, thread_manager):
        """Test getting metrics with data in history"""
        # Add mock metrics to history
        metrics = ThreadOrchestrationMetrics(
            session_id="test",
            operation_type="broadcast",
            start_time=1234567890.0,
            end_time=1234567892.0,
            total_threads=2,
            successful_threads=1,
            failed_threads=1,
        )
        thread_manager.metrics_manager.metrics_history.append(metrics)

        result = await thread_manager.get_thread_metrics()

        assert "thread_summary" in result
        assert result["thread_summary"]["total_operations"] == 1
        assert "backend_performance" in result
        assert "metrics_storage" in result

    def test_circuit_breaker_state_initialization(self):
        """Test CircuitBreakerState initialization"""
        breaker = CircuitBreakerState(backend=ModelBackend.BEDROCK)

        assert breaker.backend == ModelBackend.BEDROCK
        assert breaker.failure_count == 0
        assert breaker.state == "CLOSED"
        assert breaker.should_allow_request() is True

    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opening after threshold"""
        breaker = CircuitBreakerState(backend=ModelBackend.BEDROCK, failure_threshold=2)

        # Should allow requests initially
        assert breaker.should_allow_request() is True

        # After failures, should still allow until threshold
        breaker.failure_count = 1
        assert breaker.should_allow_request() is True

        # After threshold, should open and block requests
        breaker.failure_count = 2
        breaker.state = "OPEN"
        breaker.last_failure_time = datetime.now(timezone.utc)
        assert breaker.should_allow_request() is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_record_success(self):
        """Test circuit breaker success recording"""
        breaker = CircuitBreakerState(backend=ModelBackend.BEDROCK)
        breaker.failure_count = 5
        breaker.state = "HALF_OPEN"

        with patch(
            "context_switcher_mcp.circuit_breaker_manager.save_circuit_breaker_state"
        ) as mock_save:
            mock_save.return_value = None
            await breaker.record_success()

            assert breaker.failure_count == 0
            assert breaker.state == "CLOSED"
            assert breaker.last_failure_time is None
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_record_failure(self):
        """Test circuit breaker failure recording"""
        breaker = CircuitBreakerState(backend=ModelBackend.BEDROCK, failure_threshold=2)

        with patch(
            "context_switcher_mcp.circuit_breaker_manager.save_circuit_breaker_state"
        ) as mock_save:
            mock_save.return_value = None
            await breaker.record_failure()

            assert breaker.failure_count == 1
            assert breaker.last_failure_time is not None
            # Should still be closed after first failure
            assert breaker.state == "CLOSED"

            # Second failure should open the breaker
            await breaker.record_failure()
            assert breaker.failure_count == 2
            assert breaker.state == "OPEN"

    def test_get_circuit_breaker_status(self, thread_manager):
        """Test getting circuit breaker status"""
        # Set up a breaker with some state
        breaker = thread_manager.circuit_breaker_manager.circuit_breakers[
            ModelBackend.BEDROCK
        ]
        breaker.failure_count = 3
        breaker.state = "OPEN"
        breaker.last_failure_time = datetime(2023, 1, 1, 12, 0, 0)

        status = thread_manager.get_circuit_breaker_status()

        assert "bedrock" in status
        assert status["bedrock"]["failure_count"] == 3
        assert status["bedrock"]["state"] == "OPEN"
        assert status["bedrock"]["last_failure"] == "2023-01-01T12:00:00"

    def test_reset_circuit_breakers(self, thread_manager):
        """Test resetting all circuit breakers"""
        # Set up breakers with failure states
        for breaker in thread_manager.circuit_breaker_manager.circuit_breakers.values():
            breaker.failure_count = 5
            breaker.state = "OPEN"
            breaker.last_failure_time = datetime.now(timezone.utc)

        reset_status = thread_manager.reset_circuit_breakers()

        # All breakers should be reset
        for (
            backend,
            breaker,
        ) in thread_manager.circuit_breaker_manager.circuit_breakers.items():
            assert breaker.state == "CLOSED"
            assert breaker.failure_count == 0
            assert breaker.last_failure_time is None
            assert backend.value in reset_status
            assert "CLOSED" in reset_status[backend.value]

    def test_thread_metrics_properties(self):
        """Test ThreadMetrics property calculations"""
        metrics = ThreadMetrics(
            thread_name="test", start_time=1234567890.0, end_time=1234567892.0
        )

        assert metrics.execution_time == 2.0

        # Test with no end time
        metrics_incomplete = ThreadMetrics(thread_name="test", start_time=1234567890.0)
        assert metrics_incomplete.execution_time is None

    def test_orchestration_metrics_properties(self):
        """Test ThreadOrchestrationMetrics property calculations"""
        metrics = ThreadOrchestrationMetrics(
            session_id="test",
            operation_type="broadcast",
            start_time=1234567890.0,
            end_time=1234567895.0,
            total_threads=10,
            successful_threads=7,
        )

        assert metrics.execution_time == 5.0
        assert metrics.success_rate == 70.0

        # Test with zero threads
        empty_metrics = ThreadOrchestrationMetrics(
            session_id="test",
            operation_type="broadcast",
            start_time=1234567890.0,
            total_threads=0,
        )
        assert empty_metrics.success_rate == 0.0


class TestThreadManagerIntegration:
    """Integration tests for ThreadManager"""

    @pytest.mark.asyncio
    async def test_full_broadcast_workflow(self):
        """Test complete broadcast workflow with mocked backends"""
        manager = ThreadManager(max_retries=1, retry_delay=0.01)

        # Create test threads
        threads = {}
        for name in ["perspective1", "perspective2"]:
            thread = MagicMock(spec=Thread)
            thread.name = name
            thread.model_backend = ModelBackend.BEDROCK
            thread.conversation_history = []
            thread.add_message = MagicMock()
            threads[name] = thread

        # Mock the backend factory in the lifecycle manager
        with patch(
            "context_switcher_mcp.thread_lifecycle_manager.BackendFactory.get_backend"
        ) as mock_get_backend:
            mock_backend = AsyncMock()
            mock_backend.call_model.side_effect = [
                "Perspective 1 response",
                "Perspective 2 response",
            ]
            mock_get_backend.return_value = mock_backend

            # Mock circuit breaker state loading
            with patch.object(
                manager.circuit_breaker_manager, "ensure_states_loaded"
            ) as mock_load:
                mock_load.return_value = None

                result = await manager.broadcast_message(
                    threads, "Analyze this topic", "test_session"
                )

                assert len(result) == 2
                assert result["perspective1"] == "Perspective 1 response"
                assert result["perspective2"] == "Perspective 2 response"

                # Verify backend was called for each thread
                assert mock_backend.call_model.call_count == 2
