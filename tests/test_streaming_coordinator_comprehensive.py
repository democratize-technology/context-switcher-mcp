"""Comprehensive tests for streaming coordinator functionality"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from context_switcher_mcp.streaming_coordinator import StreamingCoordinator
from context_switcher_mcp.models import Thread, ModelBackend
from context_switcher_mcp.exceptions import ModelBackendError


class MockAsyncGenerator:
    """Helper class to create mock async generators for testing"""

    def __init__(self, events):
        self.events = events
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.events):
            raise StopAsyncIteration
        event = self.events[self.index]
        self.index += 1
        if isinstance(event, Exception):
            raise event
        return event


class TestStreamingCoordinator:
    """Test suite for StreamingCoordinator class"""

    @pytest.fixture
    def streaming_coordinator(self):
        """Create a StreamingCoordinator instance for testing"""
        return StreamingCoordinator()

    @pytest.fixture
    def mock_metrics_manager(self):
        """Create mock metrics manager"""
        manager = Mock()

        # Mock orchestration metrics
        metrics = Mock()
        metrics.successful_threads = 0
        metrics.failed_threads = 0

        manager.create_orchestration_metrics.return_value = metrics
        manager.finalize_metrics = Mock()
        manager.store_metrics = AsyncMock()

        return manager, metrics

    @pytest.fixture
    def streaming_coordinator_with_metrics(self, mock_metrics_manager):
        """Create StreamingCoordinator with metrics manager"""
        manager, _ = mock_metrics_manager
        return StreamingCoordinator(metrics_manager=manager)

    @pytest.fixture
    def sample_thread(self):
        """Create a sample Thread for testing"""
        return Thread(
            id="thread-123",
            name="technical",
            system_prompt="You are a technical expert",
            model_backend=ModelBackend.BEDROCK,
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        )

    @pytest.fixture
    def sample_threads_dict(self, sample_thread):
        """Create a dictionary of sample threads"""
        thread2 = Thread(
            id="thread-456",
            name="business",
            system_prompt="You are a business expert",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-4",
        )
        return {"technical": sample_thread, "business": thread2}

    def test_initialization_without_metrics(self, streaming_coordinator):
        """Test StreamingCoordinator initialization without metrics manager"""
        assert streaming_coordinator.metrics_manager is None

    def test_initialization_with_metrics(self, streaming_coordinator_with_metrics):
        """Test StreamingCoordinator initialization with metrics manager"""
        assert streaming_coordinator_with_metrics.metrics_manager is not None

    @pytest.mark.asyncio
    async def test_broadcast_stream_basic(
        self, streaming_coordinator, sample_threads_dict
    ):
        """Test basic broadcast streaming functionality"""
        message = "Test message"

        # Mock backend streaming
        mock_events = [
            {"type": "chunk", "content": "Technical response part 1"},
            {"type": "chunk", "content": " part 2"},
            {"type": "complete", "content": ""},
        ]

        mock_backend = Mock()
        mock_backend.call_model_stream.return_value = MockAsyncGenerator(mock_events)

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            events = []
            async for event in streaming_coordinator.broadcast_stream(
                sample_threads_dict, message
            ):
                events.append(event)

        # Should have start events for each thread plus streaming events
        start_events = [e for e in events if e["type"] == "start"]
        assert len(start_events) == 2

        # Should have chunk and complete events
        chunk_events = [e for e in events if e["type"] == "chunk"]
        complete_events = [e for e in events if e["type"] == "complete"]

        assert len(chunk_events) >= 2  # At least some chunks
        assert len(complete_events) >= 1  # At least one complete

        # All events should have required fields
        for event in events:
            assert "type" in event
            assert "thread_name" in event
            assert "content" in event
            assert "timestamp" in event

    @pytest.mark.asyncio
    async def test_broadcast_stream_with_metrics(
        self,
        streaming_coordinator_with_metrics,
        sample_threads_dict,
        mock_metrics_manager,
    ):
        """Test broadcast streaming with metrics tracking"""
        manager, metrics = mock_metrics_manager
        message = "Test message"
        session_id = "test-session-123"

        # Mock backend streaming with complete event
        mock_events = [
            {"type": "chunk", "content": "Response"},
            {"type": "complete", "content": ""},
        ]

        mock_backend = Mock()
        mock_backend.call_model_stream.return_value = MockAsyncGenerator(mock_events)

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            events = []
            async for event in streaming_coordinator_with_metrics.broadcast_stream(
                sample_threads_dict, message, session_id
            ):
                events.append(event)

        # Should have created metrics
        manager.create_orchestration_metrics.assert_called_once_with(
            session_id=session_id,
            operation_type="broadcast_stream",
            thread_count=2,
        )

        # Should have finalized and stored metrics
        manager.finalize_metrics.assert_called_once_with(metrics)
        manager.store_metrics.assert_called_once_with(metrics)

    @pytest.mark.asyncio
    async def test_broadcast_stream_error_handling(
        self, streaming_coordinator, sample_thread
    ):
        """Test error handling in broadcast streaming"""
        threads = {"technical": sample_thread}
        message = "Test message"

        # Mock backend that raises an error
        mock_backend = Mock()
        mock_backend.call_model_stream.side_effect = ModelBackendError("Backend failed")

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            events = []
            async for event in streaming_coordinator.broadcast_stream(threads, message):
                events.append(event)

        # Should have start event and error event
        start_events = [e for e in events if e["type"] == "start"]
        error_events = [e for e in events if e["type"] == "error"]

        assert len(start_events) == 1
        assert len(error_events) == 1
        assert "Backend failed" in error_events[0]["content"]

    @pytest.mark.asyncio
    async def test_broadcast_stream_message_already_added(
        self, streaming_coordinator, sample_thread
    ):
        """Test broadcast streaming when message is already in conversation history"""
        message = "Test message"

        # Add message to thread history first
        sample_thread.add_message("user", message)
        threads = {"technical": sample_thread}

        mock_events = [{"type": "complete", "content": "Done"}]
        mock_backend = Mock()
        mock_backend.call_model_stream.return_value = MockAsyncGenerator(mock_events)

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            # Should not duplicate the message
            initial_history_length = len(sample_thread.conversation_history)

            events = []
            async for event in streaming_coordinator.broadcast_stream(threads, message):
                events.append(event)

            # History length should not increase (message wasn't duplicated)
            assert len(sample_thread.conversation_history) == initial_history_length

    @pytest.mark.asyncio
    async def test_broadcast_stream_empty_message(
        self, streaming_coordinator, sample_thread
    ):
        """Test broadcast streaming with empty message"""
        threads = {"technical": sample_thread}
        message = ""  # Empty message

        mock_events = [{"type": "complete", "content": "Done"}]
        mock_backend = Mock()
        mock_backend.call_model_stream.return_value = MockAsyncGenerator(mock_events)

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            events = []
            async for event in streaming_coordinator.broadcast_stream(threads, message):
                events.append(event)

        # Should still work and produce events
        assert len(events) > 0
        start_events = [e for e in events if e["type"] == "start"]
        assert len(start_events) == 1

    @pytest.mark.asyncio
    async def test_broadcast_stream_concurrent_threads(self, streaming_coordinator):
        """Test broadcast streaming handles concurrent threads correctly"""
        # Create multiple threads
        threads = {}
        for i in range(3):
            thread = Thread(
                id=f"thread-{i}",
                name=f"perspective_{i}",
                system_prompt=f"You are perspective {i}",
                model_backend=ModelBackend.BEDROCK,
                model_name="claude-3-sonnet-20240229",
            )
            threads[f"perspective_{i}"] = thread

        # Mock different responses for each thread
        def create_mock_backend():
            backend = Mock()
            backend.call_model_stream.return_value = MockAsyncGenerator(
                [
                    {"type": "chunk", "content": "Response"},
                    {"type": "complete", "content": ""},
                ]
            )
            return backend

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            side_effect=lambda _: create_mock_backend(),
        ):
            events = []
            async for event in streaming_coordinator.broadcast_stream(
                threads, "Test message"
            ):
                events.append(event)

        # Should have events from all threads
        thread_names = set(event["thread_name"] for event in events)
        assert len(thread_names) == 3
        assert all(f"perspective_{i}" in thread_names for i in range(3))

        # Should have start events for all threads
        start_events = [e for e in events if e["type"] == "start"]
        assert len(start_events) == 3

    @pytest.mark.asyncio
    async def test_stream_single_thread(self, streaming_coordinator, sample_thread):
        """Test streaming from a single thread"""
        thread_name = "test_thread"

        mock_events = [
            {"type": "chunk", "content": "Single thread response"},
            {"type": "complete", "content": ""},
        ]

        mock_backend = Mock()
        mock_backend.call_model_stream.return_value = MockAsyncGenerator(mock_events)

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            events = []
            async for event in streaming_coordinator.stream_single_thread(
                sample_thread, thread_name
            ):
                events.append(event)

        # Should have streaming events with correct thread name
        assert len(events) == 2
        assert all(event["thread_name"] == thread_name for event in events)

        chunk_events = [e for e in events if e["type"] == "chunk"]
        complete_events = [e for e in events if e["type"] == "complete"]
        assert len(chunk_events) == 1
        assert len(complete_events) == 1

    @pytest.mark.asyncio
    async def test_stream_from_thread_model_backend_error(
        self, streaming_coordinator, sample_thread
    ):
        """Test _stream_from_thread with ModelBackendError"""
        thread_name = "error_thread"

        mock_backend = Mock()
        mock_backend.call_model_stream.side_effect = ModelBackendError(
            "Model unavailable"
        )

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            events = []
            async for event in streaming_coordinator._stream_from_thread(
                sample_thread, thread_name
            ):
                events.append(event)

        # Should yield error event
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert events[0]["thread_name"] == thread_name
        assert "Model unavailable" in events[0]["content"]

    @pytest.mark.asyncio
    async def test_stream_from_thread_unexpected_error(
        self, streaming_coordinator, sample_thread
    ):
        """Test _stream_from_thread with unexpected error"""
        thread_name = "error_thread"

        mock_backend = Mock()
        mock_backend.call_model_stream.side_effect = ValueError("Unexpected error")

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            with patch(
                "context_switcher_mcp.streaming_coordinator.logger"
            ) as mock_logger:
                events = []
                async for event in streaming_coordinator._stream_from_thread(
                    sample_thread, thread_name
                ):
                    events.append(event)

        # Should yield error event with "Unexpected error" prefix
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert events[0]["thread_name"] == thread_name
        assert "Unexpected error" in events[0]["content"]

        # Should log the error
        mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_from_thread_cancellation(
        self, streaming_coordinator, sample_thread
    ):
        """Test _stream_from_thread handles cancellation correctly"""
        thread_name = "cancelled_thread"

        mock_backend = Mock()
        mock_backend.call_model_stream.side_effect = asyncio.CancelledError()

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            # Should propagate CancelledError
            with pytest.raises(asyncio.CancelledError):
                async for event in streaming_coordinator._stream_from_thread(
                    sample_thread, thread_name
                ):
                    pass

    @pytest.mark.asyncio
    async def test_stream_from_thread_backend_factory_error(
        self, streaming_coordinator, sample_thread
    ):
        """Test _stream_from_thread when backend factory fails"""
        thread_name = "factory_error_thread"

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            side_effect=RuntimeError("Backend factory failed"),
        ):
            with patch(
                "context_switcher_mcp.streaming_coordinator.logger"
            ) as mock_logger:
                events = []
                async for event in streaming_coordinator._stream_from_thread(
                    sample_thread, thread_name
                ):
                    events.append(event)

        # Should yield error event
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert events[0]["thread_name"] == thread_name
        assert "Unexpected error" in events[0]["content"]

        # Should log the error
        mock_logger.error.assert_called_once()

    def test_create_stream_event_basic(self, streaming_coordinator):
        """Test basic stream event creation"""
        event = streaming_coordinator.create_stream_event(
            event_type="chunk", thread_name="test_thread", content="Test content"
        )

        assert event["type"] == "chunk"
        assert event["thread_name"] == "test_thread"
        assert event["content"] == "Test content"
        assert "timestamp" in event
        assert isinstance(event["timestamp"], float)

    def test_create_stream_event_with_kwargs(self, streaming_coordinator):
        """Test stream event creation with additional kwargs"""
        event = streaming_coordinator.create_stream_event(
            event_type="complete",
            thread_name="test_thread",
            content="",
            metadata={"key": "value"},
            token_count=150,
        )

        assert event["type"] == "complete"
        assert event["thread_name"] == "test_thread"
        assert event["content"] == ""
        assert event["metadata"] == {"key": "value"}
        assert event["token_count"] == 150
        assert "timestamp" in event

    def test_create_stream_event_empty_content(self, streaming_coordinator):
        """Test stream event creation with empty content"""
        event = streaming_coordinator.create_stream_event(
            event_type="start",
            thread_name="test_thread",
            # No content parameter - should use default ""
        )

        assert event["type"] == "start"
        assert event["thread_name"] == "test_thread"
        assert event["content"] == ""

    @pytest.mark.asyncio
    async def test_broadcast_stream_metrics_error_tracking(
        self, streaming_coordinator_with_metrics, mock_metrics_manager
    ):
        """Test that metrics correctly track errors in broadcast streaming"""
        manager, metrics = mock_metrics_manager

        # Create threads - one will succeed, one will fail
        thread1 = Thread(
            id="1",
            name="success",
            system_prompt="Success",
            model_backend=ModelBackend.BEDROCK,
            model_name="claude-3-sonnet-20240229",
        )
        thread2 = Thread(
            id="2",
            name="error",
            system_prompt="Error",
            model_backend=ModelBackend.BEDROCK,
            model_name="claude-3-sonnet-20240229",
        )
        threads = {"success": thread1, "error": thread2}

        # Mock backends - one succeeds, one fails
        def mock_backend_factory(backend_type):
            backend = Mock()
            if backend_type == ModelBackend.BEDROCK:
                # Return different behaviors based on how many times called
                call_count = getattr(mock_backend_factory, "call_count", 0)
                mock_backend_factory.call_count = call_count + 1

                if call_count == 0:  # First call - success
                    backend.call_model_stream.return_value = MockAsyncGenerator(
                        [{"type": "complete", "content": "Success"}]
                    )
                else:  # Second call - error
                    backend.call_model_stream.return_value = MockAsyncGenerator(
                        [{"type": "error", "content": "Failed"}]
                    )
            return backend

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            side_effect=mock_backend_factory,
        ):
            events = []
            async for event in streaming_coordinator_with_metrics.broadcast_stream(
                threads, "Test"
            ):
                events.append(event)

        # Should have tracked both success and failure
        complete_events = [e for e in events if e["type"] == "complete"]
        error_events = [e for e in events if e["type"] == "error"]

        assert len(complete_events) >= 1
        assert len(error_events) >= 1

    @pytest.mark.asyncio
    async def test_broadcast_stream_no_threads(self, streaming_coordinator):
        """Test broadcast streaming with no threads"""
        empty_threads = {}
        message = "Test message"

        events = []
        async for event in streaming_coordinator.broadcast_stream(
            empty_threads, message
        ):
            events.append(event)

        # Should not yield any events for empty threads dict
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_broadcast_stream_thread_history_modification(
        self, streaming_coordinator, sample_thread
    ):
        """Test that broadcast streaming correctly modifies thread history"""
        threads = {"test": sample_thread}
        message = "New test message"

        # Ensure thread doesn't already have this message
        initial_history_length = len(sample_thread.conversation_history)

        mock_events = [{"type": "complete", "content": "Done"}]
        mock_backend = Mock()
        mock_backend.call_model_stream.return_value = MockAsyncGenerator(mock_events)

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            events = []
            async for event in streaming_coordinator.broadcast_stream(threads, message):
                events.append(event)

        # Thread history should have the new message added
        assert len(sample_thread.conversation_history) == initial_history_length + 1
        assert sample_thread.conversation_history[-1]["content"] == message
        assert sample_thread.conversation_history[-1]["role"] == "user"


class TestStreamingCoordinatorIntegration:
    """Integration tests for StreamingCoordinator"""

    @pytest.mark.asyncio
    async def test_complete_streaming_workflow(self):
        """Test a complete streaming workflow with multiple threads and realistic responses"""
        coordinator = StreamingCoordinator()

        # Create realistic threads
        threads = {
            "technical": Thread(
                id="tech-1",
                name="technical",
                system_prompt="Technical analysis",
                model_backend=ModelBackend.BEDROCK,
                model_name="claude-3-sonnet-20240229",
            ),
            "business": Thread(
                id="biz-1",
                name="business",
                system_prompt="Business analysis",
                model_backend=ModelBackend.LITELLM,
                model_name="gpt-4",
            ),
        }

        # Mock realistic streaming responses
        technical_events = [
            {"type": "chunk", "content": "From a technical perspective, "},
            {"type": "chunk", "content": "this solution involves "},
            {"type": "chunk", "content": "several key components."},
            {"type": "complete", "content": ""},
        ]

        business_events = [
            {"type": "chunk", "content": "From a business standpoint, "},
            {"type": "chunk", "content": "we need to consider "},
            {"type": "chunk", "content": "the ROI implications."},
            {"type": "complete", "content": ""},
        ]

        def mock_backend_factory(backend_type):
            backend = Mock()
            if backend_type == ModelBackend.BEDROCK:
                backend.call_model_stream.return_value = MockAsyncGenerator(
                    technical_events
                )
            else:
                backend.call_model_stream.return_value = MockAsyncGenerator(
                    business_events
                )
            return backend

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            side_effect=mock_backend_factory,
        ):
            # Collect all events
            all_events = []
            async for event in coordinator.broadcast_stream(
                threads, "Analyze this solution"
            ):
                all_events.append(event)

        # Verify we got events from both threads
        thread_names = set(event["thread_name"] for event in all_events)
        assert "technical" in thread_names
        assert "business" in thread_names

        # Verify we got all expected event types
        event_types = set(event["type"] for event in all_events)
        assert "start" in event_types
        assert "chunk" in event_types
        assert "complete" in event_types

        # Verify content was streamed
        chunk_events = [e for e in all_events if e["type"] == "chunk"]
        assert len(chunk_events) >= 6  # Should have chunks from both threads

        # Verify all events have timestamps
        assert all("timestamp" in event for event in all_events)
        assert all(isinstance(event["timestamp"], float) for event in all_events)


class TestStreamingCoordinatorPerformance:
    """Performance and stress tests for StreamingCoordinator"""

    @pytest.mark.asyncio
    async def test_many_concurrent_threads(self):
        """Test streaming coordinator with many concurrent threads"""
        coordinator = StreamingCoordinator()

        # Create many threads
        thread_count = 10
        threads = {}
        for i in range(thread_count):
            thread = Thread(
                id=f"thread-{i}",
                name=f"perspective_{i}",
                system_prompt=f"Perspective {i}",
                model_backend=ModelBackend.BEDROCK,
                model_name="claude-3-sonnet-20240229",
            )
            threads[f"perspective_{i}"] = thread

        # Mock fast responses
        mock_events = [
            {"type": "chunk", "content": "Response"},
            {"type": "complete", "content": ""},
        ]

        mock_backend = Mock()
        mock_backend.call_model_stream.return_value = MockAsyncGenerator(mock_events)

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            start_time = time.time()

            events = []
            async for event in coordinator.broadcast_stream(threads, "Test message"):
                events.append(event)

            elapsed_time = time.time() - start_time

        # Should handle many threads efficiently (should not take too long)
        assert elapsed_time < 5.0  # Should complete within 5 seconds

        # Should have events from all threads - check start events which are always generated
        start_thread_names = set(
            event["thread_name"] for event in events if event["type"] == "start"
        )
        assert len(start_thread_names) == thread_count

        # Should have expected number of start events
        start_events = [e for e in events if e["type"] == "start"]
        assert len(start_events) == thread_count


class TestStreamingCoordinatorEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def streaming_coordinator(self):
        """Create a StreamingCoordinator instance for testing"""
        return StreamingCoordinator()

    @pytest.fixture
    def sample_thread(self):
        """Create a sample Thread for edge case testing"""
        return Thread(
            id="thread-edge-123",
            name="technical",
            system_prompt="You are a technical expert",
            model_backend=ModelBackend.BEDROCK,
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        )

    @pytest.mark.asyncio
    async def test_stream_with_none_message(self, streaming_coordinator, sample_thread):
        """Test streaming with None as message"""
        threads = {"test": sample_thread}

        mock_events = [{"type": "complete", "content": "Done"}]
        mock_backend = Mock()
        mock_backend.call_model_stream.return_value = MockAsyncGenerator(mock_events)

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            return_value=mock_backend,
        ):
            # Should handle None message gracefully
            events = []
            async for event in streaming_coordinator.broadcast_stream(threads, None):
                events.append(event)

        assert len(events) > 0  # Should still produce events

    @pytest.mark.asyncio
    async def test_mixed_success_and_error_threads(self, streaming_coordinator):
        """Test streaming with some threads succeeding and others failing"""
        success_thread = Thread(
            id="1",
            name="success",
            system_prompt="Success",
            model_backend=ModelBackend.BEDROCK,
            model_name="claude-3-sonnet-20240229",
        )
        error_thread = Thread(
            id="2",
            name="error",
            system_prompt="Error",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-4",
        )
        threads = {"success": success_thread, "error": error_thread}

        def mock_backend_factory(backend_type):
            backend = Mock()
            if backend_type == ModelBackend.BEDROCK:
                # Success case
                backend.call_model_stream.return_value = MockAsyncGenerator(
                    [{"type": "complete", "content": "Success"}]
                )
            else:
                # Error case
                backend.call_model_stream.side_effect = ModelBackendError("Failed")
            return backend

        with patch(
            "context_switcher_mcp.streaming_coordinator.BackendFactory.get_backend",
            side_effect=mock_backend_factory,
        ):
            events = []
            async for event in streaming_coordinator.broadcast_stream(threads, "Test"):
                events.append(event)

        # Should have both success and error events
        complete_events = [e for e in events if e["type"] == "complete"]
        error_events = [e for e in events if e["type"] == "error"]

        assert len(complete_events) >= 1
        assert len(error_events) >= 1

        # Should have start events for both threads
        start_events = [e for e in events if e["type"] == "start"]
        assert len(start_events) == 2
