"""Tests for PerspectiveOrchestrator component"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from context_switcher_mcp.perspective_orchestrator import (
    PerspectiveOrchestrator,
    PerspectiveMetrics,
)
from context_switcher_mcp.thread_manager import ThreadManager
from context_switcher_mcp.response_formatter import ResponseFormatter
from context_switcher_mcp.models import Thread, ModelBackend
from context_switcher_mcp.exceptions import OrchestrationError


@pytest.fixture
def mock_thread_manager():
    """Create mock ThreadManager for testing"""
    manager = AsyncMock(spec=ThreadManager)
    return manager


@pytest.fixture
def mock_response_formatter():
    """Create mock ResponseFormatter for testing"""
    formatter = AsyncMock(spec=ResponseFormatter)
    return formatter


@pytest.fixture
def perspective_orchestrator(mock_thread_manager, mock_response_formatter):
    """Create PerspectiveOrchestrator instance for testing"""
    return PerspectiveOrchestrator(
        thread_manager=mock_thread_manager,
        response_formatter=mock_response_formatter,
        enable_cot=False,  # Disable CoT for unit tests
    )


@pytest.fixture
def mock_threads():
    """Create dictionary of mock perspective threads"""
    threads = {}
    for name in ["technical", "business", "user"]:
        thread = MagicMock()
        thread.name = name
        thread.model_backend = MagicMock()
        thread.model_backend.value = "bedrock"  # Mock the value attribute
        thread.system_prompt = f"You are a {name} perspective."
        thread.model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
        thread.conversation_history = []
        thread.add_message = MagicMock()
        threads[name] = thread
    return threads


class TestPerspectiveOrchestrator:
    """Test PerspectiveOrchestrator functionality"""

    def test_initialization_with_defaults(self):
        """Test PerspectiveOrchestrator initialization with default components"""
        orchestrator = PerspectiveOrchestrator()

        assert orchestrator.thread_manager is not None
        assert orchestrator.response_formatter is not None
        assert isinstance(orchestrator.thread_manager, ThreadManager)
        assert isinstance(orchestrator.response_formatter, ResponseFormatter)

    def test_initialization_with_custom_components(
        self, mock_thread_manager, mock_response_formatter
    ):
        """Test PerspectiveOrchestrator initialization with custom components"""
        orchestrator = PerspectiveOrchestrator(
            thread_manager=mock_thread_manager,
            response_formatter=mock_response_formatter,
        )

        assert orchestrator.thread_manager is mock_thread_manager
        assert orchestrator.response_formatter is mock_response_formatter

    @pytest.mark.asyncio
    async def test_broadcast_to_perspectives_success(
        self, perspective_orchestrator, mock_threads, mock_thread_manager
    ):
        """Test successful broadcast to perspectives"""
        # Mock thread manager response
        mock_thread_manager.broadcast_message.return_value = {
            "technical": "Technical analysis response",
            "business": "Business analysis response",
            "user": "User experience response",
        }

        result = await perspective_orchestrator.broadcast_to_perspectives(
            mock_threads, "Analyze this feature", "test_session"
        )

        assert len(result) == 3
        assert result["technical"] == "Technical analysis response"
        assert result["business"] == "Business analysis response"
        assert result["user"] == "User experience response"

        # Verify thread manager was called correctly
        mock_thread_manager.broadcast_message.assert_called_once_with(
            mock_threads, "Analyze this feature", "test_session"
        )

    @pytest.mark.asyncio
    async def test_broadcast_to_perspectives_with_abstentions(
        self, perspective_orchestrator, mock_threads, mock_thread_manager
    ):
        """Test broadcast handling perspective abstentions"""
        # Mock responses including abstentions
        mock_thread_manager.broadcast_message.return_value = {
            "technical": "Technical analysis response",
            "business": "[NO_RESPONSE] - Not applicable to business concerns",
            "user": "User experience response",
        }

        result = await perspective_orchestrator.broadcast_to_perspectives(
            mock_threads, "Technical implementation details", "test_session"
        )

        assert len(result) == 3
        assert result["technical"] == "Technical analysis response"
        assert "[NO_RESPONSE]" in result["business"]
        assert result["user"] == "User experience response"

    @pytest.mark.asyncio
    async def test_broadcast_to_perspectives_with_errors(
        self, perspective_orchestrator, mock_threads, mock_thread_manager
    ):
        """Test broadcast handling perspective errors"""
        # Mock responses including errors
        mock_thread_manager.broadcast_message.return_value = {
            "technical": "Technical analysis response",
            "business": "ERROR: Model connection failed",
            "user": "User experience response",
        }

        result = await perspective_orchestrator.broadcast_to_perspectives(
            mock_threads, "Analyze this feature", "test_session"
        )

        assert len(result) == 3
        assert result["technical"] == "Technical analysis response"
        assert result["business"].startswith("ERROR:")
        assert result["user"] == "User experience response"

    @pytest.mark.asyncio
    async def test_broadcast_to_perspectives_thread_manager_error(
        self, perspective_orchestrator, mock_threads, mock_thread_manager
    ):
        """Test broadcast handling thread manager errors"""
        # Mock thread manager throwing an exception
        mock_thread_manager.broadcast_message.side_effect = Exception(
            "Thread manager failed"
        )

        with pytest.raises(OrchestrationError) as exc_info:
            await perspective_orchestrator.broadcast_to_perspectives(
                mock_threads, "Analyze this feature", "test_session"
            )

        assert "Perspective broadcast failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_broadcast_to_perspectives_stream(
        self, perspective_orchestrator, mock_threads, mock_thread_manager
    ):
        """Test streaming broadcast to perspectives"""

        # Mock streaming events from thread manager
        async def mock_stream():
            yield {
                "type": "start",
                "thread_name": "technical",
                "content": "",
                "timestamp": 1234567890.0,
            }
            yield {
                "type": "chunk",
                "thread_name": "technical",
                "content": "Technical analysis...",
                "timestamp": 1234567891.0,
            }
            yield {
                "type": "complete",
                "thread_name": "technical",
                "content": "",
                "timestamp": 1234567892.0,
            }

        mock_thread_manager.broadcast_message_stream.return_value = mock_stream()

        events = []
        async for event in perspective_orchestrator.broadcast_to_perspectives_stream(
            mock_threads, "Analyze this feature", "test_session"
        ):
            events.append(event)

        assert len(events) == 3

        # Verify thread_name was transformed to perspective_name
        for event in events:
            assert "perspective_name" in event
            assert "thread_name" not in event
            assert event["perspective_name"] == "technical"

    @pytest.mark.asyncio
    async def test_synthesize_perspective_responses_success(
        self, perspective_orchestrator, mock_response_formatter
    ):
        """Test successful synthesis of perspective responses"""
        responses = {
            "technical": "Technical concerns about scalability",
            "business": "Business impact looks positive",
            "user": "Users will find this intuitive",
        }

        mock_response_formatter.synthesize_responses.return_value = (
            "Synthesized analysis combining all perspectives"
        )

        result = await perspective_orchestrator.synthesize_perspective_responses(
            responses, "test_session"
        )

        assert result == "Synthesized analysis combining all perspectives"

        # Verify formatter was called with valid responses only
        mock_response_formatter.synthesize_responses.assert_called_once_with(
            responses, "test_session"
        )

    @pytest.mark.asyncio
    async def test_synthesize_perspective_responses_filter_errors(
        self, perspective_orchestrator, mock_response_formatter
    ):
        """Test synthesis filtering out errors and abstentions"""
        responses = {
            "technical": "Technical analysis response",
            "business": "ERROR: Model connection failed",
            "user": "[NO_RESPONSE] - Not applicable",
            "risk": "Risk assessment complete",
        }

        mock_response_formatter.synthesize_responses.return_value = (
            "Synthesized from valid responses"
        )

        result = await perspective_orchestrator.synthesize_perspective_responses(
            responses, "test_session"
        )

        assert result == "Synthesized from valid responses"

        # Verify only valid responses were passed to formatter
        call_args = mock_response_formatter.synthesize_responses.call_args[0]
        valid_responses = call_args[0]
        assert len(valid_responses) == 2
        assert "technical" in valid_responses
        assert "risk" in valid_responses
        assert "business" not in valid_responses  # Error filtered out
        assert "user" not in valid_responses  # Abstention filtered out

    @pytest.mark.asyncio
    async def test_synthesize_perspective_responses_no_valid_responses(
        self, perspective_orchestrator, mock_response_formatter
    ):
        """Test synthesis when no valid responses available"""
        responses = {
            "technical": "ERROR: Model timeout",
            "business": "[NO_RESPONSE] - Cannot analyze",
        }

        mock_response_formatter.format_error_response.return_value = (
            "AORP_ERROR: No valid responses"
        )

        result = await perspective_orchestrator.synthesize_perspective_responses(
            responses, "test_session"
        )

        assert result == "AORP_ERROR: No valid responses"

        # Verify error formatter was called
        mock_response_formatter.format_error_response.assert_called_once_with(
            "No valid perspective responses to synthesize",
            "synthesis_error",
            {"session_id": "test_session"},
        )

    @pytest.mark.asyncio
    async def test_synthesize_perspective_responses_formatter_error(
        self, perspective_orchestrator, mock_response_formatter
    ):
        """Test synthesis handling formatter errors"""
        responses = {"technical": "Technical analysis response"}

        # Mock formatter throwing an exception
        mock_response_formatter.synthesize_responses.side_effect = Exception(
            "Synthesis failed"
        )
        mock_response_formatter.format_error_response.return_value = (
            "AORP_ERROR: Synthesis failed"
        )

        result = await perspective_orchestrator.synthesize_perspective_responses(
            responses, "test_session"
        )

        assert result == "AORP_ERROR: Synthesis failed"

        # Verify error was formatted
        mock_response_formatter.format_error_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_perspective_metrics_empty(self, perspective_orchestrator):
        """Test getting metrics when history is empty"""
        result = await perspective_orchestrator.get_perspective_metrics()
        assert result["message"] == "No perspective metrics available"

    @pytest.mark.asyncio
    async def test_get_perspective_metrics_with_data(self, perspective_orchestrator):
        """Test getting metrics with data in history"""
        # Add mock metrics to history
        metrics = PerspectiveMetrics(
            session_id="test",
            operation_type="broadcast",
            start_time=1234567890.0,
            end_time=1234567895.0,
            total_perspectives=3,
            successful_perspectives=2,
            abstained_perspectives=1,
        )
        perspective_orchestrator.metrics_history.append(metrics)

        result = await perspective_orchestrator.get_perspective_metrics()

        assert "perspective_summary" in result
        assert result["perspective_summary"]["total_operations"] == 1
        assert result["perspective_summary"]["avg_execution_time_seconds"] == 5.0
        assert result["perspective_summary"]["overall_success_rate_percent"] == 66.7

        assert "recent_operations" in result
        assert len(result["recent_operations"]) == 1

        operation = result["recent_operations"][0]
        assert operation["session_id"] == "test"
        assert operation["operation"] == "broadcast"
        assert operation["perspectives"]["total"] == 3
        assert operation["perspectives"]["successful"] == 2
        assert operation["perspectives"]["abstained"] == 1

    def test_perspective_metrics_properties(self):
        """Test PerspectiveMetrics property calculations"""
        metrics = PerspectiveMetrics(
            session_id="test",
            operation_type="broadcast",
            start_time=1234567890.0,
            end_time=1234567895.0,
            total_perspectives=10,
            successful_perspectives=7,
        )

        assert metrics.execution_time == 5.0
        assert metrics.success_rate == 70.0

        # Test with zero perspectives
        empty_metrics = PerspectiveMetrics(
            session_id="test",
            operation_type="broadcast",
            start_time=1234567890.0,
            total_perspectives=0,
        )
        assert empty_metrics.success_rate == 0.0

    def test_metrics_storage_limit(self, perspective_orchestrator):
        """Test that metrics storage respects size limit"""
        # Add more than 100 metrics to test limit
        for i in range(150):
            metrics = PerspectiveMetrics(
                session_id=f"session_{i}",
                operation_type="broadcast",
                start_time=1234567890.0 + i,
                end_time=1234567895.0 + i,
                total_perspectives=1,
                successful_perspectives=1,
            )
            perspective_orchestrator._store_metrics(metrics)

        # Should only keep last 100
        assert len(perspective_orchestrator.metrics_history) == 100

        # Should have the most recent metrics
        last_metric = perspective_orchestrator.metrics_history[-1]
        assert last_metric.session_id == "session_149"


class TestPerspectiveOrchestratorIntegration:
    """Integration tests for PerspectiveOrchestrator"""

    @pytest.mark.asyncio
    async def test_full_perspective_workflow(self):
        """Test complete perspective workflow with real components"""
        # Create orchestrator with real components but disable CoT for testing
        orchestrator = PerspectiveOrchestrator(enable_cot=False)

        # Create test threads
        threads = {}
        for name in ["technical", "business"]:
            thread = MagicMock(spec=Thread)
            thread.name = name
            thread.model_backend = ModelBackend.BEDROCK
            thread.add_message = MagicMock()
            threads[name] = thread

        # Mock the thread manager's broadcast method
        with patch.object(
            orchestrator.thread_manager, "broadcast_message"
        ) as mock_broadcast:
            mock_broadcast.return_value = {
                "technical": "Technical perspective response",
                "business": "Business perspective response",
            }

            result = await orchestrator.broadcast_to_perspectives(
                threads, "Analyze this feature", "integration_test"
            )

            assert len(result) == 2
            assert result["technical"] == "Technical perspective response"
            assert result["business"] == "Business perspective response"

            # Verify the underlying thread manager was called
            mock_broadcast.assert_called_once_with(
                threads, "Analyze this feature", "integration_test"
            )
