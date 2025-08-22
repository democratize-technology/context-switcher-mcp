"""Tests for Chain of Thought reasoning integration"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.context_switcher_mcp.models import ModelBackend, Thread
from src.context_switcher_mcp.reasoning_orchestrator import (
    CoTProcessingError,
    CoTTimeoutError,
    PerspectiveReasoningOrchestrator,
)


@pytest.fixture
def reasoning_orchestrator():
    """Create a reasoning orchestrator instance"""
    return PerspectiveReasoningOrchestrator(cot_timeout=5.0)


@pytest.fixture
def mock_thread():
    """Create a mock thread for testing"""
    thread = Thread(
        id="test-thread-1",
        name="test-perspective",
        system_prompt="You are a test perspective.",
        model_backend=ModelBackend.BEDROCK,
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
    )
    thread.add_message("user", "Test message")
    return thread


@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client"""
    client = Mock()
    return client


class TestPerspectiveReasoningOrchestrator:
    """Test cases for PerspectiveReasoningOrchestrator"""

    def test_initialization(self, reasoning_orchestrator):
        """Test orchestrator initialization"""
        assert reasoning_orchestrator.cot_timeout == 5.0
        assert isinstance(reasoning_orchestrator._processors, dict)
        assert len(reasoning_orchestrator._processors) == 0

    def test_is_available_without_cot(self, reasoning_orchestrator):
        """Test availability check when CoT is not installed"""
        # This will depend on whether chain-of-thought-tool is installed
        # For now, we just check the property exists
        assert hasattr(reasoning_orchestrator, "is_available")
        assert isinstance(reasoning_orchestrator.is_available, bool)

    @patch("src.context_switcher_mcp.reasoning_orchestrator.COT_AVAILABLE", False)
    def test_get_cot_tools_without_cot(self, reasoning_orchestrator):
        """Test getting tools when CoT is not available"""
        tools = reasoning_orchestrator.get_cot_tools()
        assert tools == []

    @patch("src.context_switcher_mcp.reasoning_orchestrator.COT_AVAILABLE", True)
    @patch(
        "src.context_switcher_mcp.reasoning_orchestrator.TOOL_SPECS",
        [{"name": "test_tool"}],
    )
    def test_get_cot_tools_with_cot(self, reasoning_orchestrator):
        """Test getting tools when CoT is available"""
        tools = reasoning_orchestrator.get_cot_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"

    @pytest.mark.asyncio
    @patch("src.context_switcher_mcp.reasoning_orchestrator.COT_AVAILABLE", False)
    async def test_analyze_without_cot(
        self, reasoning_orchestrator, mock_thread, mock_bedrock_client
    ):
        """Test analysis when CoT is not available"""
        response, summary = await reasoning_orchestrator.analyze_with_reasoning(
            mock_thread,
            "Test prompt",
            mock_bedrock_client,
            "test-session",
            "Test topic",
        )
        assert response == "Chain of Thought not available"
        assert summary == {}

    @pytest.mark.asyncio
    @patch("src.context_switcher_mcp.reasoning_orchestrator.COT_AVAILABLE", True)
    @patch(
        "src.context_switcher_mcp.reasoning_orchestrator.AsyncChainOfThoughtProcessor"
    )
    async def test_analyze_with_cot_success(
        self,
        mock_processor_class,
        reasoning_orchestrator,
        mock_thread,
        mock_bedrock_client,
    ):
        """Test successful CoT analysis"""
        # Mock the processor
        mock_processor = AsyncMock()
        mock_processor_class.return_value = mock_processor

        # Mock the process_tool_loop response
        mock_processor.process_tool_loop.return_value = {
            "stopReason": "end_turn",
            "output": {"message": {"content": [{"text": "Test response from CoT"}]}},
        }

        # Mock the get_reasoning_summary response
        mock_processor.get_reasoning_summary.return_value = {
            "status": "success",
            "overall_confidence": 0.85,
            "total_steps": 5,
        }

        response, summary = await reasoning_orchestrator.analyze_with_reasoning(
            mock_thread,
            "Test prompt",
            mock_bedrock_client,
            "test-session",
            "Test topic",
        )

        assert response == "Test response from CoT"
        assert summary["overall_confidence"] == 0.85
        assert summary["total_steps"] == 5

        # Verify processor was called correctly
        mock_processor.process_tool_loop.assert_called_once()
        mock_processor.get_reasoning_summary.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.context_switcher_mcp.reasoning_orchestrator.COT_AVAILABLE", True)
    @patch(
        "src.context_switcher_mcp.reasoning_orchestrator.AsyncChainOfThoughtProcessor"
    )
    async def test_analyze_with_cot_timeout(
        self,
        mock_processor_class,
        reasoning_orchestrator,
        mock_thread,
        mock_bedrock_client,
    ):
        """Test CoT analysis timeout"""
        # Mock the processor
        mock_processor = AsyncMock()
        mock_processor_class.return_value = mock_processor

        # Make process_tool_loop raise timeout
        mock_processor.process_tool_loop.side_effect = asyncio.TimeoutError()

        with pytest.raises(CoTTimeoutError) as exc_info:
            await reasoning_orchestrator.analyze_with_reasoning(
                mock_thread, "Test prompt", mock_bedrock_client, "test-session"
            )

        assert exc_info.value.timeout == 5.0

    @pytest.mark.asyncio
    @patch("src.context_switcher_mcp.reasoning_orchestrator.COT_AVAILABLE", True)
    @patch(
        "src.context_switcher_mcp.reasoning_orchestrator.AsyncChainOfThoughtProcessor"
    )
    async def test_analyze_with_cot_error(
        self,
        mock_processor_class,
        reasoning_orchestrator,
        mock_thread,
        mock_bedrock_client,
    ):
        """Test CoT analysis error"""
        # Mock the processor
        mock_processor = AsyncMock()
        mock_processor_class.return_value = mock_processor

        # Make process_tool_loop raise an error
        mock_processor.process_tool_loop.side_effect = Exception("Test error")

        with pytest.raises(CoTProcessingError) as exc_info:
            await reasoning_orchestrator.analyze_with_reasoning(
                mock_thread, "Test prompt", mock_bedrock_client, "test-session"
            )

        assert "Failed to process reasoning" in str(exc_info.value)
        assert exc_info.value.stage == "analysis"

    def test_clear_processor(self, reasoning_orchestrator):
        """Test clearing a specific processor"""
        # Add a mock processor
        processor_id = "test-session-test-thread"
        mock_processor = Mock()
        mock_processor.clear_reasoning = Mock()
        reasoning_orchestrator._processors[processor_id] = mock_processor

        # Clear it
        reasoning_orchestrator.clear_processor("test-session", "test-thread")

        # Verify it was cleared
        assert processor_id not in reasoning_orchestrator._processors
        mock_processor.clear_reasoning.assert_called_once()

    def test_clear_session(self, reasoning_orchestrator):
        """Test clearing all processors for a session"""
        # Add multiple mock processors
        processors = {
            "test-session-thread1": Mock(),
            "test-session-thread2": Mock(),
            "other-session-thread1": Mock(),
        }

        for pid, proc in processors.items():
            proc.clear_reasoning = Mock()
            reasoning_orchestrator._processors[pid] = proc

        # Clear the test session
        reasoning_orchestrator.clear_session("test-session")

        # Verify only test-session processors were cleared
        assert "test-session-thread1" not in reasoning_orchestrator._processors
        assert "test-session-thread2" not in reasoning_orchestrator._processors
        assert "other-session-thread1" in reasoning_orchestrator._processors

        # Verify clear_reasoning was called on cleared processors
        processors["test-session-thread1"].clear_reasoning.assert_called_once()
        processors["test-session-thread2"].clear_reasoning.assert_called_once()
        processors["other-session-thread1"].clear_reasoning.assert_not_called()

    def test_get_active_processors(self, reasoning_orchestrator):
        """Test getting list of active processors"""
        # Add some processors
        reasoning_orchestrator._processors = {
            "session1-thread1": Mock(),
            "session2-thread1": Mock(),
        }

        active = reasoning_orchestrator.get_active_processors()

        assert len(active) == 2
        assert "session1-thread1" in active
        assert "session2-thread1" in active
