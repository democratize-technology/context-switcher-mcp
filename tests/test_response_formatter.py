"""Tests for ResponseFormatter component"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from context_switcher_mcp.exceptions import ModelBackendError  # noqa: E402
from context_switcher_mcp.models import ModelBackend  # noqa: E402
from context_switcher_mcp.response_formatter import ResponseFormatter  # noqa: E402


@pytest.fixture
def response_formatter():
    """Create ResponseFormatter instance for testing"""
    return ResponseFormatter()


class TestResponseFormatter:
    """Test ResponseFormatter functionality"""

    def test_initialization(self, response_formatter):
        """Test ResponseFormatter initialization"""
        assert response_formatter is not None

    def test_format_error_response(self, response_formatter):
        """Test error response formatting"""
        result = response_formatter.format_error_response(
            error_message="Test error occurred",
            error_type="test_error",
            context={"key": "value"},
            recoverable=True,
        )

        assert result.startswith("AORP_ERROR: ")

        # Parse the JSON part
        json_part = result[len("AORP_ERROR: ") :]
        error_data = json.loads(json_part)

        # Check AORP structure
        assert error_data["immediate"]["status"] == "error"
        assert "Test error occurred" in error_data["immediate"]["key_insight"]
        assert error_data["quality"]["indicators"]["error_type"] == "test_error"
        assert error_data["quality"]["indicators"]["recoverable"] is True

    def test_format_error_response_minimal(self, response_formatter):
        """Test error response formatting with minimal parameters"""
        result = response_formatter.format_error_response(
            error_message="Simple error", error_type="simple_error"
        )

        assert result.startswith("AORP_ERROR: ")

        json_part = result[len("AORP_ERROR: ") :]
        error_data = json.loads(json_part)

        assert error_data["immediate"]["status"] == "error"
        assert "Simple error" in error_data["immediate"]["key_insight"]
        assert error_data["quality"]["indicators"]["error_type"] == "simple_error"
        assert (
            error_data["quality"]["indicators"]["recoverable"] is True
        )  # Default value

    def test_format_abstention_response(self, response_formatter):
        """Test abstention response formatting"""
        result = response_formatter.format_abstention_response(
            perspective_name="technical", reason="Not applicable to this domain"
        )

        assert result == "[NO_RESPONSE] - Not applicable to this domain"

    def test_format_abstention_response_no_reason(self, response_formatter):
        """Test abstention response formatting without reason"""
        result = response_formatter.format_abstention_response(
            perspective_name="business"
        )

        assert result == "[NO_RESPONSE]"

    def test_format_success_response(self, response_formatter):
        """Test success response formatting"""
        result = response_formatter.format_success_response(
            content="This is a successful response",
            metadata={"perspective": "technical"},
        )

        # For now, should just return the content
        assert result == "This is a successful response"

    def test_format_success_response_no_metadata(self, response_formatter):
        """Test success response formatting without metadata"""
        result = response_formatter.format_success_response(
            content="Simple success response"
        )

        assert result == "Simple success response"

    def test_is_error_response(self, response_formatter):
        """Test error response detection"""
        assert (
            response_formatter.is_error_response("ERROR: Something went wrong") is True
        )
        assert (
            response_formatter.is_error_response('AORP_ERROR: {"error": "test"}')
            is True
        )
        assert (
            response_formatter.is_error_response("This is a normal response") is False
        )

    def test_is_abstention_response(self, response_formatter):
        """Test abstention response detection"""
        assert response_formatter.is_abstention_response("[NO_RESPONSE]") is True
        assert (
            response_formatter.is_abstention_response("[NO_RESPONSE] - Reason") is True
        )
        assert (
            response_formatter.is_abstention_response("This is a normal response")
            is False
        )

    def test_extract_error_info_aorp_error(self, response_formatter):
        """Test extracting error info from AORP error response"""
        error_response = 'AORP_ERROR: {"error_message": "Test error", "error_type": "test", "recoverable": false}'

        result = response_formatter.extract_error_info(error_response)

        assert result["error_message"] == "Test error"
        assert result["error_type"] == "test"
        assert result["recoverable"] is False

    def test_extract_error_info_simple_error(self, response_formatter):
        """Test extracting error info from simple error response"""
        error_response = "ERROR: Connection timeout"

        result = response_formatter.extract_error_info(error_response)

        assert result["error_message"] == "Connection timeout"
        assert result["error_type"] == "generic_error"
        assert result["recoverable"] is True

    def test_extract_error_info_unknown_format(self, response_formatter):
        """Test extracting error info from unknown error format"""
        error_response = "Some unknown error format"

        result = response_formatter.extract_error_info(error_response)

        assert result["error_message"] == "Some unknown error format"
        assert result["error_type"] == "unknown_error"
        assert result["recoverable"] is True

    def test_extract_error_info_invalid_json(self, response_formatter):
        """Test extracting error info from invalid JSON"""
        error_response = "AORP_ERROR: {invalid json}"

        result = response_formatter.extract_error_info(error_response)

        assert result["error_message"] == "AORP_ERROR: {invalid json}"
        assert result["error_type"] == "parse_error"
        assert result["recoverable"] is True

    def test_format_perspective_summary(self, response_formatter):
        """Test formatting perspective summary"""
        responses = {
            "technical": "Technical analysis complete",
            "business": "ERROR: Model failed",
            "user": "[NO_RESPONSE] - Not applicable",
            "risk": "Risk assessment done",
        }

        summary = response_formatter.format_perspective_summary(responses)

        assert summary["total_perspectives"] == 4
        assert summary["successful_responses"] == 2  # technical, risk
        assert summary["error_responses"] == 1  # business
        assert summary["abstained_responses"] == 1  # user
        assert summary["success_rate"] == 50.0

        assert "technical" in summary["perspective_names"]
        assert "business" in summary["perspective_names"]
        assert "user" in summary["perspective_names"]
        assert "risk" in summary["perspective_names"]

        assert summary["response_lengths"]["technical"] == len(
            "Technical analysis complete"
        )

    def test_format_streaming_event(self, response_formatter):
        """Test formatting streaming events"""
        event = response_formatter.format_streaming_event(
            event_type="chunk",
            content="Partial response content",
            perspective_name="technical",
            timestamp=1234567890.0,
            metadata={"chunk_number": 1},
        )

        assert event["type"] == "chunk"
        assert event["content"] == "Partial response content"
        assert event["perspective_name"] == "technical"
        assert event["timestamp"] == 1234567890.0
        assert event["metadata"]["chunk_number"] == 1

    def test_format_streaming_event_minimal(self, response_formatter):
        """Test formatting streaming events with minimal parameters"""
        with patch("time.time", return_value=1234567890.0):
            event = response_formatter.format_streaming_event(
                event_type="start", content=""
            )

        assert event["type"] == "start"
        assert event["content"] == ""
        assert event["timestamp"] == 1234567890.0
        assert "perspective_name" not in event
        assert "metadata" not in event

    @pytest.mark.asyncio
    async def test_synthesize_responses_success(self, response_formatter):
        """Test successful response synthesis"""
        responses = {
            "technical": "Technical analysis shows scalability concerns",
            "business": "Business impact is positive overall",
            "user": "User experience will be improved",
        }

        # Mock the backend factory
        with patch(
            "context_switcher_mcp.response_formatter.BackendFactory.get_backend"
        ) as mock_get_backend:
            mock_backend = AsyncMock()
            mock_backend.call_model.return_value = (
                "Synthesized analysis combining all perspectives"
            )
            mock_get_backend.return_value = mock_backend

            result = await response_formatter.synthesize_responses(
                responses, "test_session", ModelBackend.BEDROCK
            )

            assert result == "Synthesized analysis combining all perspectives"

            # Verify backend was called
            mock_backend.call_model.assert_called_once()

            # Verify the synthesis prompt was created properly
            call_args = mock_backend.call_model.call_args[0]
            thread = call_args[0]
            assert thread.name == "synthesis"
            assert (
                "synthesize insights"
                in thread.conversation_history[0]["content"].lower()
            )

    @pytest.mark.asyncio
    async def test_synthesize_responses_empty_input(self, response_formatter):
        """Test synthesis with empty responses"""
        result = await response_formatter.synthesize_responses({}, "test_session")

        assert result.startswith("AORP_ERROR:")

        # Verify it's a proper error response
        error_info = response_formatter.extract_error_info(result)
        assert "synthesis_input_error" in error_info["error_type"]

    @pytest.mark.asyncio
    async def test_synthesize_responses_no_valid_responses(self, response_formatter):
        """Test synthesis with no valid responses"""
        responses = {
            "technical": "ERROR: Model failed",
            "business": "[NO_RESPONSE] - Cannot analyze",
        }

        result = await response_formatter.synthesize_responses(
            responses, "test_session"
        )

        assert result.startswith("AORP_ERROR:")

        error_info = response_formatter.extract_error_info(result)
        assert "synthesis_no_valid_input" in error_info["error_type"]

    @pytest.mark.asyncio
    async def test_synthesize_responses_model_error(self, response_formatter):
        """Test synthesis handling model backend errors"""
        responses = {"technical": "Technical analysis"}

        with patch(
            "context_switcher_mcp.response_formatter.BackendFactory.get_backend"
        ) as mock_get_backend:
            mock_backend = AsyncMock()
            mock_backend.call_model.side_effect = ModelBackendError(
                "Model connection failed"
            )
            mock_get_backend.return_value = mock_backend

            result = await response_formatter.synthesize_responses(
                responses, "test_session"
            )

            assert result.startswith("AORP_ERROR:")

            error_info = response_formatter.extract_error_info(result)
            assert "synthesis_model_error" in error_info["error_type"]

    @pytest.mark.asyncio
    async def test_synthesize_responses_unexpected_error(self, response_formatter):
        """Test synthesis handling unexpected errors"""
        responses = {"technical": "Technical analysis"}

        with patch(
            "context_switcher_mcp.response_formatter.BackendFactory.get_backend"
        ) as mock_get_backend:
            mock_get_backend.side_effect = Exception("Unexpected error")

            result = await response_formatter.synthesize_responses(
                responses, "test_session"
            )

            assert result.startswith("AORP_ERROR:")

            error_info = response_formatter.extract_error_info(result)
            assert "synthesis_unexpected_error" in error_info["error_type"]

    def test_create_synthesis_prompt(self, response_formatter):
        """Test synthesis prompt creation"""
        responses = {
            "technical": "Technical concerns about performance",
            "business": "Business value proposition is strong",
        }

        prompt = response_formatter._create_synthesis_prompt(responses)

        assert "synthesize insights" in prompt.lower()
        assert "TECHNICAL PERSPECTIVE" in prompt
        assert "BUSINESS PERSPECTIVE" in prompt
        assert "Technical concerns about performance" in prompt
        assert "Business value proposition is strong" in prompt
        assert "SYNTHESIS REQUEST" in prompt

    def test_create_synthesis_prompt_empty(self, response_formatter):
        """Test synthesis prompt creation with empty responses"""
        prompt = response_formatter._create_synthesis_prompt({})

        assert "synthesize insights" in prompt.lower()
        assert "SYNTHESIS REQUEST" in prompt


class TestResponseFormatterIntegration:
    """Integration tests for ResponseFormatter"""

    @pytest.mark.asyncio
    async def test_full_synthesis_workflow(self):
        """Test complete synthesis workflow"""
        formatter = ResponseFormatter()

        responses = {
            "technical": "The system architecture needs to handle 1000 RPS",
            "business": "ROI is projected at 150% within 2 years",
            "user": "Interface should be intuitive for non-technical users",
        }

        # Mock the synthesis backend
        with patch(
            "context_switcher_mcp.response_formatter.BackendFactory.get_backend"
        ) as mock_get_backend:
            mock_backend = AsyncMock()
            mock_backend.call_model.return_value = (
                "SYNTHESIS: The project shows strong technical feasibility with clear business value. "
                "Key considerations: ensure 1000 RPS capacity, maintain user-friendly interface, "
                "and track towards 150% ROI target."
            )
            mock_get_backend.return_value = mock_backend

            result = await formatter.synthesize_responses(responses, "integration_test")

            assert "SYNTHESIS:" in result
            assert "1000 RPS" in result
            assert "150% ROI" in result
            assert "user-friendly interface" in result

            # Verify the synthesis thread was configured correctly
            call_args = mock_backend.call_model.call_args[0]
            synthesis_thread = call_args[0]
            assert synthesis_thread.name == "synthesis"
            # Note: Thread doesn't have temperature attribute, it's part of backend config
            assert "expert analyst" in synthesis_thread.system_prompt
