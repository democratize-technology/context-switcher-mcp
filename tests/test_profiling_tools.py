"""Comprehensive tests for profiling_tools module"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from context_switcher_mcp.tools.profiling_tools import (
    ProfilingConfigRequest,
    SessionProfilingRequest,
    configure_profiling,
    get_llm_profiling_status,
    get_session_profiling_data,
    register_profiling_tools,
    reset_profiling_data,
)

# Previously skipped due to API mismatches - now fixed


class TestProfilingToolsRegistration:
    """Test profiling tools registration with FastMCP server"""

    def test_register_profiling_tools_success(self):
        """Test successful registration of all profiling tools"""
        mock_mcp = Mock()
        mock_mcp.tool = Mock()

        # This should not raise any exceptions
        register_profiling_tools(mock_mcp)

        # Verify that tool registration was called multiple times
        assert mock_mcp.tool.called
        # Should register 4 tools: status, session profiling, config, and reset
        assert mock_mcp.tool.call_count == 4


class TestRequestModels:
    """Test Pydantic request model validation"""

    def test_session_profiling_request_validation(self):
        """Test SessionProfilingRequest validation"""
        session_id = str(uuid4())
        request = SessionProfilingRequest(session_id=session_id)
        assert request.session_id == session_id

    def test_profiling_config_request_optional_fields(self):
        """Test ProfilingConfigRequest optional fields"""
        request = ProfilingConfigRequest()
        assert request.enabled is None
        assert request.sampling_rate is None
        assert request.track_costs is None
        assert request.track_memory is None

        request = ProfilingConfigRequest(enabled=True, sampling_rate=0.5, track_costs=True, track_memory=False)
        assert request.enabled is True
        assert request.sampling_rate == 0.5
        assert request.track_costs is True
        assert request.track_memory is False


class TestProfilingStatusFunction:
    """Test get_llm_profiling_status function"""

    @pytest.mark.asyncio
    async def test_get_profiling_status_success(self):
        """Test successful profiling status retrieval"""
        mock_status = {
            "enabled": True,
            "sampling_rate": 0.1,
            "statistics": {
                "total_calls": 150,
                "profiled_calls": 15,
                "success_rate": 95.5,
            },
            "storage": {"current_usage": 1024, "max_capacity": 10240},
        }

        with patch("context_switcher_mcp.tools.profiling_tools.get_global_profiler") as mock_get_profiler:
            mock_profiler = Mock()
            mock_profiler.get_configuration_status.return_value = mock_status
            mock_get_profiler.return_value = mock_profiler

            result = await get_llm_profiling_status()

            assert result == mock_status
            mock_get_profiler.assert_called_once()
            mock_profiler.get_configuration_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_profiling_status_error(self):
        """Test profiling status retrieval with error"""
        with patch("context_switcher_mcp.tools.profiling_tools.get_global_profiler") as mock_get_profiler:
            mock_get_profiler.side_effect = Exception("Profiler not initialized")

            result = await get_llm_profiling_status()

            assert "error" in result
            assert "message" in result
            assert result["error"] == "Failed to get profiling status"
            assert "Profiler not initialized" in result["message"]


class TestSessionProfilingFunction:
    """Test get_session_profiling_data function"""

    @pytest.fixture
    def mock_session_metrics(self):
        """Create mock session profiling metrics"""
        session_id = str(uuid4())
        base_time = datetime.now(timezone.utc)

        return [
            Mock(
                session_id=session_id,
                timestamp=base_time,
                thread_name="thread-1",
                backend="bedrock",
                success=True,
                total_latency=1.2,
                estimated_cost_usd=0.05,
                total_tokens=150,
                error_type=None,
            ),
            Mock(
                session_id=session_id,
                timestamp=base_time,
                thread_name="thread-2",
                backend="litellm",
                success=False,
                total_latency=2.1,
                estimated_cost_usd=None,
                total_tokens=None,
                error_type="timeout",
            ),
        ]

    @pytest.mark.asyncio
    async def test_session_profiling_success(self):
        """Test successful session profiling data retrieval"""
        session_id = str(uuid4())
        base_time = datetime.now(timezone.utc)

        # Create mock session metrics with the same session_id
        mock_session_metrics = [
            Mock(
                session_id=session_id,
                timestamp=base_time,
                thread_name="thread-1",
                backend="bedrock",
                success=True,
                total_latency=1.2,
                estimated_cost_usd=0.05,
                total_tokens=150,
                error_type=None,
            ),
            Mock(
                session_id=session_id,
                timestamp=base_time,
                thread_name="thread-2",
                backend="litellm",
                success=False,
                total_latency=2.1,
                estimated_cost_usd=None,
                total_tokens=None,
                error_type="timeout",
            ),
        ]

        with (
            patch("context_switcher_mcp.tools.profiling_tools.validate_session_id") as mock_validate,
            patch("context_switcher_mcp.tools.profiling_tools.get_global_profiler") as mock_get_profiler,
        ):
            mock_validate.return_value = (True, None)

            mock_profiler = Mock()
            mock_metrics_history = Mock()
            mock_metrics_history.is_empty.return_value = False
            mock_metrics_history.get_all.return_value = mock_session_metrics
            mock_profiler.metrics_history = mock_metrics_history
            mock_get_profiler.return_value = mock_profiler

            result = await get_session_profiling_data(session_id)

            assert result["session_id"] == session_id
            assert "summary" in result
            assert "backend_usage" in result
            assert "thread_activity" in result
            assert "timeline" in result

            # Verify summary calculations
            summary = result["summary"]
            assert summary["total_calls"] == 2
            assert summary["success_rate"] == 50.0  # 1 out of 2 successful

    @pytest.mark.asyncio
    async def test_session_profiling_invalid_session(self):
        """Test session profiling with invalid session ID"""
        with patch("context_switcher_mcp.validation.validate_session_id") as mock_validate:
            mock_validate.return_value = (False, "Invalid UUID format")

            result = await get_session_profiling_data("invalid-id")

            assert "error" in result
            assert result["error"] == "Invalid session ID"
            assert "not found or expired" in result["message"]

    @pytest.mark.asyncio
    async def test_session_profiling_no_data(self):
        """Test session profiling when no data found for session"""
        session_id = str(uuid4())

        with (
            patch("context_switcher_mcp.tools.profiling_tools.validate_session_id") as mock_validate,
            patch("context_switcher_mcp.tools.profiling_tools.get_global_profiler") as mock_get_profiler,
        ):
            mock_validate.return_value = (True, None)

            mock_profiler = Mock()
            mock_metrics_history = Mock()
            mock_metrics_history.is_empty.return_value = False
            mock_metrics_history.get_all.return_value = []  # No metrics for this session
            mock_profiler.metrics_history = mock_metrics_history
            mock_get_profiler.return_value = mock_profiler

            result = await get_session_profiling_data(session_id)

            assert result["session_id"] == session_id
            assert result["total_calls"] == 0
            assert "No profiling data found" in result["message"]


class TestResetProfilingFunction:
    """Test reset_profiling_data function"""

    @pytest.mark.asyncio
    async def test_reset_profiling_success(self):
        """Test successful profiling data reset"""
        mock_old_status = {
            "statistics": {"total_calls": 250, "profiled_calls": 25},
            "storage": {"current_usage": 2048},
        }

        with patch("context_switcher_mcp.tools.profiling_tools.get_global_profiler") as mock_get_profiler:
            mock_profiler = Mock()
            mock_profiler.get_configuration_status.return_value = mock_old_status
            mock_profiler.metrics_history = Mock()
            mock_profiler._total_calls = 250
            mock_profiler._profiled_calls = 25
            mock_get_profiler.return_value = mock_profiler

            result = await reset_profiling_data()

            assert result["status"] == "success"
            assert "All profiling data has been reset" in result["message"]
            assert "previous_stats" in result
            assert result["previous_stats"]["total_calls"] == 250
            assert "cannot be undone" in result["warning"]

            # Verify reset operations were called
            mock_profiler.metrics_history.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_profiling_error(self):
        """Test profiling data reset with error"""
        with patch("context_switcher_mcp.tools.profiling_tools.get_global_profiler") as mock_get_profiler:
            mock_get_profiler.side_effect = Exception("Profiler access denied")

            result = await reset_profiling_data()

            assert "error" in result
            assert result["error"] == "Failed to reset profiling data"


class TestConfigureProfilingFunction:
    """Test configure_profiling function"""

    @pytest.mark.asyncio
    async def test_configure_profiling_success(self):
        """Test successful profiling configuration update"""
        mock_updated_config = {
            "enabled": True,
            "sampling_rate": 0.2,
            "track_costs": True,
            "track_memory": False,
        }

        with patch("context_switcher_mcp.tools.profiling_tools.get_global_profiler") as mock_get_profiler:
            mock_profiler = Mock()
            mock_config = Mock()
            mock_profiler.config = mock_config
            mock_profiler.get_configuration_status.return_value = mock_updated_config
            mock_get_profiler.return_value = mock_profiler

            result = await configure_profiling(enabled=True, sampling_rate=0.2, track_costs=True, track_memory=False)

            assert result["status"] == "success"
            assert "configuration updated" in result["message"]
            assert result["updated_config"] == mock_updated_config

            # Verify configuration updates
            assert mock_config.enabled
            assert mock_config.sampling_rate == 0.2
            assert mock_config.track_costs
            assert not mock_config.track_memory

    @pytest.mark.asyncio
    async def test_configure_profiling_invalid_sampling_rate(self):
        """Test profiling configuration with invalid sampling rate"""
        result = await configure_profiling(sampling_rate=1.5)

        assert "error" in result
        assert result["error"] == "Invalid sampling rate"
        assert "must be between 0.0 and 1.0" in result["message"]

        result = await configure_profiling(sampling_rate=-0.1)

        assert "error" in result
        assert result["error"] == "Invalid sampling rate"

    @pytest.mark.asyncio
    async def test_configure_profiling_partial_update(self):
        """Test profiling configuration with only some parameters"""
        with patch("context_switcher_mcp.tools.profiling_tools.get_global_profiler") as mock_get_profiler:
            mock_profiler = Mock()
            mock_config = Mock()
            mock_profiler.config = mock_config
            mock_profiler.get_configuration_status.return_value = {"enabled": False}
            mock_get_profiler.return_value = mock_profiler

            result = await configure_profiling(enabled=False)

            assert result["status"] == "success"
            assert not mock_config.enabled
            # Other properties should not be modified


class TestProfilingToolsIntegration:
    """Test integration between profiling tools and MCP registration"""

    def test_tools_registration_complete(self):
        """Test that all expected tools are registered"""
        mock_mcp = Mock()
        registered_tools = {}

        def capture_tool(description):
            def decorator(func):
                registered_tools[func.__name__] = {
                    "function": func,
                    "description": description,
                }
                return func

            return decorator

        mock_mcp.tool = capture_tool
        register_profiling_tools(mock_mcp)

        # Verify all expected tools are registered
        expected_tools = [
            "get_profiling_status",
            "get_session_profiling_analysis",
            "configure_profiling_settings",
            "reset_profiling_metrics",
        ]

        for tool_name in expected_tools:
            assert tool_name in registered_tools
            assert "description" in registered_tools[tool_name]
            assert callable(registered_tools[tool_name]["function"])

    @pytest.mark.asyncio
    async def test_tool_integration_with_pydantic_models(self):
        """Test that tools work correctly with Pydantic request models"""
        mock_mcp = Mock()
        registered_tools = {}

        def capture_tool(description):
            def decorator(func):
                registered_tools[func.__name__] = func
                return func

            return decorator

        mock_mcp.tool = capture_tool
        register_profiling_tools(mock_mcp)

        # Test session profiling tool with Pydantic model
        session_id = str(uuid4())

        with (
            patch("context_switcher_mcp.tools.profiling_tools.validate_session_id") as mock_validate,
            patch("context_switcher_mcp.tools.profiling_tools.get_global_profiler") as mock_get_profiler,
        ):
            mock_validate.return_value = (True, None)

            mock_profiler = Mock()
            mock_metrics_history = Mock()
            mock_metrics_history.is_empty.return_value = False
            mock_metrics_history.get_all.return_value = []
            mock_profiler.metrics_history = mock_metrics_history
            mock_get_profiler.return_value = mock_profiler

            session_tool = registered_tools["get_session_profiling_analysis"]
            request = SessionProfilingRequest(session_id=session_id)

            result = await session_tool(request)

            assert result["session_id"] == session_id
            assert result["total_calls"] == 0
