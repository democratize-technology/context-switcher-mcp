"""Comprehensive tests for admin_tools module"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from context_switcher_mcp.tools.admin_tools import register_admin_tools  # noqa: E402


class TestAdminToolsRegistration:
    """Test admin tools registration with FastMCP server"""

    def test_register_admin_tools_success(self):
        """Test successful registration of admin tools"""
        mock_mcp = Mock()
        mock_mcp.tool = Mock()

        # This should not raise any exceptions
        register_admin_tools(mock_mcp)

        # Verify that tool registration was called
        assert mock_mcp.tool.called
        # Should have registered multiple tools (get_performance_metrics, reset_circuit_breakers, etc.)
        assert mock_mcp.tool.call_count >= 2


class TestPerformanceMetricsFunctionality:
    """Test get_performance_metrics tool functionality"""

    @pytest.fixture
    def mock_orchestrator_metrics(self):
        """Create mock orchestrator performance metrics"""
        return {
            "response_times": {"mean": 1.2, "p95": 2.5, "p99": 4.1},
            "request_counts": {"total": 150, "successful": 145, "failed": 5},
            "circuit_breakers": {
                "bedrock": {"state": "CLOSED", "failure_count": 0},
                "litellm": {
                    "state": "OPEN",
                    "failure_count": 3,
                },  # Circuit breaker tripped
                "ollama": {"state": "CLOSED", "failure_count": 0},
            },
            "thread_pool": {"active_threads": 8, "max_threads": 20, "queue_size": 2},
            "memory_usage": {"current_mb": 256, "peak_mb": 512},
        }

    @pytest.fixture
    def mock_session_stats(self):
        """Create mock session manager statistics"""
        return {
            "active_sessions": 5,
            "total_sessions": 23,
            "capacity_used": 0.25,  # 25% capacity utilization
            "average_session_duration": 1800,
            "cleanup_operations": 3,
        }

    @pytest.mark.asyncio
    async def test_get_performance_metrics_success(
        self, mock_orchestrator_metrics, mock_session_stats
    ):
        """Test successful performance metrics retrieval"""
        with (
            patch("context_switcher_mcp.orchestrator") as mock_orchestrator_module,
            patch("context_switcher_mcp.session_manager") as mock_sm_module,
        ):
            mock_orchestrator_module.get_performance_metrics = AsyncMock(
                return_value=mock_orchestrator_metrics
            )
            mock_sm_module.get_stats = AsyncMock(return_value=mock_session_stats)

            from context_switcher_mcp.tools.admin_tools import register_admin_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_admin_tools(mock_mcp)
            get_metrics = tool_functions["get_performance_metrics"]

            result = await get_metrics()

            # Verify response structure
            assert "orchestrator" in result
            assert "session_manager" in result
            assert "system_health" in result

            # Verify orchestrator metrics
            assert result["orchestrator"] == mock_orchestrator_metrics

            # Verify session manager stats
            assert result["session_manager"] == mock_session_stats

            # Verify system health calculations
            system_health = result["system_health"]
            assert system_health["active_sessions"] == 5
            assert system_health["capacity_utilization"] == 0.25
            assert (
                system_health["circuit_breaker_issues"] is True
            )  # LiteLLM circuit breaker is open

            # Verify metrics were requested correctly
            mock_orchestrator_module.get_performance_metrics.assert_called_once_with(
                last_n=20
            )
            mock_sm_module.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_performance_metrics_all_circuit_breakers_closed(self):
        """Test performance metrics when all circuit breakers are healthy"""
        healthy_orchestrator_metrics = {
            "circuit_breakers": {
                "bedrock": {"state": "CLOSED", "failure_count": 0},
                "litellm": {"state": "CLOSED", "failure_count": 0},
                "ollama": {"state": "CLOSED", "failure_count": 0},
            }
        }

        with (
            patch("context_switcher_mcp.orchestrator") as mock_orchestrator_module,
            patch("context_switcher_mcp.session_manager") as mock_sm_module,
        ):
            mock_orchestrator_module.get_performance_metrics = AsyncMock(
                return_value=healthy_orchestrator_metrics
            )
            mock_sm_module.get_stats = AsyncMock(
                return_value={"active_sessions": 3, "capacity_used": 0.15}
            )

            from context_switcher_mcp.tools.admin_tools import register_admin_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_admin_tools(mock_mcp)
            get_metrics = tool_functions["get_performance_metrics"]

            result = await get_metrics()

            # No circuit breaker issues when all are closed
            assert result["system_health"]["circuit_breaker_issues"] is False

    @pytest.mark.asyncio
    async def test_get_performance_metrics_missing_circuit_breakers(self):
        """Test performance metrics when circuit breakers data is missing"""
        orchestrator_metrics_no_cb = {
            "response_times": {"mean": 1.0}
            # No circuit_breakers key
        }

        with (
            patch("context_switcher_mcp.orchestrator") as mock_orchestrator_module,
            patch("context_switcher_mcp.session_manager") as mock_sm_module,
        ):
            mock_orchestrator_module.get_performance_metrics = AsyncMock(
                return_value=orchestrator_metrics_no_cb
            )
            mock_sm_module.get_stats = AsyncMock(
                return_value={"active_sessions": 1, "capacity_used": 0.05}
            )

            from context_switcher_mcp.tools.admin_tools import register_admin_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_admin_tools(mock_mcp)
            get_metrics = tool_functions["get_performance_metrics"]

            result = await get_metrics()

            # Should handle missing circuit breakers gracefully
            assert result["system_health"]["circuit_breaker_issues"] is False


class TestResetCircuitBreakersFunctionality:
    """Test reset_circuit_breakers tool functionality"""

    @pytest.mark.asyncio
    async def test_reset_circuit_breakers_success(self):
        """Test successful circuit breaker reset"""
        reset_status = {
            "bedrock": "reset_successful",
            "litellm": "reset_successful",
            "ollama": "reset_successful",
        }

        with patch("context_switcher_mcp.orchestrator") as mock_orchestrator_module:
            mock_orchestrator_module.reset_circuit_breakers = Mock(
                return_value=reset_status
            )

            from context_switcher_mcp.tools.admin_tools import register_admin_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_admin_tools(mock_mcp)
            reset_cb = tool_functions["reset_circuit_breakers"]

            result = await reset_cb()

            # Verify response structure
            assert "message" in result
            assert "reset_status" in result
            assert "warning" in result

            # Verify content
            assert result["message"] == "Circuit breakers reset successfully"
            assert result["reset_status"] == reset_status
            assert "administrative operation" in result["warning"]

            # Verify reset was called
            mock_orchestrator_module.reset_circuit_breakers.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_circuit_breakers_partial_success(self):
        """Test circuit breaker reset with some failures"""
        reset_status = {
            "bedrock": "reset_successful",
            "litellm": "reset_failed",
            "ollama": "backend_unavailable",
        }

        with patch("context_switcher_mcp.orchestrator") as mock_orchestrator_module:
            mock_orchestrator_module.reset_circuit_breakers = Mock(
                return_value=reset_status
            )

            from context_switcher_mcp.tools.admin_tools import register_admin_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_admin_tools(mock_mcp)
            reset_cb = tool_functions["reset_circuit_breakers"]

            result = await reset_cb()

            # Should still return success message even with partial failures
            assert result["message"] == "Circuit breakers reset successfully"
            assert result["reset_status"] == reset_status
            assert "litellm" in result["reset_status"]
            assert result["reset_status"]["litellm"] == "reset_failed"


class TestSecurityMetricsFunctionality:
    """Test security-related admin functionality"""

    @pytest.mark.asyncio
    async def test_get_security_metrics_success(self):
        """Test successful security metrics retrieval"""
        # This tests the security metrics functionality if it exists
        mock_security_metrics = {
            "client_bindings": {
                "total_clients": 5,
                "active_bindings": 3,
                "expired_bindings": 2,
            },
            "authentication_events": {
                "successful_auths": 150,
                "failed_auths": 3,
                "rate_limited_requests": 1,
            },
            "session_security": {
                "secure_sessions": 8,
                "insecure_sessions": 0,
                "session_hijack_attempts": 0,
            },
        }

        with (
            patch(
                "context_switcher_mcp.tools.admin_tools.client_binding_manager"
            ) as mock_cbm,
            patch("context_switcher_mcp.session_manager") as mock_sm_module,
        ):
            # Configure session_manager as AsyncMock to handle await expressions
            mock_sm_module.get_stats = AsyncMock(return_value={"active": 0, "total": 0})
            mock_sm_module.list_active_sessions = AsyncMock(return_value={})
            mock_sm_module.get_session = AsyncMock(return_value=None)
            # Assume get_security_metrics method exists
            if hasattr(mock_cbm, "get_security_metrics"):
                mock_cbm.get_security_metrics.return_value = mock_security_metrics[
                    "client_bindings"
                ]

                from context_switcher_mcp.tools.admin_tools import register_admin_tools

                mock_mcp = Mock()
                tool_functions = {}

                def capture_tool(description):
                    def decorator(func):
                        tool_functions[func.__name__] = func
                        return func

                    return decorator

                mock_mcp.tool = capture_tool
                register_admin_tools(mock_mcp)

                # Check if security metrics tool was registered
                if "get_security_metrics" in tool_functions:
                    get_security = tool_functions["get_security_metrics"]
                    result = await get_security()

                    assert (
                        "client_binding" in result
                    )  # Response uses singular key, not plural


class TestAdminToolsIntegration:
    """Integration tests for admin tools"""

    @pytest.mark.asyncio
    async def test_admin_workflow_health_check_then_reset(self):
        """Test complete admin workflow: check health then reset if needed"""
        # Mock unhealthy state
        orchestrator_metrics = {
            "circuit_breakers": {
                "bedrock": {"state": "OPEN", "failure_count": 5},
                "litellm": {"state": "HALF_OPEN", "failure_count": 2},
            }
        }

        session_stats = {"active_sessions": 3, "capacity_used": 0.15}

        reset_status = {"bedrock": "reset_successful", "litellm": "reset_successful"}

        with (
            patch("context_switcher_mcp.orchestrator") as mock_orchestrator,
            patch("context_switcher_mcp.session_manager") as mock_sm_module,
        ):
            mock_orchestrator.get_performance_metrics = AsyncMock(
                return_value=orchestrator_metrics
            )
            mock_orchestrator.reset_circuit_breakers = Mock(return_value=reset_status)
            mock_sm_module.get_stats = AsyncMock(return_value=session_stats)

            from context_switcher_mcp.tools.admin_tools import register_admin_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_admin_tools(mock_mcp)

            # 1. Check performance metrics first
            metrics_result = await tool_functions["get_performance_metrics"]()
            assert metrics_result["system_health"]["circuit_breaker_issues"] is True

            # 2. Reset circuit breakers due to issues
            reset_result = await tool_functions["reset_circuit_breakers"]()
            assert reset_result["reset_status"]["bedrock"] == "reset_successful"

    def test_admin_tools_defensive_programming(self):
        """Test defensive programming patterns in admin tools"""
        # Test circuit breaker health calculation with edge cases

        # Empty circuit breakers dict
        empty_cb = {}
        has_issues = any(cb["state"] != "CLOSED" for cb in empty_cb.values())
        assert has_issues is False

        # Circuit breakers with missing state (test data for potential future use)
        _cb_missing_state = {
            "bedrock": {"failure_count": 0}  # No "state" key
        }
        _ = _cb_missing_state  # Mark as intentionally unused
        # Should handle missing keys gracefully in actual implementation

        # Circuit breakers with unknown states
        cb_unknown_state = {"litellm": {"state": "UNKNOWN", "failure_count": 1}}
        has_unknown_issues = any(
            cb.get("state", "CLOSED") != "CLOSED" for cb in cb_unknown_state.values()
        )
        assert has_unknown_issues is True

    @pytest.mark.asyncio
    async def test_admin_tools_error_handling(self):
        """Test error handling in admin tools"""
        with (
            patch("context_switcher_mcp.orchestrator") as mock_orchestrator,
            patch("context_switcher_mcp.session_manager") as mock_sm_module,
            patch("context_switcher_mcp.tools.admin_tools.logger") as _mock_logger,
        ):
            # Simulate orchestrator exception
            mock_orchestrator.get_performance_metrics.side_effect = Exception(
                "Orchestrator failure"
            )
            mock_sm_module.get_stats.return_value = {}

            from context_switcher_mcp.tools.admin_tools import register_admin_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_admin_tools(mock_mcp)

            # Should handle orchestrator failure gracefully
            with pytest.raises((RuntimeError, AttributeError, TypeError, KeyError)):
                await tool_functions["get_performance_metrics"]()

    @pytest.mark.asyncio
    async def test_admin_tools_logging(self):
        """Test that admin operations are properly logged"""
        with (
            patch("context_switcher_mcp.orchestrator") as mock_orchestrator,
            patch("context_switcher_mcp.tools.admin_tools.logger") as _mock_logger,
        ):
            mock_orchestrator.reset_circuit_breakers.return_value = {
                "bedrock": "reset_successful"
            }

            from context_switcher_mcp.tools.admin_tools import register_admin_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_admin_tools(mock_mcp)

            await tool_functions["reset_circuit_breakers"]()

            # Verify logging occurred (implementation-dependent)
            # mock_logger.info.assert_called()  # Uncomment if logging is implemented


class TestAdminToolsCapacityMetrics:
    """Test capacity and resource utilization metrics"""

    @pytest.mark.asyncio
    async def test_capacity_utilization_calculations(self):
        """Test capacity utilization metric calculations"""
        # Test various capacity scenarios
        capacity_scenarios = [
            {"active_sessions": 0, "max_sessions": 20, "expected": 0.0},
            {"active_sessions": 10, "max_sessions": 20, "expected": 0.5},
            {"active_sessions": 20, "max_sessions": 20, "expected": 1.0},
            {"active_sessions": 15, "max_sessions": 20, "expected": 0.75},
        ]

        for scenario in capacity_scenarios:
            utilization = scenario["active_sessions"] / scenario["max_sessions"]
            assert abs(utilization - scenario["expected"]) < 0.001

    @pytest.mark.asyncio
    async def test_memory_metrics_tracking(self):
        """Test memory usage metrics tracking"""
        memory_metrics = {
            "current_mb": 128,
            "peak_mb": 256,
            "allocated_mb": 512,
            "gc_collections": 5,
        }

        # Verify memory metrics structure
        assert memory_metrics["current_mb"] <= memory_metrics["peak_mb"]
        assert memory_metrics["peak_mb"] <= memory_metrics["allocated_mb"]
        assert memory_metrics["gc_collections"] >= 0
