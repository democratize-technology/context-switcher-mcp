"""Comprehensive tests for session_tools module"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from context_switcher_mcp.tools.session_tools import (
    register_session_tools,  # noqa: E402
)

# Tests updated to match current API behavior


class TestSessionToolsRegistration:
    """Test session tools registration with FastMCP server"""

    def test_register_session_tools_success(self):
        """Test successful registration of all session tools"""
        mock_mcp = Mock()
        mock_mcp.tool = Mock()

        # This should not raise any exceptions
        register_session_tools(mock_mcp)

        # Verify that tool registration was called multiple times
        assert mock_mcp.tool.called
        # Should register 3 tools: list_sessions, list_templates, current_session
        assert mock_mcp.tool.call_count >= 3


class TestListSessionsFunction:
    """Test list_sessions tool functionality"""

    @pytest.fixture
    def mock_active_sessions(self):
        """Create mock active sessions data"""
        session_id_1 = str(uuid4())
        session_id_2 = str(uuid4())
        base_time = datetime.now(timezone.utc)

        session_1 = Mock()
        session_1.created_at = base_time
        session_1.threads = {"technical": Mock(), "business": Mock()}
        session_1.analyses = [
            {"prompt": "test analysis 1", "active_count": 2},
            {"prompt": "test analysis 2", "active_count": 2},
        ]
        session_1.topic = "API Design Decision"

        session_2 = Mock()
        session_2.created_at = base_time
        session_2.threads = {"risk": Mock()}
        session_2.analyses = [{"prompt": "risk analysis", "active_count": 1}]
        session_2.topic = "Security Review"

        return {session_id_1: session_1, session_id_2: session_2}

    @pytest.fixture
    def mock_session_stats(self):
        """Create mock session manager statistics"""
        return {
            "active_sessions": 2,
            "total_sessions": 15,
            "capacity_used": 0.4,
            "average_session_duration": 2400,
        }

    @pytest.mark.asyncio
    async def test_list_sessions_success(
        self, mock_active_sessions, mock_session_stats
    ):
        """Test successful session listing"""
        with patch("context_switcher_mcp.session_manager") as mock_sm:
            mock_sm.list_active_sessions = AsyncMock(return_value=mock_active_sessions)
            mock_sm.get_stats = AsyncMock(return_value=mock_session_stats)

            from context_switcher_mcp.tools.session_tools import register_session_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_session_tools(mock_mcp)
            list_sessions_func = tool_functions["list_sessions"]

            result = await list_sessions_func()

            # Verify response structure
            assert "sessions" in result
            assert "total_sessions" in result
            assert "stats" in result

            # Verify sessions data
            sessions = result["sessions"]
            assert len(sessions) == 2

            session_1 = next(s for s in sessions if s["topic"] == "API Design Decision")
            assert len(session_1["perspectives"]) == 2
            assert "technical" in session_1["perspectives"]
            assert "business" in session_1["perspectives"]
            assert session_1["analyses_count"] == 2

            session_2 = next(s for s in sessions if s["topic"] == "Security Review")
            assert len(session_2["perspectives"]) == 1
            assert "risk" in session_2["perspectives"]
            assert session_2["analyses_count"] == 1

            # Verify stats
            assert result["total_sessions"] == 2
            assert result["stats"] == mock_session_stats

            mock_sm.list_active_sessions.assert_called_once()
            mock_sm.get_stats.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self):
        """Test session listing when no sessions exist"""
        with patch("context_switcher_mcp.session_manager") as mock_sm:
            mock_sm.list_active_sessions = AsyncMock(return_value={})
            mock_sm.get_stats = AsyncMock(return_value={"active_sessions": 0})

            from context_switcher_mcp.tools.session_tools import register_session_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_session_tools(mock_mcp)
            list_sessions_func = tool_functions["list_sessions"]

            result = await list_sessions_func()

            assert result["sessions"] == []
            assert result["total_sessions"] == 0
            assert result["stats"]["active_sessions"] == 0


class TestListTemplatesFunction:
    """Test list_templates tool functionality"""

    @pytest.mark.asyncio
    async def test_list_templates_success(self):
        """Test successful template listing"""
        mock_templates = {
            "architecture_decision": {
                "perspectives": ["technical", "business", "user", "risk"],
                "custom": [],
            },
            "debugging_analysis": {
                "perspectives": ["technical", "user", "performance"],
                "custom": [("domain_expert", "Domain-specific debugging insights")],
            },
        }

        with patch(
            "context_switcher_mcp.tools.session_tools.PERSPECTIVE_TEMPLATES",
            mock_templates,
        ):
            from context_switcher_mcp.tools.session_tools import register_session_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_session_tools(mock_mcp)
            list_templates_func = tool_functions["list_templates"]

            result = await list_templates_func()

            # Verify response structure
            assert "templates" in result
            assert "usage" in result
            assert "example" in result

            templates = result["templates"]

            # Verify architecture_decision template
            arch_template = templates["architecture_decision"]
            assert arch_template["description"] == "Architecture Decision"
            assert len(arch_template["perspectives"]) == 4
            assert arch_template["total_perspectives"] == 4
            assert "technical" in arch_template["perspectives"]
            assert "business" in arch_template["perspectives"]

            # Verify debugging_analysis template with custom perspectives
            debug_template = templates["debugging_analysis"]
            assert debug_template["description"] == "Debugging Analysis"
            assert len(debug_template["perspectives"]) == 4  # 3 base + 1 custom
            assert debug_template["total_perspectives"] == 4
            assert "domain_expert (custom)" in debug_template["perspectives"]

            # Verify usage information
            assert "start_context_analysis" in result["usage"]
            assert "template" in result["example"]

    @pytest.mark.asyncio
    async def test_list_templates_empty(self):
        """Test template listing when no templates exist"""
        with patch(
            "context_switcher_mcp.tools.session_tools.PERSPECTIVE_TEMPLATES", {}
        ):
            from context_switcher_mcp.tools.session_tools import register_session_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_session_tools(mock_mcp)
            list_templates_func = tool_functions["list_templates"]

            result = await list_templates_func()

            assert result["templates"] == {}
            assert "usage" in result
            assert "example" in result


class TestCurrentSessionFunction:
    """Test current_session tool functionality"""

    @pytest.mark.asyncio
    async def test_current_session_success(self):
        """Test successful current session retrieval"""
        session_id = str(uuid4())
        base_time = datetime.now(timezone.utc)

        mock_session = Mock()
        mock_session.created_at = base_time
        mock_session.threads = {"technical": Mock(), "business": Mock()}
        mock_session.analyses = [
            {"prompt": "Short prompt", "active_count": 2},
            {
                "prompt": "This is a very long prompt that should be truncated because it exceeds 100 characters and we want to show only first part",
                "active_count": 2,
            },
        ]
        mock_session.topic = "Current Analysis Topic"

        mock_active_sessions = {session_id: mock_session}

        with patch("context_switcher_mcp.session_manager") as mock_sm:
            mock_sm.list_active_sessions = AsyncMock(return_value=mock_active_sessions)

            from context_switcher_mcp.tools.session_tools import register_session_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_session_tools(mock_mcp)
            current_session_func = tool_functions["current_session"]

            result = await current_session_func()

            # Verify response structure
            assert "session_id" in result
            assert "topic" in result
            assert "perspectives" in result
            assert "analyses_run" in result
            assert "created" in result
            assert "last_analysis" in result

            # Verify content
            assert result["session_id"] == session_id
            assert result["topic"] == "Current Analysis Topic"
            assert len(result["perspectives"]) == 2
            assert "technical" in result["perspectives"]
            assert "business" in result["perspectives"]
            assert result["analyses_run"] == 2

            # Verify last analysis is truncated
            last_analysis = result["last_analysis"]
            assert len(last_analysis["prompt"]) <= 103  # 100 chars + "..."
            assert last_analysis["prompt"].endswith("...")

    @pytest.mark.asyncio
    async def test_current_session_no_sessions(self):
        """Test current session when no sessions exist"""
        with patch("context_switcher_mcp.session_manager") as mock_sm:
            mock_sm.list_active_sessions = AsyncMock(return_value={})

            from context_switcher_mcp.tools.session_tools import register_session_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_session_tools(mock_mcp)
            current_session_func = tool_functions["current_session"]

            result = await current_session_func()

            assert result["status"] == "No active sessions"
            assert "start_context_analysis" in result["hint"]

    @pytest.mark.asyncio
    async def test_current_session_multiple_sessions(self):
        """Test current session selects the most recent one"""
        base_time = datetime.now(timezone.utc)

        session_id_1 = str(uuid4())
        session_1 = Mock()
        session_1.created_at = base_time
        session_1.topic = "Older Session"

        session_id_2 = str(uuid4())
        session_2 = Mock()
        session_2.created_at = base_time.replace(
            microsecond=base_time.microsecond + 1
        )  # Slightly newer
        session_2.threads = {"risk": Mock()}
        session_2.analyses = []
        session_2.topic = "Newer Session"

        mock_active_sessions = {session_id_1: session_1, session_id_2: session_2}

        with patch("context_switcher_mcp.session_manager") as mock_sm:
            mock_sm.list_active_sessions = AsyncMock(return_value=mock_active_sessions)

            from context_switcher_mcp.tools.session_tools import register_session_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_session_tools(mock_mcp)
            current_session_func = tool_functions["current_session"]

            result = await current_session_func()

            # Should select the newer session
            assert result["session_id"] == session_id_2
            assert result["topic"] == "Newer Session"
            assert result["analyses_run"] == 0
            assert result["last_analysis"] is None


class TestSessionToolsIntegration:
    """Test integration between session tools and overall functionality"""

    def test_all_tools_registered(self):
        """Test that all expected session tools are registered"""
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
        register_session_tools(mock_mcp)

        # Verify all expected tools are registered
        expected_tools = ["list_sessions", "list_templates", "current_session"]

        for tool_name in expected_tools:
            assert tool_name in registered_tools
            assert "description" in registered_tools[tool_name]
            assert callable(registered_tools[tool_name]["function"])

    @pytest.mark.asyncio
    async def test_session_workflow_integration(self):
        """Test a typical session management workflow"""
        session_id = str(uuid4())
        mock_session = Mock()
        mock_session.created_at = datetime.now(timezone.utc)
        mock_session.threads = {"technical": Mock()}
        mock_session.analyses = [{"prompt": "test analysis", "active_count": 1}]
        mock_session.topic = "Workflow Test"

        with patch("context_switcher_mcp.session_manager") as mock_sm:
            mock_sm.list_active_sessions = AsyncMock(
                return_value={session_id: mock_session}
            )
            mock_sm.get_stats = AsyncMock(return_value={"active_sessions": 1})

            from context_switcher_mcp.tools.session_tools import register_session_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_session_tools(mock_mcp)

            # Test workflow: list sessions -> get current session
            list_result = await tool_functions["list_sessions"]()
            current_result = await tool_functions["current_session"]()

            # Verify workflow consistency
            assert len(list_result["sessions"]) == 1
            assert list_result["sessions"][0]["session_id"] == session_id
            assert current_result["session_id"] == session_id
            assert current_result["topic"] == "Workflow Test"

    @pytest.mark.asyncio
    async def test_error_handling_in_session_tools(self):
        """Test error handling in session tools"""
        with patch("context_switcher_mcp.session_manager") as mock_sm:
            # Simulate session_manager failure
            mock_sm.list_active_sessions = AsyncMock(
                side_effect=Exception("Session manager unavailable")
            )

            from context_switcher_mcp.tools.session_tools import register_session_tools

            mock_mcp = Mock()
            tool_functions = {}

            def capture_tool(description):
                def decorator(func):
                    tool_functions[func.__name__] = func
                    return func

                return decorator

            mock_mcp.tool = capture_tool
            register_session_tools(mock_mcp)

            # Should raise the exception (tools don't have built-in error handling)
            with pytest.raises(Exception) as exc_info:
                await tool_functions["list_sessions"]()

            assert "Session manager unavailable" in str(exc_info.value)
