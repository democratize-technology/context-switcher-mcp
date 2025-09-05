"""Enhanced pytest configuration for comprehensive testing"""

import asyncio
import sys  # noqa: E402
import uuid
from datetime import datetime

try:
    from datetime import UTC
except ImportError:
    # Python < 3.11 compatibility
    from datetime import timezone

    UTC = timezone.utc
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))  # noqa: E402


@pytest.fixture(autouse=True)
def reset_global_config_state():
    """Reset global configuration state before each test"""
    import context_switcher_mcp.config

    # Reset the global config instance before each test
    context_switcher_mcp.config._global_config_instance = None

    yield

    # Clean up after test
    context_switcher_mcp.config._global_config_instance = None


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    manager = AsyncMock()
    manager.get_session.return_value = Mock(
        session_id="test-session-123",
        status="active",
        created_at="2024-01-01T00:00:00Z",
    )
    manager.create_session.return_value = Mock(
        session_id=str(uuid.uuid4()),
        status="active",
        created_at=datetime.now(UTC),
    )
    return manager


@pytest.fixture
def mock_reasoning_orchestrator():
    """Mock reasoning orchestrator for testing"""
    orchestrator = AsyncMock()
    orchestrator.analyze_from_perspectives.return_value = {
        "perspectives": [
            {"name": "technical", "response": "Technical analysis"},
            {"name": "business", "response": "Business analysis"},
        ],
        "synthesis": "Combined analysis results",
        "confidence": 0.85,
    }
    return orchestrator


@pytest.fixture
def mock_backend():
    """Mock LLM backend for testing"""
    backend = AsyncMock()
    backend.call_model.return_value = "Test response from mock backend"
    return backend


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing tool registration"""
    server = Mock()
    server.tool = Mock()
    return server


@pytest.fixture
def sample_session_id():
    """Generate a sample session ID for testing"""
    return str(uuid.uuid4())


@pytest.fixture
def sample_thread_config():
    """Sample thread configuration for testing"""
    return {
        "thread_id": "test-thread-123",
        "perspective_name": "technical",
        "model_backend": "bedrock",
        "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "temperature": 0.7,
    }


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing"""
    limiter = Mock()
    limiter.is_rate_limited.return_value = False
    limiter.acquire_token.return_value = True
    return limiter


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def suppress_external_calls():
    """Auto-fixture to prevent actual external API calls during testing"""
    import unittest.mock

    # Mock common external calls
    with (
        unittest.mock.patch("boto3.client"),
        unittest.mock.patch("litellm.acompletion"),
        unittest.mock.patch("httpx.AsyncClient"),
    ):
        yield
