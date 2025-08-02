"""Tests for Context-Switcher MCP"""

import pytest
from datetime import datetime

from src.context_switcher_mcp.models import Thread, ContextSwitcherSession, ModelBackend
from src.context_switcher_mcp.orchestrator import ThreadOrchestrator
from src.context_switcher_mcp.session_manager import SessionManager
from src.context_switcher_mcp import (
    session_manager,
)


@pytest.fixture
def mock_thread():
    """Create a mock thread for testing"""
    return Thread(
        id="test-thread-1",
        name="test",
        system_prompt="Test perspective",
        model_backend=ModelBackend.BEDROCK,
        model_name=None,
    )


@pytest.fixture
def mock_session():
    """Create a mock session for testing"""
    session = ContextSwitcherSession(
        session_id="test-session-1", created_at=datetime.utcnow()
    )
    return session


class TestThread:
    """Test Thread class"""

    def test_thread_creation(self, mock_thread):
        """Test thread is created correctly"""
        assert mock_thread.id == "test-thread-1"
        assert mock_thread.name == "test"
        assert mock_thread.system_prompt == "Test perspective"
        assert mock_thread.model_backend == ModelBackend.BEDROCK
        assert len(mock_thread.conversation_history) == 0

    def test_add_message(self, mock_thread):
        """Test adding messages to thread history"""
        mock_thread.add_message("user", "Test message")
        assert len(mock_thread.conversation_history) == 1
        assert mock_thread.conversation_history[0]["role"] == "user"
        assert mock_thread.conversation_history[0]["content"] == "Test message"
        assert "timestamp" in mock_thread.conversation_history[0]


class TestContextSwitcherSession:
    """Test ContextSwitcherSession class"""

    def test_session_creation(self, mock_session):
        """Test session is created correctly"""
        assert mock_session.session_id == "test-session-1"
        assert isinstance(mock_session.created_at, datetime)
        assert len(mock_session.threads) == 0
        assert len(mock_session.analyses) == 0

    def test_add_thread(self, mock_session, mock_thread):
        """Test adding thread to session"""
        mock_session.add_thread(mock_thread)
        assert len(mock_session.threads) == 1
        assert "test" in mock_session.threads
        assert mock_session.threads["test"] == mock_thread


class TestThreadOrchestrator:
    """Test ThreadOrchestrator class"""

    @pytest.mark.asyncio
    async def test_orchestrator_creation(self):
        """Test orchestrator is created with correct backends"""
        orchestrator = ThreadOrchestrator()
        assert ModelBackend.BEDROCK in orchestrator.backends
        assert ModelBackend.LITELLM in orchestrator.backends
        assert ModelBackend.OLLAMA in orchestrator.backends

    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality"""
        orchestrator = ThreadOrchestrator()

        # Check that circuit breakers are initialized
        assert ModelBackend.BEDROCK in orchestrator.circuit_breakers
        assert ModelBackend.LITELLM in orchestrator.circuit_breakers
        assert ModelBackend.OLLAMA in orchestrator.circuit_breakers

        # Test circuit breaker state
        cb = orchestrator.circuit_breakers[ModelBackend.BEDROCK]
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

        # Test recording failures
        for _ in range(5):  # Failure threshold is 5
            cb.record_failure()
        assert cb.state == "OPEN"

        # Test success recording
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0


class TestMCPTools:
    """Test MCP tool functions - testing core session logic"""

    def test_session_creation_logic(self):
        """Test session creation logic without MCP wrapper"""
        # Clear any existing sessions
        session_manager.sessions.clear()

        # Test basic session creation
        session = ContextSwitcherSession(
            session_id="test-session", created_at=datetime.utcnow()
        )
        session.topic = "Test topic"

        # Test adding default perspectives
        from src.context_switcher_mcp import DEFAULT_PERSPECTIVES

        for name, prompt in DEFAULT_PERSPECTIVES.items():
            thread = Thread(
                id=f"thread-{name}",
                name=name,
                system_prompt=prompt,
                model_backend=ModelBackend.BEDROCK,
                model_name=None,
            )
            session.add_thread(thread)

        assert len(session.threads) == 4
        assert "technical" in session.threads
        assert "business" in session.threads
        assert "user" in session.threads
        assert "risk" in session.threads

    def test_validation_functions(self):
        """Test input validation functions"""
        from src.context_switcher_mcp import validate_topic, validate_session_id

        # Test topic validation
        valid, error = validate_topic("Valid topic")
        assert valid is True
        assert error == ""

        valid, error = validate_topic("")
        assert valid is False
        assert "empty" in error.lower()

        valid, error = validate_topic("x" * 1001)  # Too long
        assert valid is False
        assert "too long" in error.lower()

        # Test session ID validation (without existing session)
        valid, error = validate_session_id("non-existent-session")
        assert valid is False
        assert "not found" in error.lower()


class TestSessionManager:
    """Test SessionManager class"""

    def test_session_manager_creation(self):
        """Test session manager is created correctly"""
        sm = SessionManager(max_sessions=10, session_ttl_hours=1)
        assert sm.max_sessions == 10
        assert sm.session_ttl.total_seconds() == 3600
        assert len(sm.sessions) == 0

    def test_add_session(self):
        """Test adding sessions to manager"""
        sm = SessionManager(max_sessions=2)

        # Add first session
        session1 = ContextSwitcherSession(
            session_id="test-1", created_at=datetime.utcnow()
        )
        assert sm.add_session(session1) is True
        assert len(sm.sessions) == 1

        # Add second session
        session2 = ContextSwitcherSession(
            session_id="test-2", created_at=datetime.utcnow()
        )
        assert sm.add_session(session2) is True
        assert len(sm.sessions) == 2

        # Try to add third session (should fail)
        session3 = ContextSwitcherSession(
            session_id="test-3", created_at=datetime.utcnow()
        )
        assert sm.add_session(session3) is False
        assert len(sm.sessions) == 2

    def test_get_session(self):
        """Test retrieving sessions"""
        sm = SessionManager()
        session = ContextSwitcherSession(
            session_id="test-1", created_at=datetime.utcnow()
        )
        sm.add_session(session)

        # Get existing session
        retrieved = sm.get_session("test-1")
        assert retrieved is not None
        assert retrieved.session_id == "test-1"

        # Get non-existent session
        assert sm.get_session("test-999") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
