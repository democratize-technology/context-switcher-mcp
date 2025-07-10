"""Tests for Context-Switcher MCP"""

import pytest
from datetime import datetime

from src.context_switcher_mcp import (
    Thread,
    ContextSwitcherSession,
    ThreadOrchestrator,
    ModelBackend,
    start_context_analysis,
    add_perspective,
    StartContextAnalysisRequest,
    AddPerspectiveRequest,
    NO_RESPONSE,
    sessions
)


@pytest.fixture
def mock_thread():
    """Create a mock thread for testing"""
    return Thread(
        id="test-thread-1",
        name="test",
        system_prompt="Test perspective",
        model_backend=ModelBackend.BEDROCK,
        model_name=None
    )


@pytest.fixture
def mock_session():
    """Create a mock session for testing"""
    session = ContextSwitcherSession(
        session_id="test-session-1",
        created_at=datetime.utcnow()
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
    
    @pytest.mark.asyncio
    async def test_broadcast_with_mock_responses(self, mock_session):
        """Test broadcasting with mocked responses"""
        orchestrator = ThreadOrchestrator()
        
        # Create test threads
        thread1 = Thread(
            id="thread-1",
            name="perspective1",
            system_prompt="Test perspective 1",
            model_backend=ModelBackend.BEDROCK,
            model_name=None
        )
        thread2 = Thread(
            id="thread-2",
            name="perspective2", 
            system_prompt="Test perspective 2",
            model_backend=ModelBackend.BEDROCK,
            model_name=None
        )
        
        mock_session.add_thread(thread1)
        mock_session.add_thread(thread2)
        
        # Mock the backend calls
        async def mock_bedrock_response(thread):
            if thread.name == "perspective1":
                return "Response from perspective 1"
            else:
                return NO_RESPONSE
        
        orchestrator._call_bedrock = mock_bedrock_response
        
        # Test broadcast
        responses = await orchestrator.broadcast_message(
            mock_session.threads,
            "Test message"
        )
        
        assert len(responses) == 2
        assert responses["perspective1"] == "Response from perspective 1"
        assert responses["perspective2"] == NO_RESPONSE


class TestMCPTools:
    """Test MCP tool functions"""
    
    @pytest.mark.asyncio
    async def test_start_context_analysis(self):
        """Test starting a new analysis session"""
        # Clear any existing sessions
        sessions.clear()
        
        request = StartContextAnalysisRequest(
            topic="Test topic",
            model_backend=ModelBackend.BEDROCK
        )
        
        result = await start_context_analysis(request)
        
        assert "session_id" in result
        assert result["topic"] == "Test topic"
        assert len(result["perspectives"]) == 4  # Default perspectives
        assert "technical" in result["perspectives"]
        assert "business" in result["perspectives"]
        assert "user" in result["perspectives"]
        assert "risk" in result["perspectives"]
    
    @pytest.mark.asyncio
    async def test_add_perspective(self):
        """Test adding a custom perspective"""
        # First create a session
        sessions.clear()
        start_request = StartContextAnalysisRequest(
            topic="Test topic",
            model_backend=ModelBackend.BEDROCK
        )
        start_result = await start_context_analysis(start_request)
        session_id = start_result["session_id"]
        
        # Add a perspective
        add_request = AddPerspectiveRequest(
            session_id=session_id,
            name="security",
            description="Focus on security implications"
        )
        
        result = await add_perspective(add_request)
        
        assert result["perspective_added"] == "security"
        assert result["total_perspectives"] == 5
        assert "security" in result["all_perspectives"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
