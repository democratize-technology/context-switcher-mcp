"""Comprehensive tests for session data models"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from context_switcher_mcp.models import ModelBackend, Thread
from context_switcher_mcp.session_data import (
    AnalysisRecord,
    SessionData,
)


class TestAnalysisRecord:
    """Test suite for AnalysisRecord dataclass"""

    @pytest.fixture
    def sample_analysis_record(self):
        """Create a sample AnalysisRecord for testing"""
        return AnalysisRecord(
            prompt="Test analysis prompt",
            timestamp=datetime(2023, 6, 15, 12, 30, 45, tzinfo=timezone.utc),
            responses={
                "technical": "Technical response",
                "business": "Business response",
            },
            active_count=2,
            abstained_count=0,
        )

    def test_analysis_record_creation(self, sample_analysis_record):
        """Test AnalysisRecord creation and attributes"""
        assert sample_analysis_record.prompt == "Test analysis prompt"
        assert sample_analysis_record.timestamp == datetime(
            2023, 6, 15, 12, 30, 45, tzinfo=timezone.utc
        )
        assert sample_analysis_record.responses == {
            "technical": "Technical response",
            "business": "Business response",
        }
        assert sample_analysis_record.active_count == 2
        assert sample_analysis_record.abstained_count == 0

    def test_analysis_record_to_dict(self, sample_analysis_record):
        """Test conversion of AnalysisRecord to dictionary"""
        result = sample_analysis_record.to_dict()

        expected = {
            "prompt": "Test analysis prompt",
            "timestamp": "2023-06-15T12:30:45+00:00",
            "responses": {
                "technical": "Technical response",
                "business": "Business response",
            },
            "active_count": 2,
            "abstained_count": 0,
        }

        assert result == expected

    def test_analysis_record_from_dict(self):
        """Test creation of AnalysisRecord from dictionary"""
        data = {
            "prompt": "Test prompt from dict",
            "timestamp": "2023-06-15T12:30:45+00:00",
            "responses": {"perspective1": "Response 1", "perspective2": "Response 2"},
            "active_count": 2,
            "abstained_count": 1,
        }

        record = AnalysisRecord.from_dict(data)

        assert record.prompt == "Test prompt from dict"
        assert record.timestamp == datetime(
            2023, 6, 15, 12, 30, 45, tzinfo=timezone.utc
        )
        assert record.responses == {
            "perspective1": "Response 1",
            "perspective2": "Response 2",
        }
        assert record.active_count == 2
        assert record.abstained_count == 1

    def test_analysis_record_roundtrip_serialization(self, sample_analysis_record):
        """Test roundtrip serialization (to_dict -> from_dict)"""
        dict_data = sample_analysis_record.to_dict()
        reconstructed = AnalysisRecord.from_dict(dict_data)

        assert reconstructed.prompt == sample_analysis_record.prompt
        assert reconstructed.timestamp == sample_analysis_record.timestamp
        assert reconstructed.responses == sample_analysis_record.responses
        assert reconstructed.active_count == sample_analysis_record.active_count
        assert reconstructed.abstained_count == sample_analysis_record.abstained_count


class TestSessionData:
    """Test suite for SessionData dataclass"""

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
    def another_thread(self):
        """Create another sample Thread for testing"""
        return Thread(
            id="thread-456",
            name="business",
            system_prompt="You are a business expert",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-4",
        )

    @pytest.fixture
    def sample_session_data(self, sample_thread):
        """Create a sample SessionData for testing"""
        session_data = SessionData(
            session_id="session-789",
            created_at=datetime(2023, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
            topic="Test Topic",
        )
        session_data.add_thread(sample_thread)
        return session_data

    def test_session_data_creation(self):
        """Test SessionData creation with default values"""
        session_data = SessionData(
            session_id="test-session",
            created_at=datetime.now(timezone.utc),
        )

        assert session_data.session_id == "test-session"
        assert session_data.topic is None
        assert len(session_data.threads) == 0
        assert len(session_data.analyses) == 0
        assert isinstance(session_data.created_at, datetime)

    def test_session_data_with_topic(self):
        """Test SessionData creation with topic"""
        session_data = SessionData(
            session_id="test-session",
            created_at=datetime.now(timezone.utc),
            topic="AI Ethics Discussion",
        )

        assert session_data.topic == "AI Ethics Discussion"

    def test_add_thread(self, sample_session_data, another_thread):
        """Test adding threads to session"""
        initial_count = len(sample_session_data.threads)

        sample_session_data.add_thread(another_thread)

        assert len(sample_session_data.threads) == initial_count + 1
        assert "business" in sample_session_data.threads
        assert sample_session_data.threads["business"] == another_thread

    def test_get_thread(self, sample_session_data):
        """Test retrieving threads by name"""
        # Existing thread
        thread = sample_session_data.get_thread("technical")
        assert thread is not None
        assert thread.name == "technical"

        # Non-existing thread
        thread = sample_session_data.get_thread("nonexistent")
        assert thread is None

    def test_record_analysis(self, sample_session_data):
        """Test recording analysis results"""
        responses = {"technical": "Tech analysis", "business": "Business analysis"}
        prompt = "Analyze this scenario"

        initial_count = len(sample_session_data.analyses)

        with patch("context_switcher_mcp.session_data.datetime") as mock_datetime:
            mock_time = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            sample_session_data.record_analysis(prompt, responses, 2, 0)

        assert len(sample_session_data.analyses) == initial_count + 1

        latest_analysis = sample_session_data.analyses[-1]
        assert latest_analysis.prompt == prompt
        assert latest_analysis.responses == responses
        assert latest_analysis.active_count == 2
        assert latest_analysis.abstained_count == 0
        assert latest_analysis.timestamp == mock_time

    def test_get_last_analysis(self, sample_session_data):
        """Test getting the most recent analysis"""
        # No analyses initially
        assert sample_session_data.get_last_analysis() is None

        # Add an analysis
        sample_session_data.record_analysis("Test prompt", {"tech": "response"}, 1, 0)

        last_analysis = sample_session_data.get_last_analysis()
        assert last_analysis is not None
        assert last_analysis.prompt == "Test prompt"

        # Add another analysis
        sample_session_data.record_analysis(
            "Second prompt", {"tech": "response2"}, 1, 0
        )

        last_analysis = sample_session_data.get_last_analysis()
        assert last_analysis.prompt == "Second prompt"

    def test_get_thread_count(self, sample_session_data, another_thread):
        """Test getting thread count"""
        assert sample_session_data.get_thread_count() == 1

        sample_session_data.add_thread(another_thread)
        assert sample_session_data.get_thread_count() == 2

    def test_get_analysis_count(self, sample_session_data):
        """Test getting analysis count"""
        assert sample_session_data.get_analysis_count() == 0

        sample_session_data.record_analysis("Test", {}, 0, 0)
        assert sample_session_data.get_analysis_count() == 1

        sample_session_data.record_analysis("Test 2", {}, 0, 0)
        assert sample_session_data.get_analysis_count() == 2

    def test_get_thread_names(self, sample_session_data, another_thread):
        """Test getting list of thread names"""
        names = sample_session_data.get_thread_names()
        assert names == ["technical"]

        sample_session_data.add_thread(another_thread)
        names = sample_session_data.get_thread_names()
        assert set(names) == {"technical", "business"}

    def test_remove_thread_existing(self, sample_session_data, another_thread):
        """Test removing an existing thread"""
        sample_session_data.add_thread(another_thread)
        initial_count = sample_session_data.get_thread_count()

        result = sample_session_data.remove_thread("business")

        assert result is True
        assert sample_session_data.get_thread_count() == initial_count - 1
        assert sample_session_data.get_thread("business") is None

    def test_remove_thread_nonexistent(self, sample_session_data):
        """Test removing a non-existent thread"""
        initial_count = sample_session_data.get_thread_count()

        result = sample_session_data.remove_thread("nonexistent")

        assert result is False
        assert sample_session_data.get_thread_count() == initial_count

    def test_clear_analyses(self, sample_session_data):
        """Test clearing all analysis history"""
        # Add some analyses
        sample_session_data.record_analysis("Test 1", {}, 0, 0)
        sample_session_data.record_analysis("Test 2", {}, 0, 0)

        assert sample_session_data.get_analysis_count() == 2

        sample_session_data.clear_analyses()

        assert sample_session_data.get_analysis_count() == 0
        assert sample_session_data.get_last_analysis() is None

    def test_get_analyses_summary_no_analyses(self, sample_session_data):
        """Test analyses summary when no analyses exist"""
        summary = sample_session_data.get_analyses_summary()

        assert summary == {"count": 0, "message": "No analyses recorded"}

    def test_get_analyses_summary_with_analyses(self, sample_session_data):
        """Test analyses summary with existing analyses"""
        # Add multiple analyses
        sample_session_data.record_analysis(
            "Test 1", {"tech": "resp1", "business": "resp2"}, 2, 0
        )
        sample_session_data.record_analysis("Test 2", {"tech": "resp3"}, 1, 1)

        summary = sample_session_data.get_analyses_summary()

        assert summary["count"] == 2
        assert summary["total_responses"] == 3  # 2 + 1 responses

        last_analysis = summary["last_analysis"]
        assert last_analysis["active_count"] == 1
        assert last_analysis["abstained_count"] == 1
        assert "timestamp" in last_analysis

    def test_to_dict_serialization(self, sample_session_data):
        """Test conversion of SessionData to dictionary"""
        sample_session_data.record_analysis("Test analysis", {"tech": "response"}, 1, 0)

        result = sample_session_data.to_dict()

        assert result["session_id"] == "session-789"
        assert result["created_at"] == "2023-06-15T10:00:00+00:00"
        assert result["topic"] == "Test Topic"

        # Check threads serialization
        assert "technical" in result["threads"]
        thread_data = result["threads"]["technical"]
        assert thread_data["name"] == "technical"
        assert thread_data["model_backend"] == "bedrock"
        assert thread_data["system_prompt"] == "You are a technical expert"

        # Check analyses serialization
        assert len(result["analyses"]) == 1
        assert result["analyses"][0]["prompt"] == "Test analysis"

    def test_from_dict_deserialization(self, sample_thread):
        """Test creation of SessionData from dictionary"""
        data = {
            "session_id": "reconstructed-session",
            "created_at": "2023-06-15T10:00:00+00:00",
            "topic": "Reconstructed Topic",
            "threads": {
                "technical": {
                    "id": "thread-123",
                    "name": "technical",
                    "system_prompt": "You are a technical expert",
                    "model_backend": "bedrock",
                    "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "conversation_history": [{"role": "user", "content": "Hello"}],
                }
            },
            "analyses": [
                {
                    "prompt": "Test analysis",
                    "timestamp": "2023-06-15T12:30:45+00:00",
                    "responses": {"tech": "response"},
                    "active_count": 1,
                    "abstained_count": 0,
                }
            ],
        }

        session_data = SessionData.from_dict(data)

        assert session_data.session_id == "reconstructed-session"
        assert session_data.topic == "Reconstructed Topic"
        assert session_data.created_at == datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=timezone.utc
        )

        # Check thread reconstruction
        assert session_data.get_thread_count() == 1
        technical_thread = session_data.get_thread("technical")
        assert technical_thread is not None
        assert technical_thread.model_backend == ModelBackend.BEDROCK
        assert len(technical_thread.conversation_history) == 1

        # Check analysis reconstruction
        assert session_data.get_analysis_count() == 1
        analysis = session_data.get_last_analysis()
        assert analysis.prompt == "Test analysis"
        assert analysis.active_count == 1

    def test_roundtrip_serialization(self, sample_session_data, another_thread):
        """Test complete roundtrip serialization (to_dict -> from_dict)"""
        # Add more data
        sample_session_data.add_thread(another_thread)
        sample_session_data.record_analysis("First analysis", {"tech": "resp1"}, 1, 0)
        sample_session_data.record_analysis(
            "Second analysis", {"tech": "resp2", "business": "resp3"}, 2, 0
        )

        # Serialize and deserialize
        dict_data = sample_session_data.to_dict()
        reconstructed = SessionData.from_dict(dict_data)

        # Compare key attributes
        assert reconstructed.session_id == sample_session_data.session_id
        assert reconstructed.topic == sample_session_data.topic
        assert reconstructed.created_at == sample_session_data.created_at
        assert (
            reconstructed.get_thread_count() == sample_session_data.get_thread_count()
        )
        assert (
            reconstructed.get_analysis_count()
            == sample_session_data.get_analysis_count()
        )

        # Compare thread names
        assert set(reconstructed.get_thread_names()) == set(
            sample_session_data.get_thread_names()
        )

        # Compare last analysis
        original_last = sample_session_data.get_last_analysis()
        reconstructed_last = reconstructed.get_last_analysis()
        assert original_last.prompt == reconstructed_last.prompt
        assert original_last.responses == reconstructed_last.responses


class TestSessionDataEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def sample_thread(self):
        """Create a sample Thread for edge case testing"""
        return Thread(
            id="thread-123",
            name="technical",
            system_prompt="You are a technical expert",
            model_backend=ModelBackend.BEDROCK,
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        )

    @pytest.fixture
    def another_thread(self):
        """Create another sample Thread for edge case testing"""
        return Thread(
            id="thread-456",
            name="business",
            system_prompt="You are a business expert",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-4",
        )

    @pytest.fixture
    def sample_session_data(self, sample_thread):
        """Create a sample SessionData for edge case testing"""
        session = SessionData(
            session_id="test-session-123",
            topic="Test topic",
            created_at=datetime(2023, 6, 15, 10, 0, 0, tzinfo=timezone.utc),
        )
        session.add_thread(sample_thread)
        return session

    def test_from_dict_missing_optional_fields(self):
        """Test deserialization with missing optional fields"""
        minimal_data = {
            "session_id": "minimal-session",
            "created_at": "2023-06-15T10:00:00+00:00",
            # No topic, threads, or analyses
        }

        session_data = SessionData.from_dict(minimal_data)

        assert session_data.session_id == "minimal-session"
        assert session_data.topic is None
        assert session_data.get_thread_count() == 0
        assert session_data.get_analysis_count() == 0

    def test_from_dict_thread_missing_optional_fields(self):
        """Test thread deserialization with missing optional fields"""
        data = {
            "session_id": "test-session",
            "created_at": "2023-06-15T10:00:00+00:00",
            "threads": {
                "minimal": {
                    "id": "thread-minimal",
                    "name": "minimal",
                    "system_prompt": "Minimal thread",
                    "model_backend": "bedrock",
                    # No model_name or conversation_history
                }
            },
        }

        session_data = SessionData.from_dict(data)

        thread = session_data.get_thread("minimal")
        assert thread is not None
        assert thread.model_name is None
        assert thread.conversation_history == []

    def test_thread_replacement(self, sample_session_data, sample_thread):
        """Test that adding a thread with the same name replaces the existing one"""
        original_thread = sample_session_data.get_thread("technical")
        assert original_thread is not None

        # Create new thread with same name
        new_thread = Thread(
            id="new-thread-id",
            name="technical",  # Same name
            system_prompt="Updated technical prompt",
            model_backend=ModelBackend.OLLAMA,
            model_name="llama2",
        )

        sample_session_data.add_thread(new_thread)

        # Should still have only one thread, but it should be the new one
        assert sample_session_data.get_thread_count() == 1
        retrieved_thread = sample_session_data.get_thread("technical")
        assert retrieved_thread.id == "new-thread-id"
        assert retrieved_thread.system_prompt == "Updated technical prompt"
        assert retrieved_thread.model_backend == ModelBackend.OLLAMA


class TestSessionDataPerformance:
    """Test performance characteristics"""

    def test_large_number_of_analyses(self):
        """Test handling of large number of analyses"""
        session_data = SessionData(
            session_id="perf-test",
            created_at=datetime.now(timezone.utc),
        )

        # Add many analyses
        num_analyses = 1000
        for i in range(num_analyses):
            session_data.record_analysis(
                f"Analysis {i}",
                {f"perspective_{j}": f"response_{i}_{j}" for j in range(3)},
                3,
                0,
            )

        assert session_data.get_analysis_count() == num_analyses

        # Test that recent operations are still fast
        last_analysis = session_data.get_last_analysis()
        assert last_analysis.prompt == f"Analysis {num_analyses - 1}"

        summary = session_data.get_analyses_summary()
        assert summary["count"] == num_analyses
        assert summary["total_responses"] == num_analyses * 3
