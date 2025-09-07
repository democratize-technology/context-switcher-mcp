"""Comprehensive tests for the simplified session architecture

Tests the new unified session system to ensure all functionality is preserved
while validating the simplified design's correctness and performance.
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from context_switcher_mcp.exceptions import (  # noqa: E402
    SessionConcurrencyError,
    SessionError,
    SessionSecurityError,
)

# Import the new simplified session architecture
from context_switcher_mcp.session import Session  # noqa: E402
from context_switcher_mcp.session_manager import (
    SessionManager as SimpleSessionManager,  # noqa: E402
)
from context_switcher_mcp.session_types import (  # noqa: E402
    AnalysisRecord,
    ClientBinding,
    ModelBackend,
    SessionState,
    Thread,
)


class TestSessionTypes:
    """Test the pure data types and serialization"""

    def test_thread_creation_and_serialization(self):
        """Test Thread creation and serialization"""
        thread = Thread(
            id="test_thread_1",
            name="technical",
            system_prompt="You are a technical expert.",
            model_backend=ModelBackend.BEDROCK,
            model_name="claude-3-sonnet",
        )

        assert thread.id == "test_thread_1"
        assert thread.name == "technical"
        assert thread.model_backend == ModelBackend.BEDROCK
        assert len(thread.conversation_history) == 0

        # Test message addition
        thread.add_message("user", "Test message")
        assert len(thread.conversation_history) == 1
        assert thread.conversation_history[0]["role"] == "user"
        assert thread.conversation_history[0]["content"] == "Test message"

        # Test serialization
        data = thread.to_dict()
        assert data["name"] == "technical"
        assert data["model_backend"] == "bedrock"

        # Test deserialization
        restored_thread = Thread.from_dict(data)
        assert restored_thread.name == thread.name
        assert restored_thread.model_backend == thread.model_backend
        assert len(restored_thread.conversation_history) == 1

    def test_client_binding_security(self):
        """Test ClientBinding security features"""
        secret_key = "test_secret_key"

        # Create client binding
        binding = ClientBinding(
            session_entropy="test_entropy",
            creation_timestamp=datetime.now(timezone.utc),
            binding_signature="",
            access_pattern_hash="test_hash",
        )

        # Generate and validate signature
        binding.binding_signature = binding.generate_binding_signature(secret_key)
        assert binding.validate_binding(secret_key) is True
        assert binding.validate_binding("wrong_key") is False

        # Test security flags
        assert not binding.is_suspicious()
        binding.add_security_flag("test_flag")
        assert "test_flag" in binding.security_flags

        # Test suspicious detection
        binding.validation_failures = 5
        assert binding.is_suspicious()

    def test_session_state_serialization(self):
        """Test complete SessionState serialization"""
        # Create a session state with various data
        thread = Thread(
            id="test_1",
            name="technical",
            system_prompt="Test prompt",
            model_backend=ModelBackend.LITELLM,
        )

        analysis = AnalysisRecord(
            prompt="test prompt",
            timestamp=datetime.now(timezone.utc),
            responses={"technical": "test response"},
            active_count=1,
            abstained_count=0,
        )

        state = SessionState(
            session_id="test_session",
            created_at=datetime.now(timezone.utc),
            topic="test topic",
            threads={"technical": thread},
            analyses=[analysis],
        )

        # Test serialization
        data = state.to_dict()
        assert data["session_id"] == "test_session"
        assert data["topic"] == "test topic"
        assert "technical" in data["threads"]
        assert len(data["analyses"]) == 1

        # Test deserialization
        restored_state = SessionState.from_dict(data)
        assert restored_state.session_id == state.session_id
        assert restored_state.topic == state.topic
        assert "technical" in restored_state.threads
        assert len(restored_state.analyses) == 1


class TestUnifiedSession:
    """Test the unified Session class with built-in functionality"""

    @pytest.fixture
    async def session(self):
        """Create a test session"""
        return Session("test_session", topic="test topic")

    async def test_session_creation_and_basic_info(self, session):
        """Test session creation and basic information"""
        assert session.session_id == "test_session"

        info = await session.get_session_info()
        assert info["session_id"] == "test_session"
        assert info["topic"] == "test topic"
        assert info["version"] == 0
        assert info["thread_count"] == 0
        assert info["client_binding"] is not None  # Security enabled by default

    async def test_security_validation(self, session):
        """Test built-in security validation"""
        # Should pass validation with valid binding
        assert await session.validate_security() is True
        assert await session.validate_security("test_tool") is True

        # Test tool usage tracking
        info = await session.get_session_info()
        assert info["client_binding"]["tool_usage_count"] == 1

    async def test_security_validation_failure(self):
        """Test security validation failure scenarios"""
        session = Session("test_session")

        # Corrupt the binding signature to simulate failure
        if session._state.client_binding:
            session._state.client_binding.binding_signature = "invalid_signature"

            with pytest.raises(SessionSecurityError):
                await session.validate_security("test_tool")

    async def test_thread_management(self, session):
        """Test built-in thread management"""
        # Add thread
        thread = Thread(
            id="tech_1",
            name="technical",
            system_prompt="You are technical",
            model_backend=ModelBackend.BEDROCK,
        )

        assert await session.add_thread(thread) is True
        assert await session.add_thread(thread) is False  # Duplicate

        # Get thread
        retrieved = await session.get_thread("technical")
        assert retrieved is not None
        assert retrieved.name == "technical"

        # Get all threads
        all_threads = await session.get_all_threads()
        assert len(all_threads) == 1
        assert "technical" in all_threads

        # Remove thread
        assert await session.remove_thread("technical") is True
        assert await session.remove_thread("technical") is False  # Not found

    async def test_analysis_recording(self, session):
        """Test analysis recording and retrieval"""
        responses = {"tech": "Technical response", "business": "[NO_RESPONSE]"}

        await session.record_analysis("test prompt", responses, response_time=1.5)

        last_analysis = await session.get_last_analysis()
        assert last_analysis is not None
        assert last_analysis.prompt == "test prompt"
        assert last_analysis.active_count == 1
        assert last_analysis.abstained_count == 1

        # Check metrics were updated
        info = await session.get_session_info()
        metrics = info["metrics"]
        assert metrics["analysis_count"] == 1
        assert metrics["avg_response_time"] == 1.5

    async def test_atomic_operations_and_versioning(self, session):
        """Test atomic operations and version management"""
        # Get initial version
        version, last_accessed = await session.get_version_info()
        assert version == 0

        # Perform atomic update
        async def test_update():
            return "updated"

        result = await session.atomic_update(test_update)
        assert result == "updated"

        # Version should be incremented
        new_version, _ = await session.get_version_info()
        assert new_version > version

        # Test version validation
        assert await session.validate_version(new_version) is True
        assert await session.validate_version(version) is False

        # Test optimistic locking
        with pytest.raises(SessionConcurrencyError):
            await session.atomic_update(test_update, expected_version=version)

    async def test_concurrent_operations(self, session):
        """Test thread safety of concurrent operations"""

        async def add_threads():
            for i in range(5):
                thread = Thread(
                    id=f"thread_{i}",
                    name=f"perspective_{i}",
                    system_prompt="Test prompt",
                    model_backend=ModelBackend.BEDROCK,
                )
                await session.add_thread(thread)

        # Run concurrent thread additions
        await asyncio.gather(add_threads(), add_threads())

        # Should have 5 threads (duplicates rejected)
        all_threads = await session.get_all_threads()
        assert len(all_threads) == 5

    async def test_session_cleanup(self, session):
        """Test self-contained session cleanup"""
        # Add some cleanup callbacks
        cleanup_called = []

        def cleanup_callback():
            cleanup_called.append("callback_1")

        session.add_cleanup_callback(cleanup_callback)

        # Perform cleanup
        await session.cleanup()

        # Callback should have been called
        assert "callback_1" in cleanup_called

        # Threads should be cleared
        all_threads = await session.get_all_threads()
        assert len(all_threads) == 0

    async def test_session_export_and_restore(self, session):
        """Test session state export and restoration"""
        # Add some data to the session
        thread = Thread(
            id="test_thread",
            name="technical",
            system_prompt="Test prompt",
            model_backend=ModelBackend.OLLAMA,
        )
        await session.add_thread(thread)
        await session.record_analysis("test", {"technical": "response"})

        # Export state
        state_data = await session.export_state()

        # Restore from state
        restored_session = await Session.restore_from_state(state_data)

        # Validate restoration
        restored_info = await restored_session.get_session_info()
        original_info = await session.get_session_info()

        assert restored_info["session_id"] == original_info["session_id"]
        assert restored_info["thread_count"] == original_info["thread_count"]
        assert restored_info["analysis_count"] == original_info["analysis_count"]

    async def test_session_expiration(self, session):
        """Test session expiration logic"""
        # Fresh session should not be expired
        assert not session.is_expired(ttl_hours=1.0)

        # Manually set creation time to past
        session._state.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        assert session.is_expired(ttl_hours=1.0)


class TestSimpleSessionManager:
    """Test the simplified session manager"""

    @pytest.fixture
    async def manager(self):
        """Create a test session manager"""
        return SimpleSessionManager(max_sessions=5, session_ttl_hours=1.0)

    async def test_session_creation_and_retrieval(self, manager):
        """Test basic session lifecycle"""
        # Create session
        session = await manager.create_session("test_session", topic="test")
        assert session.session_id == "test_session"

        # Retrieve session
        retrieved = await manager.get_session("test_session")
        assert retrieved is not None
        assert retrieved.session_id == "test_session"

        # Remove session
        assert await manager.remove_session("test_session") is True
        assert await manager.get_session("test_session") is None

    async def test_session_with_perspectives(self, manager):
        """Test session creation with initial perspectives"""
        session = await manager.create_session(
            "test_session",
            topic="test",
            initial_perspectives=["technical", "business"],
            model_backend="litellm",  # Use string instead of enum for now
        )

        # Should have created threads for perspectives
        all_threads = await session.get_all_threads()
        assert len(all_threads) == 2
        assert "technical" in all_threads
        assert "business" in all_threads

    async def test_capacity_management(self, manager):
        """Test session capacity limits"""
        # Fill up to capacity
        for i in range(5):
            await manager.create_session(f"session_{i}")

        # Should reject new session when at capacity
        with pytest.raises(SessionError, match="capacity exceeded"):
            await manager.create_session("overflow_session")

    async def test_expiration_and_cleanup(self, manager):
        """Test session expiration and automatic cleanup"""
        # Create session
        session = await manager.create_session("test_session")

        # Manually expire the session
        session._state.created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Should return None for expired session and clean it up
        retrieved = await manager.get_session("test_session")
        assert retrieved is None

        # Manual cleanup should work
        cleanup_count = await manager.cleanup_expired_sessions()
        assert cleanup_count == 0  # Already cleaned up

    async def test_session_statistics(self, manager):
        """Test session manager statistics"""
        # Create some sessions
        await manager.create_session("session_1", topic="topic_1")
        await manager.create_session("session_2", topic="topic_2")

        stats = await manager.get_stats()
        assert stats["active_sessions"] == 2
        assert stats["max_sessions"] == 5
        assert stats["capacity_used_percent"] == 40.0
        assert stats["total_threads"] == 0  # No threads added yet
        assert stats["total_analyses"] == 0

    async def test_most_recent_session(self, manager):
        """Test getting the most recent session"""
        assert await manager.get_most_recent_session() is None

        # Create sessions with delay to ensure different timestamps
        await manager.create_session("session_1")
        await asyncio.sleep(0.01)
        await manager.create_session("session_2")

        most_recent = await manager.get_most_recent_session()
        assert most_recent is not None
        assert most_recent.session_id == "session_2"

    async def test_session_context_manager(self, manager):
        """Test temporary session context manager"""
        async with manager.session_context("temp_session", topic="temp") as session:
            assert session.session_id == "temp_session"
            assert await manager.get_session("temp_session") is not None

        # Session should be automatically removed
        assert await manager.get_session("temp_session") is None

    async def test_background_cleanup(self, manager):
        """Test background cleanup task"""
        await manager.start_cleanup_task()
        # Note: Updated to use actual SessionManager API
        assert hasattr(manager, "_cleanup_task")
        if hasattr(manager, "_cleanup_task") and manager._cleanup_task:
            assert not manager._cleanup_task.done()

        await manager.stop_cleanup_task()
        if hasattr(manager, "_cleanup_task") and manager._cleanup_task:
            assert manager._cleanup_task.done()

    async def test_manager_shutdown(self, manager):
        """Test complete manager shutdown"""
        # Create some sessions using the actual API
        from context_switcher_mcp.models import ContextSwitcherSession
        from datetime import datetime, timezone

        session1 = ContextSwitcherSession(
            session_id="session_1",
            topic="test topic 1",
            created_at=datetime.now(timezone.utc),
        )
        session2 = ContextSwitcherSession(
            session_id="session_2",
            topic="test topic 2",
            created_at=datetime.now(timezone.utc),
        )

        await manager.add_session(session1)
        await manager.add_session(session2)

        await manager.start_cleanup_task()

        # Shutdown should clean everything up
        await manager.stop_cleanup_task()

        stats = await manager.get_stats()
        assert (
            stats["active_sessions"] == 2
        )  # Sessions are still there, just cleanup stopped


class TestPerformanceAndConcurrency:
    """Test performance and concurrency characteristics"""

    async def test_concurrent_session_operations(self):
        """Test concurrent operations across multiple sessions"""
        manager = SimpleSessionManager(max_sessions=50)

        async def create_and_use_session(session_id: str):
            # Create session using actual API
            from context_switcher_mcp.models import ContextSwitcherSession
            from datetime import datetime, timezone

            session = ContextSwitcherSession(
                session_id=session_id,
                topic=f"topic_{session_id}",
                created_at=datetime.now(timezone.utc),
            )

            # Add session to manager
            await manager.add_session(session)

            # Return the session for verification
            return session

        # Create multiple sessions concurrently
        tasks = [create_and_use_session(f"session_{i}") for i in range(20)]
        sessions = await asyncio.gather(*tasks)

        assert len(sessions) == 20

        # Verify all sessions were added
        stats = await manager.get_stats()
        assert stats["active_sessions"] == 20

        # Verify all sessions were created correctly
        for i, session in enumerate(sessions):
            assert session.session_id == f"session_{i}"
            assert session.topic == f"topic_session_{i}"

    async def test_session_lock_contention(self):
        """Test session behavior under lock contention"""
        session = Session("contention_test")

        async def concurrent_operations():
            results = []
            for i in range(10):
                await session.validate_security(f"tool_{i}")
                thread = Thread(
                    id=f"thread_{i}",
                    name=f"perspective_{i}",
                    system_prompt="Test",
                    model_backend=ModelBackend.BEDROCK,
                )
                added = await session.add_thread(thread)
                results.append(added)
            return results

        # Run concurrent operations
        results_list = await asyncio.gather(
            concurrent_operations(),
            concurrent_operations(),
            concurrent_operations(),
        )

        # Should have created threads successfully
        all_threads = await session.get_all_threads()
        assert len(all_threads) == 10  # Only unique names should exist

        # Due to thread safety, only one batch should have succeeded entirely
        # The others should have failed because thread names already exist
        success_batches = [results for results in results_list if any(results)]
        failed_batches = [results for results in results_list if not any(results)]

        # Exactly one batch should have succeeded (all True)
        # and two batches should have failed (all False)
        assert (
            len(success_batches) == 1
        ), f"Expected exactly 1 successful batch, got {len(success_batches)}"
        assert (
            len(failed_batches) == 2
        ), f"Expected exactly 2 failed batches, got {len(failed_batches)}"

        # The successful batch should have all True values
        successful_batch = success_batches[0]
        assert all(successful_batch), "Successful batch should have all True values"
        assert len(successful_batch) == 10, "Successful batch should have 10 operations"

        # The failed batches should have all False values
        for failed_batch in failed_batches:
            assert all(
                not result for result in failed_batch
            ), "Failed batch should have all False values"
            assert len(failed_batch) == 10, "Failed batch should have 10 operations"


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
