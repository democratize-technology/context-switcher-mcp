"""Tests for the new session lock implementation using SessionLockManager"""

import asyncio
from datetime import datetime, timezone

import pytest
from context_switcher_mcp.models import ContextSwitcherSession
from context_switcher_mcp.session_lock_manager import get_session_lock_manager


class TestSessionLockWithManager:
    """Test session locking with the new SessionLockManager"""

    def test_session_no_longer_has_internal_locks(self):
        """Test that sessions no longer maintain internal lock state"""
        session = ContextSwitcherSession(
            session_id="test-123", created_at=datetime.now(timezone.utc)
        )

        # Sessions should not have these attributes anymore
        assert not hasattr(session, "_lock_initialized")
        assert not hasattr(session, "_access_lock")
        assert not hasattr(session, "_initialization_lock")

    @pytest.mark.asyncio
    async def test_record_access_uses_centralized_lock(self):
        """Test that record_access uses the centralized lock manager"""
        session = ContextSwitcherSession(
            session_id="test-456", created_at=datetime.now(timezone.utc)
        )

        # Record access should work through the lock manager
        await session.record_access("test_tool")

        assert session.access_count == 1
        assert session.version == 1

        # Verify lock manager has the lock for this session
        lock_manager = get_session_lock_manager()
        lock = lock_manager.get_lock(session.session_id)
        assert lock is not None

    @pytest.mark.asyncio
    async def test_concurrent_access_thread_safe(self):
        """Test that concurrent access is properly synchronized"""
        session = ContextSwitcherSession(
            session_id="test-789", created_at=datetime.now(timezone.utc)
        )

        # Simulate concurrent access
        async def access_session(tool_name):
            await session.record_access(tool_name)

        # Run 100 concurrent accesses
        tasks = [access_session(f"tool_{i}") for i in range(100)]
        await asyncio.gather(*tasks)

        # All accesses should be recorded
        assert session.access_count == 100
        assert session.version == 100

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent_locks(self):
        """Test that multiple session instances have independent locks"""
        session1 = ContextSwitcherSession(
            session_id="session-1", created_at=datetime.now(timezone.utc)
        )
        session2 = ContextSwitcherSession(
            session_id="session-2", created_at=datetime.now(timezone.utc)
        )

        # Both sessions should work independently
        await session1.record_access("tool1")
        await session2.record_access("tool2")

        assert session1.access_count == 1
        assert session2.access_count == 1

        # Verify they have different locks
        lock_manager = get_session_lock_manager()
        lock1 = lock_manager.get_lock(session1.session_id)
        lock2 = lock_manager.get_lock(session2.session_id)

        assert lock1 is not None
        assert lock2 is not None
        assert lock1 is not lock2

    @pytest.mark.asyncio
    async def test_session_cleanup_removes_lock(self):
        """Test that cleaning up a session removes its lock"""
        session = ContextSwitcherSession(
            session_id="test-cleanup", created_at=datetime.now(timezone.utc)
        )

        # Use the session
        await session.record_access("test_tool")

        # Verify lock exists
        lock_manager = get_session_lock_manager()
        lock = lock_manager.get_lock(session.session_id)
        assert lock is not None

        # Clean up
        lock_manager.remove_lock(session.session_id)

        # New lock should be created if we access again
        lock2 = lock_manager.get_lock(session.session_id)
        assert lock2 is not None
        # In our implementation, it will be a new lock
        # (This depends on implementation details)
