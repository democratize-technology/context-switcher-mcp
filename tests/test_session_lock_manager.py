"""Tests for the centralized session lock manager"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from context_switcher_mcp.session_lock_manager import (  # noqa: E402
    SessionLockManager,
    get_session_lock_manager,
)


class TestSessionLockManager:
    """Test the SessionLockManager class"""

    @pytest.mark.asyncio
    async def test_get_lock_creates_lock_in_async_context(self):
        """Test that get_lock creates a lock when in async context"""
        manager = SessionLockManager()
        session_id = "test-session-1"

        # In async context, should create lock
        lock = manager.get_lock(session_id)
        assert lock is not None
        assert isinstance(lock, asyncio.Lock)

        # Getting again should return same lock
        lock2 = manager.get_lock(session_id)
        assert lock2 is lock

    def test_get_lock_returns_none_without_event_loop(self):
        """Test that get_lock returns None when no event loop exists"""
        manager = SessionLockManager()
        session_id = "test-session-2"

        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            lock = manager.get_lock(session_id)
            assert lock is None

    @pytest.mark.asyncio
    async def test_acquire_lock_context_manager(self):
        """Test the acquire_lock context manager"""
        manager = SessionLockManager()
        session_id = "test-session-3"

        # Test that we can acquire and release lock
        async with manager.acquire_lock(session_id):
            # Lock should be held here
            lock = manager.get_lock(session_id)
            assert lock is not None
            assert lock.locked()

        # Lock should be released after context
        lock = manager.get_lock(session_id)
        assert not lock.locked()

    @pytest.mark.asyncio
    async def test_multiple_sessions_have_independent_locks(self):
        """Test that different sessions have independent locks"""
        manager = SessionLockManager()
        session1 = "test-session-4"
        session2 = "test-session-5"

        lock1 = manager.get_lock(session1)
        lock2 = manager.get_lock(session2)

        assert lock1 is not None
        assert lock2 is not None
        assert lock1 is not lock2

    @pytest.mark.asyncio
    async def test_concurrent_access_to_same_session(self):
        """Test that concurrent access to same session is properly synchronized"""
        manager = SessionLockManager()
        session_id = "test-session-6"
        counter = 0

        async def increment():
            nonlocal counter
            async with manager.acquire_lock(session_id):
                temp = counter
                await asyncio.sleep(0.01)  # Simulate work
                counter = temp + 1

        # Run multiple increments concurrently
        tasks = [increment() for _ in range(10)]
        await asyncio.gather(*tasks)

        # All increments should have been applied
        assert counter == 10

    def test_remove_lock(self):
        """Test that remove_lock removes the lock for a session"""
        manager = SessionLockManager()
        session_id = "test-session-7"

        # Create a lock
        with patch("asyncio.get_running_loop", return_value=MagicMock()):
            lock1 = manager.get_lock(session_id)
            assert lock1 is not None

            # Remove it
            manager.remove_lock(session_id)

            # Getting again should create a new lock
            lock2 = manager.get_lock(session_id)
            assert lock2 is not None
            assert lock2 is not lock1

    def test_clear_all(self):
        """Test that clear_all removes all locks"""
        manager = SessionLockManager()

        with patch("asyncio.get_running_loop", return_value=MagicMock()):
            # Create multiple locks
            for i in range(5):
                manager.get_lock(f"session-{i}")

            # Clear all
            manager.clear_all()

            # All locks should be gone (would create new ones)
            for i in range(5):
                lock = manager.get_lock(f"session-{i}")
                assert lock is not None  # New lock created

    def test_get_session_lock_manager_returns_singleton(self):
        """Test that get_session_lock_manager returns the same instance"""
        manager1 = get_session_lock_manager()
        manager2 = get_session_lock_manager()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_lock_context_manager_without_lock(self):
        """Test that context manager works even when no lock is available"""
        manager = SessionLockManager()
        session_id = "test-session-8"

        # Patch to make get_lock return None
        with patch.object(manager, "get_lock", return_value=None):
            # Should not raise an error
            async with manager.acquire_lock(session_id):
                pass  # Should work fine without a lock
