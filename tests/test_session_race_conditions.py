"""Tests for session manager race condition fixes"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from context_switcher_mcp.models import ContextSwitcherSession
from context_switcher_mcp.session_manager import SessionManager


@pytest.fixture
def session_manager():
    """Create a session manager for testing"""
    return SessionManager(max_sessions=10, session_ttl_hours=1)


@pytest.fixture
def test_session():
    """Create a test session"""
    session = ContextSwitcherSession(
        session_id="test-session-123",
        created_at=datetime.now(timezone.utc),
        topic="Test Session",
    )
    return session


@pytest.fixture
def expired_session():
    """Create an expired test session"""
    session = ContextSwitcherSession(
        session_id="expired-session-123",
        created_at=datetime.now(timezone.utc) - timedelta(hours=2),  # Expired
        topic="Expired Session",
    )
    return session


class TestRaceConditionFixes:
    """Test race condition fixes in session management"""

    @pytest.mark.asyncio
    async def test_concurrent_session_access(self, session_manager, test_session):
        """Test concurrent access to the same session"""
        await session_manager.add_session(test_session)

        async def access_session():
            return await session_manager.get_session(test_session.session_id)

        # Run multiple concurrent accesses
        tasks = [access_session() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed and return the same session
        assert all(result is not None for result in results)
        assert all(result.session_id == test_session.session_id for result in results)

    @pytest.mark.asyncio
    async def test_concurrent_expiration_handling(
        self, session_manager, expired_session
    ):
        """Test concurrent access to expired sessions"""
        await session_manager.add_session(expired_session)

        async def access_expired_session():
            return await session_manager.get_session(expired_session.session_id)

        # Run multiple concurrent accesses to expired session
        tasks = [access_expired_session() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should return None (expired)
        assert all(result is None for result in results)

        # Session should be completely removed
        assert expired_session.session_id not in session_manager.sessions

    @pytest.mark.asyncio
    async def test_atomic_session_removal(self, session_manager, test_session):
        """Test atomic session removal prevents double-deletion races"""
        await session_manager.add_session(test_session)

        async def remove_session():
            return await session_manager.remove_session(test_session.session_id)

        # Run multiple concurrent removals
        tasks = [remove_session() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # Only one should succeed (return True), others should return False
        true_count = sum(1 for result in results if result)
        false_count = sum(1 for result in results if not result)

        assert true_count == 1  # Only one successful removal
        assert false_count == 4  # Four should fail (already removed)

    @pytest.mark.asyncio
    async def test_session_version_tracking(self, session_manager, test_session):
        """Test that session versions are properly tracked"""
        await session_manager.add_session(test_session)

        initial_version = test_session.version

        # Record access should increment version
        await test_session.record_access("test_tool")

        assert test_session.version == initial_version + 1

    @pytest.mark.asyncio
    async def test_concurrent_session_access_recording(
        self, session_manager, test_session
    ):
        """Test concurrent access recording is thread-safe"""
        await session_manager.add_session(test_session)
        initial_access_count = test_session.access_count

        async def record_access():
            await test_session.record_access("test_tool")

        # Run multiple concurrent access recordings
        num_accesses = 20
        tasks = [record_access() for _ in range(num_accesses)]
        await asyncio.gather(*tasks)

        # Access count should be exactly initial + num_accesses
        assert test_session.access_count == initial_access_count + num_accesses

    @pytest.mark.asyncio
    async def test_atomic_get_with_version(self, session_manager, test_session):
        """Test atomic get with version for optimistic locking"""
        await session_manager.add_session(test_session)

        session, version = await session_manager.get_session_atomic(
            test_session.session_id
        )

        assert session is not None
        assert session.session_id == test_session.session_id
        assert version == test_session.version

    @pytest.mark.asyncio
    async def test_atomic_get_expired_session(self, session_manager, expired_session):
        """Test atomic get properly handles expired sessions"""
        await session_manager.add_session(expired_session)

        session, version = await session_manager.get_session_atomic(
            expired_session.session_id
        )

        assert session is None
        assert version == -1
        assert expired_session.session_id not in session_manager.sessions

    @pytest.mark.asyncio
    async def test_version_validation(self, session_manager, test_session):
        """Test session version validation"""
        await session_manager.add_session(test_session)

        current_version = test_session.version

        # Validation should succeed with correct version
        assert await session_manager.validate_session_version(
            test_session.session_id, current_version
        )

        # Validation should fail with wrong version
        assert not await session_manager.validate_session_version(
            test_session.session_id, current_version + 999
        )

    @pytest.mark.asyncio
    async def test_cleanup_with_concurrent_access(self, session_manager):
        """Test cleanup doesn't interfere with concurrent access"""
        # Add a mix of regular and expired sessions
        regular_session = ContextSwitcherSession(
            session_id="regular-session",
            created_at=datetime.now(timezone.utc),
            topic="Regular Session",
        )
        expired_session = ContextSwitcherSession(
            session_id="expired-session",
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            topic="Expired Session",
        )

        await session_manager.add_session(regular_session)
        await session_manager.add_session(expired_session)

        async def access_regular_session():
            return await session_manager.get_session("regular-session")

        async def cleanup_sessions():
            await session_manager._cleanup_expired_sessions()

        # Run cleanup concurrently with access
        access_task = asyncio.create_task(access_regular_session())
        cleanup_task = asyncio.create_task(cleanup_sessions())

        session_result, _ = await asyncio.gather(access_task, cleanup_task)

        # Regular session should still be accessible
        assert session_result is not None
        assert session_result.session_id == "regular-session"

        # Expired session should be removed
        assert "expired-session" not in session_manager.sessions

    @pytest.mark.asyncio
    async def test_resource_cleanup_called(self, session_manager, test_session):
        """Test that resource cleanup is properly called"""
        with patch.object(
            session_manager, "_cleanup_session_resources"
        ) as mock_cleanup:
            await session_manager.add_session(test_session)
            await session_manager.remove_session(test_session.session_id)

            mock_cleanup.assert_called_once_with(test_session.session_id)

    @pytest.mark.asyncio
    async def test_resource_cleanup_exception_handling(
        self, session_manager, test_session
    ):
        """Test that resource cleanup exceptions don't break session operations"""
        with patch.object(
            session_manager, "_cleanup_session_resources"
        ) as mock_cleanup:
            mock_cleanup.side_effect = Exception("Cleanup failed")

            await session_manager.add_session(test_session)
            result = await session_manager.remove_session(test_session.session_id)

            # Session should still be removed despite cleanup failure
            assert result is True
            assert test_session.session_id not in session_manager.sessions

    @pytest.mark.asyncio
    async def test_concurrent_add_and_cleanup(self, session_manager):
        """Test concurrent session addition and cleanup operations"""
        sessions = []
        for i in range(10):
            session = ContextSwitcherSession(
                session_id=f"session-{i}",
                created_at=datetime.now(timezone.utc)
                if i % 2 == 0
                else datetime.now(timezone.utc) - timedelta(hours=2),
                topic=f"Session {i}",
            )
            sessions.append(session)

        async def add_sessions():
            for session in sessions:
                await session_manager.add_session(session)

        async def cleanup_sessions():
            await asyncio.sleep(0.01)  # Small delay to allow some additions
            async with session_manager._lock:
                await session_manager._cleanup_expired_sessions()

        # Run addition and cleanup concurrently
        await asyncio.gather(add_sessions(), cleanup_sessions())

        # Only non-expired sessions should remain
        remaining_sessions = await session_manager.list_active_sessions()
        assert len(remaining_sessions) == 5  # Half were expired

        for session_id in remaining_sessions:
            assert (
                int(session_id.split("-")[1]) % 2 == 0
            )  # Only even-numbered (non-expired)
