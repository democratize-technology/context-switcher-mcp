"""Comprehensive tests for session concurrency management"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from context_switcher_mcp.session_concurrency import (
    SessionConcurrency,
    create_session_concurrency,
)


class TestSessionConcurrency:
    """Test suite for SessionConcurrency class"""

    @pytest.fixture
    def session_concurrency(self):
        """Create a SessionConcurrency instance for testing"""
        return SessionConcurrency("test-session-123")

    @pytest.fixture
    def mock_lock_manager(self):
        """Create mock lock manager"""
        mock_lock = AsyncMock()
        mock_lock.__aenter__ = AsyncMock(return_value=mock_lock)
        mock_lock.__aexit__ = AsyncMock(return_value=None)

        mock_manager = Mock()
        mock_manager.acquire_lock.return_value = mock_lock
        mock_manager.remove_lock = Mock()

        return mock_manager

    def test_initialization(self, session_concurrency):
        """Test SessionConcurrency initialization"""
        assert session_concurrency.session_id == "test-session-123"
        assert session_concurrency.access_count == 0
        assert session_concurrency.version == 0
        assert isinstance(session_concurrency.last_accessed, datetime)
        assert session_concurrency.last_accessed.tzinfo is not None

    def test_get_lock_manager_imports_correctly(self, session_concurrency):
        """Test that _get_lock_manager imports the lock manager correctly"""
        with patch(
            "context_switcher_mcp.session_lock_manager.get_session_lock_manager"
        ) as mock_get:
            mock_manager = Mock()
            mock_get.return_value = mock_manager

            result = session_concurrency._get_lock_manager()

            mock_get.assert_called_once()
            assert result == mock_manager

    @pytest.mark.asyncio
    async def test_record_access_basic(self, session_concurrency, mock_lock_manager):
        """Test basic access recording functionality"""
        with patch.object(
            session_concurrency, "_get_lock_manager", return_value=mock_lock_manager
        ):
            initial_count = session_concurrency.access_count
            initial_version = session_concurrency.version
            initial_time = session_concurrency.last_accessed

            await session_concurrency.record_access()

            assert session_concurrency.access_count == initial_count + 1
            assert session_concurrency.version == initial_version + 1
            assert session_concurrency.last_accessed > initial_time
            mock_lock_manager.acquire_lock.assert_called_once_with("test-session-123")

    @pytest.mark.asyncio
    async def test_record_access_with_tool_name(
        self, session_concurrency, mock_lock_manager
    ):
        """Test access recording with tool name logging"""
        with patch.object(
            session_concurrency, "_get_lock_manager", return_value=mock_lock_manager
        ):
            with patch(
                "context_switcher_mcp.session_concurrency.logger"
            ) as mock_logger:
                await session_concurrency.record_access("test_tool")

                mock_logger.debug.assert_called_once_with(
                    "Session test-session-123 accessed tool: test_tool"
                )

    @pytest.mark.asyncio
    async def test_record_access_concurrent(
        self, session_concurrency, mock_lock_manager
    ):
        """Test concurrent access recording with proper locking"""
        with patch.object(
            session_concurrency, "_get_lock_manager", return_value=mock_lock_manager
        ):
            # Start multiple concurrent access recordings
            tasks = [session_concurrency.record_access(f"tool_{i}") for i in range(5)]

            await asyncio.gather(*tasks)

            # Should have recorded all 5 accesses
            assert session_concurrency.access_count == 5
            assert session_concurrency.version == 5
            # Each call should acquire the lock
            assert mock_lock_manager.acquire_lock.call_count == 5

    def test_check_version_conflict(self, session_concurrency):
        """Test version conflict detection"""
        # No conflict with matching version
        assert not session_concurrency.check_version_conflict(0)

        # Set version to 5
        session_concurrency.version = 5

        # No conflict with matching version
        assert not session_concurrency.check_version_conflict(5)

        # Conflict with different version
        assert session_concurrency.check_version_conflict(4)
        assert session_concurrency.check_version_conflict(6)

    def test_get_access_stats(self, session_concurrency):
        """Test access statistics retrieval"""
        # Set some test values
        session_concurrency.access_count = 10
        session_concurrency.version = 15
        test_time = datetime(2023, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
        session_concurrency.last_accessed = test_time

        stats = session_concurrency.get_access_stats()

        assert stats["session_id"] == "test-session-123"
        assert stats["access_count"] == 10
        assert stats["version"] == 15
        assert stats["last_accessed"] == test_time.isoformat()

    @pytest.mark.asyncio
    async def test_synchronized_update_sync_function(
        self, session_concurrency, mock_lock_manager
    ):
        """Test synchronized update with synchronous function"""
        with patch.object(
            session_concurrency, "_get_lock_manager", return_value=mock_lock_manager
        ):
            test_value = {"counter": 0}

            def update_func():
                test_value["counter"] += 1

            initial_version = session_concurrency.version
            initial_time = session_concurrency.last_accessed

            await session_concurrency.synchronized_update(update_func)

            # Function should have been called
            assert test_value["counter"] == 1

            # Metadata should be updated
            assert session_concurrency.version == initial_version + 1
            assert session_concurrency.last_accessed > initial_time

            # Lock should have been acquired
            mock_lock_manager.acquire_lock.assert_called_once_with("test-session-123")

    @pytest.mark.asyncio
    async def test_synchronized_update_async_function(
        self, session_concurrency, mock_lock_manager
    ):
        """Test synchronized update with asynchronous function"""
        with patch.object(
            session_concurrency, "_get_lock_manager", return_value=mock_lock_manager
        ):
            test_value = {"counter": 0}

            async def async_update_func():
                test_value["counter"] += 1
                await asyncio.sleep(0.01)  # Small async operation

            await session_concurrency.synchronized_update(async_update_func)

            # Function should have been called
            assert test_value["counter"] == 1

            # Version should be incremented
            assert session_concurrency.version == 1

    @pytest.mark.asyncio
    async def test_synchronized_update_with_version_check(
        self, session_concurrency, mock_lock_manager
    ):
        """Test synchronized update with version checking"""
        with patch.object(
            session_concurrency, "_get_lock_manager", return_value=mock_lock_manager
        ):
            session_concurrency.version = 5

            def update_func():
                pass

            # Should succeed with correct expected version
            await session_concurrency.synchronized_update(
                update_func, expected_version=5
            )

            # Version should be incremented after successful update
            assert session_concurrency.version == 6

    @pytest.mark.asyncio
    async def test_synchronized_update_version_conflict(
        self, session_concurrency, mock_lock_manager
    ):
        """Test synchronized update with version conflict"""
        with patch.object(
            session_concurrency, "_get_lock_manager", return_value=mock_lock_manager
        ):
            session_concurrency.version = 5

            def update_func():
                pass

            # Should raise ValueError with incorrect expected version
            with pytest.raises(ValueError, match="Version conflict: expected 3, got 5"):
                await session_concurrency.synchronized_update(
                    update_func, expected_version=3
                )

            # Version should not be incremented after failed update
            assert session_concurrency.version == 5

    def test_cleanup(self, session_concurrency, mock_lock_manager):
        """Test cleanup of concurrency resources"""
        with patch.object(
            session_concurrency, "_get_lock_manager", return_value=mock_lock_manager
        ):
            with patch(
                "context_switcher_mcp.session_concurrency.logger"
            ) as mock_logger:
                session_concurrency.cleanup()

                # Should remove lock and log cleanup
                mock_lock_manager.remove_lock.assert_called_once_with(
                    "test-session-123"
                )
                mock_logger.debug.assert_called_once_with(
                    "Cleaned up concurrency resources for session test-session-123"
                )


class TestSessionConcurrencyIntegration:
    """Integration tests for SessionConcurrency"""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_operations(self):
        """Test multiple concurrent operations on the same session"""
        session = SessionConcurrency("integration-test")

        # Mock lock manager for this test
        mock_lock = AsyncMock()
        mock_lock.__aenter__ = AsyncMock(return_value=mock_lock)
        mock_lock.__aexit__ = AsyncMock(return_value=None)

        mock_manager = Mock()
        mock_manager.acquire_lock.return_value = mock_lock
        mock_manager.remove_lock = Mock()

        with patch.object(session, "_get_lock_manager", return_value=mock_manager):
            # Perform multiple operations concurrently
            operations = []

            # Add some access recordings
            for i in range(3):
                operations.append(session.record_access(f"tool_{i}"))

            # Add some synchronized updates
            test_counters = {"sync": 0, "async": 0}

            def sync_update():
                test_counters["sync"] += 1

            async def async_update():
                test_counters["async"] += 1

            operations.extend(
                [
                    session.synchronized_update(sync_update),
                    session.synchronized_update(async_update),
                ]
            )

            # Execute all operations concurrently
            await asyncio.gather(*operations)

            # Verify final state
            assert session.access_count == 3  # Only record_access calls increment this
            assert session.version == 5  # All operations increment version
            assert test_counters["sync"] == 1
            assert test_counters["async"] == 1

            # Verify all operations acquired the lock
            assert mock_manager.acquire_lock.call_count == 5


class TestCreateSessionConcurrency:
    """Test the factory function"""

    def test_create_session_concurrency(self):
        """Test the factory function creates a proper SessionConcurrency instance"""
        session_id = "factory-test-session"
        session = create_session_concurrency(session_id)

        assert isinstance(session, SessionConcurrency)
        assert session.session_id == session_id
        assert session.access_count == 0
        assert session.version == 0


class TestSessionConcurrencyErrorHandling:
    """Test error handling scenarios"""

    @pytest.mark.asyncio
    async def test_lock_manager_error_propagation(self):
        """Test that lock manager errors are properly propagated"""
        session = SessionConcurrency("error-test")

        mock_manager = Mock()
        mock_manager.acquire_lock.side_effect = RuntimeError("Lock acquisition failed")

        with patch.object(session, "_get_lock_manager", return_value=mock_manager):
            # Should propagate the error from lock manager
            with pytest.raises(RuntimeError, match="Lock acquisition failed"):
                await session.record_access()

    @pytest.mark.asyncio
    async def test_update_function_error_propagation(self):
        """Test that errors in update functions are properly propagated"""
        session = SessionConcurrency("error-test")

        mock_lock = AsyncMock()
        mock_lock.__aenter__ = AsyncMock(return_value=mock_lock)
        mock_lock.__aexit__ = AsyncMock(return_value=None)

        mock_manager = Mock()
        mock_manager.acquire_lock.return_value = mock_lock

        with patch.object(session, "_get_lock_manager", return_value=mock_manager):

            def failing_update():
                raise ValueError("Update function failed")

            # Should propagate the error from update function
            with pytest.raises(ValueError, match="Update function failed"):
                await session.synchronized_update(failing_update)

            # Version should not be incremented on failure
            assert session.version == 0


class TestSessionConcurrencyThreadSafety:
    """Test thread safety aspects"""

    @pytest.mark.asyncio
    async def test_version_consistency_under_concurrent_access(self):
        """Test that version numbers remain consistent under concurrent access"""
        session = SessionConcurrency("thread-safety-test")

        # Create a real lock to test actual synchronization
        real_lock = asyncio.Lock()
        mock_manager = Mock()
        mock_manager.acquire_lock.return_value = real_lock
        mock_manager.remove_lock = Mock()

        with patch.object(session, "_get_lock_manager", return_value=mock_manager):
            # Create many concurrent operations
            num_operations = 20
            operations = [
                session.record_access(f"concurrent_tool_{i}")
                for i in range(num_operations)
            ]

            # Execute all operations concurrently
            await asyncio.gather(*operations)

            # Final version should equal number of operations
            assert session.version == num_operations
            assert session.access_count == num_operations
