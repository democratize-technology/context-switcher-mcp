"""Test suite for critical security and reliability fixes"""

import asyncio
import json
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest
import concurrent.futures

from context_switcher_mcp.circuit_breaker_store import CircuitBreakerStore
from context_switcher_mcp.models import ContextSwitcherSession
from context_switcher_mcp.session_manager import SessionManager
from context_switcher_mcp.exceptions import SessionCleanupError


class TestCircuitBreakerPathValidation:
    """Test simplified path validation in CircuitBreakerStore"""

    def test_default_path_is_allowed(self):
        """Test that default path is properly created"""
        store = CircuitBreakerStore()
        assert store.storage_path.parent.name == ".context_switcher"
        assert store.storage_path.name == "circuit_breakers.json"

    def test_allowed_paths_accepted(self):
        """Test that paths within allowed directories are accepted"""
        # Test home directory path
        home_path = Path.home() / ".context_switcher" / "test.json"
        store = CircuitBreakerStore(str(home_path))
        assert store.storage_path == home_path.resolve()

        # Test temp directory path
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            temp_path = Path(tf.name)
            store = CircuitBreakerStore(str(temp_path))
            assert store.storage_path == temp_path.resolve()
            temp_path.unlink()

    def test_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected"""
        # Test path outside allowed directories
        with pytest.raises(ValueError, match="must be within allowed directories"):
            CircuitBreakerStore("/etc/passwd.json")

        # Test path with traversal - blocked before resolution for security
        # The path /tmp/../etc/passwd.json contains ".." and is blocked immediately
        with pytest.raises(ValueError, match="Path traversal attempt detected"):
            CircuitBreakerStore("/tmp/../etc/passwd.json")

    def test_non_json_file_rejected(self):
        """Test that non-JSON files are rejected"""
        with pytest.raises(ValueError, match="must be a .json file"):
            CircuitBreakerStore(str(Path.home() / ".context_switcher" / "test.txt"))

    def test_symlink_validation(self):
        """Test that symlinks are properly resolved and validated"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid target
            valid_target = Path(tmpdir) / "valid.json"
            valid_target.write_text("{}")

            # Create a symlink to the valid target
            symlink = Path(tmpdir) / "link.json"
            symlink.symlink_to(valid_target)

            # This should work as both are in temp directory
            store = CircuitBreakerStore(str(symlink))
            assert store.storage_path == symlink.resolve()

    @pytest.mark.asyncio
    async def test_atomic_write_prevents_corruption(self):
        """Test that atomic writes prevent state corruption"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            store = CircuitBreakerStore(tf.name)

            # Save state multiple times concurrently
            tasks = []
            for i in range(10):
                task = store.save_state(f"backend_{i}", {"count": i})
                tasks.append(task)

            await asyncio.gather(*tasks)

            # Verify the file is valid JSON
            content = Path(tf.name).read_text()
            data = json.loads(content)  # Should not raise
            assert len(data) == 10

            Path(tf.name).unlink()


class TestSessionLockInitialization:
    """Test race condition fix in ContextSwitcherSession"""

    def test_lock_initialized_in_post_init(self):
        """Test that locks are properly initialized in __post_init__"""
        session = ContextSwitcherSession(
            session_id="test-123", created_at=datetime.utcnow()
        )

        assert session._lock_initialized is True
        # Lock may be None if no event loop is running, which is fine

    @pytest.mark.asyncio
    async def test_record_access_handles_lazy_lock_creation(self):
        """Test that record_access creates lock lazily if needed"""
        session = ContextSwitcherSession(
            session_id="test-456", created_at=datetime.utcnow()
        )

        # Clear the lock to simulate lazy creation scenario
        session._access_lock = None

        # This should create the lock and work properly
        await session.record_access("test_tool")

        assert session._access_lock is not None
        assert session.access_count == 1
        assert session.version == 1

    @pytest.mark.asyncio
    async def test_concurrent_access_no_race_condition(self):
        """Test that concurrent access doesn't cause race conditions"""
        session = ContextSwitcherSession(
            session_id="test-789", created_at=datetime.utcnow()
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

    def test_multiple_instances_independent_locks(self):
        """Test that multiple session instances have independent locks"""
        session1 = ContextSwitcherSession(
            session_id="session-1", created_at=datetime.utcnow()
        )
        session2 = ContextSwitcherSession(
            session_id="session-2", created_at=datetime.utcnow()
        )

        # Both should be initialized
        assert session1._lock_initialized is True
        assert session2._lock_initialized is True

        # Locks should be independent (or both None if no event loop)
        if session1._access_lock is not None and session2._access_lock is not None:
            assert session1._access_lock is not session2._access_lock


class TestSessionManagerResourceCleanup:
    """Test memory leak fix in SessionManager"""

    @pytest.mark.asyncio
    async def test_cleanup_always_attempted(self):
        """Test that cleanup is always attempted even with errors"""
        manager = SessionManager()
        session = ContextSwitcherSession(
            session_id="cleanup-test", created_at=datetime.utcnow()
        )
        manager.sessions["cleanup-test"] = session

        # Make session expired by setting session_ttl to 0 seconds
        manager.session_ttl = timedelta(seconds=0)

        # Mock the _cleanup_session_resources to simulate an error
        with patch.object(
            manager,
            "_cleanup_session_resources",
            side_effect=SessionCleanupError("Cleanup failed"),
        ):
            # This should log the error but not prevent session removal
            await manager._cleanup_expired_sessions()

            # Session should still be removed despite cleanup error
            assert "cleanup-test" not in manager.sessions

    @pytest.mark.asyncio
    async def test_cleanup_multiple_errors_collected(self):
        """Test that multiple cleanup errors are collected and reported"""
        manager = SessionManager()

        # Mock multiple cleanup steps failing
        with patch.object(manager, "_cleanup_session_resources") as mock_cleanup:
            mock_cleanup.side_effect = SessionCleanupError("Multiple errors occurred")

            session = ContextSwitcherSession(
                session_id="multi-error-test", created_at=datetime.utcnow()
            )
            # Make it expired by setting created_at to an old date
            from datetime import timedelta

            session.created_at = datetime.utcnow() - timedelta(hours=2)
            manager.sessions["multi-error-test"] = session
            manager.session_ttl = timedelta(seconds=0)  # Force expiration

            # Run cleanup
            await manager._cleanup_expired_sessions()

            # Session should still be removed despite errors
            assert "multi-error-test" not in manager.sessions

    @pytest.mark.asyncio
    async def test_cleanup_partial_success(self):
        """Test that partial cleanup success is handled properly"""
        manager = SessionManager()

        # Create a mock that succeeds for some operations
        call_count = 0

        def cleanup_side_effect(session_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds
                return
            else:
                # Subsequent calls fail
                raise Exception("Partial failure")

        # Mock the import of rate_limiter module
        mock_limiter = MagicMock()
        mock_limiter.cleanup_session = Mock(side_effect=cleanup_side_effect)
        with patch.dict(
            "sys.modules", {"context_switcher_mcp.rate_limiter": mock_limiter}
        ):
            # Add multiple expired sessions
            for i in range(3):
                session = ContextSwitcherSession(
                    session_id=f"partial-{i}", created_at=datetime.utcnow()
                )
                manager.sessions[f"partial-{i}"] = session

            manager.session_ttl = timedelta(seconds=0)  # Force expiration

            # Run cleanup
            await manager._cleanup_expired_sessions()

            # All sessions should be removed regardless of cleanup errors
            assert len(manager.sessions) == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_import_error(self):
        """Test that missing rate_limiter module doesn't break cleanup"""
        manager = SessionManager()

        session = ContextSwitcherSession(
            session_id="import-error-test", created_at=datetime.utcnow()
        )
        manager.sessions["import-error-test"] = session
        manager.session_ttl = timedelta(seconds=0)  # Force expiration

        # Simulate ImportError for rate_limiter by removing it from sys.modules
        with patch.dict("sys.modules", {"context_switcher_mcp.rate_limiter": None}):
            # This should not raise
            await manager._cleanup_expired_sessions()

            # Session should still be removed
            assert "import-error-test" not in manager.sessions

    @pytest.mark.asyncio
    async def test_cleanup_resources_comprehensive(self):
        """Test comprehensive resource cleanup behavior"""
        manager = SessionManager()

        # Test that cleanup doesn't raise when rate_limiter module exists
        # but doesn't have cleanup_session at module level (which is the current state)
        await manager._cleanup_session_resources("test-normal")

        # Test cleanup with error by adding a mock cleanup_session
        from context_switcher_mcp import rate_limiter

        # Add a mock cleanup_session that raises an error
        rate_limiter.cleanup_session = Mock(side_effect=Exception("Test error"))
        try:
            with pytest.raises(SessionCleanupError):
                await manager._cleanup_session_resources("test-error")
        finally:
            # Clean up the mock attribute
            if hasattr(rate_limiter, "cleanup_session"):
                del rate_limiter.cleanup_session

        # Test again without cleanup_session - should not raise
        await manager._cleanup_session_resources("test-missing")


class TestIntegration:
    """Integration tests for all critical fixes"""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle_with_fixes(self):
        """Test complete session lifecycle with all fixes applied"""
        # Create a circuit breaker store with safe path
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            cb_store = CircuitBreakerStore(tf.name)

            # Create a session manager
            manager = SessionManager()

            # Create a session manually
            session = ContextSwitcherSession(
                session_id="integration-test", created_at=datetime.utcnow()
            )
            await manager.add_session(session)

            # Simulate concurrent access (tests lock fix)
            async def access_session():
                await session.record_access("test_tool")

            tasks = [access_session() for _ in range(10)]
            await asyncio.gather(*tasks)

            assert session.access_count == 10

            # Save circuit breaker state (tests atomic write)
            await cb_store.save_state("test_backend", {"failures": 0})

            # Force session expiration and cleanup (tests cleanup fix)
            manager.session_ttl = timedelta(seconds=0)
            await manager._cleanup_expired_sessions()

            assert session.session_id not in manager.sessions

            # Verify circuit breaker state persisted correctly
            state = await cb_store.load_state("test_backend")
            assert state["failures"] == 0

            Path(tf.name).unlink()

    def test_thread_safety_of_fixes(self):
        """Test that all fixes are thread-safe"""

        def create_session():
            session = ContextSwitcherSession(
                session_id=f"thread-{threading.current_thread().ident}", created_at=None
            )
            return session._lock_initialized

        # Test concurrent session creation
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_session) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

            # All should be initialized
            assert all(results)
