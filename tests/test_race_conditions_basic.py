"""Basic test for race condition fixes validation"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import asyncio  # noqa: E402
import pytest  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from context_switcher_mcp.session_manager import SessionManager  # noqa: E402
from context_switcher_mcp.models import ContextSwitcherSession  # noqa: E402


@pytest.mark.asyncio
async def test_session_creation_and_access():
    """Test basic session creation and access"""
    session_manager = SessionManager(max_sessions=10, session_ttl_hours=1)

    # Create a test session
    session = ContextSwitcherSession(
        session_id="test-session-123",
        created_at=datetime.utcnow(),
        topic="Test Session",
    )

    # Add session
    result = await session_manager.add_session(session)
    assert result is True

    # Get session
    retrieved_session = await session_manager.get_session("test-session-123")
    assert retrieved_session is not None
    assert retrieved_session.session_id == "test-session-123"


@pytest.mark.asyncio
async def test_session_expiration():
    """Test session expiration handling"""
    session_manager = SessionManager(max_sessions=10, session_ttl_hours=1)

    # Create an expired session
    expired_session = ContextSwitcherSession(
        session_id="expired-session-123",
        created_at=datetime.utcnow() - timedelta(hours=2),  # Expired
        topic="Expired Session",
    )

    # Add expired session
    await session_manager.add_session(expired_session)

    # Try to get expired session - should return None and remove it
    retrieved_session = await session_manager.get_session("expired-session-123")
    assert retrieved_session is None

    # Verify it's removed from sessions dict
    assert "expired-session-123" not in session_manager.sessions


@pytest.mark.asyncio
async def test_concurrent_access_recording():
    """Test that concurrent access recording works"""
    session = ContextSwitcherSession(
        session_id="test-session", created_at=datetime.utcnow(), topic="Concurrent Test"
    )

    initial_count = session.access_count

    # Record multiple accesses concurrently
    tasks = []
    for i in range(5):
        task = session.record_access(f"tool_{i}")
        tasks.append(task)

    await asyncio.gather(*tasks)

    # Access count should be updated properly
    assert session.access_count == initial_count + 5


@pytest.mark.asyncio
async def test_atomic_session_get():
    """Test atomic session get with version"""
    session_manager = SessionManager(max_sessions=10, session_ttl_hours=1)

    session = ContextSwitcherSession(
        session_id="atomic-test", created_at=datetime.utcnow(), topic="Atomic Test"
    )

    await session_manager.add_session(session)

    # Get session atomically
    retrieved_session, version = await session_manager.get_session_atomic("atomic-test")

    assert retrieved_session is not None
    assert retrieved_session.session_id == "atomic-test"
    assert version == session.version


@pytest.mark.asyncio
async def test_version_validation():
    """Test session version validation"""
    session_manager = SessionManager(max_sessions=10, session_ttl_hours=1)

    session = ContextSwitcherSession(
        session_id="version-test", created_at=datetime.utcnow(), topic="Version Test"
    )

    await session_manager.add_session(session)
    current_version = session.version

    # Valid version should return True
    assert await session_manager.validate_session_version(
        "version-test", current_version
    )

    # Invalid version should return False
    assert not await session_manager.validate_session_version(
        "version-test", current_version + 999
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
