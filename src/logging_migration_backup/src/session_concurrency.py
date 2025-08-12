"""Concurrency control for session access management"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SessionConcurrency:
    """Manages concurrency control for session access"""

    def __init__(self, session_id: str):
        """Initialize concurrency control for a session

        Args:
            session_id: The session identifier
        """
        self.session_id = session_id
        self.access_count = 0
        self.last_accessed = datetime.now(timezone.utc)
        self.version = 0  # Version for optimistic locking and race condition detection

    def _get_lock_manager(self):
        """Get the session lock manager"""
        from .session_lock_manager import get_session_lock_manager

        return get_session_lock_manager()

    async def record_access(self, tool_name: Optional[str] = None) -> None:
        """Record session access in a thread-safe manner

        Args:
            tool_name: Optional name of the tool being accessed
        """
        lock_manager = self._get_lock_manager()
        async with lock_manager.acquire_lock(self.session_id):
            self.access_count += 1
            self.last_accessed = datetime.now(timezone.utc)
            self.version += 1  # Increment version for change tracking

            if tool_name:
                logger.debug(f"Session {self.session_id} accessed tool: {tool_name}")

    def check_version_conflict(self, expected_version: int) -> bool:
        """Check if there's a version conflict

        Args:
            expected_version: The expected version number

        Returns:
            True if there's a conflict, False otherwise
        """
        return self.version != expected_version

    def get_access_stats(self) -> Dict[str, Any]:
        """Get access statistics for the session

        Returns:
            Dictionary containing access statistics
        """
        return {
            "session_id": self.session_id,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "version": self.version,
        }

    async def synchronized_update(
        self, update_func: Callable, expected_version: Optional[int] = None
    ) -> None:
        """Perform a synchronized update with optional version checking

        Args:
            update_func: Function to execute while holding the lock
            expected_version: Optional expected version for optimistic locking

        Raises:
            ValueError: If version conflict is detected
        """
        lock_manager = self._get_lock_manager()
        async with lock_manager.acquire_lock(self.session_id):
            # Check for version conflict if expected version provided
            if expected_version is not None and self.check_version_conflict(
                expected_version
            ):
                raise ValueError(
                    f"Version conflict: expected {expected_version}, got {self.version}"
                )

            # Execute the update function
            if asyncio.iscoroutinefunction(update_func):
                await update_func()
            else:
                update_func()

            # Update access metadata
            self.last_accessed = datetime.now(timezone.utc)
            self.version += 1

    def cleanup(self) -> None:
        """Cleanup session concurrency resources"""
        lock_manager = self._get_lock_manager()
        lock_manager.remove_lock(self.session_id)
        logger.debug(f"Cleaned up concurrency resources for session {self.session_id}")


def create_session_concurrency(session_id: str) -> SessionConcurrency:
    """Factory function to create SessionConcurrency instance

    Args:
        session_id: The session identifier

    Returns:
        SessionConcurrency instance
    """
    return SessionConcurrency(session_id)
