"""Centralized lock management for session concurrency control"""

import asyncio
import threading
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class _AsyncLockContext:
    """Async context manager for session locks"""

    def __init__(self, lock_manager: "SessionLockManager", session_id: str):
        self.lock_manager = lock_manager
        self.session_id = session_id
        self.lock = None

    async def __aenter__(self):
        """Acquire the lock"""
        self.lock = self.lock_manager.get_lock(self.session_id)
        if self.lock:
            await self.lock.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release the lock"""
        if self.lock:
            self.lock.release()


class SessionLockManager:
    """Manages async locks for sessions with thread-safe initialization"""

    def __init__(self):
        """Initialize the lock manager"""
        self._locks: Dict[str, asyncio.Lock] = {}
        self._init_lock = threading.Lock()  # Thread lock for safe initialization

    def get_lock(self, session_id: str) -> Optional[asyncio.Lock]:
        """Get or create an async lock for a session

        Args:
            session_id: The session identifier

        Returns:
            asyncio.Lock if in async context, None otherwise
        """
        # Fast path - check if lock already exists
        if session_id in self._locks:
            return self._locks[session_id]

        # Slow path - need to create lock
        with self._init_lock:
            # Double-check inside lock
            if session_id in self._locks:
                return self._locks[session_id]

            # Try to create async lock
            try:
                asyncio.get_running_loop()
                self._locks[session_id] = asyncio.Lock()
                logger.debug(f"Created async lock for session {session_id}")
                return self._locks[session_id]
            except RuntimeError:
                # No running event loop
                logger.debug(
                    f"No event loop for session {session_id}, deferring lock creation"
                )
                return None

    def acquire_lock(self, session_id: str):
        """Get an async context manager for the session lock

        Args:
            session_id: The session identifier

        Returns:
            An async context manager that acquires/releases the lock

        Usage:
            async with lock_manager.acquire_lock(session_id):
                # Critical section
                pass
        """
        return _AsyncLockContext(self, session_id)

    def remove_lock(self, session_id: str) -> None:
        """Remove lock for a session during cleanup

        Args:
            session_id: The session identifier
        """
        with self._init_lock:
            if session_id in self._locks:
                del self._locks[session_id]
                logger.debug(f"Removed lock for session {session_id}")

    def clear_all(self) -> None:
        """Clear all locks (used in tests or shutdown)"""
        with self._init_lock:
            self._locks.clear()
            logger.debug("Cleared all session locks")


# Global singleton instance
_lock_manager = SessionLockManager()


def get_session_lock_manager() -> SessionLockManager:
    """Get the global session lock manager instance"""
    return _lock_manager
