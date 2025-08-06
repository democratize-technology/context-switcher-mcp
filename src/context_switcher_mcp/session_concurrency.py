"""Concurrency control for session access management"""

import asyncio
import threading
import logging
from typing import Optional
from datetime import datetime

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
        self.last_accessed = datetime.utcnow()
        self.version = 0  # Version for optimistic locking and race condition detection

        # Concurrency control attributes
        self._access_lock: Optional[asyncio.Lock] = None
        self._lock_initialized = False

        # CRITICAL: Class-level lock for thread-safe initialization
        # This prevents race conditions during async lock creation
        self._initialization_lock = threading.Lock()

    def initialize_locks(self) -> None:
        """Initialize async locks safely

        This method handles the complex initialization of asyncio locks
        in a way that's safe across different execution contexts.
        """
        if self._lock_initialized:
            return

        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we can create asyncio locks
                self._access_lock = asyncio.Lock()
            else:
                # If no loop or not running, we'll create the lock lazily
                # when first needed in an async context
                self._access_lock = None
        except RuntimeError:
            # No event loop exists yet, will create lock lazily
            self._access_lock = None

        self._lock_initialized = True

    async def record_access(self, tool_name: str = None) -> None:
        """Record session access in a thread-safe manner

        Args:
            tool_name: Optional name of the tool being accessed
        """
        # Ensure access lock is available using double-checked locking pattern
        await self._ensure_access_lock()

        async with self._access_lock:
            self.access_count += 1
            self.last_accessed = datetime.utcnow()
            self.version += 1  # Increment version for change tracking

            if tool_name:
                logger.debug(f"Session {self.session_id} accessed tool: {tool_name}")

    async def _ensure_access_lock(self) -> None:
        """Ensure access lock is initialized using double-checked locking pattern"""
        # Fast path - if lock exists, return immediately
        if self._access_lock is not None:
            return

        # Acquire thread lock for initialization
        with self._initialization_lock:
            # Double-check inside the lock (another thread may have initialized)
            if self._access_lock is None:
                # Now safe to create the asyncio lock
                self._access_lock = asyncio.Lock()

    async def acquire_access_lock(self):
        """Acquire the access lock for external use

        Returns:
            Context manager for the access lock

        Example:
            async with session_concurrency.acquire_access_lock():
                # Critical section code here
                pass
        """
        await self._ensure_access_lock()
        return self._access_lock

    def get_access_info(self) -> dict:
        """Get current access information

        Returns:
            Dictionary containing access information
        """
        return {
            "session_id": self.session_id,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "version": self.version,
            "lock_initialized": self._lock_initialized,
        }

    def increment_version(self) -> int:
        """Increment and return the current version

        Returns:
            The new version number
        """
        self.version += 1
        return self.version

    def check_version_conflict(self, expected_version: int) -> bool:
        """Check if there's a version conflict (optimistic locking)

        Args:
            expected_version: The expected version number

        Returns:
            True if there's a conflict (version mismatch)
        """
        return self.version != expected_version

    async def safe_update(self, update_func, expected_version: Optional[int] = None):
        """Perform a safe update with concurrency control

        Args:
            update_func: Function to execute while holding the lock
            expected_version: Optional expected version for optimistic locking

        Raises:
            ValueError: If version conflict is detected
        """
        await self._ensure_access_lock()

        async with self._access_lock:
            # Check for version conflict if expected version provided
            if expected_version is not None and self.check_version_conflict(
                expected_version
            ):
                raise ValueError(
                    f"Version conflict: expected {expected_version}, got {self.version}"
                )

            # Execute the update function
            result = (
                await update_func()
                if asyncio.iscoroutinefunction(update_func)
                else update_func()
            )

            # Increment version after successful update
            self.increment_version()
            self.last_accessed = datetime.utcnow()

            return result

    def reset_access_tracking(self) -> None:
        """Reset access tracking (useful for testing)"""
        self.access_count = 0
        self.version = 0
        self.last_accessed = datetime.utcnow()

    def get_concurrency_status(self) -> dict:
        """Get detailed concurrency status

        Returns:
            Dictionary with concurrency status information
        """
        return {
            "session_id": self.session_id,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "version": self.version,
            "lock_initialized": self._lock_initialized,
            "has_access_lock": self._access_lock is not None,
            "initialization_lock_acquired": not self._initialization_lock.acquire(
                blocking=False
            ),
        }
