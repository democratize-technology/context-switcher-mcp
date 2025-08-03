"""Session management with automatic cleanup for Context-Switcher MCP"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from .models import ContextSwitcherSession
from .config import get_config

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages sessions with automatic expiration and cleanup"""

    def __init__(
        self,
        max_sessions: int = None,
        session_ttl_hours: int = None,
        cleanup_interval_minutes: int = None,
    ):
        """Initialize session manager

        Args:
            max_sessions: Maximum number of sessions to keep (uses config default if None)
            session_ttl_hours: Hours before a session expires (uses config default if None)
            cleanup_interval_minutes: Minutes between cleanup runs (uses config default if None)
        """
        config = get_config()
        self.sessions: Dict[str, ContextSwitcherSession] = {}
        self.max_sessions = (
            max_sessions
            if max_sessions is not None
            else config.session.max_active_sessions
        )
        self.session_ttl = timedelta(
            hours=session_ttl_hours
            if session_ttl_hours is not None
            else config.session.default_ttl_hours
        )
        self.cleanup_interval = timedelta(
            seconds=config.session.cleanup_interval_seconds
        )
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def add_session(self, session: ContextSwitcherSession) -> bool:
        """Add a new session

        Returns:
            True if added successfully, False if at capacity
        """
        async with self._lock:
            if len(self.sessions) >= self.max_sessions:
                # Try to clean up expired sessions first
                await self._cleanup_expired_sessions()

                if len(self.sessions) >= self.max_sessions:
                    logger.warning(f"Session limit reached ({self.max_sessions})")
                    return False

            self.sessions[session.session_id] = session
            logger.info(
                f"Added session {session.session_id}, total: {len(self.sessions)}"
            )
            return True

    async def get_session(self, session_id: str) -> Optional[ContextSwitcherSession]:
        """Get a session by ID"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session and self._is_expired(session):
                # Remove expired session
                del self.sessions[session_id]
                logger.info(f"Removed expired session {session_id}")
                return None
            return session

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session"""
        async with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Removed session {session_id}")

                # Clean up rate limiter state if available
                try:
                    from . import rate_limiter

                    rate_limiter.cleanup_session(session_id)
                except ImportError:
                    # Rate limiter not available
                    pass

                return True
            return False

    async def list_active_sessions(self) -> Dict[str, ContextSwitcherSession]:
        """List all active (non-expired) sessions"""
        async with self._lock:
            # Clean up first
            await self._cleanup_expired_sessions()
            return self.sessions.copy()

    def _is_expired(self, session: ContextSwitcherSession) -> bool:
        """Check if a session has expired"""
        age = datetime.utcnow() - session.created_at
        return age > self.session_ttl

    async def _cleanup_expired_sessions(self):
        """Remove expired sessions (internal, assumes lock is held)"""
        expired = []
        for session_id, session in self.sessions.items():
            if self._is_expired(session):
                expired.append(session_id)

        for session_id in expired:
            del self.sessions[session_id]

            # Clean up rate limiter state if available
            try:
                from . import rate_limiter

                rate_limiter.cleanup_session(session_id)
            except ImportError:
                # Rate limiter not available
                pass

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started session cleanup task")

    async def stop_cleanup_task(self):
        """Stop the background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                # Expected when task is cancelled - safe to ignore
                pass
            logger.info("Stopped session cleanup task")

    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())

                async with self._lock:
                    before_count = len(self.sessions)
                    await self._cleanup_expired_sessions()
                    after_count = len(self.sessions)

                if before_count != after_count:
                    logger.info(
                        f"Periodic cleanup: {before_count} -> {after_count} sessions"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        async with self._lock:
            active_sessions = len(self.sessions)
            oldest_session = None
            newest_session = None

            if self.sessions:
                sessions_by_age = sorted(
                    self.sessions.values(), key=lambda s: s.created_at
                )
                oldest_session = sessions_by_age[0].created_at
                newest_session = sessions_by_age[-1].created_at

            return {
                "active_sessions": active_sessions,
                "max_sessions": self.max_sessions,
                "session_ttl_hours": self.session_ttl.total_seconds() / 3600,
                "oldest_session": oldest_session.isoformat()
                if oldest_session
                else None,
                "newest_session": newest_session.isoformat()
                if newest_session
                else None,
                "capacity_used": f"{(active_sessions / self.max_sessions) * 100:.1f}%",
            }
