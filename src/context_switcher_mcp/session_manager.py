"""Session management with automatic cleanup for Context-Switcher MCP"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .session import Session

from .config import get_config
from .error_context import suppress_and_log
from .error_logging import log_error_with_context
from .exceptions import (
    SessionCleanupError,
    SessionError,
)
from .logging_base import get_logger
from .models import ContextSwitcherSession

logger = get_logger(__name__)


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
        self.sessions: dict[str, ContextSwitcherSession] = {}
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
        self._cleanup_task: asyncio.Task | None = None

    async def create_session(
        self,
        session_id: str,
        topic: str | None = None,
        initial_perspectives: list[str] | None = None,
        model_backend: str | None = None,
        **kwargs,
    ) -> "Session":
        """Create a new session and add it to the manager

        Args:
            session_id: Unique session identifier
            topic: Optional topic for the session
            initial_perspectives: Optional list of initial perspectives
            model_backend: Optional model backend to use
            **kwargs: Additional arguments passed to Session constructor

        Returns:
            The created Session object

        Raises:
            SessionError: If session creation fails
        """
        # Import Session here to avoid circular imports
        from .session import Session

        # Create the new unified session
        session = Session(session_id=session_id, topic=topic, **kwargs)

        # Add initial perspectives if requested
        if initial_perspectives:
            from .session_types import Thread, ModelBackend as MB

            # Convert string backend to enum if provided
            backend = (
                getattr(MB, model_backend.upper()) if model_backend else MB.BEDROCK
            )

            for perspective in initial_perspectives:
                thread = Thread(
                    id=f"{perspective}_1",
                    name=perspective,
                    system_prompt=f"You are a {perspective} perspective analyzer.",
                    model_backend=backend,
                )
                await session.add_thread(thread)

        # Store session (convert to ContextSwitcherSession for storage)
        # For now, create a compatible session object for storage
        storage_session = ContextSwitcherSession(
            session_id=session_id,
            created_at=session._state.created_at,
            topic=topic,
            client_binding=session._state.client_binding,
            access_count=session._state.metrics.access_count,
            last_accessed=session._state.created_at,
        )

        # Store the original session object as well for retrieval
        storage_session._unified_session = session

        success = await self.add_session(storage_session)
        if not success:
            raise SessionError(
                f"Session capacity exceeded - manager at capacity ({self.max_sessions})"
            )

        return session

    async def add_session(self, session: ContextSwitcherSession) -> bool:
        """Add a new session

        Returns:
            True if added successfully, False if at capacity
        """
        async with self._lock:
            if len(self.sessions) >= self.max_sessions:
                await self._cleanup_expired_sessions()

                if len(self.sessions) >= self.max_sessions:
                    logger.warning(f"Session limit reached ({self.max_sessions})")
                    return False

            self.sessions[session.session_id] = session
            logger.info(
                f"Added session {session.session_id}, total: {len(self.sessions)}"
            )
            return True

    async def get_session(self, session_id: str) -> ContextSwitcherSession | None:
        """Get a session by ID, atomically handling expiration"""
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None

            # Atomic check-and-delete for expired sessions
            if self._is_expired(session):
                # Remove from sessions first to prevent race conditions
                removed_session = self.sessions.pop(session_id, None)
                if removed_session:
                    logger.info(f"Removed expired session {session_id}")
                    # Cleanup rate limiter outside the critical session operation
                    async with suppress_and_log(
                        ImportError,
                        AttributeError,
                        Exception,
                        operation_name="session_cleanup_expired",
                        log_level=logging.WARNING,
                    ):
                        await self._cleanup_session_resources(session_id)
                return None

            # Return unified session if available, otherwise return storage session
            return getattr(session, "_unified_session", session)

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session (idempotent and race-condition safe)"""
        async with self._lock:
            removed_session = self.sessions.pop(session_id, None)
            if removed_session:
                logger.info(f"Removed session {session_id}")
                # Clean up resources outside the critical section, don't let failures break removal
                async with suppress_and_log(
                    ImportError,
                    AttributeError,
                    Exception,
                    operation_name="session_cleanup_removed",
                    log_level=logging.WARNING,
                ):
                    await self._cleanup_session_resources(session_id)
                return True
            return False

    async def list_active_sessions(self) -> dict[str, ContextSwitcherSession]:
        """List all active (non-expired) sessions"""
        async with self._lock:
            await self._cleanup_expired_sessions()
            return self.sessions.copy()

    def _is_expired(self, session: ContextSwitcherSession) -> bool:
        """Check if a session has expired"""
        # Check unified session state if available for accurate expiration timing
        unified_session = getattr(session, "_unified_session", None)
        if unified_session:
            age = datetime.now(timezone.utc) - unified_session._state.created_at
        else:
            age = datetime.now(timezone.utc) - session.created_at
        return age > self.session_ttl

    async def _cleanup_expired_sessions(self):
        """Remove expired sessions (internal, assumes lock is held)"""
        # Create a snapshot to avoid dictionary modification during iteration
        session_snapshot = list(self.sessions.items())
        expired_sessions = []

        # Identify expired sessions
        for session_id, session in session_snapshot:
            if self._is_expired(session):
                expired_sessions.append(session_id)

        # Atomically remove expired sessions with guaranteed cleanup
        for session_id in expired_sessions:
            removed_session = self.sessions.pop(session_id, None)
            if removed_session:
                # Always attempt cleanup, even if errors occur
                # The cleanup method now handles errors internally
                try:
                    await self._cleanup_session_resources(session_id)
                except SessionCleanupError as e:
                    # Cleanup had errors but was attempted
                    from .security import sanitize_error_message

                    logger.warning(
                        f"Session {session_id} cleanup completed with errors: {sanitize_error_message(str(e))}"
                    )
                except Exception as e:
                    # Unexpected error - this shouldn't happen with new cleanup
                    cleanup_error = SessionCleanupError(
                        f"Unexpected cleanup error: {str(e)}"
                    )
                    log_error_with_context(
                        error=cleanup_error,
                        operation_name="session_cleanup_batch",
                        session_id=session_id,
                        additional_context={"cleanup_type": "batch_expired"},
                    )
                finally:
                    # Ensure session is marked as cleaned up even if errors occurred
                    # This prevents resource accumulation
                    logger.debug(f"Session {session_id} removed from active sessions")

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def _cleanup_session_resources(self, session_id: str) -> None:
        """Clean up external resources for a session (rate limiter, etc.)

        This method ensures resources are cleaned up even if errors occur.
        All cleanup steps are attempted, and errors are logged but don't
        prevent other cleanup steps from executing.
        """
        cleanup_errors = []

        # Cleanup rate limiter resources
        try:
            from . import rate_limiter

            if hasattr(rate_limiter, "cleanup_session"):
                rate_limiter.cleanup_session(session_id)
        except ImportError:
            # Rate limiter not available, skip cleanup
            pass
        except AttributeError as e:
            # cleanup_session method not available
            logger.warning(f"Rate limiter missing cleanup_session method: {e}")
        except Exception as e:
            # Log error but continue with other cleanup
            cleanup_error = SessionCleanupError(
                f"Rate limiter cleanup failed: {str(e)}"
            )
            correlation_id = log_error_with_context(
                error=cleanup_error,
                operation_name="rate_limiter_cleanup",
                session_id=session_id,
                additional_context={"resource_type": "rate_limiter"},
            )
            cleanup_errors.append(f"Rate limiter cleanup failed (ID: {correlation_id})")

        # Future: Add other resource cleanup here (e.g., cache, temp files)
        # Each cleanup step should be in its own try/except block

        # If there were critical errors, raise them after all cleanup attempts
        if cleanup_errors:
            # Still raise the error so caller knows cleanup had issues,
            # but all cleanup steps were attempted
            raise SessionCleanupError(
                f"Session cleanup had {len(cleanup_errors)} error(s): {'; '.join(cleanup_errors)}"
            )

    async def get_session_atomic(
        self, session_id: str
    ) -> tuple[ContextSwitcherSession | None, int]:
        """Get a session atomically with its version for optimistic locking

        Returns:
            Tuple of (session, version) or (None, -1) if not found/expired
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None, -1

            # Atomic check-and-delete for expired sessions
            if self._is_expired(session):
                removed_session = self.sessions.pop(session_id, None)
                if removed_session:
                    logger.info(
                        f"Removed expired session {session_id} during atomic get"
                    )
                    try:
                        await self._cleanup_session_resources(session_id)
                    except (ImportError, AttributeError) as e:
                        # Module or attribute not available - non-critical
                        from .security import sanitize_error_message

                        logger.warning(
                            f"Failed to cleanup resources for expired session {session_id}: {sanitize_error_message(str(e))}"
                        )
                    except Exception as e:
                        # Unexpected errors - log but don't fail
                        cleanup_error = SessionCleanupError(
                            f"Unexpected cleanup error in get_session: {str(e)}"
                        )
                        log_error_with_context(
                            error=cleanup_error,
                            operation_name="session_cleanup_get",
                            session_id=session_id,
                            additional_context={
                                "operation": "get_session_with_version"
                            },
                        )
                return None, -1

            return session, session.version

    async def validate_session_version(
        self, session_id: str, expected_version: int
    ) -> bool:
        """Validate that a session still exists and has the expected version

        Returns:
            True if session exists and version matches, False otherwise
        """
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is None or self._is_expired(session):
                return False
            return session.version == expected_version

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
                # Unexpected errors - log but continue running
                cleanup_error = SessionCleanupError(f"Periodic cleanup error: {str(e)}")
                log_error_with_context(
                    error=cleanup_error,
                    operation_name="periodic_session_cleanup",
                    additional_context={"cleanup_type": "periodic_background"},
                )
                # Sleep a bit extra to avoid tight error loops
                await asyncio.sleep(10)

    async def get_stats(self) -> dict[str, Any]:
        """Get session manager statistics"""
        async with self._lock:
            active_sessions = len(self.sessions)
            oldest_session = None
            newest_session = None
            total_threads = 0
            total_analyses = 0

            if self.sessions:
                sessions_by_age = sorted(
                    self.sessions.values(), key=lambda s: s.created_at
                )
                oldest_session = sessions_by_age[0].created_at
                newest_session = sessions_by_age[-1].created_at

                # Count threads and analyses across all sessions
                for session in self.sessions.values():
                    # Count threads from unified session if available
                    unified_session = getattr(session, "_unified_session", None)
                    if unified_session:
                        total_threads += len(unified_session._state.threads)
                        total_analyses += len(unified_session._state.analyses)
                    else:
                        # Fallback to storage session
                        total_threads += len(session.threads)
                        total_analyses += len(session.analyses)

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
                "capacity_used_percent": (active_sessions / self.max_sessions) * 100.0,
                "total_threads": total_threads,
                "total_analyses": total_analyses,
            }

    async def get_most_recent_session(self) -> "Session | None":
        """Get the most recently created session

        Returns:
            The most recent Session object or None if no sessions exist
        """
        async with self._lock:
            if not self.sessions:
                return None

            # Find the most recent session by creation time
            most_recent = max(self.sessions.values(), key=lambda s: s.created_at)

            # Return unified session if available, otherwise return storage session
            return getattr(most_recent, "_unified_session", most_recent)

    @asynccontextmanager
    async def session_context(
        self, session_id: str, topic: str | None = None, **kwargs
    ) -> "Session":
        """Context manager for temporary sessions that are automatically cleaned up

        Args:
            session_id: Unique session identifier
            topic: Optional topic for the session
            **kwargs: Additional arguments passed to create_session

        Yields:
            The created Session object

        The session is automatically removed when the context exits
        """
        session = await self.create_session(session_id, topic, **kwargs)
        try:
            yield session
        finally:
            # Ensure cleanup even if there are errors
            await self.remove_session(session_id)

    async def cleanup_expired_sessions(self) -> int:
        """Public method to clean up expired sessions

        Returns:
            Number of sessions that were cleaned up
        """
        async with self._lock:
            before_count = len(self.sessions)
            await self._cleanup_expired_sessions()
            after_count = len(self.sessions)
            cleaned_up = before_count - after_count

            if cleaned_up > 0:
                logger.info(f"Manual cleanup: removed {cleaned_up} expired sessions")

            return cleaned_up
