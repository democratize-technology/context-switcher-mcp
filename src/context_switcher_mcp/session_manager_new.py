"""Simplified session manager with unified session pool management

This module provides a clean, simple session manager that works with the
unified Session class. It removes all complex interdependencies and provides
a straightforward interface for session lifecycle management.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager

try:
    from .session import Session
    from .session_types import Thread, ModelBackend
    from .config import get_config
    from .exceptions import SessionError, SessionCleanupError
    from .logging_base import get_logger
except ImportError:
    # Direct imports for standalone usage
    from session import Session
    from session_types import Thread, ModelBackend
    # Mock config and exceptions for standalone usage
    class MockConfig:
        class session:
            max_active_sessions = 50
            default_ttl_hours = 1.0
            cleanup_interval_seconds = 300
    def get_config(): return MockConfig()
    class SessionError(Exception): pass
    class SessionCleanupError(SessionError): pass
    class MockLogger:
        def debug(self, msg): pass
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    def get_logger(name): return MockLogger()

logger = get_logger(__name__)


class SimpleSessionManager:
    """Simplified session manager with clean interface and minimal dependencies
    
    This replaces the complex session management system with a straightforward
    implementation that focuses on essential functionality:
    - Session pool management with TTL
    - Automatic cleanup
    - Simple statistics
    - Thread-safe operations with a single global lock
    """

    def __init__(
        self,
        max_sessions: Optional[int] = None,
        session_ttl_hours: Optional[float] = None,
        cleanup_interval_seconds: Optional[int] = None,
    ):
        """Initialize the simplified session manager
        
        Args:
            max_sessions: Maximum number of concurrent sessions (uses config default if None)
            session_ttl_hours: Hours before a session expires (uses config default if None)  
            cleanup_interval_seconds: Seconds between cleanup runs (uses config default if None)
        """
        config = get_config()
        
        # Simple session pool - just a dict with a single global lock
        self._sessions: Dict[str, Session] = {}
        self._global_lock = asyncio.Lock()
        
        # Configuration
        self.max_sessions = (
            max_sessions if max_sessions is not None
            else config.session.max_active_sessions
        )
        self.session_ttl_hours = (
            session_ttl_hours if session_ttl_hours is not None
            else config.session.default_ttl_hours
        )
        self.cleanup_interval = timedelta(
            seconds=cleanup_interval_seconds if cleanup_interval_seconds is not None
            else config.session.cleanup_interval_seconds
        )
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(
            f"Initialized SimpleSessionManager: max_sessions={self.max_sessions}, "
            f"ttl={self.session_ttl_hours}h, cleanup_interval={self.cleanup_interval.total_seconds()}s"
        )

    async def create_session(
        self,
        session_id: str,
        topic: Optional[str] = None,
        initial_perspectives: Optional[List[str]] = None,
        model_backend: ModelBackend = ModelBackend.BEDROCK,
        model_name: Optional[str] = None,
    ) -> Session:
        """Create a new session with optional initial perspectives
        
        Args:
            session_id: Unique session identifier
            topic: Optional topic for the session
            initial_perspectives: List of initial perspective names to create
            model_backend: Model backend to use for perspectives
            model_name: Optional specific model name
            
        Returns:
            New Session instance
            
        Raises:
            SessionError: If session already exists or capacity exceeded
        """
        async with self._global_lock:
            # Check if session already exists
            if session_id in self._sessions:
                raise SessionError(f"Session '{session_id}' already exists")
            
            # Check capacity and cleanup if needed
            if len(self._sessions) >= self.max_sessions:
                await self._cleanup_expired_sessions_internal()
                
                if len(self._sessions) >= self.max_sessions:
                    raise SessionError(f"Session capacity exceeded ({self.max_sessions})")
            
            # Create new session with built-in security
            session = Session(session_id=session_id, topic=topic, create_client_binding=True)
            
            # Add initial perspectives if provided
            if initial_perspectives:
                from .templates import get_perspective_system_prompt
                
                for perspective_name in initial_perspectives:
                    system_prompt = get_perspective_system_prompt(perspective_name)
                    if system_prompt:
                        thread = Thread(
                            id=f"{session_id}_{perspective_name}",
                            name=perspective_name,
                            system_prompt=system_prompt,
                            model_backend=model_backend,
                            model_name=model_name,
                        )
                        await session.add_thread(thread)
                    else:
                        logger.warning(f"Unknown perspective '{perspective_name}', skipping")
            
            # Add to session pool
            self._sessions[session_id] = session
            
            logger.info(
                f"Created session '{session_id}' with {len(initial_perspectives or [])} perspectives "
                f"(pool size: {len(self._sessions)})"
            )
            
            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get an active session by ID
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if found and not expired, None otherwise
        """
        async with self._global_lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            
            # Check if session has expired
            if session.is_expired(self.session_ttl_hours):
                logger.info(f"Removing expired session '{session_id}'")
                await self._remove_session_internal(session_id)
                return None
            
            return session

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session from the pool
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was removed, False if not found
        """
        async with self._global_lock:
            return await self._remove_session_internal(session_id)

    async def _remove_session_internal(self, session_id: str) -> bool:
        """Remove session (internal method, assumes lock is held)"""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        
        # Perform session cleanup
        try:
            await session.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup of session '{session_id}': {e}")
            # Don't fail the removal due to cleanup errors
        
        logger.info(f"Removed session '{session_id}' (pool size: {len(self._sessions)})")
        return True

    async def list_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active sessions
        
        Returns:
            Dictionary mapping session_id -> session info
        """
        async with self._global_lock:
            # Clean up expired sessions first
            await self._cleanup_expired_sessions_internal()
            
            result = {}
            for session_id, session in self._sessions.items():
                try:
                    result[session_id] = await session.get_session_info()
                except Exception as e:
                    logger.warning(f"Error getting info for session '{session_id}': {e}")
                    result[session_id] = {"error": str(e)}
            
            return result

    async def get_most_recent_session(self) -> Optional[Session]:
        """Get the most recently created session
        
        Returns:
            Most recent Session if any exist, None otherwise
        """
        async with self._global_lock:
            if not self._sessions:
                return None
            
            # Find most recent session by created_at time
            most_recent = None
            most_recent_time = None
            
            for session in self._sessions.values():
                session_info = await session.get_session_info()
                created_at = datetime.fromisoformat(session_info["created_at"])
                
                if most_recent_time is None or created_at > most_recent_time:
                    most_recent = session
                    most_recent_time = created_at
            
            return most_recent

    async def cleanup_expired_sessions(self) -> int:
        """Manually trigger cleanup of expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        async with self._global_lock:
            return await self._cleanup_expired_sessions_internal()

    async def _cleanup_expired_sessions_internal(self) -> int:
        """Clean up expired sessions (internal method, assumes lock is held)"""
        expired_sessions = []
        
        # Identify expired sessions
        for session_id, session in self._sessions.items():
            if session.is_expired(self.session_ttl_hours):
                expired_sessions.append(session_id)
        
        # Remove expired sessions
        cleanup_count = 0
        for session_id in expired_sessions:
            if await self._remove_session_internal(session_id):
                cleanup_count += 1
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} expired sessions")
        
        return cleanup_count

    async def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics
        
        Returns:
            Dictionary containing statistics and metrics
        """
        async with self._global_lock:
            active_count = len(self._sessions)
            
            # Calculate age statistics
            oldest_session_age = None
            newest_session_age = None
            total_threads = 0
            total_analyses = 0
            
            if self._sessions:
                current_time = datetime.now(timezone.utc)
                
                for session in self._sessions.values():
                    session_info = await session.get_session_info()
                    created_at = datetime.fromisoformat(session_info["created_at"])
                    age = current_time - created_at
                    
                    if oldest_session_age is None or age > oldest_session_age:
                        oldest_session_age = age
                    if newest_session_age is None or age < newest_session_age:
                        newest_session_age = age
                    
                    total_threads += session_info["thread_count"]
                    total_analyses += session_info["analysis_count"]
            
            return {
                "active_sessions": active_count,
                "max_sessions": self.max_sessions,
                "capacity_used_percent": round((active_count / self.max_sessions) * 100, 1),
                "session_ttl_hours": self.session_ttl_hours,
                "cleanup_interval_seconds": self.cleanup_interval.total_seconds(),
                "total_threads": total_threads,
                "total_analyses": total_analyses,
                "oldest_session_age_seconds": oldest_session_age.total_seconds() if oldest_session_age else None,
                "newest_session_age_seconds": newest_session_age.total_seconds() if newest_session_age else None,
                "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done(),
            }

    async def start_background_cleanup(self) -> None:
        """Start the background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._shutdown_event.clear()
            self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())
            logger.info("Started background session cleanup task")

    async def stop_background_cleanup(self) -> None:
        """Stop the background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            logger.info("Stopped background session cleanup task")

    async def _background_cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while not self._shutdown_event.is_set():
            try:
                # Wait for either shutdown signal or cleanup interval
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.cleanup_interval.total_seconds()
                )
                # If we get here, shutdown was signaled
                break
            except asyncio.TimeoutError:
                # Timeout is normal - time to run cleanup
                pass
            
            try:
                cleanup_count = await self.cleanup_expired_sessions()
                if cleanup_count > 0:
                    logger.debug(f"Background cleanup removed {cleanup_count} expired sessions")
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
                # Sleep a bit to avoid tight error loops
                await asyncio.sleep(10)

    async def shutdown(self) -> None:
        """Shutdown the session manager and clean up all resources"""
        logger.info("Shutting down session manager...")
        
        # Stop background cleanup
        await self.stop_background_cleanup()
        
        # Clean up all sessions
        async with self._global_lock:
            session_ids = list(self._sessions.keys())
            for session_id in session_ids:
                try:
                    await self._remove_session_internal(session_id)
                except Exception as e:
                    logger.warning(f"Error cleaning up session '{session_id}' during shutdown: {e}")
        
        logger.info("Session manager shutdown complete")

    @asynccontextmanager
    async def session_context(self, session_id: str, **create_kwargs):
        """Context manager for temporary sessions that are automatically cleaned up
        
        Args:
            session_id: Session identifier
            **create_kwargs: Arguments passed to create_session
            
        Yields:
            Session instance
            
        Example:
            async with manager.session_context("temp_session", topic="test") as session:
                # Use session here
                pass
            # Session is automatically removed
        """
        session = await self.create_session(session_id, **create_kwargs)
        try:
            yield session
        finally:
            await self.remove_session(session_id)

    def __repr__(self) -> str:
        return (
            f"SimpleSessionManager(active={len(self._sessions)}/{self.max_sessions}, "
            f"ttl={self.session_ttl_hours}h)"
        )


# Global singleton instance for backward compatibility
_session_manager: Optional[SimpleSessionManager] = None


def get_session_manager() -> SimpleSessionManager:
    """Get the global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SimpleSessionManager()
    return _session_manager


async def initialize_session_manager(**kwargs) -> SimpleSessionManager:
    """Initialize and start the global session manager
    
    Args:
        **kwargs: Arguments passed to SimpleSessionManager constructor
        
    Returns:
        Initialized session manager
    """
    global _session_manager
    _session_manager = SimpleSessionManager(**kwargs)
    await _session_manager.start_background_cleanup()
    return _session_manager


async def shutdown_session_manager() -> None:
    """Shutdown the global session manager"""
    global _session_manager
    if _session_manager:
        await _session_manager.shutdown()
        _session_manager = None