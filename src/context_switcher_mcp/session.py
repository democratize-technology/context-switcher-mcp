"""Unified session management with built-in security, concurrency, and data management

This module provides a self-contained Session class that integrates all session
functionality without external dependencies on separate lock, security, or data modules.
"""

import asyncio
import hashlib
import secrets
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

try:
    from .exceptions import (
        SessionCleanupError,
        SessionConcurrencyError,
        SessionError,
        SessionSecurityError,
    )
    from .logging_base import get_logger
    from .session_types import (
        AnalysisRecord,
        ClientBinding,
        SecurityEvent,
        SessionState,
        Thread,
    )
except ImportError:
    # Direct imports for standalone usage
    from session_types import (
        AnalysisRecord,
        ClientBinding,
        SecurityEvent,
        SessionState,
        Thread,
    )

    # Mock exceptions for standalone usage
    class SessionError(Exception):
        pass

    class SessionSecurityError(SessionError):
        pass

    class SessionConcurrencyError(SessionError):
        pass

    class SessionCleanupError(SessionError):
        pass

    # Mock logger for standalone usage
    class MockLogger:
        def debug(self, msg):
            pass

        def info(self, msg):
            logger.error(f"INFO: {msg}")

        def warning(self, msg):
            logger.error(f"WARNING: {msg}")

        def error(self, msg):
            logger.error(f"ERROR: {msg}")

    def get_logger(name):
        return MockLogger()


logger = get_logger(__name__)


class Session:
    """Unified session with built-in security, concurrency, and data management

    This class replaces the complex multi-module session system with a single
    self-contained implementation that handles all session operations atomically.
    """

    def __init__(
        self,
        session_id: str,
        topic: str | None = None,
        secret_key: str | None = None,
        create_client_binding: bool = True,
    ):
        """Initialize a new session with built-in security and concurrency

        Args:
            session_id: Unique session identifier
            topic: Optional topic for the session
            secret_key: Secret key for client binding security (generated if None)
            create_client_binding: Whether to create client binding for security
        """
        self.session_id = session_id
        self.secret_key = secret_key or secrets.token_urlsafe(32)

        # Core session state
        self._state = SessionState(
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
            topic=topic,
        )

        # Built-in concurrency control - single lock per session
        self._lock = asyncio.Lock()
        self._cleanup_callbacks: list[Callable[[], None]] = []

        # Create client binding for security if requested
        if create_client_binding:
            self._create_client_binding()

        logger.info(f"Created session {session_id} with built-in security and concurrency")

    def _create_client_binding(self) -> None:
        """Create a secure client binding for this session"""
        session_entropy = secrets.token_urlsafe(32)
        creation_timestamp = datetime.now(timezone.utc)

        # Generate default access pattern hash
        default_data = f"{self.session_id}:{creation_timestamp.isoformat()}"
        access_pattern_hash = hashlib.sha256(default_data.encode()).hexdigest()

        binding = ClientBinding(
            session_entropy=session_entropy,
            creation_timestamp=creation_timestamp,
            binding_signature="",  # Will be set after creation
            access_pattern_hash=access_pattern_hash,
        )

        # Generate and set the binding signature
        binding.binding_signature = binding.generate_binding_signature(self.secret_key)
        self._state.client_binding = binding

        logger.debug(f"Created client binding for session {self.session_id}")

    async def _atomic_operation(self, operation_name: str, operation: Callable) -> Any:
        """Execute an operation atomically with proper return value propagation"""
        async with self._lock:
            start_time = datetime.now(timezone.utc)
            try:
                logger.debug(f"Starting atomic operation '{operation_name}' for session {self.session_id}")

                # Execute the operation and capture its return value
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()

                # Update version and metrics on successful operation
                self._state.version += 1
                self._state.metrics.record_access()

                return result

            except Exception as e:
                # Record error in metrics and re-raise with context
                self._state.metrics.record_error()
                self._record_security_event(
                    f"operation_error_{operation_name}",
                    {"error": str(e), "operation": operation_name},
                )

                logger.error(f"Error in atomic operation '{operation_name}' for session {self.session_id}: {e}")

                # Don't wrap SessionError subclasses - re-raise them as-is
                if isinstance(e, SessionError):
                    raise
                else:
                    raise SessionError(f"Operation '{operation_name}' failed: {e}") from e

            finally:
                # Always record operation time
                end_time = datetime.now(timezone.utc)
                operation_time = (end_time - start_time).total_seconds()
                logger.debug(f"Completed operation '{operation_name}' in {operation_time:.3f}s")

    async def validate_security(self, tool_name: str | None = None) -> bool:
        """Validate session security and record access patterns

        Args:
            tool_name: Optional name of tool being accessed

        Returns:
            True if security validation passes

        Raises:
            SessionSecurityError: If security validation fails
        """

        def _validate_security_impl():
            # Always allow sessions without client binding (backward compatibility)
            if not self._state.client_binding:
                if tool_name:
                    logger.debug(f"Session {self.session_id} accessed tool: {tool_name} (no binding)")
                return True

            # Validate client binding
            is_valid = self._state.client_binding.validate_binding(self.secret_key)

            if not is_valid:
                self._state.client_binding.validation_failures += 1
                self._record_security_event(
                    "binding_validation_failed",
                    {
                        "tool_name": tool_name,
                        "validation_failures": self._state.client_binding.validation_failures,
                    },
                )

                # Check if session is now suspicious
                if self._state.client_binding.is_suspicious():
                    self._record_security_event("session_flagged_suspicious", {"reason": "excessive_failures"})
                    raise SessionSecurityError(
                        f"Session {self.session_id} flagged as suspicious due to excessive validation failures"
                    )

                raise SessionSecurityError(f"Client binding validation failed for session {self.session_id}")

            # Update binding metadata on successful validation
            self._state.client_binding.last_validated = datetime.now(timezone.utc)

            # Record tool usage pattern
            if tool_name:
                self._update_tool_usage_pattern(tool_name)
                logger.debug(f"Session {self.session_id} validated access to tool: {tool_name}")

            return True

        return await self._atomic_operation("validate_security", _validate_security_impl)

    def _update_tool_usage_pattern(self, tool_name: str) -> None:
        """Update tool usage pattern in client binding (internal, assumes lock held)"""
        if self._state.client_binding and len(self._state.client_binding.tool_usage_sequence) < 10:
            self._state.client_binding.tool_usage_sequence.append(tool_name)

    def _record_security_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Record a security event (internal, assumes lock held)"""
        event = SecurityEvent(event_type=event_type, timestamp=datetime.now(timezone.utc), details=details)

        self._state.security_events.append(event)
        self._state.metrics.record_security_event()

        # Add security flag to client binding if present
        if self._state.client_binding:
            self._state.client_binding.add_security_flag(event_type)

        logger.info(f"Security event '{event_type}' recorded for session {self.session_id}")

    async def add_thread(self, thread: Thread) -> bool:
        """Add a perspective thread to the session

        Args:
            thread: The Thread to add

        Returns:
            True if thread was added, False if already exists
        """

        def _add_thread_impl():
            if thread.name in self._state.threads:
                logger.warning(f"Thread '{thread.name}' already exists in session {self.session_id}")
                return False

            self._state.threads[thread.name] = thread
            self._state.metrics.thread_count = len(self._state.threads)

            logger.info(f"Added thread '{thread.name}' to session {self.session_id}")
            return True

        return await self._atomic_operation("add_thread", _add_thread_impl)

    async def get_thread(self, name: str) -> Thread | None:
        """Get a thread by name

        Args:
            name: Name of the thread

        Returns:
            Thread if found, None otherwise
        """

        def _get_thread_impl():
            thread = self._state.threads.get(name)
            if thread:
                logger.debug(f"Retrieved thread '{name}' from session {self.session_id}")
            return thread

        return await self._atomic_operation("get_thread", _get_thread_impl)

    async def remove_thread(self, name: str) -> bool:
        """Remove a thread by name

        Args:
            name: Name of the thread to remove

        Returns:
            True if thread was removed, False if not found
        """

        def _remove_thread_impl():
            if name in self._state.threads:
                del self._state.threads[name]
                self._state.metrics.thread_count = len(self._state.threads)
                logger.info(f"Removed thread '{name}' from session {self.session_id}")
                return True
            return False

        return await self._atomic_operation("remove_thread", _remove_thread_impl)

    async def get_all_threads(self) -> dict[str, Thread]:
        """Get all threads in the session

        Returns:
            Dictionary of thread name -> Thread
        """

        def _get_all_threads_impl():
            return self._state.threads.copy()

        return await self._atomic_operation("get_all_threads", _get_all_threads_impl)

    async def record_analysis(
        self,
        prompt: str,
        responses: dict[str, str],
        response_time: float = 0.0,
    ) -> None:
        """Record an analysis execution with metrics

        Args:
            prompt: The analysis prompt
            responses: Map of thread name -> response
            response_time: Time taken for analysis in seconds
        """

        def _record_analysis_impl():
            active_count = sum(1 for r in responses.values() if r != "[NO_RESPONSE]")
            abstained_count = len(responses) - active_count

            analysis = AnalysisRecord(
                prompt=prompt,
                timestamp=datetime.now(timezone.utc),
                responses=responses,
                active_count=active_count,
                abstained_count=abstained_count,
            )

            self._state.analyses.append(analysis)
            self._state.metrics.record_analysis(response_time)

            logger.info(
                f"Recorded analysis for session {self.session_id}: "
                f"{active_count} active, {abstained_count} abstained responses"
            )

        return await self._atomic_operation("record_analysis", _record_analysis_impl)

    async def get_last_analysis(self) -> AnalysisRecord | None:
        """Get the most recent analysis

        Returns:
            Latest AnalysisRecord if any exist, None otherwise
        """

        def _get_last_analysis_impl():
            return self._state.analyses[-1] if self._state.analyses else None

        return await self._atomic_operation("get_last_analysis", _get_last_analysis_impl)

    async def get_session_info(self) -> dict[str, Any]:
        """Get comprehensive session information

        Returns:
            Dictionary containing complete session state and metrics
        """

        def _get_session_info_impl():
            security_info = None
            if self._state.client_binding:
                security_info = {
                    "validation_failures": self._state.client_binding.validation_failures,
                    "last_validated": self._state.client_binding.last_validated.isoformat(),
                    "security_flags_count": len(self._state.client_binding.security_flags),
                    "is_suspicious": self._state.client_binding.is_suspicious(),
                    "tool_usage_count": len(self._state.client_binding.tool_usage_sequence),
                }

            return {
                "session_id": self.session_id,
                "created_at": self._state.created_at.isoformat(),
                "topic": self._state.topic,
                "version": self._state.version,
                "thread_count": len(self._state.threads),
                "thread_names": list(self._state.threads.keys()),
                "analysis_count": len(self._state.analyses),
                "security_event_count": len(self._state.security_events),
                "client_binding": security_info,
                "metrics": self._state.metrics.to_dict(),
                "recent_security_events": [event.to_dict() for event in self._state.security_events[-5:]],
            }

        return await self._atomic_operation("get_session_info", _get_session_info_impl)

    async def get_version_info(self) -> tuple[int, datetime]:
        """Get current version and last access time for concurrency control

        Returns:
            Tuple of (version, last_accessed)
        """
        async with self._lock:  # Simple lock for read-only operation
            return self._state.version, self._state.metrics.last_accessed

    async def validate_version(self, expected_version: int) -> bool:
        """Validate expected version for optimistic concurrency control

        Args:
            expected_version: The expected version number

        Returns:
            True if version matches, False if there's a conflict
        """
        async with self._lock:  # Simple lock for read-only operation
            return self._state.version == expected_version

    async def atomic_update(
        self,
        update_func: Callable,
        expected_version: int | None = None,
    ) -> Any:
        """Perform an atomic update with optional optimistic locking

        Args:
            update_func: Function to execute atomically
            expected_version: Optional expected version for conflict detection

        Returns:
            Result of update_func

        Raises:
            SessionConcurrencyError: If version conflict detected
        """

        async def _atomic_update_impl():
            # Check for version conflict if expected version provided
            if expected_version is not None and self._state.version != expected_version:
                raise SessionConcurrencyError(
                    f"Version conflict in session {self.session_id}: "
                    f"expected {expected_version}, got {self._state.version}"
                )

            # Execute the update function
            if asyncio.iscoroutinefunction(update_func):
                result = await update_func()
            else:
                result = update_func()

            # Version is automatically incremented by _atomic_operation
            return result

        return await self._atomic_operation("atomic_update", _atomic_update_impl)

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to be executed during session cleanup

        Args:
            callback: Function to call during cleanup (should not raise)
        """
        self._cleanup_callbacks.append(callback)

    async def cleanup(self) -> None:
        """Clean up session resources and external dependencies

        This method is self-contained and handles all cleanup without external
        dependencies. It ensures resources are cleaned up even if errors occur.
        """
        cleanup_errors = []

        async with self._lock:  # Ensure no concurrent operations during cleanup
            logger.info(f"Starting cleanup for session {self.session_id}")

            # Execute cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    cleanup_errors.append(f"Callback cleanup failed: {e!s}")
                    logger.warning(f"Cleanup callback failed for session {self.session_id}: {e}")

            # Clean up rate limiter resources if available
            try:
                from . import rate_limiter

                if hasattr(rate_limiter, "cleanup_session"):
                    rate_limiter.cleanup_session(self.session_id)
            except ImportError:
                pass  # Rate limiter not available
            except Exception as e:
                cleanup_errors.append(f"Rate limiter cleanup failed: {e!s}")
                logger.warning(f"Rate limiter cleanup failed for session {self.session_id}: {e}")

            # Clear session state
            try:
                self._state.threads.clear()
                self._state.analyses.clear()
                # Keep security events for audit purposes until final cleanup
                if len(self._state.security_events) > 100:
                    # Keep only recent events to prevent memory leaks
                    self._state.security_events = self._state.security_events[-50:]
            except Exception as e:
                cleanup_errors.append(f"State cleanup failed: {e!s}")
                logger.warning(f"State cleanup failed for session {self.session_id}: {e}")

            # Log cleanup completion
            if cleanup_errors:
                logger.warning(f"Session {self.session_id} cleanup completed with {len(cleanup_errors)} errors")
                # Don't raise exception - cleanup should be best-effort
            else:
                logger.info(f"Session {self.session_id} cleanup completed successfully")

    async def export_state(self) -> dict[str, Any]:
        """Export complete session state for persistence/backup

        Returns:
            Dictionary containing complete serializable session state
        """
        async with self._lock:
            return self._state.to_dict()

    @classmethod
    async def restore_from_state(
        cls,
        state_data: dict[str, Any],
        secret_key: str | None = None,
    ) -> "Session":
        """Restore session from exported state data

        Args:
            state_data: Previously exported session state
            secret_key: Secret key for client binding (generated if None)

        Returns:
            Restored Session instance
        """
        # Create session without client binding first
        session = cls(
            session_id=state_data["session_id"],
            topic=state_data.get("topic"),
            secret_key=secret_key,
            create_client_binding=False,  # Will restore from state
        )

        # Restore complete state
        session._state = SessionState.from_dict(state_data)

        logger.info(f"Restored session {session.session_id} from state data")
        return session

    def is_expired(self, ttl_hours: float) -> bool:
        """Check if session has expired based on TTL

        Args:
            ttl_hours: Time-to-live in hours

        Returns:
            True if session has expired
        """
        age = datetime.now(timezone.utc) - self._state.created_at
        return age.total_seconds() > (ttl_hours * 3600)

    def __repr__(self) -> str:
        return (
            f"Session(id='{self.session_id}', "
            f"threads={len(self._state.threads)}, "
            f"analyses={len(self._state.analyses)}, "
            f"version={self._state.version})"
        )
