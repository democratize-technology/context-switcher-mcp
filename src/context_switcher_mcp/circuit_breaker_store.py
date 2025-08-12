"""Circuit breaker state persistence for production reliability"""

import json
import asyncio
from .logging_base import get_logger
from typing import Dict, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

from .exceptions import (
    CircuitBreakerStateError,
    StorageError,
    SerializationError,
)


logger = get_logger(__name__)


class CircuitBreakerStore:
    """Persistent storage for circuit breaker states

    Supports file-based persistence with automatic background saving.
    Can be extended to support Redis or other backends in the future.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize circuit breaker store

        Args:
            storage_path: Path to store circuit breaker state (default: ~/.context_switcher/circuit_breakers.json)
        """
        validated_path = self._resolve_and_validate_storage_path(storage_path)
        self.storage_path = Path(validated_path)
        self._lock: Optional[asyncio.Lock] = None
        self._auto_save_task: Optional[asyncio.Task] = None

        # Ensure parent directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_and_validate_storage_path(self, storage_path: Optional[str]) -> Path:
        """Resolve and validate the storage path with security checks"""
        if storage_path is None:
            return self._get_default_storage_path()

        return self._validate_custom_storage_path(storage_path)

    def _get_default_storage_path(self) -> Path:
        """Get the default storage path"""
        config_dir = Path.home() / ".context_switcher"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "circuit_breakers.json"

    def _validate_custom_storage_path(self, storage_path: str) -> Path:
        """Validate custom storage path with comprehensive security checks"""
        path = Path(storage_path)

        # Security validation pipeline
        self._check_path_traversal(path)
        resolved_path = self._resolve_path_safely(path)
        self._validate_symlink_security(resolved_path)
        self._validate_allowed_directory(resolved_path)
        self._validate_file_extension(resolved_path)
        self._validate_hidden_directories(resolved_path)

        return resolved_path

    def _check_path_traversal(self, path: Path):
        """Check for path traversal attempts"""
        if ".." in str(path):
            raise ValueError(f"Path traversal attempt detected in path: {path}")

    def _resolve_path_safely(self, path: Path) -> Path:
        """Resolve path handling user home and relative paths"""
        return path.expanduser().resolve(strict=False)

    def _validate_symlink_security(self, path: Path):
        """Validate symlink security"""
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed for security reasons: {path}")

    def _validate_allowed_directory(self, path: Path):
        """Validate path is within allowed base directories"""
        allowed_bases = self._get_allowed_base_directories()

        for base in allowed_bases:
            if self._is_path_within_base(path, base):
                return

        raise ValueError(
            f"Storage path must be within allowed directories: "
            f"{', '.join(str(b) for b in allowed_bases)}. "
            f"Resolved path was: {path}"
        )

    def _get_allowed_base_directories(self) -> list[Path]:
        """Get list of allowed base directories"""
        import tempfile

        allowed_bases = [
            Path.home() / ".context_switcher",  # Default config directory
            Path.home() / ".config",  # Standard config directory
            Path.home() / ".local",  # Local data directory
            Path(tempfile.gettempdir()),  # System temp directory
        ]

        # Add /tmp for Unix systems
        try:
            allowed_bases.append(Path("/tmp").resolve())
        except (OSError, RuntimeError):
            pass  # /tmp may not exist on all systems

        return allowed_bases

    def _is_path_within_base(self, path: Path, base: Path) -> bool:
        """Check if path is within base directory"""
        try:
            base_resolved = base.resolve()
            path.relative_to(base_resolved)
            return True
        except (ValueError, OSError):
            return False

    def _validate_file_extension(self, path: Path):
        """Validate file has proper JSON extension"""
        if path.suffix != ".json":
            raise ValueError("Storage path must be a .json file")

    def _validate_hidden_directories(self, path: Path):
        """Validate hidden directories are safe"""
        safe_hidden_dirs = {".context_switcher", ".config", ".local"}
        path_parts = path.parts

        for part in path_parts:
            if part.startswith(".") and part not in safe_hidden_dirs:
                raise ValueError(
                    f"Hidden directories not allowed except {safe_hidden_dirs}: {part}"
                )

    async def save_state(self, backend: str, state_data: Dict[str, Any]) -> None:
        """Save circuit breaker state for a backend

        Args:
            backend: Backend name (e.g., 'bedrock', 'litellm', 'ollama')
            state_data: Circuit breaker state dictionary
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            try:
                # Load existing data
                all_states = await self._load_all_states()

                # Update with new state
                all_states[backend] = {
                    **state_data,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                }

                # Save to file
                await self._save_all_states(all_states)
                logger.debug(f"Saved circuit breaker state for {backend}")

            except (OSError, IOError) as e:
                logger.error(f"Failed to save circuit breaker state for {backend}: {e}")
                raise StorageError(f"Failed to save state for {backend}: {e}") from e
            except Exception as e:
                logger.error(
                    f"Unexpected error saving circuit breaker state for {backend}: {e}",
                    exc_info=True,
                )
                raise CircuitBreakerStateError(
                    f"Unexpected error saving state: {e}"
                ) from e

    async def load_state(self, backend: str) -> Optional[Dict[str, Any]]:
        """Load circuit breaker state for a backend

        Args:
            backend: Backend name

        Returns:
            Circuit breaker state dictionary or None if not found
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            try:
                all_states = await self._load_all_states()
                state = all_states.get(backend)

                if state:
                    logger.debug(f"Loaded circuit breaker state for {backend}")
                    return state

                return None

            except (OSError, IOError, json.JSONDecodeError) as e:
                logger.error(f"Failed to load circuit breaker state for {backend}: {e}")
                return None
            except Exception as e:
                logger.error(
                    f"Unexpected error loading circuit breaker state for {backend}: {e}",
                    exc_info=True,
                )
                return None

    async def clear_state(self, backend: str) -> None:
        """Clear circuit breaker state for a backend

        Args:
            backend: Backend name
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            try:
                all_states = await self._load_all_states()
                if backend in all_states:
                    del all_states[backend]
                    await self._save_all_states(all_states)
                    logger.debug(f"Cleared circuit breaker state for {backend}")

            except (OSError, IOError) as e:
                logger.error(
                    f"Failed to clear circuit breaker state for {backend}: {e}"
                )
                raise StorageError(f"Failed to clear state for {backend}: {e}") from e
            except Exception as e:
                logger.error(
                    f"Unexpected error clearing circuit breaker state for {backend}: {e}",
                    exc_info=True,
                )
                raise CircuitBreakerStateError(
                    f"Unexpected error clearing state: {e}"
                ) from e

    async def clear_all_states(self) -> None:
        """Clear all circuit breaker states"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            try:
                await self._save_all_states({})
                logger.info("Cleared all circuit breaker states")

            except (OSError, IOError) as e:
                logger.error(f"Failed to clear all circuit breaker states: {e}")
                raise StorageError(f"Failed to clear all states: {e}") from e
            except Exception as e:
                logger.error(
                    f"Unexpected error clearing all circuit breaker states: {e}",
                    exc_info=True,
                )
                raise CircuitBreakerStateError(
                    f"Unexpected error clearing all states: {e}"
                ) from e

    async def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all circuit breaker states

        Returns:
            Dictionary mapping backend names to their states
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            return await self._load_all_states()

    async def start_auto_save(self, interval_seconds: int = 30) -> None:
        """Start background auto-save task

        Args:
            interval_seconds: How often to save state
        """
        if self._auto_save_task is None or self._auto_save_task.done():
            self._auto_save_task = asyncio.create_task(
                self._auto_save_loop(interval_seconds)
            )
            logger.info(
                f"Started circuit breaker auto-save (every {interval_seconds}s)"
            )

    async def stop_auto_save(self) -> None:
        """Stop background auto-save task"""
        if self._auto_save_task and not self._auto_save_task.done():
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped circuit breaker auto-save")

    async def _load_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Load all circuit breaker states from storage"""
        try:
            if not self.storage_path.exists():
                return {}

            # Use asyncio to read file to avoid blocking
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None, self.storage_path.read_text, "utf-8"
            )

            return json.loads(content)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load circuit breaker states: {e}")
            return {}

    async def _save_all_states(self, states: Dict[str, Dict[str, Any]]) -> None:
        """Save all circuit breaker states to storage using atomic write"""
        try:
            content = json.dumps(states, indent=2, default=str)

            # Use atomic write: write to temp file then rename
            # This prevents partial writes from corrupting the state
            import tempfile

            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.storage_path.parent, prefix=".circuit_breaker_", suffix=".tmp"
            )

            try:
                # Write to temp file
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: Path(temp_path).write_text(content, "utf-8")
                )

                # Atomic rename (on POSIX systems, rename is atomic)
                await loop.run_in_executor(
                    None, lambda: Path(temp_path).replace(self.storage_path)
                )

            finally:
                # Clean up temp file if it still exists
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass  # Best effort cleanup

        except (OSError, IOError) as e:
            logger.error(f"Failed to save circuit breaker states: {e}")
            raise StorageError(f"Failed to write state file: {e}") from e
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize circuit breaker states: {e}")
            raise SerializationError(f"Failed to serialize states: {e}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error saving circuit breaker states: {e}", exc_info=True
            )
            raise CircuitBreakerStateError(f"Unexpected save error: {e}") from e

    async def _auto_save_loop(self, interval_seconds: int) -> None:
        """Background loop for auto-saving states"""
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                # Auto-save is mainly for ensuring periodic writes
                # The main saving happens in save_state()
                logger.debug("Circuit breaker auto-save tick")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error in circuit breaker auto-save loop: {e}",
                    exc_info=True,
                )
                # Continue loop despite errors
                await asyncio.sleep(5)  # Brief pause to avoid tight error loops


# Global store instance
_store: Optional[CircuitBreakerStore] = None


def get_circuit_breaker_store() -> CircuitBreakerStore:
    """Get the global circuit breaker store instance"""
    global _store
    if _store is None:
        _store = CircuitBreakerStore()
    return _store


async def save_circuit_breaker_state(
    backend: str, failure_count: int, last_failure_time: Optional[datetime], state: str
) -> None:
    """Helper function to save circuit breaker state

    Args:
        backend: Backend name
        failure_count: Number of failures
        last_failure_time: Time of last failure
        state: Circuit breaker state (CLOSED, OPEN, HALF_OPEN)
    """
    store = get_circuit_breaker_store()
    state_data = {
        "failure_count": failure_count,
        "last_failure_time": last_failure_time.isoformat()
        if last_failure_time
        else None,
        "state": state,
    }
    await store.save_state(backend, state_data)


async def load_circuit_breaker_state(backend: str) -> Optional[Dict[str, Any]]:
    """Helper function to load circuit breaker state

    Args:
        backend: Backend name

    Returns:
        State dictionary or None if not found
    """
    store = get_circuit_breaker_store()
    return await store.load_state(backend)
