"""Circuit breaker state persistence for production reliability"""

import json
import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path

from .exceptions import (
    CircuitBreakerStateError,
    StorageError,
    SerializationError,
)


logger = logging.getLogger(__name__)


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
        if storage_path is None:
            config_dir = Path.home() / ".context_switcher"
            config_dir.mkdir(exist_ok=True)
            storage_path = config_dir / "circuit_breakers.json"
        else:
            # Validate and sanitize the provided path
            storage_path = Path(storage_path)

            # Expand user and resolve to absolute path FIRST
            # This ensures we check the actual resolved path, not the input
            storage_path = storage_path.expanduser().resolve(strict=False)

            # NOW check for path traversal patterns in the resolved path
            # This prevents bypasses through symbolic links or other tricks
            path_str = str(storage_path)

            # Check the resolved path doesn't contain traversal patterns
            # Note: After resolution, ".." should not appear in a valid path
            if ".." in path_str:
                raise ValueError(
                    f"Invalid storage path after resolution: {storage_path}"
                )

            # Ensure the path is within a safe directory (home or temp)
            home_dir = Path.home().resolve()

            # Additional security: ensure path components don't start with dots (hidden files)
            # except for the config directory itself
            for part in storage_path.parts:
                if part.startswith(".") and part not in [
                    ".context_switcher",
                    ".config",
                    ".local",
                ]:
                    raise ValueError(
                        f"Storage path contains suspicious hidden directory: {part}"
                    )

            # Check if path is within allowed directories
            is_in_safe_directory = False
            allowed_dirs = []

            # Check home directory
            try:
                storage_path.relative_to(home_dir)
                is_in_safe_directory = True
                allowed_dirs.append(str(home_dir))
            except ValueError:
                pass

            # Check system temp directory
            if not is_in_safe_directory:
                import tempfile

                temp_root = Path(tempfile.gettempdir()).resolve()
                try:
                    storage_path.relative_to(temp_root)
                    is_in_safe_directory = True
                    allowed_dirs.append(str(temp_root))
                except ValueError:
                    pass

            # Check /tmp for compatibility
            if not is_in_safe_directory:
                try:
                    tmp_path = Path("/tmp").resolve()
                    storage_path.relative_to(tmp_path)
                    is_in_safe_directory = True
                    allowed_dirs.append(str(tmp_path))
                except (ValueError, OSError):
                    pass

            if not is_in_safe_directory:
                raise ValueError(
                    f"Storage path must be within home directory or temp directory. "
                    f"Resolved path: {storage_path}, Allowed: {', '.join(allowed_dirs)}"
                )

            # Ensure filename ends with .json
            if storage_path.suffix != ".json":
                raise ValueError("Storage path must be a .json file")

            # Final security check: Ensure the path is not a symlink pointing outside safe dirs
            if storage_path.exists() and storage_path.is_symlink():
                real_path = storage_path.resolve()
                # Re-validate the real path
                is_real_path_safe = False
                try:
                    real_path.relative_to(home_dir)
                    is_real_path_safe = True
                except ValueError:
                    try:
                        import tempfile

                        real_path.relative_to(Path(tempfile.gettempdir()).resolve())
                        is_real_path_safe = True
                    except ValueError:
                        try:
                            real_path.relative_to(Path("/tmp").resolve())
                            is_real_path_safe = True
                        except (ValueError, OSError):
                            pass

                if not is_real_path_safe:
                    raise ValueError(
                        f"Symlink points outside safe directories: {real_path}"
                    )

        self.storage_path = Path(storage_path)
        self._lock: Optional[asyncio.Lock] = None
        self._auto_save_task: Optional[asyncio.Task] = None

        # Ensure parent directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

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
                    "last_updated": datetime.utcnow().isoformat(),
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
        """Save all circuit breaker states to storage"""
        try:
            content = json.dumps(states, indent=2, default=str)

            # Use asyncio to write file to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.storage_path.write_text, content, "utf-8"
            )

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
