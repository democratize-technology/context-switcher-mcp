"""Circuit breaker pattern implementation for resilient backend failure handling"""

import logging
from typing import Dict
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import ModelBackend
from .config import get_config
from .circuit_breaker_store import (
    save_circuit_breaker_state,
    load_circuit_breaker_state,
)

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for model backends"""

    backend: ModelBackend
    failure_count: int = 0
    last_failure_time: datetime = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = None
    timeout_seconds: int = None

    def __post_init__(self):
        """Initialize with config defaults if not provided"""
        config = get_config()
        if self.failure_threshold is None:
            self.failure_threshold = config.circuit_breaker.failure_threshold
        if self.timeout_seconds is None:
            self.timeout_seconds = config.circuit_breaker.timeout_seconds

    def should_allow_request(self) -> bool:
        """Check if requests should be allowed through circuit breaker"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time:
                time_since_failure = (
                    datetime.now(timezone.utc) - self.last_failure_time
                ).total_seconds()
                if time_since_failure > self.timeout_seconds:
                    self.state = "HALF_OPEN"
                    return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        return False

    async def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            # Only transition to CLOSED from HALF_OPEN after successful request
            self.state = "CLOSED"
        self.last_failure_time = None

        # Save state with proper error handling
        try:
            await save_circuit_breaker_state(
                self.backend.value,
                self.failure_count,
                self.last_failure_time,
                self.state,
            )
        except Exception as e:
            from .security import sanitize_error_message

            logger.error(
                f"Failed to save circuit breaker state: {sanitize_error_message(str(e))}"
            )
            # Don't raise - just log the error

    async def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

        # Save state with proper error handling
        try:
            await save_circuit_breaker_state(
                self.backend.value,
                self.failure_count,
                self.last_failure_time,
                self.state,
            )
        except Exception as e:
            from .security import sanitize_error_message

            logger.error(
                f"Failed to save circuit breaker state: {sanitize_error_message(str(e))}"
            )
            # Don't raise - just log the error


class CircuitBreakerManager:
    """Manages circuit breaker states for all model backends"""

    def __init__(self):
        """Initialize circuit breaker manager"""
        # Circuit breakers for each backend
        self.circuit_breakers: Dict[ModelBackend, CircuitBreakerState] = {
            backend: CircuitBreakerState(backend=backend) for backend in ModelBackend
        }

        # Circuit breaker states will be loaded on first use
        self._states_loaded = False

    async def ensure_states_loaded(self) -> None:
        """Ensure circuit breaker states are loaded from persistence"""
        if not self._states_loaded:
            await self._load_circuit_breaker_states()
            self._states_loaded = True

    def get_circuit_breaker(self, backend: ModelBackend) -> CircuitBreakerState:
        """Get circuit breaker for a specific backend"""
        return self.circuit_breakers[backend]

    def should_allow_request(self, backend: ModelBackend) -> bool:
        """Check if a request should be allowed for a backend"""
        circuit_breaker = self.circuit_breakers[backend]
        return circuit_breaker.should_allow_request()

    async def record_success(self, backend: ModelBackend) -> None:
        """Record successful request for a backend"""
        circuit_breaker = self.circuit_breakers[backend]
        await circuit_breaker.record_success()

    async def record_failure(self, backend: ModelBackend) -> None:
        """Record failed request for a backend"""
        circuit_breaker = self.circuit_breakers[backend]
        await circuit_breaker.record_failure()

    def get_status_summary(self) -> Dict[str, Dict[str, any]]:
        """Get current circuit breaker status for all backends"""
        circuit_status = {}
        for backend, breaker in self.circuit_breakers.items():
            circuit_status[backend.value] = {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat()
                if breaker.last_failure_time
                else None,
                "failure_threshold": breaker.failure_threshold,
                "timeout_seconds": breaker.timeout_seconds,
            }
        return circuit_status

    def reset_all_circuit_breakers(self) -> Dict[str, str]:
        """Reset all circuit breakers to CLOSED state"""
        reset_status = {}
        for backend, breaker in self.circuit_breakers.items():
            old_state = breaker.state
            breaker.state = "CLOSED"
            breaker.failure_count = 0
            breaker.last_failure_time = None
            reset_status[backend.value] = f"{old_state} -> CLOSED"
            logger.info(
                f"Reset circuit breaker for {backend.value}: {old_state} -> CLOSED"
            )

        return reset_status

    def reset_circuit_breaker(self, backend: ModelBackend) -> str:
        """Reset a specific circuit breaker to CLOSED state"""
        breaker = self.circuit_breakers[backend]
        old_state = breaker.state
        breaker.state = "CLOSED"
        breaker.failure_count = 0
        breaker.last_failure_time = None

        status = f"{old_state} -> CLOSED"
        logger.info(f"Reset circuit breaker for {backend.value}: {status}")
        return status

    async def save_all_states(self) -> None:
        """Manually save all circuit breaker states"""
        for backend, breaker in self.circuit_breakers.items():
            try:
                await save_circuit_breaker_state(
                    backend.value,
                    breaker.failure_count,
                    breaker.last_failure_time,
                    breaker.state,
                )
            except Exception as e:
                from .security import sanitize_error_message

                logger.error(
                    f"Failed to save circuit breaker state for {backend.value}: "
                    f"{sanitize_error_message(str(e))}"
                )

    async def _load_circuit_breaker_states(self) -> None:
        """Load persisted circuit breaker states on startup"""
        try:
            for backend in ModelBackend:
                stored_state = await load_circuit_breaker_state(backend.value)
                if stored_state:
                    breaker = self.circuit_breakers[backend]
                    breaker.failure_count = stored_state.get("failure_count", 0)
                    breaker.state = stored_state.get("state", "CLOSED")

                    # Parse last_failure_time if it exists
                    if stored_state.get("last_failure_time"):
                        try:
                            breaker.last_failure_time = datetime.fromisoformat(
                                stored_state["last_failure_time"]
                            )
                        except ValueError:
                            # Invalid timestamp, reset to None
                            breaker.last_failure_time = None

                    logger.info(
                        f"Restored circuit breaker state for {backend.value}: "
                        f"state={breaker.state}, failures={breaker.failure_count}"
                    )

        except (OSError, IOError, ValueError) as e:
            # File system or parsing errors - log but continue
            logger.warning(f"Failed to load circuit breaker states: {e}")
        except Exception as e:
            # Unexpected errors - log with full trace but continue
            logger.error(
                f"Unexpected error loading circuit breaker states: {e}", exc_info=True
            )
