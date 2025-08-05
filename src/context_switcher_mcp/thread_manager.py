"""Thread management for parallel LLM execution with circuit breaker pattern"""

import asyncio
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .models import Thread, ModelBackend
from .config import get_config
from .circular_buffer import CircularBuffer
from .aorp import create_error_response
from .circuit_breaker_store import (
    save_circuit_breaker_state,
    load_circuit_breaker_state,
)
from .backend_interface import get_backend_interface
from .exceptions import (
    CircuitBreakerStateError,
    CircuitBreakerOpenError,
    OrchestrationError,
    ModelBackendError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelRateLimitError,
    ModelAuthenticationError,
)

logger = logging.getLogger(__name__)

# Constants
NO_RESPONSE = "[NO_RESPONSE]"


@dataclass
class ThreadMetrics:
    """Metrics for thread execution tracking"""

    thread_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    model_backend: Optional[str] = None
    retry_count: int = 0

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time in seconds"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class ThreadOrchestrationMetrics:
    """Aggregate metrics for thread orchestration operations"""

    session_id: str
    operation_type: str  # 'broadcast', 'synthesis', 'single_thread'
    start_time: float
    end_time: Optional[float] = None
    thread_metrics: Dict[str, ThreadMetrics] = field(default_factory=dict)
    total_threads: int = 0
    successful_threads: int = 0
    failed_threads: int = 0
    abstained_threads: int = 0

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate total operation execution time"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_threads == 0:
            return 0.0
        return (self.successful_threads / self.total_threads) * 100


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for model backends"""

    backend: ModelBackend
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
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
                    datetime.utcnow() - self.last_failure_time
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
        except (OSError, IOError) as e:
            from .security import sanitize_error_message

            logger.error(
                f"Failed to save circuit breaker state: {sanitize_error_message(str(e))}"
            )
            raise CircuitBreakerStateError(
                f"Failed to save circuit breaker state: {e}"
            ) from e

    async def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
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
        except (OSError, IOError) as e:
            from .security import sanitize_error_message

            logger.error(
                f"Failed to save circuit breaker state: {sanitize_error_message(str(e))}"
            )
            raise CircuitBreakerStateError(
                f"Failed to save circuit breaker state: {e}"
            ) from e


class ThreadManager:
    """Manages parallel thread execution with circuit breaker pattern"""

    def __init__(self, max_retries: int = None, retry_delay: float = None):
        """Initialize thread manager

        Args:
            max_retries: Maximum number of retries for failed calls (uses config default if None)
            retry_delay: Initial delay between retries (uses config default if None)
        """
        config = get_config()
        self.max_retries = (
            max_retries if max_retries is not None else config.retry.max_retries
        )
        self.retry_delay = (
            retry_delay if retry_delay is not None else config.retry.initial_delay
        )
        self.backends = {
            ModelBackend.BEDROCK: self._call_unified_backend,
            ModelBackend.LITELLM: self._call_unified_backend,
            ModelBackend.OLLAMA: self._call_unified_backend,
        }

        # Circuit breakers for each backend
        self.circuit_breakers: Dict[ModelBackend, CircuitBreakerState] = {
            backend: CircuitBreakerState(backend=backend) for backend in ModelBackend
        }

        # Circuit breaker states will be loaded on first use
        self._circuit_breaker_states_loaded = False

        # Metrics storage with circular buffer to prevent memory leaks
        self.metrics_history = CircularBuffer[ThreadOrchestrationMetrics](
            config.metrics.max_history_size
        )
        self.metrics_lock = asyncio.Lock()  # Protect metrics operations

    async def broadcast_message(
        self, threads: Dict[str, Thread], message: str, session_id: str = "unknown"
    ) -> Dict[str, str]:
        """Broadcast message to all threads and collect responses"""
        metrics = ThreadOrchestrationMetrics(
            session_id=session_id,
            operation_type="broadcast",
            start_time=time.time(),
            total_threads=len(threads),
        )

        tasks = []
        thread_names = []

        for name, thread in threads.items():
            thread.add_message("user", message)

            task = self._get_thread_response_with_metrics(thread, metrics)
            tasks.append(task)
            thread_names.append(name)

        # Execute all threads in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Build response dictionary and update metrics
        result = {}
        for name, response in zip(thread_names, responses):
            if isinstance(response, Exception):
                from .security import sanitize_error_message

                sanitized_error = sanitize_error_message(str(response))
                logger.error(f"Error in thread {name}: {sanitized_error}")
                result[name] = f"ERROR: {sanitized_error}"
                metrics.failed_threads += 1
            else:
                # Add assistant response to thread history
                threads[name].add_message("assistant", response)
                result[name] = response

                # Classify response type for metrics
                if NO_RESPONSE in response:
                    metrics.abstained_threads += 1
                elif response.startswith("ERROR:"):
                    metrics.failed_threads += 1
                else:
                    metrics.successful_threads += 1

        # Finalize metrics atomically
        metrics.end_time = time.time()
        async with self.metrics_lock:
            self._store_metrics(metrics)

        # Log performance summary
        logger.info(
            f"Thread broadcast completed: {metrics.execution_time:.2f}s, "
            f"Success: {metrics.successful_threads}/{metrics.total_threads}, "
            f"Rate: {metrics.success_rate:.1f}%"
        )

        return result

    async def broadcast_message_stream(
        self, threads: Dict[str, Thread], message: str, session_id: str = "unknown"
    ):
        """
        Broadcast message to all threads with streaming responses

        Yields events in format:
        {
            "type": "chunk" | "complete" | "error" | "start",
            "thread_name": str,
            "content": str,
            "timestamp": float
        }
        """
        metrics = ThreadOrchestrationMetrics(
            session_id=session_id,
            operation_type="broadcast_stream",
            start_time=time.time(),
            total_threads=len(threads),
        )

        tasks = []
        for name, thread in threads.items():
            thread.add_message("user", message)

            # Yield start event for this thread
            yield {
                "type": "start",
                "thread_name": name,
                "content": "",
                "timestamp": time.time(),
            }

            task = asyncio.create_task(self._stream_from_thread(thread, name))
            tasks.append(task)

        # Stream responses as they arrive
        async def stream_handler(task):
            try:
                if hasattr(task, "__aiter__"):  # Streaming response
                    async for event in task:
                        yield event
                else:  # Non-streaming response
                    result = await task
                    yield result
            except (ModelBackendError, OrchestrationError) as e:
                yield {
                    "type": "error",
                    "thread_name": "unknown",
                    "content": str(e),
                    "timestamp": time.time(),
                }
            except asyncio.CancelledError:
                # Let cancellation propagate
                raise
            except Exception as e:
                # Unexpected errors - log and wrap
                logger.error(f"Unexpected error in stream handler: {e}", exc_info=True)
                yield {
                    "type": "error",
                    "thread_name": "unknown",
                    "content": f"Unexpected error: {str(e)}",
                    "timestamp": time.time(),
                }

        # Use asyncio.as_completed to yield results as they come
        for task in asyncio.as_completed(tasks):
            async for event in stream_handler(task):
                yield event

        # Finalize metrics atomically
        metrics.end_time = time.time()
        async with self.metrics_lock:
            self._store_metrics(metrics)

    async def _stream_from_thread(self, thread: Thread, thread_name: str):
        """Stream response from a single thread using unified backend"""
        try:
            backend_interface = get_backend_interface(thread.model_backend.value)
            async for event in backend_interface.call_model_stream(thread):
                event["timestamp"] = time.time()
                yield event
        except ModelBackendError as e:
            # Expected model errors - pass through
            yield {
                "type": "error",
                "thread_name": thread_name,
                "content": str(e),
                "timestamp": time.time(),
            }
        except asyncio.CancelledError:
            # Let cancellation propagate
            raise
        except Exception as e:
            # Unexpected errors - log and wrap
            logger.error(
                f"Unexpected error streaming from thread {thread_name}: {e}",
                exc_info=True,
            )
            yield {
                "type": "error",
                "thread_name": thread_name,
                "content": f"Unexpected error: {str(e)}",
                "timestamp": time.time(),
            }

    async def _get_thread_response_with_metrics(
        self, thread: Thread, orchestration_metrics: ThreadOrchestrationMetrics
    ) -> str:
        """Get response from a single thread with metrics tracking"""
        thread_metrics = ThreadMetrics(
            thread_name=thread.name,
            start_time=time.time(),
            model_backend=thread.model_backend.value,
        )
        orchestration_metrics.thread_metrics[thread.name] = thread_metrics

        try:
            response = await self._get_thread_response(thread)
            thread_metrics.success = not response.startswith("ERROR:")
            thread_metrics.end_time = time.time()
            return response
        except (ModelBackendError, OrchestrationError, CircuitBreakerOpenError) as e:
            # Expected errors - record and re-raise
            thread_metrics.success = False
            thread_metrics.error_message = str(e)
            thread_metrics.end_time = time.time()
            raise
        except Exception as e:
            # Unexpected errors - log, record, wrap and raise
            logger.error(
                f"Unexpected error in thread {thread.name}: {e}", exc_info=True
            )
            thread_metrics.success = False
            thread_metrics.error_message = f"Unexpected error: {str(e)}"
            thread_metrics.end_time = time.time()
            raise OrchestrationError(
                f"Unexpected error in thread {thread.name}: {e}"
            ) from e

    async def _get_thread_response(self, thread: Thread) -> str:
        """Get response from a single thread with retry logic and circuit breaker"""
        # Ensure circuit breaker states are loaded
        if not self._circuit_breaker_states_loaded:
            await self._load_circuit_breaker_states()
            self._circuit_breaker_states_loaded = True

        backend_fn = self.backends.get(thread.model_backend)
        if not backend_fn:
            raise ValueError(f"Unknown model backend: {thread.model_backend}")

        circuit_breaker = self.circuit_breakers[thread.model_backend]
        if not circuit_breaker.should_allow_request():
            logger.warning(
                f"Circuit breaker OPEN for {thread.model_backend.value}, failing fast"
            )
            error_response = create_error_response(
                error_message=f"Circuit breaker is OPEN for {thread.model_backend.value} backend due to repeated failures",
                error_type="circuit_breaker_open",
                context={
                    "backend": thread.model_backend.value,
                    "failure_count": circuit_breaker.failure_count,
                },
                recoverable=True,
            )
            return f"AORP_ERROR: {error_response}"

        # Try with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await backend_fn(thread)
                # Record success in circuit breaker
                await circuit_breaker.record_success()
                return response
            except ModelAuthenticationError:
                # Authentication errors - don't retry
                raise
            except (ModelConnectionError, ModelTimeoutError, ModelRateLimitError) as e:
                # Retryable errors
                last_error = e
                error_str = str(e).lower()

                # Record failure in circuit breaker for transient errors
                await circuit_breaker.record_failure()

            except ModelBackendError as e:
                # Other model errors - check if retryable
                last_error = e
                error_str = str(e).lower()

            except Exception as e:
                # Unexpected errors - wrap and treat as non-retryable
                logger.error(
                    f"Unexpected error calling backend for thread {thread.name}: {e}",
                    exc_info=True,
                )
                raise OrchestrationError(f"Unexpected backend error: {e}") from e

            if last_error:
                error_str = str(last_error).lower()

                # Record failure in circuit breaker for retryable errors
                if not any(
                    term in error_str
                    for term in [
                        "api_key",
                        "credentials",
                        "not found",
                        "invalid",
                        "unauthorized",
                        "forbidden",
                        "model not found",
                    ]
                ):
                    await circuit_breaker.record_failure()

                # Don't retry on non-transient errors
                if any(
                    term in error_str
                    for term in [
                        "api_key",
                        "credentials",
                        "not found",
                        "invalid",
                        "unauthorized",
                        "forbidden",
                        "model not found",
                    ]
                ):
                    from .security import sanitize_error_message

                    logger.error(
                        f"Non-retryable error for {thread.name}: {sanitize_error_message(str(last_error))}"
                    )
                    error_response = create_error_response(
                        error_message=str(last_error),
                        error_type="configuration_error",
                        context={
                            "thread": thread.name,
                            "backend": thread.model_backend.value,
                        },
                        recoverable=False,
                    )
                    return f"AORP_ERROR: {error_response}"

                # Retry on transient errors
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {thread.name}: {last_error}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    from .security import sanitize_error_message

                    logger.error(
                        f"All {self.max_retries} attempts failed for {thread.name}: {sanitize_error_message(str(last_error))}"
                    )

        # If all retries failed, return AORP error response
        from .security import sanitize_error_message

        error_response = create_error_response(
            error_message=f"Failed after {self.max_retries} attempts: {sanitize_error_message(str(last_error))}",
            error_type="retry_exhausted",
            context={
                "attempts": self.max_retries,
                "thread": thread.name,
                "backend": thread.model_backend.value,
            },
            recoverable=True,
        )
        return f"AORP_ERROR: {error_response}"

    async def _call_unified_backend(self, thread: Thread) -> str:
        """Call backend using unified interface"""
        try:
            backend_interface = get_backend_interface(thread.model_backend.value)
            return await backend_interface.call_model(thread)
        except ModelBackendError:
            # The backend interface already formats errors appropriately
            raise
        except Exception as e:
            # Unexpected errors - wrap
            logger.error(
                f"Unexpected error in unified backend call: {e}", exc_info=True
            )
            raise OrchestrationError(f"Backend call failed unexpectedly: {e}") from e

    def _store_metrics(self, metrics: ThreadOrchestrationMetrics) -> None:
        """Store metrics in circular buffer (automatically maintains size limit)"""
        self.metrics_history.append(metrics)

    async def get_thread_metrics(self, last_n: int = 10) -> Dict[str, any]:
        """Get thread-level performance metrics"""
        async with self.metrics_lock:
            if self.metrics_history.is_empty():
                return {"message": "No thread metrics available"}

            recent_metrics = self.metrics_history.get_recent(last_n)

        # Calculate aggregate statistics (outside lock)
        total_operations = len(recent_metrics)
        avg_execution_time = (
            sum(m.execution_time for m in recent_metrics if m.execution_time)
            / total_operations
        )
        overall_success_rate = (
            sum(m.success_rate for m in recent_metrics) / total_operations
        )

        # Backend performance breakdown
        backend_stats = {}
        for metrics in recent_metrics:
            for thread_name, thread_metrics in metrics.thread_metrics.items():
                backend = thread_metrics.model_backend
                if backend not in backend_stats:
                    backend_stats[backend] = {
                        "count": 0,
                        "success": 0,
                        "total_time": 0.0,
                    }

                backend_stats[backend]["count"] += 1
                if thread_metrics.success:
                    backend_stats[backend]["success"] += 1
                if thread_metrics.execution_time:
                    backend_stats[backend][
                        "total_time"
                    ] += thread_metrics.execution_time

        # Calculate backend success rates and avg times
        for backend, stats in backend_stats.items():
            stats["success_rate"] = (
                (stats["success"] / stats["count"]) * 100 if stats["count"] > 0 else 0
            )
            stats["avg_time"] = (
                stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            )

        return {
            "thread_summary": {
                "total_operations": total_operations,
                "avg_execution_time_seconds": round(avg_execution_time, 2),
                "overall_success_rate_percent": round(overall_success_rate, 1),
            },
            "backend_performance": backend_stats,
            "metrics_storage": {
                "current_size": len(self.metrics_history),
                "max_size": self.metrics_history.maxsize,
                "utilization_percent": round(
                    (len(self.metrics_history) / self.metrics_history.maxsize) * 100, 1
                ),
                "memory_usage_mb": round(self.metrics_history.memory_usage_mb, 2),
            },
        }

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, any]]:
        """Get current circuit breaker status for all backends"""
        circuit_status = {}
        for backend, breaker in self.circuit_breakers.items():
            circuit_status[backend.value] = {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat()
                if breaker.last_failure_time
                else None,
            }
        return circuit_status

    def reset_circuit_breakers(self) -> Dict[str, str]:
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

    async def save_all_circuit_breaker_states(self) -> None:
        """Manually save all circuit breaker states"""
        for backend, breaker in self.circuit_breakers.items():
            await save_circuit_breaker_state(
                backend.value,
                breaker.failure_count,
                breaker.last_failure_time,
                breaker.state,
            )
