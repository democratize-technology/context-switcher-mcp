"""Refactored Thread management with extracted components"""

import logging
from typing import Dict

from .circuit_breaker_manager import CircuitBreakerManager
from .constants import NO_RESPONSE
from .metrics_manager import MetricsManager
from .models import Thread
from .streaming_coordinator import StreamingCoordinator
from .thread_lifecycle_manager import ThreadLifecycleManager

logger = logging.getLogger(__name__)


class ThreadManager:
    """Refactored thread manager using extracted components for better separation of concerns"""

    def __init__(self, max_retries: int = None, retry_delay: float = None):
        """Initialize thread manager with extracted components

        Args:
            max_retries: Maximum number of retries for failed calls (uses config default if None)
            retry_delay: Initial delay between retries (uses config default if None)
        """
        # Initialize extracted components
        self.metrics_manager = MetricsManager()
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.streaming_coordinator = StreamingCoordinator(self.metrics_manager)
        self.thread_lifecycle_manager = ThreadLifecycleManager(
            circuit_breaker_manager=self.circuit_breaker_manager,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    async def broadcast_message(
        self, threads: Dict[str, Thread], message: str, session_id: str = "unknown"
    ) -> Dict[str, str]:
        """Broadcast message to all threads and collect responses"""
        # Create metrics tracking
        metrics = self.metrics_manager.create_orchestration_metrics(
            session_id=session_id, operation_type="broadcast", thread_count=len(threads)
        )

        # Execute threads using lifecycle manager
        result = await self.thread_lifecycle_manager.execute_threads_parallel(
            threads, message
        )

        # Update metrics based on results
        for name, response in result.items():
            thread_metrics = self.metrics_manager.create_thread_metrics(
                thread_name=name, model_backend=threads[name].model_backend.value
            )
            thread_metrics.end_time = (
                thread_metrics.start_time
            )  # Simplified for refactored version
            metrics.thread_metrics[name] = thread_metrics

            # Classify response type for metrics
            if response.startswith("ERROR:"):
                metrics.failed_threads += 1
                thread_metrics.success = False
            elif NO_RESPONSE in response:
                metrics.abstained_threads += 1
                thread_metrics.success = True
            else:
                metrics.successful_threads += 1
                thread_metrics.success = True

        # Finalize and store metrics
        self.metrics_manager.finalize_metrics(metrics)
        await self.metrics_manager.store_metrics(metrics)
        self.metrics_manager.log_performance_summary(metrics)

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
        # Delegate to streaming coordinator
        async for event in self.streaming_coordinator.broadcast_stream(
            threads, message, session_id
        ):
            yield event

    # Stream from thread method delegated to StreamingCoordinator
    # This method is no longer needed in this class

    # Thread response with metrics now handled by lifecycle manager and metrics manager
    # This method is no longer needed in this class

    # Thread response logic now delegated to ThreadLifecycleManager
    async def get_single_thread_response(self, thread: Thread) -> str:
        """Get response from a single thread (public interface)"""
        return await self.thread_lifecycle_manager.execute_thread(thread)

    # Backend calls now handled by ThreadLifecycleManager
    # This method is no longer needed in this class

    # Metrics storage now handled by MetricsManager
    # This method is no longer needed in this class

    async def get_thread_metrics(self, last_n: int = 10) -> Dict[str, any]:
        """Get thread-level performance metrics"""
        return await self.metrics_manager.get_performance_metrics(last_n)

    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, any]]:
        """Get current circuit breaker status for all backends"""
        return self.circuit_breaker_manager.get_status_summary()

    def reset_circuit_breakers(self) -> Dict[str, str]:
        """Reset all circuit breakers to CLOSED state"""
        return self.circuit_breaker_manager.reset_all_circuit_breakers()

    # Circuit breaker state loading now handled by CircuitBreakerManager
    # This method is no longer needed in this class

    async def save_all_circuit_breaker_states(self) -> None:
        """Manually save all circuit breaker states"""
        await self.circuit_breaker_manager.save_all_states()
