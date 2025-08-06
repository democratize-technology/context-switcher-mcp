"""High-level orchestration facade for the Context Switcher MCP"""

import logging
from typing import Dict

from .models import Thread
from .thread_manager import ThreadManager
from .perspective_orchestrator import PerspectiveOrchestrator
from .response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

# Constants for backward compatibility
NO_RESPONSE = "[NO_RESPONSE]"


class ThreadOrchestrator:
    """High-level orchestration facade combining thread management, perspective orchestration, and response formatting"""

    def __init__(self, max_retries: int = None, retry_delay: float = None):
        """Initialize orchestrator facade

        Args:
            max_retries: Maximum number of retries for failed calls (uses config default if None)
            retry_delay: Initial delay between retries (uses config default if None)
        """
        # Initialize component services
        self.thread_manager = ThreadManager(max_retries, retry_delay)
        self.perspective_orchestrator = PerspectiveOrchestrator(
            thread_manager=self.thread_manager
        )
        self.response_formatter = ResponseFormatter()

        # Delegate properties to thread manager for backward compatibility
        self.circuit_breakers = self.thread_manager.circuit_breakers
        self.backends = self.thread_manager.backends
        self.metrics_history = self.thread_manager.metrics_history
        self.metrics_lock = self.thread_manager.metrics_lock

    async def broadcast_message(
        self,
        threads: Dict[str, Thread],
        message: str,
        session_id: str = "unknown",
        topic: str = None,
    ) -> Dict[str, str]:
        """Broadcast message to all threads and collect responses"""
        return await self.perspective_orchestrator.broadcast_to_perspectives(
            threads, message, session_id, topic
        )

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
        async for (
            event
        ) in self.perspective_orchestrator.broadcast_to_perspectives_stream(
            threads, message, session_id
        ):
            yield event

    async def get_performance_metrics(self, last_n: int = 10) -> Dict[str, any]:
        """Get performance metrics for recent operations"""
        # Combine metrics from thread manager and perspective orchestrator
        thread_metrics = await self.thread_manager.get_thread_metrics(last_n)
        perspective_metrics = (
            await self.perspective_orchestrator.get_perspective_metrics(last_n)
        )

        # Add circuit breaker status
        circuit_status = self.thread_manager.get_circuit_breaker_status()

        return {
            "thread_level": thread_metrics,
            "perspective_level": perspective_metrics,
            "circuit_breakers": circuit_status,
        }

    def reset_circuit_breakers(self) -> Dict[str, str]:
        """Reset all circuit breakers to CLOSED state"""
        return self.thread_manager.reset_circuit_breakers()

    async def save_all_circuit_breaker_states(self) -> None:
        """Manually save all circuit breaker states"""
        await self.thread_manager.save_all_circuit_breaker_states()
