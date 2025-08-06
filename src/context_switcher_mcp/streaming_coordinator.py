"""Streaming coordination for real-time response handling"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict

from .models import Thread
from .backend_factory import BackendFactory
from .exceptions import ModelBackendError

logger = logging.getLogger(__name__)


class StreamingCoordinator:
    """Coordinates streaming responses from multiple threads"""

    def __init__(self, metrics_manager=None):
        """Initialize streaming coordinator

        Args:
            metrics_manager: Optional metrics manager for tracking streaming operations
        """
        self.metrics_manager = metrics_manager

    async def broadcast_stream(
        self, threads: Dict[str, Thread], message: str, session_id: str = "unknown"
    ) -> AsyncGenerator[Dict[str, Any], None]:
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
        # Initialize metrics if manager is available
        metrics = None
        if self.metrics_manager:
            metrics = self.metrics_manager.create_orchestration_metrics(
                session_id=session_id,
                operation_type="broadcast_stream",
                thread_count=len(threads),
            )

        # Create tasks for streaming from each thread
        async def stream_thread_wrapper(thread, name):
            """Wrapper coroutine to stream from a thread"""
            events = []
            try:
                async for event in self._stream_from_thread(thread, name):
                    events.append(event)
            except Exception as e:
                events.append(
                    {
                        "type": "error",
                        "thread_name": name,
                        "content": str(e),
                        "timestamp": time.time(),
                    }
                )
            return name, events

        tasks = []
        for name, thread in threads.items():
            # Only add message if not already present (CoT may have added it)
            if message and (
                not thread.conversation_history
                or thread.conversation_history[-1].get("content") != message
            ):
                thread.add_message("user", message)

            # Yield start event for this thread
            yield {
                "type": "start",
                "thread_name": name,
                "content": "",
                "timestamp": time.time(),
            }

            # Create task for this thread
            task = asyncio.create_task(stream_thread_wrapper(thread, name))
            tasks.append(task)

        # Yield events as threads complete
        for task in asyncio.as_completed(tasks):
            name, events = await task
            for event in events:
                yield event

                # Update metrics if available
                if metrics:
                    if event.get("type") == "complete":
                        metrics.successful_threads += 1
                    elif event.get("type") == "error":
                        metrics.failed_threads += 1

        # Finalize metrics if manager is available
        if self.metrics_manager and metrics:
            self.metrics_manager.finalize_metrics(metrics)
            await self.metrics_manager.store_metrics(metrics)

    async def stream_single_thread(
        self, thread: Thread, thread_name: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from a single thread"""
        async for event in self._stream_from_thread(thread, thread_name):
            yield event

    async def _stream_from_thread(self, thread: Thread, thread_name: str):
        """Stream response from a single thread using unified backend"""
        try:
            backend = BackendFactory.get_backend(thread.model_backend)
            async for event in backend.call_model_stream(thread):
                event["timestamp"] = time.time()
                event["thread_name"] = thread_name
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

    def create_stream_event(
        self, event_type: str, thread_name: str, content: str = "", **kwargs
    ) -> Dict[str, Any]:
        """Create a standardized streaming event"""
        event = {
            "type": event_type,
            "thread_name": thread_name,
            "content": content,
            "timestamp": time.time(),
        }
        event.update(kwargs)
        return event
