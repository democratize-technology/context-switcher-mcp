"""Streaming coordination for real-time response handling"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, List

from .models import Thread
from .backend_factory import BackendFactory
from .exceptions import ModelBackendError, OrchestrationError
from .error_helpers import wrap_generic_exception
from .error_logging import log_error_with_context
from .error_context import error_context
from .security import sanitize_error_message

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
        metrics = self._initialize_stream_metrics(session_id, len(threads))

        # Setup and yield start events, collect tasks
        tasks = []
        async for start_event, task in self._setup_streaming_tasks(threads, message):
            if start_event:
                yield start_event
            if task:
                tasks.append(task)

        # Stream results as they complete
        async for event in self._process_streaming_tasks(tasks, metrics):
            yield event

        await self._finalize_stream_metrics(metrics)

    def _initialize_stream_metrics(
        self, session_id: str, thread_count: int
    ) -> Optional[Any]:
        """Initialize metrics for streaming operation"""
        if not self.metrics_manager:
            return None

        return self.metrics_manager.create_orchestration_metrics(
            session_id=session_id,
            operation_type="broadcast_stream",
            thread_count=thread_count,
        )

    async def _setup_streaming_tasks(
        self, threads: Dict[str, Thread], message: str
    ) -> AsyncGenerator[tuple, None]:
        """Setup streaming tasks for all threads"""
        for name, thread in threads.items():
            self._prepare_thread_for_streaming(thread, message)

            # Create start event
            start_event = {
                "type": "start",
                "thread_name": name,
                "content": "",
                "timestamp": time.time(),
            }

            # Create streaming task
            task = asyncio.create_task(self._stream_thread_wrapper(thread, name))

            yield start_event, task

    def _prepare_thread_for_streaming(self, thread: Thread, message: str) -> None:
        """Prepare thread by adding message if needed"""
        should_add_message = message and (
            not thread.conversation_history
            or thread.conversation_history[-1].get("content") != message
        )

        if should_add_message:
            thread.add_message("user", message)

    async def _stream_thread_wrapper(
        self, thread: Thread, name: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Wrapper coroutine to stream from a thread with structured error handling"""
        events = []
        try:
            async for event in self._stream_from_thread(thread, name):
                events.append(event)
        except ModelBackendError as e:
            # Expected model errors - create structured error event
            log_error_with_context(
                error=e,
                operation_name="thread_streaming",
                additional_context={
                    "thread_name": name,
                    "backend": thread.model_backend,
                    "error_category": "model_backend",
                },
            )
            events.append(
                self._create_error_event(
                    name, f"Model error: {sanitize_error_message(str(e))}"
                )
            )
        except Exception as e:
            # Unexpected errors - wrap and log
            specific_error = wrap_generic_exception(
                e,
                f"thread_streaming_{name}",
                OrchestrationError,
                context={"thread_name": name, "backend": thread.model_backend},
            )
            log_error_with_context(
                error=specific_error,
                operation_name="thread_streaming",
                additional_context={
                    "thread_name": name,
                    "backend": thread.model_backend,
                    "error_category": "unexpected",
                },
            )
            events.append(
                self._create_error_event(name, sanitize_error_message(str(e)))
            )

        return name, events

    def _create_error_event(
        self, thread_name: str, error_message: str
    ) -> Dict[str, Any]:
        """Create error event for streaming"""
        return {
            "type": "error",
            "thread_name": thread_name,
            "content": error_message,
            "timestamp": time.time(),
        }

    async def _process_streaming_tasks(
        self, tasks: List[Any], metrics: Optional[Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process streaming tasks and update metrics"""
        for task in asyncio.as_completed(tasks):
            name, events = await task
            for event in events:
                yield event
                self._update_metrics_from_event(metrics, event)

    def _update_metrics_from_event(
        self, metrics: Optional[Any], event: Dict[str, Any]
    ) -> None:
        """Update metrics based on streaming event"""
        if not metrics:
            return

        event_type = event.get("type")
        if event_type == "complete":
            metrics.successful_threads += 1
        elif event_type == "error":
            metrics.failed_threads += 1

    async def _finalize_stream_metrics(self, metrics: Optional[Any]) -> None:
        """Finalize and store streaming metrics"""
        if self.metrics_manager and metrics:
            self.metrics_manager.finalize_metrics(metrics)
            await self.metrics_manager.store_metrics(metrics)

    async def stream_single_thread(
        self, thread: Thread, thread_name: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from a single thread"""
        async for event in self._stream_from_thread(thread, thread_name):
            yield event

    async def _stream_from_thread(
        self, thread: Thread, thread_name: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from a single thread using unified backend with structured error handling"""
        async with error_context(
            operation_name="single_thread_streaming",
            user_context={
                "thread_name": thread_name,
                "backend": thread.model_backend,
                "conversation_length": len(thread.conversation_history),
            },
        ) as ctx:
            try:
                backend = BackendFactory.get_backend(thread.model_backend)

                event_count = 0
                async for event in backend.call_model_stream(thread):
                    event["timestamp"] = time.time()
                    event["thread_name"] = thread_name
                    event_count += 1
                    yield event

                ctx["events_streamed"] = event_count

            except ModelBackendError as e:
                # Expected model errors - create structured error event
                ctx["model_error"] = True
                log_error_with_context(
                    error=e,
                    operation_name="single_thread_streaming",
                    additional_context={
                        "thread_name": thread_name,
                        "backend": thread.model_backend,
                        "error_category": "model_backend",
                    },
                )
                yield {
                    "type": "error",
                    "thread_name": thread_name,
                    "content": f"Model backend error: {sanitize_error_message(str(e))}",
                    "timestamp": time.time(),
                    "error_category": "model_backend",
                }
            except asyncio.CancelledError:
                # Let cancellation propagate
                ctx["cancelled"] = True
                raise
            except Exception as e:
                # Unexpected errors - wrap and create structured error
                ctx["unexpected_error"] = True
                specific_error = wrap_generic_exception(
                    e,
                    f"single_thread_streaming_{thread_name}",
                    OrchestrationError,
                    context={
                        "thread_name": thread_name,
                        "backend": thread.model_backend,
                    },
                )

                log_error_with_context(
                    error=specific_error,
                    operation_name="single_thread_streaming",
                    additional_context={
                        "thread_name": thread_name,
                        "backend": thread.model_backend,
                        "error_category": "unexpected",
                    },
                )

                yield {
                    "type": "error",
                    "thread_name": thread_name,
                    "content": f"Streaming error: {sanitize_error_message(str(e))}",
                    "timestamp": time.time(),
                    "error_category": "unexpected",
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
