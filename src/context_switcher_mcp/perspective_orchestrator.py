"""Perspective orchestration service for managing multi-perspective analysis"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass

from .models import Thread
from .thread_manager import ThreadManager
from .response_formatter import ResponseFormatter
from .exceptions import OrchestrationError
from .reasoning_orchestrator import (
    PerspectiveReasoningOrchestrator,
    CoTTimeoutError,
    CoTProcessingError,
)

logger = logging.getLogger(__name__)

# Constants
NO_RESPONSE = "[NO_RESPONSE]"


@dataclass
class PerspectiveMetrics:
    """Metrics for perspective-level operations"""

    session_id: str
    operation_type: str  # 'broadcast', 'synthesis', 'single_thread'
    start_time: float
    end_time: Optional[float] = None
    total_perspectives: int = 0
    successful_perspectives: int = 0
    failed_perspectives: int = 0
    abstained_perspectives: int = 0

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate total operation execution time"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_perspectives == 0:
            return 0.0
        return (self.successful_perspectives / self.total_perspectives) * 100


class PerspectiveOrchestrator:
    """Orchestrates parallel perspective analysis using thread manager"""

    def __init__(
        self,
        thread_manager: ThreadManager = None,
        response_formatter: ResponseFormatter = None,
        enable_cot: bool = True,
        cot_timeout: float = 30.0,
    ):
        """Initialize perspective orchestrator

        Args:
            thread_manager: Thread manager for parallel execution (creates default if None)
            response_formatter: Response formatter for AORP formatting (creates default if None)
            enable_cot: Enable Chain of Thought reasoning for Bedrock (default: True)
            cot_timeout: Timeout for CoT processing in seconds (default: 30.0)
        """
        self.thread_manager = thread_manager or ThreadManager()
        self.response_formatter = response_formatter or ResponseFormatter()

        # Initialize reasoning orchestrator if enabled
        self.enable_cot = enable_cot
        self.reasoning_orchestrator = None
        if enable_cot:
            self.reasoning_orchestrator = PerspectiveReasoningOrchestrator(cot_timeout)
            if self.reasoning_orchestrator.is_available:
                logger.info(
                    "Chain of Thought reasoning enabled for perspective analysis"
                )
            else:
                logger.warning(
                    "Chain of Thought tool not available - falling back to standard analysis"
                )

        # Metrics storage
        self.metrics_history = []
        self.metrics_lock = asyncio.Lock()

    async def broadcast_to_perspectives(
        self, threads: Dict[str, Thread], message: str, session_id: str = "unknown"
    ) -> Dict[str, str]:
        """Broadcast message to all perspective threads and collect responses"""
        # Initialize metrics
        metrics = PerspectiveMetrics(
            session_id=session_id,
            operation_type="broadcast",
            start_time=time.time(),
            total_perspectives=len(threads),
        )

        try:
            # Check if we should use CoT for any threads
            use_cot_threads = {}
            standard_threads = {}

            for name, thread in threads.items():
                if (
                    self.enable_cot
                    and self.reasoning_orchestrator
                    and self.reasoning_orchestrator.is_available
                    and thread.model_backend.value == "bedrock"
                ):
                    use_cot_threads[name] = thread
                else:
                    standard_threads[name] = thread

            # Process threads in parallel
            responses = {}

            # Handle CoT threads if any
            if use_cot_threads:
                cot_responses = await self._broadcast_with_cot(
                    use_cot_threads, message, session_id
                )
                responses.update(cot_responses)

            # Handle standard threads
            if standard_threads:
                standard_responses = await self.thread_manager.broadcast_message(
                    standard_threads, message, session_id
                )
                responses.update(standard_responses)

            # Classify responses for metrics
            for perspective_name, response in responses.items():
                if isinstance(response, str):
                    if NO_RESPONSE in response:
                        metrics.abstained_perspectives += 1
                    elif response.startswith("ERROR:"):
                        metrics.failed_perspectives += 1
                    else:
                        metrics.successful_perspectives += 1
                else:
                    metrics.failed_perspectives += 1

            # Store metrics
            metrics.end_time = time.time()
            async with self.metrics_lock:
                self._store_metrics(metrics)

            # Log perspective summary
            logger.info(
                f"Perspective broadcast completed: {metrics.execution_time:.2f}s, "
                f"Success: {metrics.successful_perspectives}/{metrics.total_perspectives}, "
                f"Rate: {metrics.success_rate:.1f}%"
            )

            return responses

        except Exception as e:
            metrics.end_time = time.time()
            metrics.failed_perspectives = metrics.total_perspectives
            async with self.metrics_lock:
                self._store_metrics(metrics)

            logger.error(f"Perspective broadcast failed: {e}")
            raise OrchestrationError(f"Perspective broadcast failed: {e}") from e

    async def _broadcast_with_cot(
        self, threads: Dict[str, Thread], message: str, session_id: str
    ) -> Dict[str, str]:
        """Broadcast to threads using Chain of Thought reasoning"""
        import boto3

        # Get Bedrock client
        bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Process each thread with CoT
        tasks = []
        for name, thread in threads.items():
            # Add the message to thread history
            thread.add_message("user", message)

            # Create task for CoT processing
            task = asyncio.create_task(
                self._process_thread_with_cot(
                    name, thread, message, bedrock_client, session_id
                )
            )
            tasks.append((name, task))

        # Gather results
        responses = {}
        for name, task in tasks:
            try:
                response = await task
                responses[name] = response
            except Exception as e:
                logger.error(f"CoT processing failed for {name}: {e}")
                responses[name] = f"ERROR: {str(e)}"

        return responses

    async def _process_thread_with_cot(
        self,
        name: str,
        thread: Thread,
        message: str,
        bedrock_client: Any,
        session_id: str,
    ) -> str:
        """Process a single thread with Chain of Thought reasoning"""
        try:
            (
                response_text,
                reasoning_summary,
            ) = await self.reasoning_orchestrator.analyze_with_reasoning(
                thread, message, bedrock_client, session_id
            )

            # Add response to thread history
            thread.add_message("assistant", response_text)

            # Log CoT summary if available
            if reasoning_summary and "overall_confidence" in reasoning_summary:
                logger.info(
                    f"CoT analysis for {name}: confidence={reasoning_summary['overall_confidence']}, "
                    f"steps={reasoning_summary.get('total_steps', 0)}"
                )

            return response_text

        except CoTTimeoutError as e:
            logger.warning(f"CoT timeout for {name}: {e}")
            # Fallback to standard processing
            return await self._fallback_to_standard(thread, name)
        except CoTProcessingError as e:
            logger.error(f"CoT processing error for {name}: {e}")
            # Fallback to standard processing
            return await self._fallback_to_standard(thread, name)
        except Exception as e:
            logger.error(f"Unexpected error in CoT for {name}: {e}")
            raise

    async def _fallback_to_standard(self, thread: Thread, name: str) -> str:
        """Fallback to standard processing when CoT fails"""
        logger.info(f"Falling back to standard processing for {name}")
        try:
            # Use thread_manager's single thread processing
            single_thread_dict = {name: thread}
            responses = await self.thread_manager.broadcast_message(
                single_thread_dict, "", "fallback"
            )
            return responses.get(name, "ERROR: Fallback failed")
        except Exception as e:
            logger.error(f"Fallback processing failed for {name}: {e}")
            return f"ERROR: Both CoT and fallback failed: {str(e)}"

    async def broadcast_to_perspectives_stream(
        self, threads: Dict[str, Thread], message: str, session_id: str = "unknown"
    ):
        """
        Broadcast message to all perspective threads with streaming responses

        Yields events in format:
        {
            "type": "chunk" | "complete" | "error" | "start",
            "perspective_name": str,
            "content": str,
            "timestamp": float
        }
        """
        metrics = PerspectiveMetrics(
            session_id=session_id,
            operation_type="broadcast_stream",
            start_time=time.time(),
            total_perspectives=len(threads),
        )

        try:
            # Use thread manager for streaming
            async for event in self.thread_manager.broadcast_message_stream(
                threads, message, session_id
            ):
                # Transform thread events to perspective events
                if "thread_name" in event:
                    event["perspective_name"] = event.pop("thread_name")
                yield event

            # Store metrics
            metrics.end_time = time.time()
            async with self.metrics_lock:
                self._store_metrics(metrics)

        except Exception as e:
            metrics.end_time = time.time()
            metrics.failed_perspectives = metrics.total_perspectives
            async with self.metrics_lock:
                self._store_metrics(metrics)

            logger.error(f"Perspective streaming failed: {e}")
            raise OrchestrationError(f"Perspective streaming failed: {e}") from e

    async def synthesize_perspective_responses(
        self, responses: Dict[str, str], session_id: str = "unknown"
    ) -> str:
        """Synthesize responses from multiple perspectives into a coherent analysis"""
        try:
            # Filter out errors and abstentions
            valid_responses = {
                name: response
                for name, response in responses.items()
                if not response.startswith("ERROR:") and NO_RESPONSE not in response
            }

            if not valid_responses:
                return self.response_formatter.format_error_response(
                    "No valid perspective responses to synthesize",
                    "synthesis_error",
                    {"session_id": session_id},
                )

            # Use response formatter for synthesis logic
            return await self.response_formatter.synthesize_responses(
                valid_responses, session_id
            )

        except Exception as e:
            logger.error(f"Perspective synthesis failed: {e}")
            return self.response_formatter.format_error_response(
                f"Synthesis failed: {str(e)}",
                "synthesis_error",
                {"session_id": session_id},
            )

    def _store_metrics(self, metrics: PerspectiveMetrics) -> None:
        """Store perspective metrics"""
        self.metrics_history.append(metrics)
        # Keep only last 100 metrics to prevent memory leaks
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

    async def get_perspective_metrics(self, last_n: int = 10) -> Dict[str, any]:
        """Get perspective-level performance metrics"""
        async with self.metrics_lock:
            if not self.metrics_history:
                return {"message": "No perspective metrics available"}

            recent_metrics = self.metrics_history[-last_n:]

        # Calculate aggregate statistics
        total_operations = len(recent_metrics)
        avg_execution_time = (
            sum(m.execution_time for m in recent_metrics if m.execution_time)
            / total_operations
        )
        overall_success_rate = (
            sum(m.success_rate for m in recent_metrics) / total_operations
        )

        return {
            "perspective_summary": {
                "total_operations": total_operations,
                "avg_execution_time_seconds": round(avg_execution_time, 2),
                "overall_success_rate_percent": round(overall_success_rate, 1),
            },
            "recent_operations": [
                {
                    "session_id": m.session_id,
                    "operation": m.operation_type,
                    "execution_time": m.execution_time,
                    "success_rate": m.success_rate,
                    "perspectives": {
                        "total": m.total_perspectives,
                        "successful": m.successful_perspectives,
                        "failed": m.failed_perspectives,
                        "abstained": m.abstained_perspectives,
                    },
                }
                for m in recent_metrics
            ],
        }
