"""Perspective orchestration service for managing multi-perspective analysis"""

import asyncio
import time
from typing import Any, Dict, Optional, AsyncGenerator
from dataclasses import dataclass

from .models import Thread
from .thread_manager import ThreadManager
from .response_formatter import ResponseFormatter
from .exceptions import (
    OrchestrationError,
    PerspectiveError,
    AnalysisError,
    PerformanceTimeoutError,
)
from .constants import NO_RESPONSE
from .reasoning_orchestrator import (
    PerspectiveReasoningOrchestrator,
    CoTTimeoutError,
    CoTProcessingError,
)
from .error_helpers import (
    wrap_generic_exception,
)
from .error_logging import log_error_with_context
from .error_context import error_context
from .security import sanitize_error_message
from .logging_config import get_logger
from .logging_utils import performance_timer

logger = get_logger(__name__)


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
        max_retries: int = None,
        retry_delay: float = None,
    ):
        """Initialize perspective orchestrator

        Args:
            thread_manager: Thread manager for parallel execution (creates default if None)
            response_formatter: Response formatter for AORP formatting (creates default if None)
            enable_cot: Enable Chain of Thought reasoning for Bedrock (default: True)
            cot_timeout: Timeout for CoT processing in seconds (default: 30.0)
            max_retries: Maximum number of retries for failed calls (uses config default if None)
            retry_delay: Initial delay between retries (uses config default if None)
        """
        self.thread_manager = thread_manager or ThreadManager(max_retries, retry_delay)
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

    @performance_timer(
        "perspective_broadcast", threshold_seconds=2.0, warn_threshold_seconds=10.0
    )
    async def broadcast_to_perspectives(
        self,
        threads: Dict[str, Thread],
        message: str,
        session_id: str = "unknown",
        topic: str = None,
    ) -> Dict[str, str]:
        """Broadcast message to all perspective threads and collect responses"""
        async with error_context(
            operation_name="perspective_broadcast",
            session_id=session_id,
            user_context={
                "thread_count": len(threads),
                "topic": topic,
                "enable_cot": self.enable_cot,
            },
        ) as ctx:
            metrics = self._initialize_broadcast_metrics(session_id, len(threads))
            ctx["metrics_id"] = id(metrics)

            try:
                # Partition threads by processing strategy
                cot_threads, standard_threads = self._partition_threads_by_strategy(
                    threads
                )
                ctx["cot_threads"] = len(cot_threads)
                ctx["standard_threads"] = len(standard_threads)

                # Execute broadcasts in parallel
                responses = await self._execute_parallel_broadcasts(
                    cot_threads, standard_threads, message, session_id, topic
                )

                # Update metrics and finalize
                self._classify_responses_for_metrics(responses, metrics)
                await self._finalize_broadcast_metrics(metrics)

                ctx["successful_responses"] = len(
                    [
                        r
                        for r in responses.values()
                        if r != NO_RESPONSE and not r.startswith("ERROR")
                    ]
                )

                return responses

            except (CoTTimeoutError, CoTProcessingError) as e:
                # Chain of Thought specific errors
                await self._handle_broadcast_error(e, metrics)
                raise PerformanceTimeoutError(
                    f"Perspective broadcast timed out in CoT processing: {sanitize_error_message(str(e))}",
                    performance_context={
                        "session_id": session_id,
                        "thread_count": len(threads),
                        "cot_enabled": self.enable_cot,
                    },
                ) from e
            except (PerspectiveError, AnalysisError) as e:
                # Already specific errors, re-raise with context
                await self._handle_broadcast_error(e, metrics)
                raise
            except Exception as e:
                # Convert generic exceptions to specific types
                await self._handle_broadcast_error(e, metrics)
                specific_error = wrap_generic_exception(
                    e,
                    "perspective_broadcast",
                    OrchestrationError,
                    context={
                        "session_id": session_id,
                        "thread_count": len(threads),
                        "topic": topic,
                    },
                )
                raise specific_error from e

    def _initialize_broadcast_metrics(
        self, session_id: str, total_threads: int
    ) -> PerspectiveMetrics:
        """Initialize metrics for broadcast operation"""
        return PerspectiveMetrics(
            session_id=session_id,
            operation_type="broadcast",
            start_time=time.time(),
            total_perspectives=total_threads,
        )

    def _partition_threads_by_strategy(self, threads: Dict[str, Thread]) -> tuple:
        """Partition threads into CoT and standard processing groups"""
        cot_threads = {}
        standard_threads = {}

        for name, thread in threads.items():
            if self._should_use_cot_for_thread(thread):
                cot_threads[name] = thread
            else:
                standard_threads[name] = thread

        return cot_threads, standard_threads

    def _should_use_cot_for_thread(self, thread: Thread) -> bool:
        """Check if thread should use Chain of Thought processing"""
        return (
            self.enable_cot
            and self.reasoning_orchestrator
            and self.reasoning_orchestrator.is_available
            and thread.model_backend.value == "bedrock"
        )

    async def _execute_parallel_broadcasts(
        self,
        cot_threads: Dict[str, Thread],
        standard_threads: Dict[str, Thread],
        message: str,
        session_id: str,
        topic: str,
    ) -> Dict[str, str]:
        """Execute broadcasts to both CoT and standard threads in parallel"""
        responses = {}

        # Handle CoT threads if any
        if cot_threads:
            cot_responses = await self._broadcast_with_cot(
                cot_threads, message, session_id, topic
            )
            responses.update(cot_responses)

        # Handle standard threads
        if standard_threads:
            standard_responses = await self.thread_manager.broadcast_message(
                standard_threads, message, session_id
            )
            responses.update(standard_responses)

        return responses

    def _classify_responses_for_metrics(
        self, responses: Dict[str, str], metrics: PerspectiveMetrics
    ) -> None:
        """Classify responses and update metrics accordingly"""
        for perspective_name, response in responses.items():
            if not isinstance(response, str):
                metrics.failed_perspectives += 1
                continue

            if NO_RESPONSE in response:
                metrics.abstained_perspectives += 1
            elif response.startswith("ERROR:"):
                metrics.failed_perspectives += 1
            else:
                metrics.successful_perspectives += 1

    async def _finalize_broadcast_metrics(self, metrics: PerspectiveMetrics) -> None:
        """Store metrics and log broadcast completion"""
        metrics.end_time = time.time()
        async with self.metrics_lock:
            self._store_metrics(metrics)

        logger.info(
            f"Perspective broadcast completed: {metrics.execution_time:.2f}s, "
            f"Success: {metrics.successful_perspectives}/{metrics.total_perspectives}, "
            f"Rate: {metrics.success_rate:.1f}%"
        )

    async def _handle_broadcast_error(
        self, error: Exception, metrics: PerspectiveMetrics
    ) -> None:
        """Handle broadcast error and update metrics with structured logging"""
        metrics.end_time = time.time()
        metrics.failed_perspectives = metrics.total_perspectives
        async with self.metrics_lock:
            self._store_metrics(metrics)

        # Use structured error logging
        log_error_with_context(
            error=error,
            operation_name="perspective_broadcast",
            session_id=metrics.session_id,
            additional_context={
                "operation_type": metrics.operation_type,
                "total_perspectives": metrics.total_perspectives,
                "execution_time": metrics.execution_time,
                "enable_cot": self.enable_cot,
            },
        )

    async def _broadcast_with_cot(
        self,
        threads: Dict[str, Thread],
        message: str,
        session_id: str,
        topic: str = None,
    ) -> Dict[str, str]:
        """Broadcast to threads using Chain of Thought reasoning"""
        import boto3

        # Get Bedrock client
        bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Process each thread with CoT
        tasks = []
        for name, thread in threads.items():
            # CRITICAL: Add the message to conversation history for CoT threads
            # This was missing and causing empty conversation history!
            thread.add_message("user", message)

            # Create task for CoT processing
            task = asyncio.create_task(
                self._process_thread_with_cot(
                    name, thread, message, bedrock_client, session_id, topic
                )
            )
            tasks.append((name, task))

        # Gather results with proper error handling
        responses = {}
        for name, task in tasks:
            try:
                response = await task
                responses[name] = response
            except (CoTTimeoutError, CoTProcessingError) as e:
                # Specific CoT errors - log and create error response
                log_error_with_context(
                    error=e,
                    operation_name="cot_processing",
                    session_id=session_id,
                    additional_context={"thread_name": name, "topic": topic},
                )
                responses[name] = (
                    f"ERROR: CoT processing failed - {sanitize_error_message(str(e))}"
                )
            except Exception as e:
                # Unexpected errors - wrap and log
                specific_error = wrap_generic_exception(
                    e,
                    f"cot_processing_{name}",
                    PerspectiveError,
                    context={"thread_name": name, "session_id": session_id},
                )
                log_error_with_context(
                    error=specific_error,
                    operation_name="cot_processing",
                    session_id=session_id,
                    additional_context={"thread_name": name, "topic": topic},
                )
                responses[name] = f"ERROR: {sanitize_error_message(str(e))}"

        return responses

    async def _process_thread_with_cot(
        self,
        name: str,
        thread: Thread,
        message: str,
        bedrock_client: Any,
        session_id: str,
        topic: str = None,
    ) -> str:
        """Process a single thread with Chain of Thought reasoning"""
        try:
            (
                response_text,
                reasoning_summary,
            ) = await self.reasoning_orchestrator.analyze_with_reasoning(
                thread, message, bedrock_client, session_id, topic
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
            # CoT timeout - fallback with structured logging
            log_error_with_context(
                error=e,
                operation_name="cot_thread_processing",
                session_id=session_id,
                additional_context={"thread_name": name, "fallback_attempted": True},
            )
            # Fallback to standard processing
            return await self._fallback_to_standard(thread, name)
        except CoTProcessingError as e:
            # CoT processing error - fallback with structured logging
            log_error_with_context(
                error=e,
                operation_name="cot_thread_processing",
                session_id=session_id,
                additional_context={"thread_name": name, "fallback_attempted": True},
            )
            # Fallback to standard processing
            return await self._fallback_to_standard(thread, name)
        except Exception as e:
            # Unexpected errors in CoT processing
            specific_error = wrap_generic_exception(
                e,
                f"cot_thread_processing_{name}",
                PerspectiveError,
                context={"thread_name": name, "session_id": session_id},
            )
            log_error_with_context(
                error=specific_error,
                operation_name="cot_thread_processing",
                session_id=session_id,
                additional_context={"thread_name": name},
            )
            raise specific_error from e

    async def _fallback_to_standard(self, thread: Thread, name: str) -> str:
        """Fallback to standard processing when CoT fails"""
        async with error_context(
            operation_name="cot_fallback_processing",
            user_context={"thread_name": name, "fallback_reason": "cot_failure"},
        ) as ctx:
            logger.info(f"Falling back to standard processing for {name}")

            try:
                # Use thread_manager's single thread processing
                single_thread_dict = {name: thread}
                responses = await self.thread_manager.broadcast_message(
                    single_thread_dict, "", "fallback"
                )

                result = responses.get(name, "ERROR: Fallback failed")
                ctx["fallback_success"] = not result.startswith("ERROR")
                return result

            except Exception as e:
                # Fallback itself failed - return sanitized error
                error_response = f"ERROR: Both CoT and fallback failed: {sanitize_error_message(str(e))}"
                ctx["double_failure"] = True

                # Log the fallback failure with context
                log_error_with_context(
                    error=e,
                    operation_name="cot_fallback_processing",
                    additional_context={
                        "thread_name": name,
                        "failure_type": "double_failure",
                        "original_error": "cot_processing_failed",
                    },
                )

                return error_response

    async def broadcast_to_perspectives_stream(
        self, threads: Dict[str, Thread], message: str, session_id: str = "unknown"
    ) -> AsyncGenerator[Dict[str, Any], None]:
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

            # Use structured error handling
            specific_error = wrap_generic_exception(
                e,
                "perspective_streaming",
                OrchestrationError,
                context={
                    "session_id": session_id,
                    "thread_count": len(threads),
                    "streaming": True,
                },
            )

            log_error_with_context(
                error=specific_error,
                operation_name="perspective_streaming",
                session_id=session_id,
                additional_context={"thread_count": len(threads)},
            )

            raise specific_error from e

    async def synthesize_perspective_responses(
        self, responses: Dict[str, str], session_id: str = "unknown"
    ) -> str:
        """Synthesize responses from multiple perspectives into a coherent analysis"""
        async with error_context(
            operation_name="perspective_synthesis",
            session_id=session_id,
            user_context={
                "total_responses": len(responses),
                "response_names": list(responses.keys()),
            },
        ) as ctx:
            try:
                # Filter out errors and abstentions
                valid_responses = {
                    name: response
                    for name, response in responses.items()
                    if not response.startswith("ERROR:") and NO_RESPONSE not in response
                }

                ctx["valid_responses"] = len(valid_responses)

                if not valid_responses:
                    return self.response_formatter.format_error_response(
                        "No valid perspective responses to synthesize",
                        "synthesis_error",
                        {"session_id": session_id, "total_responses": len(responses)},
                    )

                # Use response formatter for synthesis logic
                synthesis_result = await self.response_formatter.synthesize_responses(
                    valid_responses, session_id
                )

                ctx["synthesis_success"] = True
                return synthesis_result

            except AnalysisError as e:
                # Already a specific analysis error
                log_error_with_context(
                    error=e,
                    operation_name="perspective_synthesis",
                    session_id=session_id,
                    additional_context={"valid_responses": len(valid_responses)},
                )
                return self.response_formatter.format_error_response(
                    f"Analysis synthesis failed: {sanitize_error_message(str(e))}",
                    "synthesis_error",
                    {"session_id": session_id, "error_type": "analysis_error"},
                )
            except Exception as e:
                # Convert generic errors to AnalysisError
                specific_error = wrap_generic_exception(
                    e,
                    "perspective_synthesis",
                    AnalysisError,
                    context={
                        "session_id": session_id,
                        "response_count": len(responses),
                    },
                )

                log_error_with_context(
                    error=specific_error,
                    operation_name="perspective_synthesis",
                    session_id=session_id,
                    additional_context={"total_responses": len(responses)},
                )

                return self.response_formatter.format_error_response(
                    f"Synthesis failed: {sanitize_error_message(str(e))}",
                    "synthesis_error",
                    {"session_id": session_id, "error_type": type(e).__name__},
                )

    def _store_metrics(self, metrics: PerspectiveMetrics) -> None:
        """Store perspective metrics"""
        self.metrics_history.append(metrics)
        # Keep only last 100 metrics to prevent memory leaks
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

    async def get_perspective_metrics(self, last_n: int = 10) -> Dict[str, Any]:
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

    async def get_performance_metrics(self, last_n: int = 10) -> Dict[str, Any]:
        """Get combined performance metrics for recent operations"""
        # Get metrics from thread manager and perspective orchestrator
        thread_metrics = await self.thread_manager.get_thread_metrics(last_n)
        perspective_metrics = await self.get_perspective_metrics(last_n)

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

    # Expose circuit_breakers and backends for backward compatibility
    @property
    def circuit_breakers(self) -> Any:
        """Access to circuit breakers"""
        return self.thread_manager.circuit_breaker_manager.circuit_breakers

    @property
    def backends(self) -> Any:
        """Access to available backends - returns a dict-like interface for compatibility"""
        from .backend_factory import BackendFactory

        # Return a dict-like object that provides backend availability
        class BackendDict:
            def __contains__(self, backend):
                return BackendFactory.is_backend_available(backend)

            def __getitem__(self, backend):
                return BackendFactory.get_backend(backend)

        return BackendDict()
