"""Thread orchestration for parallel LLM execution"""

import asyncio
import logging
import os
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .models import Thread, ModelBackend

logger = logging.getLogger(__name__)

# Constants
NO_RESPONSE = "[NO_RESPONSE]"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7


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
class OrchestrationMetrics:
    """Aggregate metrics for orchestration operations"""

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
    failure_threshold: int = 5
    timeout_seconds: int = 300  # 5 minutes

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

    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class ThreadOrchestrator:
    """Orchestrates parallel thread execution with different LLM backends"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize orchestrator

        Args:
            max_retries: Maximum number of retries for failed calls
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.backends = {
            ModelBackend.BEDROCK: self._call_bedrock,
            ModelBackend.LITELLM: self._call_litellm,
            ModelBackend.OLLAMA: self._call_ollama,
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Circuit breakers for each backend
        self.circuit_breakers: Dict[ModelBackend, CircuitBreakerState] = {
            backend: CircuitBreakerState(backend=backend) for backend in ModelBackend
        }

        # Metrics storage (in production, this would be persisted)
        self.metrics_history: List[OrchestrationMetrics] = []
        self.max_metrics_history = 1000  # Keep last 1000 operations

    async def broadcast_message(
        self, threads: Dict[str, Thread], message: str, session_id: str = "unknown"
    ) -> Dict[str, str]:
        """Broadcast message to all threads and collect responses"""
        # Initialize metrics
        metrics = OrchestrationMetrics(
            session_id=session_id,
            operation_type="broadcast",
            start_time=time.time(),
            total_threads=len(threads),
        )

        tasks = []
        thread_names = []

        for name, thread in threads.items():
            # Add user message to thread history
            thread.add_message("user", message)

            # Create task for this thread with metrics
            task = self._get_thread_response_with_metrics(thread, metrics)
            tasks.append(task)
            thread_names.append(name)

        # Execute all threads in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Build response dictionary and update metrics
        result = {}
        for name, response in zip(thread_names, responses):
            if isinstance(response, Exception):
                logger.error(f"Error in thread {name}: {response}")
                result[name] = f"ERROR: {str(response)}"
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

        # Finalize metrics
        metrics.end_time = time.time()
        self._store_metrics(metrics)

        # Log performance summary
        logger.info(
            f"Broadcast completed: {metrics.execution_time:.2f}s, "
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
        # Initialize metrics
        metrics = OrchestrationMetrics(
            session_id=session_id,
            operation_type="broadcast_stream",
            start_time=time.time(),
            total_threads=len(threads),
        )

        # Create streaming tasks for each thread
        tasks = []
        for name, thread in threads.items():
            # Add user message to thread history
            thread.add_message("user", message)

            # Yield start event for this thread
            yield {
                "type": "start",
                "thread_name": name,
                "content": "",
                "timestamp": time.time(),
            }

            # Create streaming task based on backend
            if thread.model_backend == ModelBackend.BEDROCK:
                task = asyncio.create_task(self._stream_from_thread(thread, name))
                tasks.append(task)
            else:
                # For non-streaming backends, fall back to regular call
                task = asyncio.create_task(
                    self._get_thread_response_async(thread, name)
                )
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
            except Exception as e:
                yield {
                    "type": "error",
                    "thread_name": "unknown",
                    "content": str(e),
                    "timestamp": time.time(),
                }

        # Use asyncio.as_completed to yield results as they come
        for task in asyncio.as_completed(tasks):
            async for event in stream_handler(task):
                yield event

        # Finalize metrics
        metrics.end_time = time.time()
        self._store_metrics(metrics)

    async def _stream_from_thread(self, thread: Thread, thread_name: str):
        """Stream response from a single thread"""
        try:
            if thread.model_backend == ModelBackend.BEDROCK:
                async for event in self._call_bedrock_stream(thread):
                    event["timestamp"] = time.time()
                    yield event
            else:
                # Fallback for non-streaming backends
                response = await self._get_thread_response(thread)
                yield {
                    "type": "complete",
                    "thread_name": thread_name,
                    "content": response,
                    "timestamp": time.time(),
                }
        except Exception as e:
            yield {
                "type": "error",
                "thread_name": thread_name,
                "content": str(e),
                "timestamp": time.time(),
            }

    async def _get_thread_response_async(self, thread: Thread, thread_name: str):
        """Get non-streaming response formatted as streaming event"""
        try:
            response = await self._get_thread_response(thread)
            return {
                "type": "complete",
                "thread_name": thread_name,
                "content": response,
                "timestamp": time.time(),
            }
        except Exception as e:
            return {
                "type": "error",
                "thread_name": thread_name,
                "content": str(e),
                "timestamp": time.time(),
            }

    async def _get_thread_response_with_metrics(
        self, thread: Thread, orchestration_metrics: OrchestrationMetrics
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
        except Exception as e:
            thread_metrics.success = False
            thread_metrics.error_message = str(e)
            thread_metrics.end_time = time.time()
            raise

    async def _get_thread_response(self, thread: Thread) -> str:
        """Get response from a single thread with retry logic and circuit breaker"""
        backend_fn = self.backends.get(thread.model_backend)
        if not backend_fn:
            raise ValueError(f"Unknown model backend: {thread.model_backend}")

        # Check circuit breaker
        circuit_breaker = self.circuit_breakers[thread.model_backend]
        if not circuit_breaker.should_allow_request():
            logger.warning(
                f"Circuit breaker OPEN for {thread.model_backend.value}, failing fast"
            )
            return (
                f"Error: Circuit breaker OPEN for {thread.model_backend.value} backend"
            )

        # Try with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await backend_fn(thread)
                # Record success in circuit breaker
                circuit_breaker.record_success()
                return response
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

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
                    circuit_breaker.record_failure()

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
                    logger.error(f"Non-retryable error for {thread.name}: {e}")
                    return f"Error: {str(e)}"

                # Retry on transient errors
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {thread.name}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries} attempts failed for {thread.name}: {e}"
                    )

        # If all retries failed, return error message
        return f"Error after {self.max_retries} attempts: {str(last_error)}"

    async def _call_bedrock(self, thread: Thread) -> str:
        """Call AWS Bedrock model"""
        try:
            import boto3

            # Create Bedrock client
            client = boto3.client(
                service_name="bedrock-runtime", region_name="us-east-1"
            )

            # Prepare messages for Bedrock Converse API
            messages = []

            # Add conversation history (skip system message)
            for msg in thread.conversation_history:
                # Bedrock expects content as a list of content blocks
                messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}],  # Fixed: content as list
                    }
                )

            # Call Bedrock
            model_id = thread.model_name or os.environ.get(
                "BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            )

            response = client.converse(
                modelId=model_id,
                messages=messages,
                system=[{"text": thread.system_prompt}],
                inferenceConfig={
                    "maxTokens": DEFAULT_MAX_TOKENS,
                    "temperature": DEFAULT_TEMPERATURE,
                },
            )

            # Extract response
            content = response["output"]["message"]["content"][0]["text"]
            return content

        except Exception as e:
            logger.error(f"Bedrock error: {e}")
            if "inference profile" in str(e).lower():
                return "Error: Model needs inference profile ID. Try: us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            elif "credentials" in str(e).lower():
                return "Error: AWS credentials not configured. Run: aws configure"
            else:
                return f"Error calling Bedrock: {str(e)}"

    async def _call_bedrock_stream(self, thread: Thread):
        """Call AWS Bedrock model with streaming support"""
        try:
            import boto3

            # Create Bedrock client
            client = boto3.client("bedrock-runtime")

            # Prepare messages for Converse API
            messages = []
            for msg in thread.conversation_history:
                messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}],  # Converse API format
                    }
                )

            # Get model ID
            model_id = thread.model_name or os.environ.get(
                "BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            )

            # Use converse_stream for streaming responses
            response = client.converse_stream(
                modelId=model_id,
                messages=messages,
                system=[{"text": thread.system_prompt}],
                inferenceConfig={
                    "maxTokens": DEFAULT_MAX_TOKENS,
                    "temperature": DEFAULT_TEMPERATURE,
                },
            )

            # Stream response chunks
            full_content = ""
            for event in response["stream"]:
                if "contentBlockDelta" in event:
                    chunk = event["contentBlockDelta"]["delta"]["text"]
                    full_content += chunk
                    yield {
                        "type": "chunk",
                        "content": chunk,
                        "thread_name": thread.name,
                    }
                elif "messageStop" in event:
                    yield {
                        "type": "complete",
                        "content": full_content,
                        "thread_name": thread.name,
                    }

        except Exception as e:
            logger.error(f"Bedrock streaming error: {e}")
            error_msg = f"Error calling Bedrock: {str(e)}"
            yield {"type": "error", "content": error_msg, "thread_name": thread.name}

    async def _call_litellm(self, thread: Thread) -> str:
        """Call model via LiteLLM"""
        try:
            import litellm

            # Prepare messages
            messages = [{"role": "system", "content": thread.system_prompt}]

            # Add conversation history
            for msg in thread.conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Call LiteLLM
            model = thread.model_name or "gpt-4"

            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LiteLLM error: {e}")
            if "api_key" in str(e).lower():
                return "Error: Missing API key. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable"
            elif "connection" in str(e).lower():
                return "Error: Cannot connect to LiteLLM. Check LITELLM_API_BASE is set correctly"
            else:
                return f"Error calling LiteLLM: {str(e)}"

    async def _call_ollama(self, thread: Thread) -> str:
        """Call local Ollama model"""
        try:
            import httpx

            # Prepare messages
            messages = [{"role": "system", "content": thread.system_prompt}]

            # Add conversation history
            for msg in thread.conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Call Ollama API
            model = thread.model_name or "llama3.2"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                    + "/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": DEFAULT_TEMPERATURE,
                            "num_predict": DEFAULT_MAX_TOKENS,
                        },
                    },
                    timeout=60.0,
                )

                response.raise_for_status()
                result = response.json()
                return result["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            if "connection" in str(e).lower():
                return "Error: Cannot connect to Ollama. Is it running? Set OLLAMA_HOST=http://your-host:11434"
            elif "model" in str(e).lower():
                return f"Error: Model not found. Pull it first: ollama pull {model}"
            else:
                return f"Error calling Ollama: {str(e)}"

    def _store_metrics(self, metrics: OrchestrationMetrics) -> None:
        """Store metrics and maintain history limit"""
        self.metrics_history.append(metrics)

        # Trim history if it exceeds max size
        if len(self.metrics_history) > self.max_metrics_history:
            self.metrics_history = self.metrics_history[-self.max_metrics_history :]

    def get_performance_metrics(self, last_n: int = 10) -> Dict[str, any]:
        """Get performance metrics for recent operations"""
        if not self.metrics_history:
            return {"message": "No metrics available"}

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

        # Circuit breaker status
        circuit_status = {}
        for backend, breaker in self.circuit_breakers.items():
            circuit_status[backend.value] = {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat()
                if breaker.last_failure_time
                else None,
            }

        return {
            "summary": {
                "total_operations": total_operations,
                "avg_execution_time_seconds": round(avg_execution_time, 2),
                "overall_success_rate_percent": round(overall_success_rate, 1),
            },
            "backend_performance": backend_stats,
            "circuit_breakers": circuit_status,
            "last_operation": recent_metrics[-1].operation_type
            if recent_metrics
            else None,
        }

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
