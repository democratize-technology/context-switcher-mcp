"""Profiling wrapper for LLM backend interfaces

This module provides transparent profiling integration with existing backend interfaces.
It wraps backend calls with profiling logic while maintaining full compatibility.
"""

import logging
import time
from typing import Dict, Any, AsyncGenerator
from functools import wraps

from .backend_interface import ModelBackendInterface, ModelCallConfig
from .llm_profiler import get_global_profiler, LLMCallMetrics
from .models import Thread

logger = logging.getLogger(__name__)


class ProfilingBackendWrapper(ModelBackendInterface):
    """Wrapper that adds profiling to any backend interface"""

    def __init__(self, wrapped_backend: ModelBackendInterface):
        """Initialize with the backend to wrap"""
        self.wrapped_backend = wrapped_backend
        self.profiler = get_global_profiler()

        # Copy backend name and config
        super().__init__(wrapped_backend.backend_name)
        self.config = wrapped_backend.config

    def _get_model_name(self, thread: Thread) -> str:
        """Delegate to wrapped backend"""
        return self.wrapped_backend._get_model_name(thread)

    def get_model_config(self, thread: Thread) -> ModelCallConfig:
        """Delegate to wrapped backend"""
        return self.wrapped_backend.get_model_config(thread)

    async def call_model(self, thread: Thread) -> str:
        """Profile a regular model call"""
        model_config = self.get_model_config(thread)
        session_id = getattr(thread, "session_id", "unknown")

        async with self.profiler.profile_call(
            session_id=session_id,
            thread_name=thread.name,
            backend=self.backend_name,
            model_name=model_config.model_name,
            streaming=False,
        ) as metrics:
            start_time = time.time()

            try:
                # Call the wrapped backend
                response = await self.wrapped_backend.call_model(thread)

                # Record successful call metrics
                if metrics:
                    self._record_call_metrics(metrics, thread, response, start_time)

                return response

            except Exception as e:
                # Record error metrics
                if metrics:
                    metrics.error_type = type(e).__name__
                    metrics.error_message = str(e)[:200]

                raise

    async def call_model_stream(
        self, thread: Thread
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Profile a streaming model call"""
        model_config = self.get_model_config(thread)
        session_id = getattr(thread, "session_id", "unknown")

        async with self.profiler.profile_call(
            session_id=session_id,
            thread_name=thread.name,
            backend=self.backend_name,
            model_name=model_config.model_name,
            streaming=True,
        ) as metrics:
            start_time = time.time()
            first_token_time = None
            accumulated_response = ""
            chunk_count = 0

            try:
                async for event in self.wrapped_backend.call_model_stream(thread):
                    # Record time to first token for network latency
                    if first_token_time is None and event.get("type") == "chunk":
                        first_token_time = time.time()
                        if metrics:
                            self.profiler.record_network_timing(
                                metrics, first_token_time
                            )

                    # Accumulate response content for token counting
                    if event.get("type") in ["chunk", "complete"]:
                        content = event.get("content", "")
                        if event.get("type") == "chunk":
                            accumulated_response += content
                            chunk_count += 1
                        elif event.get("type") == "complete":
                            # Use complete content if available, otherwise use accumulated
                            final_response = content or accumulated_response
                            if metrics:
                                self._record_streaming_metrics(
                                    metrics, thread, final_response, chunk_count
                                )

                    yield event

            except Exception as e:
                # Record error metrics for streaming
                if metrics:
                    metrics.error_type = type(e).__name__
                    metrics.error_message = str(e)[:200]

                raise

    def _record_call_metrics(
        self, metrics: LLMCallMetrics, thread: Thread, response: str, start_time: float
    ) -> None:
        """Record metrics for a completed model call"""

        # Estimate token usage
        input_tokens = self._estimate_tokens(self._get_input_text(thread))
        output_tokens = self._estimate_tokens(response)

        # Record token usage and costs
        self.profiler.record_token_usage(
            metrics=metrics,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_length=len(self._get_input_text(thread)),
            response_length=len(response),
        )

    def _record_streaming_metrics(
        self, metrics: LLMCallMetrics, thread: Thread, response: str, chunk_count: int
    ) -> None:
        """Record metrics for a streaming model call"""

        # Estimate token usage
        input_tokens = self._estimate_tokens(self._get_input_text(thread))
        output_tokens = self._estimate_tokens(response)

        # Record token usage and costs
        self.profiler.record_token_usage(
            metrics=metrics,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_length=len(self._get_input_text(thread)),
            response_length=len(response),
        )

        # Add streaming-specific metadata
        if hasattr(metrics, "additional_context"):
            metrics.additional_context = {}
        else:
            metrics.additional_context = {}

        metrics.additional_context["chunk_count"] = chunk_count
        metrics.additional_context["streaming_response"] = True

    def _get_input_text(self, thread: Thread) -> str:
        """Extract input text from thread for token estimation"""
        input_parts = [thread.system_prompt]

        for msg in thread.conversation_history:
            if msg.get("role") in ["user", "assistant"]:
                input_parts.append(msg.get("content", ""))

        return " ".join(input_parts)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (approximately 4 characters per token for English)

        This is a fallback estimation. In production, you might want to use
        actual tokenizers for each model for accuracy.
        """
        if not text:
            return 0

        # Simple heuristic: ~4 characters per token for English text
        # Add some tokens for formatting and structure
        estimated = len(text) // 4

        # Add overhead for message structure, formatting, etc.
        overhead = max(10, len(text) // 100)

        return estimated + overhead


class TokenCountingMixin:
    """Mixin to provide accurate token counting for specific backends"""

    @staticmethod
    def count_tokens_bedrock(text: str, model_name: str) -> int:
        """Count tokens for Bedrock models using approximation

        For production use, you might want to integrate with the actual
        tokenizer for each model family.
        """

        # Claude models roughly follow this pattern
        if "claude" in model_name.lower():
            # Claude tokenizer is roughly 3.5-4 chars per token
            return max(1, len(text) // 4)

        # Fallback to general estimation
        return max(1, len(text) // 4)

    @staticmethod
    def count_tokens_litellm(text: str, model_name: str) -> int:
        """Count tokens for LiteLLM models"""

        try:
            # Try to use tiktoken for OpenAI models if available
            if model_name.startswith("gpt"):
                try:
                    import tiktoken

                    encoding = tiktoken.encoding_for_model(model_name)
                    return len(encoding.encode(text))
                except ImportError:
                    pass
                except Exception:
                    pass

            # Fallback to estimation
            return max(1, len(text) // 4)

        except Exception:
            return max(1, len(text) // 4)

    @staticmethod
    def count_tokens_ollama(text: str, model_name: str) -> int:
        """Count tokens for Ollama models"""

        # Most Ollama models use similar tokenization patterns
        # This is a rough approximation
        return max(1, len(text) // 4)


class EnhancedProfilingWrapper(ProfilingBackendWrapper, TokenCountingMixin):
    """Enhanced wrapper with accurate token counting"""

    def _estimate_tokens(self, text: str) -> int:
        """Use backend-specific token counting when available"""

        try:
            if self.backend_name == "bedrock":
                model_config = ModelCallConfig(
                    max_tokens=1000,
                    temperature=0.7,
                    model_name="default",
                    timeout_seconds=60.0,
                )
                return self.count_tokens_bedrock(text, model_config.model_name)

            elif self.backend_name == "litellm":
                model_config = ModelCallConfig(
                    max_tokens=1000,
                    temperature=0.7,
                    model_name="gpt-3.5-turbo",
                    timeout_seconds=60.0,
                )
                return self.count_tokens_litellm(text, model_config.model_name)

            elif self.backend_name == "ollama":
                return self.count_tokens_ollama(text, "default")

        except Exception as e:
            logger.debug(f"Token counting failed for {self.backend_name}: {e}")

        # Fallback to parent estimation
        return super()._estimate_tokens(text)


def create_profiling_wrapper(backend: ModelBackendInterface) -> ModelBackendInterface:
    """Factory function to create appropriate profiling wrapper"""

    # Check if profiling is enabled
    profiler = get_global_profiler()
    if not profiler.config.enabled:
        return backend

    # Return enhanced wrapper with better token counting
    return EnhancedProfilingWrapper(backend)


def profile_backend_call(backend_name: str):
    """Decorator for profiling individual backend calls"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            profiler = get_global_profiler()

            # Extract context from args/kwargs
            thread = None
            for arg in args:
                if isinstance(arg, Thread):
                    thread = arg
                    break

            if not thread:
                # No thread context, execute without profiling
                return await func(*args, **kwargs)

            session_id = getattr(thread, "session_id", "unknown")

            async with profiler.profile_call(
                session_id=session_id,
                thread_name=thread.name,
                backend=backend_name,
                model_name=getattr(thread, "model_name", "unknown"),
                streaming=False,
            ) as metrics:
                try:
                    result = await func(*args, **kwargs)

                    # Record basic metrics if available
                    if metrics and isinstance(result, str):
                        # Simple token estimation for the result
                        output_tokens = len(result) // 4
                        input_text = thread.system_prompt + " ".join(
                            msg.get("content", "")
                            for msg in thread.conversation_history
                        )
                        input_tokens = len(input_text) // 4

                        profiler.record_token_usage(
                            metrics=metrics,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                        )

                    return result

                except Exception as e:
                    if metrics:
                        metrics.error_type = type(e).__name__
                        metrics.error_message = str(e)[:200]
                    raise

        return wrapper

    return decorator
