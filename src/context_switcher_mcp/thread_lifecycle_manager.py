"""Core thread lifecycle and execution management"""

import asyncio
import logging
from typing import Dict, Optional

from .models import Thread, ModelBackend
from .config import get_config
from .backend_interface import get_backend_interface
from .aorp import create_error_response
from .exceptions import (
    OrchestrationError,
    ModelBackendError,
    ModelConnectionError,
    ModelTimeoutError,
    ModelRateLimitError,
    ModelAuthenticationError,
    ModelValidationError,
)

logger = logging.getLogger(__name__)


class ThreadLifecycleManager:
    """Manages the core lifecycle of thread execution"""

    def __init__(
        self,
        circuit_breaker_manager=None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ):
        """Initialize thread lifecycle manager

        Args:
            circuit_breaker_manager: Circuit breaker manager for failure handling
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

        self.circuit_breaker_manager = circuit_breaker_manager

        # Backend interface mapping
        self.backends = {
            ModelBackend.BEDROCK: self._call_unified_backend,
            ModelBackend.LITELLM: self._call_unified_backend,
            ModelBackend.OLLAMA: self._call_unified_backend,
        }

    async def execute_thread(self, thread: Thread) -> str:
        """Execute a single thread with full lifecycle management

        Args:
            thread: Thread to execute

        Returns:
            Response string from the thread execution

        Raises:
            OrchestrationError: On unexpected errors
            ModelBackendError: On expected model errors (non-retryable)
            CircuitBreakerOpenError: When circuit breaker is open
        """
        return await self._get_thread_response(thread)

    async def _get_thread_response(self, thread: Thread) -> str:
        """Get response from a single thread with retry logic and circuit breaker"""
        # Ensure circuit breaker states are loaded if manager is available
        if self.circuit_breaker_manager:
            await self.circuit_breaker_manager.ensure_states_loaded()

        backend_fn = self.backends.get(thread.model_backend)
        if not backend_fn:
            raise ValueError(f"Unknown model backend: {thread.model_backend}")

        # Check circuit breaker if manager is available
        if self.circuit_breaker_manager:
            if not self.circuit_breaker_manager.should_allow_request(
                thread.model_backend
            ):
                logger.warning(
                    f"Circuit breaker OPEN for {thread.model_backend.value}, failing fast"
                )
                error_response = create_error_response(
                    error_message=f"Circuit breaker is OPEN for {thread.model_backend.value} backend due to repeated failures",
                    error_type="circuit_breaker_open",
                    context={
                        "backend": thread.model_backend.value,
                    },
                    recoverable=True,
                )
                return f"AORP_ERROR: {error_response}"

        # Try with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await backend_fn(thread)
                # Record success in circuit breaker if manager is available
                if self.circuit_breaker_manager:
                    await self.circuit_breaker_manager.record_success(
                        thread.model_backend
                    )
                return response

            except (ModelConnectionError, ModelTimeoutError, ModelRateLimitError) as e:
                # Retryable errors
                last_error = e

                # Record failure in circuit breaker for transient errors
                if self.circuit_breaker_manager:
                    await self.circuit_breaker_manager.record_failure(
                        thread.model_backend
                    )

            except (ModelAuthenticationError, ModelValidationError) as e:
                # Non-retryable errors - don't record in circuit breaker
                logger.warning(f"Non-retryable error from backend: {e}")
                raise

            except ModelBackendError as e:
                # Other model errors - check if retryable
                last_error = e

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
                if self.circuit_breaker_manager and not self._is_non_retryable_error(
                    error_str
                ):
                    await self.circuit_breaker_manager.record_failure(
                        thread.model_backend
                    )

                # Don't retry on non-transient errors
                if self._is_non_retryable_error(error_str):
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

    def _is_non_retryable_error(self, error_str: str) -> bool:
        """Check if error is non-retryable based on error message"""
        non_retryable_terms = [
            "api_key",
            "credentials",
            "not found",
            "invalid",
            "unauthorized",
            "forbidden",
            "model not found",
        ]
        return any(term in error_str for term in non_retryable_terms)

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

    async def execute_threads_parallel(
        self, threads: Dict[str, Thread], message: str = None
    ) -> Dict[str, str]:
        """Execute multiple threads in parallel

        Args:
            threads: Dictionary of thread_name -> Thread objects
            message: Optional message to add to each thread before execution

        Returns:
            Dictionary of thread_name -> response
        """
        tasks = []
        thread_names = []

        for name, thread in threads.items():
            # Only add message if not already present and message is provided
            if message and (
                not thread.conversation_history
                or thread.conversation_history[-1].get("content") != message
            ):
                thread.add_message("user", message)

            task = self.execute_thread(thread)
            tasks.append(task)
            thread_names.append(name)

        # Execute all threads in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Build response dictionary
        result = {}
        for name, response in zip(thread_names, responses):
            if isinstance(response, Exception):
                from .security import sanitize_error_message

                sanitized_error = sanitize_error_message(str(response))
                logger.error(f"Error in thread {name}: {sanitized_error}")
                result[name] = f"ERROR: {sanitized_error}"
            else:
                # Add assistant response to thread history
                threads[name].add_message("assistant", response)
                result[name] = response

        return result
