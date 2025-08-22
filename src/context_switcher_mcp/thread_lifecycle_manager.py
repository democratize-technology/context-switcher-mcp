"""Core thread lifecycle and execution management"""

import asyncio
from dataclasses import dataclass
from typing import Any

from .aorp import create_error_response
from .backend_factory import BackendFactory
from .config import get_config
from .exceptions import (
    ModelAuthenticationError,
    ModelBackendError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
    ModelValidationError,
    OrchestrationError,
)
from .logging_base import get_logger
from .models import Thread

logger = get_logger(__name__)


@dataclass
class ExecutionContext:
    """Context for thread execution with all necessary information"""

    thread: Thread
    backend: Any
    max_retries: int
    retry_delay: float
    circuit_breaker_manager: Any = None


class ErrorClassifier:
    """Classifies errors as retryable or non-retryable"""

    @staticmethod
    def is_non_retryable_error(error_str: str) -> bool:
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
        return any(term in error_str.lower() for term in non_retryable_terms)

    @staticmethod
    def classify_exception(exception: Exception) -> tuple[bool, bool]:
        """Classify exception as (is_retryable, should_record_in_circuit_breaker)"""
        if isinstance(exception, ModelAuthenticationError | ModelValidationError):
            return False, False  # Non-retryable, don't record
        elif isinstance(
            exception, ModelConnectionError | ModelTimeoutError | ModelRateLimitError
        ):
            return True, True  # Retryable, record failure
        elif isinstance(exception, ModelBackendError):
            return True, True  # Generally retryable, record failure
        else:
            return False, False  # Unexpected errors are non-retryable


class CircuitBreakerHandler:
    """Handles circuit breaker operations"""

    def __init__(self, circuit_breaker_manager):
        self.circuit_breaker_manager = circuit_breaker_manager

    async def ensure_ready(self) -> None:
        """Ensure circuit breaker states are loaded"""
        if self.circuit_breaker_manager:
            await self.circuit_breaker_manager.ensure_states_loaded()

    def should_allow_request(self, backend_type) -> bool:
        """Check if request should be allowed through circuit breaker"""
        if not self.circuit_breaker_manager:
            return True
        return self.circuit_breaker_manager.should_allow_request(backend_type)

    async def record_success(self, backend_type) -> None:
        """Record successful request"""
        if self.circuit_breaker_manager:
            await self.circuit_breaker_manager.record_success(backend_type)

    async def record_failure(self, backend_type) -> None:
        """Record failed request"""
        if self.circuit_breaker_manager:
            await self.circuit_breaker_manager.record_failure(backend_type)

    def create_circuit_breaker_error_response(self, backend_type) -> str:
        """Create error response for circuit breaker open state"""
        error_response = create_error_response(
            error_message=f"Circuit breaker is OPEN for {backend_type.value} backend due to repeated failures",
            error_type="circuit_breaker_open",
            context={"backend": backend_type.value},
            recoverable=True,
        )
        return f"AORP_ERROR: {error_response}"


class ErrorResponseBuilder:
    """Builds appropriate error responses"""

    @staticmethod
    def create_retry_exhausted_response(
        context: ExecutionContext, last_error: Exception
    ) -> str:
        """Create response for retry exhaustion"""
        from .security import sanitize_error_message

        error_response = create_error_response(
            error_message=f"Failed after {context.max_retries} attempts: {sanitize_error_message(str(last_error))}",
            error_type="retry_exhausted",
            context={
                "attempts": context.max_retries,
                "thread": context.thread.name,
                "backend": context.thread.model_backend.value,
            },
            recoverable=True,
        )
        return f"AORP_ERROR: {error_response}"

    @staticmethod
    def create_non_retryable_error_response(
        context: ExecutionContext, error: Exception
    ) -> str:
        """Create response for non-retryable errors"""

        error_response = create_error_response(
            error_message=str(error),
            error_type="configuration_error",
            context={
                "thread": context.thread.name,
                "backend": context.thread.model_backend.value,
            },
            recoverable=False,
        )
        return f"AORP_ERROR: {error_response}"


class ThreadLifecycleManager:
    """Manages the core lifecycle of thread execution"""

    def __init__(
        self,
        circuit_breaker_manager=None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ):
        """Initialize thread lifecycle manager

        Args:
            circuit_breaker_manager: Circuit breaker manager for failure handling
            max_retries: Maximum number of retries for failed calls (uses config default if None)
            retry_delay: Initial delay between retries (uses config default if None)
        """
        config = get_config()
        self.max_retries = (
            max_retries if max_retries is not None else config.models.max_retries
        )
        self.retry_delay = (
            retry_delay
            if retry_delay is not None
            else config.models.retry_delay_seconds
        )

        self.circuit_breaker_manager = circuit_breaker_manager

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
        # Setup execution context and handlers
        context = ExecutionContext(
            thread=thread,
            backend=self._get_backend(thread),
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

        circuit_breaker = CircuitBreakerHandler(self.circuit_breaker_manager)
        await circuit_breaker.ensure_ready()

        # Check circuit breaker before attempting
        if not circuit_breaker.should_allow_request(thread.model_backend):
            logger.warning(
                f"Circuit breaker OPEN for {thread.model_backend.value}, failing fast"
            )
            return circuit_breaker.create_circuit_breaker_error_response(
                thread.model_backend
            )

        # Execute with retry strategy
        return await self._execute_with_retry(context, circuit_breaker)

    def _get_backend(self, thread: Thread) -> Any:
        """Get backend from factory with error handling"""
        try:
            return BackendFactory.get_backend(thread.model_backend)
        except Exception as e:
            raise ValueError(f"Backend not available: {thread.model_backend} - {e}")

    async def _execute_with_retry(
        self, context: ExecutionContext, circuit_breaker: CircuitBreakerHandler
    ) -> str:
        """Execute thread with retry logic"""
        last_error = None

        for attempt in range(context.max_retries):
            try:
                response = await context.backend.call_model(context.thread)
                await circuit_breaker.record_success(context.thread.model_backend)
                return response

            except Exception as e:
                last_error = e
                is_retryable, should_record = ErrorClassifier.classify_exception(e)

                # Handle non-retryable errors immediately
                if not is_retryable:
                    if isinstance(e, ModelAuthenticationError | ModelValidationError):
                        logger.warning(f"Non-retryable error from backend: {e}")
                        raise
                    else:
                        logger.error(
                            f"Unexpected error calling backend for thread {context.thread.name}: {e}",
                            exc_info=True,
                        )
                        raise OrchestrationError(
                            f"Unexpected backend error: {e}"
                        ) from e

                # Record failure in circuit breaker if needed
                if should_record:
                    await circuit_breaker.record_failure(context.thread.model_backend)

                # Handle retry logic
                if await self._should_retry_error(context, e, attempt):
                    continue
                else:
                    return ErrorResponseBuilder.create_non_retryable_error_response(
                        context, e
                    )

        # All retries exhausted
        from .security import sanitize_error_message

        logger.error(
            f"All {context.max_retries} attempts failed for {context.thread.name}: {sanitize_error_message(str(last_error))}"
        )
        return ErrorResponseBuilder.create_retry_exhausted_response(context, last_error)

    async def _should_retry_error(
        self, context: ExecutionContext, error: Exception, attempt: int
    ) -> bool:
        """Determine if error should be retried and handle delay"""
        error_str = str(error).lower()

        # Don't retry on non-transient errors
        if ErrorClassifier.is_non_retryable_error(error_str):
            from .security import sanitize_error_message

            logger.error(
                f"Non-retryable error for {context.thread.name}: {sanitize_error_message(str(error))}"
            )
            return False

        # If this is the last attempt, don't retry
        if attempt >= context.max_retries - 1:
            return False

        # Apply exponential backoff delay
        delay = context.retry_delay * (2**attempt)
        logger.warning(
            f"Attempt {attempt + 1} failed for {context.thread.name}: {error}. Retrying in {delay}s..."
        )
        await asyncio.sleep(delay)
        return True

    async def execute_threads_parallel(
        self, threads: dict[str, Thread], message: str | None = None
    ) -> dict[str, str]:
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
        for name, response in zip(thread_names, responses, strict=False):
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

    def _is_non_retryable_error(self, error_str: str) -> bool:
        """Check if error is non-retryable based on error message"""
        return ErrorClassifier.is_non_retryable_error(error_str)
