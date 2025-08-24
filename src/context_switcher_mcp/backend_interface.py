"""Unified backend interface to eliminate configuration duplication"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from .aorp import create_error_response
from .config import get_config
from .error_context import error_context
from .error_decorators import log_errors_with_context
from .exceptions import (
    ConfigurationError,
    ModelAuthenticationError,
    ModelBackendError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
    ModelValidationError,
    NetworkConnectivityError,
    NetworkTimeoutError,
)
from .models import Thread
from .security import sanitize_error_message


@dataclass
class ModelCallConfig:
    """Configuration for model calls"""

    max_tokens: int
    temperature: float
    model_name: str
    timeout_seconds: float = 60.0


class ModelBackendInterface(ABC):
    """Abstract interface for model backends"""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.config = get_config()

    def get_model_config(self, thread: Thread) -> ModelCallConfig:
        """Get configuration for model call"""
        return ModelCallConfig(
            max_tokens=self.config.model.default_max_tokens,
            temperature=self.config.model.default_temperature,
            model_name=self._get_model_name(thread),
            timeout_seconds=60.0,
        )

    @abstractmethod
    def _get_model_name(self, thread: Thread) -> str:
        """Get the model name for this backend"""
        pass

    @abstractmethod
    @log_errors_with_context(include_performance=True)
    async def call_model(self, thread: Thread) -> str:
        """Make a model call and return response"""
        pass

    async def call_model_stream(
        self, thread: Thread
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Make a streaming model call (fallback to regular call if not supported)"""
        try:
            response = await self.call_model(thread)
            yield {
                "type": "complete",
                "content": response,
                "thread_name": thread.name,
            }
        except ModelBackendError as e:
            # Pass through model backend errors
            yield {
                "type": "error",
                "content": self._format_error_response(
                    str(e), "streaming_error", {"thread": thread.name}
                ),
                "thread_name": thread.name,
            }
        except Exception as e:
            # Unexpected errors
            yield {
                "type": "error",
                "content": self._format_error_response(
                    f"Unexpected error: {str(e)}",
                    "streaming_error",
                    {"thread": thread.name},
                ),
                "thread_name": thread.name,
            }

    def _format_error_response(
        self, error_message: str, error_type: str, context: dict[str, Any]
    ) -> str:
        """Format error as AORP response"""
        sanitized_message = sanitize_error_message(error_message)

        error_response = create_error_response(
            error_message=f"{self.backend_name} error: {sanitized_message}",
            error_type=error_type,
            context={"backend": self.backend_name, **context},
            recoverable=True,
        )
        return f"AORP_ERROR: {error_response}"

    def _get_error_type_and_message(self, error: Exception) -> tuple[str, str]:
        """Classify error and return appropriate type and message"""
        error_str = str(error).lower()

        if any(
            term in error_str
            for term in [
                "api_key",
                "key",
                "unauthorized",
                "forbidden",
                "permissions",
                "access denied",
                "accessdenied",
            ]
        ):
            return "credentials_error", "Missing or invalid API credentials"
        elif any(term in error_str for term in ["connection", "timeout", "network"]):
            return "connection_error", "Network connection failed"
        elif any(term in error_str for term in ["model", "not found"]):
            return "model_not_found", "Model not found or unavailable"
        elif "inference profile" in error_str:
            return "model_configuration_error", "Model configuration issue"
        else:
            return "api_error", f"API call failed: {sanitize_error_message(str(error))}"

    def _get_exception_for_error(
        self, error: Exception, error_message: str
    ) -> Exception:
        """Map original exception to appropriate custom exception class"""
        # Import here to avoid circular imports
        from .exceptions import (
            ModelAuthenticationError,
            ModelConnectionError,
            ModelRateLimitError,
            ModelTimeoutError,
            ModelValidationError,
            NetworkConnectivityError,
        )

        # First, check if it's already one of our custom exceptions or standard exceptions
        # that tests expect to pass through
        if isinstance(
            error,
            ModelAuthenticationError
            | ModelConnectionError
            | ModelRateLimitError
            | ModelTimeoutError
            | ModelValidationError
            | NetworkConnectivityError
            | ConnectionError
            | OSError
            | asyncio.TimeoutError,
        ):
            return error

        # Check for httpx exceptions that should pass through
        try:
            import httpx

            if isinstance(error, httpx.HTTPError | httpx.RequestError):
                return error
        except ImportError:
            pass

        # Check for botocore exceptions that should pass through
        try:
            from botocore.exceptions import (
                BotoCoreError,
                ClientError,
                NoCredentialsError,
            )

            if isinstance(error, BotoCoreError | ClientError | NoCredentialsError):
                return error
        except ImportError:
            pass

        # Check for specific LiteLLM exceptions
        try:
            from litellm import AuthenticationError as LiteLLMAuthError
            from litellm import RateLimitError as LiteLLMRateLimit

            if isinstance(error, LiteLLMAuthError):
                return ModelAuthenticationError(error_message)
            elif isinstance(error, LiteLLMRateLimit):
                return ModelRateLimitError(error_message)
        except ImportError:
            pass

        # Check exception type
        if isinstance(error, asyncio.TimeoutError):
            return ModelTimeoutError(error_message)
        elif isinstance(error, ConnectionError | OSError):
            # These should pass through as the tests expect them
            return error
        elif isinstance(error, KeyError | ValueError):
            # JSON parsing errors should be validation errors
            return ModelValidationError(error_message)

        # Pattern matching for error content
        error_str = str(error).lower()

        if any(
            term in error_str
            for term in [
                "api_key",
                "key",
                "unauthorized",
                "forbidden",
                "permissions",
                "access denied",
                "accessdenied",
            ]
        ):
            return ModelAuthenticationError(error_message)
        elif any(
            term in error_str for term in ["rate limit", "quota", "too many requests"]
        ):
            return ModelRateLimitError(error_message)
        elif any(
            term in error_str for term in ["timeout", "timed out", "read timeout"]
        ):
            return ModelTimeoutError(error_message)
        elif any(
            term in error_str
            for term in ["connection", "connect", "network", "unreachable", "refused"]
        ):
            return ModelConnectionError(error_message)
        elif any(term in error_str for term in ["validation", "invalid", "malformed"]):
            return ModelValidationError(error_message)
        else:
            # Default to the original generic ModelBackendError for unclassified errors
            from .exceptions import ModelBackendError

            return ModelBackendError(error_message)

    def _prepare_messages(self, thread: Thread) -> list[dict[str, str]]:
        """Prepare messages in standard format"""
        messages = [{"role": "system", "content": thread.system_prompt}]

        for msg in thread.conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages


class BedrockBackend(ModelBackendInterface):
    """AWS Bedrock backend implementation"""

    def __init__(self):
        super().__init__("bedrock")

    def _get_model_name(self, thread: Thread) -> str:
        return thread.model_name or self.config.model.bedrock_model_id

    async def call_model(self, thread: Thread) -> str:
        async with error_context(
            operation_name="bedrock_model_call",
            user_context={"thread_name": thread.name, "model_backend": "bedrock"},
        ) as ctx:
            try:
                import boto3
            except ImportError as e:
                raise ConfigurationError(
                    "boto3 library not installed for Bedrock backend"
                ) from e

            try:
                model_config = self.get_model_config(thread)
                ctx["model_name"] = model_config.model_name
                client = boto3.client("bedrock-runtime", region_name="us-east-1")

                # Validate model ID with security context
                from .security import validate_model_id

                is_valid, error_msg = validate_model_id(model_config.model_name)
                if not is_valid:
                    raise ModelValidationError(
                        f"Invalid model ID: {error_msg}",
                        validation_context={
                            "model_name": model_config.model_name,
                            "validation_error": error_msg,
                        },
                    )

                # Prepare messages
                messages = []
                for msg in thread.conversation_history:
                    messages.append(
                        {
                            "role": msg["role"],
                            "content": [{"text": msg["content"]}],
                        }
                    )

                ctx["message_count"] = len(messages)

                # Make the API call
                response = client.converse(
                    modelId=model_config.model_name,
                    messages=messages,
                    system=[{"text": thread.system_prompt}],
                    inferenceConfig={
                        "maxTokens": model_config.max_tokens,
                        "temperature": model_config.temperature,
                    },
                )

                result = response["output"]["message"]["content"][0]["text"]
                ctx["response_length"] = len(result)
                return result

            except ValueError as e:
                # Model validation error (already converted above)
                if "Invalid model ID" in str(e):
                    raise  # Re-raise ModelValidationError with context
                else:
                    raise ModelValidationError(str(e)) from e

            except Exception as e:
                # Enhanced error classification with network context
                error_str = str(e).lower()

                # Network-related errors
                if any(term in error_str for term in ["connection", "dns", "resolve"]):
                    raise NetworkConnectivityError(
                        f"Network connectivity error: {str(e)}",
                        network_context={
                            "backend": "bedrock",
                            "region": "us-east-1",
                            "error_type": "connectivity",
                        },
                    ) from e
                elif any(term in error_str for term in ["timeout", "timed out"]):
                    raise NetworkTimeoutError(
                        f"Network timeout error: {str(e)}",
                        network_context={
                            "backend": "bedrock",
                            "timeout_type": "network",
                            "model": model_config.model_name,
                        },
                    ) from e

                # Authentication/Authorization errors
                elif any(
                    term in error_str
                    for term in [
                        "api_key",
                        "key",
                        "unauthorized",
                        "forbidden",
                        "credentials",
                        "access denied",
                        "accessdenied",
                    ]
                ):
                    raise ModelAuthenticationError(
                        "AWS credentials invalid or insufficient permissions",
                        security_context={
                            "backend": "bedrock",
                            "auth_error_type": "credentials",
                        },
                    ) from e

                # Rate limiting
                elif any(
                    term in error_str
                    for term in ["throttling", "throttled", "rate", "quota"]
                ):
                    raise ModelRateLimitError(
                        "Bedrock API rate limit exceeded",
                        network_context={
                            "backend": "bedrock",
                            "rate_limit_type": "api",
                        },
                    ) from e

                # Model-specific errors
                elif any(
                    term in error_str for term in ["model", "not found", "invalid"]
                ):
                    raise ModelValidationError(
                        f"Model error: {str(e)}",
                        validation_context={
                            "backend": "bedrock",
                            "model_name": model_config.model_name,
                            "error_type": "model_not_found",
                        },
                    ) from e

                # Generic model backend error with context
                else:
                    raise ModelBackendError(
                        f"Bedrock API call failed: {sanitize_error_message(str(e))}",
                        performance_context={
                            "backend": "bedrock",
                            "operation": "converse_api",
                            "model": model_config.model_name,
                        },
                    ) from e

    async def call_model_stream(
        self, thread: Thread
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            import boto3

            client = boto3.client("bedrock-runtime")
            model_config = self.get_model_config(thread)

            messages = []
            for msg in thread.conversation_history:
                messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"text": msg["content"]}],
                    }
                )

            from .security import validate_model_id

            is_valid, error_msg = validate_model_id(model_config.model_name)
            if not is_valid:
                raise ValueError(f"Invalid model ID: {error_msg}")

            response = client.converse_stream(
                modelId=model_config.model_name,
                messages=messages,
                system=[{"text": thread.system_prompt}],
                inferenceConfig={
                    "maxTokens": model_config.max_tokens,
                    "temperature": model_config.temperature,
                },
            )

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

        except ImportError:
            yield {
                "type": "error",
                "content": self._format_error_response(
                    "boto3 library not installed",
                    "configuration_error",
                    {"thread": thread.name},
                ),
                "thread_name": thread.name,
            }
        except ValueError as e:
            yield {
                "type": "error",
                "content": self._format_error_response(
                    str(e), "validation_error", {"thread": thread.name}
                ),
                "thread_name": thread.name,
            }
        except Exception as e:
            error_type, error_message = self._get_error_type_and_message(e)
            yield {
                "type": "error",
                "content": self._format_error_response(
                    error_message, error_type, {"thread": thread.name}
                ),
                "thread_name": thread.name,
            }


class LiteLLMBackend(ModelBackendInterface):
    """LiteLLM backend implementation"""

    def __init__(self):
        super().__init__("litellm")

    def _get_model_name(self, thread: Thread) -> str:
        return thread.model_name or self.config.model.litellm_model

    async def call_model(self, thread: Thread) -> str:
        try:
            import litellm

            model_config = self.get_model_config(thread)
            messages = self._prepare_messages(thread)

            # Validate model name
            from .security import validate_model_id

            is_valid, error_msg = validate_model_id(model_config.model_name)
            if not is_valid:
                raise ValueError(f"Invalid model ID: {error_msg}")

            response = await litellm.acompletion(
                model=model_config.model_name,
                messages=messages,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
            )

            return response.choices[0].message.content

        except ImportError as e:
            raise ConfigurationError("litellm library not installed") from e
        except ValueError as e:
            # Model validation error
            raise ModelValidationError(str(e)) from e
        except Exception as e:
            # Map to specific exception types based on error
            error_type, error_message = self._get_error_type_and_message(e)

            if error_type == "credentials_error":
                raise ModelAuthenticationError(error_message) from e
            elif error_type == "connection_error":
                raise ModelConnectionError(error_message) from e
            elif error_type == "model_not_found":
                raise ModelValidationError(error_message) from e
            elif "rate" in str(e).lower() or "limit" in str(e).lower():
                raise ModelRateLimitError("Rate limit exceeded") from e
            elif "timeout" in str(e).lower():
                raise ModelTimeoutError("Request timed out") from e
            else:
                # Generic model backend error
                raise ModelBackendError(
                    self._format_error_response(error_message, error_type, {})
                ) from e


class OllamaBackend(ModelBackendInterface):
    """Ollama backend implementation"""

    def __init__(self):
        super().__init__("ollama")

    def _get_model_name(self, thread: Thread) -> str:
        return thread.model_name or self.config.model.ollama_model

    async def call_model(self, thread: Thread) -> str:
        try:
            import httpx

            model_config = self.get_model_config(thread)
            messages = self._prepare_messages(thread)

            # Validate model name
            from .security import validate_model_id

            is_valid, error_msg = validate_model_id(model_config.model_name)
            if not is_valid:
                raise ValueError(f"Invalid model ID: {error_msg}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    str(self.config.model.ollama_host) + "/api/chat",
                    json={
                        "model": model_config.model_name,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": model_config.temperature,
                            "num_predict": model_config.max_tokens,
                        },
                    },
                    timeout=model_config.timeout_seconds,
                )

                response.raise_for_status()
                result = response.json()
                return result["message"]["content"]

        except ImportError as e:
            raise ConfigurationError(
                "httpx library not installed for Ollama backend"
            ) from e
        except ValueError as e:
            # Model validation error
            raise ModelValidationError(str(e)) from e
        except httpx.ConnectError as e:
            raise ModelConnectionError(
                f"Failed to connect to Ollama at {self.config.model.ollama_host}"
            ) from e
        except httpx.TimeoutException as e:
            raise ModelTimeoutError(
                f"Request to Ollama timed out after {model_config.timeout_seconds}s"
            ) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ModelValidationError(
                    f"Model '{model_config.model_name}' not found in Ollama"
                ) from e
            elif e.response.status_code == 429:
                raise ModelRateLimitError("Ollama rate limit exceeded") from e
            else:
                raise ModelBackendError(
                    f"Ollama HTTP error: {e.response.status_code}"
                ) from e
        except Exception as e:
            # Map to specific exception types based on error
            error_type, error_message = self._get_error_type_and_message(e)

            if "connection" in str(e).lower():
                raise ModelConnectionError(error_message) from e
            elif "timeout" in str(e).lower():
                raise ModelTimeoutError(error_message) from e
            else:
                # Generic model backend error
                raise ModelBackendError(
                    self._format_error_response(
                        error_message,
                        error_type,
                        {"host": self.config.model.ollama_host},
                    )
                ) from e


BACKEND_REGISTRY: dict[str, ModelBackendInterface] = {
    "bedrock": BedrockBackend(),
    "litellm": LiteLLMBackend(),
    "ollama": OllamaBackend(),
}


def get_backend_interface(backend_name: str) -> ModelBackendInterface:
    """Get backend interface by name"""
    if backend_name not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend: {backend_name}")
    return BACKEND_REGISTRY[backend_name]
