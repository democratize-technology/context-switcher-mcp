"""Factory for creating backend instances without duplication"""

from .logging_base import get_logger
from typing import Dict, Optional

from .models import ModelBackend, Thread
from .backend_interface import ModelBackendInterface, BedrockBackend
from .exceptions import ConfigurationError, ModelBackendError
from .profiling_wrapper import create_profiling_wrapper

logger = get_logger(__name__)


class LiteLLMBackend(ModelBackendInterface):
    """LiteLLM backend implementation"""

    def __init__(self):
        super().__init__("litellm")

    def _get_model_name(self, thread: Thread) -> str:
        return thread.model_name or self.config.model.litellm_model

    async def call_model(self, thread: Thread) -> str:
        try:
            from litellm import acompletion

            model_config = self.get_model_config(thread)
            messages = self._prepare_messages(thread)

            response = await acompletion(
                model=model_config.model_name,
                messages=messages,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                timeout=model_config.timeout_seconds,
            )

            return response.choices[0].message.content

        except ImportError:
            raise ConfigurationError(
                "LiteLLM not installed. Install with: pip install litellm"
            )
        except Exception as e:
            error_type, error_msg = self._get_error_type_and_message(e)
            raise ModelBackendError(error_msg) from e


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

            async with httpx.AsyncClient(
                timeout=model_config.timeout_seconds
            ) as client:
                response = await client.post(
                    f"{self.config.model.ollama_host}/api/chat",
                    json={
                        "model": model_config.model_name,
                        "messages": messages,
                        "options": {
                            "temperature": model_config.temperature,
                            "num_predict": model_config.max_tokens,
                        },
                        "stream": False,
                    },
                )
                response.raise_for_status()
                return response.json()["message"]["content"]

        except ImportError:
            raise ConfigurationError(
                "httpx not installed. Install with: pip install httpx"
            )
        except Exception as e:
            error_type, error_msg = self._get_error_type_and_message(e)
            raise ModelBackendError(error_msg) from e


class BackendFactory:
    """Factory for creating and managing backend instances"""

    _instances: Dict[ModelBackend, Optional[ModelBackendInterface]] = {}
    _config = None

    @classmethod
    def get_backend(cls, backend_type: ModelBackend) -> ModelBackendInterface:
        """Get or create a backend instance

        Args:
            backend_type: The type of backend to get

        Returns:
            Backend instance

        Raises:
            ConfigurationError: If backend is not supported
        """
        if backend_type not in cls._instances:
            cls._instances[backend_type] = cls._create_backend(backend_type)

        backend = cls._instances[backend_type]
        if backend is None:
            raise ConfigurationError(f"Backend {backend_type.value} is not available")

        return backend

    @classmethod
    def _create_backend(
        cls, backend_type: ModelBackend
    ) -> Optional[ModelBackendInterface]:
        """Create a new backend instance

        Args:
            backend_type: The type of backend to create

        Returns:
            Backend instance or None if not available
        """
        try:
            base_backend = None
            if backend_type == ModelBackend.BEDROCK:
                base_backend = BedrockBackend()
            elif backend_type == ModelBackend.LITELLM:
                base_backend = LiteLLMBackend()
            elif backend_type == ModelBackend.OLLAMA:
                base_backend = OllamaBackend()
            else:
                logger.error(f"Unknown backend type: {backend_type}")
                return None

            # Wrap with profiling if available
            if base_backend:
                return create_profiling_wrapper(base_backend)
            return None

        except Exception as e:
            logger.warning(f"Failed to create {backend_type.value} backend: {e}")
            return None

    @classmethod
    def reset(cls) -> None:
        """Reset all backend instances (useful for testing)"""
        cls._instances.clear()

    @classmethod
    def is_backend_available(cls, backend_type: ModelBackend) -> bool:
        """Check if a backend is available

        Args:
            backend_type: The backend to check

        Returns:
            True if backend is available
        """
        try:
            backend = cls.get_backend(backend_type)
            return backend is not None
        except ConfigurationError:
            return False

    @classmethod
    def get_available_backends(cls) -> list[ModelBackend]:
        """Get list of available backends

        Returns:
            List of available backend types
        """
        available = []
        for backend_type in ModelBackend:
            if cls.is_backend_available(backend_type):
                available.append(backend_type)
        return available


def get_backend_for_thread(thread: Thread) -> ModelBackendInterface:
    """Convenience function to get backend for a thread

    Args:
        thread: Thread with backend configuration

    Returns:
        Backend instance configured for the thread
    """
    return BackendFactory.get_backend(thread.model_backend)
