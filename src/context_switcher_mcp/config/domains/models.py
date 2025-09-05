"""Model and LLM backend configuration

This module handles configuration for all supported LLM backends:
- AWS Bedrock (Anthropic Claude models)
- LiteLLM (Multi-provider LLM access)
- Ollama (Local LLM hosting)

Key features:
- Backend-specific validation rules
- Token and character limits
- Model identifier validation
- URL validation for remote endpoints
- Temperature and parameter constraints
"""

import re
from typing import Any

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Configuration for LLM model backends and parameters

    Handles all model-related configuration including backend-specific
    settings, token limits, and generation parameters.
    """

    model_config = SettingsConfigDict(
        env_prefix="CS_", case_sensitive=False, extra="forbid", validate_assignment=True
    )

    # General model parameters with validation
    default_max_tokens: int = Field(
        default=2048,
        ge=1,
        le=200000,
        description="Default maximum tokens for model responses",
        alias="CS_MAX_TOKENS",
    )

    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default temperature for response generation",
        alias="CS_TEMPERATURE",
    )

    # Character limits for different model classes
    max_chars_opus: int = Field(
        default=20000,
        ge=1000,
        le=1000000,
        description="Maximum characters for Claude Opus models",
        alias="CS_MAX_CHARS_OPUS",
    )

    max_chars_haiku: int = Field(
        default=180000,
        ge=1000,
        le=2000000,
        description="Maximum characters for Claude Haiku models",
        alias="CS_MAX_CHARS_HAIKU",
    )

    # AWS Bedrock configuration
    bedrock_model_id: str = Field(
        default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        pattern=r"^[a-z0-9\.\-:]+$",
        description="AWS Bedrock model identifier",
        alias="BEDROCK_MODEL_ID",
    )

    bedrock_region: str = Field(
        default="us-east-1",
        pattern=r"^[a-z0-9\-]+$",
        description="AWS region for Bedrock access",
        alias="AWS_DEFAULT_REGION",
    )

    bedrock_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for Bedrock API calls",
        alias="CS_MODEL_BEDROCK_TIMEOUT",
    )

    # LiteLLM configuration
    litellm_model: str = Field(
        default="gpt-4",
        min_length=1,
        max_length=100,
        description="LiteLLM model identifier",
        alias="LITELLM_MODEL",
    )

    litellm_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for LiteLLM API calls",
        alias="CS_MODEL_LITELLM_TIMEOUT",
    )

    # Ollama configuration
    ollama_model: str = Field(
        default="llama3.2",
        pattern=r"^[a-zA-Z0-9\.\-_:]+$",
        description="Ollama model identifier",
        alias="OLLAMA_MODEL",
    )

    ollama_host: HttpUrl = Field(
        default="http://localhost:11434",
        description="Ollama service endpoint URL",
        alias="OLLAMA_HOST",
    )

    ollama_timeout_seconds: float = Field(
        default=60.0,  # Ollama can be slower for local inference
        ge=1.0,
        le=600.0,
        description="Timeout for Ollama API calls",
        alias="CS_MODEL_OLLAMA_TIMEOUT",
    )

    # Backend selection and circuit breaker settings
    enabled_backends: list[str] = Field(
        default=["bedrock", "litellm", "ollama"],
        description="List of enabled model backends",
        alias="CS_MODEL_ENABLED_BACKENDS",
    )

    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Failures before circuit breaker opens",
        alias="CS_MODEL_CIRCUIT_FAILURE_THRESHOLD",
    )

    circuit_breaker_timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Circuit breaker timeout before retry attempts",
        alias="CS_MODEL_CIRCUIT_TIMEOUT",
    )

    # Retry configuration for model calls
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed model calls",
        alias="CS_MODEL_MAX_RETRIES",
    )

    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Initial delay between retry attempts",
        alias="CS_MODEL_RETRY_DELAY",
    )

    retry_backoff_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier for retries",
        alias="CS_MODEL_RETRY_BACKOFF",
    )

    @field_validator("bedrock_model_id")
    @classmethod
    def validate_bedrock_model_id(cls, v: str) -> str:
        """Validate Bedrock model ID follows AWS naming convention"""
        # Pattern: region.provider.model-name-version:revision
        if not re.match(r"^[a-z]{2}(-[a-z]+)*\.[a-z]+\.[a-z0-9\-]+:[0-9]+$", v):
            raise ValueError(
                "Bedrock model ID must follow format: region.provider.model-name:version"
            )
        return v

    @field_validator("enabled_backends")
    @classmethod
    def validate_enabled_backends(cls, v: list[str]) -> list[str]:
        """Validate that enabled backends are supported"""
        valid_backends = {"bedrock", "litellm", "ollama"}
        invalid_backends = set(v) - valid_backends
        if invalid_backends:
            raise ValueError(
                f"Invalid backends: {invalid_backends}. Supported: {valid_backends}"
            )
        return v

    def get_backend_config(self, backend: str) -> dict[str, Any]:
        """Get configuration for a specific backend

        Args:
            backend: Backend name ("bedrock", "litellm", or "ollama")

        Returns:
            Backend-specific configuration dictionary

        Raises:
            ValueError: If backend is not supported
        """
        if backend not in self.enabled_backends:
            raise ValueError(f"Backend '{backend}' is not enabled")

        if backend == "bedrock":
            return {
                "model_id": self.bedrock_model_id,
                "region": self.bedrock_region,
                "timeout_seconds": self.bedrock_timeout_seconds,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay_seconds,
            }
        elif backend == "litellm":
            return {
                "model": self.litellm_model,
                "timeout_seconds": self.litellm_timeout_seconds,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay_seconds,
            }
        elif backend == "ollama":
            return {
                "model": self.ollama_model,
                "host": str(self.ollama_host),
                "timeout_seconds": self.ollama_timeout_seconds,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay_seconds,
            }
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def is_backend_enabled(self, backend: str) -> bool:
        """Check if a backend is enabled

        Args:
            backend: Backend name to check

        Returns:
            True if backend is enabled
        """
        return backend in self.enabled_backends

    def get_max_chars_for_model(self, model_name: str) -> int:
        """Get maximum character limit for a specific model

        Args:
            model_name: Model identifier

        Returns:
            Maximum character limit
        """
        # Simple heuristic based on model name
        model_lower = model_name.lower()
        if "opus" in model_lower:
            return self.max_chars_opus
        elif "haiku" in model_lower:
            return self.max_chars_haiku
        else:
            # Default to more conservative limit
            return min(self.max_chars_opus, self.max_chars_haiku)
