"""Centralized configuration management for Context Switcher MCP"""

import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model backends"""

    default_max_tokens: int = 2048
    default_temperature: float = 0.7
    max_chars_opus: int = 20000
    max_chars_haiku: int = 180000

    # Backend-specific defaults
    bedrock_model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    litellm_model: str = "gpt-4"
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5
    timeout_seconds: int = 300  # 5 minutes


@dataclass
class ValidationConfig:
    """Input validation configuration"""
    
    max_session_id_length: int = 100
    max_topic_length: int = 1000
    max_perspective_name_length: int = 100
    max_custom_prompt_length: int = 10000


@dataclass
class SessionConfig:
    """Session management configuration"""

    default_ttl_hours: int = 24
    cleanup_interval_seconds: int = 600  # 10 minutes
    max_active_sessions: int = 1000


@dataclass
class MetricsConfig:
    """Metrics collection configuration"""

    max_history_size: int = 1000
    retention_days: int = 7


@dataclass
class RetryConfig:
    """Retry configuration for LLM calls"""

    max_retries: int = 3
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0


@dataclass
class ServerConfig:
    """MCP server configuration"""

    host: str = "localhost"
    port: int = 3023
    log_level: str = "INFO"


@dataclass
class ContextSwitcherConfig:
    """Main configuration class combining all settings"""

    def __init__(self):
        """Initialize configuration from environment variables"""
        self.model = ModelConfig(
            default_max_tokens=int(os.getenv("CS_MAX_TOKENS", "2048")),
            default_temperature=float(os.getenv("CS_TEMPERATURE", "0.7")),
            max_chars_opus=int(os.getenv("CS_MAX_CHARS_OPUS", "20000")),
            max_chars_haiku=int(os.getenv("CS_MAX_CHARS_HAIKU", "180000")),
            bedrock_model_id=os.getenv(
                "BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            ),
            litellm_model=os.getenv("LITELLM_MODEL", "gpt-4"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )

        self.circuit_breaker = CircuitBreakerConfig(
            failure_threshold=int(os.getenv("CS_CIRCUIT_FAILURE_THRESHOLD", "5")),
            timeout_seconds=int(os.getenv("CS_CIRCUIT_TIMEOUT_SECONDS", "300")),
        )

        self.validation = ValidationConfig(
            max_session_id_length=int(os.getenv("CS_MAX_SESSION_ID_LENGTH", "100")),
            max_topic_length=int(os.getenv("CS_MAX_TOPIC_LENGTH", "1000")),
            max_perspective_name_length=int(os.getenv("CS_MAX_PERSPECTIVE_NAME_LENGTH", "100")),
            max_custom_prompt_length=int(os.getenv("CS_MAX_CUSTOM_PROMPT_LENGTH", "10000")),
        )

        self.session = SessionConfig(
            default_ttl_hours=int(os.getenv("CS_SESSION_TTL_HOURS", "24")),
            cleanup_interval_seconds=int(os.getenv("CS_CLEANUP_INTERVAL", "600")),
            max_active_sessions=int(os.getenv("CS_MAX_SESSIONS", "1000")),
        )

        self.metrics = MetricsConfig(
            max_history_size=int(os.getenv("CS_METRICS_HISTORY_SIZE", "1000")),
            retention_days=int(os.getenv("CS_METRICS_RETENTION_DAYS", "7")),
        )

        self.retry = RetryConfig(
            max_retries=int(os.getenv("CS_MAX_RETRIES", "3")),
            initial_delay=float(os.getenv("CS_RETRY_DELAY", "1.0")),
            backoff_factor=float(os.getenv("CS_BACKOFF_FACTOR", "2.0")),
            max_delay=float(os.getenv("CS_MAX_RETRY_DELAY", "60.0")),
        )

        self.server = ServerConfig(
            host=os.getenv("CS_HOST", "localhost"),
            port=int(os.getenv("CS_PORT", "3023")),
            log_level=os.getenv("CS_LOG_LEVEL", "INFO"),
        )


config = ContextSwitcherConfig()


def get_config() -> ContextSwitcherConfig:
    """Get the global configuration instance"""
    return config


def reload_config() -> ContextSwitcherConfig:
    """Reload configuration from environment variables"""
    global config
    config = ContextSwitcherConfig()
    return config
