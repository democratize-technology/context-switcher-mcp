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
class ReasoningConfig:
    """Configuration for Chain of Thought reasoning"""

    max_iterations: int = 20
    cot_timeout_seconds: float = 30.0
    summary_timeout_seconds: float = 5.0
    default_temperature: float = 0.7


@dataclass
class ProfilingConfig:
    """Configuration for LLM profiling and monitoring"""

    enabled: bool = True
    level: str = "standard"  # disabled, basic, standard, detailed
    sampling_rate: float = 0.1  # Profile 10% of calls by default

    # Feature flags
    track_tokens: bool = True
    track_costs: bool = True
    track_memory: bool = False  # More expensive, disabled by default
    track_network_timing: bool = True

    # Storage settings
    max_history_size: int = 10000

    # Alert thresholds
    cost_alert_threshold_usd: float = 100.0  # Daily budget alert
    latency_alert_threshold_s: float = 30.0  # High latency alert
    memory_alert_threshold_mb: float = 1000.0  # High memory usage alert

    # Sampling rules - always profile these conditions
    always_profile_errors: bool = True
    always_profile_slow_calls: bool = True
    always_profile_expensive_calls: bool = True
    always_profile_circuit_breaker: bool = True


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
            max_perspective_name_length=int(
                os.getenv("CS_MAX_PERSPECTIVE_NAME_LENGTH", "100")
            ),
            max_custom_prompt_length=int(
                os.getenv("CS_MAX_CUSTOM_PROMPT_LENGTH", "10000")
            ),
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

        self.reasoning = ReasoningConfig(
            max_iterations=int(os.getenv("CS_REASONING_MAX_ITERATIONS", "20")),
            cot_timeout_seconds=float(os.getenv("CS_COT_TIMEOUT", "30.0")),
            summary_timeout_seconds=float(os.getenv("CS_SUMMARY_TIMEOUT", "5.0")),
            default_temperature=float(os.getenv("CS_REASONING_TEMPERATURE", "0.7")),
        )

        self.server = ServerConfig(
            host=os.getenv("CS_HOST", "localhost"),
            port=int(os.getenv("CS_PORT", "3023")),
            log_level=os.getenv("CS_LOG_LEVEL", "INFO"),
        )

        self.profiling = ProfilingConfig(
            enabled=os.getenv("CS_PROFILING_ENABLED", "true").lower() == "true",
            level=os.getenv("CS_PROFILING_LEVEL", "standard"),
            sampling_rate=float(os.getenv("CS_PROFILING_SAMPLING_RATE", "0.1")),
            track_tokens=os.getenv("CS_PROFILING_TRACK_TOKENS", "true").lower()
            == "true",
            track_costs=os.getenv("CS_PROFILING_TRACK_COSTS", "true").lower() == "true",
            track_memory=os.getenv("CS_PROFILING_TRACK_MEMORY", "false").lower()
            == "true",
            track_network_timing=os.getenv("CS_PROFILING_TRACK_NETWORK", "true").lower()
            == "true",
            max_history_size=int(os.getenv("CS_PROFILING_MAX_HISTORY", "10000")),
            cost_alert_threshold_usd=float(
                os.getenv("CS_PROFILING_COST_ALERT", "100.0")
            ),
            latency_alert_threshold_s=float(
                os.getenv("CS_PROFILING_LATENCY_ALERT", "30.0")
            ),
            memory_alert_threshold_mb=float(
                os.getenv("CS_PROFILING_MEMORY_ALERT", "1000.0")
            ),
        )


# Global configuration instance with enhanced error handling
try:
    config = ContextSwitcherConfig()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    # Create a minimal fallback configuration to prevent total failure
    config = None
    warnings.warn(
        f"Configuration loading failed: {e}. Some functionality may be limited.",
        UserWarning,
    )


def get_config() -> ContextSwitcherConfig:
    """Get the global configuration instance"""
    return config


def reload_config() -> ContextSwitcherConfig:
    """Reload configuration from environment variables"""
    global config
    config = ContextSwitcherConfig()
    return config
