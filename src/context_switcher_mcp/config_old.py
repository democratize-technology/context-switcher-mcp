"""Centralized configuration management for Context Switcher MCP

This module provides the main configuration interface using the new clean
architecture design. It uses dependency injection to avoid circular
dependencies while maintaining backward compatibility.

For new code, use the dependency injection container to get configuration
instances rather than importing directly.
"""

import os
from .logging_base import get_logger
import warnings
from dataclasses import dataclass
from typing import Optional

# Import from our clean architecture modules
from .types import ModelBackend
from .config_base import (
    BaseConfigurationProvider,
    ConfigurationError,
    BackendConfiguration,
)
from .container import get_container
from .protocols import ConfigurationProvider, ConfigurationMigrator

# Import validated config if available
try:
    from .validated_config import (
        ValidatedContextSwitcherConfig,
        load_validated_config,
    )

    _VALIDATED_CONFIG_AVAILABLE = True
except ImportError as e:
    _VALIDATED_CONFIG_AVAILABLE = False
    warnings.warn(
        f"Validated configuration system not available: {e}. "
        "Using legacy configuration system.",
        UserWarning,
    )

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model backends (Legacy compatibility class)

    Note: This class is maintained for backward compatibility.
    New code should use BackendConfiguration from config_base.
    """

    default_max_tokens: int = 2048
    default_temperature: float = 0.7
    max_chars_opus: int = 20000
    max_chars_haiku: int = 180000

    def to_backend_config(self, backend: ModelBackend) -> BackendConfiguration:
        """Convert to new BackendConfiguration format"""
        return BackendConfiguration(
            backend_type=backend,
            model_specific_config={
                "default_max_tokens": self.default_max_tokens,
                "default_temperature": self.default_temperature,
                "max_chars_opus": self.max_chars_opus,
                "max_chars_haiku": self.max_chars_haiku,
            },
        )

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
class ContextSwitcherConfig(BaseConfigurationProvider):
    """Main configuration class combining all settings (Legacy compatibility class)

    Note: This class is maintained for backward compatibility but now uses
    the clean architecture design with dependency injection.
    """

    def __init__(self):
        """Initialize configuration from environment variables

        This constructor now uses the new configuration system with dependency
        injection to avoid circular dependencies.
        """
        # Initialize base configuration provider
        super().__init__()

        # Try to use validated configuration system
        if _VALIDATED_CONFIG_AVAILABLE:
            try:
                self._use_validated_config()
                logger.debug("Using validated configuration system")
                return
            except Exception as e:
                logger.warning(
                    f"Failed to use validated config, falling back to legacy: {e}"
                )

        # Fallback to legacy configuration with error handling
        self._use_legacy_config()

    def _use_validated_config(self) -> None:
        """Initialize configuration using validated system"""
        try:
            validated_config = load_validated_config()

            # Convert validated config to legacy format for compatibility
            self.model = ModelConfig(
                default_max_tokens=getattr(
                    validated_config.model, "default_max_tokens", 2048
                ),
                default_temperature=getattr(
                    validated_config.model, "default_temperature", 0.7
                ),
                max_chars_opus=getattr(validated_config.model, "max_chars_opus", 20000),
                max_chars_haiku=getattr(
                    validated_config.model, "max_chars_haiku", 180000
                ),
            )

            # Use validated config for session settings
            session_config = (
                validated_config.session
                if hasattr(validated_config, "session")
                else None
            )
            if session_config:
                self.session = SessionConfig(
                    default_ttl_hours=getattr(session_config, "default_ttl_hours", 24),
                    cleanup_interval_seconds=getattr(
                        session_config, "cleanup_interval_seconds", 600
                    ),
                    max_active_sessions=getattr(
                        session_config, "max_active_sessions", 1000
                    ),
                )
            else:
                self.session = SessionConfig()

            # Store reference to validated config for advanced features
            self._validated_config = validated_config

        except Exception as e:
            logger.warning(f"Failed to load validated config: {e}")
            raise

    def _use_legacy_config(self) -> None:
        """Initialize configuration using legacy system with error handling"""
        try:
            self.model = ModelConfig(
                default_max_tokens=self._safe_int("CS_MAX_TOKENS", 2048),
                default_temperature=self._safe_float("CS_TEMPERATURE", 0.7),
                max_chars_opus=self._safe_int("CS_MAX_CHARS_OPUS", 20000),
                max_chars_haiku=self._safe_int("CS_MAX_CHARS_HAIKU", 180000),
                bedrock_model_id=os.getenv(
                    "BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
                ),
                litellm_model=os.getenv("LITELLM_MODEL", "gpt-4"),
                ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
                ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            )

            self.circuit_breaker = CircuitBreakerConfig(
                failure_threshold=self._safe_int("CS_CIRCUIT_FAILURE_THRESHOLD", 5),
                timeout_seconds=self._safe_int("CS_CIRCUIT_TIMEOUT_SECONDS", 300),
            )

            self.validation = ValidationConfig(
                max_session_id_length=self._safe_int("CS_MAX_SESSION_ID_LENGTH", 100),
                max_topic_length=self._safe_int("CS_MAX_TOPIC_LENGTH", 1000),
                max_perspective_name_length=self._safe_int(
                    "CS_MAX_PERSPECTIVE_NAME_LENGTH", 100
                ),
                max_custom_prompt_length=self._safe_int(
                    "CS_MAX_CUSTOM_PROMPT_LENGTH", 10000
                ),
            )

            self.session = SessionConfig(
                default_ttl_hours=self._safe_int("CS_SESSION_TTL_HOURS", 24),
                cleanup_interval_seconds=self._safe_int("CS_CLEANUP_INTERVAL", 600),
                max_active_sessions=self._safe_int("CS_MAX_SESSIONS", 1000),
            )

            self.metrics = MetricsConfig(
                max_history_size=self._safe_int("CS_METRICS_HISTORY_SIZE", 1000),
                retention_days=self._safe_int("CS_METRICS_RETENTION_DAYS", 7),
            )

            self.retry = RetryConfig(
                max_retries=self._safe_int("CS_MAX_RETRIES", 3),
                initial_delay=self._safe_float("CS_RETRY_DELAY", 1.0),
                backoff_factor=self._safe_float("CS_BACKOFF_FACTOR", 2.0),
                max_delay=self._safe_float("CS_MAX_RETRY_DELAY", 60.0),
            )

            self.reasoning = ReasoningConfig(
                max_iterations=self._safe_int("CS_REASONING_MAX_ITERATIONS", 20),
                cot_timeout_seconds=self._safe_float("CS_COT_TIMEOUT", 30.0),
                summary_timeout_seconds=self._safe_float("CS_SUMMARY_TIMEOUT", 5.0),
                default_temperature=self._safe_float("CS_REASONING_TEMPERATURE", 0.7),
            )

            self.server = ServerConfig(
                host=os.getenv("CS_HOST", "localhost"),
                port=self._safe_int("CS_PORT", 3023),
                log_level=os.getenv("CS_LOG_LEVEL", "INFO"),
            )

            self.profiling = ProfilingConfig(
                enabled=self._safe_bool("CS_PROFILING_ENABLED", True),
                level=os.getenv("CS_PROFILING_LEVEL", "standard"),
                sampling_rate=self._safe_float("CS_PROFILING_SAMPLING_RATE", 0.1),
                track_tokens=self._safe_bool("CS_PROFILING_TRACK_TOKENS", True),
                track_costs=self._safe_bool("CS_PROFILING_TRACK_COSTS", True),
                track_memory=self._safe_bool("CS_PROFILING_TRACK_MEMORY", False),
                track_network_timing=self._safe_bool(
                    "CS_PROFILING_TRACK_NETWORK", True
                ),
                max_history_size=self._safe_int("CS_PROFILING_MAX_HISTORY", 10000),
                cost_alert_threshold_usd=self._safe_float(
                    "CS_PROFILING_COST_ALERT", 100.0
                ),
                latency_alert_threshold_s=self._safe_float(
                    "CS_PROFILING_LATENCY_ALERT", 30.0
                ),
                memory_alert_threshold_mb=self._safe_float(
                    "CS_PROFILING_MEMORY_ALERT", 1000.0
                ),
            )

            self._validated_config = None  # No validated config available

        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise ConfigurationError(f"Configuration initialization failed: {e}") from e

    def _safe_int(self, env_var: str, default: int) -> int:
        """Safely convert environment variable to int"""
        try:
            value = os.getenv(env_var)
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid value for {env_var}: {os.getenv(env_var)}. Using default: {default}"
            )
            return default

    def _safe_float(self, env_var: str, default: float) -> float:
        """Safely convert environment variable to float"""
        try:
            value = os.getenv(env_var)
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid value for {env_var}: {os.getenv(env_var)}. Using default: {default}"
            )
            return default

    def _safe_bool(self, env_var: str, default: bool) -> bool:
        """Safely convert environment variable to bool"""
        try:
            value = os.getenv(env_var)
            if value is None:
                return default
            return value.lower() in ("true", "1", "yes", "on")
        except (AttributeError, TypeError):
            logger.warning(
                f"Invalid value for {env_var}: {os.getenv(env_var)}. Using default: {default}"
            )
            return default

    def mask_sensitive_data(self) -> dict:
        """Return configuration with sensitive data masked for logging"""
        if hasattr(self, "_validated_config") and self._validated_config:
            return self._validated_config.mask_sensitive_data()
        else:
            # Basic masking for legacy config
            config_dict = {
                "model": {
                    "bedrock_model_id": self.model.bedrock_model_id,
                    "litellm_model": self.model.litellm_model,
                    "ollama_model": self.model.ollama_model,
                    "ollama_host": self.model.ollama_host,
                },
                "server": {
                    "host": self.server.host,
                    "port": self.server.port,
                    "log_level": self.server.log_level,
                },
            }
            return config_dict


# Exception class for configuration errors
if not _VALIDATED_CONFIG_AVAILABLE:

    class ConfigurationError(Exception):
        """Raised when configuration is invalid or unavailable"""

        pass


# Global configuration instance with enhanced error handling
config: Optional[ContextSwitcherConfig] = None

try:
    config = ContextSwitcherConfig()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    warnings.warn(
        f"Configuration loading failed: {e}. Some functionality may be limited.",
        UserWarning,
    )


def get_config() -> ContextSwitcherConfig:
    """Get the global configuration instance

    This function now uses dependency injection when available,
    falling back to the legacy singleton pattern for compatibility.

    Returns:
        The global configuration instance

    Raises:
        ConfigurationError: If configuration is not available or invalid
    """
    global config

    # Try to get from dependency container first
    try:
        container = get_container()
        if container.has_registration(ConfigurationProvider):
            provider = container.get(ConfigurationProvider)
            if isinstance(provider, ContextSwitcherConfig):
                return provider
    except Exception as e:
        logger.debug(f"Could not get config from DI container: {e}")

    # Fallback to singleton pattern
    if config is None:
        try:
            config = create_config_with_migration()
            logger.info("Configuration initialized successfully")
        except Exception as e:
            raise ConfigurationError(f"Configuration not available: {e}") from e
    return config


def reload_config() -> ContextSwitcherConfig:
    """Reload configuration from environment variables

    Returns:
        Newly loaded configuration instance

    Raises:
        ConfigurationError: If configuration reload fails
    """
    global config
    try:
        config = ContextSwitcherConfig()
        logger.info("Configuration reloaded successfully")
        return config
    except Exception as e:
        logger.error(f"Configuration reload failed: {e}")
        raise ConfigurationError(f"Failed to reload configuration: {e}") from e


# Add configuration validation utility
def validate_current_config() -> tuple[bool, list[str]]:
    """Validate the current configuration

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    try:
        current_config = get_config()

        # Basic validation checks
        if current_config.server.port <= 1024:
            issues.append("Server port should be > 1024 for non-privileged access")

        if (
            current_config.model.default_temperature < 0
            or current_config.model.default_temperature > 2
        ):
            issues.append("Temperature should be between 0 and 2")

        if current_config.retry.max_delay <= current_config.retry.initial_delay:
            issues.append("Max retry delay should be greater than initial delay")

        # Check if validated config is available for advanced validation
        if (
            hasattr(current_config, "_validated_config")
            and current_config._validated_config
        ):
            if not current_config._validated_config.is_production_ready:
                issues.append("Configuration is not production-ready")

        return len(issues) == 0, issues

    except Exception as e:
        issues.append(f"Configuration validation failed: {e}")
        return False, issues


def get_validated_config() -> ValidatedContextSwitcherConfig:
    """Get validated configuration instance (if available)

    Returns:
        Validated configuration instance

    Raises:
        ConfigurationError: If validated config is not available
    """
    if not _VALIDATED_CONFIG_AVAILABLE:
        raise ConfigurationError("Validated configuration system not available")

    current_config = get_config()
    if (
        hasattr(current_config, "_validated_config")
        and current_config._validated_config
    ):
        return current_config._validated_config

    # Fallback: create validated config directly
    return load_validated_config(validate_dependencies=False)


# Configuration factory functions using dependency injection
def create_config_with_migration() -> ContextSwitcherConfig:
    """Create configuration with migration support using dependency injection"""
    config = ContextSwitcherConfig()

    # Try to get migrator from dependency container
    try:
        container = get_container()
        if container.has_registration(ConfigurationMigrator):
            migrator = container.get(ConfigurationMigrator)
            logger.info("Found configuration migrator, checking for migration needs")

            # Check if migration is needed and apply if necessary
            # This is where we'd apply migration logic if we had old config
            # For now, just log that migrator is available
            logger.debug("Configuration migrator available for future migrations")

    except Exception as e:
        logger.debug(f"No configuration migrator available: {e}")

    # Register the config in the container
    try:
        container = get_container()
        container.register_instance(ConfigurationProvider, config)
    except Exception as e:
        logger.debug(f"Could not register config in DI container: {e}")

    return config


# Setup dependency injection for configuration
def setup_configuration_dependencies() -> None:
    """Setup configuration dependencies in the DI container"""
    try:
        container = get_container()

        # Register configuration provider factory
        def config_factory() -> ConfigurationProvider:
            return ContextSwitcherConfig()

        container.register_singleton_factory(ConfigurationProvider, config_factory)

        # Register migrator if available
        if _VALIDATED_CONFIG_AVAILABLE:
            try:
                # Import and register the migrator
                # We use a lambda to delay the import until needed
                def migrator_factory() -> ConfigurationMigrator:
                    # Import here to avoid circular dependency
                    from .config_migration import CompatibilityAdapter

                    return CompatibilityAdapter()

                container.register_singleton_factory(
                    ConfigurationMigrator, migrator_factory
                )
                logger.debug("Registered configuration migrator in DI container")

            except ImportError as e:
                logger.debug(f"Could not register configuration migrator: {e}")

        logger.debug("Configuration dependencies setup complete")

    except Exception as e:
        logger.warning(f"Failed to setup configuration dependencies: {e}")
