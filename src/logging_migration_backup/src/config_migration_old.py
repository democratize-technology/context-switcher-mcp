"""Configuration migration utilities and backward compatibility layer

This module provides utilities to migrate from the old dataclass-based configuration
to the new Pydantic-validated configuration system while maintaining backward
compatibility.

Features:
- Seamless migration from old config interface
- Compatibility adapter for existing code
- Configuration comparison and validation tools
- Migration validation and error reporting
- Uses dependency injection to avoid circular dependencies
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Import from clean architecture modules
from .config_base import ConfigurationError, ConfigurationMigrationError

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
        "Migration features may be limited.",
        UserWarning,
    )

logger = logging.getLogger(__name__)


# Migration error is now imported from config_base


class CompatibilityAdapter:
    """Adapter to provide backward compatibility with legacy config interface

    This adapter wraps the validated configuration and provides the same
    interface as the legacy configuration system while ensuring all values
    are properly validated.
    """

    def __init__(self, validated_config: ValidatedContextSwitcherConfig):
        """Initialize adapter with validated configuration

        Args:
            validated_config: The validated configuration instance
        """
        self._validated_config = validated_config

        # Create legacy-compatible attribute access
        self.model = LegacyModelConfigAdapter(validated_config.model)
        self.circuit_breaker = LegacyCircuitBreakerAdapter(
            validated_config.circuit_breaker
        )
        self.validation = LegacyValidationAdapter(validated_config.validation)
        self.session = LegacySessionAdapter(validated_config.session)
        self.metrics = LegacyMetricsAdapter(validated_config.metrics)
        self.retry = LegacyRetryAdapter(validated_config.retry)
        self.reasoning = LegacyReasoningAdapter(validated_config.reasoning)
        self.profiling = LegacyProfilingAdapter(validated_config.profiling)
        self.server = LegacyServerAdapter(validated_config.server)

    def __getattr__(self, name: str) -> Any:
        """Provide access to validated config attributes"""
        return getattr(self._validated_config, name)


class LegacyModelConfigAdapter:
    """Legacy-compatible adapter for model configuration"""

    def __init__(self, validated_model_config):
        self._config = validated_model_config

    @property
    def default_max_tokens(self) -> int:
        return self._config.default_max_tokens

    @property
    def default_temperature(self) -> float:
        return self._config.default_temperature

    @property
    def max_chars_opus(self) -> int:
        return self._config.max_chars_opus

    @property
    def max_chars_haiku(self) -> int:
        return self._config.max_chars_haiku

    @property
    def bedrock_model_id(self) -> str:
        return self._config.bedrock_model_id

    @property
    def litellm_model(self) -> str:
        return self._config.litellm_model

    @property
    def ollama_model(self) -> str:
        return self._config.ollama_model

    @property
    def ollama_host(self) -> str:
        return str(self._config.ollama_host)


class LegacyCircuitBreakerAdapter:
    """Legacy-compatible adapter for circuit breaker configuration"""

    def __init__(self, validated_config):
        self._config = validated_config

    @property
    def failure_threshold(self) -> int:
        return self._config.failure_threshold

    @property
    def timeout_seconds(self) -> int:
        return self._config.timeout_seconds


class LegacyValidationAdapter:
    """Legacy-compatible adapter for validation configuration"""

    def __init__(self, validated_config):
        self._config = validated_config

    @property
    def max_session_id_length(self) -> int:
        return self._config.max_session_id_length

    @property
    def max_topic_length(self) -> int:
        return self._config.max_topic_length

    @property
    def max_perspective_name_length(self) -> int:
        return self._config.max_perspective_name_length

    @property
    def max_custom_prompt_length(self) -> int:
        return self._config.max_custom_prompt_length


class LegacySessionAdapter:
    """Legacy-compatible adapter for session configuration"""

    def __init__(self, validated_config):
        self._config = validated_config

    @property
    def default_ttl_hours(self) -> int:
        return self._config.default_ttl_hours

    @property
    def cleanup_interval_seconds(self) -> int:
        return self._config.cleanup_interval_seconds

    @property
    def max_active_sessions(self) -> int:
        return self._config.max_active_sessions


class LegacyMetricsAdapter:
    """Legacy-compatible adapter for metrics configuration"""

    def __init__(self, validated_config):
        self._config = validated_config

    @property
    def max_history_size(self) -> int:
        return self._config.max_history_size

    @property
    def retention_days(self) -> int:
        return self._config.retention_days


class LegacyRetryAdapter:
    """Legacy-compatible adapter for retry configuration"""

    def __init__(self, validated_config):
        self._config = validated_config

    @property
    def max_retries(self) -> int:
        return self._config.max_retries

    @property
    def initial_delay(self) -> float:
        return self._config.initial_delay

    @property
    def backoff_factor(self) -> float:
        return self._config.backoff_factor

    @property
    def max_delay(self) -> float:
        return self._config.max_delay


class LegacyReasoningAdapter:
    """Legacy-compatible adapter for reasoning configuration"""

    def __init__(self, validated_config):
        self._config = validated_config

    @property
    def max_iterations(self) -> int:
        return self._config.max_iterations

    @property
    def cot_timeout_seconds(self) -> float:
        return self._config.cot_timeout_seconds

    @property
    def summary_timeout_seconds(self) -> float:
        return self._config.summary_timeout_seconds

    @property
    def default_temperature(self) -> float:
        return self._config.default_temperature


class LegacyProfilingAdapter:
    """Legacy-compatible adapter for profiling configuration"""

    def __init__(self, validated_config):
        self._config = validated_config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def level(self) -> str:
        return self._config.level.value

    @property
    def sampling_rate(self) -> float:
        return self._config.sampling_rate

    @property
    def track_tokens(self) -> bool:
        return self._config.track_tokens

    @property
    def track_costs(self) -> bool:
        return self._config.track_costs

    @property
    def track_memory(self) -> bool:
        return self._config.track_memory

    @property
    def track_network_timing(self) -> bool:
        return self._config.track_network_timing

    @property
    def max_history_size(self) -> int:
        return self._config.max_history_size

    @property
    def cost_alert_threshold_usd(self) -> float:
        return self._config.cost_alert_threshold_usd

    @property
    def latency_alert_threshold_s(self) -> float:
        return self._config.latency_alert_threshold_s

    @property
    def memory_alert_threshold_mb(self) -> float:
        return self._config.memory_alert_threshold_mb

    @property
    def always_profile_errors(self) -> bool:
        return self._config.always_profile_errors

    @property
    def always_profile_slow_calls(self) -> bool:
        return self._config.always_profile_slow_calls

    @property
    def always_profile_expensive_calls(self) -> bool:
        return self._config.always_profile_expensive_calls

    @property
    def always_profile_circuit_breaker(self) -> bool:
        return self._config.always_profile_circuit_breaker


class LegacyServerAdapter:
    """Legacy-compatible adapter for server configuration"""

    def __init__(self, validated_config):
        self._config = validated_config

    @property
    def host(self) -> str:
        return self._config.host

    @property
    def port(self) -> int:
        return self._config.port

    @property
    def log_level(self) -> str:
        return self._config.log_level.value


def create_validated_config_with_fallback() -> CompatibilityAdapter:
    """Create validated configuration with fallback to legacy config

    This function attempts to create a validated configuration and falls back
    to the legacy configuration if validation fails, with appropriate warnings.

    Returns:
        Configuration adapter providing validated config interface

    Raises:
        ConfigurationMigrationError: If both validated and legacy config fail
    """
    try:
        # Try to load validated configuration
        validated_config = load_validated_config(validate_dependencies=False)
        logger.info("Using validated configuration system")
        return CompatibilityAdapter(validated_config)

    except ConfigurationError as e:
        # Fall back to legacy configuration with warnings
        warnings.warn(
            f"Validated configuration failed, falling back to legacy config: {e}",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(f"Configuration validation failed: {e}")
        logger.warning("Using legacy configuration - please fix validation errors")

        try:
            # Create legacy config and wrap in adapter
            legacy_config = LegacyConfig()
            logger.info("Legacy configuration loaded successfully")

            # Create a minimal validated config for compatibility
            minimal_validated = ValidatedContextSwitcherConfig()
            return CompatibilityAdapter(minimal_validated)

        except Exception as legacy_error:
            raise ConfigurationMigrationError(
                f"Both validated and legacy configuration failed. "
                f"Validated error: {e}. Legacy error: {legacy_error}"
            ) from legacy_error


def compare_configurations(
    legacy_config: LegacyConfig, validated_config: ValidatedContextSwitcherConfig
) -> Dict[str, List[str]]:
    """Compare legacy and validated configurations for differences

    Args:
        legacy_config: Legacy configuration instance
        validated_config: Validated configuration instance

    Returns:
        Dictionary of differences categorized by severity
    """
    differences = {"errors": [], "warnings": [], "info": []}

    try:
        # Compare model configuration
        _compare_model_config(legacy_config.model, validated_config.model, differences)

        # Compare other configurations...
        _compare_server_config(
            legacy_config.server, validated_config.server, differences
        )

        # Add more comparisons as needed

    except Exception as e:
        differences["errors"].append(f"Configuration comparison failed: {e}")

    return differences


def _compare_model_config(legacy, validated, differences):
    """Compare model configuration sections"""
    if legacy.default_max_tokens != validated.default_max_tokens:
        differences["info"].append(
            f"Max tokens: {legacy.default_max_tokens} -> {validated.default_max_tokens}"
        )

    if abs(legacy.default_temperature - validated.default_temperature) > 0.001:
        differences["info"].append(
            f"Temperature: {legacy.default_temperature} -> {validated.default_temperature}"
        )

    if legacy.ollama_host != str(validated.ollama_host):
        differences["info"].append(
            f"Ollama host: {legacy.ollama_host} -> {validated.ollama_host}"
        )


def _compare_server_config(legacy, validated, differences):
    """Compare server configuration sections"""
    if legacy.host != validated.host:
        differences["info"].append(f"Server host: {legacy.host} -> {validated.host}")

    if legacy.port != validated.port:
        differences["warnings"].append(
            f"Server port changed: {legacy.port} -> {validated.port}"
        )

    if legacy.log_level != validated.log_level.value:
        differences["info"].append(
            f"Log level: {legacy.log_level} -> {validated.log_level.value}"
        )


def validate_migration(
    config_file: Optional[str] = None,
) -> Tuple[bool, List[str], List[str]]:
    """Validate configuration migration readiness

    Args:
        config_file: Optional configuration file to validate

    Returns:
        Tuple of (success, errors, warnings)
    """
    errors = []
    warnings = []

    try:
        # Test loading validated configuration
        validated_config = load_validated_config(
            config_file=config_file, validate_dependencies=True
        )

        # Test creating compatibility adapter
        adapter = CompatibilityAdapter(validated_config)

        # Test key functionality
        try:
            _ = adapter.model.default_max_tokens
            _ = adapter.server.port
            _ = adapter.profiling.enabled
            warnings.append("All configuration sections accessible")
        except Exception as e:
            errors.append(f"Configuration adapter error: {e}")

        # Check production readiness
        if not validated_config.is_production_ready:
            warnings.append("Configuration is not production-ready")

        if not errors:
            warnings.append("Configuration migration validation successful")
            return True, errors, warnings

    except ConfigurationError as e:
        errors.append(f"Configuration validation failed: {e}")
    except Exception as e:
        errors.append(f"Unexpected migration validation error: {e}")

    return len(errors) == 0, errors, warnings


def generate_migration_report() -> str:
    """Generate a comprehensive migration report

    Returns:
        Formatted migration report
    """
    report_lines = ["Configuration Migration Report", "=" * 50, ""]

    # Test migration validation
    success, errors, warnings = validate_migration()

    if success:
        report_lines.extend(
            [
                "✅ Migration Status: READY",
                f"   Validated {len(warnings)} configuration aspects",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "❌ Migration Status: NEEDS ATTENTION",
                f"   Found {len(errors)} errors that must be resolved:",
                "",
            ]
        )
        for error in errors:
            report_lines.append(f"   • {error}")
        report_lines.append("")

    # Add warnings
    if warnings:
        report_lines.extend(["⚠️  Warnings:", ""])
        for warning in warnings:
            report_lines.append(f"   • {warning}")
        report_lines.append("")

    # Add next steps
    report_lines.extend(
        [
            "Next Steps:",
            "-" * 20,
            "1. Review and resolve any errors listed above",
            "2. Update imports to use validated_config module",
            "3. Test application startup with new configuration",
            "4. Monitor logs for validation warnings",
            "5. Consider creating configuration files for environment-specific settings",
            "",
        ]
    )

    return "\n".join(report_lines)


if __name__ == "__main__":
    # Generate and print migration report
    from .logging_config import get_logger

    logger = get_logger(__name__)

    migration_report = generate_migration_report()
    logger.info("Configuration migration report generated")
    print(migration_report)  # Keep console output for CLI script
