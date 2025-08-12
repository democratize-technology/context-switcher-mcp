"""Migration and backward compatibility layer

This module provides backward compatibility with the legacy configuration
system while migrating to the new unified configuration architecture.

Key features:
- Legacy configuration interface compatibility
- Gradual migration with deprecation warnings  
- Automatic config format detection and conversion
- Zero-downtime migration support
- Legacy import path compatibility

The migration strategy:
1. Detect legacy vs unified configuration usage
2. Provide compatibility adapters for legacy interfaces
3. Issue deprecation warnings for legacy usage
4. Gradually phase out legacy support
"""

from ..logging_config import get_logger
import warnings
from typing import Any, Optional, Dict, Union
from pathlib import Path

from .core import ContextSwitcherConfig, ConfigurationError

logger = get_logger(__name__)


class LegacyConfigAdapter:
    """Adapter to provide legacy configuration interface compatibility
    
    This adapter wraps the new unified configuration and exposes the same
    interface as the legacy configuration system, enabling gradual migration
    without breaking existing code.
    """
    
    def __init__(self, unified_config: ContextSwitcherConfig):
        """Initialize adapter with unified configuration
        
        Args:
            unified_config: The new unified configuration instance
        """
        self._unified_config = unified_config
        self._issue_deprecation_warning()
        
        # Create legacy-compatible attribute structures
        self._create_legacy_attributes()
    
    def _issue_deprecation_warning(self):
        """Issue deprecation warning for legacy config usage"""
        warnings.warn(
            "Legacy configuration interface is deprecated. "
            "Please update to use the new unified configuration system: "
            "from context_switcher_mcp.config import get_config",
            DeprecationWarning,
            stacklevel=3
        )
    
    def _create_legacy_attributes(self):
        """Create legacy-compatible attribute structures"""
        # Model configuration compatibility
        self.model = LegacyModelConfigAdapter(self._unified_config.models)
        
        # Circuit breaker compatibility (now part of models)
        self.circuit_breaker = LegacyCircuitBreakerAdapter(self._unified_config.models)
        
        # Validation compatibility (now part of session)
        self.validation = LegacyValidationAdapter(self._unified_config.session)
        
        # Session configuration compatibility
        self.session = LegacySessionAdapter(self._unified_config.session)
        
        # Metrics compatibility (now part of monitoring)
        self.metrics = LegacyMetricsAdapter(self._unified_config.monitoring.metrics)
        
        # Retry compatibility (now part of models)
        self.retry = LegacyRetryAdapter(self._unified_config.models)
        
        # Reasoning compatibility - create synthetic config since this was removed
        self.reasoning = LegacyReasoningAdapter()
        
        # Profiling compatibility
        self.profiling = LegacyProfilingAdapter(self._unified_config.monitoring.profiling)
        
        # Server configuration compatibility
        self.server = LegacyServerAdapter(self._unified_config.server)
    
    def __getattr__(self, name: str) -> Any:
        """Provide access to unified config attributes for any missing legacy attrs"""
        if hasattr(self._unified_config, name):
            return getattr(self._unified_config, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def mask_sensitive_data(self) -> Dict[str, Any]:
        """Legacy method compatibility"""
        return self._unified_config.get_masked_dict()
    
    def validate_current_config(self) -> tuple[bool, list[str]]:
        """Legacy validation method compatibility"""
        try:
            # Basic validation - if we got here, config is valid
            dependencies = self._unified_config.validate_external_dependencies()
            warnings = [msg for msg in dependencies if msg.startswith("⚠")]
            return len(warnings) == 0, warnings
        except Exception as e:
            return False, [str(e)]


class LegacyModelConfigAdapter:
    """Legacy model configuration adapter"""
    
    def __init__(self, models_config):
        self._config = models_config
    
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
    
    def to_backend_config(self, backend) -> Dict[str, Any]:
        """Legacy method compatibility"""
        return self._config.get_backend_config(backend.value if hasattr(backend, 'value') else backend)


class LegacyCircuitBreakerAdapter:
    """Legacy circuit breaker configuration adapter"""
    
    def __init__(self, models_config):
        self._config = models_config
    
    @property
    def failure_threshold(self) -> int:
        return self._config.circuit_breaker_failure_threshold
    
    @property
    def timeout_seconds(self) -> int:
        return self._config.circuit_breaker_timeout_seconds


class LegacyValidationAdapter:
    """Legacy validation configuration adapter"""
    
    def __init__(self, session_config):
        self._config = session_config
    
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
    """Legacy session configuration adapter"""
    
    def __init__(self, session_config):
        self._config = session_config
    
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
    """Legacy metrics configuration adapter"""
    
    def __init__(self, metrics_config):
        self._config = metrics_config
    
    @property
    def max_history_size(self) -> int:
        return self._config.max_history_size
    
    @property
    def retention_days(self) -> int:
        return self._config.retention_days


class LegacyRetryAdapter:
    """Legacy retry configuration adapter"""
    
    def __init__(self, models_config):
        self._config = models_config
    
    @property
    def max_retries(self) -> int:
        return self._config.max_retries
    
    @property
    def initial_delay(self) -> float:
        return self._config.retry_delay_seconds
    
    @property
    def backoff_factor(self) -> float:
        return self._config.retry_backoff_factor
    
    @property
    def max_delay(self) -> float:
        # Calculate max delay based on formula: initial_delay * (backoff_factor ^ max_retries)
        return self._config.retry_delay_seconds * (self._config.retry_backoff_factor ** self._config.max_retries)


class LegacyReasoningAdapter:
    """Legacy reasoning configuration adapter (synthetic)"""
    
    # These values were removed from the new config, so provide legacy defaults
    @property
    def max_iterations(self) -> int:
        return 20
    
    @property
    def cot_timeout_seconds(self) -> float:
        return 30.0
    
    @property
    def summary_timeout_seconds(self) -> float:
        return 5.0
    
    @property
    def default_temperature(self) -> float:
        return 0.7


class LegacyProfilingAdapter:
    """Legacy profiling configuration adapter"""
    
    def __init__(self, profiling_config):
        self._config = profiling_config
    
    @property
    def enabled(self) -> bool:
        return self._config.enabled
    
    @property
    def level(self) -> str:
        return self._config.level.value if hasattr(self._config.level, 'value') else self._config.level
    
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
    """Legacy server configuration adapter"""
    
    def __init__(self, server_config):
        self._config = server_config
    
    @property
    def host(self) -> str:
        return self._config.host
    
    @property
    def port(self) -> int:
        return self._config.port
    
    @property
    def log_level(self) -> str:
        return self._config.log_level.value if hasattr(self._config.log_level, 'value') else self._config.log_level


# Legacy configuration factory functions
def create_legacy_config_with_migration(
    config_file: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None,
    **overrides
) -> LegacyConfigAdapter:
    """Create legacy configuration with automatic migration from unified config
    
    Args:
        config_file: Optional configuration file path
        environment: Optional environment name
        **overrides: Configuration overrides
        
    Returns:
        Legacy-compatible configuration adapter
    """
    # Import here to avoid circular import
    from . import get_config
    
    try:
        # Create unified config
        unified_config = get_config(
            environment=environment,
            config_file=config_file,
            reload=True
        )
        
        # Apply any overrides by recreating with merged data
        if overrides:
            merged_config = ContextSwitcherConfig(
                config_file=config_file,
                **overrides
            )
            unified_config = merged_config
        
        # Wrap in legacy adapter
        legacy_adapter = LegacyConfigAdapter(unified_config)
        
        logger.info("Created legacy configuration adapter with unified config backend")
        return legacy_adapter
        
    except Exception as e:
        raise ConfigurationError(f"Failed to create legacy configuration: {e}") from e


def detect_legacy_config_usage() -> Dict[str, Any]:
    """Detect and report legacy configuration usage patterns
    
    This function analyzes the current call stack and environment to detect
    whether legacy configuration patterns are being used.
    
    Returns:
        Dictionary with legacy usage analysis
    """
    import inspect
    import os
    
    analysis = {
        "legacy_imports_detected": False,
        "legacy_attributes_accessed": [],
        "migration_recommendations": [],
        "call_stack_analysis": [],
    }
    
    # Analyze call stack for legacy patterns
    stack = inspect.stack()
    for frame_info in stack[1:6]:  # Check recent frames
        filename = frame_info.filename
        lineno = frame_info.lineno
        
        # Skip this file
        if "migration.py" in filename:
            continue
        
        analysis["call_stack_analysis"].append({
            "file": filename,
            "line": lineno,
            "function": frame_info.function,
        })
        
        # Check for legacy import patterns (would require code analysis)
        if "config.py" in filename and "context_switcher_mcp" in filename:
            analysis["legacy_imports_detected"] = True
    
    # Generate migration recommendations
    if analysis["legacy_imports_detected"]:
        analysis["migration_recommendations"].extend([
            "Update imports to use unified config: from context_switcher_mcp.config import get_config",
            "Replace config.attribute with get_config().domain.attribute pattern",
            "Consider using environment-specific configurations",
        ])
    
    return analysis


# Global legacy compatibility functions
def get_legacy_config() -> LegacyConfigAdapter:
    """Get legacy-compatible configuration instance (DEPRECATED)
    
    This function provides the old configuration interface for backward
    compatibility. New code should use get_config() from the main config module.
    
    Returns:
        Legacy configuration adapter
    """
    warnings.warn(
        "get_legacy_config() is deprecated. Use get_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return create_legacy_config_with_migration()


def reload_legacy_config() -> LegacyConfigAdapter:
    """Reload legacy configuration (DEPRECATED)
    
    Returns:
        Newly loaded legacy configuration adapter
    """
    warnings.warn(
        "reload_legacy_config() is deprecated. Use reload_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import here to avoid circular import
    from . import reload_config
    
    unified_config = reload_config()
    return LegacyConfigAdapter(unified_config)


# Legacy validation functions
def validate_current_config() -> tuple[bool, list[str]]:
    """Legacy configuration validation function (DEPRECATED)"""
    warnings.warn(
        "validate_current_config() is deprecated. Use config validation methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        legacy_config = get_legacy_config()
        return legacy_config.validate_current_config()
    except Exception as e:
        return False, [str(e)]