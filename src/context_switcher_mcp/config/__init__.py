"""Unified Configuration System for Context Switcher MCP

This module provides the unified configuration interface that replaces the
fragmented legacy configuration system. It serves as the single source of
truth for all configuration needs.

Key Features:
- Pydantic-based validation with clear error messages
- Environment-specific configuration support
- Domain-specific configuration modules
- Backward compatibility with legacy imports
- Type-safe configuration access
- Production-ready defaults and validation

Usage:
    from context_switcher_mcp.config import get_config

    config = get_config()
    max_tokens = config.models.default_max_tokens
    server_port = config.server.port

For environment-specific configs:
    from context_switcher_mcp.config import get_config

    config = get_config(environment="production")

Migration from legacy config:
    This system maintains backward compatibility. Legacy imports will work
    but will issue deprecation warnings. Update imports as follows:

    OLD: from context_switcher_mcp.config import config
    NEW: from context_switcher_mcp.config import get_config; config = get_config()
"""

import logging
import warnings
from pathlib import Path

from .core import ConfigurationError, ContextSwitcherConfig
from .environments import get_environment_config
from .migration import LegacyConfigAdapter

logger = logging.getLogger(__name__)

# Configuration system availability flags (for compatibility testing)
_UNIFIED_CONFIG_AVAILABLE = True
_LEGACY_CONFIG_AVAILABLE = False

# Import-time validation to ensure at least one config system is available
if not _UNIFIED_CONFIG_AVAILABLE and not _LEGACY_CONFIG_AVAILABLE:
    raise ImportError("Neither unified nor legacy configuration system is available")

# Legacy config import (for testing compatibility)
try:
    import context_switcher_mcp.config_old as config_old
except ImportError:
    config_old = None

# Global configuration instance
_global_config_instance: ContextSwitcherConfig | None = None


# Internal functions for transition layer compatibility (used by tests)
def _new_get_config(**kwargs) -> ContextSwitcherConfig:
    """Internal function that implements the new unified config system"""
    return ContextSwitcherConfig(**kwargs)


def _old_get_config():
    """Internal function that would implement legacy config system (deprecated)"""
    # For compatibility testing - return a LegacyConfigAdapter wrapping a basic config
    if _LEGACY_CONFIG_AVAILABLE:
        # Create minimal legacy-style config for testing
        basic_config = ContextSwitcherConfig()
        return LegacyConfigAdapter(basic_config)
    else:
        # This would never be called in the current system since _LEGACY_CONFIG_AVAILABLE is False
        raise ConfigurationError("Legacy configuration system not available")


def _new_reload_config() -> ContextSwitcherConfig:
    """Internal function for unified config reload"""
    global _global_config_instance
    _global_config_instance = None
    return get_config()


def _old_reload_config():
    """Internal function for legacy config reload (deprecated)"""
    # For compatibility testing - return a fresh legacy config
    if _LEGACY_CONFIG_AVAILABLE:
        # Create fresh legacy-style config for testing
        basic_config = ContextSwitcherConfig()
        return LegacyConfigAdapter(basic_config)
    else:
        # This would never be called in the current system since _LEGACY_CONFIG_AVAILABLE is False
        raise ConfigurationError("Legacy configuration reload not available")


def get_config(
    environment: str | None = None,
    config_file: str | Path | None = None,
    reload: bool = False,
    **kwargs,
) -> ContextSwitcherConfig:
    """Get the unified configuration instance

    This is the main entry point for accessing configuration throughout
    the application. It provides a consistent, validated configuration
    object with all settings.

    Args:
        environment: Environment name (dev, staging, prod). Auto-detected if None.
        config_file: Optional path to configuration file
        reload: Force reload of configuration from environment/files
        **kwargs: Additional configuration parameters (for compatibility)

    Returns:
        Validated configuration instance

    Raises:
        ConfigurationError: If configuration validation fails

    Examples:
        # Get default configuration
        config = get_config()

        # Get production configuration
        config = get_config(environment="production")

        # Load from specific file
        config = get_config(config_file="my_config.yaml")

        # Force reload (useful for testing)
        config = get_config(reload=True)
    """
    global _global_config_instance

    # Check availability flags for testing/compatibility scenarios
    if not _UNIFIED_CONFIG_AVAILABLE:
        if _LEGACY_CONFIG_AVAILABLE:
            warnings.warn(
                "Falling back to legacy configuration system. "
                "Please upgrade to the unified configuration system.",
                DeprecationWarning,
                stacklevel=2,
            )
            return _old_get_config()
        else:
            raise ConfigurationError("No configuration system available")

    if _global_config_instance is None or reload:
        try:
            if environment:
                # Load environment-specific configuration
                _global_config_instance = get_environment_config(
                    environment, config_file
                )
                logger.info(f"Loaded {environment} environment configuration")
            else:
                # Load default configuration using the new unified system
                unified_config = _new_get_config(**kwargs)
                if config_file:
                    unified_config = ContextSwitcherConfig(config_file=config_file)

                _global_config_instance = LegacyConfigAdapter(unified_config)
                logger.info("Loaded default configuration")

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {e}") from e

    return _global_config_instance


def reload_config() -> ContextSwitcherConfig:
    """Reload configuration from environment variables and files

    This function forces a complete reload of the configuration system,
    useful for picking up runtime changes to environment variables or
    configuration files.

    Returns:
        Newly loaded configuration instance

    Raises:
        ConfigurationError: If configuration reload fails
    """
    # Check availability flags for testing/compatibility scenarios
    if not _UNIFIED_CONFIG_AVAILABLE:
        if _LEGACY_CONFIG_AVAILABLE:
            warnings.warn(
                "Using legacy configuration reload. "
                "Please upgrade to the unified configuration system.",
                DeprecationWarning,
                stacklevel=2,
            )
            return _old_reload_config()
        else:
            raise ConfigurationError("No configuration system available")

    # Use the new unified reload system
    unified_config = _new_reload_config()
    return LegacyConfigAdapter(unified_config)


def validate_current_config() -> tuple[bool, list[str]]:
    """Validate the current configuration (DEPRECATED)

    This function maintains backward compatibility with the old config system.
    New code should use the config validation methods directly.

    Returns:
        tuple[bool, list[str]]: (is_valid, validation_messages)
    """
    warnings.warn(
        "validate_current_config() is deprecated. Use config validation methods instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        config_instance = get_config()
        if hasattr(config_instance, "validate_current_config"):
            return config_instance.validate_current_config()
        return (True, [])
    except Exception as e:
        return (False, [str(e)])


def _get_global_config() -> ContextSwitcherConfig:
    """Get or create the global configuration instance

    This function maintains the global configuration instance,
    creating it if it doesn't exist, or returning the existing one.

    Returns:
        The global configuration instance
    """
    global _global_config_instance
    if _global_config_instance is None:
        _global_config_instance = get_config()
    return _global_config_instance


# Legacy compatibility functions (with deprecation warnings)
def get_legacy_config():
    """Legacy configuration accessor (DEPRECATED)

    This function maintains backward compatibility with the old config system.
    New code should use get_config() instead.

    Returns:
        Legacy-compatible configuration adapter
    """
    warnings.warn(
        "get_legacy_config() is deprecated. Use get_config() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    unified_config = get_config()
    return LegacyConfigAdapter(unified_config)


# Provide legacy compatibility at module level
def __getattr__(name: str):
    """Module-level attribute access for legacy compatibility"""
    if name == "config":
        warnings.warn(
            "Importing 'config' directly is deprecated. Use 'get_config()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _get_global_config()

    # Handle legacy system attribute access
    if not _UNIFIED_CONFIG_AVAILABLE and _LEGACY_CONFIG_AVAILABLE:
        if config_old:
            try:
                return getattr(config_old, name)
            except AttributeError:
                pass

    # Handle unified system attribute access (for testing)
    if _UNIFIED_CONFIG_AVAILABLE:
        try:
            # Try to get attribute from LegacyConfigAdapter class (for testing purposes)
            return getattr(LegacyConfigAdapter, name)
        except AttributeError:
            pass

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Public API exports
__all__ = [
    "get_config",
    "reload_config",
    "validate_current_config",
    "ConfigurationError",
    "ContextSwitcherConfig",
]
