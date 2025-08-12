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
from typing import Optional, Union
from pathlib import Path

from .core import ContextSwitcherConfig, ConfigurationError
from .migration import LegacyConfigAdapter
from .environments import get_environment_config

logger = logging.getLogger(__name__)

# Global configuration instance
_global_config: Optional[ContextSwitcherConfig] = None


def get_config(
    environment: Optional[str] = None,
    config_file: Optional[Union[str, Path]] = None,
    reload: bool = False,
) -> ContextSwitcherConfig:
    """Get the unified configuration instance
    
    This is the main entry point for accessing configuration throughout
    the application. It provides a consistent, validated configuration
    object with all settings.
    
    Args:
        environment: Environment name (dev, staging, prod). Auto-detected if None.
        config_file: Optional path to configuration file
        reload: Force reload of configuration from environment/files
        
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
    global _global_config
    
    if _global_config is None or reload:
        try:
            if environment:
                # Load environment-specific configuration
                _global_config = get_environment_config(environment, config_file)
                logger.info(f"Loaded {environment} environment configuration")
            else:
                # Load default configuration
                _global_config = ContextSwitcherConfig(config_file=config_file)
                logger.info("Loaded default configuration")
                
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize configuration: {e}") from e
    
    return _global_config


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
    return get_config(reload=True)


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
        stacklevel=2
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
            stacklevel=2
        )
        return get_legacy_config()
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Public API exports
__all__ = [
    "get_config",
    "reload_config", 
    "ConfigurationError",
    "ContextSwitcherConfig",
]