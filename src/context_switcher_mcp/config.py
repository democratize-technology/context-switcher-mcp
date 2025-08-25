"""Configuration module transition interface

This module provides the transition interface between the legacy configuration
system and the new unified configuration architecture. It maintains full
backward compatibility while routing all requests to the new system.

MIGRATION NOTICE:
This is a transition module that provides backward compatibility. The actual
configuration implementation has moved to the config/ package with a clean,
unified architecture.

For new code, prefer:
    from context_switcher_mcp.config import get_config

For legacy compatibility, this module continues to support:
    from context_switcher_mcp.config import config, get_config, etc.
"""

import warnings
from typing import Optional, Any, Dict

# Import the new unified configuration system
try:
    from .config import (
        get_config as _new_get_config,
        reload_config as _new_reload_config,
    )
    from .config.core import ConfigurationError
    from .config.migration import (
        LegacyConfigAdapter,
        create_legacy_config_with_migration,
        get_legacy_config as _get_legacy_config,
    )
    _UNIFIED_CONFIG_AVAILABLE = True
except ImportError as e:
    _UNIFIED_CONFIG_AVAILABLE = False
    warnings.warn(
        f"New unified configuration system not available: {e}. "
        "Falling back to legacy configuration system.",
        UserWarning
    )
    
    # Fallback imports for legacy system
    try:
        from .config_old import (
            get_config as _old_get_config,
            reload_config as _old_reload_config,
            ContextSwitcherConfig,
            ConfigurationError,
            config as _old_config,
        )
        _LEGACY_CONFIG_AVAILABLE = True
    except ImportError:
        _LEGACY_CONFIG_AVAILABLE = False
        raise ImportError(
            "Neither unified nor legacy configuration system is available. "
            "This indicates a serious configuration system issue."
        )


def get_config(**kwargs) -> Any:
    """Get configuration instance with automatic system selection
    
    This function provides backward compatibility by automatically selecting
    the best available configuration system and returning an appropriate
    configuration instance.
    
    Returns:
        Configuration instance (unified or legacy compatible)
        
    Raises:
        ConfigurationError: If configuration loading fails
    """
    if _UNIFIED_CONFIG_AVAILABLE:
        # Use new unified system
        unified_config = _new_get_config(**kwargs)
        
        # For maximum compatibility, wrap in legacy adapter
        # This ensures all legacy attribute access patterns work
        return LegacyConfigAdapter(unified_config)
    
    elif _LEGACY_CONFIG_AVAILABLE:
        # Fall back to old system
        warnings.warn(
            "Using legacy configuration system. Please update to unified config.",
            DeprecationWarning,
            stacklevel=2
        )
        return _old_get_config()
    
    else:
        raise ConfigurationError("No configuration system available")


def reload_config() -> Any:
    """Reload configuration with automatic system selection
    
    Returns:
        Reloaded configuration instance
        
    Raises:
        ConfigurationError: If configuration reload fails
    """
    if _UNIFIED_CONFIG_AVAILABLE:
        unified_config = _new_reload_config()
        return LegacyConfigAdapter(unified_config)
    
    elif _LEGACY_CONFIG_AVAILABLE:
        warnings.warn(
            "Using legacy configuration reload. Please update to unified config.",
            DeprecationWarning,
            stacklevel=2
        )
        return _old_reload_config()
    
    else:
        raise ConfigurationError("No configuration system available")


# Create global configuration instance for legacy compatibility
_global_config_instance: Optional[Any] = None


def _get_global_config():
    """Get or create global configuration instance"""
    global _global_config_instance
    
    if _global_config_instance is None:
        _global_config_instance = get_config()
    
    return _global_config_instance


# Legacy compatibility: provide 'config' attribute at module level
def __getattr__(name: str) -> Any:
    """Provide legacy module-level attribute access"""
    
    if name == "config":
        warnings.warn(
            "Accessing 'config' directly from the module is deprecated. "
            "Use 'get_config()' instead for better error handling.",
            DeprecationWarning,
            stacklevel=2
        )
        return _get_global_config()
    
    # Check if it's a configuration class from the old system
    if _UNIFIED_CONFIG_AVAILABLE:
        # Try to get it from the legacy migration layer
        try:
            return getattr(LegacyConfigAdapter, name)
        except AttributeError:
            pass
    
    if _LEGACY_CONFIG_AVAILABLE:
        # Try to get it from the old config module
        try:
            from . import config_old
            return getattr(config_old, name)
        except AttributeError:
            pass
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Additional legacy compatibility functions
def validate_current_config() -> tuple[bool, list[str]]:
    """Legacy configuration validation function (DEPRECATED)"""
    warnings.warn(
        "validate_current_config() is deprecated. Use config validation methods instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        config_instance = get_config()
        if hasattr(config_instance, 'validate_current_config'):
            return config_instance.validate_current_config()
        else:
            # Basic validation
            return True, []
    except Exception as e:
        return False, [str(e)]


# Export the key interfaces for backward compatibility
__all__ = [
    "get_config",
    "reload_config", 
    "validate_current_config",
    "ConfigurationError",
]

# Note: 'config' is available via __getattr__ for legacy compatibility