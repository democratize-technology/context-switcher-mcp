"""Environment-specific configuration support

This package provides environment-specific configuration presets and utilities
for different deployment scenarios: development, staging, and production.

Each environment has its own configuration defaults optimized for that use case:

- Development: Debug logging, detailed profiling, permissive security
- Staging: Production-like settings with enhanced monitoring
- Production: Secure defaults, optimized performance, minimal logging

Environment detection is automatic but can be overridden via environment
variables or explicit configuration.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

from ..core import ConfigurationError, ContextSwitcherConfig, create_config
from .base import BaseEnvironmentConfig
from .development import DevelopmentConfig
from .production import ProductionConfig
from .staging import StagingConfig

logger = logging.getLogger(__name__)

# Environment configuration registry
ENVIRONMENT_CONFIGS = {
    "development": DevelopmentConfig,
    "dev": DevelopmentConfig,
    "staging": StagingConfig,
    "stage": StagingConfig,
    "production": ProductionConfig,
    "prod": ProductionConfig,
}


def detect_environment() -> str:
    """Detect the current deployment environment

    Uses environment variables and system indicators to determine
    the current deployment environment.

    Returns:
        Environment name: "development", "staging", or "production"
    """
    # Check explicit environment variables
    env = os.getenv("ENVIRONMENT", os.getenv("ENV", "")).lower().strip()
    if env in ENVIRONMENT_CONFIGS:
        return (
            "development"
            if env in ["dev", "development"]
            else "staging"
            if env in ["stage", "staging"]
            else "production"
        )

    # Check for common CI/CD environment indicators
    if os.getenv("CI") or os.getenv("CONTINUOUS_INTEGRATION"):
        return "staging"

    # Check for container/orchestration indicators
    if os.getenv("KUBERNETES_SERVICE_HOST") or os.getenv("DOCKER_CONTAINER"):
        return "production"

    # Check Python environment indicators
    if os.getenv("PYTHON_ENV") == "production":
        return "production"
    elif os.getenv("PYTHON_ENV") in ["development", "dev"]:
        return "development"

    # Check for development indicators
    if (
        os.getenv("DEBUG", "").lower() in ["true", "1", "yes"]
        or os.getenv("LOG_LEVEL", "").upper() == "DEBUG"
        or os.path.exists(".env")  # Development environment file
        or os.path.exists("pyproject.toml")  # Development project
    ):
        return "development"

    # Default to development for safety
    return "development"


def get_environment_config(
    environment: str, config_file: str | Path | None = None, **overrides
) -> ContextSwitcherConfig:
    """Get configuration for a specific environment

    Args:
        environment: Environment name (development, staging, production)
        config_file: Optional configuration file path
        **overrides: Additional configuration overrides

    Returns:
        Configuration instance with environment-specific defaults

    Raises:
        ConfigurationError: If environment is invalid or configuration fails
    """
    environment = environment.lower().strip()

    if environment not in ENVIRONMENT_CONFIGS:
        raise ConfigurationError(
            f"Unknown environment: {environment}. "
            f"Valid environments: {list(ENVIRONMENT_CONFIGS.keys())}"
        )

    # Get environment-specific configuration class
    env_config_class = ENVIRONMENT_CONFIGS[environment]

    try:
        # Create environment configuration
        env_config = env_config_class()

        # Convert to dictionary for merging
        env_dict = env_config.get_config_dict()

        # Merge with any provided overrides
        merged_config = {**env_dict, **overrides}

        # Create main configuration with environment defaults
        config = ContextSwitcherConfig(config_file=config_file, **merged_config)

        logger.info(f"Loaded {environment} environment configuration")
        return config

    except Exception as e:
        raise ConfigurationError(
            f"Failed to load {environment} environment configuration: {e}"
        ) from e


def get_auto_environment_config(
    config_file: str | Path | None = None, **overrides
) -> ContextSwitcherConfig:
    """Get configuration with automatic environment detection

    Args:
        config_file: Optional configuration file path
        **overrides: Additional configuration overrides

    Returns:
        Configuration instance with auto-detected environment defaults
    """
    environment = detect_environment()
    logger.info(f"Auto-detected environment: {environment}")
    return get_environment_config(environment, config_file, **overrides)


def list_available_environments() -> list[str]:
    """List all available environment configurations

    Returns:
        List of available environment names
    """
    return list(ENVIRONMENT_CONFIGS.keys())


def get_environment_info(environment: str) -> dict[str, Any]:
    """Get information about a specific environment configuration

    Args:
        environment: Environment name

    Returns:
        Dictionary with environment information

    Raises:
        ConfigurationError: If environment is invalid
    """
    environment = environment.lower().strip()

    if environment not in ENVIRONMENT_CONFIGS:
        raise ConfigurationError(f"Unknown environment: {environment}")

    env_config_class = ENVIRONMENT_CONFIGS[environment]

    return {
        "name": environment,
        "class": env_config_class.__name__,
        "description": env_config_class.__doc__ or "No description available",
        "is_production_ready": getattr(env_config_class, "is_production_ready", False),
    }


# Convenience functions for specific environments
def get_development_config(
    config_file: str | Path | None = None, **overrides
) -> ContextSwitcherConfig:
    """Get development environment configuration"""
    return get_environment_config("development", config_file, **overrides)


def get_staging_config(
    config_file: str | Path | None = None, **overrides
) -> ContextSwitcherConfig:
    """Get staging environment configuration"""
    return get_environment_config("staging", config_file, **overrides)


def get_production_config(
    config_file: str | Path | None = None, **overrides
) -> ContextSwitcherConfig:
    """Get production environment configuration"""
    return get_environment_config("production", config_file, **overrides)


__all__ = [
    "detect_environment",
    "get_environment_config",
    "get_auto_environment_config",
    "list_available_environments",
    "get_environment_info",
    "get_development_config",
    "get_staging_config",
    "get_production_config",
]
