"""Base environment configuration

This module provides the base class for all environment-specific configurations.
It defines the common interface and utilities that all environment configs share.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class BaseEnvironmentConfig(BaseModel, ABC):
    """Base class for environment-specific configuration

    All environment configurations should inherit from this class and
    implement the required methods to provide environment-specific defaults.
    """

    @abstractmethod
    def get_config_dict(self) -> dict[str, Any]:
        """Get configuration dictionary for this environment

        Returns:
            Dictionary with environment-specific configuration values
        """
        pass

    @property
    @abstractmethod
    def is_production_ready(self) -> bool:
        """Check if this environment configuration is production-ready

        Returns:
            True if configuration is suitable for production deployment
        """
        pass

    @property
    @abstractmethod
    def environment_name(self) -> str:
        """Get the environment name

        Returns:
            Environment name string
        """
        pass

    def get_server_config(self) -> dict[str, Any]:
        """Get server configuration for this environment

        Override this method to provide environment-specific server settings.

        Returns:
            Dictionary with server configuration
        """
        return {}

    def get_models_config(self) -> dict[str, Any]:
        """Get models configuration for this environment

        Override this method to provide environment-specific model settings.

        Returns:
            Dictionary with models configuration
        """
        return {}

    def get_session_config(self) -> dict[str, Any]:
        """Get session configuration for this environment

        Override this method to provide environment-specific session settings.

        Returns:
            Dictionary with session configuration
        """
        return {}

    def get_security_config(self) -> dict[str, Any]:
        """Get security configuration for this environment

        Override this method to provide environment-specific security settings.

        Returns:
            Dictionary with security configuration
        """
        return {}

    def get_monitoring_config(self) -> dict[str, Any]:
        """Get monitoring configuration for this environment

        Override this method to provide environment-specific monitoring settings.

        Returns:
            Dictionary with monitoring configuration
        """
        return {}

    def validate_environment_requirements(self) -> list[str]:
        """Validate that environment requirements are met

        Override this method to add environment-specific validation.

        Returns:
            List of validation warnings or errors
        """
        return []
