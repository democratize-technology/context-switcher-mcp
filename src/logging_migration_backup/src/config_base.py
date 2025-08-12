"""
Configuration base interfaces and utilities

This module provides the foundational interfaces and utilities for the
configuration system, breaking the circular dependency between config
and config_migration modules.
"""

import os
import logging
from typing import Any, Dict, List
from dataclasses import dataclass

from .types import ModelBackend, ConfigurationData
from .protocols import ConfigurationProvider, ConfigurationMigrator

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Base exception for configuration errors"""

    pass


class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails"""

    pass


class ConfigurationMigrationError(ConfigurationError):
    """Raised when configuration migration fails"""

    pass


@dataclass
class BackendConfiguration:
    """Configuration for a specific model backend"""

    backend_type: ModelBackend
    enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    model_specific_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.model_specific_config is None:
            self.model_specific_config = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "backend_type": self.backend_type.value,
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "model_specific_config": self.model_specific_config.copy(),
        }


@dataclass
class SecurityConfiguration:
    """Security-related configuration"""

    enable_client_binding: bool = True
    max_validation_failures: int = 3
    session_entropy_length: int = 32
    binding_signature_algorithm: str = "pbkdf2_hmac"
    signature_iterations: int = 600000
    enable_access_pattern_analysis: bool = True
    suspicious_activity_threshold: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "enable_client_binding": self.enable_client_binding,
            "max_validation_failures": self.max_validation_failures,
            "session_entropy_length": self.session_entropy_length,
            "binding_signature_algorithm": self.binding_signature_algorithm,
            "signature_iterations": self.signature_iterations,
            "enable_access_pattern_analysis": self.enable_access_pattern_analysis,
            "suspicious_activity_threshold": self.suspicious_activity_threshold,
        }


class BaseConfigurationProvider(ConfigurationProvider):
    """Base implementation of configuration provider with common functionality"""

    def __init__(self):
        self.session_config = ConfigurationData()
        self.backend_configs: Dict[ModelBackend, BackendConfiguration] = {}
        self.security_config = SecurityConfiguration()
        self._validated = False

    def get_session_config(self) -> ConfigurationData:
        """Get session configuration"""
        return self.session_config

    def get_backend_config(self, backend: ModelBackend) -> Dict[str, Any]:
        """Get backend-specific configuration"""
        if backend in self.backend_configs:
            return self.backend_configs[backend].to_dict()

        # Return default configuration for backend
        default_config = BackendConfiguration(backend_type=backend)
        return default_config.to_dict()

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.security_config.to_dict()

    def set_backend_config(
        self, backend: ModelBackend, config: BackendConfiguration
    ) -> None:
        """Set backend-specific configuration"""
        self.backend_configs[backend] = config
        self._validated = False  # Re-validation needed

    def validate(self) -> bool:
        """Validate configuration completeness and correctness"""
        try:
            self._validate_session_config()
            self._validate_backend_configs()
            self._validate_security_config()
            self._validated = True
            return True
        except ConfigurationValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            self._validated = False
            return False

    def _validate_session_config(self) -> None:
        """Validate session configuration"""
        if self.session_config.max_active_sessions <= 0:
            raise ConfigurationValidationError("max_active_sessions must be positive")

        if self.session_config.default_ttl_hours <= 0:
            raise ConfigurationValidationError("default_ttl_hours must be positive")

        if self.session_config.cleanup_interval_seconds <= 0:
            raise ConfigurationValidationError(
                "cleanup_interval_seconds must be positive"
            )

        if self.session_config.timeout_seconds <= 0:
            raise ConfigurationValidationError("timeout_seconds must be positive")

    def _validate_backend_configs(self) -> None:
        """Validate backend configurations"""
        for backend, config in self.backend_configs.items():
            if config.timeout_seconds <= 0:
                raise ConfigurationValidationError(
                    f"Backend {backend.value} timeout_seconds must be positive"
                )

            if config.max_retries < 0:
                raise ConfigurationValidationError(
                    f"Backend {backend.value} max_retries must be non-negative"
                )

            if config.retry_delay_seconds < 0:
                raise ConfigurationValidationError(
                    f"Backend {backend.value} retry_delay_seconds must be non-negative"
                )

    def _validate_security_config(self) -> None:
        """Validate security configuration"""
        if self.security_config.max_validation_failures <= 0:
            raise ConfigurationValidationError(
                "max_validation_failures must be positive"
            )

        if self.security_config.session_entropy_length < 16:
            raise ConfigurationValidationError(
                "session_entropy_length must be at least 16"
            )

        if self.security_config.signature_iterations < 100000:
            raise ConfigurationValidationError(
                "signature_iterations must be at least 100000 for security"
            )

    def is_validated(self) -> bool:
        """Check if configuration has been validated"""
        return self._validated


class ConfigurationFactory:
    """Factory for creating configuration providers"""

    @staticmethod
    def create_from_dict(config_dict: Dict[str, Any]) -> BaseConfigurationProvider:
        """Create configuration provider from dictionary"""
        provider = BaseConfigurationProvider()

        # Load session config
        if "session" in config_dict:
            session_data = config_dict["session"]
            provider.session_config = ConfigurationData(
                max_active_sessions=session_data.get("max_active_sessions", 50),
                default_ttl_hours=session_data.get("default_ttl_hours", 1),
                cleanup_interval_seconds=session_data.get(
                    "cleanup_interval_seconds", 300
                ),
                max_retries=session_data.get("max_retries", 3),
                retry_delay_seconds=session_data.get("retry_delay_seconds", 1.0),
                timeout_seconds=session_data.get("timeout_seconds", 30.0),
            )

        # Load backend configs
        if "backends" in config_dict:
            for backend_name, backend_config in config_dict["backends"].items():
                try:
                    backend_type = ModelBackend(backend_name)
                    config = BackendConfiguration(
                        backend_type=backend_type,
                        enabled=backend_config.get("enabled", True),
                        timeout_seconds=backend_config.get("timeout_seconds", 30.0),
                        max_retries=backend_config.get("max_retries", 3),
                        retry_delay_seconds=backend_config.get(
                            "retry_delay_seconds", 1.0
                        ),
                        model_specific_config=backend_config.get(
                            "model_specific_config", {}
                        ),
                    )
                    provider.set_backend_config(backend_type, config)
                except ValueError:
                    logger.warning(f"Unknown backend type: {backend_name}")

        # Load security config
        if "security" in config_dict:
            security_data = config_dict["security"]
            provider.security_config = SecurityConfiguration(
                enable_client_binding=security_data.get("enable_client_binding", True),
                max_validation_failures=security_data.get("max_validation_failures", 3),
                session_entropy_length=security_data.get("session_entropy_length", 32),
                binding_signature_algorithm=security_data.get(
                    "binding_signature_algorithm", "pbkdf2_hmac"
                ),
                signature_iterations=security_data.get("signature_iterations", 600000),
                enable_access_pattern_analysis=security_data.get(
                    "enable_access_pattern_analysis", True
                ),
                suspicious_activity_threshold=security_data.get(
                    "suspicious_activity_threshold", 5
                ),
            )

        return provider

    @staticmethod
    def create_from_environment() -> BaseConfigurationProvider:
        """Create configuration provider from environment variables"""
        config_dict = {
            "session": {
                "max_active_sessions": int(os.getenv("MAX_ACTIVE_SESSIONS", "50")),
                "default_ttl_hours": int(os.getenv("DEFAULT_TTL_HOURS", "1")),
                "cleanup_interval_seconds": int(
                    os.getenv("CLEANUP_INTERVAL_SECONDS", "300")
                ),
                "max_retries": int(os.getenv("MAX_RETRIES", "3")),
                "retry_delay_seconds": float(os.getenv("RETRY_DELAY_SECONDS", "1.0")),
                "timeout_seconds": float(os.getenv("TIMEOUT_SECONDS", "30.0")),
            },
            "security": {
                "enable_client_binding": os.getenv(
                    "ENABLE_CLIENT_BINDING", "true"
                ).lower()
                == "true",
                "max_validation_failures": int(
                    os.getenv("MAX_VALIDATION_FAILURES", "3")
                ),
                "session_entropy_length": int(
                    os.getenv("SESSION_ENTROPY_LENGTH", "32")
                ),
                "signature_iterations": int(
                    os.getenv("SIGNATURE_ITERATIONS", "600000")
                ),
            },
        }

        return ConfigurationFactory.create_from_dict(config_dict)

    @staticmethod
    def create_default() -> BaseConfigurationProvider:
        """Create default configuration provider"""
        return BaseConfigurationProvider()


class BaseMigrator(ConfigurationMigrator):
    """Base implementation of configuration migrator"""

    def __init__(self):
        self.migration_rules: List[Dict[str, Any]] = []

    def add_migration_rule(
        self, from_version: str, to_version: str, migration_func: callable
    ) -> None:
        """Add a migration rule"""
        self.migration_rules.append(
            {
                "from_version": from_version,
                "to_version": to_version,
                "migrate": migration_func,
            }
        )

    def migrate_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old configuration to new format"""
        current_config = old_config.copy()
        current_version = self._get_config_version(current_config)

        # Apply migration rules in sequence
        for rule in self.migration_rules:
            if rule["from_version"] == current_version:
                logger.info(
                    f"Migrating config from {rule['from_version']} to {rule['to_version']}"
                )
                current_config = rule["migrate"](current_config)
                current_version = rule["to_version"]

        return current_config

    def is_migration_needed(self, config: Dict[str, Any]) -> bool:
        """Check if migration is needed"""
        current_version = self._get_config_version(config)

        # Check if we have any migration rules that apply
        for rule in self.migration_rules:
            if rule["from_version"] == current_version:
                return True

        return False

    def validate_migration(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> bool:
        """Validate that migration was successful"""
        try:
            # Create providers from both configs
            old_provider = ConfigurationFactory.create_from_dict(old_config)
            new_provider = ConfigurationFactory.create_from_dict(new_config)

            # Validate both
            return old_provider.validate() and new_provider.validate()
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False

    def _get_config_version(self, config: Dict[str, Any]) -> str:
        """Get configuration version"""
        return config.get("version", "unknown")


# Utility functions for working with configurations
def merge_configurations(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configuration dictionaries"""
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configurations(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_configuration_dict(config_dict: Dict[str, Any]) -> List[str]:
    """Validate configuration dictionary and return list of errors"""
    errors = []

    try:
        provider = ConfigurationFactory.create_from_dict(config_dict)
        if not provider.validate():
            errors.append("Configuration validation failed")
    except Exception as e:
        errors.append(f"Configuration creation failed: {str(e)}")

    return errors


def get_default_backend_config(backend: ModelBackend) -> Dict[str, Any]:
    """Get default configuration for a backend"""
    config = BackendConfiguration(backend_type=backend)
    return config.to_dict()
