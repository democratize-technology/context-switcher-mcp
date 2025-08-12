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

from .logging_base import get_logger
import warnings
from typing import Any, Dict, Optional

# Import from clean architecture modules
from .config_base import (
    BaseMigrator,
    ConfigurationError,
    ConfigurationMigrationError,
    BaseConfigurationProvider,
    ConfigurationFactory,
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
        "Migration features may be limited.",
        UserWarning,
    )

logger = get_logger(__name__)


class CompatibilityAdapter(BaseMigrator):
    """Adapter to provide configuration migration and backward compatibility

    This adapter uses dependency injection to avoid circular dependencies
    while providing migration capabilities between old and new configuration
    formats.
    """

    def __init__(self):
        """Initialize adapter using dependency injection"""
        super().__init__()

        # Setup migration rules
        self.add_migration_rule(
            from_version="legacy_dataclass",
            to_version="validated_pydantic",
            migration_func=self._migrate_legacy_to_validated,
        )

        self.add_migration_rule(
            from_version="unknown",
            to_version="validated_pydantic",
            migration_func=self._migrate_unknown_to_validated,
        )

    def _migrate_legacy_to_validated(
        self, old_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Migrate legacy dataclass configuration to validated format"""
        logger.info("Migrating legacy dataclass configuration to validated format")

        # Extract session configuration
        session_config = old_config.get("session", {})
        migrated_session = {
            "max_active_sessions": session_config.get("max_active_sessions", 50),
            "default_ttl_hours": session_config.get("default_ttl_hours", 1),
            "cleanup_interval_seconds": session_config.get(
                "cleanup_interval_seconds", 300
            ),
        }

        # Extract model configuration
        model_config = old_config.get("model", {})
        migrated_model = {
            "default_max_tokens": model_config.get("default_max_tokens", 2048),
            "default_temperature": model_config.get("default_temperature", 0.7),
            "backends": {},
        }

        # Add backend-specific configurations
        if "bedrock_model_id" in model_config:
            migrated_model["backends"]["bedrock"] = {
                "model_id": model_config["bedrock_model_id"],
                "enabled": True,
            }

        # Extract security configuration
        security_config = old_config.get("security", {})
        migrated_security = {
            "enable_client_binding": security_config.get("enable_client_binding", True),
            "max_validation_failures": security_config.get(
                "max_validation_failures", 3
            ),
        }

        migrated_config = {
            "version": "validated_pydantic",
            "session": migrated_session,
            "model": migrated_model,
            "security": migrated_security,
        }

        logger.info("Legacy configuration migration completed")
        return migrated_config

    def _migrate_unknown_to_validated(
        self, old_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Migrate configuration of unknown format to validated format"""
        logger.info("Migrating unknown configuration format to validated format")

        # Use defaults for unknown configuration
        migrated_config = {
            "version": "validated_pydantic",
            "session": {
                "max_active_sessions": 50,
                "default_ttl_hours": 1,
                "cleanup_interval_seconds": 300,
            },
            "model": {
                "default_max_tokens": 2048,
                "default_temperature": 0.7,
                "backends": {},
            },
            "security": {"enable_client_binding": True, "max_validation_failures": 3},
        }

        logger.info("Unknown configuration migration completed with defaults")
        return migrated_config

    def create_legacy_compatible_provider(
        self, migrated_config: Dict[str, Any]
    ) -> BaseConfigurationProvider:
        """Create a legacy-compatible configuration provider from migrated config"""
        provider = ConfigurationFactory.create_from_dict(migrated_config)

        # Validate the migrated configuration
        if not provider.validate():
            raise ConfigurationMigrationError(
                "Migrated configuration failed validation"
            )

        return provider


def create_validated_config_with_fallback() -> ValidatedContextSwitcherConfig:
    """Create validated config with fallback to legacy using dependency injection

    This function replaces the previous circular dependency approach by using
    the dependency injection container to get configuration providers.
    """
    if not _VALIDATED_CONFIG_AVAILABLE:
        raise ConfigurationError("Validated configuration system not available")

    try:
        # Try to load validated config directly
        return load_validated_config()

    except ConfigurationError:
        # Fallback: get configuration from dependency container
        logger.info(
            "Direct validated config loading failed, trying DI container fallback"
        )

        try:
            container = get_container()
            if container.has_registration(ConfigurationProvider):
                legacy_provider = container.get(ConfigurationProvider)

                # Convert legacy provider to validated config
                return _convert_legacy_provider_to_validated(legacy_provider)

        except Exception as e:
            logger.warning(f"DI container fallback failed: {e}")

        # Final fallback: create default validated config
        logger.info("Creating default validated configuration")
        return _create_default_validated_config()


def _convert_legacy_provider_to_validated(
    provider: ConfigurationProvider,
) -> ValidatedContextSwitcherConfig:
    """Convert legacy configuration provider to validated format"""
    if not _VALIDATED_CONFIG_AVAILABLE:
        raise ConfigurationError(
            "Cannot convert to validated config - system not available"
        )

    # Get configuration data from provider
    session_config = provider.get_session_config()
    security_config = provider.get_security_config()

    # Create configuration dictionary in validated format
    config_dict = {
        "session": {
            "max_active_sessions": session_config.max_active_sessions,
            "default_ttl_hours": session_config.default_ttl_hours,
            "cleanup_interval_seconds": session_config.cleanup_interval_seconds,
        },
        "security": security_config,
        "model": {"default_max_tokens": 2048, "default_temperature": 0.7},
    }

    # Create validated config from dictionary
    # Note: This would require the ValidatedContextSwitcherConfig to support dict initialization
    # For now, we'll create a basic validated config
    return _create_default_validated_config()


def _create_default_validated_config() -> ValidatedContextSwitcherConfig:
    """Create a default validated configuration"""
    if not _VALIDATED_CONFIG_AVAILABLE:
        raise ConfigurationError("Validated configuration system not available")

    # This would create a validated config with sensible defaults
    # The exact implementation depends on the ValidatedContextSwitcherConfig structure
    try:
        # Try to create with defaults
        from .validated_config import create_default_validated_config

        return create_default_validated_config()
    except (ImportError, AttributeError):
        # Fallback approach
        logger.warning(
            "Default validated config creation not available, using load_validated_config"
        )
        return load_validated_config(validate_dependencies=False)


# Migration utilities
def migrate_legacy_config_to_provider(
    legacy_config_dict: Dict[str, Any],
) -> BaseConfigurationProvider:
    """Migrate legacy configuration dictionary to new provider format"""
    migrator = CompatibilityAdapter()

    # Determine config version
    if "version" not in legacy_config_dict:
        legacy_config_dict["version"] = "legacy_dataclass"

    # Check if migration is needed
    if migrator.is_migration_needed(legacy_config_dict):
        migrated_dict = migrator.migrate_config(legacy_config_dict)
    else:
        migrated_dict = legacy_config_dict

    # Create provider from migrated config
    return migrator.create_legacy_compatible_provider(migrated_dict)


def setup_migration_in_container() -> None:
    """Setup migration-related dependencies in the DI container"""
    try:
        container = get_container()

        # Register the migrator
        def migrator_factory() -> ConfigurationMigrator:
            return CompatibilityAdapter()

        container.register_singleton_factory(ConfigurationMigrator, migrator_factory)
        logger.debug("Registered configuration migrator in DI container")

    except Exception as e:
        logger.warning(f"Failed to setup migration dependencies: {e}")


# Backward compatibility functions
def get_legacy_config_adapter() -> Optional[ConfigurationProvider]:
    """Get legacy configuration adapter from DI container if available"""
    try:
        container = get_container()
        return container.get_optional(ConfigurationProvider)
    except Exception as e:
        logger.debug(f"Could not get legacy config adapter: {e}")
        return None
