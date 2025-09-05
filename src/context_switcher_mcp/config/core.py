"""Core unified configuration system

This module provides the main ContextSwitcherConfig class that serves as the
single source of truth for all configuration. It consolidates and validates
all configuration domains while maintaining a clean, extensible architecture.

The design principles:
- Single responsibility: Each domain config handles its own validation
- Type safety: Full Pydantic validation with clear error messages
- Environment awareness: Support for dev/staging/prod configurations
- Backward compatibility: Maintains interface compatibility where possible
- Extensibility: Easy to add new configuration domains
"""

import os
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, computed_field
from pydantic_settings import BaseSettings

from ..logging_base import get_logger
from .domains.models import ModelConfig
from .domains.monitoring import MonitoringConfig, ProfilingConfig
from .domains.security import SecurityConfig
from .domains.server import ServerConfig
from .domains.session import SessionConfig

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation or loading fails"""

    pass


class ContextSwitcherConfig(BaseSettings):
    """Unified configuration for the Context Switcher MCP server

    This class consolidates all configuration domains into a single,
    validated configuration object. It uses Pydantic for validation
    and supports loading from environment variables, files, and defaults.

    Configuration Domains:
    - models: LLM backend configuration (Bedrock, LiteLLM, Ollama)
    - session: Session management settings
    - security: Security and encryption settings
    - server: MCP server configuration
    - monitoring: Profiling and metrics configuration

    Environment Variable Prefixes:
    - CS_: General configuration
    - CS_MODEL_: Model configuration
    - CS_SESSION_: Session configuration
    - CS_SECURITY_: Security configuration
    - CS_SERVER_: Server configuration
    - CS_MONITORING_: Monitoring configuration

    File Loading:
    Supports JSON and YAML configuration files. Environment variables
    take precedence over file values.
    """

    model_config = ConfigDict(
        case_sensitive=False,
        extra="forbid",  # Reject unknown configuration keys
        validate_assignment=True,  # Validate on attribute assignment
        str_strip_whitespace=True,  # Clean up string inputs
        env_nested_delimiter="__",  # Allow nested environment variables like CS_SERVER__PORT
    )

    # Configuration domains - each domain manages its own validation
    models: ModelConfig = Field(default_factory=ModelConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Configuration metadata
    config_version: str = Field(default="unified-v1")
    loaded_from: str | None = Field(default=None)

    def __init__(self, config_file: str | Path | None = None, **kwargs):
        """Initialize configuration with validation

        Args:
            config_file: Optional path to configuration file (JSON/YAML)
            **kwargs: Additional configuration overrides

        Raises:
            ConfigurationError: If configuration validation fails
        """
        # Load from file if provided
        file_data = {}
        if config_file:
            file_data = self._load_config_file(config_file)
            kwargs["loaded_from"] = str(config_file)

        # Process environment variables with nested support
        env_data = self._load_environment_variables()

        # Merge file data, environment data, and kwargs (kwargs take highest precedence)
        merged_data = {**file_data, **env_data, **kwargs}

        try:
            # Extract domain-specific data from merged_data
            domain_data = {}
            main_data = {}

            for key, value in merged_data.items():
                if key in ["models", "session", "security", "server", "monitoring"]:
                    # This is domain data, pass it as is
                    domain_data[key] = value
                else:
                    # This is main config data
                    main_data[key] = value

            # Initialize with main data first
            super().__init__(**main_data)

            # Then update domain configs with any domain-specific data
            for domain_name, domain_config in domain_data.items():
                if hasattr(self, domain_name) and isinstance(domain_config, dict):
                    # Get the current domain instance
                    current_domain = getattr(self, domain_name)

                    # Update domain config field by field to ensure proper validation
                    updated_domain = self._update_domain_config(
                        current_domain, domain_config
                    )

                    # Set the updated domain
                    setattr(self, domain_name, updated_domain)

            logger.info("Configuration initialized successfully")

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e

    @computed_field
    @property
    def is_production_ready(self) -> bool:
        """Check if configuration is suitable for production deployment

        Validates that security, logging, and monitoring settings are
        appropriate for production use.

        Returns:
            True if configuration is production-ready
        """
        return (
            # Security requirements
            self.security.secret_key is not None
            and len(self.security.secret_key) >= 32
            and
            # Logging requirements
            self.server.log_level in ["INFO", "WARNING", "ERROR"]
            and
            # Monitoring requirements
            self.monitoring.profiling.level in ["basic", "standard"]
            and
            # Session requirements
            self.session.cleanup_interval_seconds <= 3600  # Max 1 hour
        )

    @computed_field
    @property
    def deployment_environment(self) -> str:
        """Detect the current deployment environment

        Uses environment variables and configuration settings to determine
        the current deployment environment.

        Returns:
            Environment name: "development", "staging", or "production"
        """
        # Check explicit environment variable
        env = os.getenv("ENVIRONMENT", os.getenv("ENV", "")).lower()
        if env in ["dev", "development"]:
            return "development"
        elif env in ["staging", "stage"]:
            return "staging"
        elif env in ["prod", "production"]:
            return "production"

        # Infer from configuration
        if self.server.log_level == "DEBUG":
            return "development"
        elif not self.is_production_ready:
            return "development"
        else:
            return "production"

    @computed_field
    @property
    def validation(self) -> SessionConfig:
        """Backward compatibility property for validation configuration

        Returns the session configuration which contains validation settings
        like max_topic_length and max_session_id_length.

        Returns:
            SessionConfig instance with validation properties
        """
        return self.session

    @computed_field
    @property
    def profiling(self) -> ProfilingConfig:
        """Backward compatibility property for profiling configuration

        Returns the profiling configuration from the monitoring domain
        which contains settings like enabled, level, and sampling_rate.

        Returns:
            ProfilingConfig instance with profiling properties
        """
        return self.monitoring.profiling

    @computed_field
    @property
    def model(self) -> ModelConfig:
        """Backward compatibility property for model configuration

        Returns the models configuration to maintain compatibility with
        legacy code that accesses config.model instead of config.models.

        Returns:
            ModelConfig instance with model properties
        """
        return self.models

    def get_masked_dict(self) -> dict[str, Any]:
        """Get configuration as dictionary with sensitive data masked

        This is safe for logging and debugging as it masks sensitive
        information like secret keys and API credentials.

        Returns:
            Configuration dictionary with sensitive fields masked
        """
        config_dict = self.model_dump()

        # Mask sensitive security fields
        if config_dict.get("security", {}).get("secret_key"):
            config_dict["security"]["secret_key"] = "***MASKED***"

        # Mask any credentials that might be in model configuration
        models_config = config_dict.get("models", {})
        for key in ["bedrock_model_id", "litellm_model"]:
            if key in models_config:
                value = models_config[key]
                # Mask if it looks like a credential (long string with special chars)
                if len(value) > 50 or any(char in value for char in [":", "=", "/"]):
                    config_dict["models"][key] = "***MASKED***"

        return config_dict

    def validate_external_dependencies(self) -> list[str]:
        """Validate external service dependencies (non-blocking)

        Checks availability of external services and libraries without
        causing configuration loading to fail if they're unavailable.

        Returns:
            List of validation messages (info/warnings)
        """
        messages = []

        # Check Python package dependencies
        optional_deps = {
            "litellm": "LiteLLM backend functionality",
            "boto3": "AWS Bedrock backend functionality",
            "yaml": "YAML configuration file support",
            "httpx": "Ollama backend connectivity",
        }

        for package, description in optional_deps.items():
            try:
                __import__(package)
                messages.append(f"✓ {package} available for {description}")
            except ImportError:
                messages.append(
                    f"⚠ {package} not installed - {description} unavailable"
                )

        # Check critical environment variables
        if not os.getenv("CONTEXT_SWITCHER_SECRET_KEY"):
            messages.append(
                "⚠ CONTEXT_SWITCHER_SECRET_KEY not set - security features limited"
            )

        return messages

    def _load_config_file(self, config_file: str | Path) -> dict[str, Any]:
        """Load configuration from JSON or YAML file

        Args:
            config_file: Path to configuration file

        Returns:
            Configuration data dictionary

        Raises:
            ConfigurationError: If file loading fails
        """
        config_path = Path(config_file)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as f:
                if config_path.suffix.lower() == ".json":
                    import json

                    return json.load(f)
                elif config_path.suffix.lower() in [".yaml", ".yml"]:
                    import yaml

                    return yaml.safe_load(f) or {}
                else:
                    raise ConfigurationError(
                        f"Unsupported configuration file format: {config_path.suffix}"
                    )

        except ImportError as e:
            if "yaml" in str(e):
                raise ConfigurationError(
                    "PyYAML not installed. Install with: pip install pyyaml"
                ) from e
            raise ConfigurationError(f"Failed to load configuration file: {e}") from e

        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse configuration file {config_path}: {e}"
            ) from e

    def _update_domain_config(self, current_domain, updates: dict) -> Any:
        """Update a domain configuration with proper field validation

        This method updates domain configs field by field to ensure proper
        validation and type conversion while avoiding BaseSettings env processing issues.

        Args:
            current_domain: Current domain configuration instance
            updates: Dictionary of field updates

        Returns:
            Updated domain configuration instance
        """
        # Create a copy and update it field by field using setattr with validation
        updated_domain = current_domain.model_copy(deep=True)

        for field_name, new_value in updates.items():
            if hasattr(updated_domain, field_name):
                # Use setattr which will trigger Pydantic field validation
                try:
                    setattr(updated_domain, field_name, new_value)
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to set {field_name}={new_value} in {type(current_domain).__name__}: {e}"
                    ) from e
            else:
                # Field doesn't exist in this domain - log a warning but don't fail
                logger.warning(
                    f"Unknown field '{field_name}' for {type(current_domain).__name__}"
                )

        return updated_domain

    def _load_environment_variables(self) -> dict[str, Any]:
        """Load configuration from environment variables with nested support

        Handles environment variables for domains that don't follow the standard pattern.
        Most domain configs handle their own env vars via pydantic-settings.
        This handles special cases and legacy variable names.

        Returns:
            Configuration data dictionary with nested structure
        """
        import os

        env_data = {}

        # Handle legacy/non-standard environment variable names that don't match
        # the standard CS_ prefix pattern used by individual domain configs
        legacy_mappings = {
            # These are examples - the actual domain configs handle most env vars
        }

        for env_var, (domain, field) in legacy_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if domain not in env_data:
                    env_data[domain] = {}

                # Convert value to appropriate type
                try:
                    if (
                        field.endswith("_port")
                        or "max_tokens" in field
                        or "ttl_hours" in field
                    ):
                        env_data[domain][field] = int(value)
                    elif field.endswith("_temperature"):
                        env_data[domain][field] = float(value)
                    elif "enable_" in field:
                        env_data[domain][field] = value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    else:
                        env_data[domain][field] = value
                except (ValueError, TypeError):
                    # Keep as string if conversion fails
                    env_data[domain][field] = value

        return env_data


def create_config(
    environment: str | None = None,
    config_file: str | Path | None = None,
    **overrides,
) -> ContextSwitcherConfig:
    """Factory function to create configuration instances

    This is a convenience function for creating configuration instances
    with specific environments or overrides.

    Args:
        environment: Target environment (development, staging, production)
        config_file: Optional configuration file path
        **overrides: Configuration value overrides

    Returns:
        Validated configuration instance

    Raises:
        ConfigurationError: If configuration creation fails
    """
    # Apply environment-specific defaults
    if environment == "development":
        overrides.setdefault("server", {})["log_level"] = "DEBUG"
        overrides.setdefault("monitoring", {}).setdefault("profiling", {})[
            "level"
        ] = "detailed"

    elif environment == "production":
        overrides.setdefault("server", {})["log_level"] = "INFO"
        overrides.setdefault("monitoring", {}).setdefault("profiling", {})[
            "level"
        ] = "standard"

    return ContextSwitcherConfig(config_file=config_file, **overrides)


# Configuration validation utilities
def validate_config_dict(config_dict: dict[str, Any]) -> list[str]:
    """Validate a configuration dictionary without creating an instance

    Args:
        config_dict: Configuration data to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    try:
        ContextSwitcherConfig(**config_dict)
        return []
    except Exception as e:
        return [str(e)]
