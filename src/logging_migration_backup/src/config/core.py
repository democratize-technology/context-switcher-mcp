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

import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any

from pydantic import BaseModel, Field, ConfigDict, computed_field
from pydantic_settings import BaseSettings

from .domains.models import ModelConfig
from .domains.session import SessionConfig  
from .domains.security import SecurityConfig
from .domains.server import ServerConfig
from .domains.monitoring import MonitoringConfig

logger = logging.getLogger(__name__)


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
    )
    
    # Configuration domains - each domain manages its own validation
    models: ModelConfig = Field(default_factory=ModelConfig)
    session: SessionConfig = Field(default_factory=SessionConfig) 
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Configuration metadata
    _config_version: str = Field(default="unified-v1", alias="config_version")
    _loaded_from: Optional[str] = Field(default=None)
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None, **kwargs):
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
            self._loaded_from = str(config_file)
        
        # Merge file data with kwargs (kwargs take precedence)
        merged_data = {**file_data, **kwargs}
        
        try:
            super().__init__(**merged_data)
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
            self.security.secret_key is not None and 
            len(self.security.secret_key) >= 32 and
            
            # Logging requirements
            self.server.log_level in ["INFO", "WARNING", "ERROR"] and
            
            # Monitoring requirements  
            self.monitoring.profiling.level in ["basic", "standard"] and
            
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
    
    def get_masked_dict(self) -> Dict[str, Any]:
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
            "httpx": "Ollama backend connectivity"
        }
        
        for package, description in optional_deps.items():
            try:
                __import__(package)
                messages.append(f"✓ {package} available for {description}")
            except ImportError:
                messages.append(f"⚠ {package} not installed - {description} unavailable")
        
        # Check critical environment variables
        if not os.getenv("CONTEXT_SWITCHER_SECRET_KEY"):
            messages.append("⚠ CONTEXT_SWITCHER_SECRET_KEY not set - security features limited")
        
        return messages
    
    def _load_config_file(self, config_file: Union[str, Path]) -> Dict[str, Any]:
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
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.json':
                    import json
                    return json.load(f)
                elif config_path.suffix.lower() in ['.yaml', '.yml']:
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


def create_config(
    environment: Optional[str] = None,
    config_file: Optional[Union[str, Path]] = None,
    **overrides
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
        overrides.setdefault("monitoring", {}).setdefault("profiling", {})["level"] = "detailed"
        
    elif environment == "production":
        overrides.setdefault("server", {})["log_level"] = "INFO" 
        overrides.setdefault("monitoring", {}).setdefault("profiling", {})["level"] = "standard"
    
    return ContextSwitcherConfig(config_file=config_file, **overrides)


# Configuration validation utilities
def validate_config_dict(config_dict: Dict[str, Any]) -> list[str]:
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