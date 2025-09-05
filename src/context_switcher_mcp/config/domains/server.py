"""Server configuration for Context Switcher MCP

This module handles MCP server configuration including:
- Network settings (host, port, binding)
- Logging configuration
- Performance tuning
- Connection limits and timeouts
- Development vs production settings
"""

import re
from enum import Enum

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Valid logging levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ServerConfig(BaseSettings):
    """Configuration for the MCP server networking and behavior

    Controls how the server binds to network interfaces, handles connections,
    and manages logging. Critical for both development and production deployments.
    """

    model_config = SettingsConfigDict(
        env_prefix="CS_", case_sensitive=False, extra="forbid", validate_assignment=True
    )

    # Network configuration
    host: str = Field(default="localhost", description="Server host address to bind to")

    port: int = Field(default=3023, ge=1024, le=65535, description="Server port number")

    # Logging configuration
    log_level: LogLevel = Field(
        default=LogLevel.INFO, description="Server logging level"
    )

    log_format: str = Field(
        default="structured",
        description="Log format: 'structured', 'simple', or 'json'",
    )

    enable_access_logging: bool = Field(
        default=True, description="Enable HTTP access logging"
    )

    log_file_path: str | None = Field(
        default=None, description="Path to log file (None for console only)"
    )

    max_log_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum log file size in MB before rotation",
    )

    log_file_backup_count: int = Field(
        default=5, ge=1, le=100, description="Number of rotated log files to keep"
    )

    # Connection and performance settings
    max_concurrent_connections: int = Field(
        default=100, ge=1, le=10000, description="Maximum concurrent client connections"
    )

    connection_timeout_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=600.0,
        description="Client connection timeout in seconds",
    )

    request_timeout_seconds: float = Field(
        default=300.0,  # 5 minutes
        ge=1.0,
        le=3600.0,
        description="Request processing timeout in seconds",
    )

    keepalive_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=300.0, description="HTTP keep-alive timeout in seconds"
    )

    # Development and debugging settings
    enable_debug_mode: bool = Field(
        default=False, description="Enable debug mode with additional logging"
    )

    enable_hot_reload: bool = Field(
        default=False, description="Enable hot reload for development"
    )

    enable_cors: bool = Field(
        default=False, description="Enable CORS headers for web clients"
    )

    cors_allowed_origins: list[str] = Field(
        default=[], description="Allowed CORS origins (empty = all)"
    )

    # Health and monitoring endpoints
    enable_health_endpoint: bool = Field(
        default=True, description="Enable /health endpoint"
    )

    enable_metrics_endpoint: bool = Field(
        default=True, description="Enable /metrics endpoint"
    )

    enable_status_endpoint: bool = Field(
        default=True, description="Enable /status endpoint with system info"
    )

    # Performance tuning
    worker_threads: int = Field(
        default=4,
        ge=1,
        le=100,
        description="Number of worker threads for request processing",
    )

    max_request_size_mb: int = Field(
        default=10, ge=1, le=1000, description="Maximum request size in MB"
    )

    enable_compression: bool = Field(
        default=True, description="Enable response compression"
    )

    compression_threshold_bytes: int = Field(
        default=1024,
        ge=100,
        le=100000,
        description="Minimum response size for compression",
    )

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host address format"""
        # Allow common host formats
        if v in ["localhost", "0.0.0.0", "*"]:
            return v

        # Validate IP address format
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", v):
            parts = v.split(".")
            if not all(0 <= int(part) <= 255 for part in parts):
                raise ValueError("Invalid IP address format")
            return v

        # Allow hostname format (basic validation)
        if re.match(r"^[a-zA-Z0-9\.\-]+$", v):
            return v

        raise ValueError("Invalid host address format")

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format option"""
        valid_formats = {"structured", "simple", "json"}
        if v not in valid_formats:
            raise ValueError(f"Invalid log format: {v}. Valid: {valid_formats}")
        return v

    @field_validator("cors_allowed_origins")
    @classmethod
    def validate_cors_origins(cls, v: list[str]) -> list[str]:
        """Validate CORS origin URLs"""
        for origin in v:
            if origin == "*":
                continue  # Allow wildcard
            if not re.match(r"^https?://[a-zA-Z0-9\.\-:]+$", origin):
                raise ValueError(f"Invalid CORS origin format: {origin}")
        return v

    @computed_field
    @property
    def is_development_mode(self) -> bool:
        """Check if server is configured for development"""
        return (
            self.log_level == LogLevel.DEBUG
            or self.enable_debug_mode
            or self.enable_hot_reload
            or self.host == "localhost"
        )

    @computed_field
    @property
    def is_production_ready(self) -> bool:
        """Check if server configuration is production-ready"""
        return (
            self.log_level in [LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
            and not self.enable_debug_mode
            and not self.enable_hot_reload
            and self.max_concurrent_connections >= 10
            and self.request_timeout_seconds <= 600
        )

    @computed_field
    @property
    def bind_address(self) -> str:
        """Get the full bind address"""
        return f"{self.host}:{self.port}"

    def get_log_config(self) -> dict[str, any]:
        """Get logging configuration dictionary

        Returns:
            Dictionary with logging parameters
        """
        return {
            "level": self.log_level.value,
            "format": self.log_format,
            "enable_access_logging": self.enable_access_logging,
            "file_path": self.log_file_path,
            "max_file_size_mb": self.max_log_file_size_mb,
            "backup_count": self.log_file_backup_count,
        }

    def get_connection_config(self) -> dict[str, any]:
        """Get connection configuration dictionary

        Returns:
            Dictionary with connection parameters
        """
        return {
            "max_connections": self.max_concurrent_connections,
            "connection_timeout": self.connection_timeout_seconds,
            "request_timeout": self.request_timeout_seconds,
            "keepalive_timeout": self.keepalive_timeout_seconds,
        }

    def get_cors_config(self) -> dict[str, any]:
        """Get CORS configuration dictionary

        Returns:
            Dictionary with CORS parameters
        """
        return {
            "enabled": self.enable_cors,
            "allowed_origins": self.cors_allowed_origins.copy(),
        }

    def get_performance_config(self) -> dict[str, any]:
        """Get performance tuning configuration

        Returns:
            Dictionary with performance parameters
        """
        return {
            "worker_threads": self.worker_threads,
            "max_request_size_mb": self.max_request_size_mb,
            "enable_compression": self.enable_compression,
            "compression_threshold_bytes": self.compression_threshold_bytes,
        }

    def get_monitoring_endpoints(self) -> dict[str, bool]:
        """Get enabled monitoring endpoints

        Returns:
            Dictionary of endpoint names to enabled status
        """
        return {
            "health": self.enable_health_endpoint,
            "metrics": self.enable_metrics_endpoint,
            "status": self.enable_status_endpoint,
        }

    def is_secure_deployment(self) -> bool:
        """Check if deployment is configured securely

        Returns:
            True if configuration follows security best practices
        """
        return (
            self.host != "0.0.0.0"  # Not bound to all interfaces
            or (
                self.enable_cors and self.cors_allowed_origins
            )  # CORS properly configured
            or not self.enable_debug_mode  # Debug mode disabled
        )
