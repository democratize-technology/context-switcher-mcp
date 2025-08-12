"""Security configuration for Context Switcher MCP

This module handles all security-related configuration including:
- Encryption keys and secrets
- Client authentication and binding
- Rate limiting and abuse prevention
- Input validation and sanitization
- Security monitoring and alerting

Security is critical for production deployments, especially when handling
sensitive analysis topics or operating in multi-tenant environments.
"""

import re
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecurityConfig(BaseSettings):
    """Configuration for security, encryption, and access control
    
    Handles all security aspects of the Context Switcher MCP server,
    from encryption keys to client validation and abuse prevention.
    """
    
    model_config = SettingsConfigDict(
        env_prefix="CS_SECURITY_",
        case_sensitive=True,  # Security settings are case-sensitive
        extra="forbid"
    )
    
    # Core encryption and secrets
    secret_key: Optional[str] = Field(
        default=None,
        min_length=32,
        description="Master secret key for encryption (base64 encoded)",
        env="CONTEXT_SWITCHER_SECRET_KEY"
    )
    
    session_secret_key: Optional[str] = Field(
        default=None,
        min_length=32,
        description="Session-specific encryption key",
        env="CS_SESSION_SECRET_KEY"
    )
    
    # Client authentication and binding
    enable_client_binding: bool = Field(
        default=True,
        description="Enable client binding for session security",
        env="CS_ENABLE_CLIENT_BINDING"
    )
    
    client_binding_entropy_bytes: int = Field(
        default=32,
        ge=16,
        le=128,
        description="Entropy bytes for client binding signatures",
        env="CS_CLIENT_BINDING_ENTROPY"
    )
    
    binding_signature_algorithm: str = Field(
        default="pbkdf2_hmac",
        description="Algorithm for client binding signatures",
        env="CS_BINDING_SIGNATURE_ALGORITHM"
    )
    
    signature_iterations: int = Field(
        default=600000,
        ge=100000,
        le=2000000,
        description="PBKDF2 iterations for signature generation",
        env="CS_SIGNATURE_ITERATIONS"
    )
    
    # Rate limiting and abuse prevention
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting for API endpoints",
        env="CS_ENABLE_RATE_LIMITING"
    )
    
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Maximum requests per minute per client",
        env="CS_RATE_LIMIT_RPM"
    )
    
    rate_limit_burst_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Burst request allowance above rate limit",
        env="CS_RATE_LIMIT_BURST"
    )
    
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Rate limiting window duration",
        env="CS_RATE_LIMIT_WINDOW"
    )
    
    # Input validation and sanitization
    max_input_length: int = Field(
        default=1000000,  # 1MB
        ge=1000,
        le=100000000,  # 100MB max
        description="Maximum input length for any field",
        env="CS_MAX_INPUT_LENGTH"
    )
    
    enable_input_sanitization: bool = Field(
        default=True,
        description="Enable input sanitization and validation",
        env="CS_ENABLE_INPUT_SANITIZATION"
    )
    
    blocked_patterns: list[str] = Field(
        default=[
            r"<script[^>]*>.*?</script>",  # Script tags
            r"javascript:",  # JavaScript URLs
            r"data:",  # Data URLs
            r"vbscript:",  # VBScript URLs
        ],
        description="Regex patterns to block in inputs",
        env="CS_BLOCKED_PATTERNS"
    )
    
    # Access control
    max_validation_failures: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum validation failures before blocking",
        env="CS_MAX_VALIDATION_FAILURES"
    )
    
    validation_failure_window_seconds: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600,
        description="Window for counting validation failures",
        env="CS_VALIDATION_FAILURE_WINDOW"
    )
    
    enable_suspicious_activity_detection: bool = Field(
        default=True,
        description="Enable detection of suspicious activity patterns",
        env="CS_ENABLE_SUSPICIOUS_DETECTION"
    )
    
    suspicious_activity_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Threshold for flagging suspicious activity",
        env="CS_SUSPICIOUS_ACTIVITY_THRESHOLD"
    )
    
    # Security monitoring
    enable_security_logging: bool = Field(
        default=True,
        description="Enable detailed security event logging",
        env="CS_ENABLE_SECURITY_LOGGING"
    )
    
    security_log_level: str = Field(
        default="INFO",
        description="Logging level for security events",
        env="CS_SECURITY_LOG_LEVEL"
    )
    
    enable_security_alerts: bool = Field(
        default=True,
        description="Enable security alerting for critical events",
        env="CS_ENABLE_SECURITY_ALERTS"
    )
    
    # Session security
    session_timeout_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,  # Max 24 hours
        description="Session timeout in minutes",
        env="CS_SESSION_TIMEOUT_MINUTES"
    )
    
    enable_session_rotation: bool = Field(
        default=True,
        description="Enable automatic session key rotation",
        env="CS_ENABLE_SESSION_ROTATION"
    )
    
    session_rotation_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Interval for session key rotation",
        env="CS_SESSION_ROTATION_INTERVAL"
    )
    
    @field_validator("secret_key", "session_secret_key")
    @classmethod
    def validate_secret_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate secret key format and strength"""
        if v is not None:
            if len(v) < 32:
                raise ValueError("Secret keys must be at least 32 characters long")
            
            # Check for base64 format (recommended)
            if not re.match(r"^[A-Za-z0-9+/=]+$", v):
                # Allow other formats but warn
                if not re.match(r"^[A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]+$", v):
                    raise ValueError("Secret key contains invalid characters")
        return v
    
    @field_validator("binding_signature_algorithm")
    @classmethod
    def validate_signature_algorithm(cls, v: str) -> str:
        """Validate that signature algorithm is supported"""
        supported_algorithms = {"pbkdf2_hmac", "scrypt", "argon2"}
        if v not in supported_algorithms:
            raise ValueError(
                f"Unsupported signature algorithm: {v}. "
                f"Supported: {supported_algorithms}"
            )
        return v
    
    @field_validator("security_log_level")
    @classmethod
    def validate_security_log_level(cls, v: str) -> str:
        """Validate security log level"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Valid: {valid_levels}")
        return v.upper()
    
    @computed_field
    @property
    def has_encryption_keys(self) -> bool:
        """Check if encryption keys are configured"""
        return self.secret_key is not None
    
    @computed_field
    @property
    def is_production_secure(self) -> bool:
        """Check if configuration is secure for production"""
        return (
            self.secret_key is not None and
            len(self.secret_key) >= 32 and
            self.enable_client_binding and
            self.enable_rate_limiting and
            self.enable_input_sanitization and
            self.signature_iterations >= 600000
        )
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration
        
        Returns:
            Dictionary with rate limiting parameters
        """
        return {
            "enabled": self.enable_rate_limiting,
            "requests_per_minute": self.rate_limit_requests_per_minute,
            "burst_size": self.rate_limit_burst_size,
            "window_seconds": self.rate_limit_window_seconds,
        }
    
    def get_client_binding_config(self) -> Dict[str, Any]:
        """Get client binding configuration
        
        Returns:
            Dictionary with client binding parameters
        """
        return {
            "enabled": self.enable_client_binding,
            "entropy_bytes": self.client_binding_entropy_bytes,
            "algorithm": self.binding_signature_algorithm,
            "iterations": self.signature_iterations,
        }
    
    def get_input_validation_config(self) -> Dict[str, Any]:
        """Get input validation configuration
        
        Returns:
            Dictionary with input validation parameters
        """
        return {
            "enabled": self.enable_input_sanitization,
            "max_length": self.max_input_length,
            "blocked_patterns": self.blocked_patterns.copy(),
        }
    
    def get_session_security_config(self) -> Dict[str, Any]:
        """Get session security configuration
        
        Returns:
            Dictionary with session security parameters
        """
        return {
            "timeout_minutes": self.session_timeout_minutes,
            "enable_rotation": self.enable_session_rotation,
            "rotation_interval_hours": self.session_rotation_interval_hours,
            "has_session_key": self.session_secret_key is not None,
        }
    
    def should_log_security_event(self, level: str) -> bool:
        """Check if a security event should be logged
        
        Args:
            level: Event log level
            
        Returns:
            True if event should be logged
        """
        if not self.enable_security_logging:
            return False
        
        level_priority = {
            "DEBUG": 10,
            "INFO": 20, 
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50
        }
        
        return level_priority.get(level, 0) >= level_priority.get(self.security_log_level, 20)
    
    def validate_input_length(self, input_data: str) -> bool:
        """Validate input length against security limits
        
        Args:
            input_data: Input string to validate
            
        Returns:
            True if input length is acceptable
        """
        return len(input_data) <= self.max_input_length
    
    def check_blocked_patterns(self, input_data: str) -> list[str]:
        """Check input against blocked patterns
        
        Args:
            input_data: Input string to check
            
        Returns:
            List of matched blocked patterns (empty if none)
        """
        matched_patterns = []
        for pattern in self.blocked_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                matched_patterns.append(pattern)
        return matched_patterns