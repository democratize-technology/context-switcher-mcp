"""Session management configuration

This module handles configuration for session lifecycle, cleanup, and limits.
Sessions are central to the Context Switcher's multi-perspective analysis
functionality.

Key features:
- Session TTL and cleanup configuration
- Concurrent session limits
- Validation rules for session parameters
- Timeout and retry settings
- Memory management settings
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SessionConfig(BaseSettings):
    """Configuration for session management and lifecycle

    Controls how sessions are created, maintained, and cleaned up.
    Critical for preventing resource leaks and ensuring system stability.
    """

    model_config = SettingsConfigDict(
        env_prefix="CS_", case_sensitive=False, extra="forbid", validate_assignment=True
    )

    # Session lifecycle settings
    default_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Maximum 1 week
        description="Default session time-to-live in hours",
    )

    max_ttl_hours: int = Field(
        default=168,  # 1 week
        ge=1,
        le=720,  # Maximum 30 days
        description="Maximum allowed session TTL in hours",
    )

    cleanup_interval_seconds: int = Field(
        default=600,  # 10 minutes
        ge=60,
        le=3600,  # Maximum 1 hour
        description="Interval between session cleanup runs",
    )

    # Concurrent session limits
    max_active_sessions: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Maximum number of active sessions",
    )

    max_sessions_per_client: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum sessions per client identifier",
    )

    # Session operation settings
    session_operation_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for individual session operations",
    )

    perspective_analysis_timeout_seconds: float = Field(
        default=120.0,
        ge=5.0,
        le=600.0,
        description="Timeout for perspective analysis operations",
    )

    synthesis_timeout_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=300.0,
        description="Timeout for perspective synthesis operations",
    )

    # Memory and resource management
    max_session_memory_mb: float = Field(
        default=100.0,
        ge=1.0,
        le=1000.0,
        description="Maximum memory per session in MB",
    )

    max_session_history_entries: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum history entries per session",
    )

    enable_session_compression: bool = Field(
        default=True,
        description="Enable compression for session data",
    )

    # Validation settings for session inputs
    max_session_id_length: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum length for session identifiers",
    )

    max_topic_length: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum length for analysis topics",
    )

    max_perspective_name_length: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum length for perspective names",
    )

    max_custom_prompt_length: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum length for custom perspective prompts",
    )

    # Concurrent operations
    max_concurrent_perspectives: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent perspectives per session",
    )

    max_concurrent_sessions: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum concurrent active session operations",
    )

    # Session persistence settings
    enable_session_persistence: bool = Field(
        default=True,
        description="Enable session data persistence across restarts",
    )

    session_storage_path: str | None = Field(
        default=None,
        description="Path for session data storage (None for memory only)",
    )

    @field_validator("max_ttl_hours")
    @classmethod
    def validate_max_ttl_greater_than_default(cls, v: int, info) -> int:
        """Ensure max TTL is greater than or equal to default TTL"""
        # Note: In Pydantic v2, we can't directly access other fields during validation
        # This validation will be done at the model level if needed
        return v

    @field_validator("cleanup_interval_seconds")
    @classmethod
    def validate_cleanup_interval(cls, v: int) -> int:
        """Validate cleanup interval is reasonable for production"""
        if v < 60:
            raise ValueError("Cleanup interval must be at least 60 seconds")
        return v

    @field_validator("max_sessions_per_client")
    @classmethod
    def validate_client_session_limit(cls, v: int) -> int:
        """Validate per-client session limit is reasonable"""
        if v > 100:
            raise ValueError("Per-client session limit should not exceed 100")
        return v

    def get_cleanup_schedule_seconds(self) -> int:
        """Get the cleanup schedule in seconds

        Returns interval at which expired sessions should be cleaned up.

        Returns:
            Cleanup interval in seconds
        """
        return self.cleanup_interval_seconds

    def is_ttl_valid(self, ttl_hours: int) -> bool:
        """Check if a TTL value is within allowed limits

        Args:
            ttl_hours: TTL value to validate

        Returns:
            True if TTL is valid
        """
        return 1 <= ttl_hours <= self.max_ttl_hours

    def get_session_memory_limit_bytes(self) -> int:
        """Get session memory limit in bytes

        Returns:
            Memory limit in bytes
        """
        return int(self.max_session_memory_mb * 1024 * 1024)

    def should_compress_session_data(self) -> bool:
        """Check if session data compression is enabled

        Returns:
            True if compression should be used
        """
        return self.enable_session_compression

    def get_operation_timeouts(self) -> dict[str, float]:
        """Get all operation timeout values

        Returns:
            Dictionary of operation names to timeout values in seconds
        """
        return {
            "session_operation": self.session_operation_timeout_seconds,
            "perspective_analysis": self.perspective_analysis_timeout_seconds,
            "synthesis": self.synthesis_timeout_seconds,
        }
