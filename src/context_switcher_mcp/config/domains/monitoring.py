"""Monitoring and observability configuration

This module handles configuration for:
- LLM profiling and metrics collection
- Performance monitoring
- Cost tracking and alerting
- System observability
- Data retention and storage
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProfilingLevel(str, Enum):
    """Profiling detail levels"""

    DISABLED = "disabled"
    BASIC = "basic"
    STANDARD = "standard"
    DETAILED = "detailed"


class MetricsConfig(BaseModel):
    """Configuration for metrics collection and storage"""

    max_history_size: int = Field(
        default=1000,
        ge=10,
        le=1000000,
        description="Maximum metrics history entries to retain",
    )

    retention_days: int = Field(
        default=7, ge=1, le=365, description="Metrics retention period in days"
    )

    collection_interval_seconds: int = Field(
        default=60, ge=1, le=3600, description="Metrics collection interval in seconds"
    )

    enable_system_metrics: bool = Field(
        default=True, description="Enable system resource metrics collection"
    )

    enable_application_metrics: bool = Field(
        default=True, description="Enable application-level metrics"
    )


class ProfilingConfig(BaseModel):
    """Configuration for LLM profiling and detailed monitoring"""

    enabled: bool = Field(default=True, description="Enable LLM profiling")

    level: ProfilingLevel = Field(
        default=ProfilingLevel.STANDARD, description="Profiling detail level"
    )

    sampling_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Profiling sampling rate (0.0-1.0)"
    )

    # Feature flags for different types of tracking
    track_tokens: bool = Field(
        default=True, description="Track token usage in API calls"
    )

    track_costs: bool = Field(default=True, description="Track API costs and spending")

    track_memory: bool = Field(
        default=False, description="Track memory usage (expensive)"
    )

    track_network_timing: bool = Field(
        default=True, description="Track network request timing"
    )

    track_model_performance: bool = Field(
        default=True, description="Track model-specific performance metrics"
    )

    # Storage and retention settings
    max_history_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum profiling entries to retain",
    )

    # Alert thresholds
    cost_alert_threshold_usd: float = Field(
        default=100.0,
        ge=0.01,
        le=10000.0,
        description="Daily cost alert threshold in USD",
    )

    latency_alert_threshold_s: float = Field(
        default=30.0,
        ge=0.1,
        le=300.0,
        description="High latency alert threshold in seconds",
    )

    memory_alert_threshold_mb: float = Field(
        default=1000.0,
        ge=10.0,
        le=100000.0,
        description="High memory usage alert threshold in MB",
    )

    error_rate_alert_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Error rate alert threshold (0.0-1.0)"
    )

    # Conditional profiling rules
    always_profile_errors: bool = Field(
        default=True, description="Always profile error conditions"
    )

    always_profile_slow_calls: bool = Field(
        default=True, description="Always profile slow API calls"
    )

    always_profile_expensive_calls: bool = Field(
        default=True, description="Always profile expensive API calls"
    )

    always_profile_circuit_breaker: bool = Field(
        default=True, description="Always profile circuit breaker events"
    )


class AlertingConfig(BaseModel):
    """Configuration for alerting and notifications"""

    enabled: bool = Field(default=True, description="Enable alerting system")

    alert_cooldown_minutes: int = Field(
        default=15,
        ge=1,
        le=1440,
        description="Cooldown between similar alerts in minutes",
    )

    enable_cost_alerts: bool = Field(
        default=True, description="Enable cost threshold alerts"
    )

    enable_performance_alerts: bool = Field(
        default=True, description="Enable performance threshold alerts"
    )

    enable_error_alerts: bool = Field(
        default=True, description="Enable error rate alerts"
    )

    enable_security_alerts: bool = Field(
        default=True, description="Enable security event alerts"
    )


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring, profiling, and observability

    This configuration controls all aspects of system monitoring including
    performance metrics, cost tracking, and alerting.
    """

    model_config = SettingsConfigDict(
        env_prefix="CS_", case_sensitive=False, extra="forbid", validate_assignment=True
    )

    # Sub-configurations
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)

    # Global monitoring settings
    enable_monitoring: bool = Field(
        default=True, description="Enable monitoring system"
    )

    monitoring_storage_path: str | None = Field(
        default=None,
        description="Path for monitoring data storage",
    )

    enable_real_time_metrics: bool = Field(
        default=True,
        description="Enable real-time metrics streaming",
    )

    metrics_export_format: str = Field(
        default="json",
        description="Format for metrics export: 'json', 'csv', 'prometheus'",
    )

    enable_automated_reports: bool = Field(
        default=False,
        description="Enable automated performance reports",
    )

    report_generation_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Report generation interval in hours",
    )

    @field_validator("metrics_export_format")
    @classmethod
    def validate_export_format(cls, v: str) -> str:
        """Validate metrics export format"""
        valid_formats = {"json", "csv", "prometheus"}
        if v not in valid_formats:
            raise ValueError(f"Invalid export format: {v}. Valid: {valid_formats}")
        return v

    @computed_field
    @property
    def is_production_monitoring(self) -> bool:
        """Check if monitoring is configured for production"""
        return (
            self.enable_monitoring
            and self.profiling.level in [ProfilingLevel.BASIC, ProfilingLevel.STANDARD]
            and self.alerting.enabled
            and self.metrics.retention_days >= 7
        )

    @computed_field
    @property
    def estimated_storage_mb_per_day(self) -> float:
        """Estimate daily storage requirements in MB"""
        base_storage = 1.0  # Base metrics

        if self.profiling.enabled:
            if self.profiling.level == ProfilingLevel.DETAILED:
                base_storage *= 10
            elif self.profiling.level == ProfilingLevel.STANDARD:
                base_storage *= 5
            elif self.profiling.level == ProfilingLevel.BASIC:
                base_storage *= 2

        # Factor in sampling rate
        base_storage *= self.profiling.sampling_rate

        # Factor in collection frequency
        daily_collections = 86400 / self.metrics.collection_interval_seconds
        return base_storage * daily_collections

    def get_profiling_config(self) -> dict[str, Any]:
        """Get profiling configuration dictionary

        Returns:
            Dictionary with profiling parameters
        """
        return {
            "enabled": self.profiling.enabled,
            "level": self.profiling.level.value,
            "sampling_rate": self.profiling.sampling_rate,
            "track_tokens": self.profiling.track_tokens,
            "track_costs": self.profiling.track_costs,
            "track_memory": self.profiling.track_memory,
            "track_network_timing": self.profiling.track_network_timing,
            "track_model_performance": self.profiling.track_model_performance,
        }

    def get_alert_thresholds(self) -> dict[str, Any]:
        """Get alert threshold configuration

        Returns:
            Dictionary with alert thresholds
        """
        return {
            "cost_usd": self.profiling.cost_alert_threshold_usd,
            "latency_seconds": self.profiling.latency_alert_threshold_s,
            "memory_mb": self.profiling.memory_alert_threshold_mb,
            "error_rate": self.profiling.error_rate_alert_threshold,
        }

    def get_retention_config(self) -> dict[str, Any]:
        """Get data retention configuration

        Returns:
            Dictionary with retention parameters
        """
        return {
            "metrics_days": self.metrics.retention_days,
            "profiling_entries": self.profiling.max_history_size,
            "storage_path": self.monitoring_storage_path,
        }

    def should_profile_call(
        self,
        is_error: bool = False,
        latency_seconds: float | None = None,
        cost_usd: float | None = None,
        is_circuit_breaker_event: bool = False,
    ) -> bool:
        """Determine if a call should be profiled based on conditions

        Args:
            is_error: Whether the call resulted in an error
            latency_seconds: Call latency in seconds
            cost_usd: Call cost in USD
            is_circuit_breaker_event: Whether this is a circuit breaker event

        Returns:
            True if the call should be profiled
        """
        if not self.profiling.enabled:
            return False

        # Always profile certain conditions
        if is_error and self.profiling.always_profile_errors:
            return True

        if is_circuit_breaker_event and self.profiling.always_profile_circuit_breaker:
            return True

        if (
            latency_seconds
            and latency_seconds > self.profiling.latency_alert_threshold_s
        ):
            if self.profiling.always_profile_slow_calls:
                return True

        if cost_usd and cost_usd > (
            self.profiling.cost_alert_threshold_usd / 1000
        ):  # Per-call threshold
            if self.profiling.always_profile_expensive_calls:
                return True

        # Regular sampling
        import random

        return random.random() < self.profiling.sampling_rate
