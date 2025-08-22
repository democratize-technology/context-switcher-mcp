"""Staging environment configuration

Production-like configuration for testing and validation with:
- Production-like security and performance settings
- Enhanced monitoring and alerting for testing
- Moderate resource limits
- Comprehensive logging for validation
"""

from typing import Any

from .base import BaseEnvironmentConfig


class StagingConfig(BaseEnvironmentConfig):
    """Configuration preset for staging environment

    This configuration mirrors production settings but with enhanced
    monitoring and logging to facilitate testing and validation of
    changes before production deployment.
    """

    @property
    def environment_name(self) -> str:
        return "staging"

    @property
    def is_production_ready(self) -> bool:
        return True

    def get_config_dict(self) -> dict[str, Any]:
        """Get staging configuration dictionary"""
        return {
            "server": self.get_server_config(),
            "models": self.get_models_config(),
            "session": self.get_session_config(),
            "security": self.get_security_config(),
            "monitoring": self.get_monitoring_config(),
        }

    def get_server_config(self) -> dict[str, Any]:
        """Staging server configuration"""
        return {
            "host": "0.0.0.0",  # Bind to all interfaces for container deployment
            "port": 3023,
            "log_level": "INFO",  # Production-like logging
            "log_format": "json",  # Structured logging for analysis
            "enable_debug_mode": False,
            "enable_hot_reload": False,
            "enable_cors": True,  # May need CORS for staging tests
            "cors_allowed_origins": [],  # Explicit origins only
            "enable_health_endpoint": True,
            "enable_metrics_endpoint": True,
            "enable_status_endpoint": True,
            "max_concurrent_connections": 200,  # Production-like capacity
            "connection_timeout_seconds": 60.0,
            "request_timeout_seconds": 300.0,  # 5 minutes
        }

    def get_models_config(self) -> dict[str, Any]:
        """Staging models configuration"""
        return {
            "default_max_tokens": 2048,  # Production default
            "default_temperature": 0.7,  # Production default
            # All backends enabled for staging tests
            "enabled_backends": ["bedrock", "litellm", "ollama"],
            # Production-like timeouts
            "bedrock_timeout_seconds": 30.0,
            "litellm_timeout_seconds": 30.0,
            "ollama_timeout_seconds": 60.0,
            # Circuit breaker settings
            "circuit_breaker_failure_threshold": 5,
            "circuit_breaker_timeout_seconds": 300,  # 5 minutes
            # Retry settings
            "max_retries": 3,
            "retry_delay_seconds": 1.0,
            "retry_backoff_factor": 2.0,
        }

    def get_session_config(self) -> dict[str, Any]:
        """Staging session configuration"""
        return {
            "default_ttl_hours": 12,  # Moderate TTL for testing
            "max_ttl_hours": 48,  # Max 2 days for staging
            "cleanup_interval_seconds": 300,  # 5 minutes
            "max_active_sessions": 500,  # Moderate capacity
            "max_sessions_per_client": 10,
            # Production-like timeouts
            "session_operation_timeout_seconds": 30.0,
            "perspective_analysis_timeout_seconds": 120.0,
            "synthesis_timeout_seconds": 60.0,
            # Memory and resources
            "max_session_memory_mb": 100.0,
            "max_session_history_entries": 100,
            "enable_session_compression": True,
            # Concurrent operations
            "max_concurrent_perspectives": 8,
            "max_concurrent_sessions": 50,
            # Session persistence
            "enable_session_persistence": True,
            "session_storage_path": "/var/lib/context-switcher/sessions",
        }

    def get_security_config(self) -> dict[str, Any]:
        """Staging security configuration (production-like)"""
        return {
            # Production-like security
            "secret_key": None,  # Must be set via environment variable
            "enable_client_binding": True,
            # Rate limiting
            "enable_rate_limiting": True,
            "rate_limit_requests_per_minute": 120,  # Moderate limit
            "rate_limit_burst_size": 20,
            "rate_limit_window_seconds": 60,
            # Input validation
            "max_input_length": 1000000,  # 1MB limit
            "enable_input_sanitization": True,
            "blocked_patterns": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"data:",
                r"vbscript:",
            ],
            # Access control
            "max_validation_failures": 5,
            "validation_failure_window_seconds": 300,
            "enable_suspicious_activity_detection": True,
            "suspicious_activity_threshold": 5,
            # Security monitoring
            "enable_security_logging": True,
            "security_log_level": "INFO",
            "enable_security_alerts": True,  # Enable alerts for testing
            # Session security
            "session_timeout_minutes": 120,  # 2 hours
            "enable_session_rotation": True,
            "session_rotation_interval_hours": 12,
        }

    def get_monitoring_config(self) -> dict[str, Any]:
        """Staging monitoring configuration (enhanced for testing)"""
        return {
            "enable_monitoring": True,
            "enable_real_time_metrics": True,
            "metrics_export_format": "json",
            "monitoring_storage_path": "/var/lib/context-switcher/monitoring",
            # Metrics configuration
            "metrics": {
                "max_history_size": 5000,
                "retention_days": 7,  # Week retention
                "collection_interval_seconds": 60,
                "enable_system_metrics": True,
                "enable_application_metrics": True,
            },
            # Profiling configuration (standard with enhanced sampling)
            "profiling": {
                "enabled": True,
                "level": "standard",  # Standard level
                "sampling_rate": 0.3,  # Higher sampling for staging
                # Track key metrics
                "track_tokens": True,
                "track_costs": True,
                "track_memory": False,  # Disable expensive tracking
                "track_network_timing": True,
                "track_model_performance": True,
                # Storage
                "max_history_size": 10000,
                # Alert thresholds (staging-appropriate)
                "cost_alert_threshold_usd": 50.0,
                "latency_alert_threshold_s": 30.0,
                "memory_alert_threshold_mb": 1000.0,
                "error_rate_alert_threshold": 0.2,  # 20% error rate
                # Conditional profiling
                "always_profile_errors": True,
                "always_profile_slow_calls": True,
                "always_profile_expensive_calls": True,
                "always_profile_circuit_breaker": True,
            },
            # Alerting (enabled for testing)
            "alerting": {
                "enabled": True,
                "alert_cooldown_minutes": 15,
                "enable_cost_alerts": True,
                "enable_performance_alerts": True,
                "enable_error_alerts": True,
                "enable_security_alerts": True,
            },
            # Dashboard and reporting
            "enable_performance_dashboard": True,
            "dashboard_update_interval_seconds": 30,
            "enable_automated_reports": True,
            "report_generation_interval_hours": 12,  # Twice daily
        }

    def validate_environment_requirements(self) -> list[str]:
        """Validate staging environment requirements"""
        warnings = []

        import os

        # Check required environment variables
        required_env_vars = [
            "CONTEXT_SWITCHER_SECRET_KEY",
        ]

        for var in required_env_vars:
            if not os.getenv(var):
                warnings.append(f"❌ Required environment variable not set: {var}")

        # Check optional backend configuration
        if not os.getenv("AWS_PROFILE") and not os.getenv("AWS_ACCESS_KEY_ID"):
            warnings.append(
                "⚠ AWS credentials not configured - Bedrock backend unavailable"
            )

        if not os.getenv("LITELLM_API_KEY"):
            warnings.append("⚠ LITELLM_API_KEY not set - LiteLLM backend unavailable")

        # Check storage directories
        storage_paths = [
            "/var/lib/context-switcher/sessions",
            "/var/lib/context-switcher/monitoring",
        ]

        for path in storage_paths:
            if not os.path.exists(path):
                warnings.append(f"ℹ Storage directory will be created: {path}")

        # Check network configuration
        if os.getenv("CS_HOST") == "localhost":
            warnings.append(
                "⚠ Server bound to localhost - may not be accessible externally"
            )

        # Check for production-like dependencies
        optional_deps = ["boto3", "litellm", "yaml"]
        for dep in optional_deps:
            try:
                __import__(dep)
                warnings.append(f"✓ {dep} available")
            except ImportError:
                warnings.append(
                    f"⚠ {dep} not installed - some features may be unavailable"
                )

        return warnings
