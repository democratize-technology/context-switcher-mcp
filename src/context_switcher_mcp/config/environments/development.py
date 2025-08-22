"""Development environment configuration

Optimized for local development with:
- Debug logging enabled
- Detailed profiling and monitoring
- Relaxed security settings
- Fast iteration and debugging features
- Local service endpoints
"""

from typing import Any

from .base import BaseEnvironmentConfig


class DevelopmentConfig(BaseEnvironmentConfig):
    """Configuration preset for development environment

    This configuration is optimized for local development work with
    extensive logging, debugging features, and relaxed security to
    facilitate rapid iteration and troubleshooting.
    """

    @property
    def environment_name(self) -> str:
        return "development"

    @property
    def is_production_ready(self) -> bool:
        return False

    def get_config_dict(self) -> dict[str, Any]:
        """Get development configuration dictionary"""
        return {
            "server": self.get_server_config(),
            "models": self.get_models_config(),
            "session": self.get_session_config(),
            "security": self.get_security_config(),
            "monitoring": self.get_monitoring_config(),
        }

    def get_server_config(self) -> dict[str, Any]:
        """Development server configuration"""
        return {
            "host": "localhost",  # Bind to localhost only
            "port": 3023,
            "log_level": "DEBUG",  # Verbose logging
            "log_format": "structured",
            "enable_debug_mode": True,
            "enable_hot_reload": True,  # Hot reload for development
            "enable_cors": True,  # Allow cross-origin requests
            "cors_allowed_origins": ["*"],  # Allow all origins in dev
            "enable_health_endpoint": True,
            "enable_metrics_endpoint": True,
            "enable_status_endpoint": True,
            "max_concurrent_connections": 50,  # Moderate limit for dev
            "connection_timeout_seconds": 300.0,  # Longer timeout for debugging
            "request_timeout_seconds": 600.0,  # Very long timeout for debugging
        }

    def get_models_config(self) -> dict[str, Any]:
        """Development models configuration"""
        return {
            "default_max_tokens": 1024,  # Smaller tokens for faster dev cycles
            "default_temperature": 0.8,  # Slightly higher temperature for variety
            # Use local/dev-friendly backends
            "enabled_backends": ["ollama", "litellm", "bedrock"],
            # Ollama for local development
            "ollama_host": "http://localhost:11434",
            "ollama_model": "llama3.2",
            "ollama_timeout_seconds": 120.0,  # Generous timeout for local
            # Development circuit breaker settings
            "circuit_breaker_failure_threshold": 10,  # More forgiving
            "circuit_breaker_timeout_seconds": 60,  # Shorter recovery time
            # Development retry settings
            "max_retries": 2,  # Fewer retries for faster feedback
            "retry_delay_seconds": 0.5,
            "retry_backoff_factor": 1.5,
        }

    def get_session_config(self) -> dict[str, Any]:
        """Development session configuration"""
        return {
            "default_ttl_hours": 2,  # Short TTL for development
            "max_ttl_hours": 8,  # Max 8 hours for dev sessions
            "cleanup_interval_seconds": 300,  # Clean up every 5 minutes
            "max_active_sessions": 20,  # Lower limit for dev environment
            "max_sessions_per_client": 5,
            # Development timeouts
            "session_operation_timeout_seconds": 60.0,
            "perspective_analysis_timeout_seconds": 180.0,  # Longer for debugging
            "synthesis_timeout_seconds": 90.0,
            # Memory and resources
            "max_session_memory_mb": 50.0,  # Conservative for dev
            "max_session_history_entries": 50,
            "enable_session_compression": False,  # Disable for easier debugging
            # Concurrent operations
            "max_concurrent_perspectives": 5,  # Lower for development
            "max_concurrent_sessions": 10,
            # Session persistence
            "enable_session_persistence": True,
            "session_storage_path": "./dev_sessions",  # Local storage
        }

    def get_security_config(self) -> dict[str, Any]:
        """Development security configuration (relaxed)"""
        return {
            # Relaxed security for development
            "secret_key": None,  # Allow running without secret key
            "enable_client_binding": False,  # Disable for easier testing
            # Rate limiting (generous for development)
            "enable_rate_limiting": True,
            "rate_limit_requests_per_minute": 300,  # High limit
            "rate_limit_burst_size": 50,
            "rate_limit_window_seconds": 60,
            # Input validation (enabled but permissive)
            "max_input_length": 10000000,  # 10MB for dev testing
            "enable_input_sanitization": True,
            "blocked_patterns": [
                r"<script[^>]*>.*?</script>",  # Keep basic XSS protection
            ],
            # Access control (relaxed)
            "max_validation_failures": 10,  # More forgiving
            "validation_failure_window_seconds": 600,  # Longer window
            "enable_suspicious_activity_detection": False,  # Disable for dev
            # Security monitoring (minimal)
            "enable_security_logging": True,
            "security_log_level": "DEBUG",
            "enable_security_alerts": False,  # No alerts in dev
            # Session security (relaxed)
            "session_timeout_minutes": 240,  # 4 hours
            "enable_session_rotation": False,  # Disable rotation
        }

    def get_monitoring_config(self) -> dict[str, Any]:
        """Development monitoring configuration (detailed)"""
        return {
            "enable_monitoring": True,
            "enable_real_time_metrics": True,
            "metrics_export_format": "json",
            "monitoring_storage_path": "./dev_monitoring",
            # Metrics configuration
            "metrics": {
                "max_history_size": 1000,
                "retention_days": 1,  # Short retention for dev
                "collection_interval_seconds": 30,  # Frequent collection
                "enable_system_metrics": True,
                "enable_application_metrics": True,
            },
            # Profiling configuration (detailed for debugging)
            "profiling": {
                "enabled": True,
                "level": "detailed",  # Maximum detail for debugging
                "sampling_rate": 1.0,  # Profile everything in dev
                # Track everything for development debugging
                "track_tokens": True,
                "track_costs": True,
                "track_memory": True,
                "track_network_timing": True,
                "track_model_performance": True,
                # Storage
                "max_history_size": 1000,
                # Alert thresholds (high for development)
                "cost_alert_threshold_usd": 10.0,  # Low threshold for dev
                "latency_alert_threshold_s": 60.0,
                "memory_alert_threshold_mb": 500.0,
                "error_rate_alert_threshold": 0.5,  # 50% error rate
                # Always profile everything in development
                "always_profile_errors": True,
                "always_profile_slow_calls": True,
                "always_profile_expensive_calls": True,
                "always_profile_circuit_breaker": True,
            },
            # Alerting (disabled for development)
            "alerting": {
                "enabled": False,  # No alerts in development
                "alert_cooldown_minutes": 5,
                "enable_cost_alerts": False,
                "enable_performance_alerts": False,
                "enable_error_alerts": False,
                "enable_security_alerts": False,
            },
            # Dashboard and reporting
            "enable_performance_dashboard": True,
            "dashboard_update_interval_seconds": 10,  # Fast updates
            "enable_automated_reports": False,
        }

    def validate_environment_requirements(self) -> list[str]:
        """Validate development environment requirements"""
        warnings = []

        # Check for development tools
        try:
            import yaml

            warnings.append("✓ PyYAML available for configuration files")
        except ImportError:
            warnings.append("⚠ PyYAML not installed - YAML configs unavailable")

        # Check for Ollama (common in development)
        import os

        if not os.getenv("OLLAMA_HOST"):
            warnings.append("ℹ OLLAMA_HOST not set - using default localhost:11434")

        # Check for development directories
        if not os.path.exists("./dev_sessions"):
            warnings.append("ℹ Development session storage directory will be created")

        if not os.path.exists("./dev_monitoring"):
            warnings.append(
                "ℹ Development monitoring storage directory will be created"
            )

        return warnings
