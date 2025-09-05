"""Production environment configuration

Secure, optimized configuration for production deployment with:
- Security-first defaults
- Performance optimizations
- Minimal logging and monitoring overhead
- Resource limits for stability
- Circuit breakers and resilience patterns
"""

from typing import Any

from .base import BaseEnvironmentConfig


class ProductionConfig(BaseEnvironmentConfig):
    """Configuration preset for production environment

    This configuration prioritizes security, performance, and stability
    for production deployments. All settings are optimized for high
    availability and minimal resource consumption.
    """

    @property
    def environment_name(self) -> str:
        return "production"

    @property
    def is_production_ready(self) -> bool:
        return True

    def get_config_dict(self) -> dict[str, Any]:
        """Get production configuration dictionary"""
        return {
            "server": self.get_server_config(),
            "models": self.get_models_config(),
            "session": self.get_session_config(),
            "security": self.get_security_config(),
            "monitoring": self.get_monitoring_config(),
        }

    def get_server_config(self) -> dict[str, Any]:
        """Production server configuration"""
        return {
            "host": "0.0.0.0",  # Bind to all interfaces
            "port": 3023,
            "log_level": "WARNING",  # Minimal logging for performance
            "log_format": "json",  # Structured logging for analysis
            "enable_debug_mode": False,
            "enable_hot_reload": False,
            "enable_cors": False,  # Disable CORS for security
            "cors_allowed_origins": [],
            "enable_health_endpoint": True,  # Keep for monitoring
            "enable_metrics_endpoint": False,  # Disable for security
            "enable_status_endpoint": False,  # Disable for security
            "max_concurrent_connections": 1000,  # High capacity
            "connection_timeout_seconds": 30.0,  # Fast timeout
            "request_timeout_seconds": 120.0,  # 2 minutes max
            "keepalive_timeout_seconds": 15.0,  # Short keepalive
            # Performance optimizations
            "worker_threads": 8,
            "max_request_size_mb": 5,  # Limit request size
            "enable_compression": True,
            "compression_threshold_bytes": 1024,
        }

    def get_models_config(self) -> dict[str, Any]:
        """Production models configuration"""
        return {
            "default_max_tokens": 2048,
            "default_temperature": 0.7,
            # Conservative backend selection
            "enabled_backends": ["bedrock", "litellm"],  # Exclude Ollama for prod
            # Production timeouts (conservative)
            "bedrock_timeout_seconds": 30.0,
            "litellm_timeout_seconds": 30.0,
            "ollama_timeout_seconds": 45.0,  # If enabled
            # Aggressive circuit breaker
            "circuit_breaker_failure_threshold": 3,  # Fail fast
            "circuit_breaker_timeout_seconds": 600,  # 10 minutes
            # Conservative retry settings
            "max_retries": 2,  # Fewer retries
            "retry_delay_seconds": 2.0,  # Longer delay
            "retry_backoff_factor": 3.0,  # Aggressive backoff
        }

    def get_session_config(self) -> dict[str, Any]:
        """Production session configuration"""
        return {
            "default_ttl_hours": 24,  # Standard TTL
            "max_ttl_hours": 72,  # Max 3 days
            "cleanup_interval_seconds": 300,  # 5 minutes
            "max_active_sessions": 2000,  # High capacity
            "max_sessions_per_client": 5,  # Prevent abuse
            # Conservative timeouts
            "session_operation_timeout_seconds": 20.0,
            "perspective_analysis_timeout_seconds": 90.0,
            "synthesis_timeout_seconds": 45.0,
            # Memory and resources (conservative)
            "max_session_memory_mb": 50.0,
            "max_session_history_entries": 50,
            "enable_session_compression": True,  # Save memory
            # Concurrent operations (limited)
            "max_concurrent_perspectives": 6,
            "max_concurrent_sessions": 100,
            # Session persistence
            "enable_session_persistence": True,
            "session_storage_path": "/var/lib/context-switcher/sessions",
        }

    def get_security_config(self) -> dict[str, Any]:
        """Production security configuration (maximum security)"""
        return {
            # Strong security requirements (secret_key loaded from environment variable)
            "enable_client_binding": True,
            "client_binding_entropy_bytes": 64,  # Higher entropy
            "signature_iterations": 1000000,  # More iterations
            # Strict rate limiting
            "enable_rate_limiting": True,
            "rate_limit_requests_per_minute": 60,  # Conservative limit
            "rate_limit_burst_size": 5,  # Small burst
            "rate_limit_window_seconds": 60,
            # Strict input validation
            "max_input_length": 100000,  # 100KB limit
            "enable_input_sanitization": True,
            "blocked_patterns": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"data:",
                r"vbscript:",
                r"on\w+\s*=",  # Event handlers
                r"<iframe[^>]*>",  # Iframes
                r"<object[^>]*>",  # Objects
                r"<embed[^>]*>",  # Embeds
            ],
            # Strict access control
            "max_validation_failures": 3,
            "validation_failure_window_seconds": 300,
            "enable_suspicious_activity_detection": True,
            "suspicious_activity_threshold": 3,  # Low threshold
            # Security monitoring
            "enable_security_logging": True,
            "security_log_level": "WARNING",  # Only warnings and errors
            "enable_security_alerts": True,
            # Session security
            "session_timeout_minutes": 60,  # 1 hour
            "enable_session_rotation": True,
            "session_rotation_interval_hours": 8,  # Frequent rotation
        }

    def get_monitoring_config(self) -> dict[str, Any]:
        """Production monitoring configuration (minimal overhead)"""
        return {
            "enable_monitoring": True,
            "enable_real_time_metrics": False,  # Reduce overhead
            "metrics_export_format": "json",
            "monitoring_storage_path": "/var/lib/context-switcher/monitoring",
            # Metrics configuration (minimal)
            "metrics": {
                "max_history_size": 1000,  # Limited history
                "retention_days": 7,
                "collection_interval_seconds": 300,  # 5 minutes
                "enable_system_metrics": False,  # Disable for performance
                "enable_application_metrics": True,
            },
            # Profiling configuration (basic)
            "profiling": {
                "enabled": True,
                "level": "basic",  # Minimal profiling
                "sampling_rate": 0.05,  # 5% sampling only
                # Track essential metrics only
                "track_tokens": True,
                "track_costs": True,
                "track_memory": False,  # Expensive, disable
                "track_network_timing": False,  # Disable for performance
                "track_model_performance": True,
                # Limited storage
                "max_history_size": 5000,
                # Production alert thresholds
                "cost_alert_threshold_usd": 200.0,
                "latency_alert_threshold_s": 15.0,  # Strict latency
                "memory_alert_threshold_mb": 500.0,
                "error_rate_alert_threshold": 0.05,  # 5% error rate
                # Selective profiling
                "always_profile_errors": True,
                "always_profile_slow_calls": False,  # Reduce overhead
                "always_profile_expensive_calls": True,
                "always_profile_circuit_breaker": True,
            },
            # Alerting (critical only)
            "alerting": {
                "enabled": True,
                "alert_cooldown_minutes": 30,  # Longer cooldown
                "enable_cost_alerts": True,
                "enable_performance_alerts": True,
                "enable_error_alerts": True,
                "enable_security_alerts": True,
            },
            # Dashboard and reporting (minimal)
            "enable_performance_dashboard": False,  # Disable for security
            "dashboard_update_interval_seconds": 300,
            "enable_automated_reports": True,
            "report_generation_interval_hours": 24,  # Daily reports
        }

    def validate_environment_requirements(self) -> list[str]:
        """Validate production environment requirements"""
        warnings = []

        import os

        # Check critical environment variables
        critical_env_vars = [
            "CONTEXT_SWITCHER_SECRET_KEY",
        ]

        for var in critical_env_vars:
            if not os.getenv(var):
                warnings.append(
                    f"❌ CRITICAL: Required environment variable not set: {var}"
                )
            else:
                # Validate secret key strength
                if var == "CONTEXT_SWITCHER_SECRET_KEY":
                    key = os.getenv(var)
                    if len(key) < 64:
                        warnings.append(
                            "⚠ Secret key should be at least 64 characters for production"
                        )

        # Check backend configuration
        backend_warnings = []

        if not (os.getenv("AWS_PROFILE") or os.getenv("AWS_ACCESS_KEY_ID")):
            backend_warnings.append("Bedrock")

        if not os.getenv("LITELLM_API_KEY"):
            backend_warnings.append("LiteLLM")

        if backend_warnings:
            warnings.append(
                f"❌ Backend credentials missing: {', '.join(backend_warnings)}"
            )

        # Check security settings
        if os.getenv("CS_HOST") == "localhost":
            warnings.append(
                "❌ CRITICAL: Server bound to localhost - not accessible externally"
            )

        if os.getenv("CS_LOG_LEVEL", "").upper() == "DEBUG":
            warnings.append("❌ CRITICAL: Debug logging enabled in production")

        if os.getenv("CS_ENABLE_DEBUG_MODE", "").lower() in ["true", "1", "yes"]:
            warnings.append("❌ CRITICAL: Debug mode enabled in production")

        # Check storage directories
        storage_paths = [
            "/var/lib/context-switcher/sessions",
            "/var/lib/context-switcher/monitoring",
        ]

        for path in storage_paths:
            if not os.path.exists(path):
                warnings.append(f"❌ Storage directory missing: {path}")
            else:
                # Check permissions
                if not os.access(path, os.R_OK | os.W_OK):
                    warnings.append(f"❌ Insufficient permissions for: {path}")

        # Check required dependencies
        required_deps = ["boto3", "litellm"]
        for dep in required_deps:
            try:
                __import__(dep)
                warnings.append(f"✓ {dep} available")
            except ImportError:
                warnings.append(
                    f"❌ CRITICAL: Required dependency not installed: {dep}"
                )

        # Check system resources
        try:
            import psutil

            # Check available memory
            memory = psutil.virtual_memory()
            if memory.total < 2 * 1024**3:  # Less than 2GB
                warnings.append("⚠ Low system memory - may affect performance")

            # Check disk space
            disk = psutil.disk_usage("/")
            if disk.free < 1 * 1024**3:  # Less than 1GB free
                warnings.append("⚠ Low disk space - may affect logging and sessions")

        except ImportError:
            warnings.append("ℹ psutil not available - cannot check system resources")

        return warnings
