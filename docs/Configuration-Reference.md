# Context Switcher MCP Configuration Reference

This document provides comprehensive reference documentation for the unified configuration system.

## Quick Start

### Basic Usage

```python
from context_switcher_mcp.config import get_config

# Get default configuration
config = get_config()

# Access configuration domains
server_port = config.server.port
model_tokens = config.models.default_max_tokens
session_ttl = config.session.default_ttl_hours
```

### Environment-Specific Configuration

```python
# Development configuration (debug logging, detailed profiling)
dev_config = get_config(environment="development")

# Staging configuration (production-like with enhanced monitoring)  
staging_config = get_config(environment="staging")

# Production configuration (secure defaults, optimized performance)
prod_config = get_config(environment="production")
```

### Configuration Files

```python
# Load from JSON file
config = get_config(config_file="config.json")

# Load from YAML file  
config = get_config(config_file="config.yaml")

# Load with environment overrides
config = get_config(config_file="base.json", environment="production")
```

## Configuration Domains

### Models Domain

Controls LLM backend configuration and behavior.

```python
models = config.models

# Token and generation limits
models.default_max_tokens     # Default: 2048
models.default_temperature    # Default: 0.7  
models.max_chars_opus         # Default: 20000
models.max_chars_haiku        # Default: 180000

# Backend configuration
models.enabled_backends       # Default: ["bedrock", "litellm", "ollama"]
models.bedrock_model_id       # Default: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
models.litellm_model          # Default: "gpt-4"
models.ollama_model           # Default: "llama3.2"
models.ollama_host            # Default: "http://localhost:11434"

# Circuit breaker settings
models.circuit_breaker_failure_threshold  # Default: 5
models.circuit_breaker_timeout_seconds    # Default: 300

# Retry configuration  
models.max_retries            # Default: 3
models.retry_delay_seconds    # Default: 1.0
models.retry_backoff_factor   # Default: 2.0

# Backend-specific methods
backend_config = models.get_backend_config("bedrock")
is_enabled = models.is_backend_enabled("ollama")
char_limit = models.get_max_chars_for_model("claude-3-opus")
```

#### Environment Variables

```bash
# General model settings
export CS_MAX_TOKENS=4096
export CS_TEMPERATURE=0.8
export CS_MAX_CHARS_OPUS=25000
export CS_MAX_CHARS_HAIKU=200000

# Backend settings
export BEDROCK_MODEL_ID="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
export LITELLM_MODEL="gpt-4-turbo"
export OLLAMA_MODEL="llama3.1"
export OLLAMA_HOST="http://ollama.example.com:11434"

# Circuit breaker and retry
export CS_MODEL_CIRCUIT_FAILURE_THRESHOLD=3
export CS_MODEL_CIRCUIT_TIMEOUT=600
export CS_MODEL_MAX_RETRIES=5
```

### Session Domain

Manages session lifecycle, limits, and validation.

```python
session = config.session

# Session lifecycle
session.default_ttl_hours             # Default: 24
session.max_ttl_hours                 # Default: 168 (1 week)
session.cleanup_interval_seconds      # Default: 600 (10 minutes)

# Session limits
session.max_active_sessions           # Default: 1000
session.max_sessions_per_client       # Default: 10
session.max_concurrent_perspectives   # Default: 10
session.max_concurrent_sessions       # Default: 100

# Operation timeouts
session.session_operation_timeout_seconds      # Default: 30.0
session.perspective_analysis_timeout_seconds   # Default: 120.0  
session.synthesis_timeout_seconds              # Default: 60.0

# Memory and storage
session.max_session_memory_mb         # Default: 100.0
session.max_session_history_entries   # Default: 100
session.enable_session_compression    # Default: True
session.session_storage_path          # Default: None (memory only)

# Input validation limits
session.max_session_id_length         # Default: 100
session.max_topic_length              # Default: 1000
session.max_perspective_name_length   # Default: 100  
session.max_custom_prompt_length      # Default: 10000

# Utility methods
is_valid_ttl = session.is_ttl_valid(12)
memory_bytes = session.get_session_memory_limit_bytes()
timeouts = session.get_operation_timeouts()
```

#### Environment Variables

```bash
# Session lifecycle
export CS_SESSION_TTL_HOURS=48
export CS_SESSION_MAX_TTL_HOURS=336  # 2 weeks
export CS_CLEANUP_INTERVAL=300

# Session limits  
export CS_MAX_SESSIONS=2000
export CS_MAX_SESSIONS_PER_CLIENT=20
export CS_MAX_CONCURRENT_PERSPECTIVES=15

# Timeouts
export CS_SESSION_OPERATION_TIMEOUT=45
export CS_PERSPECTIVE_ANALYSIS_TIMEOUT=180
export CS_SYNTHESIS_TIMEOUT=90

# Storage and memory
export CS_SESSION_MAX_MEMORY_MB=200
export CS_SESSION_STORAGE_PATH="/var/lib/context-switcher/sessions"
```

### Security Domain

Controls security, encryption, and access control.

```python
security = config.security

# Encryption and secrets
security.secret_key                   # Default: None (must be set)
security.session_secret_key           # Default: None

# Client authentication
security.enable_client_binding        # Default: True
security.client_binding_entropy_bytes # Default: 32
security.signature_iterations         # Default: 600000

# Rate limiting
security.enable_rate_limiting         # Default: True  
security.rate_limit_requests_per_minute  # Default: 60
security.rate_limit_burst_size        # Default: 10
security.rate_limit_window_seconds    # Default: 60

# Input validation
security.max_input_length             # Default: 1000000 (1MB)
security.enable_input_sanitization    # Default: True
security.blocked_patterns             # Default: XSS protection patterns

# Access control
security.max_validation_failures      # Default: 3
security.enable_suspicious_activity_detection  # Default: True
security.suspicious_activity_threshold        # Default: 5

# Security monitoring
security.enable_security_logging      # Default: True
security.security_log_level           # Default: "INFO"
security.enable_security_alerts       # Default: True

# Configuration methods
rate_config = security.get_rate_limit_config()
binding_config = security.get_client_binding_config()  
input_config = security.get_input_validation_config()
is_valid_length = security.validate_input_length(text)
blocked = security.check_blocked_patterns(text)
```

#### Environment Variables

```bash
# Encryption (REQUIRED for production)
export CONTEXT_SWITCHER_SECRET_KEY="your-64-character-base64-encoded-secret-key"
export CS_SESSION_SECRET_KEY="session-specific-secret-key"

# Client binding
export CS_ENABLE_CLIENT_BINDING=true
export CS_CLIENT_BINDING_ENTROPY=64
export CS_SIGNATURE_ITERATIONS=1000000

# Rate limiting
export CS_ENABLE_RATE_LIMITING=true
export CS_RATE_LIMIT_RPM=120
export CS_RATE_LIMIT_BURST=20

# Security monitoring
export CS_ENABLE_SECURITY_LOGGING=true  
export CS_SECURITY_LOG_LEVEL=WARNING
export CS_ENABLE_SECURITY_ALERTS=true
```

### Server Domain

Configures MCP server networking and behavior.

```python
server = config.server

# Network settings
server.host                           # Default: "localhost"
server.port                          # Default: 3023  
server.bind_address                  # Computed: "host:port"

# Logging
server.log_level                     # Default: LogLevel.INFO
server.log_format                    # Default: "structured"  
server.enable_access_logging         # Default: True
server.log_file_path                 # Default: None (console)

# Connection settings
server.max_concurrent_connections    # Default: 100
server.connection_timeout_seconds    # Default: 60.0
server.request_timeout_seconds       # Default: 300.0
server.keepalive_timeout_seconds     # Default: 30.0

# Development features  
server.enable_debug_mode             # Default: False
server.enable_hot_reload             # Default: False
server.enable_cors                   # Default: False

# Monitoring endpoints
server.enable_health_endpoint        # Default: True
server.enable_metrics_endpoint       # Default: True  
server.enable_status_endpoint        # Default: True

# Performance tuning
server.worker_threads                # Default: 4
server.max_request_size_mb           # Default: 10
server.enable_compression            # Default: True

# Configuration methods
log_config = server.get_log_config()
conn_config = server.get_connection_config()
perf_config = server.get_performance_config()
endpoints = server.get_monitoring_endpoints()

# Environment detection
is_dev = server.is_development_mode
is_prod_ready = server.is_production_ready
is_secure = server.is_secure_deployment()
```

#### Environment Variables

```bash
# Network settings
export CS_HOST=0.0.0.0              # Bind to all interfaces
export CS_PORT=8080                 # Custom port

# Logging
export CS_LOG_LEVEL=WARNING         # Production logging
export CS_LOG_FORMAT=json           # Structured logging
export CS_LOG_FILE_PATH=/var/log/context-switcher.log

# Connection limits
export CS_MAX_CONCURRENT_CONNECTIONS=500
export CS_CONNECTION_TIMEOUT=30
export CS_REQUEST_TIMEOUT=180

# Development (disable in production)
export CS_ENABLE_DEBUG_MODE=false
export CS_ENABLE_HOT_RELOAD=false  
export CS_ENABLE_CORS=false

# Performance
export CS_WORKER_THREADS=8
export CS_MAX_REQUEST_SIZE_MB=5
export CS_ENABLE_COMPRESSION=true
```

### Monitoring Domain

Controls profiling, metrics, and observability.

```python
monitoring = config.monitoring

# General monitoring
monitoring.enable_monitoring          # Default: True
monitoring.enable_real_time_metrics   # Default: True
monitoring.metrics_export_format      # Default: "json"

# Metrics collection
metrics = monitoring.metrics
metrics.max_history_size              # Default: 1000
metrics.retention_days                # Default: 7
metrics.collection_interval_seconds   # Default: 60

# Profiling configuration
profiling = monitoring.profiling  
profiling.enabled                     # Default: True
profiling.level                       # Default: ProfilingLevel.STANDARD
profiling.sampling_rate               # Default: 0.1 (10%)

# Feature flags
profiling.track_tokens                # Default: True
profiling.track_costs                 # Default: True
profiling.track_memory                # Default: False (expensive)
profiling.track_network_timing        # Default: True

# Alert thresholds
profiling.cost_alert_threshold_usd    # Default: 100.0
profiling.latency_alert_threshold_s   # Default: 30.0
profiling.memory_alert_threshold_mb   # Default: 1000.0
profiling.error_rate_alert_threshold  # Default: 0.1 (10%)

# Conditional profiling rules
profiling.always_profile_errors       # Default: True
profiling.always_profile_slow_calls   # Default: True
profiling.always_profile_expensive_calls    # Default: True
profiling.always_profile_circuit_breaker    # Default: True

# Alerting
alerting = monitoring.alerting
alerting.enabled                      # Default: True
alerting.alert_cooldown_minutes       # Default: 15
alerting.enable_cost_alerts           # Default: True

# Configuration methods
prof_config = monitoring.get_profiling_config()
thresholds = monitoring.get_alert_thresholds() 
retention = monitoring.get_retention_config()
should_profile = monitoring.should_profile_call(is_error=True)
dashboard = monitoring.get_dashboard_config()
```

#### Environment Variables

```bash
# General monitoring
export CS_ENABLE_MONITORING=true
export CS_ENABLE_REAL_TIME_METRICS=true
export CS_MONITORING_STORAGE_PATH=/var/lib/context-switcher/monitoring

# Profiling  
export CS_PROFILING_ENABLED=true
export CS_PROFILING_LEVEL=standard   # disabled, basic, standard, detailed
export CS_PROFILING_SAMPLING_RATE=0.2

# Feature tracking
export CS_PROFILING_TRACK_TOKENS=true
export CS_PROFILING_TRACK_COSTS=true
export CS_PROFILING_TRACK_MEMORY=false
export CS_PROFILING_TRACK_NETWORK=true

# Alert thresholds
export CS_PROFILING_COST_ALERT=200.0
export CS_PROFILING_LATENCY_ALERT=15.0
export CS_PROFILING_MEMORY_ALERT=500.0

# Dashboard and reporting
export CS_ENABLE_PERFORMANCE_DASHBOARD=true
export CS_DASHBOARD_UPDATE_INTERVAL=60
export CS_ENABLE_AUTOMATED_REPORTS=true
```

## Environment Configurations

### Development Environment

Optimized for local development with debugging features:

```python
dev_config = get_config(environment="development")

# Key characteristics:
# - DEBUG logging enabled
# - Detailed profiling (100% sampling)
# - Relaxed security settings
# - Hot reload and CORS enabled
# - Local service endpoints (localhost)
# - Short session TTL (2 hours)
# - Enhanced error reporting
```

### Staging Environment

Production-like settings with enhanced monitoring:

```python
staging_config = get_config(environment="staging")

# Key characteristics:
# - INFO level logging  
# - Standard profiling (30% sampling)
# - Production-like security
# - Enhanced monitoring and alerting
# - 12-hour session TTL
# - Real-time metrics enabled
# - Comprehensive validation
```

### Production Environment

Secure, optimized settings for production deployment:

```python
prod_config = get_config(environment="production")

# Key characteristics:
# - WARNING level logging (minimal overhead)
# - Basic profiling (5% sampling)  
# - Maximum security settings
# - Optimized performance settings
# - 24-hour session TTL
# - Monitoring with minimal overhead
# - Strict validation and limits
```

## Configuration Files

### JSON Configuration

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "log_level": "INFO"
  },
  "models": {
    "default_max_tokens": 4096,
    "enabled_backends": ["bedrock", "litellm"]
  },
  "security": {
    "rate_limit_requests_per_minute": 120,
    "enable_client_binding": true
  },
  "monitoring": {
    "profiling": {
      "enabled": true,
      "level": "standard",
      "sampling_rate": 0.2
    }
  }
}
```

### YAML Configuration

```yaml
server:
  host: 0.0.0.0
  port: 8080
  log_level: INFO

models:
  default_max_tokens: 4096
  enabled_backends:
    - bedrock
    - litellm

security:
  rate_limit_requests_per_minute: 120
  enable_client_binding: true

monitoring:
  profiling:
    enabled: true
    level: standard
    sampling_rate: 0.2
```

## Validation and Error Handling

### Configuration Validation

```python
# Automatic validation on creation
config = get_config()  # Validates all settings

# Check production readiness
if config.is_production_ready:
    print("Configuration is production-ready")
else:
    print("Configuration needs updates for production")

# Validate external dependencies
warnings = config.validate_external_dependencies()
for warning in warnings:
    print(f"Dependency check: {warning}")
```

### Error Handling

```python
from context_switcher_mcp.config import ConfigurationError

try:
    config = get_config(config_file="invalid.json")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Error message includes specific validation failures
```

### Common Validation Errors

1. **Invalid Port Numbers**: Must be 1024-65535
2. **Invalid Temperature**: Must be 0.0-2.0 for model temperature
3. **Missing Secret Key**: Required for production deployment
4. **Invalid Log Level**: Must be DEBUG/INFO/WARNING/ERROR/CRITICAL
5. **Invalid Backend Names**: Must be in ["bedrock", "litellm", "ollama"]
6. **Invalid File Format**: Config files must be JSON or YAML

## Migration Guide

### From Legacy Configuration

```python
# OLD: Direct config access (still works with deprecation warning)
from context_switcher_mcp.config import config
port = config.server.port

# NEW: Recommended approach
from context_switcher_mcp.config import get_config
config = get_config()
port = config.server.port

# OLD: Module-level imports (deprecated)  
from context_switcher_mcp.config import ContextSwitcherConfig

# NEW: Factory function approach
from context_switcher_mcp.config import get_config
config = get_config()
```

### Legacy Compatibility

The system provides full backward compatibility:

```python
# All these patterns continue to work:
from context_switcher_mcp.config import config       # ⚠ Deprecated
from context_switcher_mcp.config import get_config   # ✅ Recommended

# Legacy attribute access still works:  
config.model.bedrock_model_id        # Legacy adapter
config.models.bedrock_model_id       # New unified system
```

## Best Practices

### Production Deployment

1. **Set Required Environment Variables**:
   ```bash
   export CONTEXT_SWITCHER_SECRET_KEY="your-secure-64-char-key"
   export CS_LOG_LEVEL=WARNING
   export CS_PROFILING_LEVEL=basic
   ```

2. **Use Configuration Files**:
   ```python
   prod_config = get_config(
       environment="production",
       config_file="/etc/context-switcher/config.yaml"
   )
   ```

3. **Validate Before Deployment**:
   ```python
   if not config.is_production_ready:
       raise ValueError("Configuration not production-ready")
   ```

### Development Workflow

1. **Use Environment-Specific Configs**:
   ```python
   dev_config = get_config(environment="development")
   ```

2. **Override for Testing**:
   ```python
   test_config = get_config(
       config_file="test_config.json",
       reload=True
   )
   ```

3. **Validate Configuration Changes**:
   ```python
   dependencies = config.validate_external_dependencies()
   ```

### Security Considerations

1. **Never commit secrets** to version control
2. **Use environment variables** for sensitive data
3. **Validate secret key length** (minimum 32 characters)
4. **Enable rate limiting** in production
5. **Use secure log levels** (WARNING+ for production)
6. **Enable client binding** for multi-tenant deployments

---

For additional help, see the [ADR-001 Unified Configuration System](./ADR-001-Unified-Configuration-System.md) document.