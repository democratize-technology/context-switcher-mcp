# Configuration Validation System

This document describes the comprehensive configuration validation system that eliminates runtime failures from invalid configuration parameters.

## Overview

The Context Switcher MCP server now includes a robust configuration validation system that:

- ✅ **Validates all 60+ configuration parameters** with proper type checking, range validation, and pattern matching
- ✅ **Provides fail-fast loading** with clear, actionable error messages
- ✅ **Maintains backward compatibility** while adding comprehensive validation
- ✅ **Supports configuration files** (JSON/YAML) in addition to environment variables
- ✅ **Includes security-aware validation** with sensitive data masking
- ✅ **Offers production readiness assessment** with built-in best practices

## Architecture

### Core Components

1. **`validated_config.py`** - Pydantic-based configuration models with comprehensive validation
2. **`config_migration.py`** - Backward compatibility layer and migration utilities
3. **`config.py`** - Updated main configuration module with integrated validation
4. **`config_validator.py`** - CLI tools and validation utilities
5. **Configuration Templates** - Example configuration files for different environments

### Validation Features

#### Type Safety and Range Validation
```python
default_max_tokens: int = Field(
    default=2048,
    ge=1,           # Greater than or equal to 1
    le=200000,      # Less than or equal to 200,000
    description="Default maximum tokens for model responses"
)
```

#### String Pattern Validation
```python
bedrock_model_id: str = Field(
    pattern=r"^[a-z0-9\.\-:]+$",  # Valid characters
    description="AWS Bedrock model identifier"
)

@field_validator("bedrock_model_id")
def validate_bedrock_model_id(cls, v: str) -> str:
    if not re.match(r"^[a-z]{2}\.[a-z]+\.[a-z0-9\-]+:[0-9]+$", v):
        raise ValueError("Must be in format: region.provider.model:version")
    return v
```

#### URL and Network Validation
```python
ollama_host: HttpUrl = Field(
    default="http://localhost:11434",
    description="Ollama service URL"
)
```

#### Enum Validation
```python
class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

log_level: LogLevel = Field(
    default=LogLevel.INFO,
    description="Logging level"
)
```

## Configuration Parameters

### Model Configuration
- **default_max_tokens** (1-200000): Maximum tokens for model responses
- **default_temperature** (0.0-2.0): Temperature for model responses
- **max_chars_opus** (1000-1000000): Character limit for Claude Opus
- **max_chars_haiku** (1000-2000000): Character limit for Claude Haiku
- **bedrock_model_id**: AWS Bedrock model identifier with format validation
- **litellm_model**: LiteLLM model identifier
- **ollama_model**: Ollama model identifier with pattern validation
- **ollama_host**: Ollama service URL with HTTP/HTTPS validation

### Server Configuration
- **host**: Server bind address with IP validation
- **port** (1024-65535): Server port (non-privileged range)
- **log_level**: Logging level (enum validation)

### Circuit Breaker Configuration
- **failure_threshold** (1-100): Failures before opening circuit
- **timeout_seconds** (10-3600): Timeout before reset attempt

### Session Management
- **default_ttl_hours** (1-168): Session TTL in hours (max 1 week)
- **cleanup_interval_seconds** (60-3600): Cleanup interval
- **max_active_sessions** (1-100000): Maximum concurrent sessions

### Retry Configuration
- **max_retries** (0-20): Maximum retry attempts
- **initial_delay** (0.1-60.0): Initial retry delay
- **backoff_factor** (1.0-10.0): Exponential backoff multiplier
- **max_delay** (1.0-600.0): Maximum retry delay

*Cross-validation: max_delay must be greater than initial_delay*

### Profiling Configuration
- **enabled**: Enable/disable profiling
- **level**: Profiling detail level (disabled/basic/standard/detailed)
- **sampling_rate** (0.0-1.0): Percentage of calls to profile
- **track_tokens/costs/memory/network_timing**: Feature flags
- **cost_alert_threshold_usd** (0.01-10000): Daily cost alert
- **latency_alert_threshold_s** (0.1-300): Latency alert threshold
- **memory_alert_threshold_mb** (10-100000): Memory alert threshold

### Security Configuration
- **secret_key**: Encryption key (minimum 32 characters, base64 pattern)

## Environment Variable Mapping

All configuration parameters can be set via environment variables with appropriate prefixes:

```bash
# Model configuration
CS_MAX_TOKENS=4096
CS_TEMPERATURE=0.7
BEDROCK_MODEL_ID="us.anthropic.claude-3-haiku:1"
LITELLM_MODEL="gpt-4"
OLLAMA_HOST="http://localhost:11434"

# Server configuration
CS_HOST="localhost"
CS_PORT=3023
CS_LOG_LEVEL="INFO"

# Security
CONTEXT_SWITCHER_SECRET_KEY="your-base64-encoded-secret-key"

# Profiling
CS_PROFILING_ENABLED=true
CS_PROFILING_LEVEL="standard"
CS_PROFILING_SAMPLING_RATE=0.1
```

## Configuration Files

### JSON Configuration
```json
{
  "model": {
    "default_max_tokens": 4096,
    "default_temperature": 0.7,
    "ollama_host": "http://localhost:11434"
  },
  "server": {
    "host": "localhost", 
    "port": 3023,
    "log_level": "INFO"
  }
}
```

### YAML Configuration (requires PyYAML)
```yaml
model:
  default_max_tokens: 4096
  default_temperature: 0.7
  ollama_host: "http://localhost:11434"

server:
  host: localhost
  port: 3023
  log_level: INFO
```

## Usage

### Basic Usage (Backward Compatible)
```python
from context_switcher_mcp.config import get_config

config = get_config()
print(f"Max tokens: {config.model.default_max_tokens}")
print(f"Server port: {config.server.port}")
```

### Using Validated Configuration Directly
```python
from context_switcher_mcp.validated_config import load_validated_config

# Load with validation
config = load_validated_config()

# Load from file
config = load_validated_config(config_file="config.json")

# Check production readiness
if config.is_production_ready:
    print("✅ Configuration is production-ready")
else:
    print("⚠️  Configuration needs production adjustments")

# Get masked configuration for logging
masked = config.mask_sensitive_data()
```

### Configuration Validation CLI
```bash
# Validate current environment
python -m src.context_switcher_mcp.config_validator

# Validate configuration file
python -m src.context_switcher_mcp.config_validator --config config.json

# Generate JSON report
python -m src.context_switcher_mcp.config_validator --format json --output report.json

# Check migration readiness
python -m src.context_switcher_mcp.config_validator --migration
```

## Error Handling

### Clear Validation Messages
```
Configuration validation failed:
  • port: Input should be greater than or equal to 1024
  • default_temperature: Input should be less than or equal to 2
  • bedrock_model_id: Bedrock model ID must be in format: region.provider.model:version
```

### Fallback Behavior
The system provides graceful fallback:
1. **Try validated configuration** with comprehensive error checking
2. **Fall back to legacy configuration** with basic error handling  
3. **Provide clear error messages** for both validation and runtime errors
4. **Log configuration source** (validated vs legacy) for debugging

## Migration Guide

### For Existing Code
No changes required! The system maintains full backward compatibility:

```python
# This continues to work exactly the same
from context_switcher_mcp.config import get_config
config = get_config()
```

### For New Code
Consider using validated configuration directly:

```python
# Better type safety and validation
from context_switcher_mcp.validated_config import get_validated_config
config = get_validated_config()
```

### Migration Validation
```bash
# Check if your system is ready for migration
python -m src.context_switcher_mcp.config_migration

# Output shows:
# ✅ Migration Status: READY
# Validated 15 configuration aspects
```

## Configuration Templates

### Development Configuration
- Debug logging enabled
- Higher limits for experimentation
- Detailed profiling enabled
- Local service endpoints

### Production Configuration
- INFO/WARNING logging only
- Conservative limits and timeouts
- Basic profiling for performance
- Secure defaults and required secret key
- Bind to all interfaces with firewall protection

### Example Configuration
- Balanced settings for general use
- Standard profiling level
- Reasonable limits and timeouts

## Validation Reports

### Text Report
```
Configuration Validation Report
========================================
Overall Success: ✅ PASS
Configuration Loading: ✅ SUCCESS
Parameter Issues: 0
Constraint Issues: 0
Security Issues: 0
Production Ready: ✅ YES
```

### JSON Report
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "validation_summary": {
    "overall_success": true,
    "loading_success": true,
    "parameter_issues": 0,
    "security_issues": 0,
    "production_ready": true
  },
  "detailed_results": { ... },
  "recommendations": []
}
```

## Production Readiness

The system automatically assesses production readiness:

### Requirements
- ✅ Log level is INFO/WARNING/ERROR (not DEBUG)
- ✅ Secret key is configured and secure
- ✅ Profiling level is basic/standard (not detailed)
- ✅ Resource limits are appropriate

### Recommendations
- Use environment-specific configuration files
- Set up monitoring for configuration alerts
- Regularly validate configuration in CI/CD
- Review security settings for sensitive environments

## Security Features

### Sensitive Data Masking
```python
config_dict = config.mask_sensitive_data()
# Output: {"security": {"secret_key": "***MASKED***"}}
```

### Security Validation
- Secret key length and format validation
- Host binding security warnings
- Debug logging security alerts
- Cost tracking privacy considerations

### Best Practices
- Store secret keys in environment variables, not config files
- Use different configurations per environment
- Monitor configuration changes
- Validate configuration in deployment pipelines

## Troubleshooting

### Common Issues

1. **ValidationError on startup**
   ```
   Solution: Check environment variables for invalid values
   Use: python -m src.context_switcher_mcp.config_validator
   ```

2. **Fallback to legacy configuration**
   ```
   Warning: Configuration validation failed, using legacy config
   Solution: Fix validation errors shown in logs
   ```

3. **Production readiness warnings**
   ```
   Issue: Configuration not production-ready
   Solution: Set secret key, adjust log level, review profiling settings
   ```

### Debug Steps
1. Run configuration validator CLI
2. Check environment variables
3. Validate configuration file syntax
4. Review validation error messages
5. Check system logs for warnings

## Benefits

This validation system provides:

- 🛡️ **Bulletproof Configuration**: No more runtime failures from invalid config
- 🚀 **Fail-Fast Loading**: Problems caught at startup with clear messages
- 🔒 **Security Awareness**: Proper validation of sensitive parameters
- 📊 **Production Readiness**: Built-in assessment for deployment
- 🔄 **Seamless Migration**: Full backward compatibility
- 🛠️ **Developer Tools**: CLI utilities for validation and reporting
- 📚 **Self-Documenting**: Configuration schema serves as documentation

The system eliminates the risk of runtime configuration failures while maintaining the flexibility and ease of use of the original configuration system.