# LLM Profiling and Performance Monitoring System

## Overview

The Context Switcher MCP now includes a comprehensive async LLM profiling and monitoring system that provides complete visibility into LLM operations, cost tracking, performance analysis, and optimization recommendations.

## Key Features

### 🎯 **Zero-Impact Profiling**
- Transparent profiling wrapper with <1% performance overhead
- Configurable sampling rates (default: 10% of calls)
- Always profiles errors, slow calls, and circuit breaker events
- Graceful degradation when profiling is disabled

### 💰 **Cost Tracking & Analysis**
- Real-time cost calculation for all backends (Bedrock, LiteLLM, Ollama)
- Backend-specific pricing models with automatic updates
- Daily burn rate tracking and monthly projections
- Cost per token efficiency analysis
- Expensive operation identification and alerting

### ⚡ **Performance Monitoring**
- Latency tracking with percentile analysis (P95, P99)
- Network latency vs processing time breakdown
- Throughput monitoring (calls/minute, tokens/second)
- Backend performance comparison
- Memory usage profiling (optional)

### 🔍 **Optimization Insights**
- AI-powered recommendations for cost savings
- Performance bottleneck identification
- Context optimization opportunities
- Model selection guidance
- Prompt efficiency analysis

### 📊 **Comprehensive Dashboard**
- Real-time performance metrics
- Historical trend analysis
- Executive summary reports
- Session-level profiling data
- Alert management and notifications

## Configuration

### Environment Variables

```bash
# Profiling Control
CS_PROFILING_ENABLED=true          # Enable/disable profiling
CS_PROFILING_LEVEL=standard        # disabled, basic, standard, detailed
CS_PROFILING_SAMPLING_RATE=0.1     # Profile 10% of calls

# Feature Flags
CS_PROFILING_TRACK_TOKENS=true     # Token usage tracking
CS_PROFILING_TRACK_COSTS=true      # Cost calculation
CS_PROFILING_TRACK_MEMORY=false    # Memory profiling (more expensive)
CS_PROFILING_TRACK_NETWORK=true    # Network timing

# Storage & Alerts
CS_PROFILING_MAX_HISTORY=10000     # Max metrics to store
CS_PROFILING_COST_ALERT=100.0      # Daily cost alert threshold ($)
CS_PROFILING_LATENCY_ALERT=30.0    # High latency alert (seconds)
CS_PROFILING_MEMORY_ALERT=1000.0   # Memory usage alert (MB)
```

### Profiling Levels

- **`disabled`**: No profiling
- **`basic`**: Timing only
- **`standard`**: Timing + tokens + costs (recommended)
- **`detailed`**: Everything + memory profiling

## MCP Tools

### Core Profiling Tools

#### `get_profiling_status`
Get current profiling configuration and statistics
```json
{
    "enabled": true,
    "level": "standard",
    "sampling_rate": 0.1,
    "statistics": {
        "total_calls": 1500,
        "profiled_calls": 150,
        "profiling_rate": 10.0
    }
}
```

#### `get_performance_dashboard`
Comprehensive performance dashboard with all metrics
```json
{
    "dashboard_metadata": {
        "generated_at": "2025-01-15T10:30:00Z",
        "timeframe_hours": 24,
        "total_metrics_analyzed": 150
    },
    "cost_analysis": { ... },
    "performance": { ... },
    "efficiency": { ... },
    "alerts": { ... }
}
```

### Specialized Analysis Tools

#### `get_cost_analysis_data`
Deep dive into cost breakdown and spending patterns
- Cost by backend, model, and session
- Daily burn rates and monthly projections
- Most expensive operations identification

#### `get_performance_analysis`
Performance metrics with latency analysis
- P95/P99 latency percentiles
- Backend performance comparison
- Throughput analysis
- Error rate tracking

#### `get_optimization_insights`
AI-powered optimization recommendations
- Cost reduction opportunities
- Performance improvement suggestions
- Context optimization guidance
- Model selection recommendations

#### `get_session_profiling_analysis`
Session-specific profiling data
- Timeline of operations
- Backend and thread usage
- Cost and performance breakdown

### Administrative Tools

#### `configure_profiling_settings`
Update profiling configuration
```json
{
    "enabled": true,
    "sampling_rate": 0.2,
    "track_costs": true,
    "track_memory": false
}
```

#### `reset_profiling_metrics`
**⚠️ WARNING**: Permanently deletes all profiling data

#### `export_performance_report`
Generate comprehensive performance report for stakeholders

## Architecture

### Core Components

#### LLMProfiler
- **Main profiling orchestrator**
- Manages metrics collection and storage
- Handles sampling decisions
- Provides performance statistics

#### ProfilingWrapper
- **Transparent backend wrapper**
- Intercepts LLM calls for profiling
- Records timing, tokens, and costs
- Maintains compatibility with existing code

#### PerformanceDashboard  
- **Analysis and reporting engine**
- Generates comprehensive insights
- Provides optimization recommendations
- Caches expensive calculations

#### CostCalculator
- **Backend-specific cost modeling**
- Supports Bedrock, LiteLLM pricing
- Real-time cost calculation
- Handles model-specific pricing tiers

### Data Models

#### LLMCallMetrics
```python
@dataclass
class LLMCallMetrics:
    # Identifiers
    call_id: str
    session_id: str
    thread_name: str
    backend: str
    model_name: str
    
    # Timing metrics
    start_time: float
    end_time: Optional[float]
    network_latency: Optional[float]  
    processing_time: Optional[float]
    
    # Token & cost metrics
    input_tokens: Optional[int]
    output_tokens: Optional[int] 
    estimated_cost_usd: Optional[float]
    
    # Performance metrics
    success: bool
    error_type: Optional[str]
    retry_count: int
    circuit_breaker_triggered: bool
```

## Usage Examples

### Basic Profiling Status Check

```python
# Get current profiling status
status = await get_profiling_status()
print(f"Profiling enabled: {status['enabled']}")
print(f"Sampling rate: {status['sampling_rate']}")
print(f"Total calls tracked: {status['statistics']['total_calls']}")
```

### Performance Dashboard Analysis

```python
# Get 24-hour performance dashboard
dashboard = await get_performance_dashboard({
    "hours_back": 24,
    "include_cache_stats": True
})

# Access specific metrics
cost_analysis = dashboard["cost_analysis"]
performance = dashboard["performance"]
efficiency = dashboard["efficiency"]

print(f"Total cost: ${cost_analysis['total_cost_usd']}")
print(f"Average latency: {performance['avg_latency']}s")
print(f"Success rate: {performance['success_rate']}%")
```

### Cost Optimization Analysis

```python
# Get optimization recommendations
recommendations = await get_optimization_insights({
    "hours_back": 168  # Last week
})

for rec in recommendations["recommendations"]:
    print(f"[{rec['priority']}] {rec['category']}: {rec['recommendation']}")
    print(f"Potential improvement: {rec['potential_improvement']}")
```

### Session-Level Analysis

```python
# Analyze specific session
session_data = await get_session_profiling_analysis({
    "session_id": "my-session-id"  
})

print(f"Session calls: {session_data['summary']['total_calls']}")
print(f"Total cost: ${session_data['summary']['total_cost_usd']}")
print(f"Success rate: {session_data['summary']['success_rate']}%")

# Review timeline
for event in session_data["timeline"]:
    print(f"{event['timestamp']}: {event['thread_name']} - {event['backend']} - ${event['cost']}")
```

## Performance Impact

### Profiling Overhead
- **Standard profiling**: <0.5% performance impact
- **With memory tracking**: <2% performance impact  
- **Sampling reduces overhead**: 10% sampling = 0.1% of overhead
- **Zero impact when disabled**

### Storage Efficiency
- **Circular buffer**: Automatic memory management
- **Configurable history size**: Default 10,000 metrics
- **Compression**: Time-series data optimization
- **Cache management**: 60-second TTL for expensive calculations

## Best Practices

### 1. **Production Deployment**
- Use `standard` profiling level in production
- Set sampling rate to 0.05-0.1 (5-10%)
- Enable cost tracking for budget management
- Disable memory tracking unless needed

### 2. **Development & Testing**
- Use `detailed` profiling level for development
- Set sampling rate to 1.0 (100%) for comprehensive analysis
- Enable all tracking features
- Use shorter history sizes for faster testing

### 3. **Monitoring Setup**
- Set appropriate cost alert thresholds
- Monitor daily burn rates
- Review optimization recommendations weekly
- Set up automated performance reports

### 4. **Cost Optimization**
- Review expensive operations regularly
- Consider model downgrades for simple tasks
- Implement context pruning for large prompts
- Monitor backend cost efficiency

### 5. **Performance Optimization**  
- Identify slow operations with latency analysis
- Use streaming for better perceived performance
- Monitor circuit breaker effectiveness
- Optimize prompts based on token efficiency

## Integration with Existing Systems

### Backward Compatibility
- **Zero breaking changes** to existing API
- **Optional profiling** - can be disabled completely
- **Transparent operation** - no code changes required
- **Graceful degradation** on profiler failures

### Circuit Breaker Integration
- Profiling respects circuit breaker states
- Circuit breaker events always profiled
- Performance data informs circuit breaker tuning
- Resilience patterns preserved

### Session Management Integration
- Session-level profiling data
- Client binding compatibility
- Security event tracking
- Access pattern analysis

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Reduce history size
CS_PROFILING_MAX_HISTORY=1000

# Disable memory tracking
CS_PROFILING_TRACK_MEMORY=false

# Reduce sampling rate
CS_PROFILING_SAMPLING_RATE=0.05
```

#### Performance Impact
```bash
# Reduce profiling level
CS_PROFILING_LEVEL=basic

# Lower sampling rate
CS_PROFILING_SAMPLING_RATE=0.01

# Disable expensive features
CS_PROFILING_TRACK_MEMORY=false
```

#### Missing Cost Data
```bash
# Ensure cost tracking is enabled
CS_PROFILING_TRACK_COSTS=true

# Check model pricing data
# Some models may not have pricing information
```

### Debug Logging
```bash
# Enable debug logging
CS_LOG_LEVEL=DEBUG

# Check profiler status
# Use get_profiling_status tool to verify configuration
```

## Future Enhancements

### Planned Features
- **Real-time alerting**: Slack/email notifications
- **Advanced analytics**: ML-powered anomaly detection  
- **Cost forecasting**: Predictive budget modeling
- **Multi-tenant support**: Organization-level tracking
- **Export formats**: CSV, Excel, PDF reports
- **Integration APIs**: Grafana, DataDog connectors

### Extensibility
- **Custom cost models**: Support for new pricing structures
- **Plugin architecture**: Custom metrics and analyzers
- **Webhook support**: Real-time event streaming
- **Custom dashboards**: Configurable visualization

## Support

For questions, issues, or feature requests related to the profiling system:

1. Check the configuration with `get_profiling_status`
2. Review logs for error messages
3. Verify environment variables are set correctly
4. Test with different profiling levels
5. Check storage utilization and clear old data if needed

The profiling system is designed to be self-monitoring and self-healing, with comprehensive error handling and graceful degradation capabilities.