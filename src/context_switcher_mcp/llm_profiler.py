"""Comprehensive async LLM profiling and monitoring system

This module provides transparent profiling of LLM calls with minimal performance impact.
Tracks timing, token usage, costs, memory usage, and provides optimization insights.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import psutil

from .circular_buffer import CircularBuffer
from .config import get_config
from .logging_base import get_logger

logger = get_logger(__name__)


class ProfilingLevel(Enum):
    """Profiling detail levels"""

    DISABLED = "disabled"
    BASIC = "basic"  # Timing only
    STANDARD = "standard"  # Timing + tokens + costs
    DETAILED = "detailed"  # Everything + memory profiling


@dataclass
class LLMCallMetrics:
    """Comprehensive metrics for a single LLM call"""

    # Identifiers
    call_id: str
    session_id: str
    thread_name: str
    backend: str
    model_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Timing metrics (seconds)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    network_latency: float | None = None  # Time to first token/response
    processing_time: float | None = None  # Total - network latency
    queue_time: float | None = None  # Time waiting for circuit breaker

    # Token metrics
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None

    # Performance metrics
    success: bool = False
    error_type: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    circuit_breaker_triggered: bool = False
    streaming_enabled: bool = False

    # Resource metrics
    peak_memory_mb: float | None = None
    cpu_usage_percent: float | None = None

    # Additional context
    prompt_length: int | None = None
    response_length: int | None = None
    context_length: int | None = None

    @property
    def total_latency(self) -> float | None:
        """Calculate total call latency"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float | None:
        """Calculate token generation rate"""
        if not self.total_latency or not self.output_tokens:
            return None
        return self.output_tokens / self.total_latency

    @property
    def cost_per_token(self) -> float | None:
        """Calculate cost efficiency"""
        if not self.estimated_cost_usd or not self.total_tokens:
            return None
        return self.estimated_cost_usd / self.total_tokens


@dataclass
class ProfilingConfig:
    """Configuration for LLM profiling"""

    enabled: bool = True
    level: ProfilingLevel = ProfilingLevel.STANDARD
    sampling_rate: float = 0.1  # Profile 10% of calls by default

    # Feature flags
    track_tokens: bool = True
    track_costs: bool = True
    track_memory: bool = False  # More expensive, disabled by default
    track_network_timing: bool = True

    # Storage settings
    max_history_size: int = 10000

    # Alert thresholds
    cost_alert_threshold_usd: float = 100.0  # Daily budget alert
    latency_alert_threshold_s: float = 30.0  # High latency alert
    memory_alert_threshold_mb: float = 1000.0  # High memory usage alert

    # Sampling rules - always profile these conditions
    always_profile_errors: bool = True
    always_profile_slow_calls: bool = True  # Calls > latency_alert_threshold
    always_profile_expensive_calls: bool = True  # Calls > $1.00
    always_profile_circuit_breaker: bool = True


class CostCalculator:
    """Calculates costs for different LLM backends and models"""

    # Pricing per 1000 tokens (as of 2025)
    BEDROCK_PRICING = {
        "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003, "output": 0.015},
    }

    # LiteLLM pricing varies by provider - approximate costs
    LITELLM_PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    @classmethod
    def calculate_cost(
        self, backend: str, model_name: str, input_tokens: int, output_tokens: int
    ) -> float | None:
        """Calculate cost in USD for a model call"""
        try:
            pricing_table = None

            if backend == "bedrock":
                pricing_table = self.BEDROCK_PRICING
            elif backend == "litellm":
                pricing_table = self.LITELLM_PRICING
            elif backend == "ollama":
                # Ollama is typically free (local hosting)
                return 0.0

            if not pricing_table or model_name not in pricing_table:
                logger.debug(f"No pricing data for {backend}:{model_name}")
                return None

            pricing = pricing_table[model_name]

            # Calculate cost per 1000 tokens, convert to actual token count
            input_cost = (input_tokens / 1000.0) * pricing["input"]
            output_cost = (output_tokens / 1000.0) * pricing["output"]

            return round(input_cost + output_cost, 6)  # Round to 6 decimal places

        except Exception as e:
            logger.error(f"Error calculating cost for {backend}:{model_name}: {e}")
            return None


class MemoryProfiler:
    """Profiles memory usage during LLM operations"""

    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0.0
        self._monitoring = False
        self._monitor_task = None

    async def start_monitoring(self) -> None:
        """Start continuous memory monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self.peak_memory = 0.0
        self._monitor_task = asyncio.create_task(self._monitor_memory())

    async def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage in MB"""
        if not self._monitoring:
            return 0.0

        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        return self.peak_memory

    async def _monitor_memory(self) -> None:
        """Background memory monitoring task"""
        try:
            while self._monitoring:
                memory_info = self.process.memory_info()
                current_memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                self.peak_memory = max(self.peak_memory, current_memory_mb)
                await asyncio.sleep(0.1)  # Check every 100ms
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")


class LLMProfiler:
    """Main LLM profiling orchestrator"""

    def __init__(self, config: ProfilingConfig | None = None):
        """Initialize LLM profiler with configuration"""
        self.config = config or ProfilingConfig()

        # Metrics storage
        self.metrics_history = CircularBuffer[LLMCallMetrics](
            self.config.max_history_size
        )
        self.metrics_lock = asyncio.Lock()

        # Cost tracking
        self.cost_calculator = CostCalculator()

        # Memory profiler instance (reused for efficiency)
        self.memory_profiler = MemoryProfiler() if self.config.track_memory else None

        # Performance counters
        self._total_calls = 0
        self._profiled_calls = 0
        self._sampling_decisions = {}

        logger.info(
            f"LLM Profiler initialized: level={self.config.level.value}, "
            f"sampling_rate={self.config.sampling_rate}"
        )

    def should_profile_call(
        self,
        session_id: str,
        thread_name: str,
        backend: str,
        circuit_breaker_triggered: bool = False,
    ) -> bool:
        """Determine if this call should be profiled based on sampling rules"""

        if not self.config.enabled or self.config.level == ProfilingLevel.DISABLED:
            return False

        self._total_calls += 1

        # Always profile certain conditions
        if self.config.always_profile_circuit_breaker and circuit_breaker_triggered:
            logger.debug(f"Profiling due to circuit breaker trigger: {thread_name}")
            return True

        # Sample based on rate for normal operations
        call_hash = hash(f"{session_id}:{thread_name}:{self._total_calls}")
        should_sample = (call_hash % 1000) < (self.config.sampling_rate * 1000)

        if should_sample:
            self._profiled_calls += 1

        return should_sample

    @asynccontextmanager
    async def profile_call(
        self,
        session_id: str,
        thread_name: str,
        backend: str,
        model_name: str,
        streaming: bool = False,
        circuit_breaker_triggered: bool = False,
    ):
        """Context manager for profiling an LLM call"""

        if not self.should_profile_call(
            session_id, thread_name, backend, circuit_breaker_triggered
        ):
            # No profiling - yield a dummy context
            yield None
            return

        # Initialize metrics
        call_id = str(uuid.uuid4())
        metrics = LLMCallMetrics(
            call_id=call_id,
            session_id=session_id,
            thread_name=thread_name,
            backend=backend,
            model_name=model_name,
            streaming_enabled=streaming,
            circuit_breaker_triggered=circuit_breaker_triggered,
        )

        # Start memory monitoring if enabled
        if self.memory_profiler and self.config.track_memory:
            await self.memory_profiler.start_monitoring()

        try:
            yield metrics

            # Mark as successful
            metrics.success = True

        except Exception as e:
            # Capture error information
            metrics.success = False
            metrics.error_type = type(e).__name__
            metrics.error_message = str(e)[:200]  # Truncate long messages
            raise

        finally:
            # Finalize timing
            metrics.end_time = time.time()

            # Stop memory monitoring
            if self.memory_profiler and self.config.track_memory:
                metrics.peak_memory_mb = await self.memory_profiler.stop_monitoring()

            # Store metrics
            await self._store_metrics(metrics)

    def record_token_usage(
        self,
        metrics: LLMCallMetrics | None,
        input_tokens: int,
        output_tokens: int,
        prompt_length: int | None = None,
        response_length: int | None = None,
    ) -> None:
        """Record token usage and calculate costs"""
        if not metrics or not self.config.track_tokens:
            return

        metrics.input_tokens = input_tokens
        metrics.output_tokens = output_tokens
        metrics.total_tokens = input_tokens + output_tokens
        metrics.prompt_length = prompt_length
        metrics.response_length = response_length

        # Calculate costs if enabled
        if self.config.track_costs:
            metrics.estimated_cost_usd = self.cost_calculator.calculate_cost(
                metrics.backend, metrics.model_name, input_tokens, output_tokens
            )

    def record_network_timing(
        self, metrics: LLMCallMetrics | None, first_token_time: float
    ) -> None:
        """Record network latency (time to first token)"""
        if not metrics or not self.config.track_network_timing:
            return

        metrics.network_latency = first_token_time - metrics.start_time

        if metrics.end_time:
            metrics.processing_time = metrics.total_latency - metrics.network_latency

    async def _store_metrics(self, metrics: LLMCallMetrics) -> None:
        """Store metrics in circular buffer"""
        async with self.metrics_lock:
            self.metrics_history.append(metrics)

        # Check for alert conditions
        self._check_alerts(metrics)

        # Log interesting metrics
        if metrics.total_latency and metrics.total_latency > 10.0:  # Log slow calls
            logger.info(
                f"Slow LLM call: {metrics.thread_name} took {metrics.total_latency:.2f}s "
                f"({metrics.backend}:{metrics.model_name})"
            )

        if (
            metrics.estimated_cost_usd and metrics.estimated_cost_usd > 1.0
        ):  # Log expensive calls
            logger.info(
                f"Expensive LLM call: {metrics.thread_name} cost ${metrics.estimated_cost_usd:.4f} "
                f"({metrics.total_tokens} tokens)"
            )

    def _check_alerts(self, metrics: LLMCallMetrics) -> None:
        """Check if metrics trigger any alerts"""
        alerts = []

        # Latency alerts
        if (
            metrics.total_latency
            and metrics.total_latency > self.config.latency_alert_threshold_s
        ):
            alerts.append(f"High latency: {metrics.total_latency:.2f}s")

        # Cost alerts
        if (
            metrics.estimated_cost_usd and metrics.estimated_cost_usd > 1.0
        ):  # Alert on expensive single calls
            alerts.append(f"Expensive call: ${metrics.estimated_cost_usd:.4f}")

        # Memory alerts
        if (
            metrics.peak_memory_mb
            and metrics.peak_memory_mb > self.config.memory_alert_threshold_mb
        ):
            alerts.append(f"High memory usage: {metrics.peak_memory_mb:.1f}MB")

        # Error alerts
        if not metrics.success:
            alerts.append(f"Call failed: {metrics.error_type}")

        # Log alerts
        for alert in alerts:
            logger.warning(f"PROFILER ALERT [{metrics.thread_name}]: {alert}")

    async def get_performance_metrics(self, last_n: int = 100) -> dict[str, Any]:
        """Get comprehensive performance metrics"""
        async with self.metrics_lock:
            if self.metrics_history.is_empty():
                return {"message": "No profiling data available"}

            recent_metrics = self.metrics_history.get_recent(last_n)

        # Calculate comprehensive statistics
        return self._calculate_performance_statistics(recent_metrics)

    def _calculate_performance_statistics(
        self, metrics_list: list[LLMCallMetrics]
    ) -> dict[str, Any]:
        """Calculate detailed performance statistics"""

        if not metrics_list:
            return {"message": "No metrics to analyze"}

        # Basic statistics
        total_calls = len(metrics_list)
        successful_calls = sum(1 for m in metrics_list if m.success)
        success_rate = (successful_calls / total_calls) * 100

        # Timing statistics
        latencies = [m.total_latency for m in metrics_list if m.total_latency]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Token statistics
        total_tokens_used = sum(m.total_tokens or 0 for m in metrics_list)
        total_input_tokens = sum(m.input_tokens or 0 for m in metrics_list)
        total_output_tokens = sum(m.output_tokens or 0 for m in metrics_list)

        # Cost statistics
        costs = [m.estimated_cost_usd for m in metrics_list if m.estimated_cost_usd]
        total_cost = sum(costs) if costs else 0
        avg_cost_per_call = total_cost / len(costs) if costs else 0

        # Backend breakdown
        backend_stats = self._calculate_backend_breakdown(metrics_list)

        # Performance insights
        insights = self._generate_performance_insights(metrics_list)

        return {
            "profiling_summary": {
                "total_calls_analyzed": total_calls,
                "success_rate_percent": round(success_rate, 1),
                "avg_latency_seconds": round(avg_latency, 3),
                "profiling_overhead": f"{self._profiled_calls}/{self._total_calls} calls",
                "sampling_rate": self.config.sampling_rate,
            },
            "token_usage": {
                "total_tokens": total_tokens_used,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "avg_tokens_per_call": round(total_tokens_used / total_calls, 1),
            },
            "cost_analysis": {
                "total_cost_usd": round(total_cost, 4),
                "avg_cost_per_call_usd": round(avg_cost_per_call, 4),
                "cost_per_token_usd": round(total_cost / total_tokens_used, 6)
                if total_tokens_used
                else 0,
            },
            "backend_performance": backend_stats,
            "performance_insights": insights,
        }

    def _calculate_backend_breakdown(
        self, metrics_list: list[LLMCallMetrics]
    ) -> dict[str, Any]:
        """Calculate performance breakdown by backend"""
        backend_data = {}

        for metrics in metrics_list:
            backend = metrics.backend
            if backend not in backend_data:
                backend_data[backend] = {
                    "calls": 0,
                    "successful": 0,
                    "total_latency": 0.0,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                }

            data = backend_data[backend]
            data["calls"] += 1
            if metrics.success:
                data["successful"] += 1
            if metrics.total_latency:
                data["total_latency"] += metrics.total_latency
            if metrics.estimated_cost_usd:
                data["total_cost"] += metrics.estimated_cost_usd
            if metrics.total_tokens:
                data["total_tokens"] += metrics.total_tokens

        # Calculate derived metrics
        for _backend, data in backend_data.items():
            data["success_rate"] = (data["successful"] / data["calls"]) * 100
            data["avg_latency"] = (
                data["total_latency"] / data["calls"] if data["calls"] else 0
            )
            data["avg_cost_per_call"] = (
                data["total_cost"] / data["calls"] if data["calls"] else 0
            )

        return backend_data

    def _generate_performance_insights(
        self, metrics_list: list[LLMCallMetrics]
    ) -> list[str]:
        """Generate actionable performance insights"""
        insights = []

        # Analyze latency patterns
        latencies = [m.total_latency for m in metrics_list if m.total_latency]
        if latencies:
            slow_calls = [latency for latency in latencies if latency > 10.0]
            if len(slow_calls) > len(latencies) * 0.1:  # >10% slow calls
                insights.append(
                    f"High latency detected: {len(slow_calls)} calls took >10s "
                    f"({len(slow_calls) / len(latencies) * 100:.1f}% of calls)"
                )

        # Analyze cost efficiency
        costs = [m.estimated_cost_usd for m in metrics_list if m.estimated_cost_usd]
        tokens = [m.total_tokens for m in metrics_list if m.total_tokens]
        if costs and tokens:
            avg_cost_per_1k_tokens = (sum(costs) / sum(tokens)) * 1000
            if avg_cost_per_1k_tokens > 0.1:  # High cost per token
                insights.append(
                    f"High cost efficiency: ${avg_cost_per_1k_tokens:.4f} per 1K tokens "
                    f"(consider using smaller models for simple tasks)"
                )

        # Analyze error patterns
        errors = [m for m in metrics_list if not m.success]
        if errors:
            error_rate = len(errors) / len(metrics_list)
            if error_rate > 0.05:  # >5% error rate
                common_errors = {}
                for error in errors:
                    error_type = error.error_type or "unknown"
                    common_errors[error_type] = common_errors.get(error_type, 0) + 1

                most_common = max(common_errors.items(), key=lambda x: x[1])
                insights.append(
                    f"High error rate: {error_rate * 100:.1f}% "
                    f"(most common: {most_common[0]} - {most_common[1]} occurrences)"
                )

        # Token efficiency analysis
        if tokens:
            avg_tokens = sum(tokens) / len(tokens)
            if avg_tokens > 10000:  # Very long contexts
                insights.append(
                    f"Large context usage: {avg_tokens:.0f} avg tokens per call "
                    f"(consider context optimization for cost savings)"
                )

        return insights

    def get_configuration_status(self) -> dict[str, Any]:
        """Get current profiler configuration status"""
        return {
            "enabled": self.config.enabled,
            "level": self.config.level.value,
            "sampling_rate": self.config.sampling_rate,
            "features": {
                "track_tokens": self.config.track_tokens,
                "track_costs": self.config.track_costs,
                "track_memory": self.config.track_memory,
                "track_network_timing": self.config.track_network_timing,
            },
            "thresholds": {
                "cost_alert_usd": self.config.cost_alert_threshold_usd,
                "latency_alert_s": self.config.latency_alert_threshold_s,
                "memory_alert_mb": self.config.memory_alert_threshold_mb,
            },
            "storage": {
                "max_history": self.config.max_history_size,
                "current_usage": len(self.metrics_history),
                "utilization_percent": (
                    len(self.metrics_history) / self.config.max_history_size
                )
                * 100,
            },
            "statistics": {
                "total_calls": self._total_calls,
                "profiled_calls": self._profiled_calls,
                "profiling_rate": (
                    self._profiled_calls / self._total_calls * 100
                    if self._total_calls
                    else 0
                ),
            },
        }


# Global profiler instance
_global_profiler: LLMProfiler | None = None


def get_global_profiler() -> LLMProfiler:
    """Get or create the global LLM profiler instance"""
    global _global_profiler

    if _global_profiler is None:
        # Load configuration from environment
        config = get_config()

        # Use the profiling configuration from the main config
        profiling_config = ProfilingConfig(
            enabled=config.profiling.enabled,
            level=ProfilingLevel(config.profiling.level),
            sampling_rate=config.profiling.sampling_rate,
            track_tokens=config.profiling.track_tokens,
            track_costs=config.profiling.track_costs,
            track_memory=config.profiling.track_memory,
            track_network_timing=config.profiling.track_network_timing,
            max_history_size=config.profiling.max_history_size,
            cost_alert_threshold_usd=config.profiling.cost_alert_threshold_usd,
            latency_alert_threshold_s=config.profiling.latency_alert_threshold_s,
            memory_alert_threshold_mb=config.profiling.memory_alert_threshold_mb,
            always_profile_errors=config.profiling.always_profile_errors,
            always_profile_slow_calls=config.profiling.always_profile_slow_calls,
            always_profile_expensive_calls=config.profiling.always_profile_expensive_calls,
            always_profile_circuit_breaker=config.profiling.always_profile_circuit_breaker,
        )

        _global_profiler = LLMProfiler(profiling_config)

    return _global_profiler
