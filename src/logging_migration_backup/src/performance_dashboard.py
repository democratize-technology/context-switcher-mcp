"""Performance dashboard and reporting for LLM profiling

This module provides comprehensive reporting, analysis, and optimization insights
based on collected LLM profiling data.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import statistics

from .llm_profiler import get_global_profiler, LLMCallMetrics

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Detailed cost analysis breakdown"""

    total_cost_usd: float
    cost_by_backend: Dict[str, float]
    cost_by_model: Dict[str, float]
    cost_by_session: Dict[str, float]
    avg_cost_per_call: float
    most_expensive_call: Optional[LLMCallMetrics]
    daily_burn_rate: float
    projected_monthly_cost: float


@dataclass
class PerformanceAnalysis:
    """Comprehensive performance analysis"""

    total_calls: int
    success_rate: float
    avg_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    slowest_calls: List[LLMCallMetrics]
    fastest_backend: str
    slowest_backend: str
    throughput_calls_per_minute: float


@dataclass
class EfficiencyMetrics:
    """Efficiency and optimization insights"""

    tokens_per_second: float
    cost_per_token: float
    most_efficient_backend: str
    least_efficient_backend: str
    optimization_opportunities: List[str]
    token_usage_distribution: Dict[str, int]


@dataclass
class AlertSummary:
    """Summary of triggered alerts and issues"""

    high_latency_calls: int
    expensive_calls: int
    error_calls: int
    memory_alerts: int
    circuit_breaker_triggers: int
    recent_alerts: List[Dict[str, Any]]


class PerformanceDashboard:
    """Main dashboard for LLM performance monitoring and analysis"""

    def __init__(self):
        """Initialize performance dashboard"""
        self.profiler = get_global_profiler()

        # Caching for expensive calculations
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_ttl = 60.0  # Cache for 1 minute

    def _get_cached_or_compute(
        self, cache_key: str, compute_func, ttl_override: Optional[float] = None
    ) -> Any:
        """Get cached result or compute if expired"""
        now = time.time()
        ttl = ttl_override or self._cache_ttl

        if cache_key in self._cache:
            cache_time, cached_result = self._cache[cache_key]
            if now - cache_time < ttl:
                return cached_result

        # Compute new result
        result = compute_func()
        self._cache[cache_key] = (now, result)
        return result

    async def get_comprehensive_dashboard(
        self, hours_back: int = 24, include_cache_stats: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""

        # Get recent metrics
        metrics = await self._get_metrics_for_timeframe(hours_back)

        if not metrics:
            return {
                "status": "no_data",
                "message": f"No profiling data available for the last {hours_back} hours",
                "profiler_status": self.profiler.get_configuration_status(),
            }

        # Compute all dashboard sections in parallel
        tasks = [
            self._compute_cost_breakdown(metrics),
            self._compute_performance_analysis(metrics),
            self._compute_efficiency_metrics(metrics),
            self._compute_alert_summary(metrics),
            self._compute_backend_comparison(metrics),
            self._compute_trends_analysis(metrics, hours_back),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        (
            cost_breakdown,
            performance_analysis,
            efficiency_metrics,
            alert_summary,
            backend_comparison,
            trends_analysis,
        ) = results

        dashboard = {
            "dashboard_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "timeframe_hours": hours_back,
                "total_metrics_analyzed": len(metrics),
                "profiler_status": self.profiler.get_configuration_status(),
            },
            "cost_analysis": cost_breakdown
            if not isinstance(cost_breakdown, Exception)
            else {"error": str(cost_breakdown)},
            "performance": performance_analysis
            if not isinstance(performance_analysis, Exception)
            else {"error": str(performance_analysis)},
            "efficiency": efficiency_metrics
            if not isinstance(efficiency_metrics, Exception)
            else {"error": str(efficiency_metrics)},
            "alerts": alert_summary
            if not isinstance(alert_summary, Exception)
            else {"error": str(alert_summary)},
            "backend_comparison": backend_comparison
            if not isinstance(backend_comparison, Exception)
            else {"error": str(backend_comparison)},
            "trends": trends_analysis
            if not isinstance(trends_analysis, Exception)
            else {"error": str(trends_analysis)},
        }

        if include_cache_stats:
            dashboard["cache_stats"] = self._get_cache_statistics()

        return dashboard

    async def _get_metrics_for_timeframe(self, hours_back: int) -> List[LLMCallMetrics]:
        """Get metrics for specified timeframe"""

        def compute_metrics():
            # Get all available metrics
            all_metrics = (
                self.profiler.metrics_history.get_all()
                if not self.profiler.metrics_history.is_empty()
                else []
            )

            # Filter by timeframe
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

            filtered_metrics = [m for m in all_metrics if m.timestamp >= cutoff_time]

            return filtered_metrics

        return self._get_cached_or_compute(
            f"metrics_timeframe_{hours_back}h",
            compute_metrics,
            ttl_override=30.0,  # Cache metrics for 30 seconds
        )

    async def _compute_cost_breakdown(
        self, metrics: List[LLMCallMetrics]
    ) -> CostBreakdown:
        """Compute detailed cost breakdown"""

        costs = [m.estimated_cost_usd for m in metrics if m.estimated_cost_usd]
        if not costs:
            return CostBreakdown(
                total_cost_usd=0.0,
                cost_by_backend={},
                cost_by_model={},
                cost_by_session={},
                avg_cost_per_call=0.0,
                most_expensive_call=None,
                daily_burn_rate=0.0,
                projected_monthly_cost=0.0,
            )

        total_cost = sum(costs)

        # Breakdown by backend
        cost_by_backend = defaultdict(float)
        cost_by_model = defaultdict(float)
        cost_by_session = defaultdict(float)

        most_expensive_call = None
        max_cost = 0.0

        for metric in metrics:
            if metric.estimated_cost_usd:
                cost_by_backend[metric.backend] += metric.estimated_cost_usd
                cost_by_model[metric.model_name] += metric.estimated_cost_usd
                cost_by_session[metric.session_id] += metric.estimated_cost_usd

                if metric.estimated_cost_usd > max_cost:
                    max_cost = metric.estimated_cost_usd
                    most_expensive_call = metric

        # Calculate burn rates
        timeframe_hours = (
            metrics[-1].timestamp - metrics[0].timestamp
        ).total_seconds() / 3600
        daily_burn_rate = (
            (total_cost / timeframe_hours) * 24 if timeframe_hours > 0 else 0
        )

        return CostBreakdown(
            total_cost_usd=round(total_cost, 4),
            cost_by_backend=dict(cost_by_backend),
            cost_by_model=dict(cost_by_model),
            cost_by_session=dict(cost_by_session),
            avg_cost_per_call=round(total_cost / len(costs), 6),
            most_expensive_call=most_expensive_call,
            daily_burn_rate=round(daily_burn_rate, 4),
            projected_monthly_cost=round(daily_burn_rate * 30, 2),
        )

    async def _compute_performance_analysis(
        self, metrics: List[LLMCallMetrics]
    ) -> PerformanceAnalysis:
        """Compute comprehensive performance analysis"""

        total_calls = len(metrics)
        successful_calls = sum(1 for m in metrics if m.success)
        success_rate = (successful_calls / total_calls) * 100 if total_calls > 0 else 0

        # Latency analysis
        latencies = [m.total_latency for m in metrics if m.total_latency]

        if not latencies:
            return PerformanceAnalysis(
                total_calls=total_calls,
                success_rate=success_rate,
                avg_latency=0.0,
                median_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                slowest_calls=[],
                fastest_backend="unknown",
                slowest_backend="unknown",
                throughput_calls_per_minute=0.0,
            )

        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = self._percentile(latencies, 95)
        p99_latency = self._percentile(latencies, 99)

        # Find slowest calls
        slowest_calls = sorted(
            [m for m in metrics if m.total_latency],
            key=lambda x: x.total_latency,
            reverse=True,
        )[:5]

        # Backend performance comparison
        backend_latencies = defaultdict(list)
        for metric in metrics:
            if metric.total_latency:
                backend_latencies[metric.backend].append(metric.total_latency)

        backend_avg_latencies = {
            backend: statistics.mean(latencies)
            for backend, latencies in backend_latencies.items()
        }

        fastest_backend = (
            min(backend_avg_latencies.items(), key=lambda x: x[1])[0]
            if backend_avg_latencies
            else "unknown"
        )
        slowest_backend = (
            max(backend_avg_latencies.items(), key=lambda x: x[1])[0]
            if backend_avg_latencies
            else "unknown"
        )

        # Throughput calculation
        if len(metrics) >= 2:
            timeframe_minutes = (
                metrics[-1].timestamp - metrics[0].timestamp
            ).total_seconds() / 60
            throughput = total_calls / timeframe_minutes if timeframe_minutes > 0 else 0
        else:
            throughput = 0

        return PerformanceAnalysis(
            total_calls=total_calls,
            success_rate=round(success_rate, 1),
            avg_latency=round(avg_latency, 3),
            median_latency=round(median_latency, 3),
            p95_latency=round(p95_latency, 3),
            p99_latency=round(p99_latency, 3),
            slowest_calls=slowest_calls,
            fastest_backend=fastest_backend,
            slowest_backend=slowest_backend,
            throughput_calls_per_minute=round(throughput, 2),
        )

    async def _compute_efficiency_metrics(
        self, metrics: List[LLMCallMetrics]
    ) -> EfficiencyMetrics:
        """Compute efficiency and optimization metrics"""

        # Token efficiency
        token_rates = [m.tokens_per_second for m in metrics if m.tokens_per_second]
        avg_tokens_per_second = statistics.mean(token_rates) if token_rates else 0

        # Cost efficiency
        cost_per_token_rates = [m.cost_per_token for m in metrics if m.cost_per_token]
        avg_cost_per_token = (
            statistics.mean(cost_per_token_rates) if cost_per_token_rates else 0
        )

        # Backend efficiency comparison
        backend_efficiency = defaultdict(list)
        for metric in metrics:
            if metric.cost_per_token:
                backend_efficiency[metric.backend].append(metric.cost_per_token)

        backend_avg_efficiency = {
            backend: statistics.mean(costs)
            for backend, costs in backend_efficiency.items()
        }

        most_efficient_backend = (
            min(backend_avg_efficiency.items(), key=lambda x: x[1])[0]
            if backend_avg_efficiency
            else "unknown"
        )
        least_efficient_backend = (
            max(backend_avg_efficiency.items(), key=lambda x: x[1])[0]
            if backend_avg_efficiency
            else "unknown"
        )

        # Token usage distribution
        token_distribution = {
            "small_calls_<1k": sum(
                1 for m in metrics if m.total_tokens and m.total_tokens < 1000
            ),
            "medium_calls_1k-10k": sum(
                1 for m in metrics if m.total_tokens and 1000 <= m.total_tokens < 10000
            ),
            "large_calls_10k+": sum(
                1 for m in metrics if m.total_tokens and m.total_tokens >= 10000
            ),
        }

        # Generate optimization opportunities
        optimization_opportunities = []

        # High latency analysis
        high_latency_calls = [
            m for m in metrics if m.total_latency and m.total_latency > 10.0
        ]
        if len(high_latency_calls) > len(metrics) * 0.1:
            optimization_opportunities.append(
                f"High latency detected in {len(high_latency_calls)} calls - consider optimizing prompts or using faster models"
            )

        # Cost optimization
        expensive_calls = [
            m for m in metrics if m.estimated_cost_usd and m.estimated_cost_usd > 0.1
        ]
        if expensive_calls:
            avg_expensive_cost = statistics.mean(
                [m.estimated_cost_usd for m in expensive_calls]
            )
            optimization_opportunities.append(
                f"Consider using smaller models for {len(expensive_calls)} expensive calls (avg: ${avg_expensive_cost:.4f})"
            )

        # Large context analysis
        large_context_calls = [
            m for m in metrics if m.total_tokens and m.total_tokens > 20000
        ]
        if large_context_calls:
            optimization_opportunities.append(
                f"Large context detected in {len(large_context_calls)} calls - consider context pruning or summarization"
            )

        return EfficiencyMetrics(
            tokens_per_second=round(avg_tokens_per_second, 2),
            cost_per_token=round(avg_cost_per_token, 6),
            most_efficient_backend=most_efficient_backend,
            least_efficient_backend=least_efficient_backend,
            optimization_opportunities=optimization_opportunities,
            token_usage_distribution=token_distribution,
        )

    async def _compute_alert_summary(
        self, metrics: List[LLMCallMetrics]
    ) -> AlertSummary:
        """Compute alert summary and recent issues"""

        # Count different types of alerts
        high_latency_calls = sum(
            1 for m in metrics if m.total_latency and m.total_latency > 30.0
        )
        expensive_calls = sum(
            1 for m in metrics if m.estimated_cost_usd and m.estimated_cost_usd > 1.0
        )
        error_calls = sum(1 for m in metrics if not m.success)
        memory_alerts = sum(
            1 for m in metrics if m.peak_memory_mb and m.peak_memory_mb > 1000.0
        )
        circuit_breaker_triggers = sum(
            1 for m in metrics if m.circuit_breaker_triggered
        )

        # Recent alerts (last 10)
        recent_alerts = []

        for metric in sorted(metrics, key=lambda x: x.timestamp, reverse=True)[:10]:
            alerts = []

            if metric.total_latency and metric.total_latency > 30.0:
                alerts.append(f"High latency: {metric.total_latency:.2f}s")

            if metric.estimated_cost_usd and metric.estimated_cost_usd > 1.0:
                alerts.append(f"Expensive call: ${metric.estimated_cost_usd:.4f}")

            if not metric.success:
                alerts.append(f"Error: {metric.error_type}")

            if metric.circuit_breaker_triggered:
                alerts.append("Circuit breaker triggered")

            if alerts:
                recent_alerts.append(
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "thread_name": metric.thread_name,
                        "backend": metric.backend,
                        "alerts": alerts,
                    }
                )

        return AlertSummary(
            high_latency_calls=high_latency_calls,
            expensive_calls=expensive_calls,
            error_calls=error_calls,
            memory_alerts=memory_alerts,
            circuit_breaker_triggers=circuit_breaker_triggers,
            recent_alerts=recent_alerts,
        )

    async def _compute_backend_comparison(
        self, metrics: List[LLMCallMetrics]
    ) -> Dict[str, Any]:
        """Compare performance across backends"""

        backend_stats = defaultdict(
            lambda: {
                "calls": 0,
                "successful": 0,
                "total_latency": 0.0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "errors": [],
            }
        )

        for metric in metrics:
            stats = backend_stats[metric.backend]
            stats["calls"] += 1

            if metric.success:
                stats["successful"] += 1
            else:
                stats["errors"].append(metric.error_type or "unknown")

            if metric.total_latency:
                stats["total_latency"] += metric.total_latency

            if metric.estimated_cost_usd:
                stats["total_cost"] += metric.estimated_cost_usd

            if metric.total_tokens:
                stats["total_tokens"] += metric.total_tokens

        # Calculate derived metrics
        comparison = {}
        for backend, stats in backend_stats.items():
            comparison[backend] = {
                "calls": stats["calls"],
                "success_rate": (stats["successful"] / stats["calls"]) * 100
                if stats["calls"] > 0
                else 0,
                "avg_latency": stats["total_latency"] / stats["calls"]
                if stats["calls"] > 0
                else 0,
                "total_cost": stats["total_cost"],
                "avg_cost_per_call": stats["total_cost"] / stats["calls"]
                if stats["calls"] > 0
                else 0,
                "total_tokens": stats["total_tokens"],
                "common_errors": Counter(stats["errors"]).most_common(3),
            }

        return comparison

    async def _compute_trends_analysis(
        self, metrics: List[LLMCallMetrics], hours_back: int
    ) -> Dict[str, Any]:
        """Analyze trends over time"""

        if len(metrics) < 2:
            return {"message": "Insufficient data for trend analysis"}

        # Group metrics by hour
        hourly_stats = defaultdict(
            lambda: {
                "calls": 0,
                "successful": 0,
                "total_latency": 0.0,
                "total_cost": 0.0,
            }
        )

        for metric in metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            stats = hourly_stats[hour_key]

            stats["calls"] += 1
            if metric.success:
                stats["successful"] += 1
            if metric.total_latency:
                stats["total_latency"] += metric.total_latency
            if metric.estimated_cost_usd:
                stats["total_cost"] += metric.estimated_cost_usd

        # Calculate trends
        sorted_hours = sorted(hourly_stats.items())

        if len(sorted_hours) < 2:
            return {"message": "Insufficient time periods for trend analysis"}

        # Calculate growth rates
        first_half = sorted_hours[: len(sorted_hours) // 2]
        second_half = sorted_hours[len(sorted_hours) // 2 :]

        first_half_avg_calls = statistics.mean(
            [stats["calls"] for _, stats in first_half]
        )
        second_half_avg_calls = statistics.mean(
            [stats["calls"] for _, stats in second_half]
        )

        first_half_avg_cost = statistics.mean(
            [stats["total_cost"] for _, stats in first_half]
        )
        second_half_avg_cost = statistics.mean(
            [stats["total_cost"] for _, stats in second_half]
        )

        call_trend = (
            (
                (second_half_avg_calls - first_half_avg_calls)
                / first_half_avg_calls
                * 100
            )
            if first_half_avg_calls > 0
            else 0
        )
        cost_trend = (
            ((second_half_avg_cost - first_half_avg_cost) / first_half_avg_cost * 100)
            if first_half_avg_cost > 0
            else 0
        )

        return {
            "time_periods_analyzed": len(sorted_hours),
            "call_volume_trend_percent": round(call_trend, 1),
            "cost_trend_percent": round(cost_trend, 1),
            "peak_hour": max(sorted_hours, key=lambda x: x[1]["calls"])[0].isoformat()
            if sorted_hours
            else None,
            "hourly_breakdown": [
                {
                    "hour": hour.isoformat(),
                    "calls": stats["calls"],
                    "success_rate": (stats["successful"] / stats["calls"]) * 100
                    if stats["calls"] > 0
                    else 0,
                    "avg_latency": stats["total_latency"] / stats["calls"]
                    if stats["calls"] > 0
                    else 0,
                    "total_cost": stats["total_cost"],
                }
                for hour, stats in sorted_hours[-24:]  # Last 24 hours max
            ],
        }

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset"""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = min(lower_index + 1, len(sorted_data) - 1)
            weight = index - lower_index
            return (
                sorted_data[lower_index] * (1 - weight)
                + sorted_data[upper_index] * weight
            )

    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        now = time.time()

        total_entries = len(self._cache)
        expired_entries = sum(
            1
            for cache_time, _ in self._cache.values()
            if now - cache_time > self._cache_ttl
        )

        return {
            "total_cached_entries": total_entries,
            "expired_entries": expired_entries,
            "cache_hit_ratio": round(
                (total_entries - expired_entries) / total_entries * 100, 1
            )
            if total_entries > 0
            else 0,
            "cache_ttl_seconds": self._cache_ttl,
        }

    async def get_optimization_recommendations(
        self, hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get specific optimization recommendations"""

        metrics = await self._get_metrics_for_timeframe(hours_back)
        if not metrics:
            return {"message": "No data available for recommendations"}

        recommendations = []

        # Cost optimization
        expensive_calls = [
            m for m in metrics if m.estimated_cost_usd and m.estimated_cost_usd > 0.1
        ]
        if expensive_calls:
            total_expensive_cost = sum(m.estimated_cost_usd for m in expensive_calls)
            recommendations.append(
                {
                    "category": "Cost Optimization",
                    "priority": "High",
                    "issue": f"{len(expensive_calls)} expensive calls costing ${total_expensive_cost:.4f}",
                    "recommendation": "Consider using smaller models (Claude Haiku instead of Sonnet/Opus) for simple tasks",
                    "potential_savings": f"Up to 70% cost reduction (${total_expensive_cost * 0.7:.4f} potential savings)",
                }
            )

        # Performance optimization
        slow_calls = [m for m in metrics if m.total_latency and m.total_latency > 10.0]
        if len(slow_calls) > len(metrics) * 0.1:  # >10% slow calls
            avg_slow_latency = statistics.mean([m.total_latency for m in slow_calls])
            recommendations.append(
                {
                    "category": "Performance",
                    "priority": "Medium",
                    "issue": f"{len(slow_calls)} slow calls (avg: {avg_slow_latency:.2f}s)",
                    "recommendation": "Optimize prompts, reduce context length, or use streaming for better perceived performance",
                    "potential_improvement": "50-80% latency reduction",
                }
            )

        # Context optimization
        large_context_calls = [
            m for m in metrics if m.total_tokens and m.total_tokens > 50000
        ]
        if large_context_calls:
            recommendations.append(
                {
                    "category": "Context Efficiency",
                    "priority": "Medium",
                    "issue": f"{len(large_context_calls)} calls with very large context (>50k tokens)",
                    "recommendation": "Implement context pruning, summarization, or chunking strategies",
                    "potential_improvement": "30-50% cost and latency reduction",
                }
            )

        # Error rate optimization
        error_calls = [m for m in metrics if not m.success]
        if len(error_calls) > len(metrics) * 0.05:  # >5% error rate
            common_errors = Counter([m.error_type for m in error_calls]).most_common(3)
            recommendations.append(
                {
                    "category": "Reliability",
                    "priority": "High",
                    "issue": f"High error rate: {len(error_calls) / len(metrics) * 100:.1f}%",
                    "recommendation": f"Address common errors: {', '.join([f'{error}({count})' for error, count in common_errors])}",
                    "potential_improvement": "Improved reliability and reduced retry costs",
                }
            )

        return {
            "analysis_period": f"{hours_back} hours",
            "total_calls_analyzed": len(metrics),
            "recommendations": recommendations,
            "implementation_priority": "Start with High priority recommendations for maximum impact",
        }

    async def export_detailed_report(
        self, hours_back: int = 24, format: str = "json"
    ) -> Dict[str, Any]:
        """Export detailed performance report"""

        dashboard_data = await self.get_comprehensive_dashboard(hours_back)
        recommendations = await self.get_optimization_recommendations(hours_back)

        detailed_report = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report_type": "comprehensive_performance_analysis",
                "timeframe_hours": hours_back,
                "format": format,
            },
            "executive_summary": {
                "total_calls": dashboard_data.get("performance", {}).get(
                    "total_calls", 0
                ),
                "total_cost": dashboard_data.get("cost_analysis", {}).get(
                    "total_cost_usd", 0
                ),
                "success_rate": dashboard_data.get("performance", {}).get(
                    "success_rate", 0
                ),
                "avg_latency": dashboard_data.get("performance", {}).get(
                    "avg_latency", 0
                ),
                "key_issues": len(recommendations.get("recommendations", [])),
            },
            "detailed_analysis": dashboard_data,
            "optimization_recommendations": recommendations,
            "action_items": [
                "Review high-cost operations and consider model downgrades",
                "Implement prompt optimization for slow operations",
                "Set up automated alerts for cost and performance thresholds",
                "Regular monitoring of backend performance differences",
            ],
        }

        return detailed_report


# Global dashboard instance
_dashboard_instance: Optional[PerformanceDashboard] = None


def get_performance_dashboard() -> PerformanceDashboard:
    """Get global performance dashboard instance"""
    global _dashboard_instance

    if _dashboard_instance is None:
        _dashboard_instance = PerformanceDashboard()

    return _dashboard_instance
