"""MCP tools for LLM profiling and performance monitoring"""

from ..logging_config import get_logger
from typing import Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

from pydantic import BaseModel, Field

from ..llm_profiler import get_global_profiler
from ..performance_dashboard import get_performance_dashboard
from ..validation import validate_session_id

logger = get_logger(__name__)


# Request models for profiling tools
class PerformanceDashboardRequest(BaseModel):
    hours_back: int = Field(
        default=24, description="Number of hours to look back for data (1-168)"
    )
    include_cache_stats: bool = Field(
        default=True, description="Include cache performance statistics"
    )


class CostAnalysisRequest(BaseModel):
    hours_back: int = Field(
        default=24, description="Number of hours to analyze (1-168)"
    )


class PerformanceMetricsRequest(BaseModel):
    hours_back: int = Field(
        default=24, description="Number of hours to analyze (1-168)"
    )


class OptimizationRequest(BaseModel):
    hours_back: int = Field(
        default=24, description="Number of hours to analyze for recommendations (1-168)"
    )


class DetailedReportRequest(BaseModel):
    hours_back: int = Field(
        default=24, description="Number of hours to include in report (1-168)"
    )
    format: str = Field(
        default="json", description="Report format (currently only 'json')"
    )


class SessionProfilingRequest(BaseModel):
    session_id: str = Field(description="Session ID to analyze")


class ProfilingConfigRequest(BaseModel):
    enabled: Optional[bool] = Field(
        default=None, description="Enable/disable profiling"
    )
    sampling_rate: Optional[float] = Field(
        default=None, description="Sampling rate (0.0 to 1.0)"
    )
    track_costs: Optional[bool] = Field(
        default=None, description="Enable/disable cost tracking"
    )
    track_memory: Optional[bool] = Field(
        default=None, description="Enable/disable memory tracking"
    )


async def get_llm_profiling_status() -> Dict[str, Any]:
    """Get current LLM profiling configuration and status

    Returns:
        Dict containing profiling status, configuration, and statistics
    """
    try:
        profiler = get_global_profiler()
        return profiler.get_configuration_status()
    except Exception as e:
        logger.error(f"Error getting profiling status: {e}")
        return {"error": "Failed to get profiling status", "message": str(e)}


async def get_performance_dashboard_data(
    hours_back: int = 24, include_cache_stats: bool = True
) -> Dict[str, Any]:
    """Get comprehensive performance dashboard data

    Args:
        hours_back: Number of hours to look back for data (default: 24)
        include_cache_stats: Whether to include cache performance statistics

    Returns:
        Dict containing comprehensive dashboard data
    """
    try:
        # Validate hours_back parameter
        if hours_back <= 0 or hours_back > 168:  # Max 1 week
            return {
                "error": "Invalid timeframe",
                "message": "hours_back must be between 1 and 168 (1 week)",
            }

        dashboard = get_performance_dashboard()
        return await dashboard.get_comprehensive_dashboard(
            hours_back, include_cache_stats
        )

    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        return {"error": "Failed to get dashboard data", "message": str(e)}


async def get_cost_analysis(hours_back: int = 24) -> Dict[str, Any]:
    """Get detailed cost analysis and breakdown

    Args:
        hours_back: Number of hours to analyze (default: 24)

    Returns:
        Dict containing detailed cost analysis
    """
    try:
        if hours_back <= 0 or hours_back > 168:
            return {
                "error": "Invalid timeframe",
                "message": "hours_back must be between 1 and 168 (1 week)",
            }

        dashboard = get_performance_dashboard()
        full_dashboard = await dashboard.get_comprehensive_dashboard(hours_back, False)

        return {
            "timeframe_hours": hours_back,
            "cost_analysis": full_dashboard.get("cost_analysis", {}),
            "backend_comparison": full_dashboard.get("backend_comparison", {}),
            "trends": full_dashboard.get("trends", {}),
        }

    except Exception as e:
        logger.error(f"Error getting cost analysis: {e}")
        return {"error": "Failed to get cost analysis", "message": str(e)}


async def get_performance_metrics(hours_back: int = 24) -> Dict[str, Any]:
    """Get performance metrics and latency analysis

    Args:
        hours_back: Number of hours to analyze (default: 24)

    Returns:
        Dict containing performance metrics and analysis
    """
    try:
        if hours_back <= 0 or hours_back > 168:
            return {
                "error": "Invalid timeframe",
                "message": "hours_back must be between 1 and 168 (1 week)",
            }

        dashboard = get_performance_dashboard()
        full_dashboard = await dashboard.get_comprehensive_dashboard(hours_back, False)

        return {
            "timeframe_hours": hours_back,
            "performance": full_dashboard.get("performance", {}),
            "efficiency": full_dashboard.get("efficiency", {}),
            "alerts": full_dashboard.get("alerts", {}),
            "backend_comparison": full_dashboard.get("backend_comparison", {}),
        }

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return {"error": "Failed to get performance metrics", "message": str(e)}


async def get_optimization_recommendations(hours_back: int = 24) -> Dict[str, Any]:
    """Get AI-powered optimization recommendations

    Args:
        hours_back: Number of hours to analyze for recommendations (default: 24)

    Returns:
        Dict containing optimization recommendations and insights
    """
    try:
        if hours_back <= 0 or hours_back > 168:
            return {
                "error": "Invalid timeframe",
                "message": "hours_back must be between 1 and 168 (1 week)",
            }

        dashboard = get_performance_dashboard()
        return await dashboard.get_optimization_recommendations(hours_back)

    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        return {
            "error": "Failed to get optimization recommendations",
            "message": str(e),
        }


async def get_detailed_performance_report(
    hours_back: int = 24, format: str = "json"
) -> Dict[str, Any]:
    """Export comprehensive performance report

    Args:
        hours_back: Number of hours to include in report (default: 24)
        format: Report format (currently only "json" supported)

    Returns:
        Dict containing detailed performance report
    """
    try:
        if hours_back <= 0 or hours_back > 168:
            return {
                "error": "Invalid timeframe",
                "message": "hours_back must be between 1 and 168 (1 week)",
            }

        if format not in ["json"]:
            return {
                "error": "Invalid format",
                "message": "Currently only 'json' format is supported",
            }

        dashboard = get_performance_dashboard()
        return await dashboard.export_detailed_report(hours_back, format)

    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        return {"error": "Failed to generate performance report", "message": str(e)}


async def get_session_profiling_data(session_id: str) -> Dict[str, Any]:
    """Get profiling data for a specific session

    Args:
        session_id: Session ID to analyze

    Returns:
        Dict containing session-specific profiling data
    """
    try:
        # Validate session ID
        is_valid, error_msg = validate_session_id(session_id)
        if not is_valid:
            return {"error": "Invalid session ID", "message": error_msg}

        profiler = get_global_profiler()

        # Get all metrics and filter by session
        all_metrics = []
        if not profiler.metrics_history.is_empty():
            all_metrics = profiler.metrics_history.get_all()

        session_metrics = [m for m in all_metrics if m.session_id == session_id]

        if not session_metrics:
            return {
                "session_id": session_id,
                "message": "No profiling data found for this session",
                "total_calls": 0,
            }

        # Calculate session-specific statistics
        total_calls = len(session_metrics)
        successful_calls = sum(1 for m in session_metrics if m.success)
        total_cost = sum(
            m.estimated_cost_usd for m in session_metrics if m.estimated_cost_usd
        )
        total_tokens = sum(m.total_tokens for m in session_metrics if m.total_tokens)

        latencies = [m.total_latency for m in session_metrics if m.total_latency]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Backend usage breakdown
        backend_usage = {}
        for metric in session_metrics:
            backend_usage[metric.backend] = backend_usage.get(metric.backend, 0) + 1

        # Thread activity
        thread_activity = {}
        for metric in session_metrics:
            thread_name = metric.thread_name
            if thread_name not in thread_activity:
                thread_activity[thread_name] = {
                    "calls": 0,
                    "successful": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "avg_latency": 0.0,
                }

            activity = thread_activity[thread_name]
            activity["calls"] += 1
            if metric.success:
                activity["successful"] += 1
            if metric.estimated_cost_usd:
                activity["total_cost"] += metric.estimated_cost_usd
            if metric.total_tokens:
                activity["total_tokens"] += metric.total_tokens

        # Calculate averages for thread activity
        for thread_name, activity in thread_activity.items():
            thread_latencies = [
                m.total_latency
                for m in session_metrics
                if m.thread_name == thread_name and m.total_latency
            ]
            activity["avg_latency"] = (
                sum(thread_latencies) / len(thread_latencies) if thread_latencies else 0
            )

        return {
            "session_id": session_id,
            "summary": {
                "total_calls": total_calls,
                "success_rate": round((successful_calls / total_calls) * 100, 1)
                if total_calls > 0
                else 0,
                "total_cost_usd": round(total_cost, 4),
                "total_tokens": total_tokens,
                "avg_latency_seconds": round(avg_latency, 3),
                "session_duration": (
                    session_metrics[-1].timestamp - session_metrics[0].timestamp
                ).total_seconds(),
            },
            "backend_usage": backend_usage,
            "thread_activity": thread_activity,
            "timeline": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "thread_name": m.thread_name,
                    "backend": m.backend,
                    "success": m.success,
                    "latency": m.total_latency,
                    "cost": m.estimated_cost_usd,
                    "tokens": m.total_tokens,
                    "error": m.error_type if not m.success else None,
                }
                for m in sorted(session_metrics, key=lambda x: x.timestamp)[
                    -20:
                ]  # Last 20 calls
            ],
        }

    except Exception as e:
        logger.error(f"Error getting session profiling data: {e}")
        return {"error": "Failed to get session profiling data", "message": str(e)}


async def reset_profiling_data() -> Dict[str, Any]:
    """Reset all profiling data (admin operation)

    WARNING: This will clear all collected profiling metrics!

    Returns:
        Dict containing reset confirmation
    """
    try:
        profiler = get_global_profiler()

        # Get current status before reset
        old_status = profiler.get_configuration_status()

        # Clear metrics history
        profiler.metrics_history.clear()
        profiler._total_calls = 0
        profiler._profiled_calls = 0

        # Clear dashboard cache
        dashboard = get_performance_dashboard()
        dashboard._cache.clear()

        return {
            "status": "success",
            "message": "All profiling data has been reset",
            "previous_stats": {
                "total_calls": old_status.get("statistics", {}).get("total_calls", 0),
                "profiled_calls": old_status.get("statistics", {}).get(
                    "profiled_calls", 0
                ),
                "storage_usage": old_status.get("storage", {}).get("current_usage", 0),
            },
            "warning": "This operation cannot be undone - all historical profiling data has been lost",
        }

    except Exception as e:
        logger.error(f"Error resetting profiling data: {e}")
        return {"error": "Failed to reset profiling data", "message": str(e)}


async def configure_profiling(
    enabled: Optional[bool] = None,
    sampling_rate: Optional[float] = None,
    track_costs: Optional[bool] = None,
    track_memory: Optional[bool] = None,
) -> Dict[str, Any]:
    """Update profiling configuration

    Args:
        enabled: Enable/disable profiling
        sampling_rate: Sampling rate (0.0 to 1.0)
        track_costs: Enable/disable cost tracking
        track_memory: Enable/disable memory tracking

    Returns:
        Dict containing updated configuration
    """
    try:
        profiler = get_global_profiler()

        # Update configuration
        if enabled is not None:
            profiler.config.enabled = enabled

        if sampling_rate is not None:
            if not (0.0 <= sampling_rate <= 1.0):
                return {
                    "error": "Invalid sampling rate",
                    "message": "sampling_rate must be between 0.0 and 1.0",
                }
            profiler.config.sampling_rate = sampling_rate

        if track_costs is not None:
            profiler.config.track_costs = track_costs

        if track_memory is not None:
            profiler.config.track_memory = track_memory

        return {
            "status": "success",
            "message": "Profiling configuration updated",
            "updated_config": profiler.get_configuration_status(),
        }

    except Exception as e:
        logger.error(f"Error configuring profiling: {e}")
        return {"error": "Failed to configure profiling", "message": str(e)}


def register_profiling_tools(mcp: FastMCP) -> None:
    """Register all profiling tools with the MCP server"""

    @mcp.tool(
        description="Get current LLM profiling configuration and status - "
        "shows sampling rates, feature flags, storage utilization, and statistics"
    )
    async def get_profiling_status() -> Dict[str, Any]:
        """Get profiling status and configuration"""
        return await get_llm_profiling_status()

    @mcp.tool(
        description="Get comprehensive performance dashboard with cost analysis, "
        "latency metrics, efficiency insights, and backend comparison - "
        "your mission control for LLM operations"
    )
    async def get_performance_dashboard(
        request: PerformanceDashboardRequest,
    ) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        return await get_performance_dashboard_data(
            request.hours_back, request.include_cache_stats
        )

    @mcp.tool(
        description="Deep dive into cost breakdown by backend, model, and session - "
        "identify expensive operations and track daily burn rates for budget control"
    )
    async def get_cost_analysis_data(request: CostAnalysisRequest) -> Dict[str, Any]:
        """Get detailed cost analysis and breakdown"""
        return await get_cost_analysis(request.hours_back)

    @mcp.tool(
        description="Analyze performance metrics including latency percentiles, "
        "throughput, error rates, and backend efficiency comparison"
    )
    async def get_performance_analysis(
        request: PerformanceMetricsRequest,
    ) -> Dict[str, Any]:
        """Get performance metrics and latency analysis"""
        return await get_performance_metrics(request.hours_back)

    @mcp.tool(
        description="Get AI-powered optimization recommendations based on usage patterns - "
        "discover cost savings, performance improvements, and efficiency gains"
    )
    async def get_optimization_insights(request: OptimizationRequest) -> Dict[str, Any]:
        """Get optimization recommendations and insights"""
        return await get_optimization_recommendations(request.hours_back)

    @mcp.tool(
        description="Export comprehensive performance report with executive summary, "
        "detailed analysis, recommendations, and action items for stakeholders"
    )
    async def export_performance_report(
        request: DetailedReportRequest,
    ) -> Dict[str, Any]:
        """Export detailed performance report"""
        return await get_detailed_performance_report(request.hours_back, request.format)

    @mcp.tool(
        description="Analyze profiling data for a specific session - "
        "view session timeline, backend usage, thread activity, and costs"
    )
    async def get_session_profiling_analysis(
        request: SessionProfilingRequest,
    ) -> Dict[str, Any]:
        """Get profiling data for specific session"""
        return await get_session_profiling_data(request.session_id)

    @mcp.tool(
        description="ADMIN: Update profiling configuration including sampling rates "
        "and feature flags - changes apply immediately to new operations"
    )
    async def configure_profiling_settings(
        request: ProfilingConfigRequest,
    ) -> Dict[str, Any]:
        """Update profiling configuration"""
        return await configure_profiling(
            request.enabled,
            request.sampling_rate,
            request.track_costs,
            request.track_memory,
        )

    @mcp.tool(
        description="ADMIN: Reset all profiling data - WARNING: This permanently "
        "deletes all collected metrics and cannot be undone!"
    )
    async def reset_profiling_metrics() -> Dict[str, Any]:
        """Reset all profiling data (admin operation)"""
        return await reset_profiling_data()
