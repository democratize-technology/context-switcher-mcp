"""MCP tools for LLM profiling and performance monitoring"""

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..llm_profiler import get_global_profiler
from ..logging_config import get_logger
from ..validation import validate_session_id

logger = get_logger(__name__)


# Request models for profiling tools
class SessionProfilingRequest(BaseModel):
    session_id: str = Field(description="Session ID to analyze")


class ProfilingConfigRequest(BaseModel):
    enabled: bool | None = Field(default=None, description="Enable/disable profiling")
    sampling_rate: float | None = Field(
        default=None, description="Sampling rate (0.0 to 1.0)"
    )
    track_costs: bool | None = Field(
        default=None, description="Enable/disable cost tracking"
    )
    track_memory: bool | None = Field(
        default=None, description="Enable/disable memory tracking"
    )


async def get_llm_profiling_status() -> dict[str, Any]:
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


async def get_session_profiling_data(session_id: str) -> dict[str, Any]:
    """Get profiling data for a specific session

    Args:
        session_id: Session ID to analyze

    Returns:
        Dict containing session-specific profiling data
    """
    try:
        # Validate session ID
        is_valid, error_msg = await validate_session_id(session_id, "session_profiling")
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


async def reset_profiling_data() -> dict[str, Any]:
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
    enabled: bool | None = None,
    sampling_rate: float | None = None,
    track_costs: bool | None = None,
    track_memory: bool | None = None,
) -> dict[str, Any]:
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
    async def get_profiling_status() -> dict[str, Any]:
        """Get profiling status and configuration"""
        return await get_llm_profiling_status()

    @mcp.tool(
        description="Analyze profiling data for a specific session - "
        "view session timeline, backend usage, thread activity, and costs"
    )
    async def get_session_profiling_analysis(
        request: SessionProfilingRequest,
    ) -> dict[str, Any]:
        """Get profiling data for specific session"""
        return await get_session_profiling_data(request.session_id)

    @mcp.tool(
        description="ADMIN: Update profiling configuration including sampling rates "
        "and feature flags - changes apply immediately to new operations"
    )
    async def configure_profiling_settings(
        request: ProfilingConfigRequest,
    ) -> dict[str, Any]:
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
    async def reset_profiling_metrics() -> dict[str, Any]:
        """Reset all profiling data (admin operation)"""
        return await reset_profiling_data()
