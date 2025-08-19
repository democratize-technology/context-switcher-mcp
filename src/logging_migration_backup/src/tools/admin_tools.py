"""Administrative tools for Context-Switcher MCP Server"""

import logging
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

from ..client_binding import client_binding_manager

logger = logging.getLogger(__name__)


def register_admin_tools(mcp: FastMCP) -> None:
    """Register administrative tools with the MCP server"""

    @mcp.tool(description="Get performance metrics and operational health status")
    async def get_performance_metrics() -> Dict[str, Any]:
        """Get performance metrics for monitoring operational health"""
        from .. import orchestrator

        orchestrator_metrics = await orchestrator.get_performance_metrics(last_n=20)
        from .. import session_manager

        session_stats = await session_manager.get_stats()

        return {
            "orchestrator": orchestrator_metrics,
            "session_manager": session_stats,
            "system_health": {
                "active_sessions": session_stats["active_sessions"],
                "capacity_utilization": session_stats["capacity_used"],
                "circuit_breaker_issues": any(
                    cb["state"] != "CLOSED"
                    for cb in orchestrator_metrics.get("circuit_breakers", {}).values()
                ),
            },
        }

    @mcp.tool(description="Reset circuit breakers for model backends (admin operation)")
    async def reset_circuit_breakers() -> Dict[str, Any]:
        """Reset all circuit breakers to restore backend availability"""
        from .. import orchestrator

        reset_status = orchestrator.reset_circuit_breakers()

        return {
            "message": "Circuit breakers reset successfully",
            "reset_status": reset_status,
            "warning": "This is an administrative operation. Monitor backends for continued issues.",
        }

    @mcp.tool(description="Get session security metrics and client binding status")
    async def get_security_metrics() -> Dict[str, Any]:
        """Get comprehensive security metrics for monitoring session security"""
        binding_metrics = client_binding_manager.get_security_metrics()

        from .. import session_manager

        session_stats = await session_manager.get_stats()

        # Count sessions with client binding
        active_sessions = await session_manager.list_active_sessions()
        sessions_with_binding = sum(
            1 for session in active_sessions.values() if session.client_binding
        )
        sessions_without_binding = len(active_sessions) - sessions_with_binding

        # Count suspicious sessions
        suspicious_count = sum(
            1
            for session in active_sessions.values()
            if session.client_binding and session.client_binding.is_suspicious()
        )

        # Security event summary
        security_events = {}
        for session in active_sessions.values():
            for event in session.security_events:
                event_type = event["type"]
                security_events[event_type] = security_events.get(event_type, 0) + 1

        # Build comprehensive metrics
        metrics = {
            "session_security": {
                "total_sessions": len(active_sessions),
                "sessions_with_binding": sessions_with_binding,
                "sessions_without_binding": sessions_without_binding,
                "suspicious_sessions": suspicious_count,
                "binding_coverage_percentage": (
                    sessions_with_binding / len(active_sessions) * 100
                )
                if active_sessions
                else 0,
            },
            "client_binding": binding_metrics,
            "session_manager": session_stats,
            "security_events": security_events,
            "security_status": {
                "overall_security_level": "HIGH"
                if suspicious_count == 0 and sessions_without_binding == 0
                else "MEDIUM"
                if suspicious_count < 2
                else "LOW",
                "security_issues": [
                    f"{suspicious_count} suspicious sessions detected"
                    if suspicious_count > 0
                    else None,
                    f"{sessions_without_binding} sessions without client binding"
                    if sessions_without_binding > 0
                    else None,
                ],
                "recommendations": [
                    "Monitor suspicious sessions closely"
                    if suspicious_count > 0
                    else None,
                    "Investigate unbound sessions"
                    if sessions_without_binding > 0
                    else None,
                    "All sessions properly secured"
                    if suspicious_count == 0 and sessions_without_binding == 0
                    else None,
                ],
            },
        }

        # Clean up None values
        metrics["security_status"]["security_issues"] = [
            issue for issue in metrics["security_status"]["security_issues"] if issue
        ]
        metrics["security_status"]["recommendations"] = [
            rec for rec in metrics["security_status"]["recommendations"] if rec
        ]

        return metrics
