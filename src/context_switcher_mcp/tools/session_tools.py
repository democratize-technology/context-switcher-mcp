"""Session management tools for Context-Switcher MCP Server"""

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..error_helpers import session_not_found_error
from ..logging_config import get_logger
from ..templates import PERSPECTIVE_TEMPLATES
from ..validation import validate_session_id

logger = get_logger(__name__)


def register_session_tools(mcp: FastMCP) -> None:
    """Register session management tools with the MCP server"""

    @mcp.tool(description="List all active context-switching sessions")
    async def list_sessions() -> dict[str, Any]:
        """List all active analysis sessions"""
        from .. import session_manager

        session_list = []
        active_sessions = await session_manager.list_active_sessions()

        for sid, session in active_sessions.items():
            session_list.append(
                {
                    "session_id": sid,
                    "created_at": session.created_at.isoformat(),
                    "perspectives": list(session.threads.keys()),
                    "analyses_count": len(session.analyses),
                    "topic": getattr(session, "topic", "Unknown"),
                }
            )

        stats = await session_manager.get_stats()

        return {
            "sessions": session_list,
            "total_sessions": len(active_sessions),
            "stats": stats,
        }

    @mcp.tool(
        description="See available perspective templates for common analysis patterns - architecture decisions, debugging, API design, and more"
    )
    async def list_templates() -> dict[str, Any]:
        """List all available perspective templates"""
        template_info = {}

        for name, template in PERSPECTIVE_TEMPLATES.items():
            perspectives = template["perspectives"].copy()

            # Add custom perspective names
            for custom_name, _ in template.get("custom", []):
                perspectives.append(f"{custom_name} (custom)")

            template_info[name] = {
                "description": name.replace("_", " ").title(),
                "perspectives": perspectives,
                "total_perspectives": len(template["perspectives"])
                + len(template.get("custom", [])),
            }

        return {
            "templates": template_info,
            "usage": "Use template parameter in start_context_analysis",
            "example": 'start_context_analysis(topic="...", template="architecture_decision")',
        }

    @mcp.tool(
        description="Quick check of your most recent analysis session - see perspectives and results without remembering session ID"
    )
    async def current_session() -> dict[str, Any]:
        """Get information about the most recent session"""
        from .. import session_manager

        active_sessions = await session_manager.list_active_sessions()

        if not active_sessions:
            return {
                "status": "No active sessions",
                "hint": "Start with: start_context_analysis",
            }

        recent_session_id = max(
            active_sessions.keys(), key=lambda sid: active_sessions[sid].created_at
        )
        session = active_sessions[recent_session_id]

        last_analysis = None
        if session.analyses:
            last = session.analyses[-1]
            last_analysis = {
                "prompt": last["prompt"][:100] + "..."
                if len(last["prompt"]) > 100
                else last["prompt"],
                "perspectives_responded": last["active_count"],
                "perspectives_abstained": last["abstained_count"]
                if "abstained_count" in last
                else 0,
            }

        return {
            "session_id": recent_session_id,
            "created": session.created_at.strftime("%H:%M:%S"),
            "topic": getattr(session, "topic", "Unknown"),
            "perspectives": list(session.threads.keys()),
            "total_perspectives": len(session.threads),
            "analyses_run": len(session.analyses),
            "last_analysis": last_analysis,
            "next_steps": [
                "analyze_from_perspectives - Ask a question",
                "add_perspective - Add custom viewpoint",
                "synthesize_perspectives - Find patterns",
            ]
            if last_analysis is None
            else ["synthesize_perspectives - Find patterns across viewpoints"],
        }

    class GetSessionRequest(BaseModel):
        session_id: str = Field(description="Session ID to retrieve")

    @mcp.tool(description="Get details of a specific context-switching session")
    async def get_session(request: GetSessionRequest) -> dict[str, Any]:
        """Get detailed information about a session"""
        # Validate session ID with client binding
        session_valid, session_error = await validate_session_id(
            request.session_id, "get_session"
        )
        if not session_valid:
            return session_not_found_error(request.session_id, session_error)

        from .. import session_manager

        if session_manager is None:
            return session_not_found_error(
                request.session_id, "Session manager not initialized"
            )

        session = await session_manager.get_session(request.session_id)

        # Build legacy response
        legacy_response = {
            "session_id": request.session_id,
            "created_at": session.created_at.isoformat(),
            "perspectives": {
                name: {
                    "id": thread.id,
                    "system_prompt": thread.system_prompt[:200] + "...",
                    "message_count": len(thread.conversation_history),
                    "model_backend": thread.model_backend.value,
                }
                for name, thread in session.threads.items()
            },
            "analyses": [
                {
                    "prompt": a["prompt"],
                    "timestamp": a["timestamp"],
                    "response_count": len(a["results"]),
                    "active_count": a["active_count"],
                }
                for a in session.analyses
            ],
            "topic": getattr(session, "topic", "Unknown"),
            "stats": {
                "total_perspectives": len(session.threads),
                "total_analyses": len(session.analyses),
                "session_duration_minutes": (
                    session.get_duration_minutes()
                    if hasattr(session, "get_duration_minutes")
                    else 0
                ),
            },
        }

        return legacy_response
