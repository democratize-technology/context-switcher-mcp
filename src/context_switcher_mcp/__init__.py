#!/usr/bin/env python3
"""
Context-Switcher MCP Server
Multi-perspective analysis using thread orchestration
"""

from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .config import get_config
from .models import ModelBackend
from .session_manager import SessionManager
from .handlers.session_handler import SessionHandler
from .handlers.validation_handler import ValidationHandler
from .tools.session_tools import register_session_tools
from .tools.analysis_tools import register_analysis_tools
from .tools.perspective_tools import register_perspective_tools
from .tools.admin_tools import register_admin_tools
from .tools.profiling_tools import register_profiling_tools
from .logging_config import setup_logging, get_logger
from .logging_utils import log_operation, correlation_context

__all__ = ["main", "mcp"]

# Setup unified logging configuration
setup_logging()
logger = get_logger(__name__)

mcp = FastMCP("context-switcher")
config = get_config()
session_manager = SessionManager(
    max_sessions=config.session.max_active_sessions,
    session_ttl_hours=config.session.default_ttl_hours,
    cleanup_interval_minutes=config.session.cleanup_interval_seconds // 60,
)

from .perspective_orchestrator import PerspectiveOrchestrator  # noqa: E402

orchestrator = PerspectiveOrchestrator()


# Request models for tools
class StartContextAnalysisRequest(BaseModel):
    topic: str = Field(description="The decision topic or question to analyze")
    initial_perspectives: Optional[List[str]] = Field(
        default=None,
        description="List of perspective names to use (defaults to: technical, business, user, risk)",
    )
    template: Optional[str] = Field(
        default=None,
        description="Pre-configured perspective template: architecture_decision, feature_evaluation, debugging_analysis, api_design, security_review",
    )
    model_backend: ModelBackend = Field(
        default=ModelBackend.BEDROCK, description="LLM backend to use"
    )
    model_name: Optional[str] = Field(default=None, description="Specific model to use")


# Tool registration
register_session_tools(mcp)
register_analysis_tools(mcp)
register_perspective_tools(mcp)
register_admin_tools(mcp)
register_profiling_tools(mcp)


# Session creation tool (main entry point)
@mcp.tool(
    description="When you're at a crossroads and need multiple viewpoints - "
    "analyze architecture decisions, debug blind spots, or evaluate "
    "features from technical, business, user, and risk angles "
    "simultaneously. Templates available: architecture_decision, "
    "feature_evaluation, debugging_analysis, api_design, security_review"
)
async def start_context_analysis(
    request: StartContextAnalysisRequest,
) -> Dict[str, Any]:
    """Initialize a new context-switching analysis session"""
    is_valid, error_response = ValidationHandler.validate_session_creation_request(
        request.topic, request.initial_perspectives
    )
    if not is_valid:
        return error_response
    return await SessionHandler.create_session(
        topic=request.topic,
        initial_perspectives=request.initial_perspectives,
        template=request.template,
        model_backend=request.model_backend,
        model_name=request.model_name,
    )


def main() -> None:
    """Main entry point for the MCP server"""
    mcp.run()
