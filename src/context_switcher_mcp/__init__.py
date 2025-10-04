#!/usr/bin/env python3
"""
Context-Switcher MCP Server
Multi-perspective analysis using thread orchestration
"""

from typing import Any

# Basic imports that don't require MCP/Pydantic
from .logging_base import get_logger, setup_logging

__all__ = ["create_server"]

# Setup unified logging configuration
setup_logging()
logger = get_logger(__name__)

# Global variables for server components (initialized in create_server)
mcp = None
config = None
session_manager = None
orchestrator = None


def create_server(host: str = "127.0.0.1", port: int = 8082):
    """Create and configure the MCP server with all dependencies

    Args:
        host: Host to bind to for HTTP transport (default: 127.0.0.1 - localhost only)
        port: Port to bind to for HTTP transport (default: 8082 - standard MCP HTTP port)
    """
    global mcp, config, session_manager, orchestrator  # noqa: PLW0603

    try:
        # Import MCP and other dependencies only when actually creating server
        from mcp.server.fastmcp import FastMCP
        from pydantic import BaseModel, Field

        from .config import get_config
        from .handlers.session_handler import SessionHandler
        from .handlers.validation_handler import ValidationHandler
        from .models import ModelBackend
        from .perspective_orchestrator import PerspectiveOrchestrator
        from .session_manager import SessionManager
        from .tools.admin_tools import register_admin_tools
        from .tools.analysis_tools import register_analysis_tools
        from .tools.perspective_tools import register_perspective_tools
        from .tools.profiling_tools import register_profiling_tools
        from .tools.session_tools import register_session_tools

        # Initialize server components with HTTP settings
        mcp = FastMCP("context-switcher", host=host, port=port)
        config = get_config()
        session_manager = SessionManager(
            max_sessions=config.session.max_active_sessions,
            session_ttl_hours=config.session.default_ttl_hours,
            cleanup_interval_minutes=config.session.cleanup_interval_seconds // 60,
        )
        orchestrator = PerspectiveOrchestrator()

        # Request models for tools
        class StartContextAnalysisRequest(BaseModel):
            topic: str = Field(description="The decision topic or question to analyze")
            initial_perspectives: list[str] | None = Field(
                default=None,
                description="List of perspective names to use (defaults to: technical, business, user, risk)",
            )
            template: str | None = Field(
                default=None,
                description=(
                    "Pre-configured perspective template: architecture_decision, "
                    "feature_evaluation, debugging_analysis, api_design, security_review"
                ),
            )
            model_backend: ModelBackend = Field(default=ModelBackend.BEDROCK, description="LLM backend to use")
            model_name: str | None = Field(default=None, description="Specific model to use")

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
        ) -> dict[str, Any]:
            """Initialize a new context-switching analysis session"""
            (
                is_valid,
                error_response,
            ) = ValidationHandler.validate_session_creation_request(request.topic, request.initial_perspectives)
            if not is_valid:
                return error_response
            return await SessionHandler.create_session(
                topic=request.topic,
                initial_perspectives=request.initial_perspectives,
                template=request.template,
                model_backend=request.model_backend,
                model_name=request.model_name,
            )

        logger.info("MCP server created and configured successfully")
        return mcp

    except ImportError as e:
        logger.error(f"Failed to create server - missing dependencies: {e}")
        raise ImportError(
            f"Cannot create MCP server - missing dependencies: {e}. "
            "Please install required packages with: pip install -e ."
        ) from e
    except Exception as e:
        logger.error(f"Failed to create server: {e}")
        raise
