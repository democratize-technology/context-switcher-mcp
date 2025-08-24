"""Perspective management tools for Context-Switcher MCP Server"""

from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..aorp import create_error_response
from ..exceptions import (
    ModelBackendError,
    SessionNotFoundError,
)
from ..logging_config import get_logger
from ..models import ModelBackend, Thread
from ..perspective_selector import SmartPerspectiveSelector
from ..security import (
    log_security_event,
    sanitize_error_message,
    validate_perspective_data,
)
from ..validation import validate_session_id

logger = get_logger(__name__)


class AddPerspectiveRequest(BaseModel):
    session_id: str = Field(description="Session ID to add perspective to")
    name: str = Field(description="Name of the new perspective")
    description: str = Field(
        description="Description of what this perspective should focus on"
    )
    custom_prompt: str | None = Field(
        default=None, description="Custom system prompt for this perspective"
    )


class RecommendPerspectivesRequest(BaseModel):
    topic: str = Field(description="The problem or topic to analyze")
    context: str | None = Field(
        default=None, description="Additional context about the problem"
    )
    max_recommendations: int = Field(
        default=5, description="Maximum number of recommendations"
    )
    existing_session_id: str | None = Field(
        default=None, description="Session ID to add perspectives to"
    )


async def add_perspective(request: AddPerspectiveRequest) -> dict[str, Any]:
    """Add a new perspective to an existing analysis session"""
    # Validate session ID with client binding
    session_valid, session_error = await validate_session_id(
        request.session_id, "add_perspective"
    )
    if not session_valid:
        return create_error_response(
            session_error,
            "session_not_found",
            {"session_id": request.session_id},
            recoverable=True,
        )

    # Enhanced validation for perspective data
    validation_result = validate_perspective_data(
        request.name, request.description, request.custom_prompt
    )
    if not validation_result.is_valid:
        log_security_event(
            "perspective_validation_failure",
            {
                "session_id": request.session_id,
                "perspective_name": request.name[:50],  # Truncate for logging
                "issues": validation_result.issues,
                "risk_level": validation_result.risk_level,
                "blocked_patterns": len(validation_result.blocked_patterns),
            },
            session_id=request.session_id,
        )
        return create_error_response(
            f"Invalid perspective data: {sanitize_error_message(validation_result.issues[0] if validation_result.issues else 'Security check failed')}",
            "validation_error",
            {
                "session_id": request.session_id,
                "validation_issues": len(validation_result.issues),
                "risk_level": validation_result.risk_level,
            },
            recoverable=True,
            session_id=request.session_id,
        )

    # Get session
    from .. import session_manager

    if session_manager is None:
        return create_error_response(
            "Session manager not initialized", error_code="session_manager_unavailable"
        )

    session = await session_manager.get_session(request.session_id)

    # Create prompt for new perspective
    if request.custom_prompt:
        system_prompt = request.custom_prompt
    else:
        system_prompt = f"""You are a {request.name} expert. {request.description}

**Analysis approach:**
- Focus specifically on {request.name} considerations
- Provide actionable insights and recommendations
- Consider both immediate and long-term implications
- Be specific and concrete in your analysis

**Abstention criteria:** Return [NO_RESPONSE] if the topic has no relevance to {request.name} considerations or if you cannot provide meaningful insights from this perspective."""

    # Create new thread
    new_thread = Thread(
        id=str(uuid4()),
        name=request.name,
        system_prompt=system_prompt,
        model_backend=ModelBackend.BEDROCK,  # Default backend
        model_name=None,
    )

    # Add to session
    session.threads[request.name] = new_thread

    return {
        "success": True,
        "session_id": request.session_id,
        "perspective_added": request.name,
        "thread_id": new_thread.id,
        "total_perspectives": len(session.threads),
        "next_steps": [
            "analyze_from_perspectives - Test the new perspective",
            "add_perspective - Add another perspective if needed",
            "synthesize_perspectives - Find patterns across all viewpoints",
        ],
    }


async def recommend_perspectives(
    request: RecommendPerspectivesRequest,
) -> dict[str, Any]:
    """Get AI-powered perspective recommendations for a topic"""
    try:
        selector = SmartPerspectiveSelector()

        # Get recommendations (synchronous call - no await needed)
        recommendations = selector.recommend_perspectives(
            topic=request.topic,
            context=request.context,
            max_recommendations=request.max_recommendations,
        )

        # Convert PerspectiveRecommendation objects to dictionaries
        recommendations_data = [
            {
                "name": rec.name,
                "description": rec.description,
                "relevance_score": rec.relevance_score,
                "reasoning": rec.reasoning,
            }
            for rec in recommendations
        ]

        response = {
            "topic": request.topic,
            "recommendations": recommendations_data,
            "total_recommendations": len(recommendations_data),
            "usage_hint": "Use add_perspective with these suggestions or include in start_context_analysis",
        }

        # If session ID provided, offer to add them
        if request.existing_session_id:
            # Validate session exists
            session_valid, session_error = await validate_session_id(
                request.existing_session_id, "recommend_perspectives"
            )
            if session_valid:
                response["session_id"] = request.existing_session_id
                response["next_steps"] = [
                    f"add_perspective(session_id='{request.existing_session_id}', name='{rec.name}', description='{rec.description}') - Add {rec.name} perspective"
                    for rec in recommendations[:3]  # Show top 3
                ]
            else:
                response["session_error"] = session_error

        return response

    except SessionNotFoundError as e:
        logger.error(f"Session error in recommend_perspectives: {e}")
        return create_error_response(
            f"Session error: {sanitize_error_message(str(e))}",
            "session_error",
            {"topic": request.topic},
            recoverable=True,
        )
    except ModelBackendError as e:
        logger.error(f"Model backend error in recommend_perspectives: {e}")
        return create_error_response(
            f"Model error during recommendation: {sanitize_error_message(str(e))}",
            "model_error",
            {"topic": request.topic},
            recoverable=True,
        )
    except ValueError as e:
        logger.error(f"Validation error in recommend_perspectives: {e}")
        return create_error_response(
            f"Invalid request: {sanitize_error_message(str(e))}",
            "validation_error",
            {"topic": request.topic},
            recoverable=True,
        )
    except Exception as e:
        logger.error(f"Unexpected error in recommend_perspectives: {e}", exc_info=True)
        return create_error_response(
            f"Failed to generate recommendations: {sanitize_error_message(str(e))}",
            "recommendation_error",
            {"topic": request.topic},
            recoverable=True,
        )


def register_perspective_tools(mcp: FastMCP) -> None:
    """Register perspective management tools with the MCP server"""

    @mcp.tool(
        description="When the standard perspectives miss something crucial - add specialized lenses like 'performance', 'migration_path', 'customer_support', or any domain-specific viewpoint your problem needs"
    )
    async def add_perspective_tool(request: AddPerspectiveRequest) -> dict[str, Any]:
        """Add a new perspective to an existing analysis session"""
        return await add_perspective(request)

    @mcp.tool(
        description="Get smart perspective recommendations based on problem analysis - AI suggests the most relevant expert viewpoints for your specific challenge"
    )
    async def recommend_perspectives_tool(
        request: RecommendPerspectivesRequest,
    ) -> dict[str, Any]:
        """Get AI-powered perspective recommendations for a topic"""
        return await recommend_perspectives(request)
