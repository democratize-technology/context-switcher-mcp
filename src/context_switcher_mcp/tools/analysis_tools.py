"""Analysis tools for Context-Switcher MCP Server"""

import logging
from typing import Dict, Any
from pydantic import BaseModel, Field

from ..aorp import (
    AORPBuilder,
    calculate_analysis_confidence,
    generate_synthesis_next_steps,
    create_error_response,
)
from ..compression import prepare_synthesis_input
from ..confidence_metrics import (
    ConfidenceCalibrator,
    analyze_synthesis_quality,
)
from ..helpers.analysis_helpers import (
    validate_analysis_request,
    build_analysis_aorp_response,
    store_analysis_results,
)
from ..orchestrator import ThreadOrchestrator
from ..rate_limiter import SessionRateLimiter
from ..security import sanitize_error_message
from ..validation import validate_session_id
from ..exceptions import (
    SessionNotFoundError,
    OrchestrationError,
)

logger = logging.getLogger(__name__)

# Initialize rate limiter (shared across analysis operations)
rate_limiter = SessionRateLimiter()


class AnalyzeFromPerspectivesRequest(BaseModel):
    session_id: str = Field(description="Session ID for analysis")
    prompt: str = Field(description="The specific question or topic to analyze")


class AnalyzeFromPerspectivesStreamRequest(BaseModel):
    session_id: str = Field(description="Session ID for analysis")
    prompt: str = Field(description="The specific question or topic to analyze")


class SynthesizePerspectivesRequest(BaseModel):
    session_id: str = Field(description="Session ID to synthesize")


def register_analysis_tools(mcp):
    """Register analysis tools with the MCP server"""

    @mcp.tool(
        description="When you need parallel insights NOW - broadcast your question to all perspectives simultaneously. Expect 10-30 seconds for comprehensive analysis. Perspectives can abstain with [NO_RESPONSE] if not relevant"
    )
    async def analyze_from_perspectives(
        request: AnalyzeFromPerspectivesRequest,
    ) -> Dict[str, Any]:
        """Broadcast a prompt to all perspectives and collect their responses"""
        # Validate the analysis request (rate limits, session, prompt)
        is_valid, error_response = await validate_analysis_request(
            request.session_id, request.prompt, rate_limiter
        )
        if not is_valid:
            return error_response

        # Get session
        from .. import session_manager

        session = await session_manager.get_session(request.session_id)

        # Initialize orchestrator with session threads
        orchestrator = ThreadOrchestrator(list(session.threads.values()))

        try:
            # Execute orchestrated analysis
            results = await orchestrator.execute_prompt(request.prompt)

            # Store analysis results in session
            store_analysis_results(session, request.prompt, results)

            # Calculate confidence score
            confidence = calculate_analysis_confidence(
                results.active_count,
                results.abstained_count,
                len(results.model_errors),
                results.execution_time,
            )

            # Build and return AORP response
            return build_analysis_aorp_response(
                request.prompt, results, session, confidence
            )

        except SessionNotFoundError as e:
            logger.error(f"Session error in analyze_from_perspectives: {e}")
            return create_error_response(
                f"Session error: {sanitize_error_message(str(e))}",
                "session_error",
                {"session_id": request.session_id, "prompt": request.prompt[:100]},
                recoverable=True,
                session_id=request.session_id,
            )
        except OrchestrationError as e:
            logger.error(f"Orchestration error in analyze_from_perspectives: {e}")
            return create_error_response(
                f"Analysis orchestration failed: {sanitize_error_message(str(e))}",
                "orchestration_error",
                {"session_id": request.session_id, "prompt": request.prompt[:100]},
                recoverable=True,
                session_id=request.session_id,
            )
        except ValueError as e:
            logger.error(f"Validation error in analyze_from_perspectives: {e}")
            return create_error_response(
                f"Invalid request: {sanitize_error_message(str(e))}",
                "validation_error",
                {"session_id": request.session_id, "prompt": request.prompt[:100]},
                recoverable=True,
                session_id=request.session_id,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in analyze_from_perspectives: {e}", exc_info=True
            )
            return create_error_response(
                f"Analysis execution failed: {sanitize_error_message(str(e))}",
                "execution_error",
                {"session_id": request.session_id, "prompt": request.prompt[:100]},
                recoverable=True,
                session_id=request.session_id,
            )

    # Note: Streaming tool temporarily disabled due to AsyncGenerator pydantic schema issues
    # Will be re-enabled after resolving schema generation for streaming responses

    @mcp.tool(
        description="When challenges conflict - find patterns, prioritize risks, and discover the meta-vulnerabilities across all adversarial perspectives"
    )
    async def synthesize_perspectives(
        request: SynthesizePerspectivesRequest,
    ) -> Dict[str, Any]:
        """Synthesize insights across all perspective analyses"""
        # Validate session ID with client binding
        session_valid, session_error = await validate_session_id(
            request.session_id, "synthesize_perspectives"
        )
        if not session_valid:
            return create_error_response(
                session_error,
                "session_not_found",
                {"session_id": request.session_id},
                recoverable=True,
            )

        # Get session
        from .. import session_manager

        session = await session_manager.get_session(request.session_id)

        # Check if we have analyses to synthesize
        if not session.analyses:
            return create_error_response(
                "No analyses found for synthesis. Run analyze_from_perspectives first.",
                "no_data",
                {"session_id": request.session_id},
                recoverable=True,
            )

        try:
            # Prepare synthesis input with compression
            _ = prepare_synthesis_input(session.analyses)

            # Initialize confidence calibrator
            calibrator = ConfidenceCalibrator()

            # Perform synthesis analysis
            synthesis_quality = analyze_synthesis_quality(
                len(session.threads),
                len(session.analyses),
                sum(a["active_count"] for a in session.analyses),
                sum(len(a.get("model_errors", [])) for a in session.analyses),
            )

            # Calculate synthesis confidence
            synthesis_confidence = calibrator.calculate_synthesis_confidence(
                perspective_count=len(session.threads),
                analysis_count=len(session.analyses),
                total_responses=sum(a["active_count"] for a in session.analyses),
                error_count=sum(
                    len(a.get("model_errors", [])) for a in session.analyses
                ),
            )

            # Generate next steps for synthesis
            next_steps = generate_synthesis_next_steps(
                len(session.threads), len(session.analyses)
            )

            # Build AORP response for synthesis
            builder = AORPBuilder()
            builder.set_objective(
                "Cross-perspective synthesis and pattern identification"
            )
            builder.set_confidence(synthesis_confidence)

            # Add synthesis insights
            builder.add_recommendation(
                "Pattern Analysis",
                f"Synthesized insights from {len(session.analyses)} analyses across {len(session.threads)} perspectives",
                confidence=synthesis_confidence,
            )

            # Add quality metrics
            builder.add_context("synthesis_quality", str(synthesis_quality))
            builder.add_context("total_analyses", str(len(session.analyses)))
            builder.add_context("total_perspectives", str(len(session.threads)))

            # Set next steps
            for step in next_steps:
                builder.add_next_step(step)

            return builder.build()

        except SessionNotFoundError as e:
            logger.error(f"Session error in synthesize_perspectives: {e}")
            return create_error_response(
                f"Session error: {sanitize_error_message(str(e))}",
                "session_error",
                {"session_id": request.session_id},
                recoverable=True,
                session_id=request.session_id,
            )
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Data error in synthesize_perspectives: {e}")
            return create_error_response(
                f"Invalid synthesis data: {sanitize_error_message(str(e))}",
                "data_error",
                {"session_id": request.session_id},
                recoverable=True,
                session_id=request.session_id,
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in synthesize_perspectives: {e}", exc_info=True
            )
            return create_error_response(
                f"Synthesis failed: {sanitize_error_message(str(e))}",
                "synthesis_error",
                {"session_id": request.session_id},
                recoverable=True,
                session_id=request.session_id,
            )
