"""Analysis tools for Context-Switcher MCP Server"""

import logging
from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field

from ..aorp import (
    AORPBuilder,
    generate_synthesis_next_steps,
    create_error_response,
)
from ..helpers.analysis_helpers import (
    validate_analysis_request,
    build_analysis_aorp_response,
)
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

        # Initialize orchestrator
        from ..orchestrator import ThreadOrchestrator

        orchestrator = ThreadOrchestrator()

        try:
            # Execute analysis across all perspectives, passing the session topic for context
            results = await orchestrator.broadcast_message(
                session.threads, request.prompt, request.session_id, session.topic
            )

            # Count active vs abstained responses
            active_count = sum(
                1
                for r in results.values()
                if "[NO_RESPONSE]" not in r and not r.startswith("ERROR:")
            )
            abstained_count = sum(1 for r in results.values() if "[NO_RESPONSE]" in r)
            error_count = sum(1 for r in results.values() if r.startswith("ERROR:"))

            # Store analysis in session
            session.analyses.append(
                {
                    "prompt": request.prompt,
                    "results": results,
                    "timestamp": datetime.utcnow().isoformat(),
                    "active_count": active_count,
                    "abstained_count": abstained_count,
                    "error_count": error_count,
                }
            )

            # Calculate confidence
            total_perspectives = len(session.threads)
            confidence = (
                active_count / total_perspectives if total_perspectives > 0 else 0.0
            )

            # Create a results object for the helper function
            class AnalysisResults:
                def __init__(self):
                    self.perspectives = results
                    self.active_count = active_count
                    self.abstained_count = abstained_count
                    self.model_errors = [
                        k for k, v in results.items() if v.startswith("ERROR:")
                    ]
                    self.execution_time = 0.0  # We don't track this yet
                    # Create responses in the expected format
                    self.responses = [
                        {"perspective": k, "content": v} for k, v in results.items()
                    ]

            analysis_results = AnalysisResults()

            # Build AORP response using existing helper
            return build_analysis_aorp_response(
                request.prompt, analysis_results, session, confidence
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
            # Extract the most recent analysis results for synthesis
            # session.analyses is a list where each item contains a 'results' dict
            latest_analysis = session.analyses[-1]  # Get the most recent analysis
            perspectives_data = latest_analysis.get("results", {})

            # Import and use ResponseFormatter for actual synthesis
            from ..response_formatter import ResponseFormatter
            from ..models import ModelBackend

            formatter = ResponseFormatter()

            # Perform the actual synthesis using the ResponseFormatter
            synthesis_result = await formatter.synthesize_responses(
                perspectives_data,
                session_id=request.session_id,
                synthesis_backend=ModelBackend.BEDROCK,  # Default to Bedrock
            )

            # Check if synthesis result is an error response (starts with ERROR: or AORP_ERROR:)
            # Don't use extract_error_info as it incorrectly flags successful content containing error keywords
            if synthesis_result.startswith("ERROR:") or synthesis_result.startswith(
                "AORP_ERROR:"
            ):
                # If synthesis failed, extract the error message
                error_msg = synthesis_result
                if synthesis_result.startswith("ERROR:"):
                    error_msg = synthesis_result[len("ERROR:") :].strip()
                elif synthesis_result.startswith("AORP_ERROR:"):
                    error_msg = synthesis_result[len("AORP_ERROR:") :].strip()

                return create_error_response(
                    error_msg,
                    "synthesis_error",
                    {"session_id": request.session_id},
                    recoverable=True,
                    session_id=request.session_id,
                )

            # Import the synthesis confidence calculation function
            from ..aorp import calculate_synthesis_confidence

            # Calculate synthesis confidence using the correct function
            # Using simple metrics for now - can be enhanced later
            synthesis_confidence = calculate_synthesis_confidence(
                perspectives_analyzed=len(perspectives_data),
                patterns_identified=2,  # Default estimate
                tensions_mapped=1,  # Default estimate
                synthesis_length=len(synthesis_result),
            )

            # Generate next steps for synthesis
            # Using default values for tensions and insights, with calculated confidence
            next_steps = generate_synthesis_next_steps(
                tensions_identified=1,  # Default estimate for tensions
                emergent_insights=len(
                    session.analyses
                ),  # Number of analyses as insight count
                confidence=synthesis_confidence,  # Use the calculated confidence
            )

            # Build AORP response for synthesis
            builder = AORPBuilder()

            # Set required immediate fields
            builder.status("success")
            builder.key_insight(
                f"Synthesized {len(perspectives_data)} perspectives with {synthesis_confidence:.1%} confidence"
            )
            builder.confidence(synthesis_confidence)
            builder.session_id(request.session_id)

            # Set actionable fields
            builder.next_steps(next_steps)
            builder.primary_recommendation(
                "Use synthesis insights for strategic decision-making"
            )
            builder.workflow_guidance(
                "Review synthesis, identify action items, and proceed with implementation"
            )

            # Set quality metrics
            builder.completeness(1.0)  # Synthesis is complete
            builder.reliability(synthesis_confidence)
            builder.urgency("medium" if synthesis_confidence > 0.7 else "low")

            # Set details
            builder.summary(
                f"Cross-perspective synthesis completed for {len(perspectives_data)} perspectives"
            )
            builder.data(
                {
                    "synthesis": synthesis_result,
                    "perspectives_analyzed": len(perspectives_data),
                    "total_perspectives": len(session.threads),
                    "total_analyses": len(session.analyses),
                    "confidence": synthesis_confidence,
                }
            )

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
