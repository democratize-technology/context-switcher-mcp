"""Analysis tools for Context-Switcher MCP Server"""

from datetime import datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..aorp import (
    AORPBuilder,
    create_error_response,
    generate_synthesis_next_steps,
)
from ..exceptions import (
    OrchestrationError,
    SessionNotFoundError,
)
from ..helpers.analysis_helpers import (
    build_analysis_aorp_response,
    validate_analysis_request,
)
from ..logging_config import get_logger
from ..logging_utils import get_request_logger
from ..rate_limiter import SessionRateLimiter
from ..security import sanitize_error_message
from ..validation import validate_session_id

logger = get_logger(__name__)
request_logger = get_request_logger()

# Initialize rate limiter (shared across analysis operations)
rate_limiter = SessionRateLimiter()


def safe_truncate_string(text: str, max_length: int) -> str:
    """Safely truncate a string without breaking Unicode escape sequences"""
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]

    # Check if we potentially cut off a Unicode escape sequence
    # Look for backslashes near the end that might be part of an escape
    for i in range(min(10, len(truncated))):  # Check last 10 chars
        pos = len(truncated) - 1 - i
        if pos < 0:
            break
        if truncated[pos] == "\\":
            # Found a backslash, check if it might be part of an escape
            remaining = truncated[pos:]
            if (
                (remaining.startswith("\\U") and len(remaining) < 10)
                or (remaining.startswith("\\u") and len(remaining) < 6)
                or (remaining.startswith("\\x") and len(remaining) < 4)
                or (remaining.startswith("\\") and len(remaining) < 2)
            ):
                # Potential incomplete escape, cut before it
                truncated = truncated[:pos] + "..."
                break

    return truncated


class AnalyzeFromPerspectivesRequest(BaseModel):
    session_id: str = Field(description="Session ID for analysis")
    prompt: str = Field(description="The specific question or topic to analyze")


class AnalyzeFromPerspectivesStreamRequest(BaseModel):
    session_id: str = Field(description="Session ID for analysis")
    prompt: str = Field(description="The specific question or topic to analyze")


class SynthesizePerspectivesRequest(BaseModel):
    session_id: str = Field(description="Session ID to synthesize")


class CheckConvergenceRequest(BaseModel):
    session_id: str = Field(description="Session ID to check convergence for")


def register_analysis_tools(mcp: FastMCP) -> None:
    """Register analysis tools with the MCP server"""

    @mcp.tool(
        description="When you need parallel insights NOW - broadcast your question to all perspectives simultaneously. Expect 10-30 seconds for comprehensive analysis. Perspectives can abstain with [NO_RESPONSE] if not relevant"
    )
    async def analyze_from_perspectives(
        request: AnalyzeFromPerspectivesRequest,
    ) -> dict[str, Any]:
        """Broadcast a prompt to all perspectives and collect their responses"""

        # Validate the analysis request (rate limits, session, prompt)
        is_valid, error_response = await validate_analysis_request(
            request.session_id, request.prompt, rate_limiter
        )
        if not is_valid:
            return error_response

        # Get session
        from .. import session_manager

        if session_manager is None:
            return create_error_response(
                "Session manager not initialized",
                error_code="session_manager_unavailable",
            )

        session = await session_manager.get_session(request.session_id)

        # Initialize orchestrator
        from ..perspective_orchestrator import PerspectiveOrchestrator

        orchestrator = PerspectiveOrchestrator()

        try:
            # Execute analysis across all perspectives, passing the session topic for context
            results = await orchestrator.broadcast_to_perspectives(
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_count": active_count,
                    "abstained_count": abstained_count,
                    "error_count": error_count,
                }
            )

            # Check for context convergence after storing analysis
            try:
                from ..convergence import check_context_convergence

                converged, alignment_score = await check_context_convergence(
                    session.analyses, request.session_id
                )

                if converged:
                    logger.info(
                        f"Contexts aligning at {alignment_score:.3f} for session {request.session_id}"
                    )
                    # Add convergence information to the latest analysis
                    session.analyses[-1]["convergence"] = {
                        "converged": True,
                        "alignment_score": alignment_score,
                        "message": f"Contexts converged at {alignment_score:.3f} - perspectives are aligning",
                    }
                else:
                    logger.debug(
                        f"Context alignment: {alignment_score:.3f} (below threshold) for session {request.session_id}"
                    )
                    session.analyses[-1]["convergence"] = {
                        "converged": False,
                        "alignment_score": alignment_score,
                        "message": f"Contexts still diverse - alignment at {alignment_score:.3f}",
                    }

            except Exception as e:
                logger.warning(
                    f"Convergence check failed for session {request.session_id}: {e}"
                )
                # Continue without convergence info rather than failing the entire analysis

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
                    # Add convergence information if available
                    latest_analysis = session.analyses[-1] if session.analyses else {}
                    self.convergence = latest_analysis.get(
                        "convergence",
                        {
                            "converged": False,
                            "alignment_score": 0.0,
                            "message": "Convergence data not available",
                        },
                    )

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
                {
                    "session_id": request.session_id,
                    "prompt": safe_truncate_string(request.prompt, 100),
                },
                recoverable=True,
                session_id=request.session_id,
            )
        except OrchestrationError as e:
            logger.error(f"Orchestration error in analyze_from_perspectives: {e}")
            return create_error_response(
                f"Analysis orchestration failed: {sanitize_error_message(str(e))}",
                "orchestration_error",
                {
                    "session_id": request.session_id,
                    "prompt": safe_truncate_string(request.prompt, 100),
                },
                recoverable=True,
                session_id=request.session_id,
            )
        except ValueError as e:
            logger.error(f"Validation error in analyze_from_perspectives: {e}")
            return create_error_response(
                f"Invalid request: {sanitize_error_message(str(e))}",
                "validation_error",
                {
                    "session_id": request.session_id,
                    "prompt": safe_truncate_string(request.prompt, 100),
                },
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
                {
                    "session_id": request.session_id,
                    "prompt": safe_truncate_string(request.prompt, 100),
                },
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
    ) -> dict[str, Any]:
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

        if session_manager is None:
            return create_error_response(
                "Session manager not initialized",
                error_code="session_manager_unavailable",
            )

        try:
            session = await session_manager.get_session(request.session_id)
        except SessionNotFoundError:
            return create_error_response(
                "Session not found or has expired",
                "session_not_found",
                {"session_id": request.session_id},
                recoverable=True,
            )
        except Exception as e:
            return create_error_response(
                f"Unexpected error accessing session: {str(e)}",
                "synthesis_error",
                {"session_id": request.session_id},
                recoverable=True,
            )

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
            from ..models import ModelBackend
            from ..response_formatter import ResponseFormatter

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

    @mcp.tool(
        description="Check context convergence status for a session - see how aligned different perspectives have become"
    )
    async def check_context_convergence_status(
        request: CheckConvergenceRequest,
    ) -> dict[str, Any]:
        """Check convergence status for a session's analyses"""
        # Validate session ID
        session_valid, session_error = await validate_session_id(
            request.session_id, "check_convergence"
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

        if session_manager is None:
            return create_error_response(
                "Session manager not initialized",
                error_code="session_manager_unavailable",
            )

        session = await session_manager.get_session(request.session_id)

        # Check if we have analyses to check
        if not session.analyses:
            return create_error_response(
                "No analyses found for convergence check. Run analyze_from_perspectives first.",
                "no_data",
                {"session_id": request.session_id},
                recoverable=True,
            )

        try:
            from ..convergence import (
                check_context_convergence,
                context_alignment_detector,
                measure_current_alignment,
            )

            # Check current convergence status
            converged, alignment_score = await check_context_convergence(
                session.analyses, request.session_id
            )

            # Get detailed metrics for the most recent analysis
            latest_analysis = session.analyses[-1]
            latest_responses = latest_analysis.get("results", {})

            convergence_metrics = await measure_current_alignment(
                latest_responses, request.session_id
            )

            # Build AORP response
            builder = AORPBuilder()

            builder.status("success")
            builder.key_insight(
                f"Context alignment: {alignment_score:.3f} - "
                f"{'Converged' if converged else 'Still diverse'}"
            )
            builder.confidence(0.9)  # High confidence in convergence measurement
            builder.session_id(request.session_id)

            builder.summary(
                f"Convergence analysis for {len(session.analyses)} iterations"
            )

            # Determine urgency based on convergence
            urgency = "high" if converged else "medium"

            builder.data(
                {
                    "convergence_status": {
                        "converged": converged,
                        "alignment_score": alignment_score,
                        "threshold": 0.85,
                        "message": convergence_metrics.convergence.get(
                            "message", "No message"
                        )
                        if hasattr(convergence_metrics, "convergence")
                        else "Direct check performed",
                    },
                    "session_stats": {
                        "total_analyses": len(session.analyses),
                        "total_perspectives": len(session.threads),
                        "latest_active_count": latest_analysis.get("active_count", 0),
                        "latest_abstained_count": latest_analysis.get(
                            "abstained_count", 0
                        ),
                    },
                    "cache_stats": context_alignment_detector.get_cache_stats(),
                }
            )

            builder.next_steps(
                [
                    "Continue analysis if not converged"
                    if not converged
                    else "Consider synthesis or decision-making",
                    "Review alignment trends across iterations",
                    "Check individual perspective responses for patterns",
                ]
            )

            builder.primary_recommendation(
                "Proceed with synthesis"
                if converged
                else "Continue multi-perspective analysis"
            )

            builder.workflow_guidance(
                "Convergence detected - perspectives are aligning"
                if converged
                else "Healthy diversity maintained - continue exploring different angles"
            )

            builder.completeness(1.0)
            builder.reliability(0.9)
            builder.urgency(urgency)

            return builder.build()

        except Exception as e:
            logger.error(f"Convergence check failed: {e}", exc_info=True)
            return create_error_response(
                f"Convergence check failed: {sanitize_error_message(str(e))}",
                "convergence_error",
                {"session_id": request.session_id},
                recoverable=True,
                session_id=request.session_id,
            )
