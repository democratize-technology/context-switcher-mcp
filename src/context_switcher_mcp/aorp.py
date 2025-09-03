"""
AI-Optimized Response Protocol (AORP) Implementation
Transforms MCP responses from data dumps into AI workflow enablers
"""

from datetime import UTC, datetime
from typing import Any

from .logging_base import get_logger

logger = get_logger(__name__)

# Constants for quality assessment
MIN_CONFIDENCE_THRESHOLD = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.8
CRITICAL_ERROR_TYPES = ["auth_failure", "service_unavailable", "quota_exceeded"]


class AORPBuilder:
    """Builder for AI-Optimized Response Protocol responses"""

    def __init__(self):
        self.response = {
            "immediate": {},
            "actionable": {},
            "quality": {},
            "details": {},
        }

    def status(self, status: str) -> "AORPBuilder":
        """Set response status: success, error, partial, pending"""
        self.response["immediate"]["status"] = status
        return self

    def key_insight(self, insight: str) -> "AORPBuilder":
        """Set the one-sentence primary takeaway"""
        self.response["immediate"]["key_insight"] = insight
        return self

    def confidence(self, score: float) -> "AORPBuilder":
        """Set confidence score (0.0-1.0)"""
        self.response["immediate"]["confidence"] = max(0.0, min(1.0, score))
        return self

    def session_id(self, session_id: str) -> "AORPBuilder":
        """Set session ID when applicable"""
        self.response["immediate"]["session_id"] = session_id
        return self

    def next_steps(self, steps: list[str]) -> "AORPBuilder":
        """Set prioritized next action items"""
        self.response["actionable"]["next_steps"] = steps
        return self

    def primary_recommendation(self, recommendation: str) -> "AORPBuilder":
        """Set the most important recommendation"""
        if "recommendations" not in self.response["actionable"]:
            self.response["actionable"]["recommendations"] = {}
        self.response["actionable"]["recommendations"]["primary"] = recommendation
        return self

    def secondary_recommendations(self, recommendations: list[str]) -> "AORPBuilder":
        """Set additional recommendations"""
        if "recommendations" not in self.response["actionable"]:
            self.response["actionable"]["recommendations"] = {}
        self.response["actionable"]["recommendations"]["secondary"] = recommendations
        return self

    def workflow_guidance(self, guidance: str) -> "AORPBuilder":
        """Set guidance on how AI should use this information"""
        self.response["actionable"]["workflow_guidance"] = guidance
        return self

    def completeness(self, score: float) -> "AORPBuilder":
        """Set completeness score (0.0-1.0)"""
        self.response["quality"]["completeness"] = max(0.0, min(1.0, score))
        return self

    def reliability(self, score: float) -> "AORPBuilder":
        """Set reliability score (0.0-1.0)"""
        self.response["quality"]["reliability"] = max(0.0, min(1.0, score))
        return self

    def urgency(self, level: str) -> "AORPBuilder":
        """Set urgency level: low, medium, high, critical"""
        valid_levels = ["low", "medium", "high", "critical"]
        if level not in valid_levels:
            level = "medium"
        self.response["quality"]["urgency"] = level
        return self

    def indicators(self, **kwargs) -> "AORPBuilder":
        """Set quality indicators"""
        self.response["quality"]["indicators"] = kwargs
        return self

    def summary(self, summary: str) -> "AORPBuilder":
        """Set human-readable overview"""
        self.response["details"]["summary"] = summary
        return self

    def data(self, data: Any) -> "AORPBuilder":
        """Set full dataset (backward compatibility)"""
        self.response["details"]["data"] = data
        return self

    def metadata(self, **kwargs) -> "AORPBuilder":
        """Set metadata with automatic timestamp"""
        metadata = {"timestamp": datetime.now(UTC).isoformat(), **kwargs}
        self.response["details"]["metadata"] = metadata
        return self

    def debug(self, debug_info: Any) -> "AORPBuilder":
        """Set debug information when needed"""
        self.response["details"]["debug"] = debug_info
        return self

    def build(self) -> dict[str, Any]:
        """Build the final AORP response"""
        # Validate required fields
        immediate = self.response["immediate"]
        if "status" not in immediate:
            raise ValueError("Status is required")
        if "key_insight" not in immediate:
            raise ValueError("Key insight is required")
        if "confidence" not in immediate:
            raise ValueError("Confidence score is required")

        quality = self.response["quality"]
        if "completeness" not in quality:
            quality["completeness"] = 0.5
        if "reliability" not in quality:
            quality["reliability"] = 0.5
        if "urgency" not in quality:
            quality["urgency"] = "medium"

        if "next_steps" not in self.response["actionable"]:
            self.response["actionable"]["next_steps"] = []
        if "recommendations" not in self.response["actionable"]:
            self.response["actionable"]["recommendations"] = {"primary": ""}
        if "workflow_guidance" not in self.response["actionable"]:
            self.response["actionable"]["workflow_guidance"] = ""

        return self.response


def calculate_analysis_confidence(
    perspectives_responded: int,
    total_perspectives: int,
    error_count: int,
    abstention_count: int,
    response_lengths: list[int],
) -> float:
    """Calculate confidence score for analysis responses"""

    if total_perspectives == 0:
        return 0.0

    active_responses = perspectives_responded - abstention_count
    coverage_factor = active_responses / total_perspectives

    error_penalty = max(0, 1 - (error_count * 0.15))

    avg_length = (
        sum(response_lengths) / len(response_lengths) if response_lengths else 0
    )
    quality_factor = min(1.0, max(0.3, avg_length / 300))

    abstention_penalty = max(0.5, 1 - (abstention_count * 0.1))

    confidence = coverage_factor * error_penalty * quality_factor * abstention_penalty

    return min(1.0, max(0.0, confidence))


def calculate_synthesis_confidence(
    perspectives_analyzed: int,
    patterns_identified: int,
    tensions_mapped: int,
    synthesis_length: int,
) -> float:
    """Calculate confidence score for synthesis responses"""

    if perspectives_analyzed == 0:
        return 0.0

    coverage_factor = min(1.0, perspectives_analyzed / 4)  # 4+ perspectives is ideal

    pattern_factor = min(1.0, patterns_identified / 3)  # 3+ patterns is good

    tension_factor = min(1.0, tensions_mapped / 2)  # 2+ tensions shows depth

    depth_factor = min(
        1.0, max(0.2, synthesis_length / 1000)
    )  # 1000+ chars is substantial

    confidence = (coverage_factor + pattern_factor + tension_factor + depth_factor) / 4

    return min(1.0, max(0.0, confidence))


def generate_analysis_next_steps(
    session_state: str,
    perspectives_count: int,
    error_count: int,
    has_synthesis: bool,
    confidence: float,
) -> list[str]:
    """Generate contextual next steps for analysis responses"""

    steps = []

    if error_count > 0:
        steps.append("Check perspective errors and retry if needed")

    if confidence < MIN_CONFIDENCE_THRESHOLD:
        steps.append(
            "add_perspective('<domain>') - Add more perspectives for comprehensive analysis"
        )
        steps.append("Refine question for more targeted insights")

    if confidence >= HIGH_CONFIDENCE_THRESHOLD and not has_synthesis:
        steps.append("synthesize_perspectives() - Discover patterns and tensions")

    if perspectives_count < 4:
        steps.append("add_perspective('<domain>') - Expand analysis coverage")

    if confidence >= MIN_CONFIDENCE_THRESHOLD:
        steps.append(
            "analyze_from_perspectives('<follow-up>') - Explore specific aspects"
        )

    if not steps:
        steps.append("synthesize_perspectives() - Find patterns across viewpoints")

    return steps


def generate_synthesis_next_steps(
    tensions_identified: int, emergent_insights: int, confidence: float
) -> list[str]:
    """Generate next steps for synthesis responses"""

    steps = []

    if tensions_identified > 0:
        steps.append("Address critical tensions through targeted analysis")
        steps.append("Create decision framework for resolving conflicts")

    if emergent_insights > 0:
        steps.append("Explore emergent insights with deep-dive analysis")

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        steps.append("Present findings to stakeholders for decision")
        steps.append("Begin implementation planning based on insights")
    else:
        steps.append("Gather additional perspectives to strengthen analysis")

    steps.append(
        "analyze_from_perspectives('<specific-question>') - Dig deeper into findings"
    )

    return steps


def create_error_response(
    error_message: str,
    error_type: str = "general_error",
    context: dict[str, Any] | None = None,
    recoverable: bool = True,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Create standardized error response using AORP"""

    builder = AORPBuilder()

    urgency = "critical" if error_type in CRITICAL_ERROR_TYPES else "high"

    next_steps = []
    if recoverable:
        if error_type == "session_not_found":
            next_steps = [
                "start_context_analysis() - Create new session",
                "list_sessions() - Check available sessions",
            ]
        elif error_type == "validation_error":
            next_steps = [
                "Review input parameters and retry",
                "Check documentation for correct format",
            ]
        else:
            next_steps = ["Retry the operation", "Check system status and connectivity"]
    else:
        next_steps = [
            "Contact administrator for assistance",
            "Check system documentation for resolution",
        ]

    response = (
        builder.status("error")
        .key_insight(f"Operation failed: {error_message}")
        .confidence(0.0)
        .next_steps(next_steps)
        .primary_recommendation("Address the error before proceeding")
        .workflow_guidance("Present error clearly and guide user to recovery")
        .completeness(0.0)
        .reliability(0.0)
        .urgency(urgency)
        .indicators(
            error_type=error_type, recoverable=recoverable, retry_suggested=recoverable
        )
        .summary(f"Error occurred: {error_message}")
        .data(
            {
                "error_code": error_type,
                "error_message": error_message,
                "context": context or {},
            }
        )
        .metadata(error_id=f"err_{int(datetime.now(UTC).timestamp())}")
    )

    if session_id:
        response.session_id(session_id)

    return response.build()


def convert_legacy_response(
    legacy_response: dict[str, Any], tool_type: str
) -> dict[str, Any]:
    """Convert legacy response format to AORP (backward compatibility)"""

    builder = AORPBuilder()

    if "error" in legacy_response:
        return create_error_response(
            legacy_response["error"], "legacy_error", legacy_response, recoverable=True
        )

    if tool_type == "analysis":
        return _convert_analysis_response(legacy_response, builder)
    elif tool_type == "synthesis":
        return _convert_synthesis_response(legacy_response, builder)
    elif tool_type == "session":
        return _convert_session_response(legacy_response, builder)
    else:
        # Generic conversion
        return _convert_generic_response(legacy_response, builder)


def _convert_analysis_response(
    legacy: dict[str, Any], builder: AORPBuilder
) -> dict[str, Any]:
    """Convert legacy analysis response to AORP"""

    summary = legacy.get("summary", {})
    active_count = summary.get("active_responses", 0)
    total_count = summary.get("total_perspectives", 1)
    error_count = summary.get("errors", 0)

    confidence = calculate_analysis_confidence(
        active_count,
        total_count,
        error_count,
        summary.get("abstentions", 0),
        [100] * active_count,
    )

    if active_count == 0:
        key_insight = "No perspectives provided analysis"
    elif error_count > 0:
        key_insight = f"Partial analysis: {active_count} perspectives responded, {error_count} errors"
    else:
        key_insight = (
            f"Successful analysis from {active_count} of {total_count} perspectives"
        )

    return (
        builder.status("success" if error_count == 0 else "partial")
        .key_insight(key_insight)
        .confidence(confidence)
        .session_id(legacy.get("session_id", ""))
        .next_steps(
            generate_analysis_next_steps(
                "active", total_count, error_count, False, confidence
            )
        )
        .primary_recommendation("Review perspective insights for patterns")
        .workflow_guidance(
            "Present key insights to user, then offer synthesis for deeper patterns"
        )
        .completeness(active_count / total_count)
        .reliability(confidence)
        .urgency("medium")
        .indicators(**summary)
        .summary(f"Analysis completed with {active_count} perspective responses")
        .data(legacy)
        .metadata(operation_type="analysis")
        .build()
    )


def _convert_synthesis_response(
    legacy: dict[str, Any], builder: AORPBuilder
) -> dict[str, Any]:
    """Convert legacy synthesis response to AORP"""

    metadata = legacy.get("metadata", {})
    active_count = metadata.get("active_perspectives", 0)

    synthesis_text = legacy.get("synthesis", "")
    synthesis_length = len(synthesis_text)

    confidence = calculate_synthesis_confidence(active_count, 2, 1, synthesis_length)

    key_insight = f"Synthesis discovered patterns across {active_count} perspectives"

    return (
        builder.status("success")
        .key_insight(key_insight)
        .confidence(confidence)
        .session_id(legacy.get("session_id", ""))
        .next_steps(generate_synthesis_next_steps(1, 1, confidence))
        .primary_recommendation("Use synthesis insights for strategic decision-making")
        .workflow_guidance("Present synthesis as strategic decision framework")
        .completeness(1.0)
        .reliability(confidence)
        .urgency("medium")
        .indicators(perspectives_analyzed=active_count)
        .summary("Strategic synthesis across multiple perspectives")
        .data(legacy)
        .metadata(operation_type="synthesis")
        .build()
    )


def _convert_session_response(
    legacy: dict[str, Any], builder: AORPBuilder
) -> dict[str, Any]:
    """Convert legacy session response to AORP"""

    perspectives = legacy.get("perspectives", [])
    perspective_count = len(perspectives)

    key_insight = f"Session initialized with {perspective_count} perspectives"

    return (
        builder.status("success")
        .key_insight(key_insight)
        .confidence(1.0)
        .session_id(legacy.get("session_id", ""))
        .next_steps(
            [
                "analyze_from_perspectives('<your question>') - Start analysis",
                "add_perspective('<domain>') - Add specialized viewpoint",
            ]
        )
        .primary_recommendation(
            "Begin with a focused question to get targeted insights"
        )
        .workflow_guidance("Guide user to formulate their first analysis question")
        .completeness(1.0)
        .reliability(1.0)
        .urgency("low")
        .indicators(perspectives_configured=perspective_count)
        .summary(
            f"Multi-perspective analysis session ready with {perspective_count} perspectives"
        )
        .data(legacy)
        .metadata(operation_type="session_creation")
        .build()
    )


def _convert_generic_response(
    legacy: dict[str, Any], builder: AORPBuilder
) -> dict[str, Any]:
    """Convert generic legacy response to AORP"""

    return (
        builder.status("success")
        .key_insight("Operation completed successfully")
        .confidence(0.8)
        .next_steps(["Continue with next operation"])
        .primary_recommendation("Review results and proceed")
        .workflow_guidance("Present results to user")
        .completeness(0.8)
        .reliability(0.8)
        .urgency("medium")
        .summary("Operation completed")
        .data(legacy)
        .metadata(operation_type="generic")
        .build()
    )
