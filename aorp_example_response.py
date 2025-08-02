#!/usr/bin/env python3
"""
Example of how AORP transforms Context Switcher MCP responses
"""

import json
import os
import sys

# Add the src directory to the path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "src", "context_switcher_mcp")
)

# Import after path modification is intentional
from context_switcher_mcp import aorp  # noqa: E402


def show_legacy_vs_aorp():
    """Demonstrate the difference between legacy and AORP responses"""

    print("🔄 AORP Transformation Example")
    print("=" * 60)

    # Example legacy response from analyze_from_perspectives
    legacy_response = {
        "session_id": "abc-123",
        "prompt": "Should we implement real-time collaboration?",
        "perspectives": {
            "technical": "Real-time collaboration requires WebSocket connections, operational complexity increases significantly...",
            "business": "Market research shows 73% customer demand for real-time features, competitive advantage potential...",
            "user": "Current workflow involves email back-and-forth, real-time would reduce friction significantly...",
            "risk": "Data consistency challenges, potential security vulnerabilities in real-time sync...",
        },
        "abstained": [],
        "errors": [],
        "summary": {
            "total_perspectives": 4,
            "active_responses": 4,
            "abstentions": 0,
            "errors": 0,
        },
    }

    print("📊 LEGACY RESPONSE:")
    print("Raw data dump - AI has to parse and interpret everything")
    print(json.dumps(legacy_response, indent=2)[:300] + "...")

    print("\n" + "=" * 60)

    # Create AORP response using the actual functions
    confidence = aorp.calculate_analysis_confidence(
        perspectives_responded=4,
        total_perspectives=4,
        error_count=0,
        abstention_count=0,
        response_lengths=[120, 95, 88, 110],
    )

    next_steps = aorp.generate_analysis_next_steps(
        session_state="active",
        perspectives_count=4,
        error_count=0,
        has_synthesis=False,
        confidence=confidence,
    )

    builder = aorp.AORPBuilder()
    aorp_response = (
        builder.status("success")
        .key_insight(
            "Comprehensive analysis from 4 of 4 perspectives reveals strong user demand but significant technical complexity"
        )
        .confidence(confidence)
        .session_id("abc-123")
        .next_steps(next_steps)
        .primary_recommendation(
            "synthesize_perspectives() - Discover patterns and tensions across viewpoints"
        )
        .workflow_guidance(
            "Present key insights to user, then offer synthesis for strategic decision-making"
        )
        .completeness(1.0)
        .reliability(confidence)
        .urgency("medium")
        .indicators(
            active_responses=4,
            abstentions=0,
            errors=0,
            avg_response_length=103,
            perspectives_ready_for_synthesis=True,
        )
        .summary("Multi-perspective analysis: 4 active responses from 4 perspectives")
        .data(legacy_response)
        .metadata(
            operation_type="perspective_analysis",
            prompt_length=len("Should we implement real-time collaboration?"),
            session_analyses_count=1,
        )
        .build()
    )

    print("🚀 AORP RESPONSE:")
    print("AI-optimized structure with immediate actionable insights")
    print()

    print("📍 IMMEDIATE (what AI sees first):")
    immediate = aorp_response["immediate"]
    print(f"  Status: {immediate['status']}")
    print(f"  Key Insight: {immediate['key_insight']}")
    print(f"  Confidence: {immediate['confidence']:.1%}")
    print(f"  Session: {immediate['session_id']}")

    print("\n⚡ ACTIONABLE (what AI should do):")
    actionable = aorp_response["actionable"]
    print("  Next Steps:")
    for i, step in enumerate(actionable["next_steps"], 1):
        print(f"    {i}. {step}")
    print(f"  Primary Rec: {actionable['recommendations']['primary']}")
    print(f"  Workflow: {actionable['workflow_guidance']}")

    print("\n📊 QUALITY (decision confidence):")
    quality = aorp_response["quality"]
    print(f"  Completeness: {quality['completeness']:.1%}")
    print(f"  Reliability: {quality['reliability']:.1%}")
    print(f"  Urgency: {quality['urgency']}")
    print(f"  Indicators: {quality['indicators']}")

    print("\n📁 DETAILS (legacy data preserved):")
    print(f"  Summary: {aorp_response['details']['summary']}")
    print("  Legacy Data: Available in 'data' field")
    print(f"  Metadata: {aorp_response['details']['metadata']['operation_type']}")

    print("\n" + "=" * 60)
    print("🎯 IMPACT FOR AI ASSISTANTS:")
    print("✅ Instant understanding: Key insight tells the story immediately")
    print("✅ Clear actions: Next steps guide workflow without guesswork")
    print("✅ Confidence aware: AI knows when to proceed vs. gather more data")
    print("✅ Context preserved: Workflow guidance helps AI help users better")
    print(
        "✅ Backward compatible: Legacy data still accessible for existing integrations"
    )
    print("✅ Quality indicators: AI can assess response reliability automatically")


def show_error_response_improvement():
    """Show how error responses are improved with AORP"""

    print("\n\n🚨 ERROR RESPONSE COMPARISON")
    print("=" * 60)

    print("📊 LEGACY ERROR:")
    legacy_error = {"error": "Session not found"}
    print(json.dumps(legacy_error, indent=2))

    print("\n🚀 AORP ERROR:")
    aorp_error = aorp.create_error_response(
        error_message="Session not found",
        error_type="session_not_found",
        context={"session_id": "invalid-123"},
        recoverable=True,
    )

    print("📍 IMMEDIATE:")
    print(f"  Status: {aorp_error['immediate']['status']}")
    print(f"  Key Insight: {aorp_error['immediate']['key_insight']}")
    print(f"  Confidence: {aorp_error['immediate']['confidence']}")

    print("\n⚡ ACTIONABLE:")
    print("  Next Steps:")
    for i, step in enumerate(aorp_error["actionable"]["next_steps"], 1):
        print(f"    {i}. {step}")
    print(f"  Primary Rec: {aorp_error['actionable']['recommendations']['primary']}")

    print("\n📊 QUALITY:")
    print(f"  Urgency: {aorp_error['quality']['urgency']}")
    print(f"  Indicators: {aorp_error['quality']['indicators']}")

    print("\n🎯 ERROR IMPROVEMENT:")
    print("✅ Actionable recovery: AI knows exactly how to help user recover")
    print("✅ Context preserved: Error details available for debugging")
    print("✅ Consistent structure: Same AORP format as success responses")
    print("✅ Recovery guidance: Clear path forward instead of dead end")


if __name__ == "__main__":
    show_legacy_vs_aorp()
    show_error_response_improvement()

    print("\n\n🎉 AORP IMPLEMENTATION COMPLETE!")
    print("🚀 Context Switcher MCP now provides AI-optimized responses")
    print("✨ Better user experience through smarter AI interactions")
