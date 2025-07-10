#!/usr/bin/env python3
"""
Example usage of Context-Switcher MCP
Demonstrates multi-perspective analysis
"""

import asyncio
import json
from src.context_switcher_mcp import (
    start_context_analysis,
    add_perspective,
    analyze_from_perspectives,
    synthesize_perspectives,
    StartContextAnalysisRequest,
    AddPerspectiveRequest,
    AnalyzeFromPerspectivesRequest,
    SynthesizePerspectivesRequest,
    ModelBackend
)


async def example_architecture_decision():
    """Example: Analyzing a technical architecture decision"""
    print("=== Architecture Decision Analysis ===\n")
    
    # Start analysis session
    start_request = StartContextAnalysisRequest(
        topic="Choosing between microservices and monolithic architecture",
        model_backend=ModelBackend.BEDROCK
    )
    
    result = await start_context_analysis(start_request)
    session_id = result["session_id"]
    print(f"Started session: {session_id}")
    print(f"Initial perspectives: {result['perspectives']}\n")
    
    # Add domain-specific perspectives
    add_request = AddPerspectiveRequest(
        session_id=session_id,
        name="team_capability",
        description="Current team skills, size, and experience with distributed systems"
    )
    await add_perspective(add_request)
    
    add_request = AddPerspectiveRequest(
        session_id=session_id,
        name="operational",
        description="Deployment, monitoring, debugging, and maintenance considerations"
    )
    await add_perspective(add_request)
    
    print("Added perspectives: team_capability, operational\n")
    
    # Analyze the architecture question
    analyze_request = AnalyzeFromPerspectivesRequest(
        session_id=session_id,
        prompt="Should we migrate our e-commerce platform from a monolith to microservices? We have 5 developers, moderate traffic (10k daily users), and need to scale specific features independently."
    )
    
    analysis = await analyze_from_perspectives(analyze_request)
    
    print(f"Active perspectives: {len(analysis['perspectives'])}")
    print(f"Abstained: {analysis['abstained']}\n")
    
    # Show sample responses
    for perspective, response in analysis['perspectives'].items():
        print(f"\n--- {perspective.upper()} PERSPECTIVE ---")
        print(response[:300] + "..." if len(response) > 300 else response)
    
    # Synthesize insights
    synth_request = SynthesizePerspectivesRequest(session_id=session_id)
    synthesis = await synthesize_perspectives(synth_request)
    
    print("\n\n=== SYNTHESIS ===")
    print(synthesis['synthesis'])


async def example_feature_decision():
    """Example: Evaluating a new feature"""
    print("\n\n=== Feature Implementation Analysis ===\n")
    
    # Start focused analysis
    start_request = StartContextAnalysisRequest(
        topic="Real-time collaborative editing feature",
        initial_perspectives=["technical", "user", "business"],
        model_backend=ModelBackend.BEDROCK
    )
    
    result = await start_context_analysis(start_request)
    session_id = result["session_id"]
    
    # Add specific perspective for this feature
    add_request = AddPerspectiveRequest(
        session_id=session_id,
        name="competitive",
        description="How this feature positions us against competitors like Google Docs and Notion"
    )
    await add_perspective(add_request)
    
    # Analyze
    analyze_request = AnalyzeFromPerspectivesRequest(
        session_id=session_id,
        prompt="We're considering adding real-time collaborative editing to our note-taking app. Is this worth the engineering investment?"
    )
    
    analysis = await analyze_from_perspectives(analyze_request)
    
    print(f"Perspectives consulted: {list(analysis['perspectives'].keys())}")
    print(f"\nSummary:")
    print(f"- Total perspectives: {analysis['summary']['total_perspectives']}")
    print(f"- Active responses: {analysis['summary']['active_responses']}")
    print(f"- Abstentions: {analysis['summary']['abstentions']}")


async def example_debugging_blindspot():
    """Example: Finding blind spots in problem analysis"""
    print("\n\n=== Debugging Performance Issue ===\n")
    
    # Start with standard perspectives
    start_request = StartContextAnalysisRequest(
        topic="API performance degradation",
        model_backend=ModelBackend.BEDROCK
    )
    
    result = await start_context_analysis(start_request)
    session_id = result["session_id"]
    
    # Add specialized debugging perspectives
    perspectives_to_add = [
        ("database", "Database query optimization, indexing, and connection pooling"),
        ("infrastructure", "Server resources, network latency, load balancing"),
        ("code_quality", "Algorithm complexity, memory leaks, inefficient patterns"),
        ("external_deps", "Third-party API calls, external service latencies")
    ]
    
    for name, description in perspectives_to_add:
        add_request = AddPerspectiveRequest(
            session_id=session_id,
            name=name,
            description=description
        )
        await add_perspective(add_request)
    
    # Analyze the performance issue
    analyze_request = AnalyzeFromPerspectivesRequest(
        session_id=session_id,
        prompt="Our API endpoints are taking 5-10 seconds to respond during peak hours (2-4 PM). Normal response time is under 500ms. The issue started last week after a deployment."
    )
    
    analysis = await analyze_from_perspectives(analyze_request)
    
    # Show which perspectives found relevant insights
    print("Perspectives with insights:")
    for perspective in analysis['perspectives'].keys():
        print(f"  ✓ {perspective}")
    
    if analysis['abstained']:
        print(f"\nPerspectives that abstained: {analysis['abstained']}")


async def main():
    """Run all examples"""
    print("Context-Switcher MCP Examples\n")
    print("=" * 50)
    
    # Note: These examples won't actually call LLMs without proper setup
    # They demonstrate the API usage and structure
    
    try:
        await example_architecture_decision()
    except Exception as e:
        print(f"Architecture example error: {e}")
    
    try:
        await example_feature_decision()
    except Exception as e:
        print(f"Feature example error: {e}")
    
    try:
        await example_debugging_blindspot()
    except Exception as e:
        print(f"Debugging example error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
