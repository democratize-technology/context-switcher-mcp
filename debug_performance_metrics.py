#!/usr/bin/env python3
"""
Debug script to test the get_performance_metrics function directly
"""

import traceback
import asyncio


async def test_get_performance_metrics():
    """Test the get_performance_metrics function to reproduce the error"""
    print("Testing get_performance_metrics...")

    try:
        # Create the full server to initialize all global components
        print("Creating full MCP server...")
        from src.context_switcher_mcp import create_server

        create_server()
        print("✓ Full server created successfully")

        # Now test the orchestrator directly
        print("Testing orchestrator import after server creation...")
        from src.context_switcher_mcp import orchestrator

        print(f"Orchestrator type: {type(orchestrator)}")

        if orchestrator:
            print("Calling get_performance_metrics on orchestrator...")
            result = await orchestrator.get_performance_metrics(last_n=20)
            print("✓ Function executed successfully!")
            print(
                f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
            )
        else:
            print("✗ Orchestrator is still None")

    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

        # Check if we can import the individual components
        print("\n=== COMPONENT ANALYSIS ===")
        try:
            print("Testing perspective orchestrator import...")
            from src.context_switcher_mcp.perspective_orchestrator import (
                PerspectiveOrchestrator,
            )

            orchestrator_instance = PerspectiveOrchestrator()
            print("✓ PerspectiveOrchestrator imported and instantiated")

            print("Calling get_performance_metrics...")
            result = await orchestrator_instance.get_performance_metrics(last_n=20)
            print(f"✓ Direct call successful: {result}")

        except Exception as comp_error:
            print(f"✗ Component error: {comp_error}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_get_performance_metrics())
