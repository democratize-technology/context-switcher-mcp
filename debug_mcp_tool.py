#!/usr/bin/env python3
"""
Debug script to trace the exact point where the Any() instantiation error occurs
"""

import traceback
import asyncio
import json


async def trace_mcp_tool_execution():
    """Trace the MCP tool execution to find where Any() is being instantiated"""
    print("Tracing MCP tool execution...")

    try:
        # Create the full server to initialize all global components
        from src.context_switcher_mcp import create_server

        server = create_server()
        print("✓ Server created successfully")

        # Get the tool function directly from the server
        for tool_name, tool_info in (
            server._tools.items() if hasattr(server, "_tools") else []
        ):
            print(f"Found tool: {tool_name}")

        # Let's try accessing the tool through the server's interface
        # First, let me check what methods the server has
        print(f"Server methods: {[m for m in dir(server) if not m.startswith('_')]}")

        # Try to manually call the admin tool function
        from src.context_switcher_mcp.tools.admin_tools import register_admin_tools
        from mcp.server.fastmcp import FastMCP

        # Create a minimal test server
        test_mcp = FastMCP("debug")
        register_admin_tools(test_mcp)

        print("✓ Test MCP server created with admin tools")

        # Now let's try to find and call the tool
        # Check if we can access the registered tools
        if hasattr(test_mcp, "tools"):
            print(f"Found tools attribute: {test_mcp.tools}")

        # Let's manually import and test the function being called within get_performance_metrics
        print("\n=== TESTING COMPONENTS INDIVIDUALLY ===")

        # Test session_manager.get_stats()
        from src.context_switcher_mcp import session_manager

        if session_manager:
            print("Testing session_manager.get_stats()...")
            session_stats = await session_manager.get_stats()
            print(f"✓ Session stats retrieved: {type(session_stats)}")

            # Try to serialize session stats to JSON to see if that causes issues
            try:
                json.dumps(session_stats)
                print("✓ Session stats JSON serializable")
            except Exception as json_error:
                print(f"✗ Session stats JSON serialization failed: {json_error}")
                traceback.print_exc()

        # Test orchestrator.get_performance_metrics()
        from src.context_switcher_mcp import orchestrator

        if orchestrator:
            print("Testing orchestrator.get_performance_metrics()...")
            orch_metrics = await orchestrator.get_performance_metrics(last_n=20)
            print(f"✓ Orchestrator metrics retrieved: {type(orch_metrics)}")

            # Try to serialize orchestrator metrics to JSON
            try:
                json.dumps(orch_metrics)
                print("✓ Orchestrator metrics JSON serializable")
            except Exception as json_error:
                print(f"✗ Orchestrator metrics JSON serialization failed: {json_error}")
                traceback.print_exc()

                # Let's examine what's in the metrics that might be causing issues
                print(
                    f"Orchestrator metrics keys: {list(orch_metrics.keys()) if isinstance(orch_metrics, dict) else 'Not a dict'}"
                )
                if isinstance(orch_metrics, dict):
                    for key, value in orch_metrics.items():
                        print(f"  {key}: {type(value)} = {value}")
                        try:
                            json.dumps({key: value})
                            print(f"    ✓ {key} is JSON serializable")
                        except Exception as item_error:
                            print(f"    ✗ {key} is NOT JSON serializable: {item_error}")

        # Test the full combined result
        print("\n=== TESTING FULL COMBINED RESULT ===")
        combined_result = {
            "orchestrator": orch_metrics,
            "session_manager": session_stats,
            "system_health": {
                "active_sessions": session_stats["active_sessions"],
                "capacity_utilization": session_stats["capacity_used"],
                "circuit_breaker_issues": any(
                    cb["state"] != "CLOSED"
                    for cb in orch_metrics.get("circuit_breakers", {}).values()
                ),
            },
        }

        try:
            json.dumps(combined_result)
            print("✓ Combined result is JSON serializable")
        except Exception as combined_error:
            print(f"✗ Combined result JSON serialization failed: {combined_error}")
            traceback.print_exc()

    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(trace_mcp_tool_execution())
