#!/usr/bin/env python3
"""
Debug script to test MCP tool calling directly through server.call_tool()
"""

import traceback
import asyncio


async def test_call_tool():
    """Test calling the tool through the MCP server interface"""
    print("Testing MCP server call_tool...")

    try:
        # Create the full server
        from src.context_switcher_mcp import create_server

        server = create_server()
        print("✓ Server created successfully")

        # List available tools
        tools = await server.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")

        # Check if get_performance_metrics is in the list
        perf_tool = None
        for tool in tools:
            if tool.name == "get_performance_metrics":
                perf_tool = tool
                print(f"✓ Found get_performance_metrics tool: {tool.description}")
                break

        if not perf_tool:
            print("✗ get_performance_metrics tool not found in server tools")
            return

        # Try to call the tool directly
        print("Calling get_performance_metrics tool...")
        result = await server.call_tool("get_performance_metrics", {})
        print(f"✓ Tool call successful: {type(result)}")
        print(f"Result: {result}")

    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print(f"Error type: {type(e)}")
        print("\nFull traceback:")
        traceback.print_exc()

        # Check if this is specifically the Any() instantiation error
        error_str = str(e).lower()
        if "cannot instantiate" in error_str and "typing.any" in error_str:
            print("\n🎯 FOUND THE TARGET ERROR!")
            print(
                "This is the exact 'Cannot instantiate typing.Any' error we're looking for."
            )

            # Now let's extract more details from the traceback
            import sys

            exc_type, exc_value, exc_traceback = sys.exc_info()

            print("\n=== DETAILED TRACEBACK ANALYSIS ===")
            tb_lines = traceback.format_tb(exc_traceback)
            for i, line in enumerate(tb_lines):
                print(f"Frame {i}: {line.strip()}")

            # Try to find the exact line that's causing the issue
            print("\n=== LOOKING FOR ANY() INSTANTIATION ===")
            while exc_traceback:
                frame = exc_traceback.tb_frame
                filename = frame.f_code.co_filename
                lineno = exc_traceback.tb_lineno
                print(f"File: {filename}:{lineno}")

                # Try to read the problematic line
                try:
                    with open(filename, "r") as f:
                        lines = f.readlines()
                        if lineno <= len(lines):
                            problematic_line = lines[lineno - 1].strip()
                            print(f"  Line {lineno}: {problematic_line}")

                            # Check if this line has Any() instantiation
                            if "Any(" in problematic_line:
                                print(
                                    f"  🎯 FOUND ANY() INSTANTIATION: {problematic_line}"
                                )
                except Exception:
                    pass

                exc_traceback = exc_traceback.tb_next


if __name__ == "__main__":
    asyncio.run(test_call_tool())
