#!/usr/bin/env python3
"""Quick validation that the MCP server can start"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.context_switcher_mcp import mcp, main
    print("✓ Successfully imported MCP server")
    print(f"✓ MCP name: {mcp.name}")
    print(f"✓ MCP version: {mcp.version}")
    print("✓ Tools registered:")
    for tool in mcp.list_tools():
        print(f"  - {tool.name}: {tool.description[:60]}...")
    print("\n✓ Context-Switcher MCP is ready to use!")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
