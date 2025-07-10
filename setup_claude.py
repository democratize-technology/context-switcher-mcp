#!/usr/bin/env python3
"""
Context-Switcher MCP Setup Helper
Helps configure the MCP in Claude Desktop
"""

import json
import os
import sys
from pathlib import Path

def find_claude_config():
    """Find Claude Desktop config file"""
    possible_paths = [
        Path.home() / "Library/Application Support/Claude/claude_desktop_config.json",  # macOS
        Path.home() / ".config/Claude/claude_desktop_config.json",  # Linux
        Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json",  # Windows
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    return None

def main():
    print("Context-Switcher MCP Setup Helper")
    print("=" * 40)
    
    # Find config file
    config_path = find_claude_config()
    
    if not config_path:
        print("⚠️  Could not find Claude Desktop config file")
        print("\nPlease manually add to your claude_desktop_config.json:")
        print(json.dumps({
            "mcpServers": {
                "context-switcher": {
                    "command": "python",
                    "args": ["-m", "context_switcher_mcp"]
                }
            }
        }, indent=2))
        return
    
    print(f"✓ Found config at: {config_path}")
    
    # Read existing config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except:
        config = {}
    
    # Add our server
    if "mcpServers" not in config:
        config["mcpServers"] = {}
    
    if "context-switcher" in config["mcpServers"]:
        print("✓ Context-Switcher already configured")
    else:
        config["mcpServers"]["context-switcher"] = {
            "command": "python",
            "args": ["-m", "context_switcher_mcp"]
        }
        
        # Write back
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✓ Added Context-Switcher to config")
    
    print("\n✅ Setup complete! Please restart Claude Desktop.")

if __name__ == "__main__":
    main()
