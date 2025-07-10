#!/bin/bash
# Context-Switcher MCP Installer

echo "Installing Context-Switcher MCP..."
echo "================================"

# Install the package
pip install -e .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Installation successful!"
    echo ""
    echo "Next steps:"
    echo "1. Add to your claude_desktop_config.json:"
    echo '   "context-switcher": {'
    echo '     "command": "python",'
    echo '     "args": ["-m", "context_switcher_mcp"]'
    echo '   }'
    echo ""
    echo "2. Restart Claude Desktop"
else
    echo "✗ Installation failed"
    exit 1
fi
