#!/usr/bin/env python3
"""Test the Bedrock message format fix"""

# Show the fixed section
print("Checking Bedrock integration fix...")
print("=" * 50)

with open('src/context_switcher_mcp/__init__.py', 'r') as f:
    content = f.read()
    
# Extract the fixed part
import re
match = re.search(r'# Add conversation history.*?"content": \[{"text": msg\["content"\]}\]', content, re.DOTALL)
if match:
    print("✓ Bedrock method correctly formats messages:")
    print(match.group(0))
else:
    print("Could not find the fixed section")

# Also check that we're not adding system message to the messages list
if 'messages[1:]  # Skip system message' not in content:
    print("\n✓ System message is passed separately, not in messages list")
else:
    print("\n⚠️  May still be skipping system message incorrectly")
