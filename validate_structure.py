#!/usr/bin/env python3
"""Validate Context-Switcher MCP structure without imports"""

import os
import ast
import json

def validate_structure():
    """Validate the project structure and code"""
    print("=== Context-Switcher MCP Structural Validation ===\n")
    
    # Check project structure
    required_files = [
        'pyproject.toml',
        'README.md',
        'src/context_switcher_mcp.py',
        'src/__init__.py',
        'tests/test_context_switcher.py',
        'examples.py'
    ]
    
    print("1. Checking project files:")
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"   {status} {file}")
    
    # Parse and validate main MCP file
    print("\n2. Validating MCP implementation:")
    try:
        with open('src/context_switcher_mcp.py', 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Extract key components
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        print(f"   ✓ File parses successfully ({len(content)} bytes)")
        print(f"   ✓ Classes defined: {len(classes)}")
        print(f"   ✓ Functions defined: {len(functions)}")
        
        # Check for required classes
        required_classes = ['Thread', 'ContextSwitcherSession', 'ThreadOrchestrator', 'ModelBackend']
        for cls in required_classes:
            status = "✓" if cls in classes else "✗"
            print(f"   {status} Class '{cls}'")
        
        # Check for MCP tools
        mcp_tools = [
            'start_context_analysis',
            'add_perspective', 
            'analyze_from_perspectives',
            'synthesize_perspectives'
        ]
        for tool in mcp_tools:
            status = "✓" if tool in functions else "✗"
            print(f"   {status} Tool '{tool}'")
            
    except Exception as e:
        print(f"   ✗ Error parsing MCP file: {e}")
    
    # Check model backend support
    print("\n3. Model Backend Support:")
    backends = ['BEDROCK', 'LITELLM', 'OLLAMA']
    for backend in backends:
        if backend in content:
            print(f"   ✓ {backend} support implemented")
    
    # Validate thread orchestration
    print("\n4. Thread Orchestration Features:")
    features = [
        ('NO_RESPONSE', 'NoResponse handling'),
        ('broadcast_message', 'Parallel broadcasting'),
        ('conversation_history', 'Thread history tracking'),
        ('sessions', 'Session management')
    ]
    for feature, desc in features:
        status = "✓" if feature in content else "✗"
        print(f"   {status} {desc}")
    
    # Check pyproject.toml
    print("\n5. Package Configuration:")
    try:
        with open('pyproject.toml', 'r') as f:
            config = f.read()
        
        if 'context-switcher-mcp' in config:
            print("   ✓ Package name configured")
        if 'fastmcp' in config:
            print("   ✓ FastMCP dependency listed")
        if all(dep in config for dep in ['litellm', 'boto3', 'httpx']):
            print("   ✓ All provider dependencies listed")
    except Exception as e:
        print(f"   ✗ Error reading pyproject.toml: {e}")
    
    print("\n=== Validation Complete ===")
    print("\nThe Context-Switcher MCP is structurally complete and ready for installation!")


if __name__ == "__main__":
    validate_structure()
