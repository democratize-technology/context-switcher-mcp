#!/usr/bin/env python3
"""Runtime validation script for Context-Switcher MCP"""

import sys
import traceback
from typing import List, Tuple

def test_imports() -> Tuple[bool, List[str]]:
    """Test all module imports"""
    errors = []
    modules = [
        "src.context_switcher_mcp",
        "src.context_switcher_mcp.models",
        "src.context_switcher_mcp.orchestrator",
        "src.context_switcher_mcp.session_manager",
        "src.context_switcher_mcp.compression",
        "src.context_switcher_mcp.templates"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            errors.append(f"✗ {module}: {e}")
            print(f"✗ {module}: {e}")
    
    return len(errors) == 0, errors

def test_tool_creation() -> Tuple[bool, List[str]]:
    """Test MCP tool creation"""
    errors = []
    try:
        from src.context_switcher_mcp import mcp
        tools = [tool for tool in dir(mcp) if not tool.startswith('_')]
        print(f"\n✓ MCP server created with {len(tools)} tools")
    except Exception as e:
        errors.append(f"Failed to create MCP server: {e}")
        print(f"\n✗ Failed to create MCP server: {e}")
    
    return len(errors) == 0, errors

def test_basic_functionality() -> Tuple[bool, List[str]]:
    """Test basic functionality without LLM calls"""
    errors = []
    
    try:
        from src.context_switcher_mcp import (
            session_manager,
            validate_topic,
            validate_session_id,
            ContextSwitcherSession
        )
        from datetime import datetime
        
        # Test validation functions
        assert validate_topic("test")
        assert not validate_topic("")
        assert not validate_topic("x" * 2000)
        print("\n✓ Topic validation works")
        
        # Test session creation
        session = ContextSwitcherSession(
            session_id="test-validation",
            created_at=datetime.utcnow()
        )
        session_manager.add_session(session)
        
        # Test session retrieval
        assert validate_session_id("test-validation")
        assert not validate_session_id("nonexistent")
        print("✓ Session management works")
        
        # Cleanup
        session_manager.remove_session("test-validation")
        
    except Exception as e:
        errors.append(f"Functionality test failed: {e}")
        print(f"\n✗ Functionality test failed: {e}")
        traceback.print_exc()
    
    return len(errors) == 0, errors

def main():
    """Run all validation tests"""
    print("=== Context-Switcher MCP Runtime Validation ===\n")
    
    all_passed = True
    all_errors = []
    
    # Run tests
    tests = [
        ("Import Tests", test_imports),
        ("Tool Creation", test_tool_creation),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        passed, errors = test_func()
        all_passed &= passed
        all_errors.extend(errors)
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All validation tests passed!")
        return 0
    else:
        print(f"✗ {len(all_errors)} errors found:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

if __name__ == "__main__":
    sys.exit(main())