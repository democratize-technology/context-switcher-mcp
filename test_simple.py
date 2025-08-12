#!/usr/bin/env python3
"""Simple test runner that bypasses the complex logging setup."""

import sys
import os
sys.path.insert(0, 'src')

# Disable the complex logging setup
os.environ['SIMPLE_LOGGING'] = '1'

# Now try to import and test basic functionality
try:
    from context_switcher_mcp.aorp import AORPResponseBuilder
    
    # Test basic AORP functionality
    builder = AORPResponseBuilder()
    response = builder.build_response(
        summary="Test summary",
        analysis="Test analysis",
        context="Test context"
    )
    
    print("✅ AORP basic test passed")
    print(f"Response keys: {list(response.keys())}")
    
    # Test error response
    error_response = builder.build_error_response(
        "Test error",
        "test_error_type"
    )
    
    print("✅ AORP error response test passed")
    print(f"Error response keys: {list(error_response.keys())}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()