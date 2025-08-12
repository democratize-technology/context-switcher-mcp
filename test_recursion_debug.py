#!/usr/bin/env python3
"""
Simple test to reproduce the logging recursion issue
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_logging_recursion():
    """Test that triggers the recursion in logging_config.py"""
    print("Testing logging configuration...")
    
    try:
        # Mock the problematic imports first
        import sys
        from unittest.mock import MagicMock, patch
        
        # Mock the mcp module and other dependencies
        sys.modules['mcp'] = MagicMock()
        sys.modules['mcp.server'] = MagicMock()
        sys.modules['mcp.server.fastmcp'] = MagicMock()
        
        # Mock the security logging module
        security_mock = MagicMock()
        security_mock.SecureLogger = MagicMock()
        security_mock.SecureLogFormatter = MagicMock()
        security_mock.setup_secure_logging = MagicMock()
        sys.modules['context_switcher_mcp.security.secure_logging'] = security_mock
        
        # Mock the error logging module  
        error_mock = MagicMock()
        error_mock.StructuredErrorLogger = MagicMock()
        error_mock.setup_structured_error_logging = MagicMock()
        sys.modules['context_switcher_mcp.error_logging'] = error_mock
        
        # This should trigger the recursion
        from context_switcher_mcp.logging_config import get_logger
        logger = get_logger("test")
        logger.info("This should work")
        print("✓ No recursion detected")
        return True
    except RecursionError as e:
        print(f"✗ RECURSION DETECTED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ OTHER ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_direct_import_recursion():
    """Test direct import that might trigger recursion"""
    print("Testing direct import...")
    
    try:
        # Mock dependencies first
        import sys
        from unittest.mock import MagicMock
        
        sys.modules['mcp'] = MagicMock()
        sys.modules['mcp.server'] = MagicMock()
        sys.modules['mcp.server.fastmcp'] = MagicMock()
        
        security_mock = MagicMock()
        security_mock.SecureLogger = MagicMock()
        security_mock.SecureLogFormatter = MagicMock()
        security_mock.setup_secure_logging = MagicMock()
        sys.modules['context_switcher_mcp.security.secure_logging'] = security_mock
        
        error_mock = MagicMock()
        error_mock.StructuredErrorLogger = MagicMock()
        error_mock.setup_structured_error_logging = MagicMock()
        sys.modules['context_switcher_mcp.error_logging'] = error_mock
        
        # Direct import should not cause recursion
        import context_switcher_mcp.logging_config
        print("✓ Direct import successful")
        return True
    except RecursionError as e:
        print(f"✗ RECURSION ON IMPORT: {e}")
        import traceback
        traceback.print_exc()
        return False  
    except Exception as e:
        print(f"✗ IMPORT ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== LOGGING RECURSION DEBUG TEST ===\n")
    
    success = True
    
    # Test 1: Direct import
    success &= test_direct_import_recursion()
    print()
    
    # Test 2: Using get_logger function
    success &= test_logging_recursion()
    print()
    
    if success:
        print("✓ All tests passed - no recursion detected")
        sys.exit(0)
    else:
        print("✗ Tests failed - recursion issue confirmed")
        sys.exit(1)