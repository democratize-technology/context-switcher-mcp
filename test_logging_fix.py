#!/usr/bin/env python3
"""
Test to validate the logging recursion fix
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_logging_fix():
    """Test that the logging recursion is fixed"""
    print("Testing logging configuration fix...")
    
    try:
        # Mock just what we need to test logging_config itself
        import sys
        from unittest.mock import MagicMock
        
        # Create minimal mocks for dependencies
        security_mock = MagicMock()
        security_mock.SecureLogger = lambda name: MagicMock()
        security_mock.SecureLogFormatter = MagicMock()
        security_mock.setup_secure_logging = MagicMock()
        sys.modules['context_switcher_mcp.security'] = MagicMock()
        sys.modules['context_switcher_mcp.security.secure_logging'] = security_mock
        
        error_mock = MagicMock()
        error_mock.StructuredErrorLogger = lambda **kwargs: MagicMock()
        error_mock.setup_structured_error_logging = lambda logger, **kwargs: logger
        sys.modules['context_switcher_mcp.error_logging'] = error_mock
        
        # Import logging_config directly - bypassing __init__.py
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'context_switcher_mcp'))
        import logging_config
        LoggingConfig = logging_config.LoggingConfig
        
        # Create config instance - this used to recurse
        config = LoggingConfig()
        
        # Get logger - this was the main recursion trigger
        logger = config.get_logger("test")
        
        # Try to log something
        logger.info("Test message - no recursion!")
        
        # Test multiple loggers
        logger1 = config.get_logger("test1")
        logger2 = config.get_logger("test2")
        logger_secure = config.get_logger("secure", secure=True)
        
        print("✓ Logging configuration works without recursion!")
        print("✓ Multiple loggers created successfully")
        print("✓ Secure logger created successfully")
        return True
        
    except RecursionError as e:
        print(f"✗ RECURSION STILL EXISTS: {e}")
        return False
    except Exception as e:
        print(f"✗ OTHER ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_functions():
    """Test the convenience functions don't recurse"""
    print("Testing convenience functions...")
    
    try:
        # Import logging_config directly - bypassing __init__.py
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'context_switcher_mcp'))
        import logging_config
        get_logger = logging_config.get_logger
        setup_logging = logging_config.setup_logging
        
        # This used to cause recursion
        setup_logging()
        
        # This used to cause recursion
        logger = get_logger("convenience_test")
        logger.info("Convenience function works!")
        
        print("✓ Convenience functions work without recursion!")
        return True
        
    except RecursionError as e:
        print(f"✗ RECURSION IN CONVENIENCE FUNCTIONS: {e}")
        return False
    except Exception as e:
        print(f"✗ CONVENIENCE FUNCTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== LOGGING RECURSION FIX VALIDATION ===\n")
    
    success = True
    
    # Test 1: Basic logging config
    success &= test_logging_fix()
    print()
    
    # Test 2: Convenience functions
    success &= test_convenience_functions()
    print()
    
    if success:
        print("✅ ALL TESTS PASSED - Recursion issue is FIXED!")
        sys.exit(0)
    else:
        print("❌ Tests failed - recursion issue still exists")
        sys.exit(1)