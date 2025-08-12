#!/usr/bin/env python3
"""
Validation script for the new unified configuration system

This script tests the new configuration system independently of the main
application to ensure it works correctly before integration.
"""

import sys
import os
import tempfile
import json
import warnings
from pathlib import Path

# Add src to path so we can import our new config system
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Test imports of new configuration system
    from context_switcher_mcp.config.core import ContextSwitcherConfig, ConfigurationError
    from context_switcher_mcp.config.domains.models import ModelConfig
    from context_switcher_mcp.config.domains.session import SessionConfig
    from context_switcher_mcp.config.domains.security import SecurityConfig
    from context_switcher_mcp.config.domains.server import ServerConfig
    from context_switcher_mcp.config.domains.monitoring import MonitoringConfig
    from context_switcher_mcp.config.environments import (
        detect_environment, get_development_config, get_staging_config, get_production_config
    )
    
    print("✅ All configuration module imports successful")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_core_configuration():
    """Test core configuration functionality"""
    print("\n🧪 Testing core configuration...")
    
    try:
        # Test default configuration creation
        config = ContextSwitcherConfig()
        assert config.models is not None
        assert config.session is not None
        assert config.security is not None
        assert config.server is not None
        assert config.monitoring is not None
        print("  ✅ Default configuration creation")
        
        # Test configuration with overrides
        config_with_overrides = ContextSwitcherConfig(
            server={"port": 4000, "log_level": "DEBUG"}
        )
        assert config_with_overrides.server.port == 4000
        assert config_with_overrides.server.log_level.value == "DEBUG"
        print("  ✅ Configuration with overrides")
        
        # Test production readiness
        assert not config.is_production_ready  # No secret key
        print("  ✅ Production readiness check")
        
        # Test environment detection
        env = config.deployment_environment
        assert env in ["development", "staging", "production"]
        print(f"  ✅ Environment detection: {env}")
        
        # Test sensitive data masking
        config_with_secret = ContextSwitcherConfig(
            security={"secret_key": "super-secret-key-that-should-be-masked-123456"}
        )
        masked = config_with_secret.get_masked_dict()
        assert masked["security"]["secret_key"] == "***MASKED***"
        print("  ✅ Sensitive data masking")
        
    except Exception as e:
        print(f"  ❌ Core configuration test failed: {e}")
        return False
    
    return True


def test_domain_configurations():
    """Test domain-specific configurations"""
    print("\n🧪 Testing domain configurations...")
    
    try:
        config = ContextSwitcherConfig()
        
        # Test models configuration
        models = config.models
        assert models.default_max_tokens == 2048
        assert "bedrock" in models.enabled_backends
        backend_config = models.get_backend_config("bedrock")
        assert "model_id" in backend_config
        print("  ✅ Models configuration")
        
        # Test session configuration
        session = config.session
        assert session.default_ttl_hours == 24
        assert session.is_ttl_valid(12)
        assert not session.is_ttl_valid(0)
        timeouts = session.get_operation_timeouts()
        assert "session_operation" in timeouts
        print("  ✅ Session configuration")
        
        # Test security configuration
        security = config.security
        rate_limit = security.get_rate_limit_config()
        assert rate_limit["enabled"] == security.enable_rate_limiting
        assert security.validate_input_length("short string")
        print("  ✅ Security configuration")
        
        # Test server configuration
        server = config.server
        assert server.bind_address == f"{server.host}:{server.port}"
        log_config = server.get_log_config()
        assert log_config["level"] == server.log_level.value
        print("  ✅ Server configuration")
        
        # Test monitoring configuration
        monitoring = config.monitoring
        profiling_config = monitoring.get_profiling_config()
        assert profiling_config["enabled"] == monitoring.profiling.enabled
        print("  ✅ Monitoring configuration")
        
    except Exception as e:
        print(f"  ❌ Domain configuration test failed: {e}")
        return False
    
    return True


def test_environment_configurations():
    """Test environment-specific configurations"""
    print("\n🧪 Testing environment configurations...")
    
    try:
        # Test environment detection
        env = detect_environment()
        assert env in ["development", "staging", "production"]
        print(f"  ✅ Environment detection: {env}")
        
        # Test development configuration
        dev_config = get_development_config()
        assert dev_config.server.log_level.value == "DEBUG"
        assert dev_config.server.enable_debug_mode == True
        assert not dev_config.is_production_ready
        print("  ✅ Development configuration")
        
        # Test staging configuration
        staging_config = get_staging_config()
        assert staging_config.server.log_level.value == "INFO"
        assert staging_config.server.enable_debug_mode == False
        print("  ✅ Staging configuration")
        
        # Test production configuration
        # Set secret key for production test
        os.environ["CONTEXT_SWITCHER_SECRET_KEY"] = "x" * 64
        try:
            prod_config = get_production_config()
            assert prod_config.server.log_level.value == "WARNING"
            assert prod_config.security.enable_client_binding == True
            assert prod_config.is_production_ready
            print("  ✅ Production configuration")
        finally:
            if "CONTEXT_SWITCHER_SECRET_KEY" in os.environ:
                del os.environ["CONTEXT_SWITCHER_SECRET_KEY"]
        
    except Exception as e:
        print(f"  ❌ Environment configuration test failed: {e}")
        return False
    
    return True


def test_file_loading():
    """Test configuration file loading"""
    print("\n🧪 Testing configuration file loading...")
    
    try:
        # Test JSON file loading
        config_data = {
            "server": {"port": 5000, "log_level": "WARNING"},
            "models": {"default_max_tokens": 4096}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            json_file = f.name
        
        try:
            config = ContextSwitcherConfig(config_file=json_file)
            assert config.server.port == 5000
            assert config.server.log_level.value == "WARNING"
            assert config.models.default_max_tokens == 4096
            print("  ✅ JSON file loading")
        finally:
            os.unlink(json_file)
        
        # Test YAML file loading (if PyYAML is available)
        try:
            import yaml
            yaml_content = """
server:
  port: 6000
  log_level: ERROR
models:
  default_temperature: 0.9
"""
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(yaml_content)
                yaml_file = f.name
            
            try:
                config = ContextSwitcherConfig(config_file=yaml_file)
                assert config.server.port == 6000
                assert config.server.log_level.value == "ERROR"
                assert config.models.default_temperature == 0.9
                print("  ✅ YAML file loading")
            finally:
                os.unlink(yaml_file)
                
        except ImportError:
            print("  ⚠ PyYAML not available - skipping YAML test")
        
        # Test invalid file handling
        try:
            ContextSwitcherConfig(config_file="/non/existent/file.json")
            assert False, "Should have raised ConfigurationError"
        except ConfigurationError:
            print("  ✅ Invalid file error handling")
        
    except Exception as e:
        print(f"  ❌ File loading test failed: {e}")
        return False
    
    return True


def test_validation_and_errors():
    """Test configuration validation and error handling"""
    print("\n🧪 Testing validation and error handling...")
    
    try:
        # Test invalid port number
        try:
            ContextSwitcherConfig(server={"port": -1})
            assert False, "Should have raised validation error"
        except Exception:
            print("  ✅ Invalid port validation")
        
        # Test invalid temperature
        try:
            ContextSwitcherConfig(models={"default_temperature": 5.0})
            assert False, "Should have raised validation error"
        except Exception:
            print("  ✅ Invalid temperature validation")
        
        # Test environment variable override
        os.environ["CS_SERVER_PORT"] = "7000"
        try:
            config = ContextSwitcherConfig()
            assert config.server.port == 7000
            print("  ✅ Environment variable override")
        finally:
            if "CS_SERVER_PORT" in os.environ:
                del os.environ["CS_SERVER_PORT"]
        
    except Exception as e:
        print(f"  ❌ Validation test failed: {e}")
        return False
    
    return True


def test_legacy_compatibility():
    """Test legacy compatibility layer"""
    print("\n🧪 Testing legacy compatibility...")
    
    try:
        from context_switcher_mcp.config.migration import LegacyConfigAdapter
        
        # Test legacy adapter creation
        unified_config = ContextSwitcherConfig()
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            legacy_config = LegacyConfigAdapter(unified_config)
            
            # Should issue deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            print("  ✅ Deprecation warning issued")
        
        # Test legacy attribute access
        assert hasattr(legacy_config, 'model')
        assert hasattr(legacy_config, 'server') 
        assert hasattr(legacy_config, 'session')
        assert legacy_config.model.default_max_tokens == unified_config.models.default_max_tokens
        assert legacy_config.server.port == unified_config.server.port
        print("  ✅ Legacy attribute compatibility")
        
        # Test legacy methods
        masked = legacy_config.mask_sensitive_data()
        assert isinstance(masked, dict)
        
        is_valid, issues = legacy_config.validate_current_config()
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)
        print("  ✅ Legacy method compatibility")
        
    except Exception as e:
        print(f"  ❌ Legacy compatibility test failed: {e}")
        return False
    
    return True


def main():
    """Run all validation tests"""
    print("🚀 Validating unified configuration system...\n")
    
    tests = [
        test_core_configuration,
        test_domain_configurations, 
        test_environment_configurations,
        test_file_loading,
        test_validation_and_errors,
        test_legacy_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The unified configuration system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())