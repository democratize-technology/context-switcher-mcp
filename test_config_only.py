#!/usr/bin/env python3
"""
Test the unified configuration system without triggering MCP dependencies

This script directly imports and tests the configuration modules without
going through the main package __init__.py file.
"""

import sys
import os
from pathlib import Path

# Add the config directory to the path directly
config_dir = Path(__file__).parent / "src" / "context_switcher_mcp" / "config"
sys.path.insert(0, str(config_dir.parent.parent))

print("🧪 Testing unified configuration system (direct imports)...")

try:
    # Import the specific modules we need, avoiding __init__.py
    sys.path.insert(0, str(Path(__file__).parent / "src" / "context_switcher_mcp"))
    
    # Direct import of config modules
    import config.core as core_module
    import config.migration as migration_module
    import config.environments as env_module
    
    print("✅ Direct config module imports successful")
    
    # Test core configuration creation
    config = core_module.ContextSwitcherConfig()
    print(f"✅ Configuration created: {type(config).__name__}")
    
    # Test basic attributes
    print(f"   Server port: {config.server.port}")
    print(f"   Log level: {config.server.log_level.value}")
    print(f"   Models enabled: {', '.join(config.models.enabled_backends)}")
    print(f"   Production ready: {config.is_production_ready}")
    print(f"   Environment: {config.deployment_environment}")
    
    # Test legacy adapter
    legacy_adapter = migration_module.LegacyConfigAdapter(config)
    print(f"✅ Legacy adapter created: {type(legacy_adapter).__name__}")
    
    # Test legacy attribute access
    print(f"   Legacy server.port: {legacy_adapter.server.port}")
    print(f"   Legacy model.bedrock_model_id: {legacy_adapter.model.bedrock_model_id}")
    
    # Test environment configurations
    dev_config = env_module.get_development_config()
    print(f"✅ Development config: log_level={dev_config.server.log_level.value}")
    
    staging_config = env_module.get_staging_config()
    print(f"✅ Staging config: log_level={staging_config.server.log_level.value}")
    
    # Test with secret key for production
    os.environ["CONTEXT_SWITCHER_SECRET_KEY"] = "x" * 64
    try:
        prod_config = env_module.get_production_config()
        print(f"✅ Production config: log_level={prod_config.server.log_level.value}, production_ready={prod_config.is_production_ready}")
    finally:
        if "CONTEXT_SWITCHER_SECRET_KEY" in os.environ:
            del os.environ["CONTEXT_SWITCHER_SECRET_KEY"]
    
    # Test configuration with overrides
    custom_config = core_module.ContextSwitcherConfig(
        server={"port": 8080, "log_level": "ERROR"},
        models={"default_max_tokens": 4096}
    )
    print(f"✅ Custom config: port={custom_config.server.port}, tokens={custom_config.models.default_max_tokens}")
    
    print("\n🎉 All unified configuration tests passed!")
    print("   The new configuration system is working correctly.")
    
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)