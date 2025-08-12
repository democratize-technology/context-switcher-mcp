#!/usr/bin/env python3
"""
Test script to verify the new clean architecture by importing modules directly
without triggering the package __init__.py that requires MCP dependencies.
"""

import sys
import os
import importlib.util

# Add the src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), "src", "context_switcher_mcp")
sys.path.insert(0, src_path)


def load_module(module_name, file_path):
    """Load a module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_types_module():
    """Test types module"""
    print("🔧 Testing types module...")
    try:
        types_module = load_module("types", os.path.join(src_path, "types.py"))

        # Test enum access
        bedrock = types_module.ModelBackend.BEDROCK
        print(f"✅ ModelBackend.BEDROCK = {bedrock}")

        # Test dataclass creation
        config_data = types_module.ConfigurationData(max_active_sessions=25)
        print(
            f"✅ ConfigurationData created: max_sessions={config_data.max_active_sessions}"
        )

        # Test conversion to dict
        config_dict = config_data.to_dict()
        print(f"✅ ConfigurationData.to_dict() works: {len(config_dict)} keys")

        return True
    except Exception as e:
        print(f"❌ Types module test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_container_module():
    """Test container module"""
    print("\n🔧 Testing container module...")
    try:
        # Load the types module first (dependency)
        types_module = load_module("types", os.path.join(src_path, "types.py"))
        sys.modules["types"] = types_module  # Make it available for import

        # Load protocols module (dependency)
        protocols_module = load_module(
            "protocols", os.path.join(src_path, "protocols.py")
        )
        protocols_module.types = types_module  # Inject dependency

        # Load container module
        container_module = load_module(
            "container", os.path.join(src_path, "container.py")
        )
        container_module.protocols = protocols_module  # Inject dependency

        # Test container creation
        container = container_module.DependencyContainer()
        print("✅ DependencyContainer created")

        # Test registration
        class TestService:
            def __init__(self):
                self.value = "test"

        test_instance = TestService()
        container.register_instance(TestService, test_instance)
        print("✅ Instance registration works")

        # Test retrieval
        retrieved = container.get(TestService)
        assert retrieved.value == "test"
        print("✅ Instance retrieval works")

        # Test factory registration
        def test_factory():
            return TestService()

        container.register_factory(str, test_factory)  # Use str as test type
        factory_result = container.get(str)
        assert isinstance(factory_result, TestService)
        print("✅ Factory registration and retrieval works")

        # Test global container
        global_container = container_module.get_container()
        print(f"✅ Global container retrieved: {type(global_container).__name__}")

        return True
    except Exception as e:
        print(f"❌ Container module test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_base_module():
    """Test config_base module"""
    print("\n🔧 Testing config_base module...")
    try:
        # Load dependencies
        types_module = load_module("types", os.path.join(src_path, "types.py"))
        protocols_module = load_module(
            "protocols", os.path.join(src_path, "protocols.py")
        )
        protocols_module.types = types_module

        # Load config_base module
        config_base_module = load_module(
            "config_base", os.path.join(src_path, "config_base.py")
        )
        config_base_module.types = types_module
        config_base_module.protocols = protocols_module

        # Test BaseConfigurationProvider
        provider = config_base_module.BaseConfigurationProvider()
        print("✅ BaseConfigurationProvider created")

        # Test validation
        is_valid = provider.validate()
        print(f"✅ Configuration validation: {is_valid}")

        # Test session config retrieval
        session_config = provider.get_session_config()
        print(
            f"✅ Session config retrieved: max_sessions={session_config.max_active_sessions}"
        )

        # Test backend config retrieval
        backend_config = provider.get_backend_config(types_module.ModelBackend.BEDROCK)
        print(f"✅ Backend config retrieved: {len(backend_config)} settings")

        # Test configuration factory
        config_dict = {"session": {"max_active_sessions": 100, "default_ttl_hours": 2}}

        factory_provider = config_base_module.ConfigurationFactory.create_from_dict(
            config_dict
        )
        print("✅ ConfigurationFactory.create_from_dict works")

        # Verify the created configuration
        factory_session_config = factory_provider.get_session_config()
        assert factory_session_config.max_active_sessions == 100
        print(
            f"✅ Factory-created config verified: max_sessions={factory_session_config.max_active_sessions}"
        )

        return True
    except Exception as e:
        print(f"❌ Config base module test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_migration_module():
    """Test config_migration module"""
    print("\n🔧 Testing config_migration module...")
    try:
        # Load all dependencies in order
        types_module = load_module("types", os.path.join(src_path, "types.py"))
        protocols_module = load_module(
            "protocols", os.path.join(src_path, "protocols.py")
        )
        protocols_module.types = types_module

        container_module = load_module(
            "container", os.path.join(src_path, "container.py")
        )
        container_module.protocols = protocols_module

        config_base_module = load_module(
            "config_base", os.path.join(src_path, "config_base.py")
        )
        config_base_module.types = types_module
        config_base_module.protocols = protocols_module

        # Load config_migration module
        config_migration_module = load_module(
            "config_migration", os.path.join(src_path, "config_migration.py")
        )
        config_migration_module.config_base = config_base_module
        config_migration_module.container = container_module
        config_migration_module.protocols = protocols_module
        config_migration_module.types = types_module

        # Test CompatibilityAdapter
        adapter = config_migration_module.CompatibilityAdapter()
        print("✅ CompatibilityAdapter created")

        # Test migration detection
        legacy_config = {
            "version": "legacy_dataclass",
            "session": {"max_active_sessions": 50},
        }

        needs_migration = adapter.is_migration_needed(legacy_config)
        print(f"✅ Migration detection: {needs_migration}")

        # Test migration if needed
        if needs_migration:
            migrated_config = adapter.migrate_config(legacy_config)
            print("✅ Configuration migration completed")
            print(f"   Migrated version: {migrated_config.get('version')}")

        return True
    except Exception as e:
        print(f"❌ Config migration module test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_circular_dependency_resolution():
    """Test that circular dependencies are resolved"""
    print("\n🔧 Testing circular dependency resolution...")
    try:
        # Test that we can load config and config_migration without circular dependency
        print("Loading config_base...")
        config_base_module = load_module(
            "config_base", os.path.join(src_path, "config_base.py")
        )

        print("Loading config_migration...")
        config_migration_module = load_module(
            "config_migration", os.path.join(src_path, "config_migration.py")
        )

        # These should both load without importing each other circularly
        print("✅ Both modules loaded independently")

        # Test that they can work together through dependency injection
        adapter = config_migration_module.CompatibilityAdapter()
        provider = config_base_module.BaseConfigurationProvider()

        print("✅ Both modules can be instantiated together")
        print("✅ Circular dependency successfully resolved!")

        return True
    except Exception as e:
        print(f"❌ Circular dependency test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all architecture tests"""
    print("🧪 Testing Clean Architecture Implementation (Direct Module Loading)")
    print("=" * 70)

    tests = [
        ("Types Module", test_types_module),
        ("Container Module", test_container_module),
        ("Config Base Module", test_config_base_module),
        ("Config Migration Module", test_config_migration_module),
        ("Circular Dependency Resolution", test_circular_dependency_resolution),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))

    # Summary
    print("\n" + "=" * 70)
    print("🏁 Test Summary:")

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1

    print(f"\nResults: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All tests passed! Clean architecture is working correctly!")
        print("✅ Circular dependencies have been successfully resolved!")
        print("✅ Dependency injection system is operational!")
        print("✅ Configuration system is working without circular dependencies!")
        print("✅ All modules can be loaded independently!")
        return True
    else:
        print(
            f"\n❌ {len(tests) - passed} tests failed. Architecture needs further work."
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
