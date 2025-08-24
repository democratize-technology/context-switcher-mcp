"""Comprehensive tests for config_base.py module - achieving 100% coverage

This test suite covers the configuration base interfaces and utilities,
including dataclasses, exceptions, and base provider implementations.
"""

import os
from unittest.mock import patch

import context_switcher_mcp.config_base as config_base
import pytest
from context_switcher_mcp.types import ConfigurationData, ModelBackend  # noqa: E402


class TestConfigurationExceptions:
    """Test configuration exception classes"""

    def test_configuration_error_import(self):
        """Test ConfigurationError can be imported"""
        assert hasattr(config_base, "ConfigurationError")
        assert issubclass(config_base.ConfigurationError, Exception)

    def test_configuration_error_instantiation(self):
        """Test ConfigurationError can be instantiated"""
        error = config_base.ConfigurationError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_configuration_validation_error_import(self):
        """Test ConfigurationValidationError can be imported"""
        assert hasattr(config_base, "ConfigurationValidationError")
        assert issubclass(
            config_base.ConfigurationValidationError, config_base.ConfigurationError
        )

    def test_configuration_validation_error_instantiation(self):
        """Test ConfigurationValidationError can be instantiated"""
        error = config_base.ConfigurationValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, config_base.ConfigurationError)

    def test_configuration_migration_error_import(self):
        """Test ConfigurationMigrationError can be imported"""
        assert hasattr(config_base, "ConfigurationMigrationError")
        assert issubclass(
            config_base.ConfigurationMigrationError, config_base.ConfigurationError
        )

    def test_configuration_migration_error_instantiation(self):
        """Test ConfigurationMigrationError can be instantiated"""
        error = config_base.ConfigurationMigrationError("Migration failed")
        assert str(error) == "Migration failed"
        assert isinstance(error, config_base.ConfigurationError)

    def test_exception_inheritance_chain(self):
        """Test exception inheritance chain is correct"""
        # ConfigurationValidationError -> ConfigurationError -> Exception
        assert issubclass(
            config_base.ConfigurationValidationError, config_base.ConfigurationError
        )
        assert issubclass(config_base.ConfigurationError, Exception)

        # ConfigurationMigrationError -> ConfigurationError -> Exception
        assert issubclass(
            config_base.ConfigurationMigrationError, config_base.ConfigurationError
        )
        assert issubclass(config_base.ConfigurationError, Exception)


class TestBackendConfiguration:
    """Test BackendConfiguration dataclass"""

    def test_backend_configuration_import(self):
        """Test BackendConfiguration can be imported"""
        assert hasattr(config_base, "BackendConfiguration")

    def test_backend_configuration_instantiation_minimal(self):
        """Test BackendConfiguration with minimal parameters"""
        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.BEDROCK
        )

        assert backend_config.backend_type == ModelBackend.BEDROCK
        assert backend_config.enabled is True
        assert backend_config.timeout_seconds == 30.0
        assert backend_config.max_retries == 3
        assert backend_config.retry_delay_seconds == 1.0
        assert backend_config.model_specific_config == {}

    def test_backend_configuration_instantiation_full(self):
        """Test BackendConfiguration with all parameters"""
        model_config = {"model_name": "test-model", "temperature": 0.5}

        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.LITELLM,
            enabled=False,
            timeout_seconds=60.0,
            max_retries=5,
            retry_delay_seconds=2.0,
            model_specific_config=model_config,
        )

        assert backend_config.backend_type == ModelBackend.LITELLM
        assert backend_config.enabled is False
        assert backend_config.timeout_seconds == 60.0
        assert backend_config.max_retries == 5
        assert backend_config.retry_delay_seconds == 2.0
        assert backend_config.model_specific_config == model_config

    def test_backend_configuration_post_init_none_config(self):
        """Test __post_init__ method when model_specific_config is None"""
        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.OLLAMA, model_specific_config=None
        )

        # __post_init__ should convert None to empty dict
        assert backend_config.model_specific_config == {}

    def test_backend_configuration_post_init_existing_config(self):
        """Test __post_init__ method when model_specific_config is provided"""
        original_config = {"existing": "config"}

        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.BEDROCK, model_specific_config=original_config
        )

        # __post_init__ should preserve existing config
        assert backend_config.model_specific_config == original_config

    def test_backend_configuration_to_dict(self):
        """Test to_dict method"""
        model_config = {"model": "test", "params": {"temp": 0.7}}

        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.LITELLM,
            enabled=True,
            timeout_seconds=45.0,
            max_retries=2,
            retry_delay_seconds=1.5,
            model_specific_config=model_config,
        )

        result = backend_config.to_dict()

        expected = {
            "backend_type": "litellm",  # Enum value
            "enabled": True,
            "timeout_seconds": 45.0,
            "max_retries": 2,
            "retry_delay_seconds": 1.5,
            "model_specific_config": {"model": "test", "params": {"temp": 0.7}},
        }

        assert result == expected

    def test_backend_configuration_to_dict_copy_behavior(self):
        """Test that to_dict creates a copy of model_specific_config"""
        original_config = {"mutable": "config"}

        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.BEDROCK, model_specific_config=original_config
        )

        result_dict = backend_config.to_dict()

        # Modify the returned dict
        result_dict["model_specific_config"]["new_key"] = "new_value"

        # Original config should be unchanged
        assert "new_key" not in original_config
        assert original_config == {"mutable": "config"}

    def test_backend_configuration_different_backend_types(self):
        """Test BackendConfiguration with different backend types"""
        backends = [ModelBackend.BEDROCK, ModelBackend.LITELLM, ModelBackend.OLLAMA]

        for backend in backends:
            config = config_base.BackendConfiguration(backend_type=backend)
            assert config.backend_type == backend

            # Test to_dict with different backend types
            result = config.to_dict()
            assert result["backend_type"] == backend.value


class TestSecurityConfiguration:
    """Test SecurityConfiguration dataclass"""

    def test_security_configuration_import(self):
        """Test SecurityConfiguration can be imported"""
        assert hasattr(config_base, "SecurityConfiguration")

    def test_security_configuration_instantiation_defaults(self):
        """Test SecurityConfiguration with default values"""
        security_config = config_base.SecurityConfiguration()

        assert security_config.enable_client_binding is True
        assert security_config.max_validation_failures == 3
        assert security_config.session_entropy_length == 32
        assert security_config.binding_signature_algorithm == "pbkdf2_hmac"
        assert security_config.signature_iterations == 600000
        assert security_config.enable_access_pattern_analysis is True
        assert security_config.suspicious_activity_threshold == 5

    def test_security_configuration_instantiation_custom(self):
        """Test SecurityConfiguration with custom values"""
        security_config = config_base.SecurityConfiguration(
            enable_client_binding=False,
            max_validation_failures=5,
            session_entropy_length=64,
            binding_signature_algorithm="custom_algo",
            signature_iterations=1000000,
            enable_access_pattern_analysis=False,
            suspicious_activity_threshold=10,
        )

        assert security_config.enable_client_binding is False
        assert security_config.max_validation_failures == 5
        assert security_config.session_entropy_length == 64
        assert security_config.binding_signature_algorithm == "custom_algo"
        assert security_config.signature_iterations == 1000000
        assert security_config.enable_access_pattern_analysis is False
        assert security_config.suspicious_activity_threshold == 10

    def test_security_configuration_to_dict(self):
        """Test SecurityConfiguration to_dict method"""
        security_config = config_base.SecurityConfiguration(
            enable_client_binding=True,
            max_validation_failures=3,
            session_entropy_length=32,
            binding_signature_algorithm="pbkdf2_hmac",
            signature_iterations=600000,
            enable_access_pattern_analysis=True,
            suspicious_activity_threshold=5,
        )

        result = security_config.to_dict()

        expected = {
            "enable_client_binding": True,
            "max_validation_failures": 3,
            "session_entropy_length": 32,
            "binding_signature_algorithm": "pbkdf2_hmac",
            "signature_iterations": 600000,
            "enable_access_pattern_analysis": True,
            "suspicious_activity_threshold": 5,
        }

        assert result == expected

    def test_security_configuration_edge_values(self):
        """Test SecurityConfiguration with edge values"""
        # Test with boundary values
        security_config = config_base.SecurityConfiguration(
            max_validation_failures=0,
            session_entropy_length=1,
            signature_iterations=1,
            suspicious_activity_threshold=0,
        )

        assert security_config.max_validation_failures == 0
        assert security_config.session_entropy_length == 1
        assert security_config.signature_iterations == 1
        assert security_config.suspicious_activity_threshold == 0

        # Test to_dict with edge values
        result = security_config.to_dict()
        assert result["max_validation_failures"] == 0
        assert result["session_entropy_length"] == 1
        assert result["signature_iterations"] == 1
        assert result["suspicious_activity_threshold"] == 0


class TestBaseConfigurationProvider:
    """Test BaseConfigurationProvider class"""

    def test_base_configuration_provider_import(self):
        """Test BaseConfigurationProvider can be imported"""
        assert hasattr(config_base, "BaseConfigurationProvider")

    def test_base_configuration_provider_instantiation(self):
        """Test BaseConfigurationProvider can be instantiated"""
        provider = config_base.BaseConfigurationProvider()

        assert isinstance(provider, config_base.BaseConfigurationProvider)
        assert provider.session_config is not None
        assert isinstance(provider.session_config, ConfigurationData)
        assert provider.backend_configs == {}
        assert isinstance(provider.security_config, config_base.SecurityConfiguration)
        assert provider._validated is False

    def test_base_configuration_provider_get_session_config(self):
        """Test BaseConfigurationProvider get_session_config method"""
        provider = config_base.BaseConfigurationProvider()

        result = provider.get_session_config()

        assert result is provider.session_config
        assert isinstance(result, ConfigurationData)

    def test_base_configuration_provider_get_backend_config(self):
        """Test BaseConfigurationProvider get_backend_config method"""
        provider = config_base.BaseConfigurationProvider()

        # Returns default configuration for missing backend
        result = provider.get_backend_config(ModelBackend.BEDROCK)
        expected_default = config_base.BackendConfiguration(
            backend_type=ModelBackend.BEDROCK
        ).to_dict()
        assert result == expected_default
        assert result["backend_type"] == "bedrock"
        assert result["enabled"] is True
        assert result["timeout_seconds"] == 30.0

        # Add a backend config
        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.BEDROCK, timeout_seconds=60.0
        )
        provider.backend_configs[ModelBackend.BEDROCK] = backend_config

        result = provider.get_backend_config(ModelBackend.BEDROCK)
        expected = backend_config.to_dict()
        assert result == expected
        assert result["timeout_seconds"] == 60.0

    def test_base_configuration_provider_get_security_config(self):
        """Test BaseConfigurationProvider get_security_config method"""
        provider = config_base.BaseConfigurationProvider()

        result = provider.get_security_config()
        expected = provider.security_config.to_dict()
        assert result == expected

    def test_base_configuration_provider_validate_not_validated(self):
        """Test BaseConfigurationProvider validate method when not validated"""
        provider = config_base.BaseConfigurationProvider()

        # Initially not validated, but validation should succeed with defaults
        assert provider._validated is False
        result = provider.validate()
        assert result is True  # Default configuration is valid
        assert provider._validated is True

    def test_base_configuration_provider_validate_already_validated(self):
        """Test BaseConfigurationProvider validate method when already validated"""
        provider = config_base.BaseConfigurationProvider()
        provider._validated = True

        result = provider.validate()
        assert result is True

    def test_base_configuration_provider_backend_config_missing(self):
        """Test BaseConfigurationProvider with missing backend config"""
        provider = config_base.BaseConfigurationProvider()

        # Request config for backend that doesn't exist - returns default
        result = provider.get_backend_config(ModelBackend.LITELLM)
        expected_default = config_base.BackendConfiguration(
            backend_type=ModelBackend.LITELLM
        ).to_dict()
        assert result == expected_default
        assert result["backend_type"] == "litellm"
        assert result["enabled"] is True

    def test_base_configuration_provider_multiple_backends(self):
        """Test BaseConfigurationProvider with multiple backends"""
        provider = config_base.BaseConfigurationProvider()

        # Add multiple backend configs
        bedrock_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.BEDROCK, timeout_seconds=30.0
        )
        litellm_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.LITELLM, timeout_seconds=60.0
        )

        provider.backend_configs[ModelBackend.BEDROCK] = bedrock_config
        provider.backend_configs[ModelBackend.LITELLM] = litellm_config

        # Test each backend
        bedrock_result = provider.get_backend_config(ModelBackend.BEDROCK)
        litellm_result = provider.get_backend_config(ModelBackend.LITELLM)

        assert bedrock_result["timeout_seconds"] == 30.0
        assert litellm_result["timeout_seconds"] == 60.0
        assert bedrock_result["backend_type"] == "bedrock"
        assert litellm_result["backend_type"] == "litellm"


class TestModuleLevelComponents:
    """Test module-level components and imports"""

    def test_logger_import(self):
        """Test that logger is properly imported"""
        assert hasattr(config_base, "logger")
        assert config_base.logger is not None
        assert hasattr(config_base.logger, "info")
        assert hasattr(config_base.logger, "warning")
        assert hasattr(config_base.logger, "error")

    def test_logger_name(self):
        """Test logger has correct name"""
        expected_name = "context_switcher_mcp.config_base"
        assert config_base.logger.name == expected_name

    def test_os_import(self):
        """Test that os module is imported"""
        # This test ensures the import is being used/tracked
        assert hasattr(config_base.os, "environ")

    def test_dataclass_decorator_usage(self):
        """Test that dataclass decorator is properly applied"""
        # Test that BackendConfiguration is a proper dataclass
        import dataclasses

        assert dataclasses.is_dataclass(config_base.BackendConfiguration)
        assert dataclasses.is_dataclass(config_base.SecurityConfiguration)

        # Test dataclass fields
        backend_fields = dataclasses.fields(config_base.BackendConfiguration)
        security_fields = dataclasses.fields(config_base.SecurityConfiguration)

        assert len(backend_fields) > 0
        assert len(security_fields) > 0


class TestTypeAnnotations:
    """Test type annotations and typing usage"""

    def test_type_annotations_present(self):
        """Test that type annotations are present"""
        # Test BackendConfiguration annotations
        backend_annotations = config_base.BackendConfiguration.__annotations__
        assert "backend_type" in backend_annotations
        assert "enabled" in backend_annotations
        assert "timeout_seconds" in backend_annotations

        # Test SecurityConfiguration annotations
        security_annotations = config_base.SecurityConfiguration.__annotations__
        assert "enable_client_binding" in security_annotations
        assert "max_validation_failures" in security_annotations

    def test_method_type_annotations(self):
        """Test method type annotations"""
        # Test BaseConfigurationProvider method annotations
        provider = config_base.BaseConfigurationProvider

        # get_backend_config should have annotations
        method = provider.get_backend_config
        assert hasattr(method, "__annotations__")

        # get_security_config should have annotations
        method = provider.get_security_config
        assert hasattr(method, "__annotations__")


class TestEdgeCasesAndDefensiveProgramming:
    """Test edge cases and defensive programming scenarios"""

    def test_backend_configuration_with_empty_model_config(self):
        """Test BackendConfiguration with empty model_specific_config"""
        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.OLLAMA, model_specific_config={}
        )

        assert backend_config.model_specific_config == {}

        # Test to_dict preserves empty dict
        result = backend_config.to_dict()
        assert result["model_specific_config"] == {}

    def test_security_configuration_with_string_numeric_values(self):
        """Test SecurityConfiguration behavior with edge numeric values"""
        # Test with very large values
        security_config = config_base.SecurityConfiguration(
            max_validation_failures=999999,
            session_entropy_length=1024,
            signature_iterations=10000000,
            suspicious_activity_threshold=1000,
        )

        assert security_config.max_validation_failures == 999999
        assert security_config.session_entropy_length == 1024
        assert security_config.signature_iterations == 10000000
        assert security_config.suspicious_activity_threshold == 1000

    def test_base_configuration_provider_protocol_compliance(self):
        """Test BaseConfigurationProvider implements required protocols"""
        provider = config_base.BaseConfigurationProvider()

        # Should be instance of ConfigurationProvider protocol
        from context_switcher_mcp.protocols import ConfigurationProvider

        assert isinstance(provider, ConfigurationProvider)

    def test_exception_with_unicode_messages(self):
        """Test exceptions with unicode messages"""
        unicode_message = "配置错误: Configuration failed 🚨"

        error = config_base.ConfigurationError(unicode_message)
        assert str(error) == unicode_message

        validation_error = config_base.ConfigurationValidationError(unicode_message)
        assert str(validation_error) == unicode_message

    def test_backend_configuration_with_complex_model_config(self):
        """Test BackendConfiguration with complex nested model config"""
        complex_config = {
            "model": "complex-model",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "nested": {"deep": "value", "list": [1, 2, 3], "bool": True},
            },
        }

        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.LITELLM, model_specific_config=complex_config
        )

        result = backend_config.to_dict()

        # Should preserve complex structure
        assert (
            result["model_specific_config"]["parameters"]["nested"]["deep"] == "value"
        )
        assert result["model_specific_config"]["parameters"]["nested"]["list"] == [
            1,
            2,
            3,
        ]
        assert result["model_specific_config"]["parameters"]["nested"]["bool"] is True

    def test_security_configuration_boolean_edge_cases(self):
        """Test SecurityConfiguration with boolean edge cases"""
        # Test with all booleans as False
        security_config = config_base.SecurityConfiguration(
            enable_client_binding=False, enable_access_pattern_analysis=False
        )

        result = security_config.to_dict()
        assert result["enable_client_binding"] is False
        assert result["enable_access_pattern_analysis"] is False

        # Ensure other values remain as expected
        assert result["max_validation_failures"] == 3  # Default


class TestConfigurationFactory:
    """Test ConfigurationFactory class"""

    def test_configuration_factory_import(self):
        """Test ConfigurationFactory can be imported"""
        assert hasattr(config_base, "ConfigurationFactory")

    def test_configuration_factory_create_from_dict_empty(self):
        """Test ConfigurationFactory create_from_dict with empty dict"""
        empty_dict = {}
        provider = config_base.ConfigurationFactory.create_from_dict(empty_dict)

        assert isinstance(provider, config_base.BaseConfigurationProvider)
        assert isinstance(provider.session_config, ConfigurationData)
        assert provider.backend_configs == {}
        assert isinstance(provider.security_config, config_base.SecurityConfiguration)

    def test_configuration_factory_create_from_dict_with_session(self):
        """Test ConfigurationFactory create_from_dict with session config"""
        config_dict = {
            "session": {
                "max_active_sessions": 100,
                "default_ttl_hours": 2,
                "cleanup_interval_seconds": 600,
                "max_retries": 5,
                "retry_delay_seconds": 2.0,
                "timeout_seconds": 60.0,
            }
        }

        provider = config_base.ConfigurationFactory.create_from_dict(config_dict)
        session_config = provider.get_session_config()

        assert session_config.max_active_sessions == 100
        assert session_config.default_ttl_hours == 2
        assert session_config.cleanup_interval_seconds == 600
        assert session_config.max_retries == 5
        assert session_config.retry_delay_seconds == 2.0
        assert session_config.timeout_seconds == 60.0

    def test_configuration_factory_create_from_dict_with_backends(self):
        """Test ConfigurationFactory create_from_dict with backend configs"""
        config_dict = {
            "backends": {
                "bedrock": {
                    "enabled": True,
                    "timeout_seconds": 45.0,
                    "max_retries": 4,
                    "retry_delay_seconds": 1.5,
                    "model_specific_config": {"temperature": 0.7},
                },
                "litellm": {"enabled": False, "timeout_seconds": 30.0},
            }
        }

        provider = config_base.ConfigurationFactory.create_from_dict(config_dict)

        bedrock_config = provider.get_backend_config(ModelBackend.BEDROCK)
        assert bedrock_config["enabled"] is True
        assert bedrock_config["timeout_seconds"] == 45.0
        assert bedrock_config["max_retries"] == 4
        assert bedrock_config["model_specific_config"]["temperature"] == 0.7

        litellm_config = provider.get_backend_config(ModelBackend.LITELLM)
        assert litellm_config["enabled"] is False
        assert litellm_config["timeout_seconds"] == 30.0

    def test_configuration_factory_create_from_dict_with_security(self):
        """Test ConfigurationFactory create_from_dict with security config"""
        config_dict = {
            "security": {
                "enable_client_binding": False,
                "max_validation_failures": 5,
                "session_entropy_length": 64,
                "binding_signature_algorithm": "custom_algo",
                "signature_iterations": 1000000,
                "enable_access_pattern_analysis": False,
                "suspicious_activity_threshold": 10,
            }
        }

        provider = config_base.ConfigurationFactory.create_from_dict(config_dict)
        security_config = provider.get_security_config()

        assert security_config["enable_client_binding"] is False
        assert security_config["max_validation_failures"] == 5
        assert security_config["session_entropy_length"] == 64
        assert security_config["binding_signature_algorithm"] == "custom_algo"
        assert security_config["signature_iterations"] == 1000000
        assert security_config["enable_access_pattern_analysis"] is False
        assert security_config["suspicious_activity_threshold"] == 10

    def test_configuration_factory_create_from_dict_invalid_backend(self):
        """Test ConfigurationFactory with invalid backend type"""
        config_dict = {
            "backends": {
                "invalid_backend": {"enabled": True},
                "bedrock": {"enabled": True},
            }
        }

        with patch.object(config_base.logger, "warning") as mock_warning:
            provider = config_base.ConfigurationFactory.create_from_dict(config_dict)
            mock_warning.assert_called_once_with(
                "Unknown backend type: invalid_backend"
            )

        # Should still create valid backend
        bedrock_config = provider.get_backend_config(ModelBackend.BEDROCK)
        assert bedrock_config["enabled"] is True

    def test_configuration_factory_create_from_dict_complete(self):
        """Test ConfigurationFactory create_from_dict with complete config"""
        config_dict = {
            "session": {"max_active_sessions": 25, "default_ttl_hours": 3},
            "backends": {"ollama": {"enabled": True, "timeout_seconds": 120.0}},
            "security": {"enable_client_binding": True, "max_validation_failures": 2},
        }

        provider = config_base.ConfigurationFactory.create_from_dict(config_dict)

        # Test all sections
        session_config = provider.get_session_config()
        assert session_config.max_active_sessions == 25
        assert session_config.default_ttl_hours == 3

        ollama_config = provider.get_backend_config(ModelBackend.OLLAMA)
        assert ollama_config["timeout_seconds"] == 120.0

        security_config = provider.get_security_config()
        assert security_config["max_validation_failures"] == 2

    @patch.dict(
        os.environ,
        {
            "MAX_ACTIVE_SESSIONS": "75",
            "DEFAULT_TTL_HOURS": "4",
            "CLEANUP_INTERVAL_SECONDS": "900",
            "MAX_RETRIES": "6",
            "RETRY_DELAY_SECONDS": "3.0",
            "TIMEOUT_SECONDS": "90.0",
            "ENABLE_CLIENT_BINDING": "false",
            "MAX_VALIDATION_FAILURES": "8",
            "SESSION_ENTROPY_LENGTH": "128",
            "SIGNATURE_ITERATIONS": "800000",
        },
    )
    def test_configuration_factory_create_from_environment(self):
        """Test ConfigurationFactory create_from_environment with env vars"""
        provider = config_base.ConfigurationFactory.create_from_environment()

        session_config = provider.get_session_config()
        assert session_config.max_active_sessions == 75
        assert session_config.default_ttl_hours == 4
        assert session_config.cleanup_interval_seconds == 900
        assert session_config.max_retries == 6
        assert session_config.retry_delay_seconds == 3.0
        assert session_config.timeout_seconds == 90.0

        security_config = provider.get_security_config()
        assert security_config["enable_client_binding"] is False
        assert security_config["max_validation_failures"] == 8
        assert security_config["session_entropy_length"] == 128
        assert security_config["signature_iterations"] == 800000

    def test_configuration_factory_create_from_environment_defaults(self):
        """Test ConfigurationFactory create_from_environment with defaults"""
        # Clear environment variables that might affect the test
        _env_vars = [
            "MAX_ACTIVE_SESSIONS",
            "DEFAULT_TTL_HOURS",
            "CLEANUP_INTERVAL_SECONDS",
            "MAX_RETRIES",
            "RETRY_DELAY_SECONDS",
            "TIMEOUT_SECONDS",
            "ENABLE_CLIENT_BINDING",
            "MAX_VALIDATION_FAILURES",
            "SESSION_ENTROPY_LENGTH",
            "SIGNATURE_ITERATIONS",
        ]
        _ = _env_vars  # Mark as intentionally unused

        with patch.dict(os.environ, {}, clear=True):
            provider = config_base.ConfigurationFactory.create_from_environment()

            session_config = provider.get_session_config()
            assert session_config.max_active_sessions == 50  # Default
            assert session_config.default_ttl_hours == 1  # Default

            security_config = provider.get_security_config()
            assert security_config["enable_client_binding"] is True  # Default

    def test_configuration_factory_create_default(self):
        """Test ConfigurationFactory create_default"""
        provider = config_base.ConfigurationFactory.create_default()

        assert isinstance(provider, config_base.BaseConfigurationProvider)
        assert isinstance(provider.session_config, ConfigurationData)
        assert provider.backend_configs == {}
        assert isinstance(provider.security_config, config_base.SecurityConfiguration)


class TestBaseMigrator:
    """Test BaseMigrator class"""

    def test_base_migrator_import(self):
        """Test BaseMigrator can be imported"""
        assert hasattr(config_base, "BaseMigrator")

    def test_base_migrator_instantiation(self):
        """Test BaseMigrator can be instantiated"""
        migrator = config_base.BaseMigrator()

        assert isinstance(migrator, config_base.BaseMigrator)
        assert migrator.migration_rules == []

    def test_base_migrator_add_migration_rule(self):
        """Test BaseMigrator add_migration_rule method"""
        migrator = config_base.BaseMigrator()

        def test_migration(config):
            return config

        migrator.add_migration_rule("v1.0", "v2.0", test_migration)

        assert len(migrator.migration_rules) == 1
        rule = migrator.migration_rules[0]
        assert rule["from_version"] == "v1.0"
        assert rule["to_version"] == "v2.0"
        assert rule["migrate"] == test_migration

    def test_base_migrator_migrate_config_no_rules(self):
        """Test BaseMigrator migrate_config with no applicable rules"""
        migrator = config_base.BaseMigrator()
        old_config = {"version": "v1.0", "data": "test"}

        result = migrator.migrate_config(old_config)

        # Should return unchanged copy
        assert result == old_config
        assert result is not old_config  # Should be a copy

    def test_base_migrator_migrate_config_with_rule(self):
        """Test BaseMigrator migrate_config with applicable rule"""
        migrator = config_base.BaseMigrator()

        def upgrade_config(config):
            new_config = config.copy()
            new_config["version"] = "v2.0"
            new_config["upgraded"] = True
            return new_config

        migrator.add_migration_rule("v1.0", "v2.0", upgrade_config)

        old_config = {"version": "v1.0", "data": "test"}

        with patch.object(config_base.logger, "info") as mock_info:
            result = migrator.migrate_config(old_config)
            mock_info.assert_called_once_with("Migrating config from v1.0 to v2.0")

        assert result["version"] == "v2.0"
        assert result["upgraded"] is True
        assert result["data"] == "test"

    def test_base_migrator_migrate_config_multiple_rules(self):
        """Test BaseMigrator migrate_config with multiple sequential rules"""
        migrator = config_base.BaseMigrator()

        def upgrade_v1_to_v2(config):
            new_config = config.copy()
            new_config["version"] = "v2.0"
            new_config["step1"] = True
            return new_config

        def upgrade_v2_to_v3(config):
            new_config = config.copy()
            new_config["version"] = "v3.0"
            new_config["step2"] = True
            return new_config

        migrator.add_migration_rule("v1.0", "v2.0", upgrade_v1_to_v2)
        migrator.add_migration_rule("v2.0", "v3.0", upgrade_v2_to_v3)

        old_config = {"version": "v1.0", "data": "test"}
        result = migrator.migrate_config(old_config)

        assert result["version"] == "v3.0"
        assert result["step1"] is True
        assert result["step2"] is True

    def test_base_migrator_is_migration_needed_true(self):
        """Test BaseMigrator is_migration_needed returns True when needed"""
        migrator = config_base.BaseMigrator()
        migrator.add_migration_rule("v1.0", "v2.0", lambda x: x)

        config = {"version": "v1.0"}

        assert migrator.is_migration_needed(config) is True

    def test_base_migrator_is_migration_needed_false(self):
        """Test BaseMigrator is_migration_needed returns False when not needed"""
        migrator = config_base.BaseMigrator()
        migrator.add_migration_rule("v1.0", "v2.0", lambda x: x)

        config = {"version": "v2.0"}

        assert migrator.is_migration_needed(config) is False

    def test_base_migrator_validate_migration_success(self):
        """Test BaseMigrator validate_migration with successful migration"""
        migrator = config_base.BaseMigrator()

        old_config = {
            "session": {"max_active_sessions": 50},
            "backends": {},
            "security": {},
        }

        new_config = {
            "session": {"max_active_sessions": 100},
            "backends": {},
            "security": {},
        }

        result = migrator.validate_migration(old_config, new_config)
        assert result is True

    def test_base_migrator_validate_migration_failure(self):
        """Test BaseMigrator validate_migration with config that causes exception"""
        migrator = config_base.BaseMigrator()

        old_config = {"valid": "config"}
        new_config = {"valid": "config"}

        # Mock ConfigurationFactory to raise an exception
        with patch.object(
            config_base.ConfigurationFactory,
            "create_from_dict",
            side_effect=Exception("Test exception"),
        ):
            with patch.object(config_base.logger, "error") as mock_error:
                result = migrator.validate_migration(old_config, new_config)
                mock_error.assert_called_once_with(
                    "Migration validation failed: Test exception"
                )

        assert result is False

    def test_base_migrator_get_config_version_present(self):
        """Test BaseMigrator _get_config_version with version present"""
        migrator = config_base.BaseMigrator()
        config = {"version": "v1.5"}

        version = migrator._get_config_version(config)
        assert version == "v1.5"

    def test_base_migrator_get_config_version_missing(self):
        """Test BaseMigrator _get_config_version with version missing"""
        migrator = config_base.BaseMigrator()
        config = {"no_version": "field"}

        version = migrator._get_config_version(config)
        assert version == "unknown"


class TestUtilityFunctions:
    """Test module-level utility functions"""

    def test_merge_configurations_simple(self):
        """Test merge_configurations with simple configs"""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = config_base.merge_configurations(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}
        assert result is not base  # Should be a copy

    def test_merge_configurations_nested(self):
        """Test merge_configurations with nested dictionaries"""
        base = {
            "session": {"max_sessions": 50, "ttl": 1},
            "security": {"enabled": True},
        }
        override = {"session": {"max_sessions": 100}, "backends": {"bedrock": True}}

        result = config_base.merge_configurations(base, override)

        expected = {
            "session": {"max_sessions": 100, "ttl": 1},
            "security": {"enabled": True},
            "backends": {"bedrock": True},
        }

        assert result == expected

    def test_merge_configurations_override_non_dict(self):
        """Test merge_configurations when override value is not dict"""
        base = {"nested": {"a": 1, "b": 2}}
        override = {"nested": "string_value"}

        result = config_base.merge_configurations(base, override)

        assert result == {"nested": "string_value"}

    def test_merge_configurations_empty_configs(self):
        """Test merge_configurations with empty configurations"""
        base = {}
        override = {"key": "value"}

        result = config_base.merge_configurations(base, override)
        assert result == {"key": "value"}

        base = {"key": "value"}
        override = {}

        result = config_base.merge_configurations(base, override)
        assert result == {"key": "value"}

    def test_validate_configuration_dict_valid(self):
        """Test validate_configuration_dict with valid config"""
        valid_config = {
            "session": {"max_active_sessions": 50},
            "backends": {},
            "security": {},
        }

        errors = config_base.validate_configuration_dict(valid_config)
        assert errors == []

    def test_validate_configuration_dict_invalid(self):
        """Test validate_configuration_dict with config that causes exception"""
        invalid_config = {"test": "config"}

        # Mock ConfigurationFactory to raise an exception
        with patch.object(
            config_base.ConfigurationFactory,
            "create_from_dict",
            side_effect=Exception("Test creation error"),
        ):
            errors = config_base.validate_configuration_dict(invalid_config)
            assert len(errors) > 0
            assert any(
                "Configuration creation failed: Test creation error" in error
                for error in errors
            )

    def test_get_default_backend_config(self):
        """Test get_default_backend_config utility function"""
        for backend in [
            ModelBackend.BEDROCK,
            ModelBackend.LITELLM,
            ModelBackend.OLLAMA,
        ]:
            config = config_base.get_default_backend_config(backend)

            assert config["backend_type"] == backend.value
            assert config["enabled"] is True
            assert config["timeout_seconds"] == 30.0
            assert config["max_retries"] == 3
            assert config["retry_delay_seconds"] == 1.0
            assert config["model_specific_config"] == {}


class TestAdvancedBaseConfigurationProvider:
    """Test advanced BaseConfigurationProvider functionality"""

    def test_base_configuration_provider_set_backend_config(self):
        """Test BaseConfigurationProvider set_backend_config method"""
        provider = config_base.BaseConfigurationProvider()
        provider._validated = True  # Start as validated

        backend_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.BEDROCK, timeout_seconds=120.0
        )

        provider.set_backend_config(ModelBackend.BEDROCK, backend_config)

        # Should invalidate and store config
        assert provider._validated is False
        assert ModelBackend.BEDROCK in provider.backend_configs

        result = provider.get_backend_config(ModelBackend.BEDROCK)
        assert result["timeout_seconds"] == 120.0

    def test_base_configuration_provider_is_validated(self):
        """Test BaseConfigurationProvider is_validated method"""
        provider = config_base.BaseConfigurationProvider()

        # Initially not validated
        assert provider.is_validated() is False

        # After validation
        provider.validate()
        assert provider.is_validated() is True

    def test_base_configuration_provider_validate_session_config_invalid(self):
        """Test BaseConfigurationProvider validation with invalid session config"""
        provider = config_base.BaseConfigurationProvider()

        # Set invalid session config
        provider.session_config.max_active_sessions = 0  # Invalid

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_base_configuration_provider_validate_backend_configs_invalid(self):
        """Test BaseConfigurationProvider validation with invalid backend config"""
        provider = config_base.BaseConfigurationProvider()

        # Set invalid backend config
        invalid_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.BEDROCK,
            timeout_seconds=-1,  # Invalid
        )
        provider.backend_configs[ModelBackend.BEDROCK] = invalid_config

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_base_configuration_provider_validate_security_config_invalid(self):
        """Test BaseConfigurationProvider validation with invalid security config"""
        provider = config_base.BaseConfigurationProvider()

        # Set invalid security config
        provider.security_config.session_entropy_length = 8  # Too small

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_base_configuration_provider_validate_exception_handling(self):
        """Test BaseConfigurationProvider validate exception handling"""
        provider = config_base.BaseConfigurationProvider()

        # Mock validation method to raise exception
        with patch.object(
            provider,
            "_validate_session_config",
            side_effect=config_base.ConfigurationValidationError("Test error"),
        ):
            with patch.object(config_base.logger, "error") as mock_error:
                result = provider.validate()
                mock_error.assert_called_once()
                assert (
                    "Configuration validation failed: Test error"
                    in mock_error.call_args[0][0]
                )

        assert result is False
        assert provider._validated is False


class TestComprehensiveValidationCoverage:
    """Test all validation branches for 100% coverage"""

    def test_validate_session_config_default_ttl_hours_invalid(self):
        """Test _validate_session_config with invalid default_ttl_hours"""
        provider = config_base.BaseConfigurationProvider()
        provider.session_config.default_ttl_hours = 0  # Invalid

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_validate_session_config_cleanup_interval_invalid(self):
        """Test _validate_session_config with invalid cleanup_interval_seconds"""
        provider = config_base.BaseConfigurationProvider()
        provider.session_config.cleanup_interval_seconds = 0  # Invalid

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_validate_session_config_timeout_seconds_invalid(self):
        """Test _validate_session_config with invalid timeout_seconds"""
        provider = config_base.BaseConfigurationProvider()
        provider.session_config.timeout_seconds = 0  # Invalid

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_validate_backend_configs_max_retries_invalid(self):
        """Test _validate_backend_configs with invalid max_retries"""
        provider = config_base.BaseConfigurationProvider()

        invalid_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.BEDROCK,
            max_retries=-1,  # Invalid
        )
        provider.backend_configs[ModelBackend.BEDROCK] = invalid_config

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_validate_backend_configs_retry_delay_invalid(self):
        """Test _validate_backend_configs with invalid retry_delay_seconds"""
        provider = config_base.BaseConfigurationProvider()

        invalid_config = config_base.BackendConfiguration(
            backend_type=ModelBackend.LITELLM,
            retry_delay_seconds=-0.1,  # Invalid
        )
        provider.backend_configs[ModelBackend.LITELLM] = invalid_config

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_validate_security_config_max_validation_failures_invalid(self):
        """Test _validate_security_config with invalid max_validation_failures"""
        provider = config_base.BaseConfigurationProvider()
        provider.security_config.max_validation_failures = 0  # Invalid

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_validate_security_config_signature_iterations_invalid(self):
        """Test _validate_security_config with invalid signature_iterations"""
        provider = config_base.BaseConfigurationProvider()
        provider.security_config.signature_iterations = 50000  # Too low

        result = provider.validate()
        assert result is False
        assert provider._validated is False

    def test_validate_configuration_dict_validation_failed(self):
        """Test validate_configuration_dict when validation fails"""
        # Create config that will fail validation
        config_dict = {
            "session": {"max_active_sessions": 0}  # Will cause validation failure
        }

        errors = config_base.validate_configuration_dict(config_dict)
        assert len(errors) > 0
        assert "Configuration validation failed" in errors[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
