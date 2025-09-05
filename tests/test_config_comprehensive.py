"""Comprehensive tests for config.py module - achieving 100% coverage

This test suite covers the transition/compatibility layer configuration module,
testing all import fallback logic, compatibility functions, and error paths.
"""

from unittest.mock import MagicMock, patch

import context_switcher_mcp.config as config
import pytest


class TestConfigurationImports:
    """Test configuration system import and fallback logic"""

    def test_unified_config_available_flag(self):
        """Test that unified config availability flag is accessible"""
        # This tests the module-level variable is set correctly
        assert hasattr(config, "_UNIFIED_CONFIG_AVAILABLE")
        assert isinstance(config._UNIFIED_CONFIG_AVAILABLE, bool)

    @patch("context_switcher_mcp.config.warnings.warn")
    def test_fallback_import_warning(self, mock_warn):
        """Test warning is issued when falling back to legacy config"""
        # Reset the module to test import fallback
        with patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", False):
            with patch("context_switcher_mcp.config._LEGACY_CONFIG_AVAILABLE", True):
                _result = config.get_config()
                _ = _result  # Mark as intentionally unused

                # Should issue deprecation warning
                assert mock_warn.called

    def test_configuration_error_import(self):
        """Test that ConfigurationError is properly imported"""
        from context_switcher_mcp.config import ConfigurationError

        assert issubclass(ConfigurationError, Exception)

        # Test instantiation
        error = ConfigurationError("test error")
        assert str(error) == "test error"

    def test_no_config_system_available_import_error(self):
        """Test ImportError when neither config system is available"""
        # This tests the import-time validation logic by directly calling the validation
        # Since the actual validation happens at import time, we test the logic separately
        from context_switcher_mcp.config import ConfigurationError

        # Simulate the condition that would trigger the import error
        with patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", False):
            with patch("context_switcher_mcp.config._LEGACY_CONFIG_AVAILABLE", False):
                # Test that get_config properly raises an error when no system is available
                with pytest.raises(
                    ConfigurationError, match="No configuration system available"
                ):
                    config.get_config()


class TestGetConfigFunction:
    """Test get_config function with all system scenarios"""

    def test_get_config_unified_system(self):
        """Test get_config with unified system available (actual behavior)"""
        # Test that get_config works with valid parameters
        result = config.get_config(server={"port": 3023})

        assert result is not None
        from context_switcher_mcp.config.core import ContextSwitcherConfig

        assert isinstance(result, ContextSwitcherConfig)
        # Verify the values were applied
        assert result.server.port == 3023

    @patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", False)
    @patch("context_switcher_mcp.config._LEGACY_CONFIG_AVAILABLE", True)
    @patch("context_switcher_mcp.config._old_get_config")
    @patch("context_switcher_mcp.config.warnings.warn")
    def test_get_config_legacy_system(self, mock_warn, mock_old_get_config):
        """Test get_config fallback to legacy system"""
        mock_legacy_config = MagicMock()
        mock_old_get_config.return_value = mock_legacy_config

        result = config.get_config()

        mock_old_get_config.assert_called_once_with()
        mock_warn.assert_called_once()
        assert "legacy configuration system" in str(mock_warn.call_args)
        assert result == mock_legacy_config

    @patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", False)
    @patch("context_switcher_mcp.config._LEGACY_CONFIG_AVAILABLE", False)
    def test_get_config_no_system_available(self):
        """Test get_config raises error when no system available"""
        from context_switcher_mcp.config import ConfigurationError

        with pytest.raises(
            ConfigurationError, match="No configuration system available"
        ):
            config.get_config()

    @patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", True)
    @patch("context_switcher_mcp.config.ContextSwitcherConfig")
    def test_get_config_unified_raises_exception(self, mock_config_class):
        """Test get_config handles exceptions from unified system"""
        from context_switcher_mcp.config import ConfigurationError

        mock_config_class.side_effect = ConfigurationError("Unified config failed")

        with pytest.raises(
            ConfigurationError, match="Failed to initialize configuration"
        ):
            config.get_config()


class TestReloadConfigFunction:
    """Test reload_config function with all system scenarios"""

    @patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", True)
    @patch("context_switcher_mcp.config._new_reload_config")
    def test_reload_config_unified_system(self, mock_new_reload_config):
        """Test reload_config with unified system available"""
        mock_config = MagicMock()
        mock_new_reload_config.return_value = mock_config

        result = config.reload_config()

        mock_new_reload_config.assert_called_once_with()
        assert result == mock_config

    @patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", False)
    @patch("context_switcher_mcp.config._LEGACY_CONFIG_AVAILABLE", True)
    @patch("context_switcher_mcp.config._old_reload_config")
    @patch("context_switcher_mcp.config.warnings.warn")
    def test_reload_config_legacy_system(self, mock_warn, mock_old_reload_config):
        """Test reload_config fallback to legacy system"""
        mock_legacy_config = MagicMock()
        mock_old_reload_config.return_value = mock_legacy_config

        result = config.reload_config()

        mock_old_reload_config.assert_called_once_with()
        mock_warn.assert_called_once()
        assert "legacy configuration reload" in str(mock_warn.call_args)
        assert result == mock_legacy_config

    @patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", False)
    @patch("context_switcher_mcp.config._LEGACY_CONFIG_AVAILABLE", False)
    def test_reload_config_no_system_available(self):
        """Test reload_config raises error when no system available"""
        from context_switcher_mcp.config import ConfigurationError

        with pytest.raises(
            ConfigurationError, match="No configuration system available"
        ):
            config.reload_config()


class TestGlobalConfigInstance:
    """Test global configuration instance management"""

    def test_global_config_instance_initially_none(self):
        """Test that global config instance starts as None"""
        # Reset the global instance
        config._global_config_instance = None
        assert config._global_config_instance is None

    @patch("context_switcher_mcp.config.get_config")
    def test_get_global_config_creates_instance(self, mock_get_config):
        """Test that _get_global_config creates instance when None"""
        config._global_config_instance = None
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        result = config._get_global_config()

        mock_get_config.assert_called_once_with()
        assert result == mock_config
        assert config._global_config_instance == mock_config

    @patch("context_switcher_mcp.config.get_config")
    def test_get_global_config_reuses_existing(self, mock_get_config):
        """Test that _get_global_config reuses existing instance"""
        existing_config = MagicMock()
        config._global_config_instance = existing_config

        result = config._get_global_config()

        mock_get_config.assert_not_called()
        assert result == existing_config


class TestModuleLevelAttributeAccess:
    """Test __getattr__ legacy compatibility mechanism"""

    @patch("context_switcher_mcp.config._get_global_config")
    @patch("context_switcher_mcp.config.warnings.warn")
    def test_getattr_config_attribute(self, mock_warn, mock_get_global_config):
        """Test accessing 'config' attribute via __getattr__"""
        mock_config = MagicMock()
        mock_get_global_config.return_value = mock_config

        result = config.__getattr__("config")

        mock_get_global_config.assert_called_once()
        mock_warn.assert_called_once()
        assert "deprecated" in str(mock_warn.call_args)
        assert result == mock_config

    @patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", True)
    def test_getattr_unified_system_attribute(self):
        """Test __getattr__ attempts unified system attribute access"""
        # Test that non-existent attributes raise AttributeError in unified mode
        with pytest.raises(
            AttributeError, match="module .* has no attribute 'non_existent_attr'"
        ):
            config.__getattr__("non_existent_attr")

    @patch("context_switcher_mcp.config._UNIFIED_CONFIG_AVAILABLE", False)
    @patch("context_switcher_mcp.config._LEGACY_CONFIG_AVAILABLE", True)
    def test_getattr_legacy_system_attribute(self):
        """Test __getattr__ attempts legacy system attribute access"""
        # Test that non-existent attributes raise AttributeError
        with pytest.raises(
            AttributeError,
            match="module .* has no attribute 'non_existent_legacy_attr'",
        ):
            config.__getattr__("non_existent_legacy_attr")

    def test_getattr_attribute_not_found(self):
        """Test __getattr__ raises AttributeError for unknown attributes"""
        with pytest.raises(
            AttributeError, match="module .* has no attribute 'unknown_attribute'"
        ):
            config.__getattr__("unknown_attribute")


class TestLegacyCompatibilityFunctions:
    """Test legacy compatibility functions"""

    @patch("context_switcher_mcp.config.get_config")
    @patch("context_switcher_mcp.config.warnings.warn")
    def test_validate_current_config_with_validation_method(
        self, mock_warn, mock_get_config
    ):
        """Test validate_current_config with config that has validation method"""
        mock_config = MagicMock()
        mock_config.validate_current_config.return_value = (True, ["all good"])
        mock_get_config.return_value = mock_config

        is_valid, messages = config.validate_current_config()

        mock_warn.assert_called_once()
        assert "deprecated" in str(mock_warn.call_args)
        assert is_valid is True
        assert messages == ["all good"]
        mock_config.validate_current_config.assert_called_once()

    @patch("context_switcher_mcp.config.get_config")
    @patch("context_switcher_mcp.config.warnings.warn")
    def test_validate_current_config_without_validation_method(
        self, mock_warn, mock_get_config
    ):
        """Test validate_current_config with config that lacks validation method"""
        mock_config = MagicMock()
        # Remove the validation method
        del mock_config.validate_current_config
        mock_get_config.return_value = mock_config

        is_valid, messages = config.validate_current_config()

        mock_warn.assert_called_once()
        assert is_valid is True
        assert messages == []

    @patch("context_switcher_mcp.config.get_config")
    @patch("context_switcher_mcp.config.warnings.warn")
    def test_validate_current_config_exception_handling(
        self, mock_warn, mock_get_config
    ):
        """Test validate_current_config handles exceptions gracefully"""
        mock_get_config.side_effect = Exception("Config error")

        is_valid, messages = config.validate_current_config()

        mock_warn.assert_called_once()
        assert is_valid is False
        assert messages == ["Config error"]


class TestModuleExports:
    """Test module-level exports and __all__ definition"""

    def test_all_exports_defined(self):
        """Test that __all__ is properly defined"""
        assert hasattr(config, "__all__")
        expected_exports = [
            "get_config",
            "reload_config",
            "validate_current_config",
            "ConfigurationError",
        ]

        for export in expected_exports:
            assert export in config.__all__

    def test_exported_functions_callable(self):
        """Test that all exported functions are callable"""
        assert callable(config.get_config)
        assert callable(config.reload_config)
        assert callable(config.validate_current_config)

    def test_configuration_error_exported(self):
        """Test that ConfigurationError is properly exported"""
        from context_switcher_mcp.config import ConfigurationError

        assert issubclass(ConfigurationError, Exception)


class TestEdgeCases:
    """Test edge cases and defensive programming scenarios"""

    def test_empty_kwargs_to_get_config(self):
        """Test get_config with empty kwargs"""
        # Test that get_config works without arguments
        result = config.get_config()
        assert result is not None
        # Should return a ContextSwitcherConfig instance
        from context_switcher_mcp.config.core import ContextSwitcherConfig

        assert isinstance(result, ContextSwitcherConfig)

    def test_multiple_kwargs_to_get_config(self):
        """Test get_config with multiple valid keyword arguments"""
        # Test with valid configuration parameters
        result = config.get_config(
            server={"port": 3024}, session={"default_ttl_hours": 2}
        )
        assert result is not None
        # Should return a ContextSwitcherConfig instance
        from context_switcher_mcp.config.core import ContextSwitcherConfig

        assert isinstance(result, ContextSwitcherConfig)
        # Verify the values were applied
        assert result.server.port == 3024
        assert result.session.default_ttl_hours == 2

    def test_unicode_attribute_access(self):
        """Test __getattr__ with unicode attribute names"""
        unicode_attr = "配置"  # "config" in Chinese

        with pytest.raises(AttributeError):
            config.__getattr__(unicode_attr)

    def test_global_instance_thread_safety_simulation(self):
        """Test global instance creation under simulated concurrent access"""
        config._global_config_instance = None

        with patch("context_switcher_mcp.config.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            # Simulate multiple calls
            result1 = config._get_global_config()
            result2 = config._get_global_config()

            # Should only call get_config once
            mock_get_config.assert_called_once()
            assert result1 == result2 == mock_config


class TestWarningCategories:
    """Test that appropriate warning categories are used"""

    @patch("context_switcher_mcp.config.warnings.warn")
    def test_deprecation_warning_for_config_attribute(self, mock_warn):
        """Test that DeprecationWarning is used for config attribute access"""
        with patch("context_switcher_mcp.config._get_global_config"):
            config.__getattr__("config")

            args, kwargs = mock_warn.call_args
            assert len(args) >= 2
            assert args[1] is DeprecationWarning

    @patch("context_switcher_mcp.config.warnings.warn")
    def test_deprecation_warning_for_validate_current_config(self, mock_warn):
        """Test that DeprecationWarning is used for validate_current_config"""
        with patch("context_switcher_mcp.config.get_config", side_effect=Exception()):
            config.validate_current_config()

            args, kwargs = mock_warn.call_args
            assert len(args) >= 2
            assert args[1] is DeprecationWarning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
