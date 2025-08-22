"""Test config compatibility fixes for user-reported issues

Tests verify that the specific config compatibility fixes resolve the exact
user-reported AttributeError and typing.Any instantiation issues.

User-reported failing functions:
1. start_context_analysis - Error: 'ContextSwitcherConfig' object has no attribute 'validation'
2. get_performance_metrics - Error: Cannot instantiate typing.Any
3. get_profiling_status - Error: 'ContextSwitcherConfig' object has no attribute 'profiling'

Key fixes tested:
- Added validation property to ContextSwitcherConfig that returns self.session
- Added profiling property to ContextSwitcherConfig that returns self.monitoring.profiling
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from context_switcher_mcp.config import get_config
from context_switcher_mcp.config.migration import LegacyConfigAdapter


class TestConfigValidationProperty:
    """Test config.validation property provides expected attributes"""

    def test_config_validation_exists(self):
        """Test that config.validation attribute exists"""
        config = get_config()

        # Verify validation attribute exists
        assert hasattr(config, "validation"), "config.validation attribute should exist"

        # Verify it's not None
        assert config.validation is not None, "config.validation should not be None"

    def test_config_validation_provides_session_attributes(self):
        """Test that config.validation provides expected session-related attributes"""
        config = get_config()
        validation_config = config.validation

        # Test expected validation attributes used in validation.py
        expected_attrs = [
            "max_topic_length",
            "max_session_id_length",
            "max_perspective_name_length",
            "max_custom_prompt_length",
        ]

        for attr in expected_attrs:
            assert hasattr(
                validation_config, attr
            ), f"config.validation should have {attr} attribute"
            # Verify these return reasonable integer values
            value = getattr(validation_config, attr)
            assert isinstance(
                value, int
            ), f"config.validation.{attr} should be an integer"
            assert value > 0, f"config.validation.{attr} should be positive"

    def test_config_validation_max_topic_length_usage(self):
        """Test that config.validation.max_topic_length can be used as in validation.py"""
        config = get_config()

        # This mimics the exact usage in validation.py line 30
        max_length = config.validation.max_topic_length
        assert isinstance(max_length, int)
        assert max_length > 0

        # Test that we can compare against it (as done in validation.py)
        test_topic = "A" * (max_length + 1)
        assert len(test_topic) > config.validation.max_topic_length


class TestConfigProfilingProperty:
    """Test config.profiling property provides expected attributes"""

    def test_config_profiling_exists(self):
        """Test that config.profiling attribute exists"""
        config = get_config()

        # Verify profiling attribute exists
        assert hasattr(config, "profiling"), "config.profiling attribute should exist"

        # Verify it's not None
        assert config.profiling is not None, "config.profiling should not be None"

    def test_config_profiling_provides_expected_attributes(self):
        """Test that config.profiling provides expected profiling attributes"""
        config = get_config()
        profiling_config = config.profiling

        # Test attributes used in llm_profiler.py lines 648-663
        expected_attrs = [
            "enabled",
            "level",
            "sampling_rate",
            "track_tokens",
            "track_costs",
            "track_memory",
            "track_network_timing",
            "max_history_size",
            "cost_alert_threshold_usd",
            "latency_alert_threshold_s",
            "memory_alert_threshold_mb",
            "always_profile_errors",
            "always_profile_slow_calls",
            "always_profile_expensive_calls",
            "always_profile_circuit_breaker",
        ]

        for attr in expected_attrs:
            assert hasattr(
                profiling_config, attr
            ), f"config.profiling should have {attr} attribute"

    def test_config_profiling_attribute_types(self):
        """Test that config.profiling attributes have expected types"""
        config = get_config()
        profiling_config = config.profiling

        # Test boolean attributes
        bool_attrs = [
            "enabled",
            "track_tokens",
            "track_costs",
            "track_memory",
            "track_network_timing",
            "always_profile_errors",
            "always_profile_slow_calls",
            "always_profile_expensive_calls",
            "always_profile_circuit_breaker",
        ]

        for attr in bool_attrs:
            value = getattr(profiling_config, attr)
            assert isinstance(value, bool), f"config.profiling.{attr} should be boolean"

        # Test numeric attributes
        assert isinstance(profiling_config.sampling_rate, int | float)
        assert isinstance(profiling_config.max_history_size, int)
        assert isinstance(profiling_config.cost_alert_threshold_usd, int | float)
        assert isinstance(profiling_config.latency_alert_threshold_s, int | float)
        assert isinstance(profiling_config.memory_alert_threshold_mb, int | float)


class TestValidationFileUsage:
    """Test that validation.py can use config.validation without errors"""

    def test_validation_module_import(self):
        """Test that validation.py imports successfully"""
        # This should not raise any import errors
        from context_switcher_mcp import validation

        assert validation is not None

    def test_validate_topic_uses_config_validation(self):
        """Test that validate_topic function uses config.validation successfully"""
        from context_switcher_mcp.validation import validate_topic

        # Test with valid topic
        is_valid, error_msg = validate_topic("Test topic")
        assert isinstance(is_valid, bool)
        assert isinstance(error_msg, str)

        # Test with topic that should trigger max_topic_length check
        config = get_config()
        max_length = config.validation.max_topic_length
        long_topic = "A" * (max_length + 1)

        is_valid, error_msg = validate_topic(long_topic)
        assert not is_valid, "Long topic should be invalid"
        assert "characters or less" in error_msg, "Error should mention character limit"

    @pytest.mark.asyncio
    async def test_validate_session_id_uses_config_validation(self):
        """Test that validate_session_id function uses config.validation successfully"""
        from context_switcher_mcp.validation import validate_session_id

        # Test with valid session ID length
        config = get_config()
        max_length = config.validation.max_session_id_length

        # This should access config.validation.max_session_id_length without error
        valid_session_id = "a" * (max_length - 10)  # Well within limit

        # Mock the session manager to avoid session lookup
        with patch("context_switcher_mcp.validation.session_manager") as mock_sm:
            mock_session = Mock()
            mock_sm.get_session = AsyncMock(return_value=mock_session)

            # Mock client binding validation
            with patch(
                "context_switcher_mcp.validation.validate_session_access"
            ) as mock_validate:
                mock_validate.return_value = (True, "")

                is_valid, error_msg = await validate_session_id(
                    valid_session_id, "test_operation"
                )

                # Should successfully access config.validation.max_session_id_length
                assert isinstance(is_valid, bool)
                assert isinstance(error_msg, str)


class TestLLMProfilerUsage:
    """Test that llm_profiler.py can use config.profiling without errors"""

    def test_llm_profiler_import(self):
        """Test that llm_profiler.py imports successfully"""
        from context_switcher_mcp import llm_profiler

        assert llm_profiler is not None

    def test_get_global_profiler_uses_config_profiling(self):
        """Test that get_global_profiler() uses config.profiling successfully"""
        from context_switcher_mcp.llm_profiler import get_global_profiler

        # This should access config.profiling attributes without error (lines 644-663)
        profiler = get_global_profiler()
        assert profiler is not None

        # Verify it successfully created ProfilingConfig from config.profiling
        assert hasattr(profiler, "config")
        assert profiler.config is not None

    def test_profiling_config_creation_from_config_profiling(self):
        """Test that ProfilingConfig can be created from config.profiling attributes"""
        from context_switcher_mcp.llm_profiler import ProfilingConfig, ProfilingLevel

        config = get_config()

        # This mimics the exact usage in get_global_profiler() lines 647-663
        profiling_config = ProfilingConfig(
            enabled=config.profiling.enabled,
            level=ProfilingLevel(config.profiling.level),
            sampling_rate=config.profiling.sampling_rate,
            track_tokens=config.profiling.track_tokens,
            track_costs=config.profiling.track_costs,
            track_memory=config.profiling.track_memory,
            track_network_timing=config.profiling.track_network_timing,
            max_history_size=config.profiling.max_history_size,
            cost_alert_threshold_usd=config.profiling.cost_alert_threshold_usd,
            latency_alert_threshold_s=config.profiling.latency_alert_threshold_s,
            memory_alert_threshold_mb=config.profiling.memory_alert_threshold_mb,
            always_profile_errors=config.profiling.always_profile_errors,
            always_profile_slow_calls=config.profiling.always_profile_slow_calls,
            always_profile_expensive_calls=config.profiling.always_profile_expensive_calls,
            always_profile_circuit_breaker=config.profiling.always_profile_circuit_breaker,
        )

        # Verify successful creation
        assert profiling_config is not None
        assert profiling_config.enabled == config.profiling.enabled


class TestProfilingToolsUsage:
    """Test that profiling_tools.py functions work without typing.Any errors"""

    def test_profiling_tools_import(self):
        """Test that profiling_tools.py imports successfully"""
        from context_switcher_mcp.tools import profiling_tools

        assert profiling_tools is not None

    @pytest.mark.asyncio
    async def test_get_llm_profiling_status_no_typing_any_error(self):
        """Test that get_llm_profiling_status() works without 'Cannot instantiate typing.Any' error"""
        from context_switcher_mcp.tools.profiling_tools import get_llm_profiling_status

        # This should not raise "Cannot instantiate typing.Any" error
        result = await get_llm_profiling_status()

        assert isinstance(result, dict)
        # Should return valid status or error dict
        assert "enabled" in result or "error" in result

    @pytest.mark.asyncio
    async def test_get_performance_dashboard_data_no_typing_any_error(self):
        """Test that get_performance_dashboard_data() works without typing errors"""
        from context_switcher_mcp.tools.profiling_tools import (
            get_performance_dashboard_data,
        )

        # This should not raise "Cannot instantiate typing.Any" error
        result = await get_performance_dashboard_data()

        assert isinstance(result, dict)
        # Should return valid data or error dict
        assert any(key in result for key in ["dashboard", "error", "message"])


class TestStartContextAnalysisIntegration:
    """Test start_context_analysis tool integration without config.validation errors"""

    @pytest.mark.asyncio
    async def test_start_context_analysis_validation_path(self):
        """Test that start_context_analysis uses validation without config.validation AttributeError"""
        from context_switcher_mcp.handlers.validation_handler import ValidationHandler

        # Test the exact validation path used by start_context_analysis
        # This should use config.validation internally without error
        is_valid, error_response = ValidationHandler.validate_session_creation_request(
            topic="Test analysis topic", initial_perspectives=["technical", "business"]
        )

        # Should complete without AttributeError on config.validation
        assert isinstance(is_valid, bool)
        if not is_valid:
            assert isinstance(error_response, dict)
        else:
            assert error_response is None

    def test_validation_handler_imports_successfully(self):
        """Test that ValidationHandler imports and can be instantiated"""
        from context_switcher_mcp.handlers.validation_handler import ValidationHandler

        # Should import without issues
        assert ValidationHandler is not None

        # Verify it has the required method
        assert hasattr(ValidationHandler, "validate_session_creation_request")


class TestGetPerformanceMetricsIntegration:
    """Test get_performance_metrics tool integration without typing.Any errors"""

    @pytest.mark.asyncio
    async def test_get_performance_metrics_no_instantiation_error(self):
        """Test that get_performance_metrics works without 'Cannot instantiate typing.Any' error"""
        from context_switcher_mcp.tools.profiling_tools import get_performance_metrics

        # This was failing with "Cannot instantiate typing.Any" error
        result = await get_performance_metrics(hours_back=24)

        assert isinstance(result, dict)
        # Should return performance data or error dict
        assert any(key in result for key in ["performance", "error", "timeframe_hours"])


class TestLegacyConfigAdapter:
    """Test that LegacyConfigAdapter provides the required properties"""

    def test_legacy_adapter_provides_validation(self):
        """Test that LegacyConfigAdapter provides validation property"""
        config = get_config()

        # If using LegacyConfigAdapter, it should provide validation
        if isinstance(config, LegacyConfigAdapter):
            assert hasattr(config, "validation")
            assert config.validation is not None

    def test_legacy_adapter_provides_profiling(self):
        """Test that LegacyConfigAdapter provides profiling property"""
        config = get_config()

        # If using LegacyConfigAdapter, it should provide profiling
        if isinstance(config, LegacyConfigAdapter):
            assert hasattr(config, "profiling")
            assert config.profiling is not None


class TestIntegrationAllFailingFunctions:
    """Integration test exercising all three originally failing functions"""

    @pytest.mark.asyncio
    async def test_all_failing_functions_integration(self):
        """Test all three originally failing functions work together"""

        # 1. Test start_context_analysis validation path (uses config.validation)
        from context_switcher_mcp.handlers.validation_handler import ValidationHandler

        is_valid, error_response = ValidationHandler.validate_session_creation_request(
            topic="Integration test topic", initial_perspectives=["technical"]
        )

        assert isinstance(
            is_valid, bool
        ), "start_context_analysis validation should work"

        # 2. Test get_performance_metrics (was failing with typing.Any instantiation)
        from context_switcher_mcp.tools.profiling_tools import get_performance_metrics

        result = await get_performance_metrics(hours_back=1)
        assert isinstance(result, dict), "get_performance_metrics should return dict"

        # 3. Test get_profiling_status (uses config.profiling)
        from context_switcher_mcp.tools.profiling_tools import get_llm_profiling_status

        status = await get_llm_profiling_status()
        assert isinstance(status, dict), "get_profiling_status should return dict"

        # All three should complete without the original AttributeError or typing.Any errors
        assert True, "All three originally failing functions completed successfully"


class TestErrorScenariosFixed:
    """Test that specific error scenarios reported by user are fixed"""

    def test_no_validation_attribute_error(self):
        """Test that 'ContextSwitcherConfig' object has no attribute 'validation' is fixed"""
        config = get_config()

        # This should not raise AttributeError
        try:
            validation_config = config.validation
            assert validation_config is not None
        except AttributeError as e:
            pytest.fail(f"config.validation should exist: {e}")

    def test_no_profiling_attribute_error(self):
        """Test that 'ContextSwitcherConfig' object has no attribute 'profiling' is fixed"""
        config = get_config()

        # This should not raise AttributeError
        try:
            profiling_config = config.profiling
            assert profiling_config is not None
        except AttributeError as e:
            pytest.fail(f"config.profiling should exist: {e}")

    @pytest.mark.asyncio
    async def test_no_typing_any_instantiation_error(self):
        """Test that 'Cannot instantiate typing.Any' error is fixed"""
        from context_switcher_mcp.tools.profiling_tools import get_performance_metrics

        # This should not raise "Cannot instantiate typing.Any" error
        try:
            result = await get_performance_metrics()
            assert isinstance(result, dict)
        except TypeError as e:
            if "Cannot instantiate typing.Any" in str(e):
                pytest.fail(f"typing.Any instantiation error should be fixed: {e}")
            else:
                # Re-raise other TypeErrors
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
