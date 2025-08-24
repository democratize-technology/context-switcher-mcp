"""Tests for model_validators module - achieving 100% code coverage"""

from unittest.mock import patch

import pytest
from context_switcher_mcp.model_validators import (  # noqa: E402
    ModelIdValidator,
    _model_validator,
    validate_model_id,
)


class TestModelIdValidator:
    """Test the ModelIdValidator class"""

    def test_validator_class_exists(self):
        """Test that ModelIdValidator class can be instantiated"""
        validator = ModelIdValidator()
        assert validator is not None
        assert isinstance(validator, ModelIdValidator)

    def test_validate_model_id_empty_string(self):
        """Test validation with empty string"""
        validator = ModelIdValidator()
        is_valid, error_msg = validator.validate_model_id("")

        assert is_valid is False
        assert error_msg == "Model ID must be a non-empty string"

    def test_validate_model_id_none(self):
        """Test validation with None"""
        validator = ModelIdValidator()
        is_valid, error_msg = validator.validate_model_id(None)

        assert is_valid is False
        assert error_msg == "Model ID must be a non-empty string"

    def test_validate_model_id_non_string(self):
        """Test validation with non-string input"""
        validator = ModelIdValidator()
        is_valid, error_msg = validator.validate_model_id(123)

        assert is_valid is False
        assert error_msg == "Model ID must be a non-empty string"

    def test_validate_model_id_too_long(self):
        """Test validation with overly long model ID"""
        validator = ModelIdValidator()
        long_model_id = "a" * 201  # 201 characters, exceeding the 200 limit
        is_valid, error_msg = validator.validate_model_id(long_model_id)

        assert is_valid is False
        assert error_msg == "Model ID too long (max 200 characters)"

    def test_validate_model_id_exactly_200_chars(self):
        """Test validation with exactly 200 characters (boundary condition)"""
        validator = ModelIdValidator()
        model_id_200 = "a" * 200  # Exactly 200 characters

        with patch(
            "context_switcher_mcp.model_validators.SecurityPatternMatcher"
        ) as mock_matcher:
            mock_matcher.match_model_patterns.return_value = ["valid_pattern"]

            is_valid, error_msg = validator.validate_model_id(model_id_200)

            assert is_valid is True
            assert error_msg == ""
            mock_matcher.match_model_patterns.assert_called_once_with(model_id_200)

    @patch("context_switcher_mcp.model_validators.SecurityPatternMatcher")
    def test_validate_model_id_valid_pattern(self, mock_matcher):
        """Test validation with valid model ID that matches patterns"""
        mock_matcher.match_model_patterns.return_value = ["gpt-4", "claude-3"]

        validator = ModelIdValidator()
        is_valid, error_msg = validator.validate_model_id("gpt-4")

        assert is_valid is True
        assert error_msg == ""
        mock_matcher.match_model_patterns.assert_called_once_with("gpt-4")

    @patch("context_switcher_mcp.model_validators.SecurityPatternMatcher")
    @patch("context_switcher_mcp.model_validators.logger")
    def test_validate_model_id_invalid_pattern(self, mock_logger, mock_matcher):
        """Test validation with model ID that doesn't match any patterns"""
        mock_matcher.match_model_patterns.return_value = []  # No matching patterns

        validator = ModelIdValidator()
        is_valid, error_msg = validator.validate_model_id("invalid-model")

        assert is_valid is False
        assert error_msg == "Model ID 'invalid-model' not in allowed patterns"
        mock_matcher.match_model_patterns.assert_called_once_with("invalid-model")
        mock_logger.warning.assert_called_once_with(
            "Rejected invalid model ID: invalid-model"
        )

    @patch("context_switcher_mcp.model_validators.SecurityPatternMatcher")
    def test_validate_model_id_empty_matching_patterns(self, mock_matcher):
        """Test validation when SecurityPatternMatcher returns empty list"""
        mock_matcher.match_model_patterns.return_value = []

        validator = ModelIdValidator()
        is_valid, error_msg = validator.validate_model_id("test-model")

        assert is_valid is False
        assert "not in allowed patterns" in error_msg

    @patch("context_switcher_mcp.model_validators.SecurityPatternMatcher")
    def test_validate_model_id_none_matching_patterns(self, mock_matcher):
        """Test validation when SecurityPatternMatcher returns None"""
        mock_matcher.match_model_patterns.return_value = None

        validator = ModelIdValidator()
        is_valid, error_msg = validator.validate_model_id("test-model")

        assert is_valid is False
        assert "not in allowed patterns" in error_msg


class TestGlobalValidatorFunction:
    """Test the global validate_model_id function"""

    def test_global_validator_instance_exists(self):
        """Test that global validator instance exists"""
        assert _model_validator is not None
        assert isinstance(_model_validator, ModelIdValidator)

    @patch("context_switcher_mcp.model_validators._model_validator")
    def test_global_function_delegates_to_instance(self, mock_validator):
        """Test that global function delegates to the global instance"""
        mock_validator.validate_model_id.return_value = (True, "")

        result = validate_model_id("test-model")

        assert result == (True, "")
        mock_validator.validate_model_id.assert_called_once_with("test-model")

    def test_global_function_with_valid_input(self):
        """Test global function with valid input"""
        with patch(
            "context_switcher_mcp.model_validators.SecurityPatternMatcher"
        ) as mock_matcher:
            mock_matcher.match_model_patterns.return_value = ["valid_pattern"]

            is_valid, error_msg = validate_model_id("gpt-3.5-turbo")

            assert is_valid is True
            assert error_msg == ""

    def test_global_function_with_invalid_input(self):
        """Test global function with invalid input"""
        is_valid, error_msg = validate_model_id(None)

        assert is_valid is False
        assert error_msg == "Model ID must be a non-empty string"


class TestEdgeCases:
    """Test edge cases and defensive programming scenarios"""

    def test_whitespace_only_model_id(self):
        """Test model ID with only whitespace"""
        validator = ModelIdValidator()

        with patch(
            "context_switcher_mcp.model_validators.SecurityPatternMatcher"
        ) as mock_matcher:
            mock_matcher.match_model_patterns.return_value = []

            is_valid, error_msg = validator.validate_model_id("   ")

            # Should fail pattern matching, not empty string validation
            assert is_valid is False
            assert "not in allowed patterns" in error_msg

    def test_unicode_model_id(self):
        """Test model ID with unicode characters"""
        validator = ModelIdValidator()
        unicode_model = "模型-🤖-test"

        with patch(
            "context_switcher_mcp.model_validators.SecurityPatternMatcher"
        ) as mock_matcher:
            mock_matcher.match_model_patterns.return_value = ["unicode_pattern"]

            is_valid, error_msg = validator.validate_model_id(unicode_model)

            assert is_valid is True
            assert error_msg == ""
            mock_matcher.match_model_patterns.assert_called_once_with(unicode_model)

    def test_model_id_with_special_characters(self):
        """Test model ID with special characters"""
        validator = ModelIdValidator()
        special_model = "model-v1.0_beta@company.com"

        with patch(
            "context_switcher_mcp.model_validators.SecurityPatternMatcher"
        ) as mock_matcher:
            mock_matcher.match_model_patterns.return_value = []

            is_valid, error_msg = validator.validate_model_id(special_model)

            assert is_valid is False
            mock_matcher.match_model_patterns.assert_called_once_with(special_model)

    @patch("context_switcher_mcp.model_validators.SecurityPatternMatcher")
    def test_security_pattern_matcher_exception_handling(self, mock_matcher):
        """Test behavior when SecurityPatternMatcher raises exception"""
        mock_matcher.match_model_patterns.side_effect = Exception(
            "Pattern matcher error"
        )

        validator = ModelIdValidator()

        # Should propagate the exception (no exception handling in the original code)
        with pytest.raises(Exception, match="Pattern matcher error"):
            validator.validate_model_id("test-model")


class TestModuleImports:
    """Test module-level imports and dependencies"""

    def test_logger_import(self):
        """Test that logger is properly imported and configured"""
        from context_switcher_mcp.model_validators import logger

        assert logger is not None
        # Verify it's a logger-like object
        assert hasattr(logger, "warning")
        assert callable(logger.warning)

    def test_security_pattern_matcher_import(self):
        """Test that SecurityPatternMatcher is properly imported"""
        # This test verifies the import works by actually using the class
        with patch(
            "context_switcher_mcp.model_validators.SecurityPatternMatcher"
        ) as mock_matcher:
            mock_matcher.match_model_patterns.return_value = ["test"]

            from context_switcher_mcp.model_validators import validate_model_id

            is_valid, _ = validate_model_id("test-model")

            mock_matcher.match_model_patterns.assert_called_once()

    def test_all_exports_available(self):
        """Test that all expected exports are available"""
        import context_switcher_mcp.model_validators as module

        # Check that all expected classes and functions are exported
        assert hasattr(module, "ModelIdValidator")
        assert hasattr(module, "validate_model_id")
        assert hasattr(module, "_model_validator")

        # Verify types
        assert callable(module.ModelIdValidator)
        assert callable(module.validate_model_id)
        assert isinstance(module._model_validator, module.ModelIdValidator)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
