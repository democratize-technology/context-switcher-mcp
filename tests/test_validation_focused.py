"""Focused tests for validation.py - testing actual available functions"""

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from context_switcher_mcp.validation import (  # noqa: E402
    validate_session_id,
    validate_topic,
)


class TestTopicValidation:
    """Test topic validation with comprehensive edge cases"""

    def test_valid_topics(self):
        """Test validation of valid topics"""
        valid_topics = [
            "Analyze system architecture",
            "Performance optimization strategies",
            "Code review feedback",
            "API design considerations",
            "Database schema analysis",
            "Security vulnerability assessment",
        ]

        for topic in valid_topics:
            is_valid, error_msg = validate_topic(topic)
            assert is_valid is True
            assert error_msg == ""

    def test_invalid_topics_basic(self):
        """Test validation rejects invalid topics"""
        invalid_topics = [
            "",  # Empty string
            "   ",  # Whitespace only
            None,  # None value
            123,  # Wrong type
            [],  # Wrong type
        ]

        for topic in invalid_topics:
            is_valid, error_msg = validate_topic(topic)
            assert is_valid is False
            assert len(error_msg) > 0

    def test_topic_length_limits(self):
        """Test topic length validation"""
        # Test very long topic (over 1000 chars by default)
        long_topic = "A" * 1500

        is_valid, error_msg = validate_topic(long_topic)
        assert is_valid is False
        assert "characters or less" in error_msg or "too long" in error_msg.lower()

    def test_suspicious_pattern_detection(self):
        """Test detection of suspicious patterns in topics"""
        suspicious_topics = [
            "Analyze this <script>alert('xss')</script> code",
            "Review this javascript:alert('test') function",
            "Check this <?php echo 'test'; ?> snippet",
            "Examine eval('dangerous code') pattern",
            "Look at document.cookie access",
            "Review onclick=malicious() handler",
            "Analyze prompt('user input') usage",
        ]

        for topic in suspicious_topics:
            is_valid, error_msg = validate_topic(topic)
            assert is_valid is False
            assert "suspicious pattern" in error_msg.lower()

    def test_topic_edge_cases(self):
        """Test topic validation edge cases"""
        edge_cases = [
            ("Normal topic with numbers 123", True),
            ("Topic with special chars: @#$%^&*()", True),
            ("Unicode topic with émojis 🚀", True),
            ("Topic with\nnewlines", True),  # Should be valid
            ("Topic with\ttabs", True),  # Should be valid
        ]

        for topic, should_be_valid in edge_cases:
            is_valid, error_msg = validate_topic(topic)
            if should_be_valid:
                assert (
                    is_valid is True
                ), f"Expected '{topic}' to be valid but got: {error_msg}"
            else:
                assert is_valid is False, f"Expected '{topic}' to be invalid"

    def test_case_insensitive_pattern_detection(self):
        """Test that suspicious pattern detection is case-insensitive"""
        case_variations = [
            "Topic with <SCRIPT>alert('test')</SCRIPT>",
            "Review JAVASCRIPT:alert('test')",
            "Check EVAL('code') usage",
            "Analyze DOCUMENT.COOKIE access",
        ]

        for topic in case_variations:
            is_valid, error_msg = validate_topic(topic)
            assert is_valid is False
            assert "suspicious pattern" in error_msg.lower()


class TestSessionIdValidation:
    """Test session ID validation with async patterns"""

    @pytest.mark.asyncio
    async def test_valid_session_ids(self):
        """Test validation of valid session IDs with proper mocking"""
        with patch("context_switcher_mcp.session_manager") as mock_session_manager:
            # Mock successful session lookup
            mock_session = Mock()
            mock_session_manager.get_session = AsyncMock(return_value=mock_session)

            with patch(
                "context_switcher_mcp.validation.validate_session_access"
            ) as mock_validate_access:
                mock_validate_access.return_value = (True, "")

                valid_ids = [
                    str(uuid.uuid4()),
                    "session-abc-123",
                    "valid-session-id-001",
                ]

                for session_id in valid_ids:
                    is_valid, error_msg = await validate_session_id(
                        session_id, "test_operation"
                    )
                    assert is_valid is True
                    assert error_msg == ""

    @pytest.mark.asyncio
    async def test_invalid_session_ids_basic(self):
        """Test validation rejects invalid session IDs"""
        invalid_cases = [
            ("", "must be a non-empty string"),
            (None, "must be a non-empty string"),
            (123, "must be a non-empty string"),
        ]

        for session_id, expected_error_fragment in invalid_cases:
            is_valid, error_msg = await validate_session_id(
                session_id, "test_operation"
            )
            assert is_valid is False
            assert expected_error_fragment.lower() in error_msg.lower()

    @pytest.mark.asyncio
    async def test_session_not_found(self):
        """Test behavior when session is not found"""
        with patch("context_switcher_mcp.session_manager") as mock_session_manager:
            # Mock session not found
            mock_session_manager.get_session = AsyncMock(return_value=None)

            is_valid, error_msg = await validate_session_id(
                "nonexistent-session", "test_operation"
            )
            assert is_valid is False
            assert "not found" in error_msg.lower() or "expired" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_session_access_denied(self):
        """Test behavior when session access is denied"""
        with patch("context_switcher_mcp.session_manager") as mock_session_manager:
            mock_session = Mock()
            mock_session_manager.get_session = AsyncMock(return_value=mock_session)

            with patch(
                "context_switcher_mcp.validation.validate_session_access"
            ) as mock_validate_access:
                # Mock access denied
                mock_validate_access.return_value = (
                    False,
                    "Access denied for security reasons",
                )

                is_valid, error_msg = await validate_session_id(
                    "test-session", "restricted_operation"
                )
                assert is_valid is False
                assert len(error_msg) > 0

    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self):
        """Test special handling of rate limiting errors"""
        with patch("context_switcher_mcp.session_manager") as mock_session_manager:
            mock_session = Mock()
            mock_session_manager.get_session = AsyncMock(return_value=mock_session)

            with patch(
                "context_switcher_mcp.validation.validate_session_access"
            ) as mock_validate_access:
                # Mock rate limiting error
                mock_validate_access.return_value = (
                    False,
                    "excessive_access_rate detected",
                )

                is_valid, error_msg = await validate_session_id(
                    "test-session", "high_frequency_operation"
                )
                assert is_valid is False
                assert "rate limiting" in error_msg.lower()
                assert "wait before trying again" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_long_session_id(self):
        """Test validation of very long session IDs"""
        # Create very long session ID (over 100 chars)
        long_session_id = "a" * 150

        is_valid, error_msg = await validate_session_id(
            long_session_id, "test_operation"
        )
        assert is_valid is False
        assert "characters or less" in error_msg or "too long" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_session_validation_with_different_operations(self):
        """Test session validation with different operation types"""
        with patch("context_switcher_mcp.session_manager") as mock_session_manager:
            mock_session = Mock()
            mock_session_manager.get_session = AsyncMock(return_value=mock_session)

            with patch(
                "context_switcher_mcp.validation.validate_session_access"
            ) as mock_validate_access:
                mock_validate_access.return_value = (True, "")

                operations = [
                    "analyze_from_perspectives",
                    "synthesize_perspectives",
                    "get_session",
                    "add_perspective",
                    "stream_analysis",
                ]

                for operation in operations:
                    is_valid, error_msg = await validate_session_id(
                        "test-session", operation
                    )
                    assert is_valid is True
                    assert error_msg == ""

                    # Verify the operation was passed to validate_session_access
                    mock_validate_access.assert_called_with(mock_session, operation)


class TestValidationConfiguration:
    """Test validation configuration handling"""

    def test_config_fallback(self):
        """Test that validation works with fallback config"""
        with patch("context_switcher_mcp.validation._get_config") as mock_get_config:
            # Mock config loading failure
            mock_get_config.side_effect = Exception("Config loading failed")

            # Validation should still work with fallback
            is_valid, error_msg = validate_topic("Test topic")
            # Might succeed or fail, but shouldn't crash
            assert isinstance(is_valid, bool)
            assert isinstance(error_msg, str)

    def test_config_limits_respected(self):
        """Test that configured limits are respected"""

        # Test with mocked config
        class MockValidation:
            max_topic_length = 50
            max_session_id_length = 20

        class MockConfig:
            validation = MockValidation()

        with patch(
            "context_switcher_mcp.validation._get_config", return_value=MockConfig()
        ):
            # Test topic length limit
            long_topic = "A" * 60  # Over the 50 char limit
            is_valid, error_msg = validate_topic(long_topic)
            assert is_valid is False
            assert "50 characters" in error_msg

    @pytest.mark.asyncio
    async def test_session_id_config_limits(self):
        """Test session ID length limits from config"""

        class MockValidation:
            max_session_id_length = 20

        class MockConfig:
            validation = MockValidation()

        with patch(
            "context_switcher_mcp.validation._get_config", return_value=MockConfig()
        ):
            # Test session ID length limit
            long_session_id = "a" * 30  # Over the 20 char limit
            is_valid, error_msg = await validate_session_id(long_session_id, "test")
            assert is_valid is False
            assert "20 characters" in error_msg


class TestValidationErrorHandling:
    """Test validation error handling and sanitization"""

    @pytest.mark.asyncio
    async def test_error_message_sanitization(self):
        """Test that error messages are properly sanitized"""
        with patch("context_switcher_mcp.session_manager") as mock_session_manager:
            mock_session = Mock()
            mock_session_manager.get_session = AsyncMock(return_value=mock_session)

            with patch(
                "context_switcher_mcp.validation.validate_session_access"
            ) as mock_validate_access:
                # Mock access error with potentially sensitive info
                mock_validate_access.return_value = (
                    False,
                    "Access denied: internal error details",
                )

                with patch(
                    "context_switcher_mcp.validation.sanitize_error_message"
                ) as mock_sanitize:
                    mock_sanitize.return_value = "Access denied"

                    is_valid, error_msg = await validate_session_id(
                        "test-session", "test_operation"
                    )
                    assert is_valid is False

                    # Verify sanitization was called
                    mock_sanitize.assert_called_once()

    def test_validation_exception_handling(self):
        """Test that validation handles exceptions gracefully"""
        # Test with topic validation when config access fails
        with patch("context_switcher_mcp.validation._get_config") as mock_get_config:
            mock_get_config.side_effect = Exception("Unexpected error")

            # Should not raise exception, should handle gracefully
            try:
                is_valid, error_msg = validate_topic("test topic")
                assert isinstance(is_valid, bool)
                assert isinstance(error_msg, str)
            except Exception as e:
                pytest.fail(f"Validation raised unexpected exception: {e}")


class TestValidationPerformance:
    """Test validation performance with large inputs"""

    def test_topic_validation_performance(self):
        """Test topic validation performance with large inputs"""
        import time

        # Test with large topic
        large_topic = "A" * 5000

        start_time = time.time()
        is_valid, error_msg = validate_topic(large_topic)
        end_time = time.time()

        # Should complete quickly (under 1 second)
        assert end_time - start_time < 1.0

        # Should reject due to length
        assert is_valid is False

    def test_suspicious_pattern_scan_performance(self):
        """Test performance of suspicious pattern scanning"""
        import time

        # Create topic with patterns at the end (worst case for scanning)
        topic_with_late_pattern = "A" * 500 + "<script>alert('test')</script>"

        start_time = time.time()
        is_valid, error_msg = validate_topic(topic_with_late_pattern)
        end_time = time.time()

        # Should complete quickly even with pattern at end
        assert end_time - start_time < 0.5
        assert is_valid is False
        assert "suspicious pattern" in error_msg.lower()


class TestValidationIntegration:
    """Test validation integration with other components"""

    @pytest.mark.asyncio
    async def test_session_manager_integration(self):
        """Test integration with session manager"""
        # This tests the actual import and usage pattern
        with patch("context_switcher_mcp.session_manager") as mock_session_manager:
            mock_session_manager.get_session = AsyncMock(return_value=Mock())

            with patch(
                "context_switcher_mcp.validation.validate_session_access"
            ) as mock_validate:
                mock_validate.return_value = (True, "")

                # Should successfully integrate with session manager
                is_valid, error_msg = await validate_session_id(
                    "test-session", "test_op"
                )
                assert is_valid is True

                # Verify session manager was called correctly
                mock_session_manager.get_session.assert_called_once_with("test-session")

    def test_logging_integration(self):
        """Test that validation integrates with logging"""
        # Test that logger is properly initialized
        from context_switcher_mcp.validation import logger

        assert logger is not None
        assert hasattr(logger, "error")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")

    def test_config_integration(self):
        """Test integration with config system"""
        # Test the config loading mechanism
        from context_switcher_mcp.validation import _get_config

        config = _get_config()
        assert config is not None
        assert hasattr(config, "validation")
        assert hasattr(config.validation, "max_topic_length")
        assert hasattr(config.validation, "max_session_id_length")
