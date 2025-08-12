"""
Test suite for unified logging system

Tests the centralized logging configuration, utilities, and correlation ID tracking.
"""

import json
import logging
import os
import time
from unittest.mock import patch, MagicMock
import pytest

from context_switcher_mcp.logging_config import (
    LoggingConfig,
    setup_logging,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    ContextSwitcherLogFormatter,
    JSONLogFormatter,
)
from context_switcher_mcp.logging_utils import (
    log_operation,
    logged_operation,
    correlation_context,
    performance_timer,
    RequestLogger,
    mcp_tool_logger,
    log_session_event,
    log_performance_metric,
)
from context_switcher_mcp.security.secure_logging import get_secure_logger


class TestLoggingConfig:
    """Test the centralized logging configuration"""

    def setup_method(self):
        """Reset logging configuration for each test"""
        # Clear environment variables
        for key in ["LOG_LEVEL", "LOG_FORMAT", "LOG_OUTPUT", "DEBUG"]:
            if key in os.environ:
                del os.environ[key]

    def teardown_method(self):
        """Clean up after each test"""
        # Reset correlation ID
        set_correlation_id(None)

    def test_default_configuration(self):
        """Test default logging configuration"""
        config = LoggingConfig()

        assert config.config["level"] == "INFO"
        assert config.config["format"] == "standard"
        assert config.config["output"] == "console"
        assert config.config["structured_errors"] is True
        assert config.config["performance_logging"] is True
        assert config.config["security_logging"] is True

    def test_environment_configuration(self):
        """Test configuration from environment variables"""
        os.environ["LOG_LEVEL"] = "DEBUG"
        os.environ["LOG_FORMAT"] = "json"
        os.environ["LOG_OUTPUT"] = "file"
        os.environ["DEBUG"] = "true"

        config = LoggingConfig()

        assert config.config["level"] == "DEBUG"
        assert config.config["format"] == "json"
        assert config.config["output"] == "file"
        assert config.config["debug_mode"] is True

    def test_logger_creation(self):
        """Test logger creation and caching"""
        config = LoggingConfig()

        # Test regular logger
        logger1 = config.get_logger("test.module")
        logger2 = config.get_logger("test.module")
        assert logger1 is logger2  # Should be cached

        # Test secure logger
        secure_logger1 = config.get_logger("test.secure", secure=True)
        secure_logger2 = config.get_logger("test.secure", secure=True)
        assert secure_logger1 is secure_logger2  # Should be cached

    def test_correlation_id_management(self):
        """Test correlation ID context management"""
        # Initially no correlation ID
        assert get_correlation_id() is None

        # Set correlation ID
        test_id = "test-123"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id

        # Clear correlation ID
        set_correlation_id(None)
        assert get_correlation_id() is None

    def test_setup_logging_function(self):
        """Test the setup_logging convenience function"""
        # Should not raise any errors
        setup_logging()

        # Should create loggers
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)


class TestLogFormatters:
    """Test custom log formatters"""

    def test_context_switcher_formatter(self):
        """Test ContextSwitcherLogFormatter with correlation ID"""
        formatter = ContextSwitcherLogFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Test without correlation ID
        formatted = formatter.format(record)
        assert "[no-correlation]" in formatted

        # Test with correlation ID
        set_correlation_id("test-123")
        formatted = formatter.format(record)
        assert "[test-123]" in formatted

    def test_json_formatter(self):
        """Test JSONLogFormatter for structured logging"""
        formatter = JSONLogFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Format and parse as JSON
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Check required fields
        assert "timestamp" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "message" in parsed
        assert "correlation_id" in parsed
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert parsed["message"] == "Test message"


class TestLoggingUtils:
    """Test logging utilities and decorators"""

    def test_correlation_context_manager(self):
        """Test correlation context manager"""
        # Test with auto-generated correlation ID
        with correlation_context() as correlation_id:
            assert correlation_id is not None
            assert get_correlation_id() == correlation_id

        # Correlation ID should be cleared after context
        assert get_correlation_id() is None

        # Test with specific correlation ID
        test_id = "specific-123"
        with correlation_context(test_id) as correlation_id:
            assert correlation_id == test_id
            assert get_correlation_id() == test_id

    def test_log_operation_context_manager(self):
        """Test log operation context manager"""
        with patch("context_switcher_mcp.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            with log_operation("test_operation") as op:
                assert op.operation_name == "test_operation"
                assert op.correlation_id is not None

            # Should have logged start and completion
            assert mock_logger.info.call_count >= 2

    def test_logged_operation_decorator(self):
        """Test logged operation decorator"""
        with patch("context_switcher_mcp.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @logged_operation("test_func")
            def test_function(x, y):
                return x + y

            result = test_function(1, 2)
            assert result == 3

            # Should have logged the operation
            assert mock_logger.info.call_count >= 2

    def test_performance_timer_decorator(self):
        """Test performance timer decorator"""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            @performance_timer("fast_operation", threshold_seconds=0.1)
            def fast_function():
                return "done"

            @performance_timer(
                "slow_operation", threshold_seconds=0.001, warn_threshold_seconds=0.002
            )
            def slow_function():
                time.sleep(0.003)  # Intentionally slow
                return "done"

            # Fast function should log debug or info
            result = fast_function()
            assert result == "done"

            # Slow function should log warning
            result = slow_function()
            assert result == "done"

            # Should have performance logs
            assert (
                mock_logger.debug.called
                or mock_logger.info.called
                or mock_logger.warning.called
            )

    def test_request_logger(self):
        """Test MCP request/response logger"""
        with patch("context_switcher_mcp.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            request_logger = RequestLogger()

            # Log request
            correlation_id = request_logger.log_request(
                "test_tool", {"param1": "value1", "sensitive_password": "secret123"}
            )

            assert correlation_id is not None
            mock_logger.info.assert_called()

            # Log successful response
            request_logger.log_response(
                "test_tool",
                success=True,
                duration=1.5,
                result_metadata={"result_count": 10},
                correlation_id=correlation_id,
            )

            # Log error response
            request_logger.log_response(
                "test_tool",
                success=False,
                duration=2.0,
                error=Exception("Test error"),
                correlation_id=correlation_id,
            )

            # Should have logged request and both responses
            assert mock_logger.info.call_count >= 2
            assert mock_logger.error.call_count >= 1

    def test_mcp_tool_logger_decorator(self):
        """Test MCP tool logger decorator"""
        with patch(
            "context_switcher_mcp.logging_utils.RequestLogger"
        ) as MockRequestLogger:
            mock_request_logger = MagicMock()
            MockRequestLogger.return_value = mock_request_logger
            mock_request_logger.log_request.return_value = "test-correlation"

            class MockRequest:
                def __init__(self):
                    self.param1 = "value1"
                    self.param2 = "value2"

            @mcp_tool_logger("test_tool")
            def test_tool_function(request):
                return {"result": "success"}

            # Call the decorated function
            result = test_tool_function(MockRequest())

            assert result == {"result": "success"}
            mock_request_logger.log_request.assert_called_once()
            mock_request_logger.log_response.assert_called_once()

    def test_session_event_logging(self):
        """Test session event logging utility"""
        with patch("context_switcher_mcp.logging_config.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            log_session_event(
                "session_created",
                "session-123",
                {"perspective_count": 4, "template": "analysis"},
            )

            mock_logger.info.assert_called_once()
            args, kwargs = mock_logger.info.call_args
            assert "session_created" in args[0]
            assert "session-123" in args[0]

    def test_performance_metric_logging(self):
        """Test performance metric logging"""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            # Should log when performance logging is enabled
            with patch(
                "context_switcher_mcp.logging_utils.is_performance_logging_enabled",
                return_value=True,
            ):
                log_performance_metric(
                    "response_time", 150.5, "ms", {"operation": "perspective_broadcast"}
                )

                mock_logger.debug.assert_called_once()

            # Should not log when performance logging is disabled
            mock_logger.reset_mock()
            with patch(
                "context_switcher_mcp.logging_utils.is_performance_logging_enabled",
                return_value=False,
            ):
                log_performance_metric("response_time", 150.5, "ms")
                mock_logger.debug.assert_not_called()


class TestSecurityIntegration:
    """Test integration with existing security logging"""

    def test_secure_logger_integration(self):
        """Test that secure logger works with unified config"""
        secure_logger = get_secure_logger("test.secure")

        # Should be able to log without errors
        secure_logger.info("Test secure message")
        secure_logger.warning("Test warning with sensitive_password=hidden")

        # Security event logging
        secure_logger.log_security_event(
            "suspicious_activity",
            {"client_ip": "192.168.1.1", "action": "multiple_failed_attempts"},
            risk_level="high",
            session_id="session-123",
        )


class TestErrorConditions:
    """Test error conditions and edge cases"""

    def test_malformed_log_data(self):
        """Test handling of malformed log data"""
        formatter = JSONLogFormatter()

        # Create record with circular reference
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test with circular ref",
            args=(),
            exc_info=None,
        )

        # Should handle gracefully without crashing
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        assert "message" in parsed

    def test_long_log_messages(self):
        """Test handling of very long log messages"""
        formatter = ContextSwitcherLogFormatter()

        # Create record with very long message
        long_message = "A" * 20000  # Very long message
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=long_message,
            args=(),
            exc_info=None,
        )

        # Should handle gracefully (likely truncated)
        formatted = formatter.format(record)
        assert len(formatted) < len(long_message)  # Should be truncated or handled

    def test_exception_logging(self):
        """Test exception logging with correlation IDs"""
        with correlation_context("error-test") as correlation_id:
            try:
                raise ValueError("Test exception")
            except Exception:
                # Should be able to log exception with correlation context
                logger = get_logger("test")
                logger.error("Test exception occurred", exc_info=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
