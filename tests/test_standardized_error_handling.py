"""Comprehensive tests for standardized error handling patterns"""

import asyncio

import pytest
from context_switcher_mcp.error_classification import (
    ErrorCategory,
    ErrorSeverity,
    classify_error,
    get_retry_parameters,
    is_retriable_error,
    is_transient_error,
)
from context_switcher_mcp.error_context import (
    ErrorAccumulator,
    error_context,
    resource_cleanup_context,
    suppress_and_log,
)
from context_switcher_mcp.error_decorators import (
    handle_model_errors,
    log_errors_with_context,
    retry_on_transient_errors,
    validate_parameters,
)
from context_switcher_mcp.error_logging import (
    StructuredErrorLogger,
)

# Import the new error handling components
from context_switcher_mcp.exceptions import (
    AuthenticationError,
    ConcurrencyError,
    ContextSwitcherError,
    LockTimeoutError,
    ModelAuthenticationError,
    ModelBackendError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelValidationError,
    NetworkError,
    NetworkTimeoutError,
    ParameterValidationError,
    PerformanceError,
    PerformanceTimeoutError,
    SecurityError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Test the extended exception hierarchy"""

    def test_exception_inheritance(self):
        """Test that new exceptions inherit correctly"""
        # Test model validation error
        validation_error = ModelValidationError(
            "Invalid model", validation_context={"model": "test-model"}
        )
        assert isinstance(validation_error, ModelBackendError)
        assert isinstance(validation_error, ValidationError)
        assert hasattr(validation_error, "validation_context")
        assert validation_error.validation_context["model"] == "test-model"

        # Test model authentication error
        auth_error = ModelAuthenticationError(
            "Invalid credentials", security_context={"auth_type": "api_key"}
        )
        assert isinstance(auth_error, ModelBackendError)
        assert isinstance(auth_error, SecurityError)
        assert hasattr(auth_error, "security_context")
        assert auth_error.security_context["auth_type"] == "api_key"

        # Test model rate limit error
        rate_error = ModelRateLimitError(
            "Rate limit exceeded",
            network_context={"backend": "bedrock", "limit_type": "api"},
        )
        assert isinstance(rate_error, ModelBackendError)
        assert isinstance(rate_error, NetworkError)
        assert hasattr(rate_error, "network_context")
        assert rate_error.network_context["backend"] == "bedrock"

    def test_context_preservation(self):
        """Test that error contexts are preserved correctly"""
        network_error = NetworkTimeoutError(
            "Connection timed out",
            network_context={"host": "api.example.com", "timeout": 30},
        )

        assert network_error.network_context["host"] == "api.example.com"
        assert network_error.network_context["timeout"] == 30
        assert isinstance(network_error, NetworkError)

    def test_base_exception_compatibility(self):
        """Test that all new exceptions are compatible with base Exception"""
        exceptions = [
            ModelValidationError("test"),
            NetworkTimeoutError("test"),
            SecurityError("test"),
            ConcurrencyError("test"),
            ValidationError("test"),
            PerformanceError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, Exception)
            assert isinstance(exc, ContextSwitcherError)
            assert str(exc) == "test"


class TestErrorClassification:
    """Test error classification and handling decisions"""

    def test_classify_model_errors(self):
        """Test classification of model backend errors"""
        connection_error = ModelConnectionError("Network failed")
        classification = classify_error(connection_error)

        assert classification["severity"] == ErrorSeverity.HIGH
        assert classification["category"] == ErrorCategory.TRANSIENT
        assert classification["transient"] is True
        assert classification["retriable"] is True
        assert classification["auto_recover"] is True

    def test_classify_security_errors(self):
        """Test classification of security errors"""
        auth_error = AuthenticationError("Invalid credentials")
        classification = classify_error(auth_error)

        assert classification["severity"] == ErrorSeverity.CRITICAL
        assert classification["category"] == ErrorCategory.SECURITY
        assert classification["transient"] is False
        assert classification["retriable"] is False
        assert classification["user_facing"] is True

    def test_error_utility_functions(self):
        """Test error classification utility functions"""
        # Transient error
        timeout_error = NetworkTimeoutError("Timeout")
        assert is_transient_error(timeout_error)
        assert is_retriable_error(timeout_error)

        # Permanent error
        validation_error = ParameterValidationError("Invalid parameter")
        assert not is_transient_error(validation_error)
        assert not is_retriable_error(validation_error)

    def test_retry_parameters(self):
        """Test retry parameter extraction"""
        connection_error = ModelConnectionError("Connection failed")
        retry_params = get_retry_parameters(connection_error)

        assert retry_params["should_retry"] is True
        assert retry_params["delay"] == 2.0
        assert retry_params["max_retries"] == 3


class TestErrorDecorators:
    """Test error handling decorators"""

    @pytest.mark.asyncio
    async def test_handle_model_errors_decorator(self):
        """Test model error handling decorator"""
        call_count = 0

        @handle_model_errors(fallback_result={"error": "Model unavailable"})
        async def failing_model_call():
            nonlocal call_count
            call_count += 1
            raise ModelConnectionError("Connection failed")

        result = await failing_model_call()
        assert result == {"error": "Model unavailable"}
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_decorator(self):
        """Test retry decorator with transient errors"""
        call_count = 0

        @retry_on_transient_errors(max_retries=2, base_delay=0.01)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ModelConnectionError("Temporary failure")
            return "success"

        result = await flaky_operation()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_decorator_permanent_failure(self):
        """Test retry decorator gives up on permanent errors"""
        call_count = 0

        @retry_on_transient_errors(max_retries=2, base_delay=0.01)
        async def permanent_failure():
            nonlocal call_count
            call_count += 1
            raise ValidationError("Invalid input")

        with pytest.raises(ValidationError):
            await permanent_failure()

        # Should not retry validation errors
        assert call_count == 1

    def test_parameter_validation_decorator(self):
        """Test parameter validation decorator"""

        @validate_parameters(
            name=lambda x: len(x) > 0, age=lambda x: x >= 0 and x <= 150
        )
        def create_user(name: str, age: int):
            return f"User: {name}, Age: {age}"

        # Valid parameters
        result = create_user("John", 25)
        assert result == "User: John, Age: 25"

        # Invalid name
        with pytest.raises(ValidationError) as exc_info:
            create_user("", 25)
        assert "name" in str(exc_info.value)

        # Invalid age
        with pytest.raises(ValidationError) as exc_info:
            create_user("John", -5)
        assert "age" in str(exc_info.value)


class TestErrorContext:
    """Test error context managers"""

    @pytest.mark.asyncio
    async def test_error_context_success(self):
        """Test error context manager with successful operation"""
        async with error_context(
            "test_operation", user_context={"test": "value"}
        ) as ctx:
            ctx["result"] = "success"
            # Simulate some work
            await asyncio.sleep(0.01)

        # Should complete without issues
        assert ctx["result"] == "success"

    @pytest.mark.asyncio
    async def test_error_context_failure(self):
        """Test error context manager with failed operation"""
        with pytest.raises(ValueError):
            async with error_context(
                "failing_operation", user_context={"test": "value"}
            ) as ctx:
                ctx["step"] = "before_failure"
                raise ValueError("Test failure")

    @pytest.mark.asyncio
    async def test_suppress_and_log(self):
        """Test suppress and log context manager"""
        suppressed_errors = []

        def fallback():
            suppressed_errors.append("fallback_called")

        async with suppress_and_log(
            ValueError, fallback_action=fallback, operation_name="test_suppress"
        ):
            raise ValueError("This should be suppressed")

        assert len(suppressed_errors) == 1
        assert suppressed_errors[0] == "fallback_called"

    @pytest.mark.asyncio
    async def test_resource_cleanup_context(self):
        """Test resource cleanup context manager"""
        cleaned_up = []

        def cleanup1():
            cleaned_up.append("cleanup1")

        def cleanup2():
            cleaned_up.append("cleanup2")

        try:
            async with resource_cleanup_context(
                [cleanup1, cleanup2], operation_name="test_cleanup"
            ):
                raise ValueError("Operation failed")
        except ValueError:
            pass

        # Both cleanup functions should have been called
        assert "cleanup1" in cleaned_up
        assert "cleanup2" in cleaned_up

    def test_error_accumulator(self):
        """Test error accumulator for batch operations"""
        with ErrorAccumulator("batch_test", max_errors=3) as acc:
            # Successful sub-operation
            with acc.capture_error("operation1"):
                pass

            # Failed sub-operation
            with acc.capture_error("operation2"):
                raise ValueError("Sub-operation failed")

            # Another failed sub-operation
            with acc.capture_error("operation3"):
                raise RuntimeError("Another failure")

        assert acc.has_errors()
        summary = acc.get_error_summary()
        assert summary["total_errors"] == 2
        assert "ValueError" in summary["error_types"]
        assert "RuntimeError" in summary["error_types"]


class TestStructuredLogging:
    """Test structured error logging"""

    def test_structured_error_logger_creation(self):
        """Test structured error logger initialization"""
        logger = StructuredErrorLogger()
        assert logger is not None
        assert logger.error_metrics["total_errors"] == 0

    def test_error_logging_with_context(self):
        """Test error logging with contextual information"""
        logger = StructuredErrorLogger()

        error = ModelConnectionError("Connection failed")
        correlation_id = logger.log_error(
            error=error,
            operation_name="test_operation",
            session_id="test-session",
            additional_context={"attempt": 1},
        )

        assert correlation_id is not None
        assert len(correlation_id) == 8  # UUID prefix
        assert logger.error_metrics["total_errors"] == 1

    def test_error_chain_logging(self):
        """Test logging of exception chains"""
        logger = StructuredErrorLogger()

        try:
            try:
                raise ValueError("Root cause")
            except ValueError as e:
                raise ModelBackendError("Model failed") from e
        except ModelBackendError as chain_error:
            correlation_id = logger.log_error_chain(
                error=chain_error,
                operation_name="test_chain",
                session_id="test-session",
            )

            assert correlation_id is not None
            assert logger.error_metrics["total_errors"] == 1

    def test_performance_error_logging(self):
        """Test performance error logging"""
        logger = StructuredErrorLogger()

        error = PerformanceTimeoutError("Operation too slow")
        correlation_id = logger.log_performance_error(
            error=error,
            operation_name="slow_operation",
            duration=5.5,
            performance_threshold=2.0,
            session_id="test-session",
        )

        assert correlation_id is not None


class TestRealWorldScenarios:
    """Test error handling in realistic scenarios"""

    @pytest.mark.asyncio
    async def test_model_call_with_retries(self):
        """Test model call with automatic retries and proper error handling"""
        attempt_count = 0

        @retry_on_transient_errors(max_retries=2, base_delay=0.01)
        @log_errors_with_context(include_performance=True)
        async def make_model_call(model_name: str):
            nonlocal attempt_count
            attempt_count += 1

            async with error_context(
                "model_api_call", user_context={"model": model_name}
            ) as ctx:
                ctx["attempt"] = attempt_count

                if attempt_count == 1:
                    raise NetworkTimeoutError(
                        "Request timed out",
                        network_context={"timeout": 30, "backend": "bedrock"},
                    )
                elif attempt_count == 2:
                    raise ModelRateLimitError(
                        "Rate limit exceeded", network_context={"retry_after": 1}
                    )
                else:
                    return {"response": "Success!", "tokens": 150}

        result = await make_model_call("claude-v1")
        assert result["response"] == "Success!"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_session_management_errors(self):
        """Test session management with proper error handling"""
        from context_switcher_mcp.session_manager import SessionManager

        # Mock session manager to test error scenarios
        session_manager = SessionManager(max_sessions=1)

        # Test session not found scenario
        session = await session_manager.get_session("nonexistent-session")
        assert session is None

    def test_comprehensive_error_classification(self):
        """Test error classification across all error types"""
        test_errors = [
            (ModelConnectionError("Connection failed"), ErrorSeverity.HIGH, True, True),
            (AuthenticationError("Invalid auth"), ErrorSeverity.CRITICAL, False, False),
            (ParameterValidationError("Bad param"), ErrorSeverity.MEDIUM, False, False),
            (NetworkTimeoutError("Timeout"), ErrorSeverity.MEDIUM, True, True),
            (LockTimeoutError("Lock timeout"), ErrorSeverity.MEDIUM, True, True),
            (PerformanceTimeoutError("Too slow"), ErrorSeverity.HIGH, True, True),
        ]

        for (
            error,
            expected_severity,
            expected_transient,
            expected_retriable,
        ) in test_errors:
            classification = classify_error(error)
            assert classification["severity"] == expected_severity
            assert classification["transient"] == expected_transient
            assert classification["retriable"] == expected_retriable


class TestBackwardCompatibility:
    """Test that existing code still works with new error handling"""

    def test_existing_exception_handling_still_works(self):
        """Test that existing try/except blocks still work"""
        # Old-style generic exception handling should still work
        try:
            raise ModelBackendError("Test error")
        except Exception as e:
            assert isinstance(e, ModelBackendError)
            assert str(e) == "Test error"

    def test_exception_message_compatibility(self):
        """Test that error messages are still accessible"""
        error = ModelValidationError(
            "Invalid model configuration", validation_context={"model": "test"}
        )

        # String representation should still work
        assert str(error) == "Invalid model configuration"

        # Context should be accessible as an attribute
        assert error.validation_context["model"] == "test"


if __name__ == "__main__":
    # Run basic smoke tests
    import sys

    def run_smoke_tests():
        """Run basic smoke tests to verify imports and basic functionality"""
        print("Running error handling smoke tests...")

        try:
            # Test exception creation
            error = ModelValidationError("test", validation_context={"test": "value"})
            assert str(error) == "test"
            print("✓ Exception hierarchy working")

            # Test classification
            classification = classify_error(error)
            assert classification["error_type"] == "ModelValidationError"
            print("✓ Error classification working")

            # Test structured logging
            logger = StructuredErrorLogger()
            correlation_id = logger.log_error(error, "test_operation")
            assert correlation_id is not None
            print("✓ Structured logging working")

            print(
                "\nAll smoke tests passed! Error handling standardization is working correctly."
            )
            return True

        except Exception as e:
            print(f"✗ Smoke test failed: {e}")
            return False

    if not run_smoke_tests():
        sys.exit(1)
