"""
Comprehensive tests for security context sanitization
Tests ensure sensitive context data is properly sanitized before logging
"""

from unittest.mock import patch

import pytest
from context_switcher_mcp.error_logging import StructuredErrorLogger
from context_switcher_mcp.exceptions import (
    ModelAuthenticationError,
    NetworkError,
    PerformanceError,
    SecurityError,
)
from context_switcher_mcp.security_context_sanitizer import (
    SecurityContextSanitizer,
    get_context_sanitizer,
    sanitize_context_dict,
    sanitize_exception_context,
)


class TestSecurityContextSanitizer:
    """Test the security context sanitizer"""

    def setup_method(self):
        """Set up test environment"""
        self.sanitizer = SecurityContextSanitizer()

    def test_sensitive_api_key_sanitization(self):
        """Test that API keys are properly sanitized"""
        context = {
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "openai_key": "sk-proj-abcd1234567890abcdef1234567890abcdef",
            "aws_access_key": "AKIAI234567890EXAMPLE",
            "regular_value": "safe_value",
        }

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # API keys should be hashed
        assert "api_key_hash" in sanitized
        assert "api_key_present" in sanitized
        assert sanitized["api_key_present"] is True
        assert "sk-" not in str(sanitized)
        assert "AKIA" not in str(sanitized)

        # Regular values should be preserved
        assert sanitized["regular_value"] == "safe_value"

    def test_session_id_sanitization(self):
        """Test that session IDs are properly hashed"""
        context = {
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "user_session": "abc123def456",
            "operation_name": "test_operation",  # Safe value
        }

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # Session IDs should be hashed
        assert "session_id_hash" in sanitized
        assert "550e8400" not in str(sanitized)

        # Safe values should be preserved
        assert sanitized["operation_name"] == "test_operation"

    def test_jwt_token_sanitization(self):
        """Test that JWT tokens are sanitized"""
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

        sanitized_value = self.sanitizer._sanitize_value(jwt_token)

        assert "***JWT_TOKEN***" in str(sanitized_value)
        assert "eyJ" not in str(sanitized_value)

    def test_url_credential_sanitization(self):
        """Test that URLs with credentials are sanitized"""
        context = {
            "database_url": "postgresql://user:password@localhost:5432/db",
            "api_endpoint": "https://api.example.com/v1",
            "redis_url": "redis://user:secret@redis.example.com:6379",
        }

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # Credentials in URLs should be masked
        db_url = sanitized.get("database_url", "")
        assert "://***:***@" in db_url
        assert "password" not in db_url

        redis_url = sanitized.get("redis_url", "")
        assert "://***:***@" in redis_url
        assert "secret" not in redis_url

    def test_file_path_sanitization(self):
        """Test that file paths are sanitized"""
        context = {
            "home_path": "/home/testuser/secrets/config.json",
            "user_path": "/Users/testuser/private/keys.pem",
            "windows_path": "C:\\Users\\testuser\\Documents\\secrets.txt",
        }

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # User paths should be masked
        for _key, value in sanitized.items():
            if isinstance(value, str):
                assert "testuser" not in value
                assert (
                    "/home/***" in value
                    or "/Users/***" in value
                    or "C:\\Users\\***" in value
                )

    def test_security_context_specialized_sanitization(self):
        """Test specialized sanitization for security context"""
        context = {
            "auth_type": "api_key",
            "user_id": "user123",
            "session_id": "session456",
            "api_key": "sk-secret123",
            "region": "us-east-1",
            "service": "bedrock",
        }

        sanitized = self.sanitizer._sanitize_security_context(context)

        # Safe values preserved
        assert sanitized["auth_type"] == "api_key"
        assert sanitized["region"] == "us-east-1"
        assert sanitized["service"] == "bedrock"

        # Sensitive values hashed
        assert "user_id_hash" in sanitized
        assert "session_id_hash" in sanitized
        assert "api_key_hash" in sanitized

        # Original sensitive values not present
        assert "user123" not in str(sanitized)
        assert "session456" not in str(sanitized)
        assert "sk-secret123" not in str(sanitized)

    def test_network_context_sanitization(self):
        """Test network context sanitization"""
        context = {
            "host": "api.sensitive-service.com",
            "port": 443,
            "ip": "192.168.1.100",
            "url": "https://user:pass@api.example.com/endpoint",
            "method": "POST",
            "status_code": 401,
        }

        sanitized = self.sanitizer._sanitize_network_context(context)

        # Safe values preserved
        assert sanitized["port"] == 443
        assert sanitized["method"] == "POST"
        assert sanitized["status_code"] == 401

        # IP addresses hashed
        assert "ip_hash" in sanitized
        assert "192.168.1.100" not in str(sanitized)

        # URLs sanitized
        assert "://***:***@" in sanitized.get("url", "")

    def test_performance_context_sanitization(self):
        """Test performance context sanitization"""
        context = {
            "duration": 1.23,
            "timeout": 30.0,
            "session_id": "sess_12345",
            "memory_usage": 1024,
            "operation_name": "model_call",
        }

        sanitized = self.sanitizer._sanitize_performance_context(context)

        # Safe performance metrics preserved
        assert sanitized["duration"] == 1.23
        assert sanitized["timeout"] == 30.0
        assert sanitized["memory_usage"] == 1024
        assert sanitized["operation_name"] == "model_call"

        # Identifiers hashed
        assert "session_id_hash" in sanitized
        assert "sess_12345" not in str(sanitized)

    def test_validation_context_sanitization(self):
        """Test validation context sanitization"""
        context = {
            "field_name": "password",
            "field_value": "user_secret_password",
            "validation_type": "length",
            "error_count": 1,
        }

        sanitized = self.sanitizer._sanitize_validation_context(context)

        # Safe values preserved
        assert sanitized["field_name"] == "password"
        assert sanitized["validation_type"] == "length"
        assert sanitized["error_count"] == 1

        # Sensitive input data sanitized
        assert "field_value_sanitized" in sanitized
        assert "field_value_length" in sanitized
        assert "user_secret_password" not in str(sanitized)

    def test_exception_context_sanitization(self):
        """Test sanitization of exception contexts"""
        # Create exception with sensitive context
        security_error = SecurityError(
            "Authentication failed",
            security_context={
                "api_key": "sk-secret123",
                "user_id": "user456",
                "auth_method": "bearer",
            },
        )

        sanitized = self.sanitizer.sanitize_exception_context(security_error)

        assert sanitized["exception_type"] == "SecurityError"
        assert "security_context" in sanitized

        # Check that security context was properly sanitized
        sec_ctx = sanitized["security_context"]
        assert sec_ctx["auth_method"] == "bearer"  # Safe value
        assert "api_key_hash" in sec_ctx
        assert "user_id_hash" in sec_ctx
        assert "sk-secret123" not in str(sanitized)
        assert "user456" not in str(sanitized)

    def test_nested_context_sanitization(self):
        """Test sanitization of nested context structures"""
        context = {
            "operation": "auth",
            "details": {
                "credentials": {
                    "username": "admin",
                    "password": "secret123",
                    "api_key": "sk-abcd1234",
                },
                "metadata": {
                    "session_id": "sess_789",
                    "timestamp": "2023-01-01T00:00:00Z",
                },
            },
        }

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # Should handle nested structures
        assert "operation" in sanitized
        assert "details" in sanitized

        # Sensitive data should be sanitized at all levels
        assert "secret123" not in str(sanitized)
        assert "sk-abcd1234" not in str(sanitized)
        assert "sess_789" not in str(sanitized)

    def test_safe_keys_preserved(self):
        """Test that safe keys are preserved without sanitization"""
        context = {
            "operation_name": "test_op",
            "operation_id": "op_123",
            "timestamp": 1234567890,
            "duration": 1.5,
            "thread_count": 4,
            "status": "success",
        }

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # All safe keys should be preserved exactly
        for key, value in context.items():
            assert sanitized[key] == value

    def test_large_context_truncation(self):
        """Test that large context values are truncated"""
        large_value = "x" * 1000  # Create large string
        context = {"large_data": large_value}

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # Should be truncated
        sanitized_value = sanitized["large_data"]
        assert len(sanitized_value) <= 500
        assert sanitized_value.endswith("...")

    def test_empty_and_none_values(self):
        """Test handling of empty and None values"""
        context = {
            "empty_string": "",
            "none_value": None,
            "empty_dict": {},
            "empty_list": [],
        }

        sanitized = self.sanitizer.sanitize_context_dict(context)

        assert sanitized["empty_string"] == ""
        assert sanitized["none_value"] is None
        assert sanitized["empty_dict"] == {}
        assert sanitized["empty_list"] == []

    def test_list_truncation(self):
        """Test that long lists are truncated"""
        long_list = [f"item_{i}" for i in range(20)]
        context = {"items": long_list}

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # Should be truncated to 10 items
        assert len(sanitized["items"]) == 10

    def test_hash_consistency(self):
        """Test that hashing is consistent"""
        data = "sensitive_data_123"

        hash1 = self.sanitizer._hash_sensitive_data(data)
        hash2 = self.sanitizer._hash_sensitive_data(data)

        assert hash1 == hash2
        assert len(hash1) == 8  # Truncated to 8 chars
        assert hash1 != data


class TestSecurityErrorLoggingIntegration:
    """Test integration with error logging systems"""

    def setup_method(self):
        """Set up test environment"""
        self.logger = StructuredErrorLogger()

    @patch("context_switcher_mcp.error_logging.logger")
    def test_security_error_logging_sanitization(self, mock_logger):
        """Test that security errors are properly sanitized in logs"""
        # Create security error with sensitive context
        error = ModelAuthenticationError(
            "Invalid API key",
            security_context={
                "api_key": "sk-secret123456789",
                "user_id": "user_789",
                "region": "us-west-2",
            },
        )

        # Log the error
        self.logger.log_error(
            error=error, operation_name="model_auth", session_id="session_456"
        )

        # Verify logger was called
        assert mock_logger.log.called

        # Get the logged data
        call_args = mock_logger.log.call_args
        logged_extra = call_args[1].get("extra", {})
        structured_error = logged_extra.get("structured_error", {})

        # Verify sensitive context was sanitized
        assert "sanitized_context" in structured_error
        sanitized_ctx = structured_error["sanitized_context"]

        # Original sensitive data should not be in the log
        log_str = str(structured_error)
        assert "sk-secret123456789" not in log_str
        assert "user_789" not in log_str

        # But safe data should be present
        if "security_context" in sanitized_ctx:
            assert sanitized_ctx["security_context"].get("region") == "us-west-2"

    @patch("context_switcher_mcp.error_logging.logger")
    def test_network_error_logging_sanitization(self, mock_logger):
        """Test network error context sanitization in logs"""
        error = NetworkError(
            "Connection failed",
            network_context={
                "host": "secret-api.internal.com",
                "port": 443,
                "url": "https://user:pass@secret-api.internal.com/auth",
                "headers": {"Authorization": "Bearer secret_token"},
            },
        )

        self.logger.log_error(error=error, operation_name="network_call")

        # Verify sanitization occurred
        assert mock_logger.log.called
        call_args = mock_logger.log.call_args
        structured_error = call_args[1]["extra"]["structured_error"]

        log_str = str(structured_error)
        assert "secret_token" not in log_str
        assert "user:pass" not in log_str

    def test_performance_error_context_sanitization(self):
        """Test performance error context sanitization"""
        error = PerformanceError(
            "Operation too slow",
            performance_context={
                "session_id": "perf_session_123",
                "duration": 5.2,
                "threshold": 2.0,
                "operation_details": "Detailed operation info",
            },
        )

        sanitized = sanitize_exception_context(error)

        # Performance metrics should be preserved
        perf_ctx = sanitized.get("performance_context", {})
        assert perf_ctx.get("duration") == 5.2
        assert perf_ctx.get("threshold") == 2.0

        # Session ID should be hashed
        assert "session_id_hash" in perf_ctx
        assert "perf_session_123" not in str(sanitized)


class TestGlobalSanitizerFunctions:
    """Test global convenience functions"""

    def test_get_context_sanitizer(self):
        """Test global sanitizer instance"""
        sanitizer1 = get_context_sanitizer()
        sanitizer2 = get_context_sanitizer()

        # Should return same instance
        assert sanitizer1 is sanitizer2
        assert isinstance(sanitizer1, SecurityContextSanitizer)

    def test_sanitize_exception_context_function(self):
        """Test convenience function for exception sanitization"""
        error = SecurityError(
            "Test error", security_context={"sensitive_key": "sensitive_value"}
        )

        sanitized = sanitize_exception_context(error)

        assert sanitized["exception_type"] == "SecurityError"
        assert "security_context" in sanitized
        assert "sensitive_value" not in str(sanitized)

    def test_sanitize_context_dict_function(self):
        """Test convenience function for context dict sanitization"""
        context = {"api_key": "sk-test123", "operation": "test"}

        sanitized = sanitize_context_dict(context)

        assert "sk-test123" not in str(sanitized)
        assert sanitized.get("operation") == "test"


class TestSecurityContextSanitizationEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        self.sanitizer = SecurityContextSanitizer()

    def test_non_dict_context(self):
        """Test handling of non-dictionary context"""
        result = self.sanitizer.sanitize_context_dict("not_a_dict")

        assert result["sanitized"] is True
        assert result["original_type"] == "str"

    def test_circular_reference_handling(self):
        """Test handling of circular references in context"""
        context = {"key": "value"}
        context["self_ref"] = context

        # Should not crash on circular reference
        sanitized = self.sanitizer.sanitize_context_dict(context)
        assert "key" in sanitized

    def test_very_deep_nesting(self):
        """Test handling of deeply nested structures"""
        context = {"level1": {"level2": {"level3": {"api_key": "sk-deep123"}}}}

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # Should handle deep nesting and sanitize at all levels
        assert "sk-deep123" not in str(sanitized)

    def test_mixed_type_lists(self):
        """Test lists with mixed types"""
        context = {
            "mixed_list": [
                "string",
                42,
                {"nested_key": "sk-list123"},
                ["nested", "list"],
            ]
        }

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # Should handle mixed types and sanitize appropriately
        assert "sk-list123" not in str(sanitized)

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters"""
        context = {
            "unicode_key": "测试密钥_sk-unicode123",
            "special_chars": "key!@#$%^&*()_+{}[]|\\:;\"'<>?,./",
        }

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # Should handle unicode without crashing
        assert "sk-unicode123" not in str(sanitized)

    def test_extremely_long_keys(self):
        """Test handling of extremely long key names"""
        long_key = "a" * 1000
        context = {long_key: "value"}

        sanitized = self.sanitizer.sanitize_context_dict(context)

        # Should handle long keys without issues
        assert len(list(sanitized.keys())[0]) <= 500  # Should be truncated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
