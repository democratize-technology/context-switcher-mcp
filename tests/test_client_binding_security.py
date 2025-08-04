"""
Test suite for client binding security features
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
import secrets  # noqa: E402
from datetime import datetime, timedelta  # noqa: E402

from context_switcher_mcp.models import ClientBinding, ContextSwitcherSession  # noqa: E402
from context_switcher_mcp.client_binding import (  # noqa: E402
    ClientBindingManager,
    create_secure_session_with_binding,
    validate_session_access,
    log_security_event_with_binding,
)


class TestClientBinding:
    """Test ClientBinding model security features"""

    def test_client_binding_creation(self):
        """Test secure client binding creation"""
        now = datetime.utcnow()
        binding = ClientBinding(
            session_entropy=secrets.token_urlsafe(32),
            creation_timestamp=now,
            binding_signature="",
            access_pattern_hash="test_hash",
            tool_usage_sequence=["start_context_analysis"],
        )

        # Generate signature
        secret_key = "test_secret_key"
        binding.binding_signature = binding.generate_binding_signature(secret_key)

        assert binding.session_entropy
        assert binding.creation_timestamp == now
        assert binding.binding_signature
        assert len(binding.binding_signature) == 64  # SHA256 hash length in hex

    def test_binding_signature_validation(self):
        """Test binding signature validation"""
        secret_key = "test_secret_key"
        binding = ClientBinding(
            session_entropy="test_entropy",
            creation_timestamp=datetime.utcnow(),
            binding_signature="",
            access_pattern_hash="test_hash",
        )

        # Generate valid signature
        binding.binding_signature = binding.generate_binding_signature(secret_key)

        # Valid signature should pass
        assert binding.validate_binding(secret_key) is True

        # Invalid signature should fail
        binding.binding_signature = "invalid_signature"
        assert binding.validate_binding(secret_key) is False

        # Wrong secret key should fail
        assert binding.validate_binding("wrong_key") is False

    def test_suspicious_behavior_detection(self):
        """Test suspicious behavior detection"""
        binding = ClientBinding(
            session_entropy="test",
            creation_timestamp=datetime.utcnow(),
            binding_signature="test",
            access_pattern_hash="test",
        )

        # Clean binding should not be suspicious
        assert binding.is_suspicious() is False

        # Too many validation failures
        binding.validation_failures = 5
        assert binding.is_suspicious() is True

        # Reset and test security flags
        binding.validation_failures = 0
        binding.security_flags = ["flag1", "flag2", "flag3", "flag4", "flag5", "flag6"]
        assert binding.is_suspicious() is True

        # Reset and test specific suspicious flag
        binding.security_flags = ["multiple_failed_validations"]
        assert binding.is_suspicious() is True


class TestClientBindingManager:
    """Test ClientBindingManager functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.manager = ClientBindingManager("test_secret_key")

    def test_create_client_binding(self):
        """Test client binding creation"""
        session_id = "test_session_123"
        binding = self.manager.create_client_binding(
            session_id, "start_context_analysis"
        )

        assert binding.session_entropy
        assert binding.creation_timestamp
        assert binding.binding_signature
        assert binding.access_pattern_hash
        assert binding.tool_usage_sequence == ["start_context_analysis"]
        assert binding.validate_binding(self.manager.secret_key) is True

    @pytest.mark.asyncio
    async def test_session_binding_validation_success(self):
        """Test successful session binding validation"""
        # Create session with binding
        session_id = "test_session_123"
        session = create_secure_session_with_binding(session_id, "Test topic")

        # Update binding manager's secret key to match
        session.client_binding.binding_signature = (
            session.client_binding.generate_binding_signature(self.manager.secret_key)
        )

        # Validation should succeed
        is_valid, error = await self.manager.validate_session_binding(
            session, "test_tool"
        )
        assert is_valid is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_session_binding_validation_failure(self):
        """Test failed session binding validation"""
        # Create session with binding
        session_id = "test_session_123"
        session = create_secure_session_with_binding(session_id, "Test topic")

        # Corrupt the binding signature
        session.client_binding.binding_signature = "invalid_signature"

        # Validation should fail
        is_valid, error = await self.manager.validate_session_binding(
            session, "test_tool"
        )
        assert is_valid is False
        assert "validation failed" in error.lower()
        assert session.client_binding.validation_failures > 0

    @pytest.mark.asyncio
    async def test_legacy_session_handling(self):
        """Test handling of legacy sessions without client binding"""
        # Create legacy session without binding
        session = ContextSwitcherSession(
            session_id="legacy_session",
            created_at=datetime.utcnow(),
            topic="Legacy topic",
        )

        # Validation should succeed with warning
        is_valid, error = await self.manager.validate_session_binding(
            session, "test_tool"
        )
        assert is_valid is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_suspicious_access_detection(self):
        """Test suspicious access pattern detection"""
        # Create session with binding
        session_id = "test_session_123"
        session = create_secure_session_with_binding(session_id, "Test topic")
        session.client_binding.binding_signature = (
            session.client_binding.generate_binding_signature(self.manager.secret_key)
        )

        # Simulate rapid access
        session.access_count = 200
        session.created_at = datetime.utcnow() - timedelta(hours=1)

        # Should be flagged as suspicious
        is_valid, error = await self.manager.validate_session_binding(
            session, "test_tool"
        )
        assert is_valid is False
        assert "suspicious" in error.lower()

    def test_security_metrics(self):
        """Test security metrics collection"""
        metrics = self.manager.get_security_metrics()

        assert "suspicious_sessions_count" in metrics
        assert "binding_secret_key_set" in metrics
        assert "max_validation_failures" in metrics
        assert "max_security_flags" in metrics
        assert "suspicious_access_threshold" in metrics

        assert metrics["binding_secret_key_set"] is True


class TestSecureSessionCreation:
    """Test secure session creation with client binding"""

    def test_create_secure_session_with_binding(self):
        """Test creating a session with client binding"""
        session_id = "test_session_123"
        topic = "Test analysis topic"

        session = create_secure_session_with_binding(
            session_id, topic, "start_context_analysis"
        )

        assert session.session_id == session_id
        assert session.topic == topic
        assert session.client_binding is not None
        assert session.client_binding.tool_usage_sequence == ["start_context_analysis"]
        assert session.access_count == 1  # Initial access recorded

    @pytest.mark.asyncio
    async def test_session_access_validation(self):
        """Test session access validation"""
        session = create_secure_session_with_binding("test_session", "Test topic")

        # Valid access should succeed
        is_valid, error = await validate_session_access(session, "test_tool")
        assert is_valid is True
        assert error == ""

        # Access count should increment
        initial_count = session.access_count
        await validate_session_access(session, "another_tool")
        assert session.access_count > initial_count


class TestSecurityLogging:
    """Test security event logging with binding context"""

    def test_security_event_logging_with_binding(self):
        """Test security event logging with client binding context"""
        session = create_secure_session_with_binding("test_session", "Test topic")

        # Log security event
        log_security_event_with_binding(
            "test_security_event",
            "test_session",
            {"test_detail": "test_value"},
            session,
        )

        # Event should be recorded in session
        assert len(session.security_events) > 0
        event = session.security_events[-1]
        assert event["type"] == "test_security_event"
        assert event["details"]["test_detail"] == "test_value"
        assert event["details"]["has_client_binding"] is True

    def test_security_event_logging_without_binding(self):
        """Test security event logging without client binding"""
        session = ContextSwitcherSession(
            session_id="legacy_session",
            created_at=datetime.utcnow(),
            topic="Legacy topic",
        )

        # Log security event
        log_security_event_with_binding(
            "test_security_event",
            "legacy_session",
            {"test_detail": "test_value"},
            session,
        )

        # Event should be recorded with no binding context
        assert len(session.security_events) > 0
        event = session.security_events[-1]
        assert event["details"]["has_client_binding"] is False


class TestSecurityIntegration:
    """Integration tests for security features"""

    @pytest.mark.asyncio
    async def test_end_to_end_security_validation(self):
        """Test complete security validation workflow"""
        # Create session with binding
        session = create_secure_session_with_binding("test_session", "Test topic")

        # Validate access multiple times
        for i in range(5):
            is_valid, error = await validate_session_access(session, f"tool_{i}")
            assert is_valid is True

        # Check that binding is still valid
        assert session.client_binding.validation_failures == 0
        assert not session.client_binding.is_suspicious()

        # Corrupt binding and try again
        session.client_binding.binding_signature = "corrupted"
        is_valid, error = await validate_session_access(session, "tool_fail")
        assert is_valid is False
        assert session.client_binding.validation_failures > 0

    @pytest.mark.asyncio
    async def test_session_security_lifecycle(self):
        """Test session security over its lifecycle"""
        session = create_secure_session_with_binding("test_session", "Test topic")

        # Initial state should be secure
        assert session.client_binding.validation_failures == 0
        assert len(session.security_events) == 0

        # Record normal access using async version
        await session.record_access("normal_tool")
        assert session.access_count > 0

        # Record security event
        session.record_security_event("test_event", {"detail": "value"})
        assert len(session.security_events) == 1
        assert "test_event" in session.client_binding.security_flags

        # Validate that session maintains binding integrity with global manager's key
        from context_switcher_mcp.client_binding import client_binding_manager

        assert session.is_binding_valid(client_binding_manager.secret_key)


@pytest.mark.asyncio
class TestAsyncSecurityIntegration:
    """Async integration tests for security features"""

    async def test_async_session_validation_workflow(self):
        """Test async validation workflow"""
        # This would integrate with the actual MCP tool validation
        # For now, we test the basic async pattern
        session = create_secure_session_with_binding("async_session", "Async topic")

        # Simulate async validation calls
        results = []
        for tool in ["tool1", "tool2", "tool3"]:
            is_valid, error = await validate_session_access(session, tool)
            results.append((is_valid, error))

        # All validations should succeed
        assert all(result[0] for result in results)
        assert all(error == "" for _, error in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
