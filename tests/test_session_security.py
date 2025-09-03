"""Comprehensive tests for session security management"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from context_switcher_mcp.session_security import (  # noqa: E402
    ClientBinding,
    SecurityEvent,
    SessionSecurity,
)


class TestClientBinding:
    """Test suite for ClientBinding dataclass"""

    @pytest.fixture
    def sample_client_binding(self):
        """Create a sample ClientBinding for testing"""
        return ClientBinding(
            session_entropy="test-entropy-123",
            creation_timestamp=datetime(2023, 6, 15, 10, 0, 0, tzinfo=UTC),
            binding_signature="test-signature-hash",
            access_pattern_hash="pattern-hash-123",
        )

    def test_client_binding_creation(self, sample_client_binding):
        """Test ClientBinding creation with all fields"""
        assert sample_client_binding.session_entropy == "test-entropy-123"
        assert sample_client_binding.creation_timestamp == datetime(
            2023, 6, 15, 10, 0, 0, tzinfo=UTC
        )
        assert sample_client_binding.binding_signature == "test-signature-hash"
        assert sample_client_binding.access_pattern_hash == "pattern-hash-123"

        # Default values
        assert sample_client_binding.validation_failures == 0
        assert isinstance(sample_client_binding.last_validated, datetime)
        assert sample_client_binding.security_flags == []
        assert sample_client_binding.tool_usage_sequence == []

    def test_generate_binding_signature(self, sample_client_binding):
        """Test HMAC signature generation"""
        secret_key = "test-secret-key"

        # Generate signature
        signature = sample_client_binding.generate_binding_signature(secret_key)

        # Should be a hex string
        assert isinstance(signature, str)
        assert (
            len(signature) == 64
        )  # PBKDF2-HMAC-SHA256 hex output length (32 bytes * 2)

        # Should be deterministic
        signature2 = sample_client_binding.generate_binding_signature(secret_key)
        assert signature == signature2

        # Different key should produce different signature
        different_signature = sample_client_binding.generate_binding_signature(
            "different-key"
        )
        assert signature != different_signature

    def test_validate_binding_success(self, sample_client_binding):
        """Test successful binding validation"""
        secret_key = "test-secret-key"

        # Set the correct signature
        sample_client_binding.binding_signature = (
            sample_client_binding.generate_binding_signature(secret_key)
        )

        # Validation should succeed
        assert sample_client_binding.validate_binding(secret_key) is True

    def test_validate_binding_failure(self, sample_client_binding):
        """Test failed binding validation"""
        secret_key = "test-secret-key"

        # Set an incorrect signature
        sample_client_binding.binding_signature = "wrong-signature"

        # Validation should fail
        assert sample_client_binding.validate_binding(secret_key) is False

    def test_validate_binding_different_key(self, sample_client_binding):
        """Test binding validation with different secret key"""
        original_key = "original-key"
        different_key = "different-key"

        # Set signature with original key
        sample_client_binding.binding_signature = (
            sample_client_binding.generate_binding_signature(original_key)
        )

        # Validation with different key should fail
        assert sample_client_binding.validate_binding(different_key) is False

        # Validation with original key should succeed
        assert sample_client_binding.validate_binding(original_key) is True

    def test_add_security_flag(self, sample_client_binding):
        """Test adding security flags"""
        assert sample_client_binding.security_flags == []

        # Add first flag
        sample_client_binding.add_security_flag("suspicious_activity")
        assert sample_client_binding.security_flags == ["suspicious_activity"]

        # Add another flag
        sample_client_binding.add_security_flag("rate_limit_exceeded")
        assert sample_client_binding.security_flags == [
            "suspicious_activity",
            "rate_limit_exceeded",
        ]

        # Adding duplicate flag shouldn't duplicate
        sample_client_binding.add_security_flag("suspicious_activity")
        assert sample_client_binding.security_flags == [
            "suspicious_activity",
            "rate_limit_exceeded",
        ]

    def test_is_suspicious_validation_failures(self, sample_client_binding):
        """Test suspicious detection based on validation failures"""
        assert not sample_client_binding.is_suspicious()

        # Set high validation failures
        sample_client_binding.validation_failures = 5
        assert sample_client_binding.is_suspicious()

    def test_is_suspicious_security_flags(self, sample_client_binding):
        """Test suspicious detection based on security flags count"""
        assert not sample_client_binding.is_suspicious()

        # Add many security flags
        for i in range(6):
            sample_client_binding.add_security_flag(f"flag_{i}")

        assert sample_client_binding.is_suspicious()

    def test_is_suspicious_specific_flag(self, sample_client_binding):
        """Test suspicious detection based on specific flag"""
        assert not sample_client_binding.is_suspicious()

        # Add the specific flag that makes it suspicious
        sample_client_binding.add_security_flag("multiple_failed_validations")
        assert sample_client_binding.is_suspicious()


class TestSecurityEvent:
    """Test suite for SecurityEvent dataclass"""

    @pytest.fixture
    def sample_security_event(self):
        """Create a sample SecurityEvent for testing"""
        return SecurityEvent(
            event_type="binding_validation_failed",
            timestamp=datetime(2023, 6, 15, 12, 30, 45, tzinfo=UTC),
            details={"validation_failures": 2, "client_id": "test-client"},
        )

    def test_security_event_creation(self, sample_security_event):
        """Test SecurityEvent creation"""
        assert sample_security_event.event_type == "binding_validation_failed"
        assert sample_security_event.timestamp == datetime(
            2023, 6, 15, 12, 30, 45, tzinfo=UTC
        )
        assert sample_security_event.details == {
            "validation_failures": 2,
            "client_id": "test-client",
        }

    def test_security_event_to_dict(self, sample_security_event):
        """Test SecurityEvent serialization to dictionary"""
        result = sample_security_event.to_dict()

        expected = {
            "type": "binding_validation_failed",
            "timestamp": "2023-06-15T12:30:45+00:00",
            "details": {"validation_failures": 2, "client_id": "test-client"},
        }

        assert result == expected


class TestSessionSecurity:
    """Test suite for SessionSecurity class"""

    @pytest.fixture
    def session_security(self):
        """Create a SessionSecurity instance for testing"""
        return SessionSecurity("test-session-123")

    @pytest.fixture
    def session_security_with_binding(self):
        """Create a SessionSecurity instance with client binding"""
        binding = ClientBinding(
            session_entropy="test-entropy",
            creation_timestamp=datetime.now(UTC),
            binding_signature="test-signature",
            access_pattern_hash="test-pattern",
        )
        return SessionSecurity("test-session-123", client_binding=binding)

    def test_session_security_creation(self, session_security):
        """Test SessionSecurity creation"""
        assert session_security.session_id == "test-session-123"
        assert session_security.client_binding is None
        assert session_security.security_events == []

    def test_session_security_creation_with_binding(
        self, session_security_with_binding
    ):
        """Test SessionSecurity creation with existing binding"""
        assert session_security_with_binding.session_id == "test-session-123"
        assert session_security_with_binding.client_binding is not None
        assert (
            session_security_with_binding.client_binding.session_entropy
            == "test-entropy"
        )

    @patch("context_switcher_mcp.session_security.secrets.token_urlsafe")
    @patch("context_switcher_mcp.session_security.datetime")
    def test_create_client_binding(self, mock_datetime, mock_token, session_security):
        """Test client binding creation"""
        # Mock dependencies
        mock_token.return_value = "mock-entropy-token"
        mock_time = datetime(2023, 6, 15, 14, 30, 0, tzinfo=UTC)
        mock_datetime.now.return_value = mock_time
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        secret_key = "test-secret"
        access_pattern_hash = "custom-pattern-hash"

        binding = session_security.create_client_binding(
            secret_key, access_pattern_hash
        )

        assert binding.session_entropy == "mock-entropy-token"
        assert binding.creation_timestamp == mock_time
        assert binding.access_pattern_hash == "custom-pattern-hash"
        assert binding.binding_signature != ""  # Should have generated signature

        # Should be stored in session security
        assert session_security.client_binding == binding

        # Verify signature is valid
        assert binding.validate_binding(secret_key) is True

    def test_create_client_binding_default_pattern(self, session_security):
        """Test client binding creation with default access pattern"""
        secret_key = "test-secret"

        binding = session_security.create_client_binding(secret_key)

        # Should have generated a default pattern hash
        assert binding.access_pattern_hash is not None
        assert len(binding.access_pattern_hash) == 64  # SHA256 hex length

    def test_validate_binding_no_binding(self, session_security):
        """Test binding validation when no binding exists (backward compatibility)"""
        result = session_security.validate_binding("any-key")
        assert result is True  # Should allow for backward compatibility

    def test_validate_binding_success(self, session_security):
        """Test successful binding validation"""
        secret_key = "test-secret"
        binding = session_security.create_client_binding(secret_key)

        # Validation should succeed
        result = session_security.validate_binding(secret_key)
        assert result is True

        # last_validated should be updated
        assert binding.last_validated > binding.creation_timestamp

    def test_validate_binding_failure(self, session_security):
        """Test failed binding validation"""
        secret_key = "test-secret"
        wrong_key = "wrong-secret"

        binding = session_security.create_client_binding(secret_key)
        initial_failures = binding.validation_failures

        # Validation should fail
        result = session_security.validate_binding(wrong_key)
        assert result is False

        # Should increment failure count
        assert binding.validation_failures == initial_failures + 1

        # Should record security event
        assert len(session_security.security_events) > 0
        latest_event = session_security.security_events[-1]
        assert latest_event.event_type == "binding_validation_failed"

    def test_record_security_event(self, session_security_with_binding):
        """Test recording security events"""
        initial_count = len(session_security_with_binding.security_events)

        session_security_with_binding.record_security_event(
            "suspicious_activity",
            {"description": "Multiple rapid requests", "count": 10},
        )

        # Should add event to list
        assert len(session_security_with_binding.security_events) == initial_count + 1

        latest_event = session_security_with_binding.security_events[-1]
        assert latest_event.event_type == "suspicious_activity"
        assert latest_event.details["description"] == "Multiple rapid requests"
        assert isinstance(latest_event.timestamp, datetime)

        # Should add security flag to client binding
        assert (
            "suspicious_activity"
            in session_security_with_binding.client_binding.security_flags
        )

    def test_record_security_event_no_binding(self, session_security):
        """Test recording security events without client binding"""
        session_security.record_security_event("test_event", {"data": "test"})

        # Should still record the event
        assert len(session_security.security_events) == 1
        assert session_security.security_events[0].event_type == "test_event"

    def test_update_tool_usage_pattern(self, session_security_with_binding):
        """Test updating tool usage pattern"""
        binding = session_security_with_binding.client_binding
        initial_count = len(binding.tool_usage_sequence)

        session_security_with_binding.update_tool_usage_pattern(
            "start_context_analysis"
        )

        assert len(binding.tool_usage_sequence) == initial_count + 1
        assert binding.tool_usage_sequence[-1] == "start_context_analysis"

    def test_update_tool_usage_pattern_limit(self, session_security_with_binding):
        """Test tool usage pattern limit (max 10 tools)"""
        binding = session_security_with_binding.client_binding

        # Fill up to limit
        for i in range(10):
            session_security_with_binding.update_tool_usage_pattern(f"tool_{i}")

        assert len(binding.tool_usage_sequence) == 10

        # Adding more should not increase the count
        session_security_with_binding.update_tool_usage_pattern("tool_11")
        assert len(binding.tool_usage_sequence) == 10

    def test_update_tool_usage_pattern_no_binding(self, session_security):
        """Test updating tool usage pattern without client binding"""
        # Should not raise an error
        session_security.update_tool_usage_pattern("some_tool")
        # No assertion needed - just shouldn't crash

    def test_is_session_suspicious_binding(self, session_security):
        """Test suspicious session detection based on client binding"""
        # Create suspicious binding
        binding = session_security.create_client_binding("test-key")
        binding.validation_failures = 5  # Make it suspicious

        assert session_security.is_session_suspicious() is True

    def test_is_session_suspicious_events(self, session_security):
        """Test suspicious session detection based on security events"""
        # Add many recent security events
        for i in range(12):  # More than threshold of 10
            session_security.record_security_event(f"event_{i}", {"data": i})

        assert session_security.is_session_suspicious() is True

    def test_is_session_suspicious_old_events(self, session_security):
        """Test that old events don't contribute to suspicious detection"""
        with patch("context_switcher_mcp.session_security.datetime") as mock_datetime:
            # Create old events
            old_time = datetime.now(UTC) - timedelta(hours=2)
            mock_datetime.now.return_value = old_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            for i in range(12):
                session_security.record_security_event(f"old_event_{i}", {"data": i})

            # Reset to current time
            current_time = datetime.now(UTC)
            mock_datetime.now.return_value = current_time

            # Should not be suspicious because events are old
            assert session_security.is_session_suspicious() is False

    def test_get_security_summary_with_binding(self, session_security_with_binding):
        """Test security summary with client binding"""
        # Add some security events
        session_security_with_binding.record_security_event(
            "test_event", {"data": "test"}
        )
        session_security_with_binding.update_tool_usage_pattern("test_tool")

        summary = session_security_with_binding.get_security_summary()

        assert summary["session_id"] == "test-session-123"
        assert summary["client_binding"] is not None
        assert "validation_failures" in summary["client_binding"]
        assert "last_validated" in summary["client_binding"]
        assert "is_suspicious" in summary["client_binding"]
        assert summary["client_binding"]["tool_usage_count"] == 1
        assert summary["security_events_count"] == 1
        assert len(summary["recent_events"]) == 1

    def test_get_security_summary_no_binding(self, session_security):
        """Test security summary without client binding"""
        session_security.record_security_event("test_event", {"data": "test"})

        summary = session_security.get_security_summary()

        assert summary["session_id"] == "test-session-123"
        assert summary["client_binding"] is None
        assert summary["security_events_count"] == 1
        assert len(summary["recent_events"]) == 1

    def test_get_security_summary_recent_events_limit(self, session_security):
        """Test that security summary limits recent events to 5"""
        # Add more than 5 events
        for i in range(10):
            session_security.record_security_event(f"event_{i}", {"index": i})

        summary = session_security.get_security_summary()

        assert summary["security_events_count"] == 10
        assert len(summary["recent_events"]) == 5  # Limited to 5

        # Should be the most recent events
        for i, event in enumerate(summary["recent_events"]):
            expected_index = 5 + i  # Events 5-9 (most recent 5)
            assert event["details"]["index"] == expected_index

    def test_generate_default_pattern_hash(self, session_security):
        """Test default access pattern hash generation"""
        hash1 = session_security._generate_default_pattern_hash()

        # Should be a valid SHA256 hex string
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Should be deterministic for same session and time
        with patch("context_switcher_mcp.session_security.datetime") as mock_datetime:
            mock_time = datetime(2023, 6, 15, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = mock_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            hash2 = session_security._generate_default_pattern_hash()
            hash3 = session_security._generate_default_pattern_hash()

            assert hash2 == hash3

    def test_cleanup_old_events(self, session_security):
        """Test cleanup of old security events"""
        current_time = datetime.now(UTC)

        with patch("context_switcher_mcp.session_security.datetime") as mock_datetime:
            # Create some old events
            old_time = current_time - timedelta(hours=25)  # Older than default 24 hours
            mock_datetime.now.return_value = old_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            session_security.record_security_event("old_event_1", {"data": "old1"})
            session_security.record_security_event("old_event_2", {"data": "old2"})

            # Create some recent events
            recent_time = current_time - timedelta(hours=1)  # Within 24 hours
            mock_datetime.now.return_value = recent_time

            session_security.record_security_event(
                "recent_event_1", {"data": "recent1"}
            )
            session_security.record_security_event(
                "recent_event_2", {"data": "recent2"}
            )

            # Reset to current time
            mock_datetime.now.return_value = current_time

            # Should have 4 events total
            assert len(session_security.security_events) == 4

            # Cleanup old events
            session_security.cleanup_old_events()

            # Should only have the 2 recent events
            assert len(session_security.security_events) == 2
            for event in session_security.security_events:
                assert "recent" in event.event_type

    def test_cleanup_old_events_custom_age(self, session_security):
        """Test cleanup with custom max age"""
        current_time = datetime.now(UTC)

        with patch("context_switcher_mcp.session_security.datetime") as mock_datetime:
            # Create events at different times
            times_and_labels = [
                (current_time - timedelta(hours=13), "very_old"),  # Older than 12 hours
                (current_time - timedelta(hours=6), "recent"),  # Within 12 hours
                (current_time - timedelta(minutes=30), "very_recent"),  # Very recent
            ]

            for time, label in times_and_labels:
                mock_datetime.now.return_value = time
                session_security.record_security_event(
                    f"event_{label}", {"label": label}
                )

            mock_datetime.now.return_value = current_time

            assert len(session_security.security_events) == 3

            # Cleanup with 12 hour limit
            session_security.cleanup_old_events(max_age_hours=12)

            # Should only have events within 12 hours
            assert len(session_security.security_events) == 2
            remaining_labels = [
                event.details["label"] for event in session_security.security_events
            ]
            assert "very_old" not in remaining_labels
            assert "recent" in remaining_labels
            assert "very_recent" in remaining_labels


class TestSessionSecurityIntegration:
    """Integration tests for SessionSecurity"""

    def test_complete_session_lifecycle(self):
        """Test a complete session security lifecycle"""
        session_security = SessionSecurity("integration-test-session")
        secret_key = "integration-test-key"

        # 1. Create client binding
        binding = session_security.create_client_binding(secret_key, "initial-pattern")
        assert binding is not None
        assert session_security.validate_binding(secret_key) is True

        # 2. Record some tool usage
        session_security.update_tool_usage_pattern("start_context_analysis")
        session_security.update_tool_usage_pattern("add_perspective")
        session_security.update_tool_usage_pattern("analyze_from_perspectives")

        # 3. Record some security events
        session_security.record_security_event("rate_limit_warning", {"requests": 100})
        session_security.record_security_event(
            "unusual_activity", {"pattern": "rapid_requests"}
        )

        # 4. Validate binding again
        assert session_security.validate_binding(secret_key) is True

        # 5. Simulate some failed validations
        session_security.validate_binding("wrong-key")
        session_security.validate_binding("another-wrong-key")

        # 6. Check final state
        summary = session_security.get_security_summary()

        assert summary["client_binding"]["validation_failures"] == 2
        assert summary["client_binding"]["tool_usage_count"] == 3
        assert (
            summary["security_events_count"] >= 4
        )  # At least our events + validation failures
        assert len(summary["recent_events"]) >= 2

        # 7. Clean up old events
        session_security.cleanup_old_events()

        # Events should still be there since they're recent
        assert len(session_security.security_events) >= 2


class TestSessionSecurityErrorHandling:
    """Test error handling and edge cases"""

    def test_binding_signature_generation_with_special_characters(self):
        """Test signature generation with special characters in session data"""
        binding = ClientBinding(
            session_entropy="entropy-with-special-chars!@#$%^&*()",
            creation_timestamp=datetime.now(UTC),
            binding_signature="",
            access_pattern_hash="pattern-hash",
        )

        secret_key = "secret-with-special-chars!@#$%"

        # Should not raise an error
        signature = binding.generate_binding_signature(secret_key)
        assert isinstance(signature, str)
        assert len(signature) == 64

        # Validation should work
        binding.binding_signature = signature
        assert binding.validate_binding(secret_key) is True

    def test_security_event_with_none_details(self):
        """Test security event handling with None details"""
        session_security = SessionSecurity("test-session")

        # Should not raise an error
        session_security.record_security_event("test_event", None)

        assert len(session_security.security_events) == 1
        assert session_security.security_events[0].details is None
