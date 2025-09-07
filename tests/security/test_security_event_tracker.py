"""
Test suite for SecurityEventTracker security module.
"""

import logging
import os
import sys  # noqa: E402
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))  # noqa: E402

import pytest  # noqa: E402
from context_switcher_mcp.models import (  # noqa: E402 # noqa: E402
    ClientBinding,
    ContextSwitcherSession,
)
from context_switcher_mcp.security.security_event_tracker import (  # noqa: E402 # noqa: E402
    SecurityEventTracker,
    log_binding_validation_failure,
    log_security_event_with_binding,
    log_suspicious_access_pattern,
)


class TestSecurityEventTracker:
    """Test SecurityEventTracker functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = SecurityEventTracker()

        # Create a test session with binding
        self.test_session = ContextSwitcherSession(
            session_id="test_session_123",
            created_at=datetime.now(timezone.utc),
            topic="Test topic",
        )

        # Create a client binding
        self.test_binding = ClientBinding(
            session_entropy="test_entropy",
            creation_timestamp=datetime.now(timezone.utc),
            binding_signature="test_signature",
            access_pattern_hash="test_hash",
        )
        self.test_session.client_binding = self.test_binding

    def test_tracker_initialization(self):
        """Test SecurityEventTracker initialization"""
        tracker = SecurityEventTracker()
        assert tracker.security_logger is not None
        assert isinstance(tracker.event_categories, dict)
        assert len(tracker.event_categories) > 0

    def test_custom_logger_initialization(self):
        """Test initialization with custom logger name"""
        tracker = SecurityEventTracker("custom.logger")
        assert tracker.security_logger.name == "custom.logger"

    def test_event_severity_detection(self):
        """Test event severity level detection"""
        # Test exact matches
        assert self.tracker._get_event_severity("authentication") == logging.WARNING
        assert self.tracker._get_event_severity("suspicious_access") == logging.ERROR

        # Test prefix matches
        assert (
            self.tracker._get_event_severity("authentication_failed") == logging.WARNING
        )
        assert (
            self.tracker._get_event_severity("suspicious_access_pattern")
            == logging.ERROR
        )

        # Test unknown event type defaults to WARNING
        assert self.tracker._get_event_severity("unknown_event") == logging.WARNING

    def test_log_security_event_with_session(self):
        """Test logging security event with session context"""
        event_type = "test_security_event"
        session_id = "test_session_123"
        details = {"test_key": "test_value"}

        # This should not raise an exception
        self.tracker.log_security_event(
            event_type, session_id, details, self.test_session
        )

        # Check that event was recorded in session
        assert len(self.test_session.security_events) == 1
        recorded_event = self.test_session.security_events[0]
        assert recorded_event["type"] == event_type
        assert recorded_event["details"]["test_key"] == "test_value"

    def test_log_security_event_without_session(self):
        """Test logging security event without session context"""
        event_type = "system_event"
        session_id = "orphaned_session"
        details = {"system_key": "system_value"}

        # Should not raise an exception
        self.tracker.log_security_event(event_type, session_id, details)

        # No session to verify, but should log successfully

    def test_log_security_event_binding_context(self):
        """Test that binding context is added correctly"""
        event_type = "binding_test_event"
        session_id = self.test_session.session_id
        details = {"original_key": "original_value"}

        # Capture the event record (would need a more sophisticated test setup for full verification)
        # For now, verify it doesn't crash and session gets the event
        self.tracker.log_security_event(
            event_type, session_id, details, self.test_session
        )

        # Verify event was recorded
        assert len(self.test_session.security_events) == 1
        event = self.test_session.security_events[0]
        assert "original_key" in event["details"]

    def test_log_binding_validation_failure(self):
        """Test logging binding validation failures"""
        session_id = "test_session"
        tool_name = "test_tool"
        failure_count = 2

        self.tracker.log_binding_validation_failure(
            session_id, tool_name, failure_count, self.test_session
        )

        # Verify event was logged to session
        assert len(self.test_session.security_events) == 1
        event = self.test_session.security_events[0]
        assert event["type"] == "binding_validation_failed"
        assert event["details"]["tool_name"] == tool_name
        assert event["details"]["failure_count"] == failure_count

    def test_log_binding_validation_failure_critical(self):
        """Test logging critical binding validation failures"""
        session_id = "test_session"
        tool_name = "test_tool"
        failure_count = 5  # Above threshold

        self.tracker.log_binding_validation_failure(
            session_id, tool_name, failure_count, self.test_session
        )

        # Verify critical flag is set
        event = self.test_session.security_events[0]
        assert event["details"]["is_critical"] is True

    def test_log_suspicious_access_pattern(self):
        """Test logging suspicious access patterns"""
        session_id = "test_session"
        tool_name = "test_tool"
        access_metrics = {
            "access_rate": 150,
            "unique_prompts": 2,
        }

        self.tracker.log_suspicious_access_pattern(
            session_id, tool_name, access_metrics, self.test_session
        )

        # Verify event was logged
        assert len(self.test_session.security_events) == 1
        event = self.test_session.security_events[0]
        assert event["type"] == "suspicious_access_pattern"
        assert event["details"]["tool_name"] == tool_name
        assert event["details"]["access_metrics"] == access_metrics

    def test_classify_access_pattern(self):
        """Test access pattern classification"""
        # High frequency access
        metrics = {"access_rate": 150}
        classification = self.tracker._classify_access_pattern(metrics)
        assert classification == "high_frequency_access"

        # Elevated access rate
        metrics = {"access_rate": 75}
        classification = self.tracker._classify_access_pattern(metrics)
        assert classification == "elevated_access_rate"

        # Repeated prompts
        metrics = {"unique_prompts": 2}
        classification = self.tracker._classify_access_pattern(metrics)
        assert classification == "repeated_prompts_automation"

        # Rapid tool switching
        metrics = {"rapid_tool_switching": True}
        classification = self.tracker._classify_access_pattern(metrics)
        assert classification == "rapid_tool_switching"

        # Unclassified
        metrics = {"unknown_metric": True}
        classification = self.tracker._classify_access_pattern(metrics)
        assert classification == "unclassified_suspicious"

    def test_log_key_rotation_event(self):
        """Test logging key rotation events"""
        key_info = {
            "old_key_hash": "abc123",
            "new_key_hash": "def456",
            "previous_keys_count": 1,
        }
        reason = "security_incident"

        # Should not raise an exception
        self.tracker.log_key_rotation_event(key_info, reason)

        # Verify the details structure
        # (In a real implementation, we'd capture the log output)

    def test_get_event_summary_placeholder(self):
        """Test event summary (placeholder implementation)"""
        summary = self.tracker.get_event_summary("test_session", 24)

        assert isinstance(summary, dict)
        assert "session_id" in summary
        assert "time_window_hours" in summary
        assert "implementation_status" in summary
        assert summary["implementation_status"] == "placeholder"

    def test_custom_severity_override(self):
        """Test custom severity override"""
        event_type = "custom_event"
        session_id = "test_session"
        details = {"key": "value"}
        custom_severity = logging.CRITICAL

        # Should accept custom severity
        self.tracker.log_security_event(
            event_type, session_id, details, severity=custom_severity
        )

        # Event should be logged (we'd need to capture logs to verify severity)


class TestBackwardCompatibilityFunctions:
    """Test backward compatibility functions"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_session = ContextSwitcherSession(
            session_id="test_session",
            created_at=datetime.now(timezone.utc),
            topic="Test topic",
        )

    def test_log_security_event_with_binding_function(self):
        """Test backward compatibility function"""
        # Should not raise an exception
        log_security_event_with_binding(
            "test_event", "test_session", {"key": "value"}, self.test_session
        )

        # Verify event was recorded in session
        assert len(self.test_session.security_events) == 1
        assert self.test_session.security_events[0]["type"] == "test_event"

    def test_log_binding_validation_failure_function(self):
        """Test backward compatibility validation failure function"""
        log_binding_validation_failure(
            "test_session", "test_tool", 3, self.test_session
        )

        # Verify event was recorded
        assert len(self.test_session.security_events) == 1
        event = self.test_session.security_events[0]
        assert event["details"]["failure_count"] == 3

    def test_log_suspicious_access_pattern_function(self):
        """Test backward compatibility suspicious access function"""
        metrics = {"access_rate": 200}
        log_suspicious_access_pattern(
            "test_session", "test_tool", metrics, self.test_session
        )

        # Verify event was recorded
        assert len(self.test_session.security_events) == 1
        event = self.test_session.security_events[0]
        assert event["details"]["access_metrics"]["access_rate"] == 200

    def test_functions_without_session(self):
        """Test backward compatibility functions without session"""
        # Should not raise exceptions
        log_security_event_with_binding("event", "session", {"key": "value"})
        log_binding_validation_failure("session", "tool", 1)
        log_suspicious_access_pattern("session", "tool", {"rate": 50})


class TestSessionWithoutBinding:
    """Test handling of sessions without client binding"""

    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = SecurityEventTracker()

        # Create session without binding
        self.legacy_session = ContextSwitcherSession(
            session_id="legacy_session",
            created_at=datetime.now(timezone.utc),
            topic="Legacy topic",
        )
        # Explicitly no client_binding

    def test_log_event_with_no_binding_session(self):
        """Test logging event with session that has no binding"""
        self.tracker.log_security_event(
            "legacy_event", "legacy_session", {"key": "value"}, self.legacy_session
        )

        # Should still record the event
        assert len(self.legacy_session.security_events) == 1
        event = self.legacy_session.security_events[0]
        assert event["type"] == "legacy_event"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
