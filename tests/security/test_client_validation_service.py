"""
Test suite for ClientValidationService security module.
"""

import os
import sys  # noqa: E402
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))  # noqa: E402

from context_switcher_mcp.models import (  # noqa: E402 # noqa: E402
    ClientBinding,
    ContextSwitcherSession,
)
from context_switcher_mcp.security.client_validation_service import (  # noqa: E402 # noqa: E402
    AccessPattern,
    ClientValidationService,
    cleanup_suspicious_sessions,
    get_security_metrics,
    is_suspicious_access,
    mark_session_suspicious,
)

# Tests are now enabled after fixing API mismatches


class TestAccessPattern:
    """Test AccessPattern class"""

    def test_access_pattern_creation(self):
        """Test AccessPattern object creation"""
        pattern = AccessPattern(
            is_suspicious=True,
            reason="test_reason",
            metrics={"test_metric": 123},
            severity="high",
        )

        assert pattern.is_suspicious is True
        assert pattern.reason == "test_reason"
        assert pattern.metrics["test_metric"] == 123
        assert pattern.severity == "high"
        assert isinstance(pattern.timestamp, datetime)

    def test_access_pattern_default_severity(self):
        """Test AccessPattern with default severity"""
        pattern = AccessPattern(is_suspicious=False, reason="normal", metrics={})

        assert pattern.severity == "medium"  # Default value


class TestClientValidationService:
    """Test ClientValidationService functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.service = ClientValidationService()

        # Create test session with binding
        self.test_session = self._create_test_session()
        self.test_binding = self._create_test_binding()
        self.test_session.client_binding = self.test_binding

    def _create_test_session(self, session_id="test_session", access_count=5):
        """Create a test session"""
        return ContextSwitcherSession(
            session_id=session_id,
            created_at=datetime.now(timezone.utc) - timedelta(hours=1),  # 1 hour old
            topic="Test topic",
            access_count=access_count,
        )

    def _create_test_binding(self):
        """Create a test client binding"""
        return ClientBinding(
            session_entropy="test_entropy",
            creation_timestamp=datetime.now(timezone.utc),
            binding_signature="test_signature",
            access_pattern_hash="test_hash",
        )

    def test_service_initialization(self):
        """Test ClientValidationService initialization"""
        service = ClientValidationService()

        assert isinstance(service.suspicious_sessions, dict)
        assert len(service.suspicious_sessions) == 0
        assert isinstance(service.validation_rules, dict)
        assert "max_validation_failures" in service.validation_rules
        assert "suspicious_access_threshold" in service.validation_rules

    def test_normal_access_pattern(self):
        """Test normal access pattern detection"""
        session = self._create_test_session(access_count=10)  # Normal access count
        session.client_binding = self._create_test_binding()

        pattern = self.service.is_suspicious_access(session, "test_tool")

        assert pattern.is_suspicious is False
        assert "normal" in pattern.reason
        assert isinstance(pattern.metrics, dict)

    def test_excessive_access_rate_detection(self):
        """Test excessive access rate detection"""
        session = self._create_test_session(access_count=200)  # Very high access count
        session.created_at = datetime.now(timezone.utc) - timedelta(
            hours=1
        )  # 1 hour old
        session.client_binding = self._create_test_binding()

        pattern = self.service.is_suspicious_access(session, "test_tool")

        assert pattern.is_suspicious is True
        assert "excessive_access_rate" in pattern.reason
        assert "access_rate" in pattern.metrics
        assert pattern.severity in ["medium", "high"]

    def test_access_rate_grace_period(self):
        """Test access rate check grace period for new sessions"""
        session = self._create_test_session(access_count=50)  # High count
        session.created_at = datetime.now(timezone.utc) - timedelta(
            seconds=30
        )  # Very new
        session.client_binding = self._create_test_binding()

        pattern = self.service.is_suspicious_access(session, "test_tool")

        # Should not be flagged due to grace period
        assert pattern.is_suspicious is False or "access_rate" not in pattern.reason

    def test_repeated_prompts_detection(self):
        """Test repeated prompts automation detection"""
        session = self._create_test_session()
        session.client_binding = self._create_test_binding()

        # Add analyses with repeated prompts
        session.analyses = [
            {
                "prompt": "same prompt",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "prompt": "same prompt",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "prompt": "same prompt",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "prompt": "different prompt",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ] * 3  # 12 analyses total with mostly repeated prompts

        pattern = self.service.is_suspicious_access(session, "test_tool")

        assert pattern.is_suspicious is True
        assert "repeated_prompts_automation" in pattern.reason
        assert "unique_prompts_count" in pattern.metrics

    def test_rapid_tool_switching_detection(self):
        """Test rapid tool switching detection"""
        session = self._create_test_session()
        session.client_binding = self._create_test_binding()

        # Create 11 analyses in a very short time span (need > 10 for rapid switching check)
        base_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        session.analyses = []

        for i in range(11):
            session.analyses.append(
                {
                    "prompt": f"prompt_{i}",
                    "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                }
            )

        pattern = self.service.is_suspicious_access(session, "test_tool")

        assert pattern.is_suspicious is True
        assert "rapid_tool_switching" in pattern.reason
        assert "rapid_switching_timespan_seconds" in pattern.metrics

    def test_binding_security_flags_detection(self):
        """Test client binding security flags detection"""
        session = self._create_test_session()
        binding = self._create_test_binding()

        # Set suspicious binding conditions
        binding.validation_failures = 5  # Above threshold
        session.client_binding = binding

        pattern = self.service.is_suspicious_access(session, "test_tool")

        assert pattern.is_suspicious is True
        assert "binding_marked_suspicious" in pattern.reason
        assert pattern.severity in ["high", "critical"]

    def test_long_lived_high_activity_detection(self):
        """Test long-lived session with high activity detection"""
        session = self._create_test_session(access_count=1500)
        session.created_at = datetime.now(timezone.utc) - timedelta(
            days=2
        )  # 2 days old
        session.client_binding = self._create_test_binding()

        pattern = self.service.is_suspicious_access(session, "test_tool")

        assert pattern.is_suspicious is True
        assert "long_lived_high_activity" in pattern.reason

    def test_immediate_high_activity_detection(self):
        """Test immediate high activity detection for new sessions"""
        session = self._create_test_session(access_count=15)
        session.created_at = datetime.now(timezone.utc) - timedelta(
            seconds=30
        )  # Very new
        session.client_binding = self._create_test_binding()

        pattern = self.service.is_suspicious_access(session, "test_tool")

        assert pattern.is_suspicious is True
        assert "immediate_high_activity" in pattern.reason

    def test_session_lockout_functionality(self):
        """Test session lockout functionality"""
        session_id = "test_session_lockout"

        # Session should not be locked out initially
        assert self.service.is_session_locked_out(session_id) is False

        # Mark session as suspicious
        self.service.mark_session_suspicious(session_id, "test_reason")

        # Session should now be locked out
        assert self.service.is_session_locked_out(session_id) is True

        # Verify session is in suspicious list
        assert session_id in self.service.suspicious_sessions

    def test_session_lockout_expiry(self):
        """Test session lockout expiry"""
        session_id = "test_session_expiry"

        # Mark session as suspicious in the past
        past_time = datetime.now(timezone.utc) - timedelta(hours=2)
        self.service.suspicious_sessions[session_id] = past_time

        # Should no longer be locked out (default is 1 hour)
        assert self.service.is_session_locked_out(session_id) is False

        # Session should be removed from suspicious list
        assert session_id not in self.service.suspicious_sessions

    def test_cleanup_suspicious_sessions(self):
        """Test cleanup of old suspicious sessions"""
        # Add some old and new suspicious sessions
        old_session = "old_session"
        new_session = "new_session"

        self.service.suspicious_sessions[old_session] = datetime.now(
            timezone.utc
        ) - timedelta(days=2)
        self.service.suspicious_sessions[new_session] = datetime.now(
            timezone.utc
        ) - timedelta(minutes=30)

        # Clean up
        cleaned = self.service.cleanup_suspicious_sessions()

        # Should have cleaned up the old session
        assert cleaned == 1
        assert old_session not in self.service.suspicious_sessions
        assert new_session in self.service.suspicious_sessions

    def test_get_validation_metrics(self):
        """Test validation metrics collection"""
        # Add some suspicious sessions
        self.service.mark_session_suspicious("session1", "test")
        self.service.mark_session_suspicious("session2", "test")

        metrics = self.service.get_validation_metrics()

        assert isinstance(metrics, dict)
        assert "suspicious_sessions_count" in metrics
        assert "validation_rules" in metrics
        assert "active_lockouts" in metrics
        assert "configuration" in metrics

        assert metrics["suspicious_sessions_count"] == 2
        assert metrics["active_lockouts"] == 2

    def test_update_validation_rule(self):
        """Test updating validation rules"""
        self.service.validation_rules["suspicious_access_threshold"]
        new_value = 150

        # Update rule
        result = self.service.update_validation_rule(
            "suspicious_access_threshold", new_value
        )

        assert result is True
        assert self.service.validation_rules["suspicious_access_threshold"] == new_value

        # Test updating non-existent rule
        result = self.service.update_validation_rule("non_existent_rule", 123)
        assert result is False

    def test_get_suspicious_sessions_info(self):
        """Test getting detailed suspicious sessions information"""
        session_id = "info_test_session"
        self.service.mark_session_suspicious(session_id, "test_reason")

        info = self.service.get_suspicious_sessions_info()

        assert isinstance(info, list)
        assert len(info) == 1

        session_info = info[0]
        assert session_info["session_id"] == session_id
        assert "marked_at" in session_info
        assert "age_seconds" in session_info
        assert "is_locked_out" in session_info
        assert "time_remaining_seconds" in session_info

    def test_no_binding_legacy_session(self):
        """Test handling of legacy sessions without client binding"""
        # Create a very simple session that won't trigger access rate issues
        session = ContextSwitcherSession(
            session_id="legacy_session",
            created_at=datetime.now(timezone.utc)
            - timedelta(minutes=5),  # Recent session
            topic="Legacy test",
            access_count=1,  # Very low access count
        )
        session.client_binding = None  # No binding
        # Ensure no analyses to prevent insufficient_tool_data path
        session.analyses = []

        pattern = self.service.is_suspicious_access(session, "test_tool")

        assert pattern.is_suspicious is False
        # For sessions with no binding, metrics should show has_binding: False
        assert pattern.metrics.get("has_binding") is False
        # The reason will be access_pattern_normal since no suspicious activity detected
        assert pattern.reason == "access_pattern_normal"

    def test_insufficient_analysis_data(self):
        """Test handling of sessions with insufficient analysis data"""
        session = self._create_test_session()
        session.client_binding = self._create_test_binding()
        session.analyses = []  # No analyses

        pattern = self.service.is_suspicious_access(session, "test_tool")

        # Should not be suspicious due to insufficient data
        assert (
            pattern.is_suspicious is False or "insufficient_tool_data" in pattern.reason
        )


class TestBackwardCompatibilityFunctions:
    """Test backward compatibility functions"""

    def setup_method(self):
        """Set up test fixtures"""
        self.session = ContextSwitcherSession(
            session_id="test_session",
            created_at=datetime.now(timezone.utc),
            topic="Test topic",
        )
        self.binding = ClientBinding(
            session_entropy="test_entropy",
            creation_timestamp=datetime.now(timezone.utc),
            binding_signature="test_signature",
            access_pattern_hash="test_hash",
        )
        self.session.client_binding = self.binding

    def test_is_suspicious_access_function(self):
        """Test backward compatibility is_suspicious_access function"""
        result = is_suspicious_access(self.session, "test_tool")

        assert isinstance(result, bool)

    def test_mark_session_suspicious_function(self):
        """Test backward compatibility mark_session_suspicious function"""
        session_id = "test_session_bc"

        # Should not raise an exception
        mark_session_suspicious(session_id, "test_reason")

        # Global service should have the session marked
        from context_switcher_mcp.security.client_validation_service import (
            client_validation_service,
        )

        assert session_id in client_validation_service.suspicious_sessions

    def test_cleanup_suspicious_sessions_function(self):
        """Test backward compatibility cleanup_suspicious_sessions function"""
        # Should return a number
        result = cleanup_suspicious_sessions()
        assert isinstance(result, int)

    def test_get_security_metrics_function(self):
        """Test backward compatibility get_security_metrics function"""
        metrics = get_security_metrics()

        assert isinstance(metrics, dict)
        assert "suspicious_sessions_count" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
