"""
Client validation service for suspicious behavior detection and access control.

This module provides comprehensive client validation including suspicious
access pattern detection, behavioral analysis, and security rule enforcement.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from ..models import ContextSwitcherSession

logger = logging.getLogger(__name__)

# Security configuration constants
MAX_VALIDATION_FAILURES = 3
MAX_SECURITY_FLAGS = 5
BINDING_VALIDATION_WINDOW = timedelta(hours=24)
SUSPICIOUS_ACCESS_THRESHOLD = 100  # Requests per hour
MIN_SESSION_AGE_FOR_RATE_CHECK = timedelta(
    minutes=1
)  # Minimum age before checking rate
RAPID_TOOL_SWITCHING_THRESHOLD = 10  # Number of recent analyses to check
MIN_UNIQUE_PROMPTS_THRESHOLD = 3  # Minimum unique prompts in rapid switching check
SUSPICIOUS_SESSION_LOCKOUT_DURATION = timedelta(hours=1)


class AccessPattern:
    """Represents access pattern analysis result."""

    def __init__(
        self,
        is_suspicious: bool,
        reason: str,
        metrics: Dict[str, Any],
        severity: str = "medium",
    ):
        """Initialize access pattern result.

        Args:
            is_suspicious: Whether the pattern is suspicious
            reason: Reason for the determination
            metrics: Detailed metrics about the access pattern
            severity: Severity level ('low', 'medium', 'high', 'critical')
        """
        self.is_suspicious = is_suspicious
        self.reason = reason
        self.metrics = metrics
        self.severity = severity
        self.timestamp = datetime.now(timezone.utc)


class ClientValidationService:
    """Service for validating client behavior and detecting suspicious patterns.

    This service provides comprehensive client validation including:
    - Access rate analysis
    - Behavioral pattern detection
    - Tool usage pattern analysis
    - Security flag management
    - Session lockout management

    Features:
        - Configurable thresholds and rules
        - Multiple pattern detection algorithms
        - Grace periods for new sessions
        - Suspicious session tracking
        - Comprehensive metrics collection
    """

    def __init__(self):
        """Initialize client validation service."""
        self.suspicious_sessions: Dict[str, datetime] = {}
        self.validation_rules = {
            "max_validation_failures": MAX_VALIDATION_FAILURES,
            "max_security_flags": MAX_SECURITY_FLAGS,
            "suspicious_access_threshold": SUSPICIOUS_ACCESS_THRESHOLD,
            "min_session_age_for_rate_check": MIN_SESSION_AGE_FOR_RATE_CHECK,
            "lockout_duration": SUSPICIOUS_SESSION_LOCKOUT_DURATION,
        }

    def is_suspicious_access(
        self, session: ContextSwitcherSession, tool_name: str
    ) -> AccessPattern:
        """Comprehensive suspicious access pattern analysis.

        Args:
            session: The session to analyze
            tool_name: The tool being accessed

        Returns:
            AccessPattern: Complete analysis result with metrics and reasoning

        Analysis Dimensions:
            - Access rate (requests per hour)
            - Tool usage patterns
            - Prompt repetition patterns
            - Client binding security flags
            - Session behavior over time
        """
        now = datetime.now(timezone.utc)
        metrics = {}

        # Check access rate (requests per hour)
        rate_analysis = self._analyze_access_rate(session, now, metrics)
        if rate_analysis.is_suspicious:
            return rate_analysis

        # Check for rapid tool switching with repeated prompts
        tool_analysis = self._analyze_tool_usage_patterns(session, metrics)
        if tool_analysis.is_suspicious:
            return tool_analysis

        # Check client binding security flags
        binding_analysis = self._analyze_binding_security(session, metrics)
        if binding_analysis.is_suspicious:
            return binding_analysis

        # Check for session age vs access count patterns
        age_analysis = self._analyze_session_age_patterns(session, now, metrics)
        if age_analysis.is_suspicious:
            return age_analysis

        # All checks passed
        return AccessPattern(
            is_suspicious=False,
            reason="access_pattern_normal",
            metrics=metrics,
            severity="low",
        )

    def _analyze_access_rate(
        self, session: ContextSwitcherSession, now: datetime, metrics: Dict[str, Any]
    ) -> AccessPattern:
        """Analyze session access rate for suspicious patterns.

        Args:
            session: Session to analyze
            now: Current timestamp
            metrics: Metrics dictionary to populate

        Returns:
            AccessPattern: Access rate analysis result
        """
        session_age = now - session.created_at
        session_age_hours = session_age.total_seconds() / 3600

        metrics["session_age_hours"] = session_age_hours
        metrics["access_count"] = session.access_count

        # Only check access rate for sessions older than minimum age to avoid false positives
        if session_age > self.validation_rules["min_session_age_for_rate_check"]:
            access_rate = session.access_count / session_age_hours
            metrics["access_rate"] = access_rate

            if access_rate > self.validation_rules["suspicious_access_threshold"]:
                return AccessPattern(
                    is_suspicious=True,
                    reason="excessive_access_rate",
                    metrics=metrics,
                    severity="high" if access_rate > 200 else "medium",
                )

        return AccessPattern(
            is_suspicious=False, reason="access_rate_normal", metrics=metrics
        )

    def _analyze_tool_usage_patterns(
        self, session: ContextSwitcherSession, metrics: Dict[str, Any]
    ) -> AccessPattern:
        """Analyze tool usage patterns for automation indicators.

        Args:
            session: Session to analyze
            metrics: Metrics dictionary to populate

        Returns:
            AccessPattern: Tool usage pattern analysis result
        """
        if (
            not hasattr(session, "analyses")
            or len(session.analyses) <= RAPID_TOOL_SWITCHING_THRESHOLD
        ):
            return AccessPattern(
                is_suspicious=False, reason="insufficient_tool_data", metrics=metrics
            )

        # Check recent analyses for repeated patterns
        recent_analyses = session.analyses[-RAPID_TOOL_SWITCHING_THRESHOLD:]
        unique_prompts = set(a.get("prompt", "") for a in recent_analyses)
        unique_prompts.discard("")  # Remove empty prompts

        metrics["recent_analyses_count"] = len(recent_analyses)
        metrics["unique_prompts_count"] = len(unique_prompts)
        metrics["prompt_repetition_ratio"] = (
            (len(recent_analyses) - len(unique_prompts)) / len(recent_analyses)
            if recent_analyses
            else 0
        )

        if len(unique_prompts) < MIN_UNIQUE_PROMPTS_THRESHOLD:
            return AccessPattern(
                is_suspicious=True,
                reason="repeated_prompts_automation",
                metrics=metrics,
                severity="medium",
            )

        # Check for extremely rapid tool switching
        if len(recent_analyses) == RAPID_TOOL_SWITCHING_THRESHOLD:
            # All recent analyses within a short time window might indicate automation
            recent_timestamps = [
                datetime.fromisoformat(a.get("timestamp", "").replace("Z", "+00:00"))
                for a in recent_analyses
                if a.get("timestamp")
            ]

            if len(recent_timestamps) >= 2:
                time_span = max(recent_timestamps) - min(recent_timestamps)
                metrics["rapid_switching_timespan_seconds"] = time_span.total_seconds()

                if time_span.total_seconds() < 60:  # 10 analyses in under 1 minute
                    return AccessPattern(
                        is_suspicious=True,
                        reason="rapid_tool_switching",
                        metrics=metrics,
                        severity="high",
                    )

        return AccessPattern(
            is_suspicious=False, reason="tool_usage_normal", metrics=metrics
        )

    def _analyze_binding_security(
        self, session: ContextSwitcherSession, metrics: Dict[str, Any]
    ) -> AccessPattern:
        """Analyze client binding security flags.

        Args:
            session: Session to analyze
            metrics: Metrics dictionary to populate

        Returns:
            AccessPattern: Binding security analysis result
        """
        if not session.client_binding:
            metrics["has_binding"] = False
            return AccessPattern(
                is_suspicious=False, reason="no_binding_legacy", metrics=metrics
            )

        binding = session.client_binding
        metrics["has_binding"] = True
        metrics["validation_failures"] = binding.validation_failures
        metrics["security_flags_count"] = len(binding.security_flags)
        metrics["security_flags"] = list(binding.security_flags)

        # Check if binding itself reports as suspicious
        if binding.is_suspicious():
            severity = (
                "critical"
                if binding.validation_failures >= MAX_VALIDATION_FAILURES
                else "high"
            )
            return AccessPattern(
                is_suspicious=True,
                reason="binding_marked_suspicious",
                metrics=metrics,
                severity=severity,
            )

        return AccessPattern(
            is_suspicious=False, reason="binding_security_normal", metrics=metrics
        )

    def _analyze_session_age_patterns(
        self, session: ContextSwitcherSession, now: datetime, metrics: Dict[str, Any]
    ) -> AccessPattern:
        """Analyze session age vs activity patterns.

        Args:
            session: Session to analyze
            now: Current timestamp
            metrics: Metrics dictionary to populate

        Returns:
            AccessPattern: Session age pattern analysis result
        """
        session_age = now - session.created_at
        metrics["session_age_seconds"] = session_age.total_seconds()

        # Check for sessions with suspicious longevity and high activity
        if session_age.total_seconds() > 86400:  # Older than 1 day
            if session.access_count > 1000:  # Very high access count
                return AccessPattern(
                    is_suspicious=True,
                    reason="long_lived_high_activity",
                    metrics=metrics,
                    severity="medium",
                )

        # Check for brand new sessions with immediate high activity
        if session_age.total_seconds() < 60:  # Less than 1 minute old
            if session.access_count > 10:  # High immediate activity
                return AccessPattern(
                    is_suspicious=True,
                    reason="immediate_high_activity",
                    metrics=metrics,
                    severity="medium",
                )

        return AccessPattern(
            is_suspicious=False, reason="session_age_pattern_normal", metrics=metrics
        )

    def is_session_locked_out(self, session_id: str) -> bool:
        """Check if a session is currently locked out due to suspicious activity.

        Args:
            session_id: Session ID to check

        Returns:
            bool: True if session is locked out
        """
        if session_id not in self.suspicious_sessions:
            return False

        lockout_time = self.suspicious_sessions[session_id]
        lockout_duration = self.validation_rules["lockout_duration"]

        if datetime.now(timezone.utc) - lockout_time < lockout_duration:
            return True

        # Lockout period expired, remove from tracking
        del self.suspicious_sessions[session_id]
        return False

    def mark_session_suspicious(
        self, session_id: str, reason: str = "suspicious_activity"
    ) -> None:
        """Mark a session as suspicious and apply lockout.

        Args:
            session_id: Session ID to mark
            reason: Reason for marking as suspicious
        """
        self.suspicious_sessions[session_id] = datetime.now(timezone.utc)
        logger.warning(f"Marked session {session_id} as suspicious: {reason}")

    def cleanup_suspicious_sessions(self) -> int:
        """Clean up old suspicious session markers.

        Returns:
            int: Number of sessions cleaned up
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        original_count = len(self.suspicious_sessions)

        self.suspicious_sessions = {
            sid: timestamp
            for sid, timestamp in self.suspicious_sessions.items()
            if timestamp > cutoff
        }

        cleaned_count = original_count - len(self.suspicious_sessions)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old suspicious session markers")

        return cleaned_count

    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive validation metrics for monitoring.

        Returns:
            dict: Validation service metrics
        """
        return {
            "suspicious_sessions_count": len(self.suspicious_sessions),
            "validation_rules": self.validation_rules.copy(),
            "active_lockouts": sum(
                1
                for timestamp in self.suspicious_sessions.values()
                if datetime.now(timezone.utc) - timestamp
                < self.validation_rules["lockout_duration"]
            ),
            "configuration": {
                "max_validation_failures": MAX_VALIDATION_FAILURES,
                "max_security_flags": MAX_SECURITY_FLAGS,
                "suspicious_access_threshold": SUSPICIOUS_ACCESS_THRESHOLD,
                "lockout_duration_hours": SUSPICIOUS_SESSION_LOCKOUT_DURATION.total_seconds()
                / 3600,
            },
        }

    def update_validation_rule(self, rule_name: str, value: Any) -> bool:
        """Update a validation rule (for runtime configuration).

        Args:
            rule_name: Name of the rule to update
            value: New value for the rule

        Returns:
            bool: True if rule was updated successfully
        """
        if rule_name in self.validation_rules:
            old_value = self.validation_rules[rule_name]
            self.validation_rules[rule_name] = value
            logger.info(f"Updated validation rule {rule_name}: {old_value} -> {value}")
            return True

        logger.warning(f"Unknown validation rule: {rule_name}")
        return False

    def get_suspicious_sessions_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about suspicious sessions.

        Returns:
            List[Dict]: List of suspicious session information
        """
        now = datetime.now(timezone.utc)
        return [
            {
                "session_id": session_id,
                "marked_at": timestamp.isoformat(),
                "age_seconds": (now - timestamp).total_seconds(),
                "is_locked_out": (now - timestamp)
                < self.validation_rules["lockout_duration"],
                "time_remaining_seconds": max(
                    0,
                    (
                        timestamp + self.validation_rules["lockout_duration"] - now
                    ).total_seconds(),
                ),
            }
            for session_id, timestamp in self.suspicious_sessions.items()
        ]


# Global client validation service instance
client_validation_service = ClientValidationService()


def is_suspicious_access(session: ContextSwitcherSession, tool_name: str) -> bool:
    """Check if access pattern is suspicious (backward compatibility function).

    Args:
        session: The session to check
        tool_name: The tool being accessed

    Returns:
        bool: True if access pattern is suspicious
    """
    pattern = client_validation_service.is_suspicious_access(session, tool_name)
    return pattern.is_suspicious


def mark_session_suspicious(
    session_id: str, reason: str = "suspicious_activity"
) -> None:
    """Mark a session as suspicious (backward compatibility function).

    Args:
        session_id: Session ID to mark
        reason: Reason for marking as suspicious
    """
    client_validation_service.mark_session_suspicious(session_id, reason)


def cleanup_suspicious_sessions() -> int:
    """Clean up old suspicious session markers (backward compatibility function).

    Returns:
        int: Number of sessions cleaned up
    """
    return client_validation_service.cleanup_suspicious_sessions()


def get_security_metrics() -> Dict[str, Any]:
    """Get security metrics for monitoring (backward compatibility function).

    Returns:
        dict: Security metrics
    """
    return client_validation_service.get_validation_metrics()
