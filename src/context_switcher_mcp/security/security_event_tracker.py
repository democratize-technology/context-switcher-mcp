"""
Security event tracking and audit logging for client binding operations.

This module provides comprehensive security event logging, audit trails,
and monitoring capabilities for the client binding security system.
"""

import logging
from ..logging_base import get_logger
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from ..models import ContextSwitcherSession

logger = get_logger(__name__)


class SecurityEventTracker:
    """Tracks and logs security events with comprehensive audit capabilities.

    This class provides centralized security event management including:
    - Structured security event logging
    - Audit trail management
    - Event correlation and pattern detection
    - Integration with session security context

    Features:
        - Standardized event format with timestamps
        - Client binding context awareness
        - Configurable logging levels and destinations
        - Event categorization and severity mapping
    """

    def __init__(self, logger_name: str = __name__):
        """Initialize security event tracker.

        Args:
            logger_name: Name of the logger to use for security events
        """
        self.security_logger = get_logger(logger_name)
        self.event_categories = {
            "authentication": logging.WARNING,
            "authorization": logging.WARNING,
            "binding_validation": logging.WARNING,
            "suspicious_access": logging.ERROR,
            "session_hijack": logging.CRITICAL,
            "rate_limit": logging.WARNING,
            "key_rotation": logging.INFO,
            "validation_failure": logging.WARNING,
            "access_pattern": logging.INFO,
        }

    def log_security_event(
        self,
        event_type: str,
        session_id: str,
        details: Dict[str, Any],
        session: Optional[ContextSwitcherSession] = None,
        severity: Optional[int] = None,
    ) -> None:
        """Log a security event with comprehensive context.

        Args:
            event_type: Type of security event (e.g., 'binding_validation_failed')
            session_id: Session ID associated with the event
            details: Event-specific details dictionary
            session: Optional session object for additional context
            severity: Optional custom severity level (logging.INFO, WARNING, ERROR, CRITICAL)

        Security Features:
            - Adds binding context automatically
            - Structured event format for analysis
            - Session security state correlation
            - Timestamp and severity normalization
        """
        # Determine severity level
        if severity is None:
            severity = self._get_event_severity(event_type)

        # Create comprehensive event record
        event_record = {
            "event_type": event_type,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details.copy(),  # Defensive copy
        }

        # Add binding context if session is provided
        if session and session.client_binding:
            event_record["binding_context"] = {
                "has_client_binding": True,
                "validation_failures": session.client_binding.validation_failures,
                "security_flags_count": len(session.client_binding.security_flags),
                "is_suspicious": session.client_binding.is_suspicious(),
                "last_validated": session.client_binding.last_validated.isoformat()
                if session.client_binding.last_validated
                else None,
                "creation_age_seconds": (
                    datetime.now(timezone.utc)
                    - session.client_binding.creation_timestamp
                ).total_seconds(),
            }
        else:
            event_record["binding_context"] = {"has_client_binding": False}

        # Add session context if available
        if session:
            event_record["session_context"] = {
                "access_count": session.access_count,
                "session_age_seconds": (
                    datetime.now(timezone.utc) - session.created_at
                ).total_seconds(),
                "security_events_count": len(session.security_events),
                "analyses_count": len(session.analyses)
                if hasattr(session, "analyses")
                else 0,
            }

        # Record in session if available
        if session:
            session.record_security_event(event_type, event_record["details"])

        # Log to system logger with appropriate severity
        self._log_event_with_severity(severity, event_type, session_id, event_record)

    def _get_event_severity(self, event_type: str) -> int:
        """Determine severity level for an event type.

        Args:
            event_type: The type of security event

        Returns:
            int: Logging severity level (logging.INFO, WARNING, ERROR, CRITICAL)
        """
        # Check for exact match first
        if event_type in self.event_categories:
            return self.event_categories[event_type]

        # Check for category prefixes
        for category, severity in self.event_categories.items():
            if event_type.startswith(category):
                return severity

        # Default to WARNING for unknown event types
        return logging.WARNING

    def _log_event_with_severity(
        self,
        severity: int,
        event_type: str,
        session_id: str,
        event_record: Dict[str, Any],
    ) -> None:
        """Log event with specified severity level.

        Args:
            severity: Logging severity level
            event_type: Type of security event
            session_id: Session ID
            event_record: Complete event record
        """
        message = f"Security event '{event_type}' for session {session_id}"

        # Create structured log entry
        extra_data = {
            "security_event": True,
            "event_type": event_type,
            "session_id": session_id,
            "event_record": event_record,
        }

        # Log with appropriate severity
        self.security_logger.log(severity, message, extra=extra_data)

        # For critical events, also log to console and main logger
        if severity >= logging.ERROR:
            console_message = f"🚨 SECURITY ALERT: {message}"
            if event_record.get("binding_context", {}).get("is_suspicious"):
                console_message += " [SUSPICIOUS SESSION]"

            # Log to main logger for observability and to console for immediate visibility
            logger.critical(
                console_message,
                extra={"security_alert": True, "event_record": event_record},
            )
            print(console_message)  # Keep console output for critical security alerts

    def log_binding_validation_failure(
        self,
        session_id: str,
        tool_name: str,
        failure_count: int,
        session: Optional[ContextSwitcherSession] = None,
    ) -> None:
        """Log client binding validation failure with specific context.

        Args:
            session_id: Session ID with validation failure
            tool_name: Tool that was being accessed
            failure_count: Current number of validation failures
            session: Optional session object for context
        """
        details = {
            "tool_name": tool_name,
            "failure_count": failure_count,
            "is_critical": failure_count >= 3,  # Configurable threshold
        }

        severity = logging.ERROR if failure_count >= 3 else logging.WARNING

        self.log_security_event(
            "binding_validation_failed",
            session_id,
            details,
            session,
            severity=severity,
        )

    def log_suspicious_access_pattern(
        self,
        session_id: str,
        tool_name: str,
        access_metrics: Dict[str, Any],
        session: Optional[ContextSwitcherSession] = None,
    ) -> None:
        """Log suspicious access pattern detection.

        Args:
            session_id: Session ID with suspicious access
            tool_name: Tool being accessed
            access_metrics: Metrics that triggered the suspicious detection
            session: Optional session object for context
        """
        details = {
            "tool_name": tool_name,
            "access_metrics": access_metrics,
            "pattern_type": self._classify_access_pattern(access_metrics),
        }

        self.log_security_event(
            "suspicious_access_pattern",
            session_id,
            details,
            session,
            severity=logging.ERROR,
        )

    def log_key_rotation_event(
        self, key_info: Dict[str, Any], reason: str = "scheduled"
    ) -> None:
        """Log secret key rotation event.

        Args:
            key_info: Information about the key rotation (without exposing keys)
            reason: Reason for the rotation (e.g., 'scheduled', 'security_incident')
        """
        details = {
            "reason": reason,
            "key_info": key_info,
            "rotation_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.log_security_event(
            "key_rotation",
            "SYSTEM",
            details,
            severity=logging.INFO,
        )

    def _classify_access_pattern(self, access_metrics: Dict[str, Any]) -> str:
        """Classify the type of suspicious access pattern.

        Args:
            access_metrics: Access pattern metrics

        Returns:
            str: Classification of the suspicious pattern
        """
        if "access_rate" in access_metrics:
            rate = access_metrics["access_rate"]
            if rate > 100:
                return "high_frequency_access"
            elif rate > 50:
                return "elevated_access_rate"

        if "unique_prompts" in access_metrics:
            unique_count = access_metrics["unique_prompts"]
            if unique_count < 3:
                return "repeated_prompts_automation"

        if "rapid_tool_switching" in access_metrics:
            return "rapid_tool_switching"

        return "unclassified_suspicious"

    def get_event_summary(
        self, session_id: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Any]:
        """Get summary of security events for monitoring.

        Args:
            session_id: Optional session ID to filter events
            hours: Number of hours to look back

        Returns:
            dict: Summary of security events

        Note:
            This is a placeholder for future implementation with proper
            event storage and querying capabilities.
        """
        # Future implementation would query stored events
        # For now, return basic structure
        return {
            "session_id": session_id,
            "time_window_hours": hours,
            "total_events": 0,
            "events_by_type": {},
            "critical_events": 0,
            "warning_events": 0,
            "implementation_status": "placeholder",
        }


# Global security event tracker instance
security_event_tracker = SecurityEventTracker()


def log_security_event_with_binding(
    event_type: str,
    session_id: str,
    details: Dict[str, Any],
    session: Optional[ContextSwitcherSession] = None,
) -> None:
    """Log security event with binding context (backward compatibility function).

    This function provides backward compatibility with the original API
    while delegating to the new SecurityEventTracker implementation.

    Args:
        event_type: Type of security event
        session_id: Session ID
        details: Event details
        session: Optional session object for additional context
    """
    security_event_tracker.log_security_event(event_type, session_id, details, session)


def log_binding_validation_failure(
    session_id: str,
    tool_name: str,
    failure_count: int,
    session: Optional[ContextSwitcherSession] = None,
) -> None:
    """Log binding validation failure (convenience function).

    Args:
        session_id: Session ID with validation failure
        tool_name: Tool that was being accessed
        failure_count: Current number of validation failures
        session: Optional session object for context
    """
    security_event_tracker.log_binding_validation_failure(
        session_id, tool_name, failure_count, session
    )


def log_suspicious_access_pattern(
    session_id: str,
    tool_name: str,
    access_metrics: Dict[str, Any],
    session: Optional[ContextSwitcherSession] = None,
) -> None:
    """Log suspicious access pattern (convenience function).

    Args:
        session_id: Session ID with suspicious access
        tool_name: Tool being accessed
        access_metrics: Metrics that triggered the suspicious detection
        session: Optional session object for context
    """
    security_event_tracker.log_suspicious_access_pattern(
        session_id, tool_name, access_metrics, session
    )
