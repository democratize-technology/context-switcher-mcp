"""Security event logging and monitoring utilities"""

import time
from .logging_base import get_logger
from typing import Any, Dict, Optional

logger = get_logger(__name__)


class SecurityEventLogger:
    """Centralized security event logging and monitoring"""

    def __init__(self):
        self.blocked_attempts = {}

    def log_security_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        session_id: Optional[str] = None,
        client_id: Optional[str] = None,
    ):
        """
        Enhanced logging for security-related events

        Args:
            event_type: Type of security event
            details: Event details
            session_id: Associated session ID if any
            client_id: Client identifier if available
        """
        log_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "session_id": session_id,
            "client_id": client_id,
            "details": details,
        }

        # Use structured logging for security events
        logger.warning(f"SECURITY_EVENT: {log_entry}")

        # Track blocked attempts for monitoring
        if event_type in ["content_validation_failure", "prompt_injection_blocked"]:
            self.blocked_attempts[client_id or "default"] = (
                self.blocked_attempts.get(client_id or "default", 0) + 1
            )

    def get_blocked_attempts(self, client_id: str = "default") -> int:
        """Get count of blocked attempts for a client"""
        return self.blocked_attempts.get(client_id, 0)

    def reset_blocked_attempts(self, client_id: str = "default"):
        """Reset blocked attempts counter for a client"""
        if client_id in self.blocked_attempts:
            del self.blocked_attempts[client_id]


# Global security event logger
_security_event_logger = SecurityEventLogger()


def log_security_event(
    event_type: str,
    details: Dict[str, Any],
    session_id: Optional[str] = None,
    client_id: Optional[str] = None,
):
    """
    Log a security event

    Args:
        event_type: Type of security event
        details: Event details
        session_id: Associated session ID if any
        client_id: Client identifier if available
    """
    _security_event_logger.log_security_event(
        event_type, details, session_id, client_id
    )


def get_security_metrics(client_id: str = "default") -> Dict[str, Any]:
    """
    Get security metrics for monitoring

    Args:
        client_id: Client identifier

    Returns:
        Dictionary of security metrics
    """
    return {
        "blocked_attempts": _security_event_logger.get_blocked_attempts(client_id),
        "logger_initialized": True,
    }


def reset_security_metrics(client_id: str = "default"):
    """
    Reset security metrics for a client

    Args:
        client_id: Client identifier
    """
    _security_event_logger.reset_blocked_attempts(client_id)
