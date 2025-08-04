"""
Client binding security utilities for session hijacking prevention
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
import logging

from .models import ClientBinding, ContextSwitcherSession

logger = logging.getLogger(__name__)

# Global secret key for HMAC operations (in production, load from secure config)
_BINDING_SECRET_KEY = secrets.token_urlsafe(32)

# Security configuration
MAX_VALIDATION_FAILURES = 3
MAX_SECURITY_FLAGS = 5
BINDING_VALIDATION_WINDOW = timedelta(hours=24)
SUSPICIOUS_ACCESS_THRESHOLD = 100  # Requests per hour


class ClientBindingManager:
    """Manages secure client binding for session validation"""

    def __init__(self, secret_key: Optional[str] = None):
        """Initialize client binding manager

        Args:
            secret_key: Optional secret key for HMAC operations
        """
        self.secret_key = secret_key or _BINDING_SECRET_KEY
        self.suspicious_sessions: Dict[str, datetime] = {}

    def create_client_binding(
        self, session_id: str, initial_tool: str = "start_context_analysis"
    ) -> ClientBinding:
        """Create secure client binding for a new session

        Args:
            session_id: The session ID to bind
            initial_tool: The first tool called (for behavioral fingerprinting)

        Returns:
            ClientBinding object with secure binding data
        """
        now = datetime.utcnow()

        # Generate cryptographically secure entropy
        session_entropy = secrets.token_urlsafe(32)

        # Create access pattern hash based on initial conditions
        access_pattern_data = f"{session_id}:{initial_tool}:{now.timestamp()}"
        access_pattern_hash = hashlib.sha256(access_pattern_data.encode()).hexdigest()

        # Create binding object
        binding = ClientBinding(
            session_entropy=session_entropy,
            creation_timestamp=now,
            binding_signature="",  # Will be set below
            access_pattern_hash=access_pattern_hash,
            tool_usage_sequence=[initial_tool],
            last_validated=now,
        )

        # Generate and set binding signature
        binding.binding_signature = binding.generate_binding_signature(self.secret_key)

        logger.info(f"Created client binding for session {session_id}")
        return binding

    async def validate_session_binding(
        self, session: ContextSwitcherSession, tool_name: str
    ) -> Tuple[bool, str]:
        """Validate client binding for session access

        Args:
            session: The session to validate
            tool_name: The tool being accessed

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not session.client_binding:
            # Legacy session without binding - allow but log
            logger.warning(
                f"Session {session.session_id} has no client binding (legacy)"
            )
            return True, ""

        binding = session.client_binding

        # Check if session is marked suspicious
        if session.session_id in self.suspicious_sessions:
            last_suspicious = self.suspicious_sessions[session.session_id]
            if datetime.utcnow() - last_suspicious < timedelta(hours=1):
                return False, "Session flagged as suspicious - access denied"

        # Validate binding signature
        if not binding.validate_binding(self.secret_key):
            binding.validation_failures += 1
            session.record_security_event(
                "binding_validation_failed",
                {"tool_name": tool_name, "failure_count": binding.validation_failures},
            )

            logger.warning(
                f"Binding validation failed for session {session.session_id}"
            )

            if binding.validation_failures >= MAX_VALIDATION_FAILURES:
                self._mark_session_suspicious(session.session_id)
                return False, "Session binding validation failed - session invalidated"

            return False, "Client binding validation failed"

        # Check for suspicious behavior patterns
        if self._is_suspicious_access(session, tool_name):
            session.record_security_event(
                "suspicious_access_pattern",
                {
                    "tool_name": tool_name,
                    "access_count": session.access_count,
                    "time_since_creation": (
                        datetime.utcnow() - session.created_at
                    ).total_seconds(),
                },
            )
            self._mark_session_suspicious(session.session_id)
            return False, "Suspicious access pattern detected"

        # Update binding validation timestamp
        binding.last_validated = datetime.utcnow()

        # Record successful validation
        await session.record_access(tool_name)

        return True, ""

    def _is_suspicious_access(
        self, session: ContextSwitcherSession, tool_name: str
    ) -> bool:
        """Check if access pattern is suspicious

        Args:
            session: The session to check
            tool_name: The tool being accessed

        Returns:
            True if access pattern is suspicious
        """
        now = datetime.utcnow()

        # Check access rate (requests per hour)
        session_age_hours = (now - session.created_at).total_seconds() / 3600
        # Only check access rate for sessions older than 1 minute to avoid false positives
        if session_age_hours > 0.0167:  # 1 minute in hours
            access_rate = session.access_count / session_age_hours
            if access_rate > SUSPICIOUS_ACCESS_THRESHOLD:
                return True

        # Check for rapid tool switching (potential automation)
        if len(session.analyses) > 10:
            recent_analyses = session.analyses[-10:]
            unique_prompts = len(set(a.get("prompt", "") for a in recent_analyses))
            if unique_prompts < 3:  # Same prompts repeated
                return True

        # Check binding security flags
        if session.client_binding and session.client_binding.is_suspicious():
            return True

        return False

    def _mark_session_suspicious(self, session_id: str) -> None:
        """Mark a session as suspicious

        Args:
            session_id: The session ID to mark
        """
        self.suspicious_sessions[session_id] = datetime.utcnow()
        logger.warning(f"Marked session {session_id} as suspicious")

    def cleanup_suspicious_sessions(self) -> None:
        """Clean up old suspicious session markers"""
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.suspicious_sessions = {
            sid: timestamp
            for sid, timestamp in self.suspicious_sessions.items()
            if timestamp > cutoff
        }

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics for monitoring

        Returns:
            Dictionary of security metrics
        """
        return {
            "suspicious_sessions_count": len(self.suspicious_sessions),
            "binding_secret_key_set": bool(self.secret_key),
            "max_validation_failures": MAX_VALIDATION_FAILURES,
            "max_security_flags": MAX_SECURITY_FLAGS,
            "suspicious_access_threshold": SUSPICIOUS_ACCESS_THRESHOLD,
        }


# Global client binding manager instance
client_binding_manager = ClientBindingManager()


def create_secure_session_with_binding(
    session_id: str, topic: str, initial_tool: str = "start_context_analysis"
) -> ContextSwitcherSession:
    """Create a new session with secure client binding

    Args:
        session_id: The session ID
        topic: The analysis topic
        initial_tool: The initial tool being called

    Returns:
        ContextSwitcherSession with client binding
    """
    now = datetime.utcnow()

    # Create client binding
    binding = client_binding_manager.create_client_binding(session_id, initial_tool)

    # Create session with binding
    session = ContextSwitcherSession(
        session_id=session_id,
        created_at=now,
        client_binding=binding,
        topic=topic,
        last_accessed=now,
        access_count=1,  # Initial access already counted in binding
    )

    logger.info(f"Created secure session {session_id} with client binding")
    return session


async def validate_session_access(
    session: ContextSwitcherSession, tool_name: str
) -> Tuple[bool, str]:
    """Validate session access with client binding checks

    Args:
        session: The session to validate
        tool_name: The tool being accessed

    Returns:
        Tuple of (is_valid, error_message)
    """
    return await client_binding_manager.validate_session_binding(session, tool_name)


def log_security_event_with_binding(
    event_type: str,
    session_id: str,
    details: Dict[str, Any],
    session: Optional[ContextSwitcherSession] = None,
) -> None:
    """Log security event with binding context

    Args:
        event_type: Type of security event
        session_id: Session ID
        details: Event details
        session: Optional session object for additional context
    """
    # Add binding context to details
    if session and session.client_binding:
        details.update(
            {
                "has_client_binding": True,
                "validation_failures": session.client_binding.validation_failures,
                "security_flags_count": len(session.client_binding.security_flags),
                "is_suspicious": session.client_binding.is_suspicious(),
            }
        )
    else:
        details["has_client_binding"] = False

    # Record in session if available
    if session:
        session.record_security_event(event_type, details)

    # Log to system logger
    logger.warning(f"Security event {event_type} for session {session_id}: {details}")
