"""Security management for sessions and client bindings"""

import hashlib
import secrets
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ClientBinding:
    """Secure client binding data for session validation"""

    # Core binding identifiers
    session_entropy: str  # Cryptographically secure session entropy
    creation_timestamp: datetime  # Session creation time
    binding_signature: str  # HMAC signature of binding data

    # Behavioral fingerprint
    access_pattern_hash: str  # Hash of initial access patterns
    tool_usage_sequence: List[str] = field(
        default_factory=list
    )  # Initial tool usage pattern

    # Security metadata
    validation_failures: int = 0  # Count of validation failures
    last_validated: datetime = field(default_factory=datetime.utcnow)
    security_flags: List[str] = field(default_factory=list)  # Security event flags

    def generate_binding_signature(self, secret_key: str) -> str:
        """Generate HMAC signature for binding validation"""
        data = f"{self.session_entropy}:{self.creation_timestamp.isoformat()}"
        return hashlib.pbkdf2_hmac(
            "sha256", data.encode(), secret_key.encode(), 600000
        ).hex()

    def validate_binding(self, secret_key: str) -> bool:
        """Validate the binding signature"""
        expected_signature = self.generate_binding_signature(secret_key)
        return secrets.compare_digest(self.binding_signature, expected_signature)

    def add_security_flag(self, flag: str) -> None:
        """Add a security flag to the binding"""
        if flag not in self.security_flags:
            self.security_flags.append(flag)

    def is_suspicious(self) -> bool:
        """Check if binding shows suspicious activity"""
        return (
            self.validation_failures > 3
            or len(self.security_flags) > 5
            or "multiple_failed_validations" in self.security_flags
        )


@dataclass
class SecurityEvent:
    """Represents a security event in a session"""

    event_type: str
    timestamp: datetime
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert security event to dictionary"""
        return {
            "type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class SessionSecurity:
    """Manages security aspects of sessions"""

    def __init__(self, session_id: str, client_binding: Optional[ClientBinding] = None):
        """Initialize session security manager

        Args:
            session_id: The session identifier
            client_binding: Optional client binding for validation
        """
        self.session_id = session_id
        self.client_binding = client_binding
        self.security_events: List[SecurityEvent] = []

    def create_client_binding(
        self, secret_key: str, access_pattern_hash: Optional[str] = None
    ) -> ClientBinding:
        """Create a new client binding for this session

        Args:
            secret_key: Secret key for HMAC generation
            access_pattern_hash: Optional access pattern hash

        Returns:
            New ClientBinding instance
        """
        session_entropy = secrets.token_urlsafe(32)
        creation_timestamp = datetime.now(timezone.utc)

        binding = ClientBinding(
            session_entropy=session_entropy,
            creation_timestamp=creation_timestamp,
            binding_signature="",  # Will be set after creation
            access_pattern_hash=access_pattern_hash
            or self._generate_default_pattern_hash(),
        )

        # Generate and set the binding signature
        binding.binding_signature = binding.generate_binding_signature(secret_key)

        self.client_binding = binding
        return binding

    def validate_binding(self, secret_key: str) -> bool:
        """Validate the current client binding

        Args:
            secret_key: Secret key for validation

        Returns:
            True if binding is valid or no binding exists (backward compatibility)
        """
        if not self.client_binding:
            return (
                True  # No binding = legacy session (allowed for backward compatibility)
            )

        is_valid = self.client_binding.validate_binding(secret_key)

        if not is_valid:
            self.client_binding.validation_failures += 1
            self.record_security_event(
                "binding_validation_failed",
                {"validation_failures": self.client_binding.validation_failures},
            )
        else:
            self.client_binding.last_validated = datetime.now(timezone.utc)

        return is_valid

    def record_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Record a security event for this session

        Args:
            event_type: Type of security event
            details: Additional event details
        """
        event = SecurityEvent(
            event_type=event_type, timestamp=datetime.now(timezone.utc), details=details
        )
        self.security_events.append(event)

        # Add security flag to client binding if present
        if self.client_binding:
            self.client_binding.add_security_flag(event_type)

    def update_tool_usage_pattern(self, tool_name: str) -> None:
        """Update the tool usage pattern in client binding

        Args:
            tool_name: Name of the tool being used
        """
        if self.client_binding and len(self.client_binding.tool_usage_sequence) < 10:
            self.client_binding.tool_usage_sequence.append(tool_name)

    def is_session_suspicious(self) -> bool:
        """Check if the session shows suspicious activity

        Returns:
            True if session appears suspicious
        """
        if self.client_binding and self.client_binding.is_suspicious():
            return True

        # Check for excessive security events
        recent_events = [
            event
            for event in self.security_events
            if (datetime.now(timezone.utc) - event.timestamp).total_seconds()
            < 3600  # Last hour
        ]

        return len(recent_events) > 10

    def get_security_summary(self) -> Dict[str, Any]:
        """Get a summary of security status

        Returns:
            Dictionary containing security summary
        """
        binding_info = None
        if self.client_binding:
            binding_info = {
                "validation_failures": self.client_binding.validation_failures,
                "last_validated": self.client_binding.last_validated.isoformat(),
                "security_flags_count": len(self.client_binding.security_flags),
                "is_suspicious": self.client_binding.is_suspicious(),
                "tool_usage_count": len(self.client_binding.tool_usage_sequence),
            }

        return {
            "session_id": self.session_id,
            "client_binding": binding_info,
            "security_events_count": len(self.security_events),
            "is_suspicious": self.is_session_suspicious(),
            "recent_events": [
                event.to_dict()
                for event in self.security_events[-5:]  # Last 5 events
            ],
        }

    def _generate_default_pattern_hash(self) -> str:
        """Generate a default access pattern hash"""
        default_data = f"{self.session_id}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(default_data.encode()).hexdigest()

    def cleanup_old_events(self, max_age_hours: int = 24) -> None:
        """Clean up old security events

        Args:
            max_age_hours: Maximum age of events to keep in hours
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        self.security_events = [
            event
            for event in self.security_events
            if event.timestamp.timestamp() > cutoff_time
        ]
