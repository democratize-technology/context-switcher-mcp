"""
Client binding security utilities for session hijacking prevention
"""

import hashlib
import secrets
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple, List
import logging
import json
from pathlib import Path

from .models import ClientBinding, ContextSwitcherSession

logger = logging.getLogger(__name__)


# Secret key management with rotation support
def _load_or_generate_secret_key() -> str:
    """Load secret key from environment or generate a secure one

    Priority order:
    1. CONTEXT_SWITCHER_SECRET_KEY environment variable
    2. Secret key file in ~/.context_switcher/secret_key.json
    3. Generate new key and save to file
    """
    # Try environment variable first
    env_key = os.environ.get("CONTEXT_SWITCHER_SECRET_KEY")
    if env_key:
        logger.info("Using secret key from environment variable")
        return env_key

    # Try loading from file
    config_dir = Path.home() / ".context_switcher"
    secret_file = config_dir / "secret_key.json"

    if secret_file.exists():
        try:
            with open(secret_file, "r") as f:
                data = json.load(f)
                if "current_key" in data:
                    logger.info("Loaded secret key from file")
                    return data["current_key"]
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load secret key from file: {e}")

    # Generate new key and save it
    new_key = secrets.token_urlsafe(32)
    config_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Save with rotation support structure
        data = {
            "current_key": new_key,
            "previous_keys": [],  # For key rotation support
            "created_at": datetime.utcnow().isoformat(),
            "rotation_count": 0,
        }
        with open(secret_file, "w") as f:
            json.dump(data, f, indent=2)
        # Set restrictive permissions (owner read/write only)
        secret_file.chmod(0o600)
        logger.info("Generated and saved new secret key")
    except IOError as e:
        logger.warning(f"Failed to save secret key to file: {e}")

    return new_key


# Initialize the secret key
_BINDING_SECRET_KEY = _load_or_generate_secret_key()


# Key rotation support
class SecretKeyManager:
    """Manages secret key rotation for enhanced security"""

    def __init__(self):
        self.current_key = _BINDING_SECRET_KEY
        self.previous_keys: List[str] = []
        self._load_previous_keys()

    def _load_previous_keys(self):
        """Load previous keys for validation during rotation period"""
        secret_file = Path.home() / ".context_switcher" / "secret_key.json"
        if secret_file.exists():
            try:
                with open(secret_file, "r") as f:
                    data = json.load(f)
                    self.previous_keys = data.get("previous_keys", [])[
                        :5
                    ]  # Keep last 5 keys
            except (json.JSONDecodeError, IOError):
                pass

    def rotate_key(self) -> str:
        """Rotate to a new secret key, keeping the old one for grace period"""
        new_key = secrets.token_urlsafe(32)
        self.previous_keys.insert(0, self.current_key)
        self.previous_keys = self.previous_keys[:5]  # Keep only last 5 keys

        self.current_key = new_key

        # Save the rotated keys
        secret_file = Path.home() / ".context_switcher" / "secret_key.json"
        try:
            data = {
                "current_key": new_key,
                "previous_keys": self.previous_keys,
                "rotated_at": datetime.utcnow().isoformat(),
                "rotation_count": len(self.previous_keys),
            }
            with open(secret_file, "w") as f:
                json.dump(data, f, indent=2)
            secret_file.chmod(0o600)
            logger.info("Successfully rotated secret key")
        except IOError as e:
            logger.error(f"Failed to save rotated key: {e}")

        return new_key

    def validate_with_any_key(self, data: str, signature: str) -> bool:
        """Validate signature with current or previous keys (for rotation grace period)"""
        # Try current key first
        if self._validate_signature(data, signature, self.current_key):
            return True

        # Try previous keys during grace period
        for key in self.previous_keys:
            if self._validate_signature(data, signature, key):
                logger.info("Validated with previous key during rotation grace period")
                return True

        return False

    def _validate_signature(self, data: str, signature: str, key: str) -> bool:
        """Validate a signature with a specific key"""
        expected = hashlib.sha256(f"{data}:{key}".encode()).hexdigest()
        return secrets.compare_digest(expected, signature)


# Global secret key manager instance
_SECRET_KEY_MANAGER = SecretKeyManager()

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
            secret_key: Optional secret key for HMAC operations (deprecated, use environment variable)
        """
        # Use the global secret key manager for better security
        self.key_manager = _SECRET_KEY_MANAGER
        if secret_key:
            logger.warning(
                "Passing secret_key directly is deprecated. Use CONTEXT_SWITCHER_SECRET_KEY environment variable instead."
            )
            self.key_manager.current_key = secret_key
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

        session_entropy = secrets.token_urlsafe(32)
        access_pattern_data = f"{session_id}:{initial_tool}:{now.timestamp()}"
        access_pattern_hash = hashlib.sha256(access_pattern_data.encode()).hexdigest()
        binding = ClientBinding(
            session_entropy=session_entropy,
            creation_timestamp=now,
            binding_signature="",  # Will be set below
            access_pattern_hash=access_pattern_hash,
            tool_usage_sequence=[initial_tool],
            last_validated=now,
        )

        binding.binding_signature = binding.generate_binding_signature(
            self.key_manager.current_key
        )

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

        # Validate binding signature (with key rotation support)
        if not self._validate_binding_with_rotation(binding):
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

    def _validate_binding_with_rotation(self, binding: ClientBinding) -> bool:
        """Validate binding with key rotation support

        Args:
            binding: The client binding to validate

        Returns:
            True if binding is valid with current or previous keys
        """
        # First try with current key
        if binding.validate_binding(self.key_manager.current_key):
            return True

        # Try with previous keys during rotation grace period
        for prev_key in self.key_manager.previous_keys:
            if binding.validate_binding(prev_key):
                logger.info(
                    "Validated binding with previous key during rotation grace period"
                )
                # Optionally re-sign with current key for future validations
                binding.binding_signature = binding.generate_binding_signature(
                    self.key_manager.current_key
                )
                return True

        return False

    def _mark_session_suspicious(self, session_id: str) -> None:
        """Mark a session as suspicious

        Args:
            session_id: The session ID to mark
        """
        self.suspicious_sessions[session_id] = datetime.utcnow()
        logger.warning(f"Marked session {session_id} as suspicious")

    def rotate_secret_key(self) -> None:
        """Rotate the secret key for enhanced security

        This should be called periodically (e.g., daily) or after security events
        """
        new_key = self.key_manager.rotate_key()
        logger.info(
            f"Secret key rotated. New key hash: {hashlib.sha256(new_key.encode()).hexdigest()[:8]}..."
        )

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
            "binding_secret_key_set": bool(self.key_manager.current_key),
            "key_rotation_count": len(self.key_manager.previous_keys),
            "key_source": "environment"
            if os.environ.get("CONTEXT_SWITCHER_SECRET_KEY")
            else "file",
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
