"""
Core client binding management for secure session validation.

This module provides the central orchestration of client binding operations,
coordinating between secret key management, validation services, and security
event tracking.
"""

import hashlib
import secrets
from ..logging_base import get_logger

logger = get_logger(__name__)
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Tuple

from ..models import ClientBinding, ContextSwitcherSession
from .secret_key_manager import SecretKeyManager, load_or_generate_secret_key
from .client_validation_service import (
    ClientValidationService,
    client_validation_service,
)
from .security_event_tracker import SecurityEventTracker, security_event_tracker

logger = get_logger(__name__)


class ClientBindingManager:
    """Central manager for secure client binding operations.

    This class orchestrates all client binding security features including:
    - Client binding creation and validation
    - Integration with secret key management
    - Suspicious behavior detection
    - Security event logging
    - Session lifecycle management

    Features:
        - Dependency injection for testability
        - Comprehensive error handling
        - Security metrics collection
        - Key rotation support
        - Backward compatibility
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        validation_service: Optional[ClientValidationService] = None,
        event_tracker: Optional[SecurityEventTracker] = None,
    ):
        """Initialize client binding manager.

        Args:
            secret_key: Optional secret key for HMAC operations (deprecated, use environment variable)
            validation_service: Optional validation service (uses global instance if not provided)
            event_tracker: Optional event tracker (uses global instance if not provided)
        """
        # Initialize secret key manager
        if secret_key:
            logger.warning(
                "Passing secret_key directly is deprecated. Use CONTEXT_SWITCHER_SECRET_KEY environment variable instead."
            )
            self.key_manager = SecretKeyManager(secret_key)
        else:
            # Use global secret key
            global_key = load_or_generate_secret_key()
            self.key_manager = SecretKeyManager(global_key)

        # Dependency injection for services
        self.validation_service = validation_service or client_validation_service
        self.event_tracker = event_tracker or security_event_tracker

        logger.info("ClientBindingManager initialized with security services")

    def create_client_binding(
        self, session_id: str, initial_tool: str = "start_context_analysis"
    ) -> ClientBinding:
        """Create secure client binding for a new session.

        Args:
            session_id: The session ID to bind
            initial_tool: The first tool called (for behavioral fingerprinting)

        Returns:
            ClientBinding: Secure client binding object

        Security Features:
            - Cryptographically secure entropy generation
            - Access pattern fingerprinting
            - HMAC signature generation
            - Behavioral baseline establishment
        """
        now = datetime.now(timezone.utc)

        try:
            # Generate session entropy
            session_entropy = secrets.token_urlsafe(32)

            # Create access pattern fingerprint
            access_pattern_data = f"{session_id}:{initial_tool}:{now.timestamp()}"
            access_pattern_hash = hashlib.sha256(
                access_pattern_data.encode()
            ).hexdigest()

            # Create binding object
            binding = ClientBinding(
                session_entropy=session_entropy,
                creation_timestamp=now,
                binding_signature="",  # Will be set below
                access_pattern_hash=access_pattern_hash,
                tool_usage_sequence=[initial_tool],
                last_validated=now,
            )

            # Generate secure binding signature
            binding.binding_signature = binding.generate_binding_signature(
                self.key_manager.current_key
            )

            # Log creation event
            self.event_tracker.log_security_event(
                "client_binding_created",
                session_id,
                {
                    "initial_tool": initial_tool,
                    "entropy_length": len(session_entropy),
                    "has_signature": bool(binding.binding_signature),
                },
                severity=logger.INFO,
            )

            logger.info(f"Created client binding for session {session_id}")
            return binding

        except Exception as e:
            logger.error(
                f"Failed to create client binding for session {session_id}: {e}"
            )
            # Log the failure
            self.event_tracker.log_security_event(
                "client_binding_creation_failed",
                session_id,
                {"error": str(e), "initial_tool": initial_tool},
                severity=logger.ERROR,
            )
            raise

    async def validate_session_binding(
        self, session: ContextSwitcherSession, tool_name: str
    ) -> Tuple[bool, str]:
        """Validate client binding for session access.

        Args:
            session: The session to validate
            tool_name: The tool being accessed

        Returns:
            Tuple[bool, str]: (is_valid, error_message)

        Validation Steps:
            1. Check for session lockout
            2. Legacy session handling
            3. Binding signature validation (with key rotation support)
            4. Suspicious behavior detection
            5. Access pattern analysis
            6. Security event logging
        """
        session_id = session.session_id

        try:
            # Check if session is locked out due to suspicious activity
            if self.validation_service.is_session_locked_out(session_id):
                self.event_tracker.log_security_event(
                    "session_access_denied_lockout",
                    session_id,
                    {"tool_name": tool_name, "reason": "suspicious_activity_lockout"},
                    session,
                    severity=logger.WARNING,
                )
                return False, "Session locked out due to suspicious activity"

            # Handle legacy sessions without binding
            if not session.client_binding:
                logger.warning(f"Session {session_id} has no client binding (legacy)")
                self.event_tracker.log_security_event(
                    "legacy_session_access",
                    session_id,
                    {"tool_name": tool_name, "has_binding": False},
                    session,
                    severity=logger.INFO,
                )
                return True, ""

            binding = session.client_binding

            # Validate binding signature (with key rotation support)
            if not self._validate_binding_with_rotation(binding):
                binding.validation_failures += 1

                # Log validation failure
                self.event_tracker.log_binding_validation_failure(
                    session_id, tool_name, binding.validation_failures, session
                )

                # Check if session should be locked out
                if (
                    binding.validation_failures
                    >= self.validation_service.validation_rules[
                        "max_validation_failures"
                    ]
                ):
                    self.validation_service.mark_session_suspicious(
                        session_id, "max_validation_failures_exceeded"
                    )
                    return (
                        False,
                        "Session binding validation failed - session invalidated",
                    )

                return False, "Client binding validation failed"

            # Check for suspicious behavior patterns
            access_pattern = self.validation_service.is_suspicious_access(
                session, tool_name
            )
            if access_pattern.is_suspicious:
                # Log suspicious access
                self.event_tracker.log_suspicious_access_pattern(
                    session_id, tool_name, access_pattern.metrics, session
                )

                # Mark session as suspicious
                self.validation_service.mark_session_suspicious(
                    session_id, access_pattern.reason
                )

                return (
                    False,
                    f"Suspicious access pattern detected: {access_pattern.reason}",
                )

            # Update binding validation timestamp
            binding.last_validated = datetime.now(timezone.utc)

            # Record successful access
            await session.record_access(tool_name)

            # Log successful validation (debug level to avoid noise)
            logger.debug(
                f"Successfully validated session {session_id} for tool {tool_name}"
            )

            return True, ""

        except Exception as e:
            logger.error(
                f"Error during session binding validation for {session_id}: {e}"
            )

            # Log the validation error
            self.event_tracker.log_security_event(
                "session_validation_error",
                session_id,
                {"tool_name": tool_name, "error": str(e)},
                session,
                severity=logger.ERROR,
            )

            # Fail securely - deny access on error
            return False, "Session validation error - access denied"

    def _validate_binding_with_rotation(self, binding: ClientBinding) -> bool:
        """Validate binding with key rotation support.

        Args:
            binding: The client binding to validate

        Returns:
            bool: True if binding is valid with current or previous keys

        Security Features:
            - Current key validation first (common case)
            - Previous key fallback during rotation grace period
            - Automatic re-signing with current key after successful validation
            - Comprehensive validation logging
        """
        try:
            # First try with current key (most common case)
            if binding.validate_binding(self.key_manager.current_key):
                return True

            # Try with previous keys during rotation grace period
            for i, prev_key in enumerate(self.key_manager.previous_keys):
                if binding.validate_binding(prev_key):
                    logger.info(
                        f"Validated binding with previous key #{i + 1} during rotation grace period"
                    )

                    # Re-sign with current key for future validations
                    # This helps migrate sessions to the new key
                    try:
                        binding.binding_signature = binding.generate_binding_signature(
                            self.key_manager.current_key
                        )
                        logger.debug("Re-signed binding with current key")
                    except Exception as e:
                        logger.warning(
                            f"Failed to re-sign binding with current key: {e}"
                        )

                    return True

            return False

        except Exception as e:
            logger.error(f"Error during binding validation: {e}")
            return False

    def rotate_secret_key(self, reason: str = "scheduled") -> str:
        """Rotate the secret key for enhanced security.

        Args:
            reason: Reason for the rotation (e.g., 'scheduled', 'security_incident')

        Returns:
            str: Hash of the new key (for verification, not the key itself)

        Security Features:
            - Atomic key rotation
            - Previous key preservation for grace period
            - Comprehensive audit logging
            - Error handling and rollback
        """
        try:
            old_key_info = self.key_manager.get_current_key_info()
            new_key = self.key_manager.rotate_key()

            # Get new key info for logging (without exposing the key)
            new_key_hash = hashlib.sha256(new_key.encode()).hexdigest()[:16]

            # Log key rotation event
            self.event_tracker.log_key_rotation_event(
                {
                    "old_key_hash": old_key_info["key_hash"],
                    "new_key_hash": new_key_hash,
                    "previous_keys_count": len(self.key_manager.previous_keys),
                },
                reason,
            )

            logger.info(f"Secret key rotated successfully (reason: {reason})")
            return new_key_hash

        except Exception as e:
            logger.error(f"Failed to rotate secret key: {e}")
            self.event_tracker.log_security_event(
                "key_rotation_failed",
                "SYSTEM",
                {"error": str(e), "reason": reason},
                severity=logger.ERROR,
            )
            raise

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics for monitoring.

        Returns:
            dict: Complete security metrics from all components

        Metrics Include:
            - Key management status
            - Validation service metrics
            - Suspicious session tracking
            - Security configuration
        """
        try:
            # Get validation service metrics
            validation_metrics = self.validation_service.get_validation_metrics()

            # Get key manager info
            key_info = self.key_manager.get_current_key_info()

            # Combine all metrics
            return {
                "key_management": key_info,
                "validation": validation_metrics,
                "suspicious_sessions": self.validation_service.get_suspicious_sessions_info(),
                "system_health": {
                    "key_manager_operational": True,
                    "validation_service_operational": True,
                    "event_tracker_operational": True,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_health": {
                    "key_manager_operational": False,
                    "validation_service_operational": False,
                    "event_tracker_operational": False,
                },
            }

    def cleanup_security_state(self) -> Dict[str, int]:
        """Clean up old security state (maintenance operation).

        Returns:
            dict: Cleanup statistics

        Cleanup Operations:
            - Remove expired suspicious sessions
            - Clean up old security events (if implemented)
            - Maintain key rotation history limits
        """
        cleanup_stats = {"suspicious_sessions_cleaned": 0}

        try:
            # Clean up suspicious sessions
            cleaned_sessions = self.validation_service.cleanup_suspicious_sessions()
            cleanup_stats["suspicious_sessions_cleaned"] = cleaned_sessions

            # Log cleanup operation
            if cleaned_sessions > 0:
                self.event_tracker.log_security_event(
                    "security_state_cleanup",
                    "SYSTEM",
                    {"cleaned_sessions": cleaned_sessions},
                    severity=logger.INFO,
                )

            logger.info(f"Security state cleanup completed: {cleanup_stats}")
            return cleanup_stats

        except Exception as e:
            logger.error(f"Error during security state cleanup: {e}")
            cleanup_stats["error"] = str(e)
            return cleanup_stats


# Global client binding manager instance
# This will be initialized with the global secret key
_global_client_binding_manager = None


def get_client_binding_manager() -> ClientBindingManager:
    """Get the global client binding manager instance.

    Returns:
        ClientBindingManager: Global instance (singleton pattern)
    """
    global _global_client_binding_manager
    if _global_client_binding_manager is None:
        _global_client_binding_manager = ClientBindingManager()
    return _global_client_binding_manager


# Backward compatibility alias
client_binding_manager = get_client_binding_manager()


def create_secure_session_with_binding(
    session_id: str, topic: str, initial_tool: str = "start_context_analysis"
) -> ContextSwitcherSession:
    """Create a new session with secure client binding.

    Args:
        session_id: The session ID
        topic: The analysis topic
        initial_tool: The initial tool being called

    Returns:
        ContextSwitcherSession: Session with client binding

    Security Features:
        - Automatic client binding creation
        - Behavioral fingerprinting
        - Access tracking initialization
        - Security event logging
    """
    now = datetime.now(timezone.utc)
    manager = get_client_binding_manager()

    try:
        # Create client binding
        binding = manager.create_client_binding(session_id, initial_tool)

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

    except Exception as e:
        logger.error(f"Failed to create secure session {session_id}: {e}")
        # Log the failure
        manager.event_tracker.log_security_event(
            "secure_session_creation_failed",
            session_id,
            {"topic": topic, "initial_tool": initial_tool, "error": str(e)},
            severity=logger.ERROR,
        )
        raise


async def validate_session_access(
    session: ContextSwitcherSession, tool_name: str
) -> Tuple[bool, str]:
    """Validate session access with client binding checks.

    Args:
        session: The session to validate
        tool_name: The tool being accessed

    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    manager = get_client_binding_manager()
    return await manager.validate_session_binding(session, tool_name)
