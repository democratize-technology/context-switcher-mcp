"""Data models for Context-Switcher MCP"""

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ModelBackend(str, Enum):
    """Supported model backends"""

    BEDROCK = "bedrock"
    LITELLM = "litellm"
    OLLAMA = "ollama"


@dataclass
class Thread:
    """Represents a single perspective thread"""

    id: str
    name: str
    system_prompt: str
    model_backend: ModelBackend
    model_name: str | None
    conversation_history: list[dict[str, str]] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        self.conversation_history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )


@dataclass
class ClientBinding:
    """Secure client binding data for session validation"""

    # Core binding identifiers
    session_entropy: str  # Cryptographically secure session entropy
    creation_timestamp: datetime  # Session creation time
    binding_signature: str  # HMAC signature of binding data

    # Behavioral fingerprint
    access_pattern_hash: str  # Hash of initial access patterns
    tool_usage_sequence: list[str] = field(
        default_factory=list
    )  # Initial tool usage pattern

    # Security metadata
    validation_failures: int = 0  # Count of validation failures
    last_validated: datetime = field(default_factory=lambda: datetime.now(UTC))
    security_flags: list[str] = field(default_factory=list)  # Security event flags

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
class ContextSwitcherSession:
    """Manages a context-switching analysis session with client binding"""

    session_id: str
    created_at: datetime
    client_binding: ClientBinding | None = None
    threads: dict[str, Thread] = field(default_factory=dict)
    analyses: list[dict[str, Any]] = field(default_factory=list)
    topic: str | None = None

    # Session security metadata
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    security_events: list[dict[str, Any]] = field(default_factory=list)

    # Concurrency control
    version: int = 0  # Version for optimistic locking and race condition detection

    def __post_init__(self):
        """Initialize session components

        Note: Async locks are now managed by SessionLockManager
        """
        # No async lock initialization needed here anymore
        pass

    def add_thread(self, thread: Thread) -> None:
        """Add a perspective thread to the session"""
        self.threads[thread.name] = thread

    def get_thread(self, name: str) -> Thread | None:
        """Get a thread by name"""
        return self.threads.get(name)

    async def record_access(self, tool_name: str) -> None:
        """Record session access for behavioral analysis (thread-safe async version)"""
        from .session_lock_manager import get_session_lock_manager

        lock_manager = get_session_lock_manager()
        async with lock_manager.acquire_lock(self.session_id):
            self.access_count += 1
            self.last_accessed = datetime.now(UTC)
            self.version += 1  # Increment version for change tracking

            # Update client binding tool usage pattern
            if (
                self.client_binding
                and len(self.client_binding.tool_usage_sequence) < 10
            ):
                self.client_binding.tool_usage_sequence.append(tool_name)

    def record_security_event(
        self, event_type: str, event_record: dict[str, Any]
    ) -> None:
        """Record a security event for this session"""
        # Support both old API (details dict) and new API (full event record)
        if isinstance(event_record, dict) and "event_type" in event_record:
            # New API - full event record from SecurityEventTracker
            event = event_record.copy()
            event["type"] = event_type  # Ensure consistency
        else:
            # Legacy API - just details dict
            event = {
                "type": event_type,
                "timestamp": datetime.now(UTC).isoformat(),
                "details": event_record,
            }
        self.security_events.append(event)

        # Add security flag to client binding if present
        if self.client_binding:
            self.client_binding.add_security_flag(event_type)

    def is_binding_valid(self, secret_key: str) -> bool:
        """Validate client binding if present"""
        if not self.client_binding:
            return (
                True  # No binding = legacy session (allowed for backward compatibility)
            )

        return self.client_binding.validate_binding(secret_key)

    def record_analysis(
        self,
        prompt: str,
        responses: dict[str, str],
        active_count: int,
        abstained_count: int,
    ) -> None:
        """Record an analysis for history"""
        self.analyses.append(
            {
                "prompt": prompt,
                "timestamp": datetime.now(UTC).isoformat(),
                "responses": responses,
                "active_count": active_count,
                "abstained_count": abstained_count,
            }
        )

        # Note: Analysis access tracking moved to async contexts

    def get_last_analysis(self) -> dict[str, Any] | None:
        """Get the most recent analysis"""
        return self.analyses[-1] if self.analyses else None
