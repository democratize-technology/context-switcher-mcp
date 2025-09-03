"""Pure data types and models for session management

This module contains all the data structures used by the simplified session
management system. These are pure data types with no business logic.
"""

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
    model_name: str | None = None
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

    def to_dict(self) -> dict[str, Any]:
        """Convert thread to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "model_backend": self.model_backend.value,
            "model_name": self.model_name,
            "conversation_history": self.conversation_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Thread":
        """Create thread from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            system_prompt=data["system_prompt"],
            model_backend=ModelBackend(data["model_backend"]),
            model_name=data.get("model_name"),
            conversation_history=data.get("conversation_history", []),
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
    tool_usage_sequence: list[str] = field(default_factory=list)

    # Security metadata
    validation_failures: int = 0
    last_validated: datetime = field(default_factory=lambda: datetime.now(UTC))
    security_flags: list[str] = field(default_factory=list)

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

    def to_dict(self) -> dict[str, Any]:
        """Convert client binding to dictionary for serialization"""
        return {
            "session_entropy": self.session_entropy,
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "binding_signature": self.binding_signature,
            "access_pattern_hash": self.access_pattern_hash,
            "tool_usage_sequence": self.tool_usage_sequence,
            "validation_failures": self.validation_failures,
            "last_validated": self.last_validated.isoformat(),
            "security_flags": self.security_flags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClientBinding":
        """Create client binding from dictionary"""
        return cls(
            session_entropy=data["session_entropy"],
            creation_timestamp=datetime.fromisoformat(data["creation_timestamp"]),
            binding_signature=data["binding_signature"],
            access_pattern_hash=data["access_pattern_hash"],
            tool_usage_sequence=data.get("tool_usage_sequence", []),
            validation_failures=data.get("validation_failures", 0),
            last_validated=datetime.fromisoformat(
                data.get("last_validated", datetime.now(UTC).isoformat())
            ),
            security_flags=data.get("security_flags", []),
        )


@dataclass
class SecurityEvent:
    """Represents a security event in a session"""

    event_type: str
    timestamp: datetime
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert security event to dictionary"""
        return {
            "type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityEvent":
        """Create security event from dictionary"""
        return cls(
            event_type=data["type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            details=data["details"],
        )


@dataclass
class AnalysisRecord:
    """Represents a single analysis performed in a session"""

    prompt: str
    timestamp: datetime
    responses: dict[str, str]
    active_count: int
    abstained_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert analysis record to dictionary"""
        return {
            "prompt": self.prompt,
            "timestamp": self.timestamp.isoformat(),
            "responses": self.responses,
            "active_count": self.active_count,
            "abstained_count": self.abstained_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisRecord":
        """Create analysis record from dictionary"""
        return cls(
            prompt=data["prompt"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            responses=data["responses"],
            active_count=data["active_count"],
            abstained_count=data["abstained_count"],
        )


@dataclass
class SessionMetrics:
    """Session performance and usage metrics"""

    access_count: int = 0
    analysis_count: int = 0
    thread_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    total_response_time: float = 0.0  # Total response time in seconds
    avg_response_time: float = 0.0  # Average response time in seconds
    error_count: int = 0
    security_event_count: int = 0

    def record_analysis(self, response_time: float) -> None:
        """Record an analysis execution"""
        self.analysis_count += 1
        self.total_response_time += response_time
        if self.analysis_count > 0:
            self.avg_response_time = self.total_response_time / self.analysis_count

    def record_access(self) -> None:
        """Record a session access"""
        self.access_count += 1
        self.last_accessed = datetime.now(UTC)

    def record_error(self) -> None:
        """Record an error"""
        self.error_count += 1

    def record_security_event(self) -> None:
        """Record a security event"""
        self.security_event_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "access_count": self.access_count,
            "analysis_count": self.analysis_count,
            "thread_count": self.thread_count,
            "last_accessed": self.last_accessed.isoformat(),
            "total_response_time": self.total_response_time,
            "avg_response_time": self.avg_response_time,
            "error_count": self.error_count,
            "security_event_count": self.security_event_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetrics":
        """Create metrics from dictionary"""
        return cls(
            access_count=data.get("access_count", 0),
            analysis_count=data.get("analysis_count", 0),
            thread_count=data.get("thread_count", 0),
            last_accessed=datetime.fromisoformat(
                data.get("last_accessed", datetime.now(UTC).isoformat())
            ),
            total_response_time=data.get("total_response_time", 0.0),
            avg_response_time=data.get("avg_response_time", 0.0),
            error_count=data.get("error_count", 0),
            security_event_count=data.get("security_event_count", 0),
        )


@dataclass
class SessionState:
    """Complete session state for serialization/persistence"""

    session_id: str
    created_at: datetime
    topic: str | None = None
    version: int = 0  # Version for optimistic concurrency control

    # Core data
    threads: dict[str, Thread] = field(default_factory=dict)
    analyses: list[AnalysisRecord] = field(default_factory=list)
    security_events: list[SecurityEvent] = field(default_factory=list)

    # Security
    client_binding: ClientBinding | None = None

    # Metrics
    metrics: SessionMetrics = field(default_factory=SessionMetrics)

    def to_dict(self) -> dict[str, Any]:
        """Convert complete session state to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "topic": self.topic,
            "version": self.version,
            "threads": {
                name: thread.to_dict() for name, thread in self.threads.items()
            },
            "analyses": [analysis.to_dict() for analysis in self.analyses],
            "security_events": [event.to_dict() for event in self.security_events],
            "client_binding": self.client_binding.to_dict()
            if self.client_binding
            else None,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Create session state from dictionary"""
        # Reconstruct threads
        threads = {}
        for name, thread_data in data.get("threads", {}).items():
            threads[name] = Thread.from_dict(thread_data)

        # Reconstruct analyses
        analyses = [
            AnalysisRecord.from_dict(analysis_data)
            for analysis_data in data.get("analyses", [])
        ]

        # Reconstruct security events
        security_events = [
            SecurityEvent.from_dict(event_data)
            for event_data in data.get("security_events", [])
        ]

        # Reconstruct client binding
        client_binding = None
        if data.get("client_binding"):
            client_binding = ClientBinding.from_dict(data["client_binding"])

        # Reconstruct metrics
        metrics = SessionMetrics.from_dict(data.get("metrics", {}))

        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            topic=data.get("topic"),
            version=data.get("version", 0),
            threads=threads,
            analyses=analyses,
            security_events=security_events,
            client_binding=client_binding,
            metrics=metrics,
        )
