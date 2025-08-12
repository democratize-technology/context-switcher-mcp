"""
Shared types and data structures for Context Switcher MCP

This module contains pure data types without dependencies to prevent
circular imports. All shared enums, dataclasses, and type definitions
should be defined here.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any


class ModelBackend(str, Enum):
    """Supported model backends"""

    BEDROCK = "bedrock"
    LITELLM = "litellm"
    OLLAMA = "ollama"


class SessionStatus(str, Enum):
    """Session status states"""

    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class ThreadStatus(str, Enum):
    """Thread execution status"""

    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABSTAINED = "abstained"


class AnalysisType(str, Enum):
    """Types of analysis operations"""

    BROADCAST = "broadcast"
    SYNTHESIS = "synthesis"
    STREAMING = "streaming"
    SINGLE_PERSPECTIVE = "single_perspective"


class ErrorSeverity(str, Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ThreadData:
    """Pure data structure for thread information"""

    id: str
    name: str
    system_prompt: str
    model_backend: ModelBackend
    model_name: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    status: ThreadStatus = ThreadStatus.READY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "model_backend": self.model_backend.value,
            "model_name": self.model_name,
            "conversation_history": self.conversation_history.copy(),
            "status": self.status.value,
        }


@dataclass
class SessionData:
    """Pure data structure for session information"""

    session_id: str
    created_at: datetime
    topic: Optional[str] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 0  # For optimistic locking
    status: SessionStatus = SessionStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "topic": self.topic,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "version": self.version,
            "status": self.status.value,
        }


@dataclass
class AnalysisResult:
    """Result of an analysis operation"""

    session_id: str
    analysis_type: AnalysisType
    prompt: str
    responses: Dict[str, str]
    active_count: int
    abstained_count: int
    failed_count: int = 0
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "session_id": self.session_id,
            "analysis_type": self.analysis_type.value,
            "prompt": self.prompt,
            "responses": self.responses.copy(),
            "active_count": self.active_count,
            "abstained_count": self.abstained_count,
            "failed_count": self.failed_count,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SecurityEventData:
    """Security event information"""

    event_type: str
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "event_type": self.event_type,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "details": self.details.copy(),
        }


@dataclass
class ClientBindingData:
    """Client binding information for session security"""

    session_entropy: str
    creation_timestamp: datetime
    binding_signature: str
    access_pattern_hash: str
    tool_usage_sequence: List[str] = field(default_factory=list)
    validation_failures: int = 0
    last_validated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    security_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "session_entropy": self.session_entropy,
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "binding_signature": self.binding_signature,
            "access_pattern_hash": self.access_pattern_hash,
            "tool_usage_sequence": self.tool_usage_sequence.copy(),
            "validation_failures": self.validation_failures,
            "last_validated": self.last_validated.isoformat(),
            "security_flags": self.security_flags.copy(),
        }


@dataclass
class ConfigurationData:
    """Configuration data structure"""

    max_active_sessions: int = 50
    default_ttl_hours: int = 1
    cleanup_interval_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "max_active_sessions": self.max_active_sessions,
            "default_ttl_hours": self.default_ttl_hours,
            "cleanup_interval_seconds": self.cleanup_interval_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class MetricsData:
    """Performance metrics data"""

    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    success_count: int = 0
    failure_count: int = 0
    total_operations: int = 0

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_operations == 0:
            return 0.0
        return (self.success_count / self.total_operations) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time": self.execution_time,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_operations": self.total_operations,
            "success_rate": self.success_rate,
        }


# Type aliases for complex types
ResponseMap = Dict[str, str]
ThreadMap = Dict[str, ThreadData]
SessionMap = Dict[str, SessionData]
ConfigMap = Dict[str, Any]
SecurityFlags = List[str]
ConversationHistory = List[Dict[str, str]]

# Constants that don't depend on other modules
NO_RESPONSE = "[NO_RESPONSE]"
DEFAULT_PERSPECTIVES = ["technical", "business", "user", "risk"]
MAX_CONVERSATION_HISTORY = 100
DEFAULT_TIMEOUT = 30.0
