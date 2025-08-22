"""
Protocol interfaces for Context Switcher MCP

This module defines abstract interfaces (protocols) that enable dependency
injection and loose coupling between modules. All interface contracts
should be defined here to prevent circular dependencies.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from typing import Any

from .types import (
    AnalysisResult,
    ClientBindingData,
    ConfigurationData,
    ModelBackend,
    ResponseMap,
    SecurityEventData,
    SessionData,
    SessionMap,
    ThreadData,
    ThreadMap,
)


class ConfigurationProvider(ABC):
    """Protocol for configuration providers"""

    @abstractmethod
    def get_session_config(self) -> ConfigurationData:
        """Get session configuration"""
        pass

    @abstractmethod
    def get_backend_config(self, backend: ModelBackend) -> dict[str, Any]:
        """Get backend-specific configuration"""
        pass

    @abstractmethod
    def get_security_config(self) -> dict[str, Any]:
        """Get security configuration"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration completeness and correctness"""
        pass


class ConfigurationMigrator(ABC):
    """Protocol for configuration migration"""

    @abstractmethod
    def migrate_config(self, old_config: dict[str, Any]) -> dict[str, Any]:
        """Migrate old configuration to new format"""
        pass

    @abstractmethod
    def is_migration_needed(self, config: dict[str, Any]) -> bool:
        """Check if migration is needed"""
        pass

    @abstractmethod
    def validate_migration(
        self, old_config: dict[str, Any], new_config: dict[str, Any]
    ) -> bool:
        """Validate that migration was successful"""
        pass


class SessionManagerProtocol(ABC):
    """Protocol for session management"""

    @abstractmethod
    async def add_session(self, session_data: SessionData) -> bool:
        """Add a new session"""
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> SessionData | None:
        """Get session by ID"""
        pass

    @abstractmethod
    async def remove_session(self, session_id: str) -> bool:
        """Remove a session"""
        pass

    @abstractmethod
    async def list_active_sessions(self) -> SessionMap:
        """List all active sessions"""
        pass

    @abstractmethod
    async def record_session_access(self, session_id: str, tool_name: str) -> None:
        """Record session access for behavioral analysis"""
        pass

    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions, return count of cleaned up sessions"""
        pass


class ThreadManagerProtocol(ABC):
    """Protocol for thread management"""

    @abstractmethod
    async def broadcast_message(
        self, threads: ThreadMap, message: str, session_id: str
    ) -> ResponseMap:
        """Broadcast message to all threads"""
        pass

    @abstractmethod
    async def broadcast_message_stream(
        self, threads: ThreadMap, message: str, session_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Broadcast message with streaming responses"""
        pass

    @abstractmethod
    async def execute_single_thread(
        self, thread: ThreadData, message: str, session_id: str
    ) -> str:
        """Execute single thread with message"""
        pass

    @abstractmethod
    def get_thread_metrics(self, last_n: int = 10) -> dict[str, Any]:
        """Get thread-level performance metrics"""
        pass


class PerspectiveOrchestratorProtocol(ABC):
    """Protocol for perspective orchestration"""

    @abstractmethod
    async def broadcast_to_perspectives(
        self, threads: ThreadMap, message: str, session_id: str, topic: str = None
    ) -> ResponseMap:
        """Broadcast message to all perspective threads"""
        pass

    @abstractmethod
    async def broadcast_to_perspectives_stream(
        self, threads: ThreadMap, message: str, session_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Broadcast with streaming responses"""
        pass

    @abstractmethod
    async def synthesize_perspective_responses(
        self, responses: ResponseMap, session_id: str
    ) -> str:
        """Synthesize responses from multiple perspectives"""
        pass

    @abstractmethod
    async def get_perspective_metrics(self, last_n: int = 10) -> dict[str, Any]:
        """Get perspective-level performance metrics"""
        pass


class BackendProviderProtocol(ABC):
    """Protocol for LLM backend providers"""

    @abstractmethod
    async def generate_response(
        self, thread: ThreadData, message: str, session_id: str
    ) -> str:
        """Generate response from LLM backend"""
        pass

    @abstractmethod
    async def generate_response_stream(
        self, thread: ThreadData, message: str, session_id: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM backend"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass

    @abstractmethod
    def get_backend_type(self) -> ModelBackend:
        """Get the backend type"""
        pass


class SecurityManagerProtocol(ABC):
    """Protocol for security management"""

    @abstractmethod
    def create_client_binding(self, session_id: str) -> ClientBindingData:
        """Create secure client binding for session"""
        pass

    @abstractmethod
    def validate_client_binding(
        self, session_id: str, binding_data: ClientBindingData
    ) -> bool:
        """Validate client binding"""
        pass

    @abstractmethod
    def record_security_event(self, event: SecurityEventData) -> None:
        """Record a security event"""
        pass

    @abstractmethod
    def sanitize_error_message(self, message: str) -> str:
        """Sanitize error message for safe display"""
        pass


class ValidationManagerProtocol(ABC):
    """Protocol for input validation"""

    @abstractmethod
    def validate_session_creation(
        self, topic: str, perspectives: list[str] | None
    ) -> tuple[bool, str | None]:
        """Validate session creation parameters"""
        pass

    @abstractmethod
    def validate_message_input(
        self, message: str, session_id: str
    ) -> tuple[bool, str | None]:
        """Validate message input"""
        pass

    @abstractmethod
    def validate_perspective_name(self, perspective: str) -> bool:
        """Validate perspective name"""
        pass


class MetricsManagerProtocol(ABC):
    """Protocol for metrics management"""

    @abstractmethod
    def record_operation_start(
        self, operation_name: str, context: dict[str, Any]
    ) -> str:
        """Record start of operation, return operation ID"""
        pass

    @abstractmethod
    def record_operation_end(
        self, operation_id: str, success: bool, error: str | None = None
    ) -> None:
        """Record end of operation"""
        pass

    @abstractmethod
    def get_metrics_summary(self, last_n: int = 10) -> dict[str, Any]:
        """Get metrics summary"""
        pass

    @abstractmethod
    def get_performance_health(self) -> dict[str, Any]:
        """Get overall performance health status"""
        pass


class CircuitBreakerManagerProtocol(ABC):
    """Protocol for circuit breaker management"""

    @abstractmethod
    def get_circuit_breaker_status(self, backend: ModelBackend) -> dict[str, Any]:
        """Get circuit breaker status for backend"""
        pass

    @abstractmethod
    def record_success(self, backend: ModelBackend) -> None:
        """Record successful operation"""
        pass

    @abstractmethod
    def record_failure(self, backend: ModelBackend, error: Exception) -> None:
        """Record failed operation"""
        pass

    @abstractmethod
    def is_circuit_open(self, backend: ModelBackend) -> bool:
        """Check if circuit is open for backend"""
        pass

    @abstractmethod
    def reset_circuit_breaker(self, backend: ModelBackend) -> bool:
        """Reset circuit breaker to closed state"""
        pass


class ResponseFormatterProtocol(ABC):
    """Protocol for response formatting"""

    @abstractmethod
    async def format_analysis_response(self, result: AnalysisResult) -> dict[str, Any]:
        """Format analysis result into AORP format"""
        pass

    @abstractmethod
    async def synthesize_responses(
        self, responses: ResponseMap, session_id: str
    ) -> str:
        """Synthesize multiple responses into coherent analysis"""
        pass

    @abstractmethod
    def format_error_response(
        self, error_message: str, error_type: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Format error into AORP error response"""
        pass


class LockManagerProtocol(ABC):
    """Protocol for distributed locking"""

    @abstractmethod
    async def acquire_lock(
        self, resource_id: str, timeout: float = 10.0
    ) -> AbstractAsyncContextManager:
        """Acquire lock for resource"""
        pass

    @abstractmethod
    def is_locked(self, resource_id: str) -> bool:
        """Check if resource is locked"""
        pass

    @abstractmethod
    async def release_lock(self, resource_id: str) -> bool:
        """Release lock for resource"""
        pass


class ErrorHandlerProtocol(ABC):
    """Protocol for error handling and logging"""

    @abstractmethod
    def handle_exception(self, error: Exception, context: dict[str, Any]) -> str:
        """Handle exception and return correlation ID"""
        pass

    @abstractmethod
    def log_error_with_context(
        self, error: Exception, operation_name: str, **context
    ) -> str:
        """Log error with context, return correlation ID"""
        pass

    @abstractmethod
    def wrap_generic_exception(
        self, error: Exception, operation: str, target_type: type
    ) -> Exception:
        """Wrap generic exception into specific type"""
        pass


# Protocol for dependency injection container
class ContainerProtocol(ABC):
    """Protocol for dependency injection container"""

    @abstractmethod
    def register_instance(self, interface: type, instance: Any) -> None:
        """Register singleton instance"""
        pass

    @abstractmethod
    def register_factory(self, interface: type, factory: callable) -> None:
        """Register factory function"""
        pass

    @abstractmethod
    def get(self, interface: type) -> Any:
        """Get instance of interface"""
        pass

    @abstractmethod
    def has_registration(self, interface: type) -> bool:
        """Check if interface is registered"""
        pass


# Aggregate protocol for complete session functionality
class SessionServiceProtocol(ABC):
    """High-level protocol combining session management functionality"""

    @abstractmethod
    async def create_session(
        self,
        topic: str,
        perspectives: list[str] | None = None,
        template: str | None = None,
        model_backend: ModelBackend = ModelBackend.BEDROCK,
    ) -> dict[str, Any]:
        """Create complete session with all dependencies"""
        pass

    @abstractmethod
    async def analyze_with_perspectives(
        self, session_id: str, message: str
    ) -> dict[str, Any]:
        """Perform complete perspective analysis"""
        pass

    @abstractmethod
    async def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Get complete session summary"""
        pass
