# Import Dependency Refactoring Plan
## Context Switcher MCP - Circular Import Resolution

### Executive Summary

The dependency analysis revealed **1 critical circular dependency** and several architectural issues that need systematic resolution. This plan outlines a clean, layered architecture that eliminates circular dependencies while maintaining functionality.

### Current State Analysis

#### Critical Issues Found
1. **HIGH severity circular dependency**: `config` ↔ `config_migration`
2. **Heavy coupling**: 18 modules depend on `models.py`
3. **Late imports**: 23 modules have late imports indicating dependency issues
4. **Function-level imports**: Used as workarounds for circular dependencies

#### Dependency Metrics
- **Total modules**: 69
- **Total dependencies**: 160
- **Average dependencies per module**: 2.32
- **Most depended-upon modules**: models (18), security (16), exceptions (15)

### Architecture Design Principles

#### 1. Layered Architecture
```
┌─────────────────────────────────────┐
│           API/Tools Layer           │  (MCP tools, handlers)
├─────────────────────────────────────┤
│        Business Logic Layer        │  (orchestrators, managers)
├─────────────────────────────────────┤
│         Core Domain Layer          │  (models, protocols)
├─────────────────────────────────────┤
│       Infrastructure Layer         │  (config, backends, security)
└─────────────────────────────────────┘
```

#### 2. Dependency Direction Rules
- **Upward dependencies only**: Lower layers never import from higher layers
- **Interface contracts**: Use protocols for cross-layer communication
- **Dependency injection**: Inject dependencies rather than importing directly

#### 3. Module Organization
- **Types module**: Shared data types and enums
- **Protocols module**: Interface contracts
- **Core modules**: Domain logic with minimal dependencies
- **Infrastructure**: Config, backends, external integrations

### Detailed Refactoring Plan

#### Phase 1: Create Foundation Modules

##### 1.1 Create Types Module (`types.py`)
Extract all shared types from `models.py`:
```python
# src/context_switcher_mcp/types.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

class ModelBackend(str, Enum):
    BEDROCK = "bedrock"
    LITELLM = "litellm"
    OLLAMA = "ollama"

@dataclass
class ThreadData:
    """Pure data structure for thread information"""
    id: str
    name: str
    system_prompt: str
    model_backend: ModelBackend
    model_name: Optional[str]
    conversation_history: List[Dict[str, str]]

@dataclass
class SessionData:
    """Pure data structure for session information"""
    session_id: str
    created_at: datetime
    topic: Optional[str]
    access_count: int
    last_accessed: datetime
    version: int
```

##### 1.2 Create Protocols Module (`protocols.py`)
Define interface contracts:
```python
# src/context_switcher_mcp/protocols.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator
from .types import ThreadData, SessionData, ModelBackend

class SessionManagerProtocol(ABC):
    """Protocol for session management"""

    @abstractmethod
    async def add_session(self, session: SessionData) -> bool:
        """Add a new session"""
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID"""
        pass

class ThreadManagerProtocol(ABC):
    """Protocol for thread management"""

    @abstractmethod
    async def broadcast_message(self, threads: Dict[str, ThreadData], message: str) -> Dict[str, str]:
        """Broadcast message to threads"""
        pass

class ConfigProtocol(ABC):
    """Protocol for configuration access"""

    @abstractmethod
    def get_session_config(self) -> Dict[str, Any]:
        """Get session configuration"""
        pass

    @abstractmethod
    def get_backend_config(self, backend: ModelBackend) -> Dict[str, Any]:
        """Get backend configuration"""
        pass
```

##### 1.3 Create Dependency Injection Container (`container.py`)
```python
# src/context_switcher_mcp/container.py
from typing import TypeVar, Type, Dict, Any, Optional
from .protocols import SessionManagerProtocol, ThreadManagerProtocol, ConfigProtocol

T = TypeVar('T')

class DependencyContainer:
    """Simple dependency injection container"""

    def __init__(self):
        self._instances: Dict[Type, Any] = {}
        self._factories: Dict[Type, callable] = {}

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register a singleton instance"""
        self._instances[interface] = instance

    def register_factory(self, interface: Type[T], factory: callable) -> None:
        """Register a factory function"""
        self._factories[interface] = factory

    def get(self, interface: Type[T]) -> T:
        """Get instance of interface"""
        if interface in self._instances:
            return self._instances[interface]

        if interface in self._factories:
            instance = self._factories[interface]()
            self._instances[interface] = instance
            return instance

        raise ValueError(f"No registration found for {interface}")

# Global container instance
container = DependencyContainer()
```

#### Phase 2: Fix Critical Circular Dependency

##### 2.1 Create Config Base Module (`config_base.py`)
Extract common configuration interfaces:
```python
# src/context_switcher_mcp/config_base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from .types import ModelBackend

class ConfigurationProvider(ABC):
    """Base interface for configuration providers"""

    @abstractmethod
    def get_session_config(self) -> Dict[str, Any]:
        """Get session configuration"""
        pass

    @abstractmethod
    def get_backend_config(self, backend: ModelBackend) -> Dict[str, Any]:
        """Get backend-specific configuration"""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration"""
        pass

class ConfigurationMigrator(ABC):
    """Interface for configuration migration"""

    @abstractmethod
    def migrate_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old configuration to new format"""
        pass

    @abstractmethod
    def is_migration_needed(self, config: Dict[str, Any]) -> bool:
        """Check if migration is needed"""
        pass
```

##### 2.2 Refactor Config Module
```python
# src/context_switcher_mcp/config.py (refactored)
import os
import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any

from .config_base import ConfigurationProvider
from .types import ModelBackend

# Remove circular import - use dependency injection instead
# from .config_migration import CompatibilityAdapter  # REMOVED

logger = logging.getLogger(__name__)

@dataclass
class SessionConfig:
    """Session-specific configuration"""
    max_active_sessions: int = 50
    default_ttl_hours: int = 1
    cleanup_interval_seconds: int = 300

@dataclass
class ContextSwitcherConfig(ConfigurationProvider):
    """Main configuration class implementing provider interface"""

    session: SessionConfig

    def __init__(self):
        self.session = SessionConfig()
        self._backend_configs = {}

    def get_session_config(self) -> Dict[str, Any]:
        """Get session configuration"""
        return {
            'max_active_sessions': self.session.max_active_sessions,
            'default_ttl_hours': self.session.default_ttl_hours,
            'cleanup_interval_seconds': self.session.cleanup_interval_seconds
        }

    def get_backend_config(self, backend: ModelBackend) -> Dict[str, Any]:
        """Get backend-specific configuration"""
        return self._backend_configs.get(backend, {})

    def validate(self) -> bool:
        """Validate configuration"""
        return (
            self.session.max_active_sessions > 0 and
            self.session.default_ttl_hours > 0 and
            self.session.cleanup_interval_seconds > 0
        )

# Factory function to create config with migration
def create_config_with_migration() -> ContextSwitcherConfig:
    """Create configuration with migration support"""
    from .container import container
    from .config_base import ConfigurationMigrator

    config = ContextSwitcherConfig()

    # Use dependency injection to get migrator if available
    try:
        migrator = container.get(ConfigurationMigrator)
        # Apply migration logic here
    except ValueError:
        # No migrator registered, use default config
        pass

    return config

# Global config instance
_config_instance: Optional[ContextSwitcherConfig] = None

def get_config() -> ContextSwitcherConfig:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = create_config_with_migration()
    return _config_instance
```

##### 2.3 Refactor Config Migration Module
```python
# src/context_switcher_mcp/config_migration.py (refactored)
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

from .config_base import ConfigurationMigrator
from .validated_config import (
    ConfigurationError,
    ValidatedContextSwitcherConfig,
    load_validated_config,
)

logger = logging.getLogger(__name__)

class CompatibilityAdapter(ConfigurationMigrator):
    """Adapter for migrating legacy configuration"""

    def migrate_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old configuration to new format"""
        # Migration logic here
        migrated = {}

        # Migrate session config
        if 'session' in old_config:
            migrated['session'] = old_config['session']

        return migrated

    def is_migration_needed(self, config: Dict[str, Any]) -> bool:
        """Check if migration is needed"""
        # Check for legacy format indicators
        return 'legacy_format' in config

def create_validated_config_with_fallback() -> ValidatedContextSwitcherConfig:
    """Create validated config with fallback to legacy"""
    try:
        return load_validated_config()
    except ConfigurationError:
        # Fallback to legacy configuration
        # Import here to avoid circular dependency
        from .container import container
        from .config_base import ConfigurationProvider

        legacy_provider = container.get(ConfigurationProvider)
        # Convert legacy to validated format
        return _convert_legacy_to_validated(legacy_provider)

def _convert_legacy_to_validated(provider: ConfigurationProvider) -> ValidatedContextSwitcherConfig:
    """Convert legacy configuration to validated format"""
    session_config = provider.get_session_config()
    # Conversion logic here
    return ValidatedContextSwitcherConfig(**session_config)
```

#### Phase 3: Refactor Models and Dependencies

##### 3.1 Slim Down Models Module
```python
# src/context_switcher_mcp/models.py (refactored)
"""Data models with reduced dependencies"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import hashlib
import secrets

from .types import ModelBackend, ThreadData, SessionData

@dataclass
class Thread:
    """Thread with behavior methods"""
    data: ThreadData

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.data.conversation_history.append(message)

@dataclass
class ContextSwitcherSession:
    """Session with behavior methods"""
    data: SessionData
    threads: Dict[str, Thread] = field(default_factory=dict)
    client_binding: Optional['ClientBinding'] = None
    analyses: List[Dict[str, Any]] = field(default_factory=list)
    security_events: List[Dict[str, Any]] = field(default_factory=list)

    def add_thread(self, thread: Thread) -> None:
        """Add a perspective thread to the session"""
        self.threads[thread.data.name] = thread

    def get_thread(self, name: str) -> Optional[Thread]:
        """Get a thread by name"""
        return self.threads.get(name)

    async def record_access(self, tool_name: str) -> None:
        """Record session access for behavioral analysis"""
        # Use dependency injection for lock manager
        from .container import container
        from .protocols import SessionManagerProtocol

        try:
            session_manager = container.get(SessionManagerProtocol)
            # Delegate to session manager
            await session_manager.record_session_access(self.data.session_id, tool_name)
        except ValueError:
            # Fallback to direct update
            self.data.access_count += 1
            self.data.last_accessed = datetime.now(timezone.utc)
            self.data.version += 1

# Keep ClientBinding as is since it has specific security logic
@dataclass
class ClientBinding:
    """Secure client binding data for session validation"""
    # ... existing implementation
```

#### Phase 4: Update Session Manager

##### 4.1 Refactor Session Manager with Dependency Injection
```python
# src/context_switcher_mcp/session_manager.py (refactored)
"""Session management with dependency injection"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any, Tuple

from .types import SessionData
from .models import ContextSwitcherSession
from .protocols import SessionManagerProtocol, ConfigProtocol
from .container import container
from .exceptions import (
    SessionCleanupError,
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    ConcurrencyError,
    LockTimeoutError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)

class SessionManager(SessionManagerProtocol):
    """Session manager with injected dependencies"""

    def __init__(self, config_provider: Optional[ConfigProtocol] = None):
        self.sessions: Dict[str, ContextSwitcherSession] = {}

        # Use dependency injection for configuration
        if config_provider is None:
            try:
                config_provider = container.get(ConfigProtocol)
            except ValueError:
                # Fallback to default config
                from .config import get_config
                config_provider = get_config()

        self.config_provider = config_provider
        session_config = config_provider.get_session_config()

        self.max_sessions = session_config.get('max_active_sessions', 50)
        self.session_ttl = timedelta(hours=session_config.get('default_ttl_hours', 1))
        self.cleanup_interval = timedelta(seconds=session_config.get('cleanup_interval_seconds', 300))

        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ... rest of implementation remains similar but uses injected dependencies
```

#### Phase 5: Update Import Statements

##### 5.1 Update Main Module
```python
# src/context_switcher_mcp/__init__.py (updated imports)
#!/usr/bin/env python3
"""Context-Switcher MCP Server with clean architecture"""

import logging
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Use dependency injection
from .container import container
from .protocols import SessionManagerProtocol, ConfigProtocol
from .types import ModelBackend
from .config import get_config, ContextSwitcherConfig

# Register dependencies
def setup_dependencies():
    """Setup dependency injection container"""
    config = get_config()
    container.register_instance(ConfigProtocol, config)

    from .session_manager import SessionManager
    session_manager = SessionManager(config)
    container.register_instance(SessionManagerProtocol, session_manager)

# Initialize
setup_dependencies()
config = container.get(ConfigProtocol)
session_manager = container.get(SessionManagerProtocol)

# ... rest of module
```

### Implementation Strategy

#### Step 1: Create Foundation (Day 1)
1. Create `types.py` module
2. Create `protocols.py` module
3. Create `container.py` module
4. Create `config_base.py` module

#### Step 2: Fix Circular Dependency (Day 1)
1. Refactor `config.py` to remove circular import
2. Refactor `config_migration.py` to use interfaces
3. Test configuration loading

#### Step 3: Refactor Core Models (Day 2)
1. Extract types from `models.py`
2. Update `models.py` to use dependency injection
3. Update `session_manager.py` to use protocols

#### Step 4: Update Consumers (Day 2-3)
1. Update all modules importing from old locations
2. Use dependency injection where appropriate
3. Update handlers and tools

#### Step 5: Testing and Validation (Day 3)
1. Run full test suite
2. Verify no circular dependencies
3. Performance testing
4. Documentation updates

### Benefits of This Architecture

#### 1. Eliminates Circular Dependencies
- Clean layer separation prevents circular imports
- Interface-based design enables loose coupling
- Dependency injection removes direct import dependencies

#### 2. Improves Maintainability
- Clear separation of concerns
- Testable components through dependency injection
- Protocol-based contracts ensure consistent interfaces

#### 3. Enhances Scalability
- Easy to add new implementations
- Modular design supports feature additions
- Configurable dependency resolution

#### 4. Better Testing
- Mock dependencies easily with protocols
- Isolated unit testing
- Integration testing with dependency injection

### Testing Strategy

#### Unit Tests
```python
# Example: Testing with dependency injection
def test_session_manager():
    # Create mock config
    mock_config = Mock(spec=ConfigProtocol)
    mock_config.get_session_config.return_value = {'max_active_sessions': 10}

    # Test with injected dependency
    session_manager = SessionManager(mock_config)
    assert session_manager.max_sessions == 10
```

#### Integration Tests
```python
# Example: Testing complete flow
def test_complete_session_flow():
    # Setup test container
    test_container = DependencyContainer()
    test_container.register_instance(ConfigProtocol, test_config)

    # Test with real dependencies
    session_manager = test_container.get(SessionManagerProtocol)
    # ... test operations
```

### Migration Checklist

- [ ] Create foundation modules (`types.py`, `protocols.py`, `container.py`)
- [ ] Create `config_base.py` with interfaces
- [ ] Refactor `config.py` to remove circular import
- [ ] Refactor `config_migration.py` to use interfaces
- [ ] Extract types from `models.py` to `types.py`
- [ ] Update `models.py` to use dependency injection
- [ ] Update `session_manager.py` to use protocols
- [ ] Update all importing modules
- [ ] Run dependency analyzer to verify no circular imports
- [ ] Run full test suite
- [ ] Update documentation

### Monitoring and Validation

#### Dependency Health Metrics
- Number of circular dependencies (target: 0)
- Average dependencies per module (target: < 3)
- Depth of dependency chains (target: < 5)
- Late import count (target: < 5)

#### Tools for Ongoing Monitoring
```bash
# Run dependency analysis regularly
python src/context_switcher_mcp/security/dependency_analyzer.py

# Check for circular imports
python -c "import context_switcher_mcp; print('Import successful')"

# Validate architecture
pytest tests/test_architecture_compliance.py
```

This refactoring plan provides a systematic approach to eliminating circular dependencies while establishing a clean, maintainable architecture for the Context Switcher MCP codebase.
