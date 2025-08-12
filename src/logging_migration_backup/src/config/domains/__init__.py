"""Configuration domains for Context Switcher MCP

This package contains domain-specific configuration modules that handle
different aspects of the system configuration. Each domain is responsible
for its own validation, defaults, and business logic.

Domains:
- models: LLM backend configuration (Bedrock, LiteLLM, Ollama)
- session: Session management and lifecycle settings
- security: Security, encryption, and access control
- server: MCP server and networking configuration  
- monitoring: Profiling, metrics, and observability

Each domain module exports a configuration class that uses Pydantic
for validation and integrates with the unified configuration system.
"""

from .models import ModelConfig
from .session import SessionConfig
from .security import SecurityConfig 
from .server import ServerConfig
from .monitoring import MonitoringConfig

__all__ = [
    "ModelConfig",
    "SessionConfig", 
    "SecurityConfig",
    "ServerConfig",
    "MonitoringConfig",
]