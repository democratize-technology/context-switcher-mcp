"""Data models for Context-Switcher MCP"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4


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
    model_name: Optional[str]
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })


@dataclass
class ContextSwitcherSession:
    """Manages a context-switching analysis session"""
    session_id: str
    created_at: datetime
    threads: Dict[str, Thread] = field(default_factory=dict)
    analyses: List[Dict[str, Any]] = field(default_factory=list)
    topic: Optional[str] = None
    
    def add_thread(self, thread: Thread):
        """Add a perspective thread to the session"""
        self.threads[thread.name] = thread
    
    def get_thread(self, name: str) -> Optional[Thread]:
        """Get a thread by name"""
        return self.threads.get(name)
    
    def record_analysis(self, prompt: str, responses: Dict[str, str], 
                       active_count: int, abstained_count: int):
        """Record an analysis for history"""
        self.analyses.append({
            "prompt": prompt,
            "timestamp": datetime.utcnow().isoformat(),
            "responses": responses,
            "active_count": active_count,
            "abstained_count": abstained_count
        })
    
    def get_last_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis"""
        return self.analyses[-1] if self.analyses else None