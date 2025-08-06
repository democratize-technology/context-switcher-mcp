"""Pure data models for session management"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from .models import Thread


@dataclass
class AnalysisRecord:
    """Represents a single analysis performed in a session"""

    prompt: str
    timestamp: datetime
    responses: Dict[str, str]
    active_count: int
    abstained_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis record to dictionary"""
        return {
            "prompt": self.prompt,
            "timestamp": self.timestamp.isoformat(),
            "responses": self.responses,
            "active_count": self.active_count,
            "abstained_count": self.abstained_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisRecord":
        """Create analysis record from dictionary"""
        return cls(
            prompt=data["prompt"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            responses=data["responses"],
            active_count=data["active_count"],
            abstained_count=data["abstained_count"],
        )


@dataclass
class SessionData:
    """Pure data model for session information"""

    session_id: str
    created_at: datetime
    topic: Optional[str] = None
    threads: Dict[str, Thread] = field(default_factory=dict)
    analyses: List[AnalysisRecord] = field(default_factory=list)

    def add_thread(self, thread: Thread) -> None:
        """Add a perspective thread to the session"""
        self.threads[thread.name] = thread

    def get_thread(self, name: str) -> Optional[Thread]:
        """Get a thread by name"""
        return self.threads.get(name)

    def record_analysis(
        self,
        prompt: str,
        responses: Dict[str, str],
        active_count: int,
        abstained_count: int,
    ) -> None:
        """Record an analysis for history"""
        analysis = AnalysisRecord(
            prompt=prompt,
            timestamp=datetime.now(timezone.utc),
            responses=responses,
            active_count=active_count,
            abstained_count=abstained_count,
        )
        self.analyses.append(analysis)

    def get_last_analysis(self) -> Optional[AnalysisRecord]:
        """Get the most recent analysis"""
        return self.analyses[-1] if self.analyses else None

    def get_thread_count(self) -> int:
        """Get the number of threads in this session"""
        return len(self.threads)

    def get_analysis_count(self) -> int:
        """Get the number of analyses performed"""
        return len(self.analyses)

    def get_thread_names(self) -> List[str]:
        """Get list of thread names"""
        return list(self.threads.keys())

    def remove_thread(self, name: str) -> bool:
        """Remove a thread by name

        Returns:
            True if thread was removed, False if not found
        """
        if name in self.threads:
            del self.threads[name]
            return True
        return False

    def clear_analyses(self) -> None:
        """Clear all analysis history"""
        self.analyses.clear()

    def get_analyses_summary(self) -> Dict[str, Any]:
        """Get summary of analyses"""
        if not self.analyses:
            return {"count": 0, "message": "No analyses recorded"}

        recent_analysis = self.analyses[-1]
        total_responses = sum(len(analysis.responses) for analysis in self.analyses)

        return {
            "count": len(self.analyses),
            "total_responses": total_responses,
            "last_analysis": {
                "timestamp": recent_analysis.timestamp.isoformat(),
                "active_count": recent_analysis.active_count,
                "abstained_count": recent_analysis.abstained_count,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert session data to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "topic": self.topic,
            "threads": {
                name: {
                    "id": thread.id,
                    "name": thread.name,
                    "system_prompt": thread.system_prompt,
                    "model_backend": thread.model_backend.value,
                    "model_name": thread.model_name,
                    "conversation_history": thread.conversation_history,
                }
                for name, thread in self.threads.items()
            },
            "analyses": [analysis.to_dict() for analysis in self.analyses],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        """Create session data from dictionary"""
        from .models import Thread, ModelBackend

        # Reconstruct threads
        threads = {}
        for name, thread_data in data.get("threads", {}).items():
            thread = Thread(
                id=thread_data["id"],
                name=thread_data["name"],
                system_prompt=thread_data["system_prompt"],
                model_backend=ModelBackend(thread_data["model_backend"]),
                model_name=thread_data.get("model_name"),
                conversation_history=thread_data.get("conversation_history", []),
            )
            threads[name] = thread

        # Reconstruct analyses
        analyses = [
            AnalysisRecord.from_dict(analysis_data)
            for analysis_data in data.get("analyses", [])
        ]

        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            topic=data.get("topic"),
            threads=threads,
            analyses=analyses,
        )
