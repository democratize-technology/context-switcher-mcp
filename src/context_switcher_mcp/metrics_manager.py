"""Metrics management for thread orchestration operations"""

import asyncio
from .logging_base import get_logger
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from .circular_buffer import CircularBuffer
from .config import get_config

logger = get_logger(__name__)


@dataclass
class ThreadMetrics:
    """Metrics for thread execution tracking"""

    thread_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    model_backend: Optional[str] = None
    retry_count: int = 0

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time in seconds"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class ThreadOrchestrationMetrics:
    """Aggregate metrics for thread orchestration operations"""

    session_id: str
    operation_type: str  # 'broadcast', 'synthesis', 'single_thread'
    start_time: float
    end_time: Optional[float] = None
    thread_metrics: Dict[str, ThreadMetrics] = field(default_factory=dict)
    total_threads: int = 0
    successful_threads: int = 0
    failed_threads: int = 0
    abstained_threads: int = 0

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate total operation execution time"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_threads == 0:
            return 0.0
        return (self.successful_threads / self.total_threads) * 100


class MetricsManager:
    """Manages performance metrics collection and reporting"""

    def __init__(self):
        """Initialize metrics manager with configuration"""
        config = get_config()

        # Metrics storage with circular buffer to prevent memory leaks
        self.metrics_history = CircularBuffer[ThreadOrchestrationMetrics](
            config.monitoring.metrics.max_history_size
        )
        self.metrics_lock = asyncio.Lock()  # Protect metrics operations

    def create_orchestration_metrics(
        self, session_id: str, operation_type: str, thread_count: int
    ) -> ThreadOrchestrationMetrics:
        """Create new orchestration metrics tracking object"""
        return ThreadOrchestrationMetrics(
            session_id=session_id,
            operation_type=operation_type,
            start_time=time.time(),
            total_threads=thread_count,
        )

    def create_thread_metrics(
        self, thread_name: str, model_backend: str
    ) -> ThreadMetrics:
        """Create new thread metrics tracking object"""
        return ThreadMetrics(
            thread_name=thread_name,
            start_time=time.time(),
            model_backend=model_backend,
        )

    async def store_metrics(self, metrics: ThreadOrchestrationMetrics) -> None:
        """Store metrics in circular buffer atomically"""
        async with self.metrics_lock:
            self._store_metrics_unsafe(metrics)

    def _store_metrics_unsafe(self, metrics: ThreadOrchestrationMetrics) -> None:
        """Store metrics in circular buffer (automatically maintains size limit)

        Note: This method is not thread-safe and should only be called
        while holding the metrics_lock.
        """
        self.metrics_history.append(metrics)

    def finalize_metrics(self, metrics: ThreadOrchestrationMetrics) -> None:
        """Finalize metrics by setting end time"""
        metrics.end_time = time.time()

    def log_performance_summary(self, metrics: ThreadOrchestrationMetrics) -> None:
        """Log performance summary for completed operation"""
        if metrics.execution_time is None:
            logger.warning("Cannot log performance summary: metrics not finalized")
            return

        logger.info(
            f"Thread {metrics.operation_type} completed: {metrics.execution_time:.2f}s, "
            f"Success: {metrics.successful_threads}/{metrics.total_threads}, "
            f"Rate: {metrics.success_rate:.1f}%"
        )

    async def get_performance_metrics(self, last_n: int = 10) -> Dict[str, Any]:
        """Get thread-level performance metrics"""
        recent_metrics = await self._get_recent_metrics(last_n)
        if not recent_metrics:
            return {"message": "No thread metrics available"}

        # Calculate aggregate statistics using helper methods
        summary = self._calculate_thread_summary(recent_metrics)
        backend_performance = self._calculate_backend_performance(recent_metrics)
        storage_info = self._get_metrics_storage_info()

        return {
            "thread_summary": summary,
            "backend_performance": backend_performance,
            "metrics_storage": storage_info,
        }

    async def _get_recent_metrics(self, last_n: int):
        """Get recent metrics with proper locking"""
        async with self.metrics_lock:
            if self.metrics_history.is_empty():
                return None
            return self.metrics_history.get_recent(last_n)

    def _calculate_thread_summary(self, recent_metrics) -> Dict[str, Any]:
        """Calculate aggregate thread performance summary"""
        total_operations = len(recent_metrics)

        avg_execution_time = (
            sum(m.execution_time for m in recent_metrics if m.execution_time)
            / total_operations
        )
        overall_success_rate = (
            sum(m.success_rate for m in recent_metrics) / total_operations
        )

        return {
            "total_operations": total_operations,
            "avg_execution_time_seconds": round(avg_execution_time, 2),
            "overall_success_rate_percent": round(overall_success_rate, 1),
        }

    def _calculate_backend_performance(self, recent_metrics) -> Dict[str, Any]:
        """Calculate performance breakdown by backend"""
        backend_stats = {}

        # Collect raw backend statistics
        self._collect_backend_stats(recent_metrics, backend_stats)

        # Calculate derived metrics
        self._calculate_backend_derived_metrics(backend_stats)

        return backend_stats

    def _collect_backend_stats(self, recent_metrics, backend_stats):
        """Collect raw statistics for each backend"""
        for metrics in recent_metrics:
            for thread_name, thread_metrics in metrics.thread_metrics.items():
                backend = thread_metrics.model_backend
                if backend not in backend_stats:
                    backend_stats[backend] = {
                        "count": 0,
                        "success": 0,
                        "total_time": 0.0,
                    }

                self._update_backend_stats(backend_stats[backend], thread_metrics)

    def _update_backend_stats(self, stats, thread_metrics):
        """Update statistics for a specific backend"""
        stats["count"] += 1
        if thread_metrics.success:
            stats["success"] += 1
        if thread_metrics.execution_time:
            stats["total_time"] += thread_metrics.execution_time

    def _calculate_backend_derived_metrics(self, backend_stats):
        """Calculate success rates and average times for backends"""
        for backend, stats in backend_stats.items():
            stats["success_rate"] = (
                (stats["success"] / stats["count"]) * 100 if stats["count"] > 0 else 0
            )
            stats["avg_time"] = (
                stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            )

    def _get_metrics_storage_info(self) -> Dict[str, Any]:
        """Get metrics storage utilization information"""
        return {
            "current_size": len(self.metrics_history),
            "max_size": self.metrics_history.maxsize,
            "utilization_percent": round(
                (len(self.metrics_history) / self.metrics_history.maxsize) * 100, 1
            ),
            "memory_usage_mb": round(self.metrics_history.memory_usage_mb, 2),
        }

    def get_storage_info(self) -> Dict[str, Any]:
        """Get metrics storage utilization information"""
        return {
            "current_size": len(self.metrics_history),
            "max_size": self.metrics_history.maxsize,
            "utilization_percent": round(
                (len(self.metrics_history) / self.metrics_history.maxsize) * 100, 1
            ),
            "memory_usage_mb": round(self.metrics_history.memory_usage_mb, 2),
            "is_empty": self.metrics_history.is_empty(),
        }
