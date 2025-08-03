"""Memory-efficient circular buffer for metrics storage"""

from typing import Generic, TypeVar, List, Optional
from collections import deque

T = TypeVar("T")


class CircularBuffer(Generic[T]):
    """Thread-safe circular buffer with fixed maximum size

    When the buffer reaches capacity, adding new items automatically
    removes the oldest items to maintain the size limit.
    """

    def __init__(self, maxsize: int):
        """Initialize circular buffer

        Args:
            maxsize: Maximum number of items to store
        """
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")

        self._buffer: deque[T] = deque(maxlen=maxsize)
        self._maxsize = maxsize

    def append(self, item: T) -> None:
        """Add an item to the buffer

        If buffer is at capacity, oldest item is automatically removed.
        """
        self._buffer.append(item)

    def extend(self, items: List[T]) -> None:
        """Add multiple items to the buffer"""
        self._buffer.extend(items)

    def get_recent(self, n: Optional[int] = None) -> List[T]:
        """Get the n most recent items

        Args:
            n: Number of items to return (None for all items)

        Returns:
            List of items in chronological order (oldest first)
        """
        if n is None:
            return list(self._buffer)
        return list(self._buffer)[-n:] if n > 0 else []

    def get_all(self) -> List[T]:
        """Get all items in the buffer"""
        return list(self._buffer)

    def clear(self) -> None:
        """Remove all items from the buffer"""
        self._buffer.clear()

    def __len__(self) -> int:
        """Get the current number of items in the buffer"""
        return len(self._buffer)

    def is_empty(self) -> bool:
        """Check if the buffer is empty"""
        return len(self._buffer) == 0

    def is_full(self) -> bool:
        """Check if the buffer is at capacity"""
        return len(self._buffer) == self._maxsize

    @property
    def maxsize(self) -> int:
        """Get the maximum capacity of the buffer"""
        return self._maxsize

    @property
    def memory_usage_mb(self) -> float:
        """Estimate memory usage in MB (rough approximation)"""
        import sys

        if not self._buffer:
            return 0.0

        # Rough estimate: size of one item * number of items
        sample_size = sys.getsizeof(next(iter(self._buffer)))
        total_size = sample_size * len(self._buffer)
        return total_size / (1024 * 1024)
