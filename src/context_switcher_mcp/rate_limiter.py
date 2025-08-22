"""Rate limiting for session operations"""

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

from .logging_base import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting"""

    capacity: int
    tokens: float
    last_refill: float
    refill_rate: float  # tokens per second

    def __post_init__(self):
        if self.last_refill == 0:
            self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket

        Returns:
            True if tokens were consumed, False if rate limited
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class SessionRateLimiter:
    """Rate limiter for session operations"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        analyses_per_minute: int = 10,
        session_creation_per_minute: int = 5,
    ):
        """Initialize rate limiter

        Args:
            requests_per_minute: General request limit per session
            analyses_per_minute: Analysis requests per session
            session_creation_per_minute: Global session creation limit
        """
        self.requests_per_minute = requests_per_minute
        self.analyses_per_minute = analyses_per_minute
        self.session_creation_per_minute = session_creation_per_minute

        # Per-session buckets
        self.session_buckets: dict[str, dict[str, RateLimitBucket]] = {}

        # Global session creation bucket
        self.session_creation_bucket = RateLimitBucket(
            capacity=session_creation_per_minute,
            tokens=session_creation_per_minute,
            last_refill=0,
            refill_rate=session_creation_per_minute / 60.0,
        )

        self._lock = Lock()

    def check_session_creation(self) -> tuple[bool, str]:
        """Check if session creation is allowed

        Returns:
            Tuple of (is_allowed, error_message)
        """
        with self._lock:
            if self.session_creation_bucket.consume(1):
                return True, ""

            # Calculate time until next token
            time_until_next = 60.0 / self.session_creation_per_minute

            logger.warning("Session creation rate limited")
            return (
                False,
                f"Session creation rate limited. Try again in {time_until_next:.1f} seconds.",
            )

    def check_request(
        self, session_id: str, operation_type: str = "request"
    ) -> tuple[bool, str]:
        """Check if request is allowed for session

        Args:
            session_id: Session identifier
            operation_type: Type of operation (request, analysis)

        Returns:
            Tuple of (is_allowed, error_message)
        """
        with self._lock:
            if session_id not in self.session_buckets:
                self._init_session_buckets(session_id)

            buckets = self.session_buckets[session_id]

            if operation_type == "analysis":
                bucket = buckets["analysis"]
                limit_name = "analysis"
                limit_value = self.analyses_per_minute
            else:
                bucket = buckets["request"]
                limit_name = "request"
                limit_value = self.requests_per_minute

            if bucket.consume(1):
                return True, ""

            # Calculate time until next token
            time_until_next = 60.0 / limit_value

            logger.warning(f"Session {session_id} {operation_type} rate limited")
            return (
                False,
                f"{limit_name.title()} rate limited for session. Try again in {time_until_next:.1f} seconds.",
            )

    def _init_session_buckets(self, session_id: str):
        """Initialize rate limit buckets for a session"""
        self.session_buckets[session_id] = {
            "request": RateLimitBucket(
                capacity=self.requests_per_minute,
                tokens=self.requests_per_minute,
                last_refill=0,
                refill_rate=self.requests_per_minute / 60.0,
            ),
            "analysis": RateLimitBucket(
                capacity=self.analyses_per_minute,
                tokens=self.analyses_per_minute,
                last_refill=0,
                refill_rate=self.analyses_per_minute / 60.0,
            ),
        }

    def cleanup_session(self, session_id: str):
        """Remove rate limit state for a session"""
        with self._lock:
            if session_id in self.session_buckets:
                del self.session_buckets[session_id]
                logger.debug(f"Cleaned up rate limit state for session {session_id}")

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics"""
        with self._lock:
            active_sessions = len(self.session_buckets)

            # Calculate average tokens remaining
            total_request_tokens = 0
            total_analysis_tokens = 0

            for buckets in self.session_buckets.values():
                buckets["request"]._refill()
                buckets["analysis"]._refill()
                total_request_tokens += int(buckets["request"].tokens)
                total_analysis_tokens += int(buckets["analysis"].tokens)

            avg_request_tokens = (
                total_request_tokens / active_sessions if active_sessions > 0 else 0
            )
            avg_analysis_tokens = (
                total_analysis_tokens / active_sessions if active_sessions > 0 else 0
            )

            # Refill global bucket for accurate stats
            self.session_creation_bucket._refill()

            return {
                "active_sessions": active_sessions,
                "limits": {
                    "requests_per_minute": self.requests_per_minute,
                    "analyses_per_minute": self.analyses_per_minute,
                    "session_creation_per_minute": self.session_creation_per_minute,
                },
                "average_tokens_remaining": {
                    "requests": round(avg_request_tokens, 1),
                    "analyses": round(avg_analysis_tokens, 1),
                },
                "session_creation_tokens": round(
                    self.session_creation_bucket.tokens, 1
                ),
            }
