"""Security monitoring and metrics collection"""

from ..logging_base import get_logger
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from threading import Lock

logger = get_logger(__name__)


@dataclass
class SecurityMetrics:
    """Security metrics data structure"""

    validation_failures: int = 0
    injection_attempts: int = 0
    rate_limit_violations: int = 0
    suspicious_sessions: int = 0
    blocked_requests: int = 0

    # Time-series data (last 24 hours)
    hourly_failures: List[int] = field(default_factory=lambda: [0] * 24)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_validation_failure(self):
        """Record a validation failure"""
        self.validation_failures += 1
        self._update_hourly_metric("failures")

    def add_injection_attempt(self):
        """Record an injection attempt"""
        self.injection_attempts += 1
        self._update_hourly_metric("injection")

    def add_rate_limit_violation(self):
        """Record a rate limit violation"""
        self.rate_limit_violations += 1
        self._update_hourly_metric("rate_limit")

    def add_suspicious_session(self):
        """Record a suspicious session"""
        self.suspicious_sessions += 1

    def add_blocked_request(self):
        """Record a blocked request"""
        self.blocked_requests += 1

    def _update_hourly_metric(self, metric_type: str):
        """Update hourly metrics"""
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        # Shift array if day has changed
        if (now - self.last_updated).days > 0:
            self.hourly_failures = [0] * 24

        self.last_updated = now

    def get_summary(self) -> Dict[str, any]:
        """Get security metrics summary"""
        total_security_events = (
            self.validation_failures
            + self.injection_attempts
            + self.rate_limit_violations
            + self.suspicious_sessions
        )

        return {
            "total_security_events": total_security_events,
            "validation_failures": self.validation_failures,
            "injection_attempts": self.injection_attempts,
            "rate_limit_violations": self.rate_limit_violations,
            "suspicious_sessions": self.suspicious_sessions,
            "blocked_requests": self.blocked_requests,
            "last_updated": self.last_updated.isoformat(),
            "hourly_trend": sum(self.hourly_failures[-6:]),  # Last 6 hours
        }


@dataclass
class ThreatIndicator:
    """Represents a security threat indicator"""

    indicator_type: str  # ip, session, pattern, etc.
    value: str  # The actual indicator value
    threat_level: str  # low, medium, high, critical
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int = 1
    metadata: Dict[str, any] = field(default_factory=dict)

    def update_occurrence(self, metadata: Optional[Dict[str, any]] = None):
        """Update occurrence count and last seen time"""
        self.occurrence_count += 1
        self.last_seen = datetime.now(timezone.utc)
        if metadata:
            self.metadata.update(metadata)


class SecurityMonitor:
    """Real-time security monitoring and threat detection"""

    def __init__(self):
        self.metrics = SecurityMetrics()
        self.threat_indicators: Dict[str, ThreatIndicator] = {}
        self.recent_events = deque(maxlen=1000)  # Keep last 1000 security events
        self.lock = Lock()

        # Pattern-based threat detection
        self.suspicious_patterns = {
            "sql_injection": [
                r"'\s*OR\s*'1'\s*=\s*'1'",
                r"'\s*UNION\s+SELECT",
                r"'\s*;\s*DROP\s+TABLE",
                r"'\s*;\s*DELETE\s+FROM",
            ],
            "xss_injection": [
                r"<script[^>]*>",
                r"javascript\s*:",
                r"on\w+\s*=\s*['\"]",
                r"data\s*:\s*text/html",
            ],
            "command_injection": [
                r";\s*(rm|del|format)\s+",
                r"\|\s*nc\s+",
                r"&&\s*(wget|curl)\s+",
                r">\s*/dev/null",
            ],
            "path_traversal": [
                r"\.\.[\\/]",
                r"[\\/]etc[\\/]passwd",
                r"[\\/]windows[\\/]system32",
                r"file\s*:\s*[\\/][\\/]",
            ],
        }

    def record_security_event(
        self,
        event_type: str,
        details: Dict[str, any],
        threat_level: str = "medium",
        session_id: Optional[str] = None,
    ):
        """Record a security event for monitoring"""
        with self.lock:
            timestamp = datetime.now(timezone.utc)

            # Create event record
            event = {
                "timestamp": timestamp,
                "event_type": event_type,
                "threat_level": threat_level,
                "session_id": session_id,
                "details": details,
            }

            self.recent_events.append(event)

            # Update metrics based on event type
            if event_type == "validation_failure":
                self.metrics.add_validation_failure()
            elif event_type == "injection_attempt":
                self.metrics.add_injection_attempt()
            elif event_type == "rate_limit_exceeded":
                self.metrics.add_rate_limit_violation()
            elif event_type == "suspicious_session":
                self.metrics.add_suspicious_session()
            elif event_type == "blocked_request":
                self.metrics.add_blocked_request()

            # Check for threat indicators
            self._analyze_threat_indicators(event)

            # Log high-severity events
            if threat_level in ["high", "critical"]:
                logger.warning(
                    f"High-severity security event: {event_type} - {details}"
                )

    def _analyze_threat_indicators(self, event: Dict[str, any]):
        """Analyze event for threat indicators"""
        event_type = event["event_type"]
        details = event["details"]
        threat_level = event["threat_level"]

        # Extract potential indicators
        indicators = []

        # Session-based indicators
        if event["session_id"]:
            session_key = f"session:{event['session_id']}"
            indicators.append(("session", event["session_id"], threat_level))

        # Content-based indicators
        if "content" in details or "prompt" in details:
            content = details.get("content", details.get("prompt", ""))
            if content:
                # Check for suspicious patterns
                for pattern_type, patterns in self.suspicious_patterns.items():
                    import re

                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            indicators.append(
                                ("pattern", f"{pattern_type}:{pattern[:50]}", "high")
                            )

        # IP-based indicators (if available)
        if "client_ip" in details:
            indicators.append(("ip", details["client_ip"], threat_level))

        # Update threat indicator database
        for indicator_type, value, level in indicators:
            key = f"{indicator_type}:{value}"

            if key in self.threat_indicators:
                self.threat_indicators[key].update_occurrence(details)
                # Escalate threat level if multiple occurrences
                if self.threat_indicators[key].occurrence_count > 5:
                    self.threat_indicators[key].threat_level = "high"
            else:
                self.threat_indicators[key] = ThreatIndicator(
                    indicator_type=indicator_type,
                    value=value,
                    threat_level=level,
                    first_seen=event["timestamp"],
                    last_seen=event["timestamp"],
                    metadata=details,
                )

    def get_threat_summary(self) -> Dict[str, any]:
        """Get summary of current threats"""
        with self.lock:
            threat_counts = defaultdict(int)
            high_risk_indicators = []

            for indicator in self.threat_indicators.values():
                threat_counts[indicator.threat_level] += 1

                if (
                    indicator.threat_level in ["high", "critical"]
                    or indicator.occurrence_count > 3
                ):
                    high_risk_indicators.append(
                        {
                            "type": indicator.indicator_type,
                            "value": indicator.value[:50] + "..."
                            if len(indicator.value) > 50
                            else indicator.value,
                            "threat_level": indicator.threat_level,
                            "occurrences": indicator.occurrence_count,
                            "first_seen": indicator.first_seen.isoformat(),
                            "last_seen": indicator.last_seen.isoformat(),
                        }
                    )

            return {
                "total_indicators": len(self.threat_indicators),
                "threat_levels": dict(threat_counts),
                "high_risk_indicators": high_risk_indicators[:10],  # Top 10
                "recent_events_count": len(self.recent_events),
                "metrics": self.metrics.get_summary(),
            }

    def get_security_health_score(self) -> Tuple[float, str]:
        """Calculate security health score (0-100)"""
        with self.lock:
            score = 100.0
            issues = []

            # Deduct points for security events
            if self.metrics.injection_attempts > 0:
                score -= min(self.metrics.injection_attempts * 5, 30)
                issues.append(f"{self.metrics.injection_attempts} injection attempts")

            if self.metrics.validation_failures > 10:
                score -= min((self.metrics.validation_failures - 10) * 2, 20)
                issues.append(f"{self.metrics.validation_failures} validation failures")

            if self.metrics.suspicious_sessions > 0:
                score -= min(self.metrics.suspicious_sessions * 10, 25)
                issues.append(f"{self.metrics.suspicious_sessions} suspicious sessions")

            # Check for recent attack patterns
            recent_attacks = len(
                [
                    indicator
                    for indicator in self.threat_indicators.values()
                    if indicator.threat_level in ["high", "critical"]
                    and (
                        datetime.now(timezone.utc) - indicator.last_seen
                    ).total_seconds()
                    < 3600
                ]
            )

            if recent_attacks > 0:
                score -= min(recent_attacks * 15, 40)
                issues.append(f"{recent_attacks} recent high-threat indicators")

            score = max(score, 0)

            # Determine health status
            if score >= 90:
                status = "excellent"
            elif score >= 75:
                status = "good"
            elif score >= 50:
                status = "fair"
            elif score >= 25:
                status = "poor"
            else:
                status = "critical"

            return score, status

    def cleanup_old_indicators(self, max_age_hours: int = 24):
        """Clean up old threat indicators"""
        with self.lock:
            now = datetime.now(timezone.utc)
            cutoff_time = now.timestamp() - (max_age_hours * 3600)

            old_keys = [
                key
                for key, indicator in self.threat_indicators.items()
                if indicator.last_seen.timestamp() < cutoff_time
                and indicator.threat_level in ["low", "medium"]
            ]

            for key in old_keys:
                del self.threat_indicators[key]

            logger.info(f"Cleaned up {len(old_keys)} old threat indicators")

    def export_security_report(self) -> Dict[str, any]:
        """Export comprehensive security report"""
        with self.lock:
            health_score, health_status = self.get_security_health_score()
            threat_summary = self.get_threat_summary()

            # Recent events analysis
            recent_events_by_type = defaultdict(int)
            for event in list(self.recent_events)[-100:]:  # Last 100 events
                recent_events_by_type[event["event_type"]] += 1

            return {
                "report_timestamp": datetime.now(timezone.utc).isoformat(),
                "health_score": health_score,
                "health_status": health_status,
                "threat_summary": threat_summary,
                "recent_events_summary": dict(recent_events_by_type),
                "recommendations": self._generate_security_recommendations(
                    health_score, threat_summary
                ),
            }

    def _generate_security_recommendations(
        self, health_score: float, threat_summary: Dict[str, any]
    ) -> List[str]:
        """Generate security recommendations based on current state"""
        recommendations = []

        if health_score < 50:
            recommendations.append(
                "URGENT: Security health is poor. Investigate recent security events immediately."
            )

        if threat_summary["threat_levels"].get("critical", 0) > 0:
            recommendations.append(
                "Critical threat indicators detected. Review and block malicious sources."
            )

        if threat_summary["metrics"]["injection_attempts"] > 5:
            recommendations.append(
                "Multiple injection attempts detected. Consider implementing additional input validation."
            )

        if threat_summary["metrics"]["rate_limit_violations"] > 10:
            recommendations.append(
                "High rate limit violations. Consider tightening rate limits or blocking sources."
            )

        if threat_summary["metrics"]["suspicious_sessions"] > 0:
            recommendations.append(
                "Suspicious sessions detected. Review session validation and client binding."
            )

        if not recommendations:
            recommendations.append(
                "Security posture is good. Continue monitoring for threats."
            )

        return recommendations


# Global security monitor instance
_security_monitor = SecurityMonitor()


def get_security_monitor() -> SecurityMonitor:
    """Get the global security monitor instance"""
    return _security_monitor


def record_security_event(
    event_type: str,
    details: Dict[str, any],
    threat_level: str = "medium",
    session_id: Optional[str] = None,
):
    """Convenience function to record security events"""
    _security_monitor.record_security_event(
        event_type, details, threat_level, session_id
    )


def get_security_health() -> Tuple[float, str]:
    """Get current security health score and status"""
    return _security_monitor.get_security_health_score()


# Export main functions and classes
__all__ = [
    "SecurityMonitor",
    "SecurityMetrics",
    "ThreatIndicator",
    "get_security_monitor",
    "record_security_event",
    "get_security_health",
]
