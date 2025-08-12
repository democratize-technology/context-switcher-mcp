"""Input validation classes for comprehensive content validation"""

import time
from .logging_base import get_logger
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque

from .security_patterns import (
    SecurityPatternMatcher,
    COMPILED_ADVANCED_PATTERNS,
    SECURITY_PATTERNS,
    ADVANCED_PROMPT_INJECTION_PATTERNS,
)
from .input_sanitizer import sanitize_for_llm

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of content validation"""

    is_valid: bool
    cleaned_content: str
    issues: List[str]
    risk_level: str = "low"  # low, medium, high, critical
    blocked_patterns: Optional[List[str]] = None

    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = []


@dataclass
class InjectionAttempt:
    """Details of a detected injection attempt"""

    pattern: str
    location: int
    severity: str
    description: str


class ContentValidator:
    """Advanced content validation with rate limiting and context awareness"""

    def __init__(self):
        # Rate limiting: track validation attempts per content type
        self.validation_attempts = defaultdict(lambda: deque(maxlen=100))
        self.blocked_attempts = defaultdict(int)

    def _check_rate_limit(
        self, content_type: str, client_id: str = "default"
    ) -> Tuple[bool, str]:
        """Check if validation requests are within rate limits"""
        key = f"{client_id}:{content_type}"
        now = time.time()

        # Clean old entries (older than 1 minute)
        attempts = self.validation_attempts[key]
        while attempts and attempts[0] < now - 60:
            attempts.popleft()

        # Check if too many attempts
        if len(attempts) >= 50:  # 50 validation attempts per minute
            return False, "Too many validation requests. Please slow down."

        attempts.append(now)
        return True, ""


class InputValidator:
    """Validates basic input requirements"""

    @staticmethod
    def validate_input_type_and_content(content: str) -> ValidationResult:
        """Validate input type and basic content requirements"""
        if not content or not isinstance(content, str):
            return ValidationResult(
                is_valid=False,
                cleaned_content="",
                issues=["Invalid input type"],
                risk_level="high",
            )
        return ValidationResult(is_valid=True, cleaned_content=content, issues=[])

    @staticmethod
    def validate_length(content: str, max_length: int) -> ValidationResult:
        """Validate content length"""
        if len(content) > max_length:
            return ValidationResult(
                is_valid=False,
                cleaned_content=content[:max_length],
                issues=[f"Input too long (max {max_length} chars)"],
                risk_level="medium",
            )
        return ValidationResult(is_valid=True, cleaned_content=content, issues=[])


class RateLimitValidator:
    """Handles rate limiting for validation requests"""

    def __init__(self, content_validator: ContentValidator):
        self._content_validator = content_validator

    def check_rate_limit(
        self, content_type: str, client_id: str = "default"
    ) -> ValidationResult:
        """Check if validation requests are within rate limits"""
        rate_ok, rate_error = self._content_validator._check_rate_limit(
            content_type, client_id
        )
        if not rate_ok:
            return ValidationResult(
                is_valid=False,
                cleaned_content="",
                issues=[rate_error],
                risk_level="medium",
            )
        return ValidationResult(is_valid=True, cleaned_content="", issues=[])


class SecurityPatternValidator:
    """Validates content against security patterns"""

    @staticmethod
    def validate_security_patterns(content: str) -> Tuple[List[str], List[str], str]:
        """Check for basic security patterns

        Returns:
            Tuple of (issues, blocked_patterns, risk_level)
        """
        issues = []
        blocked_patterns = []
        risk_level = "low"

        pattern_indices = SecurityPatternMatcher.match_basic_security_patterns(content)
        for i in pattern_indices:
            pattern_desc = SecurityPatternMatcher.get_pattern_description(i, "basic")
            issues.append(f"Security pattern detected: {pattern_desc}")
            blocked_patterns.append(SECURITY_PATTERNS[i])
            risk_level = "high"
            logger.warning(f"Blocked malicious input: pattern {i}")

        return issues, blocked_patterns, risk_level


class PromptInjectionValidator:
    """Validates content for advanced prompt injection attempts"""

    @staticmethod
    def validate_prompt_injection(content: str) -> Tuple[List[str], List[str], str]:
        """Check for advanced prompt injection patterns

        Returns:
            Tuple of (issues, blocked_patterns, risk_level)
        """
        issues = []
        blocked_patterns = []
        risk_level = "low"

        injection_attempts = PromptInjectionValidator._detect_advanced_prompt_injection(
            content
        )
        for attempt in injection_attempts:
            issues.append(f"Prompt injection detected: {attempt.description}")
            blocked_patterns.append(attempt.pattern)
            if attempt.severity == "critical":
                risk_level = "critical"
            elif attempt.severity == "high" and risk_level not in ["critical"]:
                risk_level = "high"
            elif attempt.severity == "medium" and risk_level == "low":
                risk_level = "medium"

        return issues, blocked_patterns, risk_level

    @staticmethod
    def _detect_advanced_prompt_injection(text: str) -> List[InjectionAttempt]:
        """
        Detect sophisticated prompt injection attempts

        Args:
            text: Text to analyze

        Returns:
            List of detected injection attempts
        """
        attempts = []

        for i, pattern in enumerate(COMPILED_ADVANCED_PATTERNS):
            matches = pattern.finditer(text)
            for match in matches:
                # Determine severity based on pattern type
                severity = "medium"
                description = "Potential prompt injection"

                pattern_str = ADVANCED_PROMPT_INJECTION_PATTERNS[i]

                if any(
                    keyword in pattern_str.lower()
                    for keyword in ["system", "forget", "override", "jailbreak"]
                ):
                    severity = "critical"
                    description = "Critical prompt injection attempt"
                elif any(
                    keyword in pattern_str.lower()
                    for keyword in [
                        "ignore",
                        "bypass",
                        "unrestricted",
                        "dan",
                        "execute",
                        "switch",
                        "human",
                        "user",
                        "assistant",
                    ]
                ):
                    severity = "high"
                    description = "High-risk prompt injection"

                attempts.append(
                    InjectionAttempt(
                        pattern=pattern_str,
                        location=match.start(),
                        severity=severity,
                        description=description,
                    )
                )

        return attempts


class ContentStructureValidator:
    """Validates content structure and format"""

    def __init__(self, content_validator: ContentValidator):
        self._content_validator = content_validator

    def validate_structure(self, content: str) -> Tuple[List[str], str]:
        """Validate content structure

        Returns:
            Tuple of (issues, risk_level)
        """
        from .input_sanitizer import structure_validator

        structure_ok, structure_issues = structure_validator.validate_content_structure(
            content
        )
        if not structure_ok:
            return structure_issues, "medium"
        return [], "low"


class ValidationOrchestrator:
    """Orchestrates all validation steps"""

    def __init__(self):
        self._content_validator = ContentValidator()
        self.input_validator = InputValidator()
        self.rate_limit_validator = RateLimitValidator(self._content_validator)
        self.security_pattern_validator = SecurityPatternValidator()
        self.prompt_injection_validator = PromptInjectionValidator()
        self.structure_validator = ContentStructureValidator(self._content_validator)

    def validate_content(
        self, content: str, content_type: str, max_length: int, client_id: str
    ) -> ValidationResult:
        """Orchestrate complete content validation"""
        # Step 1: Basic input validation
        result = self.input_validator.validate_input_type_and_content(content)
        if not result.is_valid:
            return result

        # Step 2: Rate limiting
        rate_result = self.rate_limit_validator.check_rate_limit(
            content_type, client_id
        )
        if not rate_result.is_valid:
            return rate_result

        # Step 3: Length validation
        length_result = self.input_validator.validate_length(content, max_length)
        if not length_result.is_valid:
            return length_result

        # Step 4: Security validation
        return self._perform_security_validation(content, content_type)

    def _perform_security_validation(
        self, content: str, content_type: str
    ) -> ValidationResult:
        """Perform comprehensive security validation"""
        issues = []
        blocked_patterns = []
        risk_level = "low"

        # Security pattern validation
        (
            sec_issues,
            sec_patterns,
            sec_risk,
        ) = self.security_pattern_validator.validate_security_patterns(content)
        issues.extend(sec_issues)
        blocked_patterns.extend(sec_patterns)
        risk_level = self._update_risk_level(risk_level, sec_risk)

        # Prompt injection validation
        (
            inj_issues,
            inj_patterns,
            inj_risk,
        ) = self.prompt_injection_validator.validate_prompt_injection(content)
        issues.extend(inj_issues)
        blocked_patterns.extend(inj_patterns)
        risk_level = self._update_risk_level(risk_level, inj_risk)

        # Structure validation
        struct_issues, struct_risk = self.structure_validator.validate_structure(
            content
        )
        issues.extend(struct_issues)
        risk_level = self._update_risk_level(risk_level, struct_risk)

        # Clean content and create result
        cleaned_content = sanitize_for_llm(content)[0]
        is_valid = len(issues) == 0

        # Log security events if needed
        if issues:
            self._log_validation_failure(
                content_type,
                len(issues),
                risk_level,
                len(content),
                len(blocked_patterns),
            )

        return ValidationResult(
            is_valid=is_valid,
            cleaned_content=cleaned_content,
            issues=issues,
            risk_level=risk_level,
            blocked_patterns=blocked_patterns,
        )

    def _log_validation_failure(
        self,
        content_type: str,
        issues_count: int,
        risk_level: str,
        content_length: int,
        blocked_patterns_count: int,
    ):
        """Log security validation failure event"""
        from .security_events import log_security_event

        log_security_event(
            "content_validation_failure",
            {
                "content_type": content_type,
                "issues_count": issues_count,
                "risk_level": risk_level,
                "content_length": content_length,
                "blocked_patterns": blocked_patterns_count,
            },
        )

    @staticmethod
    def _update_risk_level(current_risk: str, new_risk: str) -> str:
        """Update risk level based on hierarchy: critical > high > medium > low"""
        risk_hierarchy = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        current_level = risk_hierarchy.get(current_risk, 0)
        new_level = risk_hierarchy.get(new_risk, 0)

        if new_level > current_level:
            return new_risk
        return current_risk


# Global instances
_validation_orchestrator = ValidationOrchestrator()
_content_validator = _validation_orchestrator._content_validator


# Legacy function wrappers
def detect_advanced_prompt_injection(text: str) -> List[InjectionAttempt]:
    """Legacy wrapper for advanced prompt injection detection"""
    return PromptInjectionValidator._detect_advanced_prompt_injection(text)


def sanitize_user_input(
    text: str, max_length: int = 10000
) -> Tuple[bool, str, List[str]]:
    """Legacy wrapper for basic user input sanitization"""
    result = _validation_orchestrator.validate_content(
        text, "general", max_length, "default"
    )
    return result.is_valid, result.cleaned_content, result.issues
