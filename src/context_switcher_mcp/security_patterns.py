"""Security pattern definitions and matching utilities"""

import re

# Enhanced security patterns to detect and block
SECURITY_PATTERNS = [
    # Script injection patterns
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    # System command injection patterns
    r"\$\([^)]*\)",
    r"`[^`]*`",
    r";\s*(rm|del|format|sudo|su)\s",
    r"\|\s*(curl|wget|nc|netcat)\s",
    # Path traversal patterns
    r"\.\./+",
    r"\.\.\\+",
    # SQL injection patterns (basic)
    r";\s*(drop|delete|insert|update|select)\s+",
    r"union\s+select",
    r"or\s+1\s*=\s*1",
    # Basic prompt injection patterns
    r"ignore\s+(previous|all|above)\s+(instructions|prompts?)",
    r"system\s*:\s*you\s+are\s+now",
    r"new\s+instructions?\s*:",
    r"act\s+as\s+(if\s+you\s+are\s+)?a\s+different",
]

# Advanced prompt injection patterns
ADVANCED_PROMPT_INJECTION_PATTERNS = [
    # Role manipulation
    r"you\s+are\s+now\s+(a|an)?\s*different\s+(ai|assistant|bot|system)",
    r"forget\s+(your|all|previous)\s+(instructions|training|guidelines)",
    r"override\s+(your|all|previous)\s+(instructions|training|guidelines)",
    r"switch\s+to\s+(unrestricted|developer|debug|admin)\s+mode",
    r"activate\s+(developer|debug|admin|unrestricted)\s+mode",
    # System prompt extraction
    r"show\s+me\s+your\s+(system|initial)\s+prompt",
    r"what\s+(are|were)\s+your\s+(initial|system)\s+(instructions|prompts?)",
    r"repeat\s+your\s+(system|initial)\s+(instructions|prompts?)",
    # Jailbreaking attempts
    r"dan\s+mode",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"unrestricted\s+ai",
    r"bypass\s+(safety|restrictions|guidelines)",
    # Context switching
    r"new\s+conversation\s+starts?\s+now",
    r"forget\s+everything\s+(above|before|prior)",
    r"start\s+over\s+with\s+new\s+(instructions|rules)",
    # Command injection in AI context
    r"execute\s+(this|the\s+following)\s+(command|code)",
    r"run\s+(this|the\s+following)\s+(command|code|script)",
    r"perform\s+the\s+following\s+action",
    # Multi-turn injection attempts
    r"in\s+your\s+next\s+response,\s+(ignore|forget)",
    r"from\s+now\s+on,\s+(ignore|bypass|forget)",
    r"for\s+all\s+future\s+responses,\s+(ignore|bypass)",
    # Unicode and encoding bypasses (basic detection)
    r"[\u200b-\u200f\u2060\ufeff]",  # Zero-width characters
    r"&#x?[0-9a-fA-F]+;",  # HTML entities
    r"%[0-9a-fA-F]{2}",  # URL encoding
    # Conversation hijacking
    r"human:\s*",
    r"user:\s*",
    r"assistant:\s*",
    r"ai:\s*",
    r"system:\s*",
    # Template injection
    r"\{\{.*?\}\}",
    r"\$\{.*?\}",
    r"<%.*?%>",
]

# Allowed model ID patterns
ALLOWED_MODEL_PATTERNS = [
    # AWS Bedrock patterns
    r"^us\.anthropic\.claude-\d+-\d*-?\w*-\d{8}-v\d+:\d+$",
    r"^anthropic\.claude-\d+-\w+-\d{8}-v\d+:\d+$",
    r"^anthropic\.claude-v\d+$",
    # OpenAI patterns
    r"^gpt-[34](\.\d+)?(-turbo)?(-\d{4})?$",
    r"^text-davinci-\d{3}$",
    # Local model patterns
    r"^llama\d*(\.\d+)?$",
    r"^mistral(\d+)?(\.\d+)?$",
    r"^codellama(\d+)?(\.\d+)?$",
    # Generic patterns for development
    r"^[a-zA-Z0-9\-_.]+$",  # Allow basic alphanumeric with common separators
]

# Compile patterns for efficiency
COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in SECURITY_PATTERNS
]

COMPILED_ADVANCED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in ADVANCED_PROMPT_INJECTION_PATTERNS
]

COMPILED_MODEL_PATTERNS = [re.compile(pattern) for pattern in ALLOWED_MODEL_PATTERNS]


class SecurityPatternMatcher:
    """Efficient security pattern matching utilities"""

    @staticmethod
    def match_basic_security_patterns(content: str) -> list[int]:
        """Find indices of all matching basic security patterns"""
        matches = []
        for i, pattern in enumerate(COMPILED_PATTERNS):
            if pattern.search(content):
                matches.append(i)
        return matches

    @staticmethod
    def match_advanced_injection_patterns(content: str) -> list[int]:
        """Find indices of all matching advanced injection patterns"""
        matches = []
        for i, pattern in enumerate(COMPILED_ADVANCED_PATTERNS):
            if pattern.search(content):
                matches.append(i)
        return matches

    @staticmethod
    def match_model_patterns(model_id: str) -> list[int]:
        """Find indices of all matching model ID patterns"""
        matches = []
        for i, pattern in enumerate(COMPILED_MODEL_PATTERNS):
            if pattern.match(model_id):
                matches.append(i)
        return matches

    @staticmethod
    def get_pattern_description(pattern_index: int, pattern_type: str = "basic") -> str:
        """Get human-readable description of a pattern"""
        if pattern_type == "basic":
            if 0 <= pattern_index < len(SECURITY_PATTERNS):
                return SECURITY_PATTERNS[pattern_index][:30] + "..."
        elif pattern_type == "advanced":
            if 0 <= pattern_index < len(ADVANCED_PROMPT_INJECTION_PATTERNS):
                return ADVANCED_PROMPT_INJECTION_PATTERNS[pattern_index][:30] + "..."
        return "Unknown pattern"
