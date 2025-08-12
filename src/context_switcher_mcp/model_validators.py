"""Model validation utilities for validating model IDs and configurations"""

from .logging_base import get_logger
from typing import Tuple

from .security_patterns import SecurityPatternMatcher

logger = get_logger(__name__)


class ModelIdValidator:
    """Validates model IDs against security patterns"""

    @staticmethod
    def validate_model_id(model_id: str) -> Tuple[bool, str]:
        """
        Validate model ID against allowed patterns

        Args:
            model_id: Model ID to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not model_id or not isinstance(model_id, str):
            return False, "Model ID must be a non-empty string"

        if len(model_id) > 200:
            return False, "Model ID too long (max 200 characters)"

        # Check against allowed patterns using the pattern matcher
        matching_patterns = SecurityPatternMatcher.match_model_patterns(model_id)
        if matching_patterns:
            return True, ""

        logger.warning(f"Rejected invalid model ID: {model_id}")
        return False, f"Model ID '{model_id}' not in allowed patterns"


# Global validator instance
_model_validator = ModelIdValidator()


def validate_model_id(model_id: str) -> Tuple[bool, str]:
    """
    Validate model ID against allowed patterns

    Args:
        model_id: Model ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    return _model_validator.validate_model_id(model_id)
