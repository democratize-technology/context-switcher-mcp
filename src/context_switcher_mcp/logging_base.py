"""
Basic logging utilities without dependencies to avoid circular imports.

This module provides the core get_logger function that can be safely imported
throughout the codebase without creating circular dependencies.
"""

import logging
import os
from contextvars import ContextVar

# Context variable for correlation ID tracking
correlation_id_context: ContextVar[str | None] = ContextVar(
    "correlation_id", default=None
)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.

    This is a simplified version that avoids circular imports by not
    depending on the full logging_config infrastructure.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set up basic configuration if not already configured
    if not logger.handlers and not logging.getLogger().handlers:
        level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
        logging.basicConfig(
            level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    return logger


def setup_logging() -> None:
    """Setup basic logging configuration."""
    level = logging.DEBUG if os.environ.get("DEBUG") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override existing configuration
    )


def set_correlation_id(correlation_id: str | None) -> None:
    """Set correlation ID in context."""
    correlation_id_context.set(correlation_id)


def get_correlation_id() -> str | None:
    """Get current correlation ID from context."""
    return correlation_id_context.get()


def is_performance_logging_enabled() -> bool:
    """Check if performance logging is enabled."""
    return os.environ.get("ENABLE_PERFORMANCE_LOGGING", "").lower() in (
        "true",
        "1",
        "yes",
    )
