"""Session management handler for Context-Switcher MCP Server"""

import secrets
from typing import Any

from ..aorp import create_error_response
from ..client_binding import create_secure_session_with_binding
from ..helpers.session_helpers import (
    build_session_creation_response,
    create_session_threads,
    determine_perspectives_to_use,
)
from ..logging_config import get_logger
from ..models import ModelBackend
from ..security import sanitize_error_message

logger = get_logger(__name__)

# Remove unused imports and constants - moved to session_helpers


def generate_secure_session_id() -> str:
    """Generate cryptographically secure session ID

    Uses secrets.token_urlsafe(32) which generates 256 bits of entropy
    and is safe for URLs and session identifiers.
    """
    return secrets.token_urlsafe(32)


class SessionHandler:
    """Handler for session creation and initialization logic"""

    @staticmethod
    async def create_session(
        topic: str,
        initial_perspectives: list[str] | None = None,
        template: str | None = None,
        model_backend: ModelBackend = ModelBackend.BEDROCK,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a new context-switching analysis session

        Args:
            topic: The analysis topic
            initial_perspectives: List of perspective names to use
            template: Template name for pre-configured perspectives
            model_backend: LLM backend to use
            model_name: Specific model name

        Returns:
            Dictionary with session creation result
        """
        try:
            # Generate secure session ID
            session_id = generate_secure_session_id()

            # Create secure session with client binding
            session = create_secure_session_with_binding(session_id, topic)

            # Determine which perspectives to use
            perspectives_to_use = determine_perspectives_to_use(
                template, initial_perspectives
            )

            # Create threads for each perspective
            await create_session_threads(
                session, perspectives_to_use, model_backend, model_name, session_id
            )

            # Store session
            from .. import session_manager

            await session_manager.add_session(session)

            # Build and return success response
            return build_session_creation_response(
                session_id, topic, session, template, model_backend
            )

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return create_error_response(
                f"Session creation failed: {sanitize_error_message(str(e))}",
                "session_creation_error",
                {"topic": topic},
                recoverable=True,
            )
