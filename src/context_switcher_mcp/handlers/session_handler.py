"""Session management handler for Context-Switcher MCP Server"""

import logging
import secrets
from typing import Dict, Any, List, Optional

from ..aorp import create_error_response
from ..client_binding import create_secure_session_with_binding
from ..helpers.session_helpers import (
    determine_perspectives_to_use,
    create_session_threads,
    build_session_creation_response,
)
from ..models import ModelBackend
from ..security import sanitize_error_message

logger = logging.getLogger(__name__)

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
        initial_perspectives: Optional[List[str]] = None,
        template: Optional[str] = None,
        model_backend: ModelBackend = ModelBackend.BEDROCK,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            session = await create_secure_session_with_binding(session_id)
            session.topic = topic

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

            await session_manager.add_session(session_id, session)

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
