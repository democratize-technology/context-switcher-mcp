"""Response formatting service for AORP-compliant responses"""

import logging
from typing import Dict, Any

from .aorp import create_error_response
from .compression import compress_perspectives
from .models import Thread, ModelBackend
from .backend_interface import get_backend_interface
from .exceptions import ModelBackendError

logger = logging.getLogger(__name__)

# Constants
NO_RESPONSE = "[NO_RESPONSE]"


class ResponseFormatter:
    """Formats responses according to AORP specification and handles synthesis"""

    def __init__(self):
        """Initialize response formatter"""
        pass

    def format_error_response(
        self,
        error_message: str,
        error_type: str,
        context: Dict[str, Any] = None,
        recoverable: bool = True,
    ) -> str:
        """Format an error response according to AORP specification"""
        import json

        error_response = create_error_response(
            error_message=error_message,
            error_type=error_type,
            context=context or {},
            recoverable=recoverable,
        )
        return f"AORP_ERROR: {json.dumps(error_response)}"

    def format_abstention_response(
        self, perspective_name: str, reason: str = ""
    ) -> str:
        """Format an abstention response for perspectives that cannot respond"""
        reason_text = f" - {reason}" if reason else ""
        return f"{NO_RESPONSE}{reason_text}"

    def format_success_response(
        self, content: str, metadata: Dict[str, Any] = None
    ) -> str:
        """Format a successful response with optional metadata"""
        if metadata:
            # For now, just return the content - metadata handling can be enhanced later
            return content
        return content

    async def synthesize_responses(
        self,
        responses: Dict[str, str],
        session_id: str = "unknown",
        synthesis_backend: ModelBackend = ModelBackend.BEDROCK,
    ) -> str:
        """Synthesize multiple perspective responses into a coherent analysis"""
        try:
            if not responses:
                return self.format_error_response(
                    "No responses provided for synthesis",
                    "synthesis_input_error",
                    {"session_id": session_id},
                )

            # Filter out error responses and abstentions
            valid_responses = {
                name: response
                for name, response in responses.items()
                if not response.startswith("ERROR:") and NO_RESPONSE not in response
            }

            if not valid_responses:
                return self.format_error_response(
                    "No valid responses available for synthesis",
                    "synthesis_no_valid_input",
                    {"session_id": session_id},
                )

            # Compress responses for synthesis if needed
            compressed_responses = compress_perspectives(
                valid_responses, max_chars_per_perspective=2000
            )

            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(compressed_responses)

            # Create a temporary thread for synthesis
            synthesis_thread = Thread(
                id="synthesis_thread",
                name="synthesis",
                model_backend=synthesis_backend,
                model_name=None,
                system_prompt="You are an expert analyst synthesizing insights from multiple perspectives. "
                "Identify patterns, resolve conflicts, and provide actionable recommendations.",
            )
            synthesis_thread.add_message("user", synthesis_prompt)

            backend_interface = get_backend_interface(synthesis_backend.value)
            synthesis_response = await backend_interface.call_model(synthesis_thread)

            return self.format_success_response(
                synthesis_response,
                {"session_id": session_id, "perspectives_count": len(valid_responses)},
            )

        except ModelBackendError as e:
            logger.error(f"Model backend error during synthesis: {e}")
            return self.format_error_response(
                f"Synthesis failed due to model error: {str(e)}",
                "synthesis_model_error",
                {"session_id": session_id},
            )
        except Exception as e:
            logger.error(f"Unexpected error during synthesis: {e}")
            return self.format_error_response(
                f"Synthesis failed unexpectedly: {str(e)}",
                "synthesis_unexpected_error",
                {"session_id": session_id},
            )

    def _create_synthesis_prompt(self, responses: Dict[str, str]) -> str:
        """Create a prompt for synthesizing multiple perspective responses"""
        prompt_parts = [
            "I need you to synthesize insights from multiple perspective analyses below.",
            "Look for:",
            "1. Common themes and patterns across perspectives",
            "2. Areas of disagreement or conflicting viewpoints",
            "3. Unique insights that only appear in specific perspectives",
            "4. Actionable recommendations that emerge from the collective analysis",
            "",
            "Provide a coherent synthesis that:",
            "- Integrates the most valuable insights from each perspective",
            "- Addresses any contradictions or tensions",
            "- Offers clear, actionable next steps",
            "",
            "PERSPECTIVE ANALYSES:",
            "",
        ]

        for perspective_name, response in responses.items():
            prompt_parts.extend(
                [f"=== {perspective_name.upper()} PERSPECTIVE ===", response, ""]
            )

        prompt_parts.extend(
            [
                "=== SYNTHESIS REQUEST ===",
                "Please provide a comprehensive synthesis that integrates these perspectives into actionable insights.",
            ]
        )

        return "\n".join(prompt_parts)

    def format_streaming_event(
        self,
        event_type: str,
        content: str,
        perspective_name: str = None,
        timestamp: float = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Format a streaming event for real-time responses"""
        import time

        event = {
            "type": event_type,  # "start", "chunk", "complete", "error"
            "content": content,
            "timestamp": timestamp or time.time(),
        }

        if perspective_name:
            event["perspective_name"] = perspective_name

        if metadata:
            event["metadata"] = metadata

        return event

    def is_error_response(self, response: str) -> bool:
        """Check if a response is an error response"""
        return response.startswith("ERROR:") or response.startswith("AORP_ERROR:")

    def is_abstention_response(self, response: str) -> bool:
        """Check if a response is an abstention"""
        return NO_RESPONSE in response

    def extract_error_info(self, error_response: str) -> Dict[str, Any]:
        """Extract error information from an AORP error response"""
        try:
            if error_response.startswith("AORP_ERROR: "):
                error_json = error_response[len("AORP_ERROR: ") :]
                import json

                parsed = json.loads(error_json)

                # Check if it's a simple error format (for tests)
                if "error_message" in parsed and "error_type" in parsed:
                    return {
                        "error_message": parsed.get("error_message", ""),
                        "error_type": parsed.get("error_type", "unknown_error"),
                        "recoverable": parsed.get("recoverable", True),
                        "full_response": parsed,
                    }

                # Extract error_type from various possible locations (AORP format)
                error_type = "unknown_error"
                error_message = ""
                recoverable = True

                # Check indicators first (new format)
                if "indicators" in parsed:
                    error_type = parsed["indicators"].get("error_type", error_type)
                    recoverable = parsed["indicators"].get("recoverable", recoverable)

                # Check details.data for error_code (AORP format)
                if "details" in parsed and "data" in parsed["details"]:
                    error_type = parsed["details"]["data"].get("error_code", error_type)

                # Get error message from immediate.key_insight
                if "immediate" in parsed:
                    error_message = parsed["immediate"].get(
                        "key_insight", error_message
                    )

                return {
                    "error_message": error_message,
                    "error_type": error_type,
                    "recoverable": recoverable,
                    "full_response": parsed,
                }
            elif error_response.startswith("ERROR: "):
                return {
                    "error_message": error_response[len("ERROR: ") :],
                    "error_type": "generic_error",
                    "recoverable": True,
                }
            else:
                return {
                    "error_message": error_response,
                    "error_type": "unknown_error",
                    "recoverable": True,
                }
        except Exception as e:
            logger.warning(f"Failed to parse error response: {e}")
            return {
                "error_message": error_response,
                "error_type": "parse_error",
                "recoverable": True,
            }

    def format_perspective_summary(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """Create a summary of perspective responses for metrics and logging"""
        summary = {
            "total_perspectives": len(responses),
            "successful_responses": 0,
            "abstained_responses": 0,
            "error_responses": 0,
            "response_lengths": {},
            "perspective_names": list(responses.keys()),
        }

        for perspective_name, response in responses.items():
            summary["response_lengths"][perspective_name] = len(response)

            if self.is_error_response(response):
                summary["error_responses"] += 1
            elif self.is_abstention_response(response):
                summary["abstained_responses"] += 1
            else:
                summary["successful_responses"] += 1

        summary["success_rate"] = (
            summary["successful_responses"] / summary["total_perspectives"] * 100
            if summary["total_perspectives"] > 0
            else 0
        )

        return summary
