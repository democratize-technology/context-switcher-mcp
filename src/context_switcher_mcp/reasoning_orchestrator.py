"""Chain of Thought integration for structured perspective reasoning"""

import asyncio
import logging
from typing import Any, Dict, Tuple

try:
    from chain_of_thought import (
        TOOL_SPECS,
        AsyncChainOfThoughtProcessor,
    )

    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False

from .exceptions import OrchestrationError
from .models import Thread

logger = logging.getLogger(__name__)


class ChainOfThoughtError(OrchestrationError):
    """Base exception for Chain of Thought errors"""

    pass


class CoTTimeoutError(ChainOfThoughtError):
    """Chain of Thought processing timeout"""

    def __init__(self, timeout: float, message: str = None):
        msg = message or f"Chain of Thought processing timed out after {timeout}s"
        super().__init__(msg)
        self.timeout = timeout


class CoTProcessingError(ChainOfThoughtError):
    """Error during Chain of Thought processing"""

    def __init__(self, message: str, stage: str = None):
        super().__init__(message)
        self.stage = stage


class PerspectiveReasoningOrchestrator:
    """Orchestrates Chain of Thought reasoning for perspective analysis"""

    def __init__(self, cot_timeout: float = 30.0):
        """Initialize the reasoning orchestrator

        Args:
            cot_timeout: Maximum time in seconds for CoT processing (default: 30s)
        """
        if not COT_AVAILABLE:
            logger.warning(
                "chain-of-thought-tool not available. Install with: pip install chain-of-thought-tool"
            )
        self.cot_timeout = cot_timeout
        self._cot_available = COT_AVAILABLE
        self._processors: Dict[str, AsyncChainOfThoughtProcessor] = {}

    @property
    def is_available(self) -> bool:
        """Check if Chain of Thought reasoning is available"""
        return self._cot_available

    def get_cot_tools(self) -> list:
        """Get Chain of Thought tool specifications for Bedrock"""
        if not COT_AVAILABLE:
            return []
        return TOOL_SPECS

    async def analyze_with_reasoning(
        self,
        thread: Thread,
        prompt: str,
        bedrock_client: Any,
        session_id: str = "unknown",
    ) -> Tuple[str, Dict[str, Any]]:
        """Analyze a prompt using structured Chain of Thought reasoning

        Args:
            thread: The perspective thread to use
            prompt: The prompt to analyze
            bedrock_client: AWS Bedrock client instance
            session_id: Session ID for tracking

        Returns:
            Tuple of (response_text, reasoning_summary)
        """
        if not COT_AVAILABLE:
            # Fallback to simple analysis
            return "Chain of Thought not available", {}

        # Create or get processor for this perspective
        processor_id = f"{session_id}-{thread.name}"
        if processor_id not in self._processors:
            self._processors[processor_id] = AsyncChainOfThoughtProcessor(
                conversation_id=processor_id
            )
        processor = self._processors[processor_id]

        # Prepare enhanced system prompt with CoT instructions
        enhanced_prompt = f"""{thread.system_prompt}

You have access to Chain of Thought tools to structure your reasoning:
- Use chain_of_thought_step to work through your analysis systematically
- Start with Problem Definition stage to understand what you're analyzing
- Move through Research, Analysis, and Synthesis stages as appropriate
- End with Conclusion stage to provide your final perspective
- Set next_step_needed=false when you're ready with your complete analysis

Remember: You are analyzing from the {thread.name} perspective.
Focus on aspects most relevant to this perspective."""

        # Prepare messages - MUST start with user message for Bedrock
        messages = []

        # Only add conversation history if it exists and has content
        if thread.conversation_history:
            for msg in thread.conversation_history:
                # Skip empty messages
                if msg.get("content"):
                    messages.append(
                        {"role": msg["role"], "content": [{"text": msg["content"]}]}
                    )

        # Bedrock requires conversation to start with user message
        # If messages is empty or doesn't start with user, add the prompt as first message
        if not messages or messages[0]["role"] != "user":
            # Insert user message at the beginning
            user_message = {
                "role": "user",
                "content": [
                    {
                        "text": f"""Analyze the following from your {thread.name} perspective:

{prompt}

Use chain_of_thought_step to structure your reasoning, then provide your analysis."""
                    }
                ],
            }
            # If messages exist but don't start with user, prepend the user message
            if messages:
                messages.insert(0, user_message)
            else:
                messages.append(user_message)

        # Prepare Bedrock request with CoT tools
        request = {
            "modelId": thread.model_name or "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": messages,
            "system": [{"text": enhanced_prompt}],
            "toolConfig": {"tools": self.get_cot_tools()},
            "inferenceConfig": {
                "temperature": 0.7,  # Default temperature for CoT reasoning
                "maxTokens": 4096,  # Default max tokens
            },
        }

        try:
            # Process the tool loop with CoT and timeout
            result = await asyncio.wait_for(
                processor.process_tool_loop(
                    bedrock_client=bedrock_client,
                    initial_request=request,
                    max_iterations=20,  # Allow sufficient reasoning steps
                ),
                timeout=self.cot_timeout,
            )

            # Extract the final response text
            final_text = ""
            stop_reason = result.get("stopReason", "unknown")

            if "output" in result and "message" in result["output"]:
                content = result["output"]["message"].get("content", [])
                for item in content:
                    if "text" in item:
                        final_text += item["text"] + "\n"

            # Get reasoning summary with timeout
            try:
                reasoning_summary = await asyncio.wait_for(
                    processor.get_reasoning_summary(),
                    timeout=5.0,  # Quick timeout for summary
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout getting reasoning summary")
                reasoning_summary = {"status": "timeout"}

            # Log the stop reason for debugging
            logger.debug(f"CoT processing completed with stopReason: {stop_reason}")

            return final_text.strip(), reasoning_summary

        except asyncio.TimeoutError as e:
            logger.error(f"CoT analysis timed out after {self.cot_timeout}s")
            raise CoTTimeoutError(self.cot_timeout) from e
        except ChainOfThoughtError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error in CoT analysis: {e}", exc_info=True)
            raise CoTProcessingError(
                f"Failed to process reasoning: {str(e)}", stage="analysis"
            ) from e

    def clear_processor(self, session_id: str, thread_name: str) -> None:
        """Clear the processor for a specific perspective

        Args:
            session_id: Session ID
            thread_name: Thread/perspective name
        """
        processor_id = f"{session_id}-{thread_name}"
        if processor_id in self._processors:
            try:
                self._processors[processor_id].clear_reasoning()
            except Exception as e:
                logger.warning(f"Error clearing processor {processor_id}: {e}")
            finally:
                del self._processors[processor_id]

    def clear_session(self, session_id: str) -> None:
        """Clear all processors for a session

        Args:
            session_id: Session ID to clear
        """
        to_clear = [pid for pid in self._processors if pid.startswith(f"{session_id}-")]
        for processor_id in to_clear:
            try:
                self._processors[processor_id].clear_reasoning()
            except Exception as e:
                logger.warning(f"Error clearing processor {processor_id}: {e}")
            finally:
                del self._processors[processor_id]

    def get_active_processors(self) -> list:
        """Get list of active processor IDs"""
        return list(self._processors.keys())
