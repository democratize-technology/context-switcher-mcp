"""Chain of Thought integration for structured perspective reasoning"""

import asyncio
from .logging_base import get_logger
from typing import Any, Dict, Optional, Tuple

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
from .config import get_config

logger = get_logger(__name__)


class ChainOfThoughtError(OrchestrationError):
    """Base exception for Chain of Thought errors"""

    pass


class CoTTimeoutError(ChainOfThoughtError):
    """Chain of Thought processing timeout"""

    def __init__(self, timeout: float, message: Optional[str] = None):
        msg = message or f"Chain of Thought processing timed out after {timeout}s"
        super().__init__(msg)
        self.timeout = timeout


class CoTProcessingError(ChainOfThoughtError):
    """Error during Chain of Thought processing"""

    def __init__(self, message: str, stage: Optional[str] = None):
        super().__init__(message)
        self.stage = stage


class PerspectiveReasoningOrchestrator:
    """Orchestrates Chain of Thought reasoning for perspective analysis"""

    def __init__(self, cot_timeout: Optional[float] = None):
        """Initialize the reasoning orchestrator

        Args:
            cot_timeout: Maximum time in seconds for CoT processing (uses config default if None)
        """
        if not COT_AVAILABLE:
            logger.warning(
                "chain-of-thought-tool not available. Install with: pip install chain-of-thought-tool"
            )
        config = get_config()
        # Use defaults since reasoning config was consolidated
        self.cot_timeout = cot_timeout or 120.0  # 2 minutes default
        self.max_iterations = 5  # reasonable default for reasoning iterations
        self.summary_timeout = 60.0  # 1 minute for summary generation
        self.default_temperature = config.models.default_temperature
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
        topic: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Analyze a prompt using structured Chain of Thought reasoning

        Args:
            thread: The perspective thread to use
            prompt: The prompt to analyze
            bedrock_client: AWS Bedrock client instance
            session_id: Session ID for tracking
            topic: Optional topic context for analysis

        Returns:
            Tuple of (response_text, reasoning_summary)
        """
        if not COT_AVAILABLE:
            return "Chain of Thought not available", {}

        processor = self._get_or_create_processor(session_id, thread.name)
        enhanced_prompt = self._build_enhanced_system_prompt(thread)
        messages = self._prepare_conversation_messages(thread, prompt, topic)
        request = self._build_bedrock_request(thread, messages, enhanced_prompt)

        try:
            result = await self._execute_cot_processing(
                processor, bedrock_client, request
            )
            final_text = self._extract_response_text(result)
            reasoning_summary = await self._get_reasoning_summary_with_timeout(
                processor
            )

            logger.debug(
                f"CoT processing completed with stopReason: {result.get('stopReason', 'unknown')}"
            )
            return final_text.strip(), reasoning_summary

        except asyncio.TimeoutError as e:
            logger.error(f"CoT analysis timed out after {self.cot_timeout}s")
            raise CoTTimeoutError(self.cot_timeout) from e
        except ChainOfThoughtError:
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

    def _get_or_create_processor(
        self, session_id: str, thread_name: str
    ) -> AsyncChainOfThoughtProcessor:
        """Get existing processor or create new one for the perspective"""
        processor_id = f"{session_id}-{thread_name}"
        if processor_id not in self._processors:
            self._processors[processor_id] = AsyncChainOfThoughtProcessor(
                conversation_id=processor_id
            )
        return self._processors[processor_id]

    def _build_enhanced_system_prompt(self, thread: Thread) -> str:
        """Build enhanced system prompt with CoT instructions"""
        return f"""{thread.system_prompt}

You have access to Chain of Thought tools to structure your reasoning:
- Use chain_of_thought_step to work through your analysis systematically
- Start with Problem Definition stage to understand what you're analyzing
- Move through Research, Analysis, and Synthesis stages as appropriate
- End with Conclusion stage to provide your final perspective
- Set next_step_needed=false when you're ready with your complete analysis

Remember: You are analyzing from the {thread.name} perspective.
Focus on aspects most relevant to this perspective."""

    def _prepare_conversation_messages(
        self, thread: Thread, prompt: str, topic: Optional[str]
    ) -> list:
        """Prepare conversation messages with history and CoT prompt"""
        messages = []

        # Process conversation history, excluding the current prompt if already added
        history_to_process = self._get_conversation_history_to_process(thread, prompt)
        messages.extend(self._convert_history_to_bedrock_format(history_to_process))

        # Add the prompt with CoT instructions
        cot_prompt = self._build_cot_prompt(thread, prompt, topic)
        messages.append({"role": "user", "content": [{"text": cot_prompt}]})

        return messages

    def _get_conversation_history_to_process(self, thread: Thread, prompt: str) -> list:
        """Get conversation history excluding duplicate prompts"""
        if not thread.conversation_history:
            return []

        last_msg = thread.conversation_history[-1]
        if last_msg.get("role") == "user" and last_msg.get("content") == prompt:
            # Skip the duplicate prompt we just added
            return thread.conversation_history[:-1]
        return thread.conversation_history

    def _convert_history_to_bedrock_format(self, history: list) -> list:
        """Convert conversation history to Bedrock message format"""
        messages = []
        for msg in history:
            content = msg.get("content", "").strip()
            if content:
                messages.append({"role": msg["role"], "content": [{"text": content}]})
        return messages

    def _build_cot_prompt(
        self, thread: Thread, prompt: str, topic: Optional[str]
    ) -> str:
        """Build Chain of Thought prompt with optional topic context"""
        if topic:
            return f"""Context: You are analyzing the topic "{topic}" from your {thread.name} perspective.

Specific question to analyze:
{prompt}

Use chain_of_thought_step to structure your reasoning about "{topic}", then provide your analysis."""
        else:
            return f"""Analyze the following from your {thread.name} perspective:

{prompt}

Use chain_of_thought_step to structure your reasoning, then provide your analysis."""

    def _build_bedrock_request(
        self, thread: Thread, messages: list, enhanced_prompt: str
    ) -> dict:
        """Build Bedrock API request with CoT tools"""
        return {
            "modelId": thread.model_name or "anthropic.claude-3-sonnet-20240229-v1:0",
            "messages": messages,
            "system": [{"text": enhanced_prompt}],
            "toolConfig": {"tools": self.get_cot_tools()},
            "inferenceConfig": {
                "temperature": self.default_temperature,
                "maxTokens": 4096,
            },
        }

    async def _execute_cot_processing(
        self,
        processor: AsyncChainOfThoughtProcessor,
        bedrock_client: Any,
        request: dict,
    ) -> dict:
        """Execute CoT processing with timeout"""
        return await asyncio.wait_for(
            processor.process_tool_loop(
                bedrock_client=bedrock_client,
                initial_request=request,
                max_iterations=self.max_iterations,
            ),
            timeout=self.cot_timeout,
        )

    def _extract_response_text(self, result: dict) -> str:
        """Extract final response text from CoT processing result"""
        final_text = ""
        if "output" in result and "message" in result["output"]:
            content = result["output"]["message"].get("content", [])
            for item in content:
                if "text" in item:
                    final_text += item["text"] + "\n"
        return final_text

    async def _get_reasoning_summary_with_timeout(
        self, processor: AsyncChainOfThoughtProcessor
    ) -> dict:
        """Get reasoning summary with timeout handling"""
        try:
            return await asyncio.wait_for(
                processor.get_reasoning_summary(),
                timeout=self.summary_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout getting reasoning summary")
            return {"status": "timeout"}
