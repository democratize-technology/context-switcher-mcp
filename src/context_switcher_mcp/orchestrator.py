"""Thread orchestration for parallel LLM execution"""

import asyncio
import logging
import os
from typing import Dict

from .models import Thread, ModelBackend

logger = logging.getLogger(__name__)

# Constants
NO_RESPONSE = "[NO_RESPONSE]"


class ThreadOrchestrator:
    """Orchestrates parallel thread execution with different LLM backends"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize orchestrator
        
        Args:
            max_retries: Maximum number of retries for failed calls
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.backends = {
            ModelBackend.BEDROCK: self._call_bedrock,
            ModelBackend.LITELLM: self._call_litellm,
            ModelBackend.OLLAMA: self._call_ollama
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def broadcast_message(
        self, 
        threads: Dict[str, Thread], 
        message: str
    ) -> Dict[str, str]:
        """Broadcast message to all threads and collect responses"""
        tasks = []
        thread_names = []
        
        for name, thread in threads.items():
            # Add user message to thread history
            thread.add_message("user", message)
            
            # Create task for this thread
            task = self._get_thread_response(thread)
            tasks.append(task)
            thread_names.append(name)
        
        # Execute all threads in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build response dictionary
        result = {}
        for name, response in zip(thread_names, responses):
            if isinstance(response, Exception):
                logger.error(f"Error in thread {name}: {response}")
                result[name] = f"ERROR: {str(response)}"
            else:
                # Add assistant response to thread history
                threads[name].add_message("assistant", response)
                result[name] = response
        
        return result
    
    async def _get_thread_response(self, thread: Thread) -> str:
        """Get response from a single thread with retry logic"""
        backend_fn = self.backends.get(thread.model_backend)
        if not backend_fn:
            raise ValueError(f"Unknown model backend: {thread.model_backend}")
        
        # Try with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await backend_fn(thread)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Don't retry on non-transient errors
                if any(term in error_str for term in [
                    "api_key", "credentials", "not found", "invalid", 
                    "unauthorized", "forbidden", "model not found"
                ]):
                    logger.error(f"Non-retryable error for {thread.name}: {e}")
                    return f"Error: {str(e)}"
                
                # Retry on transient errors
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {thread.name}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries} attempts failed for {thread.name}: {e}"
                    )
        
        # If all retries failed, return error message
        return f"Error after {self.max_retries} attempts: {str(last_error)}"
    
    async def _call_bedrock(self, thread: Thread) -> str:
        """Call AWS Bedrock model"""
        try:
            import boto3
            
            # Create Bedrock client
            client = boto3.client(
                service_name='bedrock-runtime',
                region_name='us-east-1'
            )
            
            # Prepare messages for Bedrock Converse API
            messages = []
            
            # Add conversation history (skip system message)
            for msg in thread.conversation_history:
                # Bedrock expects content as a list of content blocks
                messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}]  # Fixed: content as list
                })
            
            # Call Bedrock
            model_id = thread.model_name or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
            
            response = client.converse(
                modelId=model_id,
                messages=messages,
                system=[{"text": thread.system_prompt}],
                inferenceConfig={
                    "maxTokens": 2048,
                    "temperature": 0.7,
                }
            )
            
            # Extract response
            content = response['output']['message']['content'][0]['text']
            return content
            
        except Exception as e:
            logger.error(f"Bedrock error: {e}")
            if "inference profile" in str(e).lower():
                return "Error: Model needs inference profile ID. Try: us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            elif "credentials" in str(e).lower():
                return "Error: AWS credentials not configured. Run: aws configure"
            else:
                return f"Error calling Bedrock: {str(e)}"
    
    async def _call_litellm(self, thread: Thread) -> str:
        """Call model via LiteLLM"""
        try:
            import litellm
            
            # Prepare messages
            messages = [
                {"role": "system", "content": thread.system_prompt}
            ]
            
            # Add conversation history
            for msg in thread.conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Call LiteLLM
            model = thread.model_name or "gpt-4"
            
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LiteLLM error: {e}")
            if "api_key" in str(e).lower():
                return "Error: Missing API key. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable"
            elif "connection" in str(e).lower():
                return "Error: Cannot connect to LiteLLM. Check LITELLM_API_BASE is set correctly"
            else:
                return f"Error calling LiteLLM: {str(e)}"
    
    async def _call_ollama(self, thread: Thread) -> str:
        """Call local Ollama model"""
        try:
            import httpx
            
            # Prepare messages
            messages = [
                {"role": "system", "content": thread.system_prompt}
            ]
            
            # Add conversation history
            for msg in thread.conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Call Ollama API
            model = thread.model_name or "llama3.2"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 2048
                        }
                    },
                    timeout=60.0
                )
                
                response.raise_for_status()
                result = response.json()
                return result['message']['content']
                
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            if "connection" in str(e).lower():
                return "Error: Cannot connect to Ollama. Is it running? Set OLLAMA_HOST=http://your-host:11434"
            elif "model" in str(e).lower():
                return f"Error: Model not found. Pull it first: ollama pull {model}"
            else:
                return f"Error calling Ollama: {str(e)}"