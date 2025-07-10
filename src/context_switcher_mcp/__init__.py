
#!/usr/bin/env python3
"""
Context-Switcher MCP Server
Multi-perspective analysis using thread orchestration
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .compression import prepare_synthesis_input
from .templates import PERSPECTIVE_TEMPLATES

__all__ = ["main", "mcp"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    name="context-switcher",
    version="0.1.0"
)

# Constants
NO_RESPONSE = "[NO_RESPONSE]"
DEFAULT_PERSPECTIVES = {
    'technical': """Evaluate from a technical architecture and implementation perspective.
Focus on: system design, scalability, maintainability, technical debt, complexity, performance.
Abstain if the topic has no technical aspects.""",
    
    'business': """Evaluate from a business value, ROI, and strategic perspective.
Focus on: revenue impact, cost implications, market position, competitive advantage, strategic alignment.
Abstain if the topic has no business implications.""",
    
    'user': """Evaluate from an end-user experience and usability perspective.
Focus on: ease of use, accessibility, user satisfaction, learning curve, daily workflow impact.
Abstain if the topic doesn't affect end users.""",
    
    'risk': """Evaluate from a risk, security, and compliance perspective.
Focus on: security vulnerabilities, compliance requirements, operational risks, data privacy.
Abstain if the topic has no risk implications."""
}

class ModelBackend(str, Enum):
    BEDROCK = "bedrock"
    LITELLM = "litellm"
    OLLAMA = "ollama"

@dataclass
class Thread:
    """Represents a single perspective thread"""
    id: str
    name: str
    system_prompt: str
    model_backend: ModelBackend
    model_name: Optional[str]
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

@dataclass
class ContextSwitcherSession:
    """Manages a context-switching analysis session"""
    session_id: str
    created_at: datetime
    threads: Dict[str, Thread] = field(default_factory=dict)
    analyses: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_thread(self, thread: Thread):
        """Add a perspective thread to the session"""
        self.threads[thread.name] = thread

# Global session storage
sessions: Dict[str, ContextSwitcherSession] = {}

# Helper functions
def validate_session_id(session_id: str) -> bool:
    """Validate session ID format and existence"""
    if not session_id or not isinstance(session_id, str):
        return False
    if len(session_id) > 100:  # Reasonable limit
        return False
    return session_id in sessions

def validate_topic(topic: str) -> bool:
    """Validate topic string"""
    if not topic or not isinstance(topic, str):
        return False
    if len(topic.strip()) == 0 or len(topic) > 1000:  # Reasonable limits
        return False
    return True

# Thread orchestrator for parallel execution
class ThreadOrchestrator:
    """Orchestrates parallel thread execution with different LLM backends"""
    
    def __init__(self):
        self.backends = {
            ModelBackend.BEDROCK: self._call_bedrock,
            ModelBackend.LITELLM: self._call_litellm,
            ModelBackend.OLLAMA: self._call_ollama
        }
    
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
        """Get response from a single thread"""
        backend_fn = self.backends.get(thread.model_backend)
        if not backend_fn:
            raise ValueError(f"Unknown model backend: {thread.model_backend}")
        
        return await backend_fn(thread)
    
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

# Shared orchestrator instance
orchestrator = ThreadOrchestrator()

# MCP Tool Definitions

class StartContextAnalysisRequest(BaseModel):
    topic: str = Field(description="The topic or problem to analyze from multiple perspectives")
    initial_perspectives: Optional[List[str]] = Field(
        default=None,
        description="List of perspective names to use (defaults to: technical, business, user, risk)"
    )
    model_backend: ModelBackend = Field(
        default=ModelBackend.BEDROCK,
        description="LLM backend to use"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model to use"
    )
    template: Optional[str] = Field(
        default=None,
        description="Pre-configured perspective template: architecture_decision, feature_evaluation, debugging_analysis, api_design, security_review"
    )

@mcp.tool(
    description="When you're at a crossroads and need multiple viewpoints - analyze architecture decisions, debug blind spots, or evaluate features from technical, business, user, and risk angles simultaneously. Templates available: architecture_decision, feature_evaluation, debugging_analysis, api_design, security_review"
)
async def start_context_analysis(request: StartContextAnalysisRequest) -> Dict[str, Any]:
    """Initialize a new context-switching analysis session"""
    # Validate input
    if not validate_topic(request.topic):
        return {"error": "Invalid topic: must be a non-empty string under 1000 characters"}
    
    # Create new session
    session_id = str(uuid4())
    session = ContextSwitcherSession(
        session_id=session_id,
        created_at=datetime.utcnow()
    )
    session.topic = request.topic  # Store topic for easy reference
    
    # Initialize default perspectives or use provided ones
    if request.template and request.template in PERSPECTIVE_TEMPLATES:
        # Use template perspectives
        template = PERSPECTIVE_TEMPLATES[request.template]
        perspectives_to_use = template["perspectives"]
        
        # We'll add custom perspectives after creating base ones
        custom_perspectives = template.get("custom", [])
    else:
        perspectives_to_use = request.initial_perspectives or list(DEFAULT_PERSPECTIVES.keys())
        custom_perspectives = []
    
    for perspective_name in perspectives_to_use:
        # Get the prompt for this perspective
        if perspective_name in DEFAULT_PERSPECTIVES:
            prompt = DEFAULT_PERSPECTIVES[perspective_name]
        else:
            # For custom perspectives, create a basic prompt
            prompt = f"""Analyze from the {perspective_name} perspective.
Provide insights specific to this viewpoint.
Abstain with {NO_RESPONSE} if this perspective doesn't apply."""
        
        # Create thread for this perspective
        thread = Thread(
            id=str(uuid4()),
            name=perspective_name,
            system_prompt=prompt,
            model_backend=request.model_backend,
            model_name=request.model_name
        )
        
        session.add_thread(thread)
    
    # Store session
    sessions[session_id] = session
    
    # Add custom perspectives from template
    for persp_name, persp_desc in custom_perspectives:
        custom_thread = Thread(
            id=str(uuid4()),
            name=persp_name,
            system_prompt=f"""Analyze from the {persp_name} perspective.
Focus on: {persp_desc}
Provide insights specific to this viewpoint.
Abstain with {NO_RESPONSE} if this perspective doesn't apply.""",
            model_backend=request.model_backend,
            model_name=request.model_name
        )
        session.add_thread(custom_thread)
    
    return {
        "session_id": session_id,
        "topic": request.topic,
        "perspectives": list(session.threads.keys()),
        "model_backend": request.model_backend.value,
        "model_name": request.model_name,
        "message": f"Context analysis session initialized with {len(session.threads)} perspectives"
    }

class AddPerspectiveRequest(BaseModel):
    session_id: str = Field(description="Session ID to add perspective to")
    name: str = Field(description="Name of the new perspective")
    description: str = Field(description="Description of what this perspective should focus on")
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for this perspective"
    )

@mcp.tool(
    description="When the standard perspectives miss something crucial - add specialized lenses like 'performance', 'migration_path', 'customer_support', or any domain-specific viewpoint your problem needs"
)
async def add_perspective(request: AddPerspectiveRequest) -> Dict[str, Any]:
    """Add a new perspective to an existing analysis session"""
    # Validate session ID
    if not validate_session_id(request.session_id):
        return {"error": "Invalid or non-existent session ID"}
    
    # Get session
    session = sessions.get(request.session_id)
    
    # Create prompt for new perspective
    if request.custom_prompt:
        prompt = request.custom_prompt
    else:
        prompt = f"""Analyze from the {request.name} perspective.
Focus on: {request.description}
Provide insights specific to this viewpoint.
Abstain with {NO_RESPONSE} if this perspective doesn't apply to the topic."""
    
    # Create new thread
    thread = Thread(
        id=str(uuid4()),
        name=request.name,
        system_prompt=prompt,
        model_backend=ModelBackend.BEDROCK,  # Use session default
        model_name=None
    )
    
    # Add to session
    session.add_thread(thread)
    
    return {
        "session_id": request.session_id,
        "perspective_added": request.name,
        "total_perspectives": len(session.threads),
        "all_perspectives": list(session.threads.keys())
    }

class AnalyzeFromPerspectivesRequest(BaseModel):
    session_id: str = Field(description="Session ID for analysis")
    prompt: str = Field(description="The specific question or topic to analyze")

@mcp.tool(
    description="When you need parallel insights NOW - broadcast your question to all perspectives simultaneously. Expect 10-30 seconds for comprehensive analysis. Perspectives can abstain with [NO_RESPONSE] if not relevant"
)
async def analyze_from_perspectives(request: AnalyzeFromPerspectivesRequest) -> Dict[str, Any]:
    """Broadcast a prompt to all perspectives and collect their responses"""
    # Validate session ID
    if not validate_session_id(request.session_id):
        return {"error": "Invalid or non-existent session ID"}
        
    # Get session
    session = sessions.get(request.session_id)
    
    # Broadcast to all threads
    responses = await orchestrator.broadcast_message(
        session.threads,
        request.prompt
    )
    
    # Process responses
    active_perspectives = {}
    abstained_perspectives = []
    errors = []
    
    for name, response in responses.items():
        if response.startswith("ERROR:"):
            errors.append({name: response})
        elif NO_RESPONSE in response:
            abstained_perspectives.append(name)
        else:
            active_perspectives[name] = response
    
    # Store analysis
    analysis = {
        "prompt": request.prompt,
        "timestamp": datetime.utcnow().isoformat(),
        "responses": responses,
        "active_count": len(active_perspectives),
        "abstained_count": len(abstained_perspectives)
    }
    session.analyses.append(analysis)
    
    return {
        "session_id": request.session_id,
        "prompt": request.prompt,
        "perspectives": active_perspectives,
        "abstained": abstained_perspectives,
        "errors": errors,
        "summary": {
            "total_perspectives": len(session.threads),
            "active_responses": len(active_perspectives),
            "abstentions": len(abstained_perspectives),
            "errors": len(errors)
        }
    }

class SynthesizePerspectivesRequest(BaseModel):
    session_id: str = Field(description="Session ID to synthesize")

@mcp.tool(
    description="When you need the 'aha!' moment - discover surprising tensions, hidden connections, and emergent insights across all perspectives. Often reveals solutions you hadn't considered"
)
async def synthesize_perspectives(request: SynthesizePerspectivesRequest) -> Dict[str, Any]:
    """Analyze patterns across all perspectives from the last analysis"""
    # Get session
    session = sessions.get(request.session_id)
    if not session:
        return {"error": f"Session {request.session_id} not found"}
    
    if not session.analyses:
        return {"error": "No analyses to synthesize. Run analyze_from_perspectives first."}
    
    # Get most recent analysis
    latest = session.analyses[-1]
    
    # Extract active perspectives
    active = {}
    for name, response in latest["responses"].items():
        if not response.startswith("ERROR:") and NO_RESPONSE not in response:
            active[name] = response
    
    # Prepare synthesis prompt with compression
    # Log token estimates
    total_chars = sum(len(response) for response in active.values())
    estimated_tokens = total_chars // 4  # Rough estimate
    logger.info(f"Synthesis input: {len(active)} perspectives, ~{total_chars:,} chars, ~{estimated_tokens:,} tokens")
    
        # Compress perspectives to fit token limits
    compressed_text = prepare_synthesis_input(active, max_total_chars=12000)
    
    synthesis_prompt = f"""
Based on the following perspectives on "{latest['prompt']}":

{compressed_text}

Please synthesize these viewpoints by:
1. Identifying common themes and agreements
2. Highlighting key conflicts or tensions
3. Finding unexpected connections between perspectives
4. Suggesting insights that emerge from the combination
5. Recommending which perspectives deserve deeper exploration

Focus on actionable synthesis, not just summary.
"""
    
    # Create a synthesis thread
    synthesis_thread = Thread(
        id=str(uuid4()),
        name="synthesis",
        system_prompt="You are a master synthesizer who finds patterns and insights across diverse viewpoints.",
        model_backend=list(session.threads.values())[0].model_backend,
        model_name=list(session.threads.values())[0].model_name
    )
    
    # Get synthesis
    synthesis_thread.add_message("user", synthesis_prompt)
    synthesis = await orchestrator._get_thread_response(synthesis_thread)
    
    return {
        "session_id": request.session_id,
        "analyzed_prompt": latest["prompt"],
        "synthesis": synthesis,
        "perspectives_analyzed": list(active.keys()),
        "metadata": {
            "total_perspectives": len(latest["responses"]),
            "active_perspectives": len(active),
            "analysis_timestamp": latest["timestamp"]
        }
    }

# Session management tools

@mcp.tool(description="List all active context-switching sessions")
async def list_sessions() -> Dict[str, Any]:
    """List all active analysis sessions"""
    session_list = []
    for sid, session in sessions.items():
        session_list.append({
            "session_id": sid,
            "created_at": session.created_at.isoformat(),
            "perspectives": list(session.threads.keys()),
            "analyses_count": len(session.analyses)
        })

    
    return {
        "sessions": session_list,
        "total_sessions": len(sessions)
    }

@mcp.tool(description="See available perspective templates for common analysis patterns - architecture decisions, debugging, API design, and more")
async def list_templates() -> Dict[str, Any]:
    """List all available perspective templates"""
    template_info = {}
    
    for name, template in PERSPECTIVE_TEMPLATES.items():
        perspectives = template["perspectives"].copy()
        
        # Add custom perspective names
        for custom_name, _ in template.get("custom", []):
            perspectives.append(f"{custom_name} (custom)")
        
        template_info[name] = {
            "description": name.replace("_", " ").title(),
            "perspectives": perspectives,
            "total_perspectives": len(template["perspectives"]) + len(template.get("custom", []))
        }
    
    return {
        "templates": template_info,
        "usage": "Use template parameter in start_context_analysis",
        "example": 'start_context_analysis(topic="...", template="architecture_decision")'
    }

@mcp.tool(description="Quick check of your most recent analysis session - see perspectives and results without remembering session ID")
async def current_session() -> Dict[str, Any]:
    """Get information about the most recent session"""
    if not sessions:
        return {
            "status": "No active sessions",
            "hint": "Start with: start_context_analysis"
        }
    
    # Get most recent session
    recent_session_id = max(sessions.keys(), 
                           key=lambda sid: sessions[sid].created_at)
    session = sessions[recent_session_id]
    
    # Get summary of last analysis if any
    last_analysis = None
    if session.analyses:
        last = session.analyses[-1]
        last_analysis = {
            "prompt": last["prompt"][:100] + "..." if len(last["prompt"]) > 100 else last["prompt"],
            "perspectives_responded": last["active_count"],
            "perspectives_abstained": last["abstained_count"] if "abstained_count" in last else 0
        }
    
    return {
        "session_id": recent_session_id,
        "created": session.created_at.strftime("%H:%M:%S"),
        "topic": getattr(session, 'topic', 'Unknown'),
        "perspectives": list(session.threads.keys()),
        "total_perspectives": len(session.threads),
        "analyses_run": len(session.analyses),
        "last_analysis": last_analysis,
        "next_steps": [
            "analyze_from_perspectives - Ask a question",
            "add_perspective - Add custom viewpoint", 
            "synthesize_perspectives - Find patterns"
        ] if last_analysis is None else ["synthesize_perspectives - Find patterns across viewpoints"]
    }

class GetSessionRequest(BaseModel):
    session_id: str = Field(description="Session ID to retrieve")

@mcp.tool(description="Get details of a specific context-switching session")
async def get_session(request: GetSessionRequest) -> Dict[str, Any]:
    """Get detailed information about a session"""
    session = sessions.get(request.session_id)
    if not session:
        return {"error": f"Session {request.session_id} not found"}
    
    return {
        "session_id": request.session_id,
        "created_at": session.created_at.isoformat(),
        "perspectives": {
            name: {
                "id": thread.id,
                "system_prompt": thread.system_prompt[:200] + "...",
                "message_count": len(thread.conversation_history),
                "model_backend": thread.model_backend.value
            }
            for name, thread in session.threads.items()
        },
        "analyses": [
            {
                "prompt": a["prompt"],
                "timestamp": a["timestamp"],
                "response_count": len(a["responses"]),
                "active_count": a["active_count"]
            }
            for a in session.analyses
        ]
    }

# Main entry point
def main():
    """Run the MCP server"""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
