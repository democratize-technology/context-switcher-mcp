
#!/usr/bin/env python3
"""
Context-Switcher MCP Server
Multi-perspective analysis using thread orchestration
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .compression import prepare_synthesis_input
from .models import ModelBackend, Thread, ContextSwitcherSession
from .orchestrator import ThreadOrchestrator, NO_RESPONSE
from .session_manager import SessionManager
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

# Initialize session manager
session_manager = SessionManager(
    max_sessions=100,
    session_ttl_hours=24,
    cleanup_interval_minutes=30
)

# Helper functions
def validate_session_id(session_id: str) -> bool:
    """Validate session ID format and existence"""
    if not session_id or not isinstance(session_id, str):
        return False
    if len(session_id) > 100:  # Reasonable limit
        return False
    return session_manager.get_session(session_id) is not None

def validate_topic(topic: str) -> bool:
    """Validate topic string"""
    if not topic or not isinstance(topic, str):
        return False
    if len(topic.strip()) == 0 or len(topic) > 1000:  # Reasonable limits
        return False
    return True

# Create shared orchestrator instance
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
    
    # Store session in manager
    if not session_manager.add_session(session):
        return {"error": "Session limit reached. Please try again later."}
    
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
    session = session_manager.get_session(request.session_id)
    
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
    session = session_manager.get_session(request.session_id)
    
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
    session = session_manager.get_session(request.session_id)
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
    
    # Determine token limit based on backend
    first_thread = list(session.threads.values())[0]
    if first_thread.model_backend == ModelBackend.BEDROCK:
        # Claude models have larger context windows
        max_chars = 20000 if "opus" in (first_thread.model_name or "").lower() else 12000
    elif first_thread.model_backend == ModelBackend.LITELLM:
        # Conservative for various models
        max_chars = 10000
    else:
        # Ollama models vary widely
        max_chars = 8000
    
    # Prepare synthesis prompt with compression
    compressed_text, compression_stats = prepare_synthesis_input(active, max_total_chars=max_chars)
    
    # Log compression stats
    logger.info(
        f"Synthesis compression: {compression_stats['perspectives']} perspectives, "
        f"{compression_stats['original_chars']:,} -> {compression_stats['final_chars']:,} chars "
        f"({compression_stats['compression_ratio']} reduction), "
        f"~{compression_stats['final_tokens']:,} tokens"
    )
    
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
    active_sessions = session_manager.list_active_sessions()
    
    for sid, session in active_sessions.items():
        session_list.append({
            "session_id": sid,
            "created_at": session.created_at.isoformat(),
            "perspectives": list(session.threads.keys()),
            "analyses_count": len(session.analyses),
            "topic": getattr(session, 'topic', 'Unknown')
        })
    
    # Get session manager stats
    stats = session_manager.get_stats()
    
    return {
        "sessions": session_list,
        "total_sessions": len(active_sessions),
        "stats": stats
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
    active_sessions = session_manager.list_active_sessions()
    
    if not active_sessions:
        return {
            "status": "No active sessions",
            "hint": "Start with: start_context_analysis"
        }
    
    # Get most recent session
    recent_session_id = max(active_sessions.keys(), 
                           key=lambda sid: active_sessions[sid].created_at)
    session = active_sessions[recent_session_id]
    
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
    session = session_manager.get_session(request.session_id)
    if not session:
        return {"error": f"Session {request.session_id} not found or expired"}
    
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

# Note: Session cleanup task should be started manually when needed
# FastMCP doesn't have startup/shutdown hooks in this version

# Main entry point
def main():
    """Run the MCP server"""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
