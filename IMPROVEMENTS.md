# Context-Switcher MCP Improvements

## Features Added for Better AI Experience

### 1. Perspective Templates 🎯
Pre-configured perspective sets for common analysis patterns:
- **architecture_decision**: Technical, business, user, risk + scalability, team capability, migration
- **feature_evaluation**: User, business, technical + competitive, support burden, timeline  
- **debugging_analysis**: Technical, risk + performance, data flow, dependencies, history
- **api_design**: Technical, user + versioning, documentation, consistency, extensibility
- **security_review**: Risk, technical, business + attack surface, compliance, incident response

**Usage**: `start_context_analysis(topic="...", template="architecture_decision")`

### 2. Enhanced Tool Descriptions 📝
More specific, evocative descriptions that include:
- Concrete use cases and examples
- Expected behavior and timing
- When to reach for each tool
- What to expect from results

### 3. Better Error Messages 🎯
Actionable error messages that suggest fixes:
- Bedrock: "Try: us.anthropic.claude-3-7-sonnet-20250219-v1:0"
- LiteLLM: "Set OPENAI_API_KEY environment variable"
- Ollama: "Is it running? Set OLLAMA_HOST=http://your-host:11434"

### 4. New Tools 🛠️

#### list_templates()
Shows all available perspective templates with their perspectives and usage examples.

#### current_session()
Quick access to your most recent session without remembering IDs. Shows:
- Current perspectives
- Last analysis summary
- Suggested next steps

### 5. Session Context 📍
Sessions now remember their topic, making it easier to track multiple analyses.

## Why These Matter

**Templates** eliminate the "blank canvas" problem - you don't have to think about which perspectives to include.

**Better descriptions** create "reach moments" - you know exactly when to use each tool.

**Actionable errors** turn frustration into forward progress - you know what to fix.

**Session helpers** reduce cognitive load - less mental bookkeeping, more thinking about your problem.

## Usage Flow

1. **See what's available**: `list_templates()`
2. **Start with a template**: `start_context_analysis(topic="Should we use GraphQL?", template="api_design")`
3. **Check your session**: `current_session()`
4. **Analyze**: `analyze_from_perspectives(prompt="What about performance?")`
5. **Find insights**: `synthesize_perspectives()`

## The Philosophy

These improvements follow the principle that **tools should disappear into the workflow**. When you're managing session IDs or wondering which perspectives to add, you're not thinking about your problem. 

Good tools feel inevitable, not clever. They should make you feel powerful, not confused.

---

*Built with love for AI assistants who think about thinking* 🤖💭


## Latest Fix: Token Limit Management 🔧

### Problem
When synthesizing many perspectives with long responses, Bedrock would fail due to token limits.

### Solution
Added intelligent compression that:
- Limits each perspective to ~2,000 characters before synthesis
- Truncates at sentence boundaries for readability  
- Keeps total synthesis input under 12,000 characters (~3,000 tokens)
- Logs token estimates for debugging

### Implementation
- Created `compression.py` with smart truncation utilities
- Updated `synthesize_perspectives` to compress before sending to Bedrock
- Added logging to track token usage

This ensures synthesis works reliably even with many detailed perspective responses!
