# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Context Switcher MCP Server** - a Python-based Model Context Protocol (MCP) server that enables parallel analysis from multiple perspectives using different LLM backends. It's designed to help AI assistants analyze problems from various viewpoints simultaneously.

## Architecture

The codebase follows a modular, service-oriented architecture:

- **MCP Server Pattern**: Exposes tools to AI assistants via the Model Context Protocol
- **Thread Orchestration**: Parallel execution of multiple LLM queries with different perspectives via `orchestrator.py`
- **Session Management**: Stateful sessions with TTL and automatic cleanup in `session_manager.py`
- **Provider Abstraction**: Supports AWS Bedrock, LiteLLM, and Ollama backends through `models.py`

Key components:
- `src/context_switcher_mcp/__init__.py` - MCP tool definitions and server setup
- `src/context_switcher_mcp/orchestrator.py` - Core thread orchestration logic for parallel perspectives
- `src/context_switcher_mcp/session_manager.py` - Session lifecycle and state management
- `src/context_switcher_mcp/compression.py` - Text compression for synthesis operations
- `src/context_switcher_mcp/templates.py` - Pre-configured analysis perspectives

## Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run the server
python -m context_switcher_mcp

# Alternative: Use the convenience script (creates venv automatically)
./run.sh

# Run tests
pytest tests/

# Linting and formatting
ruff check .
ruff format .
```

## Key Implementation Patterns

1. **Async-First Design**: All operations use Python's asyncio for concurrent LLM queries
2. **Thread Safety**: The orchestrator manages parallel threads with proper synchronization
3. **Abstention Handling**: Perspectives can return `[NO_RESPONSE]` when queries are outside their expertise
4. **Session Cleanup**: Automatic cleanup of expired sessions (default TTL: 1 hour)
5. **Error Propagation**: Model errors are captured and returned in thread responses

## Testing Approach

- Test files mirror source structure in `tests/`
- Use `pytest-asyncio` for async test support
- Mock LLM responses for deterministic testing
- Test both success paths and error conditions

## Configuration

The server accepts configuration via command-line arguments or environment variables:
- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 3023)
- LLM provider credentials (AWS_PROFILE, LITELLM_API_KEY, etc.)

## MCP Tools Exposed

1. `start_context_analysis` - Initialize a multi-perspective session
2. `add_perspective` - Add perspectives dynamically to a session
3. `analyze_from_perspectives` - Broadcast a query to all perspectives
4. `synthesize_perspectives` - Find patterns across perspective responses
5. `get_session_info` - Retrieve current session state
6. `list_active_sessions` - View all active sessions
7. `end_session` - Manually terminate a session
