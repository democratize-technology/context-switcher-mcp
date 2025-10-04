# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Context Switcher MCP Server** - a Python-based Model Context Protocol (MCP) server that enables parallel analysis from multiple perspectives using different LLM backends. It implements the AI-Optimized Response Protocol (AORP) for structured, actionable responses.

## Architecture

The codebase follows a modular, service-oriented architecture:

- **MCP Server Pattern**: Exposes tools to AI assistants via the Model Context Protocol
- **Thread Orchestration**: Parallel execution of multiple LLM queries with different perspectives via `orchestrator.py`
- **Session Management**: Stateful sessions with TTL and automatic cleanup in `session_manager.py`
- **Provider Abstraction**: Supports AWS Bedrock, LiteLLM, and Ollama backends through `models.py`
- **AORP Implementation**: Structured response format optimized for AI workflows in `aorp.py`
- **Circuit Breaker Pattern**: Resilient backend failure handling with automatic recovery

Key components:
- `src/context_switcher_mcp/__init__.py` - MCP tool definitions and server setup
- `src/context_switcher_mcp/orchestrator.py` - Core thread orchestration logic for parallel perspectives
- `src/context_switcher_mcp/session_manager.py` - Session lifecycle and state management
- `src/context_switcher_mcp/aorp.py` - AI-Optimized Response Protocol builder and formatter
- `src/context_switcher_mcp/compression.py` - Text compression for synthesis operations
- `src/context_switcher_mcp/templates.py` - Pre-configured analysis perspectives
- `src/context_switcher_mcp/circuit_breaker_store.py` - Persistent circuit breaker state management

## Development Commands

### Transport Modes

The server supports two transport modes:

**Stdio Transport (default - for Claude Desktop):**
```bash
# Direct execution
python -m context_switcher_mcp

# Via convenience script
./run.sh
```

**HTTP Transport (for network access):**
```bash
# Enable HTTP transport with environment variables
export MCP_TRANSPORT=http
export MCP_HTTP_HOST=127.0.0.1  # Optional, defaults to 127.0.0.1
export MCP_HTTP_PORT=8082       # Optional, defaults to 8082
python -m context_switcher_mcp

# Or inline:
MCP_TRANSPORT=http MCP_HTTP_PORT=8082 python -m context_switcher_mcp
```

HTTP transport is useful for:
- Integration with web-based MCP clients
- Load balancing and horizontal scaling
- Network-based debugging with tools like curl/httpie
- Multi-client access to the same server instance

### Development Tasks

```bash
# Install for development (includes pre-commit hooks)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_aorp_core.py -v

# Linting and formatting
ruff check .
ruff format .

# Run pre-commit hooks manually
pre-commit run --all-files
```

## Key Implementation Patterns

1. **Async-First Design**: All operations use Python's asyncio for concurrent LLM queries
2. **Thread Safety**: The orchestrator manages parallel threads with proper synchronization
3. **Abstention Handling**: Perspectives can return `[NO_RESPONSE]` when queries are outside their expertise
4. **Session Cleanup**: Automatic cleanup of expired sessions (default TTL: 1 hour)
5. **Error Propagation**: Model errors are captured and returned in AORP format
6. **Metrics Collection**: ThreadMetrics and OrchestrationMetrics track performance and errors
7. **Streaming Support**: Real-time response streaming for immediate feedback

## Testing Approach

- Test files mirror source structure in `tests/`
- Use `pytest-asyncio` for async test support
- Mock LLM responses for deterministic testing
- Test both success paths and error conditions
- AORP-specific tests in `test_aorp_*.py` files validate response structure

## Configuration

The server accepts configuration via command-line arguments or environment variables:
- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 3023)
- LLM provider credentials (AWS_PROFILE, LITELLM_API_KEY, etc.)
- `LOG_LEVEL`: Set to DEBUG for detailed logging
- `DEBUG=1`: Enable debug mode when using run.sh

## MCP Tools Exposed

1. `start_context_analysis` - Initialize a multi-perspective session
2. `add_perspective` - Add perspectives dynamically to a session
3. `analyze_from_perspectives` - Broadcast a query to all perspectives (returns AORP)
4. `analyze_from_perspectives_stream` - Real-time streaming analysis
5. `synthesize_perspectives` - Find patterns across perspective responses
6. `recommend_perspectives` - AI-powered perspective suggestions
7. `get_session` - Retrieve specific session state
8. `list_sessions` - View all active sessions
9. `current_session` - Quick access to most recent session
10. `get_performance_metrics` - Operational health and metrics
11. `reset_circuit_breakers` - Admin operation for recovery
