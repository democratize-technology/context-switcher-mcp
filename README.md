# Context-Switcher MCP

A Model Context Protocol (MCP) server that enables multi-perspective analysis through parallel thread orchestration. Break free from single-perspective thinking by analyzing problems through multiple lenses simultaneously.

## Overview

Context-Switcher MCP implements the thread orchestration pattern to provide parallel evaluation from different viewpoints. Each perspective can contribute insights or abstain with `[NO_RESPONSE]` when not relevant.

## Features

- **Parallel Perspective Analysis**: Evaluate problems through multiple viewpoints simultaneously
- **Flexible Abstention**: Perspectives can opt out with `[NO_RESPONSE]` when not applicable
- **Multi-Provider Support**: Works with AWS Bedrock, LiteLLM, and Ollama
- **Session Management**: Maintain context across multiple analyses
- **Default Perspectives**: Technical, Business, User Experience, and Risk
- **Custom Perspectives**: Add domain-specific viewpoints on the fly
- **Synthesis**: Find patterns and insights across all perspectives

## Installation

### Prerequisites

- Python 3.10 or higher
- MCP-compatible client (e.g., Claude Desktop, Cline)
- At least one LLM provider configured:
  - AWS credentials for Bedrock
  - API keys for LiteLLM-supported models
  - Ollama running locally

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd context-switcher-mcp

# Install the package
pip install -e .
```

### Install for development

```bash
pip install -e ".[dev]"
```

## Configuration

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "context-switcher": {
      "command": "python",
      "args": ["-m", "context_switcher_mcp"]
    }
  }
}
```

### Cline

Add to your Cline MCP settings:

```json
{
  "context-switcher": {
    "command": "python",
    "args": ["-m", "context_switcher_mcp"]
  }
}
```


## Alternative Installation with UV

If you're using `uv` for Python package management, you can use the provided run script:

```bash
# Clone the repository
git clone <repository-url>
cd context-switcher-mcp

# Make the run script executable (if not already)
chmod +x run.sh

# The run script will handle everything
./run.sh
```

### Claude Desktop Configuration with run.sh

For a more robust setup using the run script:

```json
{
  "mcpServers": {
    "context-switcher": {
      "command": "/path/to/context-switcher-mcp/run.sh"
    }
  }
}
```

The run script will:
- Create a virtual environment if needed
- Install/update dependencies
- Start the MCP server

### Debug Mode

To enable debug logging:

```bash
DEBUG=1 ./run.sh
```

Or in Claude Desktop config:

```json
{
  "mcpServers": {
    "context-switcher": {
      "command": "/path/to/context-switcher-mcp/run.sh",
      "env": {
        "DEBUG": "1"
      }
    }
  }
}
```


### LLM Provider Setup

#### AWS Bedrock
Ensure AWS credentials are configured:
```bash
aws configure
```

#### LiteLLM
Set appropriate API keys:
```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

#### Ollama
Ensure Ollama is running:
```bash
ollama serve
ollama pull llama3.2  # or your preferred model
```

## Usage

### Starting an Analysis

```
Use context-switcher to analyze database architecture choices
```

The tool will initialize with default perspectives: technical, business, user, and risk.

### Adding Custom Perspectives

```
Add a "maintainability" perspective focusing on long-term code health
```

### Running Analysis

```
How should we handle user authentication in our new microservice?
```

Each perspective will evaluate the question, providing insights or abstaining if not relevant.

### Synthesizing Results

```
Synthesize the perspectives to find key insights
```

This will analyze patterns, conflicts, and unexpected connections across all viewpoints.

## Available Tools

1. **start_context_analysis** - Initialize multi-perspective analysis
2. **add_perspective** - Add custom viewpoints dynamically
3. **analyze_from_perspectives** - Broadcast questions to all perspectives
4. **synthesize_perspectives** - Find patterns across viewpoints
5. **list_sessions** - View all active analysis sessions
6. **get_session** - Get details of a specific session

## Default Perspectives

- **Technical**: Architecture, implementation, scalability, performance
- **Business**: ROI, market impact, strategic alignment, costs
- **User**: Experience, usability, accessibility, workflow impact
- **Risk**: Security, compliance, operational risks, privacy

## Examples

### Architecture Decision
```
Start context analysis for "choosing between REST and GraphQL"
Add "developer experience" perspective
Analyze: What are the tradeoffs between REST and GraphQL for our team?
Synthesize the results
```

### Feature Evaluation
```
Start context analysis for "implementing real-time notifications"
Analyze: Should we build real-time notifications in-house or use a service?
```

### Debugging Perspective Blindness
```
Start context analysis for "slow API response times"
Add "infrastructure" perspective
Add "database optimization" perspective
Analyze: Why are our API endpoints taking 5+ seconds to respond?
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
ruff check .
ruff format .
```

## Thread Orchestration Pattern

This MCP implements the thread orchestration pattern with:
- Parallel execution of all perspectives
- NoResponse handling for selective participation
- Provider flexibility per thread
- Conversation history per perspective

## Credits

Inspired by the thread orchestration pattern from multz and the thinking tools ecosystem including rubber-duck-mcp, sequential-thinking-mcp, and recursive-companion-mcp.

## License

MIT




## Bedrock Model Configuration

The Context-Switcher MCP defaults to using `us.anthropic.claude-3-7-sonnet-20250219-v1:0`. To use a different model:

### Option 1: Specify model when starting analysis
```python
start_context_analysis(
    topic="Your topic",
    model_backend="bedrock",
    model_name="us.anthropic.claude-opus-4-20250514-v1:0"
)
```

### Option 2: Environment variable (add to run.sh or Claude config)
```bash
export BEDROCK_MODEL_ID="us.anthropic.claude-opus-4-20250514-v1:0"
./run.sh
```

### Available Bedrock Models

**Anthropic Models (Inference Profiles)**:
- `us.anthropic.claude-3-7-sonnet-20250219-v1:0` (default)
- `us.anthropic.claude-opus-4-20250514-v1:0`
- `us.anthropic.claude-sonnet-4-20250514-v1:0`
- `us.anthropic.claude-3-5-haiku-20241022-v1:0`

**Meta Llama Models**:
- `meta.llama3-3-70b-instruct-v1:0`
- `meta.llama3-1-70b-instruct-v1:0`
- `meta.llama3-1-8b-instruct-v1:0`

**Note**: Use inference profile IDs (with "us." prefix) for Anthropic models to support on-demand throughput.


## LiteLLM Configuration

To use your LiteLLM instance at 192.168.5.111:4000:

### Get the API key from Kubernetes:
```bash
# Find the secret name
kubectl get secrets | grep litellm

# Extract the API key
kubectl get secret <secret-name> -o jsonpath='{.data.api-key}' | base64 -d
```

### Configure the endpoint:
```bash
export LITELLM_API_BASE="http://192.168.5.111:4000"
export LITELLM_API_KEY="<your-api-key>"
```

### Use in Context-Switcher:
```python
start_context_analysis(
    topic="Your topic",
    model_backend="litellm",
    model_name="gpt-4"  # or any model available on your LiteLLM instance
)
```

## Ollama Configuration

To use your Ollama instance at 192.168.1.250:

### Configure the endpoint:
```bash
export OLLAMA_HOST="http://192.168.1.250:11434"  # Replace with actual port
```

### Use in Context-Switcher:
```python
start_context_analysis(
    topic="Your topic",
    model_backend="ollama",
    model_name="llama3.2"  # or any model available on your Ollama instance
)
```

Note: Ollama requires no authentication.
