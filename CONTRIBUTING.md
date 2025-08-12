# Contributing to Context-Switcher MCP

Thank you for your interest in contributing to Context-Switcher MCP! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Security](#security)

## Code of Conduct

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please refer to established open source community standards for guidance.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- [uv](https://docs.astral.sh/uv/) (recommended for dependency management)

### Areas for Contribution

We welcome contributions in the following areas:

- **Bug fixes**: Resolve issues in the issue tracker
- **Features**: Implement new MCP tools or perspectives
- **Documentation**: Improve README, API docs, or examples
- **Testing**: Add test coverage or improve existing tests
- **Performance**: Optimize response times or resource usage
- **Security**: Enhance security measures and practices

## Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/context-switcher-mcp.git
   cd context-switcher-mcp
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"

   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify installation**:
   ```bash
   python -m context_switcher_mcp --help
   ```

## Making Changes

### Branch Strategy

- Create a feature branch from `main`: `git checkout -b feature/your-feature-name`
- Use descriptive branch names that reflect the change type:
  - `feature/add-new-perspective`
  - `fix/circuit-breaker-timing`
  - `docs/api-examples`

### Development Workflow

1. **Make your changes** in logical, focused commits
2. **Add tests** for any new functionality
3. **Update documentation** if needed
4. **Run the test suite** locally before submitting

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=context_switcher_mcp --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py -v

# Run tests matching a pattern
pytest -k "test_session" -v
```

### Code Quality Checks

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass** and code quality checks succeed
2. **Update documentation** if you've changed APIs or added features
3. **Write clear commit messages** following conventional commit format:
   ```
   feat: add new perspective template system
   fix: resolve circuit breaker timing issue
   docs: update API documentation
   ```

4. **Submit a pull request** with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - References to any related issues
   - Screenshots or examples if applicable

### Pull Request Guidelines

- **Keep changes focused**: One feature or fix per PR
- **Write tests**: New functionality should include tests
- **Document changes**: Update README or docs as needed
- **Follow the template**: Use the provided PR template
- **Be responsive**: Address feedback promptly and professionally

## Style Guidelines

### Python Code Style

- **Follow PEP 8**: Use ruff for linting and formatting
- **Type hints**: Add type annotations for all public functions
- **Docstrings**: Use Google-style docstrings for all public methods
- **Error handling**: Include proper exception handling and validation

### Example Code Style

```python
async def analyze_from_perspectives(
    session_id: str,
    prompt: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Analyze a prompt from multiple perspectives.

    Args:
        session_id: The session identifier.
        prompt: The question or topic to analyze.
        timeout: Maximum time to wait for responses.

    Returns:
        A dictionary containing perspective responses and metadata.

    Raises:
        SessionNotFoundError: If the session doesn't exist.
        AnalysisTimeoutError: If analysis exceeds timeout.
    """
    session = await self._get_session(session_id)
    # ... implementation
```

### Commit Message Guidelines

Use conventional commit format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code formatting changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tooling changes

## Testing

### Test Categories

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Security tests**: Test security boundaries and validation
- **Performance tests**: Test response times and resource usage

### Writing Tests

```python
import pytest
from context_switcher_mcp.orchestrator import ThreadOrchestrator

@pytest.mark.asyncio
async def test_perspective_analysis():
    """Test that perspectives can analyze prompts correctly."""
    orchestrator = ThreadOrchestrator()

    result = await orchestrator.analyze_prompt(
        session_id="test-session",
        prompt="Test question",
    )

    assert "responses" in result
    assert len(result["responses"]) > 0
```

### Test Requirements

- All tests must pass before merging
- New functionality requires tests
- Aim for >80% code coverage
- Include both success and failure scenarios

## Security

### Security Guidelines

- **Input validation**: Validate all inputs and parameters
- **Error handling**: Don't expose sensitive information in errors
- **Dependencies**: Keep dependencies updated and scan for vulnerabilities
- **Secrets**: Never commit API keys or sensitive configuration

### Reporting Security Issues

Please report security vulnerabilities through our [Security Policy](SECURITY.md). Do not open public issues for security concerns.

## Getting Help

### Resources

- **Documentation**: Check the README and inline documentation
- **Issues**: Browse existing issues for similar problems
- **Discussions**: Start a discussion for questions or ideas

### Communication

- **Be respectful**: Follow community guidelines
- **Be patient**: Maintainers volunteer their time
- **Be helpful**: Help others when you can

### Issue Reporting

When reporting bugs:

1. **Use the bug report template**
2. **Include reproduction steps**
3. **Provide environment information**
4. **Include error messages and logs**

## Recognition

Contributors are recognized in:

- **CHANGELOG.md**: Major contributions noted in releases
- **GitHub contributors**: Automatic recognition
- **Release notes**: Significant contributions highlighted

Thank you for contributing to Context-Switcher MCP! Your efforts help make this project better for everyone.
