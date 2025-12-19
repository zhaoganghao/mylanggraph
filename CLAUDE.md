# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangGraph project that implements an AI agent with tool calling capabilities. The main agent performs arithmetic operations using LLM tool calling. The project is structured as a Python package with LangGraph workflows.

## Key Files and Architecture

- `src/agent/graph.py`: Main graph implementation with LLM tool calling workflow
- `src/agent/workflowdemo*.py`: Additional workflow examples with different configurations
- `langgraph.json`: Configuration file defining graphs and dependencies for LangGraph Server
- `pyproject.toml`: Project dependencies and metadata

## Development Commands

### Installation
```bash
# Install dependencies and LangGraph CLI
pip install -e . "langgraph-cli[inmem]"

# Or using uv (preferred in CI)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
uv pip install -r pyproject.toml
```

### Running the Application
```bash
# Start LangGraph Server in development mode
langgraph dev
```

### Testing
```bash
# Run unit tests
uv run pytest tests/unit_tests

# Run integration tests
uv run pytest tests/integration_tests

# Run specific test file
uv run pytest tests/unit_tests/test_configuration.py
```

### Linting and Type Checking
```bash
# Lint with ruff
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Type checking with mypy
uv run mypy --strict src/
```

### Spell Checking
```bash
# Check README spelling
codespell README.md

# Check code spelling
codespell src/
```

## Code Architecture

The main agent in `src/agent/graph.py` implements a tool-calling workflow with these components:

1. **Tools**: Simple arithmetic functions (add, multiply, divide)
2. **Nodes**:
   - `llm_call`: LLM that decides whether to call tools
   - `tool_node`: Executes tool calls and returns results
3. **Edges**: Conditional routing based on whether LLM made tool calls
4. **State**: Uses `MessagesState` for conversation history

## Workflow Structure

The agent follows this pattern:
1. LLM receives user input and decides whether to call tools
2. If tools are called, execute them and return results
3. LLM receives tool results and generates final response
4. Loop continues until LLM decides no more tools are needed

## Configuration

- `langgraph.json`: Defines which graphs are exposed by the server
- Environment variables in `.env` (copy `.env.example`)
- PostgreSQL checkpointing available in some workflow examples

## Testing

Tests are organized into:
- `tests/unit_tests/`: Fast tests for individual components
- `tests/integration_tests/`: Slower tests that may call external APIs

Use pytest markers:
- `@pytest.mark.anyio`: For async tests
- `@pytest.mark.langsmith`: For tests that should run with LangSmith tracing