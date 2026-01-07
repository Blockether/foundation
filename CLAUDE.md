# General context

## Project Overview

Python 3.13+ project using `uv` for package management. Source code in `src/blockether_foundation/`, tests in `tests/unit/` and `tests/integration/`.

## Package Management

**Use uv only.** All dependency management goes through `pyproject.toml`.

```bash
# Install dependencies
uv sync --all-extras

# Add a dependency
uv add package-name

# Run commands in venv
uv run <command>
```

## Inspecting Library Code

When you need to understand how a library works, you can read its source code directly:

```bash
# Path to installed packages
.venv/lib/python3.13/site-packages/[library-name]/
```

Example: `read .venv/lib/python3.13/site-packages/requests/sessions.py`

## Running Tests

### Run all tests
```bash
uv run pytest
```

### Run single test file
```bash
uv run pytest tests/path/to/test_file.py
```

### Run specific test
```bash
uv run pytest tests/path/to/test_file.py::test_function_name
```

### With markers
```bash
# Run unit tests (default)
uv run pytest -m unit

# Run integration tests
uv run pytest -m integration

# Run async tests
uv run pytest -m asyncio
```

Available markers: `unit`, `integration`, `agno_eval`, `performance_test`, `slow`, `asyncio`

## Quality Tools

```bash
# Linting and formatting (ruff)
uv run ruff check .
uv run ruff format .

# Type checking (pyright)
uv run pyright src/
```

## Task Runner

Use `poe` for common tasks:
```bash
uv run poe lint
uv run poe test
uv run poe test-integration
```

## Core Utilities

### Concurrency (`src/blockether_foundation/concurrency.py`)

Use `ConcurrentProcessor` for concurrent batch processing with automatic retry logic:

```python
from blockether_foundation.concurrency import ConcurrentProcessor

processor = ConcurrentProcessor(
    concurrency=5,           # Max parallel operations
    max_retries=3,           # Retry attempts
    retry_min_wait=3500,    # Min wait between retries (ms)
    retry_max_wait=15000,   # Max wait between retries (ms)
)

results = await processor.process(items, processor_fn)
```

Guarantees:
- Order preservation (results returned in same order as inputs)
- Atomic processing (all items succeed or all fail)
- Exponential backoff retry logic

**Important**: When returning multiple values from processor_fn, wrap tuples as lists:
```python
# Correct
async def processor_fn(item) -> list[Output]:
    return [result1, result2]

# Also works (tuples are treated as sequences)
async def processor_fn(item) -> tuple[Output, ...]:
    return (result1, result2)
```

### Error Handling (`src/blockether_foundation/result.py`)

Use `Result[T, E]` for explicit error handling (Rust-like):

```python
from blockether_foundation.result import Result
from blockether_foundation.errors import FoundationBaseError

# Create results
result = Result.Ok(value)
result = Result.Err(error)

# Extract values
value = result.unwrap()               # Raises if Err
value = result.unwrap_or(default)     # Returns default if Err
value = result.expect("message")      # Raises with custom message

# Chain operations
result.map(transform_fn)              # Transform Ok value
result.and_then(lambda x: ...)        # Chain Result-returning operations
result.or_else(lambda e: ...)         # Provide fallback if Err
```

## Code Conventions

- Type hints required (checked by pyright)
- Use async/await for I/O operations
- Follow ruff formatting
- Write tests for new features
- Use `Result` for explicit error handling
- Use `ConcurrentProcessor` for batch concurrent operations
