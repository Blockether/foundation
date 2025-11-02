# Developing Blockether Foundation

## Workflow

### Start with an issue before writing code

Before writing any code, please create an issue first that describes the problem
you are trying to solve with alternatives that you have considered. A little bit
of prior communication can save a lot of time on coding. Keep the problem as
small as possible. If there are two problems, make two issues. We discuss the
issue and if we reach an agreement on the approach, it's time to move on to a
PR.

### Follow up with a pull request

Post a corresponding PR with the smallest change possible to address the
issue. Then we discuss the PR, make changes as needed and if we reach an
agreement, the PR will be merged.

### Tests

Each bug fix, change or new feature should be tested well to prevent future
regressions.

If possible, tests should use public APIs. If the bug is in private/internal
code, try to trigger it from a public API.

### Force-push

Please do not use `git push --force` on your PR branch for the following
reasons:

- It makes it more difficult for others to contribute to your branch if needed.
- It makes it harder to review incremental commits.
- Links (in e.g. e-mails and notifications) go stale and you're confronted with:
  this code isn't here anymore, when clicking on them.
- GitHub Actions doesn't play well with it: it might try to fetch a commit which
  doesn't exist anymore.
- Your PR will be squashed anyway.

## Requirements

You need [Python 3.13+](https://www.python.org/downloads/) for development and [uv](https://docs.astral.sh/uv/) for package management. For testing and type checking you'll need the development dependencies.

You need `poe` installed globally via brew. Use `brew tap nat-n/poethepoet && brew install nat-n/poethepoet/poethepoet`

## Clone repository

```bash
git clone https://github.com/blockether/blockether-foundation.git
cd blockether-foundation
```

## Development Setup

Set up your development environment:

```bash
# Create virtual environment
uv venv

# Activate (Unix/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install in development mode with all dependencies
uv pip install -e ".[dev]"
```

## Running Tests

### Quick Development Cycle

```bash
# Run all tests
poe test

# Run only unit tests
poe test-unit

# Run tests with coverage
poe test-cov

# Watch for changes and re-run tests
poe test-watch
```

### Test Categories

- **Unit tests**: Fast tests that mock external dependencies (`tests/unit/`)
- **Integration tests**: Slower tests that make real API calls (`tests/integration/`)

By default, integration tests are skipped. Run them with:

```bash
poe test-integration
```

### Writing Tests

1. **Place unit tests** in `tests/unit/`
2. **Place integration tests** in `tests/integration/`
3. **Use pytest markers**: `@pytest.mark.unit` and `@pytest.mark.integration`
4. **Mock external dependencies** in unit tests using `unittest.mock`
5. **Test public APIs** when possible
6. **Each test should be independent** and not rely on test order

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
def test_encoder_function():
    # Arrange
    encoder = Encoder()

    # Act
    result = encoder.encode("test text")

    # Assert
    assert result is not None
    assert len(result) > 0

@pytest.mark.integration
def test_with_real_api():
    # Integration test with real dependencies
    pass
```

## Code Quality

### Type Checking

All new code must include type annotations. Run type checking with:

```bash
poe type-check
```

### Linting and Formatting

```bash
# Format code
poe format

# Check formatting
poe format-check

# Run linter
poe lint

# Auto-fix linting issues
poe lint-fix
```

### Running All Quality Checks

```bash
poe check-all
```

## Code Style

### Python Style Guide

- **Line length**: 100 characters (enforced by ruff)
- **Type hints**: Required for all new code
- **Docstrings**: Use Google-style docstrings
- **Import organization**: ruff will handle this automatically

### Type Annotations

```python
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class ExampleClass:
    """A brief description of the class.

    Attributes:
        name: The name of the example.
        value: An optional value.
    """
    name: str
    value: Optional[int] = None

    def process(self, input_data: List[str]) -> Dict[str, Any]:
        """Process the input data.

        Args:
            input_data: List of strings to process.

        Returns:
            Dictionary with processing results.
        """
        return {"processed": len(input_data)}
```

## Adding New Components

### 1. Create the Component

Add new components in the appropriate `src/blockether_foundation/` subdirectory:

```python
# src/blockether_foundation/new_component.py
from typing import List, Optional
from ..result import Result

class NewComponent:
    """Brief description of the component."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def process(self, data: List[str]) -> Result[List[str]]:
        """Process input data."""
        # Implementation here
        return Result.success(data)
```

### 2. Update Exports

Add the component to the appropriate `__init__.py`:

```python
# src/blockether_foundation/__init__.py
from .new_component import NewComponent

__all__ = ["NewComponent", ...]
```

### 3. Add Tests

Create corresponding tests:

```python
# tests/unit/test_new_component.py
import pytest
from blockether_foundation import NewComponent

@pytest.mark.unit
def test_new_component():
    component = NewComponent()
    result = component.process(["test"])
    assert result.is_success
```

### 4. Update Documentation

- Update README.md if the component is user-facing
- Add docstrings to all public methods
- Consider adding examples in the docstrings

## Dependencies

### Adding New Dependencies

1. **Add to pyproject.toml**:
   ```toml
   dependencies = [
       # ... existing dependencies
       "new-package>=1.0.0",
   ]
   ```

2. **Consider optional dependencies** for features not everyone needs:
   ```toml
   [project.optional-dependencies]
   extra = [
       "optional-package>=1.0.0",
   ]
   ```

3. **Update dependencies** regularly:
   ```bash
   uv pip install --upgrade -r requirements.txt
   ```

### Minimal Dependencies Principle

Keep the dependency footprint minimal. Each dependency should:
- Solve a specific problem that can't be easily solved internally
- Be actively maintained
- Have a compatible license
- Not conflict with existing dependencies

## Performance Considerations

### Memory Usage

Keep notes about how adding dependencies affects memory usage and startup time:

- **Current baseline**: ~50MB at startup
- **With embeddings**: ~120MB (Model2Vec model loading)
- **Agno integration**: ~80MB additional

### Optimization Tips

- Use `__slots__` for data classes when memory is critical
- Lazy-load heavy dependencies
- Use generators instead of lists where possible
- Profile memory usage for new features

## Windows Development

### PowerShell

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run tests
poe test

# Run linting
poe lint
```

### Command Prompt (cmd.exe)

```cmd
# Activate virtual environment
.venv\Scripts\activate.bat

# Run tests
poe test
```

### Git Configuration

Make sure Git handles line endings correctly:

```bash
git config --global core.autocrlf input
```

## Release Process

### Version Bumping

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`

### Automated Release

GitHub Actions will automatically:
- Run all tests and quality checks
- Build the package
- Create a GitHub release
- Publish to PyPI (if configured)

### Manual Release (if needed)

```bash
# Build package
uv build

# Check package
twine check dist/*

# Upload to PyPI (requires credentials)
twine upload dist/*
```

## Design Decisions

### Architecture

Some key design decisions:

#### Modular Structure

- Each module should be independently usable
- Avoid circular dependencies
- Use dependency injection for testability
- Keep interfaces simple and focused

#### Type Safety

- All public APIs must have type hints
- Use `Result` type for error handling instead of exceptions
- Validate inputs at API boundaries
- Use Pydantic for data validation when needed

#### Async Patterns

- Use async/await for I/O operations
- Provide sync wrappers for simple use cases
- Use `asyncio.run()` for entry points
- Consider thread safety for shared state

### Package Structure

Why we organized the package this way:

- **`encoder/`**: Text processing is a common need across AI agents
- **`ace/`**: AI Code Engine for programmatic AI workflows
- **`os/`**: System integration utilities
- **`assets/`**: Pre-trained models and static resources
- **`concurrency.py`**: Async utilities used across modules
- **`result.py``: Error handling pattern
- **`errors.py`**: Custom exception types

## Testing Strategy

### Test Pyramid

- **Unit tests**: 70% - Fast, isolated tests
- **Integration tests**: 20% - Test component interactions
- **End-to-end tests**: 10% - Test complete workflows

### Coverage Goals

- **Core modules**: 90%+ coverage
- **Utility functions**: 95%+ coverage
- **Integration points**: 80%+ coverage
- **Overall**: Maintain 85%+ coverage

### Test Data

- Use fixtures for common test data
- Avoid real API calls in unit tests
- Use property-based testing for pure functions
- Keep test data minimal and focused

## Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Code of Conduct**: Please follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.