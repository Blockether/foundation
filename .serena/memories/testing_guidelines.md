# Testing Guidelines for Blockether Foundation

## Testing Philosophy

The project follows a comprehensive testing strategy with three main categories:

1. **Unit Tests** - Fast tests with mocked dependencies
2. **Integration Tests** - Tests with real external dependencies
3. **Slow Tests** - Performance-intensive tests

## Test Structure and Organization

### Directory Structure
```
tests/
├── __init__.py               # Test package configuration
├── conftest.py               # Pytest configuration and fixtures
├── unit/                     # Unit tests (fast, mocked)
│   ├── test_result.py         # Result type tests
│   ├── test_errors.py         # Error handling tests
│   ├── test_encoder_potion.py # Encoder tests
│   ├── test_telegram_interface.py # Telegram interface tests
│   └── ...
└── integration/              # Integration tests (slow, real APIs)
    ├── test_end_to_end.py     # Full workflow tests
    ├── test_real_apis.py      # Real LLM API calls
    └── ...
```

### Test Naming Conventions
- **Files**: `test_*.py` prefix
- **Classes**: `Test*` prefix
- **Functions**: `test_*` prefix

## Unit Testing Guidelines

### When to Use Unit Tests
- Testing pure functions and business logic
- Testing error handling patterns
- Testing data validation and transformation
- Testing Result type operations
- Testing model validation

### Unit Test Pattern
```python
import pytest
from unittest.mock import Mock, patch
from blockether_foundation.result import Result, ResultError
from blockether_foundation.errors import FoundationBaseError

class TestResult:
    """Test suite for Result type functionality."""
    
    def test_ok_result_creation(self):
        """Test creating an Ok result."""
        result = Result.Ok("test_value")
        assert result.is_ok()
        assert not result.is_err()
        assert result.unwrap() == "test_value"
    
    def test_err_result_creation(self):
        """Test creating an Err result."""
        error = TestError("test error")
        result = Result.Err(error)
        assert result.is_err()
        assert not result.is_ok()
        assert result.unwrap_err() == error
    
    def test_result_map_success(self):
        """Test mapping over a successful result."""
        result = Result.Ok(42)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_ok()
        assert mapped.unwrap() == 84
    
    def test_result_map_failure(self):
        """Test mapping over an error result."""
        error = TestError("original error")
        result = Result.Err(error)
        mapped = result.map(lambda x: x * 2)
        assert mapped.is_err()
        assert mapped.unwrap_err() == error
    
    @patch('external_module.expensive_function')
    def test_with_mocked_dependency(self, mock_function):
        """Test function with mocked external dependency."""
        mock_function.return_value = "mocked_value"
        
        result = function_that_uses_dependency()
        
        mock_function.assert_called_once()
        assert result == Result.Ok("processed_mocked_value")

class TestError(FoundationBaseError):
    """Test error for unit testing."""
    pass
```

### Mocking Patterns

#### Mocking External Dependencies
```python
from unittest.mock import Mock, patch

@patch('blockether_foundation.encoder.potion.StaticModel')
def test_encoder_with_mock_model(mock_static_model):
    """Test encoder with mocked model."""
    mock_model_instance = Mock()
    mock_static_model.from_pretrained.return_value = mock_model_instance
    
    from blockether_foundation.encoder.potion import PotionEncoder
    
    encoder = PotionEncoder.get_instance()
    result = encoder.encode(["test"])
    
    mock_static_model.from_pretrained.assert_called_once()
    assert result is not None
```

#### Mocking Database Operations
```python
@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    session = Mock()
    session.query.return_value.all.return_value = []
    session.add.return_value = None
    session.commit.return_value = None
    return session

def test_database_operation(mock_db_session):
    """Test database operation with mocked session."""
    from blockether_foundation.storage import save_to_database
    
    result = save_to_database("test_data", mock_db_session)
    
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
```

## Integration Testing Guidelines

### When to Use Integration Tests
- Testing complete workflows
- Testing real API integrations
- Testing database operations
- Testing external service connections

### Integration Test Pattern
```python
import pytest
from blockether_foundation.os.interfaces.telegram import Telegram
from agno.agent import Agent
from agno.models.openai import OpenAIChat

@pytest.mark.integration
@pytest.mark.slow
class TestTelegramIntegration:
    """Integration tests for Telegram interface."""
    
    @pytest.fixture
    def telegram_config(self):
        """Create Telegram configuration for testing."""
        return {
            "bot_token": os.getenv("TEST_TELEGRAM_TOKEN"),
            "webhook_url": None,  # Use polling for tests
        }
    
    @pytest.fixture
    def test_agent(self):
        """Create test agent with mocked LLM calls."""
        # Use mocked model for integration tests
        return Agent(
            name="Test Agent",
            model=Mock(),  # Mock the model to avoid real API calls
            instructions="Test instructions"
        )
    
    def test_complete_workflow(self, test_agent, telegram_config):
        """Test complete Telegram workflow."""
        telegram = Telegram(
            agent=test_agent,
            bot_token=telegram_config["bot_token"],
            prefix="/test"
        )
        
        # Test router creation
        router = telegram.get_router()
        assert router is not None
        
        # Test configuration
        assert telegram.agent == test_agent
        assert telegram._get_entity_type() == "agent"
```

## Testing Fixtures

### Common Fixtures
```python
# conftest.py
import pytest
import tempfile
import os
from unittest.mock import Mock

@pytest.fixture
def temp_directory():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_llm_model():
    """Mock LLM model for testing."""
    model = Mock()
    model.run.return_value = Mock(content="Test response")
    model.arun.return_value = Mock(content="Test response")
    return model

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    }
```

### Async Test Fixtures
```python
import pytest
import asyncio

@pytest.fixture
async def async_client():
    """Async test client fixture."""
    from fastapi.testclient import TestClient
    from blockether_foundation.app import create_app
    
    app = create_app()
    with TestClient(app) as client:
        yield client

@pytest.mark.asyncio
async def test_async_function(async_client):
    """Test async functionality."""
    response = async_client.post("/test-endpoint", json={"data": "test"})
    assert response.status_code == 200
```

## Test Markers and Categories

### Markers Usage
```python
import pytest

@pytest.mark.unit
def test_pure_function():
    """Unit test marker."""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_real_api_call():
    """Integration test with real API."""
    pass

@pytest.mark.parametrize("input_data,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_data_transformation(input_data, expected):
    """Parametrized test."""
    result = transform_data(input_data)
    assert result == expected
```

### Running Tests by Category
```bash
# Default: unit tests only
poe test

# Integration tests only
poe test-integration

# Slow tests only
pytest tests/ -m slow

# Specific test files
pytest tests/unit/test_result.py -v

# All tests including integration
pytest tests/ -v --cov=blockether_foundation
```

## Testing Result Types

### Testing Success Paths
```python
def test_successful_operation():
    """Test successful operation returns Ok result."""
    result = safe_operation("valid_input")
    assert result.is_ok()
    assert result.unwrap() == "expected_output"

def test_result_chaining():
    """Test Result type chaining operations."""
    result = (Result.Ok(5)
              .map(lambda x: x * 2)
              .map(lambda x: x + 1))
    
    assert result.is_ok()
    assert result.unwrap() == 11
```

### Testing Error Paths
```python
def test_error_handling():
    """Test error handling returns Err result."""
    result = risky_operation("invalid_input")
    assert result.is_err()
    
    error = result.unwrap_err()
    assert isinstance(error, ValidationError)
    assert "validation failed" in str(error).lower()

def test_error_recovery():
    """Test error recovery with unwrap_or."""
    result = risky_operation("invalid_input")
    fallback = result.unwrap_or("default_value")
    assert fallback == "default_value"
```

## Testing Async Code

### Async Test Patterns
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_operation()
    assert result == "expected_result"

@pytest.mark.asyncio
async def test_async_result_handling():
    """Test async function returning Result."""
    result = await async_risky_operation()
    
    if result.is_ok():
        assert await process_success(result.unwrap()) is True
    else:
        assert await handle_error(result.unwrap_err()) is False
```

### Async Mocking
```python
@pytest.mark.asyncio
@patch('async_module.async_function')
async def test_async_with_mock(mock_async_function):
    """Test async function with mocked dependency."""
    mock_async_function.return_value = "mocked_result"
    
    result = await function_that_calls_async()
    
    mock_async_function.assert_called_once()
    assert result == "processed_mocked_result"
```

## Coverage and Quality Metrics

### Coverage Configuration
```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = [
    "--cov=blockether_foundation",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80",  # Fail if coverage below 80%
]

[tool.coverage.run]
source = ["src/blockether_foundation"]
omit = ["*/tests/*", "*/__pycache__/*"]
```

### Coverage Targets
- **Line Coverage**: Minimum 80%
- **Branch Coverage**: Minimum 70%
- **Function Coverage**: Minimum 90%

## Performance Testing

### Timing Tests
```python
import time
import pytest

def test_performance_critical_function():
    """Test performance of critical function."""
    start_time = time.time()
    
    result = critical_function(large_dataset)
    
    elapsed = time.time() - start_time
    assert elapsed < 1.0  # Should complete within 1 second
    assert result is not None
```

### Memory Testing
```python
def test_memory_usage():
    """Test memory usage of memory-intensive functions."""
    import tracemalloc
    
    tracemalloc.start()
    
    result = memory_intensive_function()
    
    current, peak = tracemalloc.get_traced_memory()
    
    # Ensure reasonable memory usage
    assert peak < 10_000_000  # Less than 10MB
    assert result is not None
    
    tracemalloc.stop()
```

## Test Data Management

### Test Data Factories
```python
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_agent(name: str = "Test Agent") -> dict:
        """Create test agent data."""
        return {
            "name": name,
            "model": "gpt-4o-mini",
            "instructions": f"Test instructions for {name}"
        }
    
    @staticmethod
    def create_playbook(name: str = "Test Playbook") -> dict:
        """Create test playbook data."""
        return {
            "name": name,
            "overview": "Test overview",
            "policies": ["policy1", "policy2"],
            "ground_truths": []
        }

def test_with_factory_data():
    """Test using factory data."""
    agent_data = TestDataFactory.create_agent()
    agent = Agent(**agent_data)
    
    assert agent.name == "Test Agent"
    assert agent.instructions == "Test instructions for Test Agent"
```

## CI/CD Integration

### GitHub Actions Example
```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.13'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        poe test-cov
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./htmlcov/index.html
```

## Testing Best Practices

### Do's
1. **Test both success and failure paths** for Result types
2. **Use descriptive test names** that explain what's being tested
3. **Mock external dependencies** to keep tests fast and reliable
4. **Use fixtures** for common test setup
5. **Parametrize tests** for different input combinations
6. **Add assertions** for both positive and negative cases
7. **Document complex test scenarios** with comments

### Don'ts
1. **Don't test external libraries** - assume they work
2. **Don't make real API calls** in unit tests
3. **Don't ignore test failures** - all tests should pass
4. **Don't use production data** in tests
5. **Don't hard-code absolute paths** - use relative paths
6. **Don't sleep in tests** - use mocks instead
7. **Don't ignore coverage reports** - maintain good coverage

## Troubleshooting Common Issues

### Import Errors in Tests
```python
# Ensure tests can import the module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
```

### Async Test Failures
```python
# Ensure pytest-asyncio is installed and configured
# Add to pyproject.toml:
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### Mock Not Working
```python
# Check patch path matches the actual import
# Use python -c "import module; print(module.__file__)" to verify paths
@patch('package.module.function')  # Must match actual import
def test_function():
    pass
```

This testing framework ensures comprehensive coverage of all functionality while keeping tests fast, reliable, and maintainable.