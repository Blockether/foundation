# Code Style and Conventions for Blockether Foundation

## Python Style Standards

### Python Version
- **Target**: Python 3.13+ for development
- **Mypy**: Python 3.14+ for type checking
- Use `from __future__ import annotations` at top of files

### Type Hints
- **Required**: All function signatures must have type hints
- **Strict**: No implicit optional types
- **Modern**: Use `X | None` syntax instead of `Optional[X]`
- **Generic Types**: Use `TypeVar` for generic classes and functions

```python
from __future__ import annotations
from typing import TypeVar, Generic

T = TypeVar("T")

def process_data(data: list[str]) -> dict[str, int]:
    """Process data and return results."""
    return {item: len(item) for item in data}
```

### Import Organization
- Use `isort` for import organization (handled by Ruff)
- Group imports: standard library, third-party, local imports
- Use absolute imports for local modules

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `ResultError`, `PlaybookEntry`)
- **Functions/Variables**: `snake_case` (e.g., `process_data`, `user_id`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- **Private**: Leading underscore (e.g., `_internal_method`)

### Error Handling Patterns

#### Result Type Pattern
```python
from blockether_foundation.result import Result
from blockether_foundation.errors import FoundationBaseError

def risky_operation() -> Result[str, FoundationBaseError]:
    """Return Result type for explicit error handling."""
    try:
        return Result.Ok("success_value")
    except Exception as e:
        return Result.Err(FoundationBaseError(str(e)))

# Usage
result = risky_operation()
if result.is_ok():
    value = result.unwrap()  # Safe to unwrap
else:
    error = result.unwrap_err()  # Handle error
```

#### Exception Hierarchy
```python
class FoundationBaseError(Exception):
    def __init__(self, message: str, details: BaseModel | None = None):
        super().__init__(message)
        self.message = message
        self.details = details
        self.timestamp = datetime.now(UTC)

class SpecificError(FoundationBaseError):
    """Specific error type with domain-specific details."""
    pass
```

### Documentation Standards

#### Docstring Format
```python
def complex_function(
    param1: str,
    param2: int,
    *,
    optional_param: bool = False
) -> dict[str, Any]:
    """Brief description of what the function does.

    Longer description if needed, explaining the algorithm, approach,
    or any important considerations.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        optional_param: Description of optional parameter

    Returns:
        Description of return value and its structure

    Raises:
        SpecificError: When something goes wrong

    Example:
        >>> result = complex_function("test", 42)
        >>> isinstance(result, dict)
        True
    """
```

### Class Design Patterns

#### Pydantic Models
```python
from pydantic import BaseModel, Field

class DataModel(BaseModel):
    """Pydantic model with validation and documentation."""
    
    name: str = Field(description="Human-readable name")
    count: int = Field(default=0, ge=0, description="Count must be non-negative")
    optional_field: str | None = Field(None, description="Optional description")
    
    class Config:
        extra = "forbid"  # Prevent unknown fields
        frozen = True    # Make immutable if appropriate
```

#### Abstract Base Classes
```python
from abc import ABC, abstractmethod

class Processor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, data: Any) -> Result[Any, ErrorType]:
        """Process the input data."""
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """Validate input data before processing."""
        pass
```

### File Organization

#### Package Structure
```
package/
├── __init__.py          # Package exports
├── models.py            # Data models and schemas
├── processor.py         # Main logic/processing
├── utils.py             # Utility functions
├── exceptions.py        # Custom exceptions
└── tests/
    ├── __init__.py
    ├── test_processor.py
    └── test_models.py
```

#### Module Exports
```python
# __init__.py
from .models import DataModel, AnotherModel
from .processor import Processor
from .exceptions import CustomError

__all__ = ["DataModel", "AnotherModel", "Processor", "CustomError"]
```

### Code Quality Standards

#### Linting Rules
- **Line Length**: 100 characters maximum
- **Import Style**: isort (handled by Ruff)
- **Code Style**: pep8, pyflakes, bugbear, comprehensions, pyupgrade
- **Type Safety**: Strict mypy checking enabled

#### Testing Standards
- **Test Files**: `test_*.py` naming convention
- **Test Classes**: `Test*` naming convention
- **Test Functions**: `test_*` naming convention
- **Fixtures**: Use for setup/teardown
- **Markers**: Use pytest markers for categorization

```python
import pytest

class TestResult:
    """Test Result type functionality."""
    
    def test_ok_result_creation(self):
        """Test creating an Ok result."""
        result = Result.Ok("test_value")
        assert result.is_ok()
        assert result.unwrap() == "test_value"
    
    @pytest.mark.unit
    def test_err_result_creation(self):
        """Test creating an Err result."""
        error = TestError("test error")
        result = Result.Err(error)
        assert result.is_err()
```

### Async Programming Patterns

#### Async Function Signatures
```python
async def async_operation(
    input_data: str,
    timeout: float = 30.0
) -> Result[str, AsyncError]:
    """Async operation with timeout."""
    try:
        # Async implementation
        return Result.Ok("result")
    except asyncio.TimeoutError:
        return Result.Err(AsyncError("Operation timed out"))
```

#### Async Context Managers
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def resource_manager():
    """Async context manager for resource cleanup."""
    resource = await acquire_resource()
    try:
        yield resource
    finally:
        await cleanup_resource(resource)
```

### Performance Considerations

#### Lazy Imports
```python
def expensive_function():
    """Function that uses expensive imports."""
    from heavy_library import expensive_class  # Local import
    
    # Use expensive_class
    return expensive_class()
```

#### Memory Management
```python
# Use generators for large datasets
def process_large_file(file_path: str) -> Iterator[str]:
    """Process file line by line to save memory."""
    with open(file_path) as f:
        for line in f:
            yield process_line(line)
```

### Database Patterns

#### SQLAlchemy Integration
```python
from sqlalchemy.orm import Session
from typing import Generator

def get_db_session(db_url: str) -> Generator[Session, None, None]:
    """Database session context manager."""
    engine = create_engine(db_url)
    with Session(engine) as session:
        yield session
```

### Logging Patterns

#### Agno Logging
```python
from agno.utils.log import log_debug, log_info, log_error

def process_data(data: Any) -> Result[Any, ProcessingError]:
    """Process data with proper logging."""
    log_debug(f"Starting processing: {type(data)}")
    
    try:
        result = _do_processing(data)
        log_info(f"Processing completed successfully")
        return Result.Ok(result)
    except Exception as e:
        log_error(f"Processing failed: {e}")
        return Result.Err(ProcessingError(str(e)))
```

## Design Patterns

### Singleton Pattern (When Appropriate)
```python
class _GlobalState:
    """Singleton for global state management."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.data = {}
            self._initialized = True
```

### Factory Pattern
```python
def create_processor(type: str) -> Result[Processor, CreationError]:
    """Factory function for creating processors."""
    if type == "fast":
        return Result.Ok(FastProcessor())
    elif type == "thorough":
        return Result.Ok(ThoroughProcessor())
    else:
        return Result.Err(CreationError(f"Unknown processor type: {type}"))
```

These conventions ensure consistency, maintainability, and robust error handling throughout the codebase.