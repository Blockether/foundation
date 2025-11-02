# Suggested Commands for Blockether Foundation Development

## Project Overview Commands

### Package Management
```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Install production dependencies only
uv pip install -e "."
```

### Testing Commands
```bash
# Run all unit tests (default, skips integration tests)
poe test

# Run only unit tests explicitly
poe test-unit

# Run integration tests (requires real LLM API calls, slow)
poe test-integration

# Run tests with coverage report
poe test-cov

# Run tests in watch mode for continuous testing
poe test-watch

# Run specific test file
pytest tests/unit/test_result.py -v

# Run tests with specific markers
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m slow
```

### Code Quality Commands
```bash
# Lint code (check for issues)
poe lint

# Fix linting issues automatically
poe lint-fix

# Format code (auto-format)
poe format

# Check formatting without making changes
poe format-check

# Type checking (mypy)
poe type-check

# Run all quality checks in sequence
poe check-all
```

### Development Workflow Commands
```bash
# After making changes:
poe format-check  # Check formatting
poe lint         # Check for linting issues
poe type-check   # Check types
poe test         # Run tests
# OR run all at once:
poe check-all
```

### Manual Commands (if poe not available)
```bash
# Direct pytest usage
pytest tests/ -v

# Direct ruff usage
ruff check src/ tests/
ruff check --fix src/ tests/
ruff format src/ tests/

# Direct mypy usage
mypy src/
```

### System Commands (Darwin/macOS)
```bash
# File operations
ls -la                      # List files with details
find . -name "*.py"       # Find Python files
grep -r "pattern" src/    # Search in source files

# Git operations
git status                  # Check git status
git add .                   # Stage all changes
git commit -m "message"    # Commit changes
git push                   # Push to remote
git pull                   # Pull from remote

# Directory navigation
cd src                      # Change to source directory
cd ..                       # Go up one level
pwd                         # Show current directory

# Process management
ps aux | grep python       # Find Python processes
kill <pid>                  # Kill process by PID
```

### Project-Specific Commands
```bash
# Check if virtual environment is active
which python

# Verify dependencies
pip list

# Run the application (if entry points are defined)
python -m blockether_foundation
```

## Testing Strategy

### Test Categories
- **Unit Tests**: Fast tests with mocked dependencies (marked with `unit`)
- **Integration Tests**: Real API calls and external dependencies (marked with `integration`)
- **Slow Tests**: Performance-intensive tests (marked with `slow`)

### Running Tests by Category
```bash
# Default: unit tests only
poe test

# Integration tests only
poe test-integration

# All tests (including integration)
pytest tests/ -v --cov=blockether_foundation

# Specific test markers
pytest tests/ -m unit -v
pytest tests/ -m integration -v
pytest tests/ -m slow -v
```

## Code Quality Standards

### Linting Configuration
- Line length: 100 characters
- Python version: 3.13+ (target for linting, 3.14+ for mypy)
- Tools: Ruff for linting and formatting, MyPy for type checking

### Type Checking
- Strict type checking enabled
- All function signatures must have type hints
- Return types required for all functions
- No implicit optional types

### Documentation
- All public classes and functions should have docstrings
- Use Google-style or NumPy-style docstring format
- Include parameter descriptions and return type information

## Entry Points and Running the Application

The project is a library/package, so there may not be direct entry points. However, you can:

```bash
# Test import functionality
python -c "from blockether_foundation import Result; print('Import successful')"

# Run specific modules if they have __main__ sections
python -m blockether_foundation.some_module
```

## Environment Setup

### Required Environment Variables
```bash
# For OpenAI integration (if used)
export OPENAI_API_KEY="your_openai_api_key"

# For Telegram integration (if used)
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"

# For other LLM providers (as needed)
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Virtual Environment
```bash
# Create with uv (recommended)
uv venv
source .venv/bin/activate

# Or with standard Python
python -m venv .venv
source .venv/bin/activate
```