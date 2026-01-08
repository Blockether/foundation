# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-08
**Mode:** Update (existing CLAUDE.md preserved)

## OVERVIEW

Python 3.13+ multi-agent system with graph database, using uv for package management. VRACEF consensus agents, Tantivi indices, Telegram interface.

## STRUCTURE

```
./
├── src/blockether_foundation/
│   ├── agents/        # Multi-agent consensus system (VRACEF, dual translation)
│   ├── graph/         # Graph database + Tantivi indices
│   ├── asr/           # ASR service
│   ├── tts/           # TTS service
│   └── os/            # OS interfaces (Telegram)
├── tests/unit/        # Unit tests (30 files)
└── tests/integration/ # Integration tests (12 files)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Agent consensus logic | src/blockether_foundation/agents/vracef.py | VRACEF implementation |
| Graph operations | src/blockether_foundation/graph/ | Entity resolution, indices |
| Core utilities | src/blockether_foundation/concurrency.py, result.py | ConcurrentProcessor, Result |
| Test patterns | tests/unit/ | pytest markers, fixtures |
| Config | pyproject.toml | ruff, pyright, pytest |

## CODE MAP

```
Key symbols (AST-grep discovered):
- class ConcurrentProcessor (concurrency.py) - batch concurrent ops with retry
- class Result (result.py) - Rust-like error handling
- class VRACEF (agents/vracef.py) - consensus agent
- class DualTranslationAgent (agents/dual_translation.py) - translation
- class GraphDatabase (graph/database.py) - graph operations
- class EntityResolver (graph/) - entity resolution
- class TranscriptionService (transcriber.py) - ASR/TTS
```

## CONVENTIONS

- Package management: uv only (no pip)
- Type hints: Required, checked by pyright
- Async: Use async/await for I/O operations
- Testing: pytest with markers (unit, integration, agno_eval, performance_test, slow, asyncio)
- Linting: ruff (line-length 100), pyright (strict mode)
- Error handling: Result[T, E] for explicit errors (Rust-like)
- Concurrency: ConcurrentProcessor for batch operations with retry logic

## ANTI-PATTERNS (THIS PROJECT)

- DO NOT use pip directly - use uv
- DO NOT suppress type errors with @ts-ignore / @ts-expect-error
- DO NOT use empty catch blocks in error handling
- NEVER delete failing tests to pass
- NEVER commit without explicit request
- DO NOT use generic logging - use structured logging

## UNIQUE STYLES

- XML construction in docstrings: Use """ triple-quoted for multi-line XML in response_parts lists
- ConcurrentProcessor guarantees: order preservation, atomic processing, exponential backoff
- Result pattern: Ok(value), Err(error), unwrap(), map(), and_then()
- Test markers: -m unit (default), -m integration, -m agno_eval, -m performance_test, -m slow, -m asyncio

## COMMANDS

```bash
# Package management
uv sync --all-extras          # Install all dependencies
uv add package-name           # Add dependency
uv run <command>              # Run in venv

# Testing
uv run pytest                 # Run all tests
uv run pytest -m unit         # Run unit tests
uv run pytest -m integration  # Run integration tests
uv run pytest tests/path/test_file.py::test_func  # Run specific test

# Quality
uv run ruff check .           # Lint
uv run ruff format .          # Format
uv run pyright src/           # Type check

# Task runner
uv run poe lint               # Lint
uv run poe test               # Test
uv run poe test-integration   # Integration tests
```

## NOTES

- Project uses strict pyright mode - no type suppression allowed
- Test integration tests skipped by default (-m not integration)
- ConcurrentProcessor: When returning multiple values, wrap tuples as lists
- Result: Use .unwrap_or(default) or .expect("message") instead of try/catch where possible
- XML in docstrings: Multi-line XML must use triple quotes with embedded newlines in response_parts lists
