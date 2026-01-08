# UNIT TESTS

**Generated:** 2026-01-08

## OVERVIEW
Unit test conventions with pytest markers

## STRUCTURE
```
tests/unit/
├── test_concurrency.py              # ConcurrentProcessor tests
├── test_result.py                   # Result type tests
├── test_graph_entity_resolution.py  # Graph resolution tests
├── test_telegram_handlers.py        # Telegram handler tests
├── test_transcription_service.py    # Transcriber tests
└── ... (25 more test files)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Concurrency tests | test_concurrency.py | Batch operations, retry logic |
| Result type tests | test_result.py | Ok/Err, unwrap, map |
| Graph tests | test_graph_*.py | Entity resolution, indices |
| Integration tests | tests/integration/ | Full system tests |
| Marker usage | pytest config | unit, integration, asyncio |

## CONVENTIONS
- Markers: @pytest.mark.unit (default), @pytest.mark.integration, @pytest.mark.agno_eval, @pytest.mark.performance_test, @pytest.mark.slow, @pytest.mark.asyncio
- Run: `uv run pytest` (unit only by default)
- Integration: `uv run pytest -m integration`
- Use fixtures for common test setup
- Mock external dependencies (ASR, TTS, Telegram)

## ANTI-PATTERNS
- DO NOT test implementation details - test behavior
- NEVER use sleep() for timing - use mocks
- DO NOT write integration tests in unit/
- NEVER skip tests without marker justification

## UNIQUE STYLES
- Async tests: @pytest.mark.asyncio
- Slow tests: @pytest.mark.slow (skip by default)
- Performance tests: @pytest.mark.performance_test
- Agno eval: @pytest.mark.agno_eval
