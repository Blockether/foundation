# Project Guidelines

> **IMPORTANT:** `GEMINI.md` and `AGENTS.md` are symlinks to this file (`CLAUDE.md`). Always edit `CLAUDE.md` directly when updating project guidelines.

<skills_system priority="1">

## Available Skills

<!-- SKILLS_TABLE_START -->
<usage>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

How to use skills:
- Invoke: Bash("openskills read <skill-name>")
- The skill content will load with detailed instructions on how to complete the task
- Base directory provided in output for resolving bundled resources (references/, scripts/, assets/)

Usage notes:
- Only use skills listed in <available_skills> below
- Do not invoke a skill that is already loaded in your context
- Each skill invocation is stateless
</usage>

<available_skills>

<skill>
<name>skill-creator</name>
<description>Guide for creating effective skills. This skill should be used when users want to create a new skill (or update an existing skill) that extends Claude's capabilities with specialized knowledge, workflows, or tool integrations.</description>
<location>project</location>
</skill>

</available_skills>
<!-- SKILLS_TABLE_END -->

</skills_system>

## Optional Dependencies

When adding features that require optional dependencies (e.g., local ASR, ML libraries), follow this pattern:

### 1. Define optional dependency group in `pyproject.toml`

```toml
[project.optional-dependencies]
feature_name = [
    "package>=version",
    "another-package>=version",
]
```

### 2. Use `importlib.metadata` to check availability

Check if the main package is installed using package metadata (NOT try/except imports):

```python
# Check if feature dependencies are installed
try:
    from importlib.metadata import version
    version("package_name")
    FEATURE_AVAILABLE = True
except Exception:
    FEATURE_AVAILABLE = False
```

**Why:** This is a single, deterministic check. If the main package is installed, its dependencies will be too.

### 3. Use lazy `__getattr__` for exports

Never import optional dependencies at module level in `__init__.py`. Use `__getattr__`:

```python
# __init__.py
from . import feature_module

__all__ = ["FeatureClass", "FEATURE_AVAILABLE"]

def __getattr__(name: str):
    if name == "FeatureClass":
        return feature_module.FeatureClass
    if name == "FEATURE_AVAILABLE":
        return feature_module.FEATURE_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Why:** This allows the package to be imported without the optional dependencies present. Imports only happen when the attribute is actually accessed.

**Important Note for Type Annotations:**
When using optional dependencies in function type signatures (outside of `TYPE_CHECKING` blocks), use string literal type annotations to avoid `NameError` at module import time:

```python
# ❌ WRONG - This will fail at import time if LocalWhisperAudioTranscriber isn't imported yet
def my_func(audio_transcriber: LocalWhisperAudioTranscriber | None = None):
    ...

# ✅ CORRECT - String literal avoids the NameError
def my_func(audio_transcriber: "LocalWhisperAudioTranscriber | None" = None):
    ...
```

### 4. Provide helpful error messages

    Create a helper function that raises ImportError with installation instructions:

    ```python
    def _check_feature_available() -> None:
        if not FEATURE_AVAILABLE:
            raise ImportError(
                "Feature dependencies are not installed. "
                "Install them with: uv pip install 'blockether-foundation[feature_name]'"
            )
    ```

    Call this in `__init__` methods and functions that require the optional dependencies.

### 5. Always use `uv` for dependency management

    This project uses `uv` for Python package management. Always use `uv` commands instead of `pip`:

    ```bash
    # Install all dependencies including optional extras for development
    uv sync --all-extras

    # Install specific optional dependency group
    uv pip install 'blockether-foundation[feature_name]'

    # Update dependencies
    uv sync

    # Add a new dependency
    uv add package-name
    ```

    **Why:** `uv` is significantly faster than `pip` and provides better dependency resolution, especially for projects with optional dependencies and dev extras.
