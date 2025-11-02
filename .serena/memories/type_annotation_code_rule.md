# Code Rule: Modern Python Type Annotations

## Rule: Use Union Type Syntax (|) Instead of Optional[T]

**Status**: ACTIVE ✅

**Description**: All new and updated Python code must use the modern union type syntax (`Type | None`) instead of the legacy `Optional[Type]` annotation.

**Implementation Details**:
- Replace `Optional[Type]` with `Type | None`
- Replace `Optional[Union[Type1, Type2]]` with `Type1 | Type2 | None`
- Remove `Optional` from imports when no longer needed
- This applies to all function parameters, return types, and variable annotations

**Examples**:
```python
# ❌ Old way (DO NOT USE)
from typing import Optional, Union

def process_data(data: Optional[str]) -> Optional[dict]:
    return None

def handle_request(request: Optional[Union[Agent, Team]]) -> None:
    pass

# ✅ New way (REQUIRED)
def process_data(data: str | None) -> dict | None:
    return None

def handle_request(request: Agent | Team | None) -> None:
    pass
```

**Benefits**:
- Cleaner, more readable type annotations
- Less import clutter (no need for Optional/Union imports)
- Follows PEP 604 (Union Types) standard
- Modern Python 3.10+ syntax

**Files Updated**:
- `src/blockether_foundation/os/interfaces/telegram/models.py`
- `src/blockether_foundation/os/interfaces/telegram/handlers.py`

**Enforcement**: This rule should be enforced during code review and through CCLSP configuration.

**Related**: Python 3.10+ compatibility requirement