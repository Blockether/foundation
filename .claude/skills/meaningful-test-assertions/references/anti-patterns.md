# Test Assertion Anti-Patterns

This document lists common anti-patterns in test assertions and how to fix them.

## Anti-Pattern 1: Non-Specific Existence Checks

### Problem
Assertions that only check if something exists, not what it is.

### Examples
```python
assert result is not None
assert response
assert user is not None
```

### Why It's Bad
- Doesn't verify the actual expected value
- False positives: passes when it shouldn't
- Provides no diagnostic information when it fails
- Doesn't document expected behavior

### Fix
Capture actual values and make concrete assertions:
```python
# Before
assert result is not None

# After (captured: result = Entity(name="ML", id=1))
assert result.name == "ML"
assert result.id == 1
```

---

## Anti-Pattern 2: Type Checks Only

### Problem
Assertions that only verify type, not the actual content.

### Examples
```python
assert isinstance(result, Entity)
assert type(user) == User
assert isinstance(response, dict)
```

### Why It's Bad
- Type alone doesn't verify correctness
- A wrong object of the right type passes
- Doesn't check important properties

### Fix
Check specific properties instead:
```python
# Before
assert isinstance(result, Entity)

# After
assert result.name == "Machine Learning"
assert result.id == "entity-123"
```

---

## Anti-Pattern 3: Lenient Length Checks

### Problem
Assertions that only verify minimum length, not exact length.

### Examples
```python
assert len(results) >= 1
assert len(items) > 0
assert count >= 0
```

### Why It's Bad
- Doesn't verify exact expected count
- Could pass with too many results
- Doesn't catch bugs that produce extra items

### Fix
Use exact length:
```python
# Before
assert len(results) >= 1

# After
assert len(results) == 3
```

### Exception
Use length check when exact count is unpredictable (e.g., database queries with varying data).

---

## Anti-Pattern 4: Vague Substring Checks

### Problem
Assertions that check for substring presence without specifying the full expected value.

### Examples
```python
assert "error" in error_message
assert "success" in response
assert "ML" in topics
```

### Why It's Bad
- Too broad, matches unintended strings
- Doesn't verify exact message format
- Can pass when it shouldn't (e.g., "no errors" contains "error")

### Fix
Check exact string or specific value:
```python
# Before
assert "error" in error_message

# After
assert error_message == "Connection timeout"

# Or check specific field
assert response.status == "completed"
```

---

## Anti-Pattern 5: Magic Numbers

### Problem
Numeric values in assertions without semantic meaning.

### Examples
```python
assert len(results) == 7
assert retry_count == 3
assert score >= 75
assert timeout == 30
```

### Why It's Bad
- Unclear what the number represents
- Hard to maintain when requirements change
- Doesn't express business intent

### Fix
Replace with named constants:
```python
# Before
assert len(results) == 7

# After
EXPECTED_RESULTS = 7
assert len(results) == EXPECTED_RESULTS
```

### Common Constants (Don't Need Naming)
These are so common they don't need constants:
- 0, 1, -1
- 2, 10 (in some contexts)

---

## Anti-Pattern 6: Chained Weak Assertions

### Problem
Multiple weak assertions combined with `and`.

### Examples
```python
assert result is not None and len(result) > 0
assert user is not None and isinstance(user, User)
```

### Why It's Bad
- Each part is still weak
- Hard to debug when one part fails
- Better to split into separate assertions

### Fix
Split into separate concrete assertions:
```python
# Before
assert result is not None and len(result) > 0

# After (captured values)
assert len(result) == 3
assert result[0].name == "Python"
```

---

## Anti-Pattern 7: Negation Checks

### Problem
Assertions that check something is NOT present.

### Examples
```python
assert "error" not in message
assert None not in results
assert 0 not in scores
```

### Why It's Bad
- Doesn't specify what SHOULD be present
- Can pass when completely wrong
- Doesn't verify positive requirements

### Fix
Check for expected positive values:
```python
# Before
assert "error" not in message

# After
assert message == "Success: Operation completed"
```

---

## Anti-Pattern 8: Runtime Calculations in Assertions

### Problem
Doing calculations inside assertions instead of using pre-calculated expected values.

### Examples
```python
assert len(results) == expected_count()  # Function call
assert sum(scores) / len(scores) > 70  # Calculation in assert
assert datetime.now() - timestamp < timedelta(days=1)  # Time calculation
```

### Why It's Bad
- Makes tests non-deterministic
- Harder to read and understand
- Calculation might have bugs
- Harder to debug failures

### Fix
Use pre-calculated constants or setup expected values:
```python
# Before
assert sum(scores) / len(scores) > 70

# After
AVERAGE_SCORE_THRESHOLD = 75
average = sum(scores) / len(scores)
assert average >= AVERAGE_SCORE_THRESHOLD
```

---

## Detection Rules

### Weak Assertion Detection
An assertion is weak if it matches ANY of these patterns:

1. **is not None**: `assert x is not None`
2. **isinstance**: `assert isinstance(x, Type)`
3. **Truthiness**: `assert x` (just the variable)
4. **Lenient length**: `assert len(x) >= 1`, `assert len(x) > 0`, `assert len(x) >= 0`
5. **Substring in**: `assert "substring" in x`
6. **Negation**: `assert x not in y`
7. **Empty check**: `assert x != ""`, `assert x != []`

### Magic Number Detection
An assertion has magic numbers if:

1. Contains numeric literal not in `{0, 1, -1, 2, 10}`
2. The number appears in:
   - Comparison: `assert x == 7`
   - Arithmetic: `assert x * 3 > 10`
   - Function argument: `assert len(x) == 7`

### Constants That Don't Need Naming
These numbers are so common they can be literals:
- `0`, `1`, `-1`, `2`, `10`

Everything else should be a constant:
- `3`, `5`, `7`, `15`, `30`, `100`, etc.

---

## Fix Hierarchy

When fixing assertions, follow this priority order:

1. **Weak assertions** (highest priority)
   - Capture actual values
   - Make concrete assertions
   - Split complex assertions

2. **Magic numbers** (medium priority)
   - Replace with constants
   - Agent should name constants meaningfully

3. **Concrete assertions with arbitrary values** (low priority)
   - Check if value makes sense in context
   - May need constant if not immediately meaningful
