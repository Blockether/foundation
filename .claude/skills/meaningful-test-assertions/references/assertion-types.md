# Assertion Types Classification

This guide explains the different types of test assertions and how to classify them.

## Concrete Assertions (OK)

Concrete assertions provide specific, meaningful checks that clearly define expected behavior.

### Examples

```python
# Exact value match
assert result.name == "Machine Learning"
assert len(results) == 5
assert result.status == "active"

# Specific field check
assert entity.id == "ml-123"
assert user.age >= 18

# Multiple specific conditions
assert len(results) == 3 and results[0].name == "Python"
```

**When to use:**
- You know the exact expected value
- The value is meaningful in the business logic context
- The assertion would catch a real bug if it fails

## Weak Assertions (Need Improvement)

Weak assertions are non-specific and don't clearly define expected behavior. They provide little value and need concrete actual values.

### Examples

```python
# Non-specific existence check
assert result is not None  # ❌ What is result actually?
assert result  # ❌ Truthiness check

# Type check only
assert isinstance(result, Entity)  # ❌ OK it's an Entity, but what about it?

# Lenient length check
assert len(result) >= 1  # ❌ What should the actual length be?
assert len(result) > 0  # ❌ Same issue

# Substring containment (vague)
assert "ML" in topics  # ❌ Which ML topic specifically?

# Negation checks
assert "error" not in result  # ❌ What SHOULD be in result?
```

**Why these are weak:**
- They don't specify what the expected value actually is
- They might pass when they shouldn't (false positives)
- When they fail, they don't give enough information about what's wrong
- They don't document the expected behavior clearly

**How to fix:**
Capture actual values and replace with concrete assertions:
```python
# Before
assert result is not None

# After capturing: result = Entity(name="Machine Learning")
assert result.name == "Machine Learning"
```

## Magic Numbers (Need Constants)

Magic numbers are arbitrary numeric values that should be named constants.

### Examples

```python
# Magic numbers - what do these mean?
assert len(results) == 7
assert retry_count == 3
assert timeout_seconds == 30

# Magic numbers in comparisons
assert score >= 75
assert count <= 100
```

**When is a number "magic"?**
- It's not a common constant (0, 1, -1, 2, 10)
- It represents domain-specific logic
- Its meaning isn't immediately obvious from context

**How to fix:**
Replace with named constants:
```python
# Before
assert len(results) == 7

# After
EXPECTED_RESULT_COUNT = 7
assert len(results) == EXPECTED_RESULT_COUNT
```

## Decision Tree for Assertion Classification

```
Is the assertion specific about expected value?
├─ Yes → CONCRETE (OK)
│   ├─ Has magic number? → MAGIC_NUMBER
│   └─ No magic number → CONCRETE_OK
└─ No → WEAK
    ├─ is not None? → WEAK
    ├─ isinstance? → WEAK
    ├─ len(x) >= 1 or len(x) > 0? → WEAK
    ├─ "substring" in x? → WEAK
    └─ Other non-specific? → WEAK
```

## Common Anti-Patterns

### Pattern 1: Truthiness Check
```python
# Bad
assert results

# Good
assert len(results) == 3
```

### Pattern 2: Type Check Only
```python
# Bad
assert isinstance(user, User)

# Good
assert user.name == "Alice"
```

### Pattern 3: Existence Check
```python
# Bad
assert entity is not None

# Good
assert entity.id == "user-123"
```

### Pattern 4: Vague Length
```python
# Bad
assert len(items) > 0

# Good
assert len(items) == 5
```

### Pattern 5: Substring Vague
```python
# Bad
assert "error" in log_message

# Good
assert log_message == "Error: Connection timeout"
```
