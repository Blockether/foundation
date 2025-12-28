---
name: meaningful-test-assertions
description: Analyze and improve test assertions to be more meaningful and concrete. Use when working with pytest test files to identify weak assertions (e.g., assert result is not None, assert isinstance, assert len(x) greater than or equal to 1), capture actual test values, and suggest specific, deterministic assertions with named constants. Supports both individual test files and bulk analysis across unit and integration tests.
---

# Meaningful Test Assertions

Analyze pytest test files to identify weak assertions, capture actual values during test execution, and suggest meaningful improvements with concrete assertions and named constants.

## When to Use This Skill

Use this skill when:
- Reviewing test files for assertion quality
- Finding weak/non-specific assertions that provide little value
- Converting arbitrary test assertions to deterministic, concrete checks
- Adding meaningful constants to replace magic numbers
- Working with pytest tests that use `@pytest.mark.unit` or `@pytest.mark.integration`

## Quick Start

### Analyze a Single Test File

```bash
uv run python scripts/capture_and_analyze.py --file tests/unit/test_graph.py
```

### Analyze All Unit Tests

```bash
uv run python scripts/capture_and_analyze.py --all --marker unit
```

### Analyze All Integration Tests

```bash
uv run python scripts/capture_and_analyze.py --all --marker integration
```

### Custom Output File

```bash
uv run python scripts/capture_and_analyze.py --file tests/unit/test_graph.py --output my_report.json
```

**Note:** The script always saves results to a JSON file (default: `assertion_analysis.json`) and prints the JSON content to stdout.

## Workflow

### Phase 1: Parse and Classify Assertions

The script parses test files using Python AST and classifies each assertion:

**Concrete assertions** (OK, no capture needed):
- `assert result == "value"` - exact match
- `assert len(results) == 5` - concrete length
- `assert result.field == 42` - specific field check

**Weak assertions** (need debug capture):
- `assert result is not None` - non-specific existence
- `assert isinstance(result, SomeClass)` - type check only
- `assert len(result) >= 1` / `assert len(result) > 0` - existence check
- `assert "substring" in result` - substring containment
- `assert result` - truthiness check

**Magic numbers** (need constants):
- `assert len(results) == 7` - magic number should be constant
- `assert retry_count == 3` - arbitrary numeric value

### Phase 2: Capture Actual Values (Weak Assertions Only)

For weak assertions:
1. Insert debug prints: `print(f"[ASSERT_DEBUG] {var}={var}")` for each variable
2. Run specific test with: `uv run pytest <file> -k <test_name> -s -m <marker>`
3. Parse `[ASSERT_DEBUG]` output to extract actual values
4. Clean up debug prints (restore original test, no backup file)

### Phase 3: Generate Suggestions

For each assertion:

**Weak assertions**: Use captured values to suggest concrete assertions
- Example: `assert result is not None` with `result=Entity(name="ML")` → `assert result.name == "ML"`
- For complex weak assertions (multiple variables): Split into separate assertions

**Magic numbers**: Suggest adding constants with placeholder names (agent renames)
- Example: `assert len(results) == 7` → `assert len(results) == CONSTANT_X`

**Concrete assertions**: Check if values make sense, may suggest constants

### Phase 4: Present Report to Agent

The script outputs a structured JSON report with:
- Test file and test name
- Each assertion with: current code, type, captured values, issue, suggestion
- Suggested constants to add

## Example Report

```json
{
  "test_file": "tests/unit/test_graph.py",
  "test_name": "test_search_entities",
  "markers": ["unit"],
  "assertions": [
    {
      "line_number": 42,
      "current_assertion": "assert result is not None",
      "assertion_type": "WEAK",
      "captured_values": {"result": "[Entity(id=1, name=\"Machine Learning\")]"},
      "issue": "Non-specific assertion",
      "suggestion": "assert result[0].name == \"Machine Learning\""
    },
    {
      "line_number": 45,
      "current_assertion": "assert len(results) == 7",
      "assertion_type": "MAGIC_NUMBER",
      "issue": "Magic number should be a constant",
      "suggestion": "Replace with CONSTANT_X (agent should rename)"
    }
  ],
  "suggested_constants": [
    "CONSTANT_X = 7 (for line 45)"
  ]
}
```

## Agent Workflow

After the script generates a report:

1. **Review the report** - Understand which assertions are weak and have captured values
2. **Read the test file** - Use `Read` tool to examine the full test context
3. **Apply improvements** - Use `Edit` tool to:
   - Replace weak assertions with concrete assertions using captured values
   - Add meaningful constants at the top of the test file (rename placeholder names)
   - Split complex assertions into separate checks
4. **Verify tests pass** - Run `uv run pytest <file> -k <test_name> -m <marker>`

## Resources

### scripts/capture_and_analyze.py

Main script that:
- Parses test files with Python AST
- Classifies assertions (concrete vs weak vs magic numbers)
- Inserts debug prints and runs tests to capture actual values
- Generates structured suggestions report
- Cleans up debug prints automatically

Usage:
```bash
# Single file analysis (saves to assertion_analysis.json by default)
uv run python scripts/capture_and_analyze.py --file tests/unit/test_graph.py

# All unit tests
uv run python scripts/capture_and_analyze.py --all --marker unit

# All integration tests
uv run python scripts/capture_and_analyze.py --all --marker integration

# Custom output file
uv run python scripts/capture_and_analyze.py --file tests/unit/test_graph.py --output report.json
```

Output: JSON file is always created and content is printed to stdout.

### references/assertion-types.md

Classification of weak vs concrete assertions, decision tree for classification, and examples of each type. Read this to understand how assertions are classified and what makes an assertion weak or concrete.

### references/anti-patterns.md

Common anti-patterns in test assertions with explanations of why they're problematic and how to fix them. Includes:
- Non-specific existence checks
- Type checks only
- Lenient length checks
- Vague substring checks
- Magic numbers
- Chained weak assertions

### references/examples.md

Before/after transformations showing concrete improvements for various assertion patterns. Use these as a reference for how to apply suggestions to different types of weak assertions.

## Key Principles

1. **Capture actual values** - Run the test to see what actual values are, then assert those specific values
2. **Be specific** - Use concrete assertions like `assert result.name == "value"` instead of `assert result is not None`
3. **Use constants** - Replace magic numbers (7, 15, 30) with named constants like `EXPECTED_COUNT = 7`
4. **Split complex assertions** - Break `assert a is not None and b > 0` into two separate assertions with actual values
5. **No backup files** - Debug prints are cleaned up without creating `.backup` files

## Common Weak Patterns to Fix

| Pattern | Problem | Fix |
|---------|----------|------|
| `assert x is not None` | Non-specific | Capture actual value: `assert x.field == "value"` |
| `assert isinstance(x, Type)` | Type only | Check properties: `assert x.name == "value"` |
| `assert len(x) >= 1` | Too lenient | Use exact count: `assert len(x) == 3` |
| `assert "str" in x` | Vague | Exact match: `assert x == "full string"` |
| `assert len(x) == 7` | Magic number | Constant: `assert len(x) == EXPECTED_COUNT` |
