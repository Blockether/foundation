# Code Rule: Always Run Type Checking After Changes

## Rule: CCLSP/Mypy Type Checking Required

**Status**: ACTIVE ✅

**Description**: All code changes must be followed by type checking to ensure no type errors are introduced. This is a mandatory step in the development workflow.

**Implementation Process**:
1. **After any code changes** are made to Python files
2. **Run type checking** immediately using mypy or CCLSP
3. **Fix all type errors** before considering the task complete
4. **Verify the fix** by running type checking again

**Type Checking Commands**:
```bash
# For specific module
python -m mypy src/blockether_foundation/os/interfaces/telegram/ --no-error-summary

# For entire project (when needed)
python -m mypy src/ --no-error-summary

# Alternative: Use CCLSP if available
cclsp check
```

**Common Type Issues to Fix**:
- Missing return type annotations
- Incompatible return value types  
- Type mismatches in container operations
- Missing import stubs
- Any type declarations that should be specific

**Current Type Issues Found**:
- `validation.py`: Result type incompatibilities
- `handlers.py`: Any return types and type mismatches
- `usage.py`: Missing return type annotations

**Enforcement**: This rule must be followed for ALL Python code changes. No exceptions.

**Integration**: This should be the final step in any code modification workflow.

**Verification**: Task is not complete until mypy runs with zero errors.