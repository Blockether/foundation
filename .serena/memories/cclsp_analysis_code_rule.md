# Code Rule: Comprehensive Code Analysis

## Rule: Regular Code Architecture and Usage Analysis

**Status**: ACTIVE ✅

**Description**: Perform comprehensive analysis of codebase architecture, usage patterns, and identify unused or missing functionality on a regular basis.

## Analysis Process

### When to Run Analysis
1. **After major feature completions**
2. **Before architecture changes**
3. **During code refactoring planning**
4. **Monthly architectural reviews**

### Analysis Checklist

#### 1. Import and Usage Analysis
- [ ] Check for unused imports across all modules
- [ ] Identify unused classes and functions
- [ ] Verify all defined code is actually utilized
- [ ] Check for circular dependencies

#### 2. Architecture Assessment  
- [ ] Evaluate separation of concerns
- [ ] Check for proper layering (models, handlers, validation, errors)
- [ ] Assess code organization and modularity
- [ ] Verify design patterns are properly implemented

#### 3. Missing Functionality Detection
- [ ] Identify unused models that suggest missing endpoints
- [ ] Check for validation functions not exposed via API
- [ ] Look for error types not being handled
- [ ] Assess monitoring/observability gaps

#### 4. Code Quality Metrics
- [ ] Type annotation coverage
- [ ] Error handling completeness  
- [ ] Validation coverage
- [ ] Documentation completeness

## Analysis Commands

### AST-Based Analysis (when CCLSP unavailable)
```bash
python -c "
import ast
from pathlib import Path
from collections import defaultdict

# Analyze definitions, imports, and usage patterns
# Run comprehensive architecture analysis
"
```

### CCLSP Analysis (when available)
```bash
cclsp analyze src/blockether_foundation/os/interfaces/telegram/
cclsp check-unused src/blockether_foundation/os/interfaces/telegram/
```

### Type Checking
```bash
python -m mypy src/blockether_foundation/os/interfaces/telegram/ --no-error-summary --ignore-missing-imports
```

## Findings Classification

### ✅ **Well-Architected Code**
- All error classes properly used
- Validation functions properly integrated
- Clean separation of concerns
- Proper type safety
- Good documentation

### ⚠️ **Improvement Opportunities**  
- Unused models suggesting missing endpoints
- Missing admin/management functionality
- Gaps in monitoring/observability
- Incomplete API surface area

### ❌ **Problematic Code**
- Truly unused dead code
- Circular dependencies
- Poor separation of concerns
- Missing error handling
- Type safety violations

## Action Categories

### Keep (Good Architecture)
- Code that is well-architected but currently unused
- Forward-thinking design patterns
- Prepared functionality for future features

### Implement (Missing Features)
- Endpoints that should use existing models
- API surface area gaps
- Missing admin/management functionality

### Refactor (Architecture Issues)
- Poor separation of concerns
- Circular dependencies
- Truly dead code

### Document (Knowledge Gaps)
- Unclear architecture decisions
- Missing design documentation
- Unexplained design patterns

## Enforcement

This analysis should be performed:
- **Mandatory**: After any major feature completion
- **Recommended**: Monthly architectural reviews  
- **Optional**: Before individual PRs (focused analysis)

## Tools Integration

When CCLSP is properly configured, it should be used for:
- Automated unused code detection
- Import analysis
- Architecture validation
- Dependency checking

Until CCLSP is available, use AST-based manual analysis as demonstrated in the findings.

## Documentation

All analysis results should be documented in Serena memory files:
- `code_analysis_findings` - Specific analysis results
- `architecture_decisions` - Design rationale
- `missing_functionality` - Future development roadmap