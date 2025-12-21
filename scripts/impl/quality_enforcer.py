#!/usr/bin/env python3
"""
Quality Enforcer - Self-contained agentic script to analyze and automatically fix quality violations.

This script enforces strict test quality standards and automatically fixes issues:
1. Detects and fixes magic values (unexplained numbers)
2. Finds and fixes vague assertions and non-deterministic checks
3. Ensures deterministic operations assert exact expected values
4. Removes print statements from tests
5. Removes pytest.main() calls (tests should not invoke pytest programmatically)
6. Removes if statements in tests (replaces with assertions)
7. Removes for/while loops in tests (replaces with accumulation patterns)
8. Removes try/except blocks in tests (replaces with pytest.raises)
9. Removes raise statements in tests (tests should not throw exceptions)
10. Removes return statements in test functions (tests should end with assertions)
11. Ensures test functions have proper pytest markers and type annotations
12. ALWAYS runs tests after making changes using project's proper test commands
13. ALWAYS applies automatic fixes using Claude AI

IGNORE DIRECTIVES:
For legitimate exceptions to the quality rules, use these ignore directives:

• Line-level ignore: `# ignore-quality-enforcer`
  - Ignores violations on the same line only
  - Example: `if condition:  # ignore-quality-enforcer`

• Block-level ignore: `# ignore-quality-enforcer-start` and `# ignore-quality-enforcer-end`
  - Ignores violations for the entire block between start and end
  - Example:
    ```python
    # ignore-quality-enforcer-start
    async def test_processor(item: str) -> Sequence[str]:
        if item == "fail":
            raise ValueError("Processing failure")  # Test case for error handling
        return [f"processed: {item}"]
    # ignore-quality-enforcer-end
    ```

Common legitimate use cases for ignore directives:
• Test functions that need to raise exceptions for testing error handling
• Test setup code that requires conditional logic
• Data processing helper functions within tests
• Legacy code that would be unsafe to refactor

Usage:
  python quality_enforcer.py <test_file_path> [--max-iterations N]
  python quality_enforcer.py --unused-functions

Features:
1. Test quality analysis and automatic fixes (default mode)
2. Unused function detection across entire codebase (--unused-functions)
3. Auto-fix and test execution are always enabled for maximum quality enforcement
"""

import ast
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, override

# Import utility functions from utils module
from .utils import (
    C_GREEN,
    C_RED,
    C_RESET,
    C_YELLOW,
    COMMON_CONSTANTS,
    StreamSubprocess,
    analyze_development_indicators_with_claude,
    create_section_box,
    run_claude_fix,
    tabulate_with_wrapping,
)

# Removed tabulate import - now using PrettyTable via utils.tabulate_with_wrapping

# Configure logging for proper output capture and timing
logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


DEFAULT_MAX_ITERATIONS = 3

# Box formatting constants
BOX_WIDTH = 140  # Fixed width for all boxes (increased for better table readability)
BOX_PADDING = 2  # Padding on each side inside the box

# Quality enforcer prompt template - embedded directly for self-containment
QUALITY_ENFORCER_PROMPT = """# Quality Enforcer

Please fix the following quality issues in {file_path}:

ISSUES TO FIX:
{issues_list}

REQUIREMENTS:

1. Fix all the listed issues systematically
2. Maintain the existing functionality
3. Add proper type annotations where missing (-> None for test functions)
4. Replace magic values with named constants
5. Add @override decorators where needed
6. Remove or refactor prohibited control flow statements
7. Ensure all imports are properly typed
8. Remove empty 'if TYPE_CHECKING:' blocks
9. DO NOT use 'if TYPE_CHECKING:' for imports of libraries listed in pyproject.toml dependencies (fastapi, openai, pydantic, tenacity, tantivy, agno, faster-whisper, sqlalchemy, litellm) - import them directly at top level.

## IGNORE DIRECTIVES

For legitimate exceptions to the quality rules, use these ignore directives:

- Line-level ignore: `# ignore-quality-enforcer`
  - Ignores violations on the same line only
  - Example: `if condition:  # ignore-quality-enforcer`

- Block-level ignore: `# ignore-quality-enforcer-start` and `# ignore-quality-enforcer-end`
  - Ignores violations for the entire block between start and end
  - Example:

    ```python
    # ignore-quality-enforcer-start
    async def test_processor(item: str) -> Sequence[str]:
        if item == "fail":
            raise ValueError("Processing failure")  # Test case for error handling
        return [f"processed: {item}"]
    # ignore-quality-enforcer-end
    ```

Common legitimate use cases for ignore directives:

- Test functions that need to raise exceptions for testing error handling
- Test setup code that requires conditional logic
- Data processing helper functions within tests
- Legacy code that would be unsafe to refactor

Focus on the most critical issues first (type annotations, missing variables, override decorators).

After fixing, ensure the file follows Python typing best practices and the project's quality standards.

Provide a summary of the fixes applied.

## Common Fix Patterns

### Magic Values

```python
# BAD
assert result == 42

# GOOD
EXPECTED_VALUE = 42
assert result == EXPECTED_VALUE
```

### Control Flow Replacement

```python
# BAD
if condition:
    assert something

# GOOD
assert condition and something
```

### Loop Elimination

```python
# BAD
items = []
for item in source:
    items.append(process(item))
assert len(items) > 0

# GOOD
items = [process(item) for item in source]
assert items == [expected1, expected2]
```

### Exception Testing

```python
# BAD
try:
    risky_operation()
    assert False, "Should have raised"
except ValueError:
    pass

# GOOD
with pytest.raises(ValueError):
    risky_operation()
```

### Code Documentation Standards

- Replace generic docstrings with specific, professional descriptions
- Remove redundant comments that state the obvious
- Ensure all TODO/FIXME comments have corresponding implementation tasks
"""


class UnusedFunctionAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze function usage across the codebase."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.defined_functions: dict[str, ast.FunctionDef] = {}
        self.called_functions: set[str] = set()
        self.imported_names: set[str] = set()
        self.class_methods: dict[str, set[str]] = {}  # class_name -> method_names
        self.current_class: str | None = None
        self.ignored_lines: set[int] = set()
        self._parse_ignore_directives()

    def _parse_ignore_directives(self) -> None:
        """Parse ignore directives from the source file."""
        try:
            with open(self.file_path) as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                stripped = line.strip()
                # Check for ignore directives
                if "# public-api" in stripped or "# ignore-unused" in stripped:
                    self.ignored_lines.add(i + 1)  # Lines are 1-indexed in AST
                    # Also ignore the next few lines to support placing comment above function/decorators
                    self.ignored_lines.add(i + 2)
                    self.ignored_lines.add(i + 3)
                    self.ignored_lines.add(i + 4)
        except FileNotFoundError:
            pass

    @override
    def visit_Import(self, node: ast.Import) -> None:
        """Track imported names to avoid flagging imported functions."""
        for alias in node.names:
            self.imported_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track imported names to avoid flagging imported functions."""
        for alias in node.names:
            if alias.name != "*":  # Can't track star imports precisely
                self.imported_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track class definitions and their methods."""
        previous_class = self.current_class
        self.current_class = node.name
        self.class_methods[node.name] = set()
        self.generic_visit(node)
        self.current_class = previous_class

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track function definitions."""
        # Check for ignore directives
        if node.lineno in self.ignored_lines:
            self.generic_visit(node)
            return

        # Skip special methods and private methods
        if node.name.startswith("__") and node.name.endswith("__"):
            self.generic_visit(node)
            return

        # Skip test functions
        if node.name.startswith("test_"):
            self.generic_visit(node)
            return

        function_key = node.name
        if self.current_class:
            function_key = f"{self.current_class}.{node.name}"
            self.class_methods[self.current_class].add(node.name)

        self.defined_functions[function_key] = node
        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        """Track function calls."""
        # Handle direct function calls: func()
        if isinstance(node.func, ast.Name):
            self.called_functions.add(node.func.id)

        # Handle method calls: obj.method()
        elif isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            # Track all method calls - filtering will be applied later
            self.called_functions.add(method_name)

        # Handle attribute access: module.func
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            self.called_functions.add(attr_name)

        # Check for functions used as arguments (callbacks, etc.)
        for arg in node.args:
            if isinstance(arg, ast.Name):
                self.called_functions.add(arg.id)
            elif isinstance(arg, ast.Attribute):
                self.called_functions.add(arg.attr)

        # Check for functions used as keyword arguments
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Name):
                self.called_functions.add(keyword.value.id)
            elif isinstance(keyword.value, ast.Attribute):
                self.called_functions.add(keyword.value.attr)

        self.generic_visit(node)

    def get_unused_functions(self) -> list[dict[str, Any]]:
        """Get list of unused functions with context."""
        unused: list[dict[str, Any]] = []

        for func_key, func_def in self.defined_functions.items():
            func_name = func_key.split(".")[-1]

            # Skip if function is imported from somewhere else
            if func_name in self.imported_names:
                continue

            # Skip if function appears to be called
            if func_name in self.called_functions:
                continue

            # Check if it's a class method that might be called externally
            if "." in func_key:
                method_name = func_key.split(".")[1]

                # Skip public methods that might be part of API
                if not method_name.startswith("_"):
                    # Skip special methods like __str__, __repr__ etc.
                    if not (method_name.startswith("__") and method_name.endswith("__")):
                        # Conservative approach: skip public methods
                        continue

                # Skip if method name is called anywhere (could be external call)
                if method_name in self.called_functions:
                    continue

            # Skip main/entry functions
            if func_name in ["main", "__init__", "run", "start"]:
                continue

            # Add to unused list
            unused.append(
                {
                    "function": func_key,
                    "line": getattr(func_def, "lineno", 0),
                    "file": self.file_path,
                    "location": f"{self.file_path}:{getattr(func_def, 'lineno', 0)}",
                    "is_class_method": "." in func_key,
                    "is_private": func_name.startswith("_"),
                }
            )

        return unused


class DevelopmentIndicatorAnalyzer(ast.NodeVisitor):
    """AST visitor to detect development stage keywords in code."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.dev_indicators: list[dict[str, Any]] = []
        self.keywords = [
            "minimal",  # ignore-development
            "workaround",  # ignore-development
            "incomplete",  # ignore-development
            "temporarily",  # ignore-development
        ]
        self.processed_lines: set[int] = set()
        self.ignored_lines: set[int] = set()
        self.source_code: str = ""
        self._parse_ignore_directives()

    def _parse_ignore_directives(self) -> None:
        """Parse ignore directives from the source file."""
        try:
            with open(self.file_path) as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                stripped = line.strip()
                # Check for ignore directives
                if "# ignore-development" in stripped:
                    self.ignored_lines.add(i + 1)  # Lines are 1-indexed
                    # Also ignore the next line
                    self.ignored_lines.add(i + 2)
        except FileNotFoundError:
            pass

    @override
    def visit(self, node: ast.AST) -> None:
        """Override to get the original source code."""
        # Visit children first to get most specific node (bottom-up)
        self.generic_visit(node)

        try:
            # Get the source code for this node
            if hasattr(self, "source_code"):
                start_line = getattr(node, "lineno", 0) - 1
                end_line = getattr(node, "end_lineno", start_line)

                # Only process if we have valid line numbers
                if start_line >= 0:
                    lines = self.source_code.split("\n")
                    if start_line < len(lines):
                        for line_num in range(start_line, min(end_line + 1, len(lines))):
                            # Skip if already processed or ignored
                            if (
                                line_num in self.processed_lines
                                or (line_num + 1) in self.ignored_lines
                            ):
                                continue

                            line_content = lines[line_num].lower()
                            for keyword in self.keywords:
                                if keyword in line_content:
                                    self.dev_indicators.append(
                                        {
                                            "keyword": keyword,
                                            "line": line_num + 1,
                                            "content": lines[line_num].strip(),
                                            "node_type": type(node).__name__,
                                            "file": self.file_path,
                                            "location": f"{self.file_path}:{line_num + 1}",
                                        }
                                    )
                                    self.processed_lines.add(line_num)
                                    break
        except Exception:
            pass  # Ignore errors during source extraction

    def analyze_file(self) -> dict[str, Any]:
        """Analyze the file for development stage keywords."""
        try:
            with open(self.file_path, encoding="utf-8") as f:
                self.source_code = f.read()

            # Parse the AST to walk through nodes
            tree = ast.parse(self.source_code)
            self.visit(tree)

            return {
                "dev_indicators": self.dev_indicators,
                "total_found": len(self.dev_indicators),
                "file_path": self.file_path,
            }
        except Exception as e:
            return {"error": f"Error analyzing {self.file_path}: {e}"}


class QualityAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze test quality issues."""

    def __init__(self, file_path: str, is_test_file: bool = True) -> None:
        self.file_path = file_path
        self.is_test_file = is_test_file
        self.issues: list[dict[str, Any]] = []
        self.current_function: str | None = None
        self.current_class: str | None = None
        self.test_functions: list[ast.FunctionDef] = []
        self.current_function_has_return = False
        self.current_function_has_assert = False

        # Parse ignore directives from the source code
        self.ignored_lines: set[int] = set()
        self.ignored_blocks: list[tuple[int, int]] = []  # (start_line, end_line)
        self._parse_ignore_directives()

    def _parse_ignore_directives(self) -> None:
        """Parse ignore directives from the source file."""
        try:
            with open(self.file_path) as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                # Check for line-level ignore: # ignore-quality-enforcer
                if "# ignore-quality-enforcer" in stripped:
                    self.ignored_lines.add(i + 1)  # Lines are 1-indexed in AST

                # Check for block-level ignore: # ignore-quality-enforcer-start
                elif "# ignore-quality-enforcer-start" in stripped:
                    start_line = i + 1
                    # Find the corresponding end
                    i += 1
                    while i < len(lines):
                        if "# ignore-quality-enforcer-end" in lines[i]:
                            end_line = i + 1
                            self.ignored_blocks.append((start_line, end_line))
                            break
                        i += 1

                i += 1
        except FileNotFoundError:
            pass  # File doesn't exist, will be handled elsewhere

    def _should_ignore_line(self, lineno: int) -> bool:
        """Check if a line should be ignored based on directives."""
        if lineno in self.ignored_lines:
            return True

        # Check if line is within any ignored block
        for start, end in self.ignored_blocks:
            if start <= lineno <= end:
                return True

        return False

    def _should_ignore_node(self, node: ast.AST) -> bool:
        """Check if an AST node should be ignored."""
        # Get the line range of this node
        start_line = getattr(node, "lineno", 0)
        end_line = getattr(node, "end_lineno", start_line)

        # Check if any line in this node's range should be ignored
        for line in range(start_line, end_line + 1):
            if self._should_ignore_line(line):
                return True

        return False

    @override
    def visit_Import(self, node: ast.Import) -> None:
        if self.current_function and not self._should_ignore_node(node):
            self.issues.append(
                {
                    "line": node.lineno,
                    "type": "Local Import",
                    "message": "Imports should be at the top of the file, not inside functions.",
                    "suggestion": "Move import to the top of the file.",
                }
            )
        self.generic_visit(node)

    @override
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.current_function and not self._should_ignore_node(node):
            self.issues.append(
                {
                    "line": node.lineno,
                    "type": "Local Import",
                    "message": "Imports should be at the top of the file, not inside functions.",
                    "suggestion": "Move import to the top of the file.",
                }
            )
        self.generic_visit(node)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check for fallback usage in function name
        if "fallback" in node.name.lower() and not self._should_ignore_node(node):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(
                node,
                "fallback_usage",
                f"Function name '{node.name}' contains 'fallback' at line {lineno}. This is potentially useless.",
            )

        self.current_function = node.name
        self.current_function_has_return = False
        self.current_function_has_assert = False

        # Check if this is a test function
        if node.name.startswith("test_"):
            self.test_functions.append(node)
            self._check_test_markers(node)
            self._check_type_annotations(node)

        self.generic_visit(node)

        # After visiting the function body, check return/assert requirements for test functions
        if node.name.startswith("test_"):
            self._check_function_structure(node)

        self.current_function = None
        self.current_function_has_return = False
        self.current_function_has_assert = False

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    @override
    def visit_Assert(self, node: ast.Assert) -> None:
        self.current_function_has_assert = True
        self._analyze_assertion(node)
        self.generic_visit(node)

    @override
    def visit_Return(self, node: ast.Return) -> None:
        # Check if this return statement is in a test function (only for test files)

        if (
            self.is_test_file
            and self.current_function
            and self.current_function.startswith("test_")
            and not self._should_ignore_node(node)
        ):
            self.current_function_has_return = True
            lineno = getattr(node, "lineno", 0)
            self._add_issue(
                node, "return_statement", f"Return statement at line {lineno} in test function"
            )
        self.generic_visit(node)

    @override
    def visit_Call(self, node: ast.Call) -> None:
        # Check for print statements (only in test files)
        if (
            self.is_test_file
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
            and not self._should_ignore_node(node)
        ):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(node, "print_statement", f"Print statement at line {lineno}")

        # Check for hasattr calls (in all files)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
            and not self._should_ignore_node(node)
        ):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(
                node,
                "hasattr_usage",
                f"Use of 'hasattr' at line {lineno}. Use 'isinstance' checks and cast to a specific type instead. Avoid 'Any' usage.",
            )

        # Check for getattr calls (in all files)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and not self._should_ignore_node(node)
        ):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(
                node,
                "getattr_usage",
                f"Use of 'getattr' at line {lineno}. Use 'isinstance' checks and cast to a specific type instead. Avoid 'Any' usage.",
            )

        # Check for pytest.main calls (only in test files)
        if self.is_test_file and isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            # Check for pytest.main
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pytest"
                and method_name == "main"
                and not self._should_ignore_node(node)
            ):
                lineno = getattr(node, "lineno", 0)
                self._add_issue(node, "pytest_main", f"pytest.main() call at line {lineno}")

            # Check for specific assertion methods like assertEqual, assertTrue, etc.
            if method_name.startswith("assert"):
                self._analyze_assert_method(node, method_name)
        self.generic_visit(node)

    @override
    def visit_Name(self, node: ast.Name) -> None:
        if "fallback" in node.id.lower() and not self._should_ignore_node(node):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(
                node,
                "fallback_usage",
                f"Variable name '{node.id}' contains 'fallback' at line {lineno}. This is potentially useless.",
            )
        self.generic_visit(node)

    @override
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if "fallback" in node.attr.lower() and not self._should_ignore_node(node):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(
                node,
                "fallback_usage",
                f"Attribute name '{node.attr}' contains 'fallback' at line {lineno}. This is potentially useless.",
            )
        self.generic_visit(node)

    @override
    def visit_Constant(self, node: ast.Constant) -> None:
        if (
            isinstance(node.value, str)
            and "fallback" in node.value.lower()
            and not self._should_ignore_node(node)
        ):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(
                node,
                "fallback_usage",
                f"String literal contains 'fallback' at line {lineno}. This is potentially useless.",
            )
        self.generic_visit(node)

    def _is_type_checking_block(self, node: ast.If) -> bool:
        """Check if this is an 'if TYPE_CHECKING:' block."""
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return True
        if isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING":
            return True
        return False

    def _check_prohibited_type_checking_imports(self, node: ast.If) -> None:
        """Check for imports of core dependencies inside TYPE_CHECKING blocks."""
        # List of core dependencies from pyproject.toml that should be imported directly
        PROHIBITED_IN_TYPE_CHECKING = {
            "fastapi",
            "openai",
            "pydantic",
            "tenacity",
            "tantivy",
            "agno",
            "faster_whisper",  # underscore for import name
            "sqlalchemy",
            "litellm",
        }

        for stmt in node.body:
            if isinstance(stmt, ast.Import):
                for name in stmt.names:
                    base_module = name.name.split(".")[0]
                    if base_module in PROHIBITED_IN_TYPE_CHECKING:
                        lineno = getattr(stmt, "lineno", 0)
                        self._add_issue(
                            stmt,
                            "prohibited_type_checking_import",
                            f"Import of core dependency '{base_module}' inside TYPE_CHECKING block at line {lineno}. Import directly at top level.",
                        )
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.module:
                    base_module = stmt.module.split(".")[0]
                    if base_module in PROHIBITED_IN_TYPE_CHECKING:
                        lineno = getattr(stmt, "lineno", 0)
                        self._add_issue(
                            stmt,
                            "prohibited_type_checking_import",
                            f"Import of core dependency '{base_module}' inside TYPE_CHECKING block at line {lineno}. Import directly at top level.",
                        )

    def _is_empty_type_checking_block(self, node: ast.If) -> bool:
        """Check if this is an empty 'if TYPE_CHECKING:' block."""
        if not self._is_type_checking_block(node):
            return False

        # Check body is just 'pass' or '...'
        if len(node.body) != 1:
            return False

        stmt = node.body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            return True

        return False

    @override
    def visit_If(self, node: ast.If) -> None:
        # Check for TYPE_CHECKING blocks
        if self._is_type_checking_block(node) and not self._should_ignore_node(node):
            # Check for empty blocks
            if self._is_empty_type_checking_block(node):
                lineno = getattr(node, "lineno", 0)
                self._add_issue(
                    node,
                    "empty_type_checking",
                    f"Empty 'if TYPE_CHECKING:' block at line {lineno}",
                )
            # Check for prohibited imports
            self._check_prohibited_type_checking_imports(node)

        if self.is_test_file and not self._should_ignore_node(node):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(
                node,
                "if_statement",
                f"If statement at line {lineno}. Tests should be deterministic and linear. Use assertions or split into multiple tests instead of using conditional logic.",
            )
        self.generic_visit(node)

    @override
    def visit_For(self, node: ast.For) -> None:
        if self.is_test_file and not self._should_ignore_node(node):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(node, "for_loop", f"For loop at line {lineno}")
        self.generic_visit(node)

    @override
    def visit_While(self, node: ast.While) -> None:
        if self.is_test_file and not self._should_ignore_node(node):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(node, "while_loop", f"While loop at line {lineno}")
        self.generic_visit(node)

    @override
    def visit_Try(self, node: ast.Try) -> None:
        if self.is_test_file and not self._should_ignore_node(node):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(node, "try_except", f"Try/except block at line {lineno}")
        self.generic_visit(node)

    @override
    def visit_Raise(self, node: ast.Raise) -> None:
        if self.is_test_file and not self._should_ignore_node(node):
            lineno = getattr(node, "lineno", 0)
            self._add_issue(node, "raise_statement", f"Raise statement at line {lineno}")
        self.generic_visit(node)

    def _add_issue(self, node: ast.AST, issue_type: str, message: str) -> None:
        """Helper method to add an issue to the list."""
        # AST nodes don't guarantee lineno attribute, so we need to check
        lineno = getattr(node, "lineno", 0)
        self.issues.append(
            {
                "location": f"{Path(self.file_path).name}:{lineno}",
                "function": self.current_function,
                "class": self.current_class,
                "issue_type": issue_type,
                "message": message,
                "line": lineno,
            }
        )

    def _analyze_assertion(self, node: ast.Assert) -> None:
        """Analyze assert statements for quality issues."""
        lineno = getattr(node, "lineno", 0)
        location = f"{Path(self.file_path).name}:{lineno}"

        # Look for magic values in assertions
        magic_values = self._find_magic_values(node.test)
        for value in magic_values:
            self.issues.append(
                {
                    "location": location,
                    "function": self.current_function,
                    "class": self.current_class,
                    "issue_type": "magic_value",
                    "value": value,
                    "message": f"Magic value '{value}' found in assertion",
                    "line": lineno,
                }
            )

        # Check for vague comparisons
        if self._is_vague_comparison(node.test):
            self.issues.append(
                {
                    "location": location,
                    "function": self.current_function,
                    "class": self.current_class,
                    "issue_type": "vague_assertion",
                    "message": "Vague assertion found (e.g. 'is not None' without property checks)",
                    "line": lineno,
                }
            )

    def _analyze_assert_method(self, node: ast.Call, method_name: str) -> None:
        """Analyze unittest assertion methods."""
        lineno = getattr(node, "lineno", 0)
        location = f"{Path(self.file_path).name}:{lineno}"

        # Check arguments for magic values
        for arg in node.args:
            magic_values = self._find_magic_values(arg)
            for value in magic_values:
                self.issues.append(
                    {
                        "location": location,
                        "function": self.current_function,
                        "class": self.current_class,
                        "issue_type": "magic_value",
                        "value": value,
                        "message": f"Magic value '{value}' found in assertion method",
                        "line": lineno,
                    }
                )

        # Check for vague assertions in methods like assertTrue, assertFalse
        if method_name in ["assertTrue", "assertFalse", "assertIn", "assertNotIn"]:
            if len(node.args) < 2:  # Single argument asserts are often vague
                self.issues.append(
                    {
                        "location": location,
                        "function": self.current_function,
                        "class": self.current_class,
                        "issue_type": "vague_assertion",
                        "method": method_name,
                        "message": f"Vague assertion method '{method_name}' found",
                        "line": lineno,
                    }
                )

    def _find_magic_values(self, node: ast.AST) -> list[str]:
        """Find magic values (unexplained numbers) in AST node."""
        magic_values: list[str] = []

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            # Check if it's a magic value (not a common constant)
            if node.value not in COMMON_CONSTANTS:
                magic_values.append(str(node.value))
        elif isinstance(node, ast.BinOp):
            magic_values.extend(self._find_magic_values(node.left))
            magic_values.extend(self._find_magic_values(node.right))
        elif isinstance(node, ast.UnaryOp):
            magic_values.extend(self._find_magic_values(node.operand))
        elif isinstance(node, ast.Compare):
            for comparator in node.comparators:
                magic_values.extend(self._find_magic_values(comparator))

        return magic_values

    def _has_only_vague_none_assertions(self, node: ast.FunctionDef) -> bool:
        """Check if the test function only has vague 'is not None' assertions."""
        none_assertions: list[ast.Assert] = []
        meaningful_assertions: list[ast.Assert] = []

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assert):
                # Check if this is a 'X is not None' assertion
                if (
                    isinstance(stmt.test, ast.Compare)
                    and any(isinstance(op, ast.IsNot) for op in stmt.test.ops)
                    and any(
                        isinstance(comp, ast.Constant) and comp.value is None
                        for comp in stmt.test.comparators
                    )
                ):
                    none_assertions.append(stmt)
                else:
                    # This is a meaningful assertion
                    meaningful_assertions.append(stmt)

        # If there are None assertions but no meaningful assertions, it's a vague test
        return len(none_assertions) > 0 and len(meaningful_assertions) == 0

    def _is_vague_comparison(self, node: ast.AST) -> bool:
        """Check if assertion uses vague comparisons."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # Check for functions that just check existence without verifying exact properties
            func_name = node.func.attr
            vague_functions = ["any", "len", "bool", "not"]
            if func_name in vague_functions:
                return True

        # Check for "is not None" patterns - only flag if it's NOT followed by property checks
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    # Found comparison with None - now check the operator
                    for op in node.ops:
                        if isinstance(op, ast.IsNot):
                            # Check if this is a standalone assertion by looking at the parent function context
                            if self._is_standalone_none_assertion(node):
                                return True

        return False

    def _is_standalone_none_assertion(self, node: ast.AST) -> bool:
        """Check if this is a standalone 'is not None' assertion (not followed by property checks)."""
        # This is a simple check - in a full implementation, you'd want to look at the broader context
        # At present, we'll consider it standalone unless we can see it's part of a sequence
        # In practice, developers usually write meaningful assertions right after None checks
        return False  # Be conservative - don't flag None checks that might be followed by property checks

    def _check_test_markers(self, node: ast.FunctionDef) -> None:
        """Check if test function has proper pytest markers."""
        # Look for pytest markers in decorator list
        has_unit_marker = False
        has_integration_marker = False
        has_agent_test_marker = False

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute):
                # Check for pytest.mark.unit or pytest.mark.integration
                if (
                    isinstance(decorator.value, ast.Attribute)
                    and isinstance(decorator.value.value, ast.Name)
                    and decorator.value.value.id == "pytest"
                    and decorator.value.attr == "mark"
                ):
                    if decorator.attr == "unit":
                        has_unit_marker = True
                    elif decorator.attr == "integration":
                        has_integration_marker = True
                    elif decorator.attr == "agent_test":
                        has_agent_test_marker = True

        # Require either unit or integration marker
        if not (has_unit_marker or has_integration_marker):
            self._add_issue(
                node,
                "missing_marker",
                f"Test function '{node.name}' missing required @pytest.mark.unit or @pytest.mark.integration marker",
            )

        # If agent_test marker is present, ensure integration marker is also present
        if has_agent_test_marker and not has_integration_marker:
            self._add_issue(
                node,
                "agent_test_requires_integration",
                f"Test function '{node.name}' has @pytest.mark.agent_test but missing @pytest.mark.integration marker",
            )

    def _check_type_annotations(self, node: ast.FunctionDef) -> None:
        """Check if test function has proper type annotations."""
        # Check for return type annotation (should be -> None for test functions)
        if node.returns is None:
            self._add_issue(
                node,
                "missing_return_annotation",
                f"Test function '{node.name}' missing return type annotation (should be '-> None')",
            )
        else:
            # Check if return annotation is 'None'
            if isinstance(node.returns, ast.Constant) and node.returns.value is None:
                # Good case: -> None
                pass
            elif isinstance(node.returns, ast.Name) and node.returns.id == "None":
                # Good case: -> None (name form)
                pass
            else:
                # Return type is present but not None
                return_type_str = (
                    ast.unparse(node.returns)
                    if hasattr(ast, "unparse")
                    else str(type(node.returns).__name__)
                )
                self._add_issue(
                    node,
                    "wrong_return_annotation",
                    f"Test function '{node.name}' has return type '{return_type_str}' but should be 'None'",
                )

        # Check for parameter type annotations
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            if arg.annotation is None:
                self._add_issue(
                    node,
                    "missing_parameter_annotation",
                    f"Test function '{node.name}' parameter '{arg.arg}' missing type annotation",
                )

    def _check_function_structure(self, node: ast.FunctionDef) -> None:
        """Check if test function follows proper structure (no returns, has asserts)."""
        # Check if test function has return statement
        if self.current_function_has_return:
            self._add_issue(
                node, "return_statement", f"Test function '{node.name}' contains return statement"
            )

        # Check if test function has at least one assertion
        if not self.current_function_has_assert:
            self._add_issue(
                node, "missing_assertion", f"Test function '{node.name}' has no assertions"
            )

        # Check for vague None assertions that don't have meaningful property checks
        if self._has_only_vague_none_assertions(node):
            self._add_issue(
                node,
                "vague_none_assertion",
                f"Test function '{node.name}' only has 'is not None' assertions without testing actual properties",
            )


def is_test_file(file_path: str) -> bool:
    """Determine if a file is a test file based on its path and name."""
    path = Path(file_path)
    filename = path.name
    if filename.endswith("_test.py") or filename.startswith("test_"):
        return True
    if "test" in path.parts:
        return True
    return False


def analyze_test_file(file_path: str) -> dict[str, Any]:
    """Analyze a single file for quality issues."""
    try:
        with open(file_path) as f:
            content = f.read()
    except FileNotFoundError:
        return {"error": f"File not found: {file_path}", "total_issues": 0, "issues": []}

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"error": f"Syntax error in {file_path}: {e}", "total_issues": 0, "issues": []}

    is_test = is_test_file(file_path)
    analyzer = QualityAnalyzer(file_path, is_test_file=is_test)
    analyzer.visit(tree)

    dev_analyzer = DevelopmentIndicatorAnalyzer(file_path)
    dev_analyzer.source_code = content
    dev_analyzer.visit(tree)

    for indicator in dev_analyzer.dev_indicators:
        analyzer.issues.append(
            {
                "line": indicator["line"],
                "type": "Development Indicator",
                "message": f"Found development indicator '{indicator['keyword']}': {indicator['content']}",
                "suggestion": "Review development stage and implement if needed.",
            }
        )

    ignored_lines_count = len(analyzer.ignored_lines)
    for start, end in analyzer.ignored_blocks:
        ignored_lines_count += end - start + 1

    return {
        "file_path": file_path,
        "issues": analyzer.issues,
        "total_issues": len(analyzer.issues),
        "is_test_file": is_test,
        "ignored_lines_count": ignored_lines_count,
    }


def analyze_development_indicators_file(file_path: str) -> dict[str, Any]:
    """Analyze a single file for development stage keywords."""
    try:
        analyzer = DevelopmentIndicatorAnalyzer(file_path)
        return analyzer.analyze_file()
    except Exception as e:
        return {"error": f"Error analyzing {file_path}: {e}"}


def analyze_development_indicators_codebase(root_path: str = ".") -> dict[str, Any]:
    """Analyze the src directory for development stage indicators."""
    try:
        dev_indicators: list[dict[str, Any]] = []
        analyzed_files = 0
        skipped_files = 0

        src_path = Path(root_path) / "src"
        if not src_path.exists():
            return {"error": f"src directory not found at {src_path}"}

        python_files: list[str] = []
        for root, dirs, files in os.walk(src_path):
            dirs[:] = [
                d
                for d in dirs
                if d not in ["__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"]
            ]
            for file in files:
                if file.endswith(".py"):
                    python_files.append(str(Path(root) / file))

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                if not content.strip():
                    skipped_files += 1
                    continue

                analyzer = DevelopmentIndicatorAnalyzer(file_path)
                result = analyzer.analyze_file()

                if "error" not in result and result["dev_indicators"]:
                    dev_indicators.extend(result["dev_indicators"])
                    analyzed_files += 1
                elif "error" in result:
                    skipped_files += 1
                else:
                    analyzed_files += 1

            except Exception:
                skipped_files += 1
                continue

        dev_indicators.sort(key=lambda x: (x["file"], x["line"]))

        return {
            "dev_indicators": dev_indicators,
            "total_found": len(dev_indicators),
            "analyzed_files": analyzed_files,
            "skipped_files": skipped_files,
            "total_python_files": len(python_files),
        }

    except Exception as e:
        return {"error": f"Error analyzing codebase: {e}"}


def run_pyright_check(file_path: str) -> tuple[bool, str]:
    """Run pyright on the file."""
    cmd = ["npx", "pyright", "--outputjson", file_path]
    success, output = StreamSubprocess.run_silent(cmd, cwd=Path.cwd())

    # If pyright returns success (exit code 0), check if there are actually errors/warnings in the JSON
    if success and output.strip().startswith("{"):
        try:
            data = json.loads(output)
            summary = data.get("summary", {})
            if summary.get("errorCount", 0) > 0 or summary.get("warningCount", 0) > 0:
                return False, output
        except json.JSONDecodeError:
            pass

    return success, output


def format_pyright_output(json_output: str) -> str:
    """Format pyright JSON output into a readable table."""
    try:
        data = json.loads(json_output)
    except json.JSONDecodeError:
        return json_output

    diagnostics = data.get("generalDiagnostics", [])
    if not diagnostics:
        return "No type errors found."

    table_data: list[list[Any]] = []
    for d in diagnostics:
        line = d.get("range", {}).get("start", {}).get("line", 0) + 1
        message = d.get("message", "")
        rule = d.get("rule", "")
        table_data.append([line, rule, message])

    if not table_data:
        return "No type errors found."

    return tabulate_with_wrapping(table_data, headers=["Line", "Rule", "Message"], tablefmt="grid")


def run_ruff_check(
    file_path: str, fix: bool = False, unsafe_fixes: bool = False
) -> tuple[bool, str]:
    """Run ruff check on the file."""
    cmd = ["ruff", "check", file_path]
    if fix:
        cmd.append("--fix")
    if unsafe_fixes:
        cmd.append("--unsafe-fixes")
    return StreamSubprocess.run_silent(cmd, cwd=Path.cwd())


def run_ruff_format(file_path: str) -> tuple[bool, str]:
    """Run ruff format on the file."""
    cmd = ["ruff", "format", file_path]
    return StreamSubprocess.run_silent(cmd, cwd=Path.cwd())


def format_ruff_output(output: str, title: str | None = "Ruff Linting Results") -> str:
    """Format ruff output."""
    if not output.strip():
        return "No issues found."
    return output


def run_test_file(file_path: str) -> tuple[bool, str]:
    """Run the test file using the project's proper test commands."""
    try:
        # Resolve to absolute path to ensure is_relative_to works correctly
        test_path = Path(file_path).resolve()

        if not is_test_file(file_path):
            return True, f"File {file_path} is not a test file - skipping test execution"

        # We use pytest directly instead of poe tasks because poe tasks often have hardcoded
        # directories (e.g. "pytest tests/unit/") which causes pytest to run ALL tests
        # in that directory plus the specific file we want to run.
        # By invoking pytest directly with the specific file, we ensure only that file is run.

        # Use sys.executable to ensure we use the same python environment
        base_cmd = [sys.executable, "-m", "pytest", str(test_path), "--no-cov", "-v"]

        # Determine marker based on directory structure
        integration_dir = (Path.cwd() / "tests" / "integration").resolve()
        unit_dir = (Path.cwd() / "tests" / "unit").resolve()

        if test_path.is_relative_to(integration_dir):
            cmd = base_cmd + ["-m", "integration"]
        elif test_path.is_relative_to(unit_dir):
            cmd = base_cmd + ["-m", "unit"]
        else:
            # Default to integration if location is unclear but it is a test
            cmd = base_cmd + ["-m", "integration"]

        logger.info(f"Running tests for {test_path.name}...")
        # Use run_silent to avoid printing output to stdout (it will be shown in the box)
        success, output = StreamSubprocess.run_silent(cmd, cwd=Path.cwd())
        return success, output
    except Exception as e:
        return False, f"Error running tests: {e}"


def analyze_unused_functions_codebase(root_path: str = ".") -> dict[str, Any]:
    """Analyze the codebase for unused functions.

    Definitions are collected from ``src/`` only, but usages are collected
    from both ``src/`` and ``tests/`` so that functions and methods that are
    only referenced in tests are not reported as unused.
    """
    try:
        src_path = Path(root_path) / "src"
        tests_path = Path(root_path) / "tests"

        if not src_path.exists():
            return {"error": f"src directory not found at {src_path}"}

        def _collect_python_files(base_path: Path) -> list[str]:
            python_files: list[str] = []
            if not base_path.exists():
                return python_files

            for root, dirs, files in os.walk(base_path):
                dirs[:] = [d for d in dirs if d not in ["__pycache__", "node_modules", ".git"]]
                for file in files:
                    if file.endswith(".py"):
                        python_files.append(str(Path(root) / file))
            return python_files

        src_python_files = _collect_python_files(src_path)
        test_python_files = _collect_python_files(tests_path)

        all_defined: dict[str, dict[str, ast.FunctionDef]] = {}
        all_calls: set[str] = set()

        analyzed_files = 0
        skipped_files = 0

        # First, analyze src files: collect both definitions and calls
        for file_path in src_python_files:
            try:
                analyzer = UnusedFunctionAnalyzer(file_path)
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content)
                analyzer.visit(tree)

                all_defined[file_path] = analyzer.defined_functions
                all_calls.update(analyzer.called_functions)
                analyzed_files += 1
            except Exception:
                skipped_files += 1

        # Then, analyze test files: collect only calls so test usages
        # mark functions/methods as used without treating tests as
        # part of the public API surface.
        for file_path in test_python_files:
            try:
                analyzer = UnusedFunctionAnalyzer(file_path)
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content)
                analyzer.visit(tree)

                all_calls.update(analyzer.called_functions)
            except Exception:
                skipped_files += 1

        unused_functions: list[dict[str, Any]] = []
        for file_path, defined in all_defined.items():
            for func_name, node in defined.items():
                is_used = False
                if func_name in all_calls:
                    is_used = True
                else:
                    base_name = func_name.split(".")[-1]
                    if base_name in all_calls:
                        is_used = True

                if not is_used:
                    unused_functions.append(
                        {
                            "file": file_path,
                            "line": node.lineno,
                            "function": func_name,
                            "is_private": func_name.startswith("_")
                            and not func_name.startswith("__"),
                            "is_class_method": "." in func_name,
                        }
                    )

        return {
            "unused_functions": unused_functions,
            "total_unused": len(unused_functions),
            "analyzed_files": analyzed_files,
            "skipped_files": skipped_files,
            "total_python_files": len(src_python_files),
        }
    except Exception as e:
        return {"error": f"Error analyzing codebase: {e}"}


def analyze_unused_functions_file(file_path: str) -> dict[str, Any]:
    """Analyze a single file for unused functions."""
    codebase_analysis = analyze_unused_functions_codebase()
    if "error" in codebase_analysis:
        return codebase_analysis

    unused = [
        f
        for f in codebase_analysis.get("unused_functions", [])
        if str(Path(f["file"]).resolve()) == str(Path(file_path).resolve())
        or f["file"] == file_path
    ]

    return {
        "unused_functions": unused,
        "total_unused": len(unused),
        "analyzed_files": 1,
        "skipped_files": 0,
        "total_python_files": 1,
    }


def format_unused_functions_report(analysis: dict[str, Any]) -> str:
    """Format the unused functions analysis report."""
    if "error" in analysis:
        return f"{C_RED}ERROR: {analysis['error']}{C_RESET}"

    if analysis["total_unused"] == 0:
        return (
            f"{C_GREEN}No unused functions found{C_RESET}\n"
            f"Analyzed {analysis['analyzed_files']} files "
            f"(skipped {analysis['skipped_files']} files)"
        )

    report_lines = [
        f"{C_RED}Unused functions detected{C_RESET}",
        f"Found {analysis['total_unused']} unused functions across {analysis['analyzed_files']} files",
        f"(skipped {analysis['skipped_files']} files)",
        "",
    ]

    functions_by_file: dict[str, list[dict[str, Any]]] = {}
    for func in analysis["unused_functions"]:
        file_name = func["file"]
        if file_name not in functions_by_file:
            functions_by_file[file_name] = []
        functions_by_file[file_name].append(func)

    for file_name, functions in functions_by_file.items():
        report_lines.append(f"File: {file_name}")
        table_data: list[list[Any]] = []
        for func in functions:
            function_type = (
                "Private method"
                if func["is_private"]
                else "Function"
                if not func["is_class_method"]
                else "Method"
            )
            table_data.append(
                [
                    func["line"],
                    function_type,
                    func["function"],
                    "MANUAL REVIEW" if func["is_class_method"] else "SAFE TO REMOVE?",
                ]
            )

        if table_data:
            headers = ["Line", "Type", "Function", "Suggested Action"]
            table_str = tabulate_with_wrapping(table_data, headers=headers, tablefmt="grid")
            report_lines.append(table_str)

        report_lines.append("")

    return "\n".join(report_lines)


def format_development_indicators_report(
    analysis: dict[str, Any],
    claude_analysis: tuple[bool, str, list[dict[str, Any]]] | None = None,
) -> str:
    """Format the development indicators analysis report."""
    if "error" in analysis:
        return f"{C_RED}ERROR: {analysis['error']}{C_RESET}"

    if analysis["total_found"] == 0:
        if "analyzed_files" in analysis:
            return (
                f"{C_GREEN}No development indicators found{C_RESET}\n"
                f"Analyzed {analysis['analyzed_files']} files "
                f"(skipped {analysis['skipped_files']} files)"
            )
        else:
            return f"{C_GREEN}No development indicators found{C_RESET}"

    if "analyzed_files" in analysis:
        report_lines = [
            f"{C_RED}Development indicators detected{C_RESET}",
            f"Found {analysis['total_found']} development indicators across {analysis['analyzed_files']} files",
            f"(skipped {analysis['skipped_files']} files)",
            "",
        ]
    else:
        file_name = Path(analysis.get("file_path", "")).name
        report_lines = [
            f"{C_RED}Development indicators detected{C_RESET}",
            f"Found {analysis['total_found']} development indicators in {file_name}",
            "",
        ]

    indicators_by_file: dict[str, list[dict[str, Any]]] = {}
    for indicator in analysis["dev_indicators"]:
        file_name = indicator["file"]
        if file_name not in indicators_by_file:
            indicators_by_file[file_name] = []
        indicators_by_file[file_name].append(indicator)

    for file_name, indicators in indicators_by_file.items():
        report_lines.append(f"File: {file_name}")
        table_data: list[list[Any]] = []
        for indicator in indicators:
            table_data.append(
                [
                    indicator["line"],
                    indicator["keyword"].upper(),
                    indicator["content"][:80] + "..."
                    if len(indicator["content"]) > 80
                    else indicator["content"],
                ]
            )

        if table_data:
            headers = ["Line", "Keyword", "Content"]
            table_str = tabulate_with_wrapping(table_data, headers=headers, tablefmt="grid")
            report_lines.append(table_str)

        report_lines.append("")

    if claude_analysis:
        success, claude_output, _ = claude_analysis
        if success:
            report_lines.extend(["AI ANALYSIS RESULTS:", "", claude_output, ""])
        else:
            report_lines.extend(["AI ANALYSIS FAILED:", "", claude_output, ""])

    report_lines.extend(
        [
            "RECOMMENDATIONS:",
            "",
            "• Review each occurrence to determine if it needs attention",
            "• Prioritize addressing high-confidence development needs",
            "• Consider the business impact and time investment required",
            "• Document any exceptions where indicators are acceptable",
            "",
            "Common patterns to watch for:",
            "- 'minimal' - Often indicates simple implementation",  # ignore-development
            "- 'temporarily' - Temporary implementation approach",  # ignore-development
            "- 'workaround' - Quick solutions needing refinement",  # ignore-development
            "- 'incomplete' - Implementations needing completion",  # ignore-development
        ]
    )

    return "\n".join(report_lines)


def generate_quality_report(analysis: dict[str, Any]) -> str:
    """Generate a quality report with suggestions for fixes."""
    if "error" in analysis:
        return f"{C_RED}ERROR: {analysis['error']}{C_RESET}"

    if analysis["total_issues"] == 0:
        ignored_info = ""
        if analysis.get("ignored_lines_count", 0) > 0:
            ignored_info = f" (with {analysis['ignored_lines_count']} ignored line(s))"
        return f"{C_GREEN}Tests meet quality standards{C_RESET}{ignored_info}"

    issues_by_type: dict[str, list[dict[str, Any]]] = {}
    for issue in analysis["issues"]:
        issue_type = issue["issue_type"] if "issue_type" in issue else issue.get("type", "Unknown")
        if issue_type not in issues_by_type:
            issues_by_type[issue_type] = []
        issues_by_type[issue_type].append(issue)

    report_lines = [f"{C_RED}Critical test quality issues found{C_RESET}\n"]

    for issue_type, issues in issues_by_type.items():
        report_lines.append(f"Type: {issue_type}")
        table_data: list[list[str]] = []
        for issue in issues:
            table_data.append([str(issue.get("line", "?")), issue.get("message", "")])

        if table_data:
            headers = ["Line", "Message"]
            table_str = tabulate_with_wrapping(table_data, headers=headers, tablefmt="grid")
            report_lines.append(table_str)
        report_lines.append("")

    return "\n".join(report_lines)


def apply_development_indicator_ignores(items: list[dict[str, Any]]) -> tuple[bool, str]:
    """Apply ignore directives for acceptable development indicators."""
    if not items:
        return True, "No items to ignore."

    applied_count = 0
    errors: list[str] = []

    items_by_file: dict[str, list[int]] = {}
    for item in items:
        file_path = item["file"]
        if file_path not in items_by_file:
            items_by_file[file_path] = []
        items_by_file[file_path].append(item["line"])

    for file_path, lines in items_by_file.items():
        try:
            with open(file_path, encoding="utf-8") as f:
                content_lines = f.readlines()

            lines.sort(reverse=True)

            modified = False
            for line_num in lines:
                idx = line_num - 1
                if 0 <= idx < len(content_lines):
                    if idx > 0 and "# ignore-development" in content_lines[idx - 1]:
                        continue

                    indentation = ""
                    current_line = content_lines[idx]
                    stripped = current_line.lstrip()
                    if stripped:
                        indentation = current_line[: len(current_line) - len(stripped)]

                    content_lines.insert(idx, f"{indentation}# ignore-development\n")
                    modified = True
                    applied_count += 1

            if modified:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(content_lines)

        except Exception as e:
            errors.append(f"Error updating {file_path}: {e}")

    if errors:
        return False, "\n".join(errors)

    return True, f"Applied {applied_count} ignore directives."


def main() -> None:
    """Main function to run the test quality analyzer with automatic fixes."""
    total_start_time = time.perf_counter()

    # Parse arguments
    max_iterations = DEFAULT_MAX_ITERATIONS
    check_unused = False
    no_test_execution = False
    test_file = None

    # Check for flags
    if "--unused-functions" in sys.argv:
        check_unused = True

    if "--no-test-execution" in sys.argv:
        no_test_execution = True

    # Extract max_iterations if provided
    for i, arg in enumerate(sys.argv):
        if arg == "--max-iterations" and i + 1 < len(sys.argv):
            try:
                max_iterations = int(sys.argv[i + 1])
            except ValueError:
                logger.error("ERROR: --max-iterations requires a positive integer")
                sys.exit(1)
            break

    # Get the test file if not checking unused functions only
    if not check_unused:
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                test_file = arg
                break

        if not test_file:
            logger.info("Usage: python quality_enforcer.py <test_file_path> [--max-iterations N]")
            logger.info("       python quality_enforcer.py --unused-functions")
            logger.info("Note: Auto-fix and test execution are always enabled")
            sys.exit(1)
    else:
        # If --unused-functions only, no test file analysis needed
        test_file = None

    # ALWAYS run unused functions analysis first
    analysis_start = time.perf_counter()

    # Analyze single file or full codebase based on whether test_file is provided
    if test_file:
        unused_analysis = analyze_unused_functions_file(test_file)
    else:
        unused_analysis = analyze_unused_functions_codebase()

    analysis_elapsed = time.perf_counter() - analysis_start

    # Display unused functions analysis - ALWAYS in table format
    unused_report = format_unused_functions_report(unused_analysis)
    box_title = f"UNUSED FUNCTIONS ANALYSIS - {Path(test_file).name if test_file else 'CODEBASE'}"
    analysis_box = create_section_box(box_title, unused_report, analysis_elapsed, BOX_WIDTH)
    logger.info("\n" + analysis_box)

    # Analyze development indicators (development stage keywords)
    dev_start = time.perf_counter()

    # Analyze single file or full codebase based on whether test_file is provided
    if test_file:
        dev_analysis = analyze_development_indicators_file(test_file)
    else:
        dev_analysis = analyze_development_indicators_codebase()

    dev_elapsed = time.perf_counter() - dev_start

    # Run Claude analysis for development indicators
    claude_analysis = None
    if dev_analysis.get("dev_indicators"):
        logger.info("Running AI analysis for development indicators...")
        claude_analysis = analyze_development_indicators_with_claude(
            dev_analysis["dev_indicators"], test_file
        )

        # Apply ignores if acceptable items found
        if claude_analysis and len(claude_analysis) >= 3:
            acceptable_items = claude_analysis[2]
            if acceptable_items:
                logger.info(
                    f"Found {len(acceptable_items)} acceptable development indicators. Applying ignores..."
                )
                ignore_success, ignore_msg = apply_development_indicator_ignores(acceptable_items)
                if ignore_success:
                    logger.info(f"Successfully applied ignores: {ignore_msg}")
                else:
                    logger.error(f"Failed to apply ignores: {ignore_msg}")

    # Display development indicators analysis - ALWAYS in table format
    dev_report = format_development_indicators_report(dev_analysis, claude_analysis)
    dev_box_title = (
        f"DEVELOPMENT INDICATORS ANALYSIS - {Path(test_file).name if test_file else 'CODEBASE'}"
    )
    dev_box = create_section_box(dev_box_title, dev_report, dev_elapsed, BOX_WIDTH)
    logger.info("\n" + dev_box)

    # Handle --unused-functions only mode
    if check_unused:
        # Exit with appropriate code
        if "error" in unused_analysis or unused_analysis.get("total_unused", 0) > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    # Only run test file analysis if we have a test file
    if test_file is None:
        logger.error("No test file provided for quality analysis")
        sys.exit(1)

    # Auto-fix loop - always apply fixes
    current_analysis: dict[str, Any] = {}

    # Initial analysis to check if fixes are needed
    initial_analysis_start = time.perf_counter()
    initial_analysis = analyze_test_file(test_file)
    initial_analysis_elapsed = time.perf_counter() - initial_analysis_start

    if initial_analysis["total_issues"] > 0:
        for iteration in range(max_iterations):
            logger.info(f"\nFIX ITERATION {iteration + 1}/{max_iterations}")
            logger.info("=" * 60)

            # Step 1: Analyze the file for quality issues
            analysis_start = time.perf_counter()
            current_analysis = analyze_test_file(test_file)
            quality_report = generate_quality_report(current_analysis)
            analysis_elapsed = time.perf_counter() - analysis_start

            # Display quality analysis in a box
            analysis_box = create_section_box(
                f"QUALITY ANALYSIS - ITERATION {iteration + 1}/{max_iterations}",
                quality_report,
                analysis_elapsed,
                BOX_WIDTH,
            )
            logger.info("\n" + analysis_box)

            # If no issues, we're done
            if current_analysis["total_issues"] == 0:
                success_box = create_section_box(
                    "QUALITY ISSUES FIXED",
                    "No more issues detected. File meets all quality standards.",
                    0,
                    BOX_WIDTH,
                )
                logger.info("\n" + success_box)
                break

            # Apply fixes for any issues found
            if current_analysis["total_issues"] > 0:
                fix_start = time.perf_counter()
                fix_success, fix_output = run_claude_fix(test_file, current_analysis["issues"])
                fix_elapsed = time.perf_counter() - fix_start

                if fix_success:
                    fix_content = "Fixes applied successfully!\n\nCLAUDE OUTPUT:\n" + fix_output
                    fix_box = create_section_box(
                        "AUTOMATIC FIXES APPLIED", fix_content, fix_elapsed, BOX_WIDTH
                    )
                    logger.info("\n" + fix_box)
                else:
                    fix_content = "Failed to apply fixes\n\n" + fix_output
                    fix_box = create_section_box(
                        "FIX APPLICATION FAILED", fix_content, fix_elapsed, BOX_WIDTH
                    )
                    logger.info("\n" + fix_box)
                    break

                # Brief pause to avoid rate limiting
                time.sleep(2)
            else:
                # Not in fix mode, so break after first analysis
                break
    else:
        # No issues found initially
        current_analysis = initial_analysis
        # Still display the "perfect" result for quality analysis
        quality_report = generate_quality_report(current_analysis)
        analysis_box = create_section_box(
            "QUALITY ANALYSIS",
            quality_report,
            initial_analysis_elapsed,
            BOX_WIDTH,
        )
        logger.info("\n" + analysis_box)

    # Step 2: Run pyright type checking with refine loop (after all fixes)
    type_check_pass = True
    type_check_start = time.perf_counter()
    type_success, type_output = run_pyright_check(test_file)
    type_check_elapsed = time.perf_counter() - type_check_start

    # Format pyright output as a table and display in a box
    formatted_type_output = format_pyright_output(type_output)
    pyright_box = create_section_box(
        "PYRIGHT TYPE CHECKING", formatted_type_output, type_check_elapsed, BOX_WIDTH
    )
    logger.info("\n" + pyright_box)
    type_check_pass = type_success

    # Refine loop for pyright if type errors found
    if not type_check_pass:
        # Parse error count from output or assume errors if check failed
        error_count = 0
        if "errorCount" in type_output:
            type_data: dict[str, Any] = (
                json.loads(type_output) if type_output.strip().startswith("{") else {}
            )
            error_count = type_data.get("summary", {}).get("errorCount", 0)
            # Also count warnings as errors for the purpose of refinement if they are significant
            warning_count = type_data.get("summary", {}).get("warningCount", 0)
            if error_count == 0 and warning_count > 0:
                # Treat warnings as errors for refinement
                error_count = warning_count

        if error_count > 0:
            logger.info("Running refine loop for pyright type errors...")
            logger.info(f"Found {error_count} type errors to fix")
            for refine_iteration in range(1):  # Max 1 additional iteration to prevent hangs
                logger.info(f"Pyright refinement {refine_iteration + 1}/1 (timeout: 5 minutes)")

                # Try to fix type errors with Claude
                try:
                    # Create a prompt specifically for type errors
                    # This will be updated with recheck output on subsequent iterations
                    type_fix_prompt = f"""Please fix the following type errors in {test_file}:

{formatted_type_output}

Focus on:
1. Adding missing type annotations
2. Fixing type mismatch errors
3. Adding proper imports for types
4. Ensuring all function signatures have correct types

After fixing, the file should pass pyright type checking with no errors."""

                    # Run Claude to fix type errors
                    z_ai_api_key = os.environ.get("Z_AI_API_KEY")
                    if z_ai_api_key:
                        env = os.environ.copy()
                        env.update(
                            {
                                "ANTHROPIC_AUTH_TOKEN": z_ai_api_key,
                                "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                                "ANTHROPIC_DEFAULT_OPUS_MODEL": "GLM-4.6",
                                "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM-4.6",
                                "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GLM-4.5-Air",
                                "ENABLE_BACKGROUND_TASKS": "1",
                            }
                        )

                        claude_cmd = [
                            "claude",
                            "--print",
                            "--dangerously-skip-permissions",
                            "--",
                            type_fix_prompt,
                        ]
                        logger.info(
                            "Calling Claude to fix type errors (this may take a few minutes)..."
                        )
                        claude_start = time.perf_counter()
                        claude_success, claude_output = StreamSubprocess.run(
                            claude_cmd, cwd=Path.cwd(), env=env, timeout=300, stream=False
                        )
                        claude_elapsed = time.perf_counter() - claude_start

                        if claude_success:
                            # Clean up Claude output to remove [claude] prefix
                            cleaned_output = "\n".join(
                                line.replace("[claude] ", "") for line in claude_output.split("\n")
                            )
                            claude_box = create_section_box(
                                "CLAUDE TYPE FIXES", cleaned_output, claude_elapsed, BOX_WIDTH
                            )
                            logger.info("\n" + claude_box)
                            time.sleep(2)  # Brief pause

                            # Re-run pyright to check if issues are resolved
                            recheck_success, recheck_output = run_pyright_check(test_file)
                            if recheck_success:
                                type_check_pass = True
                                logger.info("Pyright type errors resolved")
                                break
                            else:
                                logger.warning(
                                    f"Pyright still has type errors after refinement {refine_iteration + 1}"
                                )
                                # Update the prompt with the new recheck output for next iteration
                                formatted_type_output = format_pyright_output(recheck_output)
                        else:
                            logger.warning(
                                f"Claude failed to apply type fixes in refinement {refine_iteration + 1}"
                            )
                    else:
                        logger.warning("Z_AI_API_KEY not set, cannot run Claude for type fixes")
                        break

                except Exception as e:
                    logger.error(f"❌ Error in pyright refinement {refine_iteration + 1}: {e}")
                    time.sleep(1)

    # Step 2.5: Run Ruff linting and formatting
    ruff_check_pass = True
    ruff_format_pass = True
    ruff_start = time.perf_counter()

    # Run linting
    ruff_check_success, ruff_check_output = run_ruff_check(test_file)

    # If linting failed, try to auto-fix
    if not ruff_check_success:
        logger.info("Attempting to auto-fix Ruff linting issues...")
        # We don't check the result of fix run, as we'll re-check anyway
        run_ruff_check(test_file, fix=True)

        # Re-run check to see if issues persist
        ruff_check_success, ruff_check_output = run_ruff_check(test_file)
        if ruff_check_success:
            logger.info("Ruff linting issues resolved by auto-fix")
        else:
            # If issues persist, try unsafe fixes
            logger.info("Attempting to auto-fix Ruff linting issues with --unsafe-fixes...")
            run_ruff_check(test_file, fix=True, unsafe_fixes=True)

            # Re-run check again
            ruff_check_success, ruff_check_output = run_ruff_check(test_file)
            if ruff_check_success:
                logger.info("Ruff linting issues resolved by unsafe auto-fix")

    # Run formatting
    ruff_format_success, ruff_format_output = run_ruff_format(test_file)

    ruff_elapsed = time.perf_counter() - ruff_start

    # Prepare combined output
    combined_output_lines: list[str] = []

    if not ruff_check_success:
        combined_output_lines.append("RUFF LINTING ISSUES:")
        combined_output_lines.append(format_ruff_output(ruff_check_output, title=None))
        combined_output_lines.append("")
        ruff_check_pass = False

    if not ruff_format_success:
        combined_output_lines.append("RUFF FORMATTING ISSUES:")
        combined_output_lines.append(format_ruff_output(ruff_format_output, title=None))
        combined_output_lines.append("")
        ruff_format_pass = False

    if ruff_check_pass and ruff_format_pass:
        combined_output_lines.append("No linting or formatting issues found.")

    ruff_box = create_section_box(
        "RUFF LINTING & FORMATTING", "\n".join(combined_output_lines), ruff_elapsed, BOX_WIDTH
    )
    logger.info("\n" + ruff_box)

    # Step 3: Run tests
    tests_pass = True
    if not no_test_execution:
        test_start = time.perf_counter()
        success, test_output = run_test_file(test_file)
        test_elapsed = time.perf_counter() - test_start

        # Display test results in a box
        test_box = create_section_box("TEST EXECUTION", test_output, test_elapsed, BOX_WIDTH)
        logger.info("\n" + test_box)
        tests_pass = success
    else:
        logger.info(
            "\n"
            + create_section_box(
                "TEST EXECUTION", "Skipped due to --no-test-execution flag", 0, BOX_WIDTH
            )
        )

    # Final Summary
    total_elapsed = time.perf_counter() - total_start_time
    summary_lines: list[str] = []

    # Determine overall status
    quality_pass = current_analysis.get("total_issues", 0) == 0
    unused_pass = unused_analysis.get("total_unused", 0) == 0

    all_passed = (
        quality_pass
        and type_check_pass
        and ruff_check_pass
        and ruff_format_pass
        and tests_pass
        and unused_pass
    )

    # Build summary lines with checkmarks
    summary_lines.append(
        f"{C_GREEN if quality_pass else C_RED}{'✓' if quality_pass else '✗'} Quality standards met{C_RESET}"
    )
    summary_lines.append(
        f"{C_GREEN if type_check_pass else C_RED}{'✓' if type_check_pass else '✗'} Type checking passed{C_RESET}"
    )
    summary_lines.append(
        f"{C_GREEN if ruff_check_pass and ruff_format_pass else C_RED}{'✓' if ruff_check_pass and ruff_format_pass else '✗'} Ruff linting passed{C_RESET}"
    )

    if no_test_execution:
        summary_lines.append(f"{C_YELLOW}- Tests skipped{C_RESET}")
    else:
        summary_lines.append(
            f"{C_GREEN if tests_pass else C_RED}{'✓' if tests_pass else '✗'} Tests passed{C_RESET}"
        )

    summary_lines.append(
        f"{C_GREEN if unused_pass else C_RED}{'✓' if unused_pass else '✗'} No unused functions found{C_RESET}"
    )

    # Create the box
    result_text = "PERFECT" if all_passed else "FAILED"
    result_color = C_GREEN if all_passed else C_RED
    box_title = f"FINAL RESULT: {result_color}{result_text}{C_RESET}"

    summary_box = create_section_box(box_title, "\n".join(summary_lines), 0, BOX_WIDTH)
    logger.info("\n" + summary_box)

    # Print additional stats outside the box
    logger.info(f"\n⏱️  Total analysis time: {total_elapsed:.2f}s")

    if all_passed:
        logger.info(f"✅ {C_GREEN}ALL CHECKS PASSED{C_RESET}")
    else:
        logger.info(f"❌ {C_RED}SOME CHECKS FAILED{C_RESET}")

    if test_file:
        if all_passed:
            logger.info(f"✅ {test_file}: No issues found")
        else:
            logger.info(f"❌ {test_file}: Issues detected")

    logger.info("\n" + "=" * 50)
    logger.info("📊 Analysis Summary")
    logger.info("Total files analyzed: 1")
    logger.info(f"Files with issues: {0 if all_passed else 1}")
    logger.info(f"Files without issues: {1 if all_passed else 0}")

    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
