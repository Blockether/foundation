#!/usr/bin/env python3
"""
Analyze test assertions for meaningfulness.

This script captures actual values from weak assertions and suggests improvements.

Usage:
    python capture_and_analyze.py --file <test_file> [--marker unit|integration]
    python capture_and_analyze.py --all [--marker unit|integration]
"""

import argparse
import ast
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

COMMON_CONSTANTS = {0, 1, -1, 2, 10, 100, 1000}


@dataclass
class AssertSuggestion:
    """Suggestion for improving an assertion."""

    line_number: int
    current_assertion: str
    assertion_type: str  # "WEAK", "MAGIC_NUMBER", "CONCRETE_OK"
    captured_values: dict[str, str] = field(default_factory=dict)  # type: ignore
    issue: str = ""
    suggestion: str = ""
    variables: list[str] = field(default_factory=list)  # type: ignore


@dataclass
class TestAnalysis:
    """Analysis of a single test function."""

    test_name: str
    markers: list[str]
    assertions: list[AssertSuggestion] = field(default_factory=lambda: list[AssertSuggestion]())
    suggested_constants: list[str] = field(default_factory=lambda: list[str]())


class AssertClassifier(ast.NodeVisitor):
    """Classify assertions and detect variables needing debug capture."""

    def __init__(self) -> None:
        self.assertions: list[AssertSuggestion] = []
        self.variables_to_capture: set[str] = set()

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit test function and classify its assertions."""
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                self._classify_assertion(child, node)
        self.generic_visit(node)

    def _classify_assertion(self, assert_node: ast.Assert, function_node: ast.FunctionDef) -> None:
        """Classify an assertion as weak, concrete, or magic number."""
        test = assert_node.test
        assert_text = (
            ast.unparse(test)
            if hasattr(ast, "unparse") and callable(ast.unparse)
            else f"assert line {assert_node.lineno}"
        )

        # Extract variables
        variables: list[str] = []
        for sub_child in ast.walk(test):
            if isinstance(sub_child, ast.Name):
                variables.append(sub_child.id)

        classification = self._determine_type(test)
        variables_to_capture = self._get_variables_to_capture(test, classification)

        suggestion = AssertSuggestion(
            line_number=assert_node.lineno,
            current_assertion=assert_text,
            assertion_type=classification,
            variables=variables,
        )

        # For weak assertions, mark variables for capture
        if classification == "WEAK":
            for var in variables_to_capture:
                if var not in [
                    "isinstance",
                    "str",
                    "int",
                    "bool",
                    "list",
                    "dict",
                    "set",
                    "tuple",
                    "len",
                    "type",
                ]:
                    self.variables_to_capture.add(var)

        self.assertions.append(suggestion)

    def _determine_type(self, test_node: ast.AST) -> str:
        """Determine if assertion is weak, concrete, or magic number."""

        # Check for weak patterns
        if self._is_weak_assertion(test_node):
            return "WEAK"

        # Check for magic numbers in concrete assertions
        if self._has_magic_number(test_node):
            return "MAGIC_NUMBER"

        return "CONCRETE_OK"

    def _is_weak_assertion(self, node: ast.AST) -> bool:
        """Check if assertion is weak (needs actual value capture)."""

        # assert result is not None
        if isinstance(node, ast.Compare):
            if len(node.comparators) == 1:
                comp = node.comparators[0]
                if isinstance(comp, ast.Constant) and comp.value is None:
                    if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                        return True

        # assert isinstance(result, SomeClass)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
                return True

        # assert result (truthiness)
        if isinstance(node, ast.Name):
            return True

        # assert len(result) >= 1 or assert len(result) > 0
        if isinstance(node, ast.Compare):
            if self._is_len_comparison(node, [ast.GtE, ast.Gt], [1, 0]):
                return True

        # assert "substring" in result
        if isinstance(node, ast.Compare):
            if len(node.comparators) == 1:
                comp = node.comparators[0]
                if isinstance(comp, (ast.Name, ast.Subscript)) or isinstance(comp, ast.Constant):
                    if any(isinstance(op, ast.In) for op in node.ops):
                        return True

        return False

    def _is_len_comparison(
        self, node: ast.Compare, op_types: list[type[ast.cmpop]], values: list[int]
    ) -> bool:
        """Check if this is a len() comparison with specific operators/values."""
        if not isinstance(node.left, ast.Call):
            return False
        if not isinstance(node.left.func, ast.Name) or node.left.func.id != "len":
            return False
        if not node.left.args:
            return False
        if not any(isinstance(op, tuple(op_types)) for op in node.ops):
            return False
        if not any(
            isinstance(comp, ast.Constant) and comp.value in values for comp in node.comparators
        ):
            return True
        return False

    def _has_magic_number(self, node: ast.AST) -> bool:
        """Check if assertion has magic numbers that should be constants."""
        common_constants = {0, 1, -1, 2, 10}

        for child in ast.walk(node):
            if isinstance(child, ast.Constant):
                if isinstance(child.value, (int, float)):
                    if child.value not in common_constants:
                        # Check if this is in a comparison context
                        parent_context = self._get_parent_context(node, child)
                        if parent_context in ["Compare", "BinOp", "UnaryOp"]:
                            return True
        return False

    def _get_parent_context(self, root: ast.AST, target: ast.AST) -> str | None:
        """Get the type of parent node containing a child."""
        for parent in ast.walk(root):
            for child in ast.iter_child_nodes(parent):
                if child is target:
                    return type(parent).__name__
        return None

    def _get_variables_to_capture(self, node: ast.AST, classification: str) -> list[str]:
        """Get list of variables that need debug capture."""
        if classification != "WEAK":
            return []

        variables: list[str] = []

        # For simple checks, capture main variable
        if isinstance(node, ast.Name):
            variables.append(node.id)
        elif isinstance(node, ast.Compare):
            # Capture left side variable
            if isinstance(node.left, ast.Name):
                variables.append(node.left.id)
            elif (
                isinstance(node.left, ast.Call)
                and isinstance(node.left.func, ast.Name)
                and node.left.func.id == "len"
            ):
                # Capture argument to len()
                if node.left.args and isinstance(node.left.args[0], ast.Name):
                    variables.append(node.left.args[0].id)
            # Capture right side if it's a variable
            for comp in node.comparators:
                if isinstance(comp, ast.Name):
                    variables.append(comp.id)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
        ):
            # Capture first argument
            if node.args and isinstance(node.args[0], ast.Name):
                variables.append(node.args[0].id)

        return variables


def insert_debug_prints(file_path: str, test_name: str, variables: set[str]) -> None:
    """Insert debug prints before assertions in a test function."""
    with open(file_path) as f:
        lines: list[str] = f.readlines()

    # Create a new list for modified lines
    new_lines: list[str] = []

    # Find test function
    in_test = False
    indent_level = 0

    for _line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Detect test function start
        if stripped.startswith(f"def {test_name}("):
            in_test = True
            indent_level = len(line) - len(line.lstrip())
            new_lines.append(line)
            continue

        # Exit test function when dedent occurs
        if in_test and stripped and not stripped.startswith("#"):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and stripped.startswith("def "):
                in_test = False

        # Insert debug prints before assert statements
        if in_test and stripped.startswith("assert "):
            indent = " " * (len(line) - len(line.lstrip()))
            new_lines.append(f"{indent}# [ASSERT_DEBUG]\n")
            for var in sorted(variables):
                if var not in [
                    "isinstance",
                    "str",
                    "int",
                    "bool",
                    "list",
                    "dict",
                    "set",
                    "tuple",
                    "len",
                ]:
                    new_lines.append(f"{indent}print(f'[ASSERT_DEBUG] {var}={{var}}')\n")

        new_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)


def parse_debug_output(output: str) -> dict[str, str]:
    """Parse debug output to extract variable values."""
    values: dict[str, str] = {}
    for line in output.split("\n"):
        if "[ASSERT_DEBUG]" in line:
            # Match pattern: [ASSERT_DEBUG] var=value
            match = re.search(r"\[ASSERT_DEBUG\] (\w+)=(.+)", line)
            if match:
                var_name = match.group(1)
                var_value = match.group(2).strip()
                values[var_name] = var_value
    return values


def run_test_capture_output(file_path: str, test_name: str, marker: str) -> dict[str, str]:
    """Run a test with debug prints and capture output."""
    cmd = [
        "uv",
        "run",
        "pytest",
        file_path,
        "-k",
        test_name,
        "-s",  # Capture output (don't suppress)
        "--tb=short",  # Short traceback format
        "-v",  # Verbose output
        "-m",
        marker,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        # Combine stdout and stderr for full debug info
        output = (
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nReturn code: {result.returncode}"
        )
        return parse_debug_output(output)
    except subprocess.TimeoutExpired:
        return {"error": "Test execution timed out after 30 seconds"}
    except Exception as e:
        return {"error": f"Error running test: {e}"}


def cleanup_debug_prints(file_path: str, test_name: str) -> None:
    """Remove debug prints from test function."""
    with open(file_path) as f:
        lines: list[str] = f.readlines()

    # Create a new list for modified lines
    new_lines: list[str] = []

    # Find test function
    in_test = False
    indent_level = 0

    for _line_num, line in enumerate(lines, 1):
        stripped = line.strip()

        # Detect test function start
        if stripped.startswith(f"def {test_name}("):
            in_test = True
            indent_level = len(line) - len(line.lstrip())
            new_lines.append(line)
            continue

        # Exit test function when dedent occurs
        if in_test and stripped and not stripped.startswith("#"):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and stripped.startswith("def "):
                in_test = False

        # Skip debug print lines
        if in_test and ("[ASSERT_DEBUG]" in line or "print(f'[ASSERT_DEBUG]" in line):
            continue

        new_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)


def generate_suggestions(
    classifier: AssertClassifier, captured_values: dict[str, str]
) -> list[AssertSuggestion]:
    """Generate suggestions based on classification and captured values."""
    suggestions: list[AssertSuggestion] = []

    for assert_info in classifier.assertions:
        suggestion = AssertSuggestion(
            line_number=assert_info.line_number,
            current_assertion=assert_info.current_assertion,
            assertion_type=assert_info.assertion_type,
            variables=assert_info.variables,
        )

        if assert_info.assertion_type == "WEAK":
            suggestion.issue = "Non-specific assertion"
            suggestion.captured_values = captured_values

            # Generate suggestion based on captured values
            if captured_values:
                var_list = list(captured_values.keys())
                if len(var_list) == 1:
                    var = var_list[0]
                    val = captured_values[var]
                    suggestion.suggestion = f"assert {var} == {val}"
                else:
                    # Multiple variables captured
                    parts = [f"{k} == {v}" for k, v in captured_values.items()]
                    suggestion.suggestion = "assert " + " and ".join(parts)

        elif assert_info.assertion_type == "MAGIC_NUMBER":
            suggestion.issue = "Magic number should be a constant"
            suggestion.suggestion = "Replace with CONSTANT_X (agent should rename)"

        elif assert_info.assertion_type == "CONCRETE_OK":
            # Check if there are magic numbers
            if captured_values and any(v not in ["0", "1", "-1"] for v in captured_values.values()):
                suggestion.assertion_type = "MAGIC_NUMBER"
                suggestion.issue = "Magic number should be a constant"
                suggestion.suggestion = "Replace with CONSTANT_X (agent should rename)"
            else:
                suggestion.issue = ""
                suggestion.suggestion = ""

        suggestions.append(suggestion)

    return suggestions


def find_test_files(root_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find all Python test files in codebase."""
    test_files: list[pathlib.Path] = []

    # Common test file patterns - patterns should be relative to root_dir
    test_patterns = ["test_*.py", "*_test.py", "**/tests/**/*.py", "**/test/**/*.py"]

    for pattern in test_patterns:
        test_files.extend(root_dir.glob(pattern))

    # Remove duplicates while preserving order
    seen: set[pathlib.Path] = set()
    unique_test_files: list[pathlib.Path] = []
    for file_path in test_files:
        if file_path not in seen:
            seen.add(file_path)
            unique_test_files.append(file_path)

    return unique_test_files


def extract_test_cases_from_file(file_path: pathlib.Path) -> list[TestAnalysis]:
    """Extract test cases from a single Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
            content = "".join(lines)

        tree = ast.parse(content, filename=str(file_path))
        visitor = AssertClassifier()
        visitor.visit(tree)

        test_name = Path(file_path).stem
        if visitor.assertions:
            analysis = TestAnalysis(
                test_name=test_name,
                markers=["unit"],  # Default to unit
                assertions=visitor.assertions,
            )
            return [analysis]
        return []
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []


def main() -> None:
    """Main function to find and analyze test cases."""
    parser = argparse.ArgumentParser(description="Analyze test assertions for meaningfulness")
    parser.add_argument("--file", help="Path to test file")
    parser.add_argument("--all", action="store_true", help="Analyze all test files")
    parser.add_argument(
        "--marker", default="unit", choices=["unit", "integration"], help="Test marker"
    )
    parser.add_argument(
        "--output",
        default="assertion_analysis.json",
        help="Output JSON file path (default: assertion_analysis.json)",
    )

    args = parser.parse_args()

    if not args.file and not args.all:
        print("Error: Specify --file or --all", file=sys.stderr)
        sys.exit(1)

    if args.file and args.all:
        print("Error: Cannot use both --file and --all", file=sys.stderr)
        sys.exit(1)

    root_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent

    test_files: list[pathlib.Path] = []

    if args.file:
        file_path = pathlib.Path(args.file)
        if not file_path.is_absolute():
            file_path = root_dir / args.file
        if not file_path.exists():
            print(f"Error: Test file {file_path} does not exist", file=sys.stderr)
            sys.exit(1)
        test_files = [file_path]
    else:
        test_dir = root_dir / "tests"
        if not test_dir.exists():
            print(f"Error: Test directory {test_dir} does not exist", file=sys.stderr)
            sys.exit(1)
        test_patterns = ["**/test_*.py", "**/*_test.py"]
        for pattern in test_patterns:
            test_files.extend(test_dir.glob(pattern))

    if not test_files:
        print("No test files found", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(test_files)} test file(s)", file=sys.stderr)

    # Analyze each test file and collect results
    all_results: list[dict[str, object]] = []

    for test_file in test_files:
        # Try to get relative path from tests directory
        try:
            rel_path = test_file.relative_to(root_dir / "tests")
        except ValueError:
            rel_path = test_file.relative_to(root_dir)

        test_analyses = extract_test_cases_from_file(test_file)
        for analysis in test_analyses:
            # Convert TestAnalysis to dict for JSON output
            result: dict[str, object] = {
                "test_file": str(rel_path),
                "test_name": analysis.test_name,
                "markers": analysis.markers,
                "assertions": [
                    {
                        "line_number": a.line_number,
                        "current_assertion": a.current_assertion,
                        "assertion_type": a.assertion_type,
                        "captured_values": a.captured_values,
                        "issue": a.issue,
                        "suggestion": a.suggestion,
                    }
                    for a in analysis.assertions
                ],
                "suggested_constants": analysis.suggested_constants,
            }
            all_results.append(result)

    import json

    output_path = pathlib.Path(args.output)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_path}", file=sys.stderr)
    print(f"\n{'=' * 60}\nJSON Output:\n{'=' * 60}\n", file=sys.stderr)

    with open(output_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
