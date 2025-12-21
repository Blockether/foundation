#!/usr/bin/env python3
"""
Script to find all test cases in the codebase using Python AST visitor pattern
and build a structured representation with file paths, test names, and line ranges.
This is the first step in enforcing meaningful asserts in test cases.

Usage:
    python scripts/enforce_meaningful_asserts.py [--output-file OUTPUT_FILE]

Arguments:
    --output-file: Optional path to save the JSON output of test cases
                   (defaults to printing to stdout)

Examples:
    python scripts/enforce_meaningful_asserts.py
    python scripts/enforce_meaningful_asserts.py --output-file test_cases.json

Dependencies:
    - Python 3.8+
    - Pydantic
    - Standard library modules: ast, json, pathlib, sys

Output:
    JSON representation of all test cases found in the codebase with:
    - full_path_to_file: Absolute path to the test file
    - name_of_the_test_case: Name of the test function/method
    - line_numbers_range: Tuple of (start_line, end_line) for the test function
"""

import ast
import json
import pathlib
import sys
from argparse import ArgumentParser
from typing import Any, override

from pydantic import BaseModel, Field


class TestCaseInfo(BaseModel):
    """Pydantic model representing a single test case."""

    full_path_to_file: str = Field(
        description="Absolute path to the test file containing this test case"
    )
    name_of_the_test_case: str = Field(description="Name of the test function or method")
    line_numbers_range: tuple[int, int] = Field(
        description="Tuple of (start_line, end_line) for the test function"
    )


class TestFileCases(BaseModel):
    """Pydantic model representing all test cases in a single file."""

    file_path: str = Field(description="Absolute path to the test file")
    test_cases: list[TestCaseInfo] = Field(description="List of test cases in this file")


class AllTestCases(BaseModel):
    """Pydantic model representing all test cases in the codebase."""

    total_test_cases: int = Field(description="Total number of test cases found")
    test_files: list[TestFileCases] = Field(description="List of test files with their test cases")

    class Config:
        json_encoders: dict[type[tuple[Any, ...]], Any] = {
            # Ensure tuples are properly serialized as lists in JSON
            tuple: lambda v: list(v)  # type: ignore[arg-type]
        }


class TestCaseVisitor(ast.NodeVisitor):
    """AST visitor to extract test case information from Python files."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.test_cases: list[TestCaseInfo] = []
        self.current_class: str | None = None

    @override
    def visit_Module(self, node: ast.Module) -> None:
        """Visit the module and start processing."""
        self.generic_visit(node)

    @override
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to track class context for test methods."""
        # Store current class context
        old_class = self.current_class
        self.current_class = node.name

        # Visit child nodes
        self.generic_visit(node)

        # Restore previous class context
        self.current_class = old_class

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions and identify test cases."""
        if self._is_test_function(node):
            test_name = self._get_full_test_name(node)
            line_range = self._get_function_line_range(node)

            test_case = TestCaseInfo(
                full_path_to_file=self.file_path,
                name_of_the_test_case=test_name,
                line_numbers_range=line_range,
            )
            self.test_cases.append(test_case)

        self.generic_visit(node)

    def _is_test_function(self, node: ast.FunctionDef) -> bool:
        """Determine if a function is a test case."""
        # Direct test function
        if node.name.startswith("test_"):
            return True

        # Test method in a test class
        if self.current_class and self.current_class.startswith("Test"):
            if node.name.startswith("test_"):
                return True

        # Pytest parametrized tests (usually start with test_)
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "pytest":
                    return True
                if isinstance(decorator, ast.Attribute):
                    if isinstance(decorator.attr, str) and decorator.attr.endswith("parametrize"):
                        return True

        return False

    def _get_full_test_name(self, node: ast.FunctionDef) -> str:
        """Get the full test name including class context if applicable."""
        if self.current_class:
            return f"{self.current_class}.{node.name}"
        return node.name

    def _get_function_line_range(self, node: ast.FunctionDef) -> tuple[int, int]:
        """Get the line range for a function."""
        start_line = node.lineno

        # Find the end line by looking at the last node in the function body
        end_line = start_line
        if node.body:
            # Find the last line in the function body
            last_node = node.body[-1]
            # Try to get end_lineno if available (Python 3.8+)
            if isinstance(last_node, ast.stmt):
                # Check if the statement node has end_lineno attribute
                if hasattr(last_node, "end_lineno") and last_node.end_lineno is not None:
                    end_line = last_node.end_lineno
                else:
                    # Fallback: walk through the last node to find its max line
                    end_line = self._find_max_line(last_node)
            else:
                end_line = self._find_max_line(last_node)

        return (start_line, end_line)

    def _find_max_line(self, node: ast.AST) -> int:
        """Recursively find the maximum line number in an AST node."""
        max_line = 0

        for child in ast.walk(node):
            # Get lineno if available
            if isinstance(child, ast.stmt):
                max_line = max(max_line, child.lineno)
                # Check if the statement node has end_lineno attribute
                if hasattr(child, "end_lineno") and child.end_lineno is not None:
                    max_line = max(max_line, child.end_lineno)

        return max_line


def find_test_files(root_dir: pathlib.Path) -> list[pathlib.Path]:
    """Find all Python test files in the codebase."""
    test_files: list[pathlib.Path] = []

    # Common test file patterns
    test_patterns = ["**/test_*.py", "**/*_test.py", "**/tests/**/*.py", "**/test/**/*.py"]

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


def extract_test_cases_from_file(file_path: pathlib.Path) -> list[TestCaseInfo]:
    """Extract test cases from a single Python file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        visitor = TestCaseVisitor(str(file_path.absolute()))
        visitor.visit(tree)

        return visitor.test_cases

    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return []


def main() -> None:
    """Main function to find and output all test cases."""
    parser = ArgumentParser(description="Find all test cases in the codebase")
    parser.add_argument("--output-file", help="Optional path to save the JSON output of test cases")
    parser.add_argument(
        "--test-dir", default="tests", help="Directory to search for test files (default: tests)"
    )

    args = parser.parse_args()

    # Find the root directory (should be the project root)
    root_dir = pathlib.Path(__file__).parent.parent
    test_dir = root_dir / args.test_dir

    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for test files in {test_dir}...", file=sys.stderr)

    # Find all test files
    test_files = find_test_files(test_dir)

    if not test_files:
        print("No test files found", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(test_files)} test files", file=sys.stderr)

    # Extract test cases from each file
    all_test_files: list[TestFileCases] = []
    total_test_cases = 0

    for test_file in test_files:
        print(f"Processing {test_file}...", file=sys.stderr)
        test_cases = extract_test_cases_from_file(test_file)

        if test_cases:
            test_file_cases = TestFileCases(
                file_path=str(test_file.absolute()), test_cases=test_cases
            )
            all_test_files.append(test_file_cases)
            total_test_cases += len(test_cases)

    # Create the final model
    all_test_cases = AllTestCases(total_test_cases=total_test_cases, test_files=all_test_files)

    # Output results
    output_data = all_test_cases.dict()

    if args.output_file:
        output_path = pathlib.Path(args.output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}", file=sys.stderr)
    else:
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

    print(
        f"\nSummary: Found {total_test_cases} test cases across {len(all_test_files)} files",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
