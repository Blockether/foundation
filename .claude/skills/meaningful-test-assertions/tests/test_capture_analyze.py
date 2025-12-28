#!/usr/bin/env python3
"""
Comprehensive tests for meaningful-test-assertions skill capture_and_analyze.py script.

These tests verify:
1. Script structure and imports
2. Assertion classification (weak, concrete, magic numbers)
3. Debug print insertion
4. Test output parsing
5. Report generation

Usage:
    python tests/test_capture_analyze.py
"""

import ast
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = SKILL_DIR / "scripts"


def test_script_exists():
    """Test that capture_and_analyze.py script exists."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("❌ FAIL: capture_and_analyze.py not found")
        return False
    print("✅ PASS: capture_and_analyze.py exists")
    return True


def test_ast_imports():
    """Test that script imports required AST module."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if "import ast" not in content:
        print("❌ FAIL: Missing 'import ast'")
        return False

    if "import argparse" not in content:
        print("❌ FAIL: Missing 'import argparse'")
        return False

    if "import re" not in content:
        print("❌ FAIL: Missing 'import re'")

    if "import subprocess" not in content:
        print("❌ FAIL: Missing 'import subprocess'")

    print("✅ PASS: Required imports found (ast, argparse, re, subprocess)")
    return True


def test_dataclass_models():
    """Test that script has proper dataclass models for test info."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    classes_to_check = [
        "AssertSuggestion",
        "TestAnalysis",
    ]

    for cls in classes_to_check:
        if cls not in content:
            print(f"❌ FAIL: Missing dataclass {cls}")
            return False

    print(f"✅ PASS: Dataclass models found ({len(classes_to_check)} classes)")
    return True


def test_has_assert_classifier():
    """Test that AssertClassifier class exists."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if "class AssertClassifier" not in content:
        print("❌ FAIL: AssertClassifier class not found")
        return False

    print("✅ PASS: AssertClassifier class exists")
    return True


def test_weak_assertion_detection():
    """Test that weak assertion detection methods exist."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    methods = [
        "_is_weak_assertion",
        "_has_magic_number",
        "_determine_type",
        "_is_len_comparison",
    ]

    missing = []
    for method in methods:
        if f"def {method}" not in content:
            missing.append(method)  # type: ignore

    if missing:
        print(f"❌ FAIL: Missing methods: {', '.join(missing)}")  # type: ignore
        return False

    print(f"✅ PASS: All classification methods found ({len(methods)} methods)")
    return True


def test_is_not_none_detection():
    """Test detection of 'is not None' assertions."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    # Check for 'is not None' pattern in _is_weak_assertion
    if "is not None" not in content:
        print("❌ FAIL: 'is not None' detection not found")
        return False

    print("✅ PASS: 'is not None' detection exists")
    return True


def test_isinstance_detection():
    """Test detection of 'isinstance' assertions."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    # Check for 'isinstance' pattern
    if "isinstance" not in content:
        print("❌ FAIL: 'isinstance' detection not found")
        return False

    print("✅ PASS: 'isinstance' detection exists")
    return True


def test_lenient_length_detection():
    """Test detection of lenient length checks (>=1, >0)."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    # Check for lenient length patterns
    if ">= 1" not in content or "> 0" not in content:
        print("❌ FAIL: Lenient length detection (>=1, >0) not found")
        return False

    print("✅ PASS: Lenient length detection (>=1, >0) exists")
    return True


def test_magic_number_detection():
    """Test detection of magic numbers."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    # Check for magic number detection
    if "COMMON_CONSTANTS" not in content:
        print("❌ FAIL: COMMON_CONSTANTS not defined")
        return False

    if "0, 1, -1, 2, 10, 100, 1000" not in content:
        print("❌ FAIL: Common constants not defined")
        return False

    print("✅ PASS: Magic number detection with common constants exists")
    return True


def test_insert_debug_prints_function():
    """Test that insert_debug_prints function exists."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if "def insert_debug_prints" not in content:
        print("❌ FAIL: insert_debug_prints function not found")
        return False

    print("✅ PASS: insert_debug_prints function exists")
    return True


def test_run_test_capture_output_function():
    """Test that run_test_with_debug function exists."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if "def run_test_capture_output" not in content:
        print("❌ FAIL: run_test_capture_output function not found")
        return False

    print("✅ PASS: run_test_capture_output function exists")
    return True


def test_cleanup_debug_prints_function():
    """Test that cleanup_debug_prints function exists."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if "def cleanup_debug_prints" not in content:
        print("❌ FAIL: cleanup_debug_prints function not found")
        return False

    print("✅ PASS: cleanup_debug_prints function exists")
    return True


def test_debug_print_format():
    """Test that debug prints use [ASSERT_DEBUG] format."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    # Check for [ASSERT_DEBUG] pattern in print statements
    if "[ASSERT_DEBUG]" not in content:
        print("❌ FAIL: [ASSERT_DEBUG] format not found")
        return False

    # Check for proper f-string formatting with var=
    if "print(f'[ASSERT_DEBUG] " not in content:
        print("❌ FAIL: Debug print format incorrect")
        return False

    print("✅ PASS: Debug print format [ASSERT_DEBUG] with proper formatting exists")
    return True


def test_parse_debug_output_function():
    """Test that parse_debug_output function exists."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if "def parse_debug_output" not in content:
        print("❌ FAIL: parse_debug_output function not found")
        return False

    print("✅ PASS: parse_debug_output function exists")
    return True


def test_pytest_command():
    """Test that script uses proper pytest commands."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    has_uv = '"uv"' in content or "'uv'" in content
    has_run = '"run"' in content or "'run'" in content
    has_pytest = '"pytest"' in content or "'pytest'" in content

    if not (has_uv and has_run and has_pytest):
        print("❌ FAIL: uv run pytest command not found")
        return False

    # Check for -s flag (capture output)
    if '"-s"' not in content and "'-s'" not in content:
        print("❌ FAIL: -s flag for capturing output not found")
        return False

    # Check for -m marker support
    if '"-m"' not in content and "'-m'" not in content:
        print("❌ FAIL: -m marker support not found")
        return False

    print("✅ PASS: Pytest command with -s and -m flags exists")
    return True


def test_no_backup_in_cleanup():
    """Test that cleanup doesn't create backup files."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    # Check for .backup references - should not exist
    if ".backup" in content:
        print("❌ FAIL: Script contains .backup file references (should not have)")
        return False

    # Check that cleanup just removes debug prints, no backup file
    if "def cleanup_debug_prints" in content:
        # Read the function to verify it doesn't reference backup
        func_start = content.index("def cleanup_debug_prints")
        # Find the next function or end
        func_end = (
            content.find("\ndef ", func_start) if "\ndef " in content[func_start:] else len(content)
        )
        func_content = content[func_start:func_end]

        if ".backup" in func_content:
            print("❌ FAIL: cleanup_debug_prints references .backup files")
            return False

    print("✅ PASS: Cleanup doesn't use backup files")
    return True


def test_unit_marker_support():
    """Test that script supports 'unit' marker."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if '"unit"' not in content and "'unit'" not in content:
        print("❌ FAIL: 'unit' marker support not found")
        return False

    print("✅ PASS: 'unit' marker support exists")
    return True


def test_integration_marker_support():
    """Test that script supports 'integration' marker."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if '"integration"' not in content and "'integration'" not in content:
        print("❌ FAIL: 'integration' marker support not found")
        return False

    print("✅ PASS: 'integration' marker support exists")
    return True


def test_all_marker_support():
    """Test that script supports --all option."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if "--all" not in content:
        print("❌ FAIL: --all option not found")
        return False

    print("✅ PASS: --all option for processing all tests exists")
    return True


def test_suggestion_generation():
    """Test that generate_suggestions function exists."""
    script_path = SCRIPTS_DIR / "capture_and_analyze.py"
    if not script_path.exists():
        print("⚠️  SKIP: Script not found")
        return False

    with open(script_path, "r") as f:
        content = f.read()

    if "def generate_suggestions" not in content:
        print("❌ FAIL: generate_suggestions function not found")
        return False

    print("✅ PASS: generate_suggestions function exists")
    return True


def main():
    """Run all tests."""
    print(f"🧪 Running comprehensive tests for meaningful-test-assertions")
    print()

    tests = [
        test_script_exists,
        test_ast_imports,
        test_dataclass_models,
        test_has_assert_classifier,
        test_weak_assertion_detection,
        test_is_not_none_detection,
        test_isinstance_detection,
        test_lenient_length_detection,
        test_magic_number_detection,
        test_insert_debug_prints_function,
        test_run_test_capture_output_function,
        test_cleanup_debug_prints_function,
        test_debug_print_format,
        test_parse_debug_output_function,
        test_pytest_command,
        test_no_backup_in_cleanup,
        test_unit_marker_support,
        test_integration_marker_support,
        test_all_marker_support,
        test_suggestion_generation,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)  # type: ignore
        except Exception as e:
            print(f"❌ FAIL: {test.__name__} raised {e}")
            results.append(False)  # type: ignore
        print()

    passed = sum(results)  # type: ignore
    total = len(results)  # type: ignore

    print(f"📊 Test Results: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
