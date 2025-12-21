#!/bin/bash

# =============================================================================
# Quality Enforcer - Comprehensive Quality Analysis
# =============================================================================
#
# PURPOSE:
#   Analyzes Python files for quality violations including structural rules,
#   linting, type checking, and test execution. Processes each file individually
#   providing comprehensive quality feedback.
#
# USAGE:
#   ./enforce_quality.sh [OPTIONS] [FILE_OR_DIRECTORY]
#
# ARGUMENTS:
#   FILE_OR_DIRECTORY    Path to specific file or directory
#                        (Optional: defaults to analyzing all files in src/ and tests/)
#
# OPTIONS:
#   -h, --help          Show help message from header documentation and exit
#   --max-iterations N  Maximum number of fix iterations (default: 3)
#   --no-fix            Disable automatic fixing (not recommended)
#
# EXAMPLES:
#   ./enforce_quality.sh
#       # Analyze all files in src/ and tests/ directories
#
#   ./enforce_quality.sh tests/unit/test_example.py
#       # Analyze specific file
#
#   ./enforce_quality.sh src/
#       # Analyze all files in specific directory
#
# DEPENDENCIES:
#   - python3 (required)
#   - quality_enforcer.py (in impl/ directory)
#   - claude (required when using --fix option)
#   - Z_AI_API_KEY environment variable (required when using --fix option)
#   - Standard Unix tools: find, sed, echo, grep
#
# OUTPUT:
#   - Color-coded analysis results for each file
#   - Structural quality violations with line numbers and suggestions
#   - Linting errors and warnings (via ruff)
#   - Type checking errors (via mypy)
#   - Test execution results
#   - Summary statistics (total files, files with/without issues)
#   - Exit code 0: All checks pass (quality + lint + types + tests)
#   - Exit code 1: One or more checks fail
#
# QUALITY RULES ENFORCED:
#   - No magic values (unexplained numbers)
#   - No print statements
#   - No pytest.main() calls (tests should not invoke pytest programmatically)
#   - No if statements (use assertions instead)
#   - No for/while loops (use accumulation patterns)
#   - No try/except blocks (use assertRaises for exception testing)
#   - No raise statements (tests shouldn't throw exceptions)
#   - No vague assertions
#   - No return statements in test functions (tests should end with assertions)
#   - All test functions must have at least one assertion
#   - ALL test functions MUST have @pytest.mark.unit or @pytest.mark.integration
#   - Tests with @pytest.mark.agent_test MUST also have @pytest.mark.integration
#
# TESTING COMMANDS REQUIRED:
#   - Unit tests: ALWAYS use 'poe test-unit' or 'poe test-unit-cov'
#   - Integration tests: ALWAYS use 'poe test-integration' or 'poe test-integration-cov'
#   - agent_test marked tests: Use langwatch MCP server and 'poe test-integration'
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUALITY_SCRIPT="$SCRIPT_DIR/impl/quality_enforcer.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_box() {
    local title="$1"
    local content="$2"
    python3 -c "
import sys
import re

def visible_len(s):
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', s))

def print_box(title, content):
    width = 140
    inner_width = width - 2
    
    # Top border
    print('╔' + '═' * inner_width + '╗')
    
    # Title centering
    vis_len = visible_len(title)
    padding = (inner_width - vis_len) // 2
    remainder = inner_width - vis_len - padding
    print('║' + ' ' * padding + title + ' ' * remainder + '║')
    
    # Header separator
    print('╠' + '═' * inner_width + '╣')
    
    if content:
        for line in content.replace('\\n', '\n').split('\n'):
            if line == '---SEPARATOR---':
                print('╟' + '─' * inner_width + '╢')
                continue
                
            vis_len = visible_len(line)
            padding = inner_width - 2 - vis_len
            if padding >= 0:
                print(f'║ {line}' + ' ' * padding + ' ║')
            else:
                print(f'║ {line} ║')
    else:
        print(f'║ ' + ' ' * (inner_width-2) + ' ║')
            
    print('╚' + '═' * inner_width + '╝')

print_box(sys.argv[1], sys.argv[2])
" "$title" "$content"
}

print_header() {
    : # No-op, handled in main
}

print_file_header() {
    print_box "ANALYZING FILE" "$1"
}

format_time() {
    local elapsed=$1
    printf "%.2fs" "$elapsed"
}


print_result() {
    local file="$1"
    local result="$2"

    if [[ $result == *"TESTS MEET ALL QUALITY STANDARDS"* ]]; then
        echo -e "${GREEN}✅ $file: No issues found${NC}"
    else
        echo -e "${RED}❌ $file: Issues detected${NC}"
        echo "$result"
    fi
}

# Find Python files to analyze
find_python_files() {
    if [[ $# -gt 0 ]]; then
        # Process provided arguments
        for arg in "$@"; do
            if [[ -f "$arg" ]]; then
                echo "$arg"
            elif [[ -d "$arg" ]]; then
                find "$arg" -name "*.py" 2>/dev/null | sort
            fi
        done
    else
        # Default: search src/ and tests/
        if [[ -d "src" ]]; then
            find "src" -name "*.py" 2>/dev/null | sort
        fi
        if [[ -d "tests" ]]; then
            find "tests" -name "*.py" 2>/dev/null | sort
        fi
    fi
}

main() {
    # Parse arguments
    local fix_mode=true  # ALWAYS FIX by default
    local max_iterations=3
    local test_files=()

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-fix)
                fix_mode=false
                echo -e "${YELLOW}⚠️  WARNING: Fixing disabled. This is not recommended.${NC}"
                shift
                ;;
            --max-iterations)
                if [[ -n "${2:-}" && "$2" =~ ^[0-9]+$ ]]; then
                    max_iterations="$2"
                    shift 2
                else
                    echo -e "${RED}❌ --max-iterations requires a positive integer${NC}"
                    exit 1
                fi
                ;;
            -h|--help)
                echo "See header documentation for usage and options information."
                head -50 "$0" | grep -E "^# [A-Z]+:" -A 10
                exit 0
                ;;
            *)
                # Pass remaining arguments to find_test_files
                break
                ;;
        esac
    done

    # Find files based on remaining arguments
    local remaining_args=("$@")
    test_files=($(find_python_files "${remaining_args[@]}"))
    local total_files=${#test_files[@]}
    local files_with_issues=0

    if [[ $total_files -eq 0 ]]; then
        echo -e "${YELLOW}⚠️  No test files found${NC}"
        exit 0
    fi

    # Define colors for usage inside the string
    local C_BLUE=$'\033[1;34m'
    local C_GREEN=$'\033[1;32m'
    local C_YELLOW=$'\033[1;33m'
    local C_RESET=$'\033[0m'
    local C_BOLD=$'\033[1m'

    # Special case: single file - run with fix mode by default
    local header_content="Found ${C_BOLD}$total_files${C_RESET} test file(s) to analyze"
    if [[ $fix_mode == true ]]; then
        header_content="${header_content}"$'\n'"Automatic Fix Mode: ${C_GREEN}ENABLED${C_RESET} (max iterations: $max_iterations)"
    else
        header_content="${header_content}"$'\n'"Automatic Fix Mode: ${C_YELLOW}DISABLED${C_RESET} (Not Recommended)"
    fi

    local skip_file_header=false
    if [[ $total_files -eq 1 ]]; then
        local single_file="${test_files[0]}"
        single_file=$(echo "$single_file" | sed 's|//|/|g')
        header_content="${header_content}"$'\n'"---SEPARATOR---"$'\n'"Analyzing: ${C_BLUE}$single_file${C_RESET}"
        skip_file_header=true
    fi

    print_box "QUALITY ENFORCEMENT - AGENTIC ANALYSIS" "$header_content"
    echo

    for file in "${test_files[@]}"; do
        # Clean up path to remove double slashes
        file=$(echo "$file" | sed 's|//|/|g')

        if [[ ! -f "$file" ]]; then
            echo -e "${RED}❌ File not found: $file${NC}"
            continue
        fi

        if [[ $skip_file_header == false ]]; then
            print_file_header "$file"
        fi

        # Start timing for this file
        local start_time=$(date +%s.%N)

        # Temporarily disable exit on error for checks (we expect some to fail)
        set +e

        local quality_args=("$file" --no-test-execution)
        [[ $fix_mode == true ]] && quality_args+=(--fix --max-iterations "$max_iterations")
        # Change to the project root directory first
        local original_dir=$(pwd)
        cd "$(dirname "$SCRIPT_DIR")"  # Go up from scripts/ to project root
        python3 -m scripts.impl.quality_enforcer "${quality_args[@]}"
        cd "$original_dir" > /dev/null
        local quality_exit_code=$?

        # Re-enable exit on error
        set -e

        # Exit code handling is now done within the python script
        if [[ $quality_exit_code -ne 0 ]]; then
            files_with_issues=$((files_with_issues + 1))
        fi
    done

    if [[ $files_with_issues -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}



# Note: Help is now handled in main() function

# Check if the Python script exists and is executable
if [[ ! -f "$QUALITY_SCRIPT" ]]; then
    echo -e "${RED}❌ Quality analyzer script not found: $QUALITY_SCRIPT${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed${NC}"
    exit 1
fi

# Run main function with all arguments
main "$@"