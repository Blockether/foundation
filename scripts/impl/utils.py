#!/usr/bin/env python3
"""
Utility functions for scripts and agents.

This module contains common utility functions used across various scripts
and agents in the foundation project, including Claude AI integration,
subprocess handling, and text formatting utilities.
"""

import logging
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from prettytable import PrettyTable

# Configure logging for proper output capture
logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ANSI Color Codes
C_GREEN = "\033[1;32m"
C_RED = "\033[1;31m"
C_YELLOW = "\033[1;33m"
C_RESET = "\033[0m"


# Named constants for common magic values
COMMON_CONSTANTS = {0, 1, -1, 2, 10, 100, 1000}
MAX_ISSUES_FOR_CLAUDE_FIX = 10


class StreamSubprocess:
    """Helper class to run subprocess commands."""

    @staticmethod
    def run(
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        stream: bool = True,
        timeout: int = 300,  # 5 minutes default timeout
    ) -> tuple[bool, str]:
        """Run a command with optional streaming output and timeout."""
        try:
            process = subprocess.Popen(
                cmd,
                cwd=cwd or Path.cwd(),
                env=env or os.environ,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            output_lines: list[str] = []

            # Stream output in real-time if requested
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    if line.strip():
                        if stream:
                            logger.info(f"[{cmd[0]}] {line.rstrip()}")
                        output_lines.append(line.rstrip())

            # Wait for process to complete with timeout
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                logger.error(f"Command timed out after {timeout} seconds: {' '.join(cmd)}")
                return False, f"Command timed out after {timeout} seconds"

            success = return_code == 0
            return success, "\n".join(output_lines)

        except Exception as e:
            logger.error(f"Error running command {' '.join(cmd)}: {e}")
            return False, f"Error: {e}"

    @staticmethod
    def run_silent(
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 300,
    ) -> tuple[bool, str]:
        """Run a command without streaming output."""
        return StreamSubprocess.run(cmd, cwd, env, stream=False, timeout=timeout)


def visible_len(s: str) -> int:
    """Get the visible length of a string, ignoring ANSI escape codes."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return len(ansi_escape.sub("", s))


def wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to fit within specified width with proper padding.
    Word wrapping for text content.

    Args:
        text: Text to wrap
        width: Maximum width for wrapped lines (including padding)

    Returns:
        List of wrapped lines
    """
    if not text:
        return [""]

    # Account for box borders and padding (1 left space + 2 right spaces)
    max_content_width = width - 5  # 2 for borders, 1 left space, 2 right spaces

    # Split text into lines and wrap each line
    input_lines = text.split("\n")
    output_lines: list[str] = []

    for line in input_lines:
        if visible_len(line) <= max_content_width:
            output_lines.append(line)
        else:
            # Wrap long line using word wrapping
            words = line.split()
            current_line: list[str] = []
            for word in words:
                test_line = " ".join(current_line + [word]) if current_line else word
                if visible_len(test_line) <= max_content_width:
                    current_line.append(word)
                else:
                    if current_line:
                        output_lines.append(" ".join(current_line))
                        current_line = [word]
                    else:
                        # Word too long, break it
                        while visible_len(word) > max_content_width:
                            # This simple breaking doesn't handle ANSI codes well inside words
                            # But for our use case (summary), words are short.
                            # If we really need to break a word with ANSI, it's complex.
                            # Let's assume we don't need to break words with ANSI for now.
                            output_lines.append(word[:max_content_width])
                            word = word[max_content_width:]
                        current_line = [word]
            if current_line:
                output_lines.append(" ".join(current_line))

    return output_lines if output_lines else [""]


def create_section_box(title: str, content: str = "", duration: float = 0.0, box_width: int = 140) -> str:
    """Create a boxed section with title, content, and centered timing.
    All boxes have the same fixed width. Tabulated content is displayed as-is
    to preserve table formatting.

    Args:
        title: Section title
        content: Section content (can be multiline)
        duration: Time taken for this section in seconds
        box_width: Width of the box (default 140)

    Returns:
        Formatted boxed section string
    """
    lines: list[str] = []

    # Top border with centered title and timing
    timing_text = f"⏱ {duration:.2f}s" if duration > 0 else ""
    combined_text = f"{title} | {timing_text}" if timing_text else title

    # Calculate padding to center the combined text with 1 left + 2 right spaces
    max_title_length = box_width - 5  # 2 borders + 1 left space + 2 right spaces

    # Use visible length for title centering
    title_len = visible_len(combined_text)

    if title_len <= max_title_length:
        # Calculate total padding needed
        total_padding = max_title_length - title_len
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        # Build title line with exact 1 space after left border and 2 spaces before right border
        title_line = f"║{' ' * (left_padding + 1)}{combined_text}{' ' * (right_padding + 2)}║"
    else:
        # Title too long, truncate it and add 2 spaces before right border
        # Note: Truncating with ANSI codes is complex, assuming title has no colors for now
        max_title_content = max_title_length - 3  # 3 for "..."
        title_line = f"║ {combined_text[:max_title_content]}...  ║"

    # Top and bottom borders
    border = "╔" + "═" * (box_width - 2) + "╗"
    lines.append(border)
    lines.append(title_line)

    # Content separator
    lines.append("╠" + "═" * (box_width - 2) + "╣")

    # Content (if any) - wrap text properly
    if content.strip():
        wrapped_lines = wrap_text(content, box_width)
        for content_line in wrapped_lines:
            # Pad content line to box width with proper padding:
            # - 1 space after left border
            # - content text (wrapped to appropriate length)
            # - 2 spaces before right border
            max_content_length = box_width - 5  # 2 borders + 1 left space + 2 right spaces

            # Check visible length for truncation
            if visible_len(content_line) > max_content_length:
                # Simple truncation (might break ANSI codes, but wrap_text should handle most cases)
                content_line = content_line[:max_content_length]

            # Calculate padding based on visible length
            vis_len = visible_len(content_line)
            padding_needed = (box_width - 2) - vis_len - 1  # -1 for left space

            # Ensure non-negative padding
            if padding_needed < 0:
                padding_needed = 0

            # Add extra padding to compensate for invisible ANSI codes in ljust
            # Or just construct the string manually without ljust
            padded_content = f" {content_line}{' ' * padding_needed}"
            lines.append(f"║{padded_content}║")
    else:
        lines.append("║" + " " * (box_width - 2) + "║")

    # Bottom border
    bottom_border = "╚" + "═" * (box_width - 2) + "╝"
    lines.append(bottom_border)

    return "\n".join(lines)


def tabulate_with_wrapping(
    table_data: list[list[Any]], headers: list[str] | None = None, tablefmt: str = "grid"
) -> str:
    """Create a table with consistent formatting using PrettyTable. Text wrapping is handled by the box formatting."""
    # Convert None headers to empty list to avoid type issues
    safe_headers: list[str] = headers if headers is not None else []

    # Create PrettyTable instance
    table = PrettyTable()

    # Set headers if provided
    if safe_headers:
        table.field_names = safe_headers
    else:
        # If no headers, create default numbered headers based on data
        if table_data:
            num_cols = len(table_data[0]) if table_data else 0
            table.field_names = [f"Column {i + 1}" for i in range(num_cols)]

    # Set alignment to left for all columns
    table.align = "l"

    # Add data rows
    for row in table_data:
        # Ensure row has correct number of columns
        while len(row) < len(table.field_names):
            row.append("")
        table.add_row(row)

    return str(table)


def get_claude_env() -> dict[str, str]:
    """Get the environment variables for Claude API calls."""
    z_ai_api_key = os.environ.get("Z_AI_API_KEY")
    if not z_ai_api_key:
        raise ValueError("Z_AI_API_KEY not set for Claude analysis")

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
    return env


def run_claude_with_prompt(prompt: str, timeout: int = 300) -> tuple[bool, str]:
    """Run Claude with a given prompt.

    Args:
        prompt: The prompt to send to Claude
        timeout: Timeout in seconds (default 300)

    Returns:
        Tuple of (success, output)
    """
    try:
        env = get_claude_env()
        claude_cmd = ["claude", "--print", "--dangerously-skip-permissions", "--", prompt]
        return StreamSubprocess.run_silent(claude_cmd, cwd=Path.cwd(), env=env, timeout=timeout)
    except Exception as e:
        return False, f"Error running Claude: {e}"


def analyze_development_indicators_with_claude(
    indicators: list[dict[str, Any]], file_path: str | None = None
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Use Claude to analyze development indicators and determine if they should be addressed."""
    if not indicators:
        return True, "No development indicators found.", []

    indicators_by_file: dict[str, list[dict[str, Any]]] = {}
    for indicator in indicators:
        file_name = indicator["file"]
        if file_name not in indicators_by_file:
            indicators_by_file[file_name] = []
        indicators_by_file[file_name].append(indicator)

    prompt_lines = [
        f"Analyze these development stage indicators in the src/ directory{' for file: ' + file_path if file_path else ''}:",
        "",
        "Focus on identifying genuine development needs vs legitimate design choices.",
        "",
    ]

    for file_name, file_indicators in indicators_by_file.items():
        prompt_lines.append(f"File: {file_name}")
        for indicator in file_indicators:
            prompt_lines.append(
                f"  Line {indicator['line']}: '{indicator['content']}' (contains: {indicator['keyword']})"
            )
        prompt_lines.append("")

    prompt_lines.extend(
        [
            "For each instance, determine:",
            "1. Is this an UNDEVELOPED FEATURE that needs implementation?",
            "2. Or is this a legitimate, intentional simplification?",
            "3. What priority should this have for development?",
            "",
            "Return your analysis in this format:",
            "CONFIDENCE: [HIGH/MEDIUM/LOW]",
            "PRIORITY: [HIGH/MEDIUM/LOW]",
            "ACTION: [IMPLEMENT_NOW/MANUAL_REVIEW/ACCEPTABLE]",
            "ANALYSIS: [Your detailed analysis]",
            "",
            "If ACTION is ACCEPTABLE, also output exactly this line:",
            "ACCEPTABLE_ITEM: <file>:<line>",
            "",
            "Evaluation criteria:",
            "- 'minimal' implementation = NEEDS DEVELOPMENT",  # ignore-development
            "- 'temporarily' = TEMPORARY IMPLEMENTATION",  # ignore-development
            "- 'incomplete' = INCOMPLETE IMPLEMENTATION",  # ignore-development
            "",
            "Acceptable contexts:",
            "- Development indicators in test files",
            "- Simple utility functions",
            "- Documentation and comments",
            "- Code analysis tools (like this enforcer)",
            "",
            "Only recommend IMPLEMENT_NOW for critical missing functionality.",
            "If usage is ACCEPTABLE (e.g. documentation, test code), state that no action is needed.",
        ]
    )

    prompt = "\n".join(prompt_lines)

    success, output = run_claude_with_prompt(prompt)

    acceptable_items: list[dict[str, Any]] = []
    if success:
        for line in output.splitlines():
            if line.strip().startswith("ACCEPTABLE_ITEM:"):
                try:
                    parts = line.strip().split(":", 2)
                    if len(parts) >= 3:
                        file_part = parts[1].strip()
                        line_part = parts[2].strip()
                        acceptable_items.append({"file": file_part, "line": int(line_part)})
                except Exception:
                    pass

    return success, output, acceptable_items


def run_claude_fix(file_path: str, issues: list[dict[str, Any]]) -> tuple[bool, str]:
    """Run Claude to fix the identified issues."""
    if not issues:
        return True, "No issues to fix."

    prompt = f"Fix the following quality issues in {file_path}:\n\n"
    for issue in issues:
        prompt += f"- Line {issue.get('line', '?')}: {issue.get('message', '')}\n"

    prompt += "\nReturn the fixed file content. Do not include any explanation, just the code."

    return run_claude_with_prompt(prompt)


def run_claude_unused_function_removal(
    file_path: str, unused_functions: list[dict[str, Any]]
) -> tuple[bool, str]:
    """Run Claude to remove unused functions."""
    return run_claude_fix(
        file_path,
        [
            {"message": f"Remove unused function {f['function']}", "line": f["line"]}
            for f in unused_functions
        ],
    )