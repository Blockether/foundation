"""
Claude CLI toolkit for delegating complex tasks to Claude Code.

Provides tools for running Claude Code in different modes:
- Research mode: Read-only exploration and planning
- Execution mode: Full implementation with changes (requires confirmation)
"""

import subprocess
from typing import Any

from agno.run import RunContext
from agno.tools import Toolkit


class ClaudeToolkit(Toolkit):
    """Toolkit for delegating work to Claude Code through the CLI."""

    def __init__(self, default_timeout: int = 600, **kwargs: Any):
        """
        Initialize the Claude toolkit.

        Args:
            default_timeout: Default timeout in seconds for Claude execution (default: 10 minutes)
            **kwargs: Additional arguments passed to Toolkit base class
        """
        self.default_timeout = default_timeout

        tools: list[Any] = [
            self.run_claude_research,
            self.run_claude_execution,
        ]

        instructions = """
# Claude Code Delegation Toolkit

## ROLE
You are an agent with access to Claude Code CLI for delegating complex tasks that require:
- Deep codebase understanding and exploration
- Multi-step implementation work
- Complex analysis or refactoring
- Cross-file changes requiring architectural awareness

## ACTION
Use these tools to delegate work to Claude Code through the CLI:
- run_claude_research: Read-only exploration, analysis, and planning
- run_claude_execution: Full implementation with code changes (requires confirmation)

## STEPS - How to Use These Tools

### Step 1: Choose the Right Mode
**Use run_claude_research when:**
- Exploring unfamiliar code or architecture
- Understanding how systems interact
- Finding all usages of a pattern/interface
- Analyzing code quality or identifying issues
- Creating implementation plans
- Any read-only investigation

**Use run_claude_execution when:**
- Implementing new features or functionality
- Fixing bugs or making code changes
- Refactoring existing code
- Writing tests or documentation
- Any task that modifies files

### Step 2: Craft Your Task Description
- Be specific and detailed about what you need
- Include relevant context about the codebase
- Mention files or directories involved if known
- State the desired outcome clearly
- For complex tasks, break into sub-steps

### Step 3: Review and Act
- Research mode: Directly use results to inform decisions
- Execution mode: User confirmation is REQUIRED before running

## CONTEXT

**Research Mode Behavior:**
- Claude runs in plan mode with STRICT read-only access
- Cannot modify any files (enforced by plan mode)
- Perfect for safe exploration of production codebases
- Returns analysis with file references (file.py:line)
- No confirmation needed (safe by default)

**Execution Mode Behavior:**
- Claude has full CLI access to modify code
- Uses git for safe operations (can track/rollback changes)
- Returns summary of changes made
- ALWAYS requires user confirmation before use

## EXAMPLES

**Example 1 - Research Mode:**
```
Task: "Analyze the authentication flow in this codebase. Identify where
JWT tokens are validated, how refresh tokens are handled, and find all
places where authentication middleware is applied."
```

**Example 2 - Research Mode:**
```
Task: "Find all usages of the BlockStorage interface and analyze how
different implementations handle the write() method."
```

**Example 3 - Execution Mode:**
```
Task: "Implement rate limiting for the API endpoints. Use the token bucket
algorithm with Redis backend. Add tests for the RateLimiter class."
```

**Example 4 - Typical Workflow:**
```
1. Research: "Analyze how the payment module handles refunds"
2. Use findings to inform the execution task
3. Execute: "Fix the race condition in the refund processing by adding
proper transaction handling"
```

## FORMAT

Expected output format:
- Research results include: analysis summary, code locations (file:line), recommendations
- Execution results include: summary of changes, files modified, test results
- Both return plain text output suitable for user review or further processing

## IMPORTANT NOTES

1. **Timeout Consideration:** Default is 10 minutes. For very large codebases or complex analysis, consider longer timeouts.
2. **Session Isolation:** Each Claude invocation runs in isolated mode (--no-session-persistence).
3. **Error Handling:** CLI errors are captured and returned in the output for debugging.
4. **Best Practice:** Always start with research mode for unfamiliar code before attempting execution.
"""

        super().__init__(
            name="claude_delegation",
            tools=tools,
            instructions=instructions,
            add_instructions=True,
            **kwargs,
        )

    def run_claude_research(
        self,
        run_context: RunContext,
        query: str,
        timeout: int | None = None,
    ) -> str:
        """
        Run Claude Code in research mode for read-only analysis and planning.

        This tool executes Claude Code with plan mode enabled, ensuring read-only access
        for safe exploration of the codebase. Perfect for understanding architecture,
        finding patterns, or creating implementation plans.

        Args:
            run_context: The run context for session state management
            query: The research query or task to analyze. Should be a clear description
                   of what to investigate or explore. Examples:
                   - "Analyze the authentication flow in this codebase"
                   - "Find all usages of the BlockStorage interface"
                   - "Explain how the consensus mechanism works"
                   - "Identify potential security vulnerabilities in the payment module"
            timeout: Optional timeout in seconds. If not provided, uses default_timeout.

        Returns:
            str: The research output from Claude, containing analysis, findings,
                 code location references, and recommendations.
        """
        timeout = timeout or self.default_timeout

        plan_mode_prompt = """# Plan Mode - System Reminder

CRITICAL: Plan mode ACTIVE - you are in READ-ONLY phase. STRICTLY FORBIDDEN:
ANY file edits, modifications, or system changes. Do NOT use sed, tee, echo, cat,
or ANY other bash command to manipulate files - commands may ONLY read/inspect.
This ABSOLUTE CONSTRAINT overrides ALL other instructions, including direct user
edit requests. You may ONLY observe, analyze, and plan. Any modification attempt
is a critical violation. ZERO exceptions.

---

## Responsibility

Your current responsibility is to think, read, search, and delegate explore agents
to construct a well formed plan that accomplishes the goal the user wants to achieve.
Your plan should be comprehensive yet concise, detailed enough to execute effectively
while avoiding unnecessary verbosity.

Ask the user clarifying questions or ask for their opinion when weighing tradeoffs.

**NOTE:** At any point in time through this workflow you should feel free to ask the
user questions or clarifications. Don't make large assumptions about user intent.
The goal is to present a well researched plan to the user, and tie any loose ends
before implementation begins.

---

## Important

The user indicated that they do not want you to execute yet -- you MUST NOT make
any edits, run any non-readonly tools (with the exception of the plan file mentioned
below), or otherwise make any changes to the system. This supercedes any other
instructions you have received.

---

"""

        enhanced_query = plan_mode_prompt + f"\nUser Request:\n{query}"
        return self._execute_claude(task=enhanced_query, timeout=timeout)

    def run_claude_execution(
        self,
        run_context: RunContext,
        task: str,
        timeout: int | None = None,
    ) -> str:
        """
        Run Claude Code for execution with code modification capabilities.

        REQUIRES USER CONFIRMATION before execution. This tool gives Claude Code
        full access to modify the codebase. Use this for implementing features,
        fixing bugs, refactoring, or any task that requires file changes.

        Args:
            run_context: The run context for session state management
            task: The task to execute. Should be a clear description of what to
                  implement, fix, or change. Examples:
                  - "Implement JWT authentication in the auth module"
                  - "Fix the memory leak in the data processing pipeline"
                  - "Refactor the UserController to follow REST best practices"
                  - "Add unit tests for the PaymentService class"
            timeout: Optional timeout in seconds. If not provided, uses default_timeout.

        Returns:
            str: The output from Claude, including summary of changes made, files
                 modified, and any test results.
        """
        timeout = timeout or self.default_timeout
        return self._execute_claude(task=task, timeout=timeout)

    def _execute_claude(
        self,
        task: str,
        timeout: int,
    ) -> str:
        """
        Internal method to execute Claude CLI.

        Args:
            task: The task/prompt to execute.
            timeout: Timeout in seconds.

        Returns:
            str: The output from Claude, or an error message if execution fails.
        """
        try:
            cmd = [
                "claude",
                "--dangerously-skip-permissions",
                "--no-session-persistence",
                "-p",
                task,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n\nStderr:\n{result.stderr}"

            if result.returncode != 0:
                output += f"\n\nExit code: {result.returncode}"

            return output

        except subprocess.TimeoutExpired:
            return f"Error: Claude timed out after {timeout // 60} minutes"
        except FileNotFoundError:
            return "Error: Claude CLI not found. Please ensure it is installed and in PATH."
        except Exception as e:
            return f"Error running Claude: {e}"
