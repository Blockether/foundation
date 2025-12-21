# Project Guidelines

> **IMPORTANT:** `GEMINI.md` and `AGENTS.md` are symlinks to this file (`CLAUDE.md`). Always edit `CLAUDE.md` directly when updating project guidelines.

## Scripts & Agents

### Script Organization

- **scripts/** - Main entry scripts (bash scripts, CLI tools) for agents
- **scripts/impl/** - Implementation details (Python scripts, internal logic) for agents

These scripts are specifically for agents and should be used by agents when performing automated tasks. Users should not directly execute these scripts - they are designed to be called by agents as part of their workflow.

### Script Documentation Requirements

All bash scripts in `scripts/` must include comprehensive documentation at the top of the file containing:

1. **Purpose**: Clear description of what the script does
2. **Usage**: Complete usage examples with all available options
3. **Arguments**: Detailed explanation of all command-line arguments
4. **Examples**: Real-world usage examples
5. **Dependencies**: List of any external tools or scripts required
6. **Output**: Description of what the script outputs and exit codes

Implementation scripts should be placed in the `impl/` subdirectory to keep the main scripts directory clean and focused on user-facing tools.

## Code Style Guidelines

- **No emojis in code**: Do not use emojis in logging statements, comments, or any code. Use clear, descriptive text instead.
- **Use standard library for HTTP requests**: Always use Python's built-in urllib.request module for making HTTP requests instead of external HTTP libraries.

## Agno Hooks Implementation

### Post-Hooks Signature and Usage

When implementing post-hooks for Agno agents, follow these guidelines:

#### Required Imports
```python
from agno.run.agent import RunOutput
from agno.run.team import TeamRunOutput
from agno.session import AgentSession, TeamSession
```

#### Proper Post-Hook Signature
```python
async def hook(
    *,
    agent: Agent = None,                      # Agent instance (None for Team runs)
    team: Team = None,                         # Team instance (None for Agent runs)
    run_output: Union[RunOutput, TeamRunOutput],  # Output of the current run
    session: AgentSession | TeamSession,         # Session object
    user_id: str = None,                        # Optional user ID
    debug_mode: bool = None,                    # Optional debug mode flag
) -> None:
```

#### Accessing Input and Output Content

- **Input content**: Use `run_output.input.input_content_string()` to get the user's input
- **Output content**: Use `run_output.get_content_as_string()` to get the agent's response

Example:
```python
# Extract both input and output from RunOutput
input_content = ""
if run_output.input:
    input_content = run_output.input.input_content_string()

response_content = run_output.get_content_as_string()

# Combine for full context
full_context = f"User: {input_content}\nAssistant: {response_content}"
```

#### Important Notes

- Post-hooks receive `RunOutput` objects that contain both input and output
- Use keyword-only arguments (`*`) to match Agno's expected signature
- For team runs, `agent` will be None and vice versa
- The `input` attribute on RunOutput is of type `Optional[RunInput]`
- Avoid using parameters like `session_state`, `dependencies`, and `metadata` as they may not be consistently provided
