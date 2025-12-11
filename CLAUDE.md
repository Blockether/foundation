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
