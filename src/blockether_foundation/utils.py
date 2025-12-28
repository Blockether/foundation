from __future__ import annotations

import base64
import dataclasses
import inspect
import json
import logging
import secrets
from collections.abc import Callable, Coroutine
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar, cast

from agno.agent import Agent
from agno.media import Audio
from agno.models.message import Message
from agno.run.agent import RunInput, RunOutput
from agno.run.base import RunContext
from agno.run.team import TeamRunInput, TeamRunOutput
from agno.session import AgentSession, TeamSession
from agno.team import Team

T = TypeVar("T")


type UserId = str | None
type DebugMode = bool | None

type AgnoPreHook = Callable[
    [
        Agent | Team,
        RunInput,
        AgentSession | TeamSession,
        UserId,
        DebugMode,
    ],
    None | Coroutine[Any, Any, None],
]

type AgnoPostHook = Callable[
    [
        Agent | Team,
        RunOutput | TeamRunOutput,
        AgentSession | TeamSession,
        RunContext,
        UserId,
        DebugMode,
    ],
    None | Coroutine[Any, Any, None],
]

logger = logging.getLogger(__name__)

# Type alias for the input_content type
InputContentType = (
    str
    | list[Any]  # This covers List[Unknown], List[str], List[Message], etc.
    | dict[str, Any]  # This covers Dict[Unknown, Unknown]
    | Message
    | list[Message]
    | None
)


def none_invariant[T](condition: Callable[..., T | None], message: str) -> T:
    # Get the caller's frame
    caller_frame = inspect.currentframe()

    frame = caller_frame.f_back if caller_frame else None

    # Get the module from the caller's frame
    caller_module = inspect.getmodule(frame)

    # Get module name
    module_name = caller_module.__name__ if caller_module else "unknown"

    result = condition()
    assert result is not None, f"[{module_name}]: {message}"

    return result


def generate_secure_id(size: int = 8) -> str:
    """Generate a secure random ID using base64 encoding.

    Args:
        size: The desired length of the ID (default 8 characters).

    Returns:
        A URL-safe string of the requested length.
    """
    # Calculate how many bytes we need for the desired size
    # Base64 encoding produces 4 characters for every 3 bytes
    bytes_needed = (size * 3 + 3) // 4  # Round up to ensure we have enough

    # Generate secure random bytes
    random_bytes = secrets.token_bytes(bytes_needed)

    # Encode to base64 and make it URL-safe
    encoded = base64.urlsafe_b64encode(random_bytes).decode("ascii")

    # Remove padding characters and truncate to desired size
    encoded = encoded.rstrip("=")

    return encoded[:size]


def dataclass_copy[T](obj: T, **kwargs: Any) -> T:
    """Create a copy of a dataclass instance with updated fields.

    This is a safe wrapper around dataclasses.replace that handles:
    - Objects that are not dataclasses (returns the original object)
    - Invalid field names (filters them out)
    - Internal fields that are not in the __init__ signature (filters them out)
    - Objects with custom replace methods that might pass internal fields

    Args:
        obj: The dataclass instance to copy
        **kwargs: Fields to update in the copy

    Returns:
        A new dataclass instance with updated fields, or the original object
        if it's not a dataclass or no changes are needed
    """
    # Check if object is a dataclass first
    if not dataclasses.is_dataclass(obj):
        return obj

    # Get the field names of the dataclass that are actually in __init__
    init_params = set(inspect.signature(obj.__class__.__init__).parameters.keys())  # type: ignore

    # Remove 'self' from init params
    init_params.discard("self")

    # Also filter out internal fields starting with '_'
    valid_field_names = {
        f.name for f in fields(obj) if f.name in init_params and not f.name.startswith("_")
    }

    # Filter kwargs to only include valid fields
    valid_kwargs = {k: v for k, v in kwargs.items() if k in valid_field_names}

    # If no valid changes, return the original object
    if not valid_kwargs:
        return obj

    # Create a new dict with only the fields that can be passed to __init__
    # Start with current values but only for fields that are in init_params
    current_values: dict[str, Any] = {}
    for field in fields(obj):
        if field.name in init_params and not field.name.startswith("_"):
            # Safely get the field value if it exists in __dict__
            if field.name in obj.__dict__:
                current_values[field.name] = obj.__dict__[field.name]

    # Update with new values
    current_values.update(valid_kwargs)

    # Create a new instance directly with the filtered values
    # Use type: ignore to silence type checking issues with dynamic instantiation
    return obj.__class__(**current_values)  # type: ignore[arg-type, return-value]


def inject_context_to_run_input(
    run_input: RunInput,
    context_content: str,
    message_role: str = "system",
    prepend: bool = True,
) -> None:
    """Inject context content into the run_input.input_content.

    This function handles different types of input_content and injects the context
    in the appropriate way, similar to how the audio and graph hooks work.

    Args:
        run_input: The RunInput object to modify
        context_content: The context content to inject
        message_role: The role for the message if injecting as a Message (default: "system")
        prepend: Whether to prepend (True) or append (False) the context (default: True)
    """
    input_content = run_input.input_content  # type: ignore

    try:
        if isinstance(input_content, str):
            # String case
            if input_content:
                if prepend:
                    run_input.input_content = f"{context_content}\n\n{input_content}"
                else:
                    run_input.input_content = f"{input_content}\n\n{context_content}"
            else:
                run_input.input_content = context_content

        elif isinstance(input_content, list):
            # List case - could be list of Messages, strings, or other content
            if input_content and isinstance(input_content[0], Message):
                # List of Messages
                message_list: list[Message] = input_content  # type: ignore
                if prepend:
                    message_list.insert(0, Message(role=message_role, content=context_content))
                else:
                    message_list.append(Message(role=message_role, content=context_content))
                run_input.input_content = message_list
            else:
                # List of strings or other content
                if prepend:
                    cast(list[Any], input_content).insert(0, context_content)
                else:
                    cast(list[Any], input_content).append(context_content)
                run_input.input_content = input_content

        elif isinstance(input_content, Message):
            # Single Message case
            if isinstance(input_content.content, str):
                if prepend:
                    # Wrap both in a list with context first
                    run_input.input_content = [
                        Message(role=message_role, content=context_content),
                        input_content,
                    ]
                else:
                    # Append context as a new message
                    run_input.input_content = [
                        input_content,
                        Message(role=message_role, content=context_content),
                    ]

        elif input_content is None:
            # None case - initialize with context
            run_input.input_content = context_content

        elif isinstance(input_content, dict):
            # Dict case
            logger.warning("Unsupported input_content type for context injection: dict. Skipping.")

        else:
            # Fallback - try to convert to string
            if prepend:
                run_input.input_content = f"{context_content}\n\n{str(input_content)}"
            else:
                run_input.input_content = f"{str(input_content)}\n\n{context_content}"

    except Exception as e:
        logger.warning(f"Could not inject context into input_content: {e}. Skipping.")


def format_audio_transcript(audio: Audio) -> str:
    """Format audio transcript into a standard string format.

    Args:
        audio: Agno Audio object with transcript

    Returns:
        Formatted transcript string
    """
    # Handle Path objects for filepath
    filepath = None
    if audio.filepath:
        filepath = str(audio.filepath) if isinstance(audio.filepath, Path) else audio.filepath

    source = audio.id or filepath or audio.url or "unknown source"
    transcript = audio.transcript or ""

    return f"Transcript from audio ({source}):\n{transcript}"


def build_extraction_context(
    run_input: RunInput | TeamRunInput | None,
    run_output: RunOutput | TeamRunOutput | None,
) -> str:
    """Build the full context string for extraction from input and output.

    This function creates a structured XML context from agent input and output,
    useful for post-processing hooks that need to analyze the conversation.

    Args:
        run_input: The input to the agent
        run_output: The output from the agent

    Returns:
        Structured context string with XML tags
    """
    # Get input and output content
    input_content = run_input.input_content_string() if run_input else ""
    response_content = ""
    if run_output is not None:
        response_content = cast(str, run_output.get_content_as_string())

    if not response_content:
        return ""

    # Build full context for extraction
    if input_content:
        return f"<original_user_message>\n{input_content}</original_user_message>\n\n<assistant_response>\n{response_content}</assistant_response>"
    else:
        return f"<assistant_response>\n{response_content}</assistant_response>"


def create_agent_with_instructions(
    description: str,
    instructions: list[str] | str,
    model: Any | None = None,
    debug_mode: DebugMode = False,
    expected_output: str | None = None,
    output_schema: Any | None = None,
) -> Agent:
    """Create an agent with structured description and instructions.

    This is a utility function for creating specialized agents with
    consistent patterns, useful for hook implementations.

    Args:
        description: The agent's description/purpose
        instructions: List of instruction strings
        model: Optional model to use (inherits from creating agent if None)
        debug_mode: Optional debug mode flag
        output_schema: Optional output schema for structured output

    Returns:
        Configured Agent instance
    """
    agent_kwargs: dict[str, Any] = {
        "description": description,
        "instructions": instructions,
        "expected_output": expected_output,
        "debug_mode": debug_mode or False,
    }

    if model is not None:
        agent_kwargs["model"] = model
    if output_schema is not None:
        agent_kwargs["output_schema"] = output_schema

    return Agent(**agent_kwargs)


async def execute_agent_async(agent: Agent, context: str) -> RunOutput:
    """Execute an agent asynchronously with standardized error handling.

    Args:
        agent: The agent to execute
        context: The context/input to provide to the agent

    Returns:
        The agent's output
    """
    return await agent.arun(context)


def execute_agent_sync(agent: Agent, context: str) -> RunOutput:
    """Execute an agent synchronously with standardized error handling.

    Args:
        agent: The agent to execute
        context: The context/input to provide to the agent

    Returns:
        The agent's output
    """
    return agent.run(context)


def save_data_to_json_file(
    data: dict[str, Any],
    file_path: str,
    create_dirs: bool = True,
    indent: int = 2,
) -> None:
    """Save data to a JSON file with proper directory creation.

    This utility handles file path creation, directory creation, and JSON serialization
    in a consistent way for hook implementations.

    Args:
        data: The data to save
        file_path: The file path to save to
        create_dirs: Whether to create parent directories
        indent: JSON indentation level
    """
    try:
        path = Path(file_path)
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        logger.debug(f"Data saved to: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to save data to {file_path}: {e}")


def read_binary_file_safely(file_path: str | Path) -> bytes | None:
    """Read a binary file safely with proper error handling.

    This utility handles file reading with consistent error handling
    for hook implementations.

    Args:
        file_path: The path to the file to read

    Returns:
        File contents as bytes, or None if reading failed
    """
    try:
        path = Path(file_path) if isinstance(file_path, str) else file_path

        with open(path, "rb") as f:
            return f.read()

    except (FileNotFoundError, OSError, PermissionError) as e:
        logger.warning(f"Failed to read file {file_path}: {e}")
        return None
