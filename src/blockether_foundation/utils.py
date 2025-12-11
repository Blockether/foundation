import base64
import dataclasses
import inspect
import secrets
from collections.abc import Callable
from dataclasses import fields
from typing import Any, TypeVar

T = TypeVar("T")


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
    init_params = set(inspect.signature(obj.__class__.__init__).parameters.keys())
    # Remove 'self' from init params
    init_params.discard('self')

    # Also filter out internal fields starting with '_'
    valid_field_names = {f.name for f in fields(obj) if f.name in init_params and not f.name.startswith("_")}

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
            current_values[field.name] = getattr(obj, field.name)

    # Update with new values
    current_values.update(valid_kwargs)

    # Create a new instance directly with the filtered values
    # Use type: ignore to silence type checking issues with dynamic instantiation
    return obj.__class__(**current_values)  # type: ignore[arg-type, return-value]
