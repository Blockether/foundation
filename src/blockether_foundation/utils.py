import inspect
from collections.abc import Callable


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
