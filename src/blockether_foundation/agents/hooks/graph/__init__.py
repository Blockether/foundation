"""Graph database hooks package.

This package provides hooks for integrating graph database operations
with Agno agents.
"""

from .common import (
    GraphHookIterativeConfig,
    GraphHooksConfig,
    GraphIngestionIterativeConfig,
)
from .core import (
    create_post_graph_database_hook,
    create_pre_graph_database_hook,
)
from .prompts import (
    GRAPH_EXTRACTION_PROMPT,
    get_extraction_prompt,
)

__all__ = [
    # Configs
    "GraphHookIterativeConfig",
    "GraphHooksConfig",
    "GraphIngestionIterativeConfig",
    # Hook creators
    "create_post_graph_database_hook",
    "create_pre_graph_database_hook",
    # Prompts
    "GRAPH_EXTRACTION_PROMPT",
    "get_extraction_prompt",
]
