"""Configuration classes for graph database hooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agno.utils.log import log_debug, log_warning  # type: ignore

if TYPE_CHECKING:
    from ....graph.database import GraphDatabase
    from ....utils import AgnoPostHook, AgnoPreHook


@dataclass
class GraphHookIterativeConfig:
    """Configuration for graph pre-hook query generation.

    Controls the single-pass query generation with optional retry for empty results.
    """

    max_queries: int = 3  # Maximum number of queries to generate
    max_query_retries: int = 2  # Retry queries with agent feedback if they return empty


@dataclass
class GraphIngestionIterativeConfig:
    """Configuration for iterative entity resolution in graph post-hook.

    Enables multi-pass extraction with entity resolution to improve graph quality
    by deduplicating entities and normalizing names before import.
    """

    max_iterations: int = 2  # 1 extraction + up to 1 resolution pass
    quality_threshold: float = 0.8  # 0.0-1.0 scale for early termination
    enable_entity_resolution: bool = True  # Run entity deduplication agent
    enable_name_normalization: bool = True  # Fix verbose entity names
    enable_quality_assessment: bool = True  # Run quality check before resolution
    max_import_retries: int = 3  # Max retries for validation/import with agent fixes


class GraphHooksConfig:
    """Configuration for graph database hooks.

    The graph database can be provided in two ways:
    1. Pass a GraphDatabase instance directly via `graph` parameter
    2. Pass a file path via `file_path` to load/save the graph from/to a file

    If both are provided, the `graph` instance takes priority but will be
    saved to `file_path` after modifications.
    """

    def __init__(
        self,
        graph: GraphDatabase | None = None,
        file_path: str | Path | None = None,
        agentic_search: bool = False,
        agentic_ingestion: bool = True,
        async_hooks: bool = True,
        iterative_config: GraphHookIterativeConfig | None = None,
        ingestion_config: GraphIngestionIterativeConfig | None = None,
    ):
        """Initialize graph hooks configuration.

        Args:
            graph: GraphDatabase instance to use. If None and file_path exists,
                   the graph will be loaded from file. If both are None, a new
                   empty graph will be created.
            file_path: Path to load/save the graph. If provided, the graph will
                       be automatically saved after modifications in post-hook.
            agentic_search: Whether to enable agentic search in pre-hook.
            agentic_ingestion: Whether to enable agentic ingestion in post-hook.
            async_hooks: Whether to use async hooks (True) or sync hooks (False).
            iterative_config: Configuration for iterative query refinement (pre-hook).
            ingestion_config: Configuration for entity resolution (post-hook).
        """
        # Import here to avoid circular imports
        from ....graph.database import GraphDatabase as GDB

        self.file_path = str(file_path) if file_path else None
        self.agentic_search = agentic_search
        self.agentic_ingestion = agentic_ingestion
        self.async_hooks = async_hooks
        self.iterative_config = iterative_config or GraphHookIterativeConfig()
        self.ingestion_config = ingestion_config or GraphIngestionIterativeConfig()

        self._graph = self._initialize_graph(graph, GDB)

    def _initialize_graph(
        self, graph: GraphDatabase | None, GDB: type[GraphDatabase]
    ) -> GraphDatabase:
        """Initialize the graph database from provided instance or file."""
        if graph is not None:
            return graph

        # Try to load from file if path exists
        if self.file_path and Path(self.file_path).exists():
            try:
                loaded_graph = GDB.load_from_file(self.file_path)
                log_debug(
                    f"Loaded graph with {loaded_graph.entity_count} entities from {self.file_path}"
                )
                return loaded_graph
            except Exception as e:
                log_warning(f"Failed to load graph from file {self.file_path}: {e}")

        return GDB(file_path=self.file_path)

    @property
    def graph(self) -> GraphDatabase:
        """Get the graph database instance."""
        return self._graph

    def save_graph(self) -> None:
        """Save the graph to file if file_path is configured."""
        if self.file_path:
            try:
                path = Path(self.file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._graph.save_to_file(self.file_path)
                log_debug(
                    f"Saved graph with {self._graph.entity_count} entities to {self.file_path}"
                )
            except Exception as e:
                log_warning(f"Failed to save graph to file {self.file_path}: {e}")

    def pre_hook(self) -> AgnoPreHook:
        """Get the pre-hook for graph database queries."""
        from .core import create_pre_graph_database_hook

        return create_pre_graph_database_hook(
            graph=self._graph,
            agentic_search=self.agentic_search,
            config=self.iterative_config,
            return_sync_wrapper=not self.async_hooks,
        )

    def post_hook(self) -> AgnoPostHook:
        """Get the post-hook for graph database ingestion."""
        from .core import create_post_graph_database_hook

        return create_post_graph_database_hook(
            graph=self._graph,
            file_path=self.file_path,
            agentic_ingestion=self.agentic_ingestion,
            ingestion_config=self.ingestion_config,
            return_sync_wrapper=not self.async_hooks,
        )
