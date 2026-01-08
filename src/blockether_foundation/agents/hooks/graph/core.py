"""Core hook logic for graph database operations."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import cast

from agno.agent.agent import Agent
from agno.run.agent import RunInput, RunOutput
from agno.run.team import TeamRunOutput
from agno.session import AgentSession, TeamSession
from agno.team import Team
from agno.utils.log import log_debug, log_warning  # type: ignore

from ....graph.database import GraphDatabase
from ....graph.formatting import (
    format_existing_entities_for_context,
    format_graph_query_results,
)
from ....graph.models import LLMGraphQueryOperations
from ....utils import (
    AgnoPostHook,
    AgnoPreHook,
    DebugMode,
    RunContext,
    UserId,
    build_extraction_context,
    dataclass_copy,
    format_main_agent_context,
    inject_context_to_run_input,
)
from .agents import INGESTION_AGENT, get_query_agent_with_context
from .tools import create_import_tool
from .common import GraphHookIterativeConfig, GraphIngestionIterativeConfig

GRAPH_CONTEXT_KEY = "graph_query_context"


def create_graph_hooks(
    graph: GraphDatabase,
    agentic_search: bool = True,
    agentic_ingestion: bool = True,
    file_path: str | None = None,
    search_config: GraphHookIterativeConfig | None = None,
    ingestion_config: GraphIngestionIterativeConfig | None = None,  # noqa: ARG001
    return_sync_wrapper: bool = False,
) -> tuple[AgnoPreHook, AgnoPostHook]:
    """Create paired pre/post hooks that share state via closure."""
    if search_config is None:
        search_config = GraphHookIterativeConfig()

    shared_state: dict[str, str] = {}

    async def pre_hook_logic(
        agent: Agent | Team,
        run_input: RunInput,
        session: AgentSession | TeamSession,  # noqa: ARG001
        user_id: UserId,  # noqa: ARG001
        debug_mode: DebugMode,
    ) -> None:
        shared_state.clear()

        if not agentic_search or graph.entity_count == 0:
            return

        input_content = run_input.input_content_string()
        existing_entities = format_existing_entities_for_context(
            graph, limit=20, max_rels_per_entity=3, max_content_length=50
        )

        query_agent = get_query_agent_with_context(existing_entities, search_config.max_queries)
        query_agent.model = agent.model
        query_agent.debug_mode = debug_mode or False

        main_context = format_main_agent_context(agent)
        if main_context:
            query_agent.instructions = f"{main_context}\n\n{query_agent.instructions}"

        response = await query_agent.arun(input_content)
        if not response.content:
            return

        queries = cast(LLMGraphQueryOperations, response.content)
        if not queries.entity_queries and not queries.relationship_queries:
            return

        context = format_graph_query_results(graph.translate_to_queries(queries), iteration=1)
        entity_ids = set(re.findall(r'id="([^"]+)"', context))

        if entity_ids:
            shared_state[GRAPH_CONTEXT_KEY] = context
            inject_context_to_run_input(run_input, context)
            log_debug(f"Pre-hook found {len(entity_ids)} entities")

    async def post_hook_logic(
        agent: Agent | Team,
        run_output: RunOutput | TeamRunOutput,
        session: AgentSession | TeamSession,  # noqa: ARG001
        run_context: RunContext,  # noqa: ARG001
        user_id: UserId,  # noqa: ARG001
        debug_mode: DebugMode,
    ) -> None:
        if not agentic_ingestion:
            return

        if GRAPH_CONTEXT_KEY in shared_state:
            log_debug("Skipping extraction - pre-hook already retrieved context")
            _save_graph_if_needed(graph, file_path)
            return

        full_context = build_extraction_context(run_output.input, run_output)
        if not full_context:
            return

        log_debug("Extracting graph operations via tool-based agent")
        existing_entities = format_existing_entities_for_context(
            graph, limit=30, max_rels_per_entity=5, max_content_length=100
        )

        instructions = INGESTION_AGENT.instructions
        if existing_entities:
            instructions += f"\n\n<existing_entities>\n{existing_entities}\n</existing_entities>"

        extraction_agent = dataclass_copy(
            INGESTION_AGENT,
            instructions=instructions,
            tools=[create_import_tool(graph)],
            model=agent.model,
            debug_mode=debug_mode or False,
        )

        main_context = format_main_agent_context(agent)
        if main_context:
            extraction_agent.instructions = f"{main_context}\n\n{extraction_agent.instructions}"

        try:
            if return_sync_wrapper:
                extraction_agent.run(full_context)
            else:
                await extraction_agent.arun(full_context)

            _save_graph_if_needed(graph, file_path)
        except Exception as e:
            log_warning(f"Extraction failed: {e}")

    def _save_graph_if_needed(g: GraphDatabase, path: str | None) -> None:
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            g.save_to_file(path)
            log_debug(f"Saved graph with {g.entity_count} entities to {path}")
        except Exception as e:
            log_warning(f"Failed to save graph: {e}")

    if return_sync_wrapper:

        def sync_pre_hook(
            agent: Agent | Team,
            run_input: RunInput,
            session: AgentSession | TeamSession,
            user_id: UserId,
            debug_mode: DebugMode,
        ) -> None:
            asyncio.run(pre_hook_logic(agent, run_input, session, user_id, debug_mode))

        def sync_post_hook(
            agent: Agent | Team,
            run_output: RunOutput | TeamRunOutput,
            session: AgentSession | TeamSession,
            run_context: RunContext,
            user_id: UserId,
            debug_mode: DebugMode,
        ) -> None:
            asyncio.run(
                post_hook_logic(agent, run_output, session, run_context, user_id, debug_mode)
            )

        return sync_pre_hook, sync_post_hook

    return pre_hook_logic, post_hook_logic


def create_pre_graph_database_hook(
    graph: GraphDatabase,
    agentic_search: bool,
    config: GraphHookIterativeConfig | None = None,
    return_sync_wrapper: bool = False,
) -> AgnoPreHook:
    """Create standalone pre-hook (use create_graph_hooks for paired hooks)."""
    pre_hook, _ = create_graph_hooks(
        graph=graph,
        agentic_search=agentic_search,
        agentic_ingestion=False,
        search_config=config,
        return_sync_wrapper=return_sync_wrapper,
    )
    return pre_hook


def create_post_graph_database_hook(
    graph: GraphDatabase,
    file_path: str | None = None,
    agentic_ingestion: bool = True,
    ingestion_config: GraphIngestionIterativeConfig | None = None,
    return_sync_wrapper: bool = False,
) -> AgnoPostHook:
    """Create standalone post-hook (use create_graph_hooks for paired hooks)."""
    _, post_hook = create_graph_hooks(
        graph=graph,
        agentic_search=False,
        agentic_ingestion=agentic_ingestion,
        file_path=file_path,
        ingestion_config=ingestion_config,
        return_sync_wrapper=return_sync_wrapper,
    )
    return post_hook
