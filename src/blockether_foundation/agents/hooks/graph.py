"""Graph database hooks for Agno agents."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from agno.agent.agent import Agent
from agno.run.agent import RunInput, RunOutput
from agno.run.team import TeamRunOutput
from agno.session import AgentSession, TeamSession
from agno.team import Team
from agno.utils.log import log_debug, log_error, log_warning  # type: ignore

from ...graph.database import GraphDatabase
from ...graph.formatting import (
    format_existing_entities_for_context,
    format_graph_query_results,
)
from ...graph.models import LLMGraphOperations, LLMGraphQueryOperations, QualityAssessment
from ...utils import (
    DebugMode,
    UserId,
    build_agent_context,
    build_extraction_context,
    create_agent_with_instructions,
    ensure_global_graph,
    execute_agent_async,
    execute_agent_sync,
    get_global_graph,
    inject_context_to_run_input,
    save_global_graph,
)

# Type aliases for hook functions
AgnoPreHook = Callable[
    [
        Agent | Team,
        RunInput,
        AgentSession | TeamSession,
        UserId,
        DebugMode,
    ],
    Any,
]

AgnoPostHook = Callable[
    [
        Agent | Team,
        "RunOutput | TeamRunOutput",
        AgentSession | TeamSession,
        UserId,
        DebugMode,
    ],
    Any,
]

# Common instructions for graph entity and relationship types
GRAPH_ENTITY_RELATIONSHIP_INSTRUCTIONS = [
    "Valid Entity Types and their usage:",
    "- creature: Living beings, people, animals, or any sentient entities.",
    "- organization: Companies, institutions, or formal groups.",
    "- location: Places, buildings, geographic features, or virtual spaces.",
    "- event: Historical events, occurrences, or time-specific happenings.",
    "- date: Specific calendar dates, time periods, or temporal references.",
    "- document: Files, reports, articles, or written records.",
    "- attachment: Binary files, media, images, or non-text attachments.",
    "- object: Physical items, equipment, tools, or tangible artifacts.",
    "- concept: Abstract ideas, theories, principles, or intellectual concepts.",
    "- example: Specific instances, illustrations, or representative cases of concepts or rules.",
    "- rule: Logical rules, constraints, regulations, or governing principles.",
    "- pattern: Recognizable patterns, templates, or recurring structures, or like it can be used in therapy context as well.",
    "- memory: Stored experiences, learned information, or recollections.",
    "Valid Relationship Types and their usage:",
    "- part_of: Entity is a component, member, or subset of another Entity.",
    "- owned_by: Entity is owned, possessed, or controlled by another Entity.",
    "- occurs_at: Event happens at a specific time and/or place.",
    "- related_to: General association, connection, or relationship between entities.",
    "- triggers: Entity causes, initiates, or precipitates another entity or event.",
    "- originates_from: Entity has its source, origin, or starting point in another entity.",
    "",
]


@dataclass
class GraphHookIterativeConfig:
    """Configuration for iterative context improvement in graph pre-hook.

    Enables multi-iteration query refinement to improve context quality
    by assessing retrieved results and generating better queries.
    """

    enabled: bool = False  # Opt-in, backward compatible
    max_iterations: int = 2  # 1 initial + 1 refinement pass
    quality_threshold: float = 0.75  # 0.0-1.0 scale for early termination
    enable_gap_analysis: bool = True  # Generate new queries for missing info
    enable_entity_expansion: bool = True  # Use BFS on interesting entities
    enable_query_refinement: bool = True  # Refine existing queries
    bfs_expansion_depth: int = 1  # How many hops for entity expansion
    max_entities_to_expand: int = 3  # Limit BFS operations


class GraphHooksConfig:
    """Configuration for graph database hooks."""

    def __init__(
        self,
        agentic_search: bool = False,
        agentic_ingestion: bool = True,
        async_hooks: bool = True,
        file_sync: str | None = None,
        iterative_config: GraphHookIterativeConfig | None = None,
    ):
        self.agentic_search = agentic_search
        self.agentic_ingestion = agentic_ingestion
        self.async_hooks = async_hooks
        self.file_sync = file_sync
        self.iterative_config = iterative_config or GraphHookIterativeConfig()

    def pre_hook(self) -> AgnoPreHook:
        """Get the pre-hook for graph database queries.

        Returns:
            Pre-hook function configured with this config's settings
        """
        if self.async_hooks:
            return _create_async_pre_graph_database_hook(
                agentic_search=self.agentic_search,
                file_sync=self.file_sync,
                iterative_config=self.iterative_config,
            )
        else:
            return _create_sync_pre_graph_database_hook(
                agentic_search=self.agentic_search,
                file_sync=self.file_sync,
                iterative_config=self.iterative_config,
            )

    def post_hook(self) -> AgnoPostHook:
        """Get the post-hook for graph database ingestion.

        Returns:
            Post-hook function configured with this config's settings
        """
        if self.async_hooks:
            return _create_async_post_graph_database_hook(
                agentic_ingestion=self.agentic_ingestion,
                file_sync=self.file_sync,
            )
        else:
            return _create_sync_post_graph_database_hook(
                agentic_ingestion=self.agentic_ingestion,
                file_sync=self.file_sync,
            )


# Common functions for both sync and async hooks
def _initialize_or_load_global_graph(
    agent: Agent | Team,
    file_sync: str | None = None,
) -> GraphDatabase | None:
    """Initialize or load global graph database for hooks."""
    if not isinstance(agent, Agent):
        return None

    global_db = get_global_graph(agent)
    if not global_db and file_sync and Path(file_sync).exists():
        # Load graph from file and set it as the global graph
        try:
            loaded_graph = GraphDatabase.load_from_file(file_sync)
            log_debug(f"Loaded graph with {loaded_graph.entity_count} entities from {file_sync}")
            save_global_graph(agent, loaded_graph)
            global_db = loaded_graph
        except Exception as e:
            log_warning(f"Failed to load graph from file {file_sync}: {e}")

    if not global_db:
        global_db = ensure_global_graph(agent)

    return global_db


def _save_graphs_to_file(
    file_path: str,
    global_db: GraphDatabase | None,
    local_db: GraphDatabase | None,
) -> None:
    """Save graphs to file."""
    if not file_path:
        return

    try:
        graph_path = Path(file_path)
        graph_path.parent.mkdir(parents=True, exist_ok=True)

        # Decide which graph to save (prioritize local, fallback to global)
        db_to_save = local_db if local_db else global_db
        if db_to_save:
            db_to_save.save_to_file(file_path)
            log_debug(f"Saved graph with {db_to_save.entity_count} entities to {file_path}")
    except Exception as e:
        log_warning(f"Failed to save graph to file {file_path}: {e}")


def _deduplicate_and_accumulate(
    new_context: str,
    accumulated_context: str,
    accumulated_entity_ids: set[str],
) -> tuple[str, set[str]]:
    """Deduplicate entities and accumulate context across iterations.

    Extracts entity IDs from the new context XML, filters out duplicates,
    and appends new context to the accumulated context.

    Args:
        new_context: XML context from current iteration
        accumulated_context: XML context accumulated from previous iterations
        accumulated_entity_ids: Set of entity IDs already seen

    Returns:
        Tuple of (updated accumulated context, updated entity ID set)
    """
    # Extract entity IDs from new context using regex
    # Pattern matches: id="<entity_id>"
    entity_id_pattern = r'id="([^"]+)"'
    new_entity_ids = set(re.findall(entity_id_pattern, new_context))

    # Find truly new entities (not in accumulated set)
    truly_new_ids = new_entity_ids - accumulated_entity_ids

    if not truly_new_ids:
        log_debug("No new entities in this iteration, skipping accumulation")
        return accumulated_context, accumulated_entity_ids

    # Update accumulated set
    updated_entity_ids = accumulated_entity_ids | truly_new_ids

    # Append new context to accumulated context
    if accumulated_context:
        # Concatenate with newline separator
        updated_context = accumulated_context + "\n\n" + new_context
    else:
        updated_context = new_context

    log_debug(f"Accumulated {len(truly_new_ids)} new entities (total: {len(updated_entity_ids)})")

    return updated_context, updated_entity_ids


async def _assess_context_quality_async(
    user_input: str,
    accumulated_context: str,
    iteration: int,
    max_iterations: int,
    agent: Agent | Team,
    debug_mode: DebugMode,
) -> QualityAssessment:
    """Use LLM to assess if retrieved context is sufficient (async).

    Args:
        user_input: User's original request
        accumulated_context: Context retrieved so far from graph queries
        iteration: Current iteration number
        max_iterations: Maximum number of iterations allowed
        agent: Main agent instance (for model/debug settings)
        debug_mode: Debug mode flag

    Returns:
        QualityAssessment model with satisfaction, score, gaps, suggestions
    """
    assessment_agent = Agent(
        description="Expert at evaluating graph query result quality and completeness",
        instructions=[
            "You will receive:",
            "1. User's original request",
            "2. Context retrieved from graph queries so far",
            "3. Current iteration number",
            "",
            "Assess whether the context is SUFFICIENT to help answer the user's request.",
            "Consider:",
            "- Relevance: Does context relate to user's question?",
            "- Completeness: Are key entities/relationships present?",
            "- Depth: Is there enough detail?",
            "",
            "Identify:",
            "- Coverage gaps: What information is missing?",
            "- Interesting entities: Which entities should we explore deeper?",
            "- Query refinements: How to improve existing queries?",
            "",
            "Be realistic about what's needed. If basic context exists, it may be sufficient.",
            "Don't demand perfection - focus on whether the user's question can be reasonably answered.",
        ],
        output_schema=QualityAssessment,
        model=agent.model,
        debug_mode=debug_mode or False,
    )

    input_to_agent = f"""<user_request>
{user_input}
</user_request>

<retrieved_context>
{accumulated_context if accumulated_context else "No context retrieved yet."}
</retrieved_context>

<iteration_info>
Current iteration: {iteration}/{max_iterations}
</iteration_info>"""

    response = await assessment_agent.arun(input_to_agent)  # type: ignore
    return cast(QualityAssessment, response.content)


def _assess_context_quality_sync(
    user_input: str,
    accumulated_context: str,
    iteration: int,
    max_iterations: int,
    agent: Agent | Team,
    debug_mode: DebugMode,
) -> QualityAssessment:
    """Use LLM to assess if retrieved context is sufficient (sync).

    Args:
        user_input: User's original request
        accumulated_context: Context retrieved so far from graph queries
        iteration: Current iteration number
        max_iterations: Maximum number of iterations allowed
        agent: Main agent instance (for model/debug settings)
        debug_mode: Debug mode flag

    Returns:
        QualityAssessment model with satisfaction, score, gaps, suggestions
    """
    assessment_agent = Agent(
        description="Expert at evaluating graph query result quality and completeness",
        instructions=[
            "You will receive:",
            "1. User's original request",
            "2. Context retrieved from graph queries so far",
            "3. Current iteration number",
            "",
            "Assess whether the context is SUFFICIENT to help answer the user's request.",
            "Consider:",
            "- Relevance: Does context relate to user's question?",
            "- Completeness: Are key entities/relationships present?",
            "- Depth: Is there enough detail?",
            "",
            "Identify:",
            "- Coverage gaps: What information is missing?",
            "- Interesting entities: Which entities should we explore deeper?",
            "- Query refinements: How to improve existing queries?",
            "",
            "Be realistic about what's needed. If basic context exists, it may be sufficient.",
            "Don't demand perfection - focus on whether the user's question can be reasonably answered.",
        ],
        output_schema=QualityAssessment,
        model=agent.model,
        debug_mode=debug_mode or False,
    )

    input_to_agent = f"""<user_request>
{user_input}
</user_request>

<retrieved_context>
{accumulated_context if accumulated_context else "No context retrieved yet."}
</retrieved_context>

<iteration_info>
Current iteration: {iteration}/{max_iterations}
</iteration_info>"""

    response = assessment_agent.run(input_to_agent)  # type: ignore
    return cast(QualityAssessment, response.content)


async def _refine_queries_async(
    user_input: str,
    quality_assessment: QualityAssessment,
    config: GraphHookIterativeConfig,
    accumulated_entity_ids: set[str],
    agent: Agent | Team,
    debug_mode: DebugMode,
) -> LLMGraphQueryOperations:
    """Generate refined queries based on quality assessment feedback (async).

    Args:
        user_input: User's original request
        quality_assessment: Quality assessment from previous iteration
        config: Iterative configuration with strategy flags
        accumulated_entity_ids: Set of entity IDs already retrieved
        agent: Main agent instance (for model/debug settings)
        debug_mode: Debug mode flag

    Returns:
        LLMGraphQueryOperations with refined queries
    """
    refinement_agent = Agent(
        description="Expert at refining graph queries based on quality feedback",
        instructions=[
            "You will receive:",
            "1. User's original request",
            "2. Quality assessment with gaps and suggestions",
            "3. Entity IDs already retrieved (to avoid duplicates)",
            "",
            "Generate REFINED queries using these strategies:",
            "",
            "Strategy 1 - Gap Analysis:",
            "- For each coverage gap, create targeted entity/relationship queries",
            "- Focus on missing information types",
            "",
            "Strategy 2 - Entity Expansion:",
            "- For interesting entities, create queries with expand_depth parameter",
            "- Use expand_depth=1 or 2 to explore related entities via BFS",
            "- Set expand_direction to 'outgoing', 'incoming', or 'both' as appropriate",
            "",
            "Strategy 3 - Query Refinement:",
            "- Try broader search queries if previous ones were too narrow",
            "- Adjust entity types based on suggestions",
            "- Use different search terms",
            "",
            "IMPORTANT: Generate new, complementary queries to fill gaps.",
            "Don't repeat queries for information we already have.",
            *GRAPH_ENTITY_RELATIONSHIP_INSTRUCTIONS,
        ],
        output_schema=LLMGraphQueryOperations,
        model=agent.model,
        debug_mode=debug_mode or False,
    )

    gaps_text = (
        "\n- ".join(quality_assessment.coverage_gaps)
        if quality_assessment.coverage_gaps
        else "None identified"
    )
    interesting_entities_text = (
        ", ".join(quality_assessment.interesting_entities)
        if quality_assessment.interesting_entities
        else "None identified"
    )
    refinements_text = (
        "\n- ".join(quality_assessment.suggested_refinements)
        if quality_assessment.suggested_refinements
        else "None identified"
    )

    input_to_agent = f"""<user_request>
{user_input}
</user_request>

<quality_assessment>
Quality Score: {quality_assessment.quality_score}
Satisfied: {quality_assessment.satisfied}

Coverage Gaps:
- {gaps_text}

Interesting Entities to Expand:
{interesting_entities_text}

Suggested Refinements:
- {refinements_text}
</quality_assessment>

<configuration>
Gap Analysis: {"enabled" if config.enable_gap_analysis else "disabled"}
Entity Expansion: {"enabled" if config.enable_entity_expansion else "disabled"}
  - Max entities to expand: {config.max_entities_to_expand}
  - BFS expansion depth: {config.bfs_expansion_depth}
Query Refinement: {"enabled" if config.enable_query_refinement else "disabled"}
</configuration>

<already_retrieved_entities>
Number of entities already retrieved: {len(accumulated_entity_ids)}
(You don't need to query for these again)
</already_retrieved_entities>

Generate new queries to address the gaps and improve context quality."""

    response = await refinement_agent.arun(input_to_agent)  # type: ignore
    return cast(LLMGraphQueryOperations, response.content)


def _refine_queries_sync(
    user_input: str,
    quality_assessment: QualityAssessment,
    config: GraphHookIterativeConfig,
    accumulated_entity_ids: set[str],
    agent: Agent | Team,
    debug_mode: DebugMode,
) -> LLMGraphQueryOperations:
    """Generate refined queries based on quality assessment feedback (sync).

    Args:
        user_input: User's original request
        quality_assessment: Quality assessment from previous iteration
        config: Iterative configuration with strategy flags
        accumulated_entity_ids: Set of entity IDs already retrieved
        agent: Main agent instance (for model/debug settings)
        debug_mode: Debug mode flag

    Returns:
        LLMGraphQueryOperations with refined queries
    """
    refinement_agent = Agent(
        description="Expert at refining graph queries based on quality feedback",
        instructions=[
            "You will receive:",
            "1. User's original request",
            "2. Quality assessment with gaps and suggestions",
            "3. Entity IDs already retrieved (to avoid duplicates)",
            "",
            "Generate REFINED queries using these strategies:",
            "",
            "Strategy 1 - Gap Analysis:",
            "- For each coverage gap, create targeted entity/relationship queries",
            "- Focus on missing information types",
            "",
            "Strategy 2 - Entity Expansion:",
            "- For interesting entities, create queries with expand_depth parameter",
            "- Use expand_depth=1 or 2 to explore related entities via BFS",
            "- Set expand_direction to 'outgoing', 'incoming', or 'both' as appropriate",
            "",
            "Strategy 3 - Query Refinement:",
            "- Try broader search queries if previous ones were too narrow",
            "- Adjust entity types based on suggestions",
            "- Use different search terms",
            "",
            "IMPORTANT: Generate new, complementary queries to fill gaps.",
            "Don't repeat queries for information we already have.",
            *GRAPH_ENTITY_RELATIONSHIP_INSTRUCTIONS,
        ],
        output_schema=LLMGraphQueryOperations,
        model=agent.model,
        debug_mode=debug_mode or False,
    )

    gaps_text = (
        "\n- ".join(quality_assessment.coverage_gaps)
        if quality_assessment.coverage_gaps
        else "None identified"
    )
    interesting_entities_text = (
        ", ".join(quality_assessment.interesting_entities)
        if quality_assessment.interesting_entities
        else "None identified"
    )
    refinements_text = (
        "\n- ".join(quality_assessment.suggested_refinements)
        if quality_assessment.suggested_refinements
        else "None identified"
    )

    input_to_agent = f"""<user_request>
{user_input}
</user_request>

<quality_assessment>
Quality Score: {quality_assessment.quality_score}
Satisfied: {quality_assessment.satisfied}

Coverage Gaps:
- {gaps_text}

Interesting Entities to Expand:
{interesting_entities_text}

Suggested Refinements:
- {refinements_text}
</quality_assessment>

<configuration>
Gap Analysis: {"enabled" if config.enable_gap_analysis else "disabled"}
Entity Expansion: {"enabled" if config.enable_entity_expansion else "disabled"}
  - Max entities to expand: {config.max_entities_to_expand}
  - BFS expansion depth: {config.bfs_expansion_depth}
Query Refinement: {"enabled" if config.enable_query_refinement else "disabled"}
</configuration>

<already_retrieved_entities>
Number of entities already retrieved: {len(accumulated_entity_ids)}
(You don't need to query for these again)
</already_retrieved_entities>

Generate new queries to address the gaps and improve context quality."""

    response = refinement_agent.run(input_to_agent)  # type: ignore
    return cast(LLMGraphQueryOperations, response.content)


def _create_async_pre_graph_database_hook(
    agentic_search: bool,
    file_sync: str | None = None,
    iterative_config: GraphHookIterativeConfig | None = None,
) -> AgnoPreHook:
    """Create async pre-hook for graph database operations."""
    # Ensure config is set
    if iterative_config is None:
        iterative_config = GraphHookIterativeConfig()

    async def hook(
        agent: Agent | Team,
        run_input: RunInput,
        session: AgentSession | TeamSession,
        user_id: UserId,
        debug_mode: DebugMode,
    ) -> None:
        """Hook implementation for pre-graph database operations."""
        log_debug("Running async pre-hook for graph database queries")
        if not agentic_search:
            return

        # Get or create global graph database
        global_db = None
        if isinstance(agent, Agent):
            global_db = get_global_graph(agent)
            if not global_db and file_sync and Path(file_sync).exists():
                # Load graph from file and set it as the global graph
                try:
                    loaded_graph = GraphDatabase.load_from_file(file_sync)
                    log_debug(
                        f"Loaded graph with {loaded_graph.entity_count} entities from {file_sync}"
                    )
                    save_global_graph(agent, loaded_graph)
                    global_db = loaded_graph
                except Exception as e:
                    log_warning(f"Failed to load graph from file {file_sync}: {e}")
            if not global_db:
                global_db = ensure_global_graph(agent)

        if not global_db:
            log_debug("No agent available, cannot create global graph")
            return

        input_content = run_input.input_content_string()
        log_debug(f"Generating graph queries for input: {input_content[:50]}...")

        # Create agent to generate graph queries
        query_agent = Agent(
            description="You are an expert at querying knowledge graphs.",
            instructions=[
                "Analyze the user's input to identify key entities and relationships.",
                "Generate a set of graph queries (entities and relationships) to retrieve relevant context.",
                "Focus on finding information that would help answer the user's request.",
                "Use specific entity types and relationship types where possible.",
                *GRAPH_ENTITY_RELATIONSHIP_INSTRUCTIONS,
            ],
            output_schema=LLMGraphQueryOperations,
            model=agent.model,
            debug_mode=debug_mode or False,
        )

        # Initialize iteration state
        accumulated_context = ""
        accumulated_entity_ids: set[str] = set()
        assessment: QualityAssessment | None = None
        iteration = 0
        max_iterations = iterative_config.max_iterations if iterative_config.enabled else 1

        try:
            while iteration < max_iterations:
                iteration += 1
                log_debug(f"Iteration {iteration}/{max_iterations}")

                # 1. Generate or refine queries
                if iteration == 1:
                    # Initial query generation
                    response = await query_agent.arun(input_content)  # type: ignore
                    if not response.content:
                        log_warning("No queries generated in first iteration")
                        break
                    queries = cast(LLMGraphQueryOperations, response.content)
                else:
                    # Refinement based on quality assessment
                    if assessment is None:
                        log_warning("No assessment available for refinement")
                        break
                    queries = await _refine_queries_async(
                        input_content,
                        assessment,
                        iterative_config,
                        accumulated_entity_ids,
                        agent,
                        debug_mode,
                    )

                # 2. Execute queries and accumulate context
                if queries.entity_queries or queries.relationship_queries:
                    query_builders = global_db.translate_to_queries(queries)
                    new_context = format_graph_query_results(query_builders, iteration=iteration)

                    # Deduplicate and accumulate
                    accumulated_context, accumulated_entity_ids = _deduplicate_and_accumulate(
                        new_context, accumulated_context, accumulated_entity_ids
                    )
                else:
                    log_debug("No queries to execute in this iteration")

                # 3. Assess quality (skip on last iteration)
                if iterative_config.enabled and iteration < max_iterations:
                    try:
                        assessment = await _assess_context_quality_async(
                            input_content,
                            accumulated_context,
                            iteration,
                            max_iterations,
                            agent,
                            debug_mode,
                        )

                        log_debug(
                            f"Quality assessment: score={assessment.quality_score}, satisfied={assessment.satisfied}"
                        )

                        # 4. Check termination
                        if (
                            assessment.satisfied
                            or assessment.quality_score >= iterative_config.quality_threshold
                        ):
                            log_debug(
                                f"Context quality satisfied (score: {assessment.quality_score}), stopping iteration"
                            )
                            break
                    except Exception as e:
                        log_warning(
                            f"Quality assessment failed: {e}, continuing with accumulated context"
                        )
                        break

            # Inject accumulated context
            if accumulated_context:
                inject_context_to_run_input(run_input, accumulated_context)
            else:
                log_debug("No context accumulated from any iteration")

        except Exception as e:
            log_error(f"Iterative improvement failed: {e}. Using single-pass.")
            # Single-pass: Try single-pass if iterative loop fails
            try:
                response = await query_agent.arun(input_content)  # type: ignore
                if response.content:
                    queries = cast(LLMGraphQueryOperations, response.content)
                    if queries.entity_queries or queries.relationship_queries:
                        query_builders = global_db.translate_to_queries(queries)
                        context_msg = format_graph_query_results(query_builders)
                        inject_context_to_run_input(run_input, context_msg)
            except Exception as single_pass_error:
                log_error(f"Single-pass failed: {single_pass_error}")

    return hook


def _create_sync_pre_graph_database_hook(
    agentic_search: bool,
    file_sync: str | None = None,
    iterative_config: GraphHookIterativeConfig | None = None,
) -> AgnoPreHook:
    """Create sync pre-hook for graph database operations."""
    # Ensure config is set
    if iterative_config is None:
        iterative_config = GraphHookIterativeConfig()

    def hook(
        agent: Agent | Team,
        run_input: RunInput,
        session: AgentSession | TeamSession,
        user_id: UserId,
        debug_mode: DebugMode,
    ) -> None:
        """Hook implementation for pre-graph database operations (sync)."""
        log_debug("Running sync pre-hook for graph database queries")
        if not agentic_search:
            return

        # Initialize or load global graph database
        global_db = _initialize_or_load_global_graph(agent, file_sync)

        if not global_db:
            log_debug("No agent available, cannot create global graph")
            return

        input_content = run_input.input_content_string()
        log_debug(f"Generating graph queries for input: {input_content[:50]}...")

        # Create agent to generate graph queries
        query_agent = Agent(
            description="You are an expert at querying knowledge graphs.",
            instructions=[
                "Analyze the user's input to identify key entities and relationships.",
                "Generate a set of graph queries (entities and relationships) to retrieve relevant context.",
                "Focus on finding information that would help answer the user's request.",
                "Use specific entity types and relationship types where possible.",
                *GRAPH_ENTITY_RELATIONSHIP_INSTRUCTIONS,
            ],
            output_schema=LLMGraphQueryOperations,
            model=agent.model,
            debug_mode=debug_mode or False,
        )

        # Initialize iteration state
        accumulated_context = ""
        accumulated_entity_ids: set[str] = set()
        assessment: QualityAssessment | None = None
        iteration = 0
        max_iterations = iterative_config.max_iterations if iterative_config.enabled else 1

        try:
            while iteration < max_iterations:
                iteration += 1
                log_debug(f"Iteration {iteration}/{max_iterations}")

                # 1. Generate or refine queries
                if iteration == 1:
                    # Initial query generation
                    response = query_agent.run(input_content)  # type: ignore
                    if not response.content:
                        log_warning("No queries generated in first iteration")
                        break
                    queries = cast(LLMGraphQueryOperations, response.content)
                else:
                    # Refinement based on quality assessment
                    if assessment is None:
                        log_warning("No assessment available for refinement")
                        break
                    queries = _refine_queries_sync(
                        input_content,
                        assessment,
                        iterative_config,
                        accumulated_entity_ids,
                        agent,
                        debug_mode,
                    )

                # 2. Execute queries and accumulate context
                if queries.entity_queries or queries.relationship_queries:
                    query_builders = global_db.translate_to_queries(queries)
                    new_context = format_graph_query_results(query_builders, iteration=iteration)

                    # Deduplicate and accumulate
                    accumulated_context, accumulated_entity_ids = _deduplicate_and_accumulate(
                        new_context, accumulated_context, accumulated_entity_ids
                    )
                else:
                    log_debug("No queries to execute in this iteration")

                # 3. Assess quality (skip on last iteration)
                if iterative_config.enabled and iteration < max_iterations:
                    try:
                        assessment = _assess_context_quality_sync(
                            input_content,
                            accumulated_context,
                            iteration,
                            max_iterations,
                            agent,
                            debug_mode,
                        )

                        log_debug(
                            f"Quality assessment: score={assessment.quality_score}, satisfied={assessment.satisfied}"
                        )

                        # 4. Check termination
                        if (
                            assessment.satisfied
                            or assessment.quality_score >= iterative_config.quality_threshold
                        ):
                            log_debug(
                                f"Context quality satisfied (score: {assessment.quality_score}), stopping iteration"
                            )
                            break
                    except Exception as e:
                        log_warning(
                            f"Quality assessment failed: {e}, continuing with accumulated context"
                        )
                        break

            # Inject accumulated context
            if accumulated_context:
                inject_context_to_run_input(run_input, accumulated_context)
            else:
                log_debug("No context accumulated from any iteration")

        except Exception as e:
            log_error(f"Iterative improvement failed: {e}. Using single-pass.")
            # Single-pass: Try single-pass if iterative loop fails
            try:
                response = query_agent.run(input_content)  # type: ignore
                if response.content:
                    queries = cast(LLMGraphQueryOperations, response.content)
                    if queries.entity_queries or queries.relationship_queries:
                        query_builders = global_db.translate_to_queries(queries)
                        context_msg = format_graph_query_results(query_builders)
                        inject_context_to_run_input(run_input, context_msg)
            except Exception as single_pass_error:
                log_error(f"Single-pass failed: {single_pass_error}")

    return hook


def _create_async_post_graph_database_hook(
    agentic_ingestion: bool,
    file_sync: str | None = None,
) -> AgnoPostHook:
    """Create async post-hook for graph database operations."""

    async def hook(
        agent: Agent | Team,
        run_output: RunOutput | TeamRunOutput,
        session: AgentSession | TeamSession,
        user_id: UserId,
        debug_mode: DebugMode,
    ) -> None:
        """Hook implementation for post-graph database operations."""
        if not agentic_ingestion:
            return

        # Initialize or load global graph database using reusable function
        global_db = _initialize_or_load_global_graph(agent, file_sync)

        if not global_db:
            log_debug("No agent available, cannot create global graph")
            return

        # Build extraction context using utils
        full_context = build_extraction_context(run_output.input, run_output)
        if not full_context:
            return

        log_debug("Extracting graph operations from conversation context")

        # Get existing entities and relationships from the graph to provide context
        existing_context = ""
        if global_db and global_db.get_entity_count() > 0:
            existing_context = format_existing_entities_for_context(
                global_db, limit=20, max_rels_per_entity=10, max_content_length=200
            )
            existing_context = "\n\n" + existing_context if existing_context else ""
            log_debug("Providing context with existing entities in XML format")

        # Get agent's main instructions for context using utils
        agent_context = build_agent_context(agent)

        # Create extraction agent using utils with simplified agent creation
        extraction_agent = create_agent_with_instructions(
            description="You are an expert at extracting meaningful entities and relationships from conversations.",
            instructions=[
                "You will receive a conversation with XML tags marking different parts:",
                "- <user_message> contains what the user asked",
                "- <assistant_response> contains the assistant's reply",
                "",
                "You may also see <existing_entities> showing entities already in the knowledge graph.",
                "You may also see <agent_context> showing the main agent's purpose and domain focus.",
                "",
                "USE THIS CONTEXT to:",
                "1. Reference existing entities when they match what's mentioned",
                "2. Create relationships between new entities and existing ones",
                "3. Build a denser, more connected graph",
                "4. Avoid creating duplicate entities",
                "5. Prioritize entities that align with the agent's main purpose and domain",
                "",
                "IMPORTANT: Extract ONLY SPECIFIC, NAMED entities with real informational value.",
                "Focus on concrete facts and knowledge that would be useful to store long-term.",
                "",
                *GRAPH_ENTITY_RELATIONSHIP_INSTRUCTIONS,
                "",
                "Be precise and accurate with the information you extract.",
                "Only extract facts that are explicitly stated in the conversation.",
                "When you see an entity that already exists in <existing_entities>, reference it by name instead of creating a duplicate.",
            ],
            output_schema=LLMGraphOperations,
            model=agent.model,  # Use agent.model directly for both Agent and Team
            debug_mode=debug_mode or False,
        )

        try:
            # Perform extraction using utils
            input_to_agent = full_context + existing_context + agent_context
            log_debug("Running async extraction agent to get graph operations")
            extraction_response = await execute_agent_async(extraction_agent, input_to_agent)

            # Get the LLMGraphOperations object from structured output
            operations = cast(LLMGraphOperations, extraction_response.content)
            log_debug(f"Extracted operations: {operations}")

            # Import operations to graph (might fail but still save the graph)
            try:
                global_db.import_operations(operations)
                log_debug(f"Graph post-hook completed. Imported operations: {operations}")
            except Exception as import_error:
                log_warning(f"Failed to import operations to graph: {import_error}")

            # Save the modified graph back to the session to persist changes
            save_global_graph(agent, global_db)
            log_debug("Graph saved to session")

            # Sync to file if configured
            if file_sync:
                _save_graphs_to_file(file_sync, global_db, None)
        except Exception as e:
            log_warning(f"Failed to process extraction in async post-hook: {e}")
            return

    return hook


def _create_sync_post_graph_database_hook(
    agentic_ingestion: bool,
    file_sync: str | None = None,
) -> AgnoPostHook:
    """Create sync post-hook for graph database operations."""

    def hook(
        agent: Agent | Team,
        run_output: RunOutput | TeamRunOutput,
        session: AgentSession | TeamSession,
        user_id: UserId,
        debug_mode: DebugMode,
    ) -> None:
        """Hook implementation for post-graph database operations (sync)."""
        if not agentic_ingestion:
            return

        # Initialize or load global graph database
        global_db = _initialize_or_load_global_graph(agent, file_sync)

        if not global_db:
            log_debug("No agent available, cannot create global graph")
            return

        # Build extraction context using utils
        full_context = build_extraction_context(run_output.input, run_output)
        if not full_context:
            return

        log_debug("Extracting graph operations from conversation context")

        # Get existing entities and relationships from the graph to provide context
        existing_context = ""
        if global_db and global_db.get_entity_count() > 0:
            existing_context = format_existing_entities_for_context(
                global_db, limit=20, max_rels_per_entity=10, max_content_length=200
            )
            existing_context = "\n\n" + existing_context if existing_context else ""
            log_debug("Providing context with existing entities in XML format")

        # Get agent's main instructions for context using utils
        agent_context = build_agent_context(agent)

        # Create extraction agent using utils with simplified agent creation
        extraction_agent = create_agent_with_instructions(
            description="You are an expert at extracting meaningful entities and relationships from conversations.",
            instructions=[
                "You will receive a conversation with XML tags marking different parts:",
                "- <user_message> contains what the user asked",
                "- <assistant_response> contains the assistant's reply",
                "",
                "You may also see <existing_entities> showing entities already in the knowledge graph.",
                "You may also see <agent_context> showing the main agent's purpose and domain focus.",
                "",
                "USE THIS CONTEXT to:",
                "1. Reference existing entities when they match what's mentioned",
                "2. Create relationships between new entities and existing ones",
                "3. Build a denser, more connected graph",
                "4. Avoid creating duplicate entities",
                "5. Prioritize entities that align with the agent's main purpose and domain",
                "",
                "IMPORTANT: Extract ONLY SPECIFIC, NAMED entities with real informational value.",
                "Focus on concrete facts and knowledge that would be useful to store long-term.",
                "",
                *GRAPH_ENTITY_RELATIONSHIP_INSTRUCTIONS,
                "",
                "Be precise and accurate with the information you extract.",
                "Only extract facts that are explicitly stated in the conversation.",
                "When you see an entity that already exists in <existing_entities>, reference it by name instead of creating a duplicate.",
            ],
            output_schema=LLMGraphOperations,
            model=agent.model,  # Use agent.model directly for both Agent and Team
            debug_mode=debug_mode or False,
        )

        try:
            # Perform extraction using utils
            input_to_agent = full_context + existing_context + agent_context
            log_debug("Running sync extraction agent to get graph operations")
            extraction_response = execute_agent_sync(extraction_agent, input_to_agent)

            # Get the LLMGraphOperations object from structured output
            operations = cast(LLMGraphOperations, extraction_response.content)
            log_debug(f"Extracted operations: {operations}")

            # Import operations to graph (might fail but still save the graph)
            try:
                global_db.import_operations(operations)
                log_debug(f"Graph post-hook completed. Imported operations: {operations}")
            except Exception as import_error:
                log_warning(f"Failed to import operations to graph: {import_error}")

            # Save the modified graph back to the session to persist changes
            save_global_graph(agent, global_db)
            log_debug("Graph saved to session")

            # Sync to file if configured
            if file_sync:
                _save_graphs_to_file(file_sync, global_db, None)
        except Exception as e:
            log_warning(f"Failed to process extraction in sync post-hook: {e}")
            return

    return hook
