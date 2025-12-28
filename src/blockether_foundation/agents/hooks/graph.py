"""Graph database hooks for Agno agents."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import cast

from agno.agent.agent import Agent
from agno.run.agent import RunInput, RunOutput
from agno.run.team import TeamRunOutput
from agno.session import AgentSession, TeamSession
from agno.team import Team
from agno.utils.log import log_debug, log_warning  # type: ignore

from ...graph.common import GRAPH_SCHEMA_XML
from ...graph.database import GraphDatabase
from ...graph.formatting import (
    format_entities_with_relationships_for_prompt,
    format_existing_entities_for_context,
    format_graph_query_results,
)
from ...graph.models import LLMGraphOperations, LLMGraphQueryOperations, QualityAssessment
from ...utils import (
    AgnoPostHook,
    AgnoPreHook,
    DebugMode,
    RunContext,
    UserId,
    build_extraction_context,
    create_agent_with_instructions,
    execute_agent_async,
    execute_agent_sync,
    inject_context_to_run_input,
)


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

    # Coverage thresholds for stopping iteration based on entity coverage
    small_graph_threshold: int = 10  # Graph size considered "small"
    small_graph_coverage_threshold: float = 0.8  # 80% coverage for small graphs
    medium_graph_threshold: int = 50  # Graph size considered "medium"
    medium_graph_coverage_threshold: float = 0.6  # 60% coverage for medium graphs
    large_graph_coverage_threshold: float = 0.4  # 40% coverage for large graphs

    # Limits for large graphs
    large_graph_threshold: int = 100  # Graph size considered "large"
    large_graph_entity_limit: int = 50  # Stop after retrieving this many entities from large graphs
    large_graph_min_iteration: int = 2  # Only apply limit after this iteration


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
            iterative_config: Configuration for iterative query refinement.
        """
        self.file_path = str(file_path) if file_path else None
        self.agentic_search = agentic_search
        self.agentic_ingestion = agentic_ingestion
        self.async_hooks = async_hooks
        self.iterative_config = iterative_config or GraphHookIterativeConfig()

        # Initialize graph database
        self._graph = self._initialize_graph(graph)

    def _initialize_graph(self, graph: GraphDatabase | None) -> GraphDatabase:
        """Initialize the graph database from provided instance or file."""
        if graph is not None:
            return graph

        # Try to load from file if path exists
        if self.file_path and Path(self.file_path).exists():
            try:
                loaded_graph = GraphDatabase.load_from_file(self.file_path)
                log_debug(
                    f"Loaded graph with {loaded_graph.entity_count} entities from {self.file_path}"
                )
                return loaded_graph
            except Exception as e:
                log_warning(f"Failed to load graph from file {self.file_path}: {e}")

        # Create new empty graph
        return GraphDatabase()

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
        """Get the pre-hook for graph database queries.

        Returns:
            Pre-hook function configured with this config's settings
        """
        return _create_pre_graph_database_hook(
            graph=self._graph,
            agentic_search=self.agentic_search,
            iterative_config=self.iterative_config,
            return_sync_wrapper=not self.async_hooks,
        )

    def post_hook(self) -> AgnoPostHook:
        """Get the post-hook for graph database ingestion.

        Returns:
            Post-hook function configured with this config's settings
        """
        return _create_post_graph_database_hook(
            graph=self._graph,
            file_path=self.file_path,
            agentic_ingestion=self.agentic_ingestion,
            return_sync_wrapper=not self.async_hooks,
        )


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


def _should_continue_iteration(
    accumulated_entity_ids: set[str],
    total_graph_entities: int,
    iteration: int,
    max_iterations: int,
    config: GraphHookIterativeConfig,
) -> bool:
    """Check if it makes sense to continue iterating based on entity coverage.

    Prevents unnecessary iterations when we've already retrieved most relevant entities.

    Args:
        accumulated_entity_ids: Set of entity IDs already retrieved
        total_graph_entities: Total number of entities in the graph
        iteration: Current iteration number
        max_iterations: Maximum number of iterations allowed
        config: Iterative configuration

    Returns:
        True if iteration should continue, False otherwise
    """
    # Don't continue if we're at the max iteration
    if iteration >= max_iterations:
        return False

    # Determine coverage threshold based on graph size (using config)
    if total_graph_entities <= config.small_graph_threshold:
        coverage_threshold = config.small_graph_coverage_threshold
    elif total_graph_entities <= config.medium_graph_threshold:
        coverage_threshold = config.medium_graph_coverage_threshold
    else:
        coverage_threshold = config.large_graph_coverage_threshold

    current_coverage = len(accumulated_entity_ids) / max(total_graph_entities, 1)

    # If we've already retrieved a significant portion, don't continue
    if current_coverage >= coverage_threshold:
        log_debug(
            f"Stopping iteration: retrieved {len(accumulated_entity_ids)}/{total_graph_entities} "
            f"entities ({current_coverage:.1%} >= {coverage_threshold:.1%} threshold)"
        )
        return False

    # For larger graphs with many entities, be more conservative about continuing
    # Only stop if we have a LOT of entities AND we're past the minimum iteration
    if (
        total_graph_entities > config.large_graph_threshold
        and len(accumulated_entity_ids) >= config.large_graph_entity_limit
        and iteration >= config.large_graph_min_iteration
    ):
        log_debug(
            f"Stopping iteration: large graph but already retrieved {len(accumulated_entity_ids)} entities "
            f"after iteration {iteration}"
        )
        return False

    return True


async def _assess_context_quality(
    user_input: str,
    accumulated_context: str,
    iteration: int,
    max_iterations: int,
    agent: Agent | Team,
    debug_mode: DebugMode,
    async_hooks: bool = False,
) -> QualityAssessment:
    """Use LLM to assess if retrieved context is sufficient.

    Args:
        user_input: User's original request
        accumulated_context: Context retrieved so far from graph queries
        iteration: Current iteration number
        max_iterations: Maximum number of iterations allowed
        agent: Main agent instance (for model/debug settings)
        debug_mode: Debug mode flag
        async_hooks: If True, runs agent synchronously

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

    input_to_agent = f"""<original_user_request>
{user_input}
</original_user_request>

<retrieved_context>
{accumulated_context if accumulated_context else "No context retrieved yet."}
</retrieved_context>

<iteration_info>
Current iteration: {iteration}/{max_iterations}
</iteration_info>"""

    if async_hooks:
        response = assessment_agent.run(input_to_agent)
    else:
        response = await assessment_agent.arun(input_to_agent)
    return cast(QualityAssessment, response.content)


async def _assess_context_quality_async(
    user_input: str,
    accumulated_context: str,
    iteration: int,
    max_iterations: int,
    agent: Agent | Team,
    debug_mode: DebugMode,
) -> QualityAssessment:
    """Use LLM to assess if retrieved context is sufficient (async)."""
    return await _assess_context_quality(
        user_input,
        accumulated_context,
        iteration,
        max_iterations,
        agent,
        debug_mode,
        async_hooks=False,
    )


async def _refine_queries(
    user_input: str,
    quality_assessment: QualityAssessment,
    config: GraphHookIterativeConfig,
    accumulated_entity_ids: set[str],
    agent: Agent | Team,
    debug_mode: DebugMode,
    global_db: GraphDatabase,
    async_hooks: bool = False,
) -> LLMGraphQueryOperations:
    """Generate refined queries based on quality assessment feedback.

    Args:
        user_input: User's original request
        quality_assessment: Quality assessment from previous iteration
        config: Iterative configuration with strategy flags
        accumulated_entity_ids: Set of entity IDs already retrieved
        agent: Main agent instance (for model/debug settings)
        debug_mode: Debug mode flag
        async_hooks: If True, runs agent synchronously

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
            GRAPH_SCHEMA_XML,
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

    input_to_agent = f"""{user_input}

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

{format_entities_with_relationships_for_prompt(accumulated_entity_ids, global_db)}

Generate new queries to address the gaps and improve context quality."""

    if async_hooks:
        response = refinement_agent.run(input_to_agent)
    else:
        response = await refinement_agent.arun(input_to_agent)
    return cast(LLMGraphQueryOperations, response.content)


async def _refine_queries_async(
    user_input: str,
    quality_assessment: QualityAssessment,
    config: GraphHookIterativeConfig,
    accumulated_entity_ids: set[str],
    agent: Agent | Team,
    debug_mode: DebugMode,
    global_db: GraphDatabase,
) -> LLMGraphQueryOperations:
    """Generate refined queries based on quality assessment feedback (async)."""
    return await _refine_queries(
        user_input,
        quality_assessment,
        config,
        accumulated_entity_ids,
        agent,
        debug_mode,
        global_db,
        async_hooks=False,
    )


async def _handle_initial_query_generation(
    query_agent: Agent,
    input_content: str,
) -> LLMGraphQueryOperations | None:
    """Generate initial queries for graph database search.

    Args:
        query_agent: Agent configured for query generation
        input_content: User's input content

    Returns:
        LLMGraphQueryOperations or None if generation failed
    """
    response = await query_agent.arun(input_content)
    if not response.content:
        log_warning("No queries generated in first iteration")
        return None
    return cast(LLMGraphQueryOperations, response.content)


async def _handle_query_refinement(
    input_content: str,
    assessment: QualityAssessment | None,
    iterative_config: GraphHookIterativeConfig,
    accumulated_entity_ids: set[str],
    agent: Agent | Team,
    debug_mode: DebugMode,
    graph: GraphDatabase,
) -> LLMGraphQueryOperations | None:
    """Refine queries based on quality assessment.

    Args:
        input_content: User's input content
        assessment: Quality assessment from previous iteration
        iterative_config: Iterative configuration
        accumulated_entity_ids: Set of entity IDs already retrieved
        agent: Main agent instance
        debug_mode: Debug mode flag
        graph: Graph database instance

    Returns:
        LLMGraphQueryOperations or None if refinement failed
    """
    if assessment is None:
        log_warning("No assessment available for refinement")
        return None

    return await _refine_queries_async(
        input_content,
        assessment,
        iterative_config,
        accumulated_entity_ids,
        agent,
        debug_mode,
        graph,
    )


def _execute_and_accumulate_queries(
    queries: LLMGraphQueryOperations,
    graph: GraphDatabase,
    accumulated_context: str,
    accumulated_entity_ids: set[str],
    iteration: int,
) -> tuple[str, set[str]]:
    """Execute queries and accumulate context.

    Args:
        queries: Queries to execute
        graph: Graph database instance
        accumulated_context: Context from previous iterations
        accumulated_entity_ids: Entity IDs from previous iterations
        iteration: Current iteration number

    Returns:
        Tuple of (updated context, updated entity IDs)
    """
    if not queries.entity_queries and not queries.relationship_queries:
        log_debug("No queries to execute in this iteration")
        return accumulated_context, accumulated_entity_ids

    query_builders = graph.translate_to_queries(queries)
    new_context = format_graph_query_results(query_builders, iteration=iteration)

    return _deduplicate_and_accumulate(new_context, accumulated_context, accumulated_entity_ids)


async def _check_quality_and_continue(
    input_content: str,
    accumulated_context: str,
    accumulated_entity_ids: set[str],
    total_entities: int,
    iteration: int,
    max_iterations: int,
    iterative_config: GraphHookIterativeConfig,
    agent: Agent | Team,
    debug_mode: DebugMode,
) -> bool:
    """Check if iteration should continue based on quality assessment.

    Args:
        input_content: User's input content
        accumulated_context: Context accumulated so far
        accumulated_entity_ids: Entity IDs retrieved so far
        total_entities: Total entities in graph
        iteration: Current iteration number
        max_iterations: Maximum allowed iterations
        iterative_config: Iterative configuration
        agent: Main agent instance
        debug_mode: Debug mode flag

    Returns:
        True if iteration should continue, False otherwise
    """
    # Check entity coverage
    if not _should_continue_iteration(
        accumulated_entity_ids,
        total_entities,
        iteration,
        max_iterations,
        iterative_config,
    ):
        entity_names = ", ".join(list(accumulated_entity_ids)[:5])
        if len(accumulated_entity_ids) > 5:
            entity_names += f" ... and {len(accumulated_entity_ids) - 5} more"
        log_debug(
            f"Entity coverage check suggests stopping at iteration {iteration} "
            f"({len(accumulated_entity_ids)}/{total_entities} entities retrieved: {entity_names})"
        )
        return False

    # Skip quality assessment on last iteration
    if iteration >= max_iterations:
        return False

    # Run quality assessment
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

        # Check if quality is sufficient
        if assessment.satisfied or assessment.quality_score >= iterative_config.quality_threshold:
            log_debug(
                f"Context quality satisfied (score: {assessment.quality_score}), stopping iteration"
            )
            return False

        return True

    except Exception as e:
        log_warning(
            f"Quality assessment failed: {e}, stopping iteration and using accumulated context"
        )
        return False


def _create_pre_graph_database_hook(
    graph: GraphDatabase,
    agentic_search: bool,
    iterative_config: GraphHookIterativeConfig | None = None,
    return_sync_wrapper: bool = False,
) -> AgnoPreHook:
    """Create pre-hook for graph database operations.

    Args:
        graph: GraphDatabase instance to query
        agentic_search: Whether to enable agentic search
        iterative_config: Configuration for iterative query refinement
        return_sync_wrapper: If True, returns a sync wrapper that runs async logic synchronously

    Returns:
        Pre-hook function (async or sync wrapper based on return_sync_wrapper parameter)
    """
    # Ensure config is set
    if iterative_config is None:
        iterative_config = GraphHookIterativeConfig()

    async def hook_logic(
        agent: Agent | Team,
        run_input: RunInput,
        session: AgentSession | TeamSession,
        user_id: UserId,
        debug_mode: DebugMode,
    ) -> None:
        """Shared hook implementation for pre-graph database operations."""
        log_debug(
            f"Running pre-hook for graph database queries ({'sync' if return_sync_wrapper else 'async'})"
        )
        if not agentic_search:
            return

        total_entities = graph.entity_count
        if total_entities == 0:
            log_debug("Graph is empty, skipping graph query hook")
            return

        input_content = run_input.input_content_string()
        existing_entities = format_existing_entities_for_context(
            graph, limit=20, max_rels_per_entity=3, max_content_length=50
        )

        # Create agent to generate graph queries
        query_agent = create_agent_with_instructions(
            description="You are an expert in producing graph database queries to retrieve relevant context.",
            instructions=(
                dedent("""
                    <task_description>
                        You are an expert in producing graph database queries to retrieve relevant context.
                        You will be given the user's message/request/question, potentially denoted with <user_message> or <original_user_message> tags.
                        Your task is to first analyze the user request and assess what information is needed from the graph database to help answer the user's request.
                        For trivial requests, you may decide that no graph information is needed.
                    </task_description>
                    <xml_tag_reference>
                        <tag name="user_message">Denotes the user's message, request, or question.</tag>
                        <tag name="original_user_message">Denotes the original user input that started the conversation.</tag>
                        <tag name="graph_schema">Denotes the graph schema definitions with valid entity and relationship types.</tag>
                        <tag name="existing_entities">Denotes current top entities in the graph with the highest information density.</tag>
                    </xml_tag_reference>
                    <quidelines>
                        <guideline>You will be given the graph schema definitions denoted by <graph_schema> to help you formulate valid queries.</guideline>
                        <guideline>You will be given current top entities in the graph with the highest information density, denoted by <existing_entities>.</guideline>
                        <guideline>For fuzzy search queries, use the search_query parameter on entity_queries. ENSURE THE SEARCH QUERY contains no formatted text, emoticons, or special characters.</guideline>
                        <guideline>Use specific entity types and relationship types where possible to narrow down results.</guideline>
                    </quidelines>""")
                + "\n"
                + GRAPH_SCHEMA_XML
                + "\n"
                + existing_entities
            ),
            expected_output=dedent("""
                Expectations for your output:
                - Return an LLMGraphQueryOperations object with two lists: entity_queries and relationship_queries.
                - Each query should be relevant to the user's request and aim to retrieve useful context.
                - If no graph information is needed, return empty lists for both entity_queries and relationship_queries.
                - Prefer concise queries with precise search terms and specific entity/relationship types.
                - Avoid overly broad queries that return too much irrelevant data.
                - Ensure queries do not duplicate information already present in the existing entities.
            """),
            output_schema=LLMGraphQueryOperations,
            model=agent.model,
            debug_mode=debug_mode or False,
        )

        # Initialize iteration state
        accumulated_context = ""
        accumulated_entity_ids: set[str] = set()
        iteration = 0
        max_iterations = iterative_config.max_iterations if iterative_config.enabled else 1
        assessment: QualityAssessment | None = None

        while iteration < max_iterations:
            iteration += 1
            log_debug(f"Iteration {iteration}/{max_iterations}")

            # 1. Generate or refine queries
            if iteration == 1:
                queries = await _handle_initial_query_generation(query_agent, input_content)
                if queries is None:
                    break
            else:
                queries = await _handle_query_refinement(
                    input_content,
                    assessment,
                    iterative_config,
                    accumulated_entity_ids,
                    agent,
                    debug_mode,
                    graph,
                )
                if queries is None:
                    break

            # 2. Execute queries and accumulate context
            accumulated_context, accumulated_entity_ids = _execute_and_accumulate_queries(
                queries, graph, accumulated_context, accumulated_entity_ids, iteration
            )

            # 3. Check if we should continue
            if iterative_config.enabled:
                should_continue = await _check_quality_and_continue(
                    input_content,
                    accumulated_context,
                    accumulated_entity_ids,
                    total_entities,
                    iteration,
                    max_iterations,
                    iterative_config,
                    agent,
                    debug_mode,
                )
                if not should_continue:
                    break

            # Inject accumulated context
            if accumulated_context:
                inject_context_to_run_input(run_input, accumulated_context)
            else:
                log_debug("No context accumulated from any iteration")

    # Return appropriate hook based on return_sync_wrapper parameter
    if return_sync_wrapper:

        def sync_hook(
            agent: Agent | Team,
            run_input: RunInput,
            session: AgentSession | TeamSession,
            user_id: UserId,
            debug_mode: DebugMode,
        ) -> None:
            """Sync wrapper that runs the async hook logic."""
            asyncio.run(hook_logic(agent, run_input, session, user_id, debug_mode))

        return sync_hook
    else:
        return hook_logic


def _create_post_graph_database_hook(
    graph: GraphDatabase,
    file_path: str | None = None,
    agentic_ingestion: bool = True,
    return_sync_wrapper: bool = False,
) -> AgnoPostHook:
    """Create post-hook for graph database operations.

    Args:
        graph: GraphDatabase instance to modify
        file_path: Optional file path to save graph after modifications
        agentic_ingestion: Whether to enable agentic ingestion
        return_sync_wrapper: If True, returns a sync wrapper that runs async logic synchronously

    Returns:
        Post-hook function (async or sync wrapper based on return_sync_wrapper parameter)
    """

    async def hook_logic(
        agent: Agent | Team,
        run_output: RunOutput | TeamRunOutput,
        session: AgentSession | TeamSession,
        run_context: RunContext,
        user_id: UserId,
        debug_mode: DebugMode,
    ) -> None:
        """Shared hook implementation for post-graph database operations."""
        if not agentic_ingestion:
            return

        # Build extraction context using utils
        full_context = build_extraction_context(run_output.input, run_output)
        if not full_context:
            return

        log_debug("Extracting graph operations from conversation context")

        log_debug(f"<FULL_CONTEXT>\n{full_context}\n</FULL_CONTEXT>")

        # Create extraction agent using utils with simplified agent creation
        extraction_agent = create_agent_with_instructions(
            description="You are an expert graph knowledge extractor specialist.",
            instructions=dedent(f"""
            <graph_extraction_instructions>
                <task_description>
                    You are an expert graph knowledge extractor specialist.
                    You will be given context denoted by various XML tags.
                </task_description>
                <xml_tag_reference>
                    <tag name="original_user_message">Denotes the original user input that started the conversation.</tag>
                    <tag name="agent_responses">Denotes the response to the original user message by the agent.</tag>
                    <tag name="conversation_history">Denotes the full conversation between the user and the agent.</tag>
                    <tag name="extracted_entities">Denotes entities that have already been extracted in this conversation.</tag>
                    <tag name="extracted_relationships">Denotes relationships that have already been extracted in this conversation.</tag>
                </xml_tag_reference>
                <schema_reference>
                    You are given the allowed entity and relationship types in the graph schema below.
                    {GRAPH_SCHEMA_XML}
                </schema_reference>
                <quidelines>
                    <guideline>When creating an entity ensure it's </guideline>
                </guidelines>
            </graph_extraction_instructions>
            """),
            expected_output=dedent("""
                Expectations for your output:
                - Return an LLMGraphOperations object with entities to create/update/delete and relationships to establish.
                - Extract new entities and relationships that are not already present in <extracted_entities> and <extracted_relationships>.
                - Use valid entity types and relationship types as defined in the <graph_schema>.
                - Prefer MANY entities connected by relationships over one large entity with lots of content:

                  <example id="1">
                    <user_message>My name is Alice and I work at Acme Corp as a software engineer since 2015. I will be retiring in 3 years.</user_message>
                    <graph_operations>
                        <entities_add_operations>
                            <entity name="Alice" type="person">
                                <content>Software engineer at Acme Corp, planning retirement in 3 years</content>
                            </entity>
                            <entity name="Acme Corp" type="organization">
                                <content>Company employing Alice since 2015</content>
                            </entity>
                            <entity name="Software Engineer" type="concept">
                                <content>Professional role held by Alice at Acme Corp</content>
                            </entity>
                            <entity name="Alice works at Acme Corp since 2015" type="fact">
                                <content>Fact: Alice has been employed at Acme Corp as a software engineer since 2015</content>
                            </entity>
                            <entity name="Alice plans to retire in 3 years" type="fact">
                                <content>Fact: Alice plans to retire from her position in 3 years</content>
                            </entity>
                        </entities_add_operations>
                        <relationships_add_operations>
                            <rel source_name="Alice" target_name="Acme Corp" type="owned_by" />
                            <rel source_name="Alice" target_name="Software Engineer" type="related_to" />
                            <rel source_name="Alice" target_name="Alice works at Acme Corp since 2015" type="is_a_fact" />
                            <rel source_name="Alice" target_name="Alice plans to retire in 3 years" type="is_a_fact" />
                        </relationships_add_operations>
                    </graph_operations>
                  </example>
            """),
            output_schema=LLMGraphOperations,
            model=agent.model,
            debug_mode=debug_mode or False,
        )

        try:
            input_to_agent = full_context
            log_debug(
                f"Running {'async' if not return_sync_wrapper else 'sync'} extraction agent to get graph operations"
            )

            if return_sync_wrapper:
                extraction_response = execute_agent_sync(extraction_agent, input_to_agent)
            else:
                extraction_response = await execute_agent_async(extraction_agent, input_to_agent)

            operations = cast(LLMGraphOperations, extraction_response.content)
            log_debug(f"Extracted operations: {operations}")

            try:
                graph.import_operations(operations)
                log_debug(f"Graph post-hook completed. Imported operations: {operations}")
            except Exception as import_error:
                log_warning(f"Failed to import operations to graph: {import_error}")

            if file_path:
                try:
                    path = Path(file_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    graph.save_to_file(file_path)
                    log_debug(f"Saved graph with {graph.entity_count} entities to {file_path}")
                except Exception as save_error:
                    log_warning(f"Failed to save graph to file {file_path}: {save_error}")

        except Exception as e:
            log_warning(
                f"Failed to process extraction in {'async' if not return_sync_wrapper else 'sync'} post-hook: {e}"
            )
            return

    # Return appropriate hook based on return_sync_wrapper parameter
    if return_sync_wrapper:

        def sync_hook(
            agent: Agent | Team,
            run_output: RunOutput | TeamRunOutput,
            session: AgentSession | TeamSession,
            run_context: RunContext,
            user_id: UserId,
            debug_mode: DebugMode,
        ) -> None:
            """Sync wrapper that runs the async hook logic."""
            asyncio.run(hook_logic(agent, run_output, session, run_context, user_id, debug_mode))

        return sync_hook
    else:
        return hook_logic
