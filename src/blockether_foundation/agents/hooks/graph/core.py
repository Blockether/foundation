"""Core hook logic for graph database operations."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

from agno.agent.agent import Agent
from agno.run.agent import RunInput, RunOutput
from agno.run.team import TeamRunOutput
from agno.session import AgentSession, TeamSession
from agno.team import Team
from agno.utils.log import log_debug, log_warning  # type: ignore

from ....graph.common import (
    ENTITY_TYPE_DEFINITIONS,
    RELATIONSHIP_TYPE_DEFINITIONS,
)
from ....graph.database import GraphDatabase
from ....graph.formatting import (
    format_existing_entities_for_context,
    format_graph_query_results,
)
from ....graph.models import (
    LLMGraphAddEntity,
    LLMGraphOperations,
    LLMGraphQueryOperations,
)
from ....utils import (
    AgnoPostHook,
    AgnoPreHook,
    DebugMode,
    RunContext,
    UserId,
    build_extraction_context,
    create_agent_with_instructions,
    execute_agent_async,
    execute_agent_sync,
    format_main_agent_context,
    inject_context_to_run_input,
)
from .common import GraphHookIterativeConfig, GraphIngestionIterativeConfig
from .prompts import (
    FIX_EMPTY_QUERIES_INSTRUCTIONS,
    FIX_OPERATIONS_INSTRUCTIONS,
    QUERY_GENERATION_DESCRIPTION,
    QUERY_GENERATION_EXPECTED_OUTPUT,
    build_fix_empty_queries_input,
    build_fix_operations_input,
    get_extraction_prompt,
    get_query_generation_instructions,
)

if TYPE_CHECKING:
    pass


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


async def _fix_empty_queries_with_agent(
    input_content: str,
    failed_queries: LLMGraphQueryOperations,
    graph: GraphDatabase,
    agent: Agent | Team,
    debug_mode: DebugMode,
    async_hooks: bool = False,
) -> LLMGraphQueryOperations | None:
    """Use agent to fix queries that returned no results.

    Args:
        input_content: User's original request
        failed_queries: Queries that returned empty results
        graph: Graph database instance
        agent: Main agent instance
        debug_mode: Debug mode flag
        async_hooks: If True, uses sync execution

    Returns:
        Fixed LLMGraphQueryOperations or None if fixing failed
    """
    # Get main agent context
    main_agent_context = format_main_agent_context(agent)

    fix_agent = Agent(
        description="Expert at fixing graph queries that returned no results",
        instructions=f"{main_agent_context}\n\n{FIX_EMPTY_QUERIES_INSTRUCTIONS}",
        output_schema=LLMGraphQueryOperations,
        model=agent.model,
        debug_mode=debug_mode or False,
    )

    # Format failed queries for display
    failed_entity_queries: list[str] = []
    for q in failed_queries.entity_queries:
        query_desc = f"EntityQuery(type={q.entity_types}"
        if q.search_query:
            query_desc += f", search='{q.search_query}'"
        query_desc += ")"
        failed_entity_queries.append(query_desc)

    failed_rel_queries: list[str] = []
    for q in failed_queries.relationship_queries:
        query_desc = (
            f"RelQuery(type={q.relationship_types}, "
            f"source={q.source_entity_name}, target={q.target_entity_name})"
        )
        failed_rel_queries.append(query_desc)

    existing_entities = format_existing_entities_for_context(
        graph, limit=30, max_rels_per_entity=3, max_content_length=100
    )

    input_to_agent = build_fix_empty_queries_input(
        input_content, failed_entity_queries, failed_rel_queries, existing_entities
    )

    try:
        if async_hooks:
            response = fix_agent.run(input_to_agent)
        else:
            response = await fix_agent.arun(input_to_agent)

        return cast(LLMGraphQueryOperations, response.content)
    except Exception as e:
        log_warning(f"Query fix agent failed: {e}")
        return None


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


def create_pre_graph_database_hook(
    graph: GraphDatabase,
    agentic_search: bool,
    config: GraphHookIterativeConfig | None = None,
    return_sync_wrapper: bool = False,
) -> AgnoPreHook:
    """Create pre-hook for graph database operations.

    Args:
        graph: GraphDatabase instance to query
        agentic_search: Whether to enable agentic search
        config: Configuration for query generation (max queries, retries)
        return_sync_wrapper: If True, returns a sync wrapper that runs async logic synchronously

    Returns:
        Pre-hook function (async or sync wrapper based on return_sync_wrapper parameter)
    """
    if config is None:
        config = GraphHookIterativeConfig()

    async def hook_logic(
        agent: Agent | Team,
        run_input: RunInput,
        session: AgentSession | TeamSession,  # noqa: ARG001
        user_id: UserId,  # noqa: ARG001
        debug_mode: DebugMode,
    ) -> None:
        """Single-pass hook for graph database queries (max 3 queries)."""
        log_debug(
            f"Running pre-hook for graph database queries ({'sync' if return_sync_wrapper else 'async'})"
        )
        if not agentic_search:
            return

        if graph.entity_count == 0:
            log_debug("Graph is empty, skipping graph query hook")
            return

        input_content = run_input.input_content_string()
        existing_entities = format_existing_entities_for_context(
            graph, limit=20, max_rels_per_entity=3, max_content_length=50
        )

        # Get main agent context
        main_agent_context = format_main_agent_context(agent)

        # Create agent to generate graph queries (max 3)
        query_agent = create_agent_with_instructions(
            description=QUERY_GENERATION_DESCRIPTION,
            instructions=f"{main_agent_context}\n\n{get_query_generation_instructions(existing_entities, config.max_queries)}",
            expected_output=QUERY_GENERATION_EXPECTED_OUTPUT,
            output_schema=LLMGraphQueryOperations,
            model=agent.model,
            debug_mode=debug_mode or False,
        )

        # Generate queries
        queries = await _handle_initial_query_generation(query_agent, input_content)
        if queries is None:
            log_debug("No queries generated")
            return

        # Execute queries
        accumulated_context = ""
        accumulated_entity_ids: set[str] = set()
        accumulated_context, accumulated_entity_ids = _execute_and_accumulate_queries(
            queries, graph, accumulated_context, accumulated_entity_ids, iteration=1
        )

        # Self-correction: If queries returned no entities, try to fix them
        query_retry_count = 0
        while len(accumulated_entity_ids) == 0 and query_retry_count < config.max_query_retries:
            log_debug(
                f"Queries returned no entities (retry {query_retry_count + 1}/{config.max_query_retries}), "
                "asking agent to fix queries"
            )
            fixed_queries = await _fix_empty_queries_with_agent(
                input_content,
                queries,
                graph,
                agent,
                debug_mode,
                async_hooks=return_sync_wrapper,
            )

            if fixed_queries is None:
                break

            query_retry_count += 1
            queries = fixed_queries
            accumulated_context, accumulated_entity_ids = _execute_and_accumulate_queries(
                fixed_queries,
                graph,
                accumulated_context,
                accumulated_entity_ids,
                iteration=1,
            )
            if len(accumulated_entity_ids) > 0:
                log_debug(f"Fixed queries found {len(accumulated_entity_ids)} entities")

        # Inject context
        if accumulated_context:
            inject_context_to_run_input(run_input, accumulated_context)
        else:
            log_debug("No context retrieved from queries")

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


async def _validate_and_import_with_retry(
    graph: GraphDatabase,
    operations: LLMGraphOperations,
    agent: Agent | Team,
    debug_mode: DebugMode,
    async_hooks: bool = False,
    max_retries: int = 2,
) -> tuple[bool, str]:
    """Validate and import operations with retry mechanism.

    Args:
        graph: Graph database instance
        operations: Operations to import
        agent: Agent for retry feedback
        debug_mode: Debug mode flag
        async_hooks: If True, uses async execution
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (success, message)
    """
    retry_count = 0
    # Pre-filter existing entities before validation to avoid common errors
    current_operations = _filter_existing_entities(graph, operations)

    while retry_count <= max_retries:
        if retry_count > 0:
            log_debug(f"Import retry {retry_count}/{max_retries}")

        # Step 1: Validate operations
        validation_errors = _validate_operations(graph, current_operations)

        if validation_errors:
            error_message = f"Validation failed (attempt {retry_count + 1}):\n" + "\n".join(
                f"  - {err}" for err in validation_errors
            )

            if retry_count >= max_retries:
                log_warning(f"Max retries ({max_retries}) reached. Import failed:\n{error_message}")
                return False, error_message

            # Step 2: Use agent to fix errors
            try:
                log_debug("Asking agent to fix validation errors...")
                fixed_operations = await _fix_operations_with_agent(
                    current_operations,
                    validation_errors,
                    agent,
                    debug_mode,
                    async_hooks,
                )

                if not fixed_operations:
                    log_warning("Agent failed to produce fixed operations")
                    return False, error_message

                current_operations = fixed_operations
                retry_count += 1
                continue
            except Exception as fix_error:
                log_warning(f"Failed to fix operations: {fix_error}")
                return False, error_message

        # Step 3: Import operations (validation passed)
        try:
            graph.import_operations(current_operations)
            log_debug(
                f"Successfully imported {len(current_operations.add_entity_ops)} entities "
                f"and {len(current_operations.add_relationship_ops)} relationships"
            )
            return True, "Import successful"

        except ValueError as e:
            error_msg = str(e)

            if retry_count >= max_retries:
                log_warning(f"Max retries ({max_retries}) reached. Import failed:\n{error_msg}")
                return False, f"Import failed after {max_retries} retries: {error_msg}"

            # Step 4: Use agent to fix import errors
            try:
                log_debug("Asking agent to fix import errors...")
                fixed_operations = await _fix_operations_with_agent(
                    current_operations,
                    [error_msg],
                    agent,
                    debug_mode,
                    async_hooks,
                )

                if not fixed_operations:
                    log_warning("Agent failed to produce fixed operations")
                    return False, error_msg

                current_operations = fixed_operations
                retry_count += 1
            except Exception as fix_error:
                log_warning(f"Failed to fix operations: {fix_error}")
                return False, error_msg

        except Exception as e:
            log_warning(f"Unexpected import error: {e}")
            return False, f"Unexpected error during import: {e}"

    return False, "Unknown error"


def _filter_existing_entities(
    graph: GraphDatabase, operations: LLMGraphOperations
) -> LLMGraphOperations:
    """Filter out entities that already exist in the graph.

    This prevents the common validation error where the LLM tries to
    re-create entities that already exist. Relationships can still
    reference existing entities by name.

    Args:
        graph: Graph database instance
        operations: Operations that may contain existing entities

    Returns:
        Filtered operations with existing entities removed
    """
    filtered_entities: list[LLMGraphAddEntity] = []
    removed_count = 0

    for entity in operations.add_entity_ops:
        existing = graph.get_entity_by_name(entity.name)
        if existing is not None and existing.type == entity.type:
            log_debug(f"Filtering out existing entity: '{entity.name}' (type: {entity.type})")
            removed_count += 1
        else:
            filtered_entities.append(entity)

    if removed_count > 0:
        log_debug(f"Filtered {removed_count} existing entities from add_entity_ops")

    return LLMGraphOperations(
        add_entity_ops=filtered_entities,
        update_entity_ops=operations.update_entity_ops,
        delete_entity_ops=operations.delete_entity_ops,
        add_relationship_ops=operations.add_relationship_ops,
        delete_relationship_ops=operations.delete_relationship_ops,
    )


def _validate_operations(graph: GraphDatabase, operations: LLMGraphOperations) -> list[str]:
    """Validate operations before import.

    Args:
        graph: Graph database instance
        operations: Operations to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []
    entity_names: set[str] = set()
    relationship_pairs: set[tuple[str, str, str]] = set()

    # Validate entity operations
    for entity in operations.add_entity_ops:
        # Check for empty names
        if not entity.name or not entity.name.strip():
            errors.append("Entity name cannot be empty")

        # Check for duplicate entity names within batch
        if entity.name in entity_names:
            errors.append(f"Duplicate entity name: '{entity.name}'")

        # Check if entity already exists in graph with same name and type
        existing_entity = graph.get_entity_by_name(entity.name)
        if existing_entity is not None and existing_entity.type == entity.type:
            errors.append(
                f"Entity '{entity.name}' (type: {entity.type}) already exists in the graph. "
                "REMOVE this entity from add_entity_ops - relationships can still reference it."
            )
        entity_names.add(entity.name)

        # Check entity name length
        if len(entity.name) > 60:
            errors.append(
                f"Entity name exceeds 60 characters: '{entity.name}' ({len(entity.name)} chars)"
            )

        # Check for type info embedded in entity name (e.g., "mode (Person)" or "Person mode")
        entity_type_str = entity.type if isinstance(entity.type, str) else entity.type.value
        type_patterns = [
            f"({entity_type_str})",
            f" {entity_type_str}",
            f"{entity_type_str} ",
        ]
        name_lower = entity.name.lower()
        if any(pattern.lower() in name_lower for pattern in type_patterns):
            errors.append(
                f"Entity name '{entity.name}' contains its type '{entity_type_str}'. "
                "Remove the type from the name - it's redundant."
            )

        # Check for parenthetical person references like "(Mark)" at the end
        if entity.name.endswith(")") and "(" in entity.name:
            paren_content = entity.name[entity.name.rfind("(") + 1 : -1]
            if len(paren_content) < 30 and paren_content.replace(" ", "").isalpha():
                errors.append(
                    f"Entity name '{entity.name}' has parenthetical suffix '({paren_content})'. "
                    "Use relationships instead of embedding references in names."
                )

        # Check for non-ASCII characters (emojis, special chars)
        if any(ord(char) > 127 for char in entity.name):
            errors.append(
                f"Entity name contains non-ASCII characters: '{entity.name}'. Use plain text only."
            )

        # Check for markdown in content
        if entity.content and (
            "**" in entity.content or "##" in entity.content or "```" in entity.content
        ):
            errors.append(
                f"Entity content for '{entity.name}' contains markdown. Use plain text only."
            )

        # Check content length
        if entity.content and len(entity.content) > 400:
            errors.append(
                f"Entity content exceeds 400 characters for '{entity.name}' ({len(entity.content)} chars)"
            )

        # Validate entity type
        if entity.type not in ENTITY_TYPE_DEFINITIONS:
            errors.append(f"Invalid entity type: '{entity.type}'")

    # Validate relationship operations
    for rel in operations.add_relationship_ops:
        # Validate relationship type
        if rel.type not in RELATIONSHIP_TYPE_DEFINITIONS:
            errors.append(f"Invalid relationship type: '{rel.type}'")

        # Check for self-referencing
        if rel.source_name == rel.target_name:
            errors.append(
                f"Self-referencing relationship: '{rel.source_name}' -> '{rel.target_name}'"
            )

        # Check for duplicate relationships
        rel_key = (rel.source_name, rel.target_name, rel.type)
        if rel_key in relationship_pairs:
            errors.append(
                f"Duplicate relationship: '{rel.source_name}' -> '{rel.target_name}' ({rel.type})"
            )
        relationship_pairs.add(rel_key)

        # Check that entities exist
        source_exists = (
            rel.source_name in entity_names or graph.get_entity_by_name(rel.source_name) is not None
        )
        target_exists = (
            rel.target_name in entity_names or graph.get_entity_by_name(rel.target_name) is not None
        )

        if not source_exists:
            errors.append(
                f"Relationship source entity not found: '{rel.source_name}'. "
                "Create this entity first."
            )

        if not target_exists:
            errors.append(
                f"Relationship target entity not found: '{rel.target_name}'. "
                "Create this entity first."
            )

    return errors


async def _fix_operations_with_agent(
    operations: LLMGraphOperations,
    errors: list[str],
    agent: Agent | Team,
    debug_mode: DebugMode,
    async_hooks: bool = False,
) -> LLMGraphOperations | None:
    """Use agent to fix validation errors in operations.

    Args:
        operations: Operations with errors
        errors: List of validation error messages
        agent: Agent to use for fixing
        debug_mode: Debug mode flag
        async_hooks: If True, uses async execution

    Returns:
        Fixed LLMGraphOperations or None if fixing failed
    """
    # Get main agent context
    main_agent_context = format_main_agent_context(agent)

    fix_agent = Agent(
        description="Expert at fixing graph operation validation errors",
        instructions=f"{main_agent_context}\n\n{FIX_OPERATIONS_INSTRUCTIONS}",
        output_schema=LLMGraphOperations,
        model=agent.model,
        debug_mode=debug_mode or False,
    )

    valid_entity_types_list = sorted(ENTITY_TYPE_DEFINITIONS.keys())
    valid_entity_types_str = ", ".join(valid_entity_types_list)

    valid_relationship_types_list = sorted(RELATIONSHIP_TYPE_DEFINITIONS.keys())
    valid_relationship_types_str = ", ".join(valid_relationship_types_list)

    operations_text = format_entities_and_relationships_text(operations)
    input_to_agent = build_fix_operations_input(
        operations_text, errors, valid_entity_types_str, valid_relationship_types_str
    )

    try:
        if async_hooks:
            response = fix_agent.run(input_to_agent)
        else:
            response = await fix_agent.arun(input_to_agent)

        return cast(LLMGraphOperations, response.content)
    except Exception as e:
        log_warning(f"Fix agent failed: {e}")
        return None


def format_entities_and_relationships_text(
    operations: LLMGraphOperations,
) -> str:
    """Format operations as readable text for agent feedback.

    Args:
        operations: Operations to format

    Returns:
        Formatted text representation
    """
    lines = ["<entities>"]
    for entity in operations.add_entity_ops:
        lines.append(
            f"  - {entity.name} (type: {entity.type}, content: {entity.content or 'None'})"
        )
    lines.append("</entities>")

    lines.append("<relationships>")
    for rel in operations.add_relationship_ops:
        lines.append(f"  - {rel.source_name} -> {rel.target_name} (type: {rel.type})")
    lines.append("</relationships>")

    return "\n".join(lines)


def create_post_graph_database_hook(
    graph: GraphDatabase,
    file_path: str | None = None,
    agentic_ingestion: bool = True,
    ingestion_config: GraphIngestionIterativeConfig | None = None,
    return_sync_wrapper: bool = False,
) -> AgnoPostHook:
    """Create post-hook for graph database operations.

    Uses a single agent call that extracts entities/relationships AND
    self-assesses quality with deduplication suggestions.
    """
    if ingestion_config is None:
        ingestion_config = GraphIngestionIterativeConfig()

    async def hook_logic(
        agent: Agent | Team,
        run_output: RunOutput | TeamRunOutput,
        session: AgentSession | TeamSession,  # noqa: ARG001
        run_context: RunContext,  # noqa: ARG001
        user_id: UserId,  # noqa: ARG001
        debug_mode: DebugMode,
    ) -> None:
        """Single-agent extraction with self-assessment and deduplication."""
        if not agentic_ingestion:
            return

        # Build extraction context
        full_context = build_extraction_context(run_output.input, run_output)
        if not full_context:
            return

        log_debug("Extracting graph operations (single-agent with self-assessment)")

        existing_entities_context = format_existing_entities_for_context(
            graph, limit=30, max_rels_per_entity=5, max_content_length=100
        )

        # Get main agent context
        main_agent_context = format_main_agent_context(agent)

        # Single extraction agent with self-assessment
        extraction_prompt = get_extraction_prompt(existing_entities_context)
        extraction_agent = create_agent_with_instructions(
            description="Knowledge extractor with self-assessment.",
            instructions=f"{main_agent_context}\n\n{extraction_prompt}",
            expected_output="Extract entities/relationships and self-assess quality.",
            output_schema=LLMGraphOperations,
            model=agent.model,
            debug_mode=debug_mode or False,
        )

        try:
            log_debug(f"Running {'sync' if return_sync_wrapper else 'async'} extraction agent")

            if return_sync_wrapper:
                response = execute_agent_sync(extraction_agent, full_context)
            else:
                response = await execute_agent_async(extraction_agent, full_context)

            operations = cast(LLMGraphOperations, response.content)

            # Validate and import with retry
            try:
                success, message = await _validate_and_import_with_retry(
                    graph,
                    operations,
                    agent,
                    debug_mode,
                    return_sync_wrapper,
                    max_retries=ingestion_config.max_import_retries,
                )

                if success:
                    log_debug(
                        f"Graph post-hook completed. Imported {len(operations.add_entity_ops)} entities"
                    )
                else:
                    log_warning(f"Graph post-hook import failed after retries: {message}")
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
