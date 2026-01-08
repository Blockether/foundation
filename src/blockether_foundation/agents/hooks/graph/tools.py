from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from agno.exceptions import RetryAgentRun
from agno.run import RunContext
from agno.utils.log import log_debug, log_warning  # type: ignore

from ....graph.common import ENTITY_TYPE_DEFINITIONS, RELATIONSHIP_TYPE_DEFINITIONS
from ....graph.models import LLMGraphOperations

if TYPE_CHECKING:
    from ....graph.database import GraphDatabase


def create_import_tool(graph: GraphDatabase) -> Callable[[RunContext, LLMGraphOperations], str]:
    def import_to_graph(run_context: RunContext, operations: LLMGraphOperations) -> str:
        """Import extracted entities and relationships into the graph database.

        Call this tool ONCE after you have extracted all entities and relationships.
        If import fails, you will receive error details to fix your extraction.
        """
        if not operations.add_entity_ops and not operations.add_relationship_ops:
            return "No operations to import. Extraction complete."

        filtered_ops = _filter_existing_entities(graph, operations)

        errors = _validate_operations(graph, filtered_ops)
        if errors:
            error_msg = _format_validation_errors(errors)
            log_debug(f"Validation failed, asking agent to retry: {error_msg}")
            raise RetryAgentRun(error_msg)

        try:
            graph.import_operations(filtered_ops)
            entity_count = len(filtered_ops.add_entity_ops)
            rel_count = len(filtered_ops.add_relationship_ops)
            log_debug(f"Imported {entity_count} entities and {rel_count} relationships")
            return f"Successfully imported {entity_count} entities and {rel_count} relationships."
        except ValueError as e:
            log_warning(f"Import failed: {e}")
            raise RetryAgentRun(
                f"Import failed: {e}\n\nFix the operations and call import_to_graph again."
            ) from e

    return import_to_graph


def _filter_existing_entities(
    graph: GraphDatabase, operations: LLMGraphOperations
) -> LLMGraphOperations:
    from ....graph.models import LLMGraphAddEntity

    filtered: list[LLMGraphAddEntity] = []
    for entity in operations.add_entity_ops:
        existing = graph.get_entity_by_name(entity.name)
        if existing is None or existing.type != entity.type:
            filtered.append(entity)

    return LLMGraphOperations(
        add_entity_ops=filtered,
        update_entity_ops=operations.update_entity_ops,
        delete_entity_ops=operations.delete_entity_ops,
        add_relationship_ops=operations.add_relationship_ops,
        delete_relationship_ops=operations.delete_relationship_ops,
    )


def _validate_operations(graph: GraphDatabase, operations: LLMGraphOperations) -> list[str]:
    errors: list[str] = []
    entity_names: set[str] = set()
    rel_pairs: set[tuple[str, str, str]] = set()

    for e in operations.add_entity_ops:
        if not e.name or not e.name.strip():
            errors.append("Entity name cannot be empty")
        if e.name in entity_names:
            errors.append(f"Duplicate entity: '{e.name}'")
        if graph.get_entity_by_name(e.name) is not None:
            errors.append(f"Entity '{e.name}' already exists - remove from add_entity_ops")
        entity_names.add(e.name)

        if len(e.name) > 60:
            errors.append(f"Entity name too long (>60): '{e.name[:30]}...'")
        if any(ord(c) > 127 for c in e.name):
            errors.append(f"Entity name has non-ASCII chars: '{e.name}'")
        if e.content and len(e.content) > 400:
            errors.append(f"Entity content too long (>400) for '{e.name}'")
        if e.type not in ENTITY_TYPE_DEFINITIONS:
            errors.append(f"Invalid entity type: '{e.type}'")

    for r in operations.add_relationship_ops:
        if r.type not in RELATIONSHIP_TYPE_DEFINITIONS:
            errors.append(f"Invalid relationship type: '{r.type}'")
        if r.source_name == r.target_name:
            errors.append(f"Self-reference: '{r.source_name}' -> '{r.target_name}'")

        key = (r.source_name, r.target_name, r.type)
        if key in rel_pairs:
            errors.append(f"Duplicate relationship: {key}")
        rel_pairs.add(key)

        src_exists = r.source_name in entity_names or graph.get_entity_by_name(r.source_name)
        tgt_exists = r.target_name in entity_names or graph.get_entity_by_name(r.target_name)
        if not src_exists:
            errors.append(f"Source entity not found: '{r.source_name}'")
        if not tgt_exists:
            errors.append(f"Target entity not found: '{r.target_name}'")

    return errors


def _format_validation_errors(errors: list[str]) -> str:
    valid_entities = ", ".join(sorted(ENTITY_TYPE_DEFINITIONS.keys()))
    valid_rels = ", ".join(sorted(RELATIONSHIP_TYPE_DEFINITIONS.keys()))

    return f"""Validation errors - fix and retry:
{chr(10).join(f"- {e}" for e in errors)}

Valid entity types: {valid_entities}
Valid relationship types: {valid_rels}

Fix these issues and call import_to_graph again with corrected operations."""
