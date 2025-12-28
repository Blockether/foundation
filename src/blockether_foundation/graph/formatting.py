"""Query formatting utilities for graph database results."""

from __future__ import annotations

from typing import cast
from xml.sax.saxutils import escape, quoteattr

from .database import EntityQuery, GraphDatabase, RelationshipQuery
from .models import Entity, Relationship


def _represent_query(query: EntityQuery | RelationshipQuery) -> str:
    """Create a string representation of a query for XML output.

    Args:
        query: EntityQuery or RelationshipQuery to represent

    Returns:
        String representation of the query
    """
    if isinstance(query, EntityQuery):
        parts = ["EntityQuery:"]

        # Add filters
        for filter_obj in query.filters:
            filter_repr = str(filter_obj)
            parts.append(f"  Filter: {filter_repr}")

        # Add sort info
        if query.sort_field:
            direction = "ASC" if query.sort_ascending else "DESC"
            parts.append(f"  Sort: {query.sort_field} {direction}")

        # Add limit
        if query.limit_count:
            parts.append(f"  Limit: {query.limit_count}")

    elif isinstance(query, RelationshipQuery):
        parts = ["RelationshipQuery:"]

        # Add filters
        for filter_obj in query.filters:
            filter_repr = str(filter_obj)
            parts.append(f"  Filter: {filter_repr}")

        # Add sort info
        if query.sort_field:
            direction = "ASC" if query.sort_ascending else "DESC"
            parts.append(f"  Sort: {query.sort_field} {direction}")

        # Add limit
        if query.limit_count:
            parts.append(f"  Limit: {query.limit_count}")
    else:
        parts = [f"UnknownQuery: {type(query)}"]

    return "\n".join(parts)


def format_graph_query_results(
    query_builders: list[EntityQuery | RelationshipQuery],
    *,
    max_content_length: int | None = None,
    max_results_per_query: int = 10,
    iteration: int = 1,
) -> str:
    """Format graph query results into structured XML format.

    Args:
        query_builders: List of EntityQuery and/or RelationshipQuery objects to execute
        max_content_length: Maximum length of entity content before truncation. None for no truncation.
        max_results_per_query: Maximum number of results to include per query
        iteration: Iteration number for tracking multi-pass queries (default: 1)

    Returns:
        Formatted XML string with the query results
    """
    xml_parts: list[str] = []
    xml_parts.append(f"<!-- Iteration {iteration} -->")
    xml_parts.append("<graph_knowledge>")

    for i, query in enumerate(query_builders):
        xml_parts.append(f'  <query_with_results index="{i + 1}">')

        # Add query representation
        query_repr = _represent_query(query)
        xml_parts.append("    <query>")
        xml_parts.append(f"      {escape(query_repr)}")
        xml_parts.append("    </query>")

        # Execute query and get results
        result = query.execute()

        xml_parts.append("    <results>")

        if result.results:
            # Check if it's an Entity query result
            first_result = result.results[0]
            if (
                isinstance(first_result, dict) and "name" in first_result and "type" in first_result
            ) or isinstance(first_result, Entity):
                # Track entities to avoid duplicates
                seen_entities: set[str] = set()

                for entity in result.results[:max_results_per_query]:
                    # Cast to Entity for type checker
                    entity_typed = cast("Entity", entity)
                    entity_key = f"{entity_typed.name}:{entity_typed.type}"

                    if entity_key not in seen_entities:
                        seen_entities.add(entity_key)

                        # Build attributes string
                        attrs = f"id={quoteattr(str(entity_typed.id))} name={quoteattr(entity_typed.name)} type={quoteattr(entity_typed.type)}"

                        xml_parts.append(f"      <entity {attrs}>")

                        # Add content if present, with optional truncation
                        if entity_typed.content:
                            content = entity_typed.content
                            if max_content_length is not None and len(content) > max_content_length:
                                content = content[:max_content_length] + "..."
                            xml_parts.append(f"        <content>{escape(content)}</content>")

                        xml_parts.append("      </entity>")

            # Check if it's a Relationship query result
            elif isinstance(result.results[0], tuple) and len(result.results[0]) == 3:
                seen_entities: set[str] = set()

                for rel_tuple in result.results[:max_results_per_query]:
                    # Type the tuple properly
                    source, target, relationship = cast(
                        "tuple[Entity, Entity, Relationship]", rel_tuple
                    )
                    source_key = f"{source.name}:{source.type}"
                    target_key = f"{target.name}:{target.type}"

                    # Ensure both entities are included and deduplicated
                    for entity, key in [(source, source_key), (target, target_key)]:
                        if key not in seen_entities:
                            seen_entities.add(key)

                            # Build attributes string
                            attrs = f"id={quoteattr(str(entity.id))} name={quoteattr(entity.name)} type={quoteattr(entity.type)}"

                            xml_parts.append(f"      <entity {attrs}>")

                            # Add content if present, with optional truncation
                            if entity.content:
                                content = entity.content
                                if (
                                    max_content_length is not None
                                    and len(content) > max_content_length
                                ):
                                    content = content[:max_content_length] + "..."
                                xml_parts.append(f"        <content>{escape(content)}</content>")

                            xml_parts.append("      </entity>")

        xml_parts.append("    </results>")
        xml_parts.append("  </query_with_results>")

    xml_parts.append("</graph_knowledge>")
    return "\n".join(xml_parts)


def format_existing_entities_for_context(
    graph: GraphDatabase,
    *,
    limit: int = 20,
    max_rels_per_entity: int = 10,
    max_content_length: int | None = 200,
) -> str:
    """Format existing entities with relationships as XML for LLM context.

    This method replaces the plain text format with structured XML that matches
    the same structure as query results, making it consistent for LLM consumption.

    Args:
        graph: GraphDatabase instance to query
        limit: Maximum number of entities to include
        max_rels_per_entity: Maximum relationships to show per entity
        max_content_length: Maximum length of entity content before truncation

    Returns:
        Formatted XML string with existing entities and relationships
    """
    xml_parts: list[str] = []
    xml_parts.append("<existing_entities>")

    most_connected = graph.get_most_connected_entities(limit)

    for entity, degree in most_connected:
        # Build attributes string
        attrs = f"id={quoteattr(str(entity.id))} name={quoteattr(entity.name)} type={quoteattr(entity.type)} connections={quoteattr(str(degree))}"

        xml_parts.append(f"  <entity {attrs}>")

        # Add content if present, with optional truncation
        if entity.content:
            content = entity.content
            if max_content_length is not None and len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            xml_parts.append(f"    <content>{escape(content)}</content>")

        # Add relationships
        xml_parts.append("    <rels>")

        # Get relationships for this entity
        relationships = graph.get_entity_relationships(entity.id)
        rels_to_show = min(len(relationships), max_rels_per_entity)

        # Sort relationships (outgoing first, then incoming)
        outgoing_first = sorted(
            relationships,
            key=lambda r: (
                0 if r.source == entity.id else 1,  # Outgoing first
                r.type,  # Then by type for consistency
            ),
        )[:rels_to_show]

        # Get the index once for efficiency
        idx = graph.index

        for rel in outgoing_first:
            if rel.source == entity.id:
                # Outgoing relationship - show target
                target_entity = idx.entity_by_id.get(rel.target)
                if target_entity:
                    xml_parts.append(
                        f'      <rel type={quoteattr(rel.type)} dir="out" to_id={quoteattr(target_entity.id)} to_name={quoteattr(target_entity.name)} to_type={quoteattr(target_entity.type)} />'
                    )
            else:
                # Incoming relationship - show source
                source_entity = idx.entity_by_id.get(rel.source)
                if source_entity:
                    xml_parts.append(
                        f'      <rel type={quoteattr(rel.type)} dir="in" from_id={quoteattr(source_entity.id)} from_name={quoteattr(source_entity.name)} from_type={quoteattr(source_entity.type)} />'
                    )

        xml_parts.append("    </rels>")
        xml_parts.append("  </entity>")

    xml_parts.append("</existing_entities>")
    return "\n".join(xml_parts)


def format_entities_with_relationships_for_prompt(
    entity_ids: set[str],
    graph: GraphDatabase,
) -> str:
    """Format entities with their relationships for the LLM prompt in XML format.

    This function formats specific entities (by ID) with all their relationships
    for inclusion in LLM prompts. Unlike format_existing_entities_for_context,
    this doesn't limit the number of entities and uses the exact entity IDs provided.

    Args:
        entity_ids: Set of entity IDs to format
        graph: GraphDatabase instance to query

    Returns:
        Formatted XML string with entities and their relationships
    """
    if not entity_ids:
        return '<already_retrieved_entities count="0"></already_retrieved_entities>'

    xml_parts: list[str] = []
    xml_parts.append(f'<already_retrieved_entities count="{len(entity_ids)}">')

    # Get the index once for efficiency
    idx = graph.index

    for entity_id in sorted(entity_ids):
        entity = graph.get_entity(entity_id)
        if entity:
            # Build entity attributes
            attrs = f"id={quoteattr(str(entity.id))} name={quoteattr(entity.name)} type={quoteattr(entity.type)}"
            xml_parts.append(f"  <entity {attrs}>")

            # Add relationships
            xml_parts.append("    <rels>")
            relationships = graph.get_entity_relationships(entity_id)

            # Sort relationships (outgoing first, then incoming)
            outgoing_first = sorted(
                relationships,
                key=lambda r: (
                    0 if entity and r.source == entity.id else 1,  # Outgoing first
                    r.type,  # Then by type for consistency
                ),
            )

            for rel in outgoing_first:
                if rel.source == entity.id:
                    # Outgoing relationship - show target
                    target_entity = idx.entity_by_id.get(rel.target)
                    if target_entity:
                        xml_parts.append(
                            f'      <rel type={quoteattr(rel.type)} dir="out" to_id={quoteattr(target_entity.id)} to_name={quoteattr(target_entity.name)} to_type={quoteattr(target_entity.type)} />'
                        )
                else:
                    # Incoming relationship - show source
                    source_entity = idx.entity_by_id.get(rel.source)
                    if source_entity:
                        xml_parts.append(
                            f'      <rel type={quoteattr(rel.type)} dir="in" from_id={quoteattr(source_entity.id)} from_name={quoteattr(source_entity.name)} from_type={quoteattr(source_entity.type)} />'
                        )

            xml_parts.append("    </rels>")
            xml_parts.append("  </entity>")
        else:
            xml_parts.append(
                f'  <entity id={quoteattr(entity_id)} type="unknown" name="[Entity not found]"/>'
            )

    xml_parts.append("</already_retrieved_entities>")
    return "\n".join(xml_parts)
