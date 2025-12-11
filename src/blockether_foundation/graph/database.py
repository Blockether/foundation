"""Main graph database implementation with CRUD and query operations."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, TypeVar, cast, override

import tantivy

from blockether_foundation.graph.indices import GraphIndex
from blockether_foundation.graph.models import (
    Entity,
    EntityType,
    LLMEntityQuery,
    LLMGraphOperations,
    LLMGraphQueryOperations,
    LLMRelationshipQuery,
    Relationship,
    RelationType,
)

# Constants
MIN_WORD_LENGTH_FOR_INDEX = 2
DEFAULT_SELECTIVITY = 0.1
RELATIONSHIP_FILTER_SELECTIVITY = 0.05
DATE_FILTER_SELECTIVITY = 0.15
DEFAULT_MAX_DEPTH = 10
DEFAULT_TOP_K = 10

# Direction constants for neighbor queries
DIRECTION_OUTGOING = "outgoing"
DIRECTION_INCOMING = "incoming"
DIRECTION_BOTH = "both"

# Sort field constants
SORT_FIELD_NAME = "name"
SORT_FIELD_CREATED_AT = "created_at"
SORT_FIELD_UPDATED_AT = "updated_at"

# Value constants
ZERO_SELECTIVITY = 0.0
MIN_DIVISOR = 1

T = TypeVar("T", Entity, Relationship)


@dataclass
class FilterStats:
    """Statistics for a single filter."""

    name: str
    selectivity: float
    matched_count: int


@dataclass
class QueryStats:
    """Query execution statistics."""

    filters: list[FilterStats]
    total_results: int
    execution_time_ms: float


@dataclass
class QueryResult[T]:
    """Generic query result with statistics."""

    results: list[T]
    stats: QueryStats


class GraphDatabase:
    """In-memory graph database with Tantivy search."""

    def __init__(self) -> None:
        """Initialize graph database."""
        self._index: GraphIndex = GraphIndex()
        self._tantivy_index: tantivy.Index | None = None  # Lazy-loaded Tantivy index
        self._tantivy_searcher: tantivy.Searcher | None = None  # Cache searcher

    @property
    def index(self) -> GraphIndex:
        """Get the graph index."""
        return self._index

    def add_entity(self, entity: Entity) -> None:
        """Add entity to the graph.

        Args:
            entity: Entity to add.

        Raises:
            ValueError: If entity with this ID already exists.
        """
        # Use the efficient batch method for single entity
        self.add_entities([entity])

    def add_entities(self, entities: list[Entity]) -> None:
        """Add multiple entities to the graph efficiently.

        Args:
            entities: List of entities to add.

        Raises:
            ValueError: If any entity ID already exists.
        """
        # Check for ID duplicates first
        existing_ids: list[str] = [e.id for e in entities if e.id in self._index.entity_by_id]
        if existing_ids:
            raise ValueError(f"Entities with IDs already exist: {existing_ids}")

        # Check for name uniqueness within entity type
        name_conflicts: list[str] = []
        for entity in entities:
            existing_name_entity_id = self._index.entity_by_name.get(entity.name)
            if existing_name_entity_id and existing_name_entity_id != entity.id:
                existing_entity = self._index.entity_by_id.get(existing_name_entity_id)
                if existing_entity and existing_entity.type == entity.type:
                    name_conflicts.append(f"'{entity.name}' (type: {entity.type})")

        if name_conflicts:
            raise ValueError(
                f"Entity names must be unique within type. Conflicts: {name_conflicts}"
            )

        # Add all entities to graph index
        for entity in entities:
            self._index.add_entity(entity)

        # Batch add to Tantivy index for efficiency
        self._incremental_add_entities_to_tantivy(entities)

    def update_entity(self, entity: Entity) -> None:
        """Update existing entity.

        Args:
            entity: Entity with updated data.

        Raises:
            ValueError: If entity does not exist.
        """
        if entity.id not in self._index.entity_by_id:
            raise ValueError(f"Entity with id '{entity.id}' does not exist")

        entity.updated_at = datetime.now(UTC)
        old_entity = self._index.entity_by_id[entity.id]
        self._index.update_entity(old_entity, entity)

        # Incremental Tantivy update - update single entity in search index
        self._incremental_update_entity_in_tantivy(entity.id, entity)

    def delete_entity(self, entity_id: str) -> None:
        """Delete entity from the graph.

        Args:
            entity_id: ID of entity to delete.

        Raises:
            ValueError: If entity does not exist.
        """
        if entity_id not in self._index.entity_by_id:
            raise ValueError(f"Entity with id '{entity_id}' does not exist")

        entity: Entity = self._index.entity_by_id[entity_id]

        outgoing_rel_ids: list[str] = list(self._index.outgoing_edges.get(entity_id, set()))
        incoming_rel_ids: list[str] = list(self._index.incoming_edges.get(entity_id, set()))

        for rel_id in outgoing_rel_ids + incoming_rel_ids:
            if rel_id in self._index.relationship_by_id:
                self.delete_relationship(rel_id)

        self._index.remove_entity(entity)

        # Incremental Tantivy update - remove single entity from search index
        self._incremental_remove_entity_from_tantivy(entity_id)

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get entity by ID.

        Args:
            entity_id: Entity ID.

        Returns:
            Entity if found, None otherwise.
        """
        return self._index.entity_by_id.get(entity_id)

    def get_entity_by_name(self, name: str, entity_type: EntityType | None = None) -> Entity | None:
        """Get entity by name and optionally type.

        Args:
            name: Entity name.
            entity_type: Optional entity type to filter by.

        Returns:
            Entity if found, None otherwise.
        """
        entity_id: str | None = self._index.entity_by_name.get(name)
        if entity_id:
            entity: Entity | None = self._index.entity_by_id.get(entity_id)
            if entity and (entity_type is None or entity.type == entity_type):
                return entity
        return None

    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to the graph.

        Args:
            relationship: Relationship to add.

        Raises:
            ValueError: If relationship ID exists or source/target entities don't exist.
        """
        if relationship.id in self._index.relationship_by_id:
            raise ValueError(f"Relationship with id '{relationship.id}' already exists")
        if relationship.source not in self._index.entity_by_id:
            raise ValueError(f"Source entity '{relationship.source}' does not exist")
        if relationship.target not in self._index.entity_by_id:
            raise ValueError(f"Target entity '{relationship.target}' does not exist")

        self._index.add_relationship(relationship)

    def create_relationship(
        self, source: str, target: str, relationship_type: RelationType
    ) -> Relationship:
        """Create and add a relationship without needing to specify an ID.

        Args:
            source: Source entity ID.
            target: Target entity ID.
            relationship_type: Type of relationship.

        Returns:
            The created relationship.

        Raises:
            ValueError: If source/target entities don't exist or relationship already exists.
        """
        if source not in self._index.entity_by_id:
            raise ValueError(f"Source entity '{source}' does not exist")
        if target not in self._index.entity_by_id:
            raise ValueError(f"Target entity '{target}' does not exist")

        # Auto-generate ID and create relationship
        relationship_id = f"{source}_{target}"
        if relationship_id in self._index.relationship_by_id:
            raise ValueError(f"Relationship between '{source}' and '{target}' already exists")

        relationship = Relationship(
            source=source,
            target=target,
            type=relationship_type,
        )
        # The ID will be auto-generated in __post_init__

        self._index.add_relationship(relationship)
        return relationship

    def update_relationship(self, relationship: Relationship) -> None:
        """Update existing relationship.

        Args:
            relationship: Relationship with updated data.

        Raises:
            ValueError: If relationship does not exist.
        """
        if relationship.id not in self._index.relationship_by_id:
            raise ValueError(f"Relationship with id '{relationship.id}' does not exist")

        relationship.updated_at = datetime.now(UTC)
        old_relationship = self._index.relationship_by_id[relationship.id]
        self._index.update_relationship(old_relationship, relationship)

    def delete_relationship(self, relationship_id: str) -> None:
        """Delete relationship from the graph.

        Args:
            relationship_id: ID of relationship to delete.

        Raises:
            ValueError: If relationship does not exist.
        """
        if relationship_id not in self._index.relationship_by_id:
            raise ValueError(f"Relationship with id '{relationship_id}' does not exist")

        relationship = self._index.relationship_by_id[relationship_id]
        self._index.remove_relationship(relationship)

    def get_relationship(self, relationship_id: str) -> Relationship | None:
        """Get relationship by ID.

        Args:
            relationship_id: Relationship ID.

        Returns:
            Relationship if found, None otherwise.
        """
        return self._index.relationship_by_id.get(relationship_id)

    def get_relationship_by_entities(self, source: str, target: str) -> Relationship | None:
        """Get relationship between two entities.

        Args:
            source: Source entity ID.
            target: Target entity ID.

        Returns:
            Relationship if found, None otherwise.
        """
        relationship_id = f"{source}_{target}"
        return self._index.relationship_by_id.get(relationship_id)

    def delete_relationship_by_entities(self, source: str, target: str) -> None:
        """Delete relationship between two entities.

        Args:
            source: Source entity ID.
            target: Target entity ID.

        Raises:
            ValueError: If relationship does not exist.
        """
        relationship_id = f"{source}_{target}"
        if relationship_id not in self._index.relationship_by_id:
            raise ValueError(f"Relationship between '{source}' and '{target}' does not exist")

        relationship = self._index.relationship_by_id[relationship_id]
        self._index.remove_relationship(relationship)

    def import_operations(self, operations: LLMGraphOperations) -> None:
        """Import and execute LLM graph operations on the database.

        Executes operations in the correct order to maintain database integrity:
        1. Delete relationships (remove edges before nodes)
        2. Delete entities (remove nodes)
        3. Add entities (create new nodes)
        4. Update entities (modify existing nodes)
        5. Add relationships (create edges after nodes exist)

        Args:
            operations: LLMGraphOperations instance containing operations to execute.

        Raises:
            ValueError: If entities referenced in operations don't exist.
        """
        # Validate operations type
        if not isinstance(operations, LLMGraphOperations):
            raise TypeError(f"Expected LLMGraphOperations, got {type(operations).__name__}")

        # Trigger normalization by accessing ops (handles duplicate edges).
        _ = operations.ops

        # Step 1: Delete relationships first (edges before nodes)
        delete_relationships = operations.delete_relationship_ops
        for delete_op in delete_relationships:
            self.delete_relationship(delete_op.id)

        # Step 2: Delete entities
        delete_entities = operations.delete_entity_ops
        for delete_entity_op in delete_entities:
            self.delete_entity(delete_entity_op.entity_id)

        # Step 3: Add new entities
        add_entities = operations.add_entity_ops
        for add_entity_op in add_entities:
            entity = Entity(
                name=add_entity_op.name,
                type=add_entity_op.type,
                content=add_entity_op.content,
            )
            self.add_entity(entity)

        # Step 4: Update existing entities
        update_entities = operations.update_entity_ops
        for update_entity_op in update_entities:
            existing_entity = self.get_entity(update_entity_op.id)
            if not existing_entity:
                raise ValueError(f"Entity with id '{update_entity_op.id}' does not exist")

            updated_entity = Entity(
                id=existing_entity.id,
                name=update_entity_op.name,
                type=update_entity_op.type,
                content=update_entity_op.content,
                created_at=existing_entity.created_at,
            )
            self.update_entity(updated_entity)

        # Step 5: Add relationships (after entities exist)
        add_relationships = operations.add_relationship_ops
        for add_relationship_op in add_relationships:
            # Look up source and target entities by name
            source_entity = self.get_entity_by_name(add_relationship_op.source_name)
            if not source_entity:
                raise ValueError(
                    f"Source entity with name '{add_relationship_op.source_name}' not found"
                )

            target_entity = self.get_entity_by_name(add_relationship_op.target_name)
            if not target_entity:
                raise ValueError(
                    f"Target entity with name '{add_relationship_op.target_name}' not found"
                )

            self.create_relationship(
                source=source_entity.id,
                target=target_entity.id,
                relationship_type=add_relationship_op.type,
            )

    def find_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Find all entities of a given type.

        Args:
            entity_type: Type of entities to find.

        Returns:
            List of entities matching the type.
        """
        entity_ids = self._index.entities_by_type.get(entity_type, set())
        return [self._index.entity_by_id[eid] for eid in entity_ids]

    def find_relationships_by_type(self, relationship_type: RelationType) -> list[Relationship]:
        """Find all relationships of a given type.

        Args:
            relationship_type: Type of relationships to find.

        Returns:
            List of relationships matching the type.
        """
        rel_ids = self._index.relationships_by_type.get(relationship_type, set())
        return [self._index.relationship_by_id[rid] for rid in rel_ids]

    def find_entities_by_name_pattern(self, pattern: str, exact: bool = True) -> list[Entity]:
        """Find entities by name pattern using prefix index.

        Args:
            pattern: Name pattern to match.
            exact: If True, exact match; otherwise prefix match.

        Returns:
            List of entities matching the name pattern.
        """
        pattern_lower = pattern.lower()

        if exact:
            # Exact match - use name mapping
            entity_id = self._index.entity_by_name.get(pattern)
            if entity_id:
                entity = self._index.entity_by_id.get(entity_id)
                return [entity] if entity else []
            return []
        else:
            # Prefix match - use name prefix index
            matching_ids: set[str] = set()
            for prefix, entity_ids in self._index.entities_by_name_prefix.items():
                if prefix.startswith(pattern_lower):
                    matching_ids.update(entity_ids)

            return [
                self._index.entity_by_id[eid]
                for eid in matching_ids
                if eid in self._index.entity_by_id
            ]

    def find_entities_by_timerange(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        use_created: bool = True,
    ) -> list[Entity]:
        """Find entities created or updated within a time range.

        Args:
            start_date: Start of time range (inclusive).
            end_date: End of time range (inclusive).
            use_created: If True, filter by created_at; otherwise by updated_at.

        Returns:
            List of entities in the time range.
        """
        index_to_use = (
            self._index.entities_by_created_date
            if use_created
            else self._index.entities_by_updated_date
        )

        matching_entity_ids: set[str] = set()

        for query_date, entity_ids in index_to_use.items():
            if start_date and query_date < start_date:
                continue
            if end_date and query_date > end_date:
                continue
            matching_entity_ids.update(entity_ids)

        return [self._index.entity_by_id[eid] for eid in matching_entity_ids]

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = DIRECTION_BOTH,
        relationship_type: RelationType | None = None,
    ) -> list[Entity]:
        """Get neighboring entities via relationships.

        Args:
            entity_id: Source entity ID.
            direction: 'outgoing', 'incoming', or 'both'.
            relationship_type: Optional filter by relationship type.

        Returns:
            List of neighboring entities.

        Raises:
            ValueError: If entity does not exist or invalid direction.
        """
        if entity_id not in self._index.entity_by_id:
            raise ValueError(f"Entity with id '{entity_id}' does not exist")
        if direction not in (DIRECTION_OUTGOING, DIRECTION_INCOMING, DIRECTION_BOTH):
            raise ValueError(f"Invalid direction: {direction}")

        neighbor_ids: set[str] = set()

        if direction in (DIRECTION_OUTGOING, DIRECTION_BOTH):
            for rel_id in self._index.outgoing_edges.get(entity_id, set()):
                rel: Relationship = self._index.relationship_by_id[rel_id]
                if relationship_type is None or rel.type == relationship_type:
                    neighbor_ids.add(rel.target)

        if direction in (DIRECTION_INCOMING, DIRECTION_BOTH):
            for rel_id in self._index.incoming_edges.get(entity_id, set()):
                rel: Relationship = self._index.relationship_by_id[rel_id]
                if relationship_type is None or rel.type == relationship_type:
                    neighbor_ids.add(rel.source)

        return [self._index.entity_by_id[nid] for nid in neighbor_ids]

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = DEFAULT_MAX_DEPTH
    ) -> list[str] | None:
        """Find shortest path between two entities using BFS.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            max_depth: Maximum path length to search.

        Returns:
            List of entity IDs representing the path, or None if no path found.

        Raises:
            ValueError: If source or target entity does not exist.
        """
        if source_id not in self._index.entity_by_id:
            raise ValueError(f"Source entity '{source_id}' does not exist")
        if target_id not in self._index.entity_by_id:
            raise ValueError(f"Target entity '{target_id}' does not exist")

        if source_id == target_id:
            return [source_id]

        queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])
        visited: set[str] = {source_id}

        while queue:
            current_id: str
            path: list[str]
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            for neighbor in self.get_neighbors(current_id, direction=DIRECTION_BOTH):
                if neighbor.id in visited:
                    continue

                new_path: list[str] = path + [neighbor.id]

                if neighbor.id == target_id:
                    return new_path

                visited.add(neighbor.id)
                queue.append((neighbor.id, new_path))

        return None

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[Entity, float]]:
        """Search entities using pure Tantivy.

        Args:
            query: Query string to search for.
            top_k: Number of results to return.

        Returns:
            List of (entity, score) tuples sorted by relevance.
        """
        # Initialize index if needed (for initial setup only)
        if self._tantivy_index is None:
            self._initialize_tantivy_index()

        if self._tantivy_index is None:
            return []  # No entities or tantivy not available

        # Create fresh searcher for each search
        searcher = self._tantivy_index.searcher()

        # Create query and search
        tantivy_query = self._tantivy_index.parse_query(query, ["content"])
        results = searcher.search(tantivy_query, limit=top_k)

        # Map results back to entities
        entity_scores: list[tuple[Entity, float]] = []
        entity_id_to_entity: dict[str, Entity] = {
            e.id: e for e in self._index.entity_by_id.values()
        }

        if results.hits:
            for score, doc_address in results.hits[:top_k]:
                doc = searcher.doc(doc_address)
                entity_id: str | None = doc.get_first("id")
                if entity_id and entity_id in entity_id_to_entity:
                    entity_scores.append((entity_id_to_entity[entity_id], float(score)))

        return entity_scores

    def query_entities(self) -> EntityQuery:
        """Create a query builder for entity queries.

        Returns:
            EntityQuery builder for chaining queries.
        """
        return EntityQuery(database=self)

    def query_relationships(self) -> RelationshipQuery:
        """Create a query builder for relationship queries.

        Returns:
            RelationshipQuery builder for chaining queries.
        """
        return RelationshipQuery(database=self)

    def translate_to_queries(
        self, operations: LLMGraphQueryOperations
    ) -> list[EntityQuery | RelationshipQuery]:
        """Translate LLM query operations into executable query builders.

        Args:
            operations: LLM query operations with entity and relationship queries.

        Returns:
            List of configured EntityQuery and/or RelationshipQuery ready for execution.
        """
        # Trigger normalization by accessing ops (handles duplicate queries).
        _ = operations.ops

        queries: list[EntityQuery | RelationshipQuery] = []

        # Translate entity queries
        for entity_query in operations.entity_queries:
            queries.append(self._translate_entity_query(entity_query))

        # Translate relationship queries
        for relationship_query in operations.relationship_queries:
            queries.append(self._translate_relationship_query(relationship_query))

        return queries

    def translate_to_query(
        self, operations: LLMGraphQueryOperations
    ) -> EntityQuery | RelationshipQuery:
        """Translate LLM query operations into a single executable query builder.

        This method maintains backward compatibility by returning the first query.
        Use translate_to_queries() for handling multiple queries.

        Args:
            operations: LLM query operations with entity and relationship queries.

        Returns:
            First configured EntityQuery or RelationshipQuery ready for execution.
        """
        queries = self.translate_to_queries(operations)

        if not queries:
            raise ValueError("No queries provided in operations")

        return queries[0]

    def _translate_entity_query(self, entity_query: LLMEntityQuery) -> EntityQuery:
        query = self.query_entities()

        # Apply types (OR logic)
        if entity_query.entity_types:
            query.type(entity_query.entity_types[0])
            for t in entity_query.entity_types[1:]:
                query.or_type(t)

        # Apply search
        if entity_query.search_query:
            query.search(entity_query.search_query)

        # Apply date filter
        if entity_query.created_after or entity_query.created_before:
            start_date = (
                date.fromisoformat(entity_query.created_after)
                if entity_query.created_after
                else None
            )
            end_date = (
                date.fromisoformat(entity_query.created_before)
                if entity_query.created_before
                else None
            )
            query.created_between(start_date, end_date)

        # Apply expansion
        if entity_query.expand_depth:
            query.expand_neighbors(
                k=entity_query.expand_depth,
                direction=entity_query.expand_direction,
                relationship_type=entity_query.expand_relationship_type,
            )

        # Common options
        query.limit(entity_query.limit)
        query.order_by(entity_query.order_by, ascending=entity_query.order_ascending)

        return query

    def _translate_relationship_query(
        self, relationship_query: LLMRelationshipQuery
    ) -> RelationshipQuery:
        query = self.query_relationships()

        # Apply types (OR logic)
        if relationship_query.relationship_types:
            query.type(relationship_query.relationship_types[0])
            for t in relationship_query.relationship_types[1:]:
                query.or_type(t)

        # Source/Target filters
        if relationship_query.source_entity_name:
            query.from_entity(entity_name=relationship_query.source_entity_name)
        if relationship_query.target_entity_name:
            query.to_entity(entity_name=relationship_query.target_entity_name)

        # Always include entities
        query.with_entities()

        # Common options
        query.limit(relationship_query.limit)
        query.order_by(relationship_query.order_by, ascending=relationship_query.order_ascending)

        return query

    def _initialize_tantivy_index(self) -> None:
        """Initialize Tantivy index with current entities."""
        entities = list(self._index.entity_by_id.values())

        if not entities:
            self._tantivy_index = None
            self._tantivy_searcher = None
            return

        # Create schema
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("id", stored=True)
        schema_builder.add_text_field("name", stored=True)
        schema_builder.add_text_field("content", stored=True)
        schema_builder.add_text_field("type", stored=True)
        schema = schema_builder.build()

        # Create index
        self._tantivy_index = tantivy.Index(schema)

        # Add all entities to the index
        writer = self._tantivy_index.writer()
        for entity in entities:
            writer.add_document(
                tantivy.Document(
                    id=[entity.id],
                    name=[entity.name],
                    content=[
                        f"{entity.name} {entity.content}"
                    ],  # Include name in content for search
                    type=[entity.type],
                )
            )
        writer.commit()
        writer.wait_merging_threads()
        self._tantivy_index.reload()
        self._tantivy_searcher = self._tantivy_index.searcher()

    def _incremental_add_entities_to_tantivy(self, entities: list[Entity]) -> None:
        """Add multiple entities to Tantivy index incrementally in one batch."""
        # Initialize Tantivy index if needed
        if self._tantivy_index is None:
            self._initialize_tantivy_index()
            return  # Entities will be added in full rebuild

        # Add all documents in one batch for efficiency
        writer = self._tantivy_index.writer()
        for entity in entities:
            writer.add_document(
                tantivy.Document(
                    id=[entity.id],
                    name=[entity.name],
                    content=[f"{entity.name} {entity.content}"],
                    type=[entity.type],
                )
            )

        # Single commit for all entities
        writer.commit()
        writer.wait_merging_threads()
        self._tantivy_index.reload()
        self._tantivy_searcher = self._tantivy_index.searcher()

    def _incremental_update_entity_in_tantivy(self, entity_id: str, entity: Entity) -> None:
        """Update single entity in Tantivy index incrementally."""
        # Initialize Tantivy index if needed
        if self._tantivy_index is None:
            self._initialize_tantivy_index()
            return

        # Delete old document and add new one
        writer = self._tantivy_index.writer()

        # Delete by term query on ID field
        delete_query = self._tantivy_index.parse_query(f'"{entity_id}"', ["id"])
        writer.delete_documents_by_query(delete_query)

        # Add updated document
        writer.add_document(
            tantivy.Document(
                id=[entity.id],
                name=[entity.name],
                content=[f"{entity.name} {entity.content}"],
                type=[entity.type],
            )
        )

        writer.commit()
        writer.wait_merging_threads()
        self._tantivy_index.reload()
        self._tantivy_searcher = self._tantivy_index.searcher()

    def _incremental_remove_entity_from_tantivy(self, entity_id: str) -> None:
        """Remove single entity from Tantivy index incrementally."""
        # If Tantivy index doesn't exist, nothing to do
        if self._tantivy_index is None:
            return

        # Delete document by ID
        writer = self._tantivy_index.writer()
        delete_query = self._tantivy_index.parse_query(f'"{entity_id}"', ["id"])
        writer.delete_documents_by_query(delete_query)
        writer.commit()
        writer.wait_merging_threads()
        self._tantivy_index.reload()
        self._tantivy_searcher = self._tantivy_index.searcher()

    def to_dict(self) -> dict[str, object]:
        """Serialize graph to dictionary.

        Returns:
            Dictionary with entities, relationships, and search index.
        """
        return {
            "entities": [e.to_dict() for e in self._index.entity_by_id.values()],
            "relationships": [r.to_dict() for r in self._index.relationship_by_id.values()],
            "index": self._index.to_dict(),
            "search_index_initialized": self._tantivy_index is not None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GraphDatabase:
        """Deserialize graph from dictionary.

        Args:
            data: Dictionary with entities, relationships, and search index.

        Returns:
            Completely restored GraphDatabase with all entities, relationships,
            and fully functional Tantivy search index.
        """
        db = cls()

        # Try to restore from index first (more efficient)
        index_data = data.get("index")
        if index_data and isinstance(index_data, dict):
            # Restore GraphIndex from saved data
            db._index = GraphIndex.from_dict(cast(dict[str, Any], index_data))

            # Extract entities and relationships from the restored index
            entities = list(db._index.entity_by_id.values())
        else:
            # Fallback to the old approach - restore entities from the entities list
            entities: list[Entity] = []
            entities_data = data.get("entities", [])
            if isinstance(entities_data, list):
                for entity_data in entities_data:  # type: ignore[assignment]
                    entity_data_dict = cast(dict[str, Any], entity_data)
                    entity = Entity.from_dict(entity_data_dict)
                    entities.append(entity)
                    # Add to GraphIndex directly (without triggering Tantivy indexing)
                    db._index.add_entity(entity)

            # Restore relationships
            relationships_data = data.get("relationships", [])
            if isinstance(relationships_data, list):
                for rel_data in relationships_data:  # type: ignore[assignment]
                    rel_data_dict = cast(dict[str, Any], rel_data)
                    relationship = Relationship.from_dict(rel_data_dict)
                    # Use GraphIndex.add_relationship to maintain consistency
                    db._index.add_relationship(relationship)

        # Rebuild Tantivy search index if it was originally initialized
        if data.get("search_index_initialized", False) and entities:
            # Initialize Tantivy index with entities (this creates the index and adds all entities)
            db._initialize_tantivy_index()
            # Don't add entities again since _initialize_tantivy_index already adds them when there are entities

        return db

    @property
    def entity_count(self) -> int:
        """Get total number of entities."""
        return len(self._index.entity_by_id)

    @property
    def relationship_count(self) -> int:
        """Get total number of relationships."""
        return len(self._index.relationship_by_id)

    def clear(self) -> None:
        """Clear all entities and relationships from the database."""
        # Clear the graph index
        self._index.clear()

        # Clear Tantivy search index if it exists
        if self._tantivy_index is not None:
            self._tantivy_index = None
            self._tantivy_searcher = None


# New unified query classes


class Filter[T](ABC):
    """Base class for all filters."""

    @abstractmethod
    def apply(self, items: Iterable[T], index: GraphIndex | None = None) -> set[str]:
        """Apply filter and return set of matching IDs."""
        pass

    @abstractmethod
    def get_selectivity(self, index: GraphIndex | None = None) -> float:
        """Estimate filter selectivity (0.0 to 1.0, lower is more selective)."""
        pass


class EntityFilter(Filter[Entity]):
    """Base class for entity filters."""

    pass


class RelationshipFilter(Filter[Relationship]):
    """Base class for relationship filters."""

    pass


@dataclass
class EntityTypeFilter(EntityFilter):
    """Filter by entity type."""

    entity_type: EntityType

    @override
    def apply(self, items: Iterable[Entity], index: GraphIndex | None = None) -> set[str]:
        if index:
            return index.entities_by_type.get(self.entity_type, set())
        return {item.id for item in items if item.type == self.entity_type}

    @override
    def get_selectivity(self, index: GraphIndex | None = None) -> float:
        if index and self.entity_type in index.entities_by_type:
            return len(index.entities_by_type[self.entity_type]) / max(
                len(index.entity_by_id), MIN_DIVISOR
            )
        return DEFAULT_SELECTIVITY


@dataclass
class RelationshipTypeFilter(RelationshipFilter):
    """Filter by relationship type."""

    relationship_type: RelationType

    @override
    def apply(self, items: Iterable[Relationship], index: GraphIndex | None = None) -> set[str]:
        if index:
            return index.relationships_by_type.get(self.relationship_type, set())
        return {item.id for item in items if item.type == self.relationship_type}

    @override
    def get_selectivity(self, index: GraphIndex | None = None) -> float:
        if index and self.relationship_type in index.relationships_by_type:
            return len(index.relationships_by_type[self.relationship_type]) / max(
                len(index.relationship_by_id), MIN_DIVISOR
            )
        return DEFAULT_SELECTIVITY


@dataclass
class RelationshipSourceFilter(RelationshipFilter):
    """Filter by source entity."""

    entity_id: str | None = None
    entity_name: str | None = None

    def __post_init__(self) -> None:
        assert self.entity_id or self.entity_name, (
            "Either entity_id or entity_name must be provided"
        )

    @override
    def apply(self, items: Iterable[Relationship], index: GraphIndex | None = None) -> set[str]:
        source_id = self.entity_id
        if index and self.entity_name:
            source_id = index.entity_by_name.get(self.entity_name)

        if source_id:
            if index:
                return index.outgoing_edges.get(source_id, set())
            else:
                return {item.id for item in items if item.source == source_id}
        return set()

    @override
    def get_selectivity(self, index: GraphIndex | None = None) -> float:
        return RELATIONSHIP_FILTER_SELECTIVITY


@dataclass
class RelationshipTargetFilter(RelationshipFilter):
    """Filter by target entity."""

    entity_id: str | None = None
    entity_name: str | None = None

    def __post_init__(self) -> None:
        assert self.entity_id or self.entity_name, (
            "Either entity_id or entity_name must be provided"
        )

    @override
    def apply(self, items: Iterable[Relationship], index: GraphIndex | None = None) -> set[str]:
        target_id = self.entity_id
        if index and self.entity_name:
            target_id = index.entity_by_name.get(self.entity_name)

        if target_id:
            if index:
                return index.incoming_edges.get(target_id, set())
            else:
                return {item.id for item in items if item.target == target_id}
        return set()

    @override
    def get_selectivity(self, index: GraphIndex | None = None) -> float:
        return RELATIONSHIP_FILTER_SELECTIVITY


@dataclass
class OrFilter(Filter[T]):
    """OR filter that combines multiple filters with OR logic."""

    filters: list[Filter[T]]

    @override
    def apply(self, items: Iterable[T], index: GraphIndex | None = None) -> set[str]:
        if not self.filters:
            return set()

        result: set[str] = set()
        for filter_obj in self.filters:
            filter_result = filter_obj.apply(items, index)
            result.update(filter_result)
        return result

    @override
    def get_selectivity(self, index: GraphIndex | None = None) -> float:
        if not self.filters:
            return ZERO_SELECTIVITY
        return max(f.get_selectivity(index) for f in self.filters)


class EntityQuery:
    """Query builder for entity queries with OR/AND support."""

    database: GraphDatabase
    filters: list[Filter[Entity]]
    sort_field: str | None
    sort_ascending: bool
    limit_count: int | None

    def __init__(self, database: GraphDatabase) -> None:
        """Initialize entity query builder."""
        self.database = database
        self.filters: list[Filter[Entity]] = []
        self.sort_field: str | None = None
        self.sort_ascending: bool = True
        self.limit_count: int | None = None
        self._expand_depth: int | None = None
        self._expand_direction: str = DIRECTION_BOTH
        self._expand_relationship_type: RelationType | None = None

    def type(self, entity_type: EntityType) -> EntityQuery:
        """Filter by entity type."""
        filter_obj = EntityTypeFilter(entity_type=entity_type)
        self.filters.append(filter_obj)
        return self

    def or_type(self, entity_type: EntityType) -> EntityQuery:
        """Add OR condition for entity type."""
        new_filter = EntityTypeFilter(entity_type=entity_type)
        existing_type_filters = [f for f in self.filters if isinstance(f, EntityTypeFilter)]

        if existing_type_filters:
            or_filters = existing_type_filters + [new_filter]
            or_filter = OrFilter(cast(list[Filter[Entity]], or_filters))
            self.filters = [f for f in self.filters if not isinstance(f, EntityTypeFilter)]
            self.filters.append(or_filter)
        else:
            or_filter = OrFilter([new_filter])
            self.filters.append(or_filter)

        return self

    def created_between(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> EntityQuery:
        """Filter by creation date range."""

        class EntityCreatedDateFilter(EntityFilter):
            start_date: date | None
            end_date: date | None

            def __init__(
                self, start_date: date | None = None, end_date: date | None = None
            ) -> None:
                self.start_date = start_date
                self.end_date = end_date

            @override
            def apply(self, items: Iterable[Entity], index: GraphIndex | None = None) -> set[str]:
                result: set[str] = set()
                if index:
                    # Optimization: Iterate over existing dates in index instead of day-by-day
                    # This is much faster when the range is large but data is sparse
                    for check_date, entity_ids in index.entities_by_created_date.items():
                        if self.start_date and check_date < self.start_date:
                            continue
                        if self.end_date and check_date > self.end_date:
                            continue
                        result.update(cast(Iterable[str], entity_ids))
                else:
                    for item in items:
                        item_date = item.created_at.date()
                        if self.start_date and item_date < self.start_date:
                            continue
                        if self.end_date and item_date > self.end_date:
                            continue
                        result.add(item.id)
                return result

            @override
            def get_selectivity(self, index: GraphIndex | None = None) -> float:
                return DATE_FILTER_SELECTIVITY

        filter_obj = EntityCreatedDateFilter(start_date=start_date, end_date=end_date)
        self.filters.append(filter_obj)
        return self

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> EntityQuery:
        """Full-text search using Tantivy."""
        search_results = self.database.search(query, top_k)
        search_ids = {result[0].id for result in search_results}

        class SearchFilter(EntityFilter):
            search_ids: set[str]

            def __init__(self, search_ids: set[str]) -> None:
                self.search_ids = search_ids

            @override
            def apply(self, items: Iterable[Entity], index: GraphIndex | None = None) -> set[str]:
                return self.search_ids

            @override
            def get_selectivity(self, index: GraphIndex | None = None) -> float:
                return len(self.search_ids) / max(
                    len(index.entity_by_id) if index else MIN_DIVISOR, MIN_DIVISOR
                )

        search_filter = SearchFilter(search_ids)
        self.filters.append(search_filter)
        return self

    def order_by(self, field: str, ascending: bool = True) -> EntityQuery:
        """Set sort order for results."""
        self.sort_field = field
        self.sort_ascending = ascending
        return self

    def limit(self, count: int) -> EntityQuery:
        """Limit number of results."""
        self.limit_count = count
        return self

    def expand_neighbors(
        self,
        k: int,
        direction: str = DIRECTION_BOTH,
        relationship_type: RelationType | None = None,
    ) -> EntityQuery:
        """Expand query results to include neighbors up to depth k.

        Args:
            k: Maximum depth to expand (1 = direct neighbors, 2 = neighbors of neighbors, etc.)
            direction: Direction to traverse - 'outgoing', 'incoming', or 'both'.
            relationship_type: Optional filter by relationship type.

        Returns:
            Self for method chaining.
        """
        assert k > 0, "Expansion depth k must be positive"
        assert direction in (DIRECTION_OUTGOING, DIRECTION_INCOMING, DIRECTION_BOTH), (
            f"Invalid direction: {direction}"
        )

        self._expand_depth = k
        self._expand_direction = direction
        self._expand_relationship_type = relationship_type
        return self

    def execute(self) -> QueryResult[Entity]:
        """Execute the query and return matching entities with statistics."""
        start_time = time.perf_counter()

        # Sort filters by selectivity (most selective first) for optimization
        sorted_filters = (
            sorted(self.filters, key=lambda f: f.get_selectivity(self.database.index))
            if self.filters
            else []
        )

        # Track filter statistics
        filter_stats_list: list[FilterStats] = []

        if not sorted_filters:
            results = list(self.database.index.entity_by_id.values())
        else:
            # Start with the most selective filter
            first_filter = sorted_filters[0]
            # Pass empty list as items since we are using index
            entity_ids = first_filter.apply([], self.database.index)

            filter_stats_list.append(
                FilterStats(
                    name=type(first_filter).__name__,
                    selectivity=first_filter.get_selectivity(self.database.index),
                    matched_count=len(entity_ids),
                )
            )

            # Intersect with remaining filters
            for filter_obj in sorted_filters[1:]:
                selectivity = filter_obj.get_selectivity(self.database.index)
                filter_name = type(filter_obj).__name__

                # Pass empty list as items since we are using index
                filter_result = filter_obj.apply([], self.database.index)
                matched_count = len(filter_result)

                filter_stats_list.append(
                    FilterStats(
                        name=filter_name,
                        selectivity=selectivity,
                        matched_count=matched_count,
                    )
                )

                entity_ids.intersection_update(filter_result)

                # Optimization: if empty, stop
                if not entity_ids:
                    break

            results = [
                self.database.index.entity_by_id[eid]
                for eid in entity_ids
                if eid in self.database.index.entity_by_id
            ]

        # Apply neighbor expansion if configured
        if self._expand_depth and self._expand_depth > 0:
            expanded_ids = self._bfs_expand(
                {entity.id for entity in results},
                self._expand_depth,
                self._expand_direction,
                self._expand_relationship_type,
            )
            results = [
                self.database.index.entity_by_id[eid]
                for eid in expanded_ids
                if eid in self.database.index.entity_by_id
            ]

        # Apply sorting
        if self.sort_field and results:
            reverse = not self.sort_ascending

            # Use a safe sorting key function that handles missing attributes
            def sort_key(entity: Entity) -> str:
                if self.sort_field is None:
                    return ""
                value = getattr(entity, self.sort_field, "")
                return str(value) if value is not None else ""

            results.sort(key=sort_key, reverse=reverse)

        # Apply limit
        if self.limit_count:
            results = results[: self.limit_count]

        # Calculate execution time
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Build statistics
        stats = QueryStats(
            filters=filter_stats_list,
            total_results=len(results),
            execution_time_ms=execution_time_ms,
        )

        return QueryResult(results=results, stats=stats)

    def _bfs_expand(
        self,
        start_ids: set[str],
        max_depth: int,
        direction: str,
        relationship_type: RelationType | None,
    ) -> set[str]:
        """Perform BFS expansion from start entities up to max_depth.

        Args:
            start_ids: Set of entity IDs to start expansion from.
            max_depth: Maximum depth to expand.
            direction: Direction to traverse.
            relationship_type: Optional relationship type filter.

        Returns:
            Set of all entity IDs found during expansion (including start_ids).
        """
        all_entities: set[str] = set(start_ids)
        current_level: set[str] = start_ids
        visited: set[str] = set(start_ids)

        for _ in range(max_depth):
            next_level: set[str] = set()

            for entity_id in current_level:
                neighbors = self.database.get_neighbors(
                    entity_id, direction=direction, relationship_type=relationship_type
                )

                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        next_level.add(neighbor.id)
                        all_entities.add(neighbor.id)

            if not next_level:
                break

            current_level = next_level

        return all_entities


class RelationshipQuery:
    """Query builder for relationship queries with OR/AND support."""

    def __init__(self, database: GraphDatabase) -> None:
        """Initialize relationship query builder."""
        self.database = database
        self.filters: list[Filter[Relationship]] = []
        self.sort_field: str | None = None
        self.sort_ascending: bool = True
        self.limit_count: int | None = None

    def type(self, relationship_type: RelationType) -> RelationshipQuery:
        """Filter by relationship type."""
        filter_obj = RelationshipTypeFilter(relationship_type=relationship_type)
        self.filters.append(filter_obj)
        return self

    def or_type(self, relationship_type: RelationType) -> RelationshipQuery:
        """Add OR condition for relationship type."""
        new_filter = RelationshipTypeFilter(relationship_type=relationship_type)
        existing_type_filters = [f for f in self.filters if isinstance(f, RelationshipTypeFilter)]

        if existing_type_filters:
            or_filters = existing_type_filters + [new_filter]
            or_filter = OrFilter(cast(list[Filter[Relationship]], or_filters))
            self.filters = [f for f in self.filters if not isinstance(f, RelationshipTypeFilter)]
            self.filters.append(or_filter)
        else:
            or_filter = OrFilter([new_filter])
            self.filters.append(or_filter)

        return self

    def from_entity(
        self, entity_id: str | None = None, entity_name: str | None = None
    ) -> RelationshipQuery:
        """Filter by source entity."""
        filter_obj = RelationshipSourceFilter(entity_id=entity_id, entity_name=entity_name)
        self.filters.append(filter_obj)
        return self

    def to_entity(
        self, entity_id: str | None = None, entity_name: str | None = None
    ) -> RelationshipQuery:
        """Filter by target entity."""
        filter_obj = RelationshipTargetFilter(entity_id=entity_id, entity_name=entity_name)
        self.filters.append(filter_obj)
        return self

    def with_entities(self) -> RelationshipQuery:
        """Include entities in the query results."""
        # This method is kept for backward compatibility but doesn't need to do anything
        # since execute() always returns (source, target, relationship) tuples
        return self

    def order_by(self, field: str, ascending: bool = True) -> RelationshipQuery:
        """Set sort order for results."""
        self.sort_field = field
        self.sort_ascending = ascending
        return self

    def limit(self, count: int) -> RelationshipQuery:
        """Limit number of results."""
        self.limit_count = count
        return self

    def execute(
        self,
    ) -> QueryResult[tuple[Entity, Entity, Relationship]]:
        """Execute the query and return matching relationships with their entities as (source, target, relationship) tuples."""
        start_time = time.perf_counter()

        # Sort filters by selectivity (most selective first) for optimization
        sorted_filters = (
            sorted(self.filters, key=lambda f: f.get_selectivity(self.database.index))
            if self.filters
            else []
        )

        # Track filter statistics
        filter_stats_list: list[FilterStats] = []

        if not sorted_filters:
            results = list(self.database.index.relationship_by_id.values())
        else:
            # Start with the most selective filter
            first_filter = sorted_filters[0]
            # Pass empty list as items since we are using index
            relationship_ids = first_filter.apply([], self.database.index)

            filter_stats_list.append(
                FilterStats(
                    name=type(first_filter).__name__,
                    selectivity=first_filter.get_selectivity(self.database.index),
                    matched_count=len(relationship_ids),
                )
            )

            # Intersect with remaining filters
            for filter_obj in sorted_filters[1:]:
                selectivity = filter_obj.get_selectivity(self.database.index)
                filter_name = type(filter_obj).__name__

                # Pass empty list as items since we are using index
                filter_result = filter_obj.apply([], self.database.index)
                matched_count = len(filter_result)

                filter_stats_list.append(
                    FilterStats(
                        name=filter_name,
                        selectivity=selectivity,
                        matched_count=matched_count,
                    )
                )

                relationship_ids.intersection_update(filter_result)

                # Optimization: if empty, stop
                if not relationship_ids:
                    break

            results = [
                self.database.index.relationship_by_id[rid]
                for rid in relationship_ids
                if rid in self.database.index.relationship_by_id
            ]

        # Apply sorting
        if self.sort_field and results:
            reverse = not self.sort_ascending

            # Use a safe sorting key function that handles missing attributes
            def sort_key(relationship: Relationship) -> str:
                if self.sort_field is None:
                    return ""
                value = getattr(relationship, self.sort_field, "")
                return str(value) if value is not None else ""

            results.sort(key=sort_key, reverse=reverse)

        # Apply limit
        if self.limit_count:
            results = results[: self.limit_count]

        # Calculate execution time
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Always build (source, target, relationship) tuples
        result_tuples: list[tuple[Entity, Entity, Relationship]] = []
        for rel in results:
            source_entity = self.database.index.entity_by_id.get(rel.source)
            target_entity = self.database.index.entity_by_id.get(rel.target)
            if source_entity and target_entity:
                result_tuples.append((source_entity, target_entity, rel))

        # Build statistics
        stats = QueryStats(
            filters=filter_stats_list,
            total_results=len(result_tuples),
            execution_time_ms=execution_time_ms,
        )
        return QueryResult(results=result_tuples, stats=stats)
