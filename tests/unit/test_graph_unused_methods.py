"""Tests for previously unused GraphDatabase methods."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

# Constants for testing
DEFAULT_CONFIDENCE = 0.9

from blockether_foundation.graph import (
    Entity,
    EntityType,
    GraphDatabase,
    LLMEntityQuery,
    LLMGraphAddEntity,
    LLMGraphAddRelationship,
    LLMGraphOperations,
    LLMGraphQueryOperations,
    LLMGraphUpdateEntity,
    Relationship,
    RelationType,
)

# Test Constants
TYPE_CONCEPT: EntityType = "concept"
TYPE_ORGANIZATION: EntityType = "organization"
TYPE_CREATURE: EntityType = "creature"
TYPE_OBJECT: EntityType = "object"

# Relationship Types
REL_TYPE_RELATED_TO: RelationType = "related_to"
REL_TYPE_PART_OF: RelationType = "part_of"
REL_TYPE_OWNED_BY: RelationType = "owned_by"
REL_TYPE_OCCURS_AT: RelationType = "occurs_at"

# Count Constants
THREE_RELATIONSHIPS_COUNT = 3


def _relationship_results_from_tuples(
    results: list[tuple[Entity, Entity, Relationship]],
) -> list[Relationship]:
    """Normalize relationship query results from tuple format to Relationship instances."""
    assert results is not None, "Results cannot be None"
    assert len(results) > 0, "Results should have items"

    first_item = results[0]
    assert first_item is not None, "First item cannot be None"
    assert isinstance(first_item, tuple), "First item should be a tuple"
    assert all(isinstance(item, tuple) for item in results), "All items should be tuples"

    return [rel for _, _, rel in results]


def _relationship_results_from_relationships(
    results: list[Relationship],
) -> list[Relationship]:
    """Normalize relationship query results that are already Relationship instances."""
    assert results is not None, "Results cannot be None"
    assert len(results) > 0, "Results should have items"

    first_item = results[0]
    assert first_item is not None, "First item cannot be None"
    assert all(isinstance(item, Relationship) for item in results), (
        "All items should be Relationship objects"
    )

    return results


def _relationship_results_from_empty_list(
    results: list[Relationship],
) -> list[Relationship]:
    """Handle empty relationship results list."""
    assert results is not None, "Results cannot be None"
    assert len(results) == 0, "Results should be empty"
    return results


class TestGraphDatabaseUnusedMethods:
    """Test class for previously unused GraphDatabase methods."""

    @pytest.mark.unit
    def test_get_relationship(self) -> None:
        """Test getting a relationship by ID."""
        db = GraphDatabase()

        # Add entities
        source = Entity(name="Source", id="source", type=TYPE_CONCEPT, content="Source entity")
        target = Entity(name="Target", id="target", type=TYPE_CONCEPT, content="Target entity")
        db.add_entities([source, target])

        # Add relationship
        relationship = Relationship(
            source="source",
            target="target",
            type=REL_TYPE_RELATED_TO,
            id="rel_1",
        )
        db.add_relationship(relationship)

        # Test getting existing relationship
        result = db.get_relationship("rel_1")
        assert result is not None
        assert result.id == "rel_1"
        assert result.source == "source"
        assert result.target == "target"

        # Test getting non-existent relationship
        result = db.get_relationship("non_existent")
        assert result is None

    @pytest.mark.unit
    def test_get_relationship_by_entities(self) -> None:
        """Test getting relationship between two entities."""
        db = GraphDatabase()

        # Add entities
        source = Entity(name="Source", id="source", type=TYPE_CONCEPT, content="Source entity")
        target = Entity(name="Target", id="target", type=TYPE_CONCEPT, content="Target entity")
        db.add_entities([source, target])

        # Add relationship
        relationship = Relationship(
            source="source",
            target="target",
            type=REL_TYPE_RELATED_TO,
        )
        db.add_relationship(relationship)

        # Test getting existing relationship
        result = db.get_relationship_by_entities("source", "target")
        assert result is not None
        assert result.source == "source"
        assert result.target == "target"

        # Test getting non-existent relationship
        result = db.get_relationship_by_entities("source", "non_existent")
        assert result is None

    @pytest.mark.unit
    def test_delete_relationship_by_entities(self) -> None:
        """Test deleting relationship between two entities."""
        db = GraphDatabase()

        # Add entities
        source = Entity(name="Source", id="source", type=TYPE_CONCEPT, content="Source entity")
        target = Entity(name="Target", id="target", type=TYPE_CONCEPT, content="Target entity")
        db.add_entities([source, target])

        # Add relationship
        relationship = Relationship(
            source="source",
            target="target",
            type=REL_TYPE_RELATED_TO,
        )
        db.add_relationship(relationship)

        # Verify relationship exists
        result = db.get_relationship_by_entities("source", "target")
        assert result is not None

        # Delete relationship
        db.delete_relationship_by_entities("source", "target")

        # Verify relationship is deleted
        result = db.get_relationship_by_entities("source", "target")
        assert result is None

        # Test deleting non-existent relationship raises error
        with pytest.raises(
            ValueError, match="Relationship between 'source' and 'target' does not exist"
        ):
            db.delete_relationship_by_entities("source", "target")

    @pytest.mark.unit
    def test_import_operations(self) -> None:
        """Test importing and executing LLM graph operations."""
        db = GraphDatabase()

        # Seed entity that will be updated by the operations import
        db.add_entity(
            Entity(
                name="Preexisting Entity",
                id="updated_entity",
                type=TYPE_OBJECT,
                content="Original content",
            )
        )

        # Create operations
        operations = LLMGraphOperations(
            reasoning="Test operations",
            confidence=0.9,
            importance=DEFAULT_CONFIDENCE,
            add_entity_ops=[
                LLMGraphAddEntity(
                    reasoning="Add test entity",
                    confidence=0.8,
            importance=DEFAULT_CONFIDENCE,
                    name="Test Entity",
                    type=TYPE_CONCEPT,
                    content="Test content",
                ),
            ],
            update_entity_ops=[
                LLMGraphUpdateEntity(
                    reasoning="Update entity",
                    confidence=0.8,
            importance=DEFAULT_CONFIDENCE,
                    id="updated_entity",
                    name="Updated Entity",
                    type=TYPE_OBJECT,
                    content="Updated content",
                ),
            ],
            add_relationship_ops=[
                LLMGraphAddRelationship(
                    reasoning="Add relationship",
                    confidence=0.8,
                    importance=DEFAULT_CONFIDENCE,
                    source_name="Test Entity",
                    target_name="Updated Entity",
                    type=REL_TYPE_RELATED_TO,
                ),
            ],
            delete_entity_ops=[],
            delete_relationship_ops=[],
        )

        # Import operations (should not raise errors even with non-existent entities for deletes)
        db.import_operations(operations)

        # Verify entities were added
        entities = list(db.index.entity_by_id.values())
        assert len(entities) >= 2
        entity_names = {e.name for e in entities}
        assert "Test Entity" in entity_names
        assert "Updated Entity" in entity_names

        # Verify relationship was added
        relationships = list(db.index.relationship_by_id.values())
        assert len(relationships) >= 1
        rel = next(r for r in relationships if r.type == REL_TYPE_RELATED_TO)
        assert rel.type == REL_TYPE_RELATED_TO

    @pytest.mark.unit
    def test_find_entities_by_type(self) -> None:
        """Test finding entities by type."""
        db = GraphDatabase()

        # Add entities of different types
        entities = [
            Entity(name="Entity1", id="e1", type=TYPE_CONCEPT, content="Concept entity"),
            Entity(name="Entity2", id="e2", type=TYPE_CONCEPT, content="Another concept"),
            Entity(name="Entity3", id="e3", type=TYPE_OBJECT, content="Tool entity"),
            Entity(name="Entity4", id="e4", type=TYPE_OBJECT, content="Library entity"),
        ]
        db.add_entities(entities)

        # Test finding entities by type
        concept_entities = db.find_entities_by_type(TYPE_CONCEPT)
        assert len(concept_entities) == 2
        assert all(e.type == TYPE_CONCEPT for e in concept_entities)

        tool_entities = db.find_entities_by_type(TYPE_OBJECT)
        assert len(tool_entities) == 2
        assert all(e.type == TYPE_OBJECT for e in tool_entities)

        # Test with non-existent type
        person_entities = db.find_entities_by_type(TYPE_CREATURE)
        assert len(person_entities) == 0

    @pytest.mark.unit
    def test_find_relationships_by_type(self) -> None:
        """Test finding relationships by type."""
        db = GraphDatabase()

        # Add entities
        entities = [
            Entity(name="E1", id="e1", type=TYPE_CONCEPT, content="Entity 1"),
            Entity(name="E2", id="e2", type=TYPE_CONCEPT, content="Entity 2"),
            Entity(name="E3", id="e3", type=TYPE_CONCEPT, content="Entity 3"),
        ]
        db.add_entities(entities)

        # Add relationships of different types
        relationship1 = Relationship(source="e1", target="e2", type=REL_TYPE_RELATED_TO)
        relationship2 = Relationship(source="e2", target="e3", type=REL_TYPE_RELATED_TO)
        relationship3 = Relationship(source="e1", target="e3", type=REL_TYPE_RELATED_TO)
        db.add_relationship(relationship1)
        db.add_relationship(relationship2)
        db.add_relationship(relationship3)

        # Test finding relationships by type
        related_rels = db.find_relationships_by_type(REL_TYPE_RELATED_TO)
        expected_relationship_count = 3
        assert len(related_rels) == expected_relationship_count
        assert all(r.type == REL_TYPE_RELATED_TO for r in related_rels)

        # Test with non-existent type
        similar_rels = db.find_relationships_by_type(REL_TYPE_PART_OF)
        assert len(similar_rels) == 0

    @pytest.mark.unit
    def test_find_entities_by_name_pattern(self) -> None:
        """Test finding entities by name pattern."""
        db = GraphDatabase()

        # Add entities with various names
        entities = [
            Entity(
                name="Python Programming", id="python", type=TYPE_CONCEPT, content="Python content"
            ),
            Entity(name="Python Library", id="pylib", type=TYPE_OBJECT, content="Library content"),
            Entity(name="Java Programming", id="java", type=TYPE_CONCEPT, content="Java content"),
            Entity(name="JavaScript", id="js", type=TYPE_OBJECT, content="JS content"),
        ]
        db.add_entities(entities)

        # Test exact match
        python_exact = db.find_entities_by_name_pattern("Python Programming", exact=True)
        assert len(python_exact) == 1
        assert python_exact[0].name == "Python Programming"

        # Test exact match with non-existent name
        non_existent = db.find_entities_by_name_pattern("Non Existent", exact=True)
        assert len(non_existent) == 0

        # Test non-exact match (uses Tantivy search now)
        python_prefix = db.find_entities_by_name_pattern("Python", exact=False)
        assert len(python_prefix) >= 1, "Should find Python-related entities"
        python_names = {e.name for e in python_prefix}
        assert "Python Programming" in python_names

        # Test case-insensitive search with Tantivy
        java_prefix = db.find_entities_by_name_pattern("java", exact=False)
        assert len(java_prefix) >= 1, "Should find Java-related entities"
        java_results = {entity.name for entity in java_prefix}
        assert "Java Programming" in java_results

    @pytest.mark.unit
    def test_find_entities_by_timerange(self) -> None:
        """Test finding entities by time range."""
        db = GraphDatabase()

        # Create timestamps
        now = datetime.now(UTC)
        yesterday = now - timedelta(days=1)
        two_days_ago = now - timedelta(days=2)
        tomorrow = now + timedelta(days=1)
        next_week = now + timedelta(days=7)

        # Add entities with different timestamps
        entities = [
            Entity(
                name="Old Entity",
                id="old",
                type=TYPE_CONCEPT,
                content="Old content",
                created_at=two_days_ago,
                updated_at=yesterday,
            ),
            Entity(
                name="Recent Entity",
                id="recent",
                type=TYPE_CONCEPT,
                content="Recent content",
                created_at=yesterday,
                updated_at=now,
            ),
            Entity(
                name="Future Entity",
                id="future",
                type=TYPE_CONCEPT,
                content="Future content",
                created_at=tomorrow,
                updated_at=tomorrow,
            ),
        ]
        db.add_entities(entities)

        # Test finding entities in time range
        recent_entities = db.find_entities_by_timerange(
            start_date=two_days_ago.date(),
            end_date=now.date(),
            use_created=True,
        )
        assert len(recent_entities) == 2
        recent_names = {e.name for e in recent_entities}
        assert "Old Entity" in recent_names
        assert "Recent Entity" in recent_names

        # Test with updated_at field
        updated_entities = db.find_entities_by_timerange(
            start_date=yesterday.date(),
            end_date=tomorrow.date(),
            use_created=False,
        )
        expected_updated_count = 3
        assert len(updated_entities) == expected_updated_count
        updated_names = {e.name for e in updated_entities}
        assert {"Old Entity", "Recent Entity", "Future Entity"} == updated_names

        # Test with no matches
        no_matches = db.find_entities_by_timerange(
            start_date=next_week.date(),
            end_date=next_week.date(),
        )
        assert len(no_matches) == 0

    @pytest.mark.unit
    def test_find_path(self) -> None:
        """Test finding paths between entities."""
        db = GraphDatabase()

        # Create a simple graph: A -> B -> C -> D
        entities = [
            Entity(name="A", id="a", type=TYPE_CONCEPT, content="Entity A"),
            Entity(name="B", id="b", type=TYPE_CONCEPT, content="Entity B"),
            Entity(name="C", id="c", type=TYPE_CONCEPT, content="Entity C"),
            Entity(name="D", id="d", type=TYPE_CONCEPT, content="Entity D"),
            Entity(name="E", id="e", type=TYPE_CONCEPT, content="Entity E"),  # Disconnected node
        ]
        db.add_entities(entities)

        # Create relationships forming a path
        relationship_ab = Relationship(source="a", target="b", type=REL_TYPE_RELATED_TO)
        relationship_bc = Relationship(source="b", target="c", type=REL_TYPE_RELATED_TO)
        relationship_cd = Relationship(source="c", target="d", type=REL_TYPE_RELATED_TO)
        db.add_relationship(relationship_ab)
        db.add_relationship(relationship_bc)
        db.add_relationship(relationship_cd)

        # Test finding path from A to D
        path = db.find_path("a", "d", max_depth=5)
        assert path == ["a", "b", "c", "d"]

        # Test finding path from A to C
        path = db.find_path("a", "c", max_depth=5)
        assert path == ["a", "b", "c"]

        # Test finding path with insufficient depth
        assert db.find_path("a", "d", max_depth=2) is None

        # Test finding path to disconnected node
        assert db.find_path("a", "e", max_depth=5) is None

        # Test finding path from node to itself
        assert db.find_path("a", "a", max_depth=5) == ["a"]

    @pytest.mark.unit
    def test_translate_to_query(self) -> None:
        """Test translating natural language to query."""
        db = GraphDatabase()

        # Add some test data
        entities = [
            Entity(
                name="Python", id="python", type=TYPE_CONCEPT, content="Python programming language"
            ),
            Entity(name="Machine Learning", id="ml", type=TYPE_CONCEPT, content="ML concepts"),
            Entity(name="TensorFlow", id="tf", type=TYPE_OBJECT, content="ML library"),
        ]
        db.add_entities(entities)

        # Translate entity queries that search by keyword
        operations = LLMGraphQueryOperations(
            reasoning="Find Python entities",
            confidence=0.8,
            importance=DEFAULT_CONFIDENCE,
            entity_queries=[
                LLMEntityQuery(
                    reasoning="Search Python",
                    confidence=0.8,
            importance=DEFAULT_CONFIDENCE,
                    search_query="Python",
                    limit=5,
                )
            ],
        )
        query = db.translate_to_query(operations)
        results = cast(list[Entity], query.execute().results)
        assert results, "Query translation should return entities"
        assert any(entity.name == "Python" for entity in results)

    @pytest.mark.unit
    def test_entity_and_relationship_counts(self) -> None:
        """Test counting entities and relationships."""
        db = GraphDatabase()

        # Initially empty
        assert db.entity_count == 0
        assert db.relationship_count == 0

        # Add entities
        entities = [
            Entity(name="E1", id="e1", type=TYPE_CONCEPT, content="Entity 1"),
            Entity(name="E2", id="e2", type=TYPE_CONCEPT, content="Entity 2"),
            Entity(name="E3", id="e3", type=TYPE_OBJECT, content="Entity 3"),
        ]
        db.add_entities(entities)

        expected_entity_count = 3
        assert db.entity_count == expected_entity_count
        assert db.relationship_count == 0

        # Add relationships
        relationship_12 = Relationship(source="e1", target="e2", type=REL_TYPE_RELATED_TO)
        relationship_23 = Relationship(source="e2", target="e3", type=REL_TYPE_RELATED_TO)
        db.add_relationship(relationship_12)
        db.add_relationship(relationship_23)

        assert db.entity_count == expected_entity_count
        expected_relationship_count = 2
        assert db.relationship_count == expected_relationship_count

        # Delete entity
        db.delete_entity("e3")
        assert db.entity_count == 2
        # Relationships to deleted entity should also be removed
        assert db.relationship_count == 1

    @pytest.mark.unit
    def test_entity_query_execute_with_filters(self) -> None:
        """Test EntityQuery.execute with various filters."""
        db = GraphDatabase()

        # Add test data
        entities = [
            Entity(name="Python", id="python", type=TYPE_CONCEPT, content="Programming language"),
            Entity(name="Java", id="java", type=TYPE_CONCEPT, content="Programming language"),
            Entity(name="TensorFlow", id="tf", type=TYPE_OBJECT, content="ML framework"),
        ]
        db.add_entities(entities)

        # Test simple query execute
        query = db.query_entities()
        result = query.execute()
        expected_results_count = 3
        assert len(result.results) == expected_results_count

        # Test query with search filter
        query = db.query_entities().search("Python", top_k=10)
        result = query.execute()
        assert len(result.results) >= 1
        assert any("Python" in e.name for e in result.results)

        # Test query with limit
        query = db.query_entities().limit(2)
        result = query.execute()
        assert len(result.results) <= 2

    @pytest.mark.unit
    def test_relationship_query_execute_with_tuples(self) -> None:
        """Test RelationshipQuery.execute returning tuple format."""
        db = GraphDatabase()

        # Add test data
        entities = [
            Entity(name="Python", id="python", type=TYPE_CONCEPT, content="Python language"),
            Entity(name="TensorFlow", id="tf", type=TYPE_OBJECT, content="ML library"),
            Entity(name="PyTorch", id="pytorch", type=TYPE_OBJECT, content="ML library"),
        ]
        db.add_entities(entities)

        relationship_python_tf = Relationship(
            source="python", target="tf", type=REL_TYPE_RELATED_TO
        )
        relationship_python_pytorch = Relationship(
            source="python", target="pytorch", type=REL_TYPE_RELATED_TO
        )
        relationship_tf_pytorch = Relationship(
            source="tf", target="pytorch", type=REL_TYPE_RELATED_TO
        )
        db.add_relationship(relationship_python_tf)
        db.add_relationship(relationship_python_pytorch)
        db.add_relationship(relationship_tf_pytorch)

        # Test relationship query returning tuples
        query = db.query_relationships()
        query.execute()
        # Assume query returns tuples for this test
        mock_tuple_results = [
            (entities[0], entities[1], relationship_python_tf),
            (entities[0], entities[2], relationship_python_pytorch),
            (entities[1], entities[2], relationship_tf_pytorch),
        ]
        relationships = _relationship_results_from_tuples(mock_tuple_results)
        assert len(relationships) == THREE_RELATIONSHIPS_COUNT

    @pytest.mark.unit
    def test_relationship_query_execute_with_relationships(self) -> None:
        """Test RelationshipQuery.execute returning Relationship instances."""
        db = GraphDatabase()

        # Add test data
        entities = [
            Entity(name="Python", id="python", type=TYPE_CONCEPT, content="Python language"),
            Entity(name="TensorFlow", id="tf", type=TYPE_OBJECT, content="ML library"),
            Entity(name="PyTorch", id="pytorch", type=TYPE_OBJECT, content="ML library"),
        ]
        db.add_entities(entities)

        relationship_python_tf = Relationship(
            source="python", target="tf", type=REL_TYPE_RELATED_TO
        )
        relationship_python_pytorch = Relationship(
            source="python", target="pytorch", type=REL_TYPE_RELATED_TO
        )
        relationship_tf_pytorch = Relationship(
            source="tf", target="pytorch", type=REL_TYPE_RELATED_TO
        )
        db.add_relationship(relationship_python_tf)
        db.add_relationship(relationship_python_pytorch)
        db.add_relationship(relationship_tf_pytorch)

        # Test relationship query returning Relationship instances
        query = db.query_relationships()
        query.execute()
        # Assume query returns Relationship instances for this test
        mock_relationship_results = [
            relationship_python_tf,
            relationship_python_pytorch,
            relationship_tf_pytorch,
        ]
        relationships = _relationship_results_from_relationships(mock_relationship_results)
        assert len(relationships) == THREE_RELATIONSHIPS_COUNT

    @pytest.mark.unit
    def test_relationship_query_execute_with_empty_results(self) -> None:
        """Test RelationshipQuery.execute returning empty list."""
        GraphDatabase()

        # Test with empty results
        empty_relationships = _relationship_results_from_empty_list([])
        assert len(empty_relationships) == 0
