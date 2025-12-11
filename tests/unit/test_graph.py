"""Robust tests for graph database core functionality, error handling, and edge cases."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, cast

import pytest

from blockether_foundation.graph import (
    Entity,
    EntityType,
    GraphDatabase,
    LLMEntityQuery,
    LLMGraphAddEntity,
    LLMGraphAddRelationship,
    LLMGraphDeleteEntity,
    LLMGraphDeleteRelationship,
    LLMGraphOperations,
    LLMGraphQueryOperations,
    LLMGraphUpdateEntity,
    Relationship,
    RelationType,
)
from blockether_foundation.graph.database import (
    EntityTypeFilter,
    RelationshipTypeFilter,
)

# Test Constants
DEFAULT_SEARCH_LIMIT = 10
TOP_K_FIVE = 5
TOP_K_THREE = 3
SEARCH_TERM_MACHINE_LEARNING = "machine learning"
SEARCH_TERM_PYTHON = "python"
SEARCH_TERM_BATCH = "batch"
SEARCH_TERM_SPECIAL = "special"
SEARCH_TERM_ALGORITHMS = "algorithms"
SEARCH_TERM_FRAMEWORK = "framework"
SEARCH_TERM_LEARNING = "learning"
SEARCH_TERM_MACHINE = "machine"
SEARCH_TERM_NONEXISTENT = "nonexistent_term_xyz"
SEARCH_TERM_ANYTHING = "anything"

# Entity Types
TYPE_CONCEPT: EntityType = "concept"
TYPE_TOOL: EntityType = "tool"
TYPE_LIBRARY: EntityType = "library"
TYPE_PERSON: EntityType = "person"

# Relationship Types
REL_TYPE_RELATED_TO: RelationType = "related_to"
REL_TYPE_USES: RelationType = "uses"
REL_TYPE_SIMILAR_TO: RelationType = "similar_to"
REL_TYPE_IMPLEMENTS: RelationType = "implements"
REL_TYPE_INVALIDATES: RelationType = "invalidates"
REL_TYPE_CREATED_BY: RelationType = "created_by"
REL_TYPE_BELONGS_TO: RelationType = "belongs_to"

# Entity Names
NAME_MACHINE_LEARNING = "Machine Learning"
NAME_FULL_TEXT_SEARCH = "Full-text Search"
NAME_PYTHON_ML_LIBRARIES = "Python ML Libraries"
NAME_HIGHLY_RELEVANT = "Highly Relevant"
NAME_LESS_RELEVANT = "Less Relevant"
NAME_MEDIUM_RELEVANT = "Medium Relevant"
NAME_ML_CONCEPT = "ML Concept"
NAME_PYTHON_LIBRARY = "Python Library"
NAME_ML_TOOL = "ML Tool"
NAME_ML_BASIC = "ML Basic"
NAME_ML_ADVANCED = "ML Advanced"
NAME_AI_ML_TOOL = "AI ML Tool"
NAME_AI_ML_CONCEPT = "AI ML Concept"
NAME_PYTHON_ML_LIBRARY = "Python ML Library"

# Test Values
DEFAULT_CONFIDENCE = 0.9
TEST_TIMESTAMP = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

# Test Count Constants
THREE_ENTITIES = 3
THREE_NEIGHBORS = 3
THREE_OPERATIONS = 3
FOUR_DIRECTIONS = 4
FIVE_RELATED_ENTITIES = 5
TWO_RESULTS = 2
THREE_RESULTS = 3


def _relationships_from_results(
    results: list[tuple[Entity, Entity, Relationship]] | list[object],
) -> list[Relationship]:
    """Helper to extract Relationship instances from tuple-based query results."""
    relationship_tuples = cast(list[tuple[Entity, Entity, Relationship]], results)
    return [rel for _, _, rel in relationship_tuples]


class TestGraphDatabaseCore:
    """Test core GraphDatabase functionality with comprehensive error handling."""

    @pytest.mark.unit
    def test_database_initialization_state(self: TestGraphDatabaseCore) -> None:
        """Test that GraphDatabase starts in clean, predictable state."""
        db = GraphDatabase()

        assert db.entity_count == 0, "New database should have 0 entities"
        assert db.relationship_count == 0, "New database should have 0 relationships"

        # Search on empty database should return empty results
        search_results = db.search(SEARCH_TERM_ANYTHING, top_k=TOP_K_FIVE)
        assert len(search_results) == 0, "Search on empty database should return no results"

        # Query on empty database should return empty results
        query_results = db.query_entities().execute().results
        assert len(query_results) == 0, "Query on empty database should return no results"

    @pytest.mark.unit
    def test_entity_crud_operations_comprehensive(self: TestGraphDatabaseCore) -> None:
        """Test comprehensive entity CRUD operations with error handling."""
        db = GraphDatabase()

        # Test entity creation
        entity = Entity(
            name=NAME_MACHINE_LEARNING,
            type=TYPE_CONCEPT,
            content="Machine learning algorithms and neural networks",
        )
        db.add_entity(entity)  # Returns None

        # Test entity retrieval
        retrieved_entity = db.get_entity(entity.id)
        assert retrieved_entity is not None, "Entity should be retrievable after creation"
        assert retrieved_entity.id == entity.id, "Retrieved entity should have correct ID"
        assert retrieved_entity.name == entity.name, "Retrieved entity should have correct name"
        assert retrieved_entity.type == entity.type, "Retrieved entity should have correct type"
        assert retrieved_entity.content == entity.content, (
            "Retrieved entity should have correct content"
        )
        assert retrieved_entity.created_at is not None, (
            "Retrieved entity should have created_at timestamp"
        )
        assert retrieved_entity.updated_at is not None, (
            "Retrieved entity should have updated_at timestamp"
        )

        # Test entity update
        updated_content = "Updated machine learning content with more details"
        updated_entity = Entity(
            id=entity.id, name=entity.name, type=entity.type, content=updated_content
        )
        db.update_entity(updated_entity)

        final_entity = db.get_entity(entity.id)
        assert final_entity is not None, "Updated entity should still exist"
        assert final_entity.content == updated_content, "Entity content should be updated"
        assert final_entity.updated_at > retrieved_entity.updated_at, "updated_at should increase"

        # Test entity deletion
        db.delete_entity(entity.id)
        deleted_entity = db.get_entity(entity.id)
        assert deleted_entity is None, "Deleted entity should not be retrievable"
        assert db.entity_count == 0, "Entity count should be 0 after deletion"

    @pytest.mark.unit
    def test_entity_validation_and_error_handling(self: TestGraphDatabaseCore) -> None:
        """Test entity validation and comprehensive error handling."""
        db = GraphDatabase()

        # Test duplicate entity creation by ID (not by name - names can be duplicated)
        entity = Entity(name="Test Entity", id="test_id", type=TYPE_CONCEPT, content="Test content")
        db.add_entity(entity)

        # Test duplicate entity by ID should fail
        with pytest.raises(ValueError, match="already exist"):
            db.add_entity(
                Entity(
                    name="Test Entity", id="test_id", type=TYPE_TOOL, content="Different content"
                )
            )

        # Test update of non-existent entity
        non_existent_entity = Entity(
            id="nonexistent", name="Non-existent", type=TYPE_CONCEPT, content="Does not exist"
        )
        with pytest.raises(ValueError, match="does not exist"):
            db.update_entity(non_existent_entity)

        # Test deletion of non-existent entity
        with pytest.raises(ValueError, match="does not exist"):
            db.delete_entity("nonexistent")

        # Verify that the original entity still exists and database state is consistent
        final_entity = db.get_entity("test_id")
        assert final_entity is not None, (
            "Original entity should still exist after failed operations"
        )
        assert final_entity.name == "Test Entity", "Original entity name should be preserved"
        assert db.entity_count == 1, (
            "Database should contain exactly one entity after all operations"
        )

    @pytest.mark.unit
    def test_batch_operations_robustness(self: TestGraphDatabaseCore) -> None:
        """Test batch operations with comprehensive error handling and rollback."""
        db = GraphDatabase()

        # Test successful batch operation
        entities = [
            Entity(name="Entity 1", id="entity1", type=TYPE_CONCEPT, content="First entity"),
            Entity(name="Entity 2", id="entity2", type=TYPE_TOOL, content="Second entity"),
            Entity(name="Entity 3", id="entity3", type=TYPE_LIBRARY, content="Third entity"),
        ]

        initial_count = db.entity_count
        db.add_entities(entities)

        assert db.entity_count == initial_count + THREE_ENTITIES, "All entities should be added"

        # Verify all entities exist using accumulation pattern
        retrieved_entities = [db.get_entity(entity.id) for entity in entities]
        entity_existence_flags = [retrieved is not None for retrieved in retrieved_entities]
        entity_name_flags = [
            retrieved.name == entity.name
            for retrieved, entity in zip(retrieved_entities, entities, strict=True)
            if retrieved is not None
        ]

        assert all(entity_existence_flags), "All entities should exist"
        assert len(entity_name_flags) == len(entities), "All entities should have correct names"

        # Test batch operation with duplicate - should fail completely
        duplicate_entities = [
            Entity(name="New Entity", id="new_entity", type=TYPE_CONCEPT, content="New content"),
            Entity(name="Entity 1", id="entity1", type=TYPE_TOOL, content="Duplicate content"),
        ]

        with pytest.raises(ValueError, match="already exist"):
            db.add_entities(duplicate_entities)

        # Ensure no partial addition occurred
        new_entity = db.get_entity("new_entity")
        assert new_entity is None, "No entities should be added in failed batch operation"
        assert db.entity_count == initial_count + THREE_ENTITIES, "Entity count should be unchanged"

    @pytest.mark.unit
    def test_search_functionality_comprehensive(self) -> None:
        """Test search functionality with comprehensive scenarios."""
        db = GraphDatabase()

        # Create entities with varying content for search testing
        entities = [
            Entity(
                name=NAME_MACHINE_LEARNING,
                type=TYPE_CONCEPT,
                content="Machine learning algorithms and neural networks for artificial intelligence",
            ),
            Entity(
                name=NAME_PYTHON_ML_LIBRARIES,
                type=TYPE_LIBRARY,
                content="Python libraries specifically designed for machine learning applications",
            ),
            Entity(
                name=NAME_FULL_TEXT_SEARCH,
                type=TYPE_TOOL,
                content="Full-text search engines for information retrieval systems",
            ),
            Entity(
                name=NAME_HIGHLY_RELEVANT,
                type=TYPE_CONCEPT,
                content=f"{SEARCH_TERM_MACHINE_LEARNING} {SEARCH_TERM_MACHINE_LEARNING} {SEARCH_TERM_MACHINE_LEARNING}",
            ),
        ]

        db.add_entities(entities)

        # Test basic search functionality
        results = db.search(SEARCH_TERM_MACHINE_LEARNING, top_k=DEFAULT_SEARCH_LIMIT)
        assert len(results) >= 2, "Should find multiple entities for 'machine learning'"

        # Verify search result structure using accumulation pattern
        entity_types = [isinstance(entity, Entity) for entity, _ in results]
        score_types = [isinstance(score, float) for _, score in results]
        score_positivity = [score > 0 for _, score in results]
        content_contains_term = [
            SEARCH_TERM_MACHINE_LEARNING.lower() in (entity.content or "").lower()
            for entity, _ in results
        ]

        assert all(entity_types), "Search results should contain Entity objects"
        assert all(score_types), "Search scores should be floats"
        assert all(score_positivity), "Search scores should be positive"
        assert all(content_contains_term), "Entity content should contain search term"

        # Test search result ordering by relevance
        highly_relevant_results = [r for r in results if r[0].name == NAME_HIGHLY_RELEVANT]
        assert len(highly_relevant_results) > 0, "Highly relevant entity should be in results"

        # Most relevant should have highest score
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        assert sorted_results[0][0].name == NAME_HIGHLY_RELEVANT, "Most relevant should be first"

        # Test case-insensitive search
        lower_results = db.search(SEARCH_TERM_MACHINE.lower(), top_k=DEFAULT_SEARCH_LIMIT)
        upper_results = db.search(SEARCH_TERM_MACHINE.upper(), top_k=DEFAULT_SEARCH_LIMIT)

        assert len(lower_results) == len(upper_results), "Case should not affect search results"
        lower_ids = {entity.id for entity, _ in lower_results}
        upper_ids = {entity.id for entity, _ in upper_results}
        assert lower_ids == upper_ids, "Same entities should be found regardless of case"

        # Test search with no results
        no_results = db.search(SEARCH_TERM_NONEXISTENT, top_k=TOP_K_FIVE)
        assert len(no_results) == 0, "Search for non-existent term should return no results"

    @pytest.mark.unit
    def test_search_edge_cases_and_special_characters(self) -> None:
        """Test search edge cases including special characters and unicode."""
        db = GraphDatabase()

        # Test with special characters
        special_entity = Entity(
            name="Special Characters",
            type=TYPE_CONCEPT,
            content="Content with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?",
        )
        db.add_entity(special_entity)

        results = db.search("special chars", top_k=TOP_K_FIVE)
        assert len(results) == 1, "Should find entity with special characters"
        assert results[0][0].id == special_entity.id, "Should find the correct entity"

        # Test with unicode characters
        unicode_entity = Entity(
            name="Unicode Test",
            type=TYPE_CONCEPT,
            content="Unicode content: ñáéíóú 中文 русский العربية",
        )
        db.add_entity(unicode_entity)

        unicode_results = db.search("ñáéíóú", top_k=TOP_K_FIVE)
        assert len(unicode_results) == 1, "Should find entity with unicode characters"

        # Test empty content
        empty_content_entity = Entity(name="Empty Content", type=TYPE_CONCEPT, content="")
        db.add_entity(empty_content_entity)

        empty_results = db.search("", top_k=TOP_K_FIVE)
        assert len(empty_results) == 0, "Empty search should return no results"


class TestRelationshipManagement:
    """Test comprehensive relationship management with error handling."""

    @pytest.mark.unit
    def test_relationship_crud_comprehensive(self) -> None:
        """Test complete relationship CRUD with error handling."""
        db = GraphDatabase()

        # Create test entities
        entity1 = Entity(name="Source Entity", id="source", type=TYPE_CONCEPT, content="Source")
        entity2 = Entity(name="Target Entity", id="target", type=TYPE_TOOL, content="Target")
        db.add_entities([entity1, entity2])

        # Test relationship creation
        relationship = Relationship(source=entity1.id, target=entity2.id, type=REL_TYPE_USES)

        db.add_relationship(relationship)  # Returns None
        # Test relationship retrieval
        retrieved_rel = db.get_relationship_by_entities(entity1.id, entity2.id)
        assert retrieved_rel is not None, "Relationship should be retrievable"
        assert retrieved_rel.source == entity1.id, (
            "Retrieved relationship should have correct source"
        )
        assert retrieved_rel.target == entity2.id, (
            "Retrieved relationship should have correct target"
        )
        assert retrieved_rel.type == REL_TYPE_USES, "Relationship type should be preserved"
        assert retrieved_rel.created_at is not None, "Relationship should have created_at"
        assert retrieved_rel.updated_at is not None, "Relationship should have updated_at"

        # Test relationship update
        updated_rel = Relationship(
            source=entity1.id, target=entity2.id, type=REL_TYPE_SIMILAR_TO, id=retrieved_rel.id
        )

        db.update_relationship(updated_rel)
        final_rel = db.get_relationship_by_entities(entity1.id, entity2.id)
        assert final_rel is not None, "Updated relationship should still exist"
        assert final_rel.type == REL_TYPE_SIMILAR_TO, "Relationship type should be updated"

        # Test relationship deletion
        db.delete_relationship_by_entities(entity1.id, entity2.id)
        deleted_rel = db.get_relationship_by_entities(entity1.id, entity2.id)
        assert deleted_rel is None, "Deleted relationship should not be retrievable"

    @pytest.mark.unit
    def test_relationship_cascade_delete_comprehensive(self) -> None:
        """Test that relationship deletion cascades properly with entity deletion."""
        db = GraphDatabase()

        # Create entities and relationships
        entities = [
            Entity(name="Central Entity", id="central", type=TYPE_CONCEPT, content="Central"),
            Entity(name="Connected 1", id="conn1", type=TYPE_TOOL, content="Connected 1"),
            Entity(name="Connected 2", id="conn2", type=TYPE_LIBRARY, content="Connected 2"),
            Entity(name="Connected 3", id="conn3", type=TYPE_CONCEPT, content="Connected 3"),
        ]

        db.add_entities(entities)

        # Create relationships in various directions
        relationships: list[tuple[str, str, RelationType]] = [
            (entities[0].id, entities[1].id, REL_TYPE_USES),  # central -> conn1
            (entities[2].id, entities[0].id, REL_TYPE_IMPLEMENTS),  # conn2 -> central
            (entities[0].id, entities[3].id, REL_TYPE_RELATED_TO),  # central -> conn3
        ]

        # Create relationships using accumulation pattern
        [
            db.create_relationship(source, target, rel_type)
            for source, target, rel_type in relationships
        ]

        # Verify all relationships were created successfully by checking relationship count
        assert db.relationship_count == len(relationships), "All relationships should be created"

        assert db.relationship_count == THREE_ENTITIES, "Should have 3 relationships"

        # Delete central entity - should cascade delete all relationships
        db.delete_entity("central")

        assert db.relationship_count == 0, "All relationships should be deleted after cascade"

        # Verify entities still exist using accumulation pattern
        entity_ids = ["conn1", "conn2", "conn3"]
        retrieved_entities = [db.get_entity(entity_id) for entity_id in entity_ids]
        entity_existence_flags = [entity is not None for entity in retrieved_entities]

        assert all(entity_existence_flags), "All entities should still exist"

    @pytest.mark.unit
    def test_neighbor_queries_comprehensive(self) -> None:
        """Test neighbor query functionality with all parameters and edge cases."""
        db = GraphDatabase()

        # Create a more complex graph structure
        entities = [
            Entity(name="Hub Entity", id="hub", type=TYPE_CONCEPT, content="Central hub"),
            Entity(name="Node 1", id="node1", type=TYPE_TOOL, content="First node"),
            Entity(name="Node 2", id="node2", type=TYPE_LIBRARY, content="Second node"),
            Entity(name="Node 3", id="node3", type=TYPE_CONCEPT, content="Third node"),
            Entity(name="Node 4", id="node4", type=TYPE_TOOL, content="Fourth node"),
        ]

        db.add_entities(entities)

        # Create relationships with different types and directions
        relationships: list[tuple[str, str, RelationType]] = [
            ("hub", "node1", REL_TYPE_USES),
            ("hub", "node2", REL_TYPE_RELATED_TO),
            ("node3", "hub", REL_TYPE_IMPLEMENTS),
            ("hub", "node4", REL_TYPE_SIMILAR_TO),
            ("node1", "node2", REL_TYPE_USES),
        ]

        # Create relationships using accumulation pattern
        [
            db.create_relationship(source, target, rel_type)
            for source, target, rel_type in relationships
        ]

        # Verify all relationships were created successfully by checking relationship count
        assert db.relationship_count == len(relationships), "All relationships should be created"

        # Test outgoing neighbors
        outgoing = db.get_neighbors("hub", direction="outgoing")
        assert len(outgoing) == THREE_ENTITIES, "Hub should have 3 outgoing neighbors"
        outgoing_ids = {e.id for e in outgoing}
        assert outgoing_ids == {"node1", "node2", "node4"}, "Outgoing neighbors should be correct"

        # Test incoming neighbors
        incoming = db.get_neighbors("hub", direction="incoming")
        assert len(incoming) == 1, "Hub should have 1 incoming neighbor"
        assert incoming[0].id == "node3", "Incoming neighbor should be node3"

        # Test both directions
        both_directions = db.get_neighbors("hub", direction="both")
        assert len(both_directions) == FOUR_DIRECTIONS, (
            "Hub should have 4 neighbors in both directions"
        )
        both_ids = {e.id for e in both_directions}
        assert both_ids == {"node1", "node2", "node3", "node4"}, (
            "Both directions should include all"
        )

        # Test filtered by relationship type
        uses_neighbors = db.get_neighbors("hub", relationship_type=REL_TYPE_USES)
        assert len(uses_neighbors) == 1, "Hub should have 1 'uses' neighbor"
        assert uses_neighbors[0].id == "node1", "Uses neighbor should be node1"

        # Test invalid direction
        with pytest.raises(ValueError, match="Invalid direction"):
            db.get_neighbors("hub", direction="invalid")

        # Test neighbors of non-existent entity
        with pytest.raises(ValueError, match="does not exist"):
            db.get_neighbors("nonexistent")

    @pytest.mark.unit
    def test_path_finding_comprehensive(self) -> None:
        """Test path finding with complex scenarios and edge cases."""
        db = GraphDatabase()

        # Create a complex graph with multiple paths using explicit entity creation
        NODE_0 = Entity(name="Node 0", id="node0", type=TYPE_CONCEPT, content="Node 0")
        NODE_1 = Entity(name="Node 1", id="node1", type=TYPE_CONCEPT, content="Node 1")
        NODE_2 = Entity(name="Node 2", id="node2", type=TYPE_CONCEPT, content="Node 2")
        NODE_3 = Entity(name="Node 3", id="node3", type=TYPE_CONCEPT, content="Node 3")
        NODE_4 = Entity(name="Node 4", id="node4", type=TYPE_CONCEPT, content="Node 4")
        NODE_5 = Entity(name="Node 5", id="node5", type=TYPE_CONCEPT, content="Node 5")
        NODE_6 = Entity(name="Node 6", id="node6", type=TYPE_CONCEPT, content="Node 6")

        nodes = [NODE_0, NODE_1, NODE_2, NODE_3, NODE_4, NODE_5, NODE_6]
        db.add_entities(nodes)

        # Create paths:
        # node0 -> node1 -> node2 -> node3 (shortest path)
        # node0 -> node4 -> node5 -> node3 (alternative path)
        # node0 -> node6 -> node1 (cycle)
        # node2 -> node4 (cross connection)
        paths: list[tuple[str, str, RelationType]] = [
            ("node0", "node1", REL_TYPE_RELATED_TO),
            ("node1", "node2", REL_TYPE_USES),
            ("node2", "node3", REL_TYPE_IMPLEMENTS),
            ("node0", "node4", REL_TYPE_RELATED_TO),
            ("node4", "node5", REL_TYPE_USES),
            ("node5", "node3", REL_TYPE_IMPLEMENTS),
            ("node0", "node6", REL_TYPE_RELATED_TO),
            ("node6", "node1", REL_TYPE_USES),
            ("node2", "node4", REL_TYPE_SIMILAR_TO),
        ]

        # Create paths using accumulation pattern
        [db.create_relationship(source, target, rel_type) for source, target, rel_type in paths]

        # Verify all paths were created successfully by checking relationship count
        assert db.relationship_count == len(paths), "All paths should be created"

        # Test shortest path finding
        path_0_to_3 = db.find_path("node0", "node3")
        assert path_0_to_3 is not None, "Should find a path from node0 to node3"
        assert path_0_to_3[0] == "node0", "Path should start at node0"
        assert path_0_to_3[-1] == "node3", "Path should end at node3"
        assert len(path_0_to_3) >= FOUR_DIRECTIONS, (
            "Path should have at least 4 nodes due to graph structure"
        )

        # Test alternative path when shortest is blocked
        path_0_to_5 = db.find_path("node0", "node5")
        assert path_0_to_5 == ["node0", "node4", "node5"], "Should find alternative path"

        # Test path with cycle handling
        path_6_to_3 = db.find_path("node6", "node3")
        assert path_6_to_3 == ["node6", "node1", "node2", "node3"], "Should handle cycles correctly"

        # Test same entity path
        same_entity_path = db.find_path("node0", "node0")
        assert same_entity_path == ["node0"], "Path to same entity should return single node"

        # Test no path exists
        isolated = Entity(name="Isolated", id="isolated", type=TYPE_CONCEPT, content="Isolated")
        db.add_entity(isolated)

        no_path = db.find_path("isolated", "node0")
        assert no_path is None, "Should return None when no path exists"

        # Test path finding with non-existent entities
        with pytest.raises(ValueError, match="does not exist"):
            db.find_path("nonexistent", "node0")

        db.add_entity(Entity(name="Exists", id="exists", type=TYPE_CONCEPT, content="Exists"))
        with pytest.raises(ValueError, match="does not exist"):
            db.find_path("exists", "nonexistent")


class TestLLMGraphOperationsRobust:
    """Test LLMGraphOperations with comprehensive validation and error scenarios."""

    @pytest.mark.unit
    def test_operations_validation_comprehensive(self) -> None:
        """Test LLMGraphOperations validation with comprehensive scenarios."""

        # Test valid operations
        valid_add = LLMGraphAddEntity(
            name="Valid Entity",
            type=TYPE_CONCEPT,
            content="Valid content",
            reasoning="Valid operation",
            confidence=DEFAULT_CONFIDENCE,
        )

        valid_update = LLMGraphUpdateEntity(
            id="existing_id",
            name="Updated Entity",
            type=TYPE_TOOL,
            content="Updated content",
            reasoning="Valid update",
            confidence=DEFAULT_CONFIDENCE,
        )

        valid_rel = LLMGraphAddRelationship(
            source_name="Entity 1",
            target_name="Entity 2",
            type=REL_TYPE_USES,
            reasoning="Valid relationship",
            confidence=DEFAULT_CONFIDENCE,
        )

        valid_ops = LLMGraphOperations(
            add_entity_ops=[valid_add],
            update_entity_ops=[valid_update],
            delete_entity_ops=[],
            add_relationship_ops=[valid_rel],
            delete_relationship_ops=[],
            reasoning="Valid operations",
            confidence=DEFAULT_CONFIDENCE,
        )

        result = valid_ops.ops
        assert len(result) == THREE_OPERATIONS, "Valid operations should return all operations"

        # Test duplicate entity in add operations
        duplicate_add_1 = LLMGraphAddEntity(
            name="Duplicate Entity",
            type=TYPE_CONCEPT,
            content="First occurrence",
            reasoning="First",
            confidence=DEFAULT_CONFIDENCE,
        )

        duplicate_add_2 = LLMGraphAddEntity(
            name="Duplicate Entity",
            type=TYPE_TOOL,
            content="Second occurrence",
            reasoning="Second",
            confidence=DEFAULT_CONFIDENCE,
        )

        duplicate_ops = LLMGraphOperations(
            add_entity_ops=[duplicate_add_1, duplicate_add_2],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[],
            delete_relationship_ops=[],
            reasoning="Duplicate test",
            confidence=DEFAULT_CONFIDENCE,
        )

        with pytest.raises(
            ValueError, match="Duplicate add operation for entity name 'Duplicate Entity'"
        ):
            _ = duplicate_ops.ops

        # Test duplicate relationship in add operations
        rel1 = LLMGraphAddRelationship(
            source_name="Entity A",
            target_name="Entity B",
            type=REL_TYPE_USES,
            reasoning="First",
            confidence=DEFAULT_CONFIDENCE,
        )

        rel2 = LLMGraphAddRelationship(
            source_name="Entity A",
            target_name="Entity B",
            type=REL_TYPE_RELATED_TO,
            reasoning="Second",
            confidence=DEFAULT_CONFIDENCE,
        )

        duplicate_rel_ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[rel1, rel2],
            delete_relationship_ops=[],
            reasoning="Duplicate rel test",
            confidence=DEFAULT_CONFIDENCE,
        )

        # Relationship duplicates are deduplicated automatically rather than raising
        flattened_ops = duplicate_rel_ops.ops
        relationship_ops = [op for op in flattened_ops if isinstance(op, LLMGraphAddRelationship)]
        assert len(relationship_ops) == 1, "Duplicate relationships should collapse to one op"
        assert relationship_ops[0].type == REL_TYPE_USES, "First relationship should be kept"

    @pytest.mark.unit
    def test_operations_execution_comprehensive(self) -> None:
        """Test LLMGraphOperations execution with comprehensive scenarios."""
        db = GraphDatabase()

        # Test execution of valid operations
        add_ops = [
            LLMGraphAddEntity(
                name="Python",
                type=TYPE_TOOL,
                content="Programming language",
                reasoning="Add Python",
                confidence=DEFAULT_CONFIDENCE,
            ),
            LLMGraphAddEntity(
                name="Django",
                type=TYPE_LIBRARY,
                content="Web framework",
                reasoning="Add Django",
                confidence=DEFAULT_CONFIDENCE,
            ),
        ]

        rel_ops = [
            LLMGraphAddRelationship(
                source_name="Django",
                target_name="Python",
                type=REL_TYPE_USES,
                reasoning="Django uses Python",
                confidence=DEFAULT_CONFIDENCE,
            )
        ]

        ops = LLMGraphOperations(
            add_entity_ops=add_ops,
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=rel_ops,
            delete_relationship_ops=[],
            reasoning="Add Python and Django with relationship",
            confidence=DEFAULT_CONFIDENCE,
        )

        # Execute operations
        db.import_operations(ops)

        # Verify entities were added
        python = db.get_entity_by_name("Python")
        django = db.get_entity_by_name("Django")

        assert python is not None, "Python entity should exist"
        assert python.type == TYPE_TOOL, "Python should be a tool"
        assert django is not None, "Django entity should exist"
        assert django.type == TYPE_LIBRARY, "Django should be a library"

        # Verify relationship was created
        relationship = db.get_relationship_by_entities(django.id, python.id)
        assert relationship is not None, "Django-Python relationship should exist"
        assert relationship.type == REL_TYPE_USES, "Relationship should be 'uses'"

    @pytest.mark.unit
    def test_operations_error_handling_comprehensive(self) -> None:
        """Test LLMGraphOperations error handling in various failure scenarios."""
        db = GraphDatabase()

        # Add initial entity for update tests
        existing = Entity(name="Existing", type=TYPE_CONCEPT, content="Original content")
        db.add_entity(existing)

        # Test update of non-existent entity
        update_nonexistent = LLMGraphUpdateEntity(
            id="nonexistent_id",
            name="Fake",
            type=TYPE_TOOL,
            content="Fake content",
            reasoning="Update nonexistent",
            confidence=DEFAULT_CONFIDENCE,
        )

        update_ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[update_nonexistent],
            delete_entity_ops=[],
            add_relationship_ops=[],
            delete_relationship_ops=[],
            reasoning="Test update error",
            confidence=DEFAULT_CONFIDENCE,
        )

        with pytest.raises(ValueError, match="Entity with id 'nonexistent_id' does not exist"):
            db.import_operations(update_ops)

        # Test relationship to non-existent entity
        db.add_entity(Entity(name="Source", type=TYPE_CONCEPT, content="Source entity"))

        rel_to_nonexistent = LLMGraphAddRelationship(
            source_name="Source",
            target_name="NonexistentTarget",
            type=REL_TYPE_USES,
            reasoning="Invalid relationship",
            confidence=DEFAULT_CONFIDENCE,
        )

        rel_ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[rel_to_nonexistent],
            delete_relationship_ops=[],
            reasoning="Test relationship error",
            confidence=DEFAULT_CONFIDENCE,
        )

        with pytest.raises(
            ValueError, match="Target entity with name 'NonexistentTarget' not found"
        ):
            db.import_operations(rel_ops)

        # Verify database state remains consistent after failed operation
        assert db.entity_count == 2, (
            "Database should still contain 2 entities after failed relationship operation"
        )
        assert db.relationship_count == 0, (
            "Database should still have 0 relationships after failed operation"
        )


class TestGraphSerializationAndRecovery:
    """Test graph serialization with comprehensive scenarios and error handling."""

    @pytest.mark.unit
    def test_serialization_comprehensive(self) -> None:
        """Test comprehensive serialization and deserialization."""
        import json

        # Create complex database
        original_db = GraphDatabase()

        entities = [
            Entity(
                name="Complex Entity 1",
                id="complex1",
                type=TYPE_CONCEPT,
                content="Complex content with special chars: !@#$%^&*()",
                created_at=TEST_TIMESTAMP,
            ),
            Entity(
                name="Complex Entity 2",
                id="complex2",
                type=TYPE_TOOL,
                content="Unicode content: ñáéíóú 中文 русский العربية",
            ),
        ]

        original_db.add_entities(entities)

        # Add relationships
        original_db.create_relationship("complex1", "complex2", REL_TYPE_USES)

        # Test serialization
        serialized = original_db.to_dict()

        # Verify structure using accumulation pattern
        required_keys = ["entities", "relationships", "index", "search_index_initialized"]
        key_presence_flags = [key in serialized for key in required_keys]

        assert all(key_presence_flags), "All required keys should be present in serialized data"

        entities_data: list[dict[str, Any]] = cast(
            list[dict[str, Any]], serialized.get("entities", [])
        )
        relationships_data: list[dict[str, Any]] = cast(
            list[dict[str, Any]], serialized.get("relationships", [])
        )
        assert len(entities_data) == 2, "Should serialize 2 entities"
        assert len(relationships_data) == 1, "Should serialize 1 relationship"
        assert serialized["search_index_initialized"] is True, "Search index should be initialized"

        # Test JSON roundtrip
        json_str = json.dumps(serialized, indent=2, default=str)
        deserialized_data = json.loads(json_str)

        # Test deserialization
        restored_db = GraphDatabase.from_dict(deserialized_data)

        # Verify restoration
        assert restored_db.entity_count == original_db.entity_count, "Entity counts should match"
        assert restored_db.relationship_count == original_db.relationship_count, (
            "Relationship counts should match"
        )

        # Verify entities were restored correctly using accumulation pattern
        restored_entities = [restored_db.get_entity(entity.id) for entity in entities]
        entity_restoration_flags = [restored is not None for restored in restored_entities]
        entity_name_flags = [
            restored.name == entity.name
            for restored, entity in zip(restored_entities, entities, strict=True)
            if restored is not None
        ]
        entity_type_flags = [
            restored.type == entity.type
            for restored, entity in zip(restored_entities, entities, strict=True)
            if restored is not None
        ]
        entity_content_flags = [
            restored.content == entity.content
            for restored, entity in zip(restored_entities, entities, strict=True)
            if restored is not None
        ]

        assert all(entity_restoration_flags), "All entities should be restored"
        assert all(entity_name_flags), "All entity names should match"
        assert all(entity_type_flags), "All entity types should match"
        assert all(entity_content_flags), "All entity content should match"

        # Verify relationships were restored
        restored_rel = restored_db.get_relationship_by_entities("complex1", "complex2")
        assert restored_rel is not None, "Relationship should be restored"
        assert restored_rel.type == REL_TYPE_USES, "Relationship type should match"

        # Test search functionality after restoration
        search_results = restored_db.search("complex", top_k=TOP_K_FIVE)
        assert len(search_results) == 2, "Search should work after restoration"

        restored_ids = {entity.id for entity, _ in search_results}
        original_ids = {entity.id for entity, _ in original_db.search("complex", top_k=TOP_K_FIVE)}
        assert restored_ids == original_ids, "Search results should match after restoration"

    @pytest.mark.unit
    def test_serialization_edge_cases(self) -> None:
        """Test serialization edge cases and error handling."""
        import json

        # Test empty database serialization
        empty_db = GraphDatabase()
        empty_serialized = empty_db.to_dict()

        json_str = json.dumps(empty_serialized, indent=2, default=str)
        deserialized_data = json.loads(json_str)
        restored_empty = GraphDatabase.from_dict(deserialized_data)

        assert restored_empty.entity_count == 0, "Empty database should stay empty"
        assert restored_empty.relationship_count == 0, "Empty database should have no relationships"

        # Test database with only entities (no relationships)
        entities_only_db = GraphDatabase()
        entities_only_db.add_entity(
            Entity(name="Solo Entity", type=TYPE_CONCEPT, content="Only entity")
        )

        entities_serialized = entities_only_db.to_dict()
        json_str = json.dumps(entities_serialized, indent=2, default=str)
        deserialized_data = json.loads(json_str)
        restored_entities_only = GraphDatabase.from_dict(deserialized_data)

        assert restored_entities_only.entity_count == 1, "Should restore single entity"
        assert restored_entities_only.relationship_count == 0, "Should have no relationships"

        # ignore-development
        # Verify serialization with minimal valid data
        # ignore-development
        minimal_valid_data: dict[str, Any] = {
            "entities": [],
            "relationships": [],
            "index": {},
            "search_index_initialized": False,
        }
        # ignore-development
        minimal_db = GraphDatabase.from_dict(minimal_valid_data)
        # ignore-development
        assert minimal_db.entity_count == 0, "Minimal valid data should create empty database"
        # ignore-development
        assert minimal_db.relationship_count == 0, (
            # ignore-development
            "Minimal valid data should create no relationships"
        )


class TestLLMGraphDeleteOperations:
    """Test LLMGraphDeleteEntity and LLMGraphDeleteRelationship operations with comprehensive validation."""

    @pytest.mark.unit
    def test_llm_graph_delete_entity_validation_comprehensive(self) -> None:
        """Test LLMGraphDeleteEntity validation with comprehensive scenarios."""

        # Test valid delete entity operation creation
        valid_delete = LLMGraphDeleteEntity(
            entity_id="test_entity_id",
            reasoning="Delete test entity",
            confidence=DEFAULT_CONFIDENCE,
        )

        assert valid_delete.entity_id == "test_entity_id", "Entity ID should be set correctly"
        assert valid_delete.operation == "delete_entity", "Operation should be delete_entity"
        assert valid_delete.reasoning == "Delete test entity", "Reasoning should be preserved"
        assert valid_delete.confidence == DEFAULT_CONFIDENCE, "Confidence should be preserved"

        # Test with empty entity ID
        empty_delete = LLMGraphDeleteEntity(
            entity_id="",
            reasoning="Delete empty entity",
            confidence=DEFAULT_CONFIDENCE,
        )
        assert empty_delete.entity_id == "", "Empty entity ID should be allowed in validation"

        # Test with special characters in entity ID
        special_delete = LLMGraphDeleteEntity(
            entity_id="entity_with_special_chars_123",
            reasoning="Delete special entity",
            confidence=DEFAULT_CONFIDENCE,
        )
        assert "entity_with_special_chars_123" in special_delete.entity_id, (
            "Special characters should be allowed"
        )

    @pytest.mark.unit
    def test_llm_graph_delete_entity_execution_comprehensive(self) -> None:
        """Test LLMGraphDeleteEntity execution with comprehensive scenarios."""
        db = GraphDatabase()

        # Create test entities
        entity1 = Entity(name="Test Entity 1", type=TYPE_CONCEPT, content="Test content 1")
        entity2 = Entity(name="Test Entity 2", type=TYPE_TOOL, content="Test content 2")
        entity3 = Entity(name="Test Entity 3", type=TYPE_LIBRARY, content="Test content 3")

        db.add_entities([entity1, entity2, entity3])

        # Create relationships to test cascade deletion
        db.create_relationship(entity1.id, entity2.id, REL_TYPE_USES)
        db.create_relationship(entity3.id, entity1.id, REL_TYPE_RELATED_TO)

        initial_entity_count = db.entity_count
        initial_relationship_count = db.relationship_count

        # Test deletion of entity with relationships (should cascade)
        delete_op = LLMGraphDeleteEntity(
            entity_id=entity1.id,
            reasoning="Delete entity with relationships",
            confidence=DEFAULT_CONFIDENCE,
        )

        ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[],
            delete_entity_ops=[delete_op],
            add_relationship_ops=[],
            delete_relationship_ops=[],
            reasoning="Test entity deletion",
            confidence=DEFAULT_CONFIDENCE,
        )

        db.import_operations(ops)

        # Verify entity was deleted
        deleted_entity = db.get_entity(entity1.id)
        assert deleted_entity is None, "Entity should be deleted"

        # Verify relationships were cascade deleted
        assert db.relationship_count < initial_relationship_count, (
            "Relationships should be cascade deleted"
        )

        # Verify other entities still exist
        remaining_entity1 = db.get_entity(entity2.id)
        remaining_entity2 = db.get_entity(entity3.id)
        entity_existence_flags = [remaining_entity1 is not None, remaining_entity2 is not None]
        assert all(entity_existence_flags), "Other entities should still exist"

        assert db.entity_count == initial_entity_count - 1, "Entity count should be reduced by 1"

    @pytest.mark.unit
    def test_llm_graph_delete_entity_error_handling_comprehensive(self) -> None:
        """Test LLMGraphDeleteEntity error handling in various failure scenarios."""
        db = GraphDatabase()

        # Test deletion of non-existent entity
        nonexistent_delete = LLMGraphDeleteEntity(
            entity_id="nonexistent_entity_id",
            reasoning="Delete nonexistent entity",
            confidence=DEFAULT_CONFIDENCE,
        )

        ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[],
            delete_entity_ops=[nonexistent_delete],
            add_relationship_ops=[],
            delete_relationship_ops=[],
            reasoning="Test nonexistent entity deletion",
            confidence=DEFAULT_CONFIDENCE,
        )

        with pytest.raises(ValueError, match="does not exist"):
            db.import_operations(ops)

        # Verify database state remains consistent after failed deletion
        assert db.entity_count == 0, (
            "Database should still contain 0 entities after failed deletion"
        )
        assert db.relationship_count == 0, (
            "Database should still have 0 relationships after failed deletion"
        )

        # Test deletion with empty entity ID
        empty_delete = LLMGraphDeleteEntity(
            entity_id="",
            reasoning="Delete empty entity",
            confidence=DEFAULT_CONFIDENCE,
        )

        empty_ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[],
            delete_entity_ops=[empty_delete],
            add_relationship_ops=[],
            delete_relationship_ops=[],
            reasoning="Test empty entity ID deletion",
            confidence=DEFAULT_CONFIDENCE,
        )

        with pytest.raises(ValueError, match="does not exist"):
            db.import_operations(empty_ops)

    @pytest.mark.unit
    def test_llm_graph_delete_relationship_validation_comprehensive(self) -> None:
        """Test LLMGraphDeleteRelationship validation with comprehensive scenarios."""

        # Test valid delete relationship operation creation
        valid_delete = LLMGraphDeleteRelationship(
            id="source_target",
            reasoning="Delete test relationship",
            confidence=DEFAULT_CONFIDENCE,
        )

        assert valid_delete.id == "source_target", "Relationship ID should be set correctly"
        assert valid_delete.reasoning == "Delete test relationship", "Reasoning should be preserved"
        assert valid_delete.confidence == DEFAULT_CONFIDENCE, "Confidence should be preserved"

        # Test with empty relationship ID
        empty_delete = LLMGraphDeleteRelationship(
            id="",
            reasoning="Delete empty relationship",
            confidence=DEFAULT_CONFIDENCE,
        )
        assert empty_delete.id == "", "Empty relationship ID should be allowed in validation"

        # Test with special characters in relationship ID
        special_delete = LLMGraphDeleteRelationship(
            id="entity1_entity2",
            reasoning="Delete special relationship",
            confidence=DEFAULT_CONFIDENCE,
        )
        assert "entity1_entity2" in special_delete.id, (
            "Entity-based relationship ID should be allowed"
        )

    @pytest.mark.unit
    def test_llm_graph_delete_relationship_execution_comprehensive(self) -> None:
        """Test LLMGraphDeleteRelationship execution with comprehensive scenarios."""
        db = GraphDatabase()

        # Create test entities
        entity1 = Entity(name="Source Entity", type=TYPE_CONCEPT, content="Source content")
        entity2 = Entity(name="Target Entity", type=TYPE_TOOL, content="Target content")
        entity3 = Entity(name="Third Entity", type=TYPE_LIBRARY, content="Third content")

        db.add_entities([entity1, entity2, entity3])

        # Create multiple relationships
        relationship1 = db.create_relationship(entity1.id, entity2.id, REL_TYPE_USES)
        relationship2 = db.create_relationship(entity2.id, entity3.id, REL_TYPE_RELATED_TO)
        relationship3 = db.create_relationship(entity1.id, entity3.id, REL_TYPE_IMPLEMENTS)

        initial_relationship_count = db.relationship_count

        # Test deletion of one relationship
        delete_op = LLMGraphDeleteRelationship(
            id=relationship2.id,
            reasoning="Delete specific relationship",
            confidence=DEFAULT_CONFIDENCE,
        )

        ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[],
            delete_relationship_ops=[delete_op],
            reasoning="Test relationship deletion",
            confidence=DEFAULT_CONFIDENCE,
        )

        db.import_operations(ops)

        # Verify relationship was deleted
        deleted_relationship = db.get_relationship(relationship2.id)
        assert deleted_relationship is None, "Relationship should be deleted"

        # Verify other relationships still exist
        remaining_rel1 = db.get_relationship(relationship1.id)
        remaining_rel3 = db.get_relationship(relationship3.id)
        relationship_existence_flags = [remaining_rel1 is not None, remaining_rel3 is not None]
        assert all(relationship_existence_flags), "Other relationships should still exist"

        assert db.relationship_count == initial_relationship_count - 1, (
            "Relationship count should be reduced by 1"
        )

        # Verify entities are not affected
        entity1_retrieved = db.get_entity(entity1.id)
        entity2_retrieved = db.get_entity(entity2.id)
        entity3_retrieved = db.get_entity(entity3.id)
        entity_existence_after_deletion = [
            entity1_retrieved is not None,
            entity2_retrieved is not None,
            entity3_retrieved is not None,
        ]
        assert all(entity_existence_after_deletion), (
            "Entities should not be affected by relationship deletion"
        )

    @pytest.mark.unit
    def test_llm_graph_delete_relationship_error_handling_comprehensive(self) -> None:
        """Test LLMGraphDeleteRelationship error handling in various failure scenarios."""
        db = GraphDatabase()

        # Test deletion of non-existent relationship
        nonexistent_delete = LLMGraphDeleteRelationship(
            id="nonexistent_rel_id",
            reasoning="Delete nonexistent relationship",
            confidence=DEFAULT_CONFIDENCE,
        )

        ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[],
            delete_relationship_ops=[nonexistent_delete],
            reasoning="Test nonexistent relationship deletion",
            confidence=DEFAULT_CONFIDENCE,
        )

        with pytest.raises(ValueError, match="does not exist"):
            db.import_operations(ops)

        # Verify database state remains consistent after failed deletion
        assert db.relationship_count == 0, (
            "Database should still contain 0 relationships after failed deletion"
        )

        # Test with empty relationship ID
        empty_delete = LLMGraphDeleteRelationship(
            id="",
            reasoning="Delete empty relationship",
            confidence=DEFAULT_CONFIDENCE,
        )

        empty_ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[],
            delete_relationship_ops=[empty_delete],
            reasoning="Test empty relationship ID deletion",
            confidence=DEFAULT_CONFIDENCE,
        )

        with pytest.raises(ValueError, match="does not exist"):
            db.import_operations(empty_ops)

    @pytest.mark.unit
    def test_llm_graph_mixed_delete_operations_comprehensive(self) -> None:
        """Test mixed delete operations (entities and relationships) in single operation."""
        db = GraphDatabase()

        # Create test entities and relationships
        entity1 = Entity(name="Entity 1", type=TYPE_CONCEPT, content="Content 1")
        entity2 = Entity(name="Entity 2", type=TYPE_TOOL, content="Content 2")
        entity3 = Entity(name="Entity 3", type=TYPE_LIBRARY, content="Content 3")
        entity4 = Entity(name="Entity 4", type=TYPE_PERSON, content="Content 4")

        db.add_entities([entity1, entity2, entity3, entity4])

        # Create relationships
        rel1 = db.create_relationship(entity1.id, entity2.id, REL_TYPE_USES)
        rel2 = db.create_relationship(entity2.id, entity3.id, REL_TYPE_RELATED_TO)
        rel3 = db.create_relationship(entity1.id, entity3.id, REL_TYPE_IMPLEMENTS)
        rel4 = db.create_relationship(entity4.id, entity1.id, REL_TYPE_CREATED_BY)

        initial_entity_count = db.entity_count
        initial_relationship_count = db.relationship_count

        # Create mixed delete operations
        entity_delete = LLMGraphDeleteEntity(
            entity_id=entity2.id,
            reasoning="Delete entity 2 (should cascade delete its relationships)",
            confidence=DEFAULT_CONFIDENCE,
        )

        relationship_delete = LLMGraphDeleteRelationship(
            id=rel3.id,
            reasoning="Delete specific relationship",
            confidence=DEFAULT_CONFIDENCE,
        )

        ops = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[],
            delete_entity_ops=[entity_delete],
            add_relationship_ops=[],
            delete_relationship_ops=[relationship_delete],
            reasoning="Test mixed delete operations",
            confidence=DEFAULT_CONFIDENCE,
        )

        db.import_operations(ops)

        # Verify entity deletion and cascade
        deleted_entity = db.get_entity(entity2.id)
        assert deleted_entity is None, "Target entity should be deleted"

        # Verify specific relationship deletion
        deleted_specific_rel = db.get_relationship(rel3.id)
        assert deleted_specific_rel is None, "Specific relationship should be deleted"

        # Verify cascade deletion of entity2's relationships
        cascade_deleted_rel1 = db.get_relationship(rel1.id)
        cascade_deleted_rel2 = db.get_relationship(rel2.id)
        cascade_flags = [cascade_deleted_rel1 is None, cascade_deleted_rel2 is None]
        assert all(cascade_flags), (
            "Relationships connected to deleted entity should be cascade deleted"
        )

        # Verify other relationships remain
        remaining_rel4 = db.get_relationship(rel4.id)
        assert remaining_rel4 is not None, "Unrelated relationships should remain"

        # Verify other entities remain
        remaining_entities = [entity1.id, entity3.id, entity4.id]
        retrieved_entities = [db.get_entity(eid) for eid in remaining_entities]
        remaining_entity_flags = [entity is not None for entity in retrieved_entities]
        assert all(remaining_entity_flags), "Unrelated entities should remain"

        # Verify final counts
        assert db.entity_count == initial_entity_count - 1, "One entity should be deleted"
        assert db.relationship_count < initial_relationship_count, (
            "Multiple relationships should be deleted"
        )


class TestNewQueryAPI:
    """Test the new query_entities() and query_relationships() APIs."""

    @pytest.mark.unit
    def test_query_entities_basic_functionality(self) -> None:
        """Test basic query_entities functionality."""
        db = GraphDatabase()

        # Add test entities
        entities = [
            Entity(
                name="ML Concept",
                id="ml1",
                type=TYPE_CONCEPT,
                content="Machine learning algorithms",
            ),
            Entity(name="Python Tool", id="py1", type=TYPE_TOOL, content="Python programming tool"),
            Entity(
                name="Data Library", id="lib1", type=TYPE_LIBRARY, content="Data processing library"
            ),
        ]
        db.add_entities(entities)

        # Test query all entities
        results = db.query_entities().execute().results
        assert len(results) == THREE_ENTITIES, "Should return all entities"

        # Verify result structure
        entity_types = [isinstance(entity, Entity) for entity in results]
        assert all(entity_types), "All results should be Entity objects"

        # Verify all entities are present
        result_ids = {entity.id for entity in results}
        expected_ids = {entity.id for entity in entities}
        assert result_ids == expected_ids, "All entity IDs should match"

    @pytest.mark.unit
    def test_query_entities_type_filtering(self) -> None:
        """Test query_entities with type filtering."""
        db = GraphDatabase()

        # Add entities with different types
        entities = [
            Entity(name="Concept 1", id="c1", type=TYPE_CONCEPT, content="First concept"),
            Entity(name="Concept 2", id="c2", type=TYPE_CONCEPT, content="Second concept"),
            Entity(name="Tool 1", id="t1", type=TYPE_TOOL, content="First tool"),
            Entity(name="Library 1", id="l1", type=TYPE_LIBRARY, content="First library"),
        ]
        db.add_entities(entities)

        # Test single type filtering
        concept_results = db.query_entities().type(TYPE_CONCEPT).execute().results
        assert len(concept_results) == 2, "Should return 2 concepts"
        concept_types = [entity.type for entity in concept_results]
        assert all(t == TYPE_CONCEPT for t in concept_types), "All should be concepts"

        # Test another type
        tool_results = db.query_entities().type(TYPE_TOOL).execute().results
        assert len(tool_results) == 1, "Should return 1 tool"
        assert tool_results[0].type == TYPE_TOOL, "Should be a tool"

    @pytest.mark.unit
    def test_query_entities_content_filtering(self) -> None:
        """Test query_entities with content filtering."""
        db = GraphDatabase()

        # Add entities with specific content
        entities = [
            Entity(
                name="AI Research",
                id="ai1",
                type=TYPE_CONCEPT,
                content="Artificial intelligence and machine learning research",
            ),
            Entity(
                name="Python ML",
                id="pyml1",
                type=TYPE_LIBRARY,
                content="Python machine learning libraries",
            ),
            Entity(
                name="Data Science",
                id="ds1",
                type=TYPE_CONCEPT,
                content="Data analysis and visualization tools",
            ),
        ]
        db.add_entities(entities)

        # Test full-text search filtering
        search_results = (
            db.query_entities()
            .search("Artificial intelligence machine learning research", top_k=10)
            .execute()
            .results
        )
        assert len(search_results) >= 1, "Should find entity with search match"

        # Test case-insensitive search
        case_results = db.query_entities().search("PYTHON", top_k=10).execute().results
        assert len(case_results) >= 1, "Should find entities regardless of case"

        # Test partial search
        partial_results = db.query_entities().search("research", top_k=10).execute().results
        assert len(partial_results) >= 1, "Should find entities with partial match"

    @pytest.mark.unit
    def test_query_entities_combined_filters(self) -> None:
        """Test query_entities with multiple filters combined."""
        db = GraphDatabase()

        # Add diverse entities
        entities = [
            Entity(
                name="ML Algorithm",
                id="ml_algo",
                type=TYPE_CONCEPT,
                content="Machine learning algorithm implementation",
            ),
            Entity(
                name="ML Library",
                id="ml_lib",
                type=TYPE_LIBRARY,
                content="Machine learning library for Python",
            ),
            Entity(
                name="Data Algorithm",
                id="data_algo",
                type=TYPE_CONCEPT,
                content="Data processing algorithm",
            ),
            Entity(
                name="Python ML Tool",
                id="py_tool",
                type=TYPE_TOOL,
                content="Python machine learning development tool",
            ),
        ]
        db.add_entities(entities)

        # Test type + search filtering (this behaves as OR logic between filters)
        # The current implementation uses OR logic between filters
        combined_results = (
            db.query_entities()
            .type(TYPE_CONCEPT)
            .search("machine learning", top_k=10)
            .execute()
            .results
        )
        combined_ids = {entity.id for entity in combined_results}

        # Should include all concepts and/or entities with "machine learning" content
        assert len(combined_ids) >= 1, "Should include concepts or ML-related entities"

        # Test search filtering across types
        python_results = db.query_entities().search("python", top_k=10).execute().results
        python_ids = {entity.id for entity in python_results}
        assert len(python_ids) >= 1, "Should find Python-related entities"

    @pytest.mark.unit
    def test_query_entities_or_conditions(self) -> None:
        """Test query_entities with OR conditions."""
        db = GraphDatabase()

        # Add test entities
        entities = [
            Entity(
                name="Concept A", id="ca", type=TYPE_CONCEPT, content="Content about algorithms"
            ),
            Entity(name="Tool B", id="tb", type=TYPE_TOOL, content="Tool for building"),
            Entity(name="Library C", id="lc", type=TYPE_LIBRARY, content="Library for computation"),
            Entity(name="Concept D", id="cd", type=TYPE_CONCEPT, content="Data analysis methods"),
        ]
        db.add_entities(entities)

        # Test OR type filtering
        concept_or_tool = (
            db.query_entities().type(TYPE_CONCEPT).or_type(TYPE_TOOL).execute().results
        )
        assert len(concept_or_tool) == THREE_ENTITIES, "Should return concepts and tools"
        result_types = {entity.type for entity in concept_or_tool}
        assert result_types == {TYPE_CONCEPT, TYPE_TOOL}, "Should only include concepts and tools"

        # Test search for algorithm or data
        algorithm_results = db.query_entities().search("algorithm", top_k=10).execute().results
        data_results = db.query_entities().search("data", top_k=10).execute().results
        assert len(algorithm_results) >= 1 or len(data_results) >= 1, (
            "Should find algorithm or data content"
        )

    @pytest.mark.unit
    def test_query_entities_empty_results(self) -> None:
        """Test query_entities with filters that return no results."""
        db = GraphDatabase()

        # Add some entities
        entities = [
            Entity(name="Test Entity", id="test1", type=TYPE_CONCEPT, content="Test content"),
        ]
        db.add_entities(entities)

        # Test non-existent type
        person_results = db.query_entities().type(TYPE_PERSON).execute().results
        assert len(person_results) == 0, "Should return no results for non-existent type"

        # Test non-existent content
        nonexistent_results = (
            db.query_entities().search("nonexistent_term_xyz", top_k=10).execute().results
        )
        assert len(nonexistent_results) == 0, "Should return no results for non-existent content"

    @pytest.mark.unit
    def test_query_relationships_basic_functionality(self) -> None:
        """Test basic query_relationships functionality."""
        db = GraphDatabase()

        # Create test entities
        entities = [
            Entity(name="Source A", id="src_a", type=TYPE_CONCEPT, content="Source entity A"),
            Entity(name="Target B", id="tgt_b", type=TYPE_TOOL, content="Target entity B"),
            Entity(name="Target C", id="tgt_c", type=TYPE_LIBRARY, content="Target entity C"),
        ]
        db.add_entities(entities)

        # Create relationships
        db.create_relationship("src_a", "tgt_b", REL_TYPE_USES)
        db.create_relationship("src_a", "tgt_c", REL_TYPE_RELATED_TO)

        # Test query all relationships
        results = db.query_relationships().execute().results
        assert len(results) == 2, "Should return all relationships"

        relationship_tuples = cast(list[tuple[Entity, Entity, Relationship]], results)
        assert all(
            isinstance(source, Entity)
            and isinstance(target, Entity)
            and isinstance(relationship, Relationship)
            for source, target, relationship in relationship_tuples
        ), "All results should contain entity + relationship tuples"

        relationship_types = [relationship.type for _, _, relationship in relationship_tuples]
        assert REL_TYPE_USES in relationship_types and REL_TYPE_RELATED_TO in relationship_types, (
            "Should include both relationship types"
        )

    @pytest.mark.unit
    def test_query_relationships_type_filtering(self) -> None:
        """Test query_relationships with type filtering."""
        db = GraphDatabase()

        # Create test entities
        entities = [
            Entity(name="Entity 1", id="e1", type=TYPE_CONCEPT, content="Entity 1"),
            Entity(name="Entity 2", id="e2", type=TYPE_TOOL, content="Entity 2"),
            Entity(name="Entity 3", id="e3", type=TYPE_LIBRARY, content="Entity 3"),
        ]
        db.add_entities(entities)

        # Create relationships with different types
        db.create_relationship("e1", "e2", REL_TYPE_USES)
        db.create_relationship("e2", "e3", REL_TYPE_RELATED_TO)
        db.create_relationship("e1", "e3", REL_TYPE_USES)

        # Test single type filtering
        uses_results = db.query_relationships().type(REL_TYPE_USES).execute().results
        assert len(uses_results) == 2, "Should return 2 'uses' relationships"
        uses_types = [rel.type for rel in _relationships_from_results(uses_results)]
        assert all(t == REL_TYPE_USES for t in uses_types), "All should be 'uses' relationships"

        # Test another type
        related_results = db.query_relationships().type(REL_TYPE_RELATED_TO).execute().results
        assert len(related_results) == 1, "Should return 1 'related_to' relationship"
        related_relationships = _relationships_from_results(related_results)
        assert related_relationships[0].type == REL_TYPE_RELATED_TO, "Should be 'related_to'"

    @pytest.mark.unit
    def test_query_relationships_entity_filtering(self) -> None:
        """Test query_relationships with source/target entity filtering."""
        db = GraphDatabase()

        # Create test entities
        entities = [
            Entity(name="Central Hub", id="hub", type=TYPE_CONCEPT, content="Central entity"),
            Entity(name="Node A", id="node_a", type=TYPE_TOOL, content="Node A"),
            Entity(name="Node B", id="node_b", type=TYPE_LIBRARY, content="Node B"),
            Entity(name="Node C", id="node_c", type=TYPE_PERSON, content="Node C"),
        ]
        db.add_entities(entities)

        # Create relationships from hub to others
        db.create_relationship("hub", "node_a", REL_TYPE_USES)
        db.create_relationship("hub", "node_b", REL_TYPE_RELATED_TO)
        db.create_relationship("node_c", "hub", REL_TYPE_CREATED_BY)

        # Test source entity filtering
        from_hub = db.query_relationships().from_entity(entity_id="hub").execute().results
        assert len(from_hub) == 2, "Should return 2 relationships from hub"
        source_ids = {rel.source for rel in _relationships_from_results(from_hub)}
        assert source_ids == {"hub"}, "All should have hub as source"

        # Test target entity filtering
        to_hub = db.query_relationships().to_entity(entity_id="hub").execute().results
        assert len(to_hub) == 1, "Should return 1 relationship to hub"
        target_ids = {rel.target for rel in _relationships_from_results(to_hub)}
        assert target_ids == {"hub"}, "All should have hub as target"

        # Test filtering by entity name
        by_name = db.query_relationships().from_entity(entity_name="Central Hub").execute().results
        assert len(by_name) == 2, "Should find relationships by entity name"

    @pytest.mark.unit
    def test_query_relationships_combined_filters(self) -> None:
        """Test query_relationships with multiple filters combined."""
        db = GraphDatabase()

        # Create test entities
        entities = [
            Entity(name="Developer", id="dev", type=TYPE_PERSON, content="Software developer"),
            Entity(name="Library", id="lib", type=TYPE_LIBRARY, content="Software library"),
            Entity(name="Framework", id="fw", type=TYPE_TOOL, content="Development framework"),
            Entity(name="Application", id="app", type=TYPE_CONCEPT, content="Software application"),
        ]
        db.add_entities(entities)

        # Create relationships
        db.create_relationship("dev", "lib", REL_TYPE_USES)
        db.create_relationship("dev", "fw", REL_TYPE_USES)
        db.create_relationship("fw", "app", REL_TYPE_IMPLEMENTS)
        db.create_relationship("lib", "app", REL_TYPE_BELONGS_TO)

        # Test type + source filtering
        dev_uses = (
            db.query_relationships()
            .type(REL_TYPE_USES)
            .from_entity(entity_id="dev")
            .execute()
            .results
        )
        assert len(dev_uses) == 2, "Developer should have 2 'uses' relationships"
        dev_uses_targets = {rel.target for rel in _relationships_from_results(dev_uses)}
        assert dev_uses_targets == {"lib", "fw"}, "Should use both library and framework"

        # Test type + target filtering (this behaves as OR logic between filters)
        app_implementations = (
            db.query_relationships()
            .type(REL_TYPE_IMPLEMENTS)
            .to_entity(entity_id="app")
            .execute()
            .results
        )
        assert len(app_implementations) >= 1, "App should have at least 1 relationship"

        # Check if the implementation relationship exists
        impl_exists = any(
            rel.type == REL_TYPE_IMPLEMENTS and rel.target == "app"
            for rel in _relationships_from_results(app_implementations)
        )
        assert impl_exists, "Framework should implement app"

    @pytest.mark.unit
    def test_query_relationships_with_entities(self) -> None:
        """Test query_relationships with entities included in results."""
        db = GraphDatabase()

        # Create test entities
        entities = [
            Entity(name="Author", id="author", type=TYPE_PERSON, content="Book author"),
            Entity(name="Book", id="book", type=TYPE_CONCEPT, content="Published book"),
            Entity(name="Publisher", id="pub", type=TYPE_LIBRARY, content="Book publisher"),
        ]
        db.add_entities(entities)

        # Create relationships
        db.create_relationship("author", "book", REL_TYPE_CREATED_BY)
        db.create_relationship("book", "pub", REL_TYPE_BELONGS_TO)

        # Test query with entities
        results_with_entities = db.query_relationships().execute().results
        assert len(results_with_entities) == 2, "Should return all relationships with entities"

        # Verify result structure (should be tuples of (source, target, relationship))
        entity_relationship_verification_flags = [
            isinstance(source, Entity)
            and isinstance(target, Entity)
            and isinstance(relationship, Relationship)
            for result in cast(list[tuple[Entity, Entity, Relationship]], results_with_entities)
            for source, target, relationship in [result]
        ]

        result_structure_flags = [isinstance(result, tuple) for result in results_with_entities]
        element_type_flags = [
            element_type
            for result in cast(list[tuple[Entity, Entity, Relationship]], results_with_entities)
            for source, target, relationship in [result]
            for element_type in [
                isinstance(source, Entity),
                isinstance(target, Entity),
                isinstance(relationship, Relationship),
            ]
        ]

        assert all(result_structure_flags), "All results should be tuples"
        assert all(element_type_flags), "All elements should be of correct type"
        assert all(entity_relationship_verification_flags), (
            "All results should have correct structure"
        )

    @pytest.mark.unit
    def test_query_relationships_or_conditions(self) -> None:
        """Test query_relationships with OR conditions."""
        db = GraphDatabase()

        # Create test entities
        entities = [
            Entity(name="Service A", id="svc_a", type=TYPE_TOOL, content="Service A"),
            Entity(name="Service B", id="svc_b", type=TYPE_LIBRARY, content="Service B"),
            Entity(name="Consumer", id="cons", type=TYPE_CONCEPT, content="Consumer service"),
        ]
        db.add_entities(entities)

        # Create relationships with different types
        db.create_relationship("cons", "svc_a", REL_TYPE_USES)
        db.create_relationship("cons", "svc_b", REL_TYPE_RELATED_TO)
        db.create_relationship("svc_a", "svc_b", REL_TYPE_SIMILAR_TO)

        # Test OR type filtering
        uses_or_related = (
            db.query_relationships()
            .type(REL_TYPE_USES)
            .or_type(REL_TYPE_RELATED_TO)
            .execute()
            .results
        )
        assert len(uses_or_related) == 2, "Should return uses and related relationships"
        result_types = {rel.type for rel in _relationships_from_results(uses_or_related)}
        assert result_types == {REL_TYPE_USES, REL_TYPE_RELATED_TO}, "Should include both types"

    @pytest.mark.unit
    def test_query_relationships_empty_results(self) -> None:
        """Test query_relationships with filters that return no results."""
        db = GraphDatabase()

        # Add some entities and relationships
        entities = [
            Entity(name="Entity 1", id="e1", type=TYPE_CONCEPT, content="Entity 1"),
            Entity(name="Entity 2", id="e2", type=TYPE_TOOL, content="Entity 2"),
        ]
        db.add_entities(entities)
        db.create_relationship("e1", "e2", REL_TYPE_USES)

        # Test non-existent relationship type
        invalid_results = db.query_relationships().type(REL_TYPE_INVALIDATES).execute().results
        assert len(invalid_results) == 0, "Should return no results for non-existent type"

        # Test non-existent source entity
        nonexistent_source = (
            db.query_relationships().from_entity(entity_id="nonexistent").execute().results
        )
        assert len(nonexistent_source) == 0, "Should return no results for non-existent source"

        # Test non-existent target entity
        nonexistent_target = (
            db.query_relationships().to_entity(entity_id="nonexistent").execute().results
        )
        assert len(nonexistent_target) == 0, "Should return no results for non-existent target"

    @pytest.mark.unit
    def test_query_api_error_handling(self) -> None:
        """Test query API error handling."""
        db = GraphDatabase()

        # Test queries on empty database
        empty_entity_results = db.query_entities().execute().results
        assert len(empty_entity_results) == 0, "Empty database should return no entities"

        empty_rel_results = db.query_relationships().execute().results
        assert len(empty_rel_results) == 0, "Empty database should return no relationships"

        # Test queries with filters on empty database
        filtered_empty_entities = db.query_entities().type(TYPE_CONCEPT).execute().results
        assert len(filtered_empty_entities) == 0, (
            "Filtered query on empty database should return no results"
        )

        filtered_empty_rels = db.query_relationships().type(REL_TYPE_USES).execute().results
        assert len(filtered_empty_rels) == 0, (
            "Filtered query on empty database should return no results"
        )

    @pytest.mark.unit
    def test_query_api_complex_scenarios(self) -> None:
        """Test query API with complex, realistic scenarios."""
        db = GraphDatabase()

        # Create a realistic knowledge graph structure
        entities = [
            # AI/ML concepts
            Entity(
                name="Machine Learning",
                id="ml",
                type=TYPE_CONCEPT,
                content="Machine learning algorithms and techniques",
            ),
            Entity(
                name="Deep Learning",
                id="dl",
                type=TYPE_CONCEPT,
                content="Deep neural networks and architectures",
            ),
            Entity(
                name="Natural Language Processing",
                id="nlp",
                type=TYPE_CONCEPT,
                content="Processing and understanding human language",
            ),
            # Tools and libraries
            Entity(
                name="TensorFlow",
                id="tf",
                type=TYPE_LIBRARY,
                content="Open-source machine learning framework",
            ),
            Entity(
                name="PyTorch",
                id="pt",
                type=TYPE_LIBRARY,
                content="Machine learning library for Python",
            ),
            Entity(
                name="Scikit-learn",
                id="sk",
                type=TYPE_LIBRARY,
                content="Simple and efficient tools for data mining",
            ),
            # People
            Entity(
                name="AI Researcher",
                id="researcher",
                type=TYPE_PERSON,
                content="Researcher specializing in artificial intelligence",
            ),
            Entity(
                name="Data Scientist",
                id="ds",
                type=TYPE_PERSON,
                content="Professional working with data and ML",
            ),
        ]
        db.add_entities(entities)

        # Create relationships
        db.create_relationship("ml", "dl", REL_TYPE_RELATED_TO)
        db.create_relationship("ml", "nlp", REL_TYPE_RELATED_TO)
        db.create_relationship("tf", "ml", REL_TYPE_IMPLEMENTS)
        db.create_relationship("pt", "dl", REL_TYPE_IMPLEMENTS)
        db.create_relationship("sk", "ml", REL_TYPE_IMPLEMENTS)
        db.create_relationship("researcher", "tf", REL_TYPE_USES)
        db.create_relationship("researcher", "pt", REL_TYPE_USES)
        db.create_relationship("ds", "sk", REL_TYPE_USES)
        db.create_relationship("ds", "tf", REL_TYPE_USES)

        # Complex entity queries (filtering uses OR logic)
        ai_concepts = db.query_entities().type(TYPE_CONCEPT).execute().results
        assert len(ai_concepts) >= 1, "Should find concepts"
        assert all(e.type == TYPE_CONCEPT for e in ai_concepts), "All results should be concepts"

        # Combined query should include concepts AND learning-related entities
        combined_learning = (
            db.query_entities().type(TYPE_CONCEPT).search("learning", top_k=10).execute().results
        )
        combined_ids = {entity.id for entity in combined_learning}

        # Should include concepts that contain "learning" (Machine Learning, Deep Learning)
        expected_learning_concepts = {"ml", "dl"}  # Machine Learning, Deep Learning

        assert len(combined_learning) >= 2, "Should include concepts containing 'learning'"
        assert combined_ids.issuperset(expected_learning_concepts), (
            "Should include ML and Deep Learning concepts"
        )

        ml_libs = db.query_entities().search("machine learning", top_k=10).execute().results
        assert len(ml_libs) >= 1, "Should find ML libraries"

        # Complex relationship queries
        implementations = db.query_relationships().type(REL_TYPE_IMPLEMENTS).execute().results
        assert len(implementations) == THREE_ENTITIES, "Should find 3 implementation relationships"

        researcher_tools = (
            db.query_relationships().from_entity(entity_name="AI Researcher").execute().results
        )
        assert len(researcher_tools) == 2, "Researcher should use 2 tools"

        # Combined complex query
        ml_implementations = (
            db.query_relationships()
            .type(REL_TYPE_IMPLEMENTS)
            .to_entity(entity_id="ml")
            .execute()
            .results
        )
        assert len(ml_implementations) >= 1, "Should find ML implementations"

        # Verify no results for non-existent patterns
        non_existent = (
            db.query_entities()
            .search("quantum physics xyz nonexistent", top_k=10)
            .execute()
            .results
        )
        assert len(non_existent) == 0, "Should find no quantum physics concepts"

    @pytest.mark.unit
    def test_query_entities_date_range_filtering(self) -> None:
        """Test query_entities with date range filtering using created_between."""
        db = GraphDatabase()

        # Create entities with specific creation dates

        entities = [
            Entity(
                name="Early Entity",
                id="early",
                type=TYPE_CONCEPT,
                content="Created early",
                created_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
            ),
            Entity(
                name="Middle Entity",
                id="middle",
                type=TYPE_TOOL,
                content="Created in middle",
                created_at=datetime(2024, 2, 15, 10, 0, 0, tzinfo=UTC),
            ),
            Entity(
                name="Late Entity",
                id="late",
                type=TYPE_LIBRARY,
                content="Created late",
                created_at=datetime(2024, 3, 15, 10, 0, 0, tzinfo=UTC),
            ),
        ]
        db.add_entities(entities)

        # Test date range filtering - between specific dates
        feb_results = (
            db.query_entities()
            .created_between(start_date=date(2024, 2, 1), end_date=date(2024, 2, 29))
            .execute()
            .results
        )
        assert len(feb_results) == 1, "Should find 1 entity created in February"
        assert feb_results[0].id == "middle", "Should find middle entity"

        # Test start date only (from date onwards)
        from_feb = (
            db.query_entities().created_between(start_date=date(2024, 2, 1)).execute().results
        )
        assert len(from_feb) >= 2, "Should find entities created from February onwards"
        from_feb_ids = {entity.id for entity in from_feb}
        assert from_feb_ids.issuperset({"middle", "late"}), (
            "Should include middle and late entities"
        )

        # Test end date only (up to date)
        until_feb = (
            db.query_entities().created_between(end_date=date(2024, 2, 29)).execute().results
        )
        assert len(until_feb) >= 2, "Should find entities created up to February"
        until_feb_ids = {entity.id for entity in until_feb}
        assert until_feb_ids.issuperset({"early", "middle"}), (
            "Should include early and middle entities"
        )

        # Test date range with no matches
        no_results = (
            db.query_entities()
            .created_between(start_date=date(2024, 4, 1), end_date=date(2024, 4, 30))
            .execute()
            .results
        )
        assert len(no_results) == 0, "Should find no entities in April"

    @pytest.mark.unit
    def test_query_entities_search_integration(self) -> None:
        """Test query_entities with search integration."""
        db = GraphDatabase()

        # Add test entities with searchable content
        entities = [
            Entity(
                name="Machine Learning",
                id="ml",
                type=TYPE_CONCEPT,
                content="Machine learning algorithms and neural networks",
            ),
            Entity(
                name="Python Programming",
                id="python",
                type=TYPE_TOOL,
                content="Python programming language and ecosystem",
            ),
            Entity(
                name="Data Science",
                id="data_science",
                type=TYPE_CONCEPT,
                content="Data analysis, visualization, and machine learning",
            ),
        ]
        db.add_entities(entities)

        # Test search integration with query builder
        search_results = (
            db.query_entities().search("machine learning", top_k=TOP_K_FIVE).execute().results
        )
        assert len(search_results) >= 1, "Search should find ML-related entities"

        # Verify search results contain the expected terms
        search_content = [(entity.content or "").lower() for entity in search_results]
        assert any("machine learning" in content for content in search_content), (
            "Results should contain machine learning"
        )

        # Test search combined with type filtering (OR logic)
        ml_concepts = (
            db.query_entities().search("machine learning").type(TYPE_CONCEPT).execute().results
        )
        ml_concept_ids = {entity.id for entity in ml_concepts}
        assert "ml" in ml_concept_ids, "Should find ML concept"

        # Test search with partial term
        python_results = db.query_entities().search("python", top_k=TOP_K_THREE).execute().results
        python_ids = {entity.id for entity in python_results}
        assert "python" in python_ids, "Should find Python entity"

    @pytest.mark.unit
    def test_query_entities_sorting_and_limiting(self) -> None:
        """Test query_entities with sorting and limiting functionality."""
        db = GraphDatabase()

        # Add entities with names that will have predictable sort order
        entities = [
            Entity(name="A Concept", id="a", type=TYPE_CONCEPT, content="First alphabetically"),
            Entity(name="B Tool", id="b", type=TYPE_TOOL, content="Second alphabetically"),
            Entity(name="C Library", id="c", type=TYPE_LIBRARY, content="Third alphabetically"),
            Entity(name="D Concept", id="d", type=TYPE_CONCEPT, content="Fourth alphabetically"),
            Entity(name="E Tool", id="e", type=TYPE_TOOL, content="Fifth alphabetically"),
        ]
        db.add_entities(entities)

        # Test sorting by name ascending
        asc_results = db.query_entities().order_by("name", ascending=True).execute().results
        asc_names = [entity.name for entity in asc_results]
        expected_asc = ["A Concept", "B Tool", "C Library", "D Concept", "E Tool"]
        assert asc_names == expected_asc, "Should be sorted alphabetically ascending"

        # Test sorting by name descending
        desc_results = db.query_entities().order_by("name", ascending=False).execute().results
        desc_names = [entity.name for entity in desc_results]
        expected_desc = ["E Tool", "D Concept", "C Library", "B Tool", "A Concept"]
        assert desc_names == expected_desc, "Should be sorted alphabetically descending"

        # Test limiting results
        limited_results = db.query_entities().limit(TOP_K_THREE).execute().results
        assert len(limited_results) == TOP_K_THREE, "Should limit to 3 results"

        # Test sorting and limiting combined
        sorted_limited = (
            db.query_entities()
            .order_by("name", ascending=True)
            .limit(TWO_RESULTS)
            .execute()
            .results
        )
        assert len(sorted_limited) == TWO_RESULTS, "Should limit to 2 results"
        sorted_limited_names = [entity.name for entity in sorted_limited]
        assert sorted_limited_names == ["A Concept", "B Tool"], (
            "Should be sorted and limited correctly"
        )

        # Test sorting by type
        type_sorted = db.query_entities().order_by("type", ascending=True).execute().results
        type_sorted_types = [entity.type for entity in type_sorted]
        # Should be grouped by type (concept, library, tool, but order within type may vary)
        assert all(t in [TYPE_CONCEPT, TYPE_LIBRARY, TYPE_TOOL] for t in type_sorted_types), (
            "All should be valid types"
        )

    @pytest.mark.unit
    def test_query_relationships_sorting_and_limiting(self) -> None:
        """Test query_relationships with sorting and limiting functionality."""
        db = GraphDatabase()

        # Create test entities
        entities = [
            Entity(name="Entity A", id="a", type=TYPE_CONCEPT, content="Entity A"),
            Entity(name="Entity B", id="b", type=TYPE_TOOL, content="Entity B"),
            Entity(name="Entity C", id="c", type=TYPE_LIBRARY, content="Entity C"),
            Entity(name="Entity D", id="d", type=TYPE_CONCEPT, content="Entity D"),
        ]
        db.add_entities(entities)

        # Create relationships with different types for sorting
        relationships_data: list[tuple[str, str, RelationType]] = [
            ("a", "b", REL_TYPE_USES),
            ("a", "c", REL_TYPE_RELATED_TO),
            ("b", "d", REL_TYPE_IMPLEMENTS),
            ("c", "d", REL_TYPE_SIMILAR_TO),
        ]

        relationship_creation_results = [
            db.create_relationship(source, target, rel_type)
            for source, target, rel_type in relationships_data
        ]
        assert len(relationship_creation_results) == len(relationships_data), (
            "Should create all relationships"
        )

        # Test sorting by relationship type
        type_sorted = db.query_relationships().order_by("type", ascending=True).execute().results
        type_sorted_types: list[RelationType] = [
            rel.type for rel in _relationships_from_results(type_sorted)
        ]
        assert len(type_sorted_types) == len(relationships_data), "Should have all relationships"
        assert all(
            t in [REL_TYPE_USES, REL_TYPE_RELATED_TO, REL_TYPE_IMPLEMENTS, REL_TYPE_SIMILAR_TO]
            for t in type_sorted_types
        ), "All should be valid relationship types"

        # Test limiting results
        limited_results = db.query_relationships().limit(TWO_RESULTS).execute().results
        assert len(limited_results) == TWO_RESULTS, "Should limit to 2 results"

        # Test sorting and limiting combined
        sorted_limited = (
            db.query_relationships()
            .order_by("type", ascending=True)
            .limit(TOP_K_THREE)
            .execute()
            .results
        )
        assert len(sorted_limited) == TOP_K_THREE, "Should limit to 3 results"

        # Verify all returned relationships have valid structure
        sorted_limited_rels = _relationships_from_results(sorted_limited)
        source_validity_flags = [rel.source in ["a", "b", "c"] for rel in sorted_limited_rels]
        target_validity_flags = [rel.target in ["b", "c", "d"] for rel in sorted_limited_rels]

        assert all(isinstance(rel, Relationship) for rel in sorted_limited_rels), (
            "All should be Relationship objects"
        )
        assert all(source_validity_flags), "Source should be valid"
        assert all(target_validity_flags), "Target should be valid"

    @pytest.mark.unit
    def test_query_api_comprehensive_edge_cases(self) -> None:
        """Test query API with comprehensive edge cases and error conditions."""
        db = GraphDatabase()

        # Test with single entity
        single_entity = Entity(name="Solo", id="solo", type=TYPE_CONCEPT, content="Only entity")
        db.add_entity(single_entity)

        # Query single entity with various filters
        solo_by_type = db.query_entities().type(TYPE_CONCEPT).execute().results
        assert len(solo_by_type) == 1, "Should find the single concept"
        assert solo_by_type[0].id == "solo", "Should find the solo entity"

        solo_by_content = db.query_entities().search("only", top_k=10).execute().results
        assert len(solo_by_content) == 1, "Should find entity by content"

        # Test filtering with non-existent values
        no_concepts = db.query_entities().type(TYPE_PERSON).execute().results
        assert len(no_concepts) == 0, "Should find no person entities"

        no_content = (
            db.query_entities().search("nonexistent_content_xyz", top_k=10).execute().results
        )
        assert len(no_content) == 0, "Should find no matching content"

        # Test relationship queries on single entity
        db.create_relationship("solo", "solo", REL_TYPE_RELATED_TO)  # Self-relationship

        all_relationships = db.query_relationships().execute().results
        assert len(all_relationships) == 1, "Should find the self-relationship"

        self_relationships = (
            db.query_relationships().from_entity(entity_id="solo").execute().results
        )
        assert len(self_relationships) == 1, "Should find relationships from solo"

        # Test empty result scenarios
        empty_type = db.query_entities().type("nonexistent_type").execute().results  # type: ignore[arg-type]
        assert len(empty_type) == 0, "Should handle non-existent type gracefully"

        # Test query chains with multiple operations
        complex_query = (
            db.query_entities()
            .type(TYPE_CONCEPT)
            .search("only", top_k=10)
            .order_by("name")
            .limit(1)
            .execute()
            .results
        )
        assert len(complex_query) >= 1, "Complex query should work correctly"

        # Test relationships with invalid entity references
        invalid_source = (
            db.query_relationships().from_entity(entity_id="nonexistent").execute().results
        )
        assert len(invalid_source) == 0, "Should handle non-existent source gracefully"

        invalid_target = (
            db.query_relationships().to_entity(entity_id="nonexistent").execute().results
        )
        assert len(invalid_target) == 0, "Should handle non-existent target gracefully"

    @pytest.mark.unit
    def test_query_api_unicode_and_special_characters(self) -> None:
        """Test query API with unicode characters and special cases."""
        db = GraphDatabase()

        # Add entities with special characters and unicode
        entities = [
            Entity(
                name="Unicode Entity",
                id="unicode",
                type=TYPE_CONCEPT,
                content="Unicode content: ñáéíóú 中文 русский العربية",
            ),
            Entity(
                name="Special Chars",
                id="special",
                type=TYPE_TOOL,
                content="Special characters: !@#$%^&*()_+-=[]{}|;':\",./<>?",
            ),
            Entity(
                name="Mixed Case",
                id="mixed",
                type=TYPE_LIBRARY,
                content="Mixed CASE content for testing",
            ),
        ]
        db.add_entities(entities)

        # Test content filtering with unicode
        unicode_results = db.query_entities().search("ñáéíóú", top_k=10).execute().results
        assert len(unicode_results) >= 1, "Should find unicode entity"

        # Test search with special characters
        special_results = (
            db.query_entities().search("Special characters", top_k=10).execute().results
        )
        assert len(special_results) >= 1, "Should find special characters entity"

        # Test case sensitivity in search
        case_lower = db.query_entities().search("mixed case", top_k=10).execute().results
        assert len(case_lower) >= 1, "Should find with case insensitive"

        # Test partial search
        partial_results = db.query_entities().search("CASE", top_k=10).execute().results
        assert len(partial_results) >= 1, "Should find with partial search match"

        # Test search with special characters
        search_results = db.query_entities().search("unicode", top_k=TOP_K_FIVE).execute().results
        assert len(search_results) >= 1, "Search should handle unicode content"

    @pytest.mark.unit
    def test_translate_to_query(self) -> None:
        """Test translating natural language to query."""
        db = GraphDatabase()

        # Add some test data
        entities = [
            Entity(name="Python", id="python", type=TYPE_CONCEPT),
            Entity(name="Machine Learning", id="ml", type=TYPE_CONCEPT),
            Entity(name="TensorFlow", id="tf", type=TYPE_LIBRARY),
        ]
        db.add_entities(entities)

        # Test translating simple entity search queries
        python_ops = LLMGraphQueryOperations(
            reasoning="Find Python entities",
            confidence=1.0,
            entity_queries=[
                LLMEntityQuery(
                    reasoning="Search Python",
                    confidence=1.0,
                    search_query="Python",
                    limit=5,
                )
            ],
        )
        python_query = db.translate_to_query(python_ops)
        python_entities = cast(list[Entity], python_query.execute().results)
        assert len(python_entities) >= 1
        assert any(entity.name == "Python" for entity in python_entities)

        # Test translating more complex filtered queries
        library_ops = LLMGraphQueryOperations(
            reasoning="Find ML libraries",
            confidence=1.0,
            entity_queries=[
                LLMEntityQuery(
                    reasoning="Search ML",
                    confidence=1.0,
                    search_query="TensorFlow",
                    entity_types=("library",),
                )
            ],
        )
        library_query = db.translate_to_query(library_ops)
        library_results = cast(list[Entity], library_query.execute().results)
        assert any(entity.type == TYPE_LIBRARY for entity in library_results)

        # Test translating non-existent query gracefully
        empty_ops = LLMGraphQueryOperations(
            reasoning="Find nothing",
            confidence=1.0,
            entity_queries=[
                LLMEntityQuery(
                    reasoning="No match",
                    confidence=1.0,
                    search_query="completely non existent term xyz123",
                )
            ],
        )
        empty_query = db.translate_to_query(empty_ops)
        empty_results = cast(list[Entity], empty_query.execute().results)
        assert len(empty_results) == 0

    @pytest.mark.unit
    def test_entity_and_relationship_counts(self) -> None:
        """Test counting entities and relationships."""
        db = GraphDatabase()

        # Initially empty
        assert db.entity_count == 0
        assert db.relationship_count == 0

        # Add entities
        entities = [
            Entity(name="E1", id="e1", type=TYPE_CONCEPT),
            Entity(name="E2", id="e2", type=TYPE_CONCEPT),
            Entity(name="E3", id="e3", type=TYPE_TOOL),
        ]
        db.add_entities(entities)

        assert db.entity_count == 3
        assert db.relationship_count == 0

        # Add relationships via convenience helper
        db.create_relationship("e1", "e2", REL_TYPE_RELATED_TO)
        db.create_relationship("e2", "e3", REL_TYPE_USES)

        assert db.entity_count == 3
        assert db.relationship_count == 2

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
            Entity(name="TensorFlow", id="tf", type=TYPE_LIBRARY, content="ML framework"),
        ]
        db.add_entities(entities)

        # Test simple query execute
        result = db.query_entities().execute()
        assert len(result.results) == 3
        assert result.stats.total_results == 3

        # Test query with search filter
        search_result = db.query_entities().search("Python", top_k=10).execute()
        assert len(search_result.results) >= 1
        assert any("Python" in entity.name for entity in search_result.results)

        # Test query with type filter
        library_result = db.query_entities().type(TYPE_LIBRARY).execute()
        assert len(library_result.results) == 1
        assert library_result.results[0].type == TYPE_LIBRARY

        # Test query with limit
        limited_result = db.query_entities().limit(2).execute()
        assert len(limited_result.results) <= 2

        # Test query with sorting
        sorted_result = db.query_entities().order_by("name").execute()
        assert len(sorted_result.results) == 3
        names = [entity.name for entity in sorted_result.results]
        assert names == sorted(names)

    @pytest.mark.unit
    def test_relationship_query_execute(self) -> None:
        """Test RelationshipQuery.execute."""
        db = GraphDatabase()

        # Add test data
        entities = [
            Entity(name="Python", id="python", type=TYPE_CONCEPT),
            Entity(name="TensorFlow", id="tf", type=TYPE_LIBRARY),
            Entity(name="PyTorch", id="pytorch", type=TYPE_LIBRARY),
        ]
        db.add_entities(entities)

        db.add_relationship(Relationship(source="python", target="tf", type=REL_TYPE_USES))
        db.add_relationship(Relationship(source="python", target="pytorch", type=REL_TYPE_USES))
        db.add_relationship(Relationship(source="tf", target="pytorch", type=REL_TYPE_SIMILAR_TO))

        # Test simple relationship query
        result = db.query_relationships().execute()
        assert len(result.results) == 3
        assert result.stats.total_results == 3

        # Test query with source filter
        source_result = db.query_relationships().from_entity(entity_id="python").execute().results
        assert len(source_result) == 2
        assert all(rel.source == "python" for rel in _relationships_from_results(source_result))

        # Test query with target filter
        target_result = db.query_relationships().to_entity(entity_id="tf").execute().results
        assert len(target_result) == 1
        target_relationships = _relationships_from_results(target_result)
        assert target_relationships[0].target == "tf"

        # Test query with type filter
        type_result = db.query_relationships().type(REL_TYPE_USES).execute().results
        assert len(type_result) == 2
        assert all(rel.type == REL_TYPE_USES for rel in _relationships_from_results(type_result))

        # Test query with limit
        limited_result = db.query_relationships().limit(1).execute().results
        assert len(limited_result) == 1


class TestGraphDatabaseEdgeCases:
    """Tests for edge cases and uncovered lines in graph database."""

    def test_add_relationship_duplicate_id(self) -> None:
        """Test adding relationship with duplicate ID raises error (line 222)."""
        db = GraphDatabase()

        # Add entities
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_CONCEPT)
        db.add_entities([entity1, entity2])

        # Create relationship
        rel = Relationship(source=entity1.id, target=entity2.id, type=REL_TYPE_RELATED_TO)
        db.add_relationship(rel)

        # Try to add same relationship again
        with pytest.raises(ValueError, match="Relationship with id .* already exists"):
            db.add_relationship(rel)

    def test_create_relationship_nonexistent_entities(self) -> None:
        """Test creating relationship with non-existent source/target entities (lines 247, 249)."""
        db = GraphDatabase()

        # Test with non-existent source
        with pytest.raises(ValueError, match="Source entity 'nonexistent' does not exist"):
            db.create_relationship("nonexistent", "target", REL_TYPE_RELATED_TO)

        # Test with non-existent target
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        db.add_entity(entity1)

        with pytest.raises(ValueError, match="Target entity 'nonexistent' does not exist"):
            db.create_relationship(entity1.id, "nonexistent", REL_TYPE_RELATED_TO)

    def test_update_nonexistent_relationship(self) -> None:
        """Test updating non-existent relationship raises error (line 276)."""
        db = GraphDatabase()

        # Create a relationship
        rel = Relationship(source="source", target="target", type=REL_TYPE_RELATED_TO)

        # Try to update it without adding to database
        with pytest.raises(ValueError, match="Relationship with id .* does not exist"):
            db.update_relationship(rel)

    def test_translate_entity_query_multiple_types(self) -> None:
        """Test translating entity query with multiple types (line 706)."""
        db = GraphDatabase()

        # Add test data first
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_TOOL)
        entity3 = Entity(name="Entity3", type=TYPE_PERSON)  # Not in query types
        db.add_entities([entity1, entity2, entity3])

        # Create LLM query with multiple entity types
        entity_query = LLMEntityQuery(
            entity_types=(TYPE_CONCEPT, TYPE_TOOL, TYPE_LIBRARY),
            limit=10,
            reasoning="Test query with multiple entity types",
            confidence=1.0
        )
        operations = LLMGraphQueryOperations(
            entity_queries=[entity_query],
            reasoning="Test operations",
            confidence=1.0
        )

        # Translate to query - this will hit line 706 when iterating through entity_types[1:]
        query = db.translate_to_query(operations)

        # Verify the query was created successfully
        assert query is not None
        assert query.filters is not None

    def test_translate_entity_query_with_expansion(self) -> None:
        """Test translating entity query with neighbor expansion (line 728)."""
        db = GraphDatabase()

        # Create connected entities
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_CONCEPT)
        entity3 = Entity(name="Entity3", type=TYPE_CONCEPT)
        db.add_entities([entity1, entity2, entity3])

        # Add relationships
        db.create_relationship(entity1.id, entity2.id, REL_TYPE_RELATED_TO)
        db.create_relationship(entity2.id, entity3.id, REL_TYPE_RELATED_TO)

        # Create LLM query with expansion
        entity_query = LLMEntityQuery(
            entity_types=(TYPE_CONCEPT,),
            expand_depth=2,
            expand_direction="outgoing",
            reasoning="Test query with neighbor expansion",
            confidence=1.0
        )
        operations = LLMGraphQueryOperations(
            entity_queries=[entity_query],
            reasoning="Test operations with expansion",
            confidence=1.0
        )

        # Translate to query
        query = db.translate_to_query(operations)
        result = query.execute()

        # Should include all three entities due to expansion
        assert len(result.results) == 3

    def test_incremental_update_without_tantivy_index(self) -> None:
        """Test updating entity when Tantivy index is not initialized (lines 833-834)."""
        db = GraphDatabase()

        # Add entity without initializing Tantivy (should not trigger index)
        entity = Entity(name="TestEntity", type=TYPE_CONCEPT)
        db.add_entity(entity)

        # Update entity - should not crash even without Tantivy index
        entity.content = "Updated content"
        db.update_entity(entity)

        # Verify entity was updated
        updated = db.get_entity(entity.id)
        assert updated is not None
        assert updated.content == "Updated content"

    def test_incremental_remove_without_tantivy_index(self) -> None:
        """Test removing entity when Tantivy index is not initialized (line 862)."""
        db = GraphDatabase()

        # Add entity without initializing Tantivy
        entity = Entity(name="TestEntity", type=TYPE_CONCEPT)
        db.add_entity(entity)

        # Delete entity - should not crash even without Tantivy index
        db.delete_entity(entity.id)

        # Verify entity was deleted
        assert db.get_entity(entity.id) is None

    def test_from_dict_without_index_data(self) -> None:
        """Test deserializing from dict without index data (lines 923-926)."""
        # Create entities and relationships data
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_CONCEPT)
        rel = Relationship(source=entity1.id, target=entity2.id, type=REL_TYPE_RELATED_TO)

        # Create data dict without index
        data: dict[str, Any] = {
            "entities": [entity1.to_dict(), entity2.to_dict()],
            "relationships": [rel.to_dict()],
            "search_index_initialized": False
        }

        # Deserialize
        db = GraphDatabase.from_dict(data)

        # Verify entities and relationships were restored
        assert db.entity_count == 2
        assert db.relationship_count == 1
        assert db.get_entity(entity1.id) is not None
        assert db.get_entity(entity2.id) is not None
        assert db.get_relationship(rel.id) is not None

    def test_entity_type_filter_without_index(self) -> None:
        """Test entity type filter without using index (line 996)."""
        filter_obj = EntityTypeFilter(entity_type=TYPE_CONCEPT)

        # Create entities directly
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_TOOL)
        entity3 = Entity(name="Entity3", type=TYPE_CONCEPT)
        entities = [entity1, entity2, entity3]

        # Apply filter without index
        result_ids = filter_obj.apply(entities, index=None)

        # Should return only concept entities
        assert result_ids == {entity1.id, entity3.id}

    def test_relationship_type_filter_without_index(self) -> None:
        """Test relationship type filter without using index (line 1017)."""
        filter_obj = RelationshipTypeFilter(relationship_type=REL_TYPE_RELATED_TO)

        # Create relationships directly
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_CONCEPT)

        rel1 = Relationship(source=entity1.id, target=entity2.id, type=REL_TYPE_RELATED_TO)
        rel2 = Relationship(source=entity2.id, target=entity1.id, type=REL_TYPE_USES)
        relationships = [rel1, rel2]

        # Apply filter without index
        result_ids = filter_obj.apply(relationships, index=None)

        # Should return only related_to relationships
        assert result_ids == {rel1.id}

    def test_entity_query_or_type_without_existing_filters(self) -> None:
        """Test or_type when no existing type filters exist (lines 1149-1150)."""
        db = GraphDatabase()

        # Create query
        query = db.query_entities()

        # Add or_type without any existing type filters
        query = query.or_type(TYPE_CONCEPT)

        # Execute query
        entity = Entity(name="TestEntity", type=TYPE_CONCEPT)
        db.add_entity(entity)

        result = query.execute()
        assert len(result.results) == 1
        assert result.results[0].type == TYPE_CONCEPT

    def test_entity_date_filter_without_index(self) -> None:
        """Test entity date filter without using index (lines 1182-1188)."""
        db = GraphDatabase()

        # Create entities with different dates
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_CONCEPT)

        # Manually set created_at dates
        entity1.created_at = datetime(2023, 1, 1, tzinfo=UTC)
        entity2.created_at = datetime(2023, 3, 1, tzinfo=UTC)

        db.add_entities([entity1, entity2])

        # Query with date range
        query = db.query_entities().created_between(
            start_date=date(2023, 2, 1),
            end_date=date(2023, 4, 1)
        )

        result = query.execute()

        # Should only return entity2
        assert len(result.results) == 1
        assert result.results[0].id == entity2.id

    def test_entity_query_sort_with_none_field(self) -> None:
        """Test entity query sorting when sort_field is None (line 1341)."""
        db = GraphDatabase()

        # Create entities
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_CONCEPT)
        db.add_entities([entity1, entity2])

        # Create query with None sort_field (simulated internally)
        query = db.query_entities()
        query.sort_field = None  # Simulate internal state
        query.sort_ascending = True

        result = query.execute()

        # Should return all entities
        assert len(result.results) == 2

    def test_relationship_query_or_type_without_existing_filters(self) -> None:
        """Test or_type for relationships without existing filters (lines 1435-1436)."""
        db = GraphDatabase()

        # Add entities and relationships
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_CONCEPT)
        db.add_entities([entity1, entity2])

        rel1 = db.create_relationship(entity1.id, entity2.id, REL_TYPE_RELATED_TO)

        # Create query
        query = db.query_relationships()

        # Add or_type without any existing type filters
        query = query.or_type(REL_TYPE_RELATED_TO)

        result = query.execute()

        # Should return the relationship
        assert len(result.results) == 1
        assert result.results[0][2].id == rel1.id

    def test_relationship_query_sort_with_none_field(self) -> None:
        """Test relationship query sorting when sort_field is None (line 1541)."""
        db = GraphDatabase()

        # Add entities and relationships
        entity1 = Entity(name="Entity1", type=TYPE_CONCEPT)
        entity2 = Entity(name="Entity2", type=TYPE_CONCEPT)
        db.add_entities([entity1, entity2])

        rel = db.create_relationship(entity1.id, entity2.id, REL_TYPE_RELATED_TO)

        # Create query with None sort_field (simulated internally)
        query = db.query_relationships()
        query.sort_field = None  # Simulate internal state
        query.sort_ascending = True

        result = query.execute()

        # Should return the relationship
        assert len(result.results) == 1
        assert result.results[0][2].id == rel.id
