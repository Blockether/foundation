"""Tests for graph indices to ensure proper indexing functionality."""

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from blockether_foundation.graph.indices import GraphIndex
from blockether_foundation.graph.models import Entity, Relationship

# Constants
SAMPLE_ENTITY_COUNT = 3


class TestGraphIndex:
    """Test cases for GraphIndex functionality."""

    @pytest.fixture
    def sample_entities(self) -> list[Entity]:
        """Create sample entities for testing."""
        now = datetime.now(UTC)
        yesterday = now - timedelta(days=1)

        return [
            Entity(
                id="entity1",
                name="Test Entity One",
                type="person",
                content="This is a test entity with some content words",
                created_at=yesterday,
                updated_at=now,
            ),
            Entity(
                id="entity2",
                name="Another Entity",
                type="organization",
                content="Different content with unique words",
                created_at=now,
                updated_at=now,
            ),
            Entity(
                id="entity3",
                name="Third Entity",
                type="concept",
                content="Project related content with technical terms",
                created_at=now,
                updated_at=now,
            ),
        ]

    @pytest.fixture
    def sample_relationships(self) -> list[Relationship]:
        """Create sample relationships for testing."""
        return [
            Relationship(
                id="rel1",
                type="belongs_to",
                source="entity1",
                target="entity2",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            ),
            Relationship(
                id="rel2",
                type="part_of",
                source="entity3",
                target="entity2",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            ),
        ]

    @pytest.mark.unit
    def test_add_entity_indexes_all_fields(self, sample_entities: list[Entity]) -> None:
        """Test that adding an entity updates all relevant indices."""
        index = GraphIndex()
        entity = sample_entities[0]

        # Add entity
        index.add_entity(entity)

        # Check all indices are populated
        assert entity.id in index.entity_by_id
        assert index.entity_by_id[entity.id] == entity

        assert entity.name in index.entity_by_name
        assert index.entity_by_name[entity.name] == entity.id

        assert entity.id in index.entities_by_type[entity.type]

        created_date = entity.created_at.date()
        assert entity.id in index.entities_by_created_date[created_date]

        updated_date = entity.updated_at.date()
        assert entity.id in index.entities_by_updated_date[updated_date]

        # Check prefix indices
        assert entity.id in index.entities_by_id_prefix["ent"]
        assert entity.id in index.entities_by_id_prefix["entity"]
        assert entity.id in index.entities_by_id_prefix[entity.id.lower()]

        # Check name prefix indices
        assert entity.id in index.entities_by_name_prefix["test"]
        assert entity.id in index.entities_by_name_prefix["test entity"]

        # Content words are no longer indexed - using Tantivy for search instead

    @pytest.mark.unit
    def test_remove_entity_cleans_all_indices(self, sample_entities: list[Entity]) -> None:
        """Test that removing an entity cleans all relevant indices."""
        index = GraphIndex()
        entity = sample_entities[0]

        # Add entity first
        index.add_entity(entity)
        assert entity.id in index.entity_by_id

        # Remove entity
        index.remove_entity(entity)

        # Check all indices are cleaned
        assert entity.id not in index.entity_by_id
        assert entity.name not in index.entity_by_name
        assert entity.id not in index.entities_by_type[entity.type]
        assert entity.id not in index.entities_by_created_date[entity.created_at.date()]
        assert entity.id not in index.entities_by_updated_date[entity.updated_at.date()]

        # Check prefix indices are cleaned
        assert entity.id not in index.entities_by_id_prefix["entity"]
        assert entity.id not in index.entities_by_name_prefix["test"]

        # Content words are no longer indexed - using Tantivy for search instead

    @pytest.mark.unit
    def test_update_entity_reindexes_correctly(self, sample_entities: list[Entity]) -> None:
        """Test that updating an entity properly reindexes it."""
        index = GraphIndex()
        old_entity = sample_entities[0]

        # Add original entity
        index.add_entity(old_entity)
        assert old_entity.name in index.entity_by_name

        # Create updated entity with different name
        updated_entity = Entity(
            id=old_entity.id,
            name="Updated Entity Name",
            type="concept",  # Different type
            content="Completely new content with different words",
            created_at=old_entity.created_at,
            updated_at=datetime.now(UTC),
        )

        # Update entity
        index.update_entity(old_entity, updated_entity)

        # Check old name is gone
        assert old_entity.name not in index.entity_by_name
        assert updated_entity.name in index.entity_by_name

        # Check type is updated
        assert old_entity.id not in index.entities_by_type[old_entity.type]
        assert old_entity.id in index.entities_by_type[updated_entity.type]

        # Content words are no longer indexed - using Tantivy for search instead

    @pytest.mark.unit
    def test_add_relationship_updates_indices(
        self, sample_relationships: list[Relationship]
    ) -> None:
        """Test that adding a relationship updates all relevant indices."""
        index = GraphIndex()
        rel = sample_relationships[0]

        # Add relationship
        index.add_relationship(rel)

        # Check all indices are populated
        assert rel.id in index.relationship_by_id
        assert index.relationship_by_id[rel.id] == rel

        assert rel.id in index.relationships_by_type[rel.type]

        assert rel.id in index.outgoing_edges[rel.source]
        assert rel.id in index.incoming_edges[rel.target]

    @pytest.mark.unit
    def test_remove_relationship_cleans_indices(
        self, sample_relationships: list[Relationship]
    ) -> None:
        """Test that removing a relationship cleans all relevant indices."""
        index = GraphIndex()
        rel = sample_relationships[0]

        # Add relationship first
        index.add_relationship(rel)
        assert rel.id in index.relationship_by_id

        # Remove relationship
        index.remove_relationship(rel)

        # Check all indices are cleaned
        assert rel.id not in index.relationship_by_id
        assert rel.id not in index.relationships_by_type[rel.type]
        assert rel.id not in index.outgoing_edges[rel.source]
        assert rel.id not in index.incoming_edges[rel.target]

    @pytest.mark.unit
    def test_clear_resets_all_indices(
        self, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ) -> None:
        """Test that clear() resets all indices to empty state."""
        index = GraphIndex()

        # Add some data
        list(map(index.add_entity, sample_entities))
        list(map(index.add_relationship, sample_relationships))

        # Verify data exists
        assert len(index.entity_by_id) > 0
        assert len(index.relationship_by_id) > 0

        # Clear all indices
        index.clear()

        # Verify all indices are empty
        assert len(index.entity_by_id) == 0
        assert len(index.relationship_by_id) == 0
        assert len(index.entity_by_name) == 0
        assert len(index.entities_by_type) == 0
        assert len(index.relationships_by_type) == 0
        assert len(index.entities_by_created_date) == 0
        assert len(index.entities_by_updated_date) == 0
        assert len(index.outgoing_edges) == 0
        assert len(index.incoming_edges) == 0
        assert len(index.entities_by_id_prefix) == 0
        assert len(index.entities_by_name_prefix) == 0

    @pytest.mark.unit
    def test_to_dict_serialization(
        self, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ) -> None:
        """Test serialization to dictionary."""
        index = GraphIndex()

        # Add test data
        list(map(index.add_entity, sample_entities))
        list(map(index.add_relationship, sample_relationships))

        # Serialize to dict
        data = index.to_dict()

        # Check structure
        assert "entity_by_id" in data
        assert "relationship_by_id" in data
        assert "entity_by_name" in data
        assert "entities_by_type" in data
        assert "relationships_by_type" in data
        assert "entities_by_created_date" in data
        assert "entities_by_updated_date" in data
        assert "outgoing_edges" in data
        assert "incoming_edges" in data
        assert "entities_by_id_prefix" in data
        assert "entities_by_name_prefix" in data

        # Check data integrity
        entity_by_id = cast(dict[str, Any], data["entity_by_id"])
        relationship_by_id = cast(dict[str, Any], data["relationship_by_id"])
        entities_by_type = cast(dict[str, list[str]], data["entities_by_type"])
        relationships_by_type = cast(dict[str, list[str]], data["relationships_by_type"])

        assert len(entity_by_id) == SAMPLE_ENTITY_COUNT
        assert len(relationship_by_id) == 2
        assert "person" in entities_by_type
        assert "belongs_to" in relationships_by_type

    @pytest.mark.unit
    def test_from_dict_deserialization(
        self, sample_entities: list[Entity], sample_relationships: list[Relationship]
    ) -> None:
        """Test deserialization from dictionary."""
        original_index = GraphIndex()

        # Add test data
        list(map(original_index.add_entity, sample_entities))
        list(map(original_index.add_relationship, sample_relationships))

        # Serialize
        data = original_index.to_dict()

        # Deserialize
        restored_index = GraphIndex.from_dict(data)

        # Verify restoration
        assert len(restored_index.entity_by_id) == len(original_index.entity_by_id)
        assert len(restored_index.relationship_by_id) == len(original_index.relationship_by_id)

        # Check specific data
        assert all(entity.id in restored_index.entity_by_id for entity in sample_entities)
        assert all(entity.name in restored_index.entity_by_name for entity in sample_entities)
        assert all(
            entity.id in restored_index.entities_by_type[entity.type] for entity in sample_entities
        )

        assert all(rel.id in restored_index.relationship_by_id for rel in sample_relationships)
        assert all(
            rel.id in restored_index.relationships_by_type[rel.type] for rel in sample_relationships
        )
        assert all(
            rel.id in restored_index.outgoing_edges[rel.source] for rel in sample_relationships
        )
        assert all(
            rel.id in restored_index.incoming_edges[rel.target] for rel in sample_relationships
        )

    @pytest.mark.unit
    def test_case_insensitive_prefix_search(self) -> None:
        """Test that prefix searches are case insensitive."""
        index = GraphIndex()

        entity = Entity(
            id="TestEntity123",
            name="Test Name",
            type="concept",
            content="Test content",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        index.add_entity(entity)

        # Test prefixes (indexing uses lowercase)
        assert entity.id in index.entities_by_id_prefix["test"]
        assert entity.id not in index.entities_by_id_prefix["TEST"]  # Uppercase not indexed
        assert entity.id not in index.entities_by_id_prefix["Test"]  # Mixed case not indexed

        assert entity.id in index.entities_by_name_prefix["test"]
        assert entity.id not in index.entities_by_name_prefix["TEST"]  # Uppercase not indexed
        assert entity.id not in index.entities_by_name_prefix["Test"]  # Mixed case not indexed

    @pytest.mark.unit
    def test_date_based_indexing(self) -> None:
        """Test that entities are properly indexed by date."""
        index = GraphIndex()

        today = datetime.now(UTC)
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)

        entities = [
            Entity(
                id="e1",
                name="Entity 1",
                type="concept",
                content="Content 1",
                created_at=yesterday,
                updated_at=today,
            ),
            Entity(
                id="e2",
                name="Entity 2",
                type="person",
                content="Content 2",
                created_at=last_week,
                updated_at=last_week,
            ),
        ]

        list(map(index.add_entity, entities))

        # Check date indices
        today_date = today.date()
        yesterday_date = yesterday.date()
        last_week_date = last_week.date()

        assert "e1" in index.entities_by_created_date[yesterday_date]
        assert "e2" in index.entities_by_created_date[last_week_date]

        assert "e1" in index.entities_by_updated_date[today_date]
        assert "e2" in index.entities_by_updated_date[last_week_date]
