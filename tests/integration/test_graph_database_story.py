"""Graph database scenario test with three little pigs story."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
import scenario

from blockether_foundation.graph import (
    Entity,
    GraphDatabase,
    LLMGraphOperations,
    Relationship,
)
from tests.integration.utils import create_agent_with_adapter  # type: ignore

# Type alias for relationship query results with entities
RelationshipWithEntities = tuple[Entity, Entity, Relationship]

# Expected counts for validation
EXPECTED_CHARACTER_COUNT = 5  # Mother Pig + 3 Little Pigs + Big Bad Wolf
EXPECTED_HOUSE_COUNT = 3  # straw, stick, brick houses
EXPECTED_LOCATION_COUNT = 4  # 3 houses + village
EXPECTED_TOTAL_ENTITY_COUNT = 9  # 5 characters + 4 locations

# Test constants
MIN_BUILD_RELATIONSHIPS = 2
MIN_WOLF_DESTRUCTIONS = 1
MIN_PIGS_IN_BRICK_HOUSE = 1
MIN_CREATED_BY_RELATIONSHIPS = 3
MIN_WOLF_RELATIONSHIPS = 2
MIN_BRICK_HOUSE_CONNECTIONS = 2
MIN_PIGS_IN_HOUSES = 2
MIN_HOUSE_BUILDING_RELATIONSHIPS = 3
MIN_EXPECTED_RELATIONSHIP_TYPES = 3
MIN_WOLF_ATTACKED_HOUSES = 2
MAX_LIMITED_RELATIONSHIPS = 2

# Character and location constants
STORY_CHARACTERS = {
    "Mother Pig",
    "First Little Pig",
    "Second Little Pig",
    "Third Little Pig",
    "Big Bad Wolf",
}

STORY_HOUSES = {
    "straw house",
    "stick house",
    "brick house",
}

REQUIRED_CHARACTERS = {
    "First Little Pig",
    "Second Little Pig",
    "Third Little Pig",
    "Big Bad Wolf",
}

REQUIRED_LOCATIONS = {
    "straw house",
    "stick house",
    "brick house",
}

STORY_RELATIONSHIP_TYPES = {
    "created_by",
    "invalidates",
    "belongs_to",
    "related_to",
}

WOLF_TARGET_HOUSES = ["straw house", "stick house"]


def validate_three_little_pigs_operations(operations: LLMGraphOperations) -> GraphDatabase:
    """Validate extracted operations and import them into a graph database.

    Args:
        operations: The graph operations extracted by the agent

    Returns:
        GraphDatabase with all operations applied

    Raises:
        AssertionError: If validation fails
    """
    # Basic validation to ensure operations exist
    assert operations.add_entity_ops, "No entities extracted"
    assert operations.add_relationship_ops, "No relationships extracted"

    character_entities = {
        entity.name: entity for entity in operations.add_entity_ops if entity.type == "person"
    }
    location_entities = {
        entity.name: entity for entity in operations.add_entity_ops if entity.type == "location"
    }

    # Validate key characters exist
    missing_characters = REQUIRED_CHARACTERS - set(character_entities.keys())
    assert not missing_characters, f"Missing key characters: {missing_characters}"

    # Validate key locations exist
    missing_locations = REQUIRED_LOCATIONS - set(location_entities.keys())
    assert not missing_locations, f"Missing key locations: {missing_locations}"

    # Validate build relationships
    relationships = operations.add_relationship_ops
    build_relationships = {
        (rel.source_name, rel.target_name) for rel in relationships if rel.type == "created_by"
    }

    # Check at least some build relationships exist
    assert len(build_relationships) >= MIN_BUILD_RELATIONSHIPS, (
        f"Expected at least {MIN_BUILD_RELATIONSHIPS} build relationships, got {len(build_relationships)}"
    )

    # Validate wolf destruction relationships - story accuracy
    destroy_relationships = {
        (rel.source_name, rel.target_name) for rel in relationships if rel.type == "invalidates"
    }

    # Wolf should destroy some houses
    wolf_destroys = {
        (source, target)
        for source, target in destroy_relationships
        if source == "Big Bad Wolf" and target in WOLF_TARGET_HOUSES
    }
    assert len(wolf_destroys) >= MIN_WOLF_DESTRUCTIONS, (
        f"Wolf should destroy at least {MIN_WOLF_DESTRUCTIONS} house, got {len(wolf_destroys)}"
    )

    # Brick house should not be destroyed
    brick_destroys = {
        (source, target) for source, target in destroy_relationships if target == "brick house"
    }
    assert not brick_destroys, "Brick house should not be destroyed"

    # Validate residence relationships
    residence_relationships = {
        (rel.source_name, rel.target_name) for rel in relationships if rel.type == "belongs_to"
    }

    # At least one pig should belong to brick house
    pigs_in_brick_house = {
        source
        for source, target in residence_relationships
        if target == "brick house" and "Pig" in source
    }
    assert len(pigs_in_brick_house) >= MIN_PIGS_IN_BRICK_HOUSE, (
        f"At least {MIN_PIGS_IN_BRICK_HOUSE} pig should be in brick house, got {len(pigs_in_brick_house)}"
    )

    # Create graph database and import operations
    # Duplicate relationships are now handled centrally by LLMGraphOperations
    # and GraphDatabase.import_operations.
    graph_db = GraphDatabase()
    graph_db.import_operations(operations)

    # Validate database has content - expect at least characters + key locations
    min_expected_entities = EXPECTED_CHARACTER_COUNT + EXPECTED_HOUSE_COUNT
    all_entities_result = graph_db.query_entities().execute()
    assert len(all_entities_result.results) >= min_expected_entities, (
        f"Should have at least {min_expected_entities} entities in database"
    )

    return graph_db


class GraphValidationStep:
    """Custom scenario step that validates and imports graph operations."""

    def __init__(self) -> None:
        """Initialize the validation step."""
        self.operations: LLMGraphOperations | None = None
        self.graph_db: GraphDatabase | None = None

    async def __call__(self, state: scenario.ScenarioState) -> None:
        """Extract operations from agent response and validate them."""
        # Get the last message from the scenario state
        last_message = state.last_message()

        # Verify it's an assistant message
        assert last_message.get("role") == "assistant", (
            f"Expected assistant message, got: {last_message.get('role')}"
        )

        # Extract content from the assistant message
        message_content = last_message.get("content")

        # Parse JSON string if needed - use comprehension for conditional parsing
        assert message_content is not None, "No content found in assistant message"

        # Parse JSON string using accumulation pattern
        # Handle different message content types
        content_str = str(message_content)
        is_json_string = content_str.strip().startswith("{")

        if is_json_string:
            parsed_content: dict[str, Any] | str = json.loads(content_str)
        else:
            if isinstance(message_content, str):
                parsed_content = message_content
            else:
                parsed_content = {"content": message_content}

        assert isinstance(parsed_content, dict), (
            f"Message content is not a dictionary: {type(parsed_content)}"
        )
        message_content = parsed_content

        # Extract operations from structured content - the LLMGraphOperations is the root object
        operations_data = message_content
        self.operations = LLMGraphOperations.model_validate(operations_data)

        # Validate and import operations
        self.graph_db = validate_three_little_pigs_operations(self.operations)


@pytest.mark.integration
@pytest.mark.agent_test
async def test_three_little_pigs_graph_extraction():
    """Test extracting entities/relationships from story using Scenario framework."""

    # The three little pigs story
    story = """
    Once upon a time, there were Three Little Pigs who lived with their Mother Pig.
    The First Little Pig decided to build a house of straw.
    The Second Little Pig built a house of sticks.
    The Third Little Pig built a house of bricks.

    One day, a Big Bad Wolf came to the village. The Wolf huffed and puffed and blew down the straw house.
    The First Little Pig ran to the Second Little Pig's stick house.
    The Wolf followed and blew down the stick house too.

    Both pigs ran to the Third Little Pig's brick house. The Wolf tried but could not blow down the brick house.
    The Wolf tried to climb down the chimney, but the Third Little Pig put a pot of boiling water on the fireplace.
    The Wolf fell into the pot and ran away, never to bother the pigs again.

    The Three Little Pigs lived happily ever after in their brick house.
    """

    # Create validation step
    validation_step = GraphValidationStep()

    # Create graph extractor agent
    graph_extractor = create_agent_with_adapter(  # type: ignore
        name="GraphExtractor",
        instructions="""
        You are an expert at extracting entities and relationships from stories.

        Extract all characters, objects, and their relationships from the story provided by the user.

        For entities:
        - Characters should be type "person" (Mother Pig, First Little Pig, Second Little Pig, Third Little Pig, Big Bad Wolf)
        - Objects should be type "location" (straw house, stick house, brick house, village)

        For relationships:
        - Use "created_by" for house → pig relationships (who built the house)
        - Use "invalidates" for wolf → house relationships (houses destroyed by wolf)
        - Use "belongs_to" for pig → house relationships (where pigs live)
        - Use "related_to" for family relationships

        Make sure all referenced entities exist before creating relationships.
        Return the operations as a single LLMGraphOperations object.
        """,
        output_schema=LLMGraphOperations,
    )

    # Run the scenario with validation step
    result = await scenario.run(
        name="Three Little Pigs Graph Extraction",
        description=f"""
        Extract entities and relationships from the Three Little Pigs story.
        The story is: {story}
        """,
        set_id="graph-database-tests",
        agents=[
            graph_extractor,
            scenario.UserSimulatorAgent(),
        ],
        script=[
            # User provides the story
            scenario.user(story),
            # Agent extracts graph operations
            scenario.agent(),
            # Validate and import the operations
            validation_step,
            # Succeed with the results
            scenario.succeed("Graph extraction and import successful"),
        ],
        max_turns=1,
    )

    assert result.success, f"Scenario failed: {result.passed_criteria}"
    assert validation_step.operations is not None, "No operations were extracted"
    assert validation_step.graph_db is not None, "Graph database was not created"

    # Validate the story content matches expectations
    validate_story_content(validation_step.graph_db)


def validate_story_content(db: GraphDatabase) -> None:
    """Validate that the Three Little Pigs story has correct entities and relationships."""

    # Test 1: Validate character entities
    characters = db.query_entities().type("person").execute().results
    character_names = {entity.name for entity in characters}
    missing_characters = STORY_CHARACTERS - character_names
    assert not missing_characters, f"Missing characters: {missing_characters}"

    # Test 2: Validate house entities
    houses = db.query_entities().search("house").execute().results
    house_names = {entity.name for entity in houses}
    missing_houses = STORY_HOUSES - house_names
    assert not missing_houses, f"Missing houses: {missing_houses}"

    # Test 3: Test relationship queries - all relationships involving the wolf
    wolf_relationships_result = (
        db.query_relationships().from_entity(entity_name="Big Bad Wolf").execute()
    )
    wolf_relationships = cast(list[RelationshipWithEntities], wolf_relationships_result.results)
    assert len(wolf_relationships) > 0, "Wolf should have relationships"

    wolf_targets = {target.name for source, target, rel in wolf_relationships}
    assert any(house in wolf_targets for house in WOLF_TARGET_HOUSES), (
        "Wolf should have house relationships"
    )

    # Test 4: Test relationship type filtering
    created_by_relationships = cast(
        list[RelationshipWithEntities],
        db.query_relationships().type("created_by").execute().results,
    )
    assert len(created_by_relationships) >= MIN_CREATED_BY_RELATIONSHIPS, (
        f"Should have at least {MIN_CREATED_BY_RELATIONSHIPS} 'created_by' relationships"
    )

    # Verify each 'created_by' relationship connects a pig to a house using accumulation
    pig_involvement = all(
        ("Pig" in source.name or "Pig" in target.name)
        and ("house" in source.name or "house" in target.name)
        for source, target, rel in created_by_relationships
    )
    assert pig_involvement, "Created_by should involve pigs and houses"

    # Test 5: Test target entity filtering - find entities that are targets
    house_targets = cast(
        list[RelationshipWithEntities],
        db.query_relationships().to_entity(entity_name="straw house").execute().results,
    )
    straw_house_sources = {source.name for source, target, rel in house_targets}
    assert "First Little Pig" in straw_house_sources, (
        "First Little Pig should be connected to straw house"
    )

    # Test 6: Test combined entity + relationship filtering
    # Find pigs and their relationships
    pigs = db.query_entities().type("person").search("Pig").execute().results
    pig_houses: list[RelationshipWithEntities] = [
        rel
        for pig in pigs
        for rel in cast(
            list[RelationshipWithEntities],
            db.query_relationships().from_entity(entity_name=pig.name).execute().results,
        )
    ]

    assert len(pig_houses) >= MIN_PIGS_IN_HOUSES, "Should have pigs with relationships"

    # Test 7: Test search + relationship filtering
    house_building_relationships = cast(
        list[RelationshipWithEntities],
        db.query_relationships().type("created_by").execute().results,
    )

    # Should find house-building relationships
    assert len(house_building_relationships) >= MIN_HOUSE_BUILDING_RELATIONSHIPS, (
        "Should find house creation relationships"
    )

    # Test 8: Test sorting relationships
    sorted_relationships = cast(
        list[RelationshipWithEntities],
        db.query_relationships()
        .type("created_by")
        .order_by("created_at", ascending=True)
        .execute()
        .results,
    )

    # Validate sorting using accumulation pattern - properly unpack tuples
    sorted_validation = all(
        sorted_relationships[i][2].created_at <= sorted_relationships[i + 1][2].created_at
        for i in range(len(sorted_relationships) - 1)
    )
    assert sorted_validation, "Relationships should be sorted by creation timestamp"

    # Test 9: Test limit on relationship results
    limited_relationships = cast(
        list[RelationshipWithEntities],
        db.query_relationships()
        .from_entity(entity_name="Big Bad Wolf")
        .limit(MAX_LIMITED_RELATIONSHIPS)
        .execute()
        .results,
    )

    assert len(limited_relationships) <= MAX_LIMITED_RELATIONSHIPS, (
        f"Should limit results to {MAX_LIMITED_RELATIONSHIPS} relationships"
    )

    # Test 10: Complex query - find all entities connected to brick house
    brick_house_incoming = cast(
        list[RelationshipWithEntities],
        db.query_relationships().to_entity(entity_name="brick house").execute().results,
    )
    brick_house_outgoing = cast(
        list[RelationshipWithEntities],
        db.query_relationships().from_entity(entity_name="brick house").execute().results,
    )

    # Combine both incoming and outgoing connections
    connected_entities: set[str] = set()
    connected_entities.update({source.name for source, target, rel in brick_house_incoming})
    connected_entities.update({target.name for source, target, rel in brick_house_outgoing})

    assert len(connected_entities) >= MIN_BRICK_HOUSE_CONNECTIONS, (
        "Brick house should have multiple connections"
    )

    # Test 11: Validate the complete story graph structure
    all_relationships = cast(
        list[RelationshipWithEntities],
        db.query_relationships().execute().results,
    )
    relationship_types = {rel.type for source, target, rel in all_relationships}

    # Should have key relationship types from the story
    found_relationship_types = STORY_RELATIONSHIP_TYPES.intersection(relationship_types)
    assert len(found_relationship_types) >= MIN_EXPECTED_RELATIONSHIP_TYPES, (
        f"Missing key relationship types. Found: {relationship_types}"
    )

    # Test 12: Verify story completeness - wolf attacked houses
    wolf_attacks_result = (
        db.query_relationships()
        .from_entity(entity_name="Big Bad Wolf")
        .type("invalidates")
        .execute()
    )
    wolf_attacks = cast(
        list[RelationshipWithEntities],
        wolf_attacks_result.results,
    )

    attacked_houses = {
        target.name for source, target, rel in wolf_attacks if "house" in target.name
    }
    assert len(attacked_houses) >= MIN_WOLF_ATTACKED_HOUSES, (
        "Wolf should have attacked multiple houses"
    )

    # Final comprehensive assertion to ensure all key story elements are present
    assert all(
        [
            len(character_names) >= EXPECTED_CHARACTER_COUNT,
            len(house_names) >= EXPECTED_HOUSE_COUNT,
            len(wolf_attacks) >= MIN_WOLF_ATTACKED_HOUSES,
            len(found_relationship_types) >= MIN_EXPECTED_RELATIONSHIP_TYPES,
        ]
    ), "Story validation failed: missing key story elements"
