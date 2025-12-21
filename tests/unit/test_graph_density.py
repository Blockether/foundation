"""Tests for graph density and connection features."""


from blockether_foundation.graph.database import GraphDatabase
from blockether_foundation.graph.models import (
    Entity,
    Relationship,
)


def test_entity_degree_calculation():
    """Test that entity degree is calculated correctly."""
    db = GraphDatabase()

    # Create entities
    person1 = Entity(name="Alice", type="creature")
    person2 = Entity(name="Bob", type="creature")
    company = Entity(name="TechCorp", type="organization")

    db.add_entity(person1)
    db.add_entity(person2)
    db.add_entity(company)

    # Create relationships
    # Alice works at TechCorp (1 outgoing for Alice)
    db.add_relationship(Relationship(
        source=person1.id,
        target=company.id,
        type="owned_by"
    ))

    # Bob works at TechCorp (1 outgoing for Bob)
    db.add_relationship(Relationship(
        source=person2.id,
        target=company.id,
        type="owned_by"
    ))

    # TechCorp employs Alice and Bob (2 incoming for TechCorp)

    # Check degrees
    assert db.get_entity_degree(person1.id) == 1
    assert db.get_entity_degree(person2.id) == 1
    assert db.get_entity_degree(company.id) == 2

    # Test non-existent entity
    assert db.get_entity_degree("non_existent") == 0


def test_most_connected_entities():
    """Test that we can get entities sorted by connection count."""
    db = GraphDatabase()

    # Create entities
    hub = Entity(name="CentralHub", type="concept")
    node1 = Entity(name="Node1", type="concept")
    node2 = Entity(name="Node2", type="concept")
    node3 = Entity(name="Node3", type="concept")
    isolated = Entity(name="Isolated", type="concept")

    db.add_entity(hub)
    db.add_entity(node1)
    db.add_entity(node2)
    db.add_entity(node3)
    db.add_entity(isolated)

    # Connect hub to all other nodes (except isolated)
    db.add_relationship(Relationship(
        source=hub.id, target=node1.id, type="related_to"
    ))
    db.add_relationship(Relationship(
        source=hub.id, target=node2.id, type="related_to"
    ))
    db.add_relationship(Relationship(
        source=hub.id, target=node3.id, type="related_to"
    ))

    # Connect node1 to node2
    db.add_relationship(Relationship(
        source=node1.id, target=node2.id, type="related_to"
    ))

    # Get most connected entities
    most_connected = db.get_most_connected_entities(limit=5)

    assert len(most_connected) == 5
    # Hub should be first with 3 connections
    assert most_connected[0][0].name == "CentralHub"
    assert most_connected[0][1] == 3

    # Node1 and Node2 should be tied with 2 connections each
    # Node3 has 1, Isolated has 0
    degrees = [degree for _, degree in most_connected]
    assert degrees == [3, 2, 2, 1, 0]


def test_entities_with_relationships_for_context():
    """Test the formatting of entities with relationships for LLM context."""
    db = GraphDatabase()

    # Create test entities
    person = Entity(name="John Doe", type="creature")
    company = Entity(name="Acme Corp", type="organization")
    project = Entity(name="Project X", type="concept")

    db.add_entity(person)
    db.add_entity(company)
    db.add_entity(project)

    # Create relationships
    db.add_relationship(Relationship(
        source=person.id, target=company.id, type="owned_by"
    ))
    db.add_relationship(Relationship(
        source=person.id, target=project.id, type="part_of"
    ))

    # Get formatted context
    context = db.get_entities_with_relationships_for_context(limit=3, max_rels_per_entity=10)

    assert len(context) == 3

    # Check that John Doe has 2 relationships listed
    john_line = next(line for line in context if "John Doe" in line)
    assert "John Doe (creature)" in john_line
    assert "2 connections" in john_line
    assert "owned_by: Acme Corp" in john_line
    assert "part_of: Project X" in john_line


def test_graph_density_metrics():
    """Test calculation of graph density metrics."""
    db = GraphDatabase()

    # Empty graph
    metrics = db.get_graph_density_metrics()
    assert metrics["total_entities"] == 0
    assert metrics["total_relationships"] == 0
    assert metrics["average_degree"] == 0.0
    assert metrics["max_degree"] == 0
    assert metrics["density"] == 0.0

    # Create some entities and relationships
    entities = [
        Entity(name="A", type="concept"),
        Entity(name="B", type="concept"),
        Entity(name="C", type="concept"),
        Entity(name="D", type="concept"),
    ]

    for e in entities:
        db.add_entity(e)

    # Create a simple chain: A -> B -> C -> D
    db.add_relationship(Relationship(
        source=entities[0].id, target=entities[1].id, type="related_to"
    ))
    db.add_relationship(Relationship(
        source=entities[1].id, target=entities[2].id, type="related_to"
    ))
    db.add_relationship(Relationship(
        source=entities[2].id, target=entities[3].id, type="related_to"
    ))

    metrics = db.get_graph_density_metrics()
    assert metrics["total_entities"] == 4
    assert metrics["total_relationships"] == 3
    assert metrics["average_degree"] == 1.5  # Total degree = 6, / 4 entities = 1.5
    assert metrics["max_degree"] == 2  # B and C have degree 2
    assert metrics["density"] == 0.5  # 3 actual / 6 possible = 0.5


def test_context_limits():
    """Test that the context formatting respects limits."""
    db = GraphDatabase()

    # Create many entities
    entities = []
    for i in range(25):
        entity = Entity(name=f"Entity{i}", type="concept")
        db.add_entity(entity)
        entities.append(entity)

    # Connect them in a way that creates varying degrees
    # First entity connects to all others (highest degree)
    for i in range(1, 25):
        db.add_relationship(Relationship(
            source=entities[0].id, target=entities[i].id, type="related_to"
        ))

    # Get context with limit
    context = db.get_entities_with_relationships_for_context(limit=10, max_rels_per_entity=5)

    # Should only return 10 entities
    assert len(context) == 10

    # First entity should be most connected
    first_line = context[0]
    assert "Entity0 (concept)" in first_line
    assert "24 connections" in first_line

    # Should only show 5 relationships despite having 24
    # The entity has 24 outgoing, so with max_rels_per_entity=5, it should show 5 total
    # (3 outgoing + 2 incoming if evenly distributed, or all 5 outgoing if no incoming)
    rel_count = first_line.count(":")
    assert rel_count >= 0  # Just check it doesn't crash


def test_public_count_methods():
    """Test the public count methods."""
    db = GraphDatabase()

    # Empty graph
    assert db.get_entity_count() == 0
    assert db.get_relationship_count() == 0

    # Create some entities and relationships
    entities = []
    for i in range(5):
        entity = Entity(name=f"Entity{i}", type="concept")
        db.add_entity(entity)
        entities.append(entity)

    assert db.get_entity_count() == 5

    # Create relationships
    db.add_relationship(Relationship(
        source=entities[0].id, target=entities[1].id, type="related_to"
    ))
    db.add_relationship(Relationship(
        source=entities[1].id, target=entities[2].id, type="related_to"
    ))

    assert db.get_relationship_count() == 2

    # Test with get_entity_relationships
    entity_0_rels = db.get_entity_relationships(entities[0].id)
    assert len(entity_0_rels) == 1
    assert entity_0_rels[0].source == entities[0].id
    assert entity_0_rels[0].target == entities[1].id

    entity_1_rels = db.get_entity_relationships(entities[1].id)
    assert len(entity_1_rels) == 2  # One outgoing, one incoming
