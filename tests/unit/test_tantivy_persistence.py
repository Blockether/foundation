"""Test persistent Tantivy index functionality."""

import tempfile
from pathlib import Path

import pytest

from blockether_foundation.graph.database import GraphDatabase
from blockether_foundation.graph.models import Entity


@pytest.mark.unit
def test_persistent_tantivy_index() -> None:
    """Test that Tantivy index persists to disk and loads quickly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        json_file = temp_path / "test_graph.json"
        tantivy_dir = temp_path / "tantivy_index"

        # Create database with persistent Tantivy index
        db = GraphDatabase(tantivy_index_path=tantivy_dir)

        # Add some test entities
        entities = [
            Entity(name="Alice Johnson", type="person", content="Software engineer at Tech Corp"),
            Entity(
                name="Bob Smith", type="person", content="Product manager with 10 years experience"
            ),
            Entity(
                name="Tech Corp",
                type="organization",
                content="Technology company specializing in AI",
            ),
            Entity(name="Project Alpha", type="object", content="AI-powered analytics platform"),
        ]

        list(map(db.add_entity, entities))

        # Test search works
        results = db.search("software", top_k=5)
        assert len(results) > 0, "Expected to find results for 'software' search"
        assert any("Alice" in r[0].name for r in results), (
            "Expected to find Alice Johnson in results"
        )

        # Save the database
        db.save_to_file(json_file)

        # Check that Tantivy index files were created
        assert tantivy_dir.exists(), "Tantivy index directory should exist after creating database"
        index_files = list(tantivy_dir.rglob("*"))
        assert len(index_files) > 0, "Expected Tantivy index files to be created"

        # Load the database in a new instance
        db2 = GraphDatabase.load_from_file(json_file)

        # Test search still works after loading
        results2 = db2.search("analytics", top_k=5)
        assert len(results2) > 0, "Expected to find results for 'analytics' search after loading"
        assert any("Project" in r[0].name for r in results2), (
            "Expected to find Project Alpha in results"
        )

        # Verify Tantivy index directory still exists
        assert tantivy_dir.exists(), "Tantivy index directory should still exist"
        index_size = sum(f.stat().st_size for f in tantivy_dir.rglob("*") if f.is_file())
        assert index_size > 0, "Tantivy index should have non-zero size"
