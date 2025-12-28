"""Test for search query parsing with special characters."""

from __future__ import annotations

import pytest

from blockether_foundation.graph import (
    Entity,
    LLMEntityQuery,
    LLMGraphQueryOperations,
)
from blockether_foundation.graph.database import GraphDatabase


class TestSearchSpecialCharacters:
    """Test that search queries handle special characters correctly."""

    @pytest.fixture
    def database_with_special_names(self) -> GraphDatabase:
        """Create database with entities having special characters in names."""
        db = GraphDatabase()

        # Create entities with special characters that caused the original error
        entities = [
            Entity(
                name="Project 2024-10-14 meeting",
                type="situation",
                content="A project meeting on October 14, 2024",
            ),
            Entity(
                name="Team Alpha-Bravo Review",
                type="concept",
                content="Review session for Alpha and Bravo teams",
            ),
            Entity(
                name="User_Alice-01",
                type="person",
                content="User Alice with ID 01",
            ),
        ]

        db.add_entities(entities)
        return db

    @pytest.mark.unit
    def test_search_with_hyphens_in_query(self, database_with_special_names: GraphDatabase) -> None:
        """Test search with hyphens in query string (reproduces original bug)."""
        # This query string matches the problematic pattern from the error:
        # "Conversation 14-10-2024 04:26-04:31" had multiple hyphens

        ops = LLMGraphQueryOperations(
            reasoning="Test search with hyphens",
            confidence=0.9,
            importance=0.9,
            entity_queries=[
                LLMEntityQuery(
                    reasoning="Search for project with date",
                    confidence=0.9,
                    importance=0.9,
                    entity_types=("situation",),
                    search_query="2024-10-14 meeting",
                    limit=10,
                )
            ],
        )

        # This should not raise ValueError: Syntax Error
        query = database_with_special_names.translate_to_query(ops)
        results = query.execute()

        # Should successfully execute
        assert len(results.results) >= 0

    @pytest.mark.unit
    def test_search_with_colons_in_query(self, database_with_special_names: GraphDatabase) -> None:
        """Test search with colons in query string."""
        ops = LLMGraphQueryOperations(
            reasoning="Test search with colons",
            confidence=0.9,
            importance=0.9,
            entity_queries=[
                LLMEntityQuery(
                    reasoning="Search for time-based entity",
                    confidence=0.9,
                    importance=0.9,
                    entity_types=("situation",),
                    search_query="04:26-04:31",
                    limit=10,
                )
            ],
        )

        # This should not raise ValueError: Syntax Error
        query = database_with_special_names.translate_to_query(ops)
        results = query.execute()

        assert len(results.results) >= 0

    @pytest.mark.unit
    def test_search_with_multiple_special_chars(
        self, database_with_special_names: GraphDatabase
    ) -> None:
        """Test search with multiple special characters together (original bug pattern)."""
        # This is the exact pattern that caused the error in the production code
        ops = LLMGraphQueryOperations(
            reasoning="Search for conversation with timestamp",
            confidence=0.9,
            importance=0.9,
            entity_queries=[
                LLMEntityQuery(
                    reasoning="Find conversation with full timestamp",
                    confidence=0.9,
                    importance=0.9,
                    entity_types=("situation",),
                    search_query="Session 2024-10-14 14:26-14:31",
                    limit=10,
                )
            ],
        )

        # Should not raise ValueError: Syntax Error: Session 2024-10-14 14:26-14:31
        query = database_with_special_names.translate_to_query(ops)
        results = query.execute()

        assert len(results.results) >= 0

    @pytest.mark.unit
    def test_search_with_underscores_and_hyphens(
        self, database_with_special_names: GraphDatabase
    ) -> None:
        """Test search with underscores and hyphens mixed."""
        ops = LLMGraphQueryOperations(
            reasoning="Test search with underscore-hyphen mix",
            confidence=0.9,
            importance=0.9,
            entity_queries=[
                LLMEntityQuery(
                    reasoning="Find user with special ID",
                    confidence=0.9,
                    importance=0.9,
                    entity_types=("person",),
                    search_query="User_Alice-01",
                    limit=10,
                )
            ],
        )

        query = database_with_special_names.translate_to_query(ops)
        results = query.execute()

        assert len(results.results) >= 0

    @pytest.mark.unit
    def test_direct_search_with_special_chars(
        self, database_with_special_names: GraphDatabase
    ) -> None:
        """Test direct search method with special characters."""
        # Test the search method directly with special character queries
        query_builder = database_with_special_names.query_entities()

        # This should not raise ValueError
        query = query_builder.search("2024-10-14", top_k=10)
        results = query.execute()

        assert len(results.results) >= 0

    @pytest.mark.unit
    def test_empty_search_query(self, database_with_special_names: GraphDatabase) -> None:
        """Test empty search query doesn't cause issues."""
        ops = LLMGraphQueryOperations(
            reasoning="Test empty search",
            confidence=0.9,
            importance=0.9,
            entity_queries=[
                LLMEntityQuery(
                    reasoning="Search with empty query",
                    confidence=0.9,
                    importance=0.9,
                    entity_types=("situation",),
                    search_query=None,
                    limit=10,
                )
            ],
        )

        query = database_with_special_names.translate_to_query(ops)
        results = query.execute()

        assert len(results.results) >= 0
