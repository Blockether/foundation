"""Simple, focused tests for optimized XML formatting functionality."""

import xml.etree.ElementTree as ET
from unittest.mock import Mock

from src.blockether_foundation.graph.formatting import (
    format_existing_entities_for_context,
    format_graph_query_results,
)
from src.blockether_foundation.graph.models import Entity, Relationship


class TestOptimizedFormatting:
    """Simple tests for NEW optimized XML formatting functionality."""

    def test_entity_formatting_uses_attributes(self):
        """Test that entity formatting uses attributes instead of elements."""
        entity = Entity(id="test123", name="Test Entity", type="concept", content="Test content")

        mock_query = Mock()
        mock_result = Mock()
        mock_result.results = [entity]
        mock_query.execute = Mock(return_value=mock_result)

        result = format_graph_query_results([mock_query])

        # Parse XML successfully
        root = ET.fromstring(result)
        entity_elem = root.find(".//entity")

        # Test NEW functionality: attributes instead of elements
        assert entity_elem.get("id") == "test123"
        assert entity_elem.get("name") == "Test Entity"
        assert entity_elem.get("type") == "concept"

        # Should NOT have verbose element names (NEW optimization)
        assert entity_elem.find("entity_id") is None
        assert entity_elem.find("entity_name") is None
        assert entity_elem.find("entity_type") is None

        # Content should be in element
        content_elem = entity_elem.find("content")
        assert content_elem.text == "Test content"

    def test_shortened_element_names_in_output(self):
        """Test that shortened element names are used in output."""
        entity = Entity(id="e1", name="Entity One", type="concept", content="Content")

        mock_query = Mock()
        mock_result = Mock()
        mock_result.results = [entity]
        mock_query.execute = Mock(return_value=mock_result)

        result = format_graph_query_results([mock_query])

        # Test NEW shortened names exist in raw output
        assert "<content>" in result
        assert 'index="1"' in result  # Shortened attribute name

        # Should NOT have verbose names
        assert "<entity_content>" not in result
        assert "query_index=" not in result

    def test_empty_content_no_element(self):
        """Test that empty content doesn't create unnecessary element."""
        empty_entity = Entity(id="empty", name="Empty Entity", type="concept", content=None)

        mock_query = Mock()
        mock_result = Mock()
        mock_result.results = [empty_entity]
        mock_query.execute = Mock(return_value=mock_result)

        result = format_graph_query_results([mock_query])

        root = ET.fromstring(result)
        entity_elem = root.find(".//entity")

        # Empty content should not create element
        content_elem = entity_elem.find("content")
        assert content_elem is None

    def test_content_truncation_works(self):
        """Test content truncation works correctly."""
        long_content = "A" * 100  # 100 characters
        long_entity = Entity(id="long", name="Long Entity", type="document", content=long_content)

        mock_query = Mock()
        mock_result = Mock()
        mock_result.results = [long_entity]
        mock_query.execute = Mock(return_value=mock_result)

        # Test with truncation
        result = format_graph_query_results([mock_query], max_content_length=50)

        root = ET.fromstring(result)
        content_elem = root.find(".//content")

        # Should be truncated with "..."
        assert len(content_elem.text) <= 53  # 50 + "..."
        assert content_elem.text.endswith("...")

        # Test without truncation
        result_no_trunc = format_graph_query_results([mock_query], max_content_length=None)
        root_no_trunc = ET.fromstring(result_no_trunc)
        content_no_trunc = root_no_trunc.find(".//content")

        # Should not be truncated
        assert len(content_no_trunc.text) == 100
        assert not content_no_trunc.text.endswith("...")

    def test_special_characters_handled_correctly(self):
        """Test that special characters are handled correctly in XML."""
        special_entity = Entity(
            id="special",
            name="Entity <with> & 'quotes'",
            type="concept",
            content="Content with <tags>"
        )

        mock_query = Mock()
        mock_result = Mock()
        mock_result.results = [special_entity]
        mock_query.execute = Mock(return_value=mock_result)

        result = format_graph_query_results([mock_query])

        # Should parse without errors (proper escaping)
        root = ET.fromstring(result)
        entity_elem = root.find(".//entity")
        content_elem = entity_elem.find("content")

        # Check that parsing succeeded and content is correct
        assert entity_elem.get("name") == "Entity <with> & 'quotes'"
        assert content_elem.text == "Content with <tags>"

    def test_context_formatting_uses_optimized_structure(self):
        """Test context formatting uses optimized structure."""
        entity = Entity(id="ctx1", name="Context Entity", type="concept", content="Context content")

        mock_graph = Mock()
        mock_graph.get_most_connected_entities.return_value = [(entity, 5)]
        mock_graph.get_entity_relationships.return_value = []

        result = format_existing_entities_for_context(mock_graph)

        # Parse XML successfully
        root = ET.fromstring(result)
        entity_elem = root.find("entity")

        # Should use optimized attributes
        assert entity_elem.get("id") == "ctx1"
        assert entity_elem.get("name") == "Context Entity"
        assert entity_elem.get("type") == "concept"
        assert entity_elem.get("connections") == "5"

        # Should use shortened container names
        rels_elem = entity_elem.find("rels")
        assert rels_elem is not None

        # Should NOT have verbose names
        assert entity_elem.find("entity_id") is None
        assert entity_elem.find("entity_relationships") is None

    def test_relationship_optimization_works(self):
        """Test relationship optimization works correctly."""
        entity1 = Entity(id="e1", name="Entity 1", type="concept")
        entity2 = Entity(id="e2", name="Entity 2", type="organization")
        relationship = Relationship(source="e1", target="e2", type="related_to")

        mock_index = Mock()
        mock_index.entity_by_id = {"e2": entity2}

        mock_graph = Mock()
        mock_graph.get_most_connected_entities.return_value = [(entity1, 1)]
        mock_graph.get_entity_relationships.return_value = [relationship]
        mock_graph.index = mock_index

        result = format_existing_entities_for_context(mock_graph)

        # Should contain optimized relationship format
        assert '<rel type="related_to" dir="out"' in result
        assert 'to_id="e2"' in result
        assert 'to_name="Entity 2"' in result

    def test_empty_queries_list(self):
        """Test empty queries list."""
        result = format_graph_query_results([])
        assert result == "<!-- Iteration 1 -->\n<graph_knowledge>\n</graph_knowledge>"

    def test_empty_results_handling(self):
        """Test queries with no results."""
        mock_query = Mock()
        mock_result = Mock()
        mock_result.results = []
        mock_query.execute = Mock(return_value=mock_result)

        result = format_graph_query_results([mock_query])

        root = ET.fromstring(result)
        entities = root.findall(".//entity")
        assert len(entities) == 0

    def test_unicode_support(self):
        """Test Unicode characters work correctly."""
        unicode_entity = Entity(
            id="unicode",
            name="Unicode Tést ñáéíóú 🎉",
            type="concept",
            content="Unicode content: 中文"
        )

        mock_query = Mock()
        mock_result = Mock()
        mock_result.results = [unicode_entity]
        mock_query.execute = Mock(return_value=mock_result)

        result = format_graph_query_results([mock_query])

        # Should parse without errors
        root = ET.fromstring(result)
        entity_elem = root.find(".//entity")
        content_elem = entity_elem.find("content")

        # Should preserve Unicode
        assert "Tést" in entity_elem.get("name")
        assert "ñáéíóú" in entity_elem.get("name")
        assert "🎉" in entity_elem.get("name")
        assert "中文" in content_elem.text

    def test_empty_graph_context(self):
        """Test empty graph context."""
        mock_graph = Mock()
        mock_graph.get_most_connected_entities.return_value = []

        result = format_existing_entities_for_context(mock_graph)
        assert result == "<existing_entities>\n</existing_entities>"

    def test_multiple_queries_indices(self):
        """Test multiple queries get correct indices."""
        entities = [
            Entity(id="e1", name="Entity 1", type="concept", content="Content 1"),
            Entity(id="e2", name="Entity 2", type="organization", content="Content 2"),
        ]

        queries = []
        for _i, entity in enumerate(entities):
            mock_query = Mock()
            mock_result = Mock()
            mock_result.results = [entity]
            mock_query.execute = Mock(return_value=mock_result)
            queries.append(mock_query)

        result = format_graph_query_results(queries)

        root = ET.fromstring(result)
        query_blocks = root.findall("query_with_results")
        assert len(query_blocks) == 2

        # Should use optimized "index" attribute
        assert query_blocks[0].get("index") == "1"
        assert query_blocks[1].get("index") == "2"

    def test_token_efficiency_verification(self):
        """Test that optimized format is token efficient."""
        entities = [
            Entity(id=f"e{i}", name=f"Entity {i}", type="concept", content=f"Content {i}")
            for i in range(5)
        ]

        mock_query = Mock()
        mock_result = Mock()
        mock_result.results = entities
        mock_query.execute = Mock(return_value=mock_result)

        result = format_graph_query_results([mock_query])

        # Count elements (proxy for token usage)
        root = ET.fromstring(result)
        elements = list(root.iter())
        element_count = len(elements)

        # Should have minimal elements due to optimization
        assert element_count <= 20

        # Verify no verbose elements
        xml_text = result
        assert "<entity_id>" not in xml_text
        assert "<entity_name>" not in xml_text
        assert "<entity_type>" not in xml_text
        assert "<entity_content>" not in xml_text

    def test_max_results_per_query(self):
        """Test max_results_per_query functionality."""
        entities = [
            Entity(id=f"e{i}", name=f"Entity {i}", type="concept", content=f"Content {i}")
            for i in range(10)
        ]

        mock_query = Mock()
        mock_result = Mock()
        mock_result.results = entities
        mock_query.execute = Mock(return_value=mock_result)

        # Test with limit
        result = format_graph_query_results([mock_query], max_results_per_query=3)

        root = ET.fromstring(result)
        entity_elems = root.findall(".//entity")
        assert len(entity_elems) == 3

        # Test with zero limit
        result_zero = format_graph_query_results([mock_query], max_results_per_query=0)
        root_zero = ET.fromstring(result_zero)
        entity_elems_zero = root_zero.findall(".//entity")
        assert len(entity_elems_zero) == 0

    def test_xml_always_well_formed(self):
        """Test that generated XML is always well-formed."""
        test_cases = [
            Entity(id="simple", name="Simple", type="concept", content="Simple content"),
            Entity(id="empty", name="Empty", type="concept", content=None),
            Entity(id="unicode", name="Unicode Tést", type="concept", content="Tëst content"),
            Entity(id="special", name="Special <chars>", type="concept", content="Content & more"),
        ]

        for test_entity in test_cases:
            mock_query = Mock()
            mock_result = Mock()
            mock_result.results = [test_entity]
            mock_query.execute = Mock(return_value=mock_result)

            result = format_graph_query_results([mock_query])

            # Should always parse without errors
            root = ET.fromstring(result)
            assert root.tag == "graph_knowledge"

            entity = root.find(".//entity")
            assert entity is not None
            assert entity.get("id") is not None
            assert entity.get("name") is not None
            assert entity.get("type") is not None

    def test_relationship_direction_shortening(self):
        """Test that relationship direction uses shortened format."""
        source_entity = Entity(id="source", name="Source", type="organization")
        target_entity = Entity(id="target", name="Target", type="location")
        relationship = Relationship(source="source", target="target", type="related_to")

        mock_index = Mock()
        mock_index.entity_by_id = {"target": target_entity}

        mock_graph = Mock()
        mock_graph.get_most_connected_entities.return_value = [(source_entity, 1)]
        mock_graph.get_entity_relationships.return_value = [relationship]
        mock_graph.index = mock_index

        result = format_existing_entities_for_context(mock_graph)

        # Should use shortened direction
        root = ET.fromstring(result)
        rel_elem = root.find(".//rel")
        assert rel_elem.get("dir") == "out"  # Source is main entity, so direction is "out"
