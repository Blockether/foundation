"""Unit tests for graph entity resolution logic."""

from __future__ import annotations

import pytest

from blockether_foundation.agents.hooks.graph import (
    GraphIngestionIterativeConfig,
    _apply_entity_resolution,
    _format_operations_for_assessment,
)
from blockether_foundation.graph.models import (
    ExtractionQualityAssessment,
    LLMEntityMerge,
    LLMEntityResolutionResult,
    LLMGraphAddEntity,
    LLMGraphAddRelationship,
    LLMGraphOperations,
)


class TestGraphIngestionIterativeConfig:
    def test_default_config_is_disabled(self):
        config = GraphIngestionIterativeConfig()
        assert config.enabled is False
        assert config.max_iterations == 2
        assert config.quality_threshold == 0.8

    def test_enabled_config(self):
        config = GraphIngestionIterativeConfig(enabled=True, max_iterations=3)
        assert config.enabled is True
        assert config.max_iterations == 3


class TestFormatOperationsForAssessment:
    def test_formats_entities_and_relationships(self):
        operations = LLMGraphOperations(
            add_entity_ops=[
                LLMGraphAddEntity(name="Alice", type="person", content="Test person"),
                LLMGraphAddEntity(name="Acme Corp", type="organization", content="Test org"),
            ],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[
                LLMGraphAddRelationship(
                    source_name="Alice", target_name="Acme Corp", type="owned_by"
                ),
            ],
            delete_relationship_ops=[],
        )

        result = _format_operations_for_assessment(operations)

        assert "<extracted_entities>" in result
        assert 'name="Alice"' in result
        assert 'type="person"' in result
        assert "Test person" in result
        assert "<extracted_relationships>" in result
        assert 'source="Alice"' in result
        assert 'target="Acme Corp"' in result

    def test_truncates_long_content(self):
        long_content = "x" * 300
        operations = LLMGraphOperations(
            add_entity_ops=[
                LLMGraphAddEntity(name="Test", type="person", content=long_content),
            ],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[],
            delete_relationship_ops=[],
        )

        result = _format_operations_for_assessment(operations)
        assert len(result) < len(long_content) + 200


class TestApplyEntityResolution:
    def test_merges_duplicate_entities(self):
        operations = LLMGraphOperations(
            add_entity_ops=[
                LLMGraphAddEntity(name="Ashley", type="person", content="Therapist"),
                LLMGraphAddEntity(name="Ashley Wojcik", type="person", content="Full name"),
            ],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[
                LLMGraphAddRelationship(
                    source_name="Ashley", target_name="Session1", type="participated_in"
                ),
            ],
            delete_relationship_ops=[],
        )

        resolution = LLMEntityResolutionResult(
            merge_operations=[
                LLMEntityMerge(
                    canonical_name="Ashley Wojcik",
                    aliases_to_merge=["Ashley"],
                    entity_type="person",
                    merged_content="Therapist. Full name",
                ),
            ],
            entities_to_rename=[],
            quality_score=0.9,
            issues_resolved=["Merged Ashley into Ashley Wojcik"],
        )

        result = _apply_entity_resolution(operations, resolution)

        entity_names = [e.name for e in result.add_entity_ops]
        assert "Ashley Wojcik" in entity_names
        assert "Ashley" not in entity_names
        assert len(result.add_entity_ops) == 1

        rel = result.add_relationship_ops[0]
        assert rel.source_name == "Ashley Wojcik"

    def test_renames_verbose_entities(self):
        operations = LLMGraphOperations(
            add_entity_ops=[
                LLMGraphAddEntity(
                    name="A conversation between Alice and Bob on Dec 28",
                    type="event",
                    content="Discussion",
                ),
            ],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[],
            delete_relationship_ops=[],
        )

        resolution = LLMEntityResolutionResult(
            merge_operations=[],
            entities_to_rename=[
                ["A conversation between Alice and Bob on Dec 28", "Dec 28 Discussion"],
            ],
            quality_score=0.9,
            issues_resolved=["Renamed verbose entity"],
        )

        result = _apply_entity_resolution(operations, resolution)

        entity_names = [e.name for e in result.add_entity_ops]
        assert "Dec 28 Discussion" in entity_names
        assert "A conversation between Alice and Bob on Dec 28" not in entity_names

    def test_updates_relationship_references_after_merge(self):
        operations = LLMGraphOperations(
            add_entity_ops=[
                LLMGraphAddEntity(name="Karol", type="person", content="Client"),
                LLMGraphAddEntity(name="Karol Wójcik", type="person", content="Full name"),
                LLMGraphAddEntity(name="Session", type="event", content="Therapy"),
            ],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[
                LLMGraphAddRelationship(
                    source_name="Karol", target_name="Session", type="participated_in"
                ),
                LLMGraphAddRelationship(
                    source_name="Karol Wójcik", target_name="Session", type="participated_in"
                ),
            ],
            delete_relationship_ops=[],
        )

        resolution = LLMEntityResolutionResult(
            merge_operations=[
                LLMEntityMerge(
                    canonical_name="Karol Wójcik",
                    aliases_to_merge=["Karol"],
                    entity_type="person",
                    merged_content="Client. Full name",
                ),
            ],
            entities_to_rename=[],
            quality_score=0.9,
            issues_resolved=[],
        )

        result = _apply_entity_resolution(operations, resolution)

        assert len(result.add_relationship_ops) == 1
        rel = result.add_relationship_ops[0]
        assert rel.source_name == "Karol Wójcik"
        assert rel.target_name == "Session"

    def test_removes_self_referential_relationships_after_merge(self):
        operations = LLMGraphOperations(
            add_entity_ops=[
                LLMGraphAddEntity(name="Alice", type="person", content="Person"),
                LLMGraphAddEntity(name="Alice Smith", type="person", content="Full name"),
            ],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[
                LLMGraphAddRelationship(
                    source_name="Alice", target_name="Alice Smith", type="related_to"
                ),
            ],
            delete_relationship_ops=[],
        )

        resolution = LLMEntityResolutionResult(
            merge_operations=[
                LLMEntityMerge(
                    canonical_name="Alice Smith",
                    aliases_to_merge=["Alice"],
                    entity_type="person",
                    merged_content="Person. Full name",
                ),
            ],
            entities_to_rename=[],
            quality_score=0.9,
            issues_resolved=[],
        )

        result = _apply_entity_resolution(operations, resolution)

        assert len(result.add_relationship_ops) == 0

    def test_no_resolution_needed_returns_original(self):
        operations = LLMGraphOperations(
            add_entity_ops=[
                LLMGraphAddEntity(name="Alice", type="person", content="Test"),
            ],
            update_entity_ops=[],
            delete_entity_ops=[],
            add_relationship_ops=[],
            delete_relationship_ops=[],
        )

        resolution = LLMEntityResolutionResult(
            merge_operations=[],
            entities_to_rename=[],
            quality_score=1.0,
            issues_resolved=[],
        )

        result = _apply_entity_resolution(operations, resolution)

        assert len(result.add_entity_ops) == 1
        assert result.add_entity_ops[0].name == "Alice"

    def test_preserves_update_and_delete_operations(self):
        from blockether_foundation.graph.models import LLMGraphDeleteEntity, LLMGraphUpdateEntity

        operations = LLMGraphOperations(
            add_entity_ops=[],
            update_entity_ops=[
                LLMGraphUpdateEntity(
                    id="123", name="Updated", type="person", content="New content"
                ),
            ],
            delete_entity_ops=[
                LLMGraphDeleteEntity(entity_id="456"),
            ],
            add_relationship_ops=[],
            delete_relationship_ops=[],
        )

        resolution = LLMEntityResolutionResult(
            merge_operations=[],
            entities_to_rename=[],
            quality_score=1.0,
            issues_resolved=[],
        )

        result = _apply_entity_resolution(operations, resolution)

        assert len(result.update_entity_ops) == 1
        assert len(result.delete_entity_ops) == 1


class TestExtractionQualityAssessmentModel:
    def test_creates_valid_assessment(self):
        assessment = ExtractionQualityAssessment(
            has_duplicate_entities=True,
            has_verbose_entity_names=False,
            has_disconnected_components=False,
            quality_score=0.6,
            needs_resolution=True,
            specific_issues=["Found duplicate: Ashley vs Ashley Wojcik"],
            duplicate_candidates=[["Ashley", "Ashley Wojcik"]],
        )

        assert assessment.has_duplicate_entities is True
        assert assessment.quality_score == 0.6
        assert len(assessment.duplicate_candidates) == 1

    def test_quality_score_bounds(self):
        with pytest.raises(ValueError):
            ExtractionQualityAssessment(
                has_duplicate_entities=False,
                has_verbose_entity_names=False,
                has_disconnected_components=False,
                quality_score=1.5,
                needs_resolution=False,
            )


class TestLLMEntityMergeModel:
    def test_creates_valid_merge(self):
        merge = LLMEntityMerge(
            canonical_name="Karol Wójcik",
            aliases_to_merge=["Karol", "K. Wójcik"],
            entity_type="person",
            merged_content="Combined content from all aliases",
        )

        assert merge.canonical_name == "Karol Wójcik"
        assert len(merge.aliases_to_merge) == 2
        assert merge.entity_type == "person"


class TestLLMEntityResolutionResultModel:
    def test_creates_valid_resolution(self):
        resolution = LLMEntityResolutionResult(
            merge_operations=[
                LLMEntityMerge(
                    canonical_name="Test",
                    aliases_to_merge=["T"],
                    entity_type="person",
                    merged_content="Content",
                ),
            ],
            entities_to_rename=[["OldName", "NewName"]],
            quality_score=0.95,
            issues_resolved=["Fixed duplicates"],
        )

        assert len(resolution.merge_operations) == 1
        assert len(resolution.entities_to_rename) == 1
        assert resolution.quality_score == 0.95
