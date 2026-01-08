"""Core data models for the knowledge graph database."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, cast

from pydantic import ConfigDict, Field

from ..models import ChainOfThoughts
from ..utils import generate_secure_id

T = TypeVar("T", bound="TimestampMixin")

EntityType = Literal[
    "creature",
    "person",
    "organization",
    "location",
    "event",
    "date",
    "document",
    "attachment",
    "object",
    "concept",
    "example",
    "rule",
    "pattern",
    "mode",
    "schema",
    "abbreviation",
    "reference",
    "memory",
    "situation",
    "fact",
]

RelationType = Literal[
    "part_of",
    "owned_by",
    "occurs_at",
    "related_to",
    "triggers",
    "originates_from",
    "expands_to",
    "explains",
    "is_a_fact",
    "participated_in",
    "alias_of",
    "created_by",
    "located_at",
]


class TimestampMixin:
    """Mixin providing automatic datetime serialization for dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with automatic datetime serialization."""
        # Convert object to dict manually since asdict doesn't work with mixins
        data: dict[str, Any] = {}
        if is_dataclass(self):
            dataclass_fields = cast(dict[str, Any], self.__dataclass_fields__)
            for key in dataclass_fields:
                value = self.__dict__.get(key)
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
                else:
                    data[key] = value
        return data

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create from dictionary with automatic datetime deserialization."""
        data = data.copy()
        if is_dataclass(cls):
            dataclass_fields = cast(dict[str, Any], cls.__dataclass_fields__)
            for field_name, field_obj in dataclass_fields.items():
                if field_name in data and isinstance(data[field_name], str):
                    # Check if field type is datetime (using string comparison to handle type differences)
                    field_type_str = str(field_obj.type)
                    if "datetime" in field_type_str:
                        data[field_name] = datetime.fromisoformat(data[field_name].__str__())

        return cls(**data)


@dataclass
class Entity(TimestampMixin):
    """Knowledge graph entity."""

    name: str  # Human-readable identifier (unique per type)
    type: EntityType
    id: str | None = None
    content: str | None = None
    provenance: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Relationship(TimestampMixin):
    """Knowledge graph relationship between entities."""

    source: str  # Entity ID (auto-generated nano ID)
    target: str  # Entity ID (auto-generated nano ID)
    type: RelationType
    id: str = ""  # Auto-generated from source_target if empty
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Auto-generate ID from source and target entity IDs if not already set."""
        if not self.id:
            self.id = f"{self.source}_{self.target}"


class LLMGraphAddEntity(ChainOfThoughts):
    """Database add entity operation for LLMGraph consumption."""

    name: str = Field(
        description="Human-readable name of the entity. Plain text only, no emojis or special characters. Concise by default, max 60 characters."
    )
    type: EntityType = Field(description="Type of the entity.")
    content: str = Field(
        description="Content or description of the entity. No markdown, plain text only. Concise by default, max 400 characters."
    )
    provenance: str | None = Field(
        default=None, description="Source or provenance of the entity information."
    )
    importance: float | None = Field(
        default=None,
        description="Importance score of the entity (0.0 to 1.0). Optional field for prioritization.",
    )

    def __post__init__(self) -> None:
        self.id = generate_secure_id(size=5)


class LLMGraphUpdateEntity(ChainOfThoughts):
    """Database update entity operation for LLMGraph consumption."""

    id: str = Field(description="Unique identifier of the entity to update.")
    name: str = Field(description="Updated human-readable name of the entity.")
    provenance: str | None = Field(
        default=None, description="Source or provenance of the entity information."
    )
    type: EntityType = Field(description="Updated type of the entity.")
    content: str = Field(description="Updated content or description of the entity.")


class LLMGraphDeleteEntity(ChainOfThoughts):
    """Database delete entity operation for LLMGraph consumption."""

    entity_id: str = Field(description="ID of the entity to be deleted from the database.")
    operation: Literal["delete_entity"] = Field(default="delete_entity")


class LLMGraphAddRelationship(ChainOfThoughts):
    """Database add relationship operation for LLMGraph consumption."""

    source_name: str = Field(
        description="Human-readable name of the source entity. SHOULD ALWAYS MATCH AN EXISTING ENTITY or REFERENCE AN ENTITY TO BE CREATED."
    )
    target_name: str = Field(
        description="Human-readable name of the target entity. SHOULD ALWAYS MATCH AN EXISTING ENTITY or REFERENCE AN ENTITY TO BE CREATED."
    )
    type: RelationType = Field(description="Type of the relationship.")

    def __post__init__(self) -> None:
        self.id = f"{self.source_name}_{self.target_name}"


class LLMGraphDeleteRelationship(ChainOfThoughts):
    """Database delete relationship operation for LLMGraph consumption."""

    id: str = Field(description="ID of the relationship to be deleted from the database.")


class LLMGraphOperations(ChainOfThoughts):
    """Database operation representation for LLMGraph consumption."""

    add_entity_ops: list[LLMGraphAddEntity] = Field(description="List of add entity operations.")
    update_entity_ops: list[LLMGraphUpdateEntity] = Field(
        description="List of update entity operations."
    )
    delete_entity_ops: list[LLMGraphDeleteEntity] = Field(
        description="List of delete entity operations."
    )
    add_relationship_ops: list[LLMGraphAddRelationship] = Field(
        description="List of add relationship operations."
    )
    delete_relationship_ops: list[LLMGraphDeleteRelationship] = Field(
        description="List of delete relationship operations."
    )

    @property
    def ops(
        self,
    ) -> list[
        LLMGraphAddEntity
        | LLMGraphUpdateEntity
        | LLMGraphDeleteEntity
        | LLMGraphAddRelationship
        | LLMGraphDeleteRelationship
    ]:
        """Get all operations in a flat list.

        Validates that each entity/relationship appears in at most one operation.
        """

        return [
            *self.add_entity_ops,
            *self.update_entity_ops,
            *self.delete_entity_ops,
            *self.add_relationship_ops,
            *self.delete_relationship_ops,
        ]


class LLMBaseQuery(ChainOfThoughts):
    """Base query options for LLMGraph consumption."""

    # NOTE: frozen=True makes models immutable and hashable
    # This enables efficient O(1) deduplication using set() instead of O(n²) JSON serialization
    model_config = ConfigDict(frozen=True)

    limit: int = Field(default=10, description="Maximum number of results to return.")
    order_ascending: bool = Field(default=False, description="Sort order (true=asc, false=desc).")
    order_by: Literal["created_at", "updated_at"] = Field(
        default="created_at", description="Field to sort by."
    )


class LLMEntityQuery(LLMBaseQuery):
    """Entity query operation for LLMGraph consumption.

    Represents a query for entities with specific filters.
    """

    # Entity Filters
    entity_types: tuple[EntityType, ...] | None = Field(
        default=None, description="Tuple of entity types to filter by (OR logic)."
    )
    search_query: str | None = Field(
        default=None, description="Full-text search query for content/name."
    )
    created_after: str | None = Field(
        default=None, description="Filter entities created after date (YYYY-MM-DD)."
    )
    created_before: str | None = Field(
        default=None, description="Filter entities created before date (YYYY-MM-DD)."
    )

    # Expansion (Entity only)
    expand_depth: int | None = Field(
        default=None,
        description="BFS expansion depth (1=neighbors, 2=neighbors of neighbors).",
    )
    expand_direction: Literal["outgoing", "incoming", "both"] = Field(
        default="both", description="Direction for expansion traversal."
    )
    expand_relationship_type: RelationType | None = Field(
        default=None, description="Filter expansion by relationship type."
    )


class LLMRelationshipQuery(LLMBaseQuery):
    """Relationship query operation for LLMGraph consumption.

    Represents a query for relationships with specific filters.
    """

    relationship_types: tuple[RelationType, ...] | None = Field(
        default=None, description="Tuple of relationship types to filter by (OR logic)."
    )
    source_entity_name: str | None = Field(
        default=None, description="Filter relationships from source entity with this name."
    )
    target_entity_name: str | None = Field(
        default=None, description="Filter relationships to target entity with this name."
    )


class LLMGraphQueryOperations(ChainOfThoughts):
    """Collection of query operations for LLMGraph consumption."""

    entity_queries: list[LLMEntityQuery] = Field(
        default_factory=list[LLMEntityQuery],
        description="List of entity query operations to perform.",
    )
    relationship_queries: list[LLMRelationshipQuery] = Field(
        default_factory=list[LLMRelationshipQuery],
        description="List of relationship query operations to perform.",
    )

    @property
    def ops(self) -> list[LLMEntityQuery | LLMRelationshipQuery]:
        """Return deduplicated query operations in evaluation order."""

        # OPTIMIZATION: Using set-based deduplication (O(1)) instead of JSON serialization (O(n²))
        # This change improves performance by ~3.36x and reduces memory usage significantly
        # Models were made hashable by:
        # 1. Adding `model_config = ConfigDict(frozen=True)` to LLMBaseQuery
        # 2. Converting list fields (entity_types, relationship_types) to tuples
        if len(self.entity_queries) != len(set(self.entity_queries)):
            self.entity_queries = list(set(self.entity_queries))

        if len(self.relationship_queries) != len(set(self.relationship_queries)):
            self.relationship_queries = list(set(self.relationship_queries))

        return [
            *self.entity_queries,
            *self.relationship_queries,
        ]


class QualityAssessment(ChainOfThoughts):
    """LLM assessment of retrieved context quality for iterative improvement.

    Used by the graph pre-hook to evaluate whether retrieved context
    is sufficient to answer the user's request, enabling iterative
    query refinement.
    """

    satisfied: bool = Field(
        description="Is the retrieved context sufficient to answer the user's request?"
    )
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality score from 0.0 (poor) to 1.0 (excellent) based on relevance and completeness",
    )
    coverage_gaps: list[str] = Field(
        default_factory=list,
        description="List of missing information or topics not covered by current context",
    )
    interesting_entities: list[str] = Field(
        default_factory=list,
        description="Entity names (from current results) worth expanding via BFS to gather more context",
    )
    suggested_refinements: list[str] = Field(
        default_factory=list,
        description="Suggestions for improving existing queries (broader search, different types, etc.)",
    )


class ExtractionQualityAssessment(ChainOfThoughts):
    """Assessment of extracted graph operations quality.

    Used by the graph post-hook to evaluate whether extracted entities
    and relationships are well-formed before importing to the graph.
    """

    has_duplicate_entities: bool = Field(
        description="True if duplicate or near-duplicate entities detected (e.g., 'Ashley' and 'Ashley Wojcik')"
    )
    has_verbose_entity_names: bool = Field(
        description="True if entity names are overly verbose (descriptions used as names instead of identifiers)"
    )
    has_disconnected_components: bool = Field(
        description="True if entities exist without proper relationships connecting them"
    )
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall extraction quality score (0.0=poor, 1.0=excellent)",
    )
    needs_resolution: bool = Field(
        description="True if entity resolution pass is recommended to fix issues"
    )
    specific_issues: list[str] = Field(
        default_factory=list,
        description="Specific issues identified in the extraction that need resolution",
    )
    duplicate_candidates: list[list[str]] = Field(
        default_factory=list[list[str]],
        description="Pairs of entity names as [old, new] that appear to be duplicates",
    )


class LLMEntityMerge(ChainOfThoughts):
    """Represents a single entity merge operation.

    When duplicate entities are detected, this describes how to merge them
    into a single canonical entity.
    """

    canonical_name: str = Field(
        description="The correct, canonical name to use for this entity (most complete/specific form)"
    )
    aliases_to_merge: list[str] = Field(
        description="List of entity names that should be merged into the canonical entity"
    )
    entity_type: EntityType = Field(description="The entity type for the merged entity")
    merged_content: str = Field(
        description="Combined content from all merged entities, preserving important information"
    )


class LLMEntityResolutionResult(ChainOfThoughts):
    """Result of entity resolution analysis.

    Contains the operations needed to deduplicate and normalize entities
    before importing to the graph database.
    """

    merge_operations: list[LLMEntityMerge] = Field(
        default_factory=list[LLMEntityMerge],
        description="List of entity merge operations to perform (for duplicates)",
    )
    entities_to_rename: list[list[str]] = Field(
        default_factory=list[list[str]],
        description="List of [old_name, new_name] pairs for renaming verbose entity names",
    )
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Quality score after resolution (0.0=still issues, 1.0=clean)",
    )
    issues_resolved: list[str] = Field(
        default_factory=list,
        description="List of issues that were resolved by this operation",
    )


@dataclass
class GraphHookIterativeConfig:
    """Configuration for graph pre-hook query generation.

    Controls the single-pass query generation with optional retry for empty results.
    """

    max_queries: int = 3  # Maximum number of queries to generate
    max_query_retries: int = 2  # Retry queries with agent feedback if they return empty


@dataclass
class GraphIngestionIterativeConfig:
    """Configuration for iterative entity resolution in graph post-hook.

    Enables multi-pass extraction with entity resolution to improve graph quality
    by deduplicating entities and normalizing names before import.
    """

    max_iterations: int = 2  # 1 extraction + up to 1 resolution pass
    quality_threshold: float = 0.8  # 0.0-1.0 scale for early termination
    enable_entity_resolution: bool = True  # Run entity deduplication agent
    enable_name_normalization: bool = True  # Fix verbose entity names
    enable_quality_assessment: bool = True  # Run quality check before resolution
    max_import_retries: int = 3  # Max retries for validation/import with agent fixes
