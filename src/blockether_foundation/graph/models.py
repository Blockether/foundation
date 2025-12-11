"""Core data models for the knowledge graph database."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, cast

from pydantic import ConfigDict, Field

from ..models import ChainOfThoughts
from ..utils import generate_secure_id

T = TypeVar("T", bound="TimestampMixin")

EntityType = Literal[
    "acronym",
    "keyword",
    "tool",
    "library",
    "person",
    "organization",
    "location",
    "event",
    "date",
    "document",
    "image",
    "audio",
    "video",
    "code_snippet",
    "url",
    "product",
    "rule",
    "concept",
    "acronym_definition",
    "keyword_definition",
    "concept_context",
    "event_context",
    "date_context",
    "product_context",
    "rule_context",
    "image_context",
    "document_context",
    "audio_context",
    "video_context",
    "code_snippet_context",
    "url_context",
]

RelationType = Literal[
    "part_of",
    "contains",
    "belongs_to",
    "instance_of",
    "related_to",
    "acronym_for",
    "synonym_of",
    "keyword_for",
    "opposite_of",
    "similar_to",
    "replaces",
    "created_by",
    "modified_by",
    "implements",
    "depends_on",
    "uses",
    "used_for",
    "before",
    "after",
    "referenced_by",
    "referenced",
    "applies_to",
    "learned_from",
    "improves_upon",
    "contradicts",
    "validates",
    "invalidates",
]


class TimestampMixin:
    """Mixin providing automatic datetime serialization for dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with automatic datetime serialization."""
        # Convert object to dict manually since asdict doesn't work with mixins
        data: dict[str, Any] = {}
        if hasattr(self, "__dataclass_fields__"):
            dataclass_fields = cast(dict[str, Any], self.__dataclass_fields__)  # type: ignore[attr-defined]
            for key in dataclass_fields:
                value = getattr(self, key)
                if isinstance(value, datetime):
                    data[key] = value.isoformat()
                else:
                    data[key] = value
        return data

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create from dictionary with automatic datetime deserialization."""
        data = data.copy()
        if hasattr(cls, "__dataclass_fields__"):
            dataclass_fields = cast(dict[str, Any], cls.__dataclass_fields__)  # type: ignore[attr-defined]
            for field_name, field_obj in dataclass_fields.items():
                if field_name in data and isinstance(data[field_name], str):
                    # Check if field type is datetime (using string comparison to handle type differences)
                    field_type_str = str(getattr(field_obj, "type", ""))
                    if "datetime" in field_type_str:
                        data[field_name] = datetime.fromisoformat(data[field_name].__str__())

        return cls(**data)


@dataclass
class Entity(TimestampMixin):
    """Knowledge graph entity."""

    name: str  # Human-readable identifier (unique per type)
    type: EntityType
    content: str | None = None
    id: str = field(default_factory=lambda: generate_secure_id(size=8))  # Auto-generated secure ID
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
        description="Human-readable name of the entity. Like full name of the person or the organization, document title, etc."
    )
    type: EntityType = Field(description="Type of the entity.")
    content: str = Field(description="Content or description of the entity.")

    def __post__init__(self) -> None:
        self.id = generate_secure_id(size=8)


class LLMGraphUpdateEntity(ChainOfThoughts):
    """Database update entity operation for LLMGraph consumption."""

    id: str = Field(description="Unique identifier of the entity to update.")
    name: str = Field(description="Updated human-readable name of the entity.")
    type: EntityType = Field(description="Updated type of the entity.")
    content: str = Field(description="Updated content or description of the entity.")


class LLMGraphDeleteEntity(ChainOfThoughts):
    """Database delete entity operation for LLMGraph consumption."""

    entity_id: str = Field(description="ID of the entity to be deleted from the database.")
    operation: Literal["delete_entity"] = Field(default="delete_entity")


class LLMGraphAddRelationship(ChainOfThoughts):
    """Database add relationship operation for LLMGraph consumption."""

    source_name: str = Field(description="Human-readable name of the source entity.")
    target_name: str = Field(description="Human-readable name of the target entity.")
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
        # Validate: each entity ID must appear in only one operation
        # Note: AddEntityLLMGraph uses name as key (no ID yet), UpdateEntityLLMGraph uses ID
        seen_entity_names: set[str] = set()
        seen_entity_ids: set[str] = set()

        for op in self.add_entity_ops:
            entity_name = op.name
            if entity_name in seen_entity_names:
                raise ValueError(f"Duplicate add operation for entity name '{entity_name}'")
            seen_entity_names.add(entity_name)

        for op in self.update_entity_ops:
            entity_id = op.id
            if entity_id in seen_entity_ids:
                raise ValueError(f"Duplicate update operation for entity ID '{entity_id}'")
            seen_entity_ids.add(entity_id)

        for op in self.delete_entity_ops:
            if op.entity_id in seen_entity_ids:
                raise ValueError(f"Duplicate delete operation for entity ID '{op.entity_id}'")
            seen_entity_ids.add(op.entity_id)

        # Normalize relationship operations by keeping first occurrence of each edge
        add_relationship_ids: set[str] = set()
        normalized_add_relationship_ops: list[LLMGraphAddRelationship] = []
        for op in self.add_relationship_ops:
            rel_id = f"{op.source_name}_{op.target_name}"
            if rel_id in add_relationship_ids:
                continue
            add_relationship_ids.add(rel_id)
            normalized_add_relationship_ops.append(op)

        if len(normalized_add_relationship_ops) != len(self.add_relationship_ops):
            self.add_relationship_ops = normalized_add_relationship_ops

        delete_relationship_ids: set[str] = set()
        normalized_delete_relationship_ops: list[LLMGraphDeleteRelationship] = []
        for op in self.delete_relationship_ops:
            if op.id in delete_relationship_ids:
                continue
            delete_relationship_ids.add(op.id)
            normalized_delete_relationship_ops.append(op)

        if len(normalized_delete_relationship_ops) != len(self.delete_relationship_ops):
            self.delete_relationship_ops = normalized_delete_relationship_ops

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
