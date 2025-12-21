"""Multi-index system for fast graph queries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, cast

from blockether_foundation.graph.models import Entity, EntityType, Relationship, RelationType


@dataclass
class GraphIndex:
    """Multi-level index for efficient graph queries."""

    entity_by_id: dict[str, Entity] = field(default_factory=lambda: {})
    relationship_by_id: dict[str, Relationship] = field(default_factory=lambda: {})
    entity_by_name: dict[str, str] = field(default_factory=lambda: {})  # name -> entity_id
    entities_by_type: dict[EntityType, set[str]] = field(default_factory=lambda: defaultdict(set))
    relationships_by_type: dict[RelationType, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    entities_by_created_date: dict[date, set[str]] = field(default_factory=lambda: defaultdict(set))
    entities_by_updated_date: dict[date, set[str]] = field(default_factory=lambda: defaultdict(set))
    outgoing_edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    incoming_edges: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    entities_by_id_prefix: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    entities_by_name_prefix: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_entity(self, entity: Entity) -> None:
        """Add entity to all relevant indices."""
        self.entity_by_id[entity.id] = entity
        self.entity_by_name[entity.name] = entity.id
        self.entities_by_type[entity.type].add(entity.id)
        self.entities_by_created_date[entity.created_at.date()].add(entity.id)
        self.entities_by_updated_date[entity.updated_at.date()].add(entity.id)

        # Index ID prefixes for fast substring searches
        entity_id_lower = entity.id.lower()
        for i in range(1, len(entity_id_lower) + 1):
            prefix = entity_id_lower[:i]
            self.entities_by_id_prefix[prefix].add(entity.id)

        # Index name prefixes for fast substring searches
        entity_name_lower = entity.name.lower()
        for i in range(1, len(entity_name_lower) + 1):
            prefix = entity_name_lower[:i]
            self.entities_by_name_prefix[prefix].add(entity.id)

    def update_entity(self, old_entity: Entity, new_entity: Entity) -> None:
        """Update entity in all relevant indices."""
        self.remove_entity(old_entity)
        self.add_entity(new_entity)

    def remove_entity(self, entity: Entity) -> None:
        """Remove entity from all relevant indices."""
        self.entity_by_id.pop(entity.id, None)
        self.entity_by_name.pop(entity.name, None)
        self.entities_by_type[entity.type].discard(entity.id)
        self.entities_by_created_date[entity.created_at.date()].discard(entity.id)
        self.entities_by_updated_date[entity.updated_at.date()].discard(entity.id)

        # Remove from ID prefix index
        entity_id_lower = entity.id.lower()
        for i in range(1, len(entity_id_lower) + 1):
            prefix = entity_id_lower[:i]
            self.entities_by_id_prefix[prefix].discard(entity.id)

        # Remove from name prefix index
        entity_name_lower = entity.name.lower()
        for i in range(1, len(entity_name_lower) + 1):
            prefix = entity_name_lower[:i]
            self.entities_by_name_prefix[prefix].discard(entity.id)

    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to all relevant indices."""
        self.relationship_by_id[relationship.id] = relationship
        self.relationships_by_type[relationship.type].add(relationship.id)
        self.outgoing_edges[relationship.source].add(relationship.id)
        self.incoming_edges[relationship.target].add(relationship.id)

    def update_relationship(
        self, old_relationship: Relationship, new_relationship: Relationship
    ) -> None:
        """Update relationship in all relevant indices."""
        self.remove_relationship(old_relationship)
        self.add_relationship(new_relationship)

    def remove_relationship(self, relationship: Relationship) -> None:
        """Remove relationship from all relevant indices."""
        self.relationship_by_id.pop(relationship.id, None)
        self.relationships_by_type[relationship.type].discard(relationship.id)
        self.outgoing_edges[relationship.source].discard(relationship.id)
        self.incoming_edges[relationship.target].discard(relationship.id)

    def clear(self) -> None:
        """Clear all indices."""
        self.entity_by_id.clear()
        self.relationship_by_id.clear()
        self.entity_by_name.clear()
        self.entities_by_type.clear()
        self.relationships_by_type.clear()
        self.entities_by_created_date.clear()
        self.entities_by_updated_date.clear()
        self.outgoing_edges.clear()
        self.incoming_edges.clear()
        self.entities_by_id_prefix.clear()
        self.entities_by_name_prefix.clear()

    def to_dict(self) -> dict[str, object]:
        """Serialize index to dictionary.

        Returns:
            Dictionary with all index data.
        """
        return {
            "entity_by_id": {eid: e.to_dict() for eid, e in self.entity_by_id.items()},
            "relationship_by_id": {rid: r.to_dict() for rid, r in self.relationship_by_id.items()},
            "entity_by_name": self.entity_by_name.copy(),
            "entities_by_type": {str(k): list(v) for k, v in self.entities_by_type.items()},
            "relationships_by_type": {
                str(k): list(v) for k, v in self.relationships_by_type.items()
            },
            "entities_by_created_date": {
                str(k): list(v) for k, v in self.entities_by_created_date.items()
            },
            "entities_by_updated_date": {
                str(k): list(v) for k, v in self.entities_by_updated_date.items()
            },
            "outgoing_edges": {k: list(v) for k, v in self.outgoing_edges.items()},
            "incoming_edges": {k: list(v) for k, v in self.incoming_edges.items()},
            "entities_by_id_prefix": {k: list(v) for k, v in self.entities_by_id_prefix.items()},
            "entities_by_name_prefix": {
                k: list(v) for k, v in self.entities_by_name_prefix.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GraphIndex:
        """Deserialize index from dictionary.

        Args:
            data: Dictionary with all index data.

        Returns:
            Restored GraphIndex.
        """
        index = cls()

        # Restore entity and relationship maps
        entity_by_id_data = cast(dict[str, dict[str, Any]], data.get("entity_by_id", {}))
        for eid, entity_data in entity_by_id_data.items():
            index.entity_by_id[eid] = Entity.from_dict(entity_data)

        relationship_by_id_data = cast(
            dict[str, dict[str, Any]], data.get("relationship_by_id", {})
        )
        for rid, rel_data in relationship_by_id_data.items():
            index.relationship_by_id[rid] = Relationship.from_dict(rel_data)

        # Restore name mapping
        entity_by_name_data = cast(dict[str, str], data.get("entity_by_name", {}))
        index.entity_by_name.update(entity_by_name_data)

        # Restore type-based indices (convert string keys back to proper types)
        entities_by_type_data = cast(dict[str, list[str]], data.get("entities_by_type", {}))
        for type_str, entity_ids in entities_by_type_data.items():
            entity_type = cast(EntityType, type_str)
            index.entities_by_type[entity_type] = set(entity_ids)

        relationships_by_type_data = cast(
            dict[str, list[str]], data.get("relationships_by_type", {})
        )
        for type_str, rel_ids in relationships_by_type_data.items():
            rel_type = cast(RelationType, type_str)
            index.relationships_by_type[rel_type] = set(rel_ids)

        # Restore date-based indices
        entities_by_created_data = cast(
            dict[str, list[str]], data.get("entities_by_created_date", {})
        )
        for date_str, entity_ids in entities_by_created_data.items():
            parsed_date = date.fromisoformat(date_str)
            index.entities_by_created_date[parsed_date] = set(entity_ids)

        entities_by_updated_data = cast(
            dict[str, list[str]], data.get("entities_by_updated_date", {})
        )
        for date_str, entity_ids in entities_by_updated_data.items():
            parsed_date = date.fromisoformat(date_str)
            index.entities_by_updated_date[parsed_date] = set(entity_ids)

        # Restore edge indices
        outgoing_edges_data = cast(dict[str, list[str]], data.get("outgoing_edges", {}))
        for source, targets in outgoing_edges_data.items():
            index.outgoing_edges[source] = set(targets)

        incoming_edges_data = cast(dict[str, list[str]], data.get("incoming_edges", {}))
        for target, sources in incoming_edges_data.items():
            index.incoming_edges[target] = set(sources)

        # Restore text search indices
        id_prefix_data = cast(dict[str, list[str]], data.get("entities_by_id_prefix", {}))
        for prefix, entity_ids in id_prefix_data.items():
            index.entities_by_id_prefix[prefix] = set(entity_ids)

        name_prefix_data = cast(dict[str, list[str]], data.get("entities_by_name_prefix", {}))
        for prefix, entity_ids in name_prefix_data.items():
            index.entities_by_name_prefix[prefix] = set(entity_ids)

        return index
