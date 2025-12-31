"""Multi-index system for fast graph queries."""

from __future__ import annotations

import bisect
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

    # Sorted date lists for O(log n) range queries
    _sorted_created_dates: list[date] = field(default_factory=list)
    _sorted_updated_dates: list[date] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> None:
        """Add entity to all relevant indices."""
        self.entity_by_id[entity.id] = entity
        self.entity_by_name[entity.name] = entity.id
        self.entities_by_type[entity.type].add(entity.id)

        created_date = entity.created_at.date()
        if created_date not in self.entities_by_created_date:
            bisect.insort(self._sorted_created_dates, created_date)
        self.entities_by_created_date[created_date].add(entity.id)

        updated_date = entity.updated_at.date()
        if updated_date not in self.entities_by_updated_date:
            bisect.insort(self._sorted_updated_dates, updated_date)
        self.entities_by_updated_date[updated_date].add(entity.id)

    def update_entity(self, old_entity: Entity, new_entity: Entity) -> None:
        """Update entity in all relevant indices."""
        self.remove_entity(old_entity)
        self.add_entity(new_entity)

    def remove_entity(self, entity: Entity) -> None:
        """Remove entity from all relevant indices."""
        self.entity_by_id.pop(entity.id, None)
        self.entity_by_name.pop(entity.name, None)
        self.entities_by_type[entity.type].discard(entity.id)

        created_date = entity.created_at.date()
        self.entities_by_created_date[created_date].discard(entity.id)
        if not self.entities_by_created_date[created_date]:
            del self.entities_by_created_date[created_date]
            idx = bisect.bisect_left(self._sorted_created_dates, created_date)
            if idx < len(self._sorted_created_dates) and self._sorted_created_dates[idx] == created_date:
                self._sorted_created_dates.pop(idx)

        updated_date = entity.updated_at.date()
        self.entities_by_updated_date[updated_date].discard(entity.id)
        if not self.entities_by_updated_date[updated_date]:
            del self.entities_by_updated_date[updated_date]
            idx = bisect.bisect_left(self._sorted_updated_dates, updated_date)
            if idx < len(self._sorted_updated_dates) and self._sorted_updated_dates[idx] == updated_date:
                self._sorted_updated_dates.pop(idx)

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
        self._sorted_created_dates.clear()
        self._sorted_updated_dates.clear()

    def get_entities_in_date_range(
        self,
        start_date: date | None,
        end_date: date | None,
        use_created: bool = True,
    ) -> set[str]:
        """Get entity IDs within a date range using O(log n) binary search.

        Args:
            start_date: Start of range (inclusive), None for no lower bound
            end_date: End of range (inclusive), None for no upper bound
            use_created: If True, use created_at dates; if False, use updated_at

        Returns:
            Set of entity IDs within the range
        """
        sorted_dates = self._sorted_created_dates if use_created else self._sorted_updated_dates
        date_index = self.entities_by_created_date if use_created else self.entities_by_updated_date

        if not sorted_dates:
            return set()

        # Find range boundaries using binary search
        if start_date is None:
            left = 0
        else:
            left = bisect.bisect_left(sorted_dates, start_date)

        if end_date is None:
            right = len(sorted_dates)
        else:
            right = bisect.bisect_right(sorted_dates, end_date)

        # Collect entities from dates in range
        result: set[str] = set()
        for i in range(left, right):
            result.update(date_index[sorted_dates[i]])
        return result

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
        # Rebuild sorted dates from keys
        index._sorted_created_dates = sorted(index.entities_by_created_date.keys())

        entities_by_updated_data = cast(
            dict[str, list[str]], data.get("entities_by_updated_date", {})
        )
        for date_str, entity_ids in entities_by_updated_data.items():
            parsed_date = date.fromisoformat(date_str)
            index.entities_by_updated_date[parsed_date] = set(entity_ids)
        # Rebuild sorted dates from keys
        index._sorted_updated_dates = sorted(index.entities_by_updated_date.keys())

        # Restore edge indices
        outgoing_edges_data = cast(dict[str, list[str]], data.get("outgoing_edges", {}))
        for source, targets in outgoing_edges_data.items():
            index.outgoing_edges[source] = set(targets)

        incoming_edges_data = cast(dict[str, list[str]], data.get("incoming_edges", {}))
        for target, sources in incoming_edges_data.items():
            index.incoming_edges[target] = set(sources)

        return index
