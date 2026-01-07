"""Common definitions for graph entity and relationship types."""

from __future__ import annotations

from textwrap import dedent

ENTITY_TYPE_DEFINITIONS: dict[str, str] = {
    "creature": "Living beings, animals, or any sentient entities which are not human.",
    "person": "Individual human beings with distinct identities and roles.",
    "organization": "Companies, institutions, or formal groups.",
    "location": "Places, buildings, geographic features, or virtual spaces, place of living of the individuals.",
    "event": "Historical events, occurrences, or time-specific happenings.",
    "date": "Specific calendar dates, time periods, or temporal references.",
    "document": "Files, reports, articles, or written records.",
    "attachment": "Binary files, media, images, or non-text attachments.",
    "object": "Physical items, equipment, tools, or tangible artifacts.",
    "concept": "Abstract ideas, theories, principles, or intellectual concepts.",
    "example": "Specific instances, illustrations, or representative cases of concepts or rules.",
    "rule": "Logical rules, constraints, regulations, or governing principles.",
    "pattern": "Recognizable patterns, templates, or recurring structures, including therapy patterns.",
    "mode": "States of operation or behavioral patterns, including schema therapy modes.",
    "schema": "Structured frameworks or mental models, including therapeutic schemas.",
    "abbreviation": "Shortened forms, acronyms, or initialisms representing longer terms.",
    "reference": "Citations, links, or pointers to other entities, documents, or sources.",
    "memory": "Stored experiences, learned information, or recollections.",
    "fact": "Specific, verifiable statements or pieces of information about entities.",
}

RELATIONSHIP_TYPE_DEFINITIONS: dict[str, str] = {
    "part_of": "Entity is a component, member, or subset of another entity.",
    "owned_by": "Entity is owned, possessed, or controlled by another entity.",
    "occurs_at": "Event happens at a specific time and/or place. Event can be a specific situation, online or physical.",
    "related_to": "General association, connection, or relationship between entities.",
    "triggers": "Entity causes, initiates, or precipitates another entity or event.",
    "expands_to": "Entity can be expanded or decomposed into more detailed components.",
    "explains": "Entity provides rationale or underlying reasons for another e.g. specific situation explaining a behavior; specific concept explaining a rule; specific example illustrating a concept or rule.",
    "causes": "Direct causation, antecedent-consequent relationships.",
    "is_a_fact": "Entity has a specific, verifiable fact associated with it.",
    "participated_in": "Entity (person/creature) participated in an event, situation, or activity.",
    "alias_of": "Entity is an alternative name or reference to another entity (used for entity resolution).",
    "created_by": "Entity was built, made, or authored by another entity.",
    "located_at": "Entity is situated, resides, or exists at a location.",
}


def build_graph_schema_xml() -> str:
    """Build XML schema definition for entity and relationship types."""
    entity_type_xml = "\n".join(
        f'            <entity_type name="{etype}" definition="{definition}" />'
        for etype, definition in ENTITY_TYPE_DEFINITIONS.items()
    )

    relationship_type_xml = "\n".join(
        f'            <relationship_type name="{rtype}" definition="{definition}" />'
        for rtype, definition in RELATIONSHIP_TYPE_DEFINITIONS.items()
    )

    return dedent(f"""
    <graph_schema>
        <valid_entity_types>
{entity_type_xml}
        </valid_entity_types>
        <valid_relationship_types>
{relationship_type_xml}
        </valid_relationship_types>
    </graph_schema>
    """)


GRAPH_SCHEMA_XML = build_graph_schema_xml()
