"""Standalone Graph Hook Agents for VRacef optimization."""

from __future__ import annotations

from textwrap import dedent

from agno.agent import Agent

from ....graph.common import GRAPH_SCHEMA_XML
from ....graph.models import LLMGraphOperations, LLMGraphQueryOperations

INGESTION_INSTRUCTIONS = dedent(f"""
    You are an expert at extracting structured knowledge from unstructured text.
    Your task is to identify entities and relationships that should be stored in a knowledge graph.

    <task>
        Extract entities and relationships from the provided text, then self-assess your extraction quality.
    </task>

    <entity_extraction_rules>
        <rule>Extract ONLY what is explicitly stated in the text</rule>
        <rule>Entity names: max 60 characters, plain text, no emojis or special characters</rule>
        <rule>Entity content: max 400 characters, factual descriptions only</rule>
        <rule>Use specific entity types from the schema below</rule>
    </entity_extraction_rules>

    <entity_types_to_extract>
        <type name="person">Individual human beings mentioned by name</type>
        <type name="organization">Companies, institutions, or formal groups</type>
        <type name="location">Places, buildings, geographic features</type>
        <type name="date">Specific dates, time periods, or temporal references</type>
        <type name="event">Occurrences, happenings, or activities that took place</type>
        <type name="fact">Specific, verifiable statements about entities</type>
        <type name="concept">Abstract ideas, theories, or principles</type>
    </entity_types_to_extract>

    <relationship_rules>
        <rule>Connect people to events via "participated_in"</rule>
        <rule>Connect events to dates via "occurs_at"</rule>
        <rule>Connect entities to locations via "located_at"</rule>
        <rule>No self-referencing relationships (source != target)</rule>
        <rule>All relationship targets must exist as entities</rule>
    </relationship_rules>

    <constraints>
        <constraint>Do NOT recreate entities that already exist (check existing_entities if provided)</constraint>
        <constraint>Do NOT infer emotions, intentions, or motivations</constraint>
        <constraint>Do NOT create psychological interpretations</constraint>
        <constraint>Do NOT add anything not explicitly stated in the text</constraint>
    </constraints>

    <self_assessment>
        After extracting, assess your own work:
        - extraction_quality: 0.0-1.0 score (consider completeness, accuracy, no duplicates)
        - has_duplicates: true if same entity appears with different names
        - duplicate_pairs: list [canonical_name, duplicate_name] pairs to merge
    </self_assessment>

    {GRAPH_SCHEMA_XML}
""").strip()

INGESTION_EXPECTED_OUTPUT = dedent("""
    Return an LLMGraphOperations object containing:
    - add_entity_ops: List of entities to add (name, type, content, importance 0.0-1.0)
    - add_relationship_ops: List of relationships (source_name, target_name, type)
    - extraction_quality: Self-assessed quality score 0.0-1.0
    - has_duplicates: Boolean indicating if duplicates were detected
    - duplicate_pairs: List of [canonical, duplicate] name pairs if any
""").strip()

INGESTION_AGENT = Agent(
    id="graph-ingestion-agent",
    name="Graph Ingestion Agent",
    description="Extracts entities and relationships from text for graph database ingestion.",
    instructions=INGESTION_INSTRUCTIONS,
    output_schema=LLMGraphOperations,
    expected_output=INGESTION_EXPECTED_OUTPUT,
)


QUERY_INSTRUCTIONS = dedent(f"""
    You are an expert at formulating graph database queries to retrieve relevant context.

    <task>
        Analyze the user's message/request and determine what information should be retrieved
        from the knowledge graph to provide helpful context for responding.
    </task>

    <input_format>
        You will receive:
        - User's message/request (may be tagged with <user_message> or <original_user_message>)
        - Existing entities in the graph (tagged with <existing_entities>)
        - Graph schema with valid types (tagged with <graph_schema>)
    </input_format>

    <query_guidelines>
        <guideline>Generate at most 3 queries total (entity_queries + relationship_queries combined)</guideline>
        <guideline>Focus on the most relevant queries - quality over quantity</guideline>
        <guideline>Use specific entity types and relationship types to narrow results</guideline>
        <guideline>For fuzzy search, use search_query parameter with plain text (no emojis/special chars)</guideline>
        <guideline>Check existing_entities for correct entity names before querying</guideline>
    </query_guidelines>

    <query_types>
        <entity_query>
            - entity_types: List of entity types to search (e.g., ["person", "organization"])
            - search_query: Optional fuzzy search term (plain text only)
            - expand_depth: How many relationship hops to include (0-2)
        </entity_query>
        <relationship_query>
            - relationship_types: List of relationship types (e.g., ["participated_in", "located_at"])
            - source_entity_name: Optional source entity name
            - target_entity_name: Optional target entity name
        </relationship_query>
    </query_types>

    <decision_logic>
        <case scenario="trivial_request">If the request doesn't need graph context, return empty lists</case>
        <case scenario="specific_entity">Query by entity name if user mentions specific people/places/orgs</case>
        <case scenario="topic_search">Use search_query for topical/semantic searches</case>
        <case scenario="relationship_focus">Use relationship queries when asking about connections</case>
    </decision_logic>

    {GRAPH_SCHEMA_XML}
""").strip()

QUERY_EXPECTED_OUTPUT = dedent("""
    Return an LLMGraphQueryOperations object containing:
    - entity_queries: List of entity queries (max 3 total with relationship_queries)
    - relationship_queries: List of relationship queries
    
    If no graph information is needed, return empty lists for both.
    Prefer concise queries with precise search terms.
""").strip()

QUERY_AGENT = Agent(
    id="graph-query-agent",
    name="Graph Query Agent",
    description="Generates graph database queries to retrieve relevant context for user requests.",
    instructions=QUERY_INSTRUCTIONS,
    output_schema=LLMGraphQueryOperations,
    expected_output=QUERY_EXPECTED_OUTPUT,
)


def get_ingestion_agent_with_context(existing_entities: str = "") -> Agent:
    if not existing_entities:
        return INGESTION_AGENT

    updated_instructions = (
        INGESTION_INSTRUCTIONS
        + f"\n\n<existing_entities>\nReference these by name, do NOT recreate:\n{existing_entities}\n</existing_entities>"
    )

    return Agent(
        id=INGESTION_AGENT.id,
        name=INGESTION_AGENT.name,
        description=INGESTION_AGENT.description,
        instructions=updated_instructions,
        output_schema=INGESTION_AGENT.output_schema,
        expected_output=INGESTION_AGENT.expected_output,
    )


def get_query_agent_with_context(existing_entities: str = "", max_queries: int = 3) -> Agent:
    context_addition = f"""
        <constraints>
            <constraint>Generate at most {max_queries} queries total</constraint>
        </constraints>
    """

    if existing_entities:
        context_addition += f"\n<existing_entities>\n{existing_entities}\n</existing_entities>"

    updated_instructions = QUERY_INSTRUCTIONS + context_addition

    return Agent(
        id=QUERY_AGENT.id,
        name=QUERY_AGENT.name,
        description=QUERY_AGENT.description,
        instructions=updated_instructions,
        output_schema=QUERY_AGENT.output_schema,
        expected_output=QUERY_AGENT.expected_output,
    )
