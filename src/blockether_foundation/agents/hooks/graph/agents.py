"""Standalone Graph Hook Agents for VRacef optimization."""

from __future__ import annotations

from textwrap import dedent

from agno.agent import Agent

from ....graph.common import GRAPH_SCHEMA_XML
from ....graph.models import LLMGraphOperations, LLMGraphQueryOperations

INGESTION_INSTRUCTIONS = dedent(f"""
    You are an expert at extracting structured knowledge from unstructured text.

    <task>
        1. Extract entities and relationships from the provided text
        2. Call the import_to_graph tool with your extracted operations
        3. If import fails, fix the errors and retry
    </task>

    <entity_extraction_rules>
        <rule>Extract ONLY what is explicitly stated in the text</rule>
        <rule>Entity names: max 60 characters, plain ASCII text only</rule>
        <rule>Entity content: max 400 characters, factual descriptions only</rule>
    </entity_extraction_rules>

    <entity_types>
        <type name="person">Individual human beings mentioned by name</type>
        <type name="organization">Companies, institutions, or formal groups</type>
        <type name="location">Places, buildings, geographic features</type>
        <type name="date">Specific dates, time periods, or temporal references</type>
        <type name="event">Occurrences, happenings, or activities</type>
        <type name="fact">Specific, verifiable statements</type>
        <type name="concept">Abstract ideas, theories, or principles</type>
    </entity_types>

    <relationship_rules>
        <rule>Connect people to events via "participated_in"</rule>
        <rule>Connect events to dates via "occurs_at"</rule>
        <rule>No self-referencing (source != target)</rule>
        <rule>All targets must exist as entities</rule>
    </relationship_rules>

    <constraints>
        <constraint>Do NOT recreate entities that already exist</constraint>
        <constraint>Do NOT infer emotions or intentions</constraint>
        <constraint>Do NOT add anything not explicitly stated</constraint>
    </constraints>

    {GRAPH_SCHEMA_XML}
""").strip()

INGESTION_EXPECTED_OUTPUT = "Extract entities/relationships and call import_to_graph tool."

QUERY_INSTRUCTIONS = dedent(f"""
    You are an expert at formulating graph database queries.

    <task>
        Analyze the user's message and determine what to retrieve from the knowledge graph.
    </task>

    <input_format>
        - User's message (may be tagged with <user_message>)
        - Existing entities (tagged with <existing_entities>)
        - Graph schema (tagged with <graph_schema>)
    </input_format>

    <guidelines>
        <guideline>Generate at most 3 queries total</guideline>
        <guideline>Use specific entity/relationship types</guideline>
        <guideline>For fuzzy search, use plain text in search_query</guideline>
    </guidelines>

    <query_types>
        <entity_query>entity_types, search_query (optional), expand_depth (0-2)</entity_query>
        <relationship_query>relationship_types, source_entity_name, target_entity_name</relationship_query>
    </query_types>

    {GRAPH_SCHEMA_XML}
""").strip()

QUERY_EXPECTED_OUTPUT = dedent("""
    Return LLMGraphQueryOperations with entity_queries and relationship_queries.
    If no graph info needed, return empty lists.
""").strip()

QUERY_AGENT = Agent(
    id="graph-query-agent",
    name="Graph Query Agent",
    description="Generates graph database queries to retrieve relevant context.",
    instructions=QUERY_INSTRUCTIONS,
    output_schema=LLMGraphQueryOperations,
    expected_output=QUERY_EXPECTED_OUTPUT,
)

INGESTION_AGENT = Agent(
    id="graph-ingestion-agent",
    name="Graph Ingestion Agent",
    description="Extracts entities/relationships and imports them to graph database.",
    instructions=INGESTION_INSTRUCTIONS,
    output_schema=LLMGraphOperations,
    expected_output=INGESTION_EXPECTED_OUTPUT,
)


def get_query_agent_with_context(existing_entities: str = "", max_queries: int = 3) -> Agent:
    context = f"<constraints><constraint>Max {max_queries} queries total</constraint></constraints>"
    if existing_entities:
        context += f"\n<existing_entities>\n{existing_entities}\n</existing_entities>"

    return Agent(
        id=QUERY_AGENT.id,
        name=QUERY_AGENT.name,
        description=QUERY_AGENT.description,
        instructions=QUERY_INSTRUCTIONS + context,
        output_schema=QUERY_AGENT.output_schema,
        expected_output=QUERY_AGENT.expected_output,
    )
