"""Prompts for graph database hooks.

All agent prompts are centralized here for maintainability.
"""

from __future__ import annotations

from textwrap import dedent

from ....graph.common import GRAPH_SCHEMA_XML

GRAPH_EXTRACTION_PROMPT = dedent("""
    Extract entities and relationships, then self-assess your extraction quality.

    EXTRACT (only what is explicitly stated):
    - People by name (type: person)
    - Organizations (type: organization)
    - Locations (type: location)
    - Dates/times (type: date)
    - Events that occurred (type: event)
    - Stated facts (type: fact)

    RULES:
    - Entity names: max 60 chars, plain text, no emojis
    - Entity content: max 400 chars, factual only
    - Connect people to events via participated_in
    - Connect events to dates via occurs_at
    - Don't recreate entities that already exist
    - No self-referencing relationships
    - All relationship targets must exist

    SELF-ASSESSMENT (required):
    After extracting, assess your own work:
    - extraction_quality: 0.0-1.0 score (consider duplicates, verbose names, completeness)
    - has_duplicates: true if same entity appears with different names (e.g., "John" and "John Smith")
    - duplicate_pairs: list [canonical_name, duplicate_name] pairs to merge

    DO NOT:
    - Infer emotions, intentions, or motivations
    - Create psychological interpretations
    - Add anything not explicitly in the text
""").strip()


def get_extraction_prompt(existing_entities: str = "") -> str:
    """Get the extraction prompt with optional existing entities context."""
    prompt = GRAPH_EXTRACTION_PROMPT

    if existing_entities:
        prompt += f"\n\nEXISTING ENTITIES (reference by name, don't recreate):\n{existing_entities}"

    return prompt


# =============================================================================
# QUERY GENERATION PROMPT
# =============================================================================

QUERY_GENERATION_DESCRIPTION = (
    "You are an expert in producing graph database queries to retrieve relevant context."
)


def get_query_generation_instructions(existing_entities: str, max_queries: int = 3) -> str:
    """Get instructions for query generation agent.

    Args:
        existing_entities: Formatted string of existing entities in the graph
        max_queries: Maximum total queries allowed (entity + relationship queries combined)
    """
    return (
        dedent(f"""
            <task_description>
                You are an expert in producing graph database queries to retrieve relevant context.
                You will be given the user's message/request/question, potentially denoted with <user_message> or <original_user_message> tags.
                Your task is to analyze the user request and determine what information is needed from the graph database.
                For trivial requests, you may decide that no graph information is needed.
            </task_description>
            <constraints>
                <constraint>Generate at most {max_queries} queries total (entity_queries + relationship_queries combined).</constraint>
                <constraint>Focus on the most relevant queries. Quality over quantity.</constraint>
            </constraints>
            <xml_tag_reference>
                <tag name="user_message">Denotes the user's message, request, or question.</tag>
                <tag name="original_user_message">Denotes the original user input that started the conversation.</tag>
                <tag name="graph_schema">Denotes the graph schema definitions with valid entity and relationship types.</tag>
                <tag name="existing_entities">Denotes current top entities in the graph with the highest information density.</tag>
            </xml_tag_reference>
            <guidelines>
                <guideline>You will be given the graph schema definitions denoted by <graph_schema> to help you formulate valid queries.</guideline>
                <guideline>You will be given current top entities in the graph with the highest information density, denoted by <existing_entities>.</guideline>
                <guideline>For fuzzy search queries, use the search_query parameter on entity_queries. ENSURE THE SEARCH QUERY contains no formatted text, emoticons, or special characters.</guideline>
                <guideline>Use specific entity types and relationship types where possible to narrow down results.</guideline>
            </guidelines>""")
        + "\n"
        + GRAPH_SCHEMA_XML
        + "\n"
        + existing_entities
    )


QUERY_GENERATION_EXPECTED_OUTPUT = dedent("""
    Expectations for your output:
    - Return an LLMGraphQueryOperations object with two lists: entity_queries and relationship_queries.
    - Each query should be relevant to the user's request and aim to retrieve useful context.
    - If no graph information is needed, return empty lists for both entity_queries and relationship_queries.
    - Prefer concise queries with precise search terms and specific entity/relationship types.
    - Avoid overly broad queries that return too much irrelevant data.
    - Ensure queries do not duplicate information already present in the existing entities.
""")


# =============================================================================
# FIX EMPTY QUERIES PROMPT
# =============================================================================

FIX_EMPTY_QUERIES_INSTRUCTIONS: list[str] = [
    "You will receive:",
    "1. User's original request",
    "2. Graph queries that returned NO RESULTS",
    "3. Existing entities in the graph to help you understand what's available",
    "",
    "Your task: Generate BETTER queries that will actually find relevant data.",
    "",
    "Common fixes:",
    "- SEARCH QUERY TOO SPECIFIC: Use broader search terms",
    "  Example: 'Dr. John Smith PhD' -> 'John Smith' or just 'Smith'",
    "- WRONG ENTITY TYPE: Check what entity types exist in the graph",
    "  Example: Looking for 'person' when data is stored as 'concept'",
    "- TYPOS IN NAMES: Check existing entity names for correct spelling",
    "- RELATIONSHIP DIRECTION: Try reversing source/target",
    "",
    "IMPORTANT:",
    "- Look at the existing entities to understand naming conventions",
    "- Use simpler, broader search queries",
    "- Try different entity types if the original ones returned nothing",
    "- Generate queries that will actually match existing data",
    GRAPH_SCHEMA_XML,
]


def build_fix_empty_queries_input(
    input_content: str,
    failed_entity_queries: list[str],
    failed_rel_queries: list[str],
    existing_entities: str,
) -> str:
    """Build input for fix empty queries agent."""
    return f"""<user_request>
{input_content}
</user_request>

<failed_queries>
Entity queries that returned NO results:
{chr(10).join(f"  - {q}" for q in failed_entity_queries) if failed_entity_queries else "  (none)"}

Relationship queries that returned NO results:
{chr(10).join(f"  - {q}" for q in failed_rel_queries) if failed_rel_queries else "  (none)"}
</failed_queries>

{existing_entities}

Generate new queries that will actually find relevant data. Use simpler search terms and check the entity types available in the graph."""


# =============================================================================
# FIX OPERATIONS PROMPT
# =============================================================================

FIX_OPERATIONS_INSTRUCTIONS: list[str] = [
    "You will receive:",
    "1. Invalid graph operations that failed validation",
    "2. Specific error messages describing what's wrong",
    "",
    "Your task: Fix ONLY the reported errors.",
    "",
    "Common fixes:",
    "- ENTITY ALREADY EXISTS: If error says entity 'already exists in the graph', "
    "  REMOVE that entity from add_entity_ops entirely. Do NOT rename it. "
    "  The entity already exists in the database, so relationships can reference it.",
    "- TYPE IN NAME: If error says name 'contains its type', remove the type word. "
    "  Example: 'Abandonment schema' -> 'Abandonment' (type is already 'schema')",
    "- PARENTHETICAL SUFFIX: If error mentions parenthetical suffix like '(Mark)', "
    "  remove it and use relationships instead. "
    "  Example: 'Detached Protector mode (Mark)' -> 'Detached Protector' with relationship to Mark",
    "- Remove emojis/special characters from names and content",
    "- Shorten names > 60 chars to identifiers",
    "- Shorten content > 400 chars",
    "- Change invalid entity/relationship types to valid ones",
    "- Create missing entities before referencing them",
    "- Remove self-referencing relationships",
    "- Remove duplicate entities or relationships",
    "",
    "IMPORTANT:",
    "- For 'already exists' errors: REMOVE the entity, don't rename it",
    "- Fix only what's reported as errors",
    "- Don't change operations that passed validation",
    "- Preserve all valid data",
    "- Output valid LLMGraphOperations object",
]


def build_fix_operations_input(
    operations_text: str,
    errors: list[str],
    valid_entity_types_str: str,
    valid_relationship_types_str: str,
) -> str:
    """Build input for fix operations agent."""
    return f"""<invalid_operations>
{operations_text}
</invalid_operations>

<validation_errors>
{chr(10).join(f"- {err}" for err in errors)}
</validation_errors>

Fix these errors and return corrected operations. Ensure all validation requirements are met:
- Entity names: Plain ASCII text, max 60 chars
- Entity content: Plain text (no markdown), max 400 chars
- Valid entity types: {valid_entity_types_str}
- Valid relationship types: {valid_relationship_types_str}
- No self-referencing relationships (source != target)
- All referenced entities must exist
- No duplicate entity names or relationships"""
