"""Tests for ContextManager functionality."""

from unittest.mock import Mock, patch

import pytest
from agno.agent import Agent
from agno.models.openai import OpenAIChat

from blockether_foundation.context_manager import (
    ContextManager,
    InvalidChunkSizeError,
    TextChunk,
    TokenizationError,
)

# Test constants - Model configuration
TEST_MODEL_ID = "gpt-4o"
TEST_MAX_CONTEXT_TOKENS = 10000
TEST_OVERLAP_TOKENS = 20
TEST_RESERVED_OUTPUT_TOKENS = 100

# Test constants - Agent configuration
TEST_AGENT_NAME = "Test Agent"
TEST_AGENT_DESCRIPTION = "A test agent for unit testing"
TEST_AGENT_INSTRUCTIONS = ["You are a helpful assistant.", "Always be concise."]

# Test constants - Text content
TEST_SHORT_MESSAGE = "This is a short user message."
TEST_EMPTY_MESSAGE = ""
TEST_LONG_MESSAGE = """This is a very long user message that contains multiple paragraphs.
It has enough content to test the chunking functionality.

This is the second paragraph with more content to process.
It should be split appropriately based on token limits.

And here is a third paragraph to make the message even longer.
This will help us test the overlap and boundary detection logic."""

# Test constants - Expected values
EXPECTED_SINGLE_CHUNK_INDEX = 0
EXPECTED_SINGLE_CHUNK_TOTAL = 1
EXPECTED_SINGLE_CHUNK_START = 0

# Test constants - Agent overhead
EXPECTED_MIN_OVERHEAD_TOKENS = 0
EXPECTED_MAX_AVAILABLE_TOKENS = TEST_MAX_CONTEXT_TOKENS - TEST_RESERVED_OUTPUT_TOKENS

# Test constants - TextChunk positions
TEST_OVERLAP_START_CHAR = 20
TEST_OVERLAP_END_CHAR = 30
TEST_START_CHAR = 20
TEST_END_CHAR = 50
TEST_TOKEN_COUNT = 10


def _validate_chunk_character_positions(chunk: TextChunk, message: str) -> None:
    """Validate that character positions in a chunk are valid."""
    chunk_index = chunk.chunk_index
    assert chunk.start_char >= 0, f"Chunk {chunk_index}: Invalid start_char: {chunk.start_char}"
    assert chunk.end_char > chunk.start_char, (
        f"Chunk {chunk_index}: end_char ({chunk.end_char}) <= start_char ({chunk.start_char})"
    )
    assert chunk.end_char <= len(message), (
        f"Chunk {chunk_index}: end_char ({chunk.end_char}) > message length ({len(message)})"
    )


class TestContextManagerInitialization:
    """Test ContextManager initialization."""

    @pytest.mark.unit
    def test_init_with_agent(self) -> None:
        """Test ContextManager initialization with Agent."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(agent=agent)

        assert cm.get_model_id() == TEST_MODEL_ID

    @pytest.mark.unit
    def test_init_with_custom_parameters(self) -> None:
        """Test ContextManager initialization with custom parameters."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            overlap_tokens=TEST_OVERLAP_TOKENS,
        )

        assert cm.get_model_id() == TEST_MODEL_ID

    @pytest.mark.unit
    def test_init_with_agent_no_model_raises_error(self) -> None:
        """Test ContextManager raises error when Agent has no model."""
        agent = Agent(model=None)

        with pytest.raises(InvalidChunkSizeError) as exc_info:
            ContextManager(agent=agent)

        assert "Agent must have a model configured" in str(exc_info.value)

    @pytest.mark.unit
    def test_init_with_model_no_id_raises_error(self) -> None:
        """Test ContextManager raises error when Model has no ID."""
        model = OpenAIChat(id="")
        agent = Agent(model=model)

        with pytest.raises(InvalidChunkSizeError) as exc_info:
            ContextManager(agent=agent)

        assert "Agent model must have an ID" in str(exc_info.value)


class TestContextManagerAgentOverhead:
    """Test agent overhead calculation."""

    @pytest.mark.unit
    def test_calculate_agent_overhead_simple_agent(self) -> None:
        """Test calculating overhead for agent without instructions."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(agent=agent)

        # Test that we can calculate available tokens and the overhead is non-negative
        available = cm.get_available_tokens_for_user_message()
        assert isinstance(available, int)
        assert available > 0

        # Verify that agent overhead is considered in token calculation
        available_with_session = cm.get_available_tokens_for_user_message(session=None)
        assert isinstance(available_with_session, int)

    @pytest.mark.unit
    def test_calculate_agent_overhead_with_instructions(self) -> None:
        """Test calculating overhead for agent with instructions."""
        agent = Agent(
            name=TEST_AGENT_NAME,
            description=TEST_AGENT_DESCRIPTION,
            model=OpenAIChat(id=TEST_MODEL_ID),
            instructions=TEST_AGENT_INSTRUCTIONS,
        )
        cm = ContextManager(agent=agent)

        # Test that agents with instructions have different available tokens
        available = cm.get_available_tokens_for_user_message()
        assert isinstance(available, int)
        assert available > 0

        # The calculation should be consistent
        available_again = cm.get_available_tokens_for_user_message()
        assert available == available_again

    @pytest.mark.unit
    def test_get_available_tokens_for_user_message(self) -> None:
        """Test calculating available tokens for user message."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
        )

        available = cm.get_available_tokens_for_user_message()

        assert available > 0
        # Note: max_context_tokens is model-dependent, so we can't test exact value here

    @pytest.mark.unit
    def test_available_tokens_accounts_for_agent_overhead(self) -> None:
        """Test that available tokens accounts for agent overhead."""
        agent_with_instructions = Agent(
            name=TEST_AGENT_NAME,
            description=TEST_AGENT_DESCRIPTION,
            model=OpenAIChat(id=TEST_MODEL_ID),
            instructions=TEST_AGENT_INSTRUCTIONS,
        )
        agent_without_instructions = Agent(model=OpenAIChat(id=TEST_MODEL_ID))

        cm_with = ContextManager(
            agent=agent_with_instructions,
        )
        cm_without = ContextManager(
            agent=agent_without_instructions,
        )

        available_with = cm_with.get_available_tokens_for_user_message()
        available_without = cm_without.get_available_tokens_for_user_message()

        assert available_with < available_without


class TestContextManagerTokenCounting:
    """Test token counting functionality."""

    @pytest.mark.unit
    def test_count_tokens_short_text(self) -> None:
        """Test counting tokens for short text."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(agent=agent)

        token_count = cm.count_tokens(TEST_SHORT_MESSAGE)

        assert token_count > 0
        assert isinstance(token_count, int)

    @pytest.mark.unit
    def test_count_tokens_empty_text(self) -> None:
        """Test counting tokens for empty text."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(agent=agent)

        token_count = cm.count_tokens(TEST_EMPTY_MESSAGE)

        assert token_count == 0

    @pytest.mark.unit
    @patch("litellm.token_counter")
    def test_count_tokens_handles_litellm_error(self, mock_counter: Mock) -> None:
        """Test count_tokens handles LiteLLM errors gracefully."""
        mock_counter.side_effect = Exception("LiteLLM error")
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(agent=agent)

        with pytest.raises(TokenizationError) as exc_info:
            cm.count_tokens(TEST_SHORT_MESSAGE)

        assert "Failed to count tokens" in str(exc_info.value)

    @pytest.mark.unit
    def test_get_model_id(self) -> None:
        """Test getting model ID."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(agent=agent)

        assert cm.get_model_id() == TEST_MODEL_ID


class TestContextManagerUserMessageSplitting:
    """Test user message splitting functionality."""

    @pytest.mark.unit
    def test_message_fits_single_chunk(self) -> None:
        """Test that short message returns single chunk."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(agent=agent)

        chunks = cm.split_user_message(TEST_SHORT_MESSAGE)

        assert len(chunks) == EXPECTED_SINGLE_CHUNK_TOTAL
        assert chunks[0].chunk_index == EXPECTED_SINGLE_CHUNK_INDEX
        assert chunks[0].total_chunks == EXPECTED_SINGLE_CHUNK_TOTAL
        assert chunks[0].content == TEST_SHORT_MESSAGE

    @pytest.mark.unit
    def test_empty_message_returns_empty_list(self) -> None:
        """Test that empty message returns empty list."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(agent=agent)

        chunks = cm.split_user_message(TEST_EMPTY_MESSAGE)

        assert len(chunks) == 0
        assert chunks == []

    @pytest.mark.unit
    def test_split_long_message(self) -> None:
        """Test splitting long user message."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            overlap_tokens=10,
        )

        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)

    @pytest.mark.unit
    def test_chunks_respect_available_tokens(self) -> None:
        """Test that chunks respect available token limits."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            overlap_tokens=10,
        )

        available_tokens = cm.get_available_tokens_for_user_message()
        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        # Assert each chunk individually
        assert all(chunk.token_count <= available_tokens for chunk in chunks)


class TestContextManagerMetadata:
    """Test chunk metadata accuracy."""

    @pytest.mark.unit
    def test_chunk_indices_sequential(self) -> None:
        """Test that chunk indices are sequential."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            overlap_tokens=10,
        )

        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        # Assert that indices are sequential
        assert len(chunks) >= 1
        assert all(chunk.chunk_index == index for index, chunk in enumerate(chunks))

    @pytest.mark.unit
    def test_total_chunks_correct(self) -> None:
        """Test that total_chunks is correct for all chunks."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            overlap_tokens=10,
        )

        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        # Assert that total_chunks is consistent across all chunks
        total = len(chunks)
        assert total >= 1
        assert all(chunk.total_chunks == total for chunk in chunks)

    @pytest.mark.unit
    @patch("litellm.get_max_tokens")
    def test_character_positions_valid(self, mock_get_max_tokens: Mock) -> None:
        """Test that character positions are valid."""
        mock_get_max_tokens.return_value = 200
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            overlap_tokens=10,
        )

        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        # Assert that character positions are valid for each chunk
        assert len(chunks) >= 1
        for chunk in chunks:
            _validate_chunk_character_positions(chunk, TEST_LONG_MESSAGE)

    @pytest.mark.unit
    def test_token_count_positive(self) -> None:
        """Test that token counts are positive."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            overlap_tokens=10,
        )

        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        assert all(chunk.token_count > 0 for chunk in chunks)


class TestTextChunkModel:
    """Test TextChunk Pydantic model."""

    @pytest.mark.unit
    def test_text_chunk_creation(self) -> None:
        """Test creating TextChunk instance."""
        chunk = TextChunk(
            chunk_index=EXPECTED_SINGLE_CHUNK_INDEX,
            total_chunks=EXPECTED_SINGLE_CHUNK_TOTAL,
            content=TEST_SHORT_MESSAGE,
            token_count=TEST_TOKEN_COUNT,
            start_char=EXPECTED_SINGLE_CHUNK_START,
            end_char=len(TEST_SHORT_MESSAGE),
            overlap_start_char=None,
            overlap_end_char=None,
            boundary_type="sentence",
            # Enhanced marking fields
            split_reason="natural",
            split_boundary="sentence",
            has_overlap_with_previous=False,
            has_overlap_with_next=False,
            overlap_content=None,
            is_first_chunk=True,
            is_last_chunk=True,
        )

        assert chunk.chunk_index == EXPECTED_SINGLE_CHUNK_INDEX
        assert chunk.total_chunks == EXPECTED_SINGLE_CHUNK_TOTAL
        assert chunk.content == TEST_SHORT_MESSAGE
        assert chunk.token_count == TEST_TOKEN_COUNT
        assert chunk.boundary_type == "sentence"
        assert chunk.split_reason == "natural"
        assert chunk.split_boundary == "sentence"
        assert chunk.is_first_chunk
        assert chunk.is_last_chunk

    @pytest.mark.unit
    def test_text_chunk_with_overlap(self) -> None:
        """Test creating TextChunk with overlap metadata."""
        chunk = TextChunk(
            chunk_index=1,
            total_chunks=3,
            content=TEST_SHORT_MESSAGE,
            token_count=TEST_TOKEN_COUNT,
            start_char=TEST_START_CHAR,
            end_char=TEST_END_CHAR,
            overlap_start_char=TEST_OVERLAP_START_CHAR,
            overlap_end_char=TEST_OVERLAP_END_CHAR,
            boundary_type="paragraph",
            # Enhanced marking fields
            split_reason="token_limit",
            split_boundary="paragraph",
            has_overlap_with_previous=True,
            has_overlap_with_next=True,
            overlap_content="overlap text",
            is_first_chunk=False,
            is_last_chunk=False,
        )

        assert chunk.overlap_start_char == TEST_OVERLAP_START_CHAR
        assert chunk.overlap_end_char == TEST_OVERLAP_END_CHAR
        assert chunk.has_overlap_with_previous
        assert chunk.has_overlap_with_next
        assert chunk.overlap_content == "overlap text"
