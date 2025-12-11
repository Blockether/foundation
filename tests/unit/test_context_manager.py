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
            max_context_tokens=TEST_MAX_CONTEXT_TOKENS,
            overlap_tokens=TEST_OVERLAP_TOKENS,
            reserved_output_tokens=TEST_RESERVED_OUTPUT_TOKENS,
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

        overhead = cm.calculate_agent_overhead()

        assert overhead >= EXPECTED_MIN_OVERHEAD_TOKENS
        assert isinstance(overhead, int)

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

        overhead = cm.calculate_agent_overhead()

        assert overhead > EXPECTED_MIN_OVERHEAD_TOKENS

    @pytest.mark.unit
    def test_get_available_tokens_for_user_message(self) -> None:
        """Test calculating available tokens for user message."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            max_context_tokens=TEST_MAX_CONTEXT_TOKENS,
            reserved_output_tokens=TEST_RESERVED_OUTPUT_TOKENS,
        )

        available = cm.get_available_tokens_for_user_message()

        assert available > 0
        assert available <= EXPECTED_MAX_AVAILABLE_TOKENS

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
            max_context_tokens=TEST_MAX_CONTEXT_TOKENS,
        )
        cm_without = ContextManager(
            agent=agent_without_instructions,
            max_context_tokens=TEST_MAX_CONTEXT_TOKENS,
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
        cm = ContextManager(agent=agent, max_context_tokens=100000)

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
            max_context_tokens=200,
            overlap_tokens=10,
            reserved_output_tokens=50,
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
            max_context_tokens=300,
            overlap_tokens=10,
            reserved_output_tokens=50,
        )

        available_tokens = cm.get_available_tokens_for_user_message()
        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        for chunk in chunks:
            assert chunk.token_count <= available_tokens


class TestContextManagerMetadata:
    """Test chunk metadata accuracy."""

    @pytest.mark.unit
    def test_chunk_indices_sequential(self) -> None:
        """Test that chunk indices are sequential."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            max_context_tokens=200,
            overlap_tokens=10,
        )

        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        if len(chunks) > 1:
            expected_indices = list(range(len(chunks)))
            actual_indices = [chunk.chunk_index for chunk in chunks]
            assert actual_indices == expected_indices

    @pytest.mark.unit
    def test_total_chunks_correct(self) -> None:
        """Test that total_chunks is correct for all chunks."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            max_context_tokens=200,
            overlap_tokens=10,
        )

        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        if len(chunks) > 0:
            total = len(chunks)
            assert all(chunk.total_chunks == total for chunk in chunks)

    @pytest.mark.unit
    def test_character_positions_valid(self) -> None:
        """Test that character positions are valid."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            max_context_tokens=200,
            overlap_tokens=10,
        )

        chunks = cm.split_user_message(TEST_LONG_MESSAGE)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert chunk.end_char <= len(TEST_LONG_MESSAGE)

    @pytest.mark.unit
    def test_token_count_positive(self) -> None:
        """Test that token counts are positive."""
        agent = Agent(model=OpenAIChat(id=TEST_MODEL_ID))
        cm = ContextManager(
            agent=agent,
            max_context_tokens=200,
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
        )

        assert chunk.chunk_index == EXPECTED_SINGLE_CHUNK_INDEX
        assert chunk.total_chunks == EXPECTED_SINGLE_CHUNK_TOTAL
        assert chunk.content == TEST_SHORT_MESSAGE
        assert chunk.token_count == TEST_TOKEN_COUNT
        assert chunk.boundary_type == "sentence"

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
        )

        assert chunk.overlap_start_char == TEST_OVERLAP_START_CHAR
        assert chunk.overlap_end_char == TEST_OVERLAP_END_CHAR
