"""Context-aware text chunking with token management.

This module provides utilities for splitting user messages into token-aware chunks
while accounting for agent overhead (instructions, system prompts, context).
"""

from __future__ import annotations

import re

import litellm
from agno.agent import Agent
from agno.session import AgentSession
from pydantic import BaseModel, Field

from .errors import FoundationBaseError


class ContextManagerError(FoundationBaseError):
    """Base error for ContextManager operations."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidChunkSizeError(ContextManagerError):
    """Raised when chunk size parameters are invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TokenizationError(ContextManagerError):
    """Raised when text tokenization fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class TextChunk(BaseModel):
    """Represents a single chunk of text with metadata."""

    chunk_index: int = Field(description="Zero-based index of this chunk")
    total_chunks: int = Field(description="Total number of chunks in the split")
    content: str = Field(description="The actual text content of this chunk")
    token_count: int = Field(description="Number of tokens in this chunk")
    start_char: int = Field(
        description="Character position where this chunk starts in original text"
    )
    end_char: int = Field(description="Character position where this chunk ends in original text")
    overlap_start_char: int | None = Field(
        default=None,
        description="Character position where overlap with previous chunk begins (None for first chunk)",
    )
    overlap_end_char: int | None = Field(
        default=None,
        description="Character position where overlap with next chunk ends (None for last chunk)",
    )
    boundary_type: str = Field(
        description='Type of boundary used for this chunk: "paragraph", "sentence", "word", or "token"'
    )


class ContextManager:
    """Manages token-aware user message chunking accounting for agent overhead."""

    def __init__(
        self,
        agent: Agent,
        max_context_tokens: int | None = None,
        overlap_tokens: int = 200,
        reserved_output_tokens: int | None = None,
    ) -> None:
        """Initialize the ContextManager.

        Args:
            agent: Agno Agent instance
            max_context_tokens: Maximum context window tokens for the model
            overlap_tokens: Target overlap size in tokens between chunks
            reserved_output_tokens: Tokens to reserve for model output

        Raises:
            InvalidChunkSizeError: If agent configuration is invalid
        """
        if agent.model is None:
            raise InvalidChunkSizeError("Agent must have a model configured")

        if not agent.model.id:
            raise InvalidChunkSizeError("Agent model must have an ID configured")

        self._agent = agent
        self._model_id = agent.model.id

        # Resolve max_context_tokens
        if max_context_tokens is not None:
            self._max_context_tokens = max_context_tokens
        elif hasattr(agent, "max_context_tokens"):
            self._max_context_tokens = getattr(agent, "max_context_tokens")
        else:
            try:
                # Try to get from litellm model cost
                model_info = litellm.model_cost.get(self._model_id)
                if model_info and "max_input_tokens" in model_info:
                    self._max_context_tokens = int(model_info["max_input_tokens"])
                else:
                    # Fallback to standard get_max_tokens
                    self._max_context_tokens = litellm.get_max_tokens(self._model_id) or 128000
            except Exception:
                # Default fallback if litellm fails
                self._max_context_tokens = 128000

        self._overlap_tokens = overlap_tokens

        # Resolve reserved_output_tokens
        if reserved_output_tokens is not None:
            self._reserved_output_tokens = reserved_output_tokens
        elif hasattr(agent, "reserved_output_tokens"):
            self._reserved_output_tokens = getattr(agent, "reserved_output_tokens")
        elif hasattr(agent.model, "max_tokens") and agent.model.max_tokens is not None:
            self._reserved_output_tokens = agent.model.max_tokens
        else:
            self._reserved_output_tokens = 4000

        self._agent_overhead_tokens: int | None = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using LiteLLM.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens in the text

        Raises:
            TokenizationError: If tokenization fails
        """
        try:
            return litellm.token_counter(model=self._model_id, text=text)  # type: ignore[reportUnknownMemberType]
        except Exception as e:
            raise TokenizationError(f"Failed to count tokens: {e}") from e

    def get_model_id(self) -> str:
        """Get the model ID being used for tokenization.

        Returns:
            Model ID string
        """
        return self._model_id

    def calculate_agent_overhead(self, session: AgentSession | None = None) -> int:
        """Calculate tokens used by agent configuration.

        Args:
            session: Optional session to get full system message with context

        Returns:
            Number of tokens used by agent overhead
        """
        overhead_parts: list[str] = []

        if self._agent.name:
            overhead_parts.append(f"Agent: {self._agent.name}")

        if self._agent.description:
            overhead_parts.append(f"Description: {self._agent.description}")

        if self._agent.instructions:  # type: ignore[reportUnknownMemberType]
            instructions = self._agent.instructions  # type: ignore[reportUnknownMemberType]
            if isinstance(instructions, list):
                instructions_text = "\n".join(instructions)  # type: ignore[reportUnknownArgumentType]
                overhead_parts.append(f"Instructions: {instructions_text}")

        if session:
            try:
                system_message = self._agent.get_system_message(session)  # type: ignore[reportUnknownMemberType]
                if system_message:
                    message_content = (
                        str(system_message.content)
                        if hasattr(system_message, "content")
                        else str(system_message)
                    )
                    overhead_parts.append(message_content)
            except Exception:
                pass

        full_overhead = "\n".join(overhead_parts)
        self._agent_overhead_tokens = self.count_tokens(full_overhead) if full_overhead else 0

        return self._agent_overhead_tokens

    def get_available_tokens_for_user_message(self, session: AgentSession | None = None) -> int:
        """Calculate how many tokens are available for user message.

        Args:
            session: Optional session for calculating full agent overhead

        Returns:
            Number of tokens available for user message
        """
        if self._agent_overhead_tokens is None:
            self.calculate_agent_overhead(session)

        available = (
            self._max_context_tokens
            - (self._agent_overhead_tokens or 0)
            - self._reserved_output_tokens
        )

        return max(available, 100)

    def split_user_message(
        self, user_message: str, session: AgentSession | None = None
    ) -> list[TextChunk]:
        """Split user message into chunks accounting for agent overhead.

        Args:
            user_message: User message to split
            session: Optional session for calculating full agent overhead

        Returns:
            List of TextChunk objects with metadata

        Raises:
            TokenizationError: If tokenization fails
        """
        if not user_message:
            return []

        available_tokens = self.get_available_tokens_for_user_message(session)
        message_tokens = self.count_tokens(user_message)

        if message_tokens <= available_tokens:
            return [self._create_single_chunk(user_message)]

        max_chunk_tokens = available_tokens - self._overlap_tokens
        paragraphs = self._split_paragraphs(user_message)
        raw_chunks = self._accumulate_chunks(paragraphs, max_chunk_tokens)

        return self._apply_overlap_and_create_chunks(raw_chunks, user_message)

    def _create_single_chunk(self, text: str) -> TextChunk:
        """Create a single chunk for text that fits within limits."""
        return TextChunk(
            chunk_index=0,
            total_chunks=1,
            content=text,
            token_count=self.count_tokens(text),
            start_char=0,
            end_char=len(text),
            overlap_start_char=None,
            overlap_end_char=None,
            boundary_type="none",
        )

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        return [p for p in text.split("\n\n") if p.strip()]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        pattern = r"(?<=[.!?])\s+"
        return [s for s in re.split(pattern, text) if s.strip()]

    def _split_words(self, text: str) -> list[str]:
        """Split text into words."""
        return [w for w in text.split() if w.strip()]

    def _accumulate_chunks(self, paragraphs: list[str], max_chunk_tokens: int) -> list[str]:
        """Accumulate paragraphs into chunks respecting token limits."""
        chunks: list[str] = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if current_tokens + para_tokens <= max_chunk_tokens:
                separator = "\n\n" if current_chunk else ""
                current_chunk = f"{current_chunk}{separator}{para}"
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if para_tokens > max_chunk_tokens:
                    chunks.extend(self._split_large_paragraph(para, max_chunk_tokens))
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = para
                    current_tokens = para_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_large_paragraph(self, para: str, max_chunk_tokens: int) -> list[str]:
        """Split a large paragraph that exceeds token limits."""
        para_tokens = self.count_tokens(para)

        if para_tokens <= max_chunk_tokens:
            return [para]

        sentences = self._split_sentences(para)
        return self._accumulate_units(sentences, "sentence", max_chunk_tokens)

    def _accumulate_units(
        self, units: list[str], unit_type: str, max_chunk_tokens: int
    ) -> list[str]:
        """Accumulate units (sentences/words) into chunks."""
        chunks: list[str] = []
        current_chunk = ""
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)

            if unit_tokens > max_chunk_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                if unit_type == "sentence":
                    words = self._split_words(unit)
                    chunks.extend(self._accumulate_units(words, "word", max_chunk_tokens))
                else:
                    chunks.extend(self._force_split_by_tokens(unit, max_chunk_tokens))
                continue

            if current_tokens + unit_tokens <= max_chunk_tokens:
                separator = " " if current_chunk else ""
                current_chunk = f"{current_chunk}{separator}{unit}"
                current_tokens += unit_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = unit
                current_tokens = unit_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _force_split_by_tokens(self, text: str, max_chunk_tokens: int) -> list[str]:
        """Force split text by character approximation when it exceeds token limits."""
        text_tokens = self.count_tokens(text)
        if text_tokens <= max_chunk_tokens:
            return [text]

        avg_chars_per_token = len(text) / max(text_tokens, 1)
        chunk_chars = int(max_chunk_tokens * avg_chars_per_token * 0.9)

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunk = text[start:end]

            while self.count_tokens(chunk) > max_chunk_tokens and len(chunk) > 1:
                end = start + int(len(chunk) * 0.9)
                chunk = text[start:end]

            chunks.append(chunk)
            start = end

        return chunks

    def _apply_overlap_and_create_chunks(
        self, raw_chunks: list[str], original_text: str
    ) -> list[TextChunk]:
        """Apply overlap between chunks and create TextChunk objects with metadata."""
        if not raw_chunks:
            return []

        if len(raw_chunks) == 1:
            return [self._create_single_chunk(raw_chunks[0])]

        result_chunks: list[TextChunk] = []
        current_pos = 0

        for i, chunk_text in enumerate(raw_chunks):
            chunk_start = original_text.find(chunk_text, current_pos)
            if chunk_start == -1:
                chunk_start = current_pos

            chunk_end = chunk_start + len(chunk_text)

            overlap_start = None
            overlap_end = None

            if i > 0:
                overlap_text = self._find_overlap_with_previous(raw_chunks[i - 1], chunk_text)
                if overlap_text:
                    overlap_start = chunk_start
                    overlap_end = chunk_start + len(overlap_text)

            if i < len(raw_chunks) - 1:
                overlap_text_next = self._find_overlap_with_next(chunk_text, raw_chunks[i + 1])
                if overlap_text_next:
                    overlap_end = chunk_end

            boundary_type = self._determine_boundary_type(chunk_text)

            result_chunks.append(
                TextChunk(
                    chunk_index=i,
                    total_chunks=len(raw_chunks),
                    content=chunk_text,
                    token_count=self.count_tokens(chunk_text),
                    start_char=chunk_start,
                    end_char=chunk_end,
                    overlap_start_char=overlap_start,
                    overlap_end_char=overlap_end,
                    boundary_type=boundary_type,
                )
            )

            current_pos = chunk_end

        return result_chunks

    def _find_overlap_with_previous(self, prev_chunk: str, current_chunk: str) -> str:
        """Find overlapping text between previous and current chunk."""
        target_chars = int(len(prev_chunk) * 0.1)

        overlap_candidates = [
            prev_chunk.split("\n\n")[-1] if "\n\n" in prev_chunk else "",
            ". ".join(prev_chunk.split(". ")[-2:]) if ". " in prev_chunk else "",
            " ".join(prev_chunk.split()[-10:]) if " " in prev_chunk else "",
            prev_chunk[-target_chars:] if target_chars > 0 else "",
        ]

        for candidate in overlap_candidates:
            if candidate and candidate in current_chunk:
                candidate_tokens = self.count_tokens(candidate)
                if candidate_tokens <= self._overlap_tokens:
                    return candidate

        return ""

    def _find_overlap_with_next(self, current_chunk: str, next_chunk: str) -> str:
        """Find overlapping text between current and next chunk."""
        target_chars = int(len(current_chunk) * 0.1)

        overlap_candidates = [
            current_chunk.split("\n\n")[-1] if "\n\n" in current_chunk else "",
            ". ".join(current_chunk.split(". ")[-2:]) if ". " in current_chunk else "",
            " ".join(current_chunk.split()[-10:]) if " " in current_chunk else "",
            current_chunk[-target_chars:] if target_chars > 0 else "",
        ]

        for candidate in overlap_candidates:
            if candidate and next_chunk.startswith(candidate):
                candidate_tokens = self.count_tokens(candidate)
                if candidate_tokens <= self._overlap_tokens:
                    return candidate

        return ""

    def _determine_boundary_type(self, chunk_text: str) -> str:
        """Determine the type of boundary used for chunking."""
        if "\n\n" in chunk_text:
            return "paragraph"
        if any(chunk_text.rstrip().endswith(p) for p in [".", "!", "?"]):
            return "sentence"
        if " " in chunk_text:
            return "word"
        return "token"
