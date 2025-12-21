"""Unit tests for audio transcription hook with XML formatting."""

from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from agno.agent import Agent
from agno.media import Audio
from agno.run.agent import RunInput

from blockether_foundation.agents.hooks.audio import AudioHooksConfig
from blockether_foundation.audio.transcription import (
    TranscriptionResult,
    TranscriptionSegment,
    Word,
)


class TestAudioHookXMLFormatting:
    """Test cases for audio hook XML formatting."""

    @pytest.fixture
    def mock_transcriber(self):
        """Create a mock transcriber."""
        with patch('blockether_foundation.agents.hooks.audio.AudioTranscriber.get_instance') as mock:
            transcriber = MagicMock()
            mock.return_value = transcriber
            yield transcriber

    @pytest.fixture
    def sample_transcription_result(self):
        """Create a sample transcription result for testing."""
        return TranscriptionResult(
            segments=[
                TranscriptionSegment(
                    start=0.0,
                    end=2.5,
                    text="Hello world, this is a test.",
                    words=[
                        Word(word="Hello", start=0.0, end=0.5, score=0.98),
                        Word(word="world", start=0.6, end=1.0, score=0.95),
                        Word(word="this", start=1.1, end=1.4, score=0.96),
                        Word(word="is", start=1.5, end=1.7, score=0.94),
                        Word(word="a", start=1.8, end=1.9, score=0.97),
                        Word(word="test", start=2.0, end=2.3, score=0.93),
                        Word(word=".", start=2.3, end=2.5, score=0.99),
                    ],
                    speaker="Speaker A"
                )
            ],
            language="en",
            language_probability=0.97,
            created_at=datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
        )

    @pytest.mark.asyncio
    async def test_audio_hook_formats_as_xml(
        self, mock_transcriber, sample_transcription_result
    ):
        """Test that the audio hook formats transcription as XML."""
        # Configure mock to return sample transcription
        mock_transcriber.transcribe = AsyncMock(return_value=sample_transcription_result)

        # Create hook
        config = AudioHooksConfig(effort=0.5)
        hook = config.pre_hook()

        # Create audio with test data (content only)
        test_audio = Audio(
            content=b"fake audio data",
            id="test-audio-123"
        )
        run_input = RunInput(
            input_content="Process this audio",
            audios=[test_audio]
        )

        # Note: No file metadata since using content directly

        # Create mock session
        from blockether_foundation.context_manager import AgentSession
        session = AgentSession(session_id="test-session")

        # Execute hook
        await hook(Agent(), run_input, session, "test-user", True)

        # Verify audio was processed
        assert len(run_input.audios) == 0, "Audio should be cleared after processing"

        # Get the content
        content = run_input.input_content_string()

        # Verify XML formatting (no file metadata since using content)
        assert "--- Audio Transcription | Source: test-audio-123" in content
        assert "<transcription>" in content
        assert 'language="en"' in content
        assert 'language_probability="0.970"' in content
        assert 'total_duration="2.500"' in content
        assert 'segment_count="1"' in content
        assert 'word_count="7"' in content
        assert '<segment index="1" start="0.000" end="2.500"' in content
        assert 'speaker="Speaker A"' in content
        # Text is now directly in the segment tag (no separate <text> tag)
        assert 'Hello world, this is a test.</segment>' in content
        # Note: Word timestamps are NOT included by default for performance
        assert '<word' not in content

    @pytest.mark.asyncio
    async def test_audio_hook_fallback_to_plain_text(
        self, mock_transcriber
    ):
        """Test that the audio hook falls back to plain text when no TranscriptionResult."""
        # Configure mock to return None (transcription failed)
        mock_transcriber.transcribe = AsyncMock(return_value=None)

        # Create hook
        config = AudioHooksConfig()
        hook = config.pre_hook()

        # Create audio
        test_audio = Audio(content=b"fake audio data")
        run_input = RunInput(input_content="Process this audio", audios=[test_audio])

        # Create mock session
        from blockether_foundation.context_manager import AgentSession
        session = AgentSession(session_id="test-session")

        # Execute hook
        await hook(Agent(), run_input, session, "test-user", True)

        # Verify audio was processed
        assert len(run_input.audios) == 0

        # Get the content - should not have XML since transcription failed
        content = run_input.input_content_string()
        assert "<transcription>" not in content
        # Should not have any transcript since transcription returned None

    @pytest.mark.asyncio
    async def test_audio_hook_no_transcript_when_none_returned(
        self, mock_transcriber
    ):
        """Test that no transcript is added when transcription returns None."""
        # Configure mock to return None
        mock_transcriber.transcribe = AsyncMock(return_value=None)

        # Create hook
        config = AudioHooksConfig()
        hook = config.pre_hook()

        # Create audio
        test_audio = Audio(content=b"fake audio data")
        run_input = RunInput(input_content="Process this audio", audios=[test_audio])

        # Create mock session
        from blockether_foundation.context_manager import AgentSession
        session = AgentSession(session_id="test-session")

        # Execute hook
        await hook(Agent(), run_input, session, "test-user", True)

        # Should not have any transcript since transcription returned None
        content = run_input.input_content_string()
        assert content == "Process this audio"  # Content should remain unchanged

    @pytest.mark.asyncio
    async def test_audio_hook_with_filepath(
        self, mock_transcriber, sample_transcription_result
    ):
        """Test audio hook with filepath audio source."""
        mock_transcriber.transcribe = AsyncMock(return_value=sample_transcription_result)

        # Create hook
        config = AudioHooksConfig()
        hook = config.pre_hook()

        # Create audio with filepath (no ID so it will use filepath)
        test_audio = Audio(filepath="/path/to/audio.wav", id=None)
        run_input = RunInput(input_content="Process this audio", audios=[test_audio])

        # Mock file reading
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b"fake audio content"

            # Create mock session
            from blockether_foundation.context_manager import AgentSession
            session = AgentSession(session_id="test-session")

            # Execute hook
            await hook(Agent(), run_input, session, "test-user", True)

            # Verify some identifier is shown in header (ID or filepath)
            content = run_input.input_content_string()
            assert "--- Audio Transcription |" in content
            assert "<transcription>" in content
            assert "language=\"en\"" in content

    @pytest.mark.asyncio
    async def test_audio_hook_max_segments_limit(
        self, mock_transcriber
    ):
        """Test that audio hook respects max_segments limit."""
        # Create transcription with many segments
        segments = []
        for i in range(100):  # 100 segments
            segments.append(TranscriptionSegment(
                start=float(i),
                end=float(i + 1),
                text=f"Segment {i + 1}",
                words=[Word(word=f"Segment", start=float(i), end=float(i + 0.5), score=0.95)],
                speaker=f"Speaker {chr(65 + (i % 3))}"  # Rotate between A, B, C
            ))

        large_transcription = TranscriptionResult(
            segments=segments,
            language="en",
            language_probability=0.98
        )

        mock_transcriber.transcribe = AsyncMock(return_value=large_transcription)

        # Create hook
        config = AudioHooksConfig()
        hook = config.pre_hook()

        # Create audio
        test_audio = Audio(content=b"fake audio data")
        run_input = RunInput(input_content="Process this audio", audios=[test_audio])

        # Create mock session
        from blockether_foundation.context_manager import AgentSession
        session = AgentSession(session_id="test-session")

        # Execute hook
        await hook(Agent(), run_input, session, "test-user", True)

        # Get the content
        content = run_input.input_content_string()

        # Should have all 100 segments (no limit in current implementation)
        segment_count = content.count('<segment index=')
        assert segment_count == 100, f"Should have 100 segments, got {segment_count}"

        # Metadata should show the correct total
        assert 'segment_count="100"' in content