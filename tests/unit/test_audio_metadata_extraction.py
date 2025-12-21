"""Tests for audio metadata extraction using PyAV."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import pytest

from blockether_foundation.agents.transcriber import extract_audio_creation_date


class TestAudioMetadataExtraction:
    """Test suite for audio metadata extraction functionality."""

    def test_extract_creation_date_from_metadata(self, tmp_path: Path) -> None:
        """Test extracting creation date from audio file metadata."""
        # Create a temporary audio file with metadata
        audio_file = tmp_path / "test_audio.m4a"

        # Create a simple audio file with metadata using PyAV
        container = av.open(str(audio_file), "w", format="mp4")
        container.metadata["creation_time"] = "2025-05-14T14:44:16.000000Z"

        # Add a simple audio stream with pcm_s16le codec (no encoding needed)
        stream = container.add_stream("pcm_s16le", rate=44100, layout="mono")

        # Write some silent audio frames
        for i in range(10):
            frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
            frame.rate = 44100
            frame.pts = i * 1024
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush the stream
        for packet in stream.encode():
            container.mux(packet)

        container.close()

        # Test extraction
        result = extract_audio_creation_date(str(audio_file))

        assert result == "14-05-2025"

    def test_extract_creation_date_no_metadata(self, tmp_path: Path) -> None:
        """Test that None is returned when no creation_time metadata exists."""
        # Create a temporary audio file without creation_time metadata
        audio_file = tmp_path / "test_audio_no_meta.m4a"

        container = av.open(str(audio_file), "w", format="mp4")

        # Add a simple audio stream without metadata
        stream = container.add_stream("pcm_s16le", rate=44100, layout="mono")

        # Write some silent audio frames
        for i in range(10):
            frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
            frame.rate = 44100
            frame.pts = i * 1024
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()

        # Test extraction
        result = extract_audio_creation_date(str(audio_file))

        assert result is None

    def test_extract_creation_date_with_mock(self) -> None:
        """Test metadata extraction using mock to verify PyAV integration."""
        mock_container = MagicMock()
        mock_container.__enter__.return_value = mock_container
        mock_container.__exit__.return_value = None
        mock_container.metadata = {"creation_time": "2024-12-20T10:30:00.000000Z"}

        with patch("av.open", return_value=mock_container):
            result = extract_audio_creation_date("fake_file.m4a")

        assert result == "20-12-2024"

    def test_extract_creation_date_invalid_file(self) -> None:
        """Test that exception is raised for invalid file."""
        with pytest.raises((OSError, av.error.FileNotFoundError)):
            extract_audio_creation_date("/nonexistent/file.m4a")

    def test_extract_creation_date_different_date_formats(self, tmp_path: Path) -> None:
        """Test extraction with different ISO date formats."""
        test_cases = [
            ("2023-01-15T08:30:45.123456Z", "15-01-2023"),
            ("2023-12-31T23:59:59.999999Z", "31-12-2023"),
            ("2025-06-01T00:00:00.000000Z", "01-06-2025"),
        ]

        for iso_date, expected_result in test_cases:
            audio_file = tmp_path / f"test_{iso_date.replace(':', '_')}.m4a"

            container = av.open(str(audio_file), "w", format="mp4")
            container.metadata["creation_time"] = iso_date

            stream = container.add_stream("pcm_s16le", rate=44100, layout="mono")

            for i in range(10):
                frame = av.AudioFrame(format="s16", layout="mono", samples=1024)
                frame.rate = 44100
                frame.pts = i * 1024
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

            container.close()

            result = extract_audio_creation_date(str(audio_file))
            assert result == expected_result
