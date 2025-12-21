"""Tests for AudioTranscriber functionality."""

import os
from unittest.mock import patch

import pytest

from blockether_foundation.audio.transcription import AudioTranscriber

# Constants for magic values
DEFAULT_BATCH_SIZE = 8
DEFAULT_BATCH_SIZE_CUDA = 8
CUSTOM_BATCH_SIZE = 16
ENV_BATCH_SIZE = 32


class TestAudioTranscriberSingleton:
    """Test AudioTranscriber singleton behavior."""

    whisper_patcher: object = None
    mock_whisper: object = None

    def setup_method(self):
        """Reset singleton before each test."""
        AudioTranscriber._instance = None  # pyright: ignore[reportPrivateUsage]
        # Mock WhisperModel to avoid loading actual models
        self.whisper_patcher = patch("blockether_foundation.audio.transcription.WhisperModel")
        self.mock_whisper = self.whisper_patcher.start()

    def teardown_method(self):
        """Clean up patches."""
        self.whisper_patcher.stop()  # type: ignore
        AudioTranscriber._instance = None  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.unit
    def test_singleton_default_behavior(self) -> None:
        """Test that the first instance becomes the singleton by default."""
        assert AudioTranscriber._instance is None  # pyright: ignore[reportPrivateUsage]

        transcriber = AudioTranscriber(model_id="tiny")

        assert AudioTranscriber._instance is transcriber  # pyright: ignore[reportPrivateUsage]
        assert AudioTranscriber.get_instance() is transcriber

    @pytest.mark.unit
    def test_multiple_instances_no_auto_assign(self) -> None:
        """Test that auto_assign_instance=False allows creating separate instances."""
        assert AudioTranscriber._instance is None  # pyright: ignore[reportPrivateUsage]

        # Create the main singleton instance
        transcriber1 = AudioTranscriber(model_id="tiny", auto_assign_instance=True)
        assert AudioTranscriber._instance is transcriber1  # pyright: ignore[reportPrivateUsage]

        # Create a second instance that shouldn't affect the singleton
        transcriber2 = AudioTranscriber(model_id="base", auto_assign_instance=False)

        # Verify singleton is still the first one
        assert AudioTranscriber._instance is transcriber1  # pyright: ignore[reportPrivateUsage]
        assert AudioTranscriber.get_instance() is transcriber1

        # Verify the second instance is distinct
        assert transcriber2 is not transcriber1
        assert transcriber2.model_id == "base"
        assert transcriber1.model_id == "tiny"

    @pytest.mark.unit
    def test_get_instance_creates_singleton(self) -> None:
        """Test that get_instance creates a singleton if none exists."""
        assert AudioTranscriber._instance is None  # pyright: ignore[reportPrivateUsage]

        transcriber = AudioTranscriber.get_instance()

        assert transcriber is not None
        assert AudioTranscriber._instance is transcriber  # pyright: ignore[reportPrivateUsage]

        # Subsequent calls return the same instance
        assert AudioTranscriber.get_instance() is transcriber

    @pytest.mark.unit
    def test_auto_assign_false_does_not_set_singleton(self) -> None:
        """Test that the first instance doesn't become singleton if auto_assign_instance=False."""
        assert AudioTranscriber._instance is None  # pyright: ignore[reportPrivateUsage]

        transcriber = AudioTranscriber(model_id="tiny", auto_assign_instance=False)

        assert AudioTranscriber._instance is None  # pyright: ignore[reportPrivateUsage]

        # get_instance should create a NEW instance since _instance is still None
        singleton = AudioTranscriber.get_instance()

        assert singleton is not transcriber
        assert AudioTranscriber._instance is singleton  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.unit
    def test_batch_size_configuration(self) -> None:
        """Test batch_size configuration."""
        # Test default should be 8 for quality
        transcriber = AudioTranscriber(model_id="tiny", device="cpu", auto_assign_instance=False)
        assert transcriber.batch_size == DEFAULT_BATCH_SIZE

        transcriber = AudioTranscriber(model_id="tiny", device="cuda", auto_assign_instance=False)
        assert transcriber.batch_size == DEFAULT_BATCH_SIZE_CUDA

        # Test explicit
        transcriber = AudioTranscriber(
            model_id="tiny", batch_size=CUSTOM_BATCH_SIZE, auto_assign_instance=False
        )
        assert transcriber.batch_size == CUSTOM_BATCH_SIZE

        # Test env var
        with patch.dict(os.environ, {"BLOCKETHER_WHISPER_BATCH_SIZE": str(ENV_BATCH_SIZE)}):
            AudioTranscriber._instance = None  # pyright: ignore[reportPrivateUsage]
            transcriber = AudioTranscriber.get_instance()
            assert transcriber.batch_size == ENV_BATCH_SIZE

    @pytest.mark.unit
    def test_default_model_is_turbo(self) -> None:
        """Test that default model is turbo for quality."""
        AudioTranscriber._instance = None  # pyright: ignore[reportPrivateUsage]
        transcriber = AudioTranscriber()
        assert transcriber.model_id == "turbo"

    @pytest.mark.unit
    def test_quality_settings(self) -> None:
        """Test that quality settings are enforced."""
        transcriber = AudioTranscriber(model_id="turbo")

        # Default batch_size should be 8 for quality
        assert transcriber.batch_size == DEFAULT_BATCH_SIZE

        # Model should be turbo by default
        assert transcriber.model_id == "turbo"

        # VAD should be enabled
        assert transcriber.vad_parameters is not None
        assert "threshold" in transcriber.vad_parameters
