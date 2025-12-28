"""Tests for LocalWhisperAudioTranscriber functionality."""

import os
from unittest.mock import patch

import pytest

from blockether_foundation.asr import LocalWhisperAudioTranscriber

# Constants for magic values
DEFAULT_BATCH_SIZE = 8
CUSTOM_BATCH_SIZE = 16
ENV_BATCH_SIZE = 32
DEFAULT_MIN_SILENCE_MS = 500
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_SPEECH_PAD_MS = 200
CUSTOM_MIN_SILENCE_MS = 1000
CUSTOM_VAD_THRESHOLD = 0.3
CUSTOM_SPEECH_PAD_MS = 100
DEFAULT_BEAM_SIZE = 5


class TestLocalWhisperAudioTranscriber:
    """Test LocalWhisperAudioTranscriber behavior."""

    whisper_patcher: object = None
    mock_whisper: object = None

    def setup_method(self):
        """Reset state before each test."""
        # Mock WhisperModel to avoid loading actual models
        self.whisper_patcher = patch("blockether_foundation.asr.local_whisper.WhisperModel")
        self.mock_whisper = self.whisper_patcher.start()

    def teardown_method(self):
        """Clean up patches."""
        self.whisper_patcher.stop()  # type: ignore

    @pytest.mark.unit
    def test_default_model_is_turbo(self) -> None:
        """Test that default model is turbo for quality."""
        transcriber = LocalWhisperAudioTranscriber(model_id="tiny")
        assert transcriber.model_id == "tiny"

        transcriber2 = LocalWhisperAudioTranscriber()
        assert transcriber2.model_id == "turbo"

    @pytest.mark.unit
    def test_quality_settings(self) -> None:
        """Test that quality settings are enforced."""
        transcriber = LocalWhisperAudioTranscriber(model_id="turbo")

        # Default batch_size should be 8 for quality
        assert transcriber.batch_size == DEFAULT_BATCH_SIZE

        # Model should be turbo by default
        assert transcriber.model_id == "turbo"

        # VAD should be enabled
        assert transcriber.vad_parameters is not None
        assert "threshold" in transcriber.vad_parameters

    @pytest.mark.unit
    def test_batch_size_configuration(self) -> None:
        """Test batch_size configuration."""
        # Test explicit
        transcriber = LocalWhisperAudioTranscriber(model_id="tiny", batch_size=CUSTOM_BATCH_SIZE)
        assert transcriber.batch_size == CUSTOM_BATCH_SIZE

    @pytest.mark.unit
    def test_device_configuration(self) -> None:
        """Test device configuration."""
        # Test explicit cpu
        transcriber = LocalWhisperAudioTranscriber(device="cpu")
        assert transcriber.device == "cpu"

        # Test explicit cuda
        transcriber = LocalWhisperAudioTranscriber(device="cuda")
        assert transcriber.device == "cuda"

        # Test auto-selection (without CUDA env var)
        with patch.dict(os.environ, {}, clear=True):
            transcriber = LocalWhisperAudioTranscriber()
            assert transcriber.device == "cpu"

        # Test auto-selection (with CUDA env var)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            transcriber = LocalWhisperAudioTranscriber()
            assert transcriber.device == "cuda"

    @pytest.mark.unit
    def test_download_root_configuration(self) -> None:
        """Test download_root configuration."""
        transcriber = LocalWhisperAudioTranscriber(download_root="/tmp/models")
        assert transcriber.download_root == "/tmp/models"

        # Default download_root
        transcriber = LocalWhisperAudioTranscriber()
        assert transcriber.download_root == "./models"

    @pytest.mark.unit
    def test_vad_parameters_default(self) -> None:
        """Test default VAD parameters."""
        transcriber = LocalWhisperAudioTranscriber()

        assert transcriber.vad_parameters is not None
        assert transcriber.vad_parameters["min_silence_duration_ms"] == DEFAULT_MIN_SILENCE_MS
        assert transcriber.vad_parameters["threshold"] == DEFAULT_VAD_THRESHOLD
        assert transcriber.vad_parameters["speech_pad_ms"] == DEFAULT_SPEECH_PAD_MS

    @pytest.mark.unit
    def test_vad_parameters_custom(self) -> None:
        """Test custom VAD parameters."""
        custom_vad = {
            "min_silence_duration_ms": CUSTOM_MIN_SILENCE_MS,
            "threshold": CUSTOM_VAD_THRESHOLD,
            "speech_pad_ms": CUSTOM_SPEECH_PAD_MS,
        }
        transcriber = LocalWhisperAudioTranscriber(vad_parameters=custom_vad)

        assert transcriber.vad_parameters["min_silence_duration_ms"] == CUSTOM_MIN_SILENCE_MS
        assert transcriber.vad_parameters["threshold"] == CUSTOM_VAD_THRESHOLD
        assert transcriber.vad_parameters["speech_pad_ms"] == CUSTOM_SPEECH_PAD_MS

    @pytest.mark.unit
    def test_beam_size_configuration(self) -> None:
        """Test beam_size configuration."""
        transcriber = LocalWhisperAudioTranscriber(beam_size=DEFAULT_BEAM_SIZE)
        assert transcriber.beam_size == DEFAULT_BEAM_SIZE

        transcriber = LocalWhisperAudioTranscriber(beam_size=1)
        assert transcriber.beam_size == 1

    @pytest.mark.unit
    async def test_transcribe_empty_audio(self) -> None:
        """Test transcription with empty audio data."""
        transcriber = LocalWhisperAudioTranscriber(model_id="tiny")

        # Empty bytes should return None
        result = await transcriber.transcribe(b"")
        assert result is None

    @pytest.mark.unit
    def test_model_initially_not_loaded(self) -> None:
        """Test that model is initially not loaded."""
        transcriber = LocalWhisperAudioTranscriber(model_id="tiny")
        assert transcriber._model is None

    @pytest.mark.unit
    def test_load_model(self) -> None:
        """Test load_model method."""
        transcriber = LocalWhisperAudioTranscriber(model_id="tiny")

        model = transcriber.load_model()
        assert model is not None

        # Calling again should return the same model
        model2 = transcriber.load_model()
        assert model is model2

    @pytest.mark.unit
    def test_unload_model(self) -> None:
        """Test unload_model method."""
        transcriber = LocalWhisperAudioTranscriber(model_id="tiny")

        # Load model first
        transcriber.load_model()
        assert transcriber._model is not None

        # Unload model
        transcriber.unload_model()
        assert transcriber._model is None
