import os
from typing import NamedTuple
from unittest.mock import Mock, patch

import numpy as np
import pytest

from blockether_foundation.audio.transcription import AudioTranscriber

# Path to sample.ogg in test resources
SAMPLE_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "../test-resources/sample.ogg")


class Word(NamedTuple):
    word: str
    start: float
    end: float
    probability: float


class Segment(NamedTuple):
    text: str
    start: float
    end: float
    words: list[Word]


class Info(NamedTuple):
    language: str
    language_probability: float


@pytest.mark.unit
def test_singleton_instance() -> None:
    # Reset singleton
    AudioTranscriber._instance = None  # type: ignore

    instance1 = AudioTranscriber.get_instance()
    instance2 = AudioTranscriber.get_instance()

    assert instance1 is instance2
    assert isinstance(instance1, AudioTranscriber)


@patch.dict(os.environ, {"BLOCKETHER_WHISPER_MODEL": "tiny"})
@pytest.mark.unit
def test_singleton_configuration() -> None:
    # Reset singleton
    AudioTranscriber._instance = None  # type: ignore

    instance = AudioTranscriber.get_instance()
    assert instance.model_id == "tiny"


@patch.dict(os.environ, {"BLOCKETHER_WHISPER_MODEL": "invalid_model"})
@pytest.mark.unit
def test_invalid_model_handling() -> None:
    # Reset singleton
    AudioTranscriber._instance = None  # type: ignore

    instance = AudioTranscriber.get_instance()
    # Should fallback to "turbo" as per implementation
    assert instance.model_id == "turbo"


@pytest.mark.skipif(
    not os.path.exists(SAMPLE_AUDIO_PATH),
    reason=f"Sample audio file not found at {SAMPLE_AUDIO_PATH}",
)
@pytest.mark.asyncio
@pytest.mark.unit
async def test_transcription_sample_ogg():
    # Reset singleton to ensure clean state
    AudioTranscriber._instance = None  # type: ignore

    # Use a smaller model for testing to be faster
    with patch.dict(os.environ, {"BLOCKETHER_WHISPER_MODEL": "tiny"}):
        with patch("blockether_foundation.audio.transcription.WhisperModel") as MockModel:
            mock_model_instance = MockModel.return_value
            # Mock the transcribe return value

            # Mock segments with words
            mock_words = [
                Word("1,", 0.0, 0.5, 0.99),
                Word("2,", 0.5, 1.0, 0.99),
                Word("3,", 1.0, 1.5, 0.99),
                Word("4,", 1.5, 2.0, 0.99),
                Word("5.", 2.0, 2.5, 0.99),
            ]
            mock_segment = Segment("1, 2, 3, 4, 5.", 0.0, 2.5, mock_words)
            mock_model_instance.transcribe.return_value = (
                iter([mock_segment]),
                Info("en", 0.99),
            )

            transcriber = AudioTranscriber.get_instance()

            with open(SAMPLE_AUDIO_PATH, "rb") as f:
                audio_data = f.read()

            result = await transcriber.transcribe(audio_data)

            # Verify result
            assert result is not None
            assert result.text == "1, 2, 3, 4, 5."


@patch.dict(os.environ, {"BLOCKETHER_WHISPER_MODEL": "distil-large-v3"})
@pytest.mark.unit
def test_distil_model_support() -> None:
    """Test that distil-large-v3 model is supported."""
    # Reset singleton
    AudioTranscriber._instance = None  # type: ignore

    instance = AudioTranscriber.get_instance()
    assert instance.model_id == "distil-large-v3"


@pytest.mark.unit
def test_unload_model() -> None:
    """Test model unloading functionality."""
    transcriber = AudioTranscriber(model_id="tiny")

    # Initially no model is loaded
    assert getattr(transcriber, "_model", None) is None

    # unload_model should work even when no model is loaded (but will log a warning)
    transcriber.unload_model()
    assert getattr(transcriber, "_model", None) is None


@pytest.mark.unit
async def test_transcribe_empty_audio() -> None:
    """Test transcription with empty audio data."""
    transcriber = AudioTranscriber(model_id="tiny")

    # Empty bytes should return None
    result = await transcriber.transcribe(b"")
    assert result is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_transcribe_error_handling() -> None:
    """Test transcription error handling with invalid audio data."""
    transcriber = AudioTranscriber(model_id="tiny")

    # Invalid audio data that should cause an error
    invalid_audio = b"This is not valid audio data at all!"
    result = await transcriber.transcribe(invalid_audio)

    # Should return None on error
    assert result is None


@pytest.mark.unit
def test_model_already_loaded() -> None:
    """Test that load_model returns existing model if already loaded."""
    transcriber = AudioTranscriber(model_id="tiny")

    # First load should create the model
    model1 = transcriber.load_model()
    assert model1 is not None

    # Second load should return the same model
    model2 = transcriber.load_model()
    assert model1 is model2


@pytest.mark.unit
def test_model_never_unloads() -> None:
    """Test that the model never unloads after transcription for maximum performance."""
    transcriber = AudioTranscriber(model_id="tiny")

    # Create a mock model
    mock_model = Mock()

    # Mock the transcribe method to return proper values
    mock_words = [Word("test", 0.0, 1.0, 0.99)]
    mock_segment = Segment("test", 0.0, 1.0, mock_words)
    mock_info = Info("en", 1.0)
    mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

    # Use setattr to set the private attribute
    transcriber._model = mock_model

    # Mock av.open and audio processing
    with patch("blockether_foundation.audio.transcription.av.open"):
        with patch("blockether_foundation.audio.transcription.np.concatenate") as mock_concat:
            mock_concat.return_value = np.array([0.0] * 16000, dtype=np.float32)

            # Call _run_whisper_inference - model should stay loaded
            # Use getattr to access the private method
            run_inference = transcriber._run_whisper_inference
            result = run_inference(b"fake audio", beam_size=1)

            # Model should still be loaded after transcription
            assert getattr(transcriber, "_model", None) is not None
            assert getattr(transcriber, "_model", None) is mock_model
            assert result.text == "test"


@pytest.mark.unit
def test_device_auto_selection() -> None:
    """Test automatic device selection."""
    # Test with CUDA available
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
        transcriber = AudioTranscriber()
        assert transcriber.device == "cuda"

    # Test without CUDA (default)
    with patch.dict(os.environ, {}, clear=True):
        transcriber = AudioTranscriber()
        assert transcriber.device == "cpu"

    # Test explicit device override
    transcriber = AudioTranscriber(device="cpu")
    assert transcriber.device == "cpu"
