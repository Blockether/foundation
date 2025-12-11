import os
from unittest.mock import patch

import pytest

from blockether_foundation.audio.transcription import AudioTranscriber

# Path to sample.ogg in test resources
SAMPLE_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "../test-resources/sample.ogg")


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
def test_invalid_model_fallback() -> None:
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
        transcriber = AudioTranscriber.get_instance()

        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        text = await transcriber.transcribe(audio_data)

        # Hardcoded expectation based on faster-whisper tiny model output
        assert text == "1, 2, 3, 4, 5."


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
    assert transcriber._model is None

    # unload_model should work even when no model is loaded (but will log a warning)
    transcriber.unload_model()
    assert transcriber._model is None


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
    from unittest.mock import Mock
    mock_model = Mock()

    # Mock the transcribe method to return proper values
    mock_segment = Mock()
    mock_segment.text = "test"
    mock_info = Mock()
    mock_info.language = "en"
    mock_info.language_probability = 1.0
    mock_model.transcribe.return_value = (iter([mock_segment]), mock_info)

    transcriber._model = mock_model

    # Call _run_whisper_inference - model should stay loaded
    # Don't patch load_model since we want to use the already loaded mock model
    result = transcriber._run_whisper_inference(b"fake audio", beam_size=1)

    # Model should still be loaded after transcription
    assert transcriber._model is not None
    assert transcriber._model is mock_model


@pytest.mark.unit
def test_load_model_fallback() -> None:
    """Test model loading fallback when compute_type fails."""
    transcriber = AudioTranscriber(model_id="tiny", device="cpu")

    # Mock WhisperModel to raise exception on first call, succeed on second
    from unittest.mock import patch, Mock
    with patch('blockether_foundation.audio.transcription.WhisperModel') as mock_model_class:
        mock_model = Mock()
        # First call raises exception, second succeeds
        mock_model_class.side_effect = [
            ValueError("Unsupported compute type"),
            mock_model
        ]

        # Load should handle the exception and fallback
        result = transcriber.load_model()
        assert result is mock_model
        assert mock_model_class.call_count == 2
        # First call should have compute_type="int8"
        assert mock_model_class.call_args_list[0][1]['compute_type'] == 'int8'
        # Second call should not have compute_type
        assert 'compute_type' not in mock_model_class.call_args_list[1][1]


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


