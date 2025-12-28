"""Integration tests for LocalCoquiTTS and VibeVoiceTTS functionality.

These tests use real TTS models and are marked with @pytest.mark.integration.
They require actual model downloads and are slower than unit tests.

Run with: pytest tests/integration/test_tts_integration.py -m integration
"""

import struct

import pytest

SPANISH_VITS_MODEL = "tts_models/es/css10/vits"
POLISH_VITS_MODEL = "tts_models/pl/mai_female/vits"
GERMAN_VITS_MODEL = "tts_models/de/thorsten/vits"
PORTUGUESE_VITS_MODEL = "tts_models/pt/cv/vits"

# VibeVoice test constants
VIBEVOICE_DEFAULT_SPEAKER = "Carter"
VIBEVOICE_MODEL = "microsoft/VibeVoice-Realtime-0.5B"


@pytest.mark.integration
def test_vibevoice_model_loading():
    import blockether_foundation.tts.local_vibevoice as vibevoice_module


    tts = vibevoice_module.VibeVoiceTTS(model_name=VIBEVOICE_MODEL)

    result = tts.synthesize("Hello world, this is a test.")

    assert tts._model is not None, "Model should be loaded after synthesis"

    assert result is not None, "Should produce audio result"

    validate_wav_format(result.audio)


@pytest.mark.integration
def test_vibevoice_english_synthesis():
    import blockether_foundation.tts.local_vibevoice as vibevoice_module

    from blockether_foundation.tts.common import SynthesisResult

    tts = vibevoice_module.VibeVoiceTTS(
        model_name=VIBEVOICE_MODEL, speaker_name=VIBEVOICE_DEFAULT_SPEAKER
    )

    text = "This is a test of the VibeVoice text to speech synthesis."
    result = tts.synthesize(text, voice=VIBEVOICE_DEFAULT_SPEAKER)

    assert result is not None, "Should produce audio result"
    assert isinstance(result, SynthesisResult), "Should return SynthesisResult"

    validate_wav_format(result.audio)

    duration = calculate_wav_duration(result.audio)
    assert duration >= MIN_AUDIO_DURATION, f"Audio duration {duration}s too short for '{text}'"


@pytest.mark.integration
def test_vibevoice_empty_text():
    import blockether_foundation.tts.local_vibevoice as vibevoice_module

    tts = vibevoice_module.VibeVoiceTTS(
        model_name=VIBEVOICE_MODEL, speaker_name=VIBEVOICE_DEFAULT_SPEAKER
    )

    result = tts.synthesize("")

    assert result is None, "Empty text should return None"
    assert tts._model is None, "Model should not be loaded for empty text"


@pytest.mark.integration
def test_vibevoice_whitespace_only():
    import blockether_foundation.tts.local_vibevoice as vibevoice_module

    tts = vibevoice_module.VibeVoiceTTS(
        model_name=VIBEVOICE_MODEL, speaker_name=VIBEVOICE_DEFAULT_SPEAKER
    )

    result = tts.synthesize("   \n\t   ")

    assert result is None, "Whitespace-only text should return None"
    assert tts._model is None, "Model should not be loaded for whitespace"


MIN_AUDIO_DURATION = 0.1
WAV_HEADER_SIZE = 44


def validate_wav_format(audio_bytes: bytes) -> None:
    """Validate that audio_bytes is a valid WAV file.

    Args:
        audio_bytes: The audio data as bytes.

    Raises:
        AssertionError: If the WAV format is invalid.
    """
    assert len(audio_bytes) >= WAV_HEADER_SIZE, "WAV file too small (must be at least 44 bytes)"
    assert audio_bytes[:4] == b"RIFF", "Invalid WAV: missing RIFF header"
    assert audio_bytes[8:12] == b"WAVE", "Invalid WAV: missing WAVE format"
    assert audio_bytes[12:16] == b"fmt ", "Invalid WAV: missing fmt chunk"
    assert b"data " in audio_bytes, "Invalid WAV: missing data chunk"


def calculate_wav_duration(audio_bytes: bytes) -> float:
    """Calculate the duration of a WAV file in seconds.

    Args:
        audio_bytes: The audio data as bytes.

    Returns:
        Duration in seconds.
    """
    if len(audio_bytes) < 44:
        return 0.0

    data_pos = audio_bytes.find(b"data ")
    if data_pos == -1:
        return 0.0

    byte_rate = struct.unpack("<I", audio_bytes[28:32])[0]

    if byte_rate == 0:
        return 0.0

    data_size = struct.unpack("<I", audio_bytes[data_pos + 4 : data_pos + 8])[0]
    duration = data_size / byte_rate

    return duration


@pytest.mark.integration
def test_real_model_loading():
    import blockether_foundation.tts.local_coqui as tts_module

    tts = tts_module.LocalCoquiTTS(model_name=SPANISH_VITS_MODEL)

    result = tts.synthesize("Hola mundo")

    assert tts._tts_model is not None, "Model should be loaded after synthesis"

    assert result is not None, "Should produce audio result"

    validate_wav_format(result.audio)


@pytest.mark.integration
def test_real_english_synthesis():
    import blockether_foundation.tts.local_coqui as tts_module
    from blockether_foundation.tts.common import SynthesisResult

    tts = tts_module.LocalCoquiTTS(model_name=PORTUGUESE_VITS_MODEL)

    text = "Olá mundo, isto é um teste."
    result = tts.synthesize(text)

    assert result is not None, "Should produce audio result"
    assert isinstance(result, SynthesisResult), "Should return SynthesisResult"

    validate_wav_format(result.audio)

    duration = calculate_wav_duration(result.audio)
    assert duration >= MIN_AUDIO_DURATION, f"Audio duration {duration}s too short for '{text}'"


@pytest.mark.integration
def test_real_synthesis_spanish():
    import blockether_foundation.tts.local_coqui as tts_module

    tts = tts_module.LocalCoquiTTS(model_name=SPANISH_VITS_MODEL)

    text = "Hola mundo"
    result = tts.synthesize(text, voice=None, language=None)

    assert result is not None, "Should produce audio result"

    validate_wav_format(result.audio)

    duration = calculate_wav_duration(result.audio)
    assert duration >= MIN_AUDIO_DURATION, f"Audio duration {duration}s too short"


@pytest.mark.integration
def test_real_long_text_synthesis():
    import blockether_foundation.tts.local_coqui as tts_module

    tts = tts_module.LocalCoquiTTS(model_name=GERMAN_VITS_MODEL)

    long_text = "Das ist ein sehr langer Text. " * 10

    result = tts.synthesize(long_text)

    assert result is not None, "Should produce audio result for long text"

    validate_wav_format(result.audio)

    duration = calculate_wav_duration(result.audio)
    assert duration > 1.0, f"Audio duration {duration}s should be longer for long text"


@pytest.mark.integration
def test_real_synthesis_empty_text():
    import blockether_foundation.tts.local_coqui as tts_module

    tts = tts_module.LocalCoquiTTS(model_name=SPANISH_VITS_MODEL)

    result = tts.synthesize("")

    assert result is None, "Empty text should return None"
    assert tts._tts_model is None, "Model should not be loaded for empty text"


@pytest.mark.integration
def test_real_synthesis_whitespace_only():
    import blockether_foundation.tts.local_coqui as tts_module

    tts = tts_module.LocalCoquiTTS(model_name=SPANISH_VITS_MODEL)

    result = tts.synthesize("   \n\t   ")

    assert result is None, "Whitespace-only text should return None"
    assert tts._tts_model is None, "Model should not be loaded for whitespace"


@pytest.mark.integration
def test_pre_download_real_model():
    import tempfile

    import blockether_foundation.tts.local_coqui as tts_module

    with tempfile.TemporaryDirectory() as temp_dir:
        download_root = temp_dir

        tts_module.LocalCoquiTTS.pre_download(
            model_name=SPANISH_VITS_MODEL,
            download_root=download_root,
            device="cpu",
        )

        tts = tts_module.LocalCoquiTTS(model_name=SPANISH_VITS_MODEL)
        result = tts.synthesize("Prueba descarga")

        assert result is not None, "Should produce audio result after pre-download"
        validate_wav_format(result.audio)
        assert calculate_wav_duration(result.audio) >= MIN_AUDIO_DURATION


@pytest.mark.integration
def test_polish_model_synthesis():
    import blockether_foundation.tts.local_coqui as tts_module

    tts = tts_module.LocalCoquiTTS(model_name=POLISH_VITS_MODEL)

    polish_text = "Witaj świecie, to jest test."
    result = tts.synthesize(polish_text)

    assert result is not None, "Should produce audio result for Polish text"

    validate_wav_format(result.audio)

    duration = calculate_wav_duration(result.audio)
    assert duration >= MIN_AUDIO_DURATION, f"Audio duration {duration}s too short for Polish text"


@pytest.mark.integration
def test_multiple_synthesis_calls():
    import blockether_foundation.tts.local_coqui as tts_module

    tts = tts_module.LocalCoquiTTS(model_name=GERMAN_VITS_MODEL)

    result1 = tts.synthesize("Erster Test")
    assert result1 is not None, "First synthesis should produce result"
    validate_wav_format(result1.audio)

    result2 = tts.synthesize("Zweiter Test")
    assert result2 is not None, "Second synthesis should produce result"
    validate_wav_format(result2.audio)

    assert tts._tts_model is not None, "Model should remain loaded"

    assert result1.audio != result2.audio, "Different texts should produce different audio"


@pytest.mark.integration
def test_device_cpu_synthesis():
    import blockether_foundation.tts.local_coqui as tts_module

    tts = tts_module.LocalCoquiTTS(model_name=PORTUGUESE_VITS_MODEL, device="cpu")

    assert tts.device == "cpu", "Device should be CPU"

    result = tts.synthesize("Olá da CPU")

    assert result is not None, "Should produce audio result on CPU"
    validate_wav_format(result.audio)

    duration = calculate_wav_duration(result.audio)
    assert duration >= MIN_AUDIO_DURATION, f"Audio duration {duration}s too short"
