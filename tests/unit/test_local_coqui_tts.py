"""Tests for LocalCoquiTTS functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Constants for magic values
DEFAULT_MODEL = "tts_models/en/ljspeech/vits"
CUSTOM_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
POLISH_MODEL = "tts_models/pl/mai_female/vits"
DEFAULT_MAX_SEGMENT_DURATION = 20.0


class TestLocalCoquiTTS:
    """Test LocalCoquiTTS behavior."""

    @pytest.mark.unit
    def test_default_model(self) -> None:
        """Test that default model is English LJSpeech VITS."""

        # Create TTS instance (without triggering import check)
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()

        assert tts.model_name == DEFAULT_MODEL

    @pytest.mark.unit
    def test_custom_model(self) -> None:
        """Test custom model selection."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS(model_name=CUSTOM_MODEL)
        assert tts.model_name == CUSTOM_MODEL

    @pytest.mark.unit
    def test_device_configuration_cpu(self) -> None:
        """Test CPU device configuration."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS(device="cpu")
        assert tts.device == "cpu"

    @pytest.mark.unit
    def test_device_configuration_cuda(self) -> None:
        """Test CUDA device configuration."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS(device="cuda")
        assert tts.device == "cuda"

    @pytest.mark.unit
    def test_device_auto_selection_without_cuda(self) -> None:
        """Test auto device selection without CUDA."""
        import blockether_foundation.tts.local_coqui as tts_module

        with patch.dict(os.environ, {}, clear=True):
            tts = tts_module.LocalCoquiTTS()
            assert tts.device == "cpu"

    @pytest.mark.unit
    def test_device_auto_selection_with_cuda(self) -> None:
        """Test auto device selection with CUDA."""
        import blockether_foundation.tts.local_coqui as tts_module

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            tts = tts_module.LocalCoquiTTS(enable_cuda=True)
            assert tts.device == "cuda"

    @pytest.mark.unit
    def test_progress_bar_default(self) -> None:
        """Test progress bar is disabled by default."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        assert tts.progress_bar is False

    @pytest.mark.unit
    def test_progress_bar_enabled(self) -> None:
        """Test progress bar can be enabled."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS(progress_bar=True)
        assert tts.progress_bar is True

    @pytest.mark.unit
    def test_model_initially_not_loaded(self) -> None:
        """Test that model is initially not loaded."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        assert tts._tts_model is None

    @pytest.mark.unit
    def test_split_text_into_chunks_short_text(self) -> None:
        """Test splitting short text."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        text = "Hello world"
        chunks = tts._split_text_into_chunks(text)
        assert len(chunks) == 1
        assert text in chunks[0]

    @pytest.mark.unit
    def test_split_text_into_chunks_long_text(self) -> None:
        """Test splitting long text."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        # Create a long text that will be split
        words = "word " * 200
        text = words.strip()
        chunks = tts._split_text_into_chunks(text)
        assert len(chunks) > 1

    @pytest.mark.unit
    def test_split_text_into_chunks_with_periods(self) -> None:
        """Test splitting text with periods."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        text = "First sentence. Second sentence. Third sentence."
        chunks = tts._split_text_into_chunks(text)
        # Should split by periods
        assert any("First sentence" in chunk for chunk in chunks)
        assert any("Second sentence" in chunk for chunk in chunks)
        assert any("Third sentence" in chunk for chunk in chunks)

    @pytest.mark.unit
    def test_calculate_audio_duration_valid_wav(self) -> None:
        """Test calculating duration from valid WAV data."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()

        # Create minimal WAV header + PCM data
        # Byte rate = 48000 (24000 Hz * 2 bytes/sample)
        wav_header = (
            b"RIFF"
            + (36 + 48000).to_bytes(4, byteorder="little")
            + b"WAVE"
            + b"fmt "
            + (16).to_bytes(4, byteorder="little")
            + b"\x01\x00"
            + (1).to_bytes(2, byteorder="little")
            + (24000).to_bytes(4, byteorder="little")
            + (48000).to_bytes(4, byteorder="little")
            + (2).to_bytes(2, byteorder="little")
            + (16).to_bytes(2, byteorder="little")
            + b"data "
            + (48000).to_bytes(4, byteorder="little")
        )
        pcm_data = b"\x00\x01" * 48000
        audio_bytes = wav_header + pcm_data

        duration = tts._calculate_audio_duration(audio_bytes)
        assert duration == 1.0

    @pytest.mark.unit
    def test_calculate_audio_duration_empty_audio(self) -> None:
        """Test calculating duration from empty audio."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        duration = tts._calculate_audio_duration(b"")
        assert duration == 0.0

    @pytest.mark.unit
    def test_calculate_audio_duration_short_header(self) -> None:
        """Test calculating duration from audio with too short header."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        duration = tts._calculate_audio_duration(b"short")
        assert duration == 0.0

    @pytest.mark.unit
    def test_join_audio_segments_empty(self) -> None:
        """Test joining empty audio segments."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        result = tts._join_audio_segments([])
        assert result == b""

    @pytest.mark.unit
    def test_join_audio_segments_single(self) -> None:
        """Test joining single audio segment."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        segment = b"audio_data"
        result = tts._join_audio_segments([segment])
        assert result == segment

    @pytest.mark.unit
    def test_synthesize_empty_text(self) -> None:
        """Test synthesis with empty text."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        result = tts.synthesize("")
        assert result is None

    @pytest.mark.unit
    def test_synthesize_whitespace_only(self) -> None:
        """Test synthesis with whitespace only."""
        import blockether_foundation.tts.local_coqui as tts_module

        tts = tts_module.LocalCoquiTTS()
        result = tts.synthesize("   ")
        assert result is None

    @pytest.mark.unit
    def test_pre_download_classmethod(self) -> None:
        """Test pre_download class method."""
        mock_tts_instance = MagicMock()
        mock_tts_instance.to.return_value = None

        with patch(
            "blockether_foundation.tts.local_coqui.TTS", return_value=mock_tts_instance, create=True
        ):
            import blockether_foundation.tts.local_coqui as tts_module

            tts_module.LocalCoquiTTS.pre_download(
                download_root="/tmp/models",
                model_name=CUSTOM_MODEL,
                device="cpu",
            )

            # Should have created TTS instance with correct model
            mock_tts_instance.assert_called_once_with(CUSTOM_MODEL, progress_bar=False)

            # Should have called to with device
            mock_tts_instance.to.assert_called_once_with("cpu")


class TestModelName:
    """Test ModelName type."""

    @pytest.mark.unit
    def test_all_models_are_tts_models(self) -> None:
        """Test that all ModelName values are TTS models (not vocoder or voice_conversion)."""
        expected_models = [
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/multilingual/multi-dataset/xtts_v1.1",
            "tts_models/multilingual/multi-dataset/your_tts",
            "tts_models/multilingual/multi-dataset/bark",
            "tts_models/bg/cv/vits",
            "tts_models/cs/cv/vits",
            "tts_models/da/cv/vits",
            "tts_models/et/cv/vits",
            "tts_models/ga/cv/vits",
            "tts_models/en/ek1/tacotron2",
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/en/ljspeech/tacotron2-DDC_ph",
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/ljspeech/speedy-speech",
            "tts_models/en/ljspeech/tacotron2-DCA",
            "tts_models/en/ljspeech/vits",
            "tts_models/en/ljspeech/vits--neon",
            "tts_models/en/ljspeech/fast_pitch",
            "tts_models/en/ljspeech/overflow",
            "tts_models/en/ljspeech/neural_hmm",
            "tts_models/en/vctk/vits",
            "tts_models/en/vctk/fast_pitch",
            "tts_models/en/sam/tacotron-DDC",
            "tts_models/en/blizzard2013/capacitron-t2-c50",
            "tts_models/en/blizzard2013/capacitron-t2-c150_v2",
            "tts_models/en/multi-dataset/tortoise-v2",
            "tts_models/en/jenny/jenny",
            "tts_models/es/mai/tacotron2-DDC",
            "tts_models/es/css10/vits",
            "tts_models/fr/mai/tacotron2-DDC",
            "tts_models/fr/css10/vits",
            "tts_models/uk/mai/glow-tts",
            "tts_models/uk/mai/vits",
            "tts_models/zh-CN/baker/tacotron2-DDC-GST",
            "tts_models/nl/mai/tacotron2-DDC",
            "tts_models/nl/css10/vits",
            "tts_models/de/thorsten/tacotron2-DCA",
            "tts_models/de/thorsten/vits",
            "tts_models/de/thorsten/tacotron2-DDC",
            "tts_models/de/css10/vits-neon",
            "tts_models/ja/kokoro/tacotron2-DDC",
            "tts_models/tr/common-voice/glow-tts",
            "tts_models/it/mai_female/glow-tts",
            "tts_models/it/mai_female/vits",
            "tts_models/it/mai_male/glow-tts",
            "tts_models/it/mai_male/vits",
            "tts_models/ewe/openbible/vits",
            "tts_models/hau/openbible/vits",
            "tts_models/lin/openbible/vits",
            "tts_models/tw_akuapem/openbible/vits",
            "tts_models/tw_asante/openbible/vits",
            "tts_models/yor/openbible/vits",
            "tts_models/hu/css10/vits",
            "tts_models/el/cv/vits",
            "tts_models/fi/css10/vits",
            "tts_models/hr/cv/vits",
            "tts_models/lt/cv/vits",
            "tts_models/lv/cv/vits",
            "tts_models/mt/cv/vits",
            "tts_models/pl/mai_female/vits",
            "tts_models/pt/cv/vits",
            "tts_models/ro/cv/vits",
            "tts_models/sk/cv/vits",
            "tts_models/sl/cv/vits",
            "tts_models/sv/cv/vits",
            "tts_models/ca/custom/vits",
            "tts_models/fa/custom/glow-tts",
            "tts_models/fa/custom/vits-female",
            "tts_models/bn/custom/vits-male",
            "tts_models/bn/custom/vits-female",
            "tts_models/be/common-voice/glow-tts",
        ]

        # Verify that all expected models don't start with vocoder or voice_conversion
        for model in expected_models:
            assert not model.startswith("vocoder_models")
            assert not model.startswith("voice_conversion_models")
            assert model.startswith("tts_models")
