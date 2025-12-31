"""Piper TTS implementation - local text-to-speech synthesis."""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Any, cast

from .common import SynthesisResult

logger = logging.getLogger(__name__)

# Default Piper voice model
PIPER_DEFAULT_MODEL = "en_US-ryan-high"

# Check if piper-tts is available
try:
    from importlib.metadata import version

    version("piper-tts")
    PIPER_AVAILABLE = True
except Exception:
    PIPER_AVAILABLE = False


def _check_piper_available() -> None:
    """Check if Piper TTS dependencies are installed."""
    if not PIPER_AVAILABLE:
        raise ImportError(
            "Piper TTS is not installed. "
            "Install it with: uv pip install 'blockether-foundation[tts_local]'"
        )


class PiperTTS:
    """Text-to-speech synthesizer using Piper TTS (local).

    This implementation provides fast, local text-to-speech synthesis using
    Piper TTS with ONNX runtime. No API key required - runs entirely on device.

    Piper supports many voices and languages. Models can be downloaded from:
    https://huggingface.co/rhasspy/piper-voices

    Example:
        ```python
        from blockether_foundation.tts import PiperTTS

        # Initialize with a model path
        tts = PiperTTS(model_path="/path/to/en_US-lessac-medium.onnx")

        # Synthesize speech
        result = tts.synthesize("Hello, world!")

        # Save to file
        tts.synthesize_to_file("Hello, world!", "output.wav")
        ```
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        use_cuda: bool = False,
        speaker_id: int | None = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ):
        """Initialize Piper TTS synthesizer.

        Args:
            model_path: Path to the Piper ONNX model file (.onnx).
                       If None, must be set later via load_model().
            config_path: Path to the model config file (.onnx.json).
                        If None, will look for {model_path}.json
            use_cuda: Whether to use CUDA for inference (requires onnxruntime-gpu).
            speaker_id: Speaker ID for multi-speaker models. None for single-speaker.
            length_scale: Speech speed. >1.0 = slower, <1.0 = faster.
            noise_scale: Generator noise scale (affects voice variation).
            noise_w: Phoneme width noise scale (affects rhythm variation).

        Raises:
            ImportError: If piper-tts package is not installed.
        """
        _check_piper_available()

        self.model_path = Path(model_path) if model_path else None
        self.config_path = Path(config_path) if config_path else None
        self.use_cuda = use_cuda
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w

        self._voice: Any = None

    def _load_voice(self) -> Any:
        """Load the Piper voice model."""
        if self._voice is not None:
            return self._voice

        if self.model_path is None:
            raise ValueError(
                "Model path is required. "
                "Provide model_path to __init__ or call load_model() first."
            )

        try:
            from piper.voice import PiperVoice
        except ImportError as e:
            raise ImportError(
                "Piper TTS is not installed. "
                "Install it with: uv pip install 'blockether-foundation[tts_local]'"
            ) from e

        logger.info(f"Loading Piper voice model from: {self.model_path}")

        # Determine config path
        config_path = self.config_path
        if config_path is None:
            # Default: look for .onnx.json next to .onnx file
            config_path = Path(str(self.model_path) + ".json")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "Piper models require a .onnx.json config file."
            )

        self._voice = PiperVoice.load(
            str(self.model_path),
            config_path=str(config_path),
            use_cuda=self.use_cuda,
        )

        logger.info("Piper voice model loaded successfully")
        return cast(Any, self._voice)

    def load_voice(self) -> Any:
        """Public method to load the voice model. Delegates to _load_voice()."""
        if self._voice is not None:
            return self._voice
        return cast(Any, self._load_voice())

    def load_model(
        self,
        model_path: str | Path,
        config_path: str | Path | None = None,
    ) -> None:
        """Load a Piper voice model.

        Args:
            model_path: Path to the Piper ONNX model file (.onnx).
            config_path: Path to the model config file (.onnx.json).
                        If None, will look for {model_path}.json
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        self._voice = None  # Reset to force reload
        self._load_voice()

    @classmethod
    def pre_download(
        cls,
        model_name: str = PIPER_DEFAULT_MODEL,
        download_root: str = "./models/piper",
    ) -> Path:
        """Pre-download Piper voice models.

        Downloads the specified voice model from Hugging Face to the local directory.

        Args:
            model_name: Name of the voice model in format 'locale-speaker-quality'
                       (e.g., 'en_US-ryan-high', 'en_GB-alba-medium', 'pl_PL-gosia-medium').
                       See https://huggingface.co/rhasspy/piper-voices for available voices.
            download_root: Directory to download models to.

        Returns:
            Path to the downloaded model file.

        Note:
            This requires huggingface_hub to be installed.
        """
        _check_piper_available()

        try:
            from huggingface_hub import (
                hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for downloading models. "
                "Install it with: uv pip install huggingface_hub"
            ) from e

        download_path = Path(download_root)
        download_path.mkdir(parents=True, exist_ok=True)

        # Parse model name: "en_US-ryan-high" -> locale="en_US", speaker="ryan", quality="high"
        parts = model_name.split("-")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid model name format: {model_name}. "
                "Expected format: 'locale-speaker-quality' (e.g., 'en_US-ryan-high')"
            )

        locale, speaker, quality = parts
        lang_short = locale.split("_")[0]  # "en_US" -> "en"

        # HuggingFace path: en/en_US/ryan/high/en_US-ryan-high.onnx
        hf_path = f"{lang_short}/{locale}/{speaker}/{quality}/{model_name}"

        # Download model and config from Hugging Face
        repo_id = "rhasspy/piper-voices"

        logger.info(f"Downloading Piper model: {model_name}")

        # Download .onnx model
        model_file: str = hf_hub_download(
            repo_id=repo_id,
            filename=f"{hf_path}.onnx",
            local_dir=str(download_path),
        )

        # Download .onnx.json config
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{hf_path}.onnx.json",
            local_dir=str(download_path),
        )

        logger.info(f"Downloaded Piper model to: {model_file}")
        return Path(model_file)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 22050,
    ) -> SynthesisResult | None:
        """Synthesize speech from text.

        Args:
            text: The text to synthesize into speech.
            voice: Not used for Piper (voice is determined by loaded model).
                  Kept for protocol compatibility.
            language: Not used for Piper (language is determined by loaded model).
                     Kept for protocol compatibility.
            sample_rate: Desired sample rate for output audio in Hz.
                        Note: Piper outputs at its native rate (usually 22050),
                        resampling is not performed.

        Returns:
            SynthesisResult with WAV audio bytes and metadata, or None if synthesis failed.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        try:
            voice_model = self._load_voice()
            logger.info(f"Synthesizing text: {text[:50]}{'...' if len(text) > 50 else ''}")

            # Use synthesize() generator to get raw audio bytes from AudioChunks
            audio_chunks: list[bytes] = []
            native_sample_rate = 22050  # Default, will be updated from chunk
            sample_channels = 1
            sample_width = 2

            for chunk in voice_model.synthesize(text):
                audio_chunks.append(chunk.audio_int16_bytes)
                native_sample_rate = chunk.sample_rate
                sample_channels = chunk.sample_channels
                sample_width = chunk.sample_width

            raw_audio = b"".join(audio_chunks)

            # Wrap raw PCM in WAV format
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(sample_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(native_sample_rate)
                wav_file.writeframes(raw_audio)

            audio_bytes = wav_buffer.getvalue()
            n_frames = len(raw_audio) // sample_width
            duration = n_frames / native_sample_rate

            logger.info(f"Synthesis complete: {duration:.2f}s of audio at {native_sample_rate}Hz")

            return SynthesisResult(
                audio=audio_bytes,
                sample_rate=native_sample_rate,
                duration=duration,
            )

        except Exception as e:
            logger.error(f"Error synthesizing text: {e}")
            return None

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 22050,
    ) -> SynthesisResult | None:
        """Synthesize speech from text and save to file.

        Args:
            text: The text to synthesize into speech.
            output_path: Path to save the synthesized audio file (WAV format).
            voice: Not used for Piper (voice is determined by loaded model).
                  Kept for protocol compatibility.
            language: Not used for Piper (language is determined by loaded model).
                     Kept for protocol compatibility.
            sample_rate: Desired sample rate for output audio in Hz.
                        Note: Piper outputs at its native rate (usually 22050),
                        resampling is not performed.

        Returns:
            SynthesisResult with WAV audio bytes and metadata, or None if synthesis failed.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        try:
            voice_model = self._load_voice()
            logger.info(f"Synthesizing text to file: {text[:50]}{'...' if len(text) > 50 else ''}")

            # Use synthesize() generator to get raw audio bytes from AudioChunks
            audio_chunks: list[bytes] = []
            native_sample_rate = 22050  # Default, will be updated from chunk
            sample_channels = 1
            sample_width = 2

            for chunk in voice_model.synthesize(text):
                audio_chunks.append(chunk.audio_int16_bytes)
                native_sample_rate = chunk.sample_rate
                sample_channels = chunk.sample_channels
                sample_width = chunk.sample_width

            raw_audio = b"".join(audio_chunks)

            # Write WAV file with proper headers
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(sample_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(native_sample_rate)
                wav_file.writeframes(raw_audio)

            # Read back to get audio bytes
            with open(output_path, "rb") as f:
                audio_bytes = f.read()

            n_frames = len(raw_audio) // sample_width
            duration = n_frames / native_sample_rate

            logger.info(f"Saved TTS audio to: {output_path} ({duration:.2f}s at {native_sample_rate}Hz)")

            return SynthesisResult(
                audio=audio_bytes,
                sample_rate=native_sample_rate,
                duration=duration,
            )

        except Exception as e:
            logger.error(f"Error synthesizing text to file: {e}")
            return None

    def list_available_voices(self) -> list[dict[str, Any]]:
        """List some commonly available Piper voices.

        Returns:
            List of dictionaries with voice information.

        Note:
            Model names follow format: locale-speaker-quality (e.g., 'en_US-ryan-high')
            For the full list, see: https://huggingface.co/rhasspy/piper-voices
        """
        return [
            {"name": "en_US-ryan-high", "language": "en_US", "quality": "high"},
            {"name": "en_US-ryan-medium", "language": "en_US", "quality": "medium"},
            {"name": "en_US-amy-medium", "language": "en_US", "quality": "medium"},
            {"name": "en_US-lessac-high", "language": "en_US", "quality": "high"},
            {"name": "en_US-lessac-medium", "language": "en_US", "quality": "medium"},
            {"name": "en_GB-alba-medium", "language": "en_GB", "quality": "medium"},
            {"name": "de_DE-thorsten-high", "language": "de_DE", "quality": "high"},
            {"name": "es_ES-davefx-medium", "language": "es_ES", "quality": "medium"},
            {"name": "fr_FR-upmc-medium", "language": "fr_FR", "quality": "medium"},
            {"name": "pl_PL-gosia-medium", "language": "pl_PL", "quality": "medium"},
        ]
