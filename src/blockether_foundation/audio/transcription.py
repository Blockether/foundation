"""Audio transcription using faster-whisper (with optional VAD)."""

import asyncio
import io
import logging
import os
from collections.abc import Mapping
from typing import (
    Any,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

from faster_whisper import WhisperModel


# Define protocols for the types we need from faster_whisper
@runtime_checkable
class HasText(Protocol):
    """Protocol for objects that have a text attribute."""

    text: str


@runtime_checkable
class HasLanguageInfo(Protocol):
    """Protocol for objects that have language info."""

    language: str
    language_probability: float


# Type aliases
Segment = HasText
TranscriptionInfo = HasLanguageInfo

logger = logging.getLogger(__name__)

WhisperModelName = Literal[
    "tiny", "base", "small", "medium", "large-v3", "large-v2", "turbo", "distil-large-v3"
]


class AudioTranscriber:
    """Audio transcriber using faster-whisper."""

    _instance: Optional["AudioTranscriber"] = None

    def __init__(
        self,
        model_id: WhisperModelName = "turbo",
        download_root: str | None = None,
        device: str | None = None,
        beam_size: int = 1,
        use_vad: bool = False,
        vad_parameters: Mapping[str, float | int] | None = None,
    ):
        """Initialize the audio transcriber.

        Args:
            model_id: The name of the Whisper model to use.
            download_root: Path to save/load the model (for faster-whisper, this is the model directory).
            device: Device to use for inference ("cpu" or "cuda"). If None, automatically selected.
            beam_size: Beam size for decoding (1=fastest, 5=more accurate). Default 1 for maximum speed.
            use_vad: Whether to enable faster-whisper's VAD filter for long/quiet audio.
            vad_parameters: Optional overrides for the VAD filter parameters.
        """
        self.model_id = model_id
        self.download_root = download_root
        self.device = device or ("cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
        self.beam_size = beam_size
        self.use_vad = use_vad
        default_vad_parameters: dict[str, float | int] = {
            "min_silence_duration_ms": 500,
            "silero_sensitivity": 0.5,
            "speech_pad_ms": 200,
        }
        self.vad_parameters: dict[str, float | int] = (
            dict(vad_parameters) if vad_parameters is not None else default_vad_parameters
        )
        self._model = None

    @classmethod
    def get_instance(cls) -> "AudioTranscriber":
        """Get or create the global AudioTranscriber instance."""
        if cls._instance is None:
            model_id = os.getenv("BLOCKETHER_WHISPER_MODEL", "turbo")
            download_root = os.getenv("BLOCKETHER_WHISPER_DOWNLOAD_ROOT")
            beam_size_str = os.getenv("BLOCKETHER_WHISPER_BEAM_SIZE", "1")
            # Allow env toggle so long-running services can enable VAD without code changes
            use_vad_env = os.getenv("BLOCKETHER_WHISPER_VAD", "0").strip().lower()

            try:
                beam_size = int(beam_size_str)
                if beam_size < 1:
                    logger.warning(f"Invalid beam_size '{beam_size_str}', falling back to 1")
                    beam_size = 1
            except ValueError:
                logger.warning(f"Invalid beam_size '{beam_size_str}', falling back to 1")
                beam_size = 1

            use_vad = use_vad_env in {"1", "true", "yes", "on"}

            # Validate model_id - include distil-large-v3 for faster-whisper
            valid_models = [
                "tiny",
                "base",
                "small",
                "medium",
                "large-v3",
                "large-v2",
                "turbo",
                "distil-large-v3",
            ]
            if model_id not in valid_models:
                logger.warning(f"Invalid model_id '{model_id}', falling back to 'turbo'")
                model_id = "turbo"

            cls._instance = cls(
                model_id=model_id,  # type: ignore
                download_root=download_root,
                beam_size=beam_size,
                use_vad=use_vad,
            )
        return cls._instance

    def load_model(self):
        """Load the Whisper model."""
        if self._model is not None:
            return self._model

        logger.info(f"Loading faster-whisper model '{self.model_id}' on device '{self.device}'...")

        # Choose compute type based on device
        compute_type = "float16" if self.device == "cuda" else "int8"

        try:
            self._model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=compute_type,
                download_root=self.download_root,
            )
            logger.info(f"Model loaded successfully with compute_type='{compute_type}'")
        except Exception:
            logger.warning(
                f"Failed to load model with compute_type='{compute_type}', falling back to default"
            )
            self._model = WhisperModel(
                self.model_id,
                device=self.device,
                download_root=self.download_root,
            )

        return self._model

    def unload_model(self):
        """Unload the model to free memory. WARNING: This should never be called in production!"""
        logger.warning("Model unloading is discouraged in production for performance reasons")
        self._model = None

    async def transcribe(self, audio_data: bytes, effort: float = 0.1) -> str | None:
        """Transcribe audio data (bytes) to text.

        Args:
            audio_data: Raw audio bytes (e.g. MP3, WAV, OGG content).
            effort: Effort level (0.0 to 1.0). Higher values = more accurate but slower.

        Returns:
            Transcribed text or None if transcription failed.

        Note:
            Voice Activity Detection (VAD) is controlled when the transcriber is created via
            constructor arguments or the BLOCKETHER_WHISPER_VAD environment variable.
        """
        if not audio_data:
            return None

        # Clamp effort to valid range
        effort = max(0.0, min(1.0, effort))

        # Translate effort to beam_size: 0.0-1.0 -> 1-5
        beam_size = max(1, min(5, int(effort * 5) + 1))

        try:
            # faster-whisper handles audio decoding internally via PyAV
            return await asyncio.to_thread(self._run_whisper_inference, audio_data, beam_size)

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    def _run_whisper_inference(self, audio_data: bytes, beam_size: int) -> str:
        """Run the actual Whisper inference."""
        model = self.load_model()
        model_any: Any = model  # faster-whisper stubs expose partially typed methods

        # faster-whisper can handle audio bytes directly via audio_file-like object
        audio_stream = io.BytesIO(audio_data)

        # Type ignore the model.transcribe call as it returns types from faster_whisper
        # that we don't have full type information for
        if self.use_vad:
            result = model_any.transcribe(
                audio_stream,
                beam_size=beam_size,
                language=None,  # Auto-detect language
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=self.vad_parameters,
            )
        else:
            result = model_any.transcribe(
                audio_stream,
                beam_size=beam_size,
                language=None,  # Auto-detect language
                condition_on_previous_text=False,
            )

        # Unpack the result
        segments, info = result

        logger.info(
            f"Detected language '{info.language}' with probability {info.language_probability:.2f} (beam_size={beam_size})"
        )

        # Collect all text from segments
        text_parts: list[str] = []
        for segment in segments:
            text_parts.append(segment.text)

        # Model stays in memory for maximum performance - never unload!

        return "".join(text_parts).strip()
