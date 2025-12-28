"""Text-to-Speech (TTS) common types and protocol."""

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of text-to-speech synthesis."""

    audio: bytes
    sample_rate: int
    duration: float


class VoiceSynthesizerProtocol(Protocol):
    """Protocol for voice synthesizers (text-to-speech)."""

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 24000,
    ) -> SynthesisResult | None:
        """Synthesize speech from text."""
        ...

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 24000,
    ) -> SynthesisResult | None:
        """Synthesize speech from text and save to file."""
        ...

    def pre_download(
        self,
        download_root: str = "./models",
        model_name: str = "tts_models/en/ljspeech/vits",
        device: str = "cpu",
    ) -> None:
        """Pre-download TTS models to the specified directory."""
        ...
