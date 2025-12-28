"""ElevenLabs TTS implementation."""

from __future__ import annotations

import logging
import os
from typing import Any, cast

from .common import SynthesisResult

logger = logging.getLogger(__name__)


class ElevenLabsTTS:
    """Text-to-speech synthesizer using ElevenLabs API.

    This implementation provides cloud-based text-to-speech synthesis using the
    ElevenLabs API, supporting multiple voices and languages.

    Environment Variables:
        BLOCKETHER_ELEVENLABS_API_KEY: ElevenLabs API key (required).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "eleven_flash_v2_5",
        voice: str = "pNInz6obpgDQGcFmaJgB",
        voice_settings: Any | None = None,
        output_format: str = "mp3_44100_128",
    ):
        """Initialize ElevenLabs TTS synthesizer.

        Args:
            api_key: ElevenLabs API key. If None, reads from BLOCKETHER_ELEVENLABS_API_KEY env var.
            model_id: ElevenLabs model ID to use. Default is 'eleven_flash_v2_5'.
            voice: Voice ID or voice name to use. If None, uses default voice '21m00Tcm4TlvDq8ikWAM'.
            voice_settings: Voice settings (stability, similarity_boost, style, use_speaker_boost).
            output_format: Audio output format (default: mp3_44100_128).

        Raises:
            ImportError: If elevenlabs package is not installed.
            ValueError: If API key is not provided and BLOCKETHER_ELEVENLABS_API_KEY is not set.
        """
        try:
            from importlib.metadata import version

            version("elevenlabs")
        except Exception:
            raise ImportError(
                "ElevenLabs is not installed. Install it with: uv pip install elevenlabs"
            ) from None

        self.api_key = api_key or os.getenv("BLOCKETHER_ELEVENLABS_API_KEY")

        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key is required. "
                "Set BLOCKETHER_ELEVENLABS_API_KEY environment variable or pass api_key parameter."
            )

        self.model_id = model_id
        self.voice = voice
        self.voice_settings = voice_settings
        self.output_format = output_format

        self._client: Any = None

    def _load_client(self) -> Any:
        """Load the ElevenLabs client."""
        if self._client is not None:
            return self._client

        try:
            from elevenlabs import ElevenLabs
        except ImportError as e:
            raise ImportError(
                "ElevenLabs is not installed. Install it with: uv pip install elevenlabs"
            ) from e

        logger.info("Initializing ElevenLabs client...")
        self._client = ElevenLabs(api_key=self.api_key)
        logger.info("ElevenLabs client initialized")

        return cast(Any, self._client)

    def load_client(self) -> Any:
        """Public method to load the client. Delegates to _load_client()."""
        if self._client is not None:
            return self._client
        return cast(Any, self._load_client())

    @classmethod
    def pre_download(
        cls,
        download_root: str = "./models",
        model_name: str = "tts_models/en/ljspeech/vits",
        device: str = "cpu",
    ) -> None:
        """Pre-download TTS models to the specified directory.

        Note: ElevenLabs is a cloud-based API and doesn't require model downloads.
        This method is a no-op for compatibility with the VoiceSynthesizerProtocol.

        Args:
            download_root: Directory to download models to (ignored).
            model_name: The model name (ignored).
            device: Device to use for model initialization (ignored).
        """
        logger.info(
            "ElevenLabs is a cloud-based API and does not require model downloads. "
            "Pre-download is not applicable."
        )

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 24000,
    ) -> SynthesisResult | None:
        """Synthesize speech from text.

        Args:
            text: The text to synthesize into speech.
            voice: Voice ID or voice name to use. If None, uses the voice from __init__ or default.
            language: Language code (e.g., 'en', 'pl', 'es'). ElevenLabs auto-detects language.
            sample_rate: Sample rate for output audio in Hz. Default 24000.
                        Note: ElevenLabs returns audio at its native rate, this param is ignored.

        Returns:
            SynthesisResult with audio bytes and metadata, or None if synthesis failed.

        Note:
            The output audio is in MP3 format (ElevenLabs default).

            The ElevenLabs API auto-detects the language from the text, so the `language`
            parameter is ignored but kept for protocol compatibility.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        try:
            client = self._load_client()
            logger.info(f"Synthesizing text: {text[:50]}{'...' if len(text) > 50 else ''}")

            voice_to_use = voice or self.voice

            if not voice_to_use:
                logger.error(
                    "Voice is required. Provide voice parameter to synthesize() or initialize with a voice."
                )
                return None

            response_iterator = client.text_to_speech.convert(
                text=text,
                voice_id=voice_to_use,
                model_id=self.model_id,
                output_format=self.output_format,
            )

            audio_bytes = b"".join(response_iterator)

            # Calculate duration (MP3 at 44.1kHz mono = ~128kbps)
            # Approximate: duration = (bytes * 8) / 128000
            duration = (len(audio_bytes) * 8) / 128000

            logger.info(f"Synthesis complete: {duration:.2f}s of audio")

            return SynthesisResult(
                audio=audio_bytes,
                sample_rate=44100,  # ElevenLabs default
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
        sample_rate: int = 24000,
    ) -> SynthesisResult | None:
        """Synthesize speech from text and save to file.

        Args:
            text: The text to synthesize into speech.
            output_path: Path to save the synthesized audio file.
            voice: Voice ID or voice name to use. If None, uses the voice from __init__ or default.
            language: Language code (e.g., 'en', 'pl', 'es'). ElevenLabs auto-detects language.
            sample_rate: Sample rate for output audio in Hz. Default 24000.
                        Note: ElevenLabs returns audio at its native rate, this param is ignored.

        Returns:
            SynthesisResult with audio bytes and metadata, or None if synthesis failed.

        Note:
            The output audio is in MP3 format (ElevenLabs default).

            The ElevenLabs API auto-detects the language from the text, so the `language`
            parameter is ignored but kept for protocol compatibility.
        """
        result = self.synthesize(text, voice, language, sample_rate)
        if result is None:
            return None

        with open(output_path, "wb") as f:
            f.write(result.audio)

        logger.info(f"Saved TTS audio to: {output_path}")
        return result
