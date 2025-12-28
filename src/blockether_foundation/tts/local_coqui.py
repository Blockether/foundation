"""Local TTS implementation using Coqui TTS."""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Literal

from .common import SynthesisResult

try:
    from torch.cuda import empty_cache

    TORCH_CUDA_AVAILABLE = True
except ImportError:
    TORCH_CUDA_AVAILABLE = False

    def empty_cache() -> None:  # noqa: ARG001
        """No-op placeholder when torch.cuda is not available."""
        pass


logger = logging.getLogger(__name__)

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

if TYPE_CHECKING:
    from TTS.api import TTS


ModelName = Literal[
    # Multilingual models
    "tts_models/multilingual/multi-dataset/xtts_v2",
    "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "tts_models/multilingual/multi-dataset/your_tts",
    "tts_models/multilingual/multi-dataset/bark",
    # Bulgarian
    "tts_models/bg/cv/vits",
    # Czech
    "tts_models/cs/cv/vits",
    # Danish
    "tts_models/da/cv/vits",
    # Estonian
    "tts_models/et/cv/vits",
    # Irish
    "tts_models/ga/cv/vits",
    # English
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
    # Spanish
    "tts_models/es/mai/tacotron2-DDC",
    "tts_models/es/css10/vits",
    # French
    "tts_models/fr/mai/tacotron2-DDC",
    "tts_models/fr/css10/vits",
    # Ukrainian
    "tts_models/uk/mai/glow-tts",
    "tts_models/uk/mai/vits",
    # Chinese
    "tts_models/zh-CN/baker/tacotron2-DDC-GST",
    # Dutch
    "tts_models/nl/mai/tacotron2-DDC",
    "tts_models/nl/css10/vits",
    # German
    "tts_models/de/thorsten/tacotron2-DCA",
    "tts_models/de/thorsten/vits",
    "tts_models/de/thorsten/tacotron2-DDC",
    "tts_models/de/css10/vits-neon",
    # Japanese
    "tts_models/ja/kokoro/tacotron2-DDC",
    # Turkish
    "tts_models/tr/common-voice/glow-tts",
    # Italian
    "tts_models/it/mai_female/glow-tts",
    "tts_models/it/mai_female/vits",
    "tts_models/it/mai_male/glow-tts",
    "tts_models/it/mai_male/vits",
    # Ewe
    "tts_models/ewe/openbible/vits",
    # Hausa
    "tts_models/hau/openbible/vits",
    # Lingala
    "tts_models/lin/openbible/vits",
    # Twi (Akuapem)
    "tts_models/tw_akuapem/openbible/vits",
    # Twi (Asante)
    "tts_models/tw_asante/openbible/vits",
    # Yoruba
    "tts_models/yor/openbible/vits",
    # Hungarian
    "tts_models/hu/css10/vits",
    # Greek
    "tts_models/el/cv/vits",
    # Finnish
    "tts_models/fi/css10/vits",
    # Croatian
    "tts_models/hr/cv/vits",
    # Lithuanian
    "tts_models/lt/cv/vits",
    # Latvian
    "tts_models/lv/cv/vits",
    # Maltese
    "tts_models/mt/cv/vits",
    # Polish
    "tts_models/pl/mai_female/vits",
    # Portuguese
    "tts_models/pt/cv/vits",
    # Romanian
    "tts_models/ro/cv/vits",
    # Slovak
    "tts_models/sk/cv/vits",
    # Slovenian
    "tts_models/sl/cv/vits",
    # Swedish
    "tts_models/sv/cv/vits",
    # Catalan
    "tts_models/ca/custom/vits",
    # Persian
    "tts_models/fa/custom/glow-tts",
    "tts_models/fa/custom/vits-female",
    # Bengali
    "tts_models/bn/custom/vits-male",
    "tts_models/bn/custom/vits-female",
    # Belarusian
    "tts_models/be/common-voice/glow-tts",
]

MAX_SEGMENT_DURATION = 20.0


class LocalCoquiTTS:
    """Text-to-speech synthesizer using Coqui TTS.

    This implementation provides local text-to-speech synthesis using the
    Coqui TTS library, supporting multiple models and languages.
    """

    @classmethod
    def pre_download(
        cls,
        download_root: str = "./models",
        model_name: ModelName = "tts_models/en/ljspeech/vits",
        device: str = "cpu",
    ) -> None:
        """Pre-download TTS models to the specified directory.

        Args:
            download_root: Directory to download models to. Defaults to './models'.
            model_name: The Coqui TTS model to download.
            device: Device to use for model initialization ('cpu' or 'cuda').

        Raises:
            ImportError: If coqui-tts is not installed.
        """
        logger.info(f"Pre-downloading models to '{download_root}'...")

        os.makedirs(download_root, exist_ok=True)

        logger.info(f"Downloading TTS model '{model_name}'...")
        try:
            from TTS.api import TTS
        except ImportError as e:
            raise ImportError(
                "Coqui TTS is not installed. "
                "Install it with: uv pip install 'blockether-foundation[tts_local]'"
            ) from e

        TTS(model_name, progress_bar=False).to(device)
        logger.info(f"TTS model '{model_name}' downloaded successfully")

    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/vits",
        device: str | None = None,
        progress_bar: bool = False,
        enable_cuda: bool = False,
        segment_output_dir: str | None = None,
    ):
        """Initialize TTS synthesizer.

        Args:
            model_name: The Coqui TTS model name. Default is English LJSpeech VITS.
                       See available models: https://github.com/coqui-ai/TTS
            device: Device to use for inference ("cpu" or "cuda"). If None, auto-selected.
            progress_bar: Whether to show progress bars during synthesis. Default False.
            enable_cuda: Whether to enable CUDA (GPU) acceleration. Default False.
            segment_output_dir: Directory to save individual TTS segments. If None, segments are not saved.

        Raises:
            ImportError: If coqui-tts is not installed.
        """
        self.model_name = model_name
        self.progress_bar = progress_bar
        self.segment_output_dir = segment_output_dir

        if device is None:
            self.device = "cuda" if enable_cuda and os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
        else:
            self.device = device

        self._tts_model: TTS | None = None
        self._synthesis_count = 0
        self._synthesis_count = 0

    def _is_multilingual_model(self) -> bool:
        return self.model_name.startswith("tts_models/multilingual/")

    def _load_model(self) -> TTS:
        """Load the Coqui TTS model."""
        if self._tts_model is not None:
            return self._tts_model

        # Import here to avoid issues when TTS is not installed
        try:
            from TTS.api import TTS
        except ImportError as e:
            raise ImportError(
                "Coqui TTS is not installed. "
                "Install it with: uv pip install 'blockether-foundation[tts_local]'"
            ) from e

        logger.info(f"Loading Coqui TTS model '{self.model_name}' on device '{self.device}'...")

        self._tts_model = TTS(self.model_name).to(self.device)

        logger.info("Model loaded successfully")

        return self._tts_model

    def load_model(self) -> TTS:
        """Public method to load the model. Delegates to _load_model()."""
        return self._load_model()

    def unload_model(self) -> None:
        """Unload the model to free memory. WARNING: This should never be called in production!"""
        logger.warning("Model unloading is discouraged in production for performance reasons")
        self._tts_model = None

    def _join_audio_segments(self, audio_segments: list[bytes]) -> bytes:
        """Join multiple audio segments into one WAV file.

        For WAV files, concatenate the PCM data and rebuild the header.
        """
        if not audio_segments:
            return b""

        if len(audio_segments) == 1:
            return audio_segments[0]

        try:
            # Extract PCM data from all segments (skip 44-byte WAV header)
            all_pcm_data: list[bytes] = []
            for segment in audio_segments:
                if len(segment) > 44:
                    all_pcm_data.append(segment[44:])
                else:
                    # Segment too short, skip
                    logger.warning("Audio segment too short to extract PCM data")

            if not all_pcm_data:
                return audio_segments[0]

            # Concatenate all PCM data
            combined_pcm = b"".join(all_pcm_data)

            # Create new WAV header for combined audio
            # Sample rate = 24000 Hz, mono = 1 channel, bits = 16
            sample_rate = 24000
            byte_rate = sample_rate * 2  # 2 bytes per sample (16-bit mono)

            # WAV header structure
            # Chunk ID: "RIFF"
            # File size: 36 + data size
            # Format: "WAVE" (16 bytes)
            # Subchunk1: "fmt " (24 bytes) - PCM format
            # Subchunk2: "data" (8 bytes) - actual audio data

            header = bytearray()
            # RIFF chunk
            header.extend(b"RIFF")
            header.extend((36 + len(combined_pcm)).to_bytes(4, byteorder="little"))
            header.extend(b"WAVE")

            # fmt chunk - PCM format
            header.extend(b"fmt ")
            header.extend((16).to_bytes(4, byteorder="little"))  # Chunk size
            header.extend(b"\x01\x00")  # Audio format (1 = PCM)
            header.extend((1).to_bytes(2, byteorder="little"))  # Number of channels (1 = mono)
            header.extend((sample_rate).to_bytes(4, byteorder="little"))  # Sample rate
            header.extend((byte_rate).to_bytes(4, byteorder="little"))  # Byte rate
            header.extend((2).to_bytes(2, byteorder="little"))  # Block align (2 bytes/sample)
            header.extend((16).to_bytes(2, byteorder="little"))  # Bits per sample (16-bit)

            # data chunk
            header.extend(b"data ")
            header.extend(len(combined_pcm).to_bytes(4, byteorder="little"))

            # Combine header with PCM data
            result = bytes(header) + combined_pcm
            logger.info(f"Joined {len(audio_segments)} audio segments into {len(result)} bytes")
            return result

        except Exception as e:
            logger.error(f"Error joining audio segments: {e}")
            # Fallback: return first segment only
            return audio_segments[0]

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 24000,
        segment_index: int | None = None,
        segment_timerange: tuple[float, float] | None = None,
    ) -> SynthesisResult | None:
        """Synthesize speech from text.

        Args:
            text: The text to synthesize into speech.
            voice: Voice model or speaker name. If None, uses default voice from model.
                   For multi-speaker models, this is speaker name/ID.
            language: Language code (e.g., 'en', 'pl', 'es'). If None, uses default language.
                      Used by multilingual models like XTTS v2.
            sample_rate: Sample rate for output audio in Hz. Default 24000.
            segment_index: Optional index for naming saved segments. Required to save segments.
            segment_timerange: Optional (start, end) in seconds for naming saved segments.

        Returns:
            SynthesisResult with audio bytes and metadata, or None if synthesis failed.

        Note:
            The output audio is in WAV format.
            For XTTS v2 multilingual model, set language appropriately (e.g., 'en', 'pl').

            Long text (>20 seconds) is automatically split into chunks and audio is concatenated.

            If segment_output_dir was set in __init__ and both segment_index and segment_timerange
            are provided, the segment will be saved to disk.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None

        for attempt in range(2):
            try:
                return self._do_synthesis(
                    text=text,
                    voice=voice,
                    language=language,
                    sample_rate=sample_rate,
                    segment_index=segment_index,
                    segment_timerange=segment_timerange,
                )
            except RuntimeError as e:
                if "Kernel size" in str(e) and attempt == 0:
                    logger.warning(f"Retrying synthesis after kernel size error: {text[:50]}")
                    continue
                logger.error(f"Error synthesizing text: {e}")
                return None

        return None

    def _do_synthesis(
        self,
        text: str,
        voice: str | None,
        language: str | None,
        sample_rate: int,
        segment_index: int | None,
        segment_timerange: tuple[float, float] | None,
    ) -> SynthesisResult | None:
        try:
            model = self._load_model()
            logger.info(f"Synthesizing text: {text[:50]}{'...' if len(text) > 50 else ''}")

            chunks = self._split_text_into_chunks(text)
            audio_segments: list[bytes] = []

            for chunk in chunks:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name

                try:
                    if (
                        self._is_multilingual_model()
                        and language is not None
                        and voice is not None
                        and voice.endswith(".wav")
                    ):
                        model.tts_to_file(
                            text=chunk,
                            file_path=temp_path,
                            language=language,
                            speaker_wav=voice,
                        )
                    elif self._is_multilingual_model() and language is not None:
                        model.tts_to_file(
                            text=chunk,
                            file_path=temp_path,
                            language=language,
                        )
                    elif voice is not None and voice.endswith(".wav"):
                        model.tts_to_file(
                            text=chunk,
                            file_path=temp_path,
                            speaker_wav=voice,
                        )
                    elif voice is not None:
                        model.tts_to_file(
                            text=chunk,
                            file_path=temp_path,
                            speaker_idx=voice,
                        )
                    else:
                        model.tts_to_file(
                            text=chunk,
                            file_path=temp_path,
                        )

                    with open(temp_path, "rb") as f:
                        audio_segments.append(f.read())

                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # Join all audio segments
            audio_bytes = self._join_audio_segments(audio_segments)

            duration = self._calculate_audio_duration(audio_bytes)
            logger.info(f"Synthesis complete: {duration:.2f}s of audio from {len(chunks)} chunk(s)")

            result = SynthesisResult(
                audio=audio_bytes,
                sample_rate=sample_rate,
                duration=duration,
            )

            # Save segment to disk if output directory and segment info provided
            if (
                self.segment_output_dir is not None
                and segment_index is not None
                and segment_timerange is not None
            ):
                os.makedirs(self.segment_output_dir, exist_ok=True)
                start, end = segment_timerange
                segment_filename = f"tts_segment_{segment_index:03d}_{start:.2f}-{end:.2f}s.wav"
                segment_path = os.path.join(self.segment_output_dir, segment_filename)

                with open(segment_path, "wb") as f:
                    f.write(audio_bytes)

                logger.info(f"Saved TTS segment: {segment_path}")

            return result

        except Exception as e:
            logger.error(f"Error synthesizing text: {e}")
            return None
        finally:
            if self.device == "cuda" and TORCH_CUDA_AVAILABLE:
                empty_cache()

            gc.collect()
            self._synthesis_count += 1
            if self._synthesis_count % 50 == 0:
                self._tts_model = None
                gc.collect()

    def _split_text_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks based on maximum segment duration.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        chunks: list[str] = []
        sentences = text.split(". ")

        current_chunk = ""
        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            if not current_chunk:
                current_chunk = sentence
            elif len(test_chunk.split()) < 150:
                current_chunk = test_chunk
            else:
                if len(current_chunk.strip()) >= 5:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk and len(current_chunk.strip()) >= 5:
            chunks.append(current_chunk)

        return chunks

    def _calculate_audio_duration(self, audio_bytes: bytes) -> float:
        try:
            if len(audio_bytes) < 44:
                return 0.0

            byte_rate = int.from_bytes(audio_bytes[28:32], byteorder="little")

            if byte_rate > 0:
                duration = (len(audio_bytes) - 44) / byte_rate
                return duration

            return 0.0
        except Exception as e:
            logger.warning(f"Could not calculate audio duration: {e}")
            return 0.0

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 24000,
        segment_index: int | None = None,
        segment_timerange: tuple[float, float] | None = None,
    ) -> SynthesisResult | None:
        """Synthesize speech from text and save to file.

        Args:
            text: The text to synthesize into speech.
            output_path: Path to save the synthesized audio file.
            voice: Voice model or speaker name. If None, uses default voice from model.
                   For multi-speaker models, this is speaker name/ID.
            language: Language code (e.g., 'en', 'pl', 'es'). If None, uses default language.
                      Used by multilingual models like XTTS v2.
            sample_rate: Sample rate for output audio in Hz. Default 24000.
            segment_index: Optional index for naming saved segments. Required to save segments.
            segment_timerange: Optional (start, end) in seconds for naming saved segments.

        Returns:
            SynthesisResult with audio bytes and metadata, or None if synthesis failed.

        Note:
            The output audio is in WAV format.
            For XTTS v2 multilingual model, set language appropriately (e.g., 'en', 'pl').

            Long text (>20 seconds) is automatically split into chunks and audio is concatenated.

            If segment_output_dir was set in __init__ and both segment_index and segment_timerange
            are provided, segment will be saved to disk.
        """
        result = self.synthesize(
            text, voice, language, sample_rate, segment_index, segment_timerange
        )
        if result is None:
            return None

        with open(output_path, "wb") as f:
            f.write(result.audio)

        logger.info(f"Saved TTS audio to: {output_path}")
        return result
