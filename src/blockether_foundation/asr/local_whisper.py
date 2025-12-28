"""Local ASR implementation using faster-whisper."""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import os
import traceback
from collections.abc import Mapping
from typing import Literal

import av
import numpy as np
from faster_whisper import WhisperModel

try:
    from torch.cuda import empty_cache

    TORCH_CUDA_AVAILABLE = True
except ImportError:
    TORCH_CUDA_AVAILABLE = False

    def empty_cache() -> None:  # noqa: ARG001
        """No-op placeholder when torch.cuda is not available."""
        pass


# Disable HuggingFace progress bars to prevent thread-safety issues
# with tqdm._lock when downloading models concurrently.
# See: https://github.com/huggingface/huggingface_hub/issues/3285
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from .common import (
    TEN_MINUTES,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionSegmentWord,
    merge_transcription_results,
    split_audio_into_chunks,
)

logger = logging.getLogger(__name__)

WhisperModelName = Literal[
    "tiny", "base", "small", "medium", "large-v3", "large-v2", "turbo", "distil-large-v3"
]


class LocalWhisperAudioTranscriber:
    _instance: LocalWhisperAudioTranscriber | None = None
    """Audio transcriber using faster-whisper with VAD."""

    @classmethod
    def pre_download(
        cls,
        download_root: str = "./models",
        whisper_model: WhisperModelName = "turbo",
        device: str = "cpu",
        hf_token: str | None = None,
        enable_diarization: bool = False,
    ) -> None:
        """Pre-download Whisper models to the specified directory.

        Args:
            download_root: Directory to download models to. Defaults to './models'.
            whisper_model: The Whisper model to download.
            device: Device to use for model initialization ('cpu' or 'cuda').
            hf_token: HuggingFace auth token (unused with faster-whisper).
            enable_diarization: Diarization not supported in this implementation.

        Raises:
            ImportError: If faster-whisper is not installed.
        """
        logger.info(f"Pre-downloading models to '{download_root}'...")

        os.makedirs(download_root, exist_ok=True)

        logger.info(f"Downloading Whisper model '{whisper_model}'...")
        compute_type = "float16" if device == "cuda" else "int8"
        WhisperModel(
            whisper_model,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )
        logger.info(f"Whisper model '{whisper_model}' downloaded successfully")

    def __init__(
        self,
        model_id: WhisperModelName = "turbo",
        download_root: str | None = "./models",
        device: str | None = None,
        beam_size: int = 1,
        batch_size: int = 8,
        vad_parameters: Mapping[str, float | int] | None = None,
        hf_token: str | None = None,
        enable_diarization: bool = False,
    ):
        """Initialize the audio transcriber.

        Args:
            model_id: The name of the Whisper model to use. Default 'turbo' (whisper-large-v3-turbo).
            download_root: Path to save/load the model. Defaults to './models'.
            device: Device to use for inference ("cpu" or "cuda"). If None, automatically selected.
            beam_size: Beam size for decoding (1=fastest, 5=more accurate). Default 1.
            batch_size: Batch size for inference. Default 8.
            vad_parameters: Optional overrides for the VAD filter parameters.
            hf_token: HuggingFace auth token (unused).
            enable_diarization: Diarization not supported in this implementation.

        Raises:
            ImportError: If faster-whisper is not installed.
        """
        self.model_id = model_id
        self.download_root = download_root

        if download_root:
            os.makedirs(download_root, exist_ok=True)

        self.device = device or ("cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
        self.beam_size = beam_size
        self.batch_size = batch_size

        default_vad_parameters: dict[str, float | int] = {
            "min_silence_duration_ms": 500,
            "threshold": 0.5,
            "speech_pad_ms": 200,
        }
        self.vad_parameters: dict[str, float | int] = (
            dict(vad_parameters) if vad_parameters is not None else default_vad_parameters
        )
        self._model: WhisperModel | None = None
        self._transcription_count = 0

    def _load_model(self) -> WhisperModel:
        """Load the Whisper model."""
        if self._model is not None:
            return self._model

        logger.info(f"Loading faster-whisper model '{self.model_id}' on device '{self.device}'...")

        compute_type = "float16" if self.device == "cuda" else "int8"

        self._model = WhisperModel(
            self.model_id,
            device=self.device,
            compute_type=compute_type,
            download_root=self.download_root,
        )

        logger.info(f"Model loaded successfully with compute_type='{compute_type}'")

        return self._model

    def load_model(self) -> WhisperModel:
        """Public method to load the model. Delegates to _load_model()."""
        return self._load_model()

    def unload_model(self) -> None:
        """Unload the model to free memory. WARNING: This should never be called in production!"""
        logger.warning("Model unloading is discouraged in production for performance reasons")
        self._model = None

    async def transcribe(
        self,
        audio: bytes,
        effort: float = 1.0,
        language: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        enable_splitting: bool = True,
        max_duration_no_split: float = TEN_MINUTES,
    ) -> TranscriptionResult | None:
        """Transcribe audio data (bytes) to text with VAD.

        Args:
            audio: Raw audio bytes (e.g. MP3, WAV, OGG content).
            effort: Effort level (0.0 to 1.0). Default 1.0 for maximum quality.
            language: Language code (e.g., 'en', 'pl', 'es'). If None, auto-detect.
            num_speakers: Unused (no diarization).
            min_speakers: Unused (no diarization).
            max_speakers: Unused (no diarization).
            enable_splitting: Whether to enable audio splitting for long files. Default True.
            max_duration_no_split: Maximum duration before splitting is triggered. Default 10 minutes.

        Returns:
            TranscriptionResult with segments, or None if transcription failed.
            Use result.text to get the full transcript as a string.

        Note:
            Voice Activity Detection (VAD) is enabled by default.
            Default effort is 1.0 (beam_size=5) for high quality transcription.
            Audio longer than max_duration_no_split will be automatically split into chunks.
        """
        if not audio:
            return None

        effort = max(0.0, min(1.0, effort))
        beam_size = max(1, min(5, int(effort * 5) + 1))

        try:
            if enable_splitting and self._should_split_audio(audio, max_duration_no_split):
                logger.info(
                    f"Audio longer than {max_duration_no_split / 60:.1f} minutes, using chunked transcription"
                )
                return await self._transcribe_with_splitting(audio, beam_size, language)
            else:
                return await asyncio.to_thread(
                    self._run_whisper_inference, audio, beam_size, language
                )
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            traceback.print_exc()
            return None

    def _should_split_audio(self, audio: bytes, max_duration: float) -> bool:
        """Check if audio should be split based on duration.

        Args:
            audio: Raw audio bytes
            max_duration: Maximum duration in seconds before splitting

        Returns:
            True if audio should be split, False otherwise
        """
        container = None
        audio_stream = None
        try:
            audio_stream = io.BytesIO(audio)
            container = av.open(audio_stream)
            audio_stream_obj = container.streams.audio[0]
            duration_val = audio_stream_obj.duration
            time_base_val = audio_stream_obj.time_base
            if duration_val is not None and time_base_val is not None:
                duration = float(duration_val) * float(time_base_val)
            else:
                duration = 0.0

            should_split = duration > max_duration
            if should_split:
                logger.info(
                    f"Audio duration: {duration:.2f}s ({duration / 60:.2f} min) exceeds threshold of {max_duration:.2f}s ({max_duration / 60:.2f} min)"
                )

            return should_split
        except Exception as e:
            logger.warning(f"Could not determine audio duration for splitting decision: {e}")
            return False
        finally:
            if container is not None:
                container.close()
            if audio_stream is not None:
                audio_stream.close()
            return False

    async def _transcribe_with_splitting(
        self, audio: bytes, beam_size: int, language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe audio by splitting it into chunks and processing each chunk.

        Args:
            audio: Raw audio bytes
            beam_size: Whisper beam size for quality
            language: Language code for transcription

        Returns:
            Merged TranscriptionResult from all chunks
        """
        chunks = split_audio_into_chunks(audio)

        if len(chunks) == 1:
            logger.info("Audio splitting resulted in single chunk, processing normally")
            return await asyncio.to_thread(self._run_whisper_inference, audio, beam_size, language)

        logger.info(f"Processing {len(chunks)} audio chunks sequentially")

        results: list[tuple[TranscriptionResult, float, float]] = []
        for i, (chunk_bytes, start_time, end_time) in enumerate(chunks):
            chunk_duration = end_time - start_time
            logger.info(
                f"Processing chunk {i + 1}/{len(chunks)}: {start_time:.2f}s - {end_time:.2f}s ({chunk_duration:.2f}s)"
            )

            try:
                chunk_result = await asyncio.to_thread(
                    self._run_whisper_inference, chunk_bytes, beam_size, language
                )
                if chunk_result:
                    results.append((chunk_result, start_time, end_time))
                    logger.info(
                        f"Chunk {i + 1} transcribed successfully: {len(chunk_result.segments)} segments"
                    )
                else:
                    logger.warning(f"Chunk {i + 1} transcription failed")
            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {e}")
                continue

        if not results:
            logger.error("All chunks failed to transcribe")
            raise ValueError("Failed to transcribe all audio chunks")

        logger.info(f"Merging results from {len(results)} successful chunks")
        return merge_transcription_results(results)

    def _run_whisper_inference(
        self, audio: bytes, beam_size: int, language: str | None = None
    ) -> TranscriptionResult:
        """Run Whisper inference."""
        container = None
        audio_stream = None
        resampler = None

        try:
            model = self._load_model()

            audio_stream = io.BytesIO(audio)
            container = av.open(audio_stream)

            resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)

            frames: list[np.ndarray] = []
            for frame in container.decode(audio=0):  # type: ignore
                resampled_frame = resampler.resample(frame)  # type: ignore
                for resampled in resampled_frame:
                    array = resampled.to_ndarray()
                    frames.append(array.flatten())

            waveform: np.ndarray = np.concatenate(frames).astype(np.float32)

            segments_gen, info = model.transcribe(
                waveform,
                beam_size=beam_size,
                language=language,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=self.vad_parameters,
                word_timestamps=True,
            )

            segments = list(segments_gen)

            logger.info(
                f"Detected language '{info.language}' with probability {info.language_probability:.2f} (beam_size={beam_size})"
            )

            transcription_segments: list[TranscriptionSegment] = []
            for segment in segments:
                words: list[TranscriptionSegmentWord] = []
                if segment.words:
                    for word in segment.words:
                        words.append(
                            TranscriptionSegmentWord(
                                word=word.word,
                                start=word.start,
                                end=word.end,
                                score=word.probability,
                                speaker=None,
                            )
                        )

                transcription_segments.append(
                    TranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text,
                        words=words,
                        speaker=None,
                    )
                )

            result = TranscriptionResult(
                segments=transcription_segments,
                language=info.language,
                language_probability=info.language_probability,
            )

            return result

        except Exception as e:
            logger.error(f"Error during Whisper inference: {e}")
            raise
        finally:
            # Cleanup PyAV resources
            if container is not None:
                container.close()
            if audio_stream is not None:
                audio_stream.close()
            if resampler is not None:
                del resampler

            # CUDA cache cleanup
            if self.device == "cuda" and TORCH_CUDA_AVAILABLE:
                empty_cache()

            # Garbage collection
            gc.collect()

            # Increment transcription counter and periodically unload model
            self._transcription_count = getattr(self, "_transcription_count", 0) + 1
            if self._transcription_count % 50 == 0:
                self._model = None
                gc.collect()
