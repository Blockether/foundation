"""Audio transcription using faster-whisper with VAD."""

import asyncio
import io
import logging
import os
import traceback
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import (
    Literal,
    Optional,
    cast,
)
from xml.sax.saxutils import escape, quoteattr

import av
import numpy as np
from faster_whisper import WhisperModel
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

WhisperModelName = Literal[
    "tiny", "base", "small", "medium", "large-v3", "large-v2", "turbo", "distil-large-v3"
]

# Audio splitting constants
TEN_MINUTES = 600.0  # 10 minutes in seconds
FIVE_MINUTES = 300.0  # 5 minutes in seconds
FIFTEEN_MINUTES = 900.0  # 15 minutes in seconds


@dataclass
class Word:
    """Individual word with timing information."""

    word: str
    start: float
    end: float
    score: float
    speaker: str | None = None

    def to_xml_dict(self) -> dict[str, str]:
        """Convert to dictionary for XML serialization."""
        return {
            "start": f"{self.start:.3f}",
            "end": f"{self.end:.3f}",
            "score": f"{self.score:.3f}",
            "speaker": self.speaker or "unknown",
            "text": self.word,
        }


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio with timing and optional speaker info."""

    start: float
    end: float
    text: str
    words: list[Word]
    speaker: str | None = None

    def to_xml_dict(self) -> dict[str, str | float | list[dict[str, str]]]:
        """Convert to dictionary for XML serialization."""
        return {
            "start": f"{self.start:.3f}",
            "end": f"{self.end:.3f}",
            "duration": f"{self.end - self.start:.3f}",
            "speaker": self.speaker or "unknown",
            "text": self.text,
            "words": [word.to_xml_dict() for word in self.words],
        }


@dataclass
class TranscriptionResult:
    """Complete transcription result with segments and metadata."""

    segments: list[TranscriptionSegment]
    language: str
    language_probability: float
    created_at: datetime | None = None
    _file_metadata: Mapping[str, str | int] | None = None

    def __post_init__(self):
        """Set created_at if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now(UTC)

    @property
    def text(self) -> str:
        """Get the full text from all segments."""
        return " ".join(seg.text for seg in self.segments).strip()

    @property
    def total_duration(self) -> float:
        """Get the total duration of the transcription."""
        if not self.segments:
            return 0.0
        return max(seg.end for seg in self.segments) - min(seg.start for seg in self.segments)

    @property
    def word_count(self) -> int:
        """Get the total word count."""
        return sum(len(seg.words) for seg in self.segments)

    @property
    def file_metadata(self) -> Mapping[str, str | int] | None:
        """Get the file metadata."""
        return self._file_metadata

    def to_xml_dict(
        self,
    ) -> dict[str, str | float | list[dict[str, str | float | list[dict[str, str]]]]]:
        """Convert to dictionary for XML serialization."""
        return {
            "language": self.language,
            "language_probability": f"{self.language_probability:.3f}",
            "total_duration": f"{self.total_duration:.3f}",
            "segment_count": str(len(self.segments)),
            "word_count": str(self.word_count),
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "segments": [seg.to_xml_dict() for seg in self.segments],
        }


def split_audio_into_chunks(
    audio: bytes,
    chunk_duration: float = TEN_MINUTES,
    min_last_chunk: float = FIVE_MINUTES,
) -> list[tuple[bytes, float, float]]:
    """Split audio into chunks with optimized last segment handling.

    Args:
        audio: Raw audio bytes
        chunk_duration: Target duration for each chunk in seconds (default: 10 minutes)
        min_last_chunk: Minimum duration for last chunk before switching to larger chunks (default: 5 minutes)

    Returns:
        List of tuples: (audio_chunk_bytes, start_time, end_time)
    """
    try:
        # First, get audio info
        audio_stream = io.BytesIO(audio)
        container = av.open(audio_stream)
        audio_stream_obj = container.streams.audio[0]
        # Handle both int and Fraction types for duration and time_base
        duration_val = audio_stream_obj.duration
        time_base_val = audio_stream_obj.time_base
        if duration_val is not None and time_base_val is not None:
            duration = float(duration_val) * float(time_base_val)
        else:
            duration = 0.0
        container.close()

        logger.info(
            f"Splitting audio with total duration: {duration:.2f}s ({duration / 60:.2f} minutes)"
        )

        # Determine chunking strategy
        if duration <= chunk_duration:
            # Audio is shorter than chunk duration, no need to split
            return [(audio, 0.0, duration)]

        # Calculate optimal number of chunks
        num_chunks = int(duration / chunk_duration) + 1  # +1 for the remainder
        last_chunk_duration = duration - ((num_chunks - 1) * chunk_duration)

        # If last chunk would be too short, adjust chunk sizes
        if last_chunk_duration < min_last_chunk:
            # Redistribute time more evenly across all chunks
            # Calculate a more even chunk size
            if num_chunks > 1:
                # Calculate how many chunks we need if each should be at least min_last_chunk
                max_chunks = int(duration / min_last_chunk)
                if max_chunks < 1:
                    max_chunks = 1

                # Use the smaller of: calculated chunks or max_chunks based on min_last_chunk
                num_chunks = min(num_chunks, max_chunks)

                # Calculate new chunk duration
                new_chunk_duration = duration / num_chunks
                chunk_duration = new_chunk_duration
                logger.info(
                    f"Adjusting chunk size to {chunk_duration / 60:.1f}min to avoid short last chunk"
                )

        # Now split using the determined chunk duration
        chunks: list[tuple[bytes, float, float]] = []
        current_time = 0.0

        while current_time < duration:
            start_time = current_time
            end_time = min(current_time + chunk_duration, duration)

            # Extract audio segment
            chunk_bytes = _extract_audio_segment_from_bytes(audio, start_time, end_time)

            if chunk_bytes:
                chunks.append((chunk_bytes, start_time, end_time))
                logger.debug(
                    f"Created chunk: {start_time / 60:.1f}min - {end_time / 60:.1f}min ({(end_time - start_time) / 60:.1f}min)"
                )

            current_time = end_time

        logger.info(
            f"Split audio into {len(chunks)} chunks using {chunk_duration / 60:.1f}min intervals"
        )
        return chunks

    except Exception as e:
        logger.error(f"Error splitting audio: {e}")
        # Return original audio as single chunk if splitting fails
        return [(audio, 0.0, 0.0)]


def _extract_audio_segment_from_bytes(
    audio: bytes, start_time: float, end_time: float
) -> bytes | None:
    """Extract a segment of audio between start_time and end_time from raw audio bytes.

    Args:
        audio: Raw audio bytes
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Raw audio bytes for the segment
    """
    try:
        # Open fresh container for each extraction
        audio_stream = io.BytesIO(audio)
        container = av.open(audio_stream)

        # Seek to start time (PyAV expects integer timestamps)
        start_time_int = int(start_time)
        audio_stream = container.streams.audio[0]
        container.seek(start_time_int, stream=audio_stream)  # type: ignore[arg-type,attr-defined]

        # Resample to 16kHz mono for consistency with transcription
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)

        frames: list[np.ndarray] = []  # type: ignore
        segment_duration = end_time - start_time
        samples_collected = 0
        target_samples = int(segment_duration * 16000)  # 16kHz sample rate

        # Collect frames until we reach the end time
        for frame in container.decode(audio=0):  # type: ignore
            # Handle potentially None values for pts and time_base
            if frame.pts is not None and frame.time_base is not None:  # type: ignore
                frame_time = float(frame.pts * frame.time_base)  # type: ignore
            else:
                continue  # Skip frame if timestamp info is missing

            # Stop if we've gone past the end time
            if frame_time >= end_time:
                break

            # Skip frames before start time (in case seeking wasn't exact)
            if frame_time < start_time:
                continue

            resampled_frame = resampler.resample(frame)  # type: ignore
            for resampled in resampled_frame:
                array = resampled.to_ndarray()
                samples_to_add = min(target_samples - samples_collected, len(array.flatten()))
                if samples_to_add > 0:
                    frames.append(array.flatten()[:samples_to_add])  # type: ignore
                    samples_collected += samples_to_add

                # Stop if we have enough samples
                if samples_collected >= target_samples:
                    break

            # Stop if we have enough samples
            if samples_collected >= target_samples:
                break

        container.close()

        if not frames:
            logger.warning(f"No frames collected for segment {start_time:.2f}s - {end_time:.2f}s")
            return None

        # Concatenate frames and convert to bytes
        waveform: np.ndarray = np.concatenate(frames).astype(np.float32)  # type: ignore

        # Create WAV format bytes

        # Create in-memory WAV file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz

            # Convert float32 to int16
            int16_data = cast(NDArray[np.int16], (waveform * 32767).astype(np.int16))
            wav_file.writeframes(int16_data.tobytes())

        return wav_buffer.getvalue()

    except Exception as e:
        logger.error(f"Error extracting audio segment {start_time:.2f}s - {end_time:.2f}s: {e}")
        return None


def _merge_transcription_results(
    results: list[tuple[TranscriptionResult, float, float]],
) -> TranscriptionResult:
    """Merge transcription results from multiple audio chunks.

    Args:
        results: List of (TranscriptionResult, start_time, end_time) tuples

    Returns:
        Merged TranscriptionResult with adjusted timestamps
    """
    if not results:
        raise ValueError("No transcription results to merge")

    if len(results) == 1:
        return results[0][0]

    # Use language from first result
    language = results[0][0].language
    language_probability = results[0][0].language_probability
    # Preserve file metadata from first result
    file_metadata = results[0][0].file_metadata

    # Merge all segments with adjusted timestamps
    all_segments: list[TranscriptionSegment] = []

    for result, chunk_start, _ in results:
        for segment in result.segments:
            # Adjust timestamps by adding the chunk offset
            adjusted_segment = TranscriptionSegment(
                start=segment.start + chunk_start,
                end=segment.end + chunk_start,
                text=segment.text,
                speaker=segment.speaker,
                words=[
                    Word(
                        word=word.word,
                        start=word.start + chunk_start,
                        end=word.end + chunk_start,
                        score=word.score,
                        speaker=word.speaker,
                    )
                    for word in segment.words
                ],
            )
            all_segments.append(adjusted_segment)

    # Sort segments by start time
    all_segments.sort(key=lambda seg: seg.start)

    return TranscriptionResult(
        segments=all_segments,
        language=language,
        language_probability=language_probability,
        created_at=datetime.now(UTC),
        _file_metadata=file_metadata,
    )


class AudioTranscriber:
    """Audio transcriber using faster-whisper with VAD."""

    _instance: Optional["AudioTranscriber"] = None

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
        auto_assign_instance: bool = True,
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
            auto_assign_instance: Whether to automatically assign this instance as the global singleton.
            enable_diarization: Diarization not supported in this implementation.
        """
        self.model_id = model_id
        self.download_root = download_root

        if self.download_root:
            os.makedirs(self.download_root, exist_ok=True)

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
        self._model = None

        if auto_assign_instance and AudioTranscriber._instance is None:
            AudioTranscriber._instance = self

    @classmethod
    def get_instance(cls) -> "AudioTranscriber":
        """Get or create the global AudioTranscriber instance."""
        if cls._instance is None:
            model_id = os.getenv("BLOCKETHER_WHISPER_MODEL", "turbo").strip().lower()
            download_root = os.getenv("BLOCKETHER_WHISPER_DOWNLOAD_ROOT")
            beam_size_str = os.getenv("BLOCKETHER_WHISPER_BEAM_SIZE", "1")
            batch_size_str = os.getenv("BLOCKETHER_WHISPER_BATCH_SIZE", "8")

            try:
                beam_size = int(beam_size_str)
                if beam_size < 1:
                    logger.warning(f"Invalid beam_size '{beam_size_str}', falling back to 1")
                    beam_size = 1
            except ValueError:
                logger.warning(f"Invalid beam_size '{beam_size_str}', falling back to 1")
                beam_size = 1

            try:
                batch_size = int(batch_size_str)
                if batch_size < 1:
                    logger.warning(f"Invalid batch_size '{batch_size_str}', falling back to 8")
                    batch_size = 8
            except ValueError:
                logger.warning(f"Invalid batch_size '{batch_size_str}', falling back to 8")
                batch_size = 8

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
                batch_size=batch_size,
            )
        return cls._instance

    def _load_model(self):
        """Load the Whisper model."""
        if self._model is not None:
            return self._model

        logger.info(f"Loading faster-whisper model '{self.model_id}' on device '{self.device}'...")

        compute_type = "float16" if self.device == "cuda" else "int8"

        # Fix tqdm disabled_tqdm _lock issue by configuring environment
        # This ensures huggingface_hub downloads work properly
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

        self._model = WhisperModel(
            self.model_id,
            device=self.device,
            compute_type=compute_type,
            download_root=self.download_root,
        )

        logger.info(f"Model loaded successfully with compute_type='{compute_type}'")

        return self._model

    def load_model(self):
        """Public method to load the model. Delegates to _load_model()."""
        return self._load_model()

    def unload_model(self):
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
            # Check if we need to split the audio
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
        try:
            # Quick duration check without full processing
            audio_stream = io.BytesIO(audio)
            container = av.open(audio_stream)
            audio_stream_obj = container.streams.audio[0]
            # Handle both int and Fraction types for duration and time_base
            duration_val = audio_stream_obj.duration
            time_base_val = audio_stream_obj.time_base
            if duration_val is not None and time_base_val is not None:
                duration = float(duration_val) * float(time_base_val)
            else:
                # Fallback to processing the whole audio if duration can't be determined
                logger.warning("Could not determine audio duration, assuming full processing")
                return False
            container.close()

            should_split = duration > max_duration
            if should_split:
                logger.info(
                    f"Audio duration: {duration:.2f}s ({duration / 60:.2f} min) exceeds threshold of {max_duration:.2f}s ({max_duration / 60:.2f} min)"
                )

            return should_split
        except Exception as e:
            logger.warning(f"Could not determine audio duration for splitting decision: {e}")
            return False  # Don't split if we can't determine duration

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
        # Split audio into chunks
        chunks = split_audio_into_chunks(audio)

        if len(chunks) == 1:
            # Only one chunk, process normally
            logger.info("Audio splitting resulted in single chunk, processing normally")
            return await asyncio.to_thread(self._run_whisper_inference, audio, beam_size, language)

        logger.info(f"Processing {len(chunks)} audio chunks sequentially")

        # Process each chunk
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
        return _merge_transcription_results(results)

    def _run_whisper_inference(
        self, audio: bytes, beam_size: int, language: str | None = None
    ) -> TranscriptionResult:
        """Run Whisper inference."""
        model = self._load_model()

        # Decode audio using PyAV
        audio_stream = io.BytesIO(audio)
        container = av.open(audio_stream)

        # Resample to 16kHz mono for Whisper
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)

        frames: list[np.ndarray] = []  # type: ignore
        for frame in container.decode(audio=0):  # type: ignore
            resampled_frame = resampler.resample(frame)  # type: ignore
            for resampled in resampled_frame:
                array = resampled.to_ndarray()
                frames.append(array.flatten())  # type: ignore

        container.close()

        # Concatenate all flattened frames into a single waveform (16kHz mono)
        waveform: np.ndarray = np.concatenate(frames).astype(np.float32)  # type: ignore

        # Run transcription with faster-whisper
        segments_gen, info = model.transcribe(  # type: ignore
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

        # Build transcription segments
        transcription_segments: list[TranscriptionSegment] = []
        for segment in segments:
            words: list[Word] = []
            if segment.words:
                for word in segment.words:
                    words.append(
                        Word(
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

        return TranscriptionResult(
            segments=transcription_segments,
            language=info.language,
            language_probability=info.language_probability,
        )


def format_transcription_for_context(
    transcription: TranscriptionResult,
    *,
    max_segment_content: int | None = None,
    include_word_timestamps: bool = True,
    max_segments: int | None = None,
) -> str:
    """Format transcription result as structured XML for LLM context.

    This function follows the same pattern as format_graph_query_results and
    format_existing_entities_for_context to maintain consistency in the codebase.

    Args:
        transcription: The TranscriptionResult to format
        max_segment_content: Maximum length of segment text before truncation. None for no truncation.
        include_word_timestamps: Whether to include individual word timestamps
        max_segments: Maximum number of segments to include. None for all segments.

    Returns:
        Formatted XML string with the transcription data
    """
    xml_parts: list[str] = []
    xml_parts.append("<transcription>")

    # Add transcription metadata
    attrs = (
        f"language={quoteattr(transcription.language)} "
        f"language_probability={quoteattr(f'{transcription.language_probability:.3f}')} "
        f"total_duration={quoteattr(f'{transcription.total_duration:.3f}')} "
        f"segment_count={quoteattr(str(len(transcription.segments)))} "
        f"word_count={quoteattr(str(transcription.word_count))}"
    )
    if transcription.created_at:
        attrs += f" created_at={quoteattr(transcription.created_at.isoformat())}"

    # Add file metadata if available
    file_metadata: dict[str, str | int] = {}
    if transcription.file_metadata is not None and isinstance(transcription.file_metadata, Mapping):
        file_metadata = dict(transcription.file_metadata)

    if file_metadata:
        filepath = file_metadata.get("filepath")
        if filepath:
            attrs += f" filepath={quoteattr(str(filepath))}"
        filename = file_metadata.get("filename")
        if filename:
            attrs += f" filename={quoteattr(str(filename))}"
        size = file_metadata.get("size")
        if size:
            attrs += f" filesize={quoteattr(str(size))}"
        modified_date = file_metadata.get("modified_date")
        if modified_date:
            attrs += f" file_modified={quoteattr(str(modified_date))}"
        created_date = file_metadata.get("created_date")
        if created_date:
            attrs += f" file_created={quoteattr(str(created_date))}"

    xml_parts.append(f"  <metadata {attrs}>")

    # Add segments
    segments_to_show = transcription.segments
    if max_segments is not None and len(segments_to_show) > max_segments:
        segments_to_show = segments_to_show[:max_segments]

    for i, segment in enumerate(segments_to_show):
        # Prepare segment attributes
        segment_attrs = (
            f'index="{i + 1}" '
            f'start="{segment.start:.3f}" '
            f'end="{segment.end:.3f}" '
            f'duration="{segment.end - segment.start:.3f}" '
            f"speaker={quoteattr(str(segment.speaker or 'unknown'))}"
        )

        # Prepare text content
        text = segment.text
        if max_segment_content is not None and len(text) > max_segment_content:
            text = text[:max_segment_content] + "..."

        if include_word_timestamps and segment.words:
            # With word timestamps: use separate tags
            xml_parts.append(f"    <segment {segment_attrs}>")
            xml_parts.append(f"      <text>{escape(text)}</text>")

            # Add word-level timestamps
            xml_parts.append("      <words>")
            for word in segment.words:
                word_attrs = (
                    f'start="{word.start:.3f}" '
                    f'end="{word.end:.3f}" '
                    f"score={quoteattr(f'{word.score:.3f}')}"
                )
                if word.speaker:
                    word_attrs += f" speaker={quoteattr(word.speaker)}"
                xml_parts.append(f"        <word {word_attrs}>{escape(word.word)}</word>")
            xml_parts.append("      </words>")
            xml_parts.append("    </segment>")
        else:
            # Without word timestamps: put text directly in segment tag
            xml_parts.append(f"    <segment {segment_attrs}>{escape(text)}</segment>")

    xml_parts.append("  </metadata>")
    xml_parts.append("</transcription>")
    return "\n".join(xml_parts)
