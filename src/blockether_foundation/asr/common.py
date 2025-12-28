"""Audio transcription common types and utilities."""

import io
import logging
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from xml.sax.saxutils import escape, quoteattr

import av
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Audio splitting constants
TEN_MINUTES = 600.0
FIVE_MINUTES = 300.0
FIFTEEN_MINUTES = 900.0


class AudioTranscriberProtocol(Protocol):
    """Protocol defining the interface for audio transcribers.

    Any class implementing this protocol can be used with audio hooks.
    """

    async def transcribe(
        self,
        audio: bytes,
        effort: float = 1.0,
        language: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        enable_splitting: bool = True,
        max_duration_no_split: float = 600.0,
    ) -> "TranscriptionResult | None":
        """Transcribe audio data.

        Args:
            audio: Raw audio bytes (e.g. MP3, WAV, OGG content).
            effort: Effort level (0.0 to 1.0). Default 1.0 for maximum quality.
            language: Language code (e.g., 'en', 'pl', 'es'). If None, auto-detect.
            num_speakers: Number of speakers (optional).
            min_speakers: Minimum number of speakers (optional).
            max_speakers: Maximum number of speakers (optional).
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
        ...


@dataclass
class TranscriptionSegmentWord:
    """A single word in a transcription segment with timing and confidence."""

    word: str
    start: float
    end: float
    score: float
    speaker: str | None = None

    def to_xml_dict(self) -> dict[str, Any]:
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
    words: list[TranscriptionSegmentWord]
    speaker: str | None = None

    def to_xml_dict(self) -> dict[str, Any]:
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

    def to_xml_dict(self) -> dict[str, Any]:
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
    overlap: float = 0.0,
) -> list[tuple[bytes, float, float]]:
    """Split audio into chunks with optimized last segment handling and optional overlap.

    Args:
        audio: Raw audio bytes
        chunk_duration: Target duration for each chunk in seconds (default: 10 minutes)
        min_last_chunk: Minimum duration for last chunk before switching to larger chunks (default: 5 minutes)
        overlap: Duration of overlap between consecutive chunks in seconds (default: 0).
                 When > 0, each chunk (except the first) will start `overlap` seconds
                 before where the previous chunk ended. This overlap allows for better
                 speaker identification and context preservation during merging.

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
        num_chunks = int(duration / chunk_duration) + 1
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
        chunk_index = 0

        # Calculate effective step (how much to advance between chunks)
        # With overlap, we advance by (chunk_duration - overlap) each time
        effective_step = chunk_duration - overlap if overlap > 0 else chunk_duration

        while current_time < duration:
            start_time = current_time
            end_time = min(current_time + chunk_duration, duration)

            # Extract audio segment
            chunk_bytes = _extract_audio_segment_from_bytes(audio, start_time, end_time)

            if chunk_bytes:
                chunks.append((chunk_bytes, start_time, end_time))
                if overlap > 0:
                    logger.debug(
                        f"Created chunk {chunk_index}: {start_time / 60:.1f}min - {end_time / 60:.1f}min "
                        f"({(end_time - start_time) / 60:.1f}min, overlap={overlap:.0f}s)"
                    )
                else:
                    logger.debug(
                        f"Created chunk: {start_time / 60:.1f}min - {end_time / 60:.1f}min ({(end_time - start_time) / 60:.1f}min)"
                    )

            # Advance by effective step (accounts for overlap)
            current_time += effective_step
            chunk_index += 1

        if overlap > 0:
            logger.info(
                f"Split audio into {len(chunks)} chunks using {chunk_duration / 60:.1f}min intervals "
                f"with {overlap:.0f}s overlap"
            )
        else:
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
        container.seek(start_time_int, stream=audio_stream)  # type: ignore[attr-defined]

        # Resample to 16kHz mono for consistency with transcription
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)

        frames: list[NDArray[np.float32]] = []
        segment_duration = end_time - start_time
        samples_collected = 0
        target_samples = int(segment_duration * 16000)

        # Collect frames until we reach the end time
        for frame in container.decode(audio=0):  # type: ignore[attr-defined]
            # Handle potentially None values for pts and time_base
            if frame.pts is not None and frame.time_base is not None:
                frame_time = float(frame.pts * frame.time_base)  # type: ignore[unknown-member-type, arg-type]
            else:
                continue

            # Stop if we've gone past the end time
            if frame_time >= end_time:
                break

            # Skip frames before start time (in case seeking wasn't exact)
            if frame_time < start_time:
                continue

            resampled_frame = resampler.resample(frame)  # type: ignore[arg-type]
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
        waveform: NDArray[np.float32] = np.concatenate(frames).astype(np.float32)

        # Create WAV format bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)

            # Convert float32 to int16
            int16_data = np.clip(waveform, -1.0, 1.0)
            int16_data = (int16_data * 32767).astype(np.int16)
            wav_file.writeframes(int16_data.tobytes())

        return wav_buffer.getvalue()

    except Exception as e:
        logger.error(f"Error extracting audio segment {start_time:.2f}s - {end_time:.2f}s: {e}")
        return None


def merge_transcription_results(
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
                    TranscriptionSegmentWord(
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


def format_transcription_for_context(
    transcription: TranscriptionResult,
    *,
    max_segment_content: int | None = None,
    include_word_timestamps: bool = True,
    max_segments: int | None = None,
) -> str:
    """Format transcription result as structured XML for LLM context.

    This function follows same pattern as format_graph_query_results and
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
