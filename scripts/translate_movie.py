"""Movie translation pipeline with voice-over dubbing.

This script processes video files to generate transcriptions, multi-language subtitles,
and TTS voice-over audio:
1. Extracts audio from video (WAV)
2. Transcribes audio using Whisper + AI agent (diarization, cleanup)
3. Translates transcription to multiple target languages
4. Generates SRT subtitle files for each language
5. Ducks original audio during speech segments
6. Generates TTS audio for translated text (optional)
7. Mixes TTS audio with ducked original (optional)
8. Muxes final audio with original video (optional)

Workflow:
- Source audio → Whisper transcription → Agent-processed JSON → Translation → SRT files
- (Optional) Translation → TTS generation → Audio mixing → Final video

 Usage Examples:

  Basic usage (Polish subtitles from existing transcription):
    python translate_movie.py \\
      --input-dir poszukiwany_poszukiwana_input \\
      --output-dir poszukiwany_poszukiwana_output \\
      --generate-subs

  Multi-language subtitles (Polish, English, Spanish):
    python translate_movie.py \\
      --input-dir my_movie_input \\
      --output-dir my_movie_output \\
      --source-language pl \\
      --target-language en \\
      --subtitles-languages pl en es \\
      --generate-subs \\
      --include-speaker

  Full dubbing pipeline with TTS:
    python translate_movie.py \\
      --input-dir my_movie_input \\
      --output-dir my_movie_output \\
      --source-language pl \\
      --target-language en \\
      --subtitles-languages en \\
      --voice jerzy \\
      --enable-tts \\
      --generate-subs

 Arguments:
  --input-dir, -i         Input directory with video and PROMPT file [required]
  --output-dir, -o         Output directory for all generated files [required]
  --subtitles-languages, -slangs  Space-separated list of language codes (default: pl)
                          Example: "pl en es" for Polish, English, Spanish
   --source-language, -sl    Source audio language (default: pl)
   --target-language, -tl    Target language for prompt substitution (default: en)
   --model, -m              Whisper model ID (default: large-v3)
                             Options: tiny, base, small, medium, large-v3, large-v2, turbo
   --generate-subs           Generate SRT subtitle files
   --skip-transcription       Skip transcription if JSON already exists
  --include-speaker          Include speaker labels in subtitles
   --duck-volume             Audio ducking volume (0.0-1.0, default: 0.25)
   --duck-lead-time          Duck audio N seconds before speech (default: 0.3)
   --duck-trail-time         Duck audio N seconds after speech (default: 0.3)
   --from-range             Start processing from this time in seconds (e.g., 120 for 2 minutes)
   --to-range               Stop processing at this time in seconds (e.g., 180 for 3 minutes)
   --voice, -v              Voice name for TTS (e.g., 'jerzy', 'tusk', 'nawrocki')
                            Requires --enable-tts
   --tts-model              Coqui TTS model name (default: tts_models/multilingual/multi-dataset/xtts_v2)
   --enable-tts             Enable TTS audio generation and mixing with voice-over
   --burn-subs              Burn subtitles into the final video (requires --generate-subs)

 Language codes:
   Supported: pl, en, es, de, fr, it, pt, ru, ja, zh, ko, ar, hi, tr, nl, sv,
   no, da, fi, cs, sk, hu, ro, bg, el, uk, he, th, vi, id, ms

  Output files:
    - {base_name}.wav              - Extracted audio
    - {base_name}.json             - Source language transcription (Polish)
    - translations/
      {base_name}_en.json + .srt   - English transcription + subtitles
      {base_name}_es.json + .srt   - Spanish transcription + subtitles
      ... (one set per language in --subtitles-languages)
    - segments/                    - TTS audio segments (if --enable-tts)
      tts_segment_000_24.57s.wav
    - partials/                    - Intermediate files for debugging
      {base_name}_ducked.wav       - Audio with ducked original volume
      {base_name}_mixed.wav        - TTS mixed with ducked audio (if --enable-tts)
      {base_name}_muxed.mp4       - Video before burning subtitles (if --burn-subs)
    - final/                       - Final deliverables
      {base_name}_final.mp4        - Final video with mixed audio and burned subtitles (if --burn-subs)
 """

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import av
import numpy as np
from agno.models.openai import OpenAIChat
from av.audio.stream import AudioStream
from pydantic import BaseModel, Field
from pydub import AudioSegment

from blockether_foundation.agents.transcriber import (
    TRANSCRIBER_AGENT,
    DialogueLine,
    Participant,
    Timerange,
    TranscriptionResult,
    dataclass_copy,
    process_audio_files,
)
from blockether_foundation.asr import LocalWhisperAudioTranscriber
from blockether_foundation.tts import (
    TTS_ELEVENLABS_AVAILABLE,
    ElevenLabsTTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger(__name__)

WhisperModelName = Literal["tiny", "base", "small", "medium", "large-v3", "large-v2", "turbo"]


class SubtitleLine(BaseModel):
    """Single subtitle line with timing information."""

    index: int = Field(..., description="Sequential index starting from 1")
    timerange: Timerange = Field(..., description="Start and end times in seconds")
    text: str = Field(..., description="Subtitle text (may contain line breaks)")


LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "pl": "Polish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "sk": "Slovak",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "el": "Greek",
    "uk": "Ukrainian",
    "he": "Hebrew",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
}

# Output directory structure
DIR_TRANSLATIONS = "translations"
DIR_SEGMENTS = "segments"
DIR_SUBTITLES = "subtitles"
DIR_PARTIALS = "partials"
DIR_FINAL = "final"


def _find_audio_stream(container: Any) -> AudioStream:
    """Find audio stream in container.

    Args:
        container: PyAV container object

    Returns:
        First audio stream found

    Raises:
        ValueError: if no audio stream exists
    """
    for stream in container.streams:
        if stream.type == "audio":
            return stream
    raise ValueError("No audio stream found in input file")


def extract_audio_to_wav(input_mp4: str, output_wav: str) -> None:
    """Extract audio track from video file to WAV format.

    Uses PyAV to decode video container and encode audio to PCM16 WAV.

    Args:
        input_mp4: Path to input video file (.mp4, .mkv)
        output_wav: Path to output WAV file
    """
    logger.info(f"Extracting audio from {input_mp4} -> {output_wav}")

    input_container = av.open(input_mp4)
    audio_stream = _find_audio_stream(input_container)

    output_container = av.open(output_wav, mode="w")
    output_stream: AudioStream = output_container.add_stream("pcm_s16le", rate=audio_stream.rate)
    output_stream.layout = audio_stream.layout

    for frame in input_container.decode(audio_stream):
        for packet in output_stream.encode(frame):
            output_container.mux(packet)

    for packet in output_stream.encode(None):
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    logger.info(f"Audio extracted successfully: {output_wav}")


def get_target_language_name(lang_code: str) -> str:
    """Get target language name from language code."""
    if lang_code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[lang_code]
    logger.warning(f"Unknown language code: {lang_code}, using as-is")
    return lang_code.upper()


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_subtitle_as_srt(subtitle: SubtitleLine) -> str:
    """Format a single subtitle line as SRT."""
    start_time = seconds_to_srt_time(subtitle.timerange.start)
    end_time = seconds_to_srt_time(subtitle.timerange.end)

    text = subtitle.text.replace("\n", "\n")

    return f"{subtitle.index}\n{start_time} --> {end_time}\n{text}\n\n"


def write_srt_file(subtitles: list[SubtitleLine], output_path: str) -> None:
    """Write subtitles to SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for subtitle in subtitles:
            f.write(format_subtitle_as_srt(subtitle))
    logger.info(f"SRT file written: {output_path}")


def adjust_srt_timestamps(
    input_srt: str, output_srt: str, offset: float, from_range: float, to_range: float | None
) -> None:
    """Adjust SRT timestamps by subtracting offset and filtering by range."""
    import re

    srt_time_pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")

    def parse_srt_time(time_str: str) -> float:
        match = srt_time_pattern.match(time_str)
        if not match:
            return 0.0
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 1000

    with open(input_srt, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    adjusted_blocks: list[str] = []
    new_index = 1

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        time_line = lines[1]
        time_match = re.match(r"(.+) --> (.+)", time_line)
        if not time_match:
            continue

        start_time = parse_srt_time(time_match.group(1))
        end_time = parse_srt_time(time_match.group(2))

        if end_time <= from_range:
            continue
        if to_range is not None and start_time >= to_range:
            continue

        new_start = max(0, start_time - offset)
        new_end = end_time - offset
        if to_range is not None:
            new_end = min(new_end, to_range - offset)

        text_lines: list[str] = lines[2:]
        text = "\n".join(text_lines)
        adjusted_blocks.append(
            f"{new_index}\n{seconds_to_srt_time(new_start)} --> {seconds_to_srt_time(new_end)}\n{text}"
        )
        new_index += 1

    with open(output_srt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(adjusted_blocks) + "\n")

    logger.info(f"Adjusted SRT written: {output_srt} (offset={offset}s, {new_index - 1} subtitles)")


def find_prompt_file(input_dir: str) -> str | None:
    """Find PROMPT file in the input directory.

    The PROMPT file contains instructions for the transcription/translation agent.
    It can contain {{TARGET_LANGUAGE}} placeholder which will be replaced.

    Args:
        input_dir: Directory to search for PROMPT file

    Returns:
        Full path to PROMPT file, or None if not found
    """
    prompt_path = os.path.join(input_dir, "PROMPT")
    if os.path.exists(prompt_path):
        return prompt_path
    logger.warning(f"No PROMPT file found in {input_dir}")
    return None


def find_video_file(input_dir: str) -> str | None:
    """Find video file in the input directory.

    Searches for common video formats (.mp4, .mkv, .MP4, .MKV).

    Args:
        input_dir: Directory to search for video file

    Returns:
        Full path to video file, or None if not found
    """
    for ext in [".mp4", ".MP4", ".mkv", ".MKV"]:
        for file in os.listdir(input_dir):
            if file.endswith(ext):
                return os.path.join(input_dir, file)
    logger.error(f"No video file found in {input_dir}")
    return None


def read_prompt(prompt_path: str, target_language: str) -> str:
    """Read and process prompt file for transcription/translation agent.

    Replaces {{TARGET_LANGUAGE}} placeholder with actual language name.

    Args:
        prompt_path: Path to PROMPT file
        target_language: Target language code (e.g., 'en', 'pl')

    Returns:
        Processed prompt string with language name substituted
    """
    with open(prompt_path) as f:
        content = f.read()

    target_lang_name = get_target_language_name(target_language)

    content = content.replace("{{TARGET_LANGUAGE}}", target_lang_name)

    logger.info(f"Loaded prompt from {prompt_path}")
    logger.info(f"Target language: {target_language} ({target_lang_name})")
    return content


async def transcribe_audio(
    wav_path: str,
    output_dir: str,
    model: OpenAIChat,
    source_language: str = "pl",
    target_language: str = "en",
    whisper_model_id: WhisperModelName = "large-v3",
    prompt: str = "",
) -> str:
    """Transcribe audio file using Whisper and AI agent.

    Performs full transcription pipeline:
    - Whisper models transcribe audio chunks
    - AI agent processes for diarization and cleanup
    - Saves agent-processed JSON transcription

    Args:
        wav_path: Path to WAV audio file
        output_dir: Directory to save transcription results
        model: OpenAIChat model for agent processing
        source_language: Source audio language code (e.g., 'pl', 'en')
        target_language: Target language for prompt substitution
        whisper_model_id: Whisper model size to use
        prompt: Additional instructions for transcription agent

    Returns:
        Path to generated transcription JSON file
    """
    logger.info(
        f"Transcribing {wav_path} with source_language={source_language}, target_language={target_language}"
    )

    transcriber = LocalWhisperAudioTranscriber(
        model_id=whisper_model_id,
        beam_size=5,
    )

    await process_audio_files(
        glob_pattern=wav_path,
        audio_transcriber=transcriber,
        output_dir=output_dir,
        input=prompt,
        language=source_language,
        save_raw_transcription=True,
        save_dir=output_dir,
        audio_chunking=True,
        chunk_duration=600.0,
        transcription_splitting=False,
        model=model,
    )

    base_name = Path(wav_path).stem
    json_path = os.path.join(output_dir, f"{base_name}.json")
    logger.info(f"Transcription saved to: {json_path}")
    return json_path


def load_transcription(json_path: str) -> TranscriptionResult:
    logger.info(f"Loading transcription from {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    result = TranscriptionResult.model_validate(data)
    logger.info(f"Loaded {len(result.conversation)} dialogue lines")
    return result


def _split_transcription_into_chunks(
    transcription: TranscriptionResult,
    chunk_size_lines: int = 100,
    overlap_lines: int = 5,
) -> list[tuple[int, int, int, list[DialogueLine]]]:
    """Split transcription into chunks with overlap for context continuity.

    Args:
        transcription: Source transcription to split
        chunk_size_lines: Number of lines per chunk (default: 100)
        overlap_lines: Number of overlapping lines between chunks (default: 5)

    Returns:
        List of (chunk_index, start_idx, end_idx, chunk_lines) tuples
    """
    chunks = []
    total_lines = len(transcription.conversation)

    for i in range(0, total_lines, chunk_size_lines):
        chunk_start = max(0, i - overlap_lines)
        chunk_end = min(total_lines, i + chunk_size_lines + overlap_lines)
        chunk_lines = transcription.conversation[chunk_start:chunk_end]
        chunk_index = len(chunks)
        chunks.append((chunk_index, chunk_start, chunk_end, chunk_lines))

    logger.info(
        f"Split {total_lines} lines into {len(chunks)} chunks "
        f"(size={chunk_size_lines}, overlap={overlap_lines})"
    )
    return chunks


async def _translate_single_chunk(
    chunk_lines: list[DialogueLine],
    target_language: str,
    model: OpenAIChat,
    prompt: str,
) -> TranscriptionResult:
    """Translate a single chunk of dialogue lines.

    Args:
        chunk_lines: List of DialogueLine to translate
        target_language: Target language code
        model: OpenAIChat model instance
        prompt: Additional instructions for translation agent

    Returns:
        TranscriptionResult with translated chunk
    """
    target_lang_name = get_target_language_name(target_language)

    translation_prompt = f"""
{prompt}

**IMPORTANT**: Translate the following conversation to {target_lang_name} (language code: {target_language}).

 INPUT FORMAT (XML-like compact format):
Each dialogue line is formatted as:
  SPEAKER [start-end]: text

Example:
  SPEAKER A [12.34-15.67]: First line of dialogue
  SPEAKER B [15.67-18.23]: Response line
  SPEAKER A [18.23-22.10]: Next line
  ...

OUTPUT FORMAT (JSON array with minimal fields):
[
  {{
    "speaker": "Speaker A",
    "text": "Translated text here",
    "timerange": {{"start": 12.34, "end": 15.67}}
  }},
  ...
]

REQUIREMENTS:
- Maintain speaker labels exactly as provided
- Preserve timing information (start/end) exactly
- Translate natural speech and idioms appropriately for {target_lang_name}
- Output ONLY the conversation array as valid JSON
- Do NOT include participants, date, or statistics fields
"""

    agent = dataclass_copy(
        TRANSCRIBER_AGENT,
        instructions=translation_prompt,
        output_schema=TranscriptionResult,
        model=model,
    )

    conversation_text = "\n".join(
        [
            f"{line.speaker} [{line.timerange.start}-{line.timerange.end}]: {line.text}"
            for line in chunk_lines
        ]
    )

    response = await agent.arun(f"Source transcription:\n\n{conversation_text}")
    translated = cast(TranscriptionResult, response.content)

    return translated


def _merge_translated_chunks(
    translated_chunks: list[TranscriptionResult],
    base_transcription: TranscriptionResult,
    chunk_size_lines: int,
    overlap_lines: int,
) -> TranscriptionResult:
    """Merge translated chunks, removing overlaps and normalizing participants.

    Args:
        translated_chunks: List of translated TranscriptionResult chunks
        base_transcription: Original transcription for preserving metadata
        chunk_size_lines: Original chunk size used for splitting
        overlap_lines: Overlap used for splitting

    Returns:
        Merged TranscriptionResult with deduplicated conversation and normalized participants
    """
    merged_conversation: list[DialogueLine] = []

    # Collect all unique participants from chunks (use .name for counting)
    all_participants: dict[str, Participant] = {}
    for chunk in translated_chunks:
        for participant in chunk.participants:
            all_participants[participant.name] = participant

    # Merge chunks sequentially, removing overlap
    for i, chunk in enumerate(translated_chunks):
        if i == 0:
            # First chunk: keep all but last overlap_lines
            keep_lines = len(chunk.conversation) - overlap_lines
            lines_to_add = chunk.conversation[:keep_lines]
        elif i == len(translated_chunks) - 1:
            # Last chunk: skip first overlap_lines
            lines_to_add = chunk.conversation[overlap_lines:]
        else:
            # Middle chunks: skip overlap on both ends
            lines_to_add = chunk.conversation[
                overlap_lines : len(chunk.conversation) - overlap_lines
            ]

        merged_conversation.extend(lines_to_add)
        logger.debug(
            f"Merged chunk {i + 1}/{len(translated_chunks)}: "
            f"kept {len(lines_to_add)} lines from {len(chunk.conversation)}"
        )

    logger.info(f"Merged {len(translated_chunks)} chunks into {len(merged_conversation)} lines")

    return TranscriptionResult(
        participants=list(all_participants.values()),
        conversation=merged_conversation,
        date=base_transcription.date,
    )


async def translate_transcription_to_language(
    base_transcription: TranscriptionResult,
    target_language: str,
    model: OpenAIChat,
    prompt: str,
    chunk_size_lines: int = 100,
    overlap_lines: int = 5,
) -> TranscriptionResult:
    """Translate transcription conversation to target language using chunked translation.

    Splits transcription into chunks with overlap for context continuity,
    translates chunks sequentially, then merges results.

    Args:
        base_transcription: Source transcription to translate
        target_language: Target language code (e.g., 'en', 'pl', 'es')
        model: OpenAIChat model instance
        prompt: Additional instructions for translation agent
        chunk_size_lines: Number of dialogue lines per chunk (default: 100)
        overlap_lines: Number of overlapping lines between chunks (default: 5)

    Returns:
        TranscribedConversation with only conversation array
    """
    target_lang_name = get_target_language_name(target_language)
    total_lines = len(base_transcription.conversation)

    logger.info(f"Translating {total_lines} lines to {target_language} ({target_lang_name})...")
    logger.info(f"Chunk size: {chunk_size_lines} lines, Overlap: {overlap_lines} lines")

    # Split transcription into chunks with overlap
    chunks = _split_transcription_into_chunks(base_transcription, chunk_size_lines, overlap_lines)

    # Translate all chunks sequentially
    logger.info(f"Translating {len(chunks)} chunks sequentially...")
    translated_chunks: list[TranscriptionResult] = []
    for i, (_, _, _, chunk_lines) in enumerate(chunks):
        logger.debug(f"Translating chunk {i + 1}/{len(chunks)}...")
        result = await _translate_single_chunk(
            chunk_lines=chunk_lines,
            target_language=target_language,
            model=model,
            prompt=prompt,
        )
        translated_chunks.append(result)

    # Merge chunks, removing overlaps
    merged_result = _merge_translated_chunks(
        translated_chunks, base_transcription, chunk_size_lines, overlap_lines
    )

    logger.info(
        f"Translation complete: {total_lines} source lines -> "
        f"{len(merged_result.conversation)} target lines"
    )

    return merged_result


def convert_transcription_to_srt(
    transcription: TranscriptionResult,
    output_path: str,
    include_speaker: bool = False,
) -> str:
    """Convert transcription to SRT format without translation.

    Args:
        transcription: Transcription result with dialogue lines
        output_path: Path where SRT file will be written
        include_speaker: If True, include speaker label in subtitle text

    Returns:
        Path to the generated SRT file
    """
    subtitle_lines: list[SubtitleLine] = []

    for i, line in enumerate(transcription.conversation):
        text = line.text if not include_speaker else f"{line.speaker}: {line.text}"

        subtitle = SubtitleLine(
            index=i + 1,
            timerange=line.timerange,
            text=text,
        )
        subtitle_lines.append(subtitle)

    write_srt_file(subtitle_lines, output_path)
    return output_path


def build_volume_adjustments(
    transcription: TranscriptionResult,
    duck_volume: float = 0.25,
    from_range: float | None = None,
    to_range: float | None = None,
    duck_lead_time: float = 0.3,
    duck_trail_time: float = 0.3,
) -> list[tuple[float, float, float]]:
    """Build volume adjustment segments for audio ducking.

    Args:
        transcription: TranscriptionResult with dialogue lines
        duck_volume: Volume level during speech (0.0-1.0)
        from_range: Start time in seconds (None = from beginning)
        to_range: End time in seconds (None = to end)
        duck_lead_time: Duck audio N seconds before speech
        duck_trail_time: Duck audio N seconds after speech

    Returns:
        List of (start_time, end_time, volume) tuples
    """
    adjustments: list[tuple[float, float, float]] = []

    for line in transcription.conversation:
        # Skip segments that end before from_range
        if from_range is not None and line.timerange.end <= from_range:
            continue
        # Skip segments that start after to_range limit
        if to_range is not None and line.timerange.start >= to_range:
            continue

        # Clamp segment to range
        seg_start = line.timerange.start
        seg_end = line.timerange.end

        if from_range is not None:
            seg_start = max(seg_start, from_range)
        if to_range is not None:
            seg_end = min(seg_end, to_range)

        # Only include if segment has positive duration after clamping
        if seg_end > seg_start:
            # Expand segment to include lead and trail time
            start_with_lead = max(0.0, seg_start - duck_lead_time)
            end_with_trail = seg_end + duck_trail_time
            adjustments.append((start_with_lead, end_with_trail, duck_volume))

    adjustments.sort(key=lambda x: x[0])
    logger.info(f"Built {len(adjustments)} volume adjustment segments")
    return adjustments


def duck_audio(
    input_wav: str,
    output_wav: str,
    volume_adjustments: list[tuple[float, float, float]],
    from_range: float | None = None,
    to_range: float | None = None,
) -> None:
    logger.info(f"Ducking audio: {input_wav} -> {output_wav}")
    logger.info(f"  {len(volume_adjustments)} segments will be ducked")
    if from_range is not None or to_range is not None:
        logger.info(f"  Range: {from_range or 0:.1f}s to {to_range or 'end'}s")

    input_container = av.open(input_wav)
    audio_stream = _find_audio_stream(input_container)

    output_container = av.open(output_wav, mode="w")
    output_stream: AudioStream = output_container.add_stream("pcm_s16le", rate=audio_stream.rate)
    output_stream.layout = audio_stream.layout

    sorted_adjustments = sorted(volume_adjustments, key=lambda x: x[0])

    def get_volume_at_time(t: float) -> float:
        for start, end, volume in sorted_adjustments:
            if start <= t < end:
                return volume
            if start > t:
                break
        return 1.0

    frames_processed = 0
    time_base = float(audio_stream.time_base) if audio_stream.time_base else 1.0

    for frame in input_container.decode(audio_stream):
        frame_time = float(frame.pts) * time_base if frame.pts is not None else 0.0

        if from_range is not None and frame_time < from_range:
            continue

        if to_range is not None and frame_time >= to_range:
            logger.info(f"Reached to_range limit ({to_range:.1f}s), stopping audio processing")
            break

        volume = get_volume_at_time(frame_time)

        if volume < 1.0:
            audio_array = frame.to_ndarray()
            scaled_array = (audio_array * volume).astype(np.int16)

            new_frame = av.AudioFrame.from_ndarray(scaled_array, format=frame.format.name)
            new_frame.rate = frame.rate
            new_frame.pts = frame.pts

            for packet in output_stream.encode(new_frame):
                output_container.mux(packet)
        else:
            for packet in output_stream.encode(frame):
                output_container.mux(packet)

        frames_processed += 1
        if frames_processed % 1000 == 0:
            logger.debug(f"Processed {frames_processed} frames (time: {frame_time:.1f}s)")

    for packet in output_stream.encode(None):
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    logger.info(f"Ducked audio saved to: {output_wav}")


def speed_up_audio_with_ffmpeg(
    audio_bytes: bytes, factor: float, output_wav: str, session_folder: str | None = None
) -> None:
    """Speed up audio without changing pitch using FFmpeg's atempo filter.

    Args:
        audio_bytes: Raw audio bytes from TTS
        factor: Speedup multiplier (e.g., 1.5 = 50% faster)
        output_wav: Path to save sped-up audio
        session_folder: Temporary directory for intermediate files
    """
    import shutil

    temp_dir = session_folder or tempfile.gettempdir()
    temp_input = os.path.join(temp_dir, "temp_speedup_input.wav")
    temp_output = os.path.join(temp_dir, "temp_speedup_output.wav")

    try:
        # Write original audio bytes to temp file
        with open(temp_input, "wb") as f:
            f.write(audio_bytes)

        # Build atempo filters: chain multiple if factor > 2.0 or < 0.5
        atempo_filters = []
        remaining_factor = factor

        while remaining_factor > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining_factor /= 2.0

        while remaining_factor < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining_factor /= 0.5

        if abs(remaining_factor - 1.0) > 0.01:
            atempo_filters.append(f"atempo={remaining_factor:.3f}")

        if not atempo_filters:
            # No speedup needed - just copy
            with open(temp_input, "rb") as f:
                audio_data = f.read()
            with open(output_wav, "wb") as f:
                f.write(audio_data)
            logger.debug(f"No speedup needed (factor={factor:.3f}), copying directly")
            return

        filter_string = ",".join(atempo_filters)
        logger.debug(f"Applying speedup: {filter_string}")

        cmd = ["ffmpeg", "-i", temp_input, "-af", filter_string, "-y", temp_output]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Copy result to output path
        shutil.copy(temp_output, output_wav)
        logger.debug(f"Speedup complete: {factor:.3f}x")

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg speedup failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise
    finally:
        # Cleanup temp files
        for temp_file in [temp_input, temp_output]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file}: {e}")


def align_tts_segments(
    tts_segments: list[tuple[float, float, bytes, float]],
    ducked_wav_path: str,
    output_wav_path: str,
    session_folder: str | None = None,
    time_offset: float = 0.0,
    min_gap_ms: int = 0,
    max_speedup: float = 1.5,
    padding_factor: float = 1.1,
) -> str:
    """Align TTS audio segments sequentially without overlap.

    ALWAYS speeds up TTS to fit in slot with padding. Calculates exact speedup needed.

    Args:
        tts_segments: List of (start, end, audio_bytes, duration_ms) tuples
        ducked_wav_path: Background audio file
        output_wav_path: Path for aligned output
        session_folder: Temp directory for intermediate files
        time_offset: Offset to subtract from segment timestamps
        min_gap_ms: Minimum gap between segments (default: 0ms)
        max_speedup: Maximum allowed speedup factor (default: 1.5x)
        padding_factor: Extra speedup for breathing room (default: 1.1 = 10% extra)

    Returns:
        Path to aligned audio file
    """
    logger.info("=" * 60)
    logger.info("[ALIGNMENT] Starting TTS sequential alignment")
    logger.info(f"  TTS segments: {len(tts_segments)}")
    logger.info(f"  Max speedup: {max_speedup}x, Padding: {padding_factor}x, Gap: {min_gap_ms}ms")
    if time_offset > 0:
        logger.info(f"  Time offset: {time_offset:.2f}s")
    logger.info("=" * 60)

    if not tts_segments:
        logger.warning("No TTS segments to align, copying background audio")
        import shutil

        shutil.copy(ducked_wav_path, output_wav_path)
        return output_wav_path

    temp_dir = session_folder or tempfile.gettempdir()
    time_offset_ms = int(time_offset * 1000)

    background = AudioSegment.from_wav(ducked_wav_path)
    background_length_ms = len(background)
    logger.info(f"  Background audio: {background_length_ms:.0f}ms")

    final_audio = background
    playhead_ms = 0

    for i, (start_time, end_time, audio_bytes, tts_duration_ms) in enumerate(tts_segments):
        ideal_start_ms = int(start_time * 1000) - time_offset_ms
        ideal_end_ms = int(end_time * 1000) - time_offset_ms
        slot_duration_ms = ideal_end_ms - ideal_start_ms

        if ideal_start_ms < 0:
            logger.warning(f"  Segment {i + 1}: negative start ({ideal_start_ms}ms), skipping")
            continue

        if ideal_start_ms >= background_length_ms:
            logger.warning(f"  Segment {i + 1}: beyond audio length, stopping")
            break

        actual_start_ms = max(ideal_start_ms, playhead_ms)
        delay_ms = actual_start_ms - ideal_start_ms

        # Detect audio format from bytes signature (MP3: ID3, WAV: RIFF)
        audio_format = "wav" if audio_bytes[:4] == b"RIFF" else "mp3"

        tts_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
        processed_tts = tts_audio
        actual_duration_ms = int(tts_duration_ms)

        target_duration_ms = slot_duration_ms - min_gap_ms
        if target_duration_ms < 100:
            target_duration_ms = slot_duration_ms

        needed_speedup = (tts_duration_ms / target_duration_ms) * padding_factor
        speedup_applied = min(needed_speedup, max_speedup)

        if speedup_applied > 1.05:
            speedup_filename = f"tts_segment_{i:03d}_{start_time:.2f}-{end_time:.2f}s_speedup.wav"
            speedup_path = os.path.join(temp_dir, speedup_filename)
            speed_up_audio_with_ffmpeg(
                audio_bytes=audio_bytes,
                factor=speedup_applied,
                output_wav=speedup_path,
                session_folder=temp_dir,
            )
            processed_tts = AudioSegment.from_wav(speedup_path)
            actual_duration_ms = len(processed_tts)

        logger.info(f"[SEGMENT {i + 1}/{len(tts_segments)}]")
        logger.info(
            f"  Slot: {slot_duration_ms}ms, TTS: {tts_duration_ms:.0f}ms, Target: {target_duration_ms}ms"
        )
        logger.info(f"  Speedup: {speedup_applied:.2f}x -> {actual_duration_ms}ms")
        if delay_ms > 0:
            logger.info(f"  Delayed: {delay_ms}ms (placed at {actual_start_ms}ms)")

        final_audio = final_audio.overlay(processed_tts, position=actual_start_ms)
        playhead_ms = actual_start_ms + actual_duration_ms + min_gap_ms

    final_audio.export(output_wav_path, format="wav")
    logger.info("=" * 60)
    logger.info(f"[ALIGNMENT] Complete: {len(final_audio):.0f}ms")
    logger.info(f"[ALIGNMENT] Saved: {output_wav_path}")
    logger.info("=" * 60)

    return output_wav_path


def generate_tts_audio(
    transcription: TranscriptionResult,
    voice_path: str | None = None,
    language: str = "en",
    eleven_model_id: str = "eleven_flash_v2_5",
    from_range: float | None = None,
    to_range: float | None = None,
    output_dir: str | None = None,
) -> list[tuple[float, float, bytes, float]]:
    logger.info(f"Generating TTS audio for {len(transcription.conversation)} dialogue lines")
    logger.info(f"  Voice: {voice_path or 'default'}")
    logger.info(f"  Language: {language}")
    logger.info(f"  Backend: ElevenLabs")
    if from_range is not None or to_range is not None:
        logger.info(f"  Range: {from_range or 0:.1f}s to {to_range or 'end'}s")
        if not TTS_ELEVENLABS_AVAILABLE:
            raise ImportError(
                "ElevenLabs TTS is not installed. "
                "Install it with: uv pip install 'blockether-foundation[tts_elevenlabs]'"
            )

        api_key = os.getenv("BLOCKETHER_ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError(
                "BLOCKETHER_ELEVENLABS_API_KEY environment variable is required for ElevenLabs TTS"
            )

        logger.info(f"  Model: {eleven_model_id} (ElevenLabs Flash)")
        tts = ElevenLabsTTS(
            api_key=api_key,
            model_id=eleven_model_id,
        )
    tts_segments: list[tuple[float, float, bytes, float]] = []

    for i, line in enumerate(transcription.conversation):
        if from_range is not None and line.timerange.end <= from_range:
            logger.debug(
                f"  Skipping line {i + 1}: ends before from range ({line.timerange.end} <= {from_range})"
            )
            continue
        if (
            to_range is not None
            and from_range is not None
            and line.timerange.start >= from_range
            and line.timerange.start < to_range
        ):
            # Line is inside or at the edge of the range
            if to_range is not None and line.timerange.end <= to_range:
                # Line is fully inside the range (END <= to_range)
                pass
            elif to_range is not None and line.timerange.end > to_range:
                # Line is beyond the range
                logger.debug(
                    f"  Skipping line {i + 1}: starts after range ({line.timerange.start} >= {to_range})"
                )
                continue
        if to_range is None:
            # No range specified, include all lines
            pass
        if to_range is not None and line.timerange.start >= to_range:
            logger.debug(
                f"  Skipping line {i + 1}: starts after to_range ({line.timerange.start} >= {to_range})"
            )
            continue

        logger.debug(f"  [{i}] Speaker: {line.speaker}")
        logger.debug(f"  [{i}] Timerange: {line.timerange.start:.2f}s - {line.timerange.end:.2f}s")
        logger.debug(f"  [{i}] Text: {line.text}")
        logger.info(
            f"  Synthesizing line {i + 1}/{len(transcription.conversation)}: {line.text[:50]}..."
        )

        # ElevenLabs doesn't support segment_index or segment_timerange
        result = tts.synthesize(
            text=line.text,
            voice=voice_path,
            language=language,
        )

        if result is not None and result.audio:
            tts_duration_ms = int(result.duration * 1000)
            tts_segments.append(
                (line.timerange.start, line.timerange.end, result.audio, tts_duration_ms)
            )
            original_duration = line.timerange.end - line.timerange.start
            logger.info(
                f"    Generated {result.duration:.2f}s audio (original: {original_duration:.2f}s)"
            )
            logger.debug(f"    Audio bytes: {len(result.audio)} bytes")
        else:
            logger.warning(f"    Failed to synthesize line {i + 1}")

    logger.info(f"TTS generation complete: {len(tts_segments)} segments created")
    return tts_segments


def mix_tts_with_audio(
    ducked_wav: str,
    tts_segments: list[tuple[float, float, bytes, float]],
    output_wav: str,
    from_range: float | None = None,
    to_range: float | None = None,
    output_dir: str | None = None,
    min_gap_ms: int = 0,
    max_speedup: float = 1.5,
    padding_factor: float = 1.1,
) -> None:
    """Mix TTS audio with background using sequential alignment.

    Args:
        ducked_wav: Path to ducked background audio
        tts_segments: List of (start, end, audio_bytes, duration_ms) tuples
        output_wav: Path for final mixed audio
        from_range: Start time offset - subtracted from segment timestamps
        to_range: End time range (for logging only)
        output_dir: Directory for intermediate files
        min_gap_ms: Minimum gap between segments (default: 0ms)
        max_speedup: Maximum speedup factor (default: 1.5x)
        padding_factor: Extra speedup for breathing room (default: 1.1 = 10% faster)
    """
    logger.info(f"Mixing TTS with background: {ducked_wav} -> {output_wav}")
    logger.info(
        f"  Segments: {len(tts_segments)}, Max speedup: {max_speedup}x, Gap: {min_gap_ms}ms"
    )
    if from_range is not None or to_range is not None:
        logger.info(f"  Range: {from_range or 0:.1f}s to {to_range or 'end'}s")

    align_tts_segments(
        tts_segments=tts_segments,
        ducked_wav_path=ducked_wav,
        output_wav_path=output_wav,
        session_folder=output_dir,
        time_offset=from_range or 0.0,
        min_gap_ms=min_gap_ms,
        max_speedup=max_speedup,
        padding_factor=padding_factor,
    )

    logger.info(f"Mixed audio saved to: {output_wav}")


def mux_audio_with_video(
    video_path: str,
    audio_path: str,
    output_video_path: str,
    from_range: float | None = None,
    to_range: float | None = None,
) -> None:
    logger.info(f"Muxing audio with video: {audio_path} + {video_path} -> {output_video_path}")
    if from_range is not None or to_range is not None:
        logger.info(f"  Range: {from_range or 0:.1f}s to {to_range or 'end'}s")

    cmd = ["ffmpeg", "-y"]

    if from_range is not None:
        cmd.extend(["-ss", str(from_range)])
    cmd.extend(["-i", video_path])

    cmd.extend(["-i", audio_path])

    if to_range is not None:
        duration = to_range - (from_range or 0)
        cmd.extend(["-t", str(duration)])

    cmd.extend(["-c:v", "copy", "-c:a", "aac", "-map", "0:v", "-map", "1:a", "-shortest"])
    cmd.append(output_video_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"FFmpeg output: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg to use video muxing.")
        raise

    logger.info(f"Final video saved to: {output_video_path}")


def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_video_path: str,
    srt_offset: float | None = None,
    from_range: float | None = None,
    to_range: float | None = None,
    font_size: int = 24,
    font_name: str = "Arial",
    margin_v: int = 30,
) -> None:
    """Burn subtitles into video.

    Args:
        video_path: Source video file
        srt_path: Subtitle file with original timestamps
        output_video_path: Output video path
        srt_offset: Offset to subtract from SRT timestamps (use when video already trimmed)
        from_range: Seek to this position in source video (use when video NOT trimmed)
        to_range: Stop at this position (used with from_range for duration calculation)
        font_size: Subtitle font size
        font_name: Subtitle font name
        margin_v: Vertical margin for subtitles
    """
    logger.info(f"Burning subtitles: {srt_path} -> {output_video_path}")
    if srt_offset is not None:
        logger.info(f"  SRT offset: {srt_offset:.1f}s")
    if from_range is not None or to_range is not None:
        logger.info(f"  Video range: {from_range or 0:.1f}s to {to_range or 'end'}s")

    adjusted_srt = srt_path
    offset_for_srt = srt_offset if srt_offset is not None else from_range
    if offset_for_srt is not None:
        temp_srt = srt_path.replace(".srt", "_adjusted.srt")
        adjust_srt_timestamps(srt_path, temp_srt, offset_for_srt, offset_for_srt, to_range)
        adjusted_srt = temp_srt

    style = (
        f"Fontname={font_name},"
        f"Fontsize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=1,"
        f"Outline=2,"
        f"Shadow=1,"
        f"MarginV={margin_v}"
    )

    srt_escaped = adjusted_srt.replace(":", r"\:").replace("'", r"\'")
    vf_filter = f"subtitles='{srt_escaped}':force_style='{style}'"

    cmd = ["ffmpeg", "-y"]

    if from_range is not None:
        cmd.extend(["-ss", str(from_range)])

    cmd.extend(["-i", video_path])

    if to_range is not None and from_range is not None:
        duration = to_range - from_range
        cmd.extend(["-t", str(duration)])

    cmd.extend(["-vf", vf_filter, "-c:a", "copy", output_video_path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Subtitles burned successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg subtitle burning failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg to burn subtitles.")
        raise

    logger.info(f"Video with burned subtitles saved to: {output_video_path}")


def _find_audio(container: Any) -> AudioStream:
    """Find audio stream in container.

    Args:
        container: PyAV container object

    Returns:
        First audio stream found

    Raises:
        ValueError: if no audio stream exists
    """
    for stream in container.streams:
        if stream.type == "audio":
            return stream
    raise ValueError("No audio stream found in input file")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate movie audio with voice-over dubbing",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Input directory containing video and PROMPT file",
    )
    parser.add_argument("--model", "-m", type=str, default="large-v3", help="Whisper model ID")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--source-language",
        "-sl",
        type=str,
        default="pl",
        help="Source language code (e.g., 'pl' for Polish, 'en' for English, 'es' for Spanish)",
    )
    parser.add_argument(
        "--target-language",
        "-tl",
        type=str,
        default="en",
        help="Target language code for translation (e.g., 'en' for English, 'pl' for Polish, 'es' for Spanish)",
    )
    parser.add_argument(
        "--subtitles-languages",
        "-slangs",
        nargs="+",
        default=["pl"],
        help="List of language codes for subtitle generation (default: pl). Space-separated. Example: 'pl en es' for Polish, English, Spanish",
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip audio transcription if {base_name}.json already exists. Use to regenerate subtitles only.",
    )
    parser.add_argument(
        "--generate-subs",
        action="store_true",
        help="Generate SRT subtitle files for all languages in --subtitles-languages",
    )
    parser.add_argument(
        "--include-speaker",
        action="store_true",
        help="Include speaker labels (Speaker A, Speaker B) in subtitle text",
    )
    parser.add_argument(
        "--duck-volume",
        type=float,
        default=0.25,
        help="Volume level for audio ducking (0.0-1.0, default: 0.25).",
    )
    parser.add_argument(
        "--duck-lead-time",
        type=float,
        default=0.3,
        help="Duck audio N seconds before speech starts (default: 0.3). Smooths transitions.",
    )
    parser.add_argument(
        "--duck-trail-time",
        type=float,
        default=0.3,
        help="Duck audio N seconds after speech ends (default: 0.3). Smooths transitions.",
    )
    parser.add_argument(
        "--from-range",
        type=float,
        default=None,
        help="Start processing from this time in seconds (e.g., 120 for 2 minutes). If not specified, starts from beginning.",
    )
    parser.add_argument(
        "--to-range",
        type=float,
        default=None,
        help="Stop processing at this time in seconds (e.g., 180 for 3 minutes). If not specified, processes to end.",
    )
    parser.add_argument(
        "--burn-subs",
        action="store_true",
        help="Burn subtitles into the final video. Requires --generate-subs.",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default=None,
        help="Voice name for TTS (e.g., 'jerzy', 'tusk', 'nawrocki'). If not specified, uses default voice.",
    )
    parser.add_argument(
        "--enable-tts",
        action="store_true",
        help="Enable TTS audio generation and mixing. Requires --voice.",
    )
    parser.add_argument(
        "--eleven-model-id",
        type=str,
        default="eleven_flash_v2_5",
        help="ElevenLabs model ID (default: eleven_flash_v2_5 for Flash multilingual, or 'eleven_flash_v2' for English-only flash).",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    source_language = args.source_language
    target_language = args.target_language

    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    prompt_path = find_prompt_file(input_dir)
    if not prompt_path:
        logger.error("PROMPT file is required in input directory")
        return

    special_instructions = read_prompt(prompt_path, target_language)

    video_path = find_video_file(input_dir)
    if not video_path:
        return

    input_path = Path(video_path)
    base_name = input_path.stem

    os.makedirs(output_dir, exist_ok=True)

    translations_dir = os.path.join(output_dir, DIR_TRANSLATIONS)
    segments_dir = os.path.join(output_dir, DIR_SEGMENTS)
    subtitles_dir = os.path.join(output_dir, DIR_SUBTITLES)
    partials_dir = os.path.join(output_dir, DIR_PARTIALS)
    final_dir = os.path.join(output_dir, DIR_FINAL)

    os.makedirs(translations_dir, exist_ok=True)
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(subtitles_dir, exist_ok=True)
    os.makedirs(partials_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    model_base_url = os.getenv("BLOCKETHER_LLM_API_BASE_URL")
    if not model_base_url:
        logger.error("BLOCKETHER_LLM_API_BASE_URL environment variable is not set!")
        logger.info("Please set BLOCKETHER_LLM_API_BASE_URL to use the transcription service.")
        return

    blockether_model = OpenAIChat(
        id="gpt-4.1",
        base_url=model_base_url,
        api_key=os.getenv("BLOCKETHER_LLM_API_KEY"),
    )
    logger.info(f"Using model: gpt-4.1 at {model_base_url}")

    wav_path = os.path.join(output_dir, f"{base_name}.wav")
    json_path = os.path.join(translations_dir, f"{base_name}_{source_language}.json")
    ducked_wav_path = os.path.join(partials_dir, f"{base_name}_ducked.wav")
    mixed_wav_path = os.path.join(partials_dir, f"{base_name}_mixed.wav")

    logger.info("=" * 60)
    logger.info("MOVIE TRANSLATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input dir:       {input_dir}")
    logger.info(f"Video file:      {video_path}")
    logger.info(f"Prompt file:      {prompt_path}")
    logger.info(
        f"Source language:  {source_language} ({get_target_language_name(source_language)})"
    )
    logger.info(
        f"Target language:  {target_language} ({get_target_language_name(target_language)})"
    )
    logger.info(f"Duck volume:     {args.duck_volume}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info("=" * 60)

    if not os.path.exists(wav_path):
        logger.info("[STEP 1-2] Extracting audio from video...")
        extract_audio_to_wav(video_path, wav_path)
    else:
        logger.info(f"[STEP 1-2] WAV already exists: {wav_path}")

    raw_chunks_exist = any(
        os.path.exists(os.path.join(output_dir, f"{base_name}_chunk_*_raw.json")) for _ in [0]
    )

    base_json_path = os.path.join(translations_dir, f"{base_name}.json")
    existing_json_path = json_path if os.path.exists(json_path) else base_json_path

    if not args.skip_transcription and not os.path.exists(existing_json_path):
        logger.info("[STEP 3] Transcribing audio...")
        whisper_model_id = cast(WhisperModelName, args.model)
        await transcribe_audio(
            wav_path=wav_path,
            output_dir=translations_dir,
            model=blockether_model,
            source_language=source_language,
            target_language=target_language,
            whisper_model_id=whisper_model_id,
            prompt=special_instructions,
        )
    else:
        logger.info(f"[STEP 3] Using existing transcription: {existing_json_path}")

    if not os.path.exists(existing_json_path):
        logger.error(f"Transcription JSON not found: {existing_json_path}")
        return

    logger.info("[STEP 4] Loading base transcription...")
    base_transcription = load_transcription(existing_json_path)

    subtitles_languages = args.subtitles_languages
    srt_paths: list[str] = []

    logger.info(
        f"[STEP 5] Processing {len(subtitles_languages)} subtitle languages: {subtitles_languages}"
    )

    for lang_code in subtitles_languages:
        lang_name = get_target_language_name(lang_code)
        logger.info(f"  Processing language: {lang_code} ({lang_name})")

        transcription_path = os.path.join(translations_dir, f"{base_name}_{lang_code}.json")

        srt_path = os.path.join(subtitles_dir, f"{base_name}_{lang_code}.srt")

        if not os.path.exists(transcription_path):
            translation = await translate_transcription_to_language(
                base_transcription=base_transcription,
                target_language=lang_code,
                model=blockether_model,
                prompt=special_instructions,
            )

            json_str = translation.model_dump_json(indent=2)
            with open(transcription_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info(f"    Saved transcription: {transcription_path}")

            if args.generate_subs:
                convert_transcription_to_srt(
                    translation,
                    srt_path,
                    include_speaker=args.include_speaker,
                )
                logger.info(f"    Subtitles generated: {srt_path}")
        else:
            logger.info(f"    Using existing transcription: {transcription_path}")

        if os.path.exists(srt_path):
            srt_paths.append(srt_path)

    volume_adjustments = build_volume_adjustments(
        base_transcription,
        args.duck_volume,
        args.from_range,
        args.to_range,
        args.duck_lead_time,
        args.duck_trail_time,
    )

    logger.info("[STEP 5] Ducking audio during speech segments...")
    logger.info(
        f"  Processing {len(volume_adjustments)} adjustment segments (from={args.from_range}, to={args.to_range})"
    )
    duck_audio(wav_path, ducked_wav_path, volume_adjustments, args.from_range, args.to_range)

    # TTS Generation and Mixing (Step 6-7)
    tts_segments: list[tuple[float, float, bytes, float]] = []
    if args.enable_tts:
        logger.info("=" * 60)
        logger.info("[STEP 6] Generating TTS audio...")

        # Load voice from registry if available
        voice_path = None
        if args.voice:
            try:
                from foundation_proprietary_voices import get_voice_path

                voice_path = get_voice_path(args.voice, index=0)
                if voice_path:
                    logger.info(f"Using voice '{args.voice}' from: {voice_path}")
            except ImportError:
                logger.warning("foundation_proprietary_voices not available, using default voice")
            except Exception as e:
                logger.warning(f"Could not load voice '{args.voice}': {e}")
        else:
            logger.info("No voice specified, using default voice from TTS model")

        target_lang_code = target_language
        tts_transcription_path = os.path.join(
            translations_dir, f"{base_name}_{target_lang_code}.json"
        )

        if os.path.exists(tts_transcription_path):
            tts_transcription = load_transcription(tts_transcription_path)
            logger.info(f"Using translated transcription for TTS: {tts_transcription_path}")
        else:
            logger.warning(
                f"Translated transcription not found: {tts_transcription_path}, using base"
            )
            tts_transcription = base_transcription

        tts_segments = generate_tts_audio(
            transcription=tts_transcription,
            voice_path=voice_path,
            language=target_lang_code,
            from_range=args.from_range,
            to_range=args.to_range,
            output_dir=segments_dir,
            eleven_model_id=args.eleven_model_id,
        )

        logger.info("[STEP 7] Mixing TTS with ducked audio...")
        mix_tts_with_audio(
            ducked_wav=ducked_wav_path,
            tts_segments=tts_segments,
            output_wav=mixed_wav_path,
            from_range=args.from_range,
            to_range=args.to_range,
            output_dir=segments_dir,
        )

    # Determine final audio for video muxing
    final_audio_path = mixed_wav_path if tts_segments else ducked_wav_path

    # logger.info("=" * 60)
    # logger.info("PIPELINE COMPLETE (Steps 1-5)")
    # logger.info("=" * 60)
    # logger.info(f"Original WAV:    {wav_path}")
    # logger.info(f"Transcription:   {json_path}")
    # logger.info(f"Ducked WAV:      {ducked_wav_path}")

    if srt_paths:
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Original WAV:  {wav_path}")
        logger.info(f"Base transcription: {json_path}")

        if args.enable_tts and tts_segments:
            logger.info(f"Ducked WAV:      {ducked_wav_path}")
            logger.info(f"Mixed WAV:      {mixed_wav_path}")
            logger.info(f"TTS segments:   {len(tts_segments)}")

        logger.info(f"Subtitles ({len(srt_paths)} languages):")
        for srt in srt_paths:
            logger.info(f"  - {srt}")
        logger.info("=" * 60)

        if args.enable_tts and tts_segments:
            logger.info("[STEP 8] Muxing final audio with original video...")
            final_video_path = os.path.join(final_dir, f"{base_name}_final.mp4")

            if args.burn_subs and srt_paths:
                muxed_video_path = os.path.join(partials_dir, f"{base_name}_muxed.mp4")
                mux_audio_with_video(
                    video_path=video_path,
                    audio_path=final_audio_path,
                    output_video_path=muxed_video_path,
                    from_range=args.from_range,
                    to_range=args.to_range,
                )

                logger.info("[STEP 9] Burning subtitles into video...")
                srt_to_burn = srt_paths[0]
                burn_subtitles(
                    video_path=muxed_video_path,
                    srt_path=srt_to_burn,
                    output_video_path=final_video_path,
                    srt_offset=args.from_range,
                )
                logger.info(f"Final video with subtitles: {final_video_path}")
            else:
                mux_audio_with_video(
                    video_path=video_path,
                    audio_path=final_audio_path,
                    output_video_path=final_video_path,
                    from_range=args.from_range,
                    to_range=args.to_range,
                )
                logger.info(f"Final video: {final_video_path}")


if __name__ == "__main__":
    asyncio.run(main())
