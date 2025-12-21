"""Transcription agent module.

This module provides the Transcriber Agent for processing and post-processing
audio transcriptions.

## Dual-Level Audio Splitting Architecture

This system implements two levels of audio splitting for maximum reliability:

Level 1: Transcription-Level (via Audio Hooks)
- Splits audio into 10-minute chunks for Whisper
- Handles Whisper's 10-minute limitation
- Merges results before agent processing

Level 2: Agent-Level (via process_audio_files_chunked)
- Splits audio into configurable agent chunks
- Each agent chunk uses Level 1 splitting internally
- Multiple agent runs, then merge results

Flow Example:
Audio → Agent Split → Each chunk: Audio Hook Split → Whisper → Agent → Merge All → Result

## XML Output Format

The transcription system outputs structured XML data that follows the same pattern
as graph entities and queries in the codebase. This ensures consistency for LLM
consumption.

### XML Structure
```xml
<transcription>
  <metadata language="en" language_probability="0.980" total_duration="5.200"
            segment_count="2" word_count="15" created_at="2025-01-15T10:30:00+00:00">
    <segment index="1" start="0.000" end="2.500" duration="2.500"
             speaker="Speaker A">
      <text>Hello world, this is a test.</text>
      <words>
        <word start="0.000" end="0.500" score="0.980">Hello</word>
        <word start="0.600" end="1.000" score="0.950">world</word>
        <word start="1.100" end="1.400" score="0.960">this</word>
        <word start="1.500" end="1.700" score="0.940">is</word>
        <word start="1.800" end="1.900" score="0.970">a</word>
        <word start="2.000" end="2.300" score="0.930">test</word>
        <word start="2.300" end="2.500" score="0.990">.</word>
      </words>
    </segment>
    <segment index="2" start="2.500" end="5.200" duration="2.700" speaker="Speaker B">
      <text>This is the second segment.</text>
      <!-- Additional words would be here -->
    </segment>
  </metadata>
</transcription>
```

### Key Features
- **Metadata**: Language, detection confidence, duration, segment/word counts
- **Segments**: Time-based chunks with speaker identification
- **Words**: Individual word timestamps and confidence scores
- **Consistency**: Follows the same XML patterns as graph queries/entities

### Integration with Hooks
The audio transcription hooks (`AudioHooksConfig`) automatically format transcription
results as XML and inject them into the agent's context. This ensures that LLMs
receive structured, consistent transcription data for processing.

The XML format is used by:
- Audio transcription hooks for real-time processing
- The `format_transcription_for_context()` function for manual formatting
- All transcription data passed to LLMs in the system

Note: The system always preserves the original language of the audio and does
not translate content during transcription processing.
"""

import asyncio
import glob
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import av
from agno.agent.agent import Agent
from agno.media import Audio
from agno.utils.log import log_error, log_info, log_warning  # type: ignore
from pydantic import BaseModel, Field, computed_field

from ..audio import AudioTranscriber
from ..audio.transcription import TEN_MINUTES, split_audio_into_chunks
from ..utils import dataclass_copy
from .hooks.audio import AudioHooksConfig


def extract_audio_creation_date(file_path: str) -> str | None:
    """Extract creation date from audio file metadata using PyAV.

    Args:
        file_path: Path to the audio file

    Returns:
        Date string in DD-MM-YYYY format, or None if not found

    Raises:
        Exception: If file cannot be opened or metadata cannot be read
    """
    with av.open(file_path) as container:
        if "creation_time" in container.metadata:
            date_str = container.metadata["creation_time"]
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            result = dt.strftime("%d-%m-%Y")
            log_info(f"Recording date from metadata: {result}")
            return result
    return None


class Participant(BaseModel):
    name: str = Field(..., description="The name of the participant.")
    role: str = Field(..., description="The inferred role of the participant in the conversation.")


class Timerange(BaseModel):
    start: float = Field(..., description="Start time of the dialogue line in seconds.")
    end: float = Field(..., description="End time of the dialogue line in seconds.")

    @computed_field(return_type=str)
    @property
    def start_formatted(self) -> str:
        """Start time formatted as HH:MM:SS."""
        hours = int(self.start // 3600)
        minutes = int((self.start % 3600) // 60)
        seconds = int(self.start % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @computed_field(return_type=str)
    @property
    def end_formatted(self) -> str:
        """End time formatted as HH:MM:SS."""
        hours = int(self.end // 3600)
        minutes = int((self.end % 3600) // 60)
        seconds = int(self.end % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @computed_field(return_type=str)
    @property
    def duration_formatted(self) -> str:
        """Duration formatted as HH:MM:SS."""
        duration = self.end - self.start
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class DialogueLine(BaseModel):
    speaker: str = Field(
        ...,
        description="The name or label of the speaker (e.g., 'Speaker A', 'John'). Name preferably inferred from context. It should match the participant name",
    )
    text: str = Field(..., description="The corrected and logically repaired spoken text.")
    timerange: Timerange = Field(...)


class SpeakerStatistics(BaseModel):
    name: str = Field(..., description="Name of the speaker.")
    total_time: float = Field(..., description="Total time spoken by this speaker in seconds.")
    message_count: int = Field(..., description="Number of dialogue lines by this speaker.")
    percentage: float = Field(
        ...,
        description="Percentage of the conversation duration dominated by this speaker.",
    )


class ConversationStatistics(BaseModel):
    total_duration: float = Field(..., description="Total duration of the conversation in seconds.")
    most_active_speaker: str = Field(..., description="Name of the speaker who spoke the most.")
    speaker_stats: list[SpeakerStatistics] = Field(
        ..., description="Detailed statistics per speaker."
    )


class TranscriptionResult(BaseModel):
    participants: list[Participant] = Field(
        ..., description="List of identified participants with their names and roles."
    )
    conversation: list[DialogueLine] = Field(
        ..., description="The full diarized and corrected conversation."
    )
    date: str | None = Field(
        None, description="The date of the conversation if it can be inferred. [DD-MM-YYYY]"
    )

    @computed_field(return_type=ConversationStatistics)
    @property
    def statistics(self) -> ConversationStatistics:
        """Conversation statistics computed deterministically from `conversation`.

        This is intentionally computed (not provided by the LLM).
        """

        conversation = self.conversation
        if not conversation:
            return ConversationStatistics(
                total_duration=0.0,
                most_active_speaker="",
                speaker_stats=[],
            )

        starts = [line.timerange.start for line in conversation]
        ends = [line.timerange.end for line in conversation]
        total_duration = max(ends) - min(starts)

        speaker_time: dict[str, float] = defaultdict(float)
        speaker_messages: dict[str, int] = defaultdict(int)

        for line in conversation:
            speaker_messages[line.speaker] += 1
            duration = line.timerange.end - line.timerange.start
            if duration > 0:
                speaker_time[line.speaker] += duration

        most_active_speaker = (
            max(speaker_time.keys(), key=lambda speaker: speaker_time[speaker])
            if speaker_time
            else ""
        )

        stats: list[SpeakerStatistics] = []
        for speaker, total_time in sorted(
            speaker_time.items(), key=lambda item: item[1], reverse=True
        ):
            percentage = (total_time / total_duration * 100.0) if total_duration > 0 else 0.0
            stats.append(
                SpeakerStatistics(
                    name=speaker,
                    total_time=total_time,
                    message_count=speaker_messages.get(speaker, 0),
                    percentage=percentage,
                )
            )

        return ConversationStatistics(
            total_duration=total_duration,
            most_active_speaker=most_active_speaker,
            speaker_stats=stats,
        )


def merge_transcription_results(results: list[TranscriptionResult]) -> TranscriptionResult:
    """Merge multiple TranscriptionResult objects into a single result.

    Args:
        results: List of TranscriptionResult objects to merge

    Returns:
        Merged TranscriptionResult with combined conversation and statistics
    """
    if not results:
        raise ValueError("No results to merge")

    if len(results) == 1:
        return results[0]

    # Combine all conversation lines
    all_conversation: list[DialogueLine] = []
    all_participants: dict[str, Participant] = {}
    chunk_offset = 0.0  # Track time offset for each chunk

    for i, result in enumerate(results):
        # Calculate chunk duration from conversation if statistics not available
        chunk_duration = 0.0
        if result.statistics.total_duration > 0:
            chunk_duration = result.statistics.total_duration
        elif result.conversation:
            # Calculate from last dialogue line
            last_line = max(result.conversation, key=lambda line: line.timerange.end)
            chunk_duration = last_line.timerange.end

        # Adjust timestamps for each chunk
        for line in result.conversation:
            # Create new dialogue line with adjusted timestamps
            adjusted_line = DialogueLine(
                speaker=line.speaker,
                text=line.text,
                timerange=Timerange(
                    start=line.timerange.start + chunk_offset, end=line.timerange.end + chunk_offset
                ),
            )
            all_conversation.append(adjusted_line)

        # Collect unique participants
        for participant in result.participants:
            all_participants[participant.name] = participant

        # Update offset for next chunk
        chunk_offset += chunk_duration
        log_info(f"Chunk {i + 1}: duration={chunk_duration:.2f}s, new offset={chunk_offset:.2f}s")

    # Sort conversation by start time
    all_conversation.sort(key=lambda line: line.timerange.start)

    # Create merged result
    merged_result = TranscriptionResult(
        participants=list(all_participants.values()),
        conversation=all_conversation,
        date=results[0].date,  # Use date from first result
    )

    return merged_result


async def _process_with_chunking(
    glob_pattern: str,
    output_dir: str = ".",
    chunk_duration: float = TEN_MINUTES,
    concurrency: int = 4,
    audio_transcriber: AudioTranscriber | None = None,
    input: str | None = None,
    save_raw_transcription: bool = False,
    save_dir: str | None = None,
    **agent_kwargs: Any,
) -> None:
    """
    Internal function to process audio files with unified chunking and parallel processing.

    1. Audio is split into chunks for agent processing
    2. Each chunk is transcribed as a whole unit (no further splitting)
    3. Individual chunk results are always saved
    4. Results are merged into final output
    5. Chunks are processed in PARALLEL for speed
    """
    files = glob.glob(glob_pattern)

    if not files:
        log_warning(f"No files found matching pattern: {glob_pattern}")
        return

    if not audio_transcriber:
        audio_transcriber = AudioTranscriber.get_instance()

    os.makedirs(output_dir, exist_ok=True)

    # Initialize pre_hooks list
    hooks_list: list[Any] = []

    # Add pre_hooks from parameter if provided
    if agent_kwargs.get("pre_hooks"):
        pre_hooks = agent_kwargs.pop("pre_hooks")
        hooks_list.extend(pre_hooks)

    # Add raw transcription hook if requested
    if save_raw_transcription:
        config = AudioHooksConfig(
            save_transcriptions=True,
            transcription_dir=save_dir or output_dir,
        )
        hooks_list.append(config.pre_hook())

    # Add hooks to agent_kwargs if any
    if hooks_list:
        agent_kwargs["pre_hooks"] = hooks_list

    agent = dataclass_copy(TRANSCRIBER_AGENT, **agent_kwargs)

    default_input = "Transcribe given audio."
    final_input = default_input

    if input:
        final_input += (
            "These special instructions should be given HIGHEST priority. "
            "THESE INSTRUCTIONS OVERRIDE ALL PREVIOUS INSTRUCTIONS IN CASE OF CONFLICT "
            f"<special_instructions>{input}</special_instructions>"
        )

    # Process all files
    for file_path in files:
        log_info(f"Processing {file_path} with parallel chunked agent processing...")

        try:
            # Read the audio file
            with open(file_path, "rb") as f:
                audio_bytes = f.read()

            # Extract original file metadata
            original_date = extract_audio_creation_date(file_path)

            # Split audio into chunks
            chunks = split_audio_into_chunks(audio_bytes, chunk_duration=chunk_duration)
            base_name = Path(file_path).stem

            log_info(f"Split {file_path} into {len(chunks)} chunks for agent processing")

            # Create semaphore to limit concurrent agent runs
            semaphore = asyncio.Semaphore(concurrency)

            async def process_chunk(
                chunk_data: tuple[bytes, float, float],
                chunk_index: int,
                chunks_ref: list[tuple[bytes, float, float]],
                base_name_ref: str,
                file_path_ref: str,
                semaphore_ref: asyncio.Semaphore,
                orig_date: str | None,
            ) -> tuple[TranscriptionResult, float, float] | None:
                """Process a single chunk with the agent."""
                chunk_bytes, start_time, end_time = chunk_data

                async with semaphore_ref:
                    chunk_duration_actual = end_time - start_time
                    log_info(
                        f"Processing chunk {chunk_index + 1}/{len(chunks_ref)}: {start_time / 60:.1f}min - {end_time / 60:.1f}min ({chunk_duration_actual / 60:.1f}min)"
                    )

                    chunk_path = None
                    try:
                        # Create temporary audio chunk
                        chunk_filename = f"{base_name_ref}_chunk_{chunk_index + 1:03d}.wav"
                        chunk_path = os.path.join(output_dir, chunk_filename)

                        # Save chunk to temporary file
                        with open(chunk_path, "wb") as chunk_file:
                            chunk_file.write(chunk_bytes)

                        # Process with Agent (this will use transcription-level splitting internally if needed)
                        response = await agent.arun(  # type: ignore
                            input=final_input, audio=[Audio(filepath=chunk_path)]
                        )

                        if response and response.content:
                            result_model = cast(TranscriptionResult, response.content)

                            # Update chunk result with original file date if available
                            if orig_date and not result_model.date:
                                # Create a new result with the original date
                                from pydantic import ValidationError

                                try:
                                    result_model = TranscriptionResult(
                                        participants=result_model.participants,
                                        conversation=result_model.conversation,
                                        date=orig_date,
                                    )
                                    log_info(
                                        f"Updated chunk {chunk_index + 1} with original date: {orig_date}"
                                    )
                                except ValidationError as e:
                                    log_warning(
                                        f"Could not update chunk {chunk_index + 1} with date: {e}"
                                    )

                            # Always save individual chunk result
                            chunk_output_path = os.path.join(
                                output_dir, f"{base_name_ref}_chunk_{chunk_index + 1:03d}.json"
                            )
                            with open(chunk_output_path, "w") as f:
                                f.write(result_model.model_dump_json(indent=2))
                            log_info(f"Saved chunk {chunk_index + 1} result to {chunk_output_path}")

                            log_info(
                                f"Chunk {chunk_index + 1} processed successfully: {len(result_model.conversation)} dialogue lines"
                            )
                            return (result_model, start_time, end_time)
                        else:
                            log_warning(
                                f"Agent failed to process chunk {chunk_index + 1} for {file_path_ref}"
                            )
                            return None

                    except Exception as e:
                        log_error(
                            f"Error processing chunk {chunk_index + 1} for {file_path_ref}: {e}"
                        )
                        return None
                    finally:
                        # Clean up temporary chunk file
                        if chunk_path and os.path.exists(chunk_path):
                            try:
                                os.remove(chunk_path)
                            except Exception:
                                pass

            # Process all chunks in parallel
            log_info(
                f"Starting parallel processing of {len(chunks)} chunks with concurrency {concurrency}"
            )
            tasks = [
                process_chunk(chunk_data, i, chunks, base_name, file_path, semaphore, original_date)
                for i, chunk_data in enumerate(chunks)
            ]

            # Wait for all chunks to complete
            chunk_results_with_none = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out failed chunks
            chunk_results: list[tuple[TranscriptionResult, float, float]] = []
            for result in chunk_results_with_none:
                if isinstance(result, Exception):
                    log_error(f"Chunk processing error: {result}")
                    continue
                if result is None:
                    continue
                if isinstance(result, tuple) and len(result) == 3:
                    chunk_results.append(cast(tuple[TranscriptionResult, float, float], result))

            if not chunk_results:
                log_error(f"All chunks failed to process for {file_path}")
                continue

            log_info(
                f"Completed parallel processing: {len(chunk_results)}/{len(chunks)} chunks successful"
            )

            # Always merge results into final output
            if len(chunk_results) > 1:
                log_info(f"Merging results from {len(chunk_results)} chunks")
                # Sort chunks by start time to maintain order
                chunk_results.sort(key=lambda chunk: chunk[1])
                # Extract results for merging
                results_only: list[TranscriptionResult] = [result for result, _, _ in chunk_results]
                merged_result = merge_transcription_results(results_only)
            else:
                merged_result = chunk_results[0][0]

            # Save final merged result
            output_path = os.path.join(output_dir, f"{base_name}.json")
            with open(output_path, "w") as f:
                f.write(merged_result.model_dump_json(indent=2))
            log_info(f"Saved final transcript to {output_path}")

        except Exception as e:
            log_error(f"Error processing {file_path} with parallel chunked processing: {e}")


TRANSCRIBER_AGENT = Agent(
    id="transcriber-agent",
    name="Transcriber Agent",
    output_schema=TranscriptionResult,
    instructions="""
You are an expert transcription post-processor. Your task is to take a raw audio transcription and transform it into a polished, diarized, and logically coherent conversation.

**BE MINDFUL OF TRANSCRIPTION QUALITY**:
Pay extreme attention to the accuracy and quality of the transcription:
- Preserve the exact meaning and intent of the original speech
- Fix grammatical errors while maintaining the speaker's voice and style
- Add proper punctuation for readability
- Ensure sentences flow naturally and make logical sense
- Use phonetic analysis to correct misheard words
- Maintain the original language - NO TRANSLATION - IF NOT SPECIFIED OTHERWISE!

**Core Tasks:**
1.  **Diarization**:
    *   Analyze the text to distinguish between different speakers.
    *   Assign labels (e.g., Speaker A, Speaker B) or names if they can be inferred from the context.
    *   Group consecutive sentences by the same speaker together.

2.  **Error Correction & Logic Repair**:
    *   Fix spelling and grammatical errors introduced by the transcription process.
    *   **Logic Repair**: If a sentence seems logically broken or nonsensical due to misheard words, reconstruct it to make sense within the conversation's context. Use phonetic similarity to guess the intended words.
    *   *Example*: If the transcript says "I need to *right* a letter", correct it to "write".
    *   **Slang / Synonyms / Nicknames Normalization**:
        - Resolve colloquial/slang terms, nicknames, abbreviations, and local synonyms into more standard, widely understood wording.
        - Example (Polish): if "maczek" clearly refers to a MacBook, output "MacBook" (optionally "MacBook (maczek)" on first mention if it helps preserve the speaker's voice).
        - If meaning is ambiguous, do not guess aggressively: keep the original word and make the sentence coherent without inventing new facts.
    *   **IMPORTANT**: Always maintain the original language of the audio. Do not translate the content to another language. Preserve all words, phrases, and expressions in their original language.

3.  **Statistics**:
    *   Do NOT calculate statistics in the LLM output.
    *   The system computes speaker statistics deterministically from the final diarized conversation timeranges.

**Input:**
The user will provide the raw text from a transcription service.

**Output:**
Return the result as a structured JSON object matching the defined schema.
Ensure the 'text' field in the dialogue lines is plain text, not markdown.

**Timestamp Format:**
The timerange includes seconds (for calculations) and these computed properties for human-readable format:
- `start_formatted`: Start time as HH:MM:SS
- `end_formatted`: End time as HH:MM:SS
- `duration_formatted`: Duration as HH:MM:SS
""",
    debug_mode=False,
)


async def process_audio_files(
    glob_pattern: str,
    output_dir: str = ".",
    audio_transcriber: AudioTranscriber | None = None,
    input: str | None = None,
    save_raw_transcription: bool = False,
    save_dir: str | None = None,
    audio_chunking: bool = True,
    chunk_duration: float = TEN_MINUTES,
    chunk_concurrency: int = 4,
    **agent_kwargs: Any,
) -> None:
    """
    Process audio files with unified audio chunking (both transcription and agent levels).

    When audio_chunking=True (default):
    1. Audio is split into chunks for agent processing
    2. Each chunk is further split for Whisper if needed
    3. Individual chunk results are always saved
    4. Results are merged into final output
    5. Chunks are processed in PARALLEL for speed

    When audio_chunking=False:
    - No chunking at any level (neither agent nor transcription)
    - Large files are processed as single units

    Args:
        glob_pattern: Glob pattern to match audio files (e.g., "data/*.mp3").
        output_dir: Directory to save the JSON transcripts.
        audio_transcriber: AudioTranscriber instance for processing.
        input: Additional instructions for the agent.
        save_raw_transcription: Whether to save raw transcription outputs.
        save_dir: Directory to save raw transcriptions.
        audio_chunking: Whether to use audio chunking (default: True).
        chunk_duration: Duration for each chunk in seconds (default: 10 minutes).
        chunk_concurrency: Number of chunks to process in parallel (default: 4).
    """
    # Check if any files match the pattern
    files = glob.glob(glob_pattern)
    if not files:
        log_warning(f"No files found matching pattern: {glob_pattern}")
        return

    if audio_chunking:
        # Use chunking - split audio into chunks for parallel processing
        log_info(
            f"Using audio chunking with {chunk_duration / 60:.1f}min chunks, concurrency: {chunk_concurrency}"
        )
        await _process_with_chunking(
            glob_pattern=glob_pattern,
            output_dir=output_dir,
            chunk_duration=chunk_duration,
            concurrency=chunk_concurrency,
            audio_transcriber=audio_transcriber,
            input=input,
            save_raw_transcription=save_raw_transcription,
            save_dir=save_dir,
            **agent_kwargs,
        )
    else:
        # No chunking - process files as whole units
        log_info("Processing files without chunking (audio_chunking=False)")

        # Use a very large chunk duration to effectively disable chunking
        no_chunk_duration = float("inf")  # Infinity ensures no chunks will be created

        await _process_with_chunking(
            glob_pattern=glob_pattern,
            output_dir=output_dir,
            chunk_duration=no_chunk_duration,
            concurrency=1,  # No parallel processing when chunking is disabled
            audio_transcriber=audio_transcriber,
            input=input,
            save_raw_transcription=save_raw_transcription,
            save_dir=save_dir,
            **agent_kwargs,
        )
