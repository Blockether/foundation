"""Transcription agent module.

This module provides the Transcriber Agent for processing and post-processing
audio transcriptions.

## Overlap-Based Chunking Architecture

This system uses overlapping audio chunks with context injection for accurate
speaker identification across long recordings:

### Phase 1: Chunked Transcription with Context Injection

```
Audio:  [====CHUNK 1====][====CHUNK 2====][====CHUNK 3====]
        ←───overlap───→  ←───overlap───→
```

When processing Chunk N (where N > 0):
- The agent receives `<pre_transcription_conversation>` containing the END of
  Chunk N-1's transcription
- This context is used ONLY for speaker identification (matching voices to names)
- The agent outputs ONLY the content from Chunk N's audio

### Phase 2: Agentic Pairwise Reduction

After all chunks are transcribed, they are merged via pairwise reduction:

```
[Chunk 1] + [Chunk 2] → MERGE_AGENT → [Merged 1-2]
                       (deduplicate overlap, match speakers)

[Merged 1-2] + [Chunk 3] → MERGE_AGENT → [Merged 1-2-3]
```

The MERGE_AGENT:
1. Identifies overlapping content (same audio transcribed in both chunks)
2. Deduplicates the overlap (keeps accumulated version)
3. Matches speakers across the boundary
4. Outputs a unified `TranscriptionResult`

### Phase 3: Deterministic Cleanup

After agentic reduction, deterministic operations can be applied:
- Sort by timestamp
- Validate continuity
- Compute statistics (already a computed field)

### Key Design Decisions

1. **No MergedResult wrapper**: MERGE_AGENT outputs `TranscriptionResult` directly
2. **Context via XML**: Previous transcription passed as `<pre_transcription_conversation>`
3. **Date as context**: Recording date passed upfront, not patched afterward
4. **Sequential processing**: Required for context passing between chunks

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
The audio transcription hooks (`TranscriptionHooksConfig`) automatically format transcription
results as XML and inject them into the agent's context. This ensures that LLMs
receive structured, consistent transcription data for processing.

The XML format is used by:
- Audio transcription hooks for real-time processing
- The `format_transcription_for_context()` function for manual formatting
- All transcription data passed to LLMs in the system

Note: The system always preserves the original language of the audio and does
not translate content during transcription processing.
"""

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

from ..asr import TEN_MINUTES, AudioTranscriberProtocol, split_audio_into_chunks
from ..utils import dataclass_copy
from .hooks import TranscriptionHooksConfig

DEFAULT_OVERLAP = 60.0


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
    name: str = Field(
        ...,
        description="The name of the participant. MUST be a clean name only (e.g., 'John Smith', 'Anna'). "
        "NO parentheses, NO roles, NO descriptions. Just the name and optional surname.",
    )
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

    def with_timestamp_offset(self, offset_seconds: float) -> "TranscriptionResult":
        """Return a new TranscriptionResult with all timestamps offset by the given amount.

        This is used to convert chunk-relative timestamps to absolute timestamps
        before merging. Timestamps from ASR are precise measurements and should
        NEVER be adjusted by an LLM - only deterministic offset is valid.

        Args:
            offset_seconds: The offset to add to all timestamps (chunk start time)

        Returns:
            New TranscriptionResult with adjusted timestamps
        """
        if offset_seconds == 0.0:
            return self

        adjusted_conversation = [
            DialogueLine(
                speaker=line.speaker,
                text=line.text,
                timerange=Timerange(
                    start=line.timerange.start + offset_seconds,
                    end=line.timerange.end + offset_seconds,
                ),
            )
            for line in self.conversation
        ]

        return TranscriptionResult(
            participants=self.participants,
            conversation=adjusted_conversation,
            date=self.date,
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
    audio_transcriber: AudioTranscriberProtocol,
    output_dir: str = ".",
    chunk_duration: float = TEN_MINUTES,
    overlap: float = 60.0,
    input: str | None = None,
    save_raw_transcription: bool = False,
    save_dir: str | None = None,
    **agent_kwargs: Any,
) -> None:
    """
    Internal function to process audio files with overlap-based chunking.

    Architecture:
    1. Audio is split into OVERLAPPING chunks (overlap for speaker context)
    2. Each chunk (except first) receives <pre_transcription_conversation> context
       from the previous chunk's transcription for speaker identification
    3. Individual chunk results are saved
    4. Results are merged agentically with overlap deduplication
    5. Chunks are processed SEQUENTIALLY to enable context passing

    Args:
        glob_pattern: Pattern to match audio files
        audio_transcriber: Transcriber instance
        output_dir: Directory for output files
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap duration between chunks in seconds (default: 60s)
        input: Additional instructions for the agent
        save_raw_transcription: Whether to save raw transcriptions
        save_dir: Directory for raw transcriptions
        **agent_kwargs: Additional arguments for the agent
    """
    files = glob.glob(glob_pattern)

    if not files:
        log_warning(f"No files found matching pattern: {glob_pattern}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Initialize pre_hooks list
    hooks_list: list[Any] = []

    # Add pre_hooks from parameter if provided
    if agent_kwargs.get("pre_hooks"):
        pre_hooks = agent_kwargs.pop("pre_hooks")
        hooks_list.extend(pre_hooks)

    # Add raw transcription hook if requested
    if save_raw_transcription:
        config = TranscriptionHooksConfig(
            save_transcriptions=True,
            transcriber=audio_transcriber,
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

            # Split audio into overlapping chunks
            chunks = split_audio_into_chunks(
                audio_bytes, chunk_duration=chunk_duration, overlap=overlap
            )
            base_name = Path(file_path).stem

            log_info(
                f"Split {file_path} into {len(chunks)} chunks "
                f"(duration={chunk_duration / 60:.1f}min, overlap={overlap:.0f}s)"
            )

            async def process_chunk(
                chunk_data: tuple[bytes, float, float],
                chunk_index: int,
                chunks_ref: list[tuple[bytes, float, float]],
                base_name_ref: str,
                file_path_ref: str,
                orig_date: str | None,
                previous_result: TranscriptionResult | None,
            ) -> tuple[TranscriptionResult, float, float] | None:
                """Process a single chunk with the agent.

                Args:
                    chunk_data: Tuple of (audio_bytes, start_time, end_time)
                    chunk_index: Index of this chunk (0-based)
                    chunks_ref: Reference to all chunks list
                    base_name_ref: Base filename
                    file_path_ref: Full file path
                    orig_date: Original file date from metadata
                    previous_result: Transcription result from previous chunk (for context)
                """
                chunk_bytes, start_time, end_time = chunk_data

                chunk_duration_actual = end_time - start_time
                log_info(
                    f"Processing chunk {chunk_index + 1}/{len(chunks_ref)}: "
                    f"{start_time / 60:.1f}min - {end_time / 60:.1f}min ({chunk_duration_actual / 60:.1f}min)"
                )

                chunk_path = None
                try:
                    # Create temporary audio chunk
                    chunk_filename = f"{base_name_ref}_chunk_{chunk_index + 1:03d}.wav"
                    chunk_path = os.path.join(output_dir, chunk_filename)

                    # Save chunk to temporary file
                    with open(chunk_path, "wb") as chunk_file:
                        chunk_file.write(chunk_bytes)

                    # Build input with context
                    chunk_input = final_input

                    # Add original date as context (instead of patching afterward)
                    if orig_date:
                        chunk_input = (
                            f"<recording_metadata>\n"
                            f"  <date>{orig_date}</date>\n"
                            f"</recording_metadata>\n\n{chunk_input}"
                        )

                    # Add previous chunk's transcription as context for speaker identification
                    if previous_result and chunk_index > 0:
                        # Format the previous transcription for context
                        # Include the last portion that corresponds to the overlap
                        prev_context_lines: list[str] = []
                        for line in previous_result.conversation[-10:]:  # Last 10 lines for context
                            prev_context_lines.append(
                                f'  <line speaker="{line.speaker}" '
                                f'start="{line.timerange.start:.1f}" '
                                f'end="{line.timerange.end:.1f}">{line.text}</line>'
                            )

                        prev_participants = ", ".join(
                            f"{p.name} ({p.role})" for p in previous_result.participants
                        )

                        pre_context = (
                            "<pre_transcription_conversation>\n"
                            "  <!-- This is the END of the PREVIOUS chunk's transcription -->\n"
                            "  <!-- Use this ONLY for speaker identification - DO NOT include in output! -->\n"
                            "  <!-- The audio you're transcribing OVERLAPS with this content -->\n"
                            f"  <participants>{prev_participants}</participants>\n"
                            "  <recent_dialogue>\n"
                            + "\n".join(prev_context_lines)
                            + "\n  </recent_dialogue>\n"
                            "</pre_transcription_conversation>\n\n"
                        )
                        chunk_input = pre_context + chunk_input
                        log_info(
                            f"Injected context from chunk {chunk_index} "
                            f"({len(prev_context_lines)} lines, participants: {prev_participants})"
                        )

                    # Process with Agent
                    response = await agent.arun(chunk_input, audio=[Audio(filepath=chunk_path)])

                    if response and response.content:
                        result_model = cast(TranscriptionResult, response.content)

                        # Always save individual chunk result
                        chunk_output_path = os.path.join(
                            output_dir, f"{base_name_ref}_chunk_{chunk_index + 1:03d}.json"
                        )
                        with open(chunk_output_path, "w") as f:
                            f.write(result_model.model_dump_json(indent=2))
                        log_info(f"Saved chunk {chunk_index + 1} result to {chunk_output_path}")

                        log_info(
                            f"Chunk {chunk_index + 1} processed successfully: "
                            f"{len(result_model.conversation)} dialogue lines"
                        )
                        return (result_model, start_time, end_time)
                    else:
                        log_warning(
                            f"Agent failed to process chunk {chunk_index + 1} for {file_path_ref}"
                        )
                        return None

                except Exception as e:
                    log_error(f"Error processing chunk {chunk_index + 1} for {file_path_ref}: {e}")
                    return None
                finally:
                    # Clean up temporary chunk file
                    if chunk_path and os.path.exists(chunk_path):
                        try:
                            os.remove(chunk_path)
                        except Exception:
                            pass

            # Process all chunks sequentially to enable context passing between chunks
            log_info(
                f"Starting sequential processing of {len(chunks)} chunks "
                f"(sequential for context passing and Whisper concurrency)"
            )
            chunk_results: list[tuple[TranscriptionResult, float, float]] = []
            previous_result: TranscriptionResult | None = None

            for i, chunk_data in enumerate(chunks):
                chunk_duration_actual = chunk_data[2] - chunk_data[1]
                log_info(
                    f"Processing chunk {i + 1}/{len(chunks)}: "
                    f"{chunk_data[1] / 60:.1f}min - {chunk_data[2] / 60:.1f}min "
                    f"({chunk_duration_actual / 60:.1f}min)"
                )

                result = await process_chunk(
                    chunk_data, i, chunks, base_name, file_path, original_date, previous_result
                )
                if result is not None:
                    chunk_results.append(result)
                    # Store this result for context in next chunk
                    previous_result = result[0]
                else:
                    log_warning(f"Chunk {i + 1} failed to process, will skip")

            if not chunk_results:
                log_error(f"All chunks failed to process for {file_path}")
                continue

            log_info(
                f"Completed sequential processing: {len(chunk_results)}/{len(chunks)} chunks successful"
            )

            # Agentic pairwise reduction of overlapping chunks
            if len(chunk_results) > 1:
                log_info(
                    f"Agentic pairwise reduction of {len(chunk_results)} overlapping chunks "
                    f"(overlap={overlap:.0f}s)"
                )
                # Sort chunks by start time to maintain order
                chunk_results.sort(key=lambda chunk: chunk[1])

                first_result, first_start, _ = chunk_results[0]
                merged_result = first_result.with_timestamp_offset(first_start)

                merge_agent_instance = dataclass_copy(MERGE_AGENT, **agent_kwargs)

                for i in range(1, len(chunk_results)):
                    chunk_result, chunk_start, _ = chunk_results[i]
                    chunk_with_absolute_timestamps = chunk_result.with_timestamp_offset(chunk_start)
                    merged_result = await merge_transcription_results_with_agent(
                        accumulated=merged_result,
                        new_chunk=chunk_with_absolute_timestamps,
                        chunk_index=i,
                        agent=merge_agent_instance,
                        overlap_duration=overlap,
                    )

                log_info("Agentic reduction complete - now deterministic cleanup can be applied")
            else:
                first_result, first_start, _ = chunk_results[0]
                merged_result = first_result.with_timestamp_offset(first_start)

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
    retries=3,
    delay_between_retries=2,
    exponential_backoff=True,
    instructions="""
You are an expert transcription post-processor. Your task is to take a raw audio transcription and transform it into a polished, diarized, and logically coherent conversation.

**CRITICAL REQUIREMENT: Participants MUST Match Speakers EXACTLY**
This is a HARD REQUIREMENT - your output will be REJECTED if violated:
- The `participants` list MUST contain an entry for EVERY unique speaker in `conversation`
- Each `speaker` name in dialogue lines MUST have a matching `Participant` with the EXACT SAME `name`
- NO speaker can exist in conversation without a corresponding participant entry
- NO participant can exist without appearing as a speaker in conversation

Example of CORRECT output:
```json
{
  "participants": [
    {"name": "Dr. Smith", "role": "Interviewer"},
    {"name": "John", "role": "Interviewee"}
  ],
  "conversation": [
    {"speaker": "Dr. Smith", "text": "...", ...},
    {"speaker": "John", "text": "...", ...}
  ]
}
```

Example of INCORRECT output (will be REJECTED):
```json
{
  "participants": [],  // WRONG: empty but speakers exist!
  "conversation": [
    {"speaker": "Dr. Smith", "text": "...", ...}
  ]
}
```

**CRITICAL: Participant and Speaker Name Format**
- Participant `name` and DialogueLine `speaker` MUST be CLEAN names only
- CORRECT: "John Smith", "Anna", "Dr. Smith", "Director"
- WRONG: "John Smith (Manager)", "Anna (the assistant)", "Dr. Smith - Interviewer"
- NO parentheses, NO dashes with roles, NO descriptions in the name
- Put roles in the separate `role` field, NOT in the `name` field

**BE MINDFUL OF TRANSCRIPTION QUALITY**:
Pay extreme attention to the accuracy and quality of the transcription:
- Preserve the exact meaning and intent of the original speech
- Fix grammatical errors while maintaining the speaker's voice and style
- Add proper punctuation for readability
- Ensure sentences flow naturally and make logical sense
- Use phonetic analysis to correct misheard words
- Maintain the original language - NO TRANSLATION - IF NOT SPECIFIED OTHERWISE!

**CRITICAL: Handling <pre_transcription_conversation> Context**
When processing chunked audio, you may receive a `<pre_transcription_conversation>` block containing
the transcription from the PREVIOUS chunk. This context is provided ONLY for:
- **Speaker identification**: Use it to match voices to existing participant names
- **Conversation continuity**: Understand who was speaking and the topic context

**DO NOT include any content from `<pre_transcription_conversation>` in your output!**
It is reference-only context. Your output should ONLY contain dialogue from the current audio chunk.

When `<pre_transcription_conversation>` is provided:
1. REUSE the EXACT participant names from the context for matching voices
2. Add the same participants to YOUR participants list
3. Ensure speaker names in your output EXACTLY match the context's participant names

**Core Tasks:**
1.  **Diarization**:
    *   Analyze the text to distinguish between different speakers.
    *   Assign labels (e.g., Speaker A, Speaker B) or names if they can be inferred from the context.
    *   **If `<pre_transcription_conversation>` is provided, REUSE participant names from it for matching voices!**
    *   Group consecutive sentences by the same speaker together.
    *   **ALWAYS populate the participants list with ALL speakers!**

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
**REMEMBER: participants list MUST be populated with ALL speakers!**

**Timestamp Format:**
The timerange includes seconds (for calculations) and these computed properties for human-readable format:
- `start_formatted`: Start time as HH:MM:SS
- `end_formatted`: End time as HH:MM:SS
- `duration_formatted`: Duration as HH:MM:SS
""",
    expected_output="""
Return a TranscriptionResult JSON object with:
1. `participants`: List of Participant objects - MUST contain ALL speakers with matching names
2. `conversation`: List of DialogueLine objects with speaker, text, and timerange
3. `date`: Optional date string in DD-MM-YYYY format if inferable

CRITICAL VALIDATION:
- Every unique `speaker` in conversation MUST have a matching entry in `participants`
- Every `participants` entry MUST appear as a `speaker` in conversation
- Failure to match participants to speakers will cause REJECTION
""",
    debug_mode=False,
)


class MergeDecision(BaseModel):
    """Agent's decision on how to merge two overlapping transcription chunks.

    Uses unified indexing: accumulated lines are 0..N-1, new_chunk lines are N..M.
    """

    duplicate_indices: list[int] = Field(
        ...,
        description="Global indices of lines that are duplicates (to skip). "
        "These will be in the new_chunk range (N and above).",
    )
    speaker_corrections: dict[str, str] = Field(
        default_factory=dict,
        description="Speaker corrections by global index. "
        "Key is global line index as string, value is corrected speaker name.",
    )


def apply_merge_decision(
    accumulated: TranscriptionResult,
    new_chunk: TranscriptionResult,
    decision: MergeDecision,
) -> TranscriptionResult:
    """Apply a MergeDecision using unified indexing (accumulated=0..N-1, new_chunk=N..)."""
    accumulated_len = len(accumulated.conversation)
    duplicate_set = set(decision.duplicate_indices)

    merged_conversation: list[DialogueLine] = []
    all_speakers: set[str] = set()

    for i, line in enumerate(accumulated.conversation):
        global_idx = i
        speaker = decision.speaker_corrections.get(str(global_idx), line.speaker)
        all_speakers.add(speaker)
        merged_conversation.append(
            DialogueLine(speaker=speaker, text=line.text, timerange=line.timerange)
        )

    for i, line in enumerate(new_chunk.conversation):
        global_idx = accumulated_len + i
        if global_idx in duplicate_set:
            continue
        speaker = decision.speaker_corrections.get(str(global_idx), line.speaker)
        all_speakers.add(speaker)
        merged_conversation.append(
            DialogueLine(speaker=speaker, text=line.text, timerange=line.timerange)
        )

    existing_participants = {p.name: p for p in accumulated.participants}
    for p in new_chunk.participants:
        if p.name not in existing_participants:
            existing_participants[p.name] = p

    final_participants: list[Participant] = []
    for speaker in all_speakers:
        if speaker in existing_participants:
            final_participants.append(existing_participants[speaker])
        else:
            final_participants.append(Participant(name=speaker, role="Unknown"))

    return TranscriptionResult(
        participants=final_participants,
        conversation=merged_conversation,
        date=accumulated.date or new_chunk.date,
    )


MERGE_AGENT = Agent(
    id="merge-agent",
    name="Merge Agent",
    output_schema=MergeDecision,
    retries=3,
    delay_between_retries=2,
    exponential_backoff=True,
    instructions="""
You analyze two OVERLAPPING transcription chunks and output a MergeDecision.

**UNIFIED INDEXING:**
Lines use GLOBAL indices across both chunks:
- Accumulated: indices 0, 1, 2, ..., N-1
- New chunk: indices N, N+1, N+2, ...
The exact offset N is provided in <merge_metadata>.

**Your ONLY tasks:**
1. Identify DUPLICATE lines (from new_chunk region) - add their GLOBAL indices to `duplicate_indices`
2. Identify speaker corrections - add to `speaker_corrections` using GLOBAL indices

**Overlap Deduplication:**
The first few lines of new_chunk overlap with the last few lines of accumulated.
Compare TEXT content to find duplicates. Add GLOBAL indices of duplicates to `duplicate_indices`.

**Speaker Corrections:**
If any line (in either chunk) has wrong speaker, add correction:
```json
{"speaker_corrections": {"5": "Dr. Smith", "23": "Anna"}}
```
Use GLOBAL index as string key.
""",
    expected_output="""
Return a MergeDecision JSON:
```json
{
  "duplicate_indices": [15, 16, 17],
  "speaker_corrections": {"5": "Dr. Smith", "18": "Anna"}
}
```
- `duplicate_indices`: GLOBAL indices of duplicate lines (from new_chunk region)
- `speaker_corrections`: GLOBAL index → correct speaker name
""",
    debug_mode=False,
)


class MergeValidationResult(BaseModel):
    """Result of merge validation check."""

    is_valid: bool = Field(..., description="Whether the merge result passed validation")
    errors: list[str] = Field(default_factory=list, description="List of validation errors")

    def __bool__(self) -> bool:
        return self.is_valid


def _validate_merge_result(
    result: TranscriptionResult,
    accumulated: TranscriptionResult,
    new_chunk: TranscriptionResult,
    overlap_duration: float,
) -> MergeValidationResult:
    """Validate a merge result for correctness.

    Checks:
    1. Participants match speakers exactly
    2. Timestamps are monotonically non-decreasing
    3. Entry count is reasonable (not losing too many lines)
    4. Time coverage is approximately correct

    Args:
        result: The merged transcription result to validate
        accumulated: Original accumulated transcription
        new_chunk: Original new chunk transcription
        overlap_duration: Duration of overlap in seconds

    Returns:
        MergeValidationResult with is_valid=True if valid, or errors list if invalid
    """
    errors: list[str] = []

    # 1. Check participants match speakers
    speakers_in_conversation = {line.speaker for line in result.conversation}
    participant_names = {p.name for p in result.participants}

    if speakers_in_conversation != participant_names:
        missing = speakers_in_conversation - participant_names
        extra = participant_names - speakers_in_conversation
        if missing:
            errors.append(f"Speakers without participant entries: {missing}")
        if extra:
            errors.append(f"Participants without speaker entries: {extra}")

    # 2. Check timestamps are STRICTLY monotonically non-decreasing
    if result.conversation:
        prev_start = -1.0
        for i, line in enumerate(result.conversation):
            if line.timerange.start < prev_start:
                errors.append(
                    f"Timestamp order violation at line {i}: "
                    f"start={line.timerange.start:.2f}s comes after previous start={prev_start:.2f}s"
                )
                break  # Only report first violation
            prev_start = line.timerange.start

    # 3. Check time coverage is correct
    if result.conversation and accumulated.conversation and new_chunk.conversation:
        # Result MUST start at or before accumulated start
        result_start = min(line.timerange.start for line in result.conversation)
        accumulated_start = min(line.timerange.start for line in accumulated.conversation)

        if result_start > accumulated_start:
            errors.append(
                f"Result start time is AFTER accumulated start: "
                f"result starts at {result_start:.2f}s, accumulated starts at {accumulated_start:.2f}s"
            )

        # Result MUST end at or after accumulated end (we're adding content, not removing)
        result_end = max(line.timerange.end for line in result.conversation)
        accumulated_end = max(line.timerange.end for line in accumulated.conversation)

        if result_end < accumulated_end:
            errors.append(
                f"Result end time is BEFORE accumulated end: "
                f"result ends at {result_end:.2f}s, accumulated ends at {accumulated_end:.2f}s"
            )

    return MergeValidationResult(is_valid=len(errors) == 0, errors=errors)


def _transcription_result_to_xml(
    result: TranscriptionResult,
    tag_name: str = "transcription",
    index_offset: int = 0,
) -> str:
    """Convert TranscriptionResult to XML with global line indices."""
    lines: list[str] = []
    lines.append(f"<{tag_name}>")

    if result.date:
        lines.append(f"  <date>{result.date}</date>")

    lines.append("  <participants>")
    for p in result.participants:
        lines.append(f'    <participant name="{p.name}" role="{p.role}" />')
    lines.append("  </participants>")

    lines.append("  <conversation>")
    for idx, line in enumerate(result.conversation):
        global_idx = index_offset + idx
        lines.append(
            f'    <line index="{global_idx}" speaker="{line.speaker}" '
            f'start="{line.timerange.start:.2f}" '
            f'end="{line.timerange.end:.2f}">{line.text}</line>'
        )
    lines.append("  </conversation>")

    lines.append(f"</{tag_name}>")
    return "\n".join(lines)


def _build_continuation_context(
    result: TranscriptionResult, max_duration_seconds: float = 60.0
) -> str:
    """Build XML continuation context from the last N seconds of a transcription.

    This context is passed to the merge agent to help match speakers at chunk boundaries.
    Uses TIME-BASED filtering to ensure we get exactly the last `max_duration_seconds`
    of dialogue, not just a fixed number of lines.

    Args:
        result: The transcription result to extract context from
        max_duration_seconds: Maximum duration of context to include (default: 60 seconds)

    Returns:
        XML-formatted string with participants and recent dialogue
    """
    # Get participant info
    participants_str = ", ".join(f"{p.name} ({p.role})" for p in result.participants)

    # Find the end time of the conversation
    if not result.conversation:
        return (
            "<continuation_context>\n"
            f"  <participants>{participants_str}</participants>\n"
            "  <recent_dialogue></recent_dialogue>\n"
            "</continuation_context>"
        )

    end_time = max(line.timerange.end for line in result.conversation)
    cutoff_time = end_time - max_duration_seconds

    # Get dialogue lines within the last max_duration_seconds
    recent_lines: list[str] = []
    for line in result.conversation:
        # Include lines that END after the cutoff (they overlap with our window)
        if line.timerange.end >= cutoff_time:
            recent_lines.append(
                f'    <line speaker="{line.speaker}" '
                f'start="{line.timerange.start:.1f}" '
                f'end="{line.timerange.end:.1f}">{line.text}</line>'
            )

    context = (
        "<continuation_context>\n"
        f"  <!-- Last {max_duration_seconds:.0f} seconds of accumulated transcription -->\n"
        "  <!-- Use for speaker matching at chunk boundaries -->\n"
        f"  <participants>{participants_str}</participants>\n"
        "  <recent_dialogue>\n" + "\n".join(recent_lines) + "\n  </recent_dialogue>\n"
        "</continuation_context>"
    )
    return context


async def merge_transcription_results_with_agent(
    accumulated: TranscriptionResult,
    new_chunk: TranscriptionResult,
    chunk_index: int,
    agent: Agent,
    overlap_duration: float = 0.0,
    max_retries: int = 2,
) -> TranscriptionResult:
    """Merge two overlapping transcription results using an LLM agent.

    The agent outputs a MergeDecision (duplicate indices + speaker corrections),
    then we apply it deterministically. Timestamps are NEVER touched by the agent.
    """
    # Build continuation context from the last portion of accumulated
    continuation_context = _build_continuation_context(accumulated)

    log_info(
        f"Merging chunk {chunk_index + 1}: "
        f"accumulated has {len(accumulated.participants)} participants, "
        f"new_chunk has {len(new_chunk.participants)} participants"
    )

    # Build base input string with all context as XML
    accumulated_xml = _transcription_result_to_xml(accumulated, "accumulated_transcription")
    new_chunk_xml = _transcription_result_to_xml(new_chunk, "new_chunk_transcription")

    accumulated_len = len(accumulated.conversation)
    new_chunk_xml = _transcription_result_to_xml(
        new_chunk, "new_chunk_transcription", index_offset=accumulated_len
    )

    base_input = f"""Analyze these overlapping transcriptions and output a MergeDecision.

<merge_metadata>
  <accumulated_indices>0 to {accumulated_len - 1}</accumulated_indices>
  <new_chunk_indices>{accumulated_len} to {accumulated_len + len(new_chunk.conversation) - 1}</new_chunk_indices>
  <overlap_duration>{overlap_duration:.1f}</overlap_duration>
</merge_metadata>

{continuation_context}

{accumulated_xml}

{new_chunk_xml}
"""

    last_validation: MergeValidationResult | None = None

    for attempt in range(max_retries + 1):
        if attempt == 0:
            input_str = base_input
        else:
            validation_errors = "\n".join(
                f"  - {e}" for e in (last_validation.errors if last_validation else [])
            )
            input_str = f"""{base_input}

<previous_attempt_failed>
Your previous MergeDecision failed validation. Please fix these errors:
{validation_errors}

Check your duplicate_indices and speaker_corrections carefully.
</previous_attempt_failed>
"""
            log_warning(
                f"Merge validation failed, retrying (attempt {attempt + 1}/{max_retries + 1})"
            )

        response = await agent.arun(input_str)

        decision = cast(MergeDecision, response.content)

        result = apply_merge_decision(accumulated, new_chunk, decision)

        validation = _validate_merge_result(result, accumulated, new_chunk, overlap_duration)

        if validation.is_valid:
            if attempt > 0:
                log_info(f"Merge succeeded after {attempt + 1} attempts")
            log_info(
                f"Merge decision: {len(decision.duplicate_indices)} duplicates, "
                f"{len(decision.speaker_corrections)} speaker corrections"
            )
            return result

        last_validation = validation
        log_warning(f"Merge validation errors: {validation.errors}")

    error_msg = (
        f"Merge validation failed after {max_retries + 1} attempts for chunk {chunk_index}. "
        f"Errors: {last_validation.errors if last_validation else 'unknown'}"
    )
    log_error(error_msg)
    raise ValueError(error_msg)


# Default overlap duration for chunked processing (60 seconds)


async def process_audio_files(
    glob_pattern: str,
    audio_transcriber: AudioTranscriberProtocol,
    output_dir: str = ".",
    input: str | None = None,
    save_raw_transcription: bool = False,
    save_dir: str | None = None,
    audio_chunking: bool = True,
    chunk_duration: float = TEN_MINUTES,
    overlap: float = DEFAULT_OVERLAP,
    **agent_kwargs: Any,
) -> None:
    """
    Process audio files with overlap-based chunking for accurate speaker identification.

    Architecture (when audio_chunking=True):
    1. Audio is split into OVERLAPPING chunks
    2. Each chunk receives context from previous chunk's transcription
       via <pre_transcription_conversation> for speaker identification
    3. Chunks are processed sequentially to enable context passing
    4. Results are merged via agentic pairwise reduction (overlap deduplication)
    5. Deterministic cleanup is applied after agentic reduction

    When audio_chunking=False:
    - No chunking - process files as single units
    - Suitable for short files that fit in memory

    Args:
        glob_pattern: Glob pattern to match audio files (e.g., "data/*.mp3").
        output_dir: Directory to save the JSON transcripts.
        audio_transcriber: AudioTranscriberProtocol instance for processing.
        input: Additional instructions for the agent.
        save_raw_transcription: Whether to save raw transcription outputs.
        save_dir: Directory to save raw transcriptions.
        audio_chunking: Whether to use audio chunking (default: True).
        chunk_duration: Duration for each chunk in seconds (default: 10 minutes).
        overlap: Overlap duration between chunks in seconds (default: 60s).
                 The overlap enables speaker identification across chunk boundaries.
    """
    # Check if any files match the pattern
    files = glob.glob(glob_pattern)
    if not files:
        log_warning(f"No files found matching pattern: {glob_pattern}")
        return

    if audio_chunking:
        # Use overlap-based chunking for speaker identification
        log_info(
            f"Using overlap-based chunking: {chunk_duration / 60:.1f}min chunks, "
            f"{overlap:.0f}s overlap"
        )
        await _process_with_chunking(
            glob_pattern=glob_pattern,
            output_dir=output_dir,
            chunk_duration=chunk_duration,
            overlap=overlap,
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
            overlap=0.0,  # No overlap needed when not chunking
            audio_transcriber=audio_transcriber,
            input=input,
            save_raw_transcription=save_raw_transcription,
            save_dir=save_dir,
            **agent_kwargs,
        )
