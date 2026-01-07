"""Transcription hooks for Agno agents."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agno.agent.agent import Agent
from agno.media import Audio
from agno.run.agent import RunInput
from agno.session import AgentSession, TeamSession
from agno.team import Team
from agno.utils.log import log_debug, log_warning  # type: ignore

from ...asr import (
    TranscriptionResult,
    format_transcription_for_context,
)
from ...utils import (
    AgnoPreHook,
    save_data_to_json_file,
)
from ...utils import (
    copy_to_clipboard as copy_to_clipboard_fn,
)

if TYPE_CHECKING:
    from ...asr import AudioTranscriberProtocol


def _prepare_transcription_data(
    result: TranscriptionResult,
    source_path: str,
) -> dict[str, Any]:
    """Prepare transcription data for JSON serialization."""
    return {
        "source_file": source_path,
        "language": result.language,
        "language_probability": result.language_probability,
        "duration": result.total_duration,
        "word_count": sum(len(segment.words) for segment in result.segments),
        "segments": [
            {
                "index": i + 1,
                "start": segment.start,
                "end": segment.end,
                "duration": segment.end - segment.start,
                "speaker": segment.speaker,
                "text": segment.text,
                "words": [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "speaker": word.speaker,
                    }
                    for word in segment.words
                ],
            }
            for i, segment in enumerate(result.segments)
        ],
    }


def _format_and_inject_transcripts(
    run_input: RunInput,
    transcripts: list[TranscriptionResult],
    max_segments: int | None,
) -> None:
    """Format transcripts and inject into run input."""
    # Store audio info before clearing
    audio_files = run_input.audios or []

    if not transcripts:
        # Still clear audio files even if no transcripts
        run_input.audios = []
        return

    # Determine header text based on audio source type
    file_count = 0
    content_count = 0

    for audio_file in audio_files:
        if audio_file.filepath:
            file_count += 1
        else:
            content_count += 1

    if content_count > 0 and file_count == 0:
        # Only audio content - show sources
        audio_ids = [audio_file.id for audio_file in audio_files if audio_file.id]
        if len(audio_ids) == 1:
            header = f"--- Audio Transcription | Source: {audio_ids[0]} ---"
        else:
            header = f"--- Audio Transcription | Sources: {', '.join(audio_ids)} ---"
    else:
        # Mixed or file-based - show filenames
        filenames = [
            Path(audio_file.filepath).name for audio_file in audio_files if audio_file.filepath
        ]
        filenames_text = ", ".join(filenames) if filenames else "audio files"
        header = f"--- Audio Transcription | {len(transcripts)} file(s): {filenames_text} ---"

    # Format all transcripts
    all_transcripts_text = ""
    for transcript in transcripts:
        transcript_text = format_transcription_for_context(
            transcription=transcript,
            max_segments=max_segments,
            include_word_timestamps=False,  # Don't include word timestamps for performance
        )
        all_transcripts_text += f"\n{transcript_text}"

    # Inject into run input
    current_content = run_input.input_content_string()
    enhanced_content = (
        f"{header}{all_transcripts_text}\n\n--- Original Message ---\n{current_content}"
    )
    run_input.input_content = enhanced_content

    # Clear the audio files after transcription
    run_input.audios = []


async def _transcribe_audio_file(
    transcriber: AudioTranscriberProtocol,
    audio_file: Audio,
    language: str | None,
    effort: float,
    async_hooks: bool = True,
) -> TranscriptionResult | None:
    """Transcribe a single audio file."""
    # Handle both content and filepath
    if audio_file.content:
        # Use audio content directly
        audio_bytes = audio_file.content
        source_desc = f"audio content ({audio_file.id})"
    elif audio_file.filepath:
        # Try to read from file
        try:
            with open(audio_file.filepath, "rb") as f:
                audio_bytes = f.read()
            source_desc = str(audio_file.filepath)
        except (FileNotFoundError, OSError):
            log_debug(f"Audio file not found: {audio_file.filepath}")
            return None
    else:
        if isinstance(audio_file, Audio) and audio_file.filepath:
            log_debug(f"No valid audio source found for: {audio_file.filepath}")
        else:
            log_debug("No valid audio source found for audio file")
        return None

    log_debug(f"Transcribing audio from: {source_desc}")

    if async_hooks:
        result: TranscriptionResult | None = asyncio.run(
            transcriber.transcribe(
                audio_bytes,
                language=language,
                effort=effort,
            )
        )
    else:
        result: TranscriptionResult | None = await transcriber.transcribe(
            audio_bytes,
            language=language,
            effort=effort,
        )

    if not result or not result.segments:
        log_debug(f"No transcription result for {source_desc}")
        return None

    log_debug(
        f"Successfully transcribed audio from {source_desc} (length: {len(result.text)} chars)"
    )

    return result


async def _save_transcription_to_file(
    result: TranscriptionResult,
    source_path: str,
    transcription_dir: str | None,
) -> None:
    """Save transcription result to file."""
    if not transcription_dir:
        return

    # Generate filename based on source audio file
    source_name = Path(source_path).stem
    output_file = f"{transcription_dir}/{source_name}_transcription.json"

    # Prepare transcription data using common function
    transcription_data = _prepare_transcription_data(result, source_path)

    # TODO: Add metadata when extract_audio_metadata is implemented
    # metadata = extract_audio_metadata(source_path)
    # if metadata:
    #     transcription_data["metadata"] = metadata

    # Save using utility function
    save_data_to_json_file(transcription_data, output_file)


def _wait_for_spacebar() -> None:
    """Wait for spacebar to be pressed (synchronous, cross-platform)."""
    import select
    import termios

    if sys.platform == "win32":
        import msvcrt

        log_debug("Press SPACE to start/stop recording...")
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b" ":
                    return
    else:
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            log_debug("Press SPACE to start/stop recording...")
            while True:
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key == " ":
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def record_interactive_audio(
    sample_rate: int = 16000,
    channels: int = 1,
    output_format: str = "wav",
) -> tuple[bytes, str]:
    """Record audio interactively with spacebar control.

    Press SPACE to start recording, press SPACE again to stop.

    Args:
        sample_rate: Audio sample rate in Hz (default: 16000 for Whisper)
        channels: Number of audio channels (default: 1 for mono)
        output_format: Output audio format (default: "wav")

    Returns:
        Tuple of (audio_bytes, filepath) where filepath is a temp file path

    Raises:
        RuntimeError: If audio recording fails
    """
    log_debug("=" * 60)
    log_debug("Interactive Audio Recording")
    log_debug("=" * 60)
    log_debug("  - Press SPACE to START recording")
    log_debug("  - Speak into your microphone")
    log_debug("  - Press SPACE to STOP recording")
    log_debug("=" * 60)

    _wait_for_spacebar()
    log_debug("🔴 Recording... Press SPACE to stop")

    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        recording_cmd, recording_args = _get_recording_command(temp_path, sample_rate, channels)

        log_debug(f"Using recording command: {recording_cmd}")

        process = subprocess.Popen(
            [recording_cmd] + recording_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if sys.platform == "win32":
            import msvcrt

            while process.poll() is None:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b" ":
                        process.terminate()
                        process.wait()
                        break
                import time

                time.sleep(0.01)
        else:
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                while process.poll() is None:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1)
                        if key == " ":
                            process.terminate()
                            process.wait()
                            break
                    import time

                    time.sleep(0.01)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        log_debug("✅ Recording stopped!")

        if not Path(temp_path).exists() or Path(temp_path).stat().st_size == 0:
            raise RuntimeError("No audio data recorded - file is empty or was not created")

        with open(temp_path, "rb") as f:
            audio_bytes = f.read()

        log_debug(f"Recorded {len(audio_bytes)} bytes to {temp_path}")
        return audio_bytes, temp_path

    except Exception as e:
        import os

        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise RuntimeError(f"Audio recording failed: {e}") from e


def _get_recording_command(
    output_path: str,
    sample_rate: int,
    channels: int,
) -> tuple[str, list[str]]:
    """Get platform-specific recording command and arguments.

    Args:
        output_path: Path to save audio file
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels

    Returns:
        Tuple of (command, args)

    Raises:
        RuntimeError: If no suitable recording tool is found
    """
    if sys.platform == "darwin":
        if shutil.which("rec"):
            return "rec", [
                "-q",
                "-r",
                str(sample_rate),
                "-c",
                str(channels),
                "-e",
                "signed",
                "-b",
                "16",
                output_path,
            ]
        elif shutil.which("ffmpeg"):
            return "ffmpeg", [
                "-f",
                "avfoundation",
                "-i",
                ":0",
                "-q:a",
                "0",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-y",
                output_path,
            ]
        else:
            raise RuntimeError(
                "No audio recording tool found on macOS. "
                "Install one of: sox (brew install sox), ffmpeg (brew install ffmpeg)"
            )

    elif sys.platform == "linux":
        if shutil.which("arecord"):
            return "arecord", [
                "-q",
                "-f",
                "S16_LE",
                "-r",
                str(sample_rate),
                "-c",
                str(channels),
                output_path,
            ]
        elif shutil.which("rec"):
            return "rec", [
                "-q",
                "-r",
                str(sample_rate),
                "-c",
                str(channels),
                "-e",
                "signed",
                "-b",
                "16",
                output_path,
            ]
        elif shutil.which("ffmpeg"):
            return "ffmpeg", [
                "-f",
                "alsa",
                "-i",
                "default",
                "-q:a",
                "0",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-y",
                output_path,
            ]
        else:
            raise RuntimeError(
                "No audio recording tool found on Linux. "
                "Install one of: alsa-utils (apt install alsa-utils), sox (apt install sox), ffmpeg (apt install ffmpeg)"
            )

    elif sys.platform == "win32":
        if shutil.which("ffmpeg"):
            return "ffmpeg", [
                "-f",
                "dshow",
                "-i",
                "audio=Microphone",
                "-q:a",
                "0",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-y",
                output_path,
            ]
        else:
            raise RuntimeError(
                "No audio recording tool found on Windows. "
                "Install ffmpeg: https://ffmpeg.org/download.html"
            )

    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


class TranscriptionHooksConfig:
    """Configuration for transcription hooks."""

    def __init__(
        self,
        transcriber: AudioTranscriberProtocol,
        language: str | None = None,
        save_transcriptions: bool = False,
        transcription_dir: str | None = None,
        async_hooks: bool = True,
        max_segments: int | None = None,
        effort: float = 1.0,
        enable_interactive_recording: bool = False,
        copy_to_clipboard: bool = False,
    ):
        self.transcriber = transcriber
        self.language = language
        self.save_transcriptions = save_transcriptions
        self.transcription_dir = transcription_dir
        self.async_hooks = async_hooks
        self.max_segments = max_segments
        self.effort = effort
        self.enable_interactive_recording = enable_interactive_recording
        self.copy_to_clipboard = copy_to_clipboard

    def pre_hook(self) -> AgnoPreHook:
        """Get the pre-hook for transcription.

        Returns:
            Pre-hook function configured with this config's settings
        """
        return _create_transcription_hook(
            transcriber=self.transcriber,
            language=self.language,
            save_transcriptions=self.save_transcriptions,
            transcription_dir=self.transcription_dir,
            max_segments=self.max_segments,
            effort=self.effort,
            async_hooks=not self.async_hooks,
            enable_interactive_recording=self.enable_interactive_recording,
            copy_to_clipboard=self.copy_to_clipboard,
        )


def _create_transcription_hook(
    transcriber: AudioTranscriberProtocol,
    language: str | None,
    save_transcriptions: bool,
    transcription_dir: str | None,
    max_segments: int | None,
    effort: float,
    async_hooks: bool = True,
    enable_interactive_recording: bool = False,
    copy_to_clipboard: bool = False,
) -> AgnoPreHook:
    """Create transcription hook.

    Args:
        transcriber: Optional audio transcriber instance
        language: Optional language code for transcription
        save_transcriptions: Whether to save transcriptions to files
        transcription_dir: Directory to save transcriptions
        max_segments: Maximum number of segments to include
        effort: Transcription effort level (0.0-1.0)
        async_hooks: If True, returns a sync hook that runs async logic synchronously
        enable_interactive_recording: If True, enables interactive recording when no audio
        copy_to_clipboard: If True, copies the plain transcription text to clipboard

    Returns:
        Hook function (async or sync based on async_hooks parameter)
    """

    async def hook_logic(
        agent: Agent | Team,
        run_input: RunInput,
        session: AgentSession | TeamSession,
        user_id: str | None,
        debug_mode: bool | None,
    ) -> None:
        """Shared hook implementation for transcription."""
        if not run_input.audios:
            if enable_interactive_recording:
                log_debug("No audio files, enabling interactive recording")
                audio_bytes, temp_path = record_interactive_audio()
                run_input.audios = [Audio(content=audio_bytes, id="interactive_recording")]
                log_debug(f"Recorded {len(audio_bytes)} bytes of audio")

                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)
            else:
                log_debug("No audio files to transcribe")
                return

        log_debug(f"Processing {len(run_input.audios)} audio(s) for transcription")

        try:
            # Transcribe all audio files
            transcripts: list[TranscriptionResult] = []
            audio_files = run_input.audios or []

            for audio_file in audio_files:
                # Transcribe audio file using shared function
                result = await _transcribe_audio_file(
                    transcriber, audio_file, language, effort, async_hooks=async_hooks
                )
                if result:
                    # Save transcription if configured
                    if save_transcriptions:
                        source_path = (
                            str(audio_file.filepath)
                            if audio_file.filepath
                            else f"audio_{audio_file.id}"
                        )
                        # Use asyncio.run for sync saving
                        if async_hooks:
                            asyncio.run(
                                _save_transcription_to_file(
                                    result,
                                    source_path,
                                    transcription_dir,
                                )
                            )
                        else:
                            await _save_transcription_to_file(
                                result,
                                source_path,
                                transcription_dir,
                            )
                    transcripts.append(result)

            # Copy plain text to clipboard if enabled
            if copy_to_clipboard and transcripts:
                # Combine all transcript texts into clean plain text
                plain_text_parts: list[str] = []
                for transcript in transcripts:
                    if transcript.text:
                        plain_text_parts.append(transcript.text.strip())

                if plain_text_parts:
                    combined_text = " ".join(plain_text_parts)
                    if copy_to_clipboard_fn(combined_text):
                        log_debug(f"Copied transcription to clipboard ({len(combined_text)} chars)")
                    else:
                        log_warning("Failed to copy transcription to clipboard")

            _format_and_inject_transcripts(run_input, transcripts, max_segments)

        except Exception as e:
            log_warning(f"Transcription failed: {e}")
            # Still clear audio files even on error

        run_input.audios = []
        log_debug("Transcription hook completed")

    # Return appropriate hook based on async_hooks parameter
    if async_hooks:

        def sync_hook(
            agent: Agent | Team,
            run_input: RunInput,
            session: AgentSession | TeamSession,
            user_id: str | None,
            debug_mode: bool | None,
        ) -> None:
            """Sync wrapper that runs the async hook logic."""
            asyncio.run(hook_logic(agent, run_input, session, user_id, debug_mode))

        return sync_hook
    else:
        return hook_logic
