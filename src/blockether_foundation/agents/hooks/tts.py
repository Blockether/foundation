"""Text-to-Speech (TTS) hooks for Agno agents."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

from agno.agent.agent import Agent
from agno.run.agent import RunOutput
from agno.run.base import RunContext
from agno.run.team import TeamRunOutput
from agno.session import AgentSession, TeamSession
from agno.team import Team
from agno.utils.log import log_debug, log_warning  # type: ignore

from ...tts.common import AudioPlayer
from ...utils import AgnoPostHook, DebugMode, UserId, normalize_text_for_tts

if TYPE_CHECKING:
    from ...tts import SynthesisResult, VoiceSynthesizerProtocol

logger = logging.getLogger(__name__)


class TTSHooksConfig:
    """Configuration for TTS (text-to-speech) hooks.

    This post-hook takes the agent's output and generates audio using
    configured voice synthesizer, then plays it.
    """

    def __init__(
        self,
        synthesizer: VoiceSynthesizerProtocol,
        output_dir: str | None = None,
        async_hooks: bool = True,
        save_to_file: bool = False,
        auto_speak: bool = True,
        language: str | None = None,
        voice: str | None = None,
        interruptible: bool = False,
    ):
        """Initialize TTS hooks configuration.

        Args:
            synthesizer: Voice synthesizer instance implementing VoiceSynthesizerProtocol
            output_dir: Optional directory to save audio files. If None, audio is
                       synthesized to bytes but not saved to disk.
            async_hooks: Whether to use async hooks (True) or sync hooks (False)
            save_to_file: Whether to save synthesized audio to file (requires output_dir)
            auto_speak: Whether to automatically play synthesized audio after generation.
                       Uses cross-platform AudioPlayer (macOS/Linux/Windows).
            language: Language code for multilingual models (e.g., 'en', 'pl', 'es').
            voice: Voice ID for multi-speaker models.
            interruptible: Whether to allow interrupting playback with spacebar.
                          When True, pressing spacebar will stop TTS playback.
        """
        self.synthesizer = synthesizer
        self.output_dir = output_dir
        self.async_hooks = async_hooks
        self.save_to_file = save_to_file
        self.auto_speak = auto_speak
        self.language = language
        self.voice = voice
        self.interruptible = interruptible

    def post_hook(self) -> AgnoPostHook:
        """Get the post-hook for TTS synthesis.

        Returns:
            Post-hook function configured with this config's settings
        """
        return _create_tts_hook(
            synthesizer=self.synthesizer,
            output_dir=self.output_dir,
            save_to_file=self.save_to_file,
            auto_speak=self.auto_speak,
            language=self.language,
            voice=self.voice,
            interruptible=self.interruptible,
            return_sync_wrapper=not self.async_hooks,
        )


def _extract_text_from_output(
    run_output: RunOutput | TeamRunOutput,
) -> str:
    """Extract text content from agent output.

    Args:
        run_output: The agent's run output

    Returns:
        Extracted text as a string
    """
    raw_text = cast(str, run_output.get_content_as_string())
    return normalize_text_for_tts(raw_text)


def _generate_output_filename(
    output_dir: str,
    output_text: str,
    timestamp: str,
) -> str:
    """Generate a filename for synthesized audio output.

    Args:
        output_dir: Directory to save file
        output_text: The text that was synthesized
        timestamp: Timestamp string

    Returns:
        Full path to the output file
    """
    safe_text = "".join(c for c in output_text[:50] if c.isalnum() or c in (" ", "-", "_")).strip()
    if not safe_text:
        safe_text = "output"

    safe_text = safe_text.replace(" ", "_")

    filename = f"{timestamp}_{safe_text}.wav"
    return str(Path(output_dir) / filename)


async def _synthesize_output(
    synthesizer: VoiceSynthesizerProtocol,
    text: str,
    output_dir: str | None,
    save_to_file: bool,
    language: str | None = None,
    voice: str | None = None,
) -> tuple[str | None, SynthesisResult | None]:
    """Synthesize text to audio and optionally save to file.

    Args:
        synthesizer: Voice synthesizer instance
        text: Text to synthesize
        output_dir: Directory to save audio file (optional)
        save_to_file: Whether to save to file
        language: Language code for multilingual models (e.g., 'en', 'pl', 'es')
        voice: Voice ID for multi-speaker models

    Returns:
        Tuple of (output_path, SynthesisResult) - output_path may be None if not saved to file
    """
    if not text or not text.strip():
        log_debug("No text to synthesize")
        return None, None

    log_debug(f"Synthesizing audio for {len(text)} characters of text")

    try:
        output_path = None
        if save_to_file and output_dir:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = _generate_output_filename(output_dir, text, timestamp)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()

        if save_to_file and output_path:
            result = await loop.run_in_executor(
                None,
                lambda: synthesizer.synthesize_to_file(
                    text=text,
                    output_path=cast(str, output_path),
                    language=language,
                    voice=voice,
                ),
            )

            if result:
                log_debug(
                    f"Successfully synthesized audio to {output_path} "
                    f"(duration: {result.duration:.2f}s, sample_rate: {result.sample_rate}Hz)"
                )
                return output_path, result
            else:
                log_warning(f"Synthesis returned None for output_path: {output_path}")
                return None, None
        else:
            result = await loop.run_in_executor(
                None,
                lambda: synthesizer.synthesize(
                    text=text,
                    language=language,
                    voice=voice,
                ),
            )

            if result:
                log_debug(
                    f"Successfully synthesized audio in memory "
                    f"(duration: {result.duration:.2f}s, size: {len(result.audio)} bytes)"
                )
                return None, result
            else:
                log_warning("Synthesis returned None")
                return None, None

    except Exception as e:
        log_warning(f"Failed to synthesize audio: {e}")
        return None, None


def _synthesize_output_sync(
    synthesizer: VoiceSynthesizerProtocol,
    text: str,
    output_dir: str | None,
    save_to_file: bool,
    language: str | None = None,
    voice: str | None = None,
) -> tuple[str | None, SynthesisResult | None]:
    """Synchronous wrapper for synthesis (for sync hooks).

    Args:
        synthesizer: Voice synthesizer instance
        text: Text to synthesize
        output_dir: Directory to save audio file (optional)
        save_to_file: Whether to save to file
        language: Language code for multilingual models (e.g., 'en', 'pl', 'es')
        voice: Voice ID for multi-speaker models

    Returns:
        Tuple of (output_path, SynthesisResult) - output_path may be None if not saved to file
    """
    if not text or not text.strip():
        log_debug("No text to synthesize")
        return None, None

    log_debug(f"Synthesizing audio for {len(text)} characters of text")

    try:
        output_path = None
        if save_to_file and output_dir:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = _generate_output_filename(output_dir, text, timestamp)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        if save_to_file and output_path:
            result = synthesizer.synthesize_to_file(
                text=text,
                output_path=cast(str, output_path),
                language=language,
                voice=voice,
            )

            if result:
                log_debug(
                    f"Successfully synthesized audio to {output_path} "
                    f"(duration: {result.duration:.2f}s, sample_rate: {result.sample_rate}Hz)"
                )
                return output_path, result
            else:
                log_warning(f"Synthesis returned None for output_path: {output_path}")
                return None, None
        else:
            result = synthesizer.synthesize(
                text=text,
                language=language,
                voice=voice,
            )

            if result:
                log_debug(
                    f"Successfully synthesized audio in memory "
                    f"(duration: {result.duration:.2f}s, size: {len(result.audio)} bytes)"
                )
                return None, result
            else:
                log_warning("Synthesis returned None")
                return None, None

    except Exception as e:
        log_warning(f"Failed to synthesize audio: {e}")
        return None, None


def _create_tts_hook(
    synthesizer: VoiceSynthesizerProtocol,
    output_dir: str | None,
    save_to_file: bool,
    auto_speak: bool = True,
    language: str | None = None,
    voice: str | None = None,
    interruptible: bool = False,
    return_sync_wrapper: bool = False,
) -> AgnoPostHook:
    """Create TTS post-hook for audio synthesis.

    Args:
        synthesizer: Voice synthesizer instance
        output_dir: Directory to save audio file (optional)
        save_to_file: Whether to save to file
        auto_speak: Whether to automatically play synthesized audio
        language: Language code for multilingual models (e.g., 'en', 'pl', 'es')
        voice: Voice ID for multi-speaker models
        interruptible: Whether to allow interrupting playback with spacebar
        return_sync_wrapper: If True, returns a sync wrapper that runs async logic synchronously

    Returns:
        Post-hook function (async or sync wrapper based on return_sync_wrapper parameter)
    """
    # Create audio player once for this hook instance
    player = AudioPlayer() if auto_speak else None

    def _play_audio(
        output_path: str | None,
        synthesis_result: SynthesisResult | None,
    ) -> bool:
        """Play audio using the embedded AudioPlayer.

        Returns:
            True if playback completed normally, False if interrupted.
        """
        if player is None:
            return True

        if output_path:
            log_debug(f"Auto-speaking audio from file: {output_path}")
            if interruptible:
                log_debug("Press SPACE to interrupt playback...")
            return player.play(output_path, interruptible=interruptible)
        elif synthesis_result:
            log_debug(f"Auto-speaking audio from bytes ({len(synthesis_result.audio)} bytes)")
            if interruptible:
                log_debug("Press SPACE to interrupt playback...")
            return player.play_bytes(
                synthesis_result.audio,
                sample_rate=synthesis_result.sample_rate,
                interruptible=interruptible,
            )
        return True

    async def hook_logic(
        agent: Agent | Team,
        run_output: RunOutput | TeamRunOutput,
        session: AgentSession | TeamSession,
        run_context: RunContext,
        user_id: UserId,
        debug_mode: DebugMode,
    ) -> None:
        """Shared hook implementation for TTS synthesis."""
        text = _extract_text_from_output(run_output)

        if not text:
            log_debug("No text content in output, skipping TTS synthesis")
            return

        log_debug(f"TTS post-hook triggered, synthesizing {len(text)} characters")

        try:
            output_path, synthesis_result = await _synthesize_output(
                synthesizer,
                text,
                output_dir,
                save_to_file,
                language,
                voice,
            )

            if auto_speak and (output_path or synthesis_result):
                completed = _play_audio(output_path, synthesis_result)
                if not completed:
                    log_debug("TTS playback was interrupted by user")
        except Exception as e:
            log_warning(f"TTS synthesis failed: {e}")

    if return_sync_wrapper:

        def sync_hook(
            agent: Agent | Team,
            run_output: RunOutput | TeamRunOutput,
            session: AgentSession | TeamSession,
            run_context: RunContext,
            user_id: UserId,
            debug_mode: DebugMode,
        ) -> None:
            """Sync wrapper that runs the async hook logic."""
            text = _extract_text_from_output(run_output)

            if not text:
                log_debug("No text content in output, skipping TTS synthesis")
                return

            log_debug(f"TTS post-hook triggered, synthesizing {len(text)} characters")

            try:
                output_path, synthesis_result = _synthesize_output_sync(
                    synthesizer,
                    text,
                    output_dir,
                    save_to_file,
                    language,
                    voice,
                )

                if auto_speak and (output_path or synthesis_result):
                    completed = _play_audio(output_path, synthesis_result)
                    if not completed:
                        log_debug("TTS playback was interrupted by user")
            except Exception as e:
                log_warning(f"TTS synthesis failed: {e}")

        return sync_hook
    else:
        return hook_logic
