"""Text-to-Speech (TTS) common types and protocol."""

from __future__ import annotations

import io
import logging
import platform
import select
import shutil
import subprocess
import sys
import tempfile
import termios
import threading
import tty
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of text-to-speech synthesis."""

    audio: bytes
    sample_rate: int
    duration: float


class VoiceSynthesizerProtocol(Protocol):
    """Protocol for voice synthesizers (text-to-speech)."""

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 24000,
    ) -> SynthesisResult | None:
        """Synthesize speech from text."""
        ...

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int = 24000,
    ) -> SynthesisResult | None:
        """Synthesize speech from text and save to file."""
        ...


class AudioPlayerProtocol(Protocol):
    """Protocol for audio players (playback of synthesized audio)."""

    def play(
        self,
        audio_path: str,
        interruptible: bool = False,
    ) -> bool:
        """Play audio from file path.

        Args:
            audio_path: Path to the audio file.
            interruptible: If True, playback can be interrupted with spacebar.

        Returns:
            True if playback completed normally, False if interrupted.
        """
        ...

    def play_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 24000,
        interruptible: bool = False,
    ) -> bool:
        """Play audio from bytes directly.

        Args:
            audio_bytes: Raw audio bytes.
            sample_rate: Sample rate in Hz.
            interruptible: If True, playback can be interrupted with spacebar.

        Returns:
            True if playback completed normally, False if interrupted.
        """
        ...

    def stop(self) -> None:
        """Stop any currently playing audio."""
        ...

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        ...


class AudioPlayer:
    """Cross-platform audio player for WAV and other audio formats.

    Supports macOS (afplay), Linux (aplay/paplay/ffplay), and Windows (PowerShell).
    Supports interruptible playback with spacebar.

    Example:
        ```python
        from blockether_foundation.tts import AudioPlayer

        player = AudioPlayer()

        # Play from file
        player.play("output.wav")

        # Play with spacebar interrupt support
        completed = player.play("output.wav", interruptible=True)
        if not completed:
            print("Playback was interrupted!")

        # Play from bytes (e.g., from PiperTTS.synthesize())
        result = tts.synthesize("Hello!")
        player.play_bytes(result.audio, result.sample_rate)
        ```
    """

    def __init__(self, player_command: str | None = None):
        """Initialize the audio player.

        Args:
            player_command: Override the default player command.
                           If None, auto-detects based on platform.
        """
        self._system = platform.system()
        self._player_command = player_command or self._detect_player()
        self._current_process: subprocess.Popen | None = None
        self._interrupted = False
        self._lock = threading.Lock()

    def _detect_player(self) -> str:
        """Detect the best available audio player for the current platform."""
        if self._system == "Darwin":  # macOS
            return "afplay"
        elif self._system == "Linux":
            # Try players in order of preference
            for player in ["paplay", "aplay", "ffplay", "mpv"]:
                if shutil.which(player):
                    return player
            raise RuntimeError(
                "No audio player found on Linux. "
                "Install one of: pulseaudio (paplay), alsa-utils (aplay), ffmpeg (ffplay), or mpv"
            )
        elif self._system == "Windows":
            return "powershell"
        else:
            raise RuntimeError(f"Unsupported platform: {self._system}")

    def stop(self) -> None:
        """Stop any currently playing audio."""
        with self._lock:
            if self._current_process is not None:
                self._interrupted = True
                try:
                    self._current_process.terminate()
                    self._current_process.wait(timeout=1)
                except Exception:
                    try:
                        self._current_process.kill()
                    except Exception:
                        pass
                self._current_process = None
                logger.debug("Audio playback stopped")

    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        with self._lock:
            if self._current_process is None:
                return False
            return self._current_process.poll() is None

    def _wait_for_spacebar_interrupt(self) -> None:
        """Monitor for spacebar press and stop playback if detected."""
        if self._system == "Windows":
            try:
                import msvcrt

                while self.is_playing():
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        if key == b" ":
                            logger.debug("Spacebar detected - interrupting playback")
                            self.stop()
                            return
                    import time

                    time.sleep(0.05)
            except Exception as e:
                logger.debug(f"Spacebar detection error: {e}")
        else:
            fd = sys.stdin.fileno()
            try:
                old_settings = termios.tcgetattr(fd)
            except termios.error:
                # Not a terminal (e.g., running in background)
                logger.debug("Not a terminal - spacebar interrupt disabled")
                return

            try:
                tty.setraw(fd)
                while self.is_playing():
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        key = sys.stdin.read(1)
                        if key == " ":
                            logger.debug("Spacebar detected - interrupting playback")
                            self.stop()
                            return
            except Exception as e:
                logger.debug(f"Spacebar detection error: {e}")
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

    def play(self, audio_path: str, interruptible: bool = False) -> bool:
        """Play audio from a file path.

        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.).
            interruptible: If True, playback can be interrupted with spacebar.

        Returns:
            True if playback completed normally, False if interrupted.

        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            RuntimeError: If playback fails.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.debug(f"Playing audio: {audio_path} with {self._player_command}")
        self._interrupted = False

        try:
            if interruptible:
                return self._play_interruptible(str(path))
            else:
                if self._system == "Darwin":
                    subprocess.run(
                        ["afplay", str(path)],
                        check=True,
                        capture_output=True,
                    )
                elif self._system == "Linux":
                    self._play_linux_blocking(str(path))
                elif self._system == "Windows":
                    self._play_windows(str(path))
                return True
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to play audio: {e.stderr.decode() if e.stderr else e}") from e

    def _play_interruptible(self, audio_path: str) -> bool:
        """Play audio with spacebar interrupt support.

        Args:
            audio_path: Path to the audio file.

        Returns:
            True if playback completed normally, False if interrupted.
        """
        cmd = self._get_play_command(audio_path)
        if cmd is None:
            raise RuntimeError(f"No play command for platform: {self._system}")

        with self._lock:
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        # Start spacebar monitor thread
        monitor_thread = threading.Thread(
            target=self._wait_for_spacebar_interrupt, daemon=True
        )
        monitor_thread.start()

        # Wait for playback to complete
        self._current_process.wait()
        monitor_thread.join(timeout=0.1)

        with self._lock:
            was_interrupted = self._interrupted
            self._current_process = None
            self._interrupted = False

        if was_interrupted:
            logger.debug("Playback was interrupted")
            return False

        logger.debug("Playback completed normally")
        return True

    def _get_play_command(self, audio_path: str) -> list[str] | None:
        """Get the command to play audio on the current platform."""
        if self._system == "Darwin":
            return ["afplay", audio_path]
        elif self._system == "Linux":
            player = self._player_command
            if player == "paplay":
                return ["paplay", audio_path]
            elif player == "aplay":
                return ["aplay", "-q", audio_path]
            elif player == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]
            elif player == "mpv":
                return ["mpv", "--no-video", "--really-quiet", audio_path]
            else:
                return [player, audio_path]
        elif self._system == "Windows":
            # Windows uses PowerShell which doesn't work well with Popen
            # Fall back to blocking play
            return None
        return None

    def _play_linux_blocking(self, audio_path: str) -> None:
        """Play audio on Linux (blocking)."""
        player = self._player_command

        if player == "paplay":
            subprocess.run(["paplay", audio_path], check=True, capture_output=True)
        elif player == "aplay":
            subprocess.run(["aplay", "-q", audio_path], check=True, capture_output=True)
        elif player == "ffplay":
            subprocess.run(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
                check=True,
                capture_output=True,
            )
        elif player == "mpv":
            subprocess.run(
                ["mpv", "--no-video", "--really-quiet", audio_path],
                check=True,
                capture_output=True,
            )
        else:
            # Generic fallback
            subprocess.run([player, audio_path], check=True, capture_output=True)

    def _play_windows(self, audio_path: str) -> None:
        """Play audio on Windows using PowerShell."""
        # Use .NET SoundPlayer for WAV files
        ps_script = f"""
        Add-Type -AssemblyName presentationCore
        $player = New-Object System.Windows.Media.MediaPlayer
        $player.Open([uri]"{audio_path}")
        $player.Play()
        Start-Sleep -Milliseconds 500
        while ($player.Position -lt $player.NaturalDuration.TimeSpan) {{
            Start-Sleep -Milliseconds 100
        }}
        $player.Close()
        """
        subprocess.run(
            ["powershell", "-Command", ps_script],
            check=True,
            capture_output=True,
        )

    def play_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 22050,
        channels: int = 1,
        sample_width: int = 2,
        interruptible: bool = False,
    ) -> bool:
        """Play audio directly from bytes.

        Args:
            audio_bytes: Raw audio bytes (WAV format expected, or raw PCM).
            sample_rate: Sample rate in Hz (default 22050, common for Piper).
            channels: Number of audio channels (default 1 = mono).
            sample_width: Bytes per sample (default 2 = 16-bit).
            interruptible: If True, playback can be interrupted with spacebar.

        Returns:
            True if playback completed normally, False if interrupted.

        Note:
            If audio_bytes is already a complete WAV file (starts with RIFF header),
            it will be played directly. Otherwise, raw PCM is assumed and wrapped in WAV.
        """
        # Check if it's already a WAV file
        is_wav = audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"

        if is_wav:
            wav_bytes = audio_bytes
        else:
            # Wrap raw PCM in WAV format
            wav_bytes = self._pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)

        # For interruptible playback, always use temp file approach
        if interruptible:
            return self._play_from_tempfile(wav_bytes, interruptible=True)

        # Try to play via stdin pipe (most efficient) or fall back to temp file
        if self._can_play_from_stdin():
            self._play_from_stdin(wav_bytes)
        else:
            self._play_from_tempfile(wav_bytes)
        return True

    def _pcm_to_wav(
        self,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        sample_width: int,
    ) -> bytes:
        """Convert raw PCM bytes to WAV format."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()

    def _can_play_from_stdin(self) -> bool:
        """Check if the current player supports stdin input."""
        return self._player_command in ("aplay", "ffplay", "mpv", "paplay")

    def _play_from_stdin(self, wav_bytes: bytes) -> None:
        """Play audio by piping to stdin."""
        player = self._player_command
        logger.debug(f"Playing {len(wav_bytes)} bytes via stdin to {player}")

        try:
            if player == "aplay":
                subprocess.run(
                    ["aplay", "-q", "-"],
                    input=wav_bytes,
                    check=True,
                    capture_output=True,
                )
            elif player == "paplay":
                # paplay can read from stdin with --raw but needs format info
                # For WAV files, use a temp file instead
                self._play_from_tempfile(wav_bytes)
            elif player == "ffplay":
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
                    input=wav_bytes,
                    check=True,
                    capture_output=True,
                )
            elif player == "mpv":
                subprocess.run(
                    ["mpv", "--no-video", "--really-quiet", "-"],
                    input=wav_bytes,
                    check=True,
                    capture_output=True,
                )
            else:
                self._play_from_tempfile(wav_bytes)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to play audio via stdin: {e.stderr.decode() if e.stderr else e}"
            ) from e

    def _play_from_tempfile(self, wav_bytes: bytes, interruptible: bool = False) -> bool:
        """Play audio by writing to a temporary file.

        Args:
            wav_bytes: WAV audio bytes.
            interruptible: If True, playback can be interrupted with spacebar.

        Returns:
            True if playback completed normally, False if interrupted.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            logger.debug(f"Playing from temp file: {tmp_path}")
            return self.play(tmp_path, interruptible=interruptible)
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    def play_synthesis_result(
        self, result: SynthesisResult, interruptible: bool = False
    ) -> bool:
        """Play audio from a SynthesisResult directly.

        Args:
            result: The SynthesisResult from a TTS synthesize() call.
            interruptible: If True, playback can be interrupted with spacebar.

        Returns:
            True if playback completed normally, False if interrupted.

        Example:
            ```python
            result = tts.synthesize("Hello, world!")
            player.play_synthesis_result(result)

            # With spacebar interrupt
            completed = player.play_synthesis_result(result, interruptible=True)
            ```
        """
        return self.play_bytes(
            result.audio, sample_rate=result.sample_rate, interruptible=interruptible
        )
