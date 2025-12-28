"""Automatic Speech Recognition (ASR) module."""

import logging
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from typing import TYPE_CHECKING, Any

from . import (
    common,
    local_whisper,  # noqa: F401 - imported for lazy export
)

if TYPE_CHECKING:
    from .common import (
        TEN_MINUTES,
        AudioTranscriberProtocol,
        TranscriptionResult,
        TranscriptionSegment,
        TranscriptionSegmentWord,
        format_transcription_for_context,
        merge_transcription_results,
        split_audio_into_chunks,
    )
    from .local_whisper import LocalWhisperAudioTranscriber, WhisperModelName

logger = logging.getLogger(__name__)

# Check if faster-whisper is available
try:
    from importlib.metadata import version

    version("faster-whisper")
    ASR_LOCAL_AVAILABLE = True
except (ImportError, PackageNotFoundError, Exception):
    ASR_LOCAL_AVAILABLE = False
    logger.debug("faster-whisper not available - local ASR disabled")

__all__ = [
    "AudioTranscriberProtocol",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionSegmentWord",
    "format_transcription_for_context",
    "merge_transcription_results",
    "split_audio_into_chunks",
    "TEN_MINUTES",
    "LocalWhisperAudioTranscriber",
    "WhisperModelName",
    "ASR_LOCAL_AVAILABLE",
]


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies."""
    if name == "AudioTranscriberProtocol":
        return common.AudioTranscriberProtocol
    if name == "TranscriptionResult":
        return common.TranscriptionResult
    if name == "TranscriptionSegment":
        return common.TranscriptionSegment
    if name == "TranscriptionSegmentWord":
        return common.TranscriptionSegmentWord
    if name == "format_transcription_for_context":
        return common.format_transcription_for_context
    if name == "merge_transcription_results":
        return common.merge_transcription_results
    if name == "split_audio_into_chunks":
        return common.split_audio_into_chunks
    if name == "TEN_MINUTES":
        return common.TEN_MINUTES
    if name == "ASR_LOCAL_AVAILABLE":
        return ASR_LOCAL_AVAILABLE

    if name == "LocalWhisperAudioTranscriber" or name == "WhisperModelName":
        _check_asr_local_available()

        if name == "LocalWhisperAudioTranscriber":
            return local_whisper.LocalWhisperAudioTranscriber
        if name == "WhisperModelName":
            return local_whisper.WhisperModelName

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _check_asr_local_available() -> None:
    """Check if local ASR dependencies are installed."""
    if not ASR_LOCAL_AVAILABLE:
        raise ImportError(
            "Local ASR dependencies are not installed. "
            "Install them with: pip install blockether-foundation[asr_local]"
        )
