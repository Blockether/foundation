"""Text-to-Speech (TTS) module."""

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING, Any

from . import (
    common,
    eleven,  # noqa: F401 - imported for lazy export
    piper,  # noqa: F401 - imported for lazy export
)

if TYPE_CHECKING:
    from .common import (
        AudioPlayer,
        AudioPlayerProtocol,
        SynthesisResult,
        VoiceSynthesizerProtocol,
    )
    from .eleven import ElevenLabsTTS
    from .piper import PIPER_DEFAULT_MODEL, PiperTTS

logger = logging.getLogger(__name__)

# Check if elevenlabs is available
try:
    pkg_version("elevenlabs")
    TTS_ELEVENLABS_AVAILABLE = True
except (ImportError, PackageNotFoundError, Exception):
    TTS_ELEVENLABS_AVAILABLE = False
    logger.debug("elevenlabs not available - ElevenLabs TTS disabled")

# Check if piper-tts is available
try:
    pkg_version("piper-tts")
    TTS_PIPER_AVAILABLE = True
except (ImportError, PackageNotFoundError, Exception):
    TTS_PIPER_AVAILABLE = False
    logger.debug("piper-tts not available - Piper TTS disabled")

__all__ = [
    "AudioPlayer",
    "AudioPlayerProtocol",
    "SynthesisResult",
    "VoiceSynthesizerProtocol",
    "ElevenLabsTTS",
    "TTS_ELEVENLABS_AVAILABLE",
    "PiperTTS",
    "PIPER_DEFAULT_MODEL",
    "TTS_PIPER_AVAILABLE",
]


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies."""
    if name == "AudioPlayer":
        return common.AudioPlayer
    if name == "AudioPlayerProtocol":
        return common.AudioPlayerProtocol
    if name == "SynthesisResult":
        return common.SynthesisResult
    if name == "VoiceSynthesizerProtocol":
        return common.VoiceSynthesizerProtocol
    if name == "TTS_ELEVENLABS_AVAILABLE":
        return TTS_ELEVENLABS_AVAILABLE
    if name == "TTS_PIPER_AVAILABLE":
        return TTS_PIPER_AVAILABLE
    if name == "PIPER_DEFAULT_MODEL":
        return piper.PIPER_DEFAULT_MODEL

    if name == "ElevenLabsTTS":
        _check_elevenlabs_available()
        return eleven.ElevenLabsTTS

    if name == "PiperTTS":
        _check_piper_available()
        return piper.PiperTTS

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _check_elevenlabs_available() -> None:
    """Check if ElevenLabs dependencies are installed."""
    if not TTS_ELEVENLABS_AVAILABLE:
        raise ImportError(
            "ElevenLabs is not installed. "
            "Install it with: uv pip install 'blockether-foundation[tts_elevenlabs]'"
        )


def _check_piper_available() -> None:
    """Check if Piper TTS dependencies are installed."""
    if not TTS_PIPER_AVAILABLE:
        raise ImportError(
            "Piper TTS is not installed. "
            "Install it with: uv pip install 'blockether-foundation[tts_local]'"
        )
