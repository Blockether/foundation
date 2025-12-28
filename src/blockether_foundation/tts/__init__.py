"""Text-to-Speech (TTS) module."""

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from . import (
    common,
    eleven,  # noqa: F401 - imported for lazy export
    local_coqui,  # noqa: F401 - imported for lazy export
)

if TYPE_CHECKING:
    from .common import (
        SynthesisResult,
        VoiceSynthesizerProtocol,
    )
    from .eleven import ElevenLabsTTS
    from .local_coqui import LocalCoquiTTS, ModelName

logger = logging.getLogger(__name__)

# Check if coqui-tts is available
try:
    version("coqui-tts")
    TTS_LOCAL_AVAILABLE = True
except (ImportError, PackageNotFoundError, Exception):
    TTS_LOCAL_AVAILABLE = False
    logger.debug("coqui-tts (coqui-tts) not available - local TTS disabled")

# Check if elevenlabs is available
try:
    version("elevenlabs")
    TTS_ELEVENLABS_AVAILABLE = True
except (ImportError, PackageNotFoundError, Exception):
    TTS_ELEVENLABS_AVAILABLE = False
    logger.debug("elevenlabs not available - ElevenLabs TTS disabled")


__all__ = [
    "SynthesisResult",
    "VoiceSynthesizerProtocol",
    "ElevenLabsTTS",
    "LocalCoquiTTS",
    "ModelName",
    "TTS_LOCAL_AVAILABLE",
    "TTS_ELEVENLABS_AVAILABLE",
]


def __getattr__(name: str) -> Any:
    """Lazy import for optional dependencies."""
    if name == "SynthesisResult":
        return common.SynthesisResult
    if name == "VoiceSynthesizerProtocol":
        return common.VoiceSynthesizerProtocol
    if name == "TTS_LOCAL_AVAILABLE":
        return TTS_LOCAL_AVAILABLE
    if name == "TTS_ELEVENLABS_AVAILABLE":
        return TTS_ELEVENLABS_AVAILABLE

    if name == "LocalCoquiTTS" or name == "ModelName":
        _check_tts_local_available()

        if name == "LocalCoquiTTS":
            return local_coqui.LocalCoquiTTS
        if name == "ModelName":
            return local_coqui.ModelName

    if name == "ElevenLabsTTS":
        _check_elevenlabs_available()
        return eleven.ElevenLabsTTS

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _check_tts_local_available() -> None:
    """Check if local TTS dependencies are installed."""
    if not TTS_LOCAL_AVAILABLE:
        raise ImportError(
            "Local TTS dependencies are not installed. "
            "Install them with: uv pip install 'blockether-foundation[tts_local]'"
        )


def _check_elevenlabs_available() -> None:
    """Check if ElevenLabs dependencies are installed."""
    if not TTS_ELEVENLABS_AVAILABLE:
        raise ImportError(
            "ElevenLabs is not installed. "
            "Install it with: uv pip install 'blockether-foundation[tts_elevenlabs]'"
        )
