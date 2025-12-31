"""Hooks for Agno agents."""

from ...tts import SynthesisResult, VoiceSynthesizerProtocol
from ...utils import AgnoPostHook, AgnoPreHook
from .consensus import (
    ConsensusHooksConfig,
    ConsensusResult,
    JudgeCriteria,
    ModelConfig,
)
from .graph import (
    GraphHookIterativeConfig,
    GraphHooksConfig,
    GraphIngestionIterativeConfig,
)
from .asr import TranscriptionHooksConfig
from .tts import TTSHooksConfig

__all__ = [
    "GraphHookIterativeConfig",
    "GraphHooksConfig",
    "GraphIngestionIterativeConfig",
    "AgnoPreHook",
    "AgnoPostHook",
    "TranscriptionHooksConfig",
    "TTSHooksConfig",
    "SynthesisResult",
    "VoiceSynthesizerProtocol",
    "ConsensusHooksConfig",
    "ConsensusResult",
    "JudgeCriteria",
    "ModelConfig",
]
