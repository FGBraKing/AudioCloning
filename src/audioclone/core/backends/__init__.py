"""模型实现层 — 可替换、可插拔的后端。

接口契约定义于 :mod:`audioclone.core.capabilities`；本目录存放各模型的具体实现。
"""

from __future__ import annotations

from audioclone.core.backends.registry import BackendRegistry, registry
from audioclone.core.capabilities import (
    AudioClassificationCapability,
    AudioToAudioCapability,
    SpeechRecognitionCapability,
    TextToAudioCapability,
    TextToSpeechCapability,
    VoiceActivityCapability,
)

__all__ = [
    "AudioClassificationCapability",
    "AudioToAudioCapability",
    "BackendRegistry",
    "SpeechRecognitionCapability",
    "TextToAudioCapability",
    "TextToSpeechCapability",
    "VoiceActivityCapability",
    "registry",
]