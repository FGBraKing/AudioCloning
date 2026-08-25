"""能力抽象层 — 多功能模型的后端契约。

一个模型后端可聚合多种 **能力(Capability)**，从而原生支持多个任务
（TTS / T2A / ASR / A2A / 音频分类 / VAD 等）。任务编排层按能力消费，而非按具体模型。
"""

from __future__ import annotations

from audioclone.core.capabilities.a2a import AudioToAudioCapability
from audioclone.core.capabilities.asr import SpeechRecognitionCapability
from audioclone.core.capabilities.audio_classification import AudioClassificationCapability
from audioclone.core.capabilities.base import Capability, ModelBackend
from audioclone.core.capabilities.t2a import TextToAudioCapability
from audioclone.core.capabilities.tts import TTS_MODES, TextToSpeechCapability
from audioclone.core.capabilities.vad import VoiceActivityCapability

__all__ = [
    "AudioClassificationCapability",
    "AudioToAudioCapability",
    "Capability",
    "ModelBackend",
    "SpeechRecognitionCapability",
    "TTS_MODES",
    "TextToAudioCapability",
    "TextToSpeechCapability",
    "VoiceActivityCapability",
]