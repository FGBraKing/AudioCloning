"""核心领域 — 能力抽象层 + 模型实现层 + 服务编排层 + 文本处理。

依赖方向: 展示层 → 服务层 → 能力接口 ← 模型实现层。
"""

from __future__ import annotations

from audioclone.core.capabilities import (
    TTS_MODES,
    AudioClassificationCapability,
    AudioToAudioCapability,
    Capability,
    ModelBackend,
    SpeechRecognitionCapability,
    TextToAudioCapability,
    TextToSpeechCapability,
    VoiceActivityCapability,
)
from audioclone.core.exceptions import ModelSlotError
from audioclone.core.model_slot import ModelSlot, model_slots
from audioclone.core.service.asr_service import ASRService
from audioclone.core.service.audio_classification_service import AudioClassificationService
from audioclone.core.service.audio_to_audio_service import AudioToAudioService
from audioclone.core.service.model_loader import (
    create_asr_backend,
    create_backend,
    create_service,
    create_tts_backend,
    list_backend_capabilities,
)
from audioclone.core.service.text_to_audio_service import TextToAudioService
from audioclone.core.service.tts_service import TTSService
from audioclone.core.service.vad_service import VoiceActivityService
from audioclone.core.text import normalize_text, split_long_text

__all__ = [
    "ASRService",
    "AudioClassificationCapability",
    "AudioClassificationService",
    "AudioToAudioCapability",
    "AudioToAudioService",
    "Capability",
    "ModelBackend",
    "ModelSlot",
    "ModelSlotError",
    "SpeechRecognitionCapability",
    "TTS_MODES",
    "TTSService",
    "TextToAudioCapability",
    "TextToAudioService",
    "TextToSpeechCapability",
    "VoiceActivityCapability",
    "VoiceActivityService",
    "create_asr_backend",
    "create_backend",
    "create_service",
    "create_tts_backend",
    "list_backend_capabilities",
    "model_slots",
    "normalize_text",
    "split_long_text",
]