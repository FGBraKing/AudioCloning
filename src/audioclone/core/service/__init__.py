"""核心服务编排层 — 每个任务对应一个 Service，按能力消费后端。

所有 Service 在构造时校验后端具备对应能力（``Capability``），
业务逻辑（校验、编排）与具体模型解耦。
"""

from __future__ import annotations

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

__all__ = [
    "ASRService",
    "AudioClassificationService",
    "AudioToAudioService",
    "TTSService",
    "TextToAudioService",
    "VoiceActivityService",
    "create_asr_backend",
    "create_backend",
    "create_service",
    "create_tts_backend",
    "list_backend_capabilities",
]