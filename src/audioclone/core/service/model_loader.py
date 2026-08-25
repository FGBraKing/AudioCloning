"""模型加载器 — 依据配置创建后端实例，并提供能力查询。"""

from __future__ import annotations

import importlib

from audioclone.config import ASR_BACKEND, ASR_BACKENDS, TTS_BACKEND, TTS_BACKENDS
from audioclone.core.capabilities import (
    AudioClassificationCapability,
    AudioToAudioCapability,
    ModelBackend,
    SpeechRecognitionCapability,
    TextToAudioCapability,
    TextToSpeechCapability,
    VoiceActivityCapability,
)

# 能力 → 人类可读名称（用于展示/日志）
CAPABILITY_LABELS = {
    TextToSpeechCapability: "text-to-speech",
    TextToAudioCapability: "text-to-audio",
    SpeechRecognitionCapability: "speech-recognition",
    AudioToAudioCapability: "audio-to-audio",
    AudioClassificationCapability: "audio-classification",
    VoiceActivityCapability: "voice-activity-detection",
}


def _resolve_class(dotted: str):
    """按 ``module.Class`` 路径导入并返回类对象。"""
    module_name, attr = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


def _create(kind: str, backends: dict, name: str | None, default: str, overrides: dict):
    name = name or default
    if name not in backends:
        raise KeyError(f"未知的{kind}后端: {name}（可用: {', '.join(backends)}）")

    spec = dict(backends[name])
    spec.update(overrides or {})
    factory_path = spec.pop("factory", None)

    # 优先从 registry 解析（收敛 dynamic import 面）；未注册时回退 config factory
    from audioclone.core.backends import registry

    cls = (
        registry.get_tts_factory(name)
        if kind == "tts"
        else registry.get_asr_factory(name)
    )
    if cls is None and factory_path:
        cls = _resolve_class(factory_path)
    if cls is None:
        raise KeyError(f"后端未在注册表登记且缺少 factory: {name}")
    return cls(**spec)


def create_tts_backend(name: str | None = None, **overrides) -> ModelBackend:
    """按名称创建 TTS 后端；未指定名则用配置默认。返回的后端应提供 ``TextToSpeechCapability``。"""
    return _create("tts", TTS_BACKENDS, name, TTS_BACKEND, overrides)


def create_asr_backend(name: str | None = None, **overrides) -> ModelBackend:
    """按名称创建 ASR 后端；未指定名则用配置默认。返回的后端应提供 ``SpeechRecognitionCapability``。"""
    return _create("asr", ASR_BACKENDS, name, ASR_BACKEND, overrides)


def list_backend_capabilities(backend: ModelBackend) -> list[str]:
    """返回某后端支持的能力的可读名称列表（反映"一个模型多任务"）。"""
    return [label for cap, label in CAPABILITY_LABELS.items() if backend.supports(cap)]


def tts_backend_names() -> tuple[str, ...]:
    """已配置的 TTS 后端名称。"""
    return tuple(TTS_BACKENDS)


def asr_backend_names() -> tuple[str, ...]:
    """已配置的 ASR 后端名称。"""
    return tuple(ASR_BACKENDS)