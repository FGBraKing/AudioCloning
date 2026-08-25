"""模型加载器 — 依据配置创建后端实例，并提供能力查询。

工厂的唯一事实源是 ``core/backends/registry.py``；config 的 ``TTS_BACKENDS/ASR_BACKENDS``
仅承载「默认选择 + 覆盖参数」（model_dir / model_id / sample_rate 等），不再含工厂字符串。
"""

from __future__ import annotations

from audioclone.config import ASR_BACKEND, ASR_BACKENDS, TTS_BACKEND, TTS_BACKENDS
from audioclone.core.backends.registry import registry
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

# 任务相关的便捷别名：``dummy`` 同时服务 TTS/ASR（单池内以 dummy-tts / dummy-asr 区分）
_TASK_ALIASES: dict[tuple[str, str], str] = {
    ("tts", "dummy"): "dummy-tts",
    ("asr", "dummy"): "dummy-asr",
}


def _create(
    kind: str,
    name: str | None,
    default: str,
    overrides: dict,
    capability: type | None = None,
) -> ModelBackend:
    name = name or default
    config = TTS_BACKENDS if kind == "tts" else ASR_BACKENDS
    if name not in config:
        raise KeyError(f"未知的{kind}后端: {name}（可用: {', '.join(config)}）")

    reg_name = _TASK_ALIASES.get((kind, name), name)
    spec = dict(config[name])
    spec.update(overrides or {})
    # 创建期按能力校验（fail-fast），由 registry.create 实施
    return registry.create(reg_name, capability=capability, **spec)


def create_tts_backend(name: str | None = None, **overrides) -> ModelBackend:
    """按名称创建 TTS 后端；未指定名则用配置默认。返回的后端应提供 ``TextToSpeechCapability``。"""
    return _create("tts", name, TTS_BACKEND, overrides, capability=TextToSpeechCapability)


def create_asr_backend(name: str | None = None, **overrides) -> ModelBackend:
    """按名称创建 ASR 后端；未指定名则用配置默认。返回的后端应提供 ``SpeechRecognitionCapability``。"""
    return _create("asr", name, ASR_BACKEND, overrides, capability=SpeechRecognitionCapability)


def list_backend_capabilities(backend: ModelBackend) -> list[str]:
    """返回某后端支持的能力的可读名称列表（反映"一个模型多任务"）。"""
    return [label for cap, label in CAPABILITY_LABELS.items() if backend.supports(cap)]


def tts_backend_names() -> tuple[str, ...]:
    """已配置的 TTS 后端名称。"""
    return tuple(TTS_BACKENDS)


def asr_backend_names() -> tuple[str, ...]:
    """已配置的 ASR 后端名称。"""
    return tuple(ASR_BACKENDS)
