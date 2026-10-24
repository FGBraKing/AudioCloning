"""模型加载器 — 依据配置创建后端实例，并提供能力查询。

工厂的唯一事实源是 ``core/backends/registry.py``；config 的 ``TTS_BACKENDS/ASR_BACKENDS``
仅承载「默认选择 + 覆盖参数」（model_dir / model_id / sample_rate 等），不再含工厂字符串。

后端名解析：显式指定（CLI/env/参数）优先；``auto`` 或未指定时按运行环境
（``core/runtime.py`` 探测的推理框架）取候选链中第一个已注册的后端。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audioclone.config import (
    A2A_BACKENDS,
    ASR_BACKENDS,
    CLASSIFICATION_BACKENDS,
    T2A_BACKENDS,
    TTS_BACKENDS,
    VAD_BACKENDS,
)
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
from audioclone.core.runtime import auto_backend_candidates

# 服务类（同包内直接导入；避免经 service/__init__ 造成循环导入）
from audioclone.core.service.asr_service import ASRService
from audioclone.core.service.audio_classification_service import AudioClassificationService
from audioclone.core.service.audio_to_audio_service import AudioToAudioService
from audioclone.core.service.text_to_audio_service import TextToAudioService
from audioclone.core.service.tts_service import TTSService
from audioclone.core.service.vad_service import VoiceActivityService
from audioclone.utils.model_path import find_cached_model, find_modelscope_model

#: 任务类型 → 业务服务类（create_service 的分派表）
KIND_SERVICE_OF: dict[str, type] = {
    "tts": TTSService,
    "asr": ASRService,
    "t2a": TextToAudioService,
    "a2a": AudioToAudioService,
    "classification": AudioClassificationService,
    "vad": VoiceActivityService,
}

logger = logging.getLogger(__name__)

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

#: 任务类型 → 后端覆盖参数表（工厂唯一事实源仍为 registry）
KIND_BACKENDS: dict[str, dict] = {
    "tts": TTS_BACKENDS,
    "asr": ASR_BACKENDS,
    "t2a": T2A_BACKENDS,
    "a2a": A2A_BACKENDS,
    "classification": CLASSIFICATION_BACKENDS,
    "vad": VAD_BACKENDS,
}


def _config_for(kind: str) -> dict:
    if kind not in KIND_BACKENDS:
        raise KeyError(f"未知任务类型: {kind}（可选: {', '.join(KIND_BACKENDS)}）")
    return KIND_BACKENDS[kind]


#: 兜底后端名集合（用于区分「自动选中」与「降级回退」日志口径）
FALLBACK_NAMES = {"dummy", "dummy-omni"}


def resolve_backend_name(kind: str, name: str | None) -> str:
    """解析后端名：显式指定原样返回；``None``/``"auto"`` 按运行环境自动选择。

    自动选择：取 ``runtime.auto_backend_candidates(kind)``（框架候选 + 兜底链）中
    **第一个已注册**的后端；全部未注册才抛 KeyError。
    """
    config = _config_for(kind)
    if name and name != "auto":
        return name
    candidates = [c for c in auto_backend_candidates(kind) if c in config]
    if not candidates:
        raise KeyError(f"{kind}: 自动选择失败，候选后端均未注册: {auto_backend_candidates(kind)}")

    # 真实后端优先于兜底后端（dummy）：模型全未就位时保持框架首选，
    # 让加载失败的下载指引指向正确的模型，而不是静默退化成 dummy
    real = [c for c in candidates if c not in FALLBACK_NAMES]
    pool = real or candidates
    available = [c for c in pool if _backend_model_available(kind, c)]
    chosen = available[0] if available else pool[0]

    if chosen in FALLBACK_NAMES:
        logger.warning(
            "[runtime] %s: 当前环境无匹配的专用后端，回退到 %s（功能受限）", kind, chosen
        )
    elif chosen != pool[0]:
        logger.warning(
            "[runtime] %s: 首选 %s 模型未就位（未下载），兜底到 %s",
            kind, pool[0], chosen,
        )
    else:
        logger.info("[runtime] %s 后端自动选择: %s", kind, chosen)
    return chosen


def _backend_model_available(kind: str, name: str) -> bool:
    """离线检查某后端的模型是否已就位（自定义目录 / HF 缓存 / ModelScope 目录）。

    仅做本地存在性检查（``local_files_only``），**绝不触发下载**；
    无 model_id 的后端（dummy 等）恒可用。
    """
    spec = _config_for(kind).get(name, {})
    model_id = spec.get("model_id")
    if not model_id:
        return True
    custom_dir = spec.get("model_dir")
    if custom_dir and Path(custom_dir).expanduser().exists():
        return True
    return bool(find_cached_model(model_id)) or bool(find_modelscope_model(model_id))


def _create(
    kind: str,
    name: str | None,
    overrides: dict,
    capability: type | None = None,
) -> ModelBackend:
    name = resolve_backend_name(kind, name)
    config = _config_for(kind)
    if name not in config:
        raise KeyError(f"未知的{kind}后端: {name}（可用: {', '.join(config)}）")

    reg_name = _TASK_ALIASES.get((kind, name), name)
    spec = dict(config[name])
    spec.update(overrides or {})
    # 创建期按能力校验（fail-fast），由 registry.create 实施
    return registry.create(reg_name, capability=capability, **spec)


#: 任务类型 → 创建期校验的能力
CAPABILITY_OF: dict[str, type | None] = {
    "tts": TextToSpeechCapability,
    "asr": SpeechRecognitionCapability,
    "t2a": TextToAudioCapability,
    "a2a": AudioToAudioCapability,
    "classification": AudioClassificationCapability,
    "vad": VoiceActivityCapability,
}


def create_tts_backend(name: str | None = None, **overrides) -> ModelBackend:
    """按名称创建 TTS 后端；未指定（或 auto）则按运行环境自动选择。返回的后端应提供 ``TextToSpeechCapability``。"""
    return _create("tts", name, overrides, capability=TextToSpeechCapability)


def create_asr_backend(name: str | None = None, **overrides) -> ModelBackend:
    """按名称创建 ASR 后端；未指定（或 auto）则按运行环境自动选择。返回的后端应提供 ``SpeechRecognitionCapability``。"""
    return _create("asr", name, overrides, capability=SpeechRecognitionCapability)


def create_backend(kind: str, name: str | None = None, **overrides) -> ModelBackend:
    """按任务类型创建后端（tts/asr/t2a/a2a/classification/vad），创建期校验能力。"""
    return _create(kind, name, overrides, capability=CAPABILITY_OF.get(kind))


def create_service(kind: str, name: str | None = None, **overrides) -> Any:
    """按任务类型创建业务服务（后端 auto 解析 + 构造期能力校验）。

    返回的服务持有后端实例；模型在服务方法首次调用时惰性加载
    （后端 ``synthesize/generate/...`` 内部均先 ``self.load()``）。
    """
    return KIND_SERVICE_OF[kind](create_backend(kind, name, **overrides))


def list_backend_capabilities(backend: ModelBackend) -> list[str]:
    """返回某后端支持的能力的可读名称列表（反映"一个模型多任务"）。"""
    return [label for cap, label in CAPABILITY_LABELS.items() if backend.supports(cap)]


def tts_backend_names() -> tuple[str, ...]:
    """已配置的 TTS 后端名称。"""
    return tuple(TTS_BACKENDS)


def asr_backend_names() -> tuple[str, ...]:
    """已配置的 ASR 后端名称。"""
    return tuple(ASR_BACKENDS)
