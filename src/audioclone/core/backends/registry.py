"""后端注册表与工厂。

单池按名注册：每个后端以唯一名称登记一个工厂类；同一模型可聚合多种能力。
任务相关的便捷别名（如 ``dummy`` 同时服务 TTS/ASR）由 ``model_loader`` 解析。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from audioclone.core.capabilities import Capability, ModelBackend
from audioclone.core.exceptions import CapabilityError

# 工厂类型：接收 **kwargs，返回后端实例
BackendFactory = Callable[..., ModelBackend]


class BackendRegistry:
    """后端注册表（单池，按名注册；按能力创建/查询）。

    用法::

        registry.register("dummy-tts", DummyTTSBackend)
        backend = registry.create("dummy-tts", sample_rate=24000)
        registry.list_backends_for(TextToSpeechCapability)  # 按能力聚合查询
    """

    def __init__(self) -> None:
        self._backends: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        """注册一个后端工厂（可按名覆盖）。"""
        self._backends[name] = factory

    def create(
        self,
        name: str,
        capability: type[Capability] | None = None,
        **cfg,
    ) -> ModelBackend:
        """按名称创建后端实例；传入 ``capability`` 时在创建期校验（fail-fast）。"""
        self._ensure(name)
        factory = self._backends[name]
        if (
            capability is not None
            and isinstance(factory, type)
            and not cast(type[ModelBackend], factory).supports(capability)
        ):
            raise CapabilityError(f"后端 {name} 不支持能力 {capability.__name__}")
        return factory(**cfg)

    def names(self) -> tuple[str, ...]:
        """已注册的后端名称。"""
        return tuple(self._backends)

    def get_factory(self, name: str) -> BackendFactory | None:
        """返回已注册的后端工厂（未注册返回 None）。"""
        return self._backends.get(name)

    def list_backends_for(self, capability: type[Capability]) -> tuple[str, ...]:
        """返回支持指定能力的所有后端名称（按 ``supports()`` 语义查询）。"""
        names: list[str] = []
        for name, factory in self._backends.items():
            if isinstance(factory, type) and cast(type[ModelBackend], factory).supports(capability):
                names.append(name)
        return tuple(names)

    def _ensure(self, name: str) -> None:
        if name not in self._backends:
            raise KeyError(f"未注册的后端: {name}（可用: {', '.join(self._backends)}）")


# 模块级单例（供需要全局注册的场景使用）
registry = BackendRegistry()


def _register_defaults() -> None:
    """预注册项目内置后端（注册表为工厂的唯一事实源）。

    选型原则：默认模型仅选 **8bit 量化（或非 LLM 极小模型）、较新较强**的小模型，
    参数量以 **≤3B** 为基线（T2A 音乐经用户决策放宽——MiniMax Music3 是唯一有
    8bit 量化版的音乐生成模型）；且优先 mlx_audio 源码内置仓库/已确认存在的
    mlx-community 仓库（不自实现、不猜仓库 ID）。默认不下载权重。
    分类经用户决策采用 mlx_audio 原生 SenseVoice（情感+事件，一模型多任务），不引入 torch。
    """
    from audioclone.core.backends.chatterbox_mlx import ChatterboxMLXBackend
    from audioclone.core.backends.deepfilternet_mlx import DeepFilterNetMLXBackend
    from audioclone.core.backends.dummy import DummyASRBackend, DummyOmnibackend, DummyTTSBackend
    from audioclone.core.backends.minimax_music_mlx import MiniMaxMusicMLXBackend
    from audioclone.core.backends.qwen3_asr_mlx import Qwen3AsrMLXBackend
    from audioclone.core.backends.qwen3_tts_mlx import (
        Qwen3TTSBaseMLXBackend,
        Qwen3TTSCustomVoiceMLXBackend,
        Qwen3TTSVoiceDesignMLXBackend,
    )
    from audioclone.core.backends.qwen3_tts_tensorrt import (
        Qwen3TTSBaseTensorRTBackend,
        Qwen3TTSCustomVoiceTensorRTBackend,
        Qwen3TTSVoiceDesignTensorRTBackend,
    )
    from audioclone.core.backends.sensevoice_mlx import SenseVoiceMLXBackend
    from audioclone.core.backends.sherpa_asr_onnx import SherpaASROnnxBackend
    from audioclone.core.backends.sherpa_tts_onnx import SherpaTTSOnnxBackend
    from audioclone.core.backends.sherpa_vad_onnx import SherpaVADOnnxBackend
    from audioclone.core.backends.silero_vad_mlx import SileroVADMLXBackend
    from audioclone.core.backends.spark_mlx import SparkMLXBackend

    registry.register("qwen3-asr-mlx", Qwen3AsrMLXBackend)
    registry.register("minimax-music-mlx", MiniMaxMusicMLXBackend)
    registry.register("deepfilternet-mlx", DeepFilterNetMLXBackend)
    registry.register("silero-vad-mlx", SileroVADMLXBackend)
    registry.register("sherpa-asr-onnx", SherpaASROnnxBackend)
    registry.register("sherpa-tts-onnx", SherpaTTSOnnxBackend)
    registry.register("sherpa-vad-onnx", SherpaVADOnnxBackend)
    registry.register("sensevoice-mlx", SenseVoiceMLXBackend)
    registry.register("qwen3-tts-base-mlx", Qwen3TTSBaseMLXBackend)
    registry.register("qwen3-tts-customvoice-mlx", Qwen3TTSCustomVoiceMLXBackend)
    registry.register("qwen3-tts-voicedesign-mlx", Qwen3TTSVoiceDesignMLXBackend)
    registry.register("chatterbox-mlx", ChatterboxMLXBackend)
    registry.register("spark-mlx", SparkMLXBackend)
    registry.register("qwen3-tts-base-tensorrt", Qwen3TTSBaseTensorRTBackend)
    registry.register("qwen3-tts-customvoice-tensorrt", Qwen3TTSCustomVoiceTensorRTBackend)
    registry.register("qwen3-tts-voicedesign-tensorrt", Qwen3TTSVoiceDesignTensorRTBackend)
    registry.register("dummy-tts", DummyTTSBackend)
    registry.register("dummy-asr", DummyASRBackend)
    registry.register("dummy-omni", DummyOmnibackend)


_register_defaults()
