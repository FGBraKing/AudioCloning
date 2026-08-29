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
    """预注册项目内置后端（注册表为工厂的唯一事实源）。"""
    from audioclone.core.backends.cosyvoice_mlx import CosyVoiceMLXBackend
    from audioclone.core.backends.dummy import DummyASRBackend, DummyOmnibackend, DummyTTSBackend
    from audioclone.core.backends.qwen3_asr_llama import Qwen3AsrLlamaBackend
    from audioclone.core.backends.qwen3_tts_mlx import (
        Qwen3TTSBaseMLXBackend,
        Qwen3TTSCustomVoiceMLXBackend,
        Qwen3TTSVoiceDesignMLXBackend,
    )

    registry.register("cosyvoice3-mlx", CosyVoiceMLXBackend)
    registry.register("qwen3-asr-llama", Qwen3AsrLlamaBackend)
    registry.register("qwen3-tts-base-mlx", Qwen3TTSBaseMLXBackend)
    registry.register("qwen3-tts-customvoice-mlx", Qwen3TTSCustomVoiceMLXBackend)
    registry.register("qwen3-tts-voicedesign-mlx", Qwen3TTSVoiceDesignMLXBackend)
    registry.register("dummy-tts", DummyTTSBackend)
    registry.register("dummy-asr", DummyASRBackend)
    registry.register("dummy-omni", DummyOmnibackend)


_register_defaults()
