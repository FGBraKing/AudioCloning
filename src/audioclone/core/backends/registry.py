"""后端注册表与工厂。"""

from __future__ import annotations

from collections.abc import Callable

from audioclone.core.capabilities import ModelBackend

# 工厂类型：接收 **kwargs，返回后端实例
BackendFactory = Callable[..., ModelBackend]


class BackendRegistry:
    """TTS / ASR 后端的命名注册表。

    用法::

        registry.register_tts("dummy", DummyTTSBackend)
        backend = registry.create_tts("dummy", sample_rate=24000)
    """

    def __init__(self) -> None:
        self._tts: dict[str, BackendFactory] = {}
        self._asr: dict[str, BackendFactory] = {}

    def register_tts(self, name: str, factory: BackendFactory) -> None:
        """注册一个 TTS 后端工厂（可按名覆盖）。"""
        self._tts[name] = factory

    def register_asr(self, name: str, factory: BackendFactory) -> None:
        """注册一个 ASR 后端工厂（可按名覆盖）。"""
        self._asr[name] = factory

    def create_tts(self, name: str, **cfg) -> ModelBackend:
        """按名称创建 TTS 后端实例。"""
        self._ensure(name, self._tts, "TTS")
        return self._tts[name](**cfg)

    def create_asr(self, name: str, **cfg) -> ModelBackend:
        """按名称创建 ASR 后端实例。"""
        self._ensure(name, self._asr, "ASR")
        return self._asr[name](**cfg)

    def tts_names(self) -> tuple[str, ...]:
        """已注册的 TTS 后端名称。"""
        return tuple(self._tts)

    def asr_names(self) -> tuple[str, ...]:
        """已注册的 ASR 后端名称。"""
        return tuple(self._asr)

    def _ensure(self, name: str, pool: dict, kind: str) -> None:
        if name not in pool:
            raise KeyError(f"未注册的{kind}后端: {name}（可用: {', '.join(pool)}）")

    def get_tts_factory(self, name: str):
        """返回已注册的 TTS 后端工厂（未注册返回 None）。"""
        return self._tts.get(name)

    def get_asr_factory(self, name: str):
        """返回已注册的 ASR 后端工厂（未注册返回 None）。"""
        return self._asr.get(name)


# 模块级单例（供需要全局注册的场景使用）
registry = BackendRegistry()


def _register_defaults() -> None:
    """预注册项目内置后端，收敛 model_loader 的 dynamic import 面。

    通过 importlib 在此处登记具体实现类；此后 model_loader 优先从 registry 解析。
    """
    from audioclone.core.backends.cosyvoice_mlx import CosyVoiceMLXBackend
    from audioclone.core.backends.dummy import DummyASRBackend, DummyOmnibackend, DummyTTSBackend
    from audioclone.core.backends.qwen3_asr_llama import Qwen3AsrLlamaBackend

    registry.register_tts("cosyvoice3-mlx", CosyVoiceMLXBackend)
    registry.register_tts("dummy", DummyTTSBackend)
    registry.register_tts("dummy-omni", DummyOmnibackend)

    registry.register_asr("qwen3-asr-llama", Qwen3AsrLlamaBackend)
    registry.register_asr("dummy", DummyASRBackend)
    registry.register_asr("dummy-omni", DummyOmnibackend)


_register_defaults()