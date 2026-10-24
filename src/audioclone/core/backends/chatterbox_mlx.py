"""Chatterbox TTS (MLX) 后端 — Resemble AI Chatterbox/Turbo，零样本克隆 + 情感控制。

与 Qwen3-TTS 互补的克隆路线：**exaggeration 情感强度旋钮**（0~2，独有能力）+
零样本克隆；不给参考音频时可用预置默认音色直读（``plain`` 模式）。

默认模型：``mlx-community/chatterbox-turbo-8bit``（已核实；标注上游 mlx-audio 库，
与本项目依赖匹配——旧版 ``Chatterbox-TTS-8bit`` 标注 mlx-audio-plus 分支，如需
可通过 model_id 覆盖）。加载经统一的 HF 缓存解析（缺失报可读错误不下载）。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

from audioclone.config import SAMPLE_RATE
from audioclone.core.capabilities import ModelBackend, TextToSpeechCapability
from audioclone.core.exceptions import InvalidAudioInput, ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.model_path import resolve_model_dir

logger = logging.getLogger(__name__)

#: mlx_audio 中承载 Chatterbox 的架构模块名
ARCH = "chatterbox"

#: 依赖缺失时的修复指引
_HINT = (
    "当前 mlx_audio 未提供 Chatterbox 架构（mlx_audio.tts.models.chatterbox）。\n"
    "  - 请升级: uv add 'mlx-audio[tts]>=0.5.0' 或 uv sync --reinstall-package mlx-audio"
)


def has_chatterbox_support() -> bool:
    """当前环境的 ``mlx_audio`` 是否提供 Chatterbox 架构。"""
    try:
        return importlib.util.find_spec(f"mlx_audio.tts.models.{ARCH}") is not None
    except (ImportError, ValueError):
        return False


class ChatterboxMLXBackend(ModelBackend, TextToSpeechCapability):
    """Chatterbox MLX 后端：零样本克隆 + 情感强度控制 + 预置音色直读。"""

    name = "chatterbox-mlx"
    default_model_id = "mlx-community/chatterbox-turbo-8bit"
    capabilities = (TextToSpeechCapability,)
    supported_modes = ("zero_shot", "plain")
    slot = ModelSlot.TTS

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        language: str = "en",
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.exaggeration = exaggeration   # 情感强度 0~2（0.4~0.7 推荐）
        self.cfg_weight = cfg_weight       # 说话速度/清晰度与情感的权衡
        self.temperature = temperature
        self.language = language
        self._model: Any = None
        self.sample_rate = SAMPLE_RATE

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return
        if not has_chatterbox_support():
            raise ModelLoadError(_HINT)

        from mlx_audio.tts.utils import load_model as mlx_load_model

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        logger.info("[TTS] 加载 Chatterbox (MLX): %s", model_path)
        self._model = self._load_model(mlx_load_model, model_path)
        self.sample_rate = int(getattr(self._model, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        logger.info("[TTS] Chatterbox 加载完成，采样率: %s", self.sample_rate)

    def _load_model(self, loader: Any, model_path: Path) -> Any:
        """显式锁定架构（规避按目录名片段误判）；旧版加载器回退默认推断。"""
        try:
            return loader(model_path, model_type=ARCH, model_name_parts=[ARCH])
        except TypeError:
            logger.warning(
                "[TTS] 当前 mlx_audio 不支持显式 model_type，回退为按目录名推断架构"
            )
            return loader(model_path)

    def _close(self) -> None:
        self._model = None

    # ─── 合成 ────────────────────────────────────────────────
    def synthesize(
        self,
        mode: str,
        text: str,
        ref_audio_path: str = "",
        ref_text: str = "",
        instruct_text: str = "",
        speed: float = 1.0,
    ) -> np.ndarray:
        self.load()
        if mode not in self.supported_modes:
            raise InvalidAudioInput(
                f"后端 {self.name} 仅支持推理模式 {', '.join(self.supported_modes)}"
                "（Chatterbox 不消费指令文本，情感控制经 exaggeration 参数）"
            )
        if self._model is None:
            raise ModelLoadError("Chatterbox 模型未加载")

        ref = ref_audio_path or None   # plain 模式：无参考音频 → 预置默认音色
        logger.info(
            "[TTS] Chatterbox mode=%s exaggeration=%.1f 文本: %s...",
            mode, self.exaggeration, text[:50],
        )
        chunks = [
            np.asarray(getattr(r, "audio", r), dtype=np.float32)
            for r in self._model.generate(
                text,
                ref_audio=ref,
                exaggeration=self.exaggeration,
                cfg_weight=self.cfg_weight,
                temperature=self.temperature,
                speed=speed,
                lang_code=self.language,
                verbose=False,
            )
        ]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            raise ModelLoadError("Chatterbox 未生成任何音频")
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


__all__ = ["ChatterboxMLXBackend", "has_chatterbox_support"]
