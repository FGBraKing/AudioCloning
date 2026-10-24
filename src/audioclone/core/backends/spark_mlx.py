"""Spark-TTS (MLX) 后端 — 0.5B 零样本克隆（Qwen2 backbone + BiCodec）。

克隆质量的社区标杆之一：离散声学 token + BiCodec 解码，zero-shot 克隆需
参考音频（``ref_audio`` 必填）与对应转录（``ref_text`` 推荐提供）。

**许可注意**：``mlx-community/Spark-TTS-0.5B-8bit`` 标注 CC BY-NC-SA 4.0
（**非商用**），商用部署前需核实 Spark-TTS 官方许可条款。

默认模型：``mlx-community/Spark-TTS-0.5B-8bit``（已核实）。加载经统一的
HF 缓存解析（缺失报可读错误不下载）。
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

#: mlx_audio 中承载 Spark-TTS 的架构模块名
ARCH = "spark"

#: 依赖缺失时的修复指引
_HINT = (
    "当前 mlx_audio 未提供 Spark-TTS 架构（mlx_audio.tts.models.spark）。\n"
    "  - 请升级: uv add 'mlx-audio[tts]>=0.5.0' 或 uv sync --reinstall-package mlx-audio"
)


def has_spark_support() -> bool:
    """当前环境的 ``mlx_audio`` 是否提供 Spark-TTS 架构。"""
    try:
        return importlib.util.find_spec(f"mlx_audio.tts.models.{ARCH}") is not None
    except (ImportError, ValueError):
        return False


class SparkMLXBackend(ModelBackend, TextToSpeechCapability):
    """Spark-TTS MLX 后端：零样本克隆（参考音频必填）。"""

    name = "spark-mlx"
    default_model_id = "mlx-community/Spark-TTS-0.5B-8bit"
    capabilities = (TextToSpeechCapability,)
    supported_modes = ("zero_shot",)   # Spark 设计上依赖参考音频，不支持无参考直读
    slot = ModelSlot.TTS

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        gender: str = "male",
        pitch: float = 1.0,
        temperature: float = 0.8,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.gender = gender             # 无参考时的性别提示（辅助采样）
        self.pitch = pitch
        self.temperature = temperature
        self._model: Any = None
        self.sample_rate = SAMPLE_RATE

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return
        if not has_spark_support():
            raise ModelLoadError(_HINT)

        from mlx_audio.tts.utils import load_model as mlx_load_model

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        logger.info("[TTS] 加载 Spark-TTS (MLX): %s", model_path)
        self._model = self._load_model(mlx_load_model, model_path)
        self.sample_rate = int(getattr(self._model, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        logger.info("[TTS] Spark-TTS 加载完成，采样率: %s", self.sample_rate)

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
                f"后端 {self.name} 仅支持推理模式 'zero_shot'（Spark 需参考音频克隆）"
            )
        if not ref_audio_path:
            raise InvalidAudioInput(
                "Spark-TTS 需提供参考音频 ref_audio（零样本克隆输入）"
            )
        if self._model is None:
            raise ModelLoadError("Spark-TTS 模型未加载")

        logger.info("[TTS] Spark-TTS zero_shot 文本: %s...", text[:50])
        result = self._model.generate(
            text,
            ref_audio=Path(ref_audio_path),
            ref_text=ref_text or None,
            gender=self.gender,
            pitch=self.pitch,
            speed=speed,
            temperature=self.temperature,
            verbose=False,
        )
        # generate 可能返回单个结果或生成器，统一收集
        items = result if hasattr(result, "__iter__") and not hasattr(result, "audio") else [result]
        chunks = [
            np.asarray(getattr(r, "audio", r), dtype=np.float32) for r in items
        ]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            raise ModelLoadError("Spark-TTS 未生成任何音频")
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


__all__ = ["SparkMLXBackend", "has_spark_support"]
