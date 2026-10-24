"""DeepFilterNet (MLX) 后端 — 基于 mlx_audio 的 ``deepfilternet`` 架构，语音增强。

覆盖 A2A 能力的「增强」部分：降噪/增强人声（DeepFilterNet3）。
「变声」暂无 MLX 原生模型（sts 模块内 mossformer2/sam_audio 为分离/增强），
可用 dummy 后端体验链路，或经 TTS VoiceDesign 变通实现（合成而非转换）。

默认模型：``mlx-community/DeepFilterNet-mlx``（mlx_audio 源码内置仓库 ID）。
加载经统一的 HF 缓存解析（缺失报可读错误不下载）。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audioclone.core.capabilities import AudioToAudioCapability, ModelBackend
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.audio import resample_audio, to_mono
from audioclone.utils.model_path import resolve_model_dir

logger = logging.getLogger(__name__)

#: mlx_audio 中承载 DeepFilterNet 的架构模块名
ARCH = "deepfilternet"

#: 依赖缺失时的修复指引
_HINT = (
    "当前 mlx_audio 未提供 DeepFilterNet 架构（mlx_audio.sts.models.deepfilternet）。\n"
    "  - 请升级: uv add 'mlx-audio[tts]>=0.5.0' 或 uv sync --reinstall-package mlx-audio"
)

#: DeepFilterNet3 工作采样率（48kHz）
DFN_SAMPLE_RATE = 48000


def has_deepfilternet_support() -> bool:
    """当前环境的 ``mlx_audio`` 是否提供 DeepFilterNet 架构。"""
    try:
        return importlib.util.find_spec(f"mlx_audio.sts.models.{ARCH}") is not None
    except (ImportError, ValueError):
        return False


class DeepFilterNetMLXBackend(ModelBackend, AudioToAudioCapability):
    """DeepFilterNet MLX 后端：输入音频 → 降噪增强 → 输出音频。"""

    name = "deepfilternet-mlx"
    default_model_id = "mlx-community/DeepFilterNet-mlx"
    capabilities = (AudioToAudioCapability,)
    slot = ModelSlot.A2A

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        sample_rate: int = DFN_SAMPLE_RATE,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._model: Any = None
        self.sample_rate = sample_rate  # DFN3 工作采样率（输入输出一致）

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return
        if not has_deepfilternet_support():
            raise ModelLoadError(_HINT)

        from mlx_audio.sts.utils import load_model as mlx_load_model

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        logger.info("[A2A] 加载 DeepFilterNet (MLX): %s", model_path)
        try:
            self._model = mlx_load_model(model_path)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"DeepFilterNet (MLX) 模型加载失败: {e}") from e
        logger.info("[A2A] DeepFilterNet 加载完成，采样率: %s", self.sample_rate)

    def _close(self) -> None:
        self._model = None

    # ─── 转换 ────────────────────────────────────────────────
    def transform(
        self, audio_path: str, instruct: str = "", speed: float = 1.0, **kwargs
    ) -> np.ndarray:
        """语音增强（降噪）；``instruct`` 仅作兼容参数（本后端忽略）。"""
        self.load()
        if self._model is None:
            raise ModelLoadError("DeepFilterNet 模型未加载")

        data, sr = sf.read(audio_path, dtype="float32")
        data = to_mono(data)
        data = resample_audio(data, int(sr), self.sample_rate)

        logger.info("[A2A] DeepFilterNet 增强: %s (sr=%s)", audio_path, self.sample_rate)
        try:
            enhanced = self._model.enhance_array(data)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"DeepFilterNet 增强失败: {e}") from e
        return np.asarray(enhanced, dtype=np.float32)


__all__ = ["DeepFilterNetMLXBackend", "has_deepfilternet_support"]
