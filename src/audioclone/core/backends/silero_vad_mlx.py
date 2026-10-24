"""Silero VAD (MLX) 后端 — 基于 mlx_audio 的 ``silero_vad`` 架构，语音活动检测。

覆盖 VAD 能力：检测音频中的说话片段起止时刻。

默认模型：``mlx-community/silero-vad-v6``（HuggingFace 已确认存在，MIT 协议，
v6 为较新版本；模型仅约 2MB，非 LLM 无量化概念）。加载经统一的 HF 缓存解析
（缺失报可读错误不下载）。
"""

from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from audioclone.core.capabilities import ModelBackend, VoiceActivityCapability
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.audio import resample_audio, to_mono
from audioclone.utils.model_path import resolve_model_dir

logger = logging.getLogger(__name__)

#: mlx_audio 中承载 Silero VAD 的架构模块名
ARCH = "silero_vad"

#: 依赖缺失时的修复指引
_HINT = (
    "当前 mlx_audio 未提供 Silero VAD 架构（mlx_audio.vad.models.silero_vad）。\n"
    "  - 请升级: uv add 'mlx-audio[tts]>=0.5.0' 或 uv sync --reinstall-package mlx-audio"
)


def has_silero_vad_support() -> bool:
    """当前环境的 ``mlx_audio`` 是否提供 Silero VAD 架构。"""
    try:
        return importlib.util.find_spec(f"mlx_audio.vad.models.{ARCH}") is not None
    except (ImportError, ValueError):
        return False


class SileroVADMLXBackend(ModelBackend, VoiceActivityCapability):
    """Silero VAD MLX 后端：返回语音片段起止时刻。"""

    name = "silero-vad-mlx"
    default_model_id = "mlx-community/silero-vad-v6"
    capabilities = (VoiceActivityCapability,)
    slot = ModelSlot.VAD

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._model: Any = None
        self.sample_rate = 16000  # Silero VAD 工作采样率

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return
        if not has_silero_vad_support():
            raise ModelLoadError(_HINT)

        from mlx_audio.vad.utils import load_model as mlx_load_model

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        logger.info("[VAD] 加载 Silero VAD (MLX): %s", model_path)
        try:
            self._model = mlx_load_model(model_path)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"Silero VAD (MLX) 模型加载失败: {e}") from e
        self.sample_rate = int(getattr(self._model, "sample_rate", self.sample_rate))
        logger.info("[VAD] Silero VAD 加载完成，采样率: %s", self.sample_rate)

    def _close(self) -> None:
        self._model = None

    # ─── 检测 ────────────────────────────────────────────────
    def vad(self, audio_path: str, **kwargs) -> list[tuple[float, float]]:
        """检测语音片段；``threshold`` 可经 kwargs 调整（默认用模型内置阈值）。"""
        self.load()
        if self._model is None:
            raise ModelLoadError("Silero VAD 模型未加载")

        data, sr = sf.read(audio_path, dtype="float32")
        data = to_mono(data)
        data = resample_audio(data, int(sr), self.sample_rate)

        threshold = kwargs.pop("threshold", None)
        logger.info("[VAD] Silero VAD 检测: %s", audio_path)
        start = time.monotonic()
        generate_kwargs = {"threshold": threshold} if threshold is not None else {}
        output = self._model.generate(data, sample_rate=self.sample_rate, **generate_kwargs)
        segments = [
            (float(seg["start"]), float(seg["end"]))
            for seg in getattr(output, "timestamps", [])
        ]
        logger.info(
            "[VAD] 检测完成，耗时 %.2fs，%d 个语音片段", time.monotonic() - start, len(segments)
        )
        return segments


__all__ = ["SileroVADMLXBackend", "has_silero_vad_support"]
