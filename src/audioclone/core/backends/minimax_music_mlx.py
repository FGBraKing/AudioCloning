"""MiniMax Music (MLX) 后端 — 基于 mlx_audio 的 ``minimax_music3`` 架构，文本生成音乐。

覆盖 T2A 能力的「音乐」部分：歌曲/乐曲生成（文本提示 + 可选歌词）。

默认模型：``mlx-community/MiniMax-Music3-8bit``（HuggingFace 已确认存在；
**经用户决策放宽参数量上限**——这是 mlx-community 唯一有 8bit 量化版的音乐生成模型，
基座 11.7B，8bit 权重约 14.2GB）。加载经统一的 HF 缓存解析（缺失报可读错误不下载）。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

from audioclone.config import SAMPLE_RATE
from audioclone.core.capabilities import ModelBackend, TextToAudioCapability
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.model_path import resolve_model_dir

logger = logging.getLogger(__name__)

#: mlx_audio 中承载 MiniMax Music 的架构模块名
ARCH = "minimax_music3"

#: 依赖缺失时的修复指引
_HINT = (
    "当前 mlx_audio 未提供 MiniMax Music 架构（mlx_audio.music.models.minimax_music3）。\n"
    "  - 请升级: uv add 'mlx-audio[tts]>=0.5.0' 或 uv sync --reinstall-package mlx-audio"
)


def has_minimax_music_support() -> bool:
    """当前环境的 ``mlx_audio`` 是否提供 MiniMax Music 架构。"""
    try:
        return importlib.util.find_spec(f"mlx_audio.music.models.{ARCH}") is not None
    except (ImportError, ValueError):
        return False


class MiniMaxMusicMLXBackend(ModelBackend, TextToAudioCapability):
    """MiniMax Music MLX 后端：文本提示（+可选歌词）生成音乐。"""

    name = "minimax-music-mlx"
    default_model_id = "mlx-community/MiniMax-Music3-8bit"
    capabilities = (TextToAudioCapability,)
    slot = ModelSlot.T2A

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        sample_rate: int | None = None,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._model: Any = None
        self.sample_rate = sample_rate or SAMPLE_RATE

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return
        if not has_minimax_music_support():
            raise ModelLoadError(_HINT)

        from mlx_audio.music.utils import load_model as mlx_load_model

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        logger.info("[T2A] 加载 MiniMax Music (MLX): %s", model_path)
        try:
            self._model = mlx_load_model(model_path)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"MiniMax Music (MLX) 模型加载失败: {e}") from e
        self.sample_rate = int(getattr(self._model, "sample_rate", self.sample_rate))
        logger.info("[T2A] MiniMax Music 加载完成，采样率: %s", self.sample_rate)

    def _close(self) -> None:
        self._model = None

    # ─── 生成 ────────────────────────────────────────────────
    def text_to_audio(
        self, text: str, duration_sec: float | None = None, **kwargs
    ) -> np.ndarray:
        """由文本提示生成音乐；``lyrics`` 可经 kwargs 传入歌词。"""
        self.load()
        if self._model is None:
            raise ModelLoadError("MiniMax Music 模型未加载")

        lyrics = kwargs.pop("lyrics", None) or ""
        logger.info("[T2A] MiniMax Music 生成: %s... (lyrics=%d 字)", text[:50], len(lyrics))
        results = self._model.generate(
            text,
            lyrics=lyrics,
            gen_duration=duration_sec,
            verbose=False,
        )
        chunks = [
            np.asarray(getattr(r, "audio", r), dtype=np.float32) for r in results
        ]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            raise ModelLoadError("MiniMax Music 未生成任何音频")
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


__all__ = ["MiniMaxMusicMLXBackend", "has_minimax_music_support"]
