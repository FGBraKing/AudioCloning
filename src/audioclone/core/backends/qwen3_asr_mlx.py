"""Qwen3-ASR (MLX) 后端 — 基于 mlx_audio 的 ``qwen3_asr`` 架构，macOS 原生推理。

CPU / AMD 核显平台的 ASR 由 sherpa-onnx（``sherpa-asr-onnx``）承担；
本后端跑在 MLX 上，是 mac (Apple Silicon) 环境的默认 ASR 后端。

加载：``mlx_audio.stt.utils.load_model``（按 model_id 解析 HF 系统缓存，缺失时报
可读错误并给下载指引，不自动下载）。转写：``model.generate(audio_path, ...)``，
长音频由 mlx_audio 内部按 ``chunk_duration`` 分段（默认 1200s，覆盖本项目的
300s 上限），后端无需自行分片。
"""

from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import Any

from audioclone.core.capabilities import ModelBackend, SpeechRecognitionCapability
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.model_path import resolve_model_dir

logger = logging.getLogger(__name__)

#: mlx_audio 中承载 Qwen3-ASR 的架构模块名
ARCH = "qwen3_asr"

#: 依赖缺失时的修复指引（日志与异常共用）
_HINT = (
    "当前 mlx_audio 未提供 Qwen3-ASR 架构（mlx_audio.stt.models.qwen3_asr）。\n"
    "  - 本项目依赖上游 mlx-audio（该模块随近期版本提供）；\n"
    "  - 请检查是否误装了 mlx-audio-plus（CosyVoice 分支，不含该模块），\n"
    "    若是请执行: uv remove mlx-audio-plus && uv add 'mlx-audio[tts]>=0.5.0'；\n"
    "  - 或直接重装: uv sync --reinstall-package mlx-audio"
)


def has_qwen3_asr_support() -> bool:
    """当前环境的 ``mlx_audio`` 是否提供 Qwen3-ASR 架构。"""
    try:
        return importlib.util.find_spec(f"mlx_audio.stt.models.{ARCH}") is not None
    except (ImportError, ValueError):  # 父包缺失 / 包名不合法
        return False


class Qwen3AsrMLXBackend(ModelBackend, SpeechRecognitionCapability):
    """Qwen3-ASR MLX 后端：进程内加载模型，直接转写音频文件。"""

    name = "qwen3-asr-mlx"
    default_model_id = "mlx-community/Qwen3-ASR-0.6B-8bit"
    capabilities = (SpeechRecognitionCapability,)
    slot = ModelSlot.ASR

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        language: str | None = None,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        # None = HF 系统缓存（加载时解析，缺失报错不下载）；显式路径优先
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.language = language   # 识别语言提示（None = 自动检测）
        self._model: Any = None
        self.sample_rate: int = 16000

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return
        if not has_qwen3_asr_support():
            raise ModelLoadError(_HINT)

        from mlx_audio.stt.utils import load_model as mlx_load_model

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        logger.info("[ASR] 加载 Qwen3-ASR (MLX): %s", model_path)
        try:
            self._model = mlx_load_model(model_path)
        except Exception as e:  # noqa: BLE001 - 包装为可读错误（权重缺失/损坏等）
            raise ModelLoadError(f"Qwen3-ASR (MLX) 模型加载失败: {e}") from e
        self.sample_rate = int(getattr(self._model, "sample_rate", 16000) or 16000)
        logger.info("[ASR] Qwen3-ASR (MLX) 加载完成，采样率: %s", self.sample_rate)

    def _close(self) -> None:
        self._model = None

    # ─── 可用性 ──────────────────────────────────────────────
    @property
    def is_model_available(self) -> bool:
        """模型已加载，或可离线解析到本地路径（缓存/自定义目录/ModelScope 目录）。"""
        if self._model is not None:
            return True
        try:
            resolve_model_dir(self.model_id, self.model_dir)
            return True
        except ModelLoadError:
            return False

    def get_status(self) -> dict[str, object]:
        return {
            "model_available": self.is_model_available,
            "model_loaded": self._model is not None,
            "model_id": self.model_id,
            "model_path": str(self.model_dir) if self.model_dir else None,
            "sample_rate": self.sample_rate,
            "language": self.language,
        }

    # ─── 识别 ────────────────────────────────────────────────
    def transcribe(self, audio_path: str) -> str:
        self.load()
        if self._model is None:
            raise ModelLoadError("Qwen3-ASR (MLX) 模型未加载")

        logger.info("[ASR] 开始识别 (MLX): %s", audio_path)
        start = time.monotonic()
        output = self._model.generate(audio_path, language=self.language, verbose=False)
        text = (getattr(output, "text", output) or "").strip()
        logger.info(
            "[ASR] 识别完成，耗时 %.2fs，结果: %s...", time.monotonic() - start, text[:80]
        )
        return text


__all__ = ["Qwen3AsrMLXBackend", "has_qwen3_asr_support"]
