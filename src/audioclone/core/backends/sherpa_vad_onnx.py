"""Silero VAD (sherpa-onnx) 后端 — onnxruntime CPU 推理，语音活动检测。

CPU / AMD 核显平台的默认 VAD：sherpa-onnx 原生消费 Silero VAD onnx 模型，
**无子进程、无 torch**，仅依赖可选组 ``onnx``。

默认模型：``onnx-community/silero-vad``（官方 onnx-community 转换，已核实存在，
约 2MB，非 LLM 无量化概念）。加载经统一的 HF 缓存解析（缺失报可读错误不下载）。
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

_HINT = (
    "sherpa-vad-onnx 后端需要 sherpa-onnx（onnx 可选依赖组，当前未安装）。\n"
    "  - 安装: uv sync --extra onnx（或 uv add sherpa-onnx）"
)


def has_sherpa_onnx_support() -> bool:
    """当前环境是否安装 sherpa-onnx（可选依赖）。"""
    try:
        return importlib.util.find_spec("sherpa_onnx") is not None
    except (ImportError, ValueError):  # 父包缺失 / 命名空间模块无 __spec__
        return False


class SherpaVADOnnxBackend(ModelBackend, VoiceActivityCapability):
    """Silero VAD sherpa-onnx 后端：返回语音片段起止时刻。"""

    name = "sherpa-vad-onnx"
    default_model_id = "onnx-community/silero-vad"
    capabilities = (VoiceActivityCapability,)
    slot = ModelSlot.VAD

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        threshold: float = 0.5,
        min_silence_sec: float = 0.25,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.threshold = threshold
        self.min_silence_sec = min_silence_sec
        self._vad: Any = None
        self._model_file: Path | None = None   # 加载后记录，vad() 重建检测器用
        self.sample_rate = 16000  # Silero VAD 工作采样率

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._vad is not None:
            return
        if not has_sherpa_onnx_support():
            raise ModelLoadError(_HINT)


        model_path = resolve_model_dir(self.model_id, self.model_dir)
        model_file = model_path / "silero_vad.onnx"
        if not model_file.is_file():
            candidates = list(model_path.glob("*.onnx"))
            if not candidates:
                raise ModelLoadError(f"模型目录缺少 .onnx 文件: {model_path}")
            model_file = candidates[0]
        logger.info("[VAD] 加载 Silero VAD (sherpa-onnx): %s", model_file)
        self._model_file = model_file
        self._vad = self._build_detector(threshold=self.threshold)
        logger.info("[VAD] Silero VAD 加载完成，采样率: %s", self.sample_rate)

    def _build_detector(self, threshold: float) -> Any:
        """构建 VAD 检测器实例（加载与每次检测共用，避免跨请求残留缓冲）。"""
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=str(self._model_file),
                threshold=threshold,
                min_silence_duration=self.min_silence_sec,
            ),
            sample_rate=self.sample_rate,
        )
        return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=3600)

    def _close(self) -> None:
        self._vad = None

    # ─── 检测 ────────────────────────────────────────────────
    def vad(self, audio_path: str, **kwargs) -> list[tuple[float, float]]:
        """检测语音片段；``threshold`` 可经 kwargs 覆盖实例默认值。"""
        self.load()
        if self._vad is None:
            raise ModelLoadError("Silero VAD (sherpa-onnx) 模型未加载")

        threshold = float(kwargs.pop("threshold", self.threshold))
        data, sr = sf.read(audio_path, dtype="float32")
        data = to_mono(data)
        data = resample_audio(data, int(sr), self.sample_rate)

        logger.info("[VAD] Silero VAD 检测 (sherpa-onnx): %s", audio_path)
        start = time.monotonic()
        detector = self._build_detector(threshold=threshold)
        window = self.sample_rate  # 1s 窗口喂入
        for i in range(0, len(data), window):
            detector.accept_waveform(data[i : i + window])
        detector.flush()
        segments = [
            (seg.start / self.sample_rate, (seg.start + len(seg.samples)) / self.sample_rate)
            for seg in detector.segments
        ]
        logger.info(
            "[VAD] 检测完成，耗时 %.2fs，%d 个语音片段", time.monotonic() - start, len(segments)
        )
        return segments

    def get_status(self) -> dict[str, object]:
        return {
            "model_loaded": self._vad is not None,
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "threshold": self.threshold,
            "sherpa_installed": has_sherpa_onnx_support(),
        }


__all__ = ["SherpaVADOnnxBackend", "has_sherpa_onnx_support"]
