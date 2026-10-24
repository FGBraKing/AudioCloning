"""Kokoro TTS (sherpa-onnx) 后端 — onnxruntime CPU 推理，纯文本合成。

CPU / AMD 核显平台的默认 TTS：Kokoro-82M int8 多语种（中/英），82M 参数，
CPU 上远超实时。**纯文本直读合成**（``plain`` 模式）——Kokoro 不做参考音频克隆，
也不消费指令文本，音色由预置 speaker（``sid``）决定。

默认模型：``csukuangfj/kokoro-int8-multi-lang-v1_1``（官方 HF 镜像，已核实：
``model.int8.onnx`` + ``voices.bin`` + ``tokens.txt`` + ``espeak-ng-data/``）。
加载经统一的 HF 缓存解析（缺失报可读错误不下载）。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

from audioclone.core.capabilities import ModelBackend, TextToSpeechCapability
from audioclone.core.exceptions import InvalidAudioInput, ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.model_path import resolve_model_dir

logger = logging.getLogger(__name__)

_HINT = (
    "sherpa-tts-onnx 后端需要 sherpa-onnx（onnx 可选依赖组，当前未安装）。\n"
    "  - 安装: uv sync --extra onnx（或 uv add sherpa-onnx）"
)


def has_sherpa_onnx_support() -> bool:
    """当前环境是否安装 sherpa-onnx（可选依赖）。"""
    try:
        return importlib.util.find_spec("sherpa_onnx") is not None
    except (ImportError, ValueError):  # 父包缺失 / 命名空间模块无 __spec__
        return False


class SherpaTTSOnnxBackend(ModelBackend, TextToSpeechCapability):
    """Kokoro sherpa-onnx 后端：``plain`` 模式纯文本合成。"""

    name = "sherpa-tts-onnx"
    default_model_id = "csukuangfj/kokoro-int8-multi-lang-v1_1"
    capabilities = (TextToSpeechCapability,)
    supported_modes = ("plain",)   # 纯文本直读（不克隆、不消费指令）
    slot = ModelSlot.TTS

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        speaker_id: int = 0,
        num_threads: int = 2,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.speaker_id = speaker_id   # Kokoro 预置音色编号
        self.num_threads = num_threads
        self._tts: Any = None
        self.sample_rate = 24000       # Kokoro 输出采样率（加载后以模型为准）

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._tts is not None:
            return
        if not has_sherpa_onnx_support():
            raise ModelLoadError(_HINT)

        import sherpa_onnx

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        logger.info("[TTS] 加载 Kokoro (sherpa-onnx): %s", model_path)
        try:
            self._tts = sherpa_onnx.OfflineTts.from_kokoro(
                vits_model=str(model_path / "model.int8.onnx"),
                voices=str(model_path / "voices.bin"),
                tokens=str(model_path / "tokens.txt"),
                data_dir=str(model_path / "espeak-ng-data"),
                dict_dir=str(model_path / "dict"),
                num_threads=self.num_threads,
            )
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(
                f"Kokoro (sherpa-onnx) 模型加载失败: {e}\n"
                "  - 注意：from_kokoro 需 sherpa-onnx >= 1.10.30，请升级后重试"
            ) from e
        self.sample_rate = int(getattr(self._tts, "sample_rate", self.sample_rate))
        logger.info("[TTS] Kokoro 加载完成，采样率: %s", self.sample_rate)

    def _close(self) -> None:
        self._tts = None

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
                f"后端 {self.name} 仅支持推理模式 'plain'（纯文本直读，不支持克隆/指令），"
                "克隆请使用 qwen3-tts-base-mlx"
            )
        if self._tts is None:
            raise ModelLoadError("Kokoro (sherpa-onnx) 模型未加载")

        logger.info("[TTS] Kokoro 合成 (sid=%s, speed=%.1f): %s...", self.speaker_id, speed, text[:50])
        import time

        start = time.monotonic()
        output = self._tts.generate(text, sid=self.speaker_id, speed=speed)
        audio = np.asarray(output.samples, dtype=np.float32)
        if audio.size == 0:
            raise ModelLoadError("Kokoro 未生成任何音频")
        self.sample_rate = int(getattr(output, "sample_rate", self.sample_rate))
        logger.info("[TTS] Kokoro 合成完成，耗时 %.2fs", time.monotonic() - start)
        return audio


__all__ = ["SherpaTTSOnnxBackend", "has_sherpa_onnx_support"]
