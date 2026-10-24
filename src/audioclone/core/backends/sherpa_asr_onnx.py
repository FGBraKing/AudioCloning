"""SenseVoice (sherpa-onnx) 后端 — onnxruntime CPU 推理，语音识别。

CPU / AMD 核显平台的默认 ASR：sherpa-onnx（Next-gen Kaldi）音频专用推理，
**无子进程、无 torch**，仅依赖可选组 ``onnx``（sherpa-onnx + onnxruntime）。

默认模型：``csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17``
（官方 HF 镜像，已核实：``model.int8.onnx`` + ``tokens.txt``，SenseVoice-Small，
中/英/日/韩/粤）。加载经统一的 HF 缓存解析（缺失报可读错误不下载）。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

from audioclone.core.capabilities import ModelBackend, SpeechRecognitionCapability
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.audio import resample_audio, to_mono
from audioclone.utils.model_path import resolve_model_dir

logger = logging.getLogger(__name__)

#: 依赖缺失时的修复指引（sherpa-onnx 为可选依赖组）
_HINT = (
    "sherpa-asr-onnx 后端需要 sherpa-onnx（onnx 可选依赖组，当前未安装）。\n"
    "  - 安装: uv sync --extra onnx（或 uv add sherpa-onnx）\n"
    "  - 不安装则该后端不可用（加载报错/候选链跳过）"
)

#: ASR 输入采样率（SenseVoice 前端要求）
ASR_ONNX_SAMPLE_RATE = 16000


def has_sherpa_onnx_support() -> bool:
    """当前环境是否安装 sherpa-onnx（可选依赖）。"""
    try:
        return importlib.util.find_spec("sherpa_onnx") is not None
    except (ImportError, ValueError):  # 父包缺失 / 命名空间模块无 __spec__
        return False


def _find_file(model_dir: Path, *names: str) -> Path:
    """在模型目录中按候选名查找文件（如 int8 优先，回退 fp32）。"""
    for name in names:
        path = model_dir / name
        if path.is_file():
            return path
    raise ModelLoadError(
        f"模型目录缺少必需文件: {' / '.join(names)}（目录: {model_dir}）"
    )


class SherpaASROnnxBackend(ModelBackend, SpeechRecognitionCapability):
    """SenseVoice sherpa-onnx 后端：onnxruntime 进程内推理。"""

    name = "sherpa-asr-onnx"
    default_model_id = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    capabilities = (SpeechRecognitionCapability,)
    slot = ModelSlot.ASR

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        num_threads: int = 2,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.num_threads = num_threads
        self._recognizer: Any = None
        self.sample_rate = ASR_ONNX_SAMPLE_RATE

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._recognizer is not None:
            return
        if not has_sherpa_onnx_support():
            raise ModelLoadError(_HINT)

        import sherpa_onnx

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        model_file = _find_file(model_path, "model.int8.onnx", "model.onnx")
        tokens_file = _find_file(model_path, "tokens.txt")
        logger.info("[ASR] 加载 SenseVoice (sherpa-onnx): %s", model_file)
        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_file),
                tokens=str(tokens_file),
                num_threads=self.num_threads,
            )
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"SenseVoice (sherpa-onnx) 模型加载失败: {e}") from e
        logger.info("[ASR] SenseVoice (sherpa-onnx) 加载完成")

    def _close(self) -> None:
        self._recognizer = None

    # ─── 可用性 ──────────────────────────────────────────────
    @property
    def is_model_available(self) -> bool:
        """sherpa-onnx 已安装且模型可离线解析（缓存/自定义目录/ModelScope 目录）。"""
        if not has_sherpa_onnx_support():
            return False
        try:
            model_path = resolve_model_dir(self.model_id, self.model_dir)
        except ModelLoadError:
            return False
        try:
            _find_file(model_path, "model.int8.onnx", "model.onnx")
            _find_file(model_path, "tokens.txt")
            return True
        except ModelLoadError:
            return False

    def is_ready(self) -> bool:
        return self._recognizer is not None

    def get_status(self) -> dict[str, object]:
        return {
            "model_available": self.is_model_available,
            "model_loaded": self._recognizer is not None,
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "sherpa_installed": has_sherpa_onnx_support(),
        }

    # ─── 识别 ────────────────────────────────────────────────
    def transcribe(self, audio_path: str) -> str:
        self.load()
        if self._recognizer is None:
            raise ModelLoadError("SenseVoice (sherpa-onnx) 模型未加载")

        import numpy as np
        import soundfile as sf

        data, sr = sf.read(audio_path, dtype="float32")
        data = to_mono(data)
        data = resample_audio(data, int(sr), self.sample_rate)

        logger.info("[ASR] 开始识别 (sherpa-onnx): %s", audio_path)
        import time

        start = time.monotonic()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, np.asarray(data, dtype=np.float32))
        self._recognizer.decode_stream(stream)
        text = (stream.result.text or "").strip()
        logger.info("[ASR] 识别完成，耗时 %.2fs，结果: %s...", time.monotonic() - start, text[:80])
        return text


__all__ = ["SherpaASROnnxBackend", "has_sherpa_onnx_support"]
