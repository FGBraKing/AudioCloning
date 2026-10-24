"""GTCRN 语音增强 (sherpa-onnx) 后端 — onnxruntime CPU 推理，降噪增强。

CPU / AMD / NVIDIA 平台的默认 A2A（增强）：GTCRN 仅 ~236K 参数（极轻），
sherpa-onnx 官方支持（``OfflineSpeechDenoiser``）。

分发：GitHub Release（``gtcrn_simple.onnx``，k2-fsa/sherpa-onnx 的
``speech-enhancement-models`` tag）——**不在 HuggingFace**，下载经
``download_github_release_asset``（本地目录 ``pretrained_models/gtcrn_simple/``）。
"""

from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audioclone.core.capabilities import AudioToAudioCapability, ModelBackend
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.audio import resample_audio, to_mono

logger = logging.getLogger(__name__)

_HINT = (
    "sherpa-denoise-onnx 后端需要 sherpa-onnx（onnx 可选依赖组，当前未安装）。\n"
    "  - 安装: uv sync --extra onnx（或 uv add sherpa-onnx）"
)

#: GTCRN 工作采样率
GTCRN_SAMPLE_RATE = 16000


def has_sherpa_onnx_support() -> bool:
    """当前环境是否安装 sherpa-onnx（可选依赖）。"""
    try:
        return importlib.util.find_spec("sherpa_onnx") is not None
    except (ImportError, ValueError):
        return False


class SherpaDenoiseOnnxBackend(ModelBackend, AudioToAudioCapability):
    """GTCRN sherpa-onnx 后端：输入音频 → 降噪增强 → 输出音频。"""

    name = "sherpa-denoise-onnx"
    # 无 HF 仓库：GitHub Release 分发（spec.github_release），model_id 置空
    default_model_id = ""
    capabilities = (AudioToAudioCapability,)
    slot = ModelSlot.A2A

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        num_threads: int = 1,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.num_threads = num_threads
        self._denoiser: Any = None
        self.sample_rate = GTCRN_SAMPLE_RATE

    # ─── 路径解析（GitHub Release 分发，不走 HF 语义） ────────
    def _resolve_model_path(self) -> Path:
        """模型目录：显式 model_dir 优先，否则 pretrained_models/gtcrn_simple/。"""
        from audioclone.config import PROJECT_ROOT

        if self.model_dir:
            if not self.model_dir.exists():
                raise ModelLoadError(f"自定义模型目录不存在: {self.model_dir}")
            return self.model_dir
        return PROJECT_ROOT / "pretrained_models" / "gtcrn_simple"

    @property
    def is_model_available(self) -> bool:
        """sherpa-onnx 已安装且本地模型目录含 .onnx 文件（离线检查）。"""
        if not has_sherpa_onnx_support():
            return False
        try:
            model_path = self._resolve_model_path()
        except ModelLoadError:
            return False
        return model_path.exists() and any(model_path.glob("*.onnx"))

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._denoiser is not None:
            return
        if not has_sherpa_onnx_support():
            raise ModelLoadError(_HINT)

        import sherpa_onnx

        model_path = self._resolve_model_path()
        model_file = model_path / "gtcrn_simple.onnx"
        if not model_file.is_file():
            candidates = list(model_path.glob("*.onnx"))
            if not candidates:
                raise ModelLoadError(
                    f"模型目录缺少 .onnx 文件: {model_path}\n"
                    "  - 下载: WebUI 总览页「模型下载」卡片（a2a 任务），或"
                    " audioclone-download-a2a；源为 GitHub Release"
                )
            model_file = candidates[0]
        logger.info("[A2A] 加载 GTCRN (sherpa-onnx): %s", model_file)
        try:
            config = sherpa_onnx.OfflineSpeechDenoiserConfig(
                model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                    gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                        model=str(model_file)
                    ),
                    debug=False,
                    num_threads=self.num_threads,
                    provider="cpu",
                )
            )
            self._denoiser = sherpa_onnx.OfflineSpeechDenoiser(config)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"GTCRN (sherpa-onnx) 模型加载失败: {e}") from e
        logger.info("[A2A] GTCRN 加载完成，采样率: %s", self.sample_rate)

    def _close(self) -> None:
        self._denoiser = None

    # ─── 转换 ────────────────────────────────────────────────
    def transform(
        self, audio_path: str, instruct: str = "", speed: float = 1.0, **kwargs
    ) -> np.ndarray:
        """语音增强（降噪）；``instruct`` 仅作兼容参数（本后端忽略）。"""
        self.load()
        if self._denoiser is None:
            raise ModelLoadError("GTCRN 模型未加载")

        data, sr = sf.read(audio_path, dtype="float32")
        data = to_mono(np.ascontiguousarray(data))
        data = resample_audio(data, int(sr), self.sample_rate)

        logger.info("[A2A] GTCRN 增强: %s (sr=%s)", audio_path, self.sample_rate)
        start = time.monotonic()
        try:
            denoised = self._denoiser(np.asarray(data, dtype=np.float32), self.sample_rate)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"GTCRN 增强失败: {e}") from e
        self.sample_rate = int(getattr(denoised, "sample_rate", self.sample_rate))
        logger.info("[A2A] GTCRN 增强完成，耗时 %.2fs", time.monotonic() - start)
        return np.asarray(denoised.samples, dtype=np.float32)

    # ─── 状态 ────────────────────────────────────────────────
    def get_status(self) -> dict[str, object]:
        return {
            "model_loaded": self._denoiser is not None,
            "sample_rate": self.sample_rate,
            "sherpa_installed": has_sherpa_onnx_support(),
        }


__all__ = ["SherpaDenoiseOnnxBackend", "has_sherpa_onnx_support"]
