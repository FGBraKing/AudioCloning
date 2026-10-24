"""CED 音频标注 (sherpa-onnx) 后端 — AudioSet 527 类通用音频事件分类。

CPU / AMD / NVIDIA 平台的默认分类：CED（Audio Spectrogram Transformer 系），
**无 torch**，仅依赖可选组 ``onnx``（sherpa-onnx + onnxruntime）。
此前 torch AST 方案（MIT/ast-finetuned-audioset）经用户决策放弃——本后端以
sherpa-onnx 官方模型达成同样的 AudioSet 527 类能力。

默认模型：``k2-fsa/sherpa-onnx-ced-small-audio-tagging-2024-04-19``
（k2-fsa 官方 HF 仓库，已核实：``model.int8.onnx`` + ``class_labels_indices.csv``，
CED-Small ~92M）。加载经统一的 HF 缓存解析（缺失报可读错误不下载）。
"""

from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audioclone.core.capabilities import AudioClassificationCapability, ModelBackend
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.audio import resample_audio, to_mono
from audioclone.utils.model_path import resolve_model_dir

logger = logging.getLogger(__name__)

_HINT = (
    "sherpa-cls-onnx 后端需要 sherpa-onnx（onnx 可选依赖组，当前未安装）。\n"
    "  - 安装: uv sync --extra onnx（或 uv add sherpa-onnx）"
)

#: CED 输入采样率
CED_SAMPLE_RATE = 16000


def has_sherpa_onnx_support() -> bool:
    """当前环境是否安装 sherpa-onnx（可选依赖）。"""
    try:
        return importlib.util.find_spec("sherpa_onnx") is not None
    except (ImportError, ValueError):
        return False


def _find_file(model_dir: Path, *names: str) -> Path:
    """按候选名查找文件（int8 优先）。"""
    for name in names:
        path = model_dir / name
        if path.is_file():
            return path
    raise ModelLoadError(
        f"模型目录缺少必需文件: {' / '.join(names)}（目录: {model_dir}）"
    )


class SherpaClsOnnxBackend(ModelBackend, AudioClassificationCapability):
    """CED sherpa-onnx 后端：AudioSet 527 类音频事件分类。"""

    name = "sherpa-cls-onnx"
    default_model_id = "k2-fsa/sherpa-onnx-ced-small-audio-tagging-2024-04-19"
    capabilities = (AudioClassificationCapability,)
    slot = ModelSlot.CLASSIFICATION

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
        self._tagger: Any = None
        self.sample_rate = CED_SAMPLE_RATE

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._tagger is not None:
            return
        if not has_sherpa_onnx_support():
            raise ModelLoadError(_HINT)

        import sherpa_onnx

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        model_file = _find_file(model_path, "model.int8.onnx", "model.onnx")
        labels_file = _find_file(model_path, "class_labels_indices.csv")
        logger.info("[CLS] 加载 CED 音频标注 (sherpa-onnx): %s", model_file)
        try:
            config = sherpa_onnx.AudioTaggingConfig(
                model=sherpa_onnx.AudioTaggingModelConfig(
                    ced=str(model_file),
                    num_threads=self.num_threads,
                    debug=False,
                    provider="cpu",
                ),
                labels=str(labels_file),
                top_k=5,
            )
            self._tagger = sherpa_onnx.AudioTagging(config)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"CED (sherpa-onnx) 模型加载失败: {e}") from e
        logger.info("[CLS] CED 加载完成（AudioSet 527 类）")

    def _close(self) -> None:
        self._tagger = None

    # ─── 分类 ────────────────────────────────────────────────
    def classify(self, audio_path: str, top_k: int = 5, **kwargs) -> list[tuple[str, float]]:
        """AudioSet 事件分类，返回按置信度降序的 ``[(标签, 置信度)]``。"""
        self.load()
        if self._tagger is None:
            raise ModelLoadError("CED 模型未加载")

        data, sr = sf.read(audio_path, dtype="float32")
        data = to_mono(np.ascontiguousarray(data))
        data = resample_audio(data, int(sr), self.sample_rate)

        logger.info("[CLS] CED 分类: %s (top_k=%d)", audio_path, top_k)
        start = time.monotonic()
        stream = self._tagger.create_stream()
        stream.accept_waveform(sample_rate=self.sample_rate, waveform=np.asarray(data, dtype=np.float32))
        result = self._tagger.compute(stream)
        elapsed = time.monotonic() - start

        pairs: list[tuple[str, float]] = []
        for entry in result:
            try:
                name, prob = entry  # 官方返回 (标签, 置信度) 对
                pairs.append((str(name), float(prob)))
            except (TypeError, ValueError):  # 兜底：字符串形态 "label prob"
                parts = str(entry).rsplit(" ", 1)
                if len(parts) == 2:
                    pairs.append((parts[0], float(parts[1])))
        pairs.sort(key=lambda t: t[1], reverse=True)
        logger.info("[CLS] 分类完成，耗时 %.2fs", elapsed)
        return pairs[: max(1, top_k)]

    # ─── 状态 ────────────────────────────────────────────────
    def get_status(self) -> dict[str, object]:
        return {
            "model_loaded": self._tagger is not None,
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "num_classes": 527,
            "sherpa_installed": has_sherpa_onnx_support(),
        }


__all__ = ["SherpaClsOnnxBackend", "has_sherpa_onnx_support"]
