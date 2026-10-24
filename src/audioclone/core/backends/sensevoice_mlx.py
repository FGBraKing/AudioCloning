"""SenseVoice (MLX) 后端 — 基于 mlx_audio 的 ``sensevoice`` 架构，音频分类。

覆盖分类能力：**情感分类**（9 类）+ **音频事件**（4 类），一次前向同时输出，
带 softmax 置信度。SenseVoice-Small 本体是语音识别模型（~234M，FunASR 出品），
分类头是其内建能力——本项目「一模型多任务」理念的典型载体。

默认模型：``mlx-community/SenseVoiceSmall``（HuggingFace 已确认存在，~936MB F32，
mlx_audio 原生架构）。加载经统一的 HF 缓存解析（缺失报可读错误不下载）。

说明：
- 情感/事件候选 token ID 与 mlx_audio 上游 ``_extract_rich_info`` 硬编码一致
  （上游变更时同步更新）；标签文本经模型自身词表解码，不额外硬编码文案。
- 通用（AudioSet 式）音频事件分类如后续需要，可按项目扩展机制自实现新后端接入。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from audioclone.core.capabilities import AudioClassificationCapability, ModelBackend
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot
from audioclone.utils.model_path import resolve_model_dir

logger = logging.getLogger(__name__)

#: mlx_audio 中承载 SenseVoice 的架构模块名
ARCH = "sensevoice"

#: 依赖缺失时的修复指引
_HINT = (
    "当前 mlx_audio 未提供 SenseVoice 架构（mlx_audio.stt.models.sensevoice）。\n"
    "  - 请升级: uv add 'mlx-audio[tts]>=0.5.0' 或 uv sync --reinstall-package mlx-audio"
)

#: SenseVoice 特殊 token：log_probs 前 4 行依次为 [0]语言 [1]情感 [2]事件 [3]文本规整
_EMOTION_ROW, _EVENT_ROW = 1, 2

#: 情感 / 事件候选 token ID（与 mlx_audio 上游 `_extract_rich_info` 保持一致）
EMOTION_TOKEN_IDS = tuple(range(25001, 25010))   # happy/sad/angry/neutral/fearful/disgusted/surprised/other/unk
EVENT_TOKEN_IDS = (24993, 24995, 24997, 24999)   # Speech/BGM/Laughter/Applause


def has_sensevoice_support() -> bool:
    """当前环境的 ``mlx_audio`` 是否提供 SenseVoice 架构。"""
    try:
        return importlib.util.find_spec(f"mlx_audio.stt.models.{ARCH}") is not None
    except (ImportError, ValueError):
        return False


class SenseVoiceMLXBackend(ModelBackend, AudioClassificationCapability):
    """SenseVoice MLX 后端：情感 + 事件分类（一次前向，softmax 置信度）。"""

    name = "sensevoice-mlx"
    default_model_id = "mlx-community/SenseVoiceSmall"
    capabilities = (AudioClassificationCapability,)
    slot = ModelSlot.CLASSIFICATION

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id or self.default_model_id
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._model: Any = None
        self.sample_rate = 16000  # SenseVoice 前端要求 16kHz

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return
        if not has_sensevoice_support():
            raise ModelLoadError(_HINT)

        from mlx_audio.stt.utils import load_model as mlx_load_model

        model_path = resolve_model_dir(self.model_id, self.model_dir)
        logger.info("[CLS] 加载 SenseVoice (MLX): %s", model_path)
        try:
            self._model = mlx_load_model(model_path)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"SenseVoice (MLX) 模型加载失败: {e}") from e
        self.sample_rate = int(self._model.config.frontend_conf.fs)
        logger.info("[CLS] SenseVoice 加载完成，采样率: %s", self.sample_rate)

    def _close(self) -> None:
        self._model = None

    # ─── 分类 ────────────────────────────────────────────────
    def classify(self, audio_path: str, top_k: int = 5, **kwargs) -> list[tuple[str, float]]:
        """情感 + 事件联合分类，返回按置信度降序的 ``[(组:标签, 置信度)]``。"""
        self.load()
        if self._model is None:
            raise ModelLoadError("SenseVoice 模型未加载")

        from mlx_audio.stt.utils import load_audio

        audio_data = load_audio(str(audio_path), sr=self.sample_rate)
        feats = self._model._extract_features(mx.array(np.asarray(audio_data, dtype=np.float32)))
        log_probs = self._model(feats[None, :, :], language="auto", use_itn=False)[0]

        results: list[tuple[str, float]] = []
        for group, row, token_ids in (
            ("emotion", _EMOTION_ROW, EMOTION_TOKEN_IDS),
            ("event", _EVENT_ROW, EVENT_TOKEN_IDS),
        ):
            # numpy 花式索引取候选 token 的 logits（对 mx.array / ndarray 输入均稳定）
            candidate = np.asarray(log_probs[row])[list(token_ids)]
            probs = np.asarray(mx.softmax(mx.array(candidate), axis=-1)).tolist()
            for token_id, prob in zip(token_ids, probs, strict=True):
                results.append((f"{group}:{self._decode(token_id)}", float(prob)))

        results.sort(key=lambda t: t[1], reverse=True)
        return results[: max(1, top_k)]

    def _decode(self, token_id: int) -> str:
        """经模型自身词表解码标签文本（不硬编码文案）。"""
        try:
            return self._model._decode_tokens([token_id]).strip() or f"token_{token_id}"
        except Exception:  # noqa: BLE001 - 词表异常时退化为 token 占位
            return f"token_{token_id}"

    # ─── 状态 ────────────────────────────────────────────────
    def get_status(self) -> dict[str, object]:
        return {
            "model_loaded": self._model is not None,
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "emotion_classes": len(EMOTION_TOKEN_IDS),
            "event_classes": len(EVENT_TOKEN_IDS),
        }


__all__ = ["SenseVoiceMLXBackend", "has_sensevoice_support"]
