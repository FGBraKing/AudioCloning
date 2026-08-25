"""TTS 服务 — 业务编排：长文本分段、逐段合成、拼接、落盘。

只依赖 ``TextToSpeechCapability`` 契约，不感知具体模型实现。
推理经可重入锁串行化，避免并发调用同一非线程安全模型（如 MLX）。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from audioclone.config import (
    MAX_TEXT_LENGTH,
    OUTPUT_DIR,
    OUTPUT_KEEP_MAX,
    TTS_NORMALIZE_LOUDNESS,
    TTS_SEGMENT_GAP_SEC,
)
from audioclone.core.capabilities import TTS_MODES, TextToSpeechCapability
from audioclone.core.exceptions import CapabilityError, InvalidAudioInput
from audioclone.core.text import (
    SentenceTextSegmenter,
    TextSegmenter,
    normalize_text,
)
from audioclone.utils.audio import loudness_normalize, prune_outputs, save_audio

logger = logging.getLogger(__name__)


def _concat_with_gap(segments: list[np.ndarray], gap_samples: int) -> np.ndarray:
    """拼接音频段，并在段间插入 ``gap_samples`` 个静音样本（改善句间听感）。"""
    if not segments:
        return np.array([], dtype=np.float32)
    if gap_samples <= 0 or len(segments) == 1:
        return np.concatenate(segments)
    silence = np.zeros(gap_samples, dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        parts.append(seg)
        if i < len(segments) - 1:
            parts.append(silence)
    return np.concatenate(parts)


class TTSService:
    """TTS 业务服务。需传入一个提供 ``TextToSpeechCapability`` 的后端。"""

    def __init__(
        self,
        backend: TextToSpeechCapability,
        segmenter: TextSegmenter | None = None,
    ) -> None:
        if not isinstance(backend, TextToSpeechCapability):
            raise CapabilityError(
                f"后端 {getattr(backend, 'name', backend)} 不支持文本转语音能力 "
                "(TextToSpeechCapability)"
            )
        self.backend = backend
        self.sample_rate = backend.sample_rate
        # 可重入锁：合成间可嵌套调用（synthesize_to_file → infer_*）
        self._lock = threading.RLock()
        self._segmenter = segmenter or SentenceTextSegmenter()

    # ─── 单段推理（暴露给直接调用的场景） ─────────────────────
    def infer_zero_shot(
        self, text: str, ref_audio_path: str, ref_text: str = "", speed: float = 1.0
    ) -> np.ndarray:
        """零样本声音克隆。"""
        text = normalize_text(text)
        logger.info("[TTS] 零样本克隆 文本: %s...", text[:50])
        with self._lock:
            return self.backend.synthesize(
                "zero_shot", text, ref_audio_path, ref_text=ref_text, speed=speed
            )

    def infer_cross_lingual(
        self, text: str, ref_audio_path: str, speed: float = 1.0
    ) -> np.ndarray:
        """跨语言克隆。"""
        text = normalize_text(text)
        logger.info("[TTS] 跨语言克隆 文本: %s...", text[:50])
        with self._lock:
            return self.backend.synthesize("cross_lingual", text, ref_audio_path, speed=speed)

    def infer_instruct(
        self, text: str, instruct_text: str, ref_audio_path: str, speed: float = 1.0
    ) -> np.ndarray:
        """自然语言指令合成。"""
        text = normalize_text(text)
        logger.info("[TTS] 指令合成 指令: %s 文本: %s...", instruct_text, text[:50])
        with self._lock:
            return self.backend.synthesize(
                "instruct", text, ref_audio_path, instruct_text=instruct_text, speed=speed
            )

    # ─── 长文本编排（分段 + 落盘） ────────────────────────────
    def synthesize_to_file(
        self,
        mode: str,
        text: str,
        ref_audio_path: str = "",
        ref_text: str = "",
        instruct_text: str = "",
        speed: float = 1.0,
        filename: str | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
        keep_segments: bool = False,
    ) -> str:
        """合成并保存到文件。

        Args:
            mode: 推理模式 (zero_shot / cross_lingual / instruct)
            text: 合成文本
            ref_audio_path: 参考音频路径
            ref_text: 参考文本
            instruct_text: 指令文本
            speed: 语速 (0.5~2.0)
            filename: 输出文件名
            progress_cb: 可选进度回调 ``(当前段号, 总段数)``，每段合成后调用
            keep_segments: 是否将每一段单独落盘到 ``outputs/segments/``（便于调试/编辑）

        Returns:
            保存的音频文件路径
        """
        if mode not in TTS_MODES:
            raise InvalidAudioInput(f"未知的推理模式: {mode}")

        segments = self._segmenter.split(text, MAX_TEXT_LENGTH)
        if not segments:
            raise InvalidAudioInput("合成文本为空")

        logger.info("[TTS] 共 %d 段, 模式=%s", len(segments), mode)

        base_stem = Path(filename).stem if filename else mode
        all_audio: list[np.ndarray] = []

        for idx, seg in enumerate(segments):
            logger.info("  段 %d/%d: %s...", idx + 1, len(segments), seg[:40])
            if mode == "zero_shot":
                audio = self.infer_zero_shot(seg, ref_audio_path, ref_text, speed)
            elif mode == "cross_lingual":
                audio = self.infer_cross_lingual(seg, ref_audio_path, speed)
            else:
                audio = self.infer_instruct(seg, instruct_text, ref_audio_path, speed)

            if keep_segments:
                self._save_segment(audio, idx + 1, base_stem)
            all_audio.append(audio)
            if progress_cb is not None:
                progress_cb(idx + 1, len(segments))

        final_audio = _concat_with_gap(all_audio, int(self.sample_rate * TTS_SEGMENT_GAP_SEC))
        if TTS_NORMALIZE_LOUDNESS:
            final_audio = loudness_normalize(final_audio, self.sample_rate)

        if not filename:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{mode}_{ts}.wav"
        output_path = str(OUTPUT_DIR / filename)
        save_audio(final_audio, output_path, self.sample_rate)
        prune_outputs(OUTPUT_DIR, OUTPUT_KEEP_MAX)

        logger.info("[TTS] 合成完成 输出: %s", output_path)
        return output_path

    @staticmethod
    def _save_segment(audio: np.ndarray, index: int, stem: str) -> None:
        """将单个分段落盘到 ``outputs/segments/<stem>_seg_<NN>.wav``。"""
        seg_dir = OUTPUT_DIR / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        path = str(seg_dir / f"{stem}_seg_{index:02d}.wav")
        save_audio(audio, path)
        logger.info("    已落盘分段: %s", path)

    def get_device_info(self) -> dict[str, Any]:
        """获取当前模型状态信息。"""
        info = getattr(self.backend, "device_info", None)
        if info is not None:
            return info
        return {
            "backend": getattr(self.backend, "name", type(self.backend).__name__),
            "sample_rate": self.sample_rate,
            "model_loaded": True,
        }