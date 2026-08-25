"""CosyVoice3 (MLX) 后端 — 调用 mlx_audio 的 CosyVoice3 模型。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from audioclone.config import MODEL_DIR, MODEL_ID, SAMPLE_RATE
from audioclone.core.capabilities import ModelBackend, TextToSpeechCapability

logger = logging.getLogger(__name__)


class CosyVoiceMLXBackend(ModelBackend, TextToSpeechCapability):
    """基于 mlx_audio 的 CosyVoice3 MLX 合成后端。"""

    name = "cosyvoice3-mlx"

    def __init__(self, model_id: str | None = None, model_dir: str | Path | None = None) -> None:
        self.model_id = model_id or MODEL_ID
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self._model = None
        self.sample_rate = SAMPLE_RATE  # 未加载时默认，加载后以模型报告为准

    # ─── 生命周期 ────────────────────────────────────────────
    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_audio.tts.utils import load_model as mlx_load_model

        logger.info("[TTS] 加载模型: %s", self.model_dir)
        self._model = mlx_load_model(self.model_dir)
        self.sample_rate = getattr(self._model, "sample_rate", SAMPLE_RATE)
        logger.info("[TTS] CosyVoice3 MLX 加载完成，采样率: %s", self.sample_rate)

    def close(self) -> None:
        self._model = None

    # ─── 合成 ────────────────────────────────────────────────
    def synthesize(
        self,
        mode: str,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        instruct_text: str = "",
        speed: float = 1.0,
    ) -> np.ndarray:
        self.load()
        ref_audio = self._load_ref_audio(ref_audio_path)

        kwargs: dict = {"speed": speed, "stt_model": None, "verbose": False}
        if mode == "zero_shot":
            kwargs.update({"ref_text": ref_text if ref_text else None})
        elif mode == "cross_lingual":
            kwargs.update({"ref_text": None})
        elif mode == "instruct":
            kwargs.update({"instruct_text": instruct_text})
        else:
            raise ValueError(f"未知的推理模式: {mode}")

        logger.info("[TTS] mode=%s | 文本: %s...", mode, text[:50])
        result = next(self._model.generate(text=text, ref_audio=ref_audio, **kwargs))
        return np.array(result.audio)

    def _load_ref_audio(self, ref_audio_path: str):
        """读取参考音频并重采样到目标采样率，返回 mx.array。"""
        import mlx.core as mx
        import soundfile as sf
        from scipy.signal import resample

        data, sr = sf.read(ref_audio_path, dtype="float32")
        if data.ndim > 1:  # 多声道取第一声道
            data = data[:, 0]
        if sr != self.sample_rate:
            num_samples = int(len(data) * self.sample_rate / sr)
            data = resample(data, num_samples).astype(np.float32)
        return mx.array(data)