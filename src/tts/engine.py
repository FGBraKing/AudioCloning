"""
TTS 推理引擎 — 封装 mlx-audio CosyVoice3 API
支持零样本克隆、跨语言合成、指令控制。
"""

import time
import logging
from typing import Optional, List
from pathlib import Path

import numpy as np
import mlx.core as mx
import soundfile as sf

from ..config import SAMPLE_RATE, OUTPUT_DIR, MODEL_DIR
from .text_processor import split_long_text, normalize_text

logger = logging.getLogger(__name__)


def load_model(model_dir=None):
    """
    加载 CosyVoice3 MLX 模型。

    Args:
        model_dir: 模型目录路径，默认使用 config.MODEL_DIR

    Returns:
        mlx_audio CosyVoice3 Model 实例
    """
    from mlx_audio.tts.utils import load_model as mlx_load_model

    target_dir = Path(model_dir) if model_dir else MODEL_DIR
    logger.info(f"[模型加载] 目录: {target_dir}")

    model = mlx_load_model(target_dir)
    logger.info(f"[模型加载] CosyVoice3 MLX 加载完成，采样率: {model.sample_rate}")

    return model


class TTSEngine:
    """CosyVoice3 MLX 推理引擎"""

    def __init__(self, mlx_model):
        """
        Args:
            mlx_model: mlx_audio load_model 返回的 Model 实例
        """
        self.model = mlx_model
        self.sample_rate = mlx_model.sample_rate

    def _load_ref_audio(self, ref_audio_path: str) -> mx.array:
        """加载参考音频并转为 24kHz mx.array"""
        data, sr = sf.read(ref_audio_path, dtype="float32")
        # 如果是多声道，取第一声道
        if data.ndim > 1:
            data = data[:, 0]
        # 重采样到 24kHz
        if sr != SAMPLE_RATE:
            from scipy.signal import resample
            num_samples = int(len(data) * SAMPLE_RATE / sr)
            data = resample(data, num_samples).astype(np.float32)
        return mx.array(data)

    def infer_zero_shot(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        speed: float = 1.0,
    ) -> np.ndarray:
        """
        零样本声音克隆。

        Args:
            text: 合成文本
            ref_audio_path: 参考音频路径
            ref_text: 参考音频对应的文字（可选，提升克隆精度）
            speed: 语速 (0.5~2.0)

        Returns:
            numpy 音频数组
        """
        text = normalize_text(text)
        ref_audio = self._load_ref_audio(ref_audio_path)

        logger.info(f"[零样本克隆] 文本: {text[:50]}... | 参考: {ref_audio_path}")
        start = time.time()

        # stt_model=None 跳过自动转录（我们提供了 ref_text 或使用跨语言模式）
        result = next(self.model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text if ref_text else None,
            speed=speed,
            stt_model=None,
            verbose=False,
        ))

        elapsed = time.time() - start
        logger.info(f"[零样本克隆] 完成，耗时 {elapsed:.2f}s")
        return np.array(result.audio)

    def infer_cross_lingual(
        self,
        text: str,
        ref_audio_path: str,
        speed: float = 1.0,
    ) -> np.ndarray:
        """
        跨语言克隆（不提供 ref_text，不使用自动转录）。

        Args:
            text: 合成文本（目标语言）
            ref_audio_path: 参考音频路径（源语言音色）
            speed: 语速 (0.5~2.0)

        Returns:
            numpy 音频数组
        """
        text = normalize_text(text)
        ref_audio = self._load_ref_audio(ref_audio_path)

        logger.info(f"[跨语言克隆] 文本: {text[:50]}...")
        start = time.time()

        # 无 ref_text + stt_model=None → 跨语言模式
        result = next(self.model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=None,
            speed=speed,
            stt_model=None,
            verbose=False,
        ))

        elapsed = time.time() - start
        logger.info(f"[跨语言克隆] 完成，耗时 {elapsed:.2f}s")
        return np.array(result.audio)

    def infer_instruct(
        self,
        text: str,
        instruct_text: str,
        ref_audio_path: str,
        speed: float = 1.0,
    ) -> np.ndarray:
        """
        自然语言指令控制合成。

        Args:
            text: 合成文本
            instruct_text: 自然语言指令（如"用开心的语气说"）
            ref_audio_path: 参考音频路径
            speed: 语速 (0.5~2.0)

        Returns:
            numpy 音频数组
        """
        text = normalize_text(text)
        ref_audio = self._load_ref_audio(ref_audio_path)

        logger.info(f"[指令合成] 指令: {instruct_text} | 文本: {text[:50]}...")
        start = time.time()

        result = next(self.model.generate(
            text=text,
            ref_audio=ref_audio,
            instruct_text=instruct_text,
            speed=speed,
            stt_model=None,
            verbose=False,
        ))

        elapsed = time.time() - start
        logger.info(f"[指令合成] 完成，耗时 {elapsed:.2f}s")
        return np.array(result.audio)

    def synthesize_to_file(
        self,
        mode: str,
        text: str,
        ref_audio_path: str = "",
        ref_text: str = "",
        instruct_text: str = "",
        speed: float = 1.0,
        filename: Optional[str] = None,
    ) -> str:
        """
        合成并保存到文件。

        Args:
            mode: 推理模式 ("zero_shot" | "cross_lingual" | "instruct")
            text: 合成文本
            ref_audio_path: 参考音频路径
            ref_text: 参考文本
            instruct_text: 指令文本
            speed: 语速 (0.5~2.0)
            filename: 输出文件名

        Returns:
            保存的音频文件路径
        """
        segments = split_long_text(text)
        if not segments:
            raise ValueError("合成文本为空")

        logger.info(f"[合成] 共 {len(segments)} 段, 模式={mode}")

        all_audio: List[np.ndarray] = []
        for idx, seg in enumerate(segments):
            logger.info(f"  段 {idx + 1}/{len(segments)}: {seg[:40]}...")

            if mode == "zero_shot":
                audio = self.infer_zero_shot(seg, ref_audio_path, ref_text, speed)
            elif mode == "cross_lingual":
                audio = self.infer_cross_lingual(seg, ref_audio_path, speed)
            elif mode == "instruct":
                audio = self.infer_instruct(seg, instruct_text, ref_audio_path, speed)
            else:
                raise ValueError(f"未知的推理模式: {mode}")

            all_audio.append(audio)

        # 拼接所有分段
        final_audio = np.concatenate(all_audio)

        # 保存文件
        if not filename:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{mode}_{ts}.wav"
        output_path = str(OUTPUT_DIR / filename)
        sf.write(output_path, final_audio, self.sample_rate)

        logger.info(f"[合成完成] 输出: {output_path}")
        return output_path

    def get_device_info(self) -> dict:
        """获取当前模型状态信息"""
        return {
            "framework": "MLX",
            "sample_rate": self.sample_rate,
            "model_loaded": True,
        }
