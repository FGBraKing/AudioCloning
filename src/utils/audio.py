"""
音频工具模块 — 验证、保存、拼接
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import REF_AUDIO_MIN_SEC, REF_AUDIO_MAX_SEC

logger = logging.getLogger(__name__)


def validate_ref_audio(path: str) -> tuple:
    """
    验证参考音频文件是否合法。

    Returns:
        (is_valid, message)
    """
    p = Path(path)
    if not p.exists():
        return False, f"文件不存在: {path}"

    try:
        info = sf.info(str(p))
    except Exception as e:
        return False, f"无法读取音频文件: {e}"

    duration = info.duration
    sr = info.samplerate

    if duration < REF_AUDIO_MIN_SEC:
        return False, f"音频时长过短 ({duration:.1f}s < {REF_AUDIO_MIN_SEC}s)"
    if duration > REF_AUDIO_MAX_SEC:
        return False, f"音频时长过长 ({duration:.1f}s > {REF_AUDIO_MAX_SEC}s)"
    if sr < 8000:
        return False, f"采样率过低 ({sr}Hz < 8000Hz)"

    return True, f"音频合法: {duration:.1f}s, {sr}Hz"


def get_audio_duration(path: str) -> float:
    """获取音频文件时长（秒）"""
    try:
        info = sf.info(str(path))
        return info.duration
    except Exception:
        return 0.0


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 24000):
    """保存音频数组到文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def concat_segments(segments: list) -> np.ndarray:
    """拼接多个音频段"""
    if not segments:
        return np.array([], dtype=np.float32)
    return np.concatenate(segments)
