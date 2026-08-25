"""音频工具模块 — 验证、数值安全、重采样、保存、拼接。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from audioclone.config import REF_AUDIO_MAX_SEC, REF_AUDIO_MIN_SEC, REF_AUDIO_MIN_SR

logger = logging.getLogger(__name__)


def to_mono(data: np.ndarray, mode: str = "first") -> np.ndarray:
    """多声道转单声道：``mode='first'`` 取第一声道（默认）；``mode='mean'`` 取声道均值。

    统一各后端/工具的声道策略（CosyVoice 与 Qwen3-ASR 均走此助手）；单声道原样返回。
    """
    if data.ndim <= 1:
        return data
    if mode == "mean":
        return data.mean(axis=1)
    return data[:, 0]


def validate_ref_audio(path: str) -> tuple[bool, str]:
    """校验参考音频是否合法，返回 ``(is_valid, message)``。

    非抛异常的兼容接口；新代码建议使用 :func:`assert_valid_ref_audio`。
    """
    p = Path(path)
    if not p.exists():
        return False, f"文件不存在: {path}"

    try:
        info = sf.info(str(p))
    except Exception as e:  # noqa: BLE001 - 对文件格式错误统一处理
        return False, f"无法读取音频文件: {e}"

    duration = info.duration
    sr = info.samplerate

    if duration < REF_AUDIO_MIN_SEC:
        return False, f"音频时长过短 ({duration:.1f}s < {REF_AUDIO_MIN_SEC}s)"
    if duration > REF_AUDIO_MAX_SEC:
        return False, f"音频时长过长 ({duration:.1f}s > {REF_AUDIO_MAX_SEC}s)"
    if sr < REF_AUDIO_MIN_SR:
        return False, f"采样率过低 ({sr}Hz < {REF_AUDIO_MIN_SR}Hz)"

    return True, f"音频合法: {duration:.1f}s, {sr}Hz"


def assert_valid_ref_audio(path: str) -> None:
    """校验参考音频，不合法时抛 ``InvalidAudioInput``（推荐用于上层业务）。"""
    from audioclone.core.exceptions import InvalidAudioInput  # 延迟导入避免循环

    ok, msg = validate_ref_audio(path)
    if not ok:
        raise InvalidAudioInput(msg)


def assert_audio_duration(path: str, max_sec: float, min_sec: float = 1.0) -> None:
    """校验音频时长在 ``[min_sec, max_sec]``，不合法抛 ``InvalidAudioInput``。

    供时长口径不同于 TTS 参考音频（1~30s）的任务（如 A2A / 分类 / VAD，上限可取
    ``ASR_AUDIO_MAX_SEC``）复用。
    """
    from audioclone.core.exceptions import InvalidAudioInput  # 延迟导入避免循环

    duration = get_audio_duration(path)
    if duration <= 0:
        raise InvalidAudioInput("无法读取音频文件")
    if duration < min_sec:
        raise InvalidAudioInput(f"音频时长过短 ({duration:.1f}s < {min_sec:.0f}s)")
    if duration > max_sec:
        raise InvalidAudioInput(f"音频时长过长 ({duration:.1f}s > {max_sec:.0f}s)")


def get_audio_duration(path: str) -> float:
    """获取音频文件时长（秒）。"""
    try:
        info = sf.info(str(path))
        return info.duration
    except Exception:  # noqa: BLE001
        return 0.0


def resample_audio(data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """将单声道 float32 音频重采样到目标采样率。

    优先使用 soxr（更快、因果、低振铃）；不可用时回退到
    ``scipy.signal.resample_poly``（polyphase，因果），避免 FFT 法的非因果振铃/预回声。
    """
    if src_sr == dst_sr or len(data) == 0:
        return np.asarray(data, dtype=np.float32)

    try:  # soxr 可选，速度快且质量高
        import soxr

        return soxr.resample(data, src_sr, dst_sr).astype(np.float32)
    except ImportError:  # pragma: no cover - 取决于运行环境
        pass

    from scipy.signal import resample_poly

    up, down = dst_sr, src_sr
    g = _gcd(up, down)
    up //= g
    down //= g
    return resample_poly(np.asarray(data, dtype=np.float32), up, down).astype(np.float32)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a or 1


def sanitize_audio(
    audio: np.ndarray,
    peak_scale: float = 0.95,
    max_peak: float = 0.99,
) -> np.ndarray:
    """音频数值安全处理：去 NaN/Inf → 移除 DC → 限幅 → 峰值归一化。

    Args:
        audio: float32 音频（单声道）
        peak_scale: 归一化目标峰值
        max_peak: 未归一化时允许的最大峰值（超出做软限幅）

    Returns:
        清理后的 ``np.float32`` 音频数组
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    # 1) 处理 NaN / Inf
    finite = np.isfinite(x)
    if not np.all(finite):
        x = np.where(finite, x, 0.0).astype(np.float32)

    # 2) 移除 DC 偏移（仅对非恒定信号）
    mean = float(np.mean(x))
    if abs(mean) > 1e-5 and np.ptp(x) > 1e-5:
        x = (x - mean).astype(np.float32)

    # 3) 削波限幅
    x = np.clip(x, -1.0, 1.0)

    # 4) 峰值归一化
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-5:
        target = min(peak_scale, max_peak)
        x = (x * (target / peak)).astype(np.float32)

    return x


def loudness_normalize(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -16.0,
) -> np.ndarray:
    """按 EBU R128 响度（LUFS）归一化；pyloudnorm 不可用时回退到峰值归一化。

    先做数值安全，再做响度匹配，最后限幅防削波。
    """
    x = sanitize_audio(audio)
    if x.size == 0:
        return x

    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(x)
        if not np.isfinite(loudness):
            return x
        x = pyln.normalize.loudness(x, loudness, target_lufs)
        x = np.clip(x, -1.0, 1.0).astype(np.float32)
    except Exception:  # noqa: BLE001 - pyloudnorm 不可用/失败时保持峰值归一化结果
        logger.debug("loudness_normalize 回退到峰值归一化")
    return x


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    """保存音频数组到文件（落盘前自动做数值安全处理）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    safe = sanitize_audio(audio)
    sf.write(path, safe, sample_rate)


def concat_segments(segments: list[np.ndarray]) -> np.ndarray:
    """拼接多个音频段。"""
    if not segments:
        return np.array([], dtype=np.float32)
    return np.concatenate(segments)


def prune_outputs(directory: str | Path, keep: int) -> None:
    """输出目录生命周期管理：保留最近 ``keep`` 个文件，删除更旧的（``keep<=0`` 不清理）。

    递归统计（含 ``segments/`` 子目录的分段落盘），并清理清空后的空目录。
    供服务层落盘后调用，避免 ``outputs`` 长期运行无限增长。
    """
    if keep <= 0:
        return
    d = Path(directory)
    if not d.exists():
        return
    files = sorted(
        (p for p in d.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in files[keep:]:
        try:
            stale.unlink()
        except OSError:
            logger.warning("清理输出失败: %s", stale)

    # 清理因删除文件而空出的子目录（从最深开始）
    for sub in sorted(d.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if sub.is_dir() and not any(sub.iterdir()):
                sub.rmdir()
        except OSError:
            logger.warning("清理空目录失败: %s", sub)