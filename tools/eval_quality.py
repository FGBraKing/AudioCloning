#!/usr/bin/env python3
"""客观质量指标 — 合成音频与参考音频的频谱距离（简化 MCD 风格）。

用于监测 TTS 合成质量随后端/版本的变化。值越小越好。

用法::

    uv run python -m tools.eval_quality --ref asset/zero_shot_prompt.wav --synth outputs/test_zero_shot.wav
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import soundfile as sf

from audioclone.utils.audio import resample_audio, sanitize_audio

logger = logging.getLogger("eval_quality")


def _spectral_frame_log_dist(ref: np.ndarray, syn: np.ndarray, n_fft: int = 512) -> float:
    """计算两段等长信号的逐帧频谱对数距离平均（简化 MCD 风格）。"""
    frames = max(1, min(len(ref), len(syn)) // n_fft)
    dists = []
    for i in range(frames):
        start = i * n_fft
        r = np.fft.rfft(ref[start : start + n_fft])
        s = np.fft.rfft(syn[start : start + n_fft])
        mag_r = np.maximum(np.abs(r), 1e-8)
        mag_s = np.maximum(np.abs(s), 1e-8)
        dists.append(np.mean(np.abs(20 * np.log10(mag_r) - 20 * np.log10(mag_s))))
    return float(np.mean(dists)) if dists else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="频谱距离客观指标")
    parser.add_argument("--ref", required=True, help="参考音频路径")
    parser.add_argument("--synth", required=True, help="待评估合成音频路径")
    parser.add_argument("--sr", type=int, default=24000, help="统一采样率")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ref, sr_r = sf.read(args.ref, dtype="float32")
    syn, sr_s = sf.read(args.synth, dtype="float32")
    if ref.ndim > 1:
        ref = ref[:, 0]
    if syn.ndim > 1:
        syn = syn[:, 0]

    ref = sanitize_audio(resample_audio(ref, int(sr_r), args.sr))
    syn = sanitize_audio(resample_audio(syn, int(sr_s), args.sr))

    # 对齐长度（取较短者）
    n = min(len(ref), len(syn))
    dist = _spectral_frame_log_dist(ref[:n], syn[:n])

    logger.info("频谱距离（越低越好）: %.3f dB", dist)
    if dist < 15:
        logger.info("判定: 频谱接近（质量良好）")
    elif dist < 30:
        logger.info("判定: 频谱中等（注意内容/时长是否对齐）")
    else:
        logger.info("判定: 频谱差异大（可能内容/语速不一致）")


if __name__ == "__main__":
    main()
