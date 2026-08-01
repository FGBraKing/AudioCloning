#!/usr/bin/env python3
"""
AudioCloning 端到端测试 — MLX 版
测试 3 种推理模式: zero_shot / cross_lingual / instruct
"""

import sys
import os
import time
import logging

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_inference")

import numpy as np
import soundfile as sf

from src.config import MODEL_DIR, ASSET_DIR, OUTPUT_DIR, SAMPLE_RATE
from src.tts import load_model, TTSEngine


def check_audio_file(path: str, label: str) -> bool:
    """验证输出音频文件"""
    if not os.path.exists(path):
        logger.error(f"[{label}] 输出文件不存在: {path}")
        return False

    size = os.path.getsize(path)
    if size == 0:
        logger.error(f"[{label}] 输出文件为空")
        return False

    info = sf.info(path)
    duration = info.duration
    if duration <= 0:
        logger.error(f"[{label}] 音频时长为 0")
        return False

    logger.info(f"[{label}] 输出: {path} | 大小: {size} bytes | 时长: {duration:.2f}s | 采样率: {info.samplerate}Hz")
    return True


def main():
    logger.info("=" * 60)
    logger.info("  AudioCloning 端到端测试 (MLX)")
    logger.info("=" * 60)

    # 检查模型目录
    if not MODEL_DIR.exists():
        logger.error(f"模型目录不存在: {MODEL_DIR}")
        logger.error("请先运行: python scripts/download_tts_model.py")
        sys.exit(1)

    # 参考音频
    ref_audio_zs = str(ASSET_DIR / "zero_shot_prompt.wav")
    ref_audio_cl = str(ASSET_DIR / "cross_lingual_prompt.wav")

    for f in [ref_audio_zs, ref_audio_cl]:
        if not os.path.exists(f):
            logger.error(f"参考音频不存在: {f}")
            sys.exit(1)

    # 加载模型
    logger.info("正在加载模型...")
    start = time.time()
    mlx_model = load_model()
    engine = TTSEngine(mlx_model)
    logger.info(f"模型加载完成，耗时 {time.time() - start:.2f}s")
    logger.info(f"采样率: {engine.sample_rate}Hz")

    results = {}
    total_start = time.time()

    # ─── 测试 1: 零样本克隆 ───────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("测试 1/3: 零样本克隆 (zero_shot)")
    logger.info("─" * 50)
    try:
        t0 = time.time()
        output_path = engine.synthesize_to_file(
            mode="zero_shot",
            text="你好，欢迎使用个人声音克隆系统。这是一段测试语音。",
            ref_audio_path=ref_audio_zs,
            ref_text="希望你以后能够做的比我还好呦。",
            filename="test_zero_shot.wav",
        )
        elapsed = time.time() - t0
        ok = check_audio_file(output_path, "zero_shot")
        results["zero_shot"] = {"ok": ok, "time": elapsed, "path": output_path}
        logger.info(f"[zero_shot] 推理耗时: {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"[zero_shot] 失败: {e}", exc_info=True)
        results["zero_shot"] = {"ok": False, "error": str(e)}

    # ─── 测试 2: 跨语言克隆 ───────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("测试 2/3: 跨语言克隆 (cross_lingual)")
    logger.info("─" * 50)
    try:
        t0 = time.time()
        output_path = engine.synthesize_to_file(
            mode="cross_lingual",
            text="Hello, this is a cross-lingual voice cloning test.",
            ref_audio_path=ref_audio_cl,
            filename="test_cross_lingual.wav",
        )
        elapsed = time.time() - t0
        ok = check_audio_file(output_path, "cross_lingual")
        results["cross_lingual"] = {"ok": ok, "time": elapsed, "path": output_path}
        logger.info(f"[cross_lingual] 推理耗时: {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"[cross_lingual] 失败: {e}", exc_info=True)
        results["cross_lingual"] = {"ok": False, "error": str(e)}

    # ─── 测试 3: 指令控制合成 ─────────────────────────────────
    logger.info("\n" + "─" * 50)
    logger.info("测试 3/3: 指令控制合成 (instruct)")
    logger.info("─" * 50)
    try:
        t0 = time.time()
        output_path = engine.synthesize_to_file(
            mode="instruct",
            text="今天天气真好，适合出去走走。",
            ref_audio_path=ref_audio_zs,
            instruct_text="请用开心的语气说",
            filename="test_instruct.wav",
        )
        elapsed = time.time() - t0
        ok = check_audio_file(output_path, "instruct")
        results["instruct"] = {"ok": ok, "time": elapsed, "path": output_path}
        logger.info(f"[instruct] 推理耗时: {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"[instruct] 失败: {e}", exc_info=True)
        results["instruct"] = {"ok": False, "error": str(e)}

    # ─── 汇总 ─────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("  测试结果汇总")
    logger.info("=" * 60)

    all_pass = True
    for mode, res in results.items():
        status = "PASS" if res.get("ok") else "FAIL"
        if not res.get("ok"):
            all_pass = False
        time_str = f"{res['time']:.2f}s" if "time" in res else res.get("error", "N/A")
        logger.info(f"  {mode:15s} | {status} | {time_str}")

    logger.info(f"\n  总耗时: {total_elapsed:.2f}s")
    logger.info("=" * 60)

    if all_pass:
        logger.info("  所有测试通过!")
    else:
        logger.error("  存在失败的测试!")
        sys.exit(1)


if __name__ == "__main__":
    main()
