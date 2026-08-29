#!/usr/bin/env python3
"""性能基准脚本 — 测量 TTS 后端冷启动、单次合成耗时。

用法::

    uv run python -m tools.benchmark --backend qwen3-tts-base-mlx --iterations 3

无真实模型时可用 ``--backend dummy-omni`` 快速跑通框架链路。
"""

from __future__ import annotations

import argparse
import logging
import time

from audioclone.core import TTSService, create_tts_backend

logger = logging.getLogger("benchmark")

TEXT = "你好，欢迎使用音频模型体验工具箱。这是一段用于性能基准测试的合成文本。"


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS 性能基准")
    parser.add_argument("--backend", type=str, default=None, help="TTS 后端名称（默认配置）")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    backend = create_tts_backend(args.backend)
    logger.info("后端: %s | 加载中...", backend.name)
    t0 = time.time()
    backend.load()
    load_sec = time.time() - t0
    logger.info("冷启动（模型加载）耗时: %.2fs", load_sec)

    svc = TTSService(backend)
    timings = []
    for i in range(args.iterations):
        t1 = time.time()
        svc.synthesize_to_file(
            mode="zero_shot",
            text=TEXT,
            ref_audio_path="asset/zero_shot_prompt.wav",
            ref_text="希望你以后能够做的比我还好呦。",
            filename=f"bench_{i}.wav",
        )
        elapsed = time.time() - t1
        timings.append(elapsed)
        logger.info("迭代 %d/%d 合成耗时: %.2fs", i + 1, args.iterations, elapsed)

    avg = sum(timings) / len(timings)
    logger.info("平均单次合成耗时: %.2fs（min %.2fs / max %.2fs）", avg, min(timings), max(timings))


if __name__ == "__main__":
    main()
