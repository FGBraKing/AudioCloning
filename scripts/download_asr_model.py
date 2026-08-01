#!/usr/bin/env python3
"""
ASR 模型下载脚本 — 从 ModelScope 下载 Qwen3-ASR-0.6B-GGUF 模型
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_asr_model")

MODEL_ID = "ggml-org/Qwen3-ASR-0.6B-GGUF"

# 项目根目录（scripts/ 的上级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_DIR = str(PROJECT_ROOT / "pretrained_models" / "Qwen3-ASR-0.6B-GGUF")


def download_model(local_dir: str):
    """从 ModelScope 下载 Qwen3-ASR GGUF 模型"""
    from modelscope import snapshot_download

    logger.info(f"正在从 ModelScope 下载 {MODEL_ID} ...")
    logger.info(f"保存目录: {local_dir}")

    snapshot_download(
        MODEL_ID,
        local_dir=local_dir,
    )
    logger.info(f"下载完成: {local_dir}")

    # 列出下载的文件
    model_dir = Path(local_dir)
    gguf_files = list(model_dir.glob("*.gguf"))
    if gguf_files:
        logger.info("GGUF 模型文件:")
        for f in gguf_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name} ({size_mb:.1f} MB)")
    else:
        logger.warning("未找到 .gguf 文件，请检查下载是否完整")


def main():
    parser = argparse.ArgumentParser(description="下载 Qwen3-ASR-0.6B-GGUF 模型")
    parser.add_argument(
        "--local_dir",
        type=str,
        default=DEFAULT_LOCAL_DIR,
        help="本地保存目录",
    )
    args = parser.parse_args()

    # 确保目录存在
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    download_model(args.local_dir)


if __name__ == "__main__":
    main()
