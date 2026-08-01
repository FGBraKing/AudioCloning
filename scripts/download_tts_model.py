#!/usr/bin/env python3
"""
模型下载脚本 — 从 ModelScope 下载 MLX 格式 CosyVoice3 模型
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_model")

MODEL_ID = "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16"

# 项目根目录（scripts/ 的上级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_DIR = str(PROJECT_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B-2512-fp16")


def download_model(local_dir: str):
    """从 ModelScope 下载 MLX 格式模型"""
    from modelscope import snapshot_download

    logger.info(f"正在从 ModelScope 下载 {MODEL_ID} ...")
    logger.info(f"保存目录: {local_dir}")

    snapshot_download(
        MODEL_ID,
        local_dir=local_dir,
    )
    logger.info(f"下载完成: {local_dir}")


def main():
    parser = argparse.ArgumentParser(description="下载 CosyVoice3 MLX 模型")
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
