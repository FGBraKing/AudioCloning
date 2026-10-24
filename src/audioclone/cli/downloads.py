"""模型下载命令 (命令行: ``uv run audioclone-download-tts`` / ``audioclone-download-asr``)。

下载逻辑统一在 ``core/model_manager.py``（与 REST API ``POST /api/models/download`` 共用）：
默认来源 **auto** —— 先 HuggingFace 进系统缓存（``~/.cache/huggingface/hub``，与默认加载
路径一致，支持 ``AUDIOCLONE_HF_ENDPOINT`` / ``HF_ENDPOINT`` 自定义镜像），失败自动回退
ModelScope（落 ``pretrained_models/<模型名>`` 或 ``--local_dir``）。本模块只做参数解析与产物清点。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from audioclone.core.model_manager import DOWNLOAD_SOURCES, download_model

logger = logging.getLogger(__name__)


def _run_download(prog: str, task: str) -> None:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument(
        "--local_dir", type=str, default=None,
        help="自定义下载目录（默认：HF 进系统缓存 / ModelScope 落 pretrained_models/<模型名>）",
    )
    parser.add_argument(
        "--backend", type=str, default=None, help="后端名称（默认取当前配置的默认后端）"
    )
    parser.add_argument(
        "--source", type=str, default="auto", choices=DOWNLOAD_SOURCES,
        help="下载来源：auto=HF 优先失败回退 ModelScope（默认）；hf / modelscope 强制指定",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = download_model(task, args.backend, args.local_dir, args.source)
    if result["status"] == "skipped" or not result["path"]:
        return

    if task == "asr":  # 清点 GGUF 产物
        gguf_files = list(Path(result["path"]).glob("*.gguf"))
        if gguf_files:
            logger.info("GGUF 模型文件:")
            for f in gguf_files:
                logger.info("  - %s (%.1f MB)", f.name, f.stat().st_size / (1024 * 1024))
        else:
            logger.warning("未找到 .gguf 文件，请检查下载是否完整")


def download_tts_main() -> None:
    """``audioclone-download-tts`` 入口。"""
    _run_download("audioclone-download-tts", "tts")


def download_asr_main() -> None:
    """``audioclone-download-asr`` 入口。"""
    _run_download("audioclone-download-asr", "asr")
