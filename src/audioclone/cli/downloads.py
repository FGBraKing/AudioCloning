"""模型下载命令 (命令行: ``uv run audioclone-download-tts`` / ``audioclone-download-asr``)。

依据当前配置的默认后端，从 ModelScope 下载对应模型。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from audioclone.config import (
    ASR_BACKEND,
    ASR_BACKENDS,
    ASR_MODEL_DIR,
    TTS_BACKEND,
    TTS_BACKENDS,
)

logger = logging.getLogger(__name__)


def _backend_model_id(backends: dict, name: str) -> str | None:
    """从后端配置中读取 model_id；无则该后端无需下载（如 dummy）。"""
    spec = backends.get(name, {})
    return spec.get("model_id")


def download_modelscope(model_id: str, local_dir: Path) -> None:
    """从 ModelScope 下载模型快照到本地目录。"""
    from modelscope import snapshot_download

    logger.info("正在从 ModelScope 下载 %s ...", model_id)
    logger.info("保存目录: %s", local_dir)
    snapshot_download(model_id, local_dir=str(local_dir))
    logger.info("下载完成: %s", local_dir)


def download_tts(local_dir: str | None = None, backend: str | None = None) -> None:
    """下载指定（或默认）TTS 后端所需的模型。"""
    name = backend or TTS_BACKEND
    if name not in TTS_BACKENDS:
        raise KeyError(f"未知的 TTS 后端: {name}（可用: {', '.join(TTS_BACKENDS)}）")
    model_id = _backend_model_id(TTS_BACKENDS, name)
    if not model_id:
        logger.info("TTS 后端 (%s) 无需下载模型", name)
        return

    actual = Path(local_dir) if local_dir else _tts_default_dir(name)
    actual.mkdir(parents=True, exist_ok=True)
    download_modelscope(model_id, actual)


def download_asr(local_dir: str | None = None, backend: str | None = None) -> None:
    """下载指定（或默认）ASR 后端所需的模型，并列出下载到的文件。"""
    name = backend or ASR_BACKEND
    if name not in ASR_BACKENDS:
        raise KeyError(f"未知的 ASR 后端: {name}（可用: {', '.join(ASR_BACKENDS)}）")
    model_id = _backend_model_id(ASR_BACKENDS, name)
    if not model_id:
        logger.info("ASR 后端 (%s) 无需下载模型", name)
        return

    target = Path(local_dir) if local_dir else ASR_MODEL_DIR
    target.mkdir(parents=True, exist_ok=True)
    download_modelscope(model_id, target)

    gguf_files = list(target.glob("*.gguf"))
    if gguf_files:
        logger.info("GGUF 模型文件:")
        for f in gguf_files:
            logger.info("  - %s (%.1f MB)", f.name, f.stat().st_size / (1024 * 1024))
    else:
        logger.warning("未找到 .gguf 文件，请检查下载是否完整")


def _tts_default_dir(backend: str | None = None) -> Path:
    """从 TTS 后端配置取模型目录（默认取全局默认后端）。"""
    from audioclone.config import QWEN3_TTS_MODEL_DIR

    entry = TTS_BACKENDS.get(backend or TTS_BACKEND, {})
    model_dir = entry.get("model_dir")
    return Path(model_dir) if isinstance(model_dir, str) else QWEN3_TTS_MODEL_DIR


def _run_download(name: str, func, default_dir_for_backend) -> None:
    """下载命令入口。

    ``--local_dir`` 未显式指定时，目录必须**跟随 ``--backend``**：不同后端的模型
    绝不能落到同一目录（否则同名权重文件互相覆盖，得到一个"混合体"模型）。
    ``default_dir_for_backend`` 为 ``backend 名 -> 默认目录`` 的回调，在解析出
    backend 之后才求值。
    """
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument(
        "--local_dir", type=str, default=None, help="本地保存目录（默认随 --backend 而定）"
    )
    parser.add_argument(
        "--backend", type=str, default=None, help="后端名称（默认取当前配置的默认后端）"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    local_dir = args.local_dir or str(default_dir_for_backend(args.backend))
    func(local_dir, args.backend)


def download_tts_main() -> None:
    """``audioclone-download-tts`` 入口。"""
    _run_download("audioclone-download-tts", download_tts, _tts_default_dir)


def download_asr_main() -> None:
    """``audioclone-download-asr`` 入口。"""
    _run_download(
        "audioclone-download-asr",
        download_asr,
        lambda backend: ASR_MODEL_DIR,
    )