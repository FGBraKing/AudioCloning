"""WebUI 媒体文件管理 — 上传落盘 / 合成结果落盘 / 临时目录生命周期。

上传文件与录音存放于进程级临时目录（退出时清理），合成结果按既有约定落盘到
``outputs/`` 目录，便于用户在磁盘上复用。
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

from audioclone.config import OUTPUT_DIR, OUTPUT_KEEP_MAX
from audioclone.utils.audio import prune_outputs, save_audio

logger = logging.getLogger(__name__)

# 允许上传落盘的音频扩展名（与 ASR 支持格式保持一致）
UPLOAD_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".mp4", ".aac"}

_tmp_root: Path | None = None


def upload_root() -> Path:
    """惰性创建并返回上传临时目录（进程级单例）。"""
    global _tmp_root  # noqa: PLW0603 - 进程级单例，与 cleanup 配对
    if _tmp_root is None or not _tmp_root.exists():
        _tmp_root = Path(tempfile.mkdtemp(prefix="audioclone-webui-"))
    return _tmp_root


def cleanup() -> None:
    """清理上传临时目录（注册于应用关闭钩子）。"""
    global _tmp_root  # noqa: PLW0603
    if _tmp_root is not None:
        shutil.rmtree(_tmp_root, ignore_errors=True)
        _tmp_root = None


def is_upload_path(path: str) -> bool:
    """判断路径是否位于上传临时目录内（防止前端回传任意路径）。"""
    try:
        return Path(path).is_relative_to(upload_root())
    except (TypeError, ValueError):
        return False


def save_upload(filename: str, data: bytes) -> str:
    """保存上传/录音字节到临时目录，返回文件绝对路径。

    扩展名缺失时补 ``.wav``，非法字符一律替换为 ``_``，避免路径穿越与读取失败。
    """
    root = upload_root()
    root.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename or "").suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        suffix = ".wav"
    stem = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    target = root / f"{stem}{suffix}"
    target.write_bytes(data)
    logger.debug("[WebUI] 已保存上传文件: %s (%d bytes)", target, len(data))
    return str(target)


def save_result(audio: np.ndarray, sample_rate: int, prefix: str = "result") -> Path:
    """保存合成/转换结果到 ``outputs/``，返回落盘路径。

    文件名带时间戳与随机后缀，避免并发覆盖与浏览器缓存命中旧音频。
    """
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}.wav"
    save_audio(audio, str(path), sample_rate)
    prune_outputs(output_dir, OUTPUT_KEEP_MAX)
    return path


__all__ = [
    "cleanup",
    "is_upload_path",
    "save_result",
    "save_upload",
    "upload_root",
]
