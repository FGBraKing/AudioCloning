"""共享测试配置。导入其他测试前先设置 no_proxy。"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")


@pytest.fixture(scope="session")
def tts_model_dir() -> str:
    """可用的 TTS 模型目录；未下载则跳过集成测试。"""
    from audioclone.config import MODEL_DIR

    if not MODEL_DIR.exists():
        pytest.skip(f"TTS 模型目录不存在，请先运行: uv run audioclone-download-tts ({MODEL_DIR})")
    return str(MODEL_DIR)


@pytest.fixture(scope="session")
def asr_ready() -> bool:
    """ASR 后端是否具备模型文件。"""
    from audioclone.core import ASRService, create_asr_backend

    return ASRService(create_asr_backend()).is_model_available