"""共享测试配置。导入其他测试前先设置 no_proxy 并清理 SOCKS 代理变量。

``ALL_PROXY=socks5://...`` 会导致 gradio 在 import 时初始化 ``httpx.Client`` 失败
（缺 socksio）；测试不应受宿主机代理影响，故在此统一清除。
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)


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