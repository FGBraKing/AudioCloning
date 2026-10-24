"""共享测试配置。导入其他测试前先设置 no_proxy 并清理 SOCKS 代理变量。

``ALL_PROXY=socks5://...`` 会导致 httpx 客户端在 import 阶段初始化失败（缺 socksio）；
测试不应受宿主机代理影响，故在此统一清除。
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
    """可用的 TTS 模型目录（本地 pretrained_models）；未下载则跳过集成测试。"""
    from audioclone.config import PROJECT_ROOT, QWEN3_TTS_MODEL_IDS

    model_dir = PROJECT_ROOT / "pretrained_models" / "qwen3_tts_base"
    if not model_dir.exists():
        pytest.skip(
            f"TTS 模型目录不存在，请先运行: "
            f"uv run audioclone-download-tts --backend qwen3-tts-base-mlx"
            f"（{QWEN3_TTS_MODEL_IDS['base']}）"
        )
    return str(model_dir)


@pytest.fixture(scope="session")
def tts_instruct_dir() -> str:
    """支持 instruct 模式的 CustomVoice 模型目录；未下载则跳过集成测试。"""
    from audioclone.config import PROJECT_ROOT, QWEN3_TTS_MODEL_IDS

    model_dir = PROJECT_ROOT / "pretrained_models" / "qwen3_tts_custom_voice"
    if not model_dir.exists():
        pytest.skip(
            f"CustomVoice 模型目录不存在，请先运行: "
            f"uv run audioclone-download-tts --backend qwen3-tts-customvoice-mlx"
            f"（{QWEN3_TTS_MODEL_IDS['custom_voice']}）"
        )
    return str(model_dir)


@pytest.fixture(scope="session")
def asr_ready() -> bool:
    """ASR 后端是否具备模型文件（sherpa-onnx；未安装/未下载则跳过集成测试）。"""
    from audioclone.core import ASRService, create_asr_backend

    backend = create_asr_backend("sherpa-asr-onnx")
    if not backend.is_model_available:
        pytest.skip(
            "ASR 模型未就位（需 uv sync --extra onnx 并下载 "
            "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17）"
        )
    return ASRService(backend).is_model_available