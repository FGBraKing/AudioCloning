"""ASR 端到端集成测试 — llama-server + Qwen3-ASR (需要本地模型与 llama.cpp)。"""

from __future__ import annotations

import logging

import pytest

from audioclone.config import ASSET_DIR
from audioclone.core import ASRService, create_asr_backend

logger = logging.getLogger(__name__)

TEST_AUDIO = ASSET_DIR / "zero_shot_prompt.wav"


@pytest.mark.integration
def test_asr_roundtrip(asr_ready: bool):
    if not asr_ready:
        pytest.skip("ASR 模型文件不完整，请先运行: uv run audioclone-download-asr")
    if not TEST_AUDIO.exists():
        pytest.skip(f"测试音频不存在: {TEST_AUDIO}")

    with ASRService(create_asr_backend()) as engine:
        text = engine.transcribe(str(TEST_AUDIO))
    assert isinstance(text, str) and len(text) > 0, "ASR 识别结果为空"