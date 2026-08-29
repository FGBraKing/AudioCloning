"""TTS 端到端集成测试 — 3 种推理模式 (需要本地模型)。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import soundfile as sf

from audioclone.config import ASSET_DIR, SAMPLE_RATE
from audioclone.core import TTSService, create_tts_backend

logger = logging.getLogger(__name__)

# 会话级服务，避免重复加载模型
_engine: TTSService | None = None
_instruct_engine: TTSService | None = None


@pytest.fixture(scope="module")
def engine(tts_model_dir: str) -> TTSService:
    global _engine
    if _engine is None:
        backend = create_tts_backend(model_dir=tts_model_dir)
        _engine = TTSService(backend)
    return _engine


@pytest.fixture(scope="module")
def instruct_engine(tts_instruct_dir: str) -> TTSService:
    """CustomVoice 后端（支持 instruct 模式），避免与 base 后端交叉加载。"""
    global _instruct_engine
    if _instruct_engine is None:
        backend = create_tts_backend(
            name="qwen3-tts-customvoice-mlx", model_dir=tts_instruct_dir
        )
        _instruct_engine = TTSService(backend)
    return _instruct_engine


def _ref_audio(name: str) -> str:
    path = ASSET_DIR / name
    if not path.exists():
        pytest.skip(f"参考音频不存在: {path}")
    return str(path)


def _assert_valid_audio(output_path: str) -> None:
    path = Path(output_path)
    assert path.exists(), "输出文件不存在"
    assert path.stat().st_size > 0, "输出文件为空"
    info = sf.info(output_path)
    assert info.duration > 0, "音频时长为 0"
    assert info.samplerate == SAMPLE_RATE


@pytest.mark.integration
def test_zero_shot_clone(engine: TTSService):
    out = engine.synthesize_to_file(
        mode="zero_shot",
        text="你好，欢迎使用个人声音克隆系统。这是一段测试语音。",
        ref_audio_path=_ref_audio("zero_shot_prompt.wav"),
        ref_text="希望你以后能够做的比我还好呦。",
        filename="test_zero_shot.wav",
    )
    _assert_valid_audio(out)


@pytest.mark.integration
def test_cross_lingual(engine: TTSService):
    out = engine.synthesize_to_file(
        mode="cross_lingual",
        text="Hello, this is a cross-lingual voice cloning test.",
        ref_audio_path=_ref_audio("cross_lingual_prompt.wav"),
        filename="test_cross_lingual.wav",
    )
    _assert_valid_audio(out)


@pytest.mark.integration
def test_instruct(instruct_engine: TTSService):
    out = instruct_engine.synthesize_to_file(
        mode="instruct",
        text="今天天气真好，适合出去走走。",
        instruct_text="请用开心的语气说",
        filename="test_instruct.wav",
    )
    _assert_valid_audio(out)