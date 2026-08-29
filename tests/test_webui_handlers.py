"""WebUI handler 单测 — 直接调用 handler（无界面运行时），覆盖成功与错误路径。"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audioclone.core import ASRService, TTSService
from audioclone.core.backends.dummy import DummyASRBackend, DummyOmnibackend, DummyTTSBackend
from audioclone.core.exceptions import AudioError
from audioclone.webui.errors import UIError
from audioclone.webui.handlers import (
    make_asr_handler,
    make_cross_lingual_handler,
    make_instruct_handler,
    make_multitask_handlers,
    make_zero_shot_handler,
    ui_error,
)


def _make_wav(path, sr: int = 16000, seconds: float = 1.0) -> str:
    data = (np.ones(int(sr * seconds), dtype=np.float32) * 0.1).astype(np.float32)
    sf.write(str(path), data, sr)
    return str(path)


# ─── @ui_error 装饰器 ────────────────────────────────────────
def test_ui_error_passthrough():
    @ui_error("t", "m")
    def fn():
        raise UIError("原始消息")

    with pytest.raises(UIError) as ei:
        fn()
    assert "原始消息" in str(ei.value)


def test_ui_error_wraps_domain_error():
    @ui_error("t", "合成失败")
    def fn():
        raise AudioError("领域错误消息")

    with pytest.raises(UIError) as ei:
        fn()
    # 领域异常语义化消息原样展示（无 "合成失败:" 前缀）
    msg = str(ei.value)
    assert "领域错误消息" in msg
    assert "合成失败" not in msg


def test_ui_error_wraps_generic_error():
    @ui_error("t", "合成失败")
    def fn():
        raise RuntimeError("boom")

    with pytest.raises(UIError) as ei:
        fn()
    assert "合成失败: boom" in str(ei.value)


# ─── 零样本克隆 handler ─────────────────────────────────────
def test_zero_shot_handler_success(tmp_path):
    engine = TTSService(DummyTTSBackend(sample_rate=24000))
    ref = _make_wav(tmp_path / "ref.wav")
    handler = make_zero_shot_handler(engine)
    sr, audio = handler("你好测试", ref, None, "参考文本", 1.0)
    assert sr == 24000
    assert len(audio) > 0


def test_zero_shot_handler_missing_audio(tmp_path):
    engine = TTSService(DummyTTSBackend(sample_rate=24000))
    handler = make_zero_shot_handler(engine)
    with pytest.raises(UIError):
        handler("你好", None, None, "", 1.0)


def test_zero_shot_handler_empty_text(tmp_path):
    engine = TTSService(DummyTTSBackend(sample_rate=24000))
    ref = _make_wav(tmp_path / "ref.wav")
    handler = make_zero_shot_handler(engine)
    with pytest.raises(UIError) as ei:
        handler("   ", ref, None, "", 1.0)
    assert "文本为空" in str(ei.value)


def test_zero_shot_handler_invalid_audio(tmp_path):
    engine = TTSService(DummyTTSBackend(sample_rate=24000))
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    handler = make_zero_shot_handler(engine)
    with pytest.raises(UIError):
        handler("你好", str(bad), None, "", 1.0)


# ─── 跨语言 / 指令 handler ──────────────────────────────────
def test_cross_lingual_handler_success(tmp_path):
    engine = TTSService(DummyTTSBackend(sample_rate=24000))
    ref = _make_wav(tmp_path / "ref.wav")
    handler = make_cross_lingual_handler(engine)
    sr, audio = handler("Hello world", ref, 1.0)
    assert sr == 24000 and len(audio) > 0


def test_instruct_handler_success(tmp_path):
    engine = TTSService(DummyTTSBackend(sample_rate=24000))
    ref = _make_wav(tmp_path / "ref.wav")
    handler = make_instruct_handler(engine)
    sr, audio = handler("今天天气真好", ref, "请用开心的语气说", 1.0)
    assert sr == 24000 and len(audio) > 0


def test_instruct_handler_missing_instruct(tmp_path):
    engine = TTSService(DummyTTSBackend(sample_rate=24000))
    ref = _make_wav(tmp_path / "ref.wav")
    handler = make_instruct_handler(engine)
    with pytest.raises(UIError) as ei:
        handler("你好", ref, "  ", 1.0)
    assert "指令" in str(ei.value)


# ─── ASR handler ────────────────────────────────────────────
def test_asr_handler_success(tmp_path):
    asr_engine = ASRService(DummyASRBackend())
    wav = _make_wav(tmp_path / "in.wav")
    handler = make_asr_handler(asr_engine)
    text, info = handler(wav, None)
    assert text == DummyASRBackend.RESULT
    assert "识别完成" in info


def test_asr_handler_bad_format(tmp_path):
    asr_engine = ASRService(DummyASRBackend())
    bad = tmp_path / "in.mp4"
    bad.write_bytes(b"x")
    handler = make_asr_handler(asr_engine)
    with pytest.raises(UIError) as ei:
        handler(str(bad), None)
    assert "不支持的音频格式" in str(ei.value)


# ─── 多任务实验台（dummy-omni 全能力） ──────────────────────
def test_multitask_omnibackend_all(tmp_path):
    handlers = make_multitask_handlers(DummyOmnibackend(sample_rate=24000))
    wav = _make_wav(tmp_path / "in.wav")

    sr, audio = handlers["t2a"]("雷雨声", 0.2)
    assert sr == 24000 and len(audio) > 0

    _, audio2 = handlers["a2a"](wav, "变男声")
    assert len(audio2) > 0

    labels = handlers["classify"](wav)
    assert "speech" in labels

    segments = handlers["vad"](wav)
    assert "0.00s" in segments


def test_multitask_guard_incapable_backend():
    handlers = make_multitask_handlers(DummyTTSBackend(sample_rate=24000))  # 仅 TTS
    with pytest.raises(UIError) as ei:
        handlers["t2a"]("雷雨声", 0.2)
    assert "不支持" in str(ei.value)
    with pytest.raises(UIError) as ei:
        handlers["classify"]("whatever.wav")
    assert "不支持" in str(ei.value)


def test_multitask_accepts_long_audio(tmp_path):
    """多任务时长口径独立于 TTS（>30s 但 <300s 应通过）。"""
    handlers = make_multitask_handlers(DummyOmnibackend(sample_rate=24000))
    long_wav = _make_wav(tmp_path / "long.wav", seconds=40)  # >30s，<300s
    _, audio = handlers["a2a"](long_wav, "变男声")
    assert len(audio) > 0
    labels = handlers["classify"](long_wav)
    assert "speech" in labels
