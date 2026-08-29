"""Qwen3-TTS (MLX) 后端测试 — 用假模型验证三个变体的模式映射与调用契约。

不需要真实权重：通过替换 ``_load_model`` 注入假模型对象，
断言「模式 → mlx_audio 生成入口」的分派是否正确。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audioclone.core import TTS_MODES, ModelSlot, TTSService, model_slots
from audioclone.core.backends.qwen3_tts_mlx import (
    Qwen3TTSBaseMLXBackend,
    Qwen3TTSCustomVoiceMLXBackend,
    Qwen3TTSMLXBackend,
    Qwen3TTSVoiceDesignMLXBackend,
    has_qwen3_tts_support,
)
from audioclone.core.capabilities import TextToSpeechCapability
from audioclone.core.exceptions import InvalidAudioInput, ModelLoadError
from audioclone.core.service.model_loader import create_tts_backend

SAMPLE_RATE = 24000


class _Result:
    """模拟 mlx_audio 的 GenerationResult。"""

    def __init__(self, n: int) -> None:
        self.audio = np.ones(n, dtype=np.float32)


class _FakeQwen3TTSModel:
    """记录调用的假 Qwen3-TTS 模型。"""

    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    # ─── 三个生成入口 ────────────────────────────────────────
    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        yield _Result(SAMPLE_RATE)

    def generate_custom_voice(self, **kwargs):
        self.calls.append(("generate_custom_voice", kwargs))
        yield _Result(SAMPLE_RATE // 2)

    def generate_voice_design(self, **kwargs):
        self.calls.append(("generate_voice_design", kwargs))
        yield _Result(SAMPLE_RATE // 2)

    def get_supported_speakers(self) -> list[str]:
        return ["Vivian", "Ryan"]

    @property
    def last(self) -> tuple[str, dict]:
        return self.calls[-1]


@pytest.fixture(autouse=True)
def _clean_slots():
    model_slots.release_all()
    yield
    model_slots.release_all()


@pytest.fixture(autouse=True)
def _assume_mlx_support(monkeypatch):
    """默认假装当前 mlx_audio 支持 qwen3_tts（真实依赖缺失的场景另有专门用例）。"""
    monkeypatch.setattr(
        "audioclone.core.backends.qwen3_tts_mlx.has_qwen3_tts_support", lambda: True
    )


def _attach(backend: Qwen3TTSMLXBackend, model: _FakeQwen3TTSModel) -> Qwen3TTSMLXBackend:
    """注入假模型：绕过真实的 mlx_audio 加载。"""
    backend._load_model = lambda loader: model  # type: ignore[method-assign]
    backend.load()
    return backend


# ─── 注册与身份 ──────────────────────────────────────────────
@pytest.mark.parametrize(
    ("name", "cls", "modes"),
    [
        ("qwen3-tts-base-mlx", Qwen3TTSBaseMLXBackend, ("zero_shot", "cross_lingual")),
        ("qwen3-tts-customvoice-mlx", Qwen3TTSCustomVoiceMLXBackend, ("instruct",)),
        ("qwen3-tts-voicedesign-mlx", Qwen3TTSVoiceDesignMLXBackend, ("instruct",)),
    ],
)
def test_backends_registered_with_mode_subsets(name, cls, modes):
    """三个后端均已注册，且各自只声明自己支持的模式。"""
    backend = create_tts_backend(name)
    assert isinstance(backend, cls)
    assert isinstance(backend, TextToSpeechCapability)
    assert backend.supported_modes == modes
    # 声明的模式必须是全局模式集合的子集（避免拼写漂移）
    assert set(backend.supported_modes) <= set(TTS_MODES)


def test_slot_is_tts_and_occupied_on_load():
    """Qwen3-TTS 后端占 TTS 槽位，保证不会与另一个 TTS 模型同时驻留。"""
    backend = _attach(Qwen3TTSBaseMLXBackend(), _FakeQwen3TTSModel())
    assert backend.slot is ModelSlot.TTS

    other = Qwen3TTSCustomVoiceMLXBackend()
    other._load_model = lambda loader: _FakeQwen3TTSModel()  # type: ignore[method-assign]
    from audioclone.core import ModelSlotError

    with pytest.raises(ModelSlotError):
        other.load()


# ─── Base：zero_shot / cross_lingual ─────────────────────────
def test_base_zero_shot_uses_icl_with_ref_text():
    """zero_shot：给出 ref_text → 走 ICL（上下文学习）克隆。"""
    model = _FakeQwen3TTSModel()
    backend = _attach(Qwen3TTSBaseMLXBackend(), model)

    audio = backend.synthesize("zero_shot", "你好世界", "ref.wav", ref_text="这是参考文本")

    entry, kwargs = model.last
    assert entry == "generate"
    assert kwargs["text"] == "你好世界"
    assert kwargs["ref_audio"] == "ref.wav"
    assert kwargs["ref_text"] == "这是参考文本"
    assert audio.dtype == np.float32


def test_base_cross_lingual_drops_ref_text():
    """cross_lingual：不依赖参考音频转录 → ref_text 置空（speaker-encoder 模式）。"""
    model = _FakeQwen3TTSModel()
    backend = _attach(Qwen3TTSBaseMLXBackend(), model)

    backend.synthesize("cross_lingual", "Hello", "ref.wav", ref_text="这段应被忽略")

    entry, kwargs = model.last
    assert entry == "generate"
    assert kwargs["ref_text"] is None  # 关键：不传转录文本


def test_base_rejects_instruct():
    model = _FakeQwen3TTSModel()
    backend = _attach(Qwen3TTSBaseMLXBackend(), model)

    with pytest.raises(InvalidAudioInput) as ei:
        backend.synthesize("instruct", "你好", "ref.wav", instruct_text="开心的语气")
    assert "不支持推理模式" in str(ei.value)


# ─── CustomVoice：预设音色 + 情绪指令 ────────────────────────
def test_custom_voice_uses_generate_custom_voice():
    model = _FakeQwen3TTSModel()
    backend = _attach(
        Qwen3TTSCustomVoiceMLXBackend(speaker="Ryan", language="English"), model
    )

    backend.synthesize("instruct", "I'm so excited!", "", instruct_text="Very happy.")

    entry, kwargs = model.last
    assert entry == "generate_custom_voice"
    assert kwargs["speaker"] == "Ryan"
    assert kwargs["language"] == "English"
    assert kwargs["instruct"] == "Very happy."


def test_custom_voice_rejects_clone_modes():
    model = _FakeQwen3TTSModel()
    backend = _attach(Qwen3TTSCustomVoiceMLXBackend(), model)

    for mode in ("zero_shot", "cross_lingual"):
        with pytest.raises(InvalidAudioInput):
            backend.synthesize(mode, "你好", "ref.wav", ref_text="x")


def test_custom_voice_exposes_speakers():
    backend = _attach(Qwen3TTSCustomVoiceMLXBackend(), _FakeQwen3TTSModel())
    assert backend.get_supported_speakers() == ["Vivian", "Ryan"]


# ─── VoiceDesign：自然语言造音色 ─────────────────────────────
def test_voice_design_uses_generate_voice_design():
    model = _FakeQwen3TTSModel()
    backend = _attach(Qwen3TTSVoiceDesignMLXBackend(language="Chinese"), model)

    backend.synthesize("instruct", "哥哥，你回来啦！", "", instruct_text="撒娇稚嫩的萝莉女声")

    entry, kwargs = model.last
    assert entry == "generate_voice_design"
    assert kwargs["instruct"] == "撒娇稚嫩的萝莉女声"
    assert kwargs["language"] == "Chinese"
    assert "speaker" not in kwargs  # VoiceDesign 无预设音色


def test_voice_design_requires_instruct():
    backend = _attach(Qwen3TTSVoiceDesignMLXBackend(), _FakeQwen3TTSModel())
    with pytest.raises(InvalidAudioInput) as ei:
        backend.synthesize("instruct", "你好", "", instruct_text="   ")
    assert "instruct_text" in str(ei.value)


# ─── 片段拼接 ────────────────────────────────────────────────
def test_multiple_chunks_are_concatenated():
    class _Multi(_FakeQwen3TTSModel):
        def generate(self, **kwargs):
            self.calls.append(("generate", kwargs))
            yield _Result(100)
            yield _Result(50)

    backend = _attach(Qwen3TTSBaseMLXBackend(), _Multi())
    audio = backend.synthesize("zero_shot", "长文本", "ref.wav", ref_text="x")
    assert len(audio) == 150


def test_empty_generation_raises():
    class _Empty(_FakeQwen3TTSModel):
        def generate(self, **kwargs):
            self.calls.append(("generate", kwargs))
            return iter(())

    backend = _attach(Qwen3TTSBaseMLXBackend(), _Empty())
    with pytest.raises(ModelLoadError):
        backend.synthesize("zero_shot", "你好", "ref.wav", ref_text="x")


# ─── 模型目录命名（规避 mlx_audio 的架构误判） ────────────────
def test_default_model_dir_avoids_arch_misdetect(tmp_path, monkeypatch):
    """目录名不得含会被 mlx_audio 切出 'qwen3'/'base' 等片段的 '-'。"""
    monkeypatch.setattr("audioclone.config.QWEN3_TTS_MODEL_DIR", tmp_path)
    for backend in (
        Qwen3TTSBaseMLXBackend(),
        Qwen3TTSCustomVoiceMLXBackend(),
        Qwen3TTSVoiceDesignMLXBackend(),
    ):
        assert backend.model_dir.parent == tmp_path
        assert "-" not in Path(backend.model_dir).name
        assert Path(backend.model_dir).name.startswith("qwen3_tts_")


# ─── 依赖缺失时的可操作报错 ──────────────────────────────────
def test_load_without_qwen3_tts_support_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(
        "audioclone.core.backends.qwen3_tts_mlx.has_qwen3_tts_support", lambda: False
    )
    backend = Qwen3TTSBaseMLXBackend()
    with pytest.raises(ModelLoadError) as ei:
        backend.load()

    msg = str(ei.value)
    assert "qwen3_tts" in msg
    assert "mlx-audio" in msg  # 给出可操作的依赖指引
    # 加载失败不得残留槽位占用
    assert model_slots.active(ModelSlot.TTS) is None


def test_support_probe_is_boolean():
    """探测函数返回布尔值，供加载前的依赖自检使用。"""
    assert isinstance(has_qwen3_tts_support(), bool)


# ─── 服务层串联：模式校验在编排前 fail-fast ──────────────────
def test_service_rejects_unsupported_mode_before_inference(tmp_path, monkeypatch):
    monkeypatch.setattr("audioclone.core.service.tts_service.OUTPUT_DIR", tmp_path)
    backend = _attach(Qwen3TTSCustomVoiceMLXBackend(), _FakeQwen3TTSModel())
    svc = TTSService(backend)

    with pytest.raises(InvalidAudioInput) as ei:
        svc.synthesize_to_file(mode="zero_shot", text="你好", ref_audio_path="ref.wav")
    assert "customvoice" in str(ei.value)


def test_service_allows_supported_mode(tmp_path, monkeypatch):
    monkeypatch.setattr("audioclone.core.service.tts_service.OUTPUT_DIR", tmp_path)
    backend = _attach(Qwen3TTSVoiceDesignMLXBackend(), _FakeQwen3TTSModel())
    svc = TTSService(backend)
    assert svc.sample_rate == SAMPLE_RATE

    svc.synthesize_to_file(
        mode="instruct",
        text="哥哥，你回来啦！",
        instruct_text="撒娇的萝莉女声",
        filename="design.wav",
    )
    assert (tmp_path / "design.wav").exists()
