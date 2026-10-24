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
    def generate(self, text=None, **kwargs):
        kwargs["text"] = text  # 兼容位置传参（chatterbox）与关键字传参（qwen3）
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


@pytest.fixture(autouse=True)
def _stub_model_resolution(monkeypatch, tmp_path):
    """模型解析返回本地临时目录：单元测试绝不触发 HF 缓存解析/网络下载。"""
    for mod in (
        "audioclone.core.backends.qwen3_tts_mlx",
        "audioclone.core.backends.chatterbox_mlx",
        "audioclone.core.backends.spark_mlx",
    ):
        monkeypatch.setattr(f"{mod}.resolve_model_dir", lambda model_id, model_dir: tmp_path)


def _attach(backend: Qwen3TTSMLXBackend, model: _FakeQwen3TTSModel) -> Qwen3TTSMLXBackend:
    """注入假模型：绕过真实的 mlx_audio 加载。"""
    backend._load_model = lambda loader, model_path: model  # type: ignore[method-assign]
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
    other._load_model = lambda loader, model_path: _FakeQwen3TTSModel()  # type: ignore[method-assign]
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


# ─── 模型路径解析（默认 HF 系统缓存 / 自定义目录覆盖） ─────────
def test_default_model_dir_is_hf_cache():
    """默认不指定本地目录：model_dir 为 None，加载时按 model_id 解析 HF 系统缓存。"""
    for backend in (
        Qwen3TTSBaseMLXBackend(),
        Qwen3TTSCustomVoiceMLXBackend(),
        Qwen3TTSVoiceDesignMLXBackend(),
    ):
        assert backend.model_dir is None
        assert backend.model_id.startswith("mlx-community/Qwen3-TTS")


def test_explicit_model_dir_wins_over_hf_cache(tmp_path):
    """显式给定 model_dir 时按原样使用（自定义路径优先于 HF 缓存）。"""
    backend = Qwen3TTSBaseMLXBackend(model_dir=tmp_path)
    assert backend.model_dir == tmp_path


def test_resolve_model_dir_custom_path(tmp_path):
    """resolve_model_dir：自定义目录存在则直接返回，不存在则报可读错误。"""
    from audioclone.utils.model_path import resolve_model_dir

    assert resolve_model_dir("some/model", tmp_path) == tmp_path
    with pytest.raises(ModelLoadError) as ei:
        resolve_model_dir("some/model", tmp_path / "missing")
    assert "不存在" in str(ei.value)


def test_resolve_model_dir_does_not_download(monkeypatch):
    """resolve_model_dir 未命中缓存时不下载，报带指引的错误。"""
    from audioclone.utils import model_path

    monkeypatch.setattr(model_path, "find_cached_model", lambda model_id: None)
    with pytest.raises(ModelLoadError) as ei:
        model_path.resolve_model_dir("some/model", None)
    assert "未下载" in str(ei.value)


def test_resolve_model_dir_requires_model_id_when_no_dir():
    """既无自定义目录又无 model_id 时无法定位模型。"""
    from audioclone.utils.model_path import resolve_model_dir

    with pytest.raises(ModelLoadError) as ei:
        resolve_model_dir("", None)
    assert "model_id" in str(ei.value)


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


# ─── Chatterbox / Spark（克隆备选，模型仓库已核实） ──────────
def test_chatterbox_identity_and_modes():
    from audioclone.core.backends.chatterbox_mlx import ChatterboxMLXBackend

    backend = ChatterboxMLXBackend()
    assert backend.name == "chatterbox-mlx"
    assert backend.model_id == "mlx-community/chatterbox-turbo-8bit"
    assert backend.supported_modes == ("zero_shot", "plain")
    assert backend.exaggeration == 0.5
    assert backend.slot is ModelSlot.TTS


def test_chatterbox_registered():
    from audioclone.core.backends.registry import registry

    assert "chatterbox-mlx" in registry.names()
    assert "spark-mlx" in registry.names()


def test_chatterbox_zero_shot_passes_ref_and_exaggeration():
    from audioclone.core.backends.chatterbox_mlx import ChatterboxMLXBackend

    model = _FakeQwen3TTSModel()  # 复用：generate 生成器 + calls 记录
    backend = _attach(ChatterboxMLXBackend(exaggeration=0.8), model)
    audio = backend.synthesize("zero_shot", "你好", "ref.wav")
    assert audio.dtype == np.float32
    entry, kwargs = model.last
    assert entry == "generate"
    assert kwargs["ref_audio"] == "ref.wav"
    assert kwargs["exaggeration"] == 0.8


def test_chatterbox_plain_without_ref():
    from audioclone.core.backends.chatterbox_mlx import ChatterboxMLXBackend

    model = _FakeQwen3TTSModel()
    backend = _attach(ChatterboxMLXBackend(), model)
    backend.synthesize("plain", "你好")
    _, kwargs = model.last
    assert kwargs["ref_audio"] is None  # plain：预置默认音色


def test_chatterbox_rejects_instruct():
    from audioclone.core.backends.chatterbox_mlx import ChatterboxMLXBackend

    model = _FakeQwen3TTSModel()
    backend = _attach(ChatterboxMLXBackend(), model)
    with pytest.raises(InvalidAudioInput) as ei:
        backend.synthesize("instruct", "你好", instruct_text="开心")
    assert "仅支持推理模式" in str(ei.value)
    assert "exaggeration" in str(ei.value)


def test_spark_identity_and_nc_license_note():
    from audioclone.core.backends.spark_mlx import SparkMLXBackend

    backend = SparkMLXBackend()
    assert backend.name == "spark-mlx"
    assert backend.model_id == "mlx-community/Spark-TTS-0.5B-8bit"
    assert backend.supported_modes == ("zero_shot",)  # 需参考音频（CC BY-NC-SA，非商用）


def test_spark_requires_ref_audio():
    from audioclone.core.backends.spark_mlx import SparkMLXBackend

    model = _FakeQwen3TTSModel()
    backend = _attach(SparkMLXBackend(), model)
    with pytest.raises(InvalidAudioInput) as ei:
        backend.synthesize("zero_shot", "你好", ref_audio_path="")
    assert "参考音频" in str(ei.value)


def test_spark_zero_shot_passes_ref(monkeypatch, tmp_path):
    """Spark 的 generate 返回单结果（非生成器）：单/多结果统一收集。"""
    from audioclone.core.backends.spark_mlx import SparkMLXBackend

    class _Single:
        audio = np.ones(50, dtype=np.float32)

    captured = {}

    class _SparkModel:
        sample_rate = SAMPLE_RATE

        def generate(self, text, **kwargs):
            captured.update(kwargs)
            return _Single()

    backend = _attach(SparkMLXBackend(), _SparkModel())
    audio = backend.synthesize("zero_shot", "你好", "ref.wav", ref_text="转录")
    assert len(audio) == 50
    assert captured["ref_audio"] == Path("ref.wav")
    assert captured["ref_text"] == "转录"
