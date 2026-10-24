"""Qwen3-ASR (MLX) 后端单元测试 — 全部注入假模型/打桩解析，离线运行。"""

from __future__ import annotations

import pytest

from audioclone.core import model_slots
from audioclone.core.backends.qwen3_asr_mlx import (
    Qwen3AsrMLXBackend,
    has_qwen3_asr_support,
)
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot


class _FakeQwen3AsrModel:
    """记录调用的假 Qwen3-ASR 模型（mlx_audio.stt qwen3_asr 接口形状）。"""

    sample_rate = 16000

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def generate(self, audio, **kwargs):
        self.calls.append(("generate", {"audio": audio, **kwargs}))

        class _Out:  # 模拟 STTOutput
            text = "  你好，世界。 "

        return _Out()


@pytest.fixture(autouse=True)
def _clean_slots():
    model_slots.release_all()
    yield
    model_slots.release_all()


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch, tmp_path):
    """假装 mlx_audio 支持 qwen3_asr、模型解析返回本地临时目录（不触网）。"""
    monkeypatch.setattr(
        "audioclone.core.backends.qwen3_asr_mlx.has_qwen3_asr_support", lambda: True
    )
    monkeypatch.setattr(
        "audioclone.core.backends.qwen3_asr_mlx.resolve_model_dir",
        lambda model_id, model_dir: tmp_path,
    )


def _attach(backend: Qwen3AsrMLXBackend, model: _FakeQwen3AsrModel) -> Qwen3AsrMLXBackend:
    """注入假模型：绕过真实的 mlx_audio 加载（走正常 load() 模板以覆盖槽位逻辑）。"""

    def _fake_load() -> None:
        backend._model = model
        backend.sample_rate = model.sample_rate

    backend._load = _fake_load  # type: ignore[method-assign]
    backend.load()
    return backend


# ─── 身份与注册 ──────────────────────────────────────────────
def test_identity_and_slot():
    backend = Qwen3AsrMLXBackend()
    assert backend.name == "qwen3-asr-mlx"
    assert backend.model_id == "mlx-community/Qwen3-ASR-0.6B-8bit"
    assert backend.slot is ModelSlot.ASR
    assert backend.model_dir is None  # 默认走 HF 系统缓存


def test_registered():
    from audioclone.core.backends.registry import registry

    assert "qwen3-asr-mlx" in registry.names()


# ─── 转写 ────────────────────────────────────────────────────
def test_transcribe_returns_stripped_text():
    model = _FakeQwen3AsrModel()
    backend = _attach(Qwen3AsrMLXBackend(), model)
    text = backend.transcribe("/tmp/a.wav")
    assert text == "你好，世界。"  # 假模型输出两侧空白被剥离


def test_transcribe_passes_language_when_set():
    model = _FakeQwen3AsrModel()
    backend = _attach(Qwen3AsrMLXBackend(language="Chinese"), model)
    backend.transcribe("/tmp/a.wav")
    _, kwargs = model.calls[-1]
    assert kwargs["language"] == "Chinese"


def test_transcribe_language_none_by_default():
    model = _FakeQwen3AsrModel()
    backend = _attach(Qwen3AsrMLXBackend(), model)
    backend.transcribe("/tmp/a.wav")
    _, kwargs = model.calls[-1]
    assert kwargs["language"] is None


def test_transcribe_loads_slot_once():
    """transcribe 自动 load：占用 ASR 槽位，重复调用不重复加载。"""
    model = _FakeQwen3AsrModel()
    backend = Qwen3AsrMLXBackend()
    backend._load = lambda: (  # type: ignore[method-assign]
        setattr(backend, "_model", model)
    )
    backend.transcribe("/tmp/a.wav")
    backend.transcribe("/tmp/b.wav")
    assert model.calls[-1][1]["audio"] == "/tmp/b.wav"
    assert model_slots.active(ModelSlot.ASR) == "qwen3-asr-mlx"


# ─── 依赖与可用性 ────────────────────────────────────────────
def test_load_without_support_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(
        "audioclone.core.backends.qwen3_asr_mlx.has_qwen3_asr_support", lambda: False
    )
    backend = Qwen3AsrMLXBackend()
    with pytest.raises(ModelLoadError) as ei:
        backend.load()
    assert "qwen3_asr" in str(ei.value)
    assert "mlx-audio" in str(ei.value)
    assert model_slots.active(ModelSlot.ASR) is None  # 失败不残留槽位


def test_is_model_available_reflects_resolution(monkeypatch):
    backend = Qwen3AsrMLXBackend()
    assert backend.is_model_available is True  # stub 解析成功

    from audioclone.core.backends import qwen3_asr_mlx as mod

    def boom(model_id, model_dir):
        raise ModelLoadError("未下载")

    monkeypatch.setattr(mod, "resolve_model_dir", boom)
    assert Qwen3AsrMLXBackend().is_model_available is False


def test_support_probe_is_boolean():
    assert isinstance(has_qwen3_asr_support(), bool)


def test_get_status():
    backend = Qwen3AsrMLXBackend(language=None)
    status = backend.get_status()
    assert status["model_id"] == "mlx-community/Qwen3-ASR-0.6B-8bit"
    assert status["model_loaded"] is False
    assert status["sample_rate"] == 16000


def test_explicit_model_id_wins():
    backend = Qwen3AsrMLXBackend(model_id="custom/repo")
    assert backend.model_id == "custom/repo"
