"""sherpa-cls-onnx（CED 分类）与 sherpa-denoise-onnx（GTCRN 增强）后端单元测试。

假 sherpa_onnx 模块注入 + 假模型目录，离线运行。
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioclone.core import model_slots
from audioclone.core.backends.sherpa_cls_onnx import SherpaClsOnnxBackend
from audioclone.core.backends.sherpa_denoise_onnx import SherpaDenoiseOnnxBackend
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot


class _FakeTagger:
    def __init__(self, config):
        self.config = config

    def create_stream(self):
        class _Stream:
            def accept_waveform(self, sample_rate, waveform):
                self._sr = sample_rate

        return _Stream()

    def compute(self, stream):
        return [("Speech", 0.8), ("Music", 0.5), ("Dog", 0.1)]


class _FakeDenoiser:
    def __init__(self, config):
        self.config = config

    def __call__(self, samples, sample_rate):
        return types.SimpleNamespace(samples=samples * 0.5, sample_rate=sample_rate)


@pytest.fixture(autouse=True)
def _clean_slots():
    model_slots.release_all()
    yield
    model_slots.release_all()


@pytest.fixture
def fake_sherpa(monkeypatch, tmp_path):
    """注入假 sherpa_onnx 模块 + 模型解析到本地临时目录（含必需文件）。"""
    mod = types.ModuleType("sherpa_onnx")
    mod.__spec__ = importlib.machinery.ModuleSpec("sherpa_onnx", None)
    mod.AudioTagging = _FakeTagger
    mod.AudioTaggingConfig = lambda **kw: types.SimpleNamespace(**kw)
    mod.AudioTaggingModelConfig = lambda **kw: types.SimpleNamespace(**kw)
    mod.OfflineSpeechDenoiser = _FakeDenoiser
    mod.OfflineSpeechDenoiserConfig = lambda **kw: types.SimpleNamespace(**kw)
    mod.OfflineSpeechDenoiserModelConfig = lambda **kw: types.SimpleNamespace(**kw)
    mod.OfflineSpeechDenoiserGtcrnModelConfig = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mod)

    cls_dir = tmp_path / "ced"
    cls_dir.mkdir()
    (cls_dir / "model.int8.onnx").touch()
    (cls_dir / "class_labels_indices.csv").touch()
    dn_dir = tmp_path / "gtcrn_simple"
    dn_dir.mkdir()
    (dn_dir / "gtcrn_simple.onnx").touch()

    monkeypatch.setattr(
        "audioclone.core.backends.sherpa_cls_onnx.resolve_model_dir",
        lambda mid, d: cls_dir,
    )
    return cls_dir, dn_dir


def _make_wav(tmp_path: Path) -> str:
    wav = tmp_path / "in.wav"
    sf.write(wav, np.ones(16000, dtype=np.float32), 16000)
    return str(wav)


# ─── 分类：CED ───────────────────────────────────────────────
def test_cls_identity_and_slot():
    backend = SherpaClsOnnxBackend()
    assert backend.name == "sherpa-cls-onnx"
    assert backend.model_id == "k2-fsa/sherpa-onnx-ced-small-audio-tagging-2024-04-19"
    assert backend.slot is ModelSlot.CLASSIFICATION


def test_cls_registered():
    from audioclone.core.backends.registry import registry

    assert "sherpa-cls-onnx" in registry.names()
    assert "sherpa-denoise-onnx" in registry.names()


def test_cls_sorted_top_k(fake_sherpa, tmp_path):
    backend = SherpaClsOnnxBackend(model_dir=str(fake_sherpa[0]))
    results = backend.classify(_make_wav(tmp_path), top_k=2)
    assert [label for label, _ in results] == ["Speech", "Music"]  # 置信度降序
    scores = [s for _, s in results]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert model_slots.active(ModelSlot.CLASSIFICATION) == "sherpa-cls-onnx"


# ─── A2A：GTCRN ──────────────────────────────────────────────
def test_denoise_identity_and_slot():
    backend = SherpaDenoiseOnnxBackend()
    assert backend.name == "sherpa-denoise-onnx"
    assert backend.slot is ModelSlot.A2A
    assert backend.sample_rate == 16000


def test_denoise_transform(fake_sherpa, tmp_path):
    backend = SherpaDenoiseOnnxBackend(model_dir=str(fake_sherpa[1]))
    out = backend.transform(_make_wav(tmp_path))
    assert out.dtype == np.float32
    assert np.allclose(out, 0.5, atol=1e-3)  # 假降噪输出 samples*0.5
    assert model_slots.active(ModelSlot.A2A) == "sherpa-denoise-onnx"


def test_denoise_missing_model_raises(tmp_path, monkeypatch):
    """模型目录缺 .onnx → 可读错误（指引 GitHub Release 下载）。"""
    mod = types.ModuleType("sherpa_onnx")
    mod.__spec__ = importlib.machinery.ModuleSpec("sherpa_onnx", None)
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mod)
    empty = tmp_path / "empty"
    empty.mkdir()
    backend = SherpaDenoiseOnnxBackend(model_dir=str(empty))
    backend._resolve_model_path = lambda: empty  # type: ignore[method-assign]
    with pytest.raises(ModelLoadError) as ei:
        backend.load()
    assert ".onnx" in str(ei.value)
