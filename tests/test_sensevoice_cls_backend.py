"""SenseVoice (MLX) 分类后端单元测试 — 注入假模型，离线运行。"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from audioclone.core import model_slots
from audioclone.core.backends.sensevoice_mlx import (
    EMOTION_TOKEN_IDS,
    EVENT_TOKEN_IDS,
    SenseVoiceMLXBackend,
    has_sensevoice_support,
)
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot


class _FakeSenseVoiceModel:
    """模拟 SenseVoiceSmall：__call__ 返回 log_probs，含词表解码。"""

    class config:  # noqa: N801 - 模拟 mlx_audio 属性名
        frontend_conf = {"fs": 16000}

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []
        # 词表：token_id -> 标签文本（与上游 emo/event map 一致的示意子集）
        self.vocab = {
            25001: "▁happy", 25002: "▁sad", 25003: "▁angry", 25004: "▁neutral",
            24993: "▁Speech", 24995: "▁BGM", 24997: "▁Laughter", 24999: "▁Applause",
        }

    def _extract_features(self, audio):
        return mx.zeros((100, 80))  # fbank 形状占位 [time, n_mels]

    def __call__(self, feats, language="auto", use_itn=False):
        self.calls.append(np.asarray(feats))
        # log_probs 形状 [1, rows, vocab]；情感行 happy>其他，事件行 Speech>BGM
        logits = np.full((1, 4, 25100), -10.0, dtype=np.float32)
        logits[0, 1, 25001] = 6.0   # happy
        logits[0, 1, 25002] = 1.0   # sad
        logits[0, 2, 24993] = 4.0   # Speech
        logits[0, 2, 24995] = 2.0   # BGM
        return logits

    def _decode_tokens(self, token_ids):
        # 与上游一致：▁ 替换为空格后 strip
        return "".join(self.vocab.get(t, f"<{t}>") for t in token_ids).replace("▁", " ").strip()


@pytest.fixture(autouse=True)
def _clean_slots():
    model_slots.release_all()
    yield
    model_slots.release_all()


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch, tmp_path):
    """假装 mlx_audio 支持 sensevoice、模型解析到本地临时目录（不触网）。"""
    monkeypatch.setattr(
        "audioclone.core.backends.sensevoice_mlx.has_sensevoice_support", lambda: True
    )
    monkeypatch.setattr(
        "audioclone.core.backends.sensevoice_mlx.resolve_model_dir",
        lambda model_id, model_dir: tmp_path,
    )


def _attach(backend: SenseVoiceMLXBackend, model: _FakeSenseVoiceModel) -> SenseVoiceMLXBackend:
    def _fake_load() -> None:
        backend._model = model
        backend.sample_rate = model.config.frontend_conf["fs"]

    backend._load = _fake_load  # type: ignore[method-assign]
    backend.load()
    return backend


# ─── 身份与注册 ──────────────────────────────────────────────
def test_identity_and_slot():
    backend = SenseVoiceMLXBackend()
    assert backend.name == "sensevoice-mlx"
    assert backend.model_id == "mlx-community/SenseVoiceSmall"
    assert backend.slot is ModelSlot.CLASSIFICATION
    assert backend.sample_rate == 16000
    assert len(EMOTION_TOKEN_IDS) == 9
    assert len(EVENT_TOKEN_IDS) == 4


def test_registered():
    from audioclone.core.backends.registry import registry

    assert "sensevoice-mlx" in registry.names()
    assert "ast-torch" not in registry.names()  # torch 路线已移除


# ─── 分类 ────────────────────────────────────────────────────
def test_classify_emotion_and_event_sorted(tmp_path):
    import soundfile as sf

    model = _FakeSenseVoiceModel()
    backend = _attach(SenseVoiceMLXBackend(), model)
    wav = tmp_path / "in.wav"
    sf.write(wav, np.ones(16000, dtype=np.float32), 16000)

    results = backend.classify(str(wav), top_k=5)
    labels = [label for label, _ in results]
    assert labels[0] == "emotion:happy"       # 情感行最高
    assert "event:Speech" in labels           # 事件行参与排序
    scores = [s for _, s in results]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)  # 降序
    assert model_slots.active(ModelSlot.CLASSIFICATION) == "sensevoice-mlx"


def test_classify_labels_decoded_via_model_vocab(tmp_path):
    """标签文本经模型词表解码（▁ 前缀剥离），不硬编码文案。"""
    import soundfile as sf

    model = _FakeSenseVoiceModel()
    backend = _attach(SenseVoiceMLXBackend(), model)
    wav = tmp_path / "in.wav"
    sf.write(wav, np.ones(16000, dtype=np.float32), 16000)

    labels = [label for label, _ in backend.classify(str(wav), top_k=12)]
    assert "emotion:happy" in labels          # ▁happy -> happy
    assert "event:Laughter" in labels         # 词表外 token 退化 <24997> 占位
    assert any(label.startswith("event:token_") or label == "event:Laughter" for label in labels)


def test_classify_top_k_bounds(tmp_path):
    import soundfile as sf

    model = _FakeSenseVoiceModel()
    backend = _attach(SenseVoiceMLXBackend(), model)
    wav = tmp_path / "in.wav"
    sf.write(wav, np.ones(16000, dtype=np.float32), 16000)

    total = len(EMOTION_TOKEN_IDS) + len(EVENT_TOKEN_IDS)
    assert len(backend.classify(str(wav), top_k=100)) == total
    assert len(backend.classify(str(wav), top_k=0)) == 1


# ─── 依赖与状态 ──────────────────────────────────────────────
def test_load_without_support_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(
        "audioclone.core.backends.sensevoice_mlx.has_sensevoice_support", lambda: False
    )
    with pytest.raises(ModelLoadError) as ei:
        SenseVoiceMLXBackend().load()
    assert "sensevoice" in str(ei.value)
    assert "mlx-audio" in str(ei.value)
    assert model_slots.active(ModelSlot.CLASSIFICATION) is None


def test_support_probe_is_boolean():
    assert isinstance(has_sensevoice_support(), bool)


def test_get_status():
    status = SenseVoiceMLXBackend().get_status()
    assert status["model_id"] == "mlx-community/SenseVoiceSmall"
    assert status["model_loaded"] is False
    assert status["emotion_classes"] == 9
    assert status["event_classes"] == 4
