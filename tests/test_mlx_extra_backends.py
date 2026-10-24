"""A2A DeepFilterNet (MLX) 后端单元测试 — 注入假模型，离线运行。

（T2A 音乐：参数量放宽例外；VAD：silero-vad-v6 已确认存在）
"""

from __future__ import annotations

import numpy as np
import pytest

from audioclone.core import model_slots
from audioclone.core.backends.deepfilternet_mlx import DeepFilterNetMLXBackend
from audioclone.core.backends.minimax_music_mlx import MiniMaxMusicMLXBackend
from audioclone.core.backends.silero_vad_mlx import SileroVADMLXBackend
from audioclone.core.exceptions import ModelLoadError
from audioclone.core.model_slot import ModelSlot


class _Result:
    def __init__(self, n: int) -> None:
        self.audio = np.ones(n, dtype=np.float32)


class _FakeDFNModel:
    sample_rate = 48000

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def enhance_array(self, audio):
        self.calls.append(audio)
        return audio * 0.5



@pytest.fixture(autouse=True)
def _clean_slots():
    model_slots.release_all()
    yield
    model_slots.release_all()


@pytest.fixture(autouse=True)
def _stub_resolutions(monkeypatch, tmp_path):
    """三个后端的模型解析统一打到本地临时目录（不触网）。"""
    for mod in (
        "audioclone.core.backends.deepfilternet_mlx",
        "audioclone.core.backends.minimax_music_mlx",
        "audioclone.core.backends.silero_vad_mlx",
    ):
        monkeypatch.setattr(f"{mod}.resolve_model_dir", lambda mid, d: tmp_path)


def _attach(backend, model):
    def _fake_load():
        backend._model = model
        backend.sample_rate = getattr(model, "sample_rate", backend.sample_rate)

    backend._load = _fake_load  # type: ignore[method-assign]
    backend.load()
    return backend


# ─── A2A: DeepFilterNet ──────────────────────────────────────
def test_dfn_identity_and_slot():
    backend = DeepFilterNetMLXBackend()
    assert backend.name == "deepfilternet-mlx"
    assert backend.model_id == "mlx-community/DeepFilterNet-mlx"
    assert backend.slot is ModelSlot.A2A
    assert backend.sample_rate == 48000


def test_dfn_registered():
    from audioclone.core.backends.registry import registry

    assert "deepfilternet-mlx" in registry.names()


def test_dnf_transform_enhances(tmp_path, monkeypatch):
    import soundfile as sf

    model = _FakeDFNModel()
    backend = _attach(DeepFilterNetMLXBackend(), model)
    wav = tmp_path / "in.wav"
    sf.write(wav, np.ones(4800, dtype=np.float32), 48000)
    out = backend.transform(str(wav))
    assert out.dtype == np.float32
    assert np.allclose(out, 0.5, atol=1e-3)  # 假模型输出 audio*0.5（WAV 16bit 量化容差）
    assert len(model.calls) == 1


def test_dfn_load_without_support_raises(monkeypatch):
    monkeypatch.setattr(
        "audioclone.core.backends.deepfilternet_mlx.has_deepfilternet_support", lambda: False
    )
    with pytest.raises(ModelLoadError) as ei:
        DeepFilterNetMLXBackend().load()
    assert "deepfilternet" in str(ei.value)




# ─── T2A: MiniMax Music（参数量放宽例外，仓库已确认存在） ────
class _FakeMusicModel:
    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def generate(self, text, *, lyrics="", gen_duration=None, verbose=False, **kw):
        self.calls.append(("generate", {"text": text, "lyrics": lyrics, "gen_duration": gen_duration}))
        yield _Result(100)
        yield _Result(50)


def test_music_identity_and_slot():
    backend = MiniMaxMusicMLXBackend()
    assert backend.name == "minimax-music-mlx"
    assert backend.model_id == "mlx-community/MiniMax-Music3-8bit"
    assert backend.slot is ModelSlot.T2A


def test_music_registered():
    from audioclone.core.backends.registry import registry

    assert "minimax-music-mlx" in registry.names()


def test_music_generate_collects_chunks():
    model = _FakeMusicModel()
    backend = _attach(MiniMaxMusicMLXBackend(), model)
    audio = backend.text_to_audio("欢快的钢琴曲", duration_sec=30.0, lyrics="la la la")
    assert len(audio) == 150  # 两段拼接
    entry, kwargs = model.calls[-1]
    assert entry == "generate"
    assert kwargs["text"] == "欢快的钢琴曲"
    assert kwargs["lyrics"] == "la la la"
    assert kwargs["gen_duration"] == 30.0
    assert model_slots.active(ModelSlot.T2A) == "minimax-music-mlx"


def test_music_load_without_support_raises(monkeypatch):
    monkeypatch.setattr(
        "audioclone.core.backends.minimax_music_mlx.has_minimax_music_support", lambda: False
    )
    with pytest.raises(ModelLoadError) as ei:
        MiniMaxMusicMLXBackend().load()
    assert "minimax_music3" in str(ei.value)
    assert model_slots.active(ModelSlot.T2A) is None


# ─── VAD: Silero（silero-vad-v6，已确认存在） ────────────────
class _FakeVADModel:
    sample_rate = 16000

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, dict]] = []

    def generate(self, audio, sample_rate=None, **kw):
        self.calls.append((audio, kw))

        class _Out:
            timestamps = [{"start": 0.0, "end": 1.5}, {"start": 2.0, "end": 3.2}]

        return _Out()


def test_vad_identity_and_slot():
    backend = SileroVADMLXBackend()
    assert backend.name == "silero-vad-mlx"
    assert backend.model_id == "mlx-community/silero-vad-v6"
    assert backend.slot is ModelSlot.VAD
    assert backend.sample_rate == 16000


def test_vad_registered():
    from audioclone.core.backends.registry import registry

    assert "silero-vad-mlx" in registry.names()


def test_vad_detects_segments(tmp_path):
    import soundfile as sf

    model = _FakeVADModel()
    backend = _attach(SileroVADMLXBackend(), model)
    wav = tmp_path / "in.wav"
    sf.write(wav, np.ones(16000, dtype=np.float32), 16000)
    segments = backend.vad(str(wav))
    assert segments == [(0.0, 1.5), (2.0, 3.2)]
    assert model_slots.active(ModelSlot.VAD) == "silero-vad-mlx"


def test_vad_threshold_passthrough(tmp_path):
    import soundfile as sf

    model = _FakeVADModel()
    backend = _attach(SileroVADMLXBackend(), model)
    wav = tmp_path / "in.wav"
    sf.write(wav, np.ones(16000, dtype=np.float32), 16000)
    backend.vad(str(wav), threshold=0.6)
    _, kwargs = model.calls[-1]
    assert kwargs == {"threshold": 0.6}


def test_vad_load_without_support_raises(monkeypatch):
    monkeypatch.setattr(
        "audioclone.core.backends.silero_vad_mlx.has_silero_vad_support", lambda: False
    )
    with pytest.raises(ModelLoadError) as ei:
        SileroVADMLXBackend().load()
    assert "silero_vad" in str(ei.value)
    assert model_slots.active(ModelSlot.VAD) is None
