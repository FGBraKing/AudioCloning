"""sherpa-onnx 后端单元测试（ASR / TTS / VAD）— 注入假模块，离线运行。

sherpa-onnx 为可选依赖（``uv sync --extra onnx``），测试通过 sys.modules 注入
假模块验证后端调用契约；未安装时后端应报可读错误。
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from audioclone.core import model_slots
from audioclone.core.backends.sherpa_asr_onnx import SherpaASROnnxBackend
from audioclone.core.backends.sherpa_tts_onnx import SherpaTTSOnnxBackend
from audioclone.core.backends.sherpa_vad_onnx import SherpaVADOnnxBackend
from audioclone.core.exceptions import InvalidAudioInput, ModelLoadError
from audioclone.core.model_slot import ModelSlot


# ─── 假 sherpa_onnx 模块 ─────────────────────────────────────
class _FakeRecognizer:
    def __init__(self, model: str, tokens: str, num_threads: int = 2, **kw):
        assert model.endswith("model.int8.onnx")
        assert tokens.endswith("tokens.txt")
        self.calls: list[np.ndarray] = []

    def create_stream(self):
        outer = self

        class _Stream:
            result = types.SimpleNamespace(text="  你好，世界。 ")

            def accept_waveform(self, sr, samples):
                outer.calls.append(samples)

        return _Stream()

    def decode_stream(self, stream):
        ...


class _FakeTTS:
    sample_rate = 24000

    def __init__(self, **kwargs):
        assert kwargs["vits_model"].endswith("model.int8.onnx")
        assert kwargs["voices"].endswith("voices.bin")
        assert kwargs["data_dir"].endswith("espeak-ng-data")

    @classmethod
    def from_kokoro(cls, **kwargs):
        return cls(**kwargs)

    def generate(self, text, sid=0, speed=1.0):
        assert sid == 0
        return types.SimpleNamespace(samples=np.ones(24000, dtype=np.float32), sample_rate=24000)


class _FakeSegment:
    def __init__(self, start: int, n: int):
        self.start = start
        self.samples = np.zeros(n, dtype=np.float32)


class _FakeDetector:
    def __init__(self, config, buffer_size_in_seconds=3600):
        self.config = config
        self._fed: list[np.ndarray] = []

    def accept_waveform(self, samples):
        self._fed.append(samples)

    def flush(self):
        ...

    @property
    def segments(self):
        return [_FakeSegment(0, 16000), _FakeSegment(32000, 16000)]


@pytest.fixture
def fake_sherpa(monkeypatch, tmp_path):
    """注入假 sherpa_onnx 模块 + 模型解析到本地临时目录（含必需文件）。"""
    import importlib.machinery

    mod = types.ModuleType("sherpa_onnx")
    mod.__spec__ = importlib.machinery.ModuleSpec("sherpa_onnx", None)  # find_spec 需要
    mod.OfflineRecognizer = type(
        "OfflineRecognizer", (), {"from_sense_voice": staticmethod(lambda **kw: _FakeRecognizer(**kw))}
    )
    mod.OfflineTts = _FakeTTS
    mod.VadModelConfig = lambda **kw: types.SimpleNamespace(**kw)
    mod.SileroVadModelConfig = lambda **kw: types.SimpleNamespace(**kw)
    mod.VoiceActivityDetector = _FakeDetector
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mod)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.int8.onnx").touch()
    (model_dir / "tokens.txt").touch()
    (model_dir / "silero_vad.onnx").touch()

    for mod_name in (
        "audioclone.core.backends.sherpa_asr_onnx",
        "audioclone.core.backends.sherpa_tts_onnx",
        "audioclone.core.backends.sherpa_vad_onnx",
    ):
        monkeypatch.setattr(f"{mod_name}.resolve_model_dir", lambda mid, d, _p=model_dir: _p)
    return model_dir


@pytest.fixture(autouse=True)
def _clean_slots():
    model_slots.release_all()
    yield
    model_slots.release_all()


# ─── ASR ─────────────────────────────────────────────────────
def test_asr_identity_and_slot():
    backend = SherpaASROnnxBackend()
    assert backend.name == "sherpa-asr-onnx"
    assert backend.model_id == (
        "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    )
    assert backend.slot is ModelSlot.ASR


def test_asr_transcribe(tmp_path, fake_sherpa):
    import soundfile as sf

    backend = SherpaASROnnxBackend(model_dir=str(fake_sherpa))
    wav = tmp_path / "in.wav"
    sf.write(wav, np.ones(16000, dtype=np.float32), 16000)
    text = backend.transcribe(str(wav))
    assert text == "你好，世界。"  # 假识别输出两侧空白被剥离
    assert model_slots.active(ModelSlot.ASR) == "sherpa-asr-onnx"


def test_asr_load_without_sherpa_raises_actionable_error(monkeypatch):
    monkeypatch.setattr(
        "audioclone.core.backends.sherpa_asr_onnx.has_sherpa_onnx_support", lambda: False
    )
    with pytest.raises(ModelLoadError) as ei:
        SherpaASROnnxBackend().load()
    assert "sherpa-onnx" in str(ei.value)
    assert "--extra onnx" in str(ei.value)  # 可操作安装指引
    assert model_slots.active(ModelSlot.ASR) is None


# ─── TTS ─────────────────────────────────────────────────────
def test_tts_identity_and_plain_mode():
    backend = SherpaTTSOnnxBackend()
    assert backend.name == "sherpa-tts-onnx"
    assert backend.model_id == "csukuangfj/kokoro-int8-multi-lang-v1_1"
    assert backend.slot is ModelSlot.TTS
    assert backend.supported_modes == ("plain",)


def test_tts_plain_synthesize(fake_sherpa):
    backend = SherpaTTSOnnxBackend(model_dir=str(fake_sherpa))
    audio = backend.synthesize("plain", "你好世界", speed=1.0)
    assert audio.dtype == np.float32 and audio.size == 24000
    assert model_slots.active(ModelSlot.TTS) == "sherpa-tts-onnx"


def test_tts_rejects_clone_modes(fake_sherpa):
    backend = SherpaTTSOnnxBackend(model_dir=str(fake_sherpa))
    with pytest.raises(InvalidAudioInput) as ei:
        backend.synthesize("zero_shot", "你好", "ref.wav")
    assert "plain" in str(ei.value)


# ─── VAD ─────────────────────────────────────────────────────
def test_vad_identity_and_slot():
    backend = SherpaVADOnnxBackend()
    assert backend.name == "sherpa-vad-onnx"
    assert backend.model_id == "onnx-community/silero-vad"
    assert backend.slot is ModelSlot.VAD


def test_vad_detects_segments(tmp_path, fake_sherpa):
    import soundfile as sf

    backend = SherpaVADOnnxBackend(model_dir=str(fake_sherpa))
    wav = tmp_path / "in.wav"
    sf.write(wav, np.ones(32000, dtype=np.float32), 16000)
    segments = backend.vad(str(wav))
    assert segments == [(0.0, 1.0), (2.0, 3.0)]
    assert model_slots.active(ModelSlot.VAD) == "sherpa-vad-onnx"


def test_vad_missing_onnx_file_raises(fake_sherpa, tmp_path):
    """模型目录无 .onnx 文件 → 可读错误。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    import audioclone.core.backends.sherpa_vad_onnx as mod

    mod.resolve_model_dir = lambda mid, d: empty  # 覆盖 fixture 的打桩
    with pytest.raises(ModelLoadError) as ei:
        SherpaVADOnnxBackend(model_dir=str(empty)).load()
    assert ".onnx" in str(ei.value)
