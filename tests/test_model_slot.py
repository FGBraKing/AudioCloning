"""模型槽位守卫测试 — 保证运行时每种任务最多只有一个已加载模型。

契约：
- TTS/ASR 各有独立槽位，互不干扰（本项目的目标形态就是「一个 TTS + 一个 ASR」）；
- 同一槽位被第二个**不同**后端占用时抛 ``ModelSlotError``；
- ``close()`` 归还槽位；``_load()`` 抛异常时自动回滚，不留悬空占用。
"""

from __future__ import annotations

import numpy as np
import pytest

from audioclone.core import ModelSlot, ModelSlotError, model_slots
from audioclone.core.capabilities import ModelBackend, TextToSpeechCapability
from audioclone.core.exceptions import ModelLoadError


class _FakeTTS(ModelBackend, TextToSpeechCapability):
    """可控的假 TTS 后端：记录 load/close 次数，可模拟加载失败。"""

    capabilities = (TextToSpeechCapability,)
    slot = ModelSlot.TTS

    def __init__(self, name: str = "fake-tts", fail: bool = False) -> None:
        self.name = name
        self._fail = fail
        self.sample_rate = 24000
        self.load_count = 0
        self.close_count = 0

    def _load(self) -> None:
        self.load_count += 1
        if self._fail:
            raise ModelLoadError("模拟加载失败")

    def _close(self) -> None:
        self.close_count += 1

    def synthesize(self, mode, text, ref_audio_path, ref_text="", instruct_text="", speed=1.0):
        return np.zeros(self.sample_rate, dtype=np.float32)


@pytest.fixture(autouse=True)
def _clean_slots():
    """每个用例前后复位槽位，避免互相污染。"""
    model_slots.release_all()
    yield
    model_slots.release_all()


# ─── 登记 / 释放 ─────────────────────────────────────────────
def test_acquire_and_release():
    backend = _FakeTTS()
    backend.load()

    assert model_slots.active(ModelSlot.TTS) == "fake-tts"
    assert model_slots.active(ModelSlot.ASR) is None

    backend.close()
    assert model_slots.active(ModelSlot.TTS) is None


def test_acquire_is_idempotent_for_same_backend():
    """同一后端重复 load() 不应触发占用冲突，也不重复加载。"""
    backend = _FakeTTS()
    backend.load()
    backend.load()

    assert backend.load_count == 1
    assert model_slots.active(ModelSlot.TTS) == "fake-tts"


def test_second_backend_in_same_slot_is_rejected():
    """核心契约：同一槽位不允许第二个不同后端。"""
    first = _FakeTTS("first-tts")
    first.load()

    second = _FakeTTS("second-tts")
    with pytest.raises(ModelSlotError) as ei:
        second.load()

    assert "first-tts" in str(ei.value)
    assert "second-tts" in str(ei.value)
    # 失败者不得占用槽位，原持有者不受影响
    assert model_slots.active(ModelSlot.TTS) == "first-tts"
    assert second.load_count == 0


def test_slot_released_then_reacquirable_by_other_backend():
    first = _FakeTTS("first-tts")
    first.load()
    first.close()

    second = _FakeTTS("second-tts")
    second.load()
    assert model_slots.active(ModelSlot.TTS) == "second-tts"


def test_failed_load_rolls_back_slot():
    """``_load()`` 抛异常时槽位必须回滚，否则后续任何加载都会被拒。"""
    broken = _FakeTTS("broken-tts", fail=True)
    with pytest.raises(ModelLoadError):
        broken.load()
    assert model_slots.active(ModelSlot.TTS) is None

    healthy = _FakeTTS("healthy-tts")
    healthy.load()
    assert model_slots.active(ModelSlot.TTS) == "healthy-tts"


# ─── 槽位隔离 ────────────────────────────────────────────────
def test_tts_and_asr_slots_are_independent():
    """TTS 与 ASR 是不同槽位，可同时各持有一个（即目标运行形态）。"""
    tts = _FakeTTS()
    tts.load()
    model_slots.acquire(ModelSlot.ASR, "fake-asr")

    snapshot = model_slots.snapshot()
    assert {k: snapshot[k] for k in ("tts", "asr")} == {"tts": "fake-tts", "asr": "fake-asr"}


# ─── 无状态后端不占槽 ────────────────────────────────────────
def test_slotless_backend_does_not_occupy():
    class Stateless(ModelBackend):
        name = "stateless"

    Stateless().load()
    snap = model_slots.snapshot()
    assert set(snap.values()) == {None}


# ─── 服务层组合：一个 TTS + 一个 ASR ─────────────────────────
class _FakeASR(ModelBackend):
    """占 ASR 槽位的假后端。"""

    name = "fake-asr"
    slot = ModelSlot.ASR

    def _load(self) -> None:
        return None


def test_one_tts_plus_one_asr_coexist():
    """端到端形态：TTS 与 ASR 各加载一个，互不排斥。"""
    tts = _FakeTTS()
    asr = _FakeASR()
    tts.load()
    asr.load()

    assert {k: model_slots.snapshot()[k] for k in ("tts", "asr")} == {
        "tts": "fake-tts", "asr": "fake-asr",
    }

    tts.close()
    asr.close()
    snap = model_slots.snapshot()
    assert set(snap.values()) == {None}


def test_stateless_backends_never_occupy_slots():
    """dummy 等无状态后端不声明槽位，因此不参与「只能一个」的约束。"""
    from audioclone.core.backends.dummy import DummyOmnibackend, DummyTTSBackend

    DummyTTSBackend().load()
    DummyOmnibackend().load()
    snap = model_slots.snapshot()
    assert set(snap.values()) == {None}
