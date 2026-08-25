"""音频工具单元测试 — 数值安全、重采样、响度、校验、输出清理。"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from audioclone.utils.audio import (
    loudness_normalize,
    prune_outputs,
    resample_audio,
    sanitize_audio,
    to_mono,
    validate_ref_audio,
)


def _make_wav(path, sr: int = 16000, seconds: float = 1.0) -> str:
    data = (np.ones(int(sr * seconds), dtype=np.float32) * 0.1).astype(np.float32)
    sf.write(str(path), data, sr)
    return str(path)


def test_sanitize_removes_nan_inf():
    x = np.array([1.0, np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
    clean = sanitize_audio(x)
    assert np.all(np.isfinite(clean))
    assert np.max(clean) <= 1.0


def test_sanitize_clips_and_normalizes():
    x = np.array([10.0, -8.0, 1.0], dtype=np.float32)
    clean = sanitize_audio(x)
    assert np.max(np.abs(clean)) <= 0.99  # 限幅 + 峰值归一化


def test_sanitize_empty():
    assert sanitize_audio(np.array([], dtype=np.float32)).size == 0


def test_resample_length():
    x = np.ones(24000, dtype=np.float32)
    out = resample_audio(x, 24000, 16000)
    assert len(out) == 16000


def test_resample_same_rate_no_op():
    x = np.ones(100, dtype=np.float32)
    out = resample_audio(x, 8000, 8000)
    assert len(out) == 100


def test_prune_outputs(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"1")
    (tmp_path / "b.wav").write_bytes(b"2")
    (tmp_path / "c.wav").write_bytes(b"3")
    prune_outputs(tmp_path, keep=1)
    remaining = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert len(remaining) == 1


def test_prune_outputs_disabled(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"1")
    prune_outputs(tmp_path, keep=0)
    assert (tmp_path / "a.wav").exists()


def test_prune_outputs_cleans_segments_subdir(tmp_path):
    """递归清理：segments/ 子目录的分段落盘也受 keep 约束，空目录被移除。"""
    seg = tmp_path / "segments"
    seg.mkdir()
    for i in range(5):
        (seg / f"s{i}.wav").write_bytes(b"1")
    (tmp_path / "main.wav").write_bytes(b"1")

    prune_outputs(tmp_path, keep=1)

    remaining = list(tmp_path.rglob("*.wav"))
    assert len(remaining) == 1
    assert not seg.exists()  # 文件被清空后，空目录一并移除


# ─── sanitize 边界 ──────────────────────────────────────────
def test_sanitize_all_zero():
    x = np.zeros(1000, dtype=np.float32)
    assert np.all(sanitize_audio(x) == 0)


def test_sanitize_all_nan():
    x = np.full(100, np.nan, dtype=np.float32)
    clean = sanitize_audio(x)
    assert np.all(np.isfinite(clean))
    assert np.all(clean == 0)


def test_sanitize_removes_constant_dc():
    # 带直流偏移的非恒定信号 → 移除 DC 后均值趋近 0
    t = np.linspace(0, 100, 10000)
    x = (0.5 + 0.2 * np.sin(t)).astype(np.float32)
    clean = sanitize_audio(x)
    assert abs(float(np.mean(clean))) < 0.01


# ─── loudness_normalize ─────────────────────────────────────
def test_loudness_normalize_finite():
    x = (np.sin(np.linspace(0, 100, 24000)) * 0.3).astype(np.float32)
    out = loudness_normalize(x, 24000)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0


def test_loudness_normalize_empty():
    assert loudness_normalize(np.array([], dtype=np.float32), 24000).size == 0


# ─── validate_ref_audio 分支 ────────────────────────────────
def test_validate_ref_audio_missing(tmp_path):
    ok, msg = validate_ref_audio(str(tmp_path / "nope.wav"))
    assert not ok
    assert "不存在" in msg


def test_validate_ref_audio_invalid_file(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"not audio")
    ok, _ = validate_ref_audio(str(p))
    assert not ok


def test_validate_ref_audio_too_short(tmp_path):
    ok, msg = validate_ref_audio(_make_wav(tmp_path / "short.wav", sr=8000, seconds=0.1))
    assert not ok
    assert "过短" in msg


def test_validate_ref_audio_valid(tmp_path):
    ok, _ = validate_ref_audio(_make_wav(tmp_path / "ok.wav"))
    assert ok


# ─── to_mono 声道统一 ───────────────────────────────────────
def test_to_mono_passthrough():
    x = np.ones(100, dtype=np.float32)
    assert to_mono(x) is x


def test_to_mono_first_channel_default():
    stereo = np.stack([np.ones(100) * 0.3, np.ones(100) * 0.9], axis=1).astype(np.float32)
    mono = to_mono(stereo)
    assert mono.ndim == 1
    assert np.allclose(mono, 0.3)


def test_to_mono_mean_mode():
    stereo = np.stack([np.ones(100) * 0.3, np.ones(100) * 0.9], axis=1).astype(np.float32)
    mono = to_mono(stereo, mode="mean")
    assert mono.ndim == 1
    assert np.allclose(mono, 0.6)