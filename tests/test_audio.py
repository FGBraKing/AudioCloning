"""音频工具单元测试 — 数值安全、重采样、输出清理。"""

from __future__ import annotations

import numpy as np

from audioclone.utils.audio import (
    prune_outputs,
    resample_audio,
    sanitize_audio,
)


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