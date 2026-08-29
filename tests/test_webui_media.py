"""WebUI 媒体落盘单测 — 上传临时目录与结果输出目录的行为契约。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audioclone.webui import media


def test_save_upload_places_file_in_upload_root():
    path = media.save_upload("ref.wav", b"RIFFfake")
    assert Path(path).exists()
    assert Path(path).read_bytes() == b"RIFFfake"
    assert media.is_upload_path(path)


def test_save_upload_normalizes_unknown_suffix():
    path = media.save_upload("clip.unknown", b"data")
    assert Path(path).suffix == ".wav"


def test_is_upload_path_rejects_foreign_path(tmp_path):
    foreign = tmp_path / "outside.wav"
    foreign.write_bytes(b"x")
    assert not media.is_upload_path(str(foreign))
    assert not media.is_upload_path("/etc/passwd")


def test_save_result_writes_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(media, "OUTPUT_DIR", tmp_path)
    audio = (np.ones(2400, dtype=np.float32) * 0.5).astype(np.float32)
    path = media.save_result(audio, 24000, prefix="unit")
    assert path.parent == tmp_path
    assert path.suffix == ".wav"
    assert path.exists()
    assert path.stat().st_size > 44


def test_cleanup_removes_upload_root():
    media.save_upload("a.wav", b"x")
    root = media.upload_root()
    assert root.exists()
    media.cleanup()
    assert not root.exists()
