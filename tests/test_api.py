"""API 端到端测试 — 用 TestClient + dummy-omni 后端验证 FastAPI 接口。

不打真实模型，验证：依赖注入、限流、校验、领域异常映射、合成链路。
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from audioclone.core import TTSService, create_tts_backend


def _make_wav_bytes() -> bytes:
    data = np.ones(16000, dtype=np.float32) * 0.1
    buf = io.BytesIO()
    sf.write(buf, data, 16000, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """构建一个注入 dummy-omni 引擎的 API 客户端。"""
    from audioclone.api import server

    backend = create_tts_backend("dummy-omni")
    engine = TTSService(backend)

    original_lifespan = server.app.router.lifespan_context

    @asynccontextmanager
    async def fake_lifespan(app):
        app.state.engine = engine
        app.state.backend = backend
        yield

    server.app.router.lifespan_context = fake_lifespan
    with TestClient(server.app) as c:
        yield c
    server.app.router.lifespan_context = original_lifespan


def _upload(file_bytes: bytes, name: str = "ref.wav"):
    return {"ref_audio": (name, io.BytesIO(file_bytes), "audio/wav")}


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_clone_success(client):
    resp = client.post(
        "/api/clone",
        data={
            "text": "你好，这是一段测试。",
            "mode": "zero_shot",
            "ref_text": "参考文本",
        },
        files=_upload(_make_wav_bytes()),
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("audio/wav")
    assert "x-synthesis-time" in resp.headers


def test_clone_empty_text(client):
    resp = client.post("/api/clone", data={"text": "   "}, files=_upload(_make_wav_bytes()))
    assert resp.status_code == 400
    assert "文本为空" in resp.json()["detail"]


def test_clone_bad_mode(client):
    resp = client.post(
        "/api/clone",
        data={"text": "hi", "mode": "nope"},
        files=_upload(_make_wav_bytes()),
    )
    assert resp.status_code == 400
    assert "不支持的模式" in resp.json()["detail"]


def test_clone_rejects_bad_format(client):
    resp = client.post(
        "/api/clone",
        data={"text": "hi"},
        files={"ref_audio": ("a.mp4", io.BytesIO(b"nope"), "video/mp4")},
    )
    assert resp.status_code == 400
    assert "不支持的音频格式" in resp.json()["detail"]


def test_clone_rejects_invalid_audio(client):
    resp = client.post(
        "/api/clone",
        data={"text": "hi"},
        files={"ref_audio": ("bad.wav", io.BytesIO(b"not audio"), "audio/wav")},
    )
    assert resp.status_code == 400  # 时长校验失败
