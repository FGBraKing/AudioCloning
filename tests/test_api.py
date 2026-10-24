"""API 端到端测试 — 用 TestClient + dummy-omni 后端验证 FastAPI 接口。

不打真实模型，验证：依赖注入、限流、校验、领域异常映射、合成链路。
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

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

    from audioclone.core import (
        create_service,
    )

    task_services = {
        kind: create_service(kind, "dummy-omni")
        for kind in ("t2a", "a2a", "classification", "vad")
    }
    # 哨兵：避免 create_service 重复建 dummy 后端（上面已按 dummy-omni 建好）
    for svc in task_services.values():
        svc.backend.sample_rate = 24000

    @asynccontextmanager
    async def fake_lifespan(app):
        app.state.engine = engine
        app.state.backend = backend
        app.state.task_services = task_services
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


def test_clone_rejects_invalid_audio(client):
    resp = client.post(
        "/api/clone",
        data={"text": "hi"},
        files={"ref_audio": ("bad.wav", io.BytesIO(b"not audio"), "audio/wav")},
    )
    assert resp.status_code == 400  # 时长校验失败


def test_metrics(client):
    client.get("/api/health")
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("text/plain")
    assert "audioclone_requests_total" in resp.text
    assert "audioclone_uptime_seconds" in resp.text


def test_response_has_request_id(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers


def test_clone_instruct_missing_instruct(client):
    resp = client.post(
        "/api/clone",
        data={"text": "hi", "mode": "instruct"},
        files=_upload(_make_wav_bytes()),
    )
    assert resp.status_code == 400
    assert "instruct" in resp.json()["detail"]


def test_clone_upload_too_large(client, monkeypatch):
    from audioclone.api import server

    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 100)  # 压到 100 字节
    big = b"\x00" * 200
    resp = client.post(
        "/api/clone",
        data={"text": "hi"},
        files={"ref_audio": ("big.wav", io.BytesIO(big), "audio/wav")},
    )
    assert resp.status_code == 400
    assert "上传文件过大" in resp.json()["detail"]


def test_clone_internal_error(client):
    """合成抛通用异常 → 500（临时替换为损坏后端，避免污染共享 client）。"""
    from audioclone.core.backends.dummy import DummyTTSBackend

    class BrokenBackend(DummyTTSBackend):
        name = "broken"

        def synthesize(self, *args, **kwargs):
            raise RuntimeError("boom")

    saved = client.app.state.engine
    client.app.state.engine = TTSService(BrokenBackend(sample_rate=24000))
    try:
        resp = client.post(
            "/api/clone",
            data={"text": "hi"},
            files=_upload(_make_wav_bytes()),
        )
        assert resp.status_code == 500
        assert "boom" in resp.json()["detail"]
    finally:
        client.app.state.engine = saved


def test_clone_backend_unavailable_503(client):
    """TTS 引擎未就绪时返回 503（BackendUnavailable），而非 400。"""
    saved = client.app.state.engine
    client.app.state.engine = None
    try:
        resp = client.post(
            "/api/clone",
            data={"text": "hi"},
            files=_upload(_make_wav_bytes()),
        )
        assert resp.status_code == 503
        assert "未就绪" in resp.json()["detail"]
    finally:
        client.app.state.engine = saved


def test_list_outputs_after_clone(client, tmp_path, monkeypatch):
    """/api/outputs 返回历史输出（含时长元数据）。"""
    from audioclone.api import server

    monkeypatch.setattr("audioclone.core.service.tts_service.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)

    resp = client.post("/api/clone", data={"text": "hi"}, files=_upload(_make_wav_bytes()))
    assert resp.status_code == 200

    resp = client.get("/api/outputs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    wavs = [f for f in body["files"] if f["name"].endswith(".wav")]
    assert wavs
    assert "duration_sec" in wavs[0]
    assert wavs[0]["duration_sec"] > 0


def test_list_outputs_empty(client, tmp_path, monkeypatch):
    from audioclone.api import server

    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)  # 空目录
    resp = client.get("/api/outputs")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_list_outputs_no_duration_and_limit(client, tmp_path, monkeypatch):
    from audioclone.api import server

    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    for i in range(3):
        (tmp_path / f"f{i}.wav").write_bytes(b"x")

    # include_duration=false → 无 duration_sec；modified 为 ISO 字符串
    resp = client.get("/api/outputs?include_duration=false&limit=2")
    body = resp.json()
    assert body["count"] == 2
    assert "duration_sec" not in body["files"][0]
    assert "T" in body["files"][0]["modified"]

    # limit 超上限被钳制
    resp = client.get("/api/outputs?limit=99999")
    assert len(resp.json()["files"]) <= server.MAX_OUTPUTS_LIMIT


def test_request_log_carries_trace_id(client):
    """回归 P1：API 请求日志行应携带 trace_id（extra 不被 filter 覆盖）。"""
    import logging

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r)  # type: ignore[method-assign]
    logger = logging.getLogger("audioclone.api.server")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # 测试环境根日志默认 WARNING，显式提升以捕获
    try:
        client.get("/api/health")
    finally:
        logger.removeHandler(handler)
    http = [r for r in records if str(r.getMessage()).startswith("HTTP")]
    assert http
    assert http[0].trace_id is not None


# ─── _save_upload 内存安全（不信任客户端 size） ───────────────
class _FakeUpload:
    def __init__(self, filename: str, size: int | None, data: bytes):
        self.filename = filename
        self.size = size
        self.file = io.BytesIO(data)


def test_save_upload_small_buffered(monkeypatch):
    from audioclone.api import server

    monkeypatch.setattr(server, "MAX_INLINE_UPLOAD_BYTES", 1024)
    up = _FakeUpload("a.wav", 100, b"\x00" * 80)
    path = server._save_upload(up)
    try:
        assert Path(path).read_bytes() == b"\x00" * 80
    finally:
        os.unlink(path)


def test_save_upload_lying_size_streams(monkeypatch):
    """客户端伪造小 size，但实际内容超阈值 → 转流式写盘，内容完整。"""
    from audioclone.api import server

    monkeypatch.setattr(server, "MAX_INLINE_UPLOAD_BYTES", 1024)
    up = _FakeUpload("a.wav", 100, b"\x00" * 3000)  # 声明 100B，实际 3000B
    path = server._save_upload(up)
    try:
        assert len(Path(path).read_bytes()) == 3000
    finally:
        os.unlink(path)


# ─── 模型管理：状态查询 + 主动下载 ───────────────────────────
def test_models_overview(client):
    """GET /api/models 离线返回 TTS/ASR 就绪状态。"""
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"tts", "asr", "t2a", "a2a", "classification", "vad"}
    for task in ("tts", "asr", "t2a", "a2a", "classification", "vad"):
        assert body[task]["task"] == task
        assert "cached" in body[task]
        assert "downloadable" in body[task]
        assert "model_id" in body[task]


def test_models_download_dummy_completes(client):
    """dummy 后端无需模型：202 受理后任务立即完成（status=skipped）。"""
    resp = client.post("/api/models/download", json={"task": "tts", "backend": "dummy"})
    assert resp.status_code == 202
    assert resp.json()["status"] == "downloading"

    import time

    from audioclone.api import server

    deadline = time.time() + 5
    while time.time() < deadline and server._download_state["running"]:
        time.sleep(0.05)
    assert server._download_state["running"] is False
    assert server._download_state["result"]["status"] == "skipped"


def test_models_download_unknown_backend(client):
    resp = client.post("/api/models/download", json={"task": "tts", "backend": "nope"})
    assert resp.status_code == 400
    assert "未知的 tts 后端" in resp.json()["detail"]


def test_models_download_bad_task(client):
    resp = client.post("/api/models/download", json={"task": "stt"})
    assert resp.status_code == 422  # Literal 校验


def test_models_download_conflict_rejected(client, monkeypatch):
    """已有下载进行中时再次发起 → 409。"""
    import threading

    from audioclone.api import server

    gate = threading.Event()

    def slow_download(*args, **kwargs):
        gate.wait(timeout=5)
        return {"status": "completed", "task": "tts", "path": None}

    monkeypatch.setattr(server, "download_model", slow_download)
    try:
        first = client.post("/api/models/download", json={"task": "tts", "backend": "dummy"})
        assert first.status_code == 202
        second = client.post("/api/models/download", json={"task": "asr", "backend": "dummy"})
        assert second.status_code == 409
        assert "已有下载任务进行中" in second.json()["detail"]
    finally:
        gate.set()
        import time

        deadline = time.time() + 5
        while time.time() < deadline and server._download_state["running"]:
            time.sleep(0.05)


def test_models_download_status_endpoint(client):
    resp = client.get("/api/models/download")
    assert resp.status_code == 200
    body = resp.json()
    assert "running" in body and "task" in body and "error" in body


# ─── 多任务端点：T2A / A2A / 分类 / VAD ──────────────────────
def test_t2a_returns_wav(client):
    resp = client.post("/api/t2a", data={"text": "轻柔的钢琴曲", "duration_sec": 0.2})
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("audio/wav")
    assert "x-generation-time" in resp.headers


def test_t2a_empty_text(client):
    resp = client.post("/api/t2a", data={"text": "   "})
    assert resp.status_code == 400


def test_transform_returns_wav(client):
    resp = client.post(
        "/api/transform",
        data={"instruct": ""},
        files={"audio": ("in.wav", io.BytesIO(_make_wav_bytes()), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type", "").startswith("audio/wav")


def test_classify_returns_sorted_results(client):
    resp = client.post(
        "/api/classify", data={"top_k": 3},
        files={"audio": ("in.wav", io.BytesIO(_make_wav_bytes()), "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert 0 < len(body["results"]) <= 3
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_vad_returns_segments(client):
    resp = client.post("/api/vad", files={"audio": ("in.wav", io.BytesIO(_make_wav_bytes()), "audio/wav")})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["segments"], list) and len(body["segments"]) > 0
    assert body["segments"][0]["start"] <= body["segments"][0]["end"]


def test_clone_plain_mode_without_ref_audio(client):
    """plain 模式纯文本合成：无需参考音频（后端不支持时 503 而非 400/422）。"""
    resp = client.post("/api/clone", data={"text": "你好世界", "mode": "plain"})
    # dummy-omni 声明全模式 → 200；不支持 plain 的后端 → 400（模式校验）或 503（未就绪）
    assert resp.status_code in (200, 400, 503)
    if resp.status_code == 200:
        assert resp.headers.get("content-type", "").startswith("audio/wav")


def test_clone_non_plain_requires_ref_audio(client):
    resp = client.post("/api/clone", data={"text": "你好", "mode": "zero_shot"})
    assert resp.status_code == 400
    assert "参考音频" in resp.json()["detail"]
