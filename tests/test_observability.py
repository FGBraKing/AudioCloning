"""可观测性组件测试 — 结构化日志 formatter、trace_id 上下文与进程内指标。"""

from __future__ import annotations

import json
import logging

from audioclone.api.metrics import SYNTH_BUCKETS, Metrics
from audioclone.utils.logging_utils import (
    JsonFormatter,
    TraceIdFilter,
    configure_logging,
    get_trace_id,
    set_trace_id,
)


def _record() -> logging.LogRecord:
    return logging.LogRecord(
        name="audioclone.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )


def test_json_formatter_outputs_valid_json():
    record = _record()
    record.trace_id = "abc123"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "audioclone.api"
    assert payload["message"] == "hello world"
    assert payload["trace_id"] == "abc123"
    assert "ts" in payload


def test_json_formatter_uses_context_trace_id():
    """服务层日志（无显式 extra）也能经 contextvars 携带 trace_id。"""
    set_trace_id("ctx-tid")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
        assert payload["trace_id"] == "ctx-tid"
        assert get_trace_id() == "ctx-tid"
    finally:
        set_trace_id(None)
    assert get_trace_id() is None


def test_configure_logging_json_sets_stream_handler():
    configure_logging(json=True)
    root = logging.getLogger()
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    configure_logging(json=False)  # 恢复文本模式，避免影响其他测试输出


def test_metrics_render_contains_counters():
    m = Metrics()
    m.inc_request("/api/health", 200)
    m.inc_request("/api/health", 200)
    m.inc_request("/api/clone", 500)
    m.record_synthesis(1.5)
    text = m.render()
    assert 'audioclone_requests_total{path="/api/health",status="200"} 2' in text
    assert 'audioclone_requests_total{path="/api/clone",status="500"} 1' in text
    assert "audioclone_synthesis_count_total 1" in text
    assert "audioclone_synthesis_seconds_total 1.5000" in text
    assert "audioclone_uptime_seconds" in text


def test_metrics_histogram_buckets():
    m = Metrics()
    for sec in (0.2, 3.0, 30.0):
        m.record_synthesis(sec)
    text = m.render()
    # 各桶累积计数按 bucket 上界递增
    assert 'audioclone_synthesis_duration_seconds_bucket{le="0.1"} 0' in text
    assert 'audioclone_synthesis_duration_seconds_bucket{le="0.25"} 1' in text
    assert 'audioclone_synthesis_duration_seconds_bucket{le="5"} 2' in text
    assert 'audioclone_synthesis_duration_seconds_bucket{le="30"} 3' in text
    assert 'audioclone_synthesis_duration_seconds_bucket{le="+Inf"} 3' in text
    assert "audioclone_synthesis_duration_seconds_count 3" in text
    assert len(SYNTH_BUCKETS) >= 2


def test_settings_env_override(monkeypatch):
    """Settings 支持 ``AUDIOCLONE_*`` 环境变量覆盖。"""
    from audioclone.config import Settings

    monkeypatch.setenv("AUDIOCLONE_MAX_UPLOAD_BYTES", "12345")
    monkeypatch.setenv("AUDIOCLONE_TTS_BACKEND", "dummy-omni")
    s = Settings()
    assert s.max_upload_bytes == 12345
    assert s.tts_backend == "dummy-omni"


# ─── TraceIdFilter：保留显式 extra / 注入上下文 ──────────────
def _capture_handler() -> tuple[logging.Handler, list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r)  # type: ignore[method-assign]
    handler.addFilter(TraceIdFilter())
    return handler, records


def test_trace_id_filter_preserves_explicit_extra():
    """显式 ``extra={"trace_id": ...}`` 不被 filter 覆盖（回归 P1 缺陷）。"""
    handler, records = _capture_handler()
    logger = logging.getLogger("test.traceid.preserve")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    set_trace_id("ctx-tid")
    try:
        logger.info("with-extra", extra={"trace_id": "explicit"})
        assert records[-1].trace_id == "explicit"

        logger.info("no-extra")  # 无 extra → 注入 contextvar
        assert records[-1].trace_id == "ctx-tid"
    finally:
        set_trace_id(None)
        logger.removeHandler(handler)


def test_metrics_custom_buckets():
    import pytest

    m = Metrics(buckets=(0.5, float("inf")))
    m.record_synthesis(0.2)
    m.record_synthesis(5.0)
    text = m.render()
    assert 'le="0.5"} 1' in text
    assert 'le="+Inf"} 2' in text

    with pytest.raises(ValueError):
        Metrics(buckets=(0.5,))  # 缺 inf 尾桶
