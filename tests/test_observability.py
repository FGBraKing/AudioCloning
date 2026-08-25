"""可观测性组件测试 — 结构化日志 formatter 与进程内指标采集。"""

from __future__ import annotations

import json
import logging

from audioclone.api.metrics import Metrics
from audioclone.utils.logging_utils import JsonFormatter, configure_logging


def test_json_formatter_outputs_valid_json():
    record = logging.LogRecord(
        name="audioclone.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.trace_id = "abc123"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "audioclone.api"
    assert payload["message"] == "hello world"
    assert payload["trace_id"] == "abc123"
    assert "ts" in payload


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
