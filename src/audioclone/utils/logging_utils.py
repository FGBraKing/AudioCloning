"""结构化日志助手 — JSON/文本 formatter、trace_id 上下文透传与统一初始化。

- ``set_trace_id/get_trace_id``：基于 ``contextvars`` 的请求级 trace_id（API 中间件 / WebUI handler 设置，
  服务层日志经 ``TraceIdFilter`` 自动携带）。
- ``configure_logging``：仅接管本模块先前添加的 handler（低侵入），文本或单行 JSON 输出。
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def set_trace_id(trace_id: str | None) -> None:
    """在当前上下文（请求/调用）设置 trace_id。"""
    _trace_id.set(trace_id)


def get_trace_id() -> str | None:
    """读取当前上下文的 trace_id。"""
    return _trace_id.get()


class TraceIdFilter(logging.Filter):
    """把当前上下文 trace_id 注入日志记录（JSON 与文本 formatter 均可消费）。

    仅当记录**未显式携带** ``trace_id`` 时注入（保留 ``extra={"trace_id": ...}`` 的值），
    避免覆盖中间件/调用方显式传入的 trace_id。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "trace_id", None) is None:
            record.trace_id = get_trace_id()
        return True


class JsonFormatter(logging.Formatter):
    """将日志记录格式化为单行 JSON（UTC ISO8601），便于采集/检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", None) or get_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """文本日志；当前上下文存在 trace_id 时追加 ``[tid=...]`` 后缀。"""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        trace_id = getattr(record, "trace_id", None) or get_trace_id()
        return f"{base} [tid={trace_id}]" if trace_id else base


def configure_logging(*, json: bool = False, level: int = logging.INFO) -> None:
    """统一配置根日志：文本（默认）或单行 JSON 输出。

    低侵入：仅移除先前由本模块添加的 handler，不破坏调用方已有的其他 handler。
    """
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, "_audioclone_managed", False)]

    handler = logging.StreamHandler()
    handler._audioclone_managed = True  # type: ignore[attr-defined]
    handler.addFilter(TraceIdFilter())
    if json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            TextFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
    root.addHandler(handler)
    root.setLevel(level)


__all__ = [
    "JsonFormatter",
    "TextFormatter",
    "TraceIdFilter",
    "configure_logging",
    "get_trace_id",
    "set_trace_id",
]
