"""结构化日志助手 — JSON formatter 与统一初始化（v0.2.2 可观测增强）。

入口通过 ``configure_logging(json=LOG_JSON)`` 统一配置根日志；
JSON 模式下每条记录为单行 JSON（含 ts/level/logger/message/trace_id/exc_info）。
"""

from __future__ import annotations

import json
import logging


class JsonFormatter(logging.Formatter):
    """将日志记录格式化为单行 JSON，便于采集/检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = getattr(record, "trace_id", None)
        if trace_id:
            payload["trace_id"] = trace_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, json: bool = False, level: int = logging.INFO) -> None:
    """统一配置根日志：文本（默认）或单行 JSON 输出。"""
    handler = logging.StreamHandler()
    if json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


__all__ = ["JsonFormatter", "configure_logging"]
