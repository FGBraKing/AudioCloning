"""极简进程内指标采集（Prometheus text 格式）— v0.2.2 可观测增强。

经 ``/api/metrics`` 暴露：HTTP 请求计数（按 path/status）、合成次数与耗时、
进程运行时长。无需外部依赖；如需接入真实 Prometheus 可替换为此实现。
"""

from __future__ import annotations

import threading
import time


class Metrics:
    """线程安全的计数器集合。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, int], int] = {}   # (path, status) -> count
        self._synth_count: int = 0
        self._synth_seconds: float = 0.0
        self._started = time.time()

    def inc_request(self, path: str, status: int) -> None:
        with self._lock:
            key = (path, status)
            self._requests[key] = self._requests.get(key, 0) + 1

    def record_synthesis(self, seconds: float) -> None:
        with self._lock:
            self._synth_count += 1
            self._synth_seconds += seconds

    def render(self) -> str:
        """渲染为 Prometheus text 格式。"""
        with self._lock:
            lines = [
                "# HELP audioclone_requests_total HTTP 请求计数（path/status 维度）",
                "# TYPE audioclone_requests_total counter",
            ]
            for (path, status), n in sorted(self._requests.items()):
                lines.append(
                    f'audioclone_requests_total{{path="{path}",status="{status}"}} {n}'
                )
            lines += [
                "# TYPE audioclone_synthesis_count_total counter",
                f"audioclone_synthesis_count_total {self._synth_count}",
                "# TYPE audioclone_synthesis_seconds_total counter",
                f"audioclone_synthesis_seconds_total {self._synth_seconds:.4f}",
                "# TYPE audioclone_uptime_seconds gauge",
                f"audioclone_uptime_seconds {time.time() - self._started:.1f}",
            ]
            return "\n".join(lines) + "\n"


# 模块级单例（进程内共享）
metrics = Metrics()

__all__ = ["Metrics", "metrics"]
