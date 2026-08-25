"""极简进程内指标采集（Prometheus text 格式）— 可观测增强。

经 ``/api/metrics`` 暴露：HTTP 请求计数（按 path/status）、合成次数/耗时与直方图、
进程运行时长。无需外部依赖；如需接入真实 Prometheus 可替换为此实现。
注意：多进程部署下各进程独立计数（进程内单例）。
"""

from __future__ import annotations

import threading
import time

# 合成耗时直方图 buckets（秒）
SYNTH_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf"))


class Metrics:
    """线程安全的计数器/直方图集合。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, int], int] = {}   # (path, status) -> count
        self._synth_count: int = 0
        self._synth_seconds: float = 0.0
        self._synth_hist: list[int] = [0] * len(SYNTH_BUCKETS)
        self._started = time.time()

    def inc_request(self, path: str, status: int) -> None:
        with self._lock:
            key = (path, status)
            self._requests[key] = self._requests.get(key, 0) + 1

    def record_synthesis(self, seconds: float) -> None:
        with self._lock:
            self._synth_count += 1
            self._synth_seconds += seconds
            # Prometheus 直方图为累积桶：落入某桶即计入其及所有更高上界桶
            for i, upper in enumerate(SYNTH_BUCKETS):
                if seconds <= upper:
                    for j in range(i, len(self._synth_hist)):
                        self._synth_hist[j] += 1
                    break

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
                "# TYPE audioclone_synthesis_duration_seconds histogram",
            ]
            for upper, n in zip(SYNTH_BUCKETS, self._synth_hist, strict=True):
                le = "+Inf" if upper == float("inf") else f"{upper:g}"
                lines.append(
                    f'audioclone_synthesis_duration_seconds_bucket{{le="{le}"}} {n}'
                )
            lines += [
                f"audioclone_synthesis_duration_seconds_sum {self._synth_seconds:.4f}",
                f"audioclone_synthesis_duration_seconds_count {self._synth_count}",
                "# TYPE audioclone_uptime_seconds gauge",
                f"audioclone_uptime_seconds {time.time() - self._started:.1f}",
            ]
            return "\n".join(lines) + "\n"


# 模块级单例（进程内共享）
metrics = Metrics()

__all__ = ["Metrics", "SYNTH_BUCKETS", "metrics"]
