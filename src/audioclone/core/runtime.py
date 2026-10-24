"""运行时环境探测 — 按硬件与系统选择后端推理框架。

选择规则（首次探测后缓存，显存/内存不足时降级）：

=================  ===============  ==============================================
环境               推理框架          说明
=================  ===============  ==============================================
macOS (Apple Silicon)  mlx          M 系列芯片统一内存，MLX 原生加速；
                                    统一内存 < AUDIOCLONE_MIN_MLX_MEM_GB（默认 8GB）→ 降级
NVIDIA GPU         tensorrt-llm     ``nvidia-smi`` / NVML 驱动在位即判定；
                                    可用显存（总量−其他进程占用）< AUDIOCLONE_MIN_CUDA_VRAM_GB
                                    （默认 8GB）→ 降级
AMD GPU (ROCm)    onnx             vLLM 不支持音频场景（经调研决策排除）；
                                    sherpa-onnx 音频专用推理，核显/独显统一走 onnx
无 GPU（纯 CPU）   onnx             sherpa-onnx（onnxruntime），音频专用全家桶，依赖轻
=================  ===============  ==============================================

降级统一落到 onnx（sherpa-onnx 音频专用推理，CPU/核显皆可跑），``downgrade_reason`` 记录原因；
探测结果与容量信息经 ``/api/health`` 的 ``runtime`` 字段暴露。查询失败（拿不到
容量读数）时**不做降级**——避免误伤探测不到容量的正常环境。

框架仅决定**候选后端链**：``model_loader`` 按链取第一个已注册的后端；链尾回退
（各任务真实后端不可用时→dummy），并在日志中说明降级原因。显式指定后端（CLI/env）
时跳过探测。
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass

from audioclone.config import settings

logger = logging.getLogger(__name__)

#: 加速器类型（探测产物）
ACCELERATORS = ("mlx", "cuda", "rocm", "none")

#: 加速器 → 推理框架
ACCELERATOR_FRAMEWORK: dict[str, str] = {
    "mlx": "mlx",
    "cuda": "tensorrt-llm",  # TTS：Edge-LLM 官方 Qwen3-TTS 流水线（引擎用户构建）
    "rocm": "onnx",          # AMD 全系：vLLM 不支持音频场景（经调研决策排除）
    "none": "onnx",
}

#: 推理框架 → 各任务的候选后端链（按优先级；未注册的候选自动跳过，
#: 为 TensorRT-LLM / vLLM 后端预留命名位，接入后无需改动本模块）
FRAMEWORK_BACKEND_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "mlx": {
        "tts": ["qwen3-tts-base-mlx", "chatterbox-mlx", "spark-mlx"],
        "asr": ["qwen3-asr-mlx"],
        "t2a": ["minimax-music-mlx"],  # 参数量放宽的例外（唯一有 8bit 的音乐生成）
        "a2a": ["deepfilternet-mlx"],
        "classification": ["sensevoice-mlx"],  # mlx_audio 原生情感+事件分类
        "vad": ["silero-vad-mlx"],
    },
    # 说明：onnx 框架（AMD/无 GPU）的 a2a/classification 候选在本文件下方 onnx 键中
    "tensorrt-llm": {
        # TTS：Edge-LLM 官方 Qwen3-TTS 三变体流水线（引擎由用户构建）；
        # 其余任务走 sherpa-onnx（onnxruntime CUDA EP，GPU 加速）
        "tts": ["qwen3-tts-base-tensorrt"],
        "asr": ["sherpa-asr-onnx"],
        "a2a": ["sherpa-denoise-onnx"],
        "classification": ["sherpa-cls-onnx"],
        "vad": ["sherpa-vad-onnx"],
    },
    "onnx": {
        # sherpa-onnx（onnxruntime）：音频专用推理全家桶，int8 模型 CPU 佳
        "tts": ["sherpa-tts-onnx"],    # Kokoro-82M int8（纯文本合成，plain 模式）
        "asr": ["sherpa-asr-onnx"],    # SenseVoice int8
        "t2a": [],                     # 音乐生成无 onnx 等价物 → 兜底 dummy
        "a2a": ["sherpa-denoise-onnx"],      # GTCRN（GitHub Release 分发）
        "classification": ["sherpa-cls-onnx"],  # CED（AudioSet 527 类）
        "vad": ["sherpa-vad-onnx"],    # Silero VAD onnx
    },
}

#: 候选链耗尽后的兜底后端（保证任何环境都能起服务；单源，model_loader 同步引用）
FALLBACK_BACKENDS: dict[str, list[str]] = {
    "tts": ["dummy"],
    "asr": ["dummy"],
    "t2a": ["dummy-omni"],
    "a2a": ["dummy-omni"],
    "classification": ["dummy-omni"],
    "vad": ["dummy-omni"],
}


@dataclass(frozen=True)
class RuntimeInfo:
    """探测结果：系统、加速器、推理框架、内存容量与降级信息。"""

    os: str                     # darwin / linux / windows ...
    machine: str                # arm64 / x86_64 ...
    accelerator: str            # mlx / cuda / rocm / none
    framework: str              # mlx / tensorrt-llm / onnx
    mem_total_gb: float | None = None    # 系统内存（Apple Silicon 为统一内存）
    gpu_vram_gb: float | None = None     # GPU 显存总量（多卡取可用最大的一张；仅 NVIDIA）
    gpu_vram_free_gb: float | None = None  # GPU 可用显存（扣除其他进程占用；仅 NVIDIA）
    downgraded: bool = False             # 容量不足是否触发降级
    downgrade_reason: str | None = None  # 降级原因（downgraded 时非空）

    def as_dict(self) -> dict[str, object]:
        return {
            "os": self.os, "machine": self.machine,
            "accelerator": self.accelerator, "framework": self.framework,
            "mem_total_gb": self.mem_total_gb,
            "gpu_vram_gb": self.gpu_vram_gb, "gpu_vram_free_gb": self.gpu_vram_free_gb,
            "downgraded": self.downgraded, "downgrade_reason": self.downgrade_reason,
        }


# ─── 容量查询（失败一律返回 None，不做降级） ─────────────────
def _query_gpu_memory_gb() -> tuple[float | None, float | None]:
    """``nvidia-smi`` 查询显存，返回 (总量, 可用量)，单位 GiB。

    多卡时取**可用显存最大**的一张（模型只需放得进一张卡）；查询失败返回 (None, None)。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        gpus: list[tuple[float, float]] = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) == 2 and all(p.replace(".", "", 1).isdigit() for p in parts):
                total_mib, used_mib = (float(p) for p in parts)
                gpus.append((total_mib, used_mib))
        if not gpus:
            return None, None
        total_mib, used_mib = max(gpus, key=lambda t: t[0] - t[1])
        return total_mib / 1024, (total_mib - used_mib) / 1024
    except Exception:  # noqa: BLE001 - 查询不可用视为未知，不参与决策
        return None, None


def _query_mem_total_gb() -> float | None:
    """查询系统内存总量（GiB）；Apple Silicon 即统一内存。"""
    os_name = platform.system().lower()
    try:
        if os_name == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=False
            )
            return int(out.stdout.strip()) / 1024**3
        if os_name == "linux":
            for line in _read_meminfo_lines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2  # kB → GiB
            return None
        if os_name == "windows":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return stat.ullTotalPhys / 1024**3
            return None
    except Exception:  # noqa: BLE001
        return None
    return None


def _read_meminfo_lines() -> list[str]:
    """读取 /proc/meminfo 行（独立小函数便于测试打桩）。"""
    with open("/proc/meminfo", encoding="ascii") as f:
        return f.readlines()


# ─── 探测主流程 ───────────────────────────────────────────────
def _detect_accelerator(os_name: str, machine: str) -> str:
    """探测加速器类型（只做存在性检查，不 import 重依赖）。"""
    if os_name == "darwin":
        # MLX 仅支持 Apple Silicon；Intel Mac 无 MLX，按无 GPU 处理
        return "mlx" if machine == "arm64" else "none"
    if shutil.which("nvidia-smi") or os.path.exists("/proc/driver/nvidia/version"):
        return "cuda"
    if shutil.which("rocminfo"):
        return "rocm"
    return "none"


_cached: RuntimeInfo | None = None


def detect_runtime(force: bool = False) -> RuntimeInfo:
    """探测当前运行环境（结果进程内缓存；``force=True`` 重新探测）。

    显存/内存不足时降级到 onnx（阈值见模块 docstring；查询失败不降级）。
    AMD **核显**平台直接映射 onnx（ROCm/vLLM 不适用核显；AMD 独显仍走 vLLM）。
    """
    global _cached
    if _cached is not None and not force:
        return _cached
    os_name = platform.system().lower()
    machine = platform.machine().lower()
    accelerator = _detect_accelerator(os_name, machine)
    framework = ACCELERATOR_FRAMEWORK.get(accelerator, "onnx")

    mem_total = _query_mem_total_gb()
    gpu_total, gpu_free = (None, None)
    if accelerator == "cuda":
        gpu_total, gpu_free = _query_gpu_memory_gb()

    downgraded, reason = False, None
    if accelerator == "mlx" and mem_total is not None and mem_total < settings.min_mlx_mem_gb:
        downgraded, reason = True, (
            f"统一内存 {mem_total:.1f}GB 低于 MLX 最低要求 {settings.min_mlx_mem_gb:.0f}GB，"
            "降级到 onnx"
        )
    elif accelerator == "cuda" and gpu_free is not None and gpu_free < settings.min_cuda_vram_gb:
        downgraded, reason = True, (
            f"GPU 可用显存 {gpu_free:.1f}GB（总量 {gpu_total:.1f}GB，含其他进程占用）"
            f"低于 TensorRT-LLM 最低要求 {settings.min_cuda_vram_gb:.0f}GB，降级到 onnx"
        )
    if downgraded:
        framework = "onnx"
        logger.warning("[runtime] %s", reason)

    _cached = RuntimeInfo(
        os=os_name, machine=machine, accelerator=accelerator, framework=framework,
        mem_total_gb=mem_total, gpu_vram_gb=gpu_total, gpu_vram_free_gb=gpu_free,
        downgraded=downgraded, downgrade_reason=reason,
    )
    logger.info(
        "[runtime] 探测到运行环境: os=%s machine=%s accelerator=%s → 推理框架=%s "
        "(mem=%s, vram=%s)",
        os_name, machine, accelerator, framework,
        f"{mem_total:.1f}GB" if mem_total else "未知",
        f"{gpu_free:.1f}GB 可用 / {gpu_total:.1f}GB" if gpu_total else "未知",
    )
    return _cached


def auto_backend_candidates(task: str) -> list[str]:
    """按当前环境返回某任务的候选后端链（框架专属候选 + 兜底链）。"""
    runtime = detect_runtime()
    specific = FRAMEWORK_BACKEND_CANDIDATES.get(runtime.framework, {}).get(task, [])
    fallback = FALLBACK_BACKENDS.get(task, ["dummy"])
    # 去重保序：框架候选在前，兜底在后
    seen: list[str] = []
    for name in [*specific, *fallback]:
        if name not in seen:
            seen.append(name)
    return seen
