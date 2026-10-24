"""runtime 运行环境探测与后端自动选择单元测试（全部 monkeypatch，不依赖真实硬件假设）。"""

from __future__ import annotations

import pytest

from audioclone.core import runtime
from audioclone.core.runtime import (
    RuntimeInfo,
    auto_backend_candidates,
    detect_runtime,
)
from audioclone.core.service.model_loader import resolve_backend_name

#: 模块导入时捕获真实实现（autouse fixture 会打桩模块属性）
_REAL_GPU_QUERY = runtime._query_gpu_memory_gb


@pytest.fixture(autouse=True)
def _fresh_runtime_cache():
    """每个用例重置探测缓存，避免跨用例污染。"""
    global_detect = runtime._cached
    runtime._cached = None
    yield
    runtime._cached = global_detect


@pytest.fixture(autouse=True)
def _stub_capacity(monkeypatch):
    """默认打桩容量查询（16GB 内存 / 24GB 显存其中 22GB 可用），保证测试确定性不跑真实 sysctl/nvidia-smi。"""
    monkeypatch.setattr(runtime, "_query_mem_total_gb", lambda: 16.0)
    monkeypatch.setattr(runtime, "_query_gpu_memory_gb", lambda: (24.0, 22.0))


def _patch_env(monkeypatch, os_name: str, machine: str, which: dict[str, str | None] | None = None):
    monkeypatch.setattr(runtime.platform, "system", lambda: os_name)
    monkeypatch.setattr(runtime.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        runtime.shutil, "which", lambda cmd: (which or {}).get(cmd)
    )
    monkeypatch.setattr(runtime.os.path, "exists", lambda p: False)


# ─── 探测：环境 → 推理框架 ───────────────────────────────────
@pytest.mark.parametrize(
    ("os_name", "machine", "which", "expected_framework"),
    [
        ("Darwin", "arm64", {}, "mlx"),                       # Apple Silicon → MLX
        ("Darwin", "x86_64", {}, "onnx"),                     # Intel Mac 无 MLX → onnx
        ("Linux", "x86_64", {"nvidia-smi": "/usr/bin/nvidia-smi"}, "tensorrt-llm"),
        ("Linux", "x86_64", {"rocminfo": "/usr/bin/rocminfo"}, "onnx"),   # AMD 全系 → onnx
        ("Linux", "x86_64", {}, "onnx"),                      # 无 GPU → onnx
        ("Windows", "AMD64", {"nvidia-smi": "C:/nvidia-smi.exe"}, "tensorrt-llm"),
    ],
)
def test_detect_runtime_framework_mapping(monkeypatch, os_name, machine, which, expected_framework):
    _patch_env(monkeypatch, os_name, machine, which)
    info = detect_runtime()
    assert isinstance(info, RuntimeInfo)
    assert info.framework == expected_framework


def test_detect_runtime_nvidia_driver_fallback(monkeypatch):
    """nvidia-smi 不在 PATH 但有 NVML 驱动文件 → 仍判定 CUDA。"""
    _patch_env(monkeypatch, "Linux", "x86_64")
    monkeypatch.setattr(runtime.os.path, "exists", lambda p: p == "/proc/driver/nvidia/version")
    assert detect_runtime().accelerator == "cuda"


def test_detect_runtime_cached(monkeypatch, caplog):
    """探测结果进程内缓存：第二次调用不再重复判定。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    first = detect_runtime()
    _patch_env(monkeypatch, "Linux", "x86_64")  # 环境突变也不影响缓存
    assert detect_runtime() is first
    assert detect_runtime(force=True).framework == "onnx"


# ─── 候选链：框架 → 候选 + 兜底 ──────────────────────────────
def test_candidates_mlx_macos(monkeypatch):
    _patch_env(monkeypatch, "Darwin", "arm64")
    assert auto_backend_candidates("tts") == [
        "qwen3-tts-base-mlx", "chatterbox-mlx", "spark-mlx", "dummy",
    ]
    assert auto_backend_candidates("asr") == ["qwen3-asr-mlx", "dummy"]


def test_candidates_cpu_only(monkeypatch):
    """无 GPU（onnx 框架）：TTS/ASR/VAD 有 sherpa 候选；音乐/增强/分类回退 dummy。"""
    _patch_env(monkeypatch, "Linux", "x86_64")
    assert auto_backend_candidates("tts") == ["sherpa-tts-onnx", "dummy"]
    assert auto_backend_candidates("asr") == ["sherpa-asr-onnx", "dummy"]
    assert auto_backend_candidates("vad") == ["sherpa-vad-onnx", "dummy-omni"]
    assert auto_backend_candidates("t2a") == ["dummy-omni"]
    assert auto_backend_candidates("a2a") == ["sherpa-denoise-onnx", "dummy-omni"]
    assert auto_backend_candidates("classification") == ["sherpa-cls-onnx", "dummy-omni"]


def test_candidates_nvidia_tts_uses_edge_llm_backend(monkeypatch):
    """NVIDIA 平台 TTS：TensorRT Edge-LLM Qwen3-TTS 后端已接入（引擎需用户构建）。"""
    _patch_env(monkeypatch, "Linux", "x86_64", {"nvidia-smi": "/usr/bin/nvidia-smi"})
    candidates = auto_backend_candidates("tts")
    assert candidates[0] == "qwen3-tts-base-tensorrt"  # 框架专属候选在前
    assert candidates[-1] == "dummy"                  # 兜底在后
    # ASR/其余任务走 sherpa-onnx（onnxruntime CUDA EP，GPU 加速）
    assert auto_backend_candidates("asr") == ["sherpa-asr-onnx", "dummy"]
    assert auto_backend_candidates("classification") == ["sherpa-cls-onnx", "dummy-omni"]


def test_candidates_unknown_task(monkeypatch):
    _patch_env(monkeypatch, "Darwin", "arm64")
    assert auto_backend_candidates("translate") == ["dummy"]


# ─── 容量判断：显存/内存不足降级 onnx ────────────────────────
def test_mlx_downgrades_on_low_unified_memory(monkeypatch):
    """Apple Silicon 但统一内存 < 8GB → 降级 onnx 并记录原因。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    monkeypatch.setattr(runtime, "_query_mem_total_gb", lambda: 4.0)
    info = detect_runtime()
    assert info.downgraded is True
    assert info.framework == "onnx"
    assert "统一内存" in info.downgrade_reason
    # 降级后候选链与 CPU 环境一致（sherpa TTS）
    assert auto_backend_candidates("tts") == ["sherpa-tts-onnx", "dummy"]


def test_mlx_kept_at_threshold_boundary(monkeypatch):
    """内存恰等于阈值（8GB）→ 不降级（>= 阈值即可用）。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    monkeypatch.setattr(runtime, "_query_mem_total_gb", lambda: 8.0)
    info = detect_runtime()
    assert info.downgraded is False
    assert info.framework == "mlx"
    assert info.mem_total_gb == 8.0


def test_cuda_downgrades_on_low_vram(monkeypatch):
    """NVIDIA 但可用显存 < 8GB（总量充足被其他进程占用）→ 降级 onnx 并记录原因。"""
    _patch_env(monkeypatch, "Linux", "x86_64", {"nvidia-smi": "/usr/bin/nvidia-smi"})
    monkeypatch.setattr(runtime, "_query_gpu_memory_gb", lambda: (24.0, 4.0))
    info = detect_runtime()
    assert info.downgraded is True
    assert info.framework == "onnx"
    assert "可用显存" in info.downgrade_reason
    assert info.gpu_vram_gb == 24.0       # 总量充足但被占用 → 仍降级
    assert info.gpu_vram_free_gb == 4.0


def test_cuda_multi_gpu_picks_most_free(monkeypatch):
    """多卡：取可用显存最大的一张做决策（模型只需放得进一张卡）。"""
    _REAL_QUERY = _REAL_GPU_QUERY  # autouse 打桩前的真实实现
    _patch_env(monkeypatch, "Linux", "x86_64", {"nvidia-smi": "/usr/bin/nvidia-smi"})
    monkeypatch.setattr(runtime, "_query_gpu_memory_gb", _REAL_QUERY)
    fake = runtime.subprocess.CompletedProcess(args=[], returncode=0)
    fake.stdout = "24576, 20480\n8188, 100\n"  # 卡0 24G/20G用(余4G)，卡1 8G/0.1G用(余7.9G)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw: fake)
    total, free = runtime._query_gpu_memory_gb()
    assert (total, free) == (8188 / 1024, 8088 / 1024)  # 卡1 可用更多 → 选卡1


def test_cuda_vram_free_boundary(monkeypatch):
    """可用显存恰等于阈值 → 不降级。"""
    _patch_env(monkeypatch, "Linux", "x86_64", {"nvidia-smi": "/usr/bin/nvidia-smi"})
    monkeypatch.setattr(runtime, "_query_gpu_memory_gb", lambda: (24.0, 8.0))
    info = detect_runtime()
    assert info.downgraded is False
    assert info.framework == "tensorrt-llm"


def test_capacity_query_failure_does_not_downgrade(monkeypatch):
    """容量查询失败（None）→ 不降级，避免误伤探测不到读数的环境。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    monkeypatch.setattr(runtime, "_query_mem_total_gb", lambda: None)
    info = detect_runtime()
    assert info.downgraded is False
    assert info.framework == "mlx"
    assert info.mem_total_gb is None


def test_thresholds_env_overridable(monkeypatch):
    """阈值可经 Settings（env AUDIOCLONE_MIN_MLX_MEM_GB）调整。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    monkeypatch.setattr(runtime, "_query_mem_total_gb", lambda: 6.0)
    monkeypatch.setattr(runtime.settings, "min_mlx_mem_gb", 4.0)
    assert detect_runtime().framework == "mlx"
    runtime._cached = None
    monkeypatch.setattr(runtime.settings, "min_mlx_mem_gb", 8.0)
    assert detect_runtime().downgraded is True


def test_gpu_vram_only_queried_for_gpu_accelerators(monkeypatch):
    """无 GPU 环境不发起显存查询。"""
    _patch_env(monkeypatch, "Linux", "x86_64")
    called = []
    monkeypatch.setattr(
        runtime, "_query_gpu_memory_gb", lambda: called.append(1) or (24.0, 22.0)
    )
    info = detect_runtime()
    assert called == []          # accelerator=none → 未查询
    assert info.gpu_vram_gb is None
    runtime._cached = None
    _patch_env(monkeypatch, "Linux", "x86_64", {"nvidia-smi": "/usr/bin/nvidia-smi"})
    detect_runtime()
    assert called == [1]         # cuda → 查询


# ─── model_loader：auto 解析 ─────────────────────────────────
def test_resolve_explicit_name_wins(monkeypatch):
    _patch_env(monkeypatch, "Linux", "x86_64")
    assert resolve_backend_name("tts", "qwen3-tts-base-mlx") == "qwen3-tts-base-mlx"
    assert resolve_backend_name("tts", "dummy") == "dummy"


def test_resolve_auto_on_macos(monkeypatch):
    _patch_env(monkeypatch, "Darwin", "arm64")
    assert resolve_backend_name("tts", None) == "qwen3-tts-base-mlx"
    assert resolve_backend_name("asr", "auto") == "qwen3-asr-mlx"  # mac 默认 MLX ASR


def test_resolve_auto_cpu_tts_uses_sherpa(monkeypatch):
    """无 GPU 环境（onnx 框架）TTS 自动选 sherpa-tts-onnx。"""
    _patch_env(monkeypatch, "Linux", "x86_64")
    assert resolve_backend_name("tts", None) == "sherpa-tts-onnx"


def test_resolve_auto_all_candidates_unregistered(monkeypatch):
    """候选链全部未注册（极端场景）→ KeyError。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    monkeypatch.setattr(runtime, "FALLBACK_BACKENDS", {"tts": ["nope"], "asr": ["nope"]})
    monkeypatch.setattr(
        runtime, "FRAMEWORK_BACKEND_CANDIDATES", {"mlx": {"tts": ["nope"], "asr": ["nope"]}}
    )
    with pytest.raises(KeyError):
        resolve_backend_name("tts", None)


def test_create_tts_backend_auto_uses_detected_framework(monkeypatch):
    """端到端：mac 环境下 create_tts_backend() 默认得到 MLX 后端。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    from audioclone.core import create_tts_backend
    from audioclone.core.backends.qwen3_tts_mlx import Qwen3TTSBaseMLXBackend

    backend = create_tts_backend()
    assert isinstance(backend, Qwen3TTSBaseMLXBackend)


# ─── 可用性感知兜底（真实后端间自动切换，绝不触发下载） ──────
@pytest.fixture
def _no_models_cached(monkeypatch):
    """默认：所有模型均未缓存（离线检查打桩）。"""
    from audioclone.core.service import model_loader

    monkeypatch.setattr(model_loader, "find_cached_model", lambda model_id: None)
    monkeypatch.setattr(model_loader, "find_modelscope_model", lambda model_id: None)


def test_fallback_prefers_first_real_even_if_not_cached(monkeypatch, _no_models_cached):
    """全未就位：保持框架首选真实后端（加载时报可读下载指引，不静默退化 dummy）。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    assert resolve_backend_name("tts", None) == "qwen3-tts-base-mlx"
    assert resolve_backend_name("asr", None) == "qwen3-asr-mlx"


def test_fallback_within_framework_when_model_cached(monkeypatch, _no_models_cached):
    """onnx 框架 ASR：sherpa 模型已缓存 → 选中；未缓存 → 保持首选（不静默退化）。"""
    _patch_env(monkeypatch, "Linux", "x86_64")
    from audioclone.core.service import model_loader

    monkeypatch.setattr(model_loader, "find_cached_model", lambda model_id: "/cache")
    assert resolve_backend_name("asr", None) == "sherpa-asr-onnx"


def test_fallback_prefers_cached_preferred_backend(monkeypatch, _no_models_cached):
    """首选已就位时不降级：MLX ASR 模型已缓存 → 保持 qwen3-asr-mlx。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    from audioclone.core.service import model_loader

    monkeypatch.setattr(model_loader, "find_cached_model", lambda model_id: "/cache")
    assert resolve_backend_name("asr", None) == "qwen3-asr-mlx"


def test_fallback_custom_dir_counts_as_available(monkeypatch, _no_models_cached, tmp_path):
    """spec 中的自定义 model_dir 存在即视为就位。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    from audioclone.core.service import model_loader

    monkeypatch.setattr(
        "audioclone.config.ASR_BACKENDS",
        {**model_loader.ASR_BACKENDS,
         "qwen3-asr-mlx": {"model_id": "mlx-community/Qwen3-ASR-0.6B-8bit",
                            "model_dir": str(tmp_path)}},
    )
    monkeypatch.setattr(model_loader, "ASR_BACKENDS",
                        {**model_loader.ASR_BACKENDS,
                         "qwen3-asr-mlx": {"model_id": "mlx-community/Qwen3-ASR-0.6B-8bit",
                                            "model_dir": str(tmp_path)}})
    assert resolve_backend_name("asr", None) == "qwen3-asr-mlx"


def test_fallback_dummy_only_when_no_real_registered(monkeypatch, _no_models_cached):
    """框架候选无真实后端（如 onnx 的 T2A）→ 兜底 dummy（功能受限告警）。"""
    _patch_env(monkeypatch, "Linux", "x86_64")
    assert resolve_backend_name("t2a", None) == "dummy-omni"


def test_fallback_never_downloads(monkeypatch, _no_models_cached):
    """选择过程只做离线检查：snapshot_download 绝不被调用（含网络路径）。"""
    _patch_env(monkeypatch, "Darwin", "arm64")
    import huggingface_hub

    def boom(*args, **kwargs):
        raise AssertionError("选择阶段不得访问网络")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    resolve_backend_name("asr", None)  # 不抛 AssertionError 即通过


# ─── AMD 核显平台：映射 onnx ─────────────────────────────────
def test_amd_gpu_maps_to_onnx(monkeypatch):
    """ROCm 在位（核显/独显均同）→ 推理框架 onnx（vLLM 不支持音频场景，经决策排除）。"""
    _patch_env(monkeypatch, "Linux", "x86_64", {"rocminfo": "/usr/bin/rocminfo"})
    info = detect_runtime()
    assert info.framework == "onnx"
    assert info.accelerator == "rocm"
    # ASR 走 onnx 专属候选链
    assert auto_backend_candidates("asr")[0] == "sherpa-asr-onnx"


