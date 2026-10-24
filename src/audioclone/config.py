"""AudioCloning 全局配置。

不产生 import 副作用：本模块不写环境变量、不创建目录。
网络代理等副作用交由应用入口显式设置；输出/日志目录需用时经 ``ensure_*_dir()`` 惰性创建。
运行时可通过环境变量（前缀 ``AUDIOCLONE_``）或 ``.env`` 覆盖配置，见 ``Settings``。
后端工厂的唯一事实源为 ``core/backends/registry.py``；本模块仅承载默认选择与覆盖参数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── 路径配置（非 env 覆盖，属仓库布局） ─────────────────────
# src/audioclone/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 模型配置（MLX / ONNX 格式）
# 默认加载路径 = HuggingFace 系统缓存（~/.cache/huggingface/hub，由 huggingface_hub 管理，
# 缺失时自动下载）；自定义路径经 Settings.model_dir（env/CLI）覆盖，见 utils/model_path.py。
ASR_MLX_MODEL_ID = "mlx-community/Qwen3-ASR-0.6B-8bit"

# Qwen3-TTS（MLX）三个变体：base=参考音频克隆 / custom_voice=预设音色+情绪指令 / voice_design=描述造音色
QWEN3_TTS_MODEL_IDS: dict[str, str] = {
    "base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "custom_voice": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "voice_design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
}

# 输出 / 日志目录
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
ASSET_DIR = PROJECT_ROOT / "asset"


def ensure_output_dir() -> Path:
    """惰性创建输出目录（避免 import 副作用）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def ensure_logs_dir() -> Path:
    """惰性创建日志目录（避免 import 副作用）。"""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


class Settings(BaseSettings):
    """运行时配置，可由 env（前缀 ``AUDIOCLONE_``）或 ``.env`` 覆盖。

    所有可调参数统一收编于此（单源化）；模块级常量由本类派生，保持既有导入面。
    """

    model_config = SettingsConfigDict(env_prefix="AUDIOCLONE_", env_file=".env", extra="ignore")

    # ── 默认后端（"auto" = 按硬件/系统自动选择推理框架，见 core/runtime.py） ──
    tts_backend: str = "auto"
    asr_backend: str = "auto"

    # ── 模型路径（None = HuggingFace 系统缓存 ~/.cache/huggingface/hub） ──
    tts_model_dir: str | None = None       # 自定义 TTS 模型目录（含权重文件的本地路径）
    asr_model_dir: str | None = None       # 自定义 ASR 模型目录（含 .gguf 的本地路径）
    t2a_model_dir: str | None = None       # 自定义 T2A（音乐生成）模型目录
    a2a_model_dir: str | None = None       # 自定义 A2A（语音增强）模型目录
    vad_model_dir: str | None = None       # 自定义 VAD 模型目录
    classification_model_dir: str | None = None  # 自定义分类模型目录
    tensorrt_engine_dir: str | None = None  # TensorRT Edge-LLM 引擎目录（qwen3-tts-*-tensorrt 专用）

    # ── 推理框架选择阈值（core/runtime.py 显存/内存不足时降级 onnx） ──
    min_mlx_mem_gb: float = 8.0            # Apple Silicon 统一内存最低要求（GB）
    min_cuda_vram_gb: float = 8.0          # NVIDIA 可用显存最低要求（GB，多卡取可用最大的一张）

    # ── 模型下载源（默认 HF，失败自动回退 ModelScope） ──────────
    hf_endpoint: str | None = None         # 自定义 HuggingFace 镜像端点（如 https://hf-mirror.com）

    # ── 音频（TTS 输出 / 参考音频校验） ─────────────────────
    sample_rate: int = 24000               # TTS 输出采样率
    ref_audio_max_sec: int = 30            # 参考音频最大时长(秒)
    ref_audio_min_sec: int = 1             # 参考音频最小时长(秒)
    ref_audio_min_sr: int = 8000           # 参考音频最低采样率(Hz)
    max_text_length: int = 300             # 单次合成最大字符数

    # ── WebUI (NiceGUI) ────────────────────────────────────
    webui_port: int = 7860
    webui_host: str = "127.0.0.1"          # 默认仅本机访问（部署护栏）
    webui_concurrency: int = 2             # 同时执行的推理任务数（保护非线程安全模型）

    # ── ASR（通用校验项；各后端自身参数见后端实现） ──────────
    asr_audio_max_sec: int = 300           # 最大音频时长（秒）
    asr_supported_formats: list[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    asr_sample_rate: int = 16000           # 输入采样率（MLX/onnx 后端一致）
    asr_num_threads: int = 2               # onnxruntime 推理线程数（sherpa-onnx）

    # ── 上传 / 资源限流 ────────────────────────────────────
    max_upload_bytes: int = 100 * 1024 * 1024   # 上传音频最大字节数
    max_inline_upload_bytes: int = 8 * 1024 * 1024  # 小文件一次读入内存的阈值（减少分块 IO）

    # ── 输出目录保留上限（<=0 表示不清理） ──────────────────
    output_keep_max: int = 0

    # ── 合成体验（v0.2.2） ─────────────────────────────────
    tts_normalize_loudness: bool = False   # 落盘前按 EBU R128 响度归一化（可选）
    tts_segment_gap_sec: float = 0.15      # 长文本分段间插入静音（秒）

    # ── 日志（v0.2.2） ─────────────────────────────────────
    log_json: bool = False                 # 结构化 JSON 日志


settings = Settings()

# ─── 派生常量（单一来源 = Settings；保持既有导入面不变） ─────
SAMPLE_RATE = settings.sample_rate
REF_AUDIO_MAX_SEC = settings.ref_audio_max_sec
REF_AUDIO_MIN_SEC = settings.ref_audio_min_sec
REF_AUDIO_MIN_SR = settings.ref_audio_min_sr
MAX_TEXT_LENGTH = settings.max_text_length

WEBUI_PORT = settings.webui_port
WEBUI_HOST = settings.webui_host
WEBUI_CONCURRENCY = settings.webui_concurrency

ASR_AUDIO_MAX_SEC = settings.asr_audio_max_sec
ASR_SUPPORTED_FORMATS = settings.asr_supported_formats
ASR_SAMPLE_RATE = settings.asr_sample_rate
ASR_NUM_THREADS = settings.asr_num_threads

MAX_UPLOAD_BYTES = settings.max_upload_bytes
MAX_INLINE_UPLOAD_BYTES = settings.max_inline_upload_bytes
OUTPUT_KEEP_MAX = settings.output_keep_max

TTS_MODEL_DIR = settings.tts_model_dir
ASR_MODEL_DIR = settings.asr_model_dir

TTS_NORMALIZE_LOUDNESS = settings.tts_normalize_loudness
TTS_SEGMENT_GAP_SEC = settings.tts_segment_gap_sec
LOG_JSON = settings.log_json

# API 上传允许的音频扩展名
ALLOWED_UPLOAD_EXT = ASR_SUPPORTED_FORMATS


# ─── 后端默认选择与覆盖参数（工厂唯一来源 = registry） ──────
# 每个键为后端名称（与 config 暴露的 CLI/env 名一致），值为构造覆盖参数。
TTS_BACKENDS: dict[str, dict[str, Any]] = {
    "dummy": {
        "sample_rate": settings.sample_rate,
    },
    "dummy-omni": {
        "sample_rate": settings.sample_rate,
    },
    "qwen3-tts-base-mlx": {
        "model_id": QWEN3_TTS_MODEL_IDS["base"],
        "model_dir": settings.tts_model_dir,   # None = HF 系统缓存
    },
    "qwen3-tts-customvoice-mlx": {
        "model_id": QWEN3_TTS_MODEL_IDS["custom_voice"],
        "model_dir": settings.tts_model_dir,
    },
    "qwen3-tts-voicedesign-mlx": {
        "model_id": QWEN3_TTS_MODEL_IDS["voice_design"],
        "model_dir": settings.tts_model_dir,
    },
    "sherpa-tts-onnx": {
        # sherpa-onnx 官方 HF 镜像：Kokoro-82M int8 多语种（zh/en，82M 参数）
        "model_id": "csukuangfj/kokoro-int8-multi-lang-v1_1",
        "model_dir": settings.tts_model_dir,
    },
    # NVIDIA 平台：TensorRT Edge-LLM 官方 Qwen3-TTS 流水线（三变体同 MLX 侧）。
    # model_dir 指 **Edge-LLM 构建产物目录**（TensorRT 引擎，env
    # AUDIOCLONE_TENSORRT_ENGINE_DIR），权重由用户在 x86 主机自行导出构建，
    # 本项目不下载；未提供时加载报可读指引。
    "qwen3-tts-base-tensorrt": {
        "model_id": QWEN3_TTS_MODEL_IDS["base"],
        "model_dir": settings.tensorrt_engine_dir,
    },
    "qwen3-tts-customvoice-tensorrt": {
        "model_id": QWEN3_TTS_MODEL_IDS["custom_voice"],
        "model_dir": settings.tensorrt_engine_dir,
    },
    "qwen3-tts-voicedesign-tensorrt": {
        "model_id": QWEN3_TTS_MODEL_IDS["voice_design"],
        "model_dir": settings.tensorrt_engine_dir,
    },
    # 情感控制 + 克隆备选（社区口碑标杆；Chatterbox Turbo 标注上游 mlx-audio）
    "chatterbox-mlx": {
        "model_id": "mlx-community/chatterbox-turbo-8bit",
        "model_dir": settings.tts_model_dir,
    },
    # 克隆质量备选；许可 CC BY-NC-SA 4.0（非商用），商用部署前需核实
    "spark-mlx": {
        "model_id": "mlx-community/Spark-TTS-0.5B-8bit",
        "model_dir": settings.tts_model_dir,
    },
}

ASR_BACKENDS: dict[str, dict[str, Any]] = {
    "qwen3-asr-mlx": {
        "model_id": ASR_MLX_MODEL_ID,
        "model_dir": settings.asr_model_dir,   # None = HF 系统缓存（mac 默认后端）
    },
    "sherpa-asr-onnx": {
        # sherpa-onnx 官方 HF 镜像：SenseVoice int8（zh/en/ja/ko/yue）
        "model_id": "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        "model_dir": settings.asr_model_dir,
    },
    "dummy": {
        "sample_rate": settings.sample_rate,
    },
    "dummy-omni": {
        "sample_rate": settings.sample_rate,
    },
}

# ─── 其余任务后端（v0.3.0：T2A / A2A / 分类 / VAD） ──────────
# 模型选型原则：默认模型仅选 **8bit 量化（或非 LLM 极小模型）、较新较强**的小模型，
# 参数量以 ≤3B 为基线（T2A 音乐经用户决策放宽）；优先 mlx_audio 源码内置仓库或
# 已确认存在的 mlx-community 仓库，不引入 torch 等重依赖；**默认不下载权重**（首启提示 + 主动下载接口）。
#
# A2A 变声无 MLX 原生模型（DeepFilterNet 为增强；VoiceDesign 可经 TTS 变通）。
T2A_BACKENDS: dict[str, dict[str, Any]] = {
    # 放宽参数量上限的例外：mlx-community 唯一有 8bit 量化版的音乐生成模型（基座 11.7B）
    "minimax-music-mlx": {
        "model_id": "mlx-community/MiniMax-Music3-8bit",
        "model_dir": settings.t2a_model_dir,
    },
    "dummy-omni": {
        "sample_rate": settings.sample_rate,
    },
}

A2A_BACKENDS: dict[str, dict[str, Any]] = {
    "deepfilternet-mlx": {
        "model_id": "mlx-community/DeepFilterNet-mlx",   # mlx_audio 源码内置仓库 ID
        "model_dir": settings.a2a_model_dir,
    },
    # 语音增强（非 Mac 平台默认）：GTCRN ~236K，GitHub Release 分发（不在 HF）
    "sherpa-denoise-onnx": {
        "model_id": None,
        "model_dir": settings.a2a_model_dir,
        "github_release": {
            "repo": "k2-fsa/sherpa-onnx",
            "tag": "speech-enhancement-models",
            "asset": "gtcrn_simple.onnx",
        },
    },
    "dummy-omni": {
        "sample_rate": settings.sample_rate,
    },
}

CLASSIFICATION_BACKENDS: dict[str, dict[str, Any]] = {
    # 情感+事件（用户决策）：SenseVoice 内建分类头（mac 默认）
    "sensevoice-mlx": {
        "model_id": "mlx-community/SenseVoiceSmall",
        "model_dir": settings.classification_model_dir,
    },
    # 通用 AudioSet 527 类事件分类（k2-fsa 官方 HF 仓库，CED-Small ~92M）
    "sherpa-cls-onnx": {
        "model_id": "k2-fsa/sherpa-onnx-ced-small-audio-tagging-2024-04-19",
        "model_dir": settings.classification_model_dir,
    },
    "dummy-omni": {
        "sample_rate": settings.sample_rate,
    },
}

VAD_BACKENDS: dict[str, dict[str, Any]] = {
    "silero-vad-mlx": {
        "model_id": "mlx-community/silero-vad-v6",   # 已确认存在，MIT，约 2MB
        "model_dir": settings.vad_model_dir,
    },
    "sherpa-vad-onnx": {
        # onnx-community 官方转换（47 赞；sherpa-onnx VAD 原生消费 silero onnx）
        "model_id": "onnx-community/silero-vad",
        "model_dir": settings.vad_model_dir,
    },
    "dummy-omni": {
        "sample_rate": settings.sample_rate,
    },
}
