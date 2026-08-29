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

# 模型配置（MLX 格式）
ASR_MODEL_ID = "ggml-org/Qwen3-ASR-0.6B-GGUF"
ASR_MODEL_DIR = PROJECT_ROOT / "pretrained_models" / "Qwen3-ASR-0.6B-GGUF"

# Qwen3-TTS（MLX）三个变体：base=参考音频克隆 / custom_voice=预设音色+情绪指令 / voice_design=描述造音色
QWEN3_TTS_MODEL_IDS: dict[str, str] = {
    "base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "custom_voice": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "voice_design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
}
# 目录名用下划线连写（qwen3_tts_<变体>）：mlx_audio 会按 "-" 切分目录名猜测架构，
# 若出现 "qwen3"/"base" 等片段会被误判为其他模型（详见 backends/qwen3_tts_mlx.py）。
QWEN3_TTS_MODEL_DIR = PROJECT_ROOT / "pretrained_models"

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

    # ── 默认后端 ───────────────────────────────────────────
    tts_backend: str = "qwen3-tts-base-mlx"
    asr_backend: str = "qwen3-asr-llama"

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

    # ── ASR ────────────────────────────────────────────────
    asr_server_host: str = "127.0.0.1"
    asr_server_port: int = 8090
    asr_ctx_size: int = 4096
    asr_n_gpu_layers: int = 99             # Metal 全量 offload
    asr_audio_max_sec: int = 300           # 最大音频时长（秒）
    asr_supported_formats: list[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    asr_sample_rate: int = 16000           # Qwen3-ASR 要求的输入采样率
    asr_http_timeout: float = 120.0        # 识别 HTTP 超时（秒）
    asr_ready_timeout: float = 60.0        # llama-server 就绪等待超时（秒）
    asr_max_segment_sec: float = 60.0      # ASR 长音频分片阈值（秒）
    asr_chunk_overlap_sec: float = 1.0     # ASR 分片间重叠（秒），避免切词

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

ASR_SERVER_HOST = settings.asr_server_host
ASR_SERVER_PORT = settings.asr_server_port
ASR_CTX_SIZE = settings.asr_ctx_size
ASR_N_GPU_LAYERS = settings.asr_n_gpu_layers
ASR_AUDIO_MAX_SEC = settings.asr_audio_max_sec
ASR_SUPPORTED_FORMATS = settings.asr_supported_formats
ASR_SAMPLE_RATE = settings.asr_sample_rate
ASR_HTTP_TIMEOUT = settings.asr_http_timeout
ASR_READY_TIMEOUT = settings.asr_ready_timeout
ASR_MAX_SEGMENT_SEC = settings.asr_max_segment_sec
ASR_CHUNK_OVERLAP_SEC = settings.asr_chunk_overlap_sec

MAX_UPLOAD_BYTES = settings.max_upload_bytes
MAX_INLINE_UPLOAD_BYTES = settings.max_inline_upload_bytes
OUTPUT_KEEP_MAX = settings.output_keep_max

TTS_BACKEND = settings.tts_backend
ASR_BACKEND = settings.asr_backend

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
        "model_dir": str(QWEN3_TTS_MODEL_DIR / "qwen3_tts_base"),
    },
    "qwen3-tts-customvoice-mlx": {
        "model_id": QWEN3_TTS_MODEL_IDS["custom_voice"],
        "model_dir": str(QWEN3_TTS_MODEL_DIR / "qwen3_tts_custom_voice"),
    },
    "qwen3-tts-voicedesign-mlx": {
        "model_id": QWEN3_TTS_MODEL_IDS["voice_design"],
        "model_dir": str(QWEN3_TTS_MODEL_DIR / "qwen3_tts_voice_design"),
    },
}

ASR_BACKENDS: dict[str, dict[str, Any]] = {
    "qwen3-asr-llama": {
        "model_id": ASR_MODEL_ID,
        "model_dir": str(ASR_MODEL_DIR),
    },
    "dummy": {
        "sample_rate": settings.sample_rate,
    },
    "dummy-omni": {
        "sample_rate": settings.sample_rate,
    },
}
