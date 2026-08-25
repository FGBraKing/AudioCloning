"""AudioCloning 全局配置。

不产生 import 副作用：本模块不写环境变量、不创建目录。
网络代理等副作用交由应用入口显式设置；输出目录需用时经 ``ensure_output_dir()`` 惰性创建。
运行时可通过环境变量（前缀 ``AUDIOCLONE_``）覆盖部分配置，见 ``Settings``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── 路径配置 ───────────────────────────────────────────────
# src/audioclone/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 模型配置（MLX 格式）
MODEL_ID = "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16"
MODEL_DIR = PROJECT_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B-2512-fp16"

# ─── 输出配置 ───────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def ensure_output_dir() -> Path:
    """惰性创建输出目录（避免 import 副作用）。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ─── 音频配置 ───────────────────────────────────────────────
SAMPLE_RATE = 24000                    # CosyVoice3 输出采样率
REF_AUDIO_MAX_SEC = 30                 # 参考音频最大时长(秒)
REF_AUDIO_MIN_SEC = 1                  # 参考音频最小时长(秒)
REF_AUDIO_MIN_SR = 8000                # 参考音频最低采样率(Hz)

# ─── 文本配置 ───────────────────────────────────────────────
MAX_TEXT_LENGTH = 300                  # 单次合成最大字符数

# ─── WebUI 配置 ─────────────────────────────────────────────
GRADIO_PORT = 7860
GRADIO_HOST = "0.0.0.0"
GRADIO_MAX_QUEUE = 4
GRADIO_CONCURRENCY = 2

# ─── 资源目录 ───────────────────────────────────────────────
ASSET_DIR = PROJECT_ROOT / "asset"

# ─── ASR 配置 ─────────────────────────────────────────────
ASR_MODEL_ID = "ggml-org/Qwen3-ASR-0.6B-GGUF"
ASR_MODEL_DIR = PROJECT_ROOT / "pretrained_models" / "Qwen3-ASR-0.6B-GGUF"
ASR_SERVER_PORT = 8090
ASR_SERVER_HOST = "127.0.0.1"
ASR_CTX_SIZE = 4096
ASR_N_GPU_LAYERS = 99           # Metal 全量 offload
ASR_AUDIO_MAX_SEC = 300         # 最大音频时长（秒）
ASR_SUPPORTED_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
ASR_SAMPLE_RATE = 16000          # Qwen3-ASR 要求的输入采样率

# ─── 后端注册表（模型可替换） ───────────────────────────────
# 每个后端: factory=加载路径(module.Class)，其余键作为构造参数。
TTS_BACKENDS: dict[str, dict[str, Any]] = {
    "cosyvoice3-mlx": {
        "factory": "audioclone.core.backends.cosyvoice_mlx.CosyVoiceMLXBackend",
        "model_id": MODEL_ID,
        "model_dir": str(MODEL_DIR),
    },
    "dummy": {
        "factory": "audioclone.core.backends.dummy.DummyTTSBackend",
        "sample_rate": SAMPLE_RATE,
    },
    "dummy-omni": {
        "factory": "audioclone.core.backends.dummy.DummyOmnibackend",
        "sample_rate": SAMPLE_RATE,
    },
}

ASR_BACKENDS: dict[str, dict[str, Any]] = {
    "qwen3-asr-llama": {
        "factory": "audioclone.core.backends.qwen3_asr_llama.Qwen3AsrLlamaBackend",
        "model_dir": str(ASR_MODEL_DIR),
    },
    "dummy": {
        "factory": "audioclone.core.backends.dummy.DummyASRBackend",
    },
    "dummy-omni": {
        "factory": "audioclone.core.backends.dummy.DummyOmnibackend",
        "sample_rate": SAMPLE_RATE,
    },
}


class Settings(BaseSettings):
    """运行时配置，可由 env（前缀 ``AUDIOCLONE_``）或 ``.env`` 覆盖。"""

    model_config = SettingsConfigDict(env_prefix="AUDIOCLONE_", env_file=".env", extra="ignore")

    # 默认后端
    tts_backend: str = "cosyvoice3-mlx"
    asr_backend: str = "qwen3-asr-llama"

    # 上传/资源限流
    max_upload_bytes: int = 100 * 1024 * 1024   # 上传音频最大字节数

    # ASR HTTP
    asr_http_timeout: float = 120.0
    asr_ready_timeout: float = 60.0

    # 输出目录保留上限（<=0 表示不清理；P2 生命周期管理）
    output_keep_max: int = 0


settings = Settings()

TTS_BACKEND = settings.tts_backend
ASR_BACKEND = settings.asr_backend
MAX_UPLOAD_BYTES = settings.max_upload_bytes
ASR_HTTP_TIMEOUT = settings.asr_http_timeout
ASR_READY_TIMEOUT = settings.asr_ready_timeout
OUTPUT_KEEP_MAX = settings.output_keep_max

# API 上传允许的音频扩展名
ALLOWED_UPLOAD_EXT = ASR_SUPPORTED_FORMATS