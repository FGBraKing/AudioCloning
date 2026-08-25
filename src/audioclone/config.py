"""AudioCloning 全局配置。

集中管理路径、模型、音频与 WebUI/API 相关常量。
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── 网络代理配置 ─────────────────────────────────────────────
# 默认绕过系统代理，确保本地服务 (Gradio/llama-server) 通信正常
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# ─── 路径配置 ───────────────────────────────────────────────
# src/audioclone/config.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 模型配置（MLX 格式）
MODEL_ID = "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16"
MODEL_DIR = PROJECT_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B-2512-fp16"

# ─── 输出配置 ───────────────────────────────────────────────
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
TTS_BACKENDS = {
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
TTS_BACKEND = os.environ.get("AUDIOCLONE_TTS_BACKEND", "cosyvoice3-mlx")

ASR_BACKENDS = {
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
ASR_BACKEND = os.environ.get("AUDIOCLONE_ASR_BACKEND", "qwen3-asr-llama")