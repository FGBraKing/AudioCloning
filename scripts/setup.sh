#!/bin/bash
# ============================================================
# AudioCloning macOS 一键安装脚本 (MLX 版)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

ENV_NAME="agent"

echo "============================================================"
echo "  AudioCloning macOS 安装脚本 (MLX)"
echo "============================================================"

# 1. 激活 conda 环境
echo "[1/5] 激活 conda 环境: $ENV_NAME ..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "  环境已激活: $CONDA_DEFAULT_ENV (Python $(python --version 2>&1 | awk '{print $2}'))"

# 2. 检查 llama.cpp (ASR 功能依赖)
echo "[2/5] 检查 llama-server..."
if command -v llama-server &> /dev/null; then
    echo "  llama-server 已安装: $(which llama-server)"
    echo "  版本: $(llama-server --version 2>&1 | head -1)"
else
    echo "  [WARNING] 未找到 llama-server，ASR 功能将不可用"
    echo "  安装方式: brew install llama.cpp"
fi

# 3. 安装 Python 依赖
echo "[3/5] 安装 Python 依赖..."
pip install -r requirements.txt

# 4. 下载 TTS 模型
echo "[4/5] 下载 MLX 格式 TTS 模型..."
python scripts/download_tts_model.py

# 5. 下载 ASR 模型
echo "[5/5] 下载 ASR 模型 (Qwen3-ASR-0.6B-GGUF)..."
python scripts/download_asr_model.py

echo ""
echo "============================================================"
echo "  安装完成!"
echo "============================================================"
echo "  启动 WebUI:  python src/app.py"
echo "  禁用 ASR:    python src/app.py --no_asr"
echo "  启动 API:    python src/api/server.py"
echo "  访问地址:    http://localhost:7860"
echo "============================================================"
