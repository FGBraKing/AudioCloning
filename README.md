# AudioCloning — 个人声音克隆系统

基于 **Fun-CosyVoice3-0.5B (MLX)** 的本地语音合成与克隆系统，运行于 Apple Silicon。  
支持零样本声音克隆、跨语言合成、自然语言指令控制，以及基于 Qwen3-ASR 的语音识别。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 零样本克隆 | 提供 3~30 秒参考音频即可克隆任意音色 |
| 跨语言合成 | 使用源语言音色合成目标语言语音 |
| 指令控制 | 通过自然语言指令控制语气、情感、语速 |
| 语音识别 (ASR) | 基于 Qwen3-ASR-0.6B-GGUF 的本地语音转文字 |
| WebUI | Gradio 交互界面，支持上传/录音 |
| REST API | FastAPI 程序化接口，便于集成 |

---

## 系统要求

- **硬件**: Apple Silicon Mac (M1/M2/M3/M4)
- **系统**: macOS 13+
- **Python**: ≥ 3.13 (conda 环境 `agent`)
- **系统依赖**: [llama.cpp](https://github.com/ggerganov/llama.cpp)（ASR 功能需要）

---

## 项目结构

```
AudioCloning/
├── scripts/                    # 安装与模型下载脚本
│   ├── setup.sh               # macOS 一键安装
│   ├── setup.bat              # Windows 参考脚本 (MLX 不支持 Windows)
│   ├── download_tts_model.py  # 下载 CosyVoice3 MLX 模型
│   └── download_asr_model.py  # 下载 Qwen3-ASR GGUF 模型
├── tests/                      # 端到端测试
│   ├── test_tts.py            # TTS 三模式推理测试
│   ├── test_asr.py            # ASR 识别测试
│   └── test_roundtrip.py      # TTS→ASR 回环交叉验证
├── src/                        # 核心源码
│   ├── config.py              # 全局配置
│   ├── app.py                 # WebUI 主入口
│   ├── tts/                   # TTS 模块
│   │   ├── engine.py          # TTSEngine 推理引擎 + 模型加载
│   │   └── text_processor.py  # 长文本分段、归一化
│   ├── asr/                   # ASR 模块
│   │   └── engine.py          # ASREngine (llama-server 管理)
│   ├── utils/                 # 工具模块
│   │   └── audio.py           # 音频验证、保存、拼接
│   ├── webui/                 # WebUI 组件
│   │   ├── components.py      # Gradio 界面布局
│   │   └── handlers.py        # 事件处理逻辑
│   └── api/                   # REST API
│       └── server.py          # FastAPI 服务
├── asset/                      # 参考音频样本
├── pretrained_models/          # 预训练模型 (需下载)
├── outputs/                    # 合成音频输出
└── requirements.txt            # Python 依赖
```

---

## 快速开始

### 1. 一键安装

```bash
# 安装 llama.cpp (ASR 功能依赖)
brew install llama.cpp

# 运行安装脚本
bash scripts/setup.sh
```

### 2. 手动安装

```bash
# 激活 conda 环境
conda activate agent

# 安装 Python 依赖
pip install -r requirements.txt

# 下载 TTS 模型 (约 1.1GB)
python scripts/download_tts_model.py

# 下载 ASR 模型 (约 600MB)
python scripts/download_asr_model.py
```

### 3. 启动服务

```bash
# 启动 WebUI (默认 http://localhost:7860)
python src/app.py

# 禁用 ASR 功能启动
python src/app.py --no_asr

# 启动 REST API (默认 http://localhost:8000)
python src/api/server.py
```

> **注意**: 启动 WebUI 前需设置 `export no_proxy=localhost,127.0.0.1`，否则 Gradio 可能连接异常。

---

## 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    用户交互层                         │
│   ┌──────────────┐         ┌──────────────────┐     │
│   │  Gradio WebUI │         │  FastAPI REST API │     │
│   │  (port 7860)  │         │  (port 8000)      │     │
│   └──────┬───────┘         └────────┬─────────┘     │
├──────────┼──────────────────────────┼───────────────┤
│          │        业务逻辑层         │               │
│   ┌──────▼──────────────────────────▼─────────┐     │
│   │              TTSEngine                     │     │
│   │  zero_shot / cross_lingual / instruct      │     │
│   └──────────────────┬────────────────────────┘     │
│   ┌──────────────────▼────────────────────────┐     │
│   │              ASREngine                     │     │
│   │  llama-server 子进程管理 + HTTP 通信        │     │
│   └───────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────┤
│                    模型层                            │
│   ┌────────────────────┐  ┌─────────────────────┐   │
│   │ CosyVoice3-0.5B    │  │ Qwen3-ASR-0.6B     │   │
│   │ (MLX, fp16)        │  │ (GGUF, Q8_0)       │   │
│   │ 24kHz 输出          │  │ 16kHz 输入          │   │
│   └────────────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 模型信息

| 模型 | 用途 | 格式 | 来源 |
|------|------|------|------|
| Fun-CosyVoice3-0.5B-2512-fp16 | TTS 语音合成 | MLX (safetensors) | mlx-community / ModelScope |
| Qwen3-ASR-0.6B-GGUF | ASR 语音识别 | GGUF (Q8_0) | ggml-org / ModelScope |

---

## API 接口

### 健康检查

```
GET /api/health
```

响应示例:
```json
{
  "status": "ok",
  "framework": "MLX",
  "model": "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16",
  "sample_rate": 24000,
  "model_loaded": true
}
```

### 声音克隆

```
POST /api/clone
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | ✓ | 合成文本 (≤300 字符) |
| `ref_audio` | file | ✓ | 参考音频文件 |
| `ref_text` | string | | 参考音频对应文字 |
| `speed` | float | | 语速 0.5~2.0，默认 1.0 |
| `mode` | string | | `zero_shot` / `cross_lingual` / `instruct` |
| `instruct_text` | string | | 自然语言指令 (instruct 模式) |

响应: WAV 音频文件流，Header 含 `X-Synthesis-Time`。

---

## 配置说明

核心配置位于 `src/config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `SAMPLE_RATE` | 24000 | TTS 输出采样率 |
| `MAX_TEXT_LENGTH` | 300 | 单次合成最大字符数 |
| `REF_AUDIO_MIN_SEC` | 1 | 参考音频最短时长 |
| `REF_AUDIO_MAX_SEC` | 30 | 参考音频最长时长 |
| `GRADIO_PORT` | 7860 | WebUI 端口 |
| `ASR_SERVER_PORT` | 8090 | llama-server 端口 |
| `ASR_CTX_SIZE` | 4096 | ASR 上下文窗口 |
| `ASR_AUDIO_MAX_SEC` | 300 | ASR 最大音频时长 |

---

## 测试

```bash
conda activate agent

# TTS 三模式端到端测试
python tests/test_tts.py

# ASR 语音识别测试
python tests/test_asr.py

# TTS→ASR 回环交叉验证 (LCS 匹配率 ≥ 80% 为通过)
python tests/test_roundtrip.py
```

---

## 依赖清单

| 包 | 版本 | 用途 |
|----|------|------|
| mlx-audio-plus | ≥0.1.7 | CosyVoice3 MLX 推理 |
| mlx-lm | ≥0.31.0 | MLX 语言模型基础 |
| modelscope | ≥1.20.0 | 模型下载 |
| gradio | ≥5.4.0 | WebUI 界面 |
| soundfile | ≥0.12.1 | 音频 I/O |
| numpy | ≥1.26 | 数值计算 |
| scipy | ≥1.11.0 | 音频重采样 |
| fastapi | ≥0.115.6 | REST API |
| uvicorn | ≥0.30.0 | ASGI 服务器 |
| httpx | ≥0.27.0 | ASR HTTP 通信 |
| llama.cpp | 系统级 | ASR 推理 (llama-server) |

---

## 使用提示

- 参考音频建议 **3~10 秒**清晰人声，采样率不低于 16kHz
- 首次加载 TTS 模型约需 30~60 秒
- ASR 服务首次启动约需 10~20 秒（模型加载到 Metal GPU）
- 长文本会自动按句子边界分段合成后拼接
