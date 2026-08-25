# AudioCloning — 音频模型体验工具箱 (v0.2.5)

一个**模型可替换**的本地音频体验工具。内置基于 **Fun-CosyVoice3-0.5B (MLX)** 的语音合成/克隆
与基于 **Qwen3-ASR** 的语音识别，运行于 Apple Silicon。功能框架（WebUI / REST API / 流程编排）与
具体模型解耦，通过统一后端接口即可切换或接入新模型。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 模型可替换 | 通过配置 / CLI 切换后端，无需改业务代码 |
| 能力驱动 | 一个模型可聚合多种能力，一模型多任务 |
| Text-to-Speech | 零样本克隆 / 跨语言 / 指令控制 / 长文本分段 |
| Text-to-Audio | 由文本提示生成音乐 / 音效 |
| Speech Recognition | 基于 Qwen3-ASR 的本地语音转文字 |
| Audio-to-Audio | 变声 / 音色转换 / 音质增强 |
| Audio Classification | 语音/音乐/环境声等分类 |
| Voice Activity Detection | 说话片段检测 (VAD) |
| WebUI | Gradio 交互界面，支持上传/录音 |
| REST API | FastAPI 程序化接口，便于集成 |
| dummy 后端 | 内置伪模型，无需 GPU 即可完整体验框架链路 |

---

## 系统要求

- **硬件**: Apple Silicon Mac (M1/M2/M3/M4)
- **系统**: macOS 13+
- **Python**: 3.13（由 uv 管理虚拟环境）
- **环境管理**: [uv](https://docs.astral.sh/uv/)（构建 Lock、解析依赖）
- **系统依赖**: [llama.cpp](https://github.com/ggerganov/llama.cpp)（ASR 功能需要）

---

## 项目结构

```
AudioCloning/
├── pyproject.toml            # 项目元数据、依赖、CLI 入口
├── uv.lock                   # 精确锁定依赖版本 (由 uv sync 生成)
├── .python-version           # 固定 Python 3.13
├── src/
│   └── audioclone/           # 主包 (src-layout)
│       ├── config.py         # 全局配置（含后端注册表）
│       ├── core/
│       │   ├── capabilities/ # 能力抽象层（一模型多任务）
│       │   │   ├── base.py      # Capability / ModelBackend / supports()
│       │   │   ├── tts.py       # TextToSpeechCapability
│       │   │   ├── t2a.py       # TextToAudioCapability
│       │   │   ├── asr.py       # SpeechRecognitionCapability
│       │   │   ├── a2a.py       # AudioToAudioCapability
│       │   │   ├── audio_classification.py # AudioClassificationCapability
│       │   │   └── vad.py       # VoiceActivityCapability
│       │   ├── backends/     # 模型实现层（可替换、可插拔）
│       │   │   ├── registry.py  # 后端注册表
│       │   │   ├── cosyvoice_mlx.py  # CosyVoice3 (MLX) TTS 后端
│       │   │   ├── qwen3_asr_llama.py # Qwen3-ASR (llama.cpp) 后端
│       │   │   └── dummy.py      # 伪模型后端（框架测试/快速演示）
│       │   ├── service/      # 服务编排层（按能力消费，与模型解耦）
│       │   │   ├── tts_service.py      # TTSService（分段/拼接/落盘）
│       │   │   ├── text_to_audio_service.py # TextToAudioService
│       │   │   ├── asr_service.py      # ASRService（格式/时长校验）
│       │   │   ├── audio_to_audio_service.py # AudioToAudioService
│       │   │   ├── audio_classification_service.py # AudioClassificationService
│       │   │   ├── vad_service.py      # VoiceActivityService
│       │   │   └── model_loader.py     # 按配置创建后端 + 能力查询
│       │   └── text.py       # 长文本分段、归一化
│       ├── webui/            # Gradio 界面（仅依赖服务层）
│       │   ├── app.py        # 入口 (audioclone-webui)
│       │   ├── components.py # 界面布局
│       │   └── handlers.py   # 事件处理
│       ├── api/              # FastAPI
│       │   └── server.py     # 入口 (audioclone-api)
│       ├── cli/              # 命令工具
│       │   └── downloads.py  # 模型下载 (audioclone-download-*)
│       └── utils/
│           └── audio.py      # 音频验证、保存、拼接
├── tests/                    # pytest（含 dummy 架构测试 + 集成测试）
├── docs/                     # 重构方案 / ADR 文档
├── asset/                    # 参考音频样本
├── pretrained_models/        # 预训练模型 (需下载)
└── outputs/                  # 合成音频输出
```

---

## 快速开始

### 1. 安装 uv 与依赖

```bash
# 安装 uv (若未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建 .venv 并安装依赖 (自动按 .python-version 选用 Python 3.13)
uv sync

# 安装系统依赖 (ASR 功能需要)
brew install llama.cpp
```

### 2. 下载模型

```bash
# 下载 MLX 格式 TTS 模型 (约 1.1GB)
uv run audioclone-download-tts

# 下载 Qwen3-ASR GGUF 模型 (约 600MB)
uv run audioclone-download-asr
```

### 3. 启动服务

```bash
# 启动 WebUI (默认 http://localhost:7860)
uv run audioclone-webui

# 禁用 ASR 功能启动
uv run audioclone-webui --no_asr

# 切换后端（无需真实模型即可完整体验框架链路）
uv run audioclone-webui --tts-backend dummy --asr-backend dummy

# 启动 REST API (默认 http://localhost:8000)
uv run audioclone-api
```

> **注**: 项目已在 `config.py` 中默认设置 `no_proxy=localhost,127.0.0.1`，代理不再造成本地回环连接异常。当前默认后端为 `cosyvoice3-mlx` + `qwen3-asr-llama`，可通过环境变量 `AUDIOCLONE_TTS_BACKEND` / `AUDIOCLONE_ASR_BACKEND` 或 CLI 参数切换。

---

## 模型可替换：如何接入新后端

框架分为四层：**展示层 → 服务层 → 能力层 → 实现层**。核心是**按能力(Capability)组织**：
一个模型后端可聚合多种能力（TTS / ASR / 变声 / 分类 / VAD 等），「任务 = 能力编排」。
服务层只认能力，不认具体模型。接入新模型需两步：

1. **实现后端接口**（继承 `ModelBackend` + 一个或多个能力接口），例如：

```python
# src/audioclone/core/backends/my_tts.py
import numpy as np
from audioclone.core.capabilities import ModelBackend, TextToSpeechCapability

class MyTTSBackend(ModelBackend, TextToSpeechCapability):
    name = "my-tts"
    sample_rate = 24000

    def synthesize(self, mode, text, ref_audio_path, ref_text="", instruct_text="", speed=1.0) -> np.ndarray:
        ...
```

   若模型同时支持语音识别，一并继承 `SpeechRecognitionCapability` 并实现 `transcribe`，即为**一模型多任务**。
   服务层 `TTSService` / `ASRService` 会按能力校验后端是否可用。查询后端能力：`list_backend_capabilities(backend)`。

2. **注册到 registry**（`core/backends/registry.py`）并在 config 登记可选覆盖参数：

```python
# core/backends/registry.py —— 工厂的唯一事实源
from audioclone.core.backends.my_tts import MyTTSBackend
registry.register("my-tts", MyTTSBackend)

# config.py —— 默认选择 + 覆盖参数（可选）
TTS_BACKENDS["my-tts"] = {"model_id": "my-org/my-model"}
```

随后即可 `uv run audioclone-webui --tts-backend my-tts` 启用。扩展新任务（如变声/分类/VAD）只需新增对应 `Capability`
接口 + 配套 Service，已有后端聚合该能力即可，WebUI / API 与其余业务无需改动。

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
| `instruct_text` | string | | 自然语言指令 (instruct 模式，必填) |

响应: WAV 音频文件流，Header 含 `X-Synthesis-Time`。

---

## 配置说明

核心配置位于 `src/audioclone/config.py`：

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
# 纯逻辑单元测试 (无需模型/GPU)
uv run pytest -m "not integration"

# 全部测试 (集成测试缺少模型时自动跳过)
uv run pytest
```

集成测试(`tests/test_tts.py`、`tests/test_asr.py`、`tests/test_roundtrip.py`)需要本地模型与 llama.cpp，未就绪会自动跳过。

---

## 开发

```bash
uv run ruff check .
uv run pytest
```

### 常用 uv 命令

| 命令 | 说明 |
|------|------|
| `uv sync` | 安装/同步依赖到 `.venv` |
| `uv add <pkg>` | 新增运行依赖 |
| `uv add --dev <pkg>` | 新增开发依赖 |
| `uv remove <pkg>` | 移除依赖 |
| `uv run <script>` | 在虚拟环境中运行命令 |

---

## 模型信息

| 模型 | 用途 | 格式 | 来源 |
|------|------|------|------|
| Fun-CosyVoice3-0.5B-2512-fp16 | TTS 语音合成 | MLX (safetensors) | mlx-community / ModelScope |
| Qwen3-ASR-0.6B-GGUF | ASR 语音识别 | GGUF (Q8_0) | ggml-org / ModelScope |

---

## 使用提示

- 参考音频建议 **3~10 秒**清晰人声，采样率不低于 16kHz
- 首次加载 TTS 模型约需 30~60 秒
- ASR 服务首次启动约需 10~20 秒（模型加载到 Metal GPU）
- 长文本会自动按句子边界分段合成后拼接