# AudioCloning — 音频模型体验工具箱 (v0.3.0)

一个**模型可替换**的本地音频体验工具。内置基于 **Qwen3-TTS-1.7B (MLX)** 的语音合成/克隆
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
| WebUI | NiceGUI 交互界面：TTS 四模式（含纯文本合成）+ ASR + 多任务实验台 + 模型一键下载卡片，深浅色响应式布局 |
| REST API | FastAPI 程序化接口，便于集成 |
| dummy 后端 | 内置伪模型，无需 GPU 即可完整体验框架链路 |

---

## 系统要求

- **硬件**: Apple Silicon Mac (M1/M2/M3/M4)
- **系统**: macOS 13+
- **Python**: 3.13（由 uv 管理虚拟环境）
- **环境管理**: [uv](https://docs.astral.sh/uv/)（构建 Lock、解析依赖）
- **可选依赖**: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（onnx 可选依赖组，CPU / AMD GPU 平台的 ASR/TTS/VAD 需要；`uv sync --extra onnx`）

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
│       │   │   ├── registry.py  # 后端注册表（工厂唯一事实源）
│       │   │   ├── qwen3_tts_mlx.py  # Qwen3-TTS (MLX) 三变体 TTS 后端
│       │   │   ├── qwen3_asr_mlx.py  # Qwen3-ASR (MLX) 后端（mac 默认）
│       │   │   ├── minimax_music_mlx.py   # MiniMax Music (MLX) 音乐生成（T2A）
│       │   │   ├── deepfilternet_mlx.py   # DeepFilterNet (MLX) 语音增强（A2A）
│       │   │   ├── sensevoice_mlx.py      # SenseVoice (MLX) 情感+事件分类
│       │   │   ├── silero_vad_mlx.py      # Silero VAD (MLX)
│       │   │   ├── qwen3_tts_tensorrt.py  # Qwen3-TTS (TensorRT Edge-LLM) 三变体
│       │   │   ├── sherpa_asr_onnx.py # SenseVoice (sherpa-onnx) 后端（CPU/AMD 默认）
│       │   │   ├── sherpa_tts_onnx.py # Kokoro TTS (sherpa-onnx) 后端（plain 模式）
│       │   │   ├── sherpa_vad_onnx.py # Silero VAD (sherpa-onnx) 后端
│       │   │   └── dummy.py      # 伪模型后端（框架测试/快速演示）
│       │   ├── service/      # 服务编排层（按能力消费，与模型解耦）
│       │   │   ├── tts_service.py      # TTSService（分段/拼接/落盘）
│       │   │   ├── text_to_audio_service.py # TextToAudioService
│       │   │   ├── asr_service.py      # ASRService（格式/时长校验）
│       │   │   ├── audio_to_audio_service.py # AudioToAudioService
│       │   │   ├── audio_classification_service.py # AudioClassificationService
│       │   │   ├── vad_service.py      # VoiceActivityService
│       │   │   └── model_loader.py     # 按配置创建后端 + 能力查询
│       │   ├── runtime.py   # 运行环境探测（硬件→推理框架，容量降级）
│       │   ├── model_slot.py # 模型槽位守卫（六任务，各槽位单模型）
│       │   ├── model_manager.py # 模型下载/状态（CLI 与 API 共用）
│       │   └── text.py       # 长文本分段、归一化
│       ├── webui/            # NiceGUI 界面（仅依赖服务层）
│       │   ├── app.py        # 入口 (audioclone-webui) + 路由装配
│       │   ├── components.py # 面板布局
│       │   ├── widgets.py    # 复合控件（音频输入/结果展示/忙态按钮）
│       │   ├── theme.py      # 设计令牌与全局样式
│       │   ├── media.py      # 上传与结果落盘
│       │   ├── errors.py     # 界面错误类型
│       │   └── handlers.py   # 事件处理（框架无关）
│       ├── api/              # FastAPI
│       │   └── server.py     # 入口 (audioclone-api)
│       ├── cli/              # 命令工具
│       │   └── downloads.py  # 模型下载 (audioclone-download-*)
│       └── utils/
│           ├── audio.py      # 音频验证、保存、拼接
│           └── model_path.py # 模型路径解析（HF 缓存查找，不下载）
├── tests/                    # pytest（含 dummy 架构测试 + 集成测试）
├── docs/                     # 重构方案 / ADR 文档
├── asset/                    # 参考音频样本
├── pretrained_models/        # 本地模型目录（可选；默认改用 HF 系统缓存）
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

# 可选：CPU / AMD GPU 平台的 sherpa-onnx 推理（ASR/TTS/VAD）
uv sync --extra onnx
```

### 2. 下载模型（首次运行必须主动执行）

模型默认从 **HuggingFace** 下载到系统缓存（`~/.cache/huggingface/hub`，可用 `HF_HOME` /
`HF_HUB_CACHE` 重定向），与运行时的默认加载路径一致；来源策略为 **auto：HF 失败自动回退
ModelScope**（落盘 `pretrained_models/<模型名>`，加载路径会回退查找该目录）。支持自定义
HF 镜像端点（`AUDIOCLONE_HF_ENDPOINT` 或标准 `HF_ENDPOINT`，如 `https://hf-mirror.com`）。
**首次运行不会自动下载**：启动时若缓存未命中，WebUI 会显示加载失败面板、API 的推理接口返回
503，日志中给出下载指引；此时主动执行下面的命令（或调用 `POST /api/models/download`）即可。

```bash
# 下载 MLX 格式 TTS 模型 (约 1.1GB) —— 默认 auto：HF 优先，失败回退 ModelScope
uv run audioclone-download-tts
# 下载 Qwen3-ASR (MLX) 模型 (约 600MB；mac 默认后端)
uv run audioclone-download-asr

# 可选：强制指定来源（默认 auto）
uv run audioclone-download-tts --source hf
uv run audioclone-download-asr --source modelscope

# 可选：自定义下载目录（需配合 --model_dir / --asr_model_dir 使用）
uv run audioclone-download-tts --local_dir /path/to/model
# 可选：自定义 HF 镜像
export AUDIOCLONE_HF_ENDPOINT=https://hf-mirror.com
```

### 3. 启动服务

```bash
# 启动 WebUI (默认 http://localhost:7860)
uv run audioclone-webui

# 禁用 ASR 功能启动
uv run audioclone-webui --no_asr

# 自定义模型目录（默认使用 HuggingFace 系统缓存）
uv run audioclone-webui --model_dir /path/to/tts_model --asr_model_dir /path/to/asr_gguf

# 切换后端（无需真实模型即可完整体验框架链路）
uv run audioclone-webui --tts-backend dummy --asr-backend dummy

# 其他可选参数：--model_dir / --asr_model_dir / --theme dark|light / --no-browser
uv run audioclone-webui --theme light --port 7870

# 启动 REST API (默认 http://localhost:8000)
uv run audioclone-api
```

### TTS 后端一览

| 后端名 | 模型 | 支持的推理模式 |
|--------|------|----------------|
| `qwen3-tts-base-mlx`（默认） | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | 零样本克隆 / 跨语言 |
| `qwen3-tts-customvoice-mlx` | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | 指令控制（预设音色 + 情绪） |
| `qwen3-tts-voicedesign-mlx` | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | 指令控制（自然语言造音色） |
| `sherpa-tts-onnx` | Kokoro-82M int8 | plain（纯文本直读） |
| `qwen3-tts-base-tensorrt` / `-customvoice-` / `-voicedesign-` | Qwen3-TTS（TensorRT Edge-LLM 引擎，用户构建） | base: 克隆；customvoice/voicedesign: instruct |
| `dummy` / `dummy-omni` | 无（伪模型） | 全部 |

切换示例（下载对应模型后再启动）：

```bash
uv run audioclone-download-tts --backend qwen3-tts-base-mlx
uv run audioclone-webui --tts-backend qwen3-tts-base-mlx
```

> **依赖说明**：三个后端依赖 `mlx_audio` 的 `qwen3_tts` 架构（**mlx-audio ≥ 0.3.0**，
> 本项目锁定 `mlx-audio[tts]>=0.5.0`）。上游 mlx-audio 0.3.0 起已移除 CosyVoice 实现，
> 因此早期的 `cosyvoice3-mlx` 后端已随本迁移一并下线。
>
> **运行时模型约束**：为保证内存可控，进程内**同时只加载一个 TTS 模型与一个 ASR 模型**
> （由核心层模型槽位守卫强制执行）。因此三个 Qwen3-TTS 变体需按需切换：
> 克隆用 `base`，情绪/音色指令用 `customvoice` 或 `voicedesign`。
> WebUI 会对当前后端不支持的模式**直接停用对应面板**并给出可切换的后端清单，
> 而不是等你点击后才报错。

> **注**: 项目已在 `config.py` 中默认设置 `no_proxy=localhost,127.0.0.1`，代理不再造成本地回环连接异常。当前默认后端为 **auto（按硬件/系统自动选择推理框架）**：macOS (Apple Silicon)→MLX、NVIDIA GPU→TensorRT-LLM、**AMD GPU（核显/独显）与无 GPU→onnx**（sherpa-onnx 音频专用推理；vLLM 不支持音频场景，经调研决策排除）。并含**容量判断**：Apple Silicon 统一内存或 NVIDIA **可用显存**（总量−其他进程占用，多卡取可用最大的一张）低于阈值（默认 8GB，`AUDIOCLONE_MIN_MLX_MEM_GB` / `AUDIOCLONE_MIN_CUDA_VRAM_GB` 可调）时自动降级 onnx。**可用性兜底**：auto 选择时离线检查各候选后端的模型是否已就位（自定义目录 / HF 缓存 / ModelScope 目录，不触发下载），已就位的真实后端优先（绝不静默退化成 dummy）；全部未就位时保持框架首选（加载报错时下载指引指向正确模型），绝不静默退化成 dummy。可通过环境变量 `AUDIOCLONE_TTS_BACKEND` / `AUDIOCLONE_ASR_BACKEND` 或 CLI 参数显式指定后端。

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
  "backend": "qwen3-tts-base-mlx",
  "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
  "supported_modes": ["zero_shot", "cross_lingual"],
  "sample_rate": 24000,
  "model_loaded": true,
  "model_slots": { "tts": "qwen3-tts-base-mlx", "asr": null }
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
| `ref_audio` | file | ✓（plain 可省略） | 参考音频文件 |
| `ref_text` | string | | 参考音频对应文字 |
| `speed` | float | | 语速 0.5~2.0，默认 1.0 |
| `mode` | string | | `zero_shot` / `cross_lingual` / `instruct` / `plain`（纯文本直读，无需参考音频） |
| `instruct_text` | string | | 自然语言指令 (instruct 模式，必填) |

响应: WAV 音频文件流，Header 含 `X-Synthesis-Time`。

### 多任务接口

| 接口 | 说明 | 响应 |
|------|------|------|
| `POST /api/t2a` | 文本转音频（音乐生成）。Form：`text`（必填）、`duration_sec`、`lyrics` | WAV 流，Header `X-Generation-Time` |
| `POST /api/transform` | 音频转换/增强。multipart：`audio` 文件、`instruct` | WAV 流，Header `X-Transform-Time` |
| `POST /api/classify` | 音频分类（情感 9 类 + 事件 4 类）。`audio`、`top_k` | JSON `{"results": [{"label", "score"}]}` |
| `POST /api/vad` | 语音活动检测。`audio` | JSON `{"segments": [{"start", "end", "duration"}], "speech_sec"}` |

四个端点的模型均在**首次调用时惰性加载**（构造不加载权重）；模型未下载时返回 503 并附下载指引。

```bash
# 文本生成音乐
curl -X POST http://localhost:8000/api/t2a -F 'text=轻柔的钢琴曲' -F 'duration_sec=10' -o music.wav

# 语音降噪增强
curl -X POST http://localhost:8000/api/transform -F 'audio=@noisy.wav' -o clean.wav

# 情感/事件分类
curl -X POST http://localhost:8000/api/classify -F 'audio=@sample.wav'

# 语音活动检测
curl -X POST http://localhost:8000/api/vad -F 'audio=@sample.wav'
```

---

## NVIDIA 平台（TensorRT Edge-LLM）

NVIDIA 平台 TTS 对接 **TensorRT Edge-LLM 官方 Qwen3-TTS 流水线**（三变体与 MLX 侧同族同能力）。工作流分两段，**本项目只封装推理步、不下载权重**：

1. **导出**（x86 主机）：`tensorrt-edgellm-export` 导出 Qwen3-TTS 三组件 ONNX（Talker / CodePredictor / Code2Wav；Base 变体含 `clone_encoders/`），CodePredictor 可选 FP8/NVFP4 量化
2. **构建**（目标设备）：`llm_build` / `audio_build` 生成 TensorRT 引擎
3. **推理**（`qwen3-tts-*-tensorrt` 后端封装）：以 `--model_dir` 指向引擎目录，后端调用 `qwen3_tts_inference` 子进程产出 WAV

未提供引擎目录时加载报可操作指引（不猜测、不下载）；ASR/T2A/A2A/分类/VAD 在该平台无 TensorRT 实现，回退 dummy。

## onnx 平台（AMD GPU / 无 GPU）

该平台使用 **sherpa-onnx**（onnxruntime）音频专用推理，ASR/TTS/VAD 三任务有真实后端；其余能力回退 dummy。默认模型均为官方 int8 量化（经 HuggingFace 核实）：

| 能力 | 后端 | 默认模型下载点（HuggingFace） |
|------|------|------------------------------|
| ASR | `sherpa-asr-onnx` | [`csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17)：`model.int8.onnx` + `tokens.txt` |
| TTS（plain 模式） | `sherpa-tts-onnx` | [`csukuangfj/kokoro-int8-multi-lang-v1_1`](https://huggingface.co/csukuangfj/kokoro-int8-multi-lang-v1_1)：`model.int8.onnx` + `voices.bin` + `tokens.txt` + `espeak-ng-data/`（82M 参数，中/英） |
| VAD | `sherpa-vad-onnx` | [`onnx-community/silero-vad`](https://huggingface.co/onnx-community/silero-vad)（约 2MB） |
| T2A / A2A / 分类 | dummy | 下载点未核实（T2A 音乐无 onnx 等价物；A2A 增强/音频标注待接入） |

> 依赖：`uv sync --extra onnx`；模型默认**不下载**，主动下载后自动命中。

## 模型管理接口

| 接口 | 说明 |
|------|------|
| `GET /api/models` | TTS / ASR 模型就绪状态（离线查询：`cached` 是否已下载、`model_path` 实际路径、`hf_endpoint` 生效端点） |
| `POST /api/models/download` | **主动下载**模型权重（后台执行，返回 202；`source` 可选 `auto`/`hf`/`modelscope`，默认 auto=HF 失败自动回退 ModelScope） |
| `GET /api/models/download` | 查询当前/最近一次下载任务进度 |

```bash
# 状态查询
curl http://localhost:8000/api/models

# 主动下载（task: tts | asr；backend 可选，缺省用当前默认后端；local_dir 可选）
curl -X POST http://localhost:8000/api/models/download \
  -H 'Content-Type: application/json' \
  -d '{"task": "asr"}'

# 查询进度
curl http://localhost:8000/api/models/download
```

下载默认进 HuggingFace 系统缓存；`local_dir` 指定自定义目录。同一时刻仅允许一个下载任务（并发请求返回 409）。

---

## 配置说明

核心配置位于 `src/audioclone/config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `SAMPLE_RATE` | 24000 | TTS 输出采样率 |
| `MAX_TEXT_LENGTH` | 300 | 单次合成最大字符数 |
| `REF_AUDIO_MIN_SEC` | 1 | 参考音频最短时长 |
| `REF_AUDIO_MAX_SEC` | 30 | 参考音频最长时长 |
| `TTS_MODEL_DIR` | 无 | 自定义 TTS 模型目录（env `AUDIOCLONE_TTS_MODEL_DIR`）；默认 HuggingFace 系统缓存 |
| `ASR_MODEL_DIR` | 无 | 自定义 ASR 模型目录（env `AUDIOCLONE_ASR_MODEL_DIR`）；默认 HuggingFace 系统缓存 |
| `WEBUI_PORT` | 7860 | WebUI 端口 |
| `WEBUI_HOST` | 127.0.0.1 | WebUI 监听地址 |
| `WEBUI_CONCURRENCY` | 2 | WebUI 推理并发上限 |
| `ASR_AUDIO_MAX_SEC` | 300 | ASR/多任务最大音频时长 |
| `ASR_NUM_THREADS` | 2 | sherpa-onnx（onnxruntime）推理线程数 |
| `TENSORRT_ENGINE_DIR` | 无 | TensorRT Edge-LLM 引擎目录（`qwen3-tts-*-tensorrt` 专用） |

---

## 测试

```bash
# 纯逻辑单元测试 (无需模型/GPU)
uv run pytest -m "not integration"

# 全部测试 (集成测试缺少模型时自动跳过)
uv run pytest
```

集成测试(`tests/test_tts.py`、`tests/test_roundtrip.py`)需要本地模型，未就绪会自动跳过。

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
| Qwen3-TTS-12Hz-1.7B-Base-8bit | TTS 参考音频克隆 | MLX (safetensors, 8bit) | mlx-community / ModelScope |
| Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit | TTS 预设音色 + 情绪指令 | MLX (safetensors, 8bit) | mlx-community / ModelScope |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit | TTS 自然语言造音色 | MLX (safetensors, 8bit) | mlx-community / ModelScope |
| Qwen3-ASR-0.6B-8bit | ASR 语音识别（mac 默认 `qwen3-asr-mlx` 后端） | MLX (safetensors, 8bit) | mlx-community / ModelScope |
| MiniMax-Music3-8bit | T2A 音乐生成（`minimax-music-mlx`；参数量放宽例外，约 14GB） | MLX (8bit) | mlx-community（已确认存在） |
| DeepFilterNet-mlx | A2A 语音增强/降噪（`deepfilternet-mlx`，约数 MB） | MLX | mlx-community（mlx_audio 源码内置） |
| silero-vad-v6 | VAD 语音活动检测（`silero-vad-mlx`，约 2MB，MIT） | MLX | mlx-community（已确认存在） |
| SenseVoiceSmall | 音频分类：情感 9 类 + 事件 4 类（`sensevoice-mlx`，234M） | MLX | mlx-community（已确认存在，mlx_audio 原生） |

**模型选型原则**：默认模型仅选 **8bit 量化（或非 LLM 极小模型）、较新较强**的小模型，
参数量以 **≤3B** 为基线；优先 mlx_audio 源码内置仓库或已确认存在的 mlx-community
仓库（不自实现、不猜仓库 ID），**不引入 torch 等重依赖**；**默认不下载权重**（首启
提示 + 主动下载接口）。两处经决策放宽：T2A 音乐（MiniMax Music3，唯一有 8bit 的音乐
生成模型）。分类采用 SenseVoice（情感+事件内建分类头，一模型多任务）；如需通用
（AudioSet 式）分类，按项目扩展机制自实现新后端接入。A2A 变声无 MLX 原生模型
（DeepFilterNet 为增强；VoiceDesign 可经 TTS 变通）。

---

## 使用提示

- 参考音频建议 **3~10 秒**清晰人声，采样率不低于 16kHz
- 首次加载 TTS 模型约需 30~60 秒
- ASR 服务首次启动约需 10~20 秒（模型加载到 Metal GPU）
- 长文本会自动按句子边界分段合成后拼接