# USER-GUIDE — AudioCloning 音频模型体验工具箱 · 使用说明

| 元信息 | 值 |
|--------|----|
| 文档版本 | 1.0 |
| **适用版本** | **v0.2.0** |
| 文档类型 | 使用说明（安装 / 配置 / 能力使用 / API / 扩展 / 故障排查） |
| 创建日期 | 2026-08-25 |
| 配套文档 | `QUALITY-REPORT.md`、`REFACTOR-001-audio-toolbox.md`、`README.md` |

---

## 1. 这是什么

一个**能力驱动、模型可替换**的本地音频模型体验工具箱（运行于 Apple Silicon + MLX）。
框架提供 6 种音频任务能力：**TTS / Text-to-Audio / ASR / Audio-to-Audio / 音频分类 / VAD**，
每种能力都有独立服务层。内置真实后端（CosyVoice3-MLX、Qwen3-ASR-llama.cpp）与全能力演示后端（dummy-omni）。

```
展示层(WebUI/API) → 服务层(Service) → 能力层(Capability) ← 实现层(后端)
```

---

## 2. 系统要求

- Apple Silicon Mac（M1/M2/M3/M4）、macOS 13+
- Python 3.13（由 uv 管理，`.python-version` 固定）
- [uv](https://docs.astral.sh/uv/)（`/opt/homebrew/bin/uv` 或加入 PATH）
- [llama.cpp](https://github.com/ggerganov/llama.cpp)（仅 ASR 需要；`brew install llama.cpp`）
- 可选真实模型：CosyVoice3-MLX、Qwen3-ASR GGUF（下载见 §6）

---

## 3. 安装

```bash
# 1) uv（若未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 进入项目并同步依赖（自动选 .python-version=3.13，生成 .venv + uv.lock）
cd AudioCloning
uv sync

# 3) ASR 系统依赖
brew install llama.cpp
```

> 全程使用 `/opt/homebrew/bin/uv` 亦可，等价。

---

## 4. 配置

### 4.1 环境变量（运行时）

| 变量 | 默认 | 说明 |
|------|------|------|
| `AUDIOCLONE_TTS_BACKEND` | `cosyvoice3-mlx` | 默认 TTS 后端（`cosyvoice3-mlx` / `dummy` / `dummy-omni`） |
| `AUDIOCLONE_ASR_BACKEND` | `qwen3-asr-llama` | 默认 ASR 后端（`qwen3-asr-llama` / `dummy` / `dummy-omni`） |
| `COSYVOICE_MODEL_DIR` | `pretrained_models/...` | API 服务覆盖 TTS 模型目录 |
| `no_proxy` / `NO_PROXY` | `localhost,127.0.0.1`（项目内置） | 绕过本地代理 |

### 4.2 后端注册表（`src/audioclone/config.py`）

新增/更换模型后端在 `TTS_BACKENDS` / `ASR_BACKENDS` 中登记：

```python
TTS_BACKENDS["my-tts"] = {
    "factory": "audioclone.core.backends.my_tts.MyTTSBackend",  # module.Class
    "model_id": "my-org/my-model",
}
```

> factory 字符串 = `模块路径.类名`，其余键作为构造参数（参考现有条目）。

### 4.3 能力矩阵（当前实现）

| 任务 | 能力接口 | Service | 真实后端 | 演示后端 |
|------|----------|---------|:---:|:---:|
| Text-to-Speech | `TextToSpeechCapability` | `TTSService` | ✅ cosyvoice_mlx | dummy/dummy-omni |
| Text-to-Audio | `TextToAudioCapability` | `TextToAudioService` | 待接入 | dummy-omni |
| Speech-Recognition | `SpeechRecognitionCapability` | `ASRService` | ✅ qwen3_asr_llama | dummy-omni |
| Audio-to-Audio | `AudioToAudioCapability` | `AudioToAudioService` | 待接入 | dummy-omni |
| Audio-Classification | `AudioClassificationCapability` | `AudioClassificationService` | 待接入 | dummy-omni |
| Voice-Activity-Detection | `VoiceActivityCapability` | `VoiceActivityService` | 待接入 | dummy-omni |

---

## 5. 快速上手

### 5.1 用真实模型启动 WebUI

```bash
uv run audioclone-download-tts    # 首次：下载 CosyVoice3-MLX（约 1.1GB）
uv run audioclone-download-asr    # 首次：下载 Qwen3-ASR GGUF（约 600MB）

uv run audioclone-webui           # http://localhost:7860
uv run audioclone-webui --no_asr  # 禁用 ASR
```

### 5.2 无模型快速体验（dummy，全能力）

```bash
uv run audioclone-webui --tts-backend dummy-omni --asr-backend dummy-omni
```

### 5.3 启动 REST API

```bash
uv run audioclone-api             # http://0.0.0.0:8000
```

---

## 6. 命令参考

| 命令 | 说明 |
|------|------|
| `uv run audioclone-webui [--port 7860] [--host 0.0.0.0] [--model_dir X] [--asr_model_dir X] [--tts-backend NAME] [--asr-backend NAME] [--no_asr]` | 启动 Gradio WebUI |
| `uv run audioclone-api [--host 0.0.0.0] [--port 8000]` | 启动 FastAPI |
| `uv run audioclone-download-tts [--local_dir X]` | 下载 TTS 模型 |
| `uv run audioclone-download-asr [--local_dir X]` | 下载 ASR 模型 |
| `uv run pytest` | 运行测试（集成默认跳过） |
| `uv run pytest -m "not integration"` | 仅单元/架构测试 |
| `uv run ruff check .` | 静态检查 |

---

## 7. API 参考

Base URL：`http://localhost:8000`

### 7.1 健康检查

```
GET /api/health
```
示例响应：
```json
{
  "status": "ok",
  "framework": "MLX",
  "model": "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16",
  "sample_rate": 24000,
  "model_loaded": true
}
```

### 7.2 声音克隆 / 合成

```
POST /api/clone
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `text` | string | ✓ | 合成文本（≤300 字符） |
| `ref_audio` | file | ✓ | 参考音频文件 |
| `ref_text` | string | | 参考音频对应文字 |
| `speed` | float | | 语速 0.5~2.0，默认 1.0 |
| `mode` | string | | `zero_shot` / `cross_lingual` / `instruct` |
| `instruct_text` | string | | 指令（instruct 模式必填） |

curl 示例：
```bash
curl -X POST http://localhost:8000/api/clone \
  -F "text=你好，这是一段测试语音。" \
  -F "ref_audio=@asset/zero_shot_prompt.wav" \
  -F "ref_text=希望你以后能够做的比我还好呦。" \
  -F "mode=zero_shot" \
  -D - -o out.wav
```

响应：WAV 音频流，Header 含 `X-Synthesis-Time`（秒）。

---

## 8. 在代码中使用（Python SDK）

```python
from audioclone.core import TTSService, create_tts_backend

# ① 创建后端并换装
backend = create_tts_backend("cosyvoice3-mlx")   # 或 "dummy-omni"
svc = TTSService(backend)                         # 构造即做能力校验

# ② 端到端（分段 + 落盘）
path = svc.synthesize_to_file(
    mode="zero_shot",
    text="你好，欢迎使用本工具箱。",
    ref_audio_path="asset/zero_shot_prompt.wav",
    ref_text="希望你以后能够做的比我还好呦。",
    filename="demo.wav",
)

# ③ 直接取数组
import numpy as np
audio: np.ndarray = svc.infer_zero_shot("你好", "asset/zero_shot_prompt.wav")
```

多任务（全能力后端）示例：
```python
from audioclone.core import AudioClassificationService, VoiceActivityService, create_tts_backend
m = create_tts_backend("dummy-omni")
print(AudioClassificationService(m).classify("asset/zero_shot_prompt.wav", top_k=3))
print(VoiceActivityService(m).vad("asset/zero_shot_prompt.wav"))
```

查询后端能力：
```python
from audioclone.core import list_backend_capabilities
print(list_backend_capabilities(m))   # ['text-to-speech', ..., 'voice-activity-detection']
```

---

## 9. 扩展开发

### 9.1 接入新后端（复用既有能力）
1. 新建 `core/backends/my_model.py`，继承 `ModelBackend` + 一个或多个能力接口。
   - TTS：实现 `synthesize` 的接口方法
   - ASR：实现 `transcribe` +（可选生命周期 `start/stop/is_ready`）
2. 在 `config.py` 登记 `factory` 路径。
3. 使用 `uv run audioclone-webui --tts-backend my-model` 启用。

### 9.2 新增能力（新任务）
1. 在 `core/capabilities/` 新增 `*Capability` 接口。
2. 在 `core/service/` 新增对应 `*Service`（构造做 `isinstance` 能力校验）。
3. 在 `model_loader.CAPABILITY_LABELS` 登记可读名。
4. 后端继承该能力即可；WebUI/API 按需接入。

---

## 10. 测试

```bash
uv run pytest -m "not integration"   # 单元/架构（无需模型）
uv run pytest                        # 含集成（缺模型自动跳过）
```

集成测试（`tests/test_tts.py` / `test_asr.py` / `test_roundtrip.py`）需先下载模型并保证 `llama-server` 可用。

---

## 11. 常见问题与故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 启动 WebUI 报 `ImportError: Using SOCKS proxy... socksio` | 环境含 `ALL_PROXY=socks5://` | `unset ALL_PROXY all_proxy` 后再启动，或安装 `httpx[socks]` |
| WebUI 端口占用 503 / 连接失败 | 默认端口被占/本地代理 | 改 `--port`；确认 `no_proxy=localhost` 已设置 |
| "ASR 模型文件未找到" | 未下载 ASR 模型或未装 llama.cpp | `uv run audioclone-download-asr`；`brew install llama.cpp` |
| TTS 加载慢（30-60s） | 首次冷加载模型到 Metal | 属正常；等待状态面板变 ✅ |
| ASR `RuntimeError: ASR 服务未就绪` | llama-server 未起/启动失败 | 查看日志、检查端口 8090、确认模型文件完整 |
| `pytest` 集成用例被跳过 | 缺模型 | 属预期（骨架已在，待模型即跑） |
| 部署到多用户 | 无鉴权、绑 0.0.0.0 | 用反向代理 + BasicAuth/OAuth；勿直接暴露公网 |

---

## 12. 生产与版本

- **版本**：当前仓库基线 `0.1.1`；本文档面向 **v0.2.0**（见 `QUALITY-REPORT.md` 的 P0/P1 清单）。
- **生产建议**：反向代理 + TLS + 鉴权 + 限流；上传大小与音频时长校验；模型与服务常驻管理。
- **可追溯**：架构决策见 `docs/REFACTOR-001-audio-toolbox.md`（ADR-001~004 + 能力演进）。