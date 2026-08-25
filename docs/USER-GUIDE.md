# USER-GUIDE — AudioCloning 音频模型体验工具箱 · 使用说明

| 元信息 | 值 |
|--------|----|
| 文档版本 | 1.0 |
| **适用版本** | **v0.2.1** |
| 文档类型 | 使用说明（安装 / 配置 / 能力使用 / API / 扩展 / 故障排查 / 部署） |
| 创建日期 | 2026-08-25 |
| 配套文档 | `QUALITY-REPORT.md`、`REFACTOR-001-audio-toolbox.md`、`README.md` |

---

## 1. 这是什么

一个**能力驱动、模型可替换**的本地音频模型体验工具箱（Apple Silicon + MLX）。
框架提供 6 种音频任务能力：**TTS / Text-to-Audio / ASR / Audio-to-Audio / 音频分类 / VAD**，
每种能力都有独立服务层（Service），按能力校验后端。内置真实后端（CosyVoice3-MLX、Qwen3-ASR-llama.cpp）
与全能力演示后端（dummy-omni）。

```
展示层(WebUI/API) → 服务层(Service) → 能力层(Capability) ← 实现层(后端)
```

v0.2.1 相比 v0.2.0 的加固点：领域异常体系、pydantic 配置、registry 收敛、能力抽象契约、
音频数值安全、因果重采样、ASR 超时与日志重定向、FastAPI 线程池、上传限流、MLX 串行锁、
模型生命周期统一、契约/API/音频工具测试与覆盖率门槛、mypy 类型检查。

---

## 2. 系统要求

- Apple Silicon Mac（M1/M2/M3/M4）、macOS 13+
- Python 3.13（由 uv 管理，`.python-version` 固定）
- [uv](https://docs.astral.sh/uv/)
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

---

## 4. 配置

### 4.1 环境变量（运行时，`pydantic-settings`，前缀 `AUDIOCLONE_`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `AUDIOCLONE_TTS_BACKEND` | `cosyvoice3-mlx` | 默认 TTS 后端 |
| `AUDIOCLONE_ASR_BACKEND` | `qwen3-asr-llama` | 默认 ASR 后端 |
| `AUDIOCLONE_MAX_UPLOAD_BYTES` | `104857600`(100MB) | API 上传音频大小上限 |
| `AUDIOCLONE_ASR_HTTP_TIMEOUT` | `120.0` | ASR 识别 HTTP 超时（秒） |
| `AUDIOCLONE_ASR_READY_TIMEOUT` | `60.0` | llama-server 就绪等待超时（秒） |
| `AUDIOCLONE_OUTPUT_KEEP_MAX` | `0` | 输出目录保留文件数（`<=0` 不清理） |
| `COSYVOICE_MODEL_DIR` | `pretrained_models/...` | API 服务覆盖 TTS 模型目录 |
| `no_proxy` / `NO_PROXY` | `localhost,127.0.0.1`（入口自动设置） | 绕过本地代理 |

也可在项目根放 `.env`（`pydantic-settings` 自动加载）。

### 4.2 后端注册表（`src/audioclone/config.py`）

```python
TTS_BACKENDS["my-tts"] = {
    "factory": "audioclone.core.backends.my_tts.MyTTSBackend",  # module.Class
    "model_id": "my-org/my-model",
}
```

> 内置后端已在 `core/backends/registry.py` 预注册；`model_loader` 优先从 registry 解析，
> 未注册时才回退 config 的 factory 字符串（受控 dynamic import）。

### 4.3 能力矩阵（当前实现）

| 任务 | 能力接口 | Service | 真实后端 | 演示后端 |
|------|----------|---------|:---:|:---:|
| Text-to-Speech | `TextToSpeechCapability` | `TTSService` | ✅ cosyvoice_mlx | dummy / dummy-omni |
| Text-to-Audio | `TextToAudioCapability` | `TextToAudioService` | 待接入 | dummy-omni |
| Speech-Recognition | `SpeechRecognitionCapability` | `ASRService` | ✅ qwen3_asr_llama | dummy-omni |
| Audio-to-Audio | `AudioToAudioCapability` | `AudioToAudioService` | 待接入 | dummy-omni |
| Audio-Classification | `AudioClassificationCapability` | `AudioClassificationService` | 待接入 | dummy-omni |
| Voice-Activity-Detection | `VoiceActivityCapability` | `VoiceActivityService` | 待接入 | dummy-omni |

> 能力接口为 `ABC` + `@abstractmethod`：后端若未实现对应能力方法将**在实例化时即报错**（非运行时）。

---

## 5. 快速上手

### 5.1 真实模型启动 WebUI

```bash
uv run audioclone-download-tts
uv run audioclone-download-asr

uv run audioclone-webui            # http://localhost:7860
uv run audioclone-webui --no_asr   # 禁用 ASR
```

### 5.2 无模型快速体验（dummy，全能力）

```bash
uv run audioclone-webui --tts-backend dummy-omni --asr-backend dummy-omni
```

### 5.3 启动 REST API

```bash
uv run audioclone-api              # http://127.0.0.1:8000（见 §12 部署护栏）
```

---

## 6. 命令参考

| 命令 | 说明 |
|------|------|
| `uv run audioclone-webui [--port 7860] [--host ...] [--model_dir X] [--asr_model_dir X] [--tts-backend NAME] [--asr-backend NAME] [--no_asr]` | 启动 Gradio WebUI |
| `uv run audioclone-api [--host 127.0.0.1] [--port 8000]` | 启动 FastAPI |
| `uv run audioclone-download-tts [--local_dir X]` | 下载 TTS 模型 |
| `uv run audioclone-download-asr [--local_dir X]` | 下载 ASR 模型 |
| `uv run pytest` | 运行测试（集成默认跳过） |
| `uv run pytest -m "not integration"` | 单元/架构/契约/API 测试 |
| `uv run pytest --cov=audioclone` | 带覆盖率（门槛 60%） |
| `uv run ruff check .` | 静态检查 |
| `uv run mypy src/audioclone` | 类型检查 |

---

## 7. API 参考

Base URL：`http://127.0.0.1:8000`

### 7.1 健康检查

```
GET /api/health
```
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
| `ref_audio` | file | ✓ | 参考音频（≤`AUDIOCLONE_MAX_UPLOAD_BYTES`，格式/时长/采样率校验） |
| `ref_text` | string | | 参考音频对应文字 |
| `speed` | float | | 语速 0.5~2.0，默认 1.0 |
| `mode` | string | | `zero_shot` / `cross_lingual` / `instruct` |
| `instruct_text` | string | | 指令（instruct 模式必填） |

curl 示例：
```bash
curl -X POST http://127.0.0.1:8000/api/clone \
  -F "text=你好，这是一段测试语音。" \
  -F "ref_audio=@asset/zero_shot_prompt.wav" \
  -F "ref_text=希望你以后能够做的比我还好呦。" \
  -F "mode=zero_shot" \
  -D - -o out.wav
```

响应：WAV 音频流，Header 含 `X-Synthesis-Time`（秒）。

### 7.3 错误响应

| 场景 | 状态码 | 说明 |
|------|:---:|------|
| 文本为空 / 超长 / 模式非法 / 格式不支持 / 音频非法 | 400 | `InvalidAudioInput` |
| TTS 后端加载失败 / 不具备能力 | 503 | `ModelLoadError` |
| 合成内部异常 | 500 | 通用错误 |

---

## 8. 在代码中使用（Python SDK）

```python
from audioclone.core import TTSService, create_tts_backend

backend = create_tts_backend("cosyvoice3-mlx")   # 或 "dummy-omni"
svc = TTSService(backend)                         # 构造即能力校验，不匹配抛 CapabilityError

path = svc.synthesize_to_file(
    mode="zero_shot",
    text="你好，欢迎使用本工具箱。",
    ref_audio_path="asset/zero_shot_prompt.wav",
    ref_text="希望你以后能够做的比我还好呦。",
    filename="demo.wav",
)  # 落盘前自动 sanitize_audio；若 OUTPUT_KEEP_MAX>0 会自动清理旧输出

import numpy as np
audio: np.ndarray = svc.infer_zero_shot("你好", "asset/zero_shot_prompt.wav")  # RLock 串行
```

多任务（全能力后端）示例：
```python
from audioclone.core import (
    AudioClassificationService, VoiceActivityService, create_tts_backend,
)
m = create_tts_backend("dummy-omni")
print(AudioClassificationService(m).classify("asset/zero_shot_prompt.wav", top_k=3))
print(VoiceActivityService(m).vad("asset/zero_shot_prompt.wav"))
```

查询后端能力：
```python
from audioclone.core import list_backend_capabilities
print(list_backend_capabilities(m))
# ['text-to-speech', 'text-to-audio', 'speech-recognition', 'audio-to-audio',
#  'audio-classification', 'voice-activity-detection']
```

---

## 9. 扩展开发

### 9.1 接入新后端（复用既有能力）
1. 新建 `core/backends/my_model.py`，继承 `ModelBackend` + 一个或多个能力接口
   （未实现能力方法会构造失败，提示你补全）。
2. 在 `core/backends/registry.py` 预注册（推荐），或 `config.py` 登记 `factory`。
3. 使用 `uv run audioclone-webui --tts-backend my-model` 启用。

### 9.2 新增能力（新任务）
1. 在 `core/capabilities/` 新增 `*Capability`（`Capability, ABC` + `@abstractmethod`）。
2. 在 `core/service/` 新增对应 `*Service`（构造做能力校验，抛 `CapabilityError`）。
3. 在 `model_loader.CAPABILITY_LABELS` 登记可读名。
4. 后端继承该能力即可；WebUI/API 按需接入。

### 9.3 领域异常
所有业务错误继承 `audioclone.core.exceptions.AudioError`：
`InvalidAudioInput`（400）/ `CapabilityError` / `BackendUnavailable`（503）/ `ModelLoadError`（503）。
API 已注册统一 `exception_handler`。

---

## 10. 测试与质量

```bash
uv run pytest -m "not integration"                 # 快速：单元/架构/契约/API
uv run pytest --cov=audioclone                     # 覆盖率（门槛 60%，实测 ~65%）
uv run ruff check .                                # 静态
uv run mypy src/audioclone                         # 类型（0 errors）
uv run pytest                                      # 含集成（缺模型自动跳过）
```

集成测试（`tests/test_tts.py` / `test_asr.py` / `test_roundtrip.py`）需先下载模型并保证 `llama-server` 可用。

---

## 11. 常见问题与故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 启动 WebUI 报 `ImportError: Using SOCKS proxy... socksio` | 环境含 `ALL_PROXY=socks5://` | `unset ALL_PROXY all_proxy` 后启动，或安装 `httpx[socks]` |
| WebUI 端口占用 503 / 连接失败 | 端口被占/本地代理 | 改 `--port`；确认 `no_proxy=localhost` 已设 |
| "ASR 模型文件未找到" | 未下载 ASR 模型 / 未装 llama.cpp | `uv run audioclone-download-asr`；`brew install llama.cpp` |
| ASR `RuntimeError: ASR 服务未就绪` | llama-server 未起/启动失败 | 查 `outputs/asr_llama_server.log`、端口 8090、模型完整性 |
| API 返回 400 上传过大 | 超过 `AUDIOCLONE_MAX_UPLOAD_BYTES` | 调大 env 或压缩音频 |
| `CapabilityError` | Service 接错后端 | 选用具备对应能力的能力后端（如 dummy-omni） |
| 合成偶发异常 | 并发/模型不稳定 | `TTSService` 已 RLock 串行；重试即可 |

---

## 12. 生产与部署护栏（重要）

- **版本**：当前仓库基线 `0.2.1`；质量报告见 `QUALITY-REPORT.md`。
- **绑定地址**：本地体验用 `--host 127.0.0.1`；**不要**直接以 `0.0.0.0` 暴露公网。
- **反向代理 + TLS**：对外服务置于 Nginx/Caddy 后并以 HTTPS 提供。
- **鉴权**：在反向代理层加 BasicAuth / OAuth；API 建议 `limit_req` 限流。
- **上传**：API 已内置大小上限与格式/时长/采样率校验。
- **常驻**：systemd/launchd 管理；优雅退出自动释放后端资源（lifespan/`finally` close）。
- **可追溯**：架构决策见 `docs/REFACTOR-001-audio-toolbox.md`（ADR-001~004 + 能力演进）。