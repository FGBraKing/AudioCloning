# USER-GUIDE — AudioCloning 音频模型体验工具箱 · 使用说明

| 元信息 | 值 |
|--------|----|
| 文档版本 | 1.5 |
| **适用版本** | **v0.3.0** |
| 文档类型 | 使用说明（安装 / 配置 / 能力使用 / API / 扩展 / 故障排查 / 部署） |
| 创建日期 | 2026-08-25（v0.2.6 修订 2026-08-26；v0.3.0 修订 2026-10-25） |
| 配套文档 | `QUALITY-REPORT.md`、`REFACTOR-001-audio-toolbox.md`、`README.md` |

---

## 1. 这是什么

一个**能力驱动、模型可替换**的本地音频模型体验工具箱（Apple Silicon + MLX）。
框架提供 6 种音频任务能力：**TTS / Text-to-Audio / ASR / Audio-to-Audio / 音频分类 / VAD**，
每种能力都有独立服务层（Service），按能力校验后端。内置真实后端（Qwen3-TTS-MLX 三变体、Qwen3-ASR/SenseVoice/Kokoro/Silero 的 MLX 与 sherpa-onnx 实现）
与全能力演示后端（dummy-omni）。

```
展示层(WebUI/API) → 服务层(Service) → 能力层(Capability) ← 实现层(后端)
```

v0.2.1 相比 v0.2.0 的加固点：领域异常体系、pydantic 配置、registry 收敛、能力抽象契约、
音频数值安全、因果重采样、ASR 超时与日志重定向、FastAPI 线程池、上传限流、MLX 串行锁、
模型生命周期统一、契约/API/音频工具测试与覆盖率门槛、mypy 类型检查。

v0.2.3（本版文档）相比 v0.2.2 的修订（按 `QUALITY-REPORT.md` §9）：**版本单一来源**（`__version__=0.2.3`，pyproject 动态读取）、
**registry 单池化**（按名注册 + 按能力查询；`dummy` 按任务别名解析）、采样率实时读取、ASR 分片重叠（`AUDIOCLONE_ASR_CHUNK_OVERLAP_SEC`）、
`prune_outputs` 递归清理（含 `segments/`）、**trace_id 贯穿服务层**、未就绪 → 503、多任务输入校验、
`pyloudnorm` 可选 extra、metrics 直方图、CI 去兜底 + benchmark 冒烟。

---

## 2. 系统要求

- **macOS** (Apple Silicon)：推理框架 MLX
- **Linux / Windows**：推理框架 onnx（sherpa-onnx，可选组 `uv sync --extra onnx`）；NVIDIA GPU 平台 TTS 走 TensorRT Edge-LLM（引擎需自行构建）
- Python 3.13（由 uv 管理，`.python-version` 固定）+ [uv](https://docs.astral.sh/uv/)
- 可选真实模型：Qwen3-TTS MLX（三变体）、Qwen3-ASR/Kokoro/Silero 的 MLX 或 onnx 版（下载见 §6）

---

## 3. 安装

```bash
# 1) uv（若未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) 进入项目并同步依赖（自动选 .python-version=3.13，生成 .venv + uv.lock）
cd AudioCloning
uv sync

# 3) 可选：CPU / AMD GPU 平台的 sherpa-onnx 推理（ASR/TTS/VAD）
uv sync --extra onnx
```

---

## 4. 配置

### 4.1 环境变量（运行时，`pydantic-settings`，前缀 `AUDIOCLONE_`）

| 变量 | 默认 | 说明 |
|------|------|------|
| `AUDIOCLONE_TTS_BACKEND` | `qwen3-tts-base-mlx` | 默认 TTS 后端 |
| `AUDIOCLONE_ASR_BACKEND` | `auto` | 默认 ASR 后端 |
| `AUDIOCLONE_SAMPLE_RATE` | `24000` | TTS 输出采样率 |
| `AUDIOCLONE_REF_AUDIO_MAX_SEC` / `MIN_SEC` / `MIN_SR` | `30` / `1` / `8000` | 参考音频校验 |
| `AUDIOCLONE_MAX_TEXT_LENGTH` | `300` | 单次合成最大字符数 |
| `AUDIOCLONE_WEBUI_PORT` / `HOST` / `CONCURRENCY` | `7860` / `127.0.0.1` / `2` | WebUI 参数（CONCURRENCY 为推理并发上限） |
| `AUDIOCLONE_ASR_AUDIO_MAX_SEC` / `SAMPLE_RATE` / `MAX_SEGMENT_SEC` | `300` / `16000` / `60` | ASR 时长/采样率/分片阈值 |
| `AUDIOCLONE_ASR_HTTP_TIMEOUT` / `READY_TIMEOUT` | `120.0` / `60.0` | ASR HTTP / 就绪等待超时（秒） |
| `AUDIOCLONE_MAX_UPLOAD_BYTES` | `104857600`(100MB) | API 上传音频大小上限 |
| `AUDIOCLONE_MAX_INLINE_UPLOAD_BYTES` | `8388608`(8MB) | 小文件一次读入内存的阈值（减少分块 IO） |
| `AUDIOCLONE_OUTPUT_KEEP_MAX` | `0` | 输出目录保留文件数（`<=0` 不清理） |
| `AUDIOCLONE_TTS_NORMALIZE_LOUDNESS` | `false` | 落盘前按 EBU R128 响度归一化 |
| `AUDIOCLONE_TTS_SEGMENT_GAP_SEC` | `0.15` | 长文本分段间插入静音（秒） |
| `AUDIOCLONE_LOG_JSON` | `false` | 启用单行 JSON 结构化日志 |
| `AUDIOCLONE_TTS_MODEL_DIR` | 无（HF 系统缓存） | 自定义 TTS 模型目录；默认用 HuggingFace 缓存（`~/.cache/huggingface/hub`） |
| `AUDIOCLONE_ASR_MODEL_DIR` | 无（HF 系统缓存） | 自定义 ASR 模型目录（含 `.gguf`）；默认用 HuggingFace 缓存 |
| `AUDIOCLONE_MIN_MLX_MEM_GB` | `8.0` | Apple Silicon 统一内存最低要求；低于则推理框架降级 onnx |
| `AUDIOCLONE_MIN_CUDA_VRAM_GB` | `8.0` | NVIDIA **可用**显存最低要求（总量−其他进程占用，多卡取可用最大的一张）；低于则降级 onnx |
| `AUDIOCLONE_ASR_NUM_THREADS` | `2` | sherpa-onnx（onnxruntime）推理线程数 |
| `AUDIOCLONE_TENSORRT_ENGINE_DIR` | 无 | TensorRT Edge-LLM 引擎目录（`qwen3-tts-*-tensorrt` 后端专用，与模型权重目录无关） |
| `AUDIOCLONE_HF_ENDPOINT` | 无（官方源） | 自定义 HuggingFace 镜像端点（如 `https://hf-mirror.com`）；也可用标准 `HF_ENDPOINT` |
| `no_proxy` / `NO_PROXY` | `localhost,127.0.0.1`（入口自动设置） | 绕过本地代理 |

完整示例见项目根 `.env.example`；也可直接复制为 `.env`（`pydantic-settings` 自动加载）。

### 4.2 后端注册表（`core/backends/registry.py`）

```python
# core/backends/registry.py —— 工厂唯一事实源
registry.register("my-tts", MyTTSBackend)

# config.py —— 默认选择 + 覆盖参数（可选）
TTS_BACKENDS["my-tts"] = {"model_id": "my-org/my-model"}
```

> 内置后端已在 `core/backends/registry.py` 预注册（单池按名注册）；config 不再承载 `factory` 字符串。
> `dummy` 为任务便捷别名，由 `model_loader` 解析为 `dummy-tts` / `dummy-asr`。

### 4.3 能力矩阵（当前实现）

| 任务 | 能力接口 | Service | 真实后端 | 演示后端 |
|------|----------|---------|:---:|:---:|
| Text-to-Speech | `TextToSpeechCapability` | `TTSService` | ✅ qwen3_tts_mlx（三变体） | dummy / dummy-omni |
| Text-to-Audio | `TextToAudioCapability` | `TextToAudioService` | 待接入 | dummy-omni |
| Speech-Recognition | `SpeechRecognitionCapability` | `ASRService` | ✅ qwen3_asr_mlx / sherpa_asr_onnx | dummy-omni |
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

### 5.3 切换 TTS 后端（含 Qwen3-TTS 三变体）

| 后端名 | 模型 | 支持模式 | 说明 |
|--------|------|----------|------|
| `qwen3-tts-base-mlx` | Qwen3-TTS-12Hz-1.7B-Base-8bit | zero_shot / cross_lingual | **默认后端**；参考音频克隆；给 `ref_text` 走 ICL，不给走 speaker-encoder |
| `qwen3-tts-customvoice-mlx` | Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit | instruct | 预设音色（默认 `Vivian`）+ 情绪/风格指令 |
| `qwen3-tts-voicedesign-mlx` | Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit | instruct | 用自然语言描述音色，无需参考音频 |

```bash
uv run audioclone-download-tts --backend qwen3-tts-base-mlx
uv run audioclone-webui --tts-backend qwen3-tts-base-mlx
```

> **依赖**：三个后端依赖 `mlx_audio` 的 `qwen3_tts` 架构（本项目锁定 `mlx-audio[tts]>=0.5.0`）。
> 依赖缺失时启动会中止并打印修复指引，不会残留半加载状态。
>
> **运行时只加载一个 TTS + 一个 ASR 模型**（核心层槽位守卫强制执行），因此三个变体按需切换：
> 克隆用 `base`，情绪/音色指令用 `customvoice` 或 `voicedesign`。
>
> **模式不匹配**：当前后端不支持的模式，其对应面板会在 WebUI 中**直接停用**
> （显示提示条 + 指引可切换的后端清单）；API 侧则在编排前返回 400，不会进入模型推理。

### 5.4 启动 REST API

```bash
uv run audioclone-api              # http://127.0.0.1:8000（见 §12 部署护栏）
```

> 结构化日志 / trace_id filter 由入口（`audioclone-api` 或 `python -m audioclone.api.server`）初始化；
> 若直接 `uvicorn audioclone.api.server:app` 启动，则不包含上述日志增强（仅功能接口可用）。

---

## 6. 命令参考

| 命令 | 说明 |
|------|------|
| `uv run audioclone-webui [--port 7860] [--host ...] [--model_dir X] [--asr_model_dir X] [--tts-backend NAME] [--asr-backend NAME] [--t2a/a2a/classification/vad-backend NAME] [--no_asr] [--theme dark\|light] [--no-browser]` | 启动 NiceGUI WebUI（含多任务实验台） |
| `uv run audioclone-api [--host 127.0.0.1] [--port 8000]` | 启动 FastAPI |
| `uv run audioclone-download-tts [--local_dir X] [--backend NAME] [--source auto\|hf\|modelscope]` | 下载 TTS 模型（默认 auto：HF 优先失败回退 ModelScope） |
| `uv run audioclone-download-asr [--local_dir X] [--backend NAME] [--source auto\|hf\|modelscope]` | 下载 ASR 模型 |
| `uv run pytest` | 运行测试（集成默认跳过） |
| `uv run pytest -m "not integration"` | 单元/架构/契约/API 测试 |
| `uv run pytest --cov=audioclone` | 带覆盖率（门槛 60%） |
| `uv run ruff check .` | 静态检查 |
| `uv run mypy src/audioclone` | 类型检查 |
| `uv run python -m tools.benchmark --backend dummy-omni --iterations 3` | TTS 性能基准（冷启动/单次合成耗时） |
| `uv run python -m tools.eval_quality --ref asset/zero_shot_prompt.wav --synth outputs/test.wav` | 合成质量客观指标（频谱距离，越低越好） |

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
  "framework": "mlx",
  "backend": "qwen3-tts-base-mlx",
  "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
  "supported_modes": ["zero_shot", "cross_lingual"],
  "sample_rate": 24000,
  "model_loaded": true,
  "model_slots": { "tts": "qwen3-tts-base-mlx", "asr": null, "t2a": null, "a2a": null,
                    "classification": null, "vad": null },
  "runtime": { "os": "darwin", "machine": "arm64", "accelerator": "mlx",
                "framework": "mlx", "mem_total_gb": 16.0, "gpu_vram_gb": null,
                "gpu_vram_free_gb": null, "downgraded": false, "downgrade_reason": null }
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
| `ref_audio` | file | ✓（plain 可省略） | 参考音频（≤`AUDIOCLONE_MAX_UPLOAD_BYTES`，格式/时长/采样率校验） |
| `ref_text` | string | | 参考音频对应文字 |
| `speed` | float | | 语速 0.5~2.0，默认 1.0 |
| `mode` | string | | `zero_shot` / `cross_lingual` / `instruct` / `plain`（纯文本直读） |
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

响应：WAV 音频流，Header 含 `X-Synthesis-Time`（秒）。所有响应均含 `X-Request-ID`（trace_id，与请求日志字段对应）。

### 7.3 指标

```
GET /api/metrics
```
Prometheus 文本格式：`audioclone_requests_total{path,status}`、`audioclone_synthesis_count_total`、
`audioclone_synthesis_seconds_total`、`audioclone_synthesis_duration_seconds_bucket{le=...}`（直方图）、
`audioclone_uptime_seconds`。指标为进程内单例，多 worker 部署需按进程聚合（见 `api/metrics.py` 说明）。

### 7.4 历史输出

```
GET /api/outputs?limit=50&include_duration=true
```
按修改时间倒序列出历史输出音频（含 `segments/` 子目录）。`limit` 上限 200（防滥用）；
`include_duration=false` 可跳过逐文件读头以加速。返回 `count` 与 `files`
（名称 / 相对路径 / 字节数 / 修改时间(UTC ISO8601) / 时长秒）。

### 7.5 错误响应

| 场景 | 状态码 | 说明 |
|------|:---:|------|
| 文本为空 / 超长 / 模式非法 / 格式不支持 / 音频非法 | 400 | `InvalidAudioInput` |
| TTS 后端加载失败 / 不具备能力 / 引擎未就绪 | 503 | `ModelLoadError` |
| 合成内部异常 | 500 | 通用错误 |

### 7.6 模型管理

```
GET /api/models            # 六任务模型就绪状态（离线查询：cached / model_path / hf_endpoint）
POST /api/models/download  # 主动下载（后台执行，202；并发 409、未知后端 400）
GET /api/models/download   # 下载任务进度
```

下载请求体：`{"task": "tts|asr|t2a|a2a|classification|vad", "backend": 可选,
"local_dir": 可选, "source": "auto|hf|modelscope"}`（默认 auto：HF 优先，失败自动回退
ModelScope；镜像端点 `AUDIOCLONE_HF_ENDPOINT`）。

### 7.7 多任务端点

| 接口 | 请求 | 响应 |
|------|------|------|
| `POST /api/t2a` | Form：`text`（必填）、`duration_sec`、`lyrics` | WAV 流，Header `X-Generation-Time` |
| `POST /api/transform` | multipart：`audio`、`instruct` | WAV 流，Header `X-Transform-Time` |
| `POST /api/classify` | multipart：`audio`、`top_k` | JSON `{"results": [{"label", "score"}]}`（情感 9 类 + 事件 4 类） |
| `POST /api/vad` | multipart：`audio` | JSON `{"segments": [{"start", "end", "duration"}], "speech_sec"}` |

四个端点的模型均在**首次调用时惰性加载**；模型未下载时返回 503 并附下载指引。

---

## 8. 在代码中使用（Python SDK）

```python
from audioclone.core import TTSService, create_tts_backend

backend = create_tts_backend("qwen3-tts-base-mlx")   # 或 "dummy-omni"
svc = TTSService(backend)                         # 构造即能力校验，不匹配抛 CapabilityError

path = svc.synthesize_to_file(
    mode="zero_shot",
    text="你好，欢迎使用本工具箱。",
    ref_audio_path="asset/zero_shot_prompt.wav",
    ref_text="希望你以后能够做的比我还好呦。",
    filename="demo.wav",
    progress_cb=lambda i, n: print(f"段 {i}/{n}"),  # 进度回调
    keep_segments=True,                              # 逐段落盘到 outputs/segments/
)  # 落盘前自动 sanitize_audio；AUDIOCLONE_TTS_NORMALIZE_LOUDNESS=1 可选响度归一化

import numpy as np
audio: np.ndarray = svc.infer_zero_shot("你好", "asset/zero_shot_prompt.wav")  # RLock 串行
```

> **模型下载**：WebUI 总览页有「模型下载」卡片，六任务状态一目了然、可一键下载
> （调用与 CLI/API 共用同一 `model_manager` 实现）。

多任务示例（推荐：`create_service(kind)` 六任务统一入口，后端按环境 auto 选择）：
```python
from audioclone.core import (
    AudioClassificationService, VoiceActivityService, create_service, create_tts_backend,
)
cls = create_service("classification")   # mac → sensevoice-mlx；CPU → dummy（模型未接入时）
print(cls.classify("asset/zero_shot_prompt.wav", top_k=3))
print(create_service("vad").vad("asset/zero_shot_prompt.wav"))

# 也可显式建 dummy-omni 全能力后端体验链路（无需模型）
m = create_tts_backend("dummy-omni")
print(VoiceActivityService(m).vad("asset/zero_shot_prompt.wav"))
```

模型下载（与 CLI/API 共用实现）：
```python
from audioclone.core import model_manager
print(model_manager.model_status("asr"))            # cached / model_path / hf_endpoint
model_manager.download_model("asr")                  # 主动下载（阻塞；API 为后台线程版）
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
2. 在 `core/backends/registry.py` 注册（工厂唯一来源）；`config.py` 登记覆盖参数
   （`*_BACKENDS` 表；含 `model_id` 时 `/api/models` 与下载接口自动管理该后端）。
3. 在 `core/runtime.py` 的 `FRAMEWORK_BACKEND_CANDIDATES` 候选链加入后端名
   （`auto` 选择的依据；显式指定后端时不需要）。
4. 使用 `uv run audioclone-webui --tts-backend my-model` 启用。

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
uv run pytest --cov=audioclone                     # 覆盖率（门槛 60%，实测 ~76%）
uv run ruff check .                                # 静态
uv run mypy src/audioclone                         # 类型（0 errors，全量 39 文件）
uv run pytest                                      # 含集成（缺模型自动跳过）
```

> 工程化：`make setup/lint/type/test/cov` 快捷命令；`uv tool run pre-commit install` 启用提交钩子（ruff+mypy+pytest）。

集成测试（`tests/test_tts.py` / `test_roundtrip.py`）需先下载模型（ASR 需 `uv sync --extra onnx`）。

---

## 11. 常见问题与故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 启动 WebUI 报 `ImportError: Using SOCKS proxy... socksio` | 环境含 `ALL_PROXY=socks5://` | `unset ALL_PROXY all_proxy` 后启动，或安装 `httpx[socks]` |
| WebUI 端口占用 503 / 连接失败 | 端口被占/本地代理 | 改 `--port`；确认 `no_proxy=localhost` 已设 |
| "ASR 模型未下载/未就位" | 模型未下载或未装 sherpa-onnx | `uv sync --extra onnx` + `uv run audioclone-download-asr`（或 `POST /api/models/download`） |
| TensorRT TTS "需提供引擎目录" | Edge-LLM 引擎未构建/未指定 | 按 README「NVIDIA 平台」章节 export+build 后 `--model_dir` 指向引擎目录 |
| API 返回 400 上传过大 | 超过 `AUDIOCLONE_MAX_UPLOAD_BYTES` | 调大 env 或压缩音频 |
| `CapabilityError` | Service 接错后端 | 选用具备对应能力的能力后端（如 dummy-omni） |
| 合成偶发异常 | 并发/模型不稳定 | `TTSService` 已 RLock 串行；重试即可 |

---

## 12. 生产与部署护栏（重要）

- **版本**：当前仓库基线 `0.3.0`（单一来源 `audioclone.__version__`）；本使用说明适用版本 `v0.3.0`；质量报告见 `QUALITY-REPORT.md`。
- **绑定地址**：本地体验用 `--host 127.0.0.1`（**默认已仅本机绑定**）；**不要**直接以 `0.0.0.0` 暴露公网。
- **反向代理 + TLS**：对外服务置于 Nginx/Caddy 后并以 HTTPS 提供。
- **鉴权**：在反向代理层加 BasicAuth / OAuth；API 建议 `limit_req` 限流。
- **上传**：API 已内置大小上限与格式/时长/采样率校验。
- **指标/输出接口**：`/api/metrics` 与 `/api/outputs` 无内建鉴权，公网部署建议随反向代理 AuthN 保护。
- **常驻**：systemd/launchd 管理；优雅退出自动释放后端资源（lifespan/`finally` close）。
- **可追溯**：架构决策见 `docs/REFACTOR-001-audio-toolbox.md`（ADR-001~004 + 能力演进）；变更记录见 `CHANGELOG.md`。