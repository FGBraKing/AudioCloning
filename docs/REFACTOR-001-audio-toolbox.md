# REFACTOR-001 — 音频模型体验工具箱：模型可替换重构方案

| 元信息 | 值 |
|--------|----|
| 文档版本 | 1.0 |
| 状态 | DRAFT（待评审） |
| 作者 | AudioCloning 团队 |
| 创建日期 | 2026-08-25 |
| 重构前版本 | v0.1.0 |
| 重构后版本 | v0.1.1 |
| 关联代码 | 见「实施步骤」章节的文件映射 |

---

## 1. 背景与动机

当前 `audioclone` 是一个**个人声音克隆系统**，但核心引擎与具体模型强耦合：

- `core/tts.py::TTSEngine` 的推理方法直接调用 `mlx_audio` 的 `model.generate(...)`，业务流程（分段、拼接、落盘）与具体模型实现挤在同一个类里。
- `core/asr.py::ASREngine` 直接内嵌 `llama-server` 子进程管理与 Qwen3-ASR 的 HTTP 协议。
- `webui/`、`api/`、`cli/` 全部直接消费 `TTSEngine` / `ASREngine` / `load_model` 上层 API。

**目标转型**：将项目从「固定模型的克隆系统」改造为**音频模型体验工具箱（Audio Model Experience Toolbox）**——
功能框架（WebUI / REST API / 长文本分段 / 音频工具）保持不变，但**底层模型可替换、可插拔**，
既可为不同模型做体验演示，也可隔离模型升级对上层界面的影响。

---

## 2. 目标与非目标

### 2.1 目标

- **模型可替换**：通过统一接口 + 配置即可切换 TTS / ASR 后端，无需改动上层框架。
- **业务抽离**：把「流程逻辑」与「模型具体实现」解耦为两层。
- **框架不变**：对外命令（`audioclone-webui` / `audioclone-api` / `audioclone-download-*`）与 WebUI/API 行为保持不变。
- **可测试**：用内置 dummy 后端做框架级测试（不依赖真实模型与 GPU）。
- **可维护**：新增后端 = 追加一个实现类 + 一次注册，不触碰业务层。

### 2.2 非目标（本轮不做）

- 不重命名顶层命令 / 包名（`audioclone` 保留，避免破坏既有集成；重命名列为可选建议见 §7.5）。
- 不引入第三方 DI 框架（依赖注入保持轻量，用工厂 + 配置）。
- 不改造 WebUI/API 的交互细节或路由，仅保证其对稳定的服务层接口依赖。

---

## 3. 现状评审（耦合点清单）

| 文件 | 现有职责 | 耦合问题 |
|------|----------|----------|
| `core/tts.py` | `TTSEngine`：分段→逐段合成→拼接→落盘；`load_model` | 业务流程与 `mlx_audio.model.generate` 强绑定；`TTSEngine` 同时承担「服务」与「后端」两职 |
| `core/asr.py` | `ASREngine`：`llama-server` 子进程生命周期 + HTTP 识别 | 子进程管理与模型协议耦合在内；仅适用于 Qwen3-ASR + llama.cpp |
| `core/text.py` | `split_long_text` / `normalize_text` | 纯业务，**无耦合**，本轮保持不变 |
| `core/__init__.py` | 导出 `TTSEngine` / `ASREngine` / `load_model` | 直接导出具体类，需改为导出抽象接口 + 工厂 |
| `webui/app.py` `handlers.py` | 直接调用 `TTSEngine.infer_*`、`ASREngine.is_ready/transcribe` | 依赖具体类的方法签名 |
| `api/server.py` | `lifespan` 内 `load_model()` + `TTSEngine` | 依赖具体模型加载接口 |
| `cli/downloads.py` | 硬编码 `MODEL_ID` / `ASR_MODEL_ID` 下载 | 下载命令未后端化 |
| `config.py` | 硬编码 `MODEL_ID` 等模型常量 | 缺少「选择后端」的配置位 |

---

## 4. 目标架构（分层设计与职责）

```
┌─────────────────────────────────────────────────────────────┐
│ 展示层（框架，保持不变）                                          │
│   webui/ (Gradio)   api/ (FastAPI)                        │
│   ── 仅依赖 service 层接口，不感知具体模型                       │
├─────────────────────────────────────────────────────────────┤
│ 服务/编排层（业务抽离后，稳定）                                     │
│   core/service/tts_service.py   -> TTSService（流程：分段/拼接/落盘）│
│   core/service/asr_service.py   -> ASRService（流程：格式/时长校验） │
│   core/service/model_loader.py  -> 后端加载与生命周期               │
├─────────────────────────────────────────────────────────────┤
│ 模型抽象层（接口契约）                                              │
│   core/backends/base.py         -> BackendSpec           │
│                                -> TTSBackend / ASRBackend │
│   core/backends/registry.py     -> BackendRegistry + 工厂   │
├─────────────────────────────────────────────────────────────┤
│ 模型实现层（可替换、可插拔）                                       │
│   core/backends/cosyvoice_mlx/  -> MLX CosyVoice3 后端       │
│   core/backends/qwen3_asr_llama/-> Qwen3-ASR llama.cpp 后端 │
│   core/backends/dummy/          -> 假后端（框架测试用）          │
├─────────────────────────────────────────────────────────────┤
│ 支撑层                                                       │
│   config.py（后端选择/参数）  utils/（音频工具）  core/text.py    │
└─────────────────────────────────────────────────────────────┘
```

### 职责划分

- **服务层（不变）**：长文本分段、参考音频校验/重采样、逐段合成编排、音频拼接与落盘、ASR 格式/时长校验——这些与「用哪个模型」无关，全部保留在服务层。
- **抽象层（契约）**：定义后端必须实现的方法签名。
- **实现层（可替换）**：真正的模型调用。新增模型只需新增实现类并完成注册。

---

## 5. 接口契约（关键抽象）

### 5.1 后端规格与工厂

```python
# core/backends/registry.py
class BackendRegistry:
    """TTS / ASR 后端的注册表，支持按名称惰性创建。"""

    def register_tts(self, name: str, builder) -> None: ...
    def register_asr(self, name: str, builder) -> None: ...
    def create_tts(self, name: str, **cfg) -> TTSBackend: ...
    def create_asr(self, name: str, **cfg) -> ASRBackend: ...
```

### 5.2 TTS 后端契约

```python
# core/backends/base.py
class TTSBackend(ABC):
    name: str
    sample_rate: int                      # 该模型输出采样率

    @abstractmethod
    def synthesize(
        self, mode: str, text: str, ref_audio_path: str,
        ref_text: str = "", instruct_text: str = "", speed: float = 1.0,
    ) -> np.ndarray:
        """按单段文本合成，返回 float32 音频数组。
        支持 mode: zero_shot / cross_lingual / instruct。
        """

    @abstractmethod
    def load(self) -> None: ...
    def close(self) -> None: ...          # 默认空实现
```

### 5.3 ASR 后端契约

```python
class ASRBackend(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio_path: str) -> str: ...
    def is_ready(self) -> bool: ...       # 默认 True
    def load(self) -> None: ...
    def close(self) -> None: ...
```

### 5.4 现有实现 → 新后端映射

| 现有 | 新位置 | 说明 |
|------|--------|------|
| `TTSEngine.infer_*` | `cosyvoice_mlx.CosyVoiceMLXBackend.synthesize` | 每段一次调用；保留现有 `model.generate` 逻辑 |
| `TTSEngine.synthesize_to_file` | `TTSService` | 流程上移，不再属于模型 |
| `ASREngine.start/stop/transcribe` | `qwen3_asr_llama.Qwen3AsrLlamaBackend` | 子进程管理保留在实现内，对外只暴露 `transcribe/is_ready/close` |
| `load_model()` | `model_loader` + `cosyvoice_mlx` 后端 `load()` | |

---

## 6. 配置化：如何“换模型”

在 `config.py` 中新增两层配置位，形式为字典，便于扩展与 env 覆盖：

```python
TTS_BACKENDS = {
    "cosyvoice3-mlx": {
        "factory": "audioclone.core.backends.cosyvoice_mlx.CosyVoiceMLXBackend",
        "model_id": "mlx-community/Fun-CosyVoice3-0.5B-2512-fp16",
        "model_dir": "pretrained_models/Fun-CosyVoice3-0.5B-2512-fp16",
    },
    "dummy": {"factory": "audioclone.core.backends.dummy.DummyTTSBackend", "sample_rate": 24000},
}
TTS_BACKEND = os.environ.get("AUDIOCLONE_TTS_BACKEND", "cosyvoice3-mlx")

ASR_BACKENDS = {
    "qwen3-asr-llama": {"factory": "audioclone.core.backends.qwen3_asr_llama.Qwen3AsrLlamaBackend", ...},
    "dummy": {"factory": "audioclone.core.backends.dummy.DummyASRBackend"},
}
ASR_BACKEND = os.environ.get("AUDIOCLONE_ASR_BACKEND", "qwen3-asr-llama")
```

**切换示例**：WebUI / API 启动时以配置名实例化后端，业务层零改动。

```bash
uv run audioclone-webui --tts-backend dummy --asr-backend dummy
```

---

## 7. 实施步骤（可追溯迁移清单）

> 每步对应一个独立的、可提交的变更，便于回滚与追溯（配合 `git log --follow`）。

| # | 步骤 | 具体改动文件 | 影响范围 |
|---|------|--------------|----------|
| 1 | 版本号更新至 v0.1.1 | `pyproject.toml`、`src/audioclone/__init__.py`、`api/server.py` 的 `version=` | 元数据 |
| 2 | 新增抽象层 | `core/backends/__init__.py`、`core/backends/base.py`、`core/backends/registry.py` | 新增文件，不动业务 |
| 3 | 迁移 TTS 为服务 + 后端 | 旧 `core/tts.py` 拆为 `core/service/tts_service.py` 与 `core/backends/cosyvoice_mlx/*` | 新增 + 移除旧类 |
| 4 | 迁移 ASR 为服务 + 后端 | `core/asr.py` → `core/service/asr_service.py` + `core/backends/qwen3_asr_llama/*` | 新增 + 移除旧类 |
| 5 | 新增 dummy 后端（含测试） | `core/backends/dummy/*`、`tests/` | 新增，用于框架测试 |
| 6 | 模型加载器与配置化 | `core/service/model_loader.py`、`config.py` | 内部 |
| 7 | 更新导出与消费者 | `core/__init__.py`、`webui/app.py`、`webui/handlers.py`、`api/server.py` | 适配新接口 |
| 8 | 后端化下载命令 | `cli/downloads.py`（按 backend factory 下载对应模型） | CLI |
| 9 | CI/测试更新 | `tests/conftest.py`（按默认后端加载）、`pyproject.toml` 版本 | 验证 |
| 10 | 文档更新 | `README.md`（新增后端扩展说明） | 文档 |

---

## 8. 兼容性与行为不变承诺

- 默认配置仍走 `cosyvoice3-mlx` + `qwen3-asr-llama`，**默认行为与 v0.1.0 完全一致**。
- 命令行入口名不变，原启动/关闭方式不破坏。
- `core/text.py`、`utils/audio.py` 等纯业务模块零改动。
- 旧类 `TTSEngine` / `ASREngine` 在新版本中移除具体模型耦合；任何外部直接引用需按 §7 中的导出更新同步调整（内部消费者全量收敛后，可安全删除旧实现）。

---

## 9. 测试策略

- **单元测试（无需模型/GPU）**：`tests/test_text.py` 保留；新增架构测试——用 `dummy` 后端走完「服务层全流程」（合成→落盘、ASR→文本），验证业务层与具体模型解耦。
- **契约测试**：用 mock/打桩断言 `TTSService` 只调用 `backend.synthesize(...)` 且不引用任何 `mlx`/`llama` 符号。
- **集成测试（需要模型）**：现有 `test_tts.py` / `test_asr.py` / `test_roundtrip.py` 改造为走 `TTSService` + 默认后端，回归验证真实链路。

---

## 10. 验收标准

1. `uv run pytest` （含 dummy 架构测试）全部通过；缺失模型时集成测试自动跳过。
2. `uv run audioclone-webui --tts-backend dummy --asr-backend dummy` 可正常启动且在**未装 / 卸载模型**时仍能完整体验上传、合成、识别流程。
3. 默认后端启动的 WebUI / API 行为与 v0.1.0 一致（回归通过）。
4. `core/` 不再出现在 `tts.py` / `asr.py` 中对 `mlx`、`mlx_audio`、`llama-server` 功能的直接引用（仅后端实现目录保留）。
5. 新增一个任意第三方模型后端时，只新增实现类 + 一行注册，无需改动 WebUI/API。

---

## 11. 决策记录（ADR 摘要）

- **ADR-001**：用「服务层 + 后端接口 + 注册表」而非抽象基类满屏继承。理由：层间依赖方向单一、新增后端成本最低。
- **ADR-002**：业务流程（分段/校验/落盘）放在服务层，后端只做「单段输入 → 音频数组 / 文本」。理由：所有模型共享流程，避免重复实现。
- **ADR-003**：不引入 DI 容器，用 `config` 字典 + 工厂。理由：规模下轻量、可读、易调试。
- **ADR-004**：包名与命令行入口保持 `audioclone` 不变。理由：最小化破坏现有集成；通用化重命名留待下一版本评估。

---

## 12. Milestone 与任务清单

- [ ] M1：版本号 → v0.1.1 + 新增抽象层（base / registry）+ dummy 后端
- [ ] M2：TTE 服务层迁出（TTSService + cosyvoice_mlx 后端）
- [ ] M3：ASR 服务层迁出（ASRService + qwen3_asr_llama 后端）
- [ ] M4：model_loader + 配置化 + CLI 适配
- [ ] M5：导出/消费者（webui/api）收敛 + dummy 架构测试
- [ ] M6：文档与回归验收

---

## 13. 能力驱动演进（Capability-driven）

> 状态：**已实施于 v0.1.1**。这是对原「按任务接口（TTSBackend/ASRBackend）」的升级，
> 将后端契约从「任务」演进为「能力(Capability)」，以原生支持**一模型多任务**。

### 13.1 动机

现代音频模型往往是**多功能**的（同一声学模型可做 TTS / 变声 / 分类 / VAD 等；同为 ASR 模型可做语音理解/翻译）。
按 task 组织的接口（一个 task 一套类）迫使多功能模型实现多个类或用适配器重复实现，且「模型能做什么」无法表达。

### 13.2 新增结构

```
core/capabilities/
  base.py   Capability（能力标记基类）+ ModelBackend（后端基类，提供 supports()）
  tts.py    TextToSpeechCapability      （synthesize + sample_rate）
  t2a.py    TextToAudioCapability       （text_to_audio：文本→音乐/音效）
  asr.py    SpeechRecognitionCapability （transcribe + ASR 生命周期默认）
  a2a.py    AudioToAudioCapability      （transform：变声/增强/风格迁移）
  audio_classification.py  AudioClassificationCapability（classify：分类打标）
  vad.py    VoiceActivityCapability     （vad：说话片段检测）
```

能力到任务映射：

| 能力接口 | 方法 | 任务 |
|----------|------|------|
| `TextToSpeechCapability` | `synthesize(...)` | text-to-speech |
| `TextToAudioCapability` | `text_to_audio(...)` | text-to-audio |
| `SpeechRecognitionCapability` | `transcribe(...)` | speech-recognition |
| `AudioToAudioCapability` | `transform(...)` | audio-to-audio |
| `AudioClassificationCapability` | `classify(...)` | audio-classification |
| `VoiceActivityCapability` | `vad(...)` | voice-activity-detection |

框架层现已完整提供上述六种能力契约，并内置全能力演示后端 `dummy-omni`（一个后端同时
支持全部六种任务，供无模型完整体验与测试）。真实模型后端除 TTS（cosyvoice_mlx）与
ASR（qwen3_asr_llama）外暂缺，接入时仅需按对应能力接口实现。

每种能力均配套一个**服务编排层（Service）**，构造时按能力校验后端：

| Service | 依赖能力 | 主方法 |
|---------|----------|--------|
| `TTSService` | `TextToSpeechCapability` | `synthesize_to_file` / `infer_*` |
| `TextToAudioService` | `TextToAudioCapability` | `generate` |
| `ASRService` | `SpeechRecognitionCapability` | `transcribe` / `is_ready` |
| `AudioToAudioService` | `AudioToAudioCapability` | `transform` |
| `AudioClassificationService` | `AudioClassificationCapability` | `classify` |
| `VoiceActivityService` | `VoiceActivityCapability` | `vad` |

- **后端 = 能力聚合**：继承 `ModelBackend` + 一个或多个 `*Capability`。
   一个后端同时继承多个能力即宣告「一模型多任务」。
- **服务层按能力消费**：`TTSService` 构造要求后端具备 `TextToSpeechCapability`，否则 `ValueError`；
   `ASRService` 同理要求 `SpeechRecognitionCapability`。避免把错误模型接到任务上。
- **能力查询**：`backend.supports(Capability)`；`list_backend_capabilities(backend)` 返回可读名称，
   WebUI 启动时打印后端能力，后续可按能力动态渲染可用任务入口。

### 13.3 与既有实现的关系

- 原 `backends/base.py`（TTSBackend / ASRBackend）已被 `capabilities/` 取代并移除。
- 三个后端（cosyvoice_mlx / qwen3_asr_llama / dummy）改为继承能力接口，`config.py` 的 factory 路径不变。
- 目标场景示例：若某变声/重音模型实现 `AudioToAudioCapability`（新增），同一声学权重后端即可同时产出
  TTS 与变声能力，WebUI/API 增加对应任务即接入，零改动既有推理。

### 13.4 测试

`tests/test_framework.py` 新增能力测试：
- `test_capability_querying`：supports() 正确反映能力
- `test_multimodal_backend_aggregates_capabilities`：一后端聚合 TTS+ASR 双能力，同时驱动两个服务
- `test_capabilities_listing`：list_backend_capabilities 输出
- `test_service_rejects_incapable_backend`：能力不匹配的后端被服务层拒绝