# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [v0.3.0] - 2026-10-25

### 变更（非 Mac 平台补全：sherpa-cls / sherpa-denoise 接入 + NVIDIA 全任务覆盖）
- 新增 `sherpa-cls-onnx` 后端：**CED-Small 音频标注（AudioSet 527 类，~92M）**，
  k2-fsa 官方 HF 仓库（已核实 `model.int8.onnx` + `class_labels_indices.csv`）——
  此前放弃的 torch AST 方案以 sherpa-onnx 路线无 torch 达成
- 新增 `sherpa-denoise-onnx` 后端：**GTCRN 语音增强（~236K）**，sherpa-onnx
  `OfflineSpeechDenoiser`；模型经 **GitHub Release 分发**（不在 HF），下载管线新增
  `github` 来源（`download_github_release_asset`，httpx 流式 + 可用性离线检查）
- **NVIDIA 平台全任务覆盖**：tensorrt-llm 框架的 ASR/VAD/分类/增强候选改为
  sherpa-onnx 后端（onnxruntime CUDA EP，GPU 加速），仅 T2A 音乐回退 dummy；
  TTS 保持 Edge-LLM 优先
- 分类/增强能力矩阵：mac→sensevoice-mlx/deepfilternet-mlx；CPU/AMD/NVIDIA→
  sherpa-cls-onnx/sherpa-denoise-onnx
- 新增 10 个单测（CED 分类/GTCRN 增强/GitHub 下载路由与状态）；mypy/ruff 全绿

### 变更（TTS 克隆备选接入：Chatterbox / Spark）
- 新增 `chatterbox-mlx` 后端（`mlx-community/chatterbox-turbo-8bit`，已核实标注上游
  mlx-audio）：**exaggeration 情感强度旋钮**（0~2，构造参数）+ 零样本克隆；
  无参考音频时支持 `plain` 模式（预置默认音色直读）
- 新增 `spark-mlx` 后端（`mlx-community/Spark-TTS-0.5B-8bit`）：零样本克隆
  （参考音频必填）；**许可 CC BY-NC-SA 4.0（非商用）**，文档与 config 注明
- mlx 框架 TTS 候选链扩展为 `["qwen3-tts-base-mlx", "chatterbox-mlx", "spark-mlx",
  "dummy"]`（默认首选不变，`--tts-backend` 可切换）；权重默认不下载
- 新增 8 个后端单测（身份/注册/克隆参数传递/plain 无参考/Spark 必填校验/单结果收集）

### 变更（TensorRT 适配落地：Edge-LLM Qwen3-TTS 推理步封装）
- 新增 `qwen3-tts-{base,customvoice,voicedesign}-tensorrt` 三后端（`core/backends/
  qwen3_tts_tensorrt.py`），对接 **TensorRT Edge-LLM 官方 Qwen3-TTS 流水线**：
  - 封装第 3 步推理（`qwen3_tts_inference` 子进程：请求 JSON → WAV 产出）；导出/构建
    为用户在 x86 主机与目标设备上的一次性离线步骤，**本项目不下载权重、不执行构建**
  - 模式映射与 MLX 侧一致（base: zero_shot/cross_lingual 含 ICL 语义；customvoice/
    voicedesign: instruct），请求 JSON 按 Edge-LLM 变体能力构造
  - 加载为校验式：引擎目录非空 + 推理工具在位，缺失时报可操作工作流指引
- runtime：NVIDIA 平台 TTS 候选链 `["qwen3-tts-base-tensorrt", "dummy"]`；
  ASR 无 TensorRT 实现（Qwen3-ASR 无 TRT 版）→ 回退 dummy
- 本机无 NVIDIA 硬件：13 个单测以假引擎目录 + 假推理工具脚本覆盖调用契约
  （请求构造/ICL 语义/失败路径/两段拼接端到端），NVIDIA 模拟环境冒烟通过

### 修复与体验（v0.3.0 收尾）
- **并发安全**：五个 Service（ASR/T2A/A2A/分类/VAD）统一加 `RLock`——后端模型非线程
  安全，此前仅 TTSService 有锁，API 线程池并发下存在竞态
- **多任务端点一致性**：`/api/t2a`、`/api/transform` 计入 `metrics.record_synthesis`；
  `_save_wav_result` 落盘接入 `prune_outputs`（`AUDIOCLONE_OUTPUT_KEEP_MAX` 对多任务输出生效）
- **`/api/models` 30s TTL 缓存**：health 高频轮询不再反复扫描模型缓存目录；下载完成后立即失效
- **WebUI 模型下载入口**：总览页新增「模型下载」卡片（六任务状态徽标 + 一键下载 +
  完成后状态刷新），与 CLI/API 共用 `model_manager`
- **配置拆分**：新增 `AUDIOCLONE_TENSORRT_ENGINE_DIR`（TensorRT 引擎目录专用），
  不再与 `AUDIOCLONE_TTS_MODEL_DIR`（HF 权重目录）混用
- mypy 修复回归 0 errors（app.py 变量重定义 / ctypes.windll 类型存根 / sensevoice tolist 收窄）

### 变更（vLLM 排除：AMD GPU 全系统一 onnx）
- 经调研决策（vLLM 仅支持音频**理解** Qwen2-Audio/Ultravox/Qwen3-Omni，无 TTS/音乐/增强/VAD，
  与本项目任务矩阵不重合），**vLLM 从适配项中排除**：
  - 框架映射：AMD GPU（核显/独显）统一 → **onnx**；移除 `vllm` 框架类别与
    `qwen3-*-vllm` 占位候选
  - 移除 AMD 核显判定逻辑（`_is_amd_igpu` / `AMD_IGPU_VRAM_GB`，随 vLLM 排除失去区分意义）
- 框架集合收敛为 **mlx / tensorrt-llm / onnx** 三个；tensorrt-llm 注释固化调研结论：
  TTS 的真实路径是 TensorRT Edge-LLM（官方支持 Qwen3-TTS 三变体流水线，FP16/FP8/NVFP4）
- 测试同步（框架映射/AMD 用例改写），全库 0 处 vLLM 引用

### 变更（CPU/核显兜底迁移 llama.cpp → sherpa-onnx；全面移除 llama.cpp）
- **新增 `onnx` 推理框架**（sherpa-onnx / onnxruntime，音频专用，依赖轻、无子进程、无 torch）：
  - `sherpa-asr-onnx`：SenseVoice int8（`csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`）
  - `sherpa-tts-onnx`：Kokoro-82M int8 多语种（`csukuangfj/kokoro-int8-multi-lang-v1_1`），
    新增 TTS **`plain` 模式**（纯文本直读，无需参考音频；`TTS_MODES` 扩展、API `ref_audio`
    改为 plain 可省略、WebUI 新增「纯文本合成」面板）
  - `sherpa-vad-onnx`：Silero VAD（`onnx-community/silero-vad`）
  - sherpa-onnx 为可选依赖组 `uv sync --extra onnx`；未安装时报可读错误并跳过候选
- 框架映射调整：无 GPU / AMD 核显 → **onnx**（原 llama.cpp）；AMD 独显仍 vLLM
- **全面移除 llama.cpp**（经用户决策，项目不在任何场景依赖）：删除 `qwen3-asr-llama`
  后端（llama-server 子进程管理）、相关 Settings（asr_server_*/ctx_size/n_gpu_layers/
  http_timeout/ready_timeout/max_segment/chunk_overlap）与集成测试；可用性兜底机制不变
- `/api/health` 的 `framework` 改为实时探测值；README/USER-GUIDE 同步（sherpa-onnx 平台
  下载点表、plain 模式、env 变更）
- 新增 9 个 sherpa 后端单测（注入假模块）与 plain 模式测试；全库 0 处 llama 依赖

### 变更（AMD 核显平台：默认 llama.cpp + GGUF 下载点核实）
- runtime 探测：AMD **核显（APU）**平台直接映射 llama.cpp 框架（ROCm/vLLM 不适用核显）；
  判定方式 `rocm-smi` 专用显存 → sysfs `mem_info_vram_total` → 均不可探测保守视为核显，
  阈值 `AMD_IGPU_VRAM_GB`（默认 4GB）；AMD 独显仍走 vLLM
- 从 HuggingFace 核实各模型 llama.cpp 量化版下载点（**仅核实未下载**，详见 README）：
  - ASR：`ggml-org/Qwen3-ASR-0.6B-GGUF`（Q8_0 主模型 + mmproj，已接入）
  - TTS：Qwen3-TTS 架构 llama.cpp 不支持——记录可选替代 `OuteAI/OuteTTS-1.0-0.6B-GGUF`
    （0.6B，未接入，需 WavTokenizer 解码器）
  - 分类：`FunAudioLLM/SenseVoiceSmall-GGUF`（q8，Apache-2.0；面向 audio.cpp 生态，
    llama.cpp 主线不支持——分类在该平台回退 dummy）
  - T2A/A2A/VAD：非 LLM 任务，无 GGUF 等价物，回退 dummy
- 新增 5 个 AMD 核显探测用例（核显映射/独显保持 vLLM/sysfs 两分支/rocm-smi JSON/默认核显）

### 变更（llama.cpp 兜底落地：可用性感知的后端自动选择）
- `resolve_backend_name` 升级为**可用性感知**：候选链逐个离线检查模型是否已就位
  （spec 自定义目录 / HF 缓存 / ModelScope 目录，`local_files_only`，**绝不触发下载**），
  已就位的真实后端优先
- **真实后端间兜底**：例如 mac 上 MLX ASR 模型未下载但 GGUF 已缓存 → 自动切
  `qwen3-asr-llama`（告警日志说明原因）；反之首选就位时不降级
- **不静默退化**：全部真实候选未就位时保持框架首选（加载失败时下载指引指向正确模型），
  dummy 仅在无真实后端注册时兜底（如 llama.cpp 框架的 TTS）
- 新增 6 个兜底行为测试（首选保持/GGUF 兜底/自定义目录/纯 dummy 场景/离线保证）

### 变更（v0.3.0 功能落地：多任务调用链 + WebUI/API 全量接入）
- **服务工厂**：`model_loader.create_service(kind)` 按任务创建业务服务（后端 auto 解析 +
  构造期能力校验）；模型在服务方法首次调用时**惰性加载**，启动零权重开销
- **API 新增四个端点**：`POST /api/t2a`（音乐生成→WAV）、`POST /api/transform`
  （降噪增强→WAV）、`POST /api/classify`（情感+事件→JSON）、`POST /api/vad`
  （语音片段→JSON）；上传复用限流/格式/时长校验；结果落盘 `outputs/`
- **WebUI 多任务实验台修正**：原实现四个任务全部错误绑定 TTS 后端（必然报"不支持"），
  现各任务绑定 `create_service(kind)` 创建的**专属后端**；任务卡片徽标显示实际后端名
  （dummy 以警示色区分）；新增 `--t2a/a2a/classification/vad-backend` CLI 参数
- API lifespan 同步创建四任务服务并统一释放；新增 5 个端点契约测试 + 多任务 handler
  测试改绑专属服务（能力校验移至服务构造期）

### 变更（v0.3.0 默认模型定版：六能力全接入 + 选型原则 + 依赖升级）
- 依赖升级：`mlx 0.29 → 0.32.2`；`mlx-audio` 0.5.0（环境内最新）
- **模型选型原则**（写入 registry/config 文档）：默认模型仅选 **8bit 量化（或非 LLM
  极小模型）、较新较强**的小模型，参数量以 **≤3B** 为基线；优先 mlx_audio 源码内置
  仓库或已确认存在的 mlx-community 仓库（不自实现、不猜仓库 ID）；**默认不下载权重**
  （首启提示 + `POST /api/models/download` 主动下载）
- **各能力默认模型定版**（全部经 HuggingFace API 核实存在）：
  - TTS：`Qwen3-TTS-12Hz-1.7B-{Base,CustomVoice,VoiceDesign}-8bit`（1.7B ✓）
  - ASR：`Qwen3-ASR-0.6B-8bit`（0.6B ✓；可选升级 `Qwen3-ASR-1.7B-8bit`）
  - T2A：`MiniMax-Music3-8bit`（**参数量放宽例外**：11.7B 基座，mlx-community 唯一
    有 8bit 量化版的音乐生成；此前猜测的仓库名 `MiniMax-Music-3` 有误，已纠正）
  - A2A：`DeepFilterNet-mlx`（mlx_audio 源码内置，语音增强）
  - VAD：`silero-vad-v6`（MIT，约 2MB；已确认存在，恢复接入）
  - 分类：`mlx-community/SenseVoiceSmall`（234M，mlx_audio 原生；**内建情感 9 类 +
    事件 4 类分类头**，softmax 置信度，一模型多任务。torch 路线（transformers AST）
    经用户决策放弃——依赖过重；通用 AudioSet 式分类如后续需要，按扩展机制自实现）
- 新增/恢复后端：`minimax-music-mlx`（T2A）、`silero-vad-mlx`（VAD）、`deepfilternet-mlx`
  （A2A）、`sensevoice-mlx`（分类）；`ModelSlot` 六任务槽位；runtime 候选链、
  `model_loader.create_backend(kind)`、`model_manager`、`/api/models`、下载接口同步支持六任务
- 新增后端单测（身份/注册/生成收集/增强/片段映射/阈值透传/依赖缺失指引），
  全部注入假模型离线运行

### 变更（mac 默认 ASR 迁移 llama.cpp → MLX）
- 新增 `qwen3-asr-mlx` 后端（`core/backends/qwen3_asr_mlx.py`）：基于 `mlx_audio.stt`
  的 `qwen3_asr` 架构，进程内加载 `mlx-community/Qwen3-ASR-0.6B-8bit`，**mac 默认 ASR
  不再依赖 llama.cpp**；长音频由 mlx_audio 内部分段（`chunk_duration` 默认 1200s），
  后端不再自行分片
- runtime 探测的 mlx 框架 ASR 候选链改为 `["qwen3-asr-mlx", "qwen3-asr-llama"]`：
  mac 自动选中 MLX 后端，`qwen3-asr-llama` 降级为 Linux/无 GPU 环境与显式指定的备选
  （llama.cpp 系统依赖随之变为可选）
- `config` 新增 `ASR_MLX_MODEL_ID` 与 `ASR_BACKENDS["qwen3-asr-mlx"]` 条目；
  registry 注册新后端；`/api/models` 与下载接口随 auto 解析自动覆盖新后端
- 集成测试显式指定 `qwen3-asr-llama`（本地权重为 GGUF）；新增 MLX 后端单测
  12 用例（转写/语言参数/槽位/依赖缺失指引/可用性判定），全部注入假模型离线运行

### 变更（模型下载源：HF 优先 + ModelScope 自动回退 + 自定义镜像）
- 下载来源新增 **auto 模式（默认）**：先 HuggingFace，失败**自动回退 ModelScope**；
  结果标记实际生效来源（`source`）与是否回退（`fallback`）；`hf` / `modelscope`
  强制指定时失败原样上抛不回退。CLI `--source` 与 API 请求体 `source` 同步支持
- **自定义 HuggingFace 镜像**：新增 `Settings.hf_endpoint`（env `AUDIOCLONE_HF_ENDPOINT`，
  兼容标准 `HF_ENDPOINT`，Settings 优先），透传 `snapshot_download(endpoint=...)`；
  `/api/models` 状态新增 `hf_endpoint` 字段
- **加载路径闭环**：`resolve_model_dir` 在 HF 缓存未命中时回退查找
  `pretrained_models/<模型名>`（ModelScope 的默认落盘位置）——回退下载的模型
  无需改启动参数即可加载；未下载错误提示同步更新（含镜像配置指引）
- CLI/ModelScope 下载日志与 `.env.example`、README、USER-GUIDE 同步更新；
  新增 7 个用例（auto 回退/强制不回退/端点优先级/端点透传/ModelScope 目录回退查找等）

### 变更（模型加载优化 2/∞：显存/内存容量判断）
- runtime 探测增加**容量判断**，不足时自动降级 llama.cpp（查询失败不做降级，避免误伤）：
  - Apple Silicon：统一内存（`sysctl hw.memsize`）< `AUDIOCLONE_MIN_MLX_MEM_GB`（默认 8GB）→ 降级
  - NVIDIA：**可用显存**（`nvidia-smi --query-gpu=memory.total,memory.used`，总量−其他进程占用；
    多卡取**可用最大**的一张，模型只需放得进一张卡）< `AUDIOCLONE_MIN_CUDA_VRAM_GB`
    （默认 8GB）→ 降级；总量充足但被占满同样触发降级
  - ROCm：无便携显存查询，暂不做容量判断（文档注明）
- `RuntimeInfo` 扩展 `mem_total_gb` / `gpu_vram_gb`（总量）/ `gpu_vram_free_gb`（可用）/
  `downgraded` / `downgrade_reason`，随 `/api/health` 的 `runtime` 字段暴露；降级时输出 warning 日志
- 阈值经 `Settings`（env 可覆盖）；探测日志带容量读数；Windows 走
  `GlobalMemoryStatusEx`，Linux 读 `/proc/meminfo`
- 新增 8 个容量判断用例（降级/阈值边界/查询失败不降级/env 调整/按需查询/多卡取最优）

### 变更（模型加载优化 1/∞：按硬件/系统自动选择后端推理框架）
- 新增 `core/runtime.py` 运行环境探测（进程内缓存，只做存在性检查、不 import 重依赖）：
  - **macOS (Apple Silicon)** → MLX；Intel Mac 视为无 GPU
  - **NVIDIA GPU**（`nvidia-smi` 或 `/proc/driver/nvidia/version`）→ **TensorRT-LLM**
  - **其他 GPU**（AMD ROCm `rocminfo`）→ **vLLM**
  - **无 GPU** → **llama.cpp**
- 默认后端改为 **`auto`**（`Settings.tts_backend/asr_backend`）：`model_loader.resolve_backend_name`
  取「框架候选链 + 兜底链」中第一个**已注册**的后端；为 `qwen3-*-tensorrt-llm` /
  `qwen3-*-vllm` 预留命名位，后端接入后无需改动 runtime 模块
- 无匹配专用后端时**降级回退并告警**（TTS→dummy；ASR→llama.cpp，任何平台可用），
  保证任何环境都能起服务；显式指定后端（CLI/env/参数）时跳过探测
- `/api/health` 新增 `runtime` 字段（os/machine/accelerator/framework）；WebUI 启动日志输出探测结果
- 移除 config 已无消费者的派生常量 `TTS_BACKEND`/`ASR_BACKEND`（`*_BACKENDS` 覆盖表不变）；
  `model_manager` 的任务→后端解析同样经 `resolve_backend_name`（`/api/models` 状态随环境一致）
- 新增 `tests/test_runtime.py`（17 用例：框架映射/驱动兜底/缓存/候选链/降级/端到端自动选择）；
  全程 monkeypatch，不依赖测试机硬件

### 变更（首次运行不自动下载 + 主动下载接口）
- **加载路径只做离线查找**：`utils/model_path.py` 的 `resolve_model_dir` 改用
  `snapshot_download(local_files_only=True)` 查 HF 缓存，**绝不触发下载**；未命中时抛
  `ModelLoadError` 并附下载指引（CLI 命令 / API 接口 / `--model_dir` 本地目录）
- **首次运行行为**：WebUI 显示可读的加载失败面板（含指引）；API 服务不再因模型缺失启动崩溃——
  lifespan 记录错误后以未就绪状态继续运行，推理接口按既有语义返回 503，
  `/api/health` 的 `model_loaded=false`
- **主动下载接口**（`core/model_manager.py`，CLI 与 API 共用）：
  - REST API 新增三个端点：`GET /api/models`（TTS/ASR 就绪状态离线查询，ASR 额外校验
    快照内含 `.gguf`）、`POST /api/models/download`（后台线程下载，立即返回 202；并发请求 409、
    未知任务/后端 400）、`GET /api/models/download`（任务进度）
  - CLI `audioclone-download-tts/asr` 改为 `model_manager` 薄封装（行为不变：
    默认 HF 系统缓存，`--source modelscope` / `--local_dir` 可选）
- 新增 `tests/test_model_manager.py`（状态查询/下载/离线解析，全部 stub 不触网）与
  API 端点测试（受理/完成/409 并发/400 校验）

### 变更（模型路径：pretrained_models → HuggingFace 系统缓存）
- **模型权重默认加载路径改为 HF 系统缓存**（`~/.cache/huggingface/hub`，可用 `HF_HOME` /
  `HF_HUB_CACHE` 重定向）：后端默认不再指向仓库内 `pretrained_models/`，加载时经
  `huggingface_hub.snapshot_download(model_id)` 解析快照目录，未缓存自动下载、命中即离线加载
- 可选自定义路径（三级覆盖，显式传入即不访问网络）：CLI `--model_dir` / `--asr_model_dir`、
  env `AUDIOCLONE_TTS_MODEL_DIR` / `AUDIOCLONE_ASR_MODEL_DIR`（`.env` 同样生效）、构造参数；
  新增 `utils/model_path.py` 统一解析（自定义目录不存在时报可读错误）
- 下载命令默认走 **HuggingFace 进系统缓存**（与默认加载路径一致）；新增 `--source modelscope`
  （国内镜像，落盘 `pretrained_models/<模型名>` 或 `--local_dir`）与通用 `--local_dir`；
  移除下载目录随 `--backend` 求值的旧逻辑（HF 缓存按 repo_id 隔离，天然不会互相覆盖）
- 新增显式依赖 `huggingface_hub>=0.30`（原先仅作为 mlx-audio 传递依赖）；三个 Qwen3-TTS
  变体的 `model_id` 下沉为后端类属性 `default_model_id`（直接构造后端亦可定位 HF 仓库）
- `config` 移除 `QWEN3_TTS_MODEL_DIR` / `ASR_MODEL_DIR` 常量与 `MODEL_ID/MODEL_DIR` 派生面，
  新增 `Settings.tts_model_dir` / `asr_model_dir`（默认 None = HF 系统缓存）
- 集成测试继续指向本地 `pretrained_models/`（显式传入，不触发网络）；单元测试 stub 掉
  `resolve_model_dir`，全程离线

### 变更（TTS 迁移：CosyVoice3 → Qwen3-TTS）
- 依赖切换：`mlx-audio-plus>=0.1.7` → **上游 `mlx-audio[tts]>=0.5.0`**（`mlx-lm` 随之放宽至 `>=0.31.1`，
  `transformers` 升至 5.x）。上游 mlx-audio 0.3.0 起移除 CosyVoice 实现，二者不可共存
- **下线 `cosyvoice3-mlx` 后端**（上游已无 CosyVoice 实现，保留即死代码），默认 TTS 后端改为
  `qwen3-tts-base-mlx`；`config.MODEL_ID/MODEL_DIR` 常量随之移除
- 修复 `audioclone-download-tts/asr` 的目录错位：`--local_dir` 默认值原先不随 `--backend` 变化，
  导致不同后端的同名权重文件互相覆盖（实测 Base 目录被 VoiceDesign 覆盖，加载时静默串模型）。
  现在 `--local_dir` 缺省时**随 `--backend`** 求值
- 修复 WebUI 合成模式与后端能力不匹配的问题：默认后端 `base` 不支持 `instruct`，
  「自然语言控制」面板此前必然报错。现在面板会感知后端能力——不支持的模式
  **渲染提示条并停用提交按钮**，并列出可切换的后端清单（`mode_guard` / `apply_mode_state`）
- 修复 SOCKS 代理导致的启动崩溃：`ALL_PROXY=socks5://` 会让 httpx 在**创建客户端时**就抛
  ImportError（与 no_proxy 无关），此前用 dummy ASR 不会触发、切到真实 ASR 后必然触发。
  webui/api 入口现在会移除 `ALL_PROXY`/`all_proxy`

### 变更（WebUI 迁移：Gradio → NiceGUI）
- 界面框架替换为 NiceGUI（`nicegui>=3.16`），移除 `gradio` 依赖；入口 `audioclone-webui` 不变
- 新增 `webui/theme.py`（设计令牌与全局样式）、`webui/widgets.py`（复合控件）、
  `webui/media.py`（上传/结果落盘）、`webui/errors.py`（框架无关的 `UIError`）
- 界面重做：顶栏 + 侧边导航（窄屏折叠为下拉菜单）的单页布局，玻璃拟态卡片，
  深浅色主题（默认深色，`--theme dark|light`），响应式栅格（≥1024px 双栏）
- 录音能力自研实现：AudioWorklet 采集 PCM → 前端编码 WAV → `POST /webui/api/recording` 落盘，
  规避 webm/opus 的服务端解码兼容问题
- 推理调用统一经 `run.io_bound` + 并发闸门（`AUDIOCLONE_WEBUI_CONCURRENCY`），
  不阻塞 asyncio 事件循环；结果落盘 `outputs/` 后可试听与下载
- handler 层与界面框架解耦：`gr.Error` → `UIError`，`@gradio_error` → `@ui_error`，
  业务逻辑与返回契约不变（单测同步迁移）
- 配置项重命名：`GRADIO_PORT/HOST/MAX_QUEUE/CONCURRENCY` → `WEBUI_PORT/HOST/CONCURRENCY`
  （`MAX_QUEUE` 无对应语义，移除）

### 变更（新增 Qwen3-TTS 后端 + 模型槽位守卫）
- 新增 `core/backends/qwen3_tts_mlx.py`，支持三个 MLX 模型变体：
  - `qwen3-tts-base-mlx` → `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`（zero_shot / cross_lingual）
  - `qwen3-tts-customvoice-mlx` → `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`（instruct，预设音色）
  - `qwen3-tts-voicedesign-mlx` → `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`（instruct，描述造音色）
- 能力层新增 `TextToSpeechCapability.supported_modes`（默认全部模式）：后端声明自己支持的
  推理模式子集，`TTSService` 在编排前 fail-fast，避免把不兼容的调用丢给模型
- 加载时显式锁定 `qwen3_tts` 架构：规避 mlx_audio 按目录名片段猜测架构导致
  `Qwen3-TTS-...` 被误判为另一个 `qwen3` 模块的问题；模型目录统一命名为 `qwen3_tts_<变体>`
- 新增 `core/model_slot.py` 模型槽位守卫（严格）：`ModelBackend.load()` 占槽、`close()` 释放，
  同一槽位被第二个不同后端占用时抛 `ModelSlotError`；`_load()` 失败自动回滚槽位。
  TTS 与 ASR 各一个槽位，即运行时恒定「一个 TTS 模型 + 一个 ASR 模型」
- `ModelBackend.load()/close()` 改为模板方法（子类实现 `_load()`/`_close()`），
  CosyVoice3 与 Qwen3-ASR 后端同步适配；无状态后端（如 dummy）不占槽
- ASR 生命周期统一：`Qwen3AsrLlamaBackend.load()` = 启动 llama-server、`close()` = 停止，
  `start()`/`stop()` 保留为兼容别名（仍返回 bool）
- `audioclone-download-tts/asr` 新增 `--backend NAME`，可为任意已登记后端下载模型
- WebUI 总览面板新增「支持模式」；TTS 后端加载失败改为可读错误（不再是原始堆栈）
- `/api/health` 新增 `model_slots` 字段，暴露当前槽位占用

### 计划
- 接入 T2A / A2A / 音频分类 / VAD 真实后端（v0.3.0，需真实模型）

## [v0.2.6] - 2026-08-26

### P2 完善（相对 v0.2.5）
- ASR 输出结构化解析：`_parse_asr_output` 优先闭合 `<asr_text>...</asr_text>` 标签对 → JSON 载荷文本字段（`text/asr_text/content/transcript`）→ 开标签 → 语言前缀清理；顺带修复 `language: xx` 前缀清理正则

### 文档
- `QUALITY-REPORT.md` 第五轮复审（适用 v0.2.6，§9 P1 全部 + P2 全部（除真实后端）实施记录）

## [v0.2.5] - 2026-08-26

### P1（缺陷修复与健壮性，按 QUALITY-REPORT v0.2.5 §9）
- trace_id 日志缺陷修复：`TraceIdFilter` 仅当记录无 `trace_id` 时注入（保留 extra），API 请求日志恢复 trace_id
- inline 上传内存安全：`_save_upload` 阈值感知读取（不信任客户端 size），超阈值转流式写盘
- `/api/outputs`：`limit` 上限 200、`modified` 改 UTC ISO8601、`include_duration` 可跳过读头
- 多任务校验口径：新增 `assert_audio_duration(path, max_sec)`，A2A/分类/VAD 使用 `ASR_AUDIO_MAX_SEC`（300s）

### P2（部分）
- `registry.create(name, capability=None)` 创建期能力 fail-fast，`model_loader` 传入对应能力
- `Metrics(buckets=...)` 可配置（缺 `inf` 尾桶抛错）
- USER-GUIDE：`/api/outputs` 参数、uvicorn 入口日志差异、`/api/metrics`+`/api/outputs` 暴露面 AuthN 注明

### 文档
- `QUALITY-REPORT.md` 第五轮复审（适用 v0.2.5，§9 P1 全部 + P2 部分实施记录）

## [v0.2.4] - 2026-08-26

### P2 完善（相对 v0.2.3）
- 新增 `GET /api/outputs` 历史输出接口（名称/路径/字节/修改时间/时长，含 `segments/` 子目录）
- 上传小文件内存直读 fast path（`AUDIOCLONE_MAX_INLINE_UPLOAD_BYTES`，默认 8MB）
- 声道统一：新增 `utils.audio.to_mono(data, mode)`，CosyVoice 与 Qwen3-ASR 统一取第一声道
- 多进程指标方案文档化（`api/metrics.py`）；USER-GUIDE 新增 `/api/outputs` 说明

### 文档
- `QUALITY-REPORT.md` 第四轮复审（适用 v0.2.4，§9 P0/P1/P2 大部分实施记录）

## [v0.2.3] - 2026-08-26

### P0（架构一致性）
- 版本单一来源：`__version__` 升级 0.2.3，pyproject `[tool.hatch.version]` 动态读取，FastAPI/WebUI 引用，消除代码与文档/tag 漂移
- registry 单池化：按名注册 + 按能力查询；`list_backends_for` 改用 `supports()`；内置后端全部声明 `capabilities`；config 去除 `factory` 字符串（`dummy` 按任务别名解析 dummy-tts/dummy-asr）
- 采样率时序：`TTSService.sample_rate` 改为属性实时读取（`load()` 后生效）
- ASR 分片重叠：段间重叠 `AUDIOCLONE_ASR_CHUNK_OVERLAP_SEC`（默认 1s）+ 尾部去重

### P1（正确性与体验）
- `prune_outputs` 递归清理（含 `segments/` 子目录）+ 空目录回收
- trace_id 贯穿服务层：`contextvars` + `TraceIdFilter`，API 中间件 / WebUI handler 注入
- 未就绪语义：`_get_engine` 改抛 `BackendUnavailable` → 503（新增 exception_handler）
- 多任务 handler（A2A/分类/VAD）复用 `assert_valid_ref_audio` 输入校验
- 后端配置 `sample_rate` 引用 `settings.sample_rate` 去重

### P2（部分）
- pyproject 可选 extra `loudness = ["pyloudnorm>=0.2.0"]`；`configure_logging` 低侵入（仅接管自身 handler）
- metrics 合成耗时直方图（累积桶）；`/api/metrics` 自身不计入请求计数；JSON 时间 UTC ISO8601
- CI 集成测试移除 `|| true` 兜底；新增 benchmark 冒烟 job；`Settings` env 覆盖专项测试

### 文档
- `QUALITY-REPORT.md` 第四轮复审（适用 v0.2.3，§9 P0/P1/P2 部分实施记录）

## [v0.2.2] - 2026-08-26

### 体验与工程（P1）
- 配置单源化：全部可调参数收编进 `Settings`（env 前缀 `AUDIOCLONE_` 可覆盖），模块级常量派生
- 共享校验：新增 `utils/validation.py`（`assert_tts_text` / `assert_valid_mode`），API 与 WebUI 复用
- ASR 子进程日志迁至 `logs/asr_llama_server.log`（独立于 `outputs/`）
- 类型覆盖：`cosyvoice_mlx.py` 与 `webui/` 纳入 mypy（0 errors / 39 文件）
- 覆盖率盲区：`webui/handlers.py`、`api/server.py` 纳入统计；补 handler/API 异常/loudness/list_backends_for/sanitize 边界/RLock 并发用例
- 函数内 import 上提（中间件 / `_json_error`）

### 音频与体验（P1/P2）
- 响度归一化接线：`AUDIOCLONE_TTS_NORMALIZE_LOUDNESS`（EBU R128，落盘前可选）
- 长文本体验：段间静音 `AUDIOCLONE_TTS_SEGMENT_GAP_SEC`（0.15s）、`progress_cb` 进度回调、`keep_segments` 逐段落盘
- WebUI 多任务 Tab：T2A / A2A / 音频分类 / VAD，按后端能力动态可用（dummy-omni 全支持）

### 可观测
- API 请求注入 trace_id（`X-Request-ID` 响应头 + 日志字段）
- `AUDIOCLONE_LOG_JSON=1` 启用单行 JSON 结构化日志
- Prometheus 风格 `/api/metrics`（请求计数 / 合成耗时 / 运行时长）

### 工程化与文档
- `.env.example`、`.pre-commit-config.yaml`（ruff+mypy+pytest）、`Makefile` 快捷命令
- 修复 `audioclone-download-asr` 因缺 `model_id` 无法下载的问题
- `QUALITY-REPORT.md` 第三轮复审（适用 v0.2.2，§9 P1/P2 已实施）
- `USER-GUIDE.md` 修订（适用 v0.2.2）

## [v0.2.1] - 2026-08-25

### 加固（相对 v0.2.0）
- 领域异常体系 `AudioError`（`InvalidAudioInput` / `CapabilityError` / `BackendUnavailable` / `ModelLoadError`）+ FastAPI 统一 `exception_handler`
- `pydantic-settings` 配置，消除 import 副作用（env 前缀 `AUDIOCLONE_`）
- 后端 registry 预注册，`model_loader` 优先从 registry 解析；新增按能力聚合查询 `list_backends_for`
- 能力接口改 `ABC + @abstractmethod`，未实现即构造失败；`ModelBackend` 支持声明式 `capabilities` 元数据
- 音频数值安全 `sanitize_audio`（去 NaN/Inf → 去 DC → 限幅 → 峰值归一化），落盘前统一调用
- 重采样 `resample_audio`（soxr → `resample_poly` 回退），替换 FFT 法
- ASR 超时可配置 `AUDIOCLONE_ASR_HTTP_TIMEOUT`；子进程 stderr 重定向日志文件防 PIPE 阻塞；长音频按 `max_segment_sec` 分片识别
- FastAPI 合成端改普通 `def`（线程池），引擎依赖注入，无模块级单例；新增请求日志中间件（方法/路径/状态码/耗时）
- 上传流式限流（大小/格式/时长校验）；校验接口收敛为 `assert_valid_ref_audio`（抛 `InvalidAudioInput`）
- `TTSService` 可重入锁串行化合成（MLX 并发安全）；文本分段接口化 `TextSegmenter` / `SentenceTextSegmenter`
- 模型生命周期统一：WebUI/API 的 `finally`/lifespan 统一 `close()` 后端
- 输出目录生命周期 `prune_outputs`（`AUDIOCLONE_OUTPUT_KEEP_MAX`）；WebUI 提示输出位置与清理策略
- WebUI handler 提取 `@gradio_error` 装饰器去样板；状态面板展示 TTS 模型加载耗时
- 默认绑定 `127.0.0.1`（部署护栏）；响度归一化 `loudness_normalize`（EBU R128，回退峰值归一化）

### 测试与质量
- 新增契约测试、API 端到端测试（`tests/test_api.py`）、音频工具测试（`tests/test_audio.py`）
- `pytest-cov` 覆盖率门槛 60%（实测 ~63%）；`mypy` 0 errors；`ruff` 全绿
- 性能基准脚本 `tools/benchmark.py`、客观质量脚本 `tools/eval_quality.py`
- GitHub Actions CI：lint（ruff）+ type（mypy）+ test + coverage

### 文档
- `QUALITY-REPORT.md` 复审版（适用 v0.2.1）
- `USER-GUIDE.md` 使用说明（适用 v0.2.1）
- `README.md` 版本同步

## [v0.2.0] - 2026-08-25

### 重构（相对 v2.0 / v0.1.1）
- 迁移为 src-layout 四层架构：展示层 → 服务层 → 能力层 → 实现层
- 能力驱动：TTS / T2A / ASR / A2A / 分类 / VAD 六种 `Capability` + 各自 Service
- 后端可替换：`cosyvoice_mlx` / `qwen3_asr_llama` / `dummy` / `dummy-omni`
- 全能力演示后端 `dummy-omni`，无模型完整体验框架链路
- 工程化：uv 管理（Python 3.13）、console scripts、ruff

## [v2.0] - 早期

个人声音克隆系统（MLX），固定模型 Fun-CosyVoice3 + Qwen3-ASR。
