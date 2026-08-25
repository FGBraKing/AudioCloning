# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划
- 接入 T2A / A2A / 音频分类 / VAD 真实后端（v0.3.0，需真实模型）
- ASR 输出结构化解析（`_parse_asr_output`，低优先级）

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
