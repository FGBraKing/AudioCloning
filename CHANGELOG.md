# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 与
[Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划
- 接入 T2A / A2A / 音频分类 / VAD 真实后端
- 结构化日志叠加 trace_id / JSON（当前为请求日志中间件）

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
