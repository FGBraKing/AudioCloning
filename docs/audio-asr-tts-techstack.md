
# 音频任务 ASR 与 TTS 技术栈调研报告

> **调研范围**：2025-08 至 2026-08（近 1 年）
> **核心样本**：Qwen3-ASR、Qwen3-TTS、CosyVoice3
> **生成日期**：2026-08-28
> **数据规模**：7 个调研对象 × 38 个字段，全部字段级内容取自 arXiv 技术报告、官方 GitHub / ModelScope / vLLM 文档等一手来源；无法核实的一律标注「待核实」

---

## 目录

- [第一部分 · 执行摘要：三条结论](#第一部分--执行摘要三条结论)
- [第二部分 · 技术栈决策：推进任务需要什么](#第二部分--技术栈决策推进任务需要什么)
- [第三部分 · 三款模型详细技术分析](#第三部分--三款模型详细技术分析)
- [第四部分 · 横向对比](#第四部分--横向对比)
- [第五部分 · 工程落地细节](#第五部分--工程落地细节)
- [第六部分 · 风险清单与待核实项](#第六部分--风险清单与待核实项)
- [附录 A · 字段定义](#附录-a--字段定义)

---

## 第一部分 · 执行摘要：三条结论

### 结论一：ASR 与 TTS 应分开选型，不要为了「统一技术栈」牺牲单项最优

三款模型虽然同出阿里系，但**技术路线完全不同**，各有明确的胜负场：

| 任务 | 首选 | 关键判据 |
|---|---|---|
| **ASR（中文/方言/长音频）** | **Qwen3-ASR-1.7B** | AISHELL-2 CER **2.71** vs Whisper-large-v3 5.06；WenetSpeech meeting CER **4.97–5.88** vs GPT-4o-Transcribe 15.30–32.27 |
| **ASR（高并发/成本敏感）** | **Qwen3-ASR-0.6B** | 并发 128 时吞吐 **2000× 实时**、RTF 0.064；并发 1 时 TTFT 仅 **92 ms** |
| **TTS（实时对话，延迟优先）** | **Qwen3-TTS-12Hz-0.6B-Base** | 端到端首包 **97 ms**，开源 TTS 最低 |
| **TTS（中文内容一致性优先）** | **CosyVoice3-0.5B-2512_RL** | Seed-TTS test-zh CER **0.81**，低于 Seed-TTS 闭源 1.12、MiniMax-Speech 0.83 之外的所有开源模型 |
| **TTS（克隆音质优先）** | **Qwen3-TTS-12Hz-1.7B-Base** | Seed-TTS test-en WER **1.24（SOTA）**；10 语种说话人相似度全部超越 MiniMax 与 ElevenLabs |
| **TTS（跨语种配音）** | **Qwen3-TTS-12Hz-1.7B-Base** | zh→ko 混错率 **4.82** vs CosyVoice3 **14.4**（相对降 66%） |
| **TTS（方言/发音修补）** | **CosyVoice3** | 9 语言 + **18 种中国方言**，支持中文拼音与英文 CMU 音素发音修补 |

### 结论二：三款模型全部 Apache 2.0，这是本次选型最大的确定性

竞品池里大量「看起来可商用」的模型其实是坑，**代码许可 ≠ 权重许可**：

| 模型 | README badge | 权重真实许可 | 商用 |
|---|---|---|---|
| **Qwen3-ASR / Qwen3-TTS / CosyVoice3** | Apache-2.0 | **Apache-2.0** | ✅ 完全可商用 |
| F5-TTS | MIT ✅ | **CC-BY-NC-4.0**（训练数据 Emilia 传导） | ❌ 非商用 |
| Fish Audio（`fish-speech` 仓库） | Apache-2.0（旧） | **FISH AUDIO RESEARCH LICENSE**（2026-03-07 整体更换） | ❌ 需书面授权 |
| IndexTTS-2 | 普遍被误传为 Apache-2.0 | **bilibili 模型使用许可协议** | ⚠️ 月活 1 亿 / 年收入 10 亿阈值 |
| Spark-TTS | Apache-2.0（推理代码） | **CC-BY-NC-SA-4.0**（已从 Apache 改过） | ❌ 非商用 |
| GPT-SoVITS | Apache-2.0 | 界定模糊，训练数据常含未授权音频 | ⚠️ 高风险 |

> **操作规范**：永远读仓库根目录的 LICENSE 原文，不读 badge、不读二手文章；权重许可看 HuggingFace 模型卡右上角。每季度复查一次。

### 结论三：真正的瓶颈不在模型权重，而在服务化层

Qwen3-TTS 的 97 ms 是**官方理想值**（内部 vLLM 引擎 + torch.compile + CUDA Graph）。第三方实测（Nari Labs，1 RPS）显示：

| 引擎 | 调优前 p95 TTFA | 调优后 p95 TTFA |
|---|---|---|
| vLLM-Omni | 277.9 ms | **56.8 ms** |
| SGLang-Omni | 1140.7 ms | 120.9 ms |
| VoxServe | 315.1 ms | **49.3 ms** |
| M* | 1160.0 ms | 104.0 ms |

调优手段只有两招，但收益巨大：**① 动态 trim 前导静音**（首帧 PCM 通常含 80–90 ms 静音，用短 RMS 窗检测持续语音起点后裁掉，直接省 ~80 ms）；**② 帧累积策略调优**（首块小、后续块递增，`codec_chunk_frames` / `codec_chunk_ramp`）。

**另外三个「模型之外」的硬约束**：
- **FlashAttention 2 是强依赖**，不是可选优化。无 FA2 时 Qwen3-TTS 在 RTX 5090 上只有约 0.3× 实时。
- **vLLM-Omni 默认为每个 stage 独立预留显存**，会独占整张 GPU，必须显式配置 `gpu_memory_utilization`（如 code_predictor 0.4 / code2wav 0.3）。
- **首包延迟会随并发急剧劣化**：Qwen3-TTS-12Hz-0.6B 从 1 并发 97 ms → 6 并发 299 ms；任何引擎在 6 RPS 时都回到 ~100 ms 以上。

---

## 第二部分 · 技术栈决策：推进任务需要什么

### 2.1 分层技术栈全景

```
L0 采集     浏览器 Web Audio API + AudioWorklet │ Python sounddevice │ ffmpeg 统一转码
   ↓
L1 传输     WebRTC(Opus over RTP) + SFU │ WebSocket(PCM16) │ SSE │ gRPC(Triton :8001)
   ↓
L2 VAD      Silero VAD（通用）│ FSMN-VAD（FunASR 栈）│ WebRTC VAD（极省资源）
   ↓
L3 特征     引擎内置前端（优先，勿重复实现）│ soxr 抗混叠重采样 │ torchaudio 变速
   ↓
L4 推理     vLLM 主线(ASR) │ vLLM-Omni(TTS) │ Triton+TRT-LLM(CosyVoice) │ llama.cpp/GGUF(边缘)
   ↓
L5 服务化   FastAPI + OpenAI 兼容 API │ Triton Inference Server │ Docker / Compose
   ↓
L6 编排     任务队列 + 并发限流 + speaker embedding / voice clone prompt 缓存
   ↓
L7 可观测   TTFT/TTFA 分位数、RTF、underrun 率、错误分类、GPU 利用率
   ↓
L8 评测     jiwer(WER/CER) │ UTMOS(自然度) │ WavLM(SIM) │ Seed-TTS-eval / TTSDS2 / TTS Arena
```

### 2.2 组件选型速查表

| 层 | 场景 | 首选 | 备选 | 关键参数 / 理由 |
|---|---|---|---|---|
| **L0** | 浏览器生产 | **Web Audio API + AudioWorklet** | MediaRecorder | AudioWorklet 跑独立音频线程，不受主线程 GC 影响；ScriptProcessorNode 已废弃，生产禁用 |
| **L0** | Python 服务端 | **sounddevice**（PortAudio） | PyAudio | NumPy 友好 |
| **L0** | 转码（必备） | **ffmpeg** `ffmpeg -i in.* -ar 16000 -ac 1 -f s16le -` | — | 统一转 **16 kHz / 单声道 / PCM16** |
| **L1** | 实时双向语音 | **WebRTC + Opus + SFU**（LiveKit / mediasoup / Pion） | 裸 WebSocket | Opus 16–64 kbps、20 ms 帧，jitter buffer 20–30 ms；内建 FEC + PLC；SFU 让「连接数」与「GPU 算力」独立扩容（SFU ~500 连接 vs GPU ~10 连接） |
| **L1** | 服务端↔服务端 | **WebSocket** 二进制帧 | SSE | 100–200 ms 分块，零编解码开销 |
| **L1** | OpenAI 生态 | OpenAI Realtime WebSocket `/v1/realtime?intent=transcription` | — | SDK 零改造 |
| **L2** | 通用实时链路 | **Silero VAD** | FSMN-VAD | 输入严格 512 样本 @16 kHz（32 ms）；`positiveSpeechThreshold=0.3`、`silenceDuration=1400ms`；起爆 ~64 ms，能滤掉咳嗽 |
| **L2** | FunASR 全链路 | **FSMN-VAD** | — | 开启动态 VAD（≤5 s→2.0 s 静音，>45 s→0.1 s），长句不会被切碎 |
| **L3** | 特征提取 | **引擎内置前端** | torchaudio / librosa | 业务层重复实现只会引入与训练不一致 |
| **L3** | 重采样 | **soxr** | — | 降采样**必须先低通**，否则 >8 kHz 分量混叠回来，等于主动给 ASR 喂噪声 |
| **L4** | ASR，GPU 高并发 | **vLLM 主线**（`pip install "vllm[audio]"`） | FunASR split-engine | day-0 支持，OpenAI 兼容 `/v1/audio/transcriptions` |
| **L4** | TTS，GPU | **vLLM-Omni** | SGLang-Omni / VoxServe | 三阶段（Talker / Code Predictor / Codec）独立调度 |
| **L4** | CosyVoice，生产 | **Triton + TensorRT-LLM** | PyTorch 原生 | README 明示相对 HF transformers **4× 加速** |
| **L4** | CPU / 边缘 | **llama.cpp + GGUF** | ONNX Runtime | SenseVoice q8 约 254 MB，C++ 单二进制、无 Python 运行时 |
| **L5** | 对外接口 | **OpenAI 兼容 API** | gRPC | 客户端零改造，便于后续换模型 |
| **L8** | WER/CER | **jiwer** | — | 英 WER、中/粤/韩 CER |
| **L8** | 自然度 | **UTMOS / UTMOSv2** | 人工 MOS | 注意：MOS > 4.5 时神经预测器饱和，无法区分前沿模型 |
| **L8** | 说话人相似度 | **WavLM-TDNN** | ECAPA-TDNN | 克隆场景目标 > 0.85 |

### 2.3 四条硬性工程规范（建议写进团队规范）

1. **ASR 输入统一 16 kHz / 单声道 / PCM16（int16 LE）**，在**网关层重采样一次**，不要每个模型各做一次。
2. **降采样必须先低通**（用 soxr）。不做低通的话 >8 kHz 分量会混叠回来，等于主动给 ASR 喂噪声。
3. **每路流一个 resampler / 一个 Opus 解码器实例**（按 SSRC）。Opus 与重采样滤波器都是有状态的，共享实例会让不同说话人的状态互相污染。
4. **Opus 解码异常恢复**：重建解码器 + 返回一帧静音 + 跳过后续 3 包让流重新同步。否则一个坏包会产出数秒噪音。

### 2.4 三条推荐落地路径

#### 路径 A · 实时语音对话（Voice Agent，延迟敏感）

```
WebRTC(Opus 20ms帧) → Silero VAD → Qwen3-ASR-0.6B (vLLM 流式, TTFT 92ms)
    → LLM → Qwen3-TTS-12Hz-0.6B-Base (vLLM-Omni 流式, 首包 97ms)
    → 动态 trim 前导静音 → WebRTC 下行
```

- **预算**：端到端目标 < 500 ms。ASR 首字 ~92 ms + LLM ~200 ms + TTS 首包 ~57 ms（调优后）+ 网络。
- **缓存**：Qwen3-TTS 的 `create_voice_clone_prompt()` 结果必须缓存，避免每次请求重算 speaker embedding。
- **打断**：需要词级时间戳时用 Qwen3-ForcedAligner-0.6B（NAR，11 语言）。

#### 路径 B · 离线批量生产（会议转写 / 字幕 / 有声书）

```
文件 → ffmpeg 16k mono → VAD 切分 → Qwen3-ASR-1.7B (vLLM 离线批推理, 并发 32~128)
    → 标点/ITN → 输出 SRT/VTT
```

- **吞吐**：并发 128 时 2000× 实时，即 1 秒处理 2000 秒音频。
- **长音频**：单段最长 20 分钟（1200 s），超过需切片。

#### 路径 C · 中文配音 / 数字人（质量优先）

```
文本 + 3~10s 参考音频 → CosyVoice3-0.5B-2512_RL 或 Qwen3-TTS-12Hz-1.7B-Base
    → Triton+TRT-LLM 或 vLLM-Omni → 24kHz WAV
```

- **要方言/发音修补** → CosyVoice3（18 方言 + 拼音/CMU 音素 inpainting）
- **要跨语种/音色设计** → Qwen3-TTS（zh→ko 4.82；VoiceDesign 在 InstructTTSEval 中文 APS 85.2 全项 SOTA）

---


## 第三部分 · 三款模型详细技术分析

> ⭐ 标记的为「架构与训练」重点字段（本轮调研强化维度）。


---

<!-- item: qwen3_asr -->

## Qwen3-ASR

> **分类**：核心模型-ASR　|　**已填字段**：38 / 38　|　**⭐ 标记为本轮重点的架构与训练字段**


### 1. 基本信息与版本演进

- **名称**：Qwen3-ASR（含 Qwen3-ASR-1.7B、Qwen3-ASR-0.6B、Qwen3-ForcedAligner-0.6B；另有闭源 API 版 Qwen3-ASR-Flash-1208）
- **团队**：阿里巴巴通义千问 Qwen Team（核心作者 Xian Shi, Xiong Wang, Zhifang Guo, ... Jin Xu†, Junyang Lin†）
- **开源时间**：2026-01-29 发布（GitHub Release / arXiv v1），arXiv v2 于 2026-01-30 修订；技术报告《Qwen3-ASR Technical Report》arXiv:2601.21337
- **基座模型**：Qwen3-Omni（arXiv:2509.17765），三个模型均由 Qwen3-Omni 后训练（post-training）而来
- **任务定位**：all-in-one 语音识别 + 语种识别（LID）+ 非自回归强制对齐时间戳预测
- **语言覆盖**：ASR 模型支持 52 种语言与方言（30 种国际语言 + 22 种中国方言）；ForcedAligner 支持 11 种语言
- **单次推理最大音频长度**：ASR 模型 1200 秒（20 分钟）；ForcedAligner 300 秒（5 分钟）
- **推理模式**：Offline / Streaming 统一（同一份权重），ForcedAligner 为 NAR（非自回归）
- **支持的音频类型**：语音、歌唱（singing voice）、带 BGM 的完整歌曲
- **代码与权重**：GitHub `QwenLM/Qwen3-ASR`、HuggingFace `Qwen/Qwen3-ASR-*`、ModelScope `Qwen/Qwen3-ASR-*`
- **官方结论定位**：1.7B 在开源 ASR 模型中取得 SOTA，并可与最强商业 API 竞争；0.6B 提供最佳「精度-规模」权衡，适合端侧/GPU 高并发部署


### 2. 开源协议与商用边界

- **开源权重**：Apache License 2.0（Qwen3-ASR-1.7B、Qwen3-ASR-0.6B、Qwen3-ForcedAligner-0.6B 三份权重全部 Apache 2.0，可商用）——来源：arXiv:2601.21337 摘要原文 "we release these models under the Apache 2.0 license"
- **推理 / 微调代码库**：Apache 2.0，包含完整推理框架与「可复现的微调配方（reproducible fine-tuning recipe）」
- **闭源部分**：Qwen3-ASR-Flash-1208 仅通过阿里云百炼 DashScope API 提供，权重未开放，属商业服务；论文附录中其评测结果仅作参考
- **注意**：FlashAttention 2（`flash-attn`）为独立第三方库，其许可证与本项目无关；仅在 `torch.float16` / `torch.bfloat16` 加载时可用


### 3. 模型变体与权重获取

| 模型 | 组成 | 参数规模 | 主要能力 | HuggingFace ID |
|---|---|---|---|---|
| Qwen3-ASR-1.7B | Qwen3-1.7B LM + Projector + AuT Encoder | LLM 1.7B；AuT Encoder 300M（hidden size 1024） | 精度旗舰：多语言/方言、复杂声学环境、复杂文本模式 | `Qwen/Qwen3-ASR-1.7B`（`-hf` 后缀为 transformers 原生格式） |
| Qwen3-ASR-0.6B | Qwen3-0.6B LM + Projector + AuT Encoder | LLM 0.6B；AuT Encoder 180M（hidden size 896） | 效率优先：精度/速度权衡最优，并发 128 时 2000x 吞吐 | `Qwen/Qwen3-ASR-0.6B` |
| Qwen3-ForcedAligner-0.6B | AuT Encoder + Qwen3-0.6B LM + timestamp prediction linear layer | 0.6B 量级（未单独公布编码器规格，论文中沿用 AuT） | NAR 时间戳预测：11 语言、最长 300s、词/字/句/段级粒度 | `Qwen/Qwen3-ForcedAligner-0.6B` |
| Qwen3-ASR-Flash-1208 | 未公开 | 未公开 | 闭源云 API，论文附录给出参考性评测结果 | 无（DashScope API） |

补充（**第三方来源，待核实**）：有技术博客（CSDN）给出更细的层配置——1.7B：Encoder 24 层 / 16 头 / d_model 1024，Decoder 28 层 / 16 头 / hidden 2048；0.6B：Encoder 18 层 / 14 头 / d_model 896，Decoder 28 层 / 16 头 / hidden 1024；词表 151,936（Qwen tokenizer）。另有资料称编码器为 32 层 Self-Attention。**论文正文未给出层数/头数，这两个数字相互冲突，建议以 `config.json` 实际值为准。**


### 4. 架构范式 ⭐

#### 范式定位：Large Audio-Language Model（LALM）

Qwen3-ASR 明确属于 **LALM 范式**，而非传统的 Transducer / AED 端到端 ASR。论文在 Introduction 中给出的范式迁移理由是：LALM 先对音频形成「高层理解」，再基于该理解条件化地生成转写，而不是自底向上的声学模式匹配。因此长音频转写、噪声鲁棒性、世界知识与命名实体识别、多语言与方言覆盖这几类问题被「更自然地解决」。

##### 具体架构范式

1. **三段式 Speech→Text 架构**：`AuT Encoder（声学前端）→ Projector（模态对齐）→ Qwen3 LM Decoder（自回归文本生成）`
   - 这是典型的「Whisper 风格音频编码器 + LLM 解码器」组合，Agent 侧可理解为：**把语音压成 12.5Hz 的连续隐序列，当作"前缀软提示"喂给一个 Qwen3 小语言模型**。
2. **AuT 编码器独立预训练**：AuT 是一个在 AED（Attention Encoder-Decoder）框架下、用大规模伪标签数据**单独预训练**的 ASR 编码器，它既独立于 Qwen3-Omni，也独立于 Qwen3-ASR 本体。论文强调该预训练编码器在**动态注意力窗口尺寸下能提供通用且稳定的音频表示**——这是「一个模型同时支持流式与离线」能成立的前提。
3. **动态 Flash Attention 窗口（1s–8s）统一流式/离线**：这是 Qwen3-ASR 最具工程价值的架构设计。同一个权重、同一套参数，通过调节注意力窗口大小在「低延迟短 chunk 流式推理」与「长上下文离线推理」之间切换，**不需要准备两套模型权重**。
4. **All-in-One 解码**：语种识别（LID）不是外挂分类器，而是让模型按输出模板先自回归生成 `language <lang>`，再生成 `<asr_text><transcript>`。因此 LID + ASR 在一次解码中完成。
5. **ForcedAligner 的范式创新**：把强制对齐重构为 **slot-filling（槽位填充）**任务——在转写文本中插入 `[time]` 特殊 token 作为时间槽，用**因果（causal）、非位移（non-shifted）**的训练方式让模型一次性并行预测所有槽位的时间戳索引。这与标准 LALM 的 next-token prediction（标签序列左移一位、用 ForCausalLMLoss）训练范式**刻意不同**，是该模型能 NAR 推理的关键。

##### 与同类方案的架构对比（论文立场）

- 传统时间戳方案：CTC / CIF 后处理（如 CTC-Segmentation），依赖语言特定词典、误差累积。
- Qwen3 方案：LALM 直接预测任意粒度时间戳；`[time]` token 与 timestamp prediction layer **不依赖语言特定音素集或词典**，因此一个模型覆盖 11 语言并支持跨语言/代码混说场景。


### 5. 模块拆解 ⭐

#### 1. Qwen3-ASR-1.7B / 0.6B（ASR 主体）

推理数据流（按论文 Figure 2 与官方代码）：

```
WAV 音频
  → 重采样到 16,000 Hz
  → 提取 128 维 Fbank 特征
  → [AuT Encoder] 8x 下采样 → 12.5 Hz 帧率的音频隐表示
  → [Projector]  映射到 Qwen3 LM 的 embedding 空间
  → [Qwen3 LM Decoder] 自回归生成文本 token
  → 按输出模板解析出 language + asr_text
```

| 组件 | 1.7B | 0.6B | 备注 |
|---|---|---|---|
| AuT Encoder | 300M 参数，hidden size 1024 | 180M 参数，hidden size 896 | AED 框架独立预训练；8x 下采样；动态 Flash Attention 窗口 1s–8s |
| Projector | 有 | 有 | 桥接编码器表示与 LLM 解码器 embedding 空间 |
| LM Decoder | Qwen3-1.7B | Qwen3-0.6B | 自回归；继承 Qwen3-Omni 的音频理解与语言建模能力 |

##### 输出模板（关键：微调时必须遵守）

模型对给定音频输出**两种类型**（取决于音频中是否存在可识别的人声）：

- 有人声：`language <语种><asr_text><转写内容>`
- 无人声：另一套「无语音」输出分支（官方 README 未以纯文本形式给出完整模板，建议直接用 `qwen_asr.parse_asr_output()` 解析）

微调时 JSONL 的 `text` 字段需写成：
```json
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
```
无语种信息时写 `language None<asr_text>...`，但**模型不会从该前缀学到语种检测**。

##### Context Biasing（上下文偏置）

SFT 阶段专门训练模型**学会使用 system prompt 中的上下文 token 作为背景知识**，用户无需微调即可通过 prompt 注入专有名词/领域术语，提升垂直场景识别率。这是工程上非常实用的能力（对比热词方案更灵活）。

#### 2. Qwen3-ForcedAligner-0.6B（时间戳模型）

```
语音 → [预训好的 AuT Encoder] → 语音 embedding
转写文本 → 每个词/字后追加起止时间戳标签 → 每个标签替换为特殊 token [time] → tokenizer
         → 拼接后送入 Qwen3-0.6B LLM
         → [Timestamp Prediction Linear Layer] 预测每个槽位的时间戳索引
         → 索引 x 80ms = 实际时间戳
```

- **时间戳离散化**：时间戳值除以 AuT 编码器输出的 **80ms 帧长**取整为索引（12.5Hz ↔ 80ms 一致）。
- **分类类别数上限 3,750**，对应最长 300 秒语音输入（300s / 0.08s = 3750）。
- **多语言能力来源**：AuT 编码器（大规模多语种语料预训练）提供帧级语音 embedding；Qwen3-0.6B 多语言 LLM 处理跨语言语义。二者结合后 `[time]` token 与预测层**不依赖语言特定音素集/词典**。


### 6. 音频编码器 ⭐

#### AuT（Audio Transformer）Encoder

| 属性 | 1.7B 配套编码器 | 0.6B 配套编码器 | 来源 |
|---|---|---|---|
| 参数量 | 300M | 180M | arXiv:2601.21337 §2.1 |
| Hidden size | 1024 | 896 | arXiv:2601.21337 §2.1 |
| 输入特征 | 128 维 Fbank | 同左 | arXiv:2601.21337 §2.1 |
| 下采样倍率 | 8x | 同左 | arXiv:2601.21337 §2.1 |
| 输出帧率 | 12.5 Hz | 同左 | arXiv:2601.21337 §2.1 |
| 注意力窗口 | 动态 Flash Attention，1s–8s | 同左 | arXiv:2601.21337 §2.1 |
| 训练框架 | AED（Attention Encoder-Decoder） | 同左 | arXiv:2601.21337 §2.1/§2.2 |
| 预训练数据量 | 约 4,000 万小时伪标签 ASR 数据（中英文占多数） | 共用同一预训练编码器预训练流程 | arXiv:2601.21337 §2.2 |

##### 设计要点（对二次开发有直接影响）

1. **独立预训练**：AuT 是「独立于 Qwen3-Omni 与 Qwen3-ASR 单独预训练」的编码器。这意味着：
   - 做领域适配时，可以只微调 Projector + LLM 而冻结 AuT（官方微调脚本默认全参数微调，冻结 AuT 属社区实践，**效果待核实**）；
   - AuT 的输出是「通用且稳定」的表示，跨任务复用性强。
2. **动态窗口是流式/离线统一的基础**：短窗口（最小 1s）→ 低延迟流式；长窗口（最大 8s）→ 长上下文离线。这不是两套权重，而是同一 attention mask 策略的调节。
3. **8x 下采样 = 12.5Hz 是时延与信息量的折中**：对比 Whisper 的 50Hz（20ms 帧移 + 2x 卷积下采样到 50Hz），Qwen3-ASR 的 12.5Hz 序列长度只有 1/4，这是它 RTF 能做到 0.0094（并发 1）的结构性原因之一。
4. **ForcedAligner 复用同一 AuT 编码器**：因此时间戳模型天然继承多语种与长音频能力。
5. **层数 / 注意力头数论文未公布**——第三方博客给出 24 层/16 头（1.7B）与 18 层/14 头（0.6B）的数字，另有来源称 32 层，**相互冲突，标注为待核实**，请以 HF `config.json` 为准。


### 7. 语音 Tokenizer 与编解码器 ⭐

#### 语音侧：无离散 tokenizer / codec

Qwen3-ASR **不使用离散语音 tokenizer（如 RVQ codec）**。AuT Encoder 输出的是**连续的 12.5Hz 隐表示**，经 Projector 直接映射到 LLM 的 embedding 空间。

这一点与 Qwen3-TTS（离散 multi-codebook 路线）形成鲜明对比，也是两者不应共用同一套音频前端的原因：

| | Qwen3-ASR | Qwen3-TTS |
|---|---|---|
| 语音表示 | 连续隐表示（AuT + Projector） | 离散 code（Qwen-TTS-Tokenizer-12Hz，16 层 RVQ） |
| 帧率 | 12.5 Hz | 12.5 Hz（12Hz 系列）/ 25 Hz（25Hz 系列） |
| 是否需要 codec 解码回波形 | 否 | 是（轻量因果 ConvNet / 块式 DiT + Flow Matching + BigVGAN） |

#### 文本侧：Qwen Tokenizer

- 使用标准 Qwen tokenizer（Qwen3 词表）。**词表大小 151,936 为第三方博客数据，待核实**（论文未给出）。
- 额外引入的特殊 token：`[time]`（ForcedAligner 的时间戳槽位，训练时随机插入）。

#### ForcedAligner 的时间戳「离散化」

严格说这不是 codec，而是一种量化方案：

- 帧长 **80ms**，与 AuT 输出的 12.5Hz 帧率严格对应；
- 时间戳索引 = round(timestamp / 80ms)；最大类别数 **3,750**（对应 300 秒上限）；
- 推理时：预测索引 × 80ms = 实际时间戳（单位 ms）。
- vLLM 中通过 config 暴露为 `timestamp_token_id` 与 `timestamp_segment_time` 两个字段（见 vLLM `forced_alignment_offline.py` / `forced_alignment_online.py` 示例）。


### 8. LLM 主干 ⭐

#### 骨干：Qwen3-1.7B / Qwen3-0.6B（源自 Qwen3-Omni）

| 属性 | Qwen3-ASR-1.7B | Qwen3-ASR-0.6B |
|---|---|---|
| LLM 骨干 | Qwen3-1.7B | Qwen3-0.6B |
| 预训练基座 | Qwen3-Omni（3 万亿 token 音/视/文多任务多模态预训练） | 同左 |
| 解码方式 | 自回归 next-token prediction | 同左 |
| 附带能力 | LID（语种识别）+ ASR 一次解码完成 | 同左 |

##### 为什么选 Qwen3-Omni 做基座

论文给出的理由：Qwen3-Omni「已被证明具备强音频理解能力」。工程上的收益：

1. **语言知识解决命名实体与方言词汇映射**（如方言「巴适」正确转写而非「巴士」）；
2. **世界知识**改善专有名词、术语识别；
3. **多模态对齐能力**使 Projector 的训练起点更好。

##### 关键的 SFT 设计决策：刻意「不听指令」

SFT 阶段明确把模型训练成 **ASR-only 模型，不遵循 prompt 中的自然语言指令**，目的是**缓解 prompt injection 与指令遵循失败**。

> 工程含义：不要指望通过自然语言 prompt 让 Qwen3-ASR 做「把转写结果翻译成英文」「总结这段音频」这类任务；它只做 ASR。唯一保留的 prompt 可控能力是 **context biasing**（system prompt 中的上下文 token 作为背景知识）。

##### LID 与 ASR 的统一解码

模型在 ASR 解码前，**通过自然语言 prompting 先做语种识别**（输出 `language English` 之类），再生成转写。语种识别准确率见 asr_metrics 字段。

论文指出剩余错误主要来自 **马来语（ms）与印尼语（id）** 这两个高度相似语言的混淆。


### 9. 声学解码器 ⭐

**N/A — ASR 模型无此项。**

Qwen3-ASR 是 Speech→Text 方向的模型，不存在将声学 token 解码为波形的 acoustic decoder / detokenizer。声学侧的解码只发生在 AuT Encoder 内部（AED 框架的 encoder 部分），其输出是送入 LLM 的连续隐表示，不回到波形域。

若需要「音频→波形」的解码组件，参见同系列 Qwen3-TTS 的 `qwen3_tts.json` 中的 `acoustic_decoder` 与 `vocoder` 字段。


### 10. 声码器

**N/A — ASR 模型无此项。**

Qwen3-ASR 不涉及波形合成，因此没有 vocoder / Code2Wav 模块。链路是「波形 → Fbank → 隐表示 → 文本」的单向编码+生成，音频在被 AuT 编码后即不再回到时域。


### 11. 上下文长度与 I/O 约束

#### 输入

| 项 | 值 | 来源 |
|---|---|---|
| 采样率 | 16,000 Hz（流式 Demo 在浏览器侧重采样到 16kHz） | 官方 README 流式 Demo 说明 |
| 单次推理最大音频长度（Seq. Len.） | **1200 秒（20 分钟）** | arXiv:2601.21337 Table 1 |
| ForcedAligner 最大音频长度 | **300 秒（5 分钟）** | arXiv:2601.21337 Table 1 |
| 输入形式 | 本地路径 / HTTP URL / base64 / `(np.ndarray, sample_rate)` 元组；支持批推理 | 官方 README |
| 支持音频类型 | Speech / Singing Voice / Songs with BGM | arXiv:2601.21337 Table 1 |

#### 上下文长度（LLM 侧）

论文未单独给出 Qwen3-ASR 的最大 context token 数。工程上需注意：12.5Hz 帧率下，20 分钟音频 = 15,000 个音频帧 token，加上文本输出与 prompt，**实际 context 需求在 16K token 量级以上**。**具体 config 值待核实**（建议直接读 HF `config.json` 的 `max_position_embeddings`）。

对比参考：同系列 Qwen3-TTS 论文明确提到预训练 S3 阶段把 max token length 从 8,192 提升到 32,768；ASR 侧论文未披露。

#### 流式推理配置（论文 §4.5 实测配置）

- **chunk size：2 秒**
- **fallback：5 token**
- **保持最后 4 个 chunk 不固定（keep the last four chunks unfixed）**

#### 输出

- 主输出：文本转写；可用 `parse_asr_output(content)` 解析出 `(language, text)`。
- 可选输出：词/字级时间戳（需额外加载 ForcedAligner，见下）。

#### 约束与注意事项

1. **流式不支持批处理**（官方 README 明示）；
2. **流式不支持返回时间戳**；
3. **时间戳仅在提供 `--aligner-checkpoint` / `forced_aligner=` 参数时才有**；不加载 aligner 时 Demo 的时间戳 UI 会自动隐藏；
4. 服务/容器必须绑定 `0.0.0.0` 而非 `127.0.0.1` 才能端口转发；
5. vLLM 不遵循 `cuda:0` 风格的设备选择，需通过 `CUDA_VISIBLE_DEVICES` 环境变量选卡。


### 12. 训练数据 ⭐

#### 各阶段数据规模总览

| 阶段 | 数据规模 | 构成 / 说明 | 来源 |
|---|---|---|---|
| (1) AuT 预训练 | **约 4,000 万小时伪标签 ASR 数据** | 中英文占多数（majority is in Chinese and English） | arXiv:2601.21337 §2.2 |
| (2) Omni 预训练 | **3 万亿 token** | 音频 + 视觉 + 文本多任务数据；Qwen3-ASR-0.6B 与 1.7B **都用 3T token** 训练 | arXiv:2601.21337 §2.2 |
| (3) ASR 有监督微调（SFT） | **规模显著更小的多语言数据集**，且与预训练语料**不相交（disjoint）** | 标准中/英/多语言 ASR 数据 + 非语音数据 + 流式增强数据 + 上下文偏置数据 | arXiv:2601.21337 §2.2 |
| (4) ASR 强化学习（GSPO） | **约 5 万条（50k utterances）** | 35% 中英文 + 35% 多语言 + 30% functional data（提升复杂环境下的转写稳定性） | arXiv:2601.21337 §2.2 |
| ForcedAligner 训练 | 未披露规模 | 由 **Montreal Forced Aligner (MFA)** 生成的**词/字级伪时间戳标签** | arXiv:2601.21337 §3.3 |

#### 关键数据设计解读

1. **SFT 数据量刻意做小且与预训练不相交**：说明 SFT 的定位是「格式/风格迁移（style transfer）」而非知识注入——把模型从通用多模态对话风格迁移到 ASR 专用输入输出格式。这对二次开发是重要信号：**领域适配应尽量用少量高质量数据 + 低 lr + 少 epoch**，而不是大规模堆数据。
2. **SFT 包含「非语音数据」**：对应「无语音时输出另一类结果」的输出模板设计，避免模型对静音/噪声产生幻觉转写。
3. **SFT 包含「流式增强数据」**：是流式模式（2s chunk）识别率仅小幅劣化的直接原因。
4. **SFT 包含「上下文偏置数据」**：支撑 context biasing 能力。
5. **RL 数据中 30% 是 functional data**：论文说明其目标是「提升复杂环境下的转写稳定性」，即 RL 主要针对难例而非通用精度。
6. **ForcedAligner 的伪标签噪声问题**：论文明确指出 MFA 伪标签**天然含噪声与系统性偏移**，Qwen3-ForcedAligner 并非简单复刻 MFA 输出，而是对其**蒸馏并平滑**，从而得到更稳定、偏移更小的时间戳。这也解释了为何它在 human-labeled 测试集上 AAS 反而低至 32.4ms。


### 13. 训练阶段 ⭐

论文 §2.2 明确将训练流程定义为 **AuT 预训练 → Omni 预训练 → ASR 后训练（SFT + RL）**，其中前两阶段与 Qwen3-Omni 完全一致。结论章节又称整体为 **四阶段训练流水线（four-stage training pipeline）**。

---

#### 阶段 1：AuT 预训练（AuT Pretraining）

- **目标**：在 AED 框架下，用大规模**有标签**（伪标签）数据得到一个预训练音频编码器。
- **数据**：约 **4,000 万小时**伪标签 ASR 数据，中英文占多数。
- **产出**：一个在**动态注意力窗口尺寸下都能提供通用、稳定音频表示**的编码器。
- **工程含义**：这是「流式/离线统一」能力的根基。若你计划替换或微调 AuT，需注意它的稳定性是在**变窗口**条件下训练的，固定窗口微调可能破坏该性质。

---

#### 阶段 2：Omni 预训练（Omni Pretraining）

- **目标**：以预训练好的 Qwen3-Omni 作为 ASR 训练的基座（foundation model）。
- **数据**：**3 万亿 token**，音频 + 视觉 + 文本多任务。
- **覆盖范围**：Qwen3-ASR-0.6B 与 Qwen3-ASR-1.7B **均用 3T token** 训练（即小模型也吃满了全量多模态预训练）。
- **详细流程**：遵循 Qwen3-Omni 技术报告（arXiv:2509.17765）。
- **工程含义**：0.6B 模型并非「1.7B 的蒸馏裁剪版」，而是同样吃满 3T token 多模态预训练的独立模型——这解释了为何 0.6B 的精度损失远小于参数量损失比例。

---

#### 阶段 3：ASR 有监督微调（ASR Supervised Fine-Tuning, SFT）

- **目标**：对 ASR 的输入/输出格式做**风格迁移（style transfer）**。
- **数据**：规模**显著更小**的多语言数据，与预训练语料**不相交**。除标准中/英/多语言 ASR 数据外，还包含：
  - **非语音数据（non-speech data）**——让模型学会在无可识别人声时输出对应的空结果分支；
  - **流式增强数据（streaming-enhancement data）**——支撑 2s chunk 流式推理；
  - **上下文偏置数据（context biasing data）**——支撑通过 system prompt 注入领域知识。
- **关键训练决策**：**刻意把模型训练为「不遵循 prompt 中自然语言指令」的 ASR-only 模型**，以缓解 prompt injection 与指令遵循失败。
- **同时训练**：模型学会**利用 system prompt 中的上下文 token 作为背景知识**，使用户无需微调即可获得定制化的 ASR 结果。

---

#### 阶段 4：ASR 强化学习（ASR Reinforcement Learning, GSPO）

- **算法**：**GSPO（Group Sequence Policy Optimization）**，引用 Zheng et al. 2025，arXiv:2507.18071。
- **数据量**：约 **5 万条 utterance**，构成 **35% 中英文 + 35% 多语言 + 30% functional data**。
- **作用（论文原文结论）**：RL 在模型的**噪声鲁棒性、转写稳定性、分析难例的能力**三方面扮演「essential role」。
- **奖励函数设计**：**论文未公开**，标注为**待核实**。

---

#### 附：Qwen3-ForcedAligner 的训练策略（§3.3）

1. **监督信号**：MFA（Montreal Forced Aligner）生成的词/字级**伪时间戳标签**（人工标注成本过高）。
2. **因果、非位移（non-shifted）训练**：
   - 标准 LALM 训练会「去掉输出序列最后一个 token、去掉标签序列第一个 token」，制造一位错位以实现 next-token prediction；
   - Qwen3-ForcedAligner **保持输出与标签序列不错位**，使模型显式地识别时间戳槽位并预测填充值；
   - **交叉熵损失只在时间戳槽位上计算**，把训练目标完全聚焦到槽位填充。
   - 附带收益：因果训练使模型预测当前槽位时可利用先验上下文，保证时间戳的**全局一致性**。
3. **动态槽位插入（dynamic slot insertion）**：训练时对每个词/字**随机决定**是否在其后插入起止时间戳槽位，以增强泛化能力。
4. **推理**：因为训练时序列不位移，用户可在**任意词或字后**插入槽位，模型用 **NAR 解码一次性并行预测所有槽位**。


### 14. 后训练与强化学习 ⭐

#### 算法：GSPO（Group Sequence Policy Optimization）

- **论文**：Zheng et al. 2025，*Group Sequence Policy Optimization*，arXiv:2507.18071（Qwen 团队自研，与 Qwen3 系列 LLM 后训练同源）。
- **在 Qwen3-ASR 中的定位**：训练流程的最后一个阶段，用于「进一步提升识别质量（further improving the quality of recognition）」。

#### 数据构成（arXiv:2601.21337 §2.2(4)）

| 类别 | 占比 | 目的 |
|---|---|---|
| 中英文数据 | 35% | 主力语种质量提升 |
| 多语言数据 | 35% | 多语种泛化 |
| Functional data | 30% | 提升复杂环境下的转写稳定性 |

**总量：约 5 万条 utterance。**

#### 论文明确归因于 RL 的三项能力

1. **噪声鲁棒性（noise robustness）**
2. **转写稳定性（transcription stability）**
3. **分析难例的能力（ability to analyze difficult cases）**

实证支撑（均来自论文内部测试集，1.7B最优成绩）：
- **ExtremeNoise（极低 SNR）：WER 16.17%**，而 GPT-4o-Transcribe 36.11、Gemini-2.5-Pro 29.06、Whisper-large-v3 **63.17**、Fun-ASR-MLT-Nano 36.55。
- **TongueTwister（绕口令）：WER 2.44%**，而 GPT-4o-Transcribe 20.87、Whisper-large-v3 16.63。
- **Elders&Kids（老人/儿童语音）：WER 3.81%**，而 Gemini-2.5-Pro 36.93、Whisper-large-v3 10.61。

对照可见：**在「分布外/高难度」子集上，Qwen3-ASR 对 Whisper-large-v3 的领先幅度（3–26 个百分点）远大于在标准公开集上的领先幅度**，这与「RL 主打鲁棒性与难例」的定位一致。

#### 未公开项（待核实）

- **奖励函数（reward）的具体设计**：论文未给出。Qwen3-TTS 论文在描述其 GSPO 阶段时明确写了「rule-based rewards（基于规则的奖励）」，但 **Qwen3-ASR 论文未说明其奖励是否为 rule-based**，不应直接套用。
- **RL 的超参**（group size、KL 系数、采样温度、训练步数、硬件规模）：均未披露。
- **RL 前后的消融数据**：论文未给出「SFT 后 vs RL 后」的对比表，因此 RL 的绝对增益量**无法从公开材料中量化**。


### 15. 数据配方 ⭐

#### 官方数据配方（从论文可还原的部分）

##### 1. AuT 编码器预训练配方
- 框架：**AED（Attention Encoder-Decoder）**
- 数据：~4,000 万小时**伪标签**（pseudo-labeled）ASR 数据，中英为主
- 隐含步骤：先用其他 ASR 系统给海量无标注音频打伪标签 → 用伪标签训练 AuT
- 关键设计：训练时**覆盖动态注意力窗口范围（1s–8s）**，使编码器在任意窗口下都稳定

##### 2. SFT 数据配比（论文点名的四类）

| 数据类型 | 作用 |
|---|---|
| 标准中文 / 英文 / 多语言 ASR 数据 | 基础转写能力 + 风格迁移 |
| **非语音数据（non-speech）** | 使模型在无可识别人声时输出正确的「空结果」分支，抑制幻觉 |
| **流式增强数据（streaming-enhancement）** | 支撑 2s chunk 流式推理，缩小流式 vs 离线的精度差 |
| **上下文偏置数据（context biasing）** | 使模型学会把 system prompt 中的上下文 token 当背景知识使用 |

约束：SFT 数据与预训练语料**不相交（disjoint）**。

##### 3. RL 数据配比
35% 中英 + 35% 多语言 + 30% functional（复杂环境稳定性）。

##### 4. ForcedAligner 数据配方
- 标签来源：MFA 伪标签（词/字级时间戳）
- 处理方式：**蒸馏 + 平滑**（非直接复刻），以降低 MFA 的噪声与系统性偏移
- 训练技巧：**动态槽位插入**——每个词/字随机决定是否在后面插入起止时间槽

#### 数据格式（微调侧，官方 `finetuning/README.md`）

JSONL，每行一个 JSON，字段：
- `audio`：WAV 文件路径
- `text`：转写文本，**必须带语言前缀**

```jsonl
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
{"audio":"/data/wavs/utt0002.wav","text":"language Chinese<asr_text>这是第二句。"}
{"audio":"/data/wavs/utt0003.wav","text":"language None<asr_text>Fine-tuning data line."}
```

**工程注意事项**：
- 语言前缀必须用模型训练时见过的语种名（如 `English` / `Chinese`），大小写与空格需与官方示例一致；
- 用 `language None` 时**模型不会学到语种检测**，若线上依赖 LID 则必须提供真实语种标签；
- 所有数据统一格式化为 ChatML 的约定来自 Qwen3-TTS 论文（ChatML to standardize inputs），**ASR 侧论文未明示使用 ChatML，待核实**。

#### 缺失信息（待核实）

- 数据清洗/过滤规则（SNR 阈值、时长过滤、去重策略、ASR 置信度过滤阈值）
- 方言数据的采集与标注方式
- 歌唱/BGM 数据的构造方式（论文只说「大规模训练数据 + RL 微调」带来该能力）
- 各语种的小时数分布


### 16. 微调支持

#### 官方支持情况

- **代码位置**：`https://github.com/QwenLM/Qwen3-TTS` 同组织下的 `QwenLM/Qwen3-ASR`，微调目录 `finetuning/`，主脚本 `qwen3_asr_sft.py`。README 顶部即写明 "a reproducible fine-tuning recipe"。
- **训练方式**：**全参数微调**，支持 `torchrun` 多卡（DDP）。
- **LoRA**：**官方微调脚本未提供 LoRA 相关参数**，官方文档未提及 LoRA 支持 → **待核实**（社区有 LoRA 需求，可通过 peft 自行接入）。
- **是否支持冻结 AuT 只训 Projector+LLM**：脚本未提供对应开关 → **待核实**。

#### 环境准备

```bash
pip install -U qwen-asr datasets
pip install -U flash-attn --no-build-isolation   # 推荐；RAM < 96GB 且 CPU 核多时用 MAX_JOBS=4
```

#### 单卡微调（官方示例）

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200 \
  --save_total_limit 5
```

#### 多卡微调

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 --grad_acc 4 --lr 2e-5 --epochs 1 --save_steps 200
```

#### 官方默认超参

| 参数 | 官方示例值 |
|---|---|
| batch_size | 32（per device） |
| grad_acc | 4 |
| lr | 2e-5 |
| epochs | 1 |
| save_steps | 200 |
| save_total_limit | 5 |
| num_workers（一键脚本） | 2 |
| prefetch_factor（一键脚本） | 2 |

> 注：官方示例的 epochs=1 与 SFT「风格迁移」定位一致；领域适配建议先跑 1–3 epoch 并密切监控，避免过拟合导致通用能力退化。

#### 断点续训

- 方式 A：`--resume_from ./qwen3-asr-finetuning-out/checkpoint-200`（显式指定）
- 方式 B：`--resume 1`（自动从 `output_dir` 下最新 checkpoint 恢复）

#### 微调后推理

```python
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained(
    "qwen3-asr-finetuning-out/checkpoint-200",
    dtype=torch.bfloat16, device_map="cuda:0",
)
```

#### 已知缺口（待核实）

- 官方未提供 eval/验证集自动评估脚本的指标说明（一键脚本含 `--eval_file`，但输出指标未文档化）；
- 无官方的 LoRA / QLoRA 配置；
- 无官方的方言适配、歌唱适配专用配方；
- 微调后是否仍保留 context biasing 与 LID 能力，官方未说明。


### 17. 训练硬件门槛

#### 官方未公开的部分（待核实）

- **预训练集群规模（GPU 卡型/数量/训练时长）**：论文完全未披露。AuT 预训练（4,000 万小时）与 Omni 预训练（3T token）显然属大规模集群作业，但无公开数字。
- **RL 阶段硬件**：未披露。
- **SFT 阶段硬件**：未披露。

#### 微调侧的可执行信息（官方 README）

| 项 | 说明 |
|---|---|
| 精度 | `torch.bfloat16`（官方示例默认） |
| 注意力实现 | 强烈推荐 FlashAttention 2，**仅支持 float16 / bfloat16 加载**，且需硬件兼容（Ampere 及以后） |
| 多卡 | `torchrun --nproc_per_node=N`（DDP） |
| 显存优化 | FA2 可显著降低显存；`batch_size=32` 为官方在示例中的取值（未说明对应卡型，**该值在单卡 24GB 上很可能 OOM，需按卡型下调**） |
| 内存提示 | 安装 flash-attn 时若 RAM < 96GB 且 CPU 核多，用 `MAX_JOBS=4`（此为编译期 RAM 约束，非推理显存） |

#### 显存需求

**官方未给出 ASR 模型的显存表格 → 待核实。**

第三方资料给出权重级估算（**来源为技术博客，非官方，仅供参考**）：
- Qwen3-ASR-0.6B：约 1 GB
- Qwen3-ASR-1.7B：约 1.6 GB

按其口径应为 **bf16 权重占用**，不含 KV cache、激活值与 vLLM 的 preallocated 显存。vLLM 部署时官方示例用 `gpu_memory_utilization=0.7~0.8`，流式 Demo 用 `0.9`。**建议以实测为准。**


### 18. 语种与方言覆盖

#### Qwen3-ASR-1.7B / 0.6B：30 种语言 + 22 种中国方言 = 52

**30 种语言（含语种代码）**：Chinese (zh)、English (en)、Cantonese (yue)、Arabic (ar)、German (de)、French (fr)、Spanish (es)、Portuguese (pt)、Indonesian (id)、Italian (it)、Korean (ko)、Russian (ru)、Thai (th)、Vietnamese (vi)、Japanese (ja)、Turkish (tr)、Hindi (hi)、Malay (ms)、Dutch (nl)、Swedish (sv)、Danish (da)、Finnish (fi)、Polish (pl)、Czech (cs)、Filipino (fil)、Persian (fa)、Greek (el)、Hungarian (hu)、Macedonian (mk)、Romanian (ro)

**22 种中国方言**：Anhui（安徽）、Dongbei（东北）、Fujian（福建）、Gansu（甘肃）、Guizhou（贵州）、Hebei（河北）、Henan（河南）、Hubei（湖北）、Hunan（湖南）、Jiangxi（江西）、Ningxia（宁夏）、Shandong（山东）、Shaanxi（陕西）、Shanxi（山西）、Sichuan（四川）、Tianjin（天津）、Yunnan（云南）、Zhejiang（浙江）、Cantonese (Hong Kong accent，粤语-香港口音)、Cantonese (Guangdong accent，粤语-广东口音)、Wu language（吴语）、Minnan language（闽南语）

**额外覆盖**：全球各国/地区的英语口音（论文内部测试集覆盖 **16 个口音组**）。

#### Qwen3-ForcedAligner-0.6B：11 种语言

Chinese、English、Cantonese、French、German、Italian、Japanese、Korean、Portuguese、Russian、Spanish

#### 已知的语言短板

1. **马来语（ms）与印尼语（id）易混淆**——论文 §4.3.2 明确指出 LID 的剩余错误主要来自这两者的混淆（声学高度相似）。
2. **Fleurs 全 30 语言子集上，Qwen3-ASR-1.7B 弱于 Whisper-large-v3**（12.60 vs 8.16）。论文自己承认「在语言多样性增加、长尾语言上仍有改进空间」。
3. **Fleurs 分语言细项可见长尾语言 WER 很高**（Qwen3-ASR-1.7B）：Hungarian 34.22、Persian 29.90、Finnish 25.23、Greek 28.08、Romanian 20.70、Czech 22.42、Danish 21.00、Macedonian 19.05、Swedish 19.36、Filipino 24.29、Hindi 17.15。**这些语言若属业务关键语种，务必先自测。**

来源：arXiv:2601.21337 Table 1、Table 5、Table A.2(b)、§4.3.2。


### 19. 流式能力

#### 架构基础

**动态 Flash Attention 窗口（1s–8s）**使 Qwen3-ASR 天然支持流式推理：**同一份权重**既做流式（短 chunk、低延迟）又做离线（长上下文、高精度），无需切换模型。

#### 实测流式配置（论文 §4.5）

- **chunk size：2 秒**
- **fallback：5 token**
- **保持最后 4 个 chunk 不固定**

#### 流式 vs 离线 WER（arXiv:2601.21337 Table 8）

| 模型 | 模式 | LibriSpeech | Fleurs-en | Fleurs-zh | 平均 |
|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | Offline | 1.63 — 3.38 | 3.35 | 2.41 | **2.69** |
| Qwen3-ASR-1.7B | Streaming | 1.95 — 4.51 | 4.02 | 2.84 | 3.33 |
| Qwen3-ASR-0.6B | Offline | 2.11 — 4.55 | 4.39 | 2.88 | **3.48** |
| Qwen3-ASR-0.6B | Streaming | 2.54 — 6.27 | 5.38 | 3.40 | 4.40 |

**结论**：流式的精度代价，1.7B 为 +0.64（2.69→3.33），0.6B 为 +0.92（3.48→4.40）。对实时字幕/语音助手场景，这个代价完全可接受。

#### 工程约束（官方 README 明示）

1. **仅 vLLM 后端支持流式推理**（transformers 后端不支持）。
2. **流式不支持批推理（no batch inference）**。
3. **流式不支持返回时间戳**。
4. 流式 Demo 命令：`qwen-asr-demo-streaming --asr-model-path Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8000`
5. 浏览器侧：捕获麦克风 → **重采样到 16,000 Hz** → 持续推送 PCM chunk。
6. 端到端延迟：官方未单独给出流式模式的 TTFT 分解；论文 Table 2 的 TTFT 数据来自 **online async（vLLM Serve 多并发请求）**设置，非纯流式麦克风场景。

#### 示例

- `examples/example_qwen3_asr_vllm_streaming.py`（官方仓库）


### 20. VAD 与后处理

#### 官方代码库明确提供

- **ForcedAligner 独立使用**：`Qwen3ForcedAligner.from_pretrained(...).align(audio=..., text=..., language=...)`，返回词/字级 `(text, start_time, end_time)`；支持本地路径 / URL / base64 / `(ndarray, sr)` 与批推理。
- **结果解析**：`from qwen_asr import parse_asr_output; language, text = parse_asr_output(content)`。
- **输入预处理**：`transcribe()` 接受本地路径、URL、base64、`(np.ndarray, sample_rate)`，内部完成解码与重采样。
- **批推理**：`max_inference_batch_size`（transformers 示例 32，vLLM 示例 128）。

#### 社区/第三方整理提到的工具链能力（**待核实**，未在论文或官方 README 主体出现）

有第三方技术站点（abeiai.cn、CSDN 系列文章）总结官方工具链包含：**音频格式自动重采样、VAD 语音活动检测、长音频智能分段、结果后处理、SRT 字幕生成**。

另有第三方 FastAPI 封装仓库（`groxaxo/Qwen3-ASR-FastAPI`）声称支持 **WAV/MP3/M4A/FLAC/OGG-Opus 自动转 16kHz mono**，并可通过 `return_timestamps=true` 输出 **SRT 字幕与 segments**。

> **这两组均非官方一手来源，VAD 与 SRT 是否属于 `qwen-asr` 包内建能力需要实测确认。** 官方 README 中未列出 VAD 相关模块。

#### 长音频处理策略

- 单次推理上限 **1200 秒**；超过需自行分段。
- 论文强调长音频场景（长转写）是 LALM 范式的优势项，20 分钟内**上下文连贯不断裂**（官方 README 未给出内部分段策略说明）。
- 社区说法：5 小时音频在 async 模式下约 10 秒完成（来自第三方博客的换算，具体为「2000x 吞吐」的推演，**待核实**）。

#### 后处理缺失项（待核实）

- **说话人分离（diarization）**：官方未提供。论文 Future Work 也未列入，但第三方评测文章将其列为改进方向。
- **标点恢复 / ITN（Inverse Text Normalization）**：论文与 README 均未说明是否内建。
- **情绪识别**：开源模型不具备；仅在 DashScope 云端 Qwen3-ASR-Flash API 中提供（第三方来源，待核实）。


### 21. 音色克隆

**N/A — ASR 模型无此项。**

Qwen3-ASR 是语音识别（Speech→Text）模型，不涉及声音克隆、说话人音色建模或 speaker embedding。

若需声音克隆能力，参见同系列 **Qwen3-TTS**（`qwen3_tts.json` 的 `voice_clone` 字段），其 Qwen3-TTS-12Hz-1.7B/0.6B-Base 支持 **3 秒零样本声音克隆**。

相关但不同的能力：Qwen3-ASR 在内部测试集中包含 **Dialog-Cantonese / Dialog-Mandarin / Dialog-Chinese Dialects** 等多说话人对话子集，说明对多说话人场景具备鲁棒性，但不输出说话人身份。


### 22. 情感与风格控制

#### 开源模型：不支持情感/风格指令控制

Qwen3-ASR 在 SFT 阶段被**刻意训练为 ASR-only 模型，不遵循 prompt 中的自然语言指令**（目的是缓解 prompt injection 与指令遵循失败）。

> **工程含义**：不能通过 prompt 让 Qwen3-ASR 做「请用 formal 语气转写」「翻译成英文」「帮我总结」这类操作。它只输出转写文本。

#### 开源模型保留的唯一 prompt 可控能力：Context Biasing

- 模型学会**利用 system prompt 中的上下文 token 作为背景知识**；
- 用法：在 system prompt 中注入专有名词、人名、产品名、领域术语；
- 价值：**无需微调即可快速适配垂直业务场景**，成本远低于微调；
- 来源：arXiv:2601.21337 §2.2(3)。

#### 其他可控项

| 可控项 | 是否支持 | 说明 |
|---|---|---|
| 强制指定语种 | ✅ | `transcribe(..., language="Chinese")`；不指定时自动 LID |
| 时间戳粒度（词/字/句/段） | ✅ | 由 ForcedAligner 提供，用户可自定义在任意词/字后插入时间槽 |
| 输出长度 | ✅ | `max_new_tokens`（vLLM 示例用 4096，评测用 1024） |
| 批处理大小 | ✅ | `max_inference_batch_size` |
| 情感/韵律/风格指令 | ❌ | 开源模型不支持 |
| 热词/偏置（非 prompt 方式） | 待核实 | 官方仅文档化 prompt 版 context biasing，未见独立热词接口 |

#### 云端 API 的额外能力（第三方来源，待核实）

第三方技术文章称 DashScope 的 **qwen3-asr-flash-realtime** API 支持「多语种高精度转写、自动语种检测、**非语音过滤**、**情绪识别**」。

> 这两项（非语音过滤、情绪识别）在开源权重中**不提供**，且未见于阿里云官方文档一手材料，**标注为待核实**。


### 23. 鲁棒性

#### 内部鲁棒性测试集（arXiv:2601.21337 Table 4，WER/CER ↓）

| 测试子集 | GPT-4o-Transcribe | Gemini-2.5-Pro | Doubao-ASR | Whisper-large-v3 | Fun-ASR-MLT-Nano | **Qwen3-ASR-0.6B** | **Qwen3-ASR-1.7B** |
|---|---|---|---|---|---|---|---|
| Accented English（16 口音均值） | 28.56 | 23.85 | 20.41 | 21.30 | 19.96 | 16.62 | **16.07** |
| Elders&Kids（老人/儿童） | 14.27 | 36.93 | 4.17 | 10.61 | 4.54 | 4.48 | **3.81** |
| ExtremeNoise（极低 SNR） | 36.11 | 29.06 | 17.04 | **63.17** | 36.55 | 17.88 | **16.17** |
| TongueTwister（绕口令） | 20.87 | 4.97 | 3.47 | 16.63 | 9.02 | 4.06 | **2.44** |
| Dialog-Mandarin | 20.73 | 12.50 | 6.61 | 14.01 | 7.32 | 7.06 | **6.54** |
| Dialog-Cantonese | 16.05 | 14.98 | 7.56 | 31.04 | 5.85 | 4.80 | **4.12** |
| Dialog-Chinese Dialects（22 方言均值） | 45.37 | 47.70 | 19.85 | 44.55 | 19.41 | 18.24 | **15.94** |

**读法**：Qwen3-ASR-1.7B 在全部 7 个内部子集上最优。相对 Whisper-large-v3 的最大优势出现在 **ExtremeNoise（63.17 → 16.17，-74%）**、**TongueTwister（16.63 → 2.44，-85%）**、**Dialog-Cantonese（31.04 → 4.12，-87%）**。

#### 歌唱与带 BGM 歌曲识别（Table 7，WER ↓）

| 子集 | GPT-4o | Gemini-2.5-Pro | Doubao-ASR-1.0 | Whisper-large-v3 | Fun-ASR-MLT-Nano | **Qwen3-ASR-1.7B** |
|---|---|---|---|---|---|---|
| M4Singer | 16.77 | 20.88 | 7.88 | 13.58 | 7.29 | **5.98** |
| MIR-1k-vocal | 11.87 | 9.85 | 6.56 | 11.71 | 8.17 | **6.25** |
| Opencpop | 7.93 | 6.49 | 3.80 | 9.52 | **2.98** | 3.08 |
| Popcs | 32.84 | 15.13 | 8.97 | 13.77 | 9.42 | **8.52** |
| EntireSongs-en（整首+BGM） | 30.71 | **12.18** | 33.51 | N/A | N/A | 14.60 |
| EntireSongs-zh（整首+BGM） | 34.86 | 18.68 | 23.99 | N/A | N/A | **13.91** |

**注意**：并非全部第一——Opencpop 上 Fun-ASR-MLT-Nano 更优（2.98 vs 3.08）；EntireSongs-en 上 Gemini-2.5-Pro 更优（12.18 vs 14.60）。Whisper-large-v3 与 Fun-ASR-MLT-Nano 在「整首歌 + BGM」长音频场景**直接退化到 N/A（性能过差）**。

#### 外部实测（第三方，待核实）

有第三方实测文章称：在极强背景音乐下，**独唱场景 WER < 6%，复杂 BGM 下 < 15%**；地铁极低 SNR 场景 WER 16.17%。第一个数字与论文 M4Singer 5.98 / EntireSongs-zh 13.91 大致吻合，可作交叉印证，但「独唱 <6% / BGM <15%」的表述方式**未见于官方材料**。

#### 鲁棒性的来源（论文归因）

1. **大规模训练数据**（4,000 万小时 AuT 预训练 + 3T token Omni 预训练）
2. **GSPO 强化学习**——论文明确把「噪声鲁棒性、转写稳定性、难例分析能力」归因于 RL
3. **LALM 范式的先验知识**——语言模型先验能纠正声学歧义

#### 已承认的鲁棒性短板

- **长尾语言**（见 languages 字段）
- **方言细分**：WenetSpeech-Chuan hard 上 Qwen3-ASR-1.7B 21.63 vs Doubao-ASR **20.20**（Doubao 更优）
- 论文未列入的改进方向（第三方整理，待核实）：**无说话人分离能力**、工厂噪声/清唱数据不足


### 24. ASR 指标

**评测统一配置**（arXiv:2601.21337 §4.1 与官方 README）：`dtype=bfloat16`、`max_new_tokens=1024`、vLLM、greedy 解码、**不指定语言参数**；字符类语言（中/粤/韩）用 CER，词分隔语言（英/德/法等）用 WER；多语言聚合用 macro-average。

---

#### 1. 公开集：英文（WER ↓）

| 测试集 | GPT-4o-Transcribe | Gemini-2.5-Pro | Doubao-ASR | Whisper-large-v3 | Fun-ASR-MLT-Nano | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** | Qwen3-ASR-Flash-1208（API 参考） |
|---|---|---|---|---|---|---|---|---|
| LibriSpeech clean — other | 1.39—3.75 | 2.89—3.56 | 2.78—5.70 | 1.51—3.97 | 1.68—4.03 | 2.11—4.55 | **1.63—3.38** | 1.33—2.40 |
| GigaSpeech | 25.50 | 9.37 | 9.55 | 9.76 | - | 8.88 | **8.45** | 8.82 |
| CV-en | 9.08 | 14.49 | 13.78 | 9.90 | 9.90 | 9.92 | **7.39** | 6.06 |
| Fleurs-en | 2.40 | 2.94 | 6.31 | 4.08 | 5.49 | 4.39 | **3.35** | 2.72 |
| MLS-en | 5.12 | 3.68 | 7.09 | 4.87 | - | 6.00 | **4.58** | 3.63 |
| Tedlium | 7.69 | 6.15 | 4.91 | 6.84 | - | **3.85** | 4.50 | 4.84 |
| VoxPopuli | 10.29 | 11.36 | 12.12 | 12.05 | - | 9.96 | **9.15** | 8.45 |

#### 2. 公开集：中文普通话（CER ↓）

| 测试集 | GPT-4o | Gemini-2.5-Pro | Doubao | Whisper-v3 | Fun-ASR | 0.6B | **1.7B** | Flash-1208 |
|---|---|---|---|---|---|---|---|---|
| WenetSpeech net — meeting | 15.30—32.27 | 14.43—13.47 | N/A | 9.86—19.11 | 6.35—- | 5.97—6.88 | **4.97—5.88** | 4.60—5.80 |
| AISHELL-2-test | 4.24 | 11.62 | 2.85 | 5.06 | - | 3.15 | **2.71** | 2.53 |
| SpeechIO | 12.86 | 5.30 | 2.93 | 7.56 | - | 3.44 | **2.88** | 2.62 |
| Fleurs-zh | 2.44 | 2.71 | 2.69 | 4.09 | 3.51 | 2.88 | **2.41** | 2.38 |
| CV-zh | 6.32 | 7.70 | 5.95 | 12.91 | 6.20 | 6.89 | **5.35** | 4.45 |

#### 3. 公开集：中文方言（CER ↓）

| 测试集 | GPT-4o | Gemini-2.5-Pro | Doubao | Whisper-v3 | 0.6B | **1.7B** | Flash-1208 |
|---|---|---|---|---|---|---|---|
| KeSpeech | 26.87 | 24.71 | 5.27 | 28.79 | 7.08 | **5.10** | 3.28 |
| Fleurs-yue | 4.98 | 9.43 | 4.98 | 9.18 | 5.79 | **3.98** | 3.50 |
| CV-yue | 11.36 | 18.76 | 13.20 | 16.23 | 9.50 | **7.57** | 4.86 |
| CV-zh-tw | 6.32 | 7.31 | 4.06 | 7.84 | 5.59 | **3.77** | 3.30 |
| WenetSpeech-Yue short — long | 15.62—25.29 | 25.19—11.23 | 9.74—11.40 | 32.26—46.64 | 7.54—9.92 | **5.82—8.85** | 5.84—8.20 |
| WenetSpeech-Chuan easy — hard | 34.81—53.98 | 43.79—67.30 | 11.40—**20.20** | 14.35—26.80 | 13.92—24.45 | **11.99**—21.63 | 11.52—20.82 |

> **注意**：WenetSpeech-Chuan hard 上 Doubao-ASR（20.20）优于 Qwen3-ASR-1.7B（21.63）。

#### 4. 多语言（WER ↓，Table 5）

| 测试集 | 语种数 | GLM-ASR-Nano-2512 | Whisper-large-v3 | Fun-ASR-MLT-Nano | 0.6B | **1.7B** |
|---|---|---|---|---|---|---|
| MLS | 8 | 13.32 | 8.62 | 28.70 | 13.19 | **8.55** |
| CommonVoice | 13 | 19.40 | 10.77 | 17.25 | 12.75 | **9.18** |
| MLC-SLM | 11 | 34.93 | 15.68 | 29.94 | 15.84 | **12.74** |
| Fleurs | 12 | 16.08 | 5.27 | 10.03 | 7.57 | **4.90** |
| Fleurs† | +8 | 20.05 | 6.85 | 31.89 | 10.37 | **6.62** |
| Fleurs††（全 30 语言） | +10 | 24.83 | **8.16** | 47.84 | 21.80 | 12.60 |
| News-Multilingual（内部，15 语种） | 15 | 49.40 | 14.80 | 65.07 | 17.39 | **12.80** |

> **重要**：Fleurs 全 30 语言子集上 **Whisper-large-v3（8.16）优于 Qwen3-ASR-1.7B（12.60）**。论文承认长尾语言仍有改进空间。

#### 5. 语种识别准确率（LID，% ↑，Table 6）

| 测试集 | Whisper-large-v3 | Qwen3-ASR-0.6B | **Qwen3-ASR-1.7B** |
|---|---|---|---|
| MLS（9 语种） | 99.9 | 99.3 | **99.9** |
| CommonVoice（13 语种） | 92.7 | 98.2 | **98.7** |
| MLC-SLM（11 语种） | 89.2 | 92.7 | **94.1** |
| Fleurs（30 语种） | 94.6 | 97.1 | **98.7** |
| **平均** | **94.1** | **96.8** | **97.9** |

#### 6. 时间戳精度：AAS（Accumulated Average Shift，ms ↓，Table 9）

AAS 定义：`AAS = (1/N) Σ|n̂ᵢ − nᵢ|`，即所有时间戳槽位上预测值与参考值绝对差的均值。

| 测试条件 | Monotonic-Aligner | NFA | WhisperX | **Qwen3-ForcedAligner-0.6B** |
|---|---|---|---|---|
| MFA-Labeled Raw — 平均 | 161.1 | 129.8 | 133.2 | **42.9** |
| ├ 中文 | 161.1 | 109.8 | - | **33.1** |
| ├ 英文 | - | 107.5 | 92.1 | **37.5** |
| MFA-Labeled Concat-300s — 平均 | 1742.4 | 246.7 | 2708.4 | **52.9** |
| ├ 中文（300s 长音频） | 1742.4 | 235.0 | - | **36.5** |
| Human-Labeled — 平均 | 141.3 | 101.2 | - | **32.4** |
| ├ Raw | 49.9 | 88.6 | - | **27.8** |
| ├ Raw-Noisy | 53.3 | 89.5 | - | **41.8** |
| ├ Concat-300s | 410.8 | 140.0 | - | **24.8** |
| └ Concat-Cross-lingual | - | - | - | **42.5** |

**论文结论**：在 human-labeled 测试集上，相对其他方法**累积平均偏移降低 67%~77%**。且基线方法在长音频上急剧退化（Monotonic-Aligner 在 300s 上从 161.1ms 恶化到 1742.4ms），Qwen3-ForcedAligner 在短/长音频上表现一致（42.9 → 52.9）。


### 25. TTS 指标

**N/A — ASR 模型无此项。**

Qwen3-ASR 不生成语音，因此没有 WER（TTS 侧的内容一致性）、SIM（说话人相似度）、UTMOS / PESQ（音质）、首包延迟等 TTS 指标。

需要 TTS 指标请见 `qwen3_tts.json` 的 `tts_metrics` 字段。

唯一与本模型相关、可能被误读为「TTS 指标」的是：**论文 §4.2.6 的「Long Speech Generation」评测中，Qwen 团队用 Qwen3-ASR 作为转写器来评测 Qwen3-TTS 的长语音 WER**——此处 Qwen3-ASR 是评测工具而非被测对象。


### 26. 延迟与吞吐

**测试条件（arXiv:2601.21337 §2.4）**：
- 引擎：**vLLM v0.14.0**，启用 **CUDA Graph**，**bfloat16** 精度，**单张典型算力资源（single typical computing resource，论文未指明具体 GPU 型号 → 卡型待核实）**
- Offline 列：vLLM 离线批生成（offline batch generation）
- Online async 列：基于 **vLLM Serve 的多并发请求**设置（更贴近工业环境）
- 输入音频长度：**ASR 约 2 分钟**，**ForcedAligner 约 1 分钟**
- TTFT avg. = 平均首 token 延迟；TTFT p95 = 95 分位 TTFT
- Throughput 单位：音频秒 / 秒（x 实时）

#### Qwen3-ASR-0.6B

| 并发 | Offline RTF | Offline 吞吐 | Online RTF | Online 吞吐 | TTFT avg. (ms) | TTFT p95 (ms) |
|---|---|---|---|---|---|---|
| 1 | 0.00923 | 108.34 | **0.00940** | **106.38** | **92** | 105 |
| 2 | 0.01124 | 177.94 | 0.01108 | 180.51 | 103 | 168 |
| 4 | 0.01284 | 311.53 | 0.01224 | 326.80 | 132 | 203 |
| 8 | 0.01600 | 500.00 | **0.01472** | **543.48** | **228** | 417 |
| 16 | 0.02384 | 671.14 | 0.01936 | 826.45 | 459 | 882 |
| 32 | 0.03808 | 840.34 | **0.02912** | **1098.90** | **820** | 1575 |
| 64 | 0.06336 | 1010.10 | 0.04352 | 1470.59 | 1631 | 3196 |
| 128 | 0.11264 | 1136.36 | **0.06400** | **2000.00** | **3210** | 6195 |
| 256 | 0.21504 | 1190.48 | - | - | - | - |
| 512 | 0.44544 | 1149.43 | - | - | - | - |

#### Qwen3-ASR-1.7B

| 并发 | Offline RTF | Offline 吞吐 | Online RTF | Online 吞吐 | TTFT avg. (ms) | TTFT p95 (ms) |
|---|---|---|---|---|---|---|
| 1 | 0.01482 | 67.48 | 0.01483 | 67.43 | **102** | 113 |
| 2 | 0.01540 | 129.87 | 0.01530 | 130.72 | 117 | 170 |
| 4 | 0.01712 | 233.64 | 0.01688 | 236.97 | 135 | 192 |
| 8 | 0.02072 | 386.10 | 0.02000 | 400.00 | 224 | 382 |
| 16 | 0.02896 | 552.49 | 0.02640 | 606.06 | 443 | 791 |
| 32 | 0.04608 | 694.44 | 0.03968 | 806.45 | 847 | 1570 |
| 64 | 0.07360 | 869.57 | 0.06208 | 1030.93 | 1597 | 2942 |
| 128 | 0.13056 | 980.39 | **0.10496** | **1219.51** | **3392** | 6227 |
| 256 | 0.24320 | 1052.63 | - | - | - | - |
| 512 | 0.50176 | 1020.41 | - | - | - | - |

#### Qwen3-ForcedAligner-0.6B（**仅 Offline**，PyTorch + FlashAttention + bf16）

| 并发 | RTF | 吞吐 |
|---|---|---|
| 1 | 0.00889 | 112.49 |
| 2 | **0.00232** | **862.07** |
| 4 | 0.00432 | 925.93 |
| 8 | 0.00832 | 961.54 |
| 16 | 0.01696 | 943.40 |
| 32 | 0.03584 | 892.86 |
| 64 | 0.08192 | 781.25 |
| 128 | 0.19712 | 649.35 |

> 论文说明：因为模型是 NAR，Transformers 与 vLLM 的推理速度差异很小，所以 FA 的基准**全部用 Transformers 跑**；高并发下 RTF 仍接近 0.001 量级，即每秒可处理约 1,000 秒音频。

#### 关键工程读数

1. **0.6B 在并发 128 达到 2000x 吞吐上限**，再往上（256/512）吞吐不再增长（1136/1190 offline），说明**已触及该单卡的算力上限**，扩容需横向加卡。
2. **1.7B 的吞吐天花板约为 1219x（并发 128，online）**，约为 0.6B 的 61%。
3. **TTFT 随并发近似线性恶化**：0.6B 从 92ms（并发 1）→ 3210ms（并发 128）。**若业务对首字延迟敏感（实时字幕/语音助手），并发必须控制在 8 以内**（0.6B 并发 8 时 TTFT 228ms）。
4. **Offline 与 Online 的 RTF 差距在高并发下显著**：0.6B 并发 128 时 offline RTF 0.113 vs online 0.064 —— online 异步服务的实际效率更高。
5. **FlashAttention 2 的收益**：官方 README 称 FA2 可「减少显存占用并加速，尤其对长输入/大批次」。具体加速幅度未量化 → **待核实**。
6. **FA2 缺失时的性能**：Qwen3-TTS 侧有明确的「无 FA2 时 RTX 5090 仅 0.3x 实时」的社区实测；**ASR 侧未见对应数据 → 待核实**。


### 27. 成本模型

#### 路径 A：自部署（开源权重，Apache 2.0）

- **许可成本**：0（Apache 2.0，可商用）
- **成本构成**：GPU 算力 + 存储 + 运维
- **吞吐换算示例**（基于论文 Table 2，0.6B，online async，输入约 2 分钟音频）：
  - 并发 1：RTF 0.0094 → 单卡理论可支撑 **约 106 倍实时**的音频流转写
  - 并发 128：RTF 0.064 → **2,000 秒音频 / 秒** ≈ 单卡每小时处理 7,200,000 秒（2,000 小时）音频
  - 按 1.7B 并发 128：1,219 秒音频/秒 ≈ 单卡每小时 4,388,400 秒（1,219 小时）音频
- **注意**：上述为**理想批吞吐**，未计入请求调度、音频下载/解码、网络 I/O、故障冗余。工程上建议按 **50%~70% 折扣**估算。
- **单位 GPU 成本**：官方未提供任何自建成本基准 → **待核实**（需结合自身机房/云 GPU 单价换算）

#### 路径 B：阿里云百炼 DashScope API（官方定价页面，help.aliyun.com/zh/model-studio/billing-for-model-studio）

**ASR 非实时（录音文件识别 / 千问ASR），按输入音频秒数计费，输出不计费：**

| 模型 ID | 地域 | 输入单价 | 免费额度 |
|---|---|---|---|
| `qwen3-asr-flash-filetrans` / `-2025-11-17` | 华北2（北京） | **0.00022 元/秒** | 36,000 秒（10 小时），自开通/发布起 90 天内 |
| `qwen3-asr-flash`（≡ `qwen3-asr-flash-2025-09-08`）/ `-2026-02-10` | 华北2（北京） | **0.00022 元/秒**（与 filetrans 同行） | 36,000 秒（10 小时） |
| `qwen3-asr-flash-filetrans` / `qwen3-asr-flash` | 国际（新加坡） | **0.00026 元/秒** | — |
| `qwen3-asr-flash-us` | 美国（弗吉尼亚） | **0.000035 元/秒** | — |

**ASR 实时（千问ASR-Realtime）：**

| 模型 ID | 地域 | 输入单价 | 免费额度 |
|---|---|---|---|
| `qwen3-asr-flash-realtime` / `-2026-02-10` / `-2025-10-27` | 华北2（北京） | **0.00033 元/秒** | 36,000 秒（10 小时） |
| 同上 | 国际（新加坡） | **0.00066 元/秒** | — |

**数量级换算（按北京非实时 0.00022 元/秒）**：
- 1 小时音频 = 3,600 秒 × 0.00022 ≈ **0.79 元**
- 1,000 小时音频 ≈ **792 元**
- 实时 API（0.00033 元/秒）：1 小时 ≈ **1.19 元**

**交叉印证**：第三方文章引用的「0.00033 CNY/s（中国大陆）/ 0.00066 CNY/s（国际）」与官方 **realtime** 行完全吻合；但该文将其标注为 `qwen3-asr-flash-realtime` 的实时价，若你用的是 filetrans（0.00022）会更便宜。

#### 成本决策建议

| 场景 | 建议 |
|---|---|
| 音频量 < 免费额度（10 小时）或 < 数百小时/月 | DashScope API（免运维、免 GPU） |
| 数据合规要求不出内网 / 高并发长音频批处理 | 自部署 vLLM（0.6B 优先） |
| 实时字幕且并发 < 8 路 | 自部署（TTFT 92–228ms 优于大多数 API 链路） |
| 需要方言/歌唱/定制微调 | 必须自部署（API 不支持微调） |

> ⚠️ 阿里云定价随时调整，且存在「限时折扣/活动优惠」，上线前请以百炼控制台实际价格为准。


### 28. 推理框架支持

#### 1. 官方 Python 包 `qwen-asr`

| 安装方式 | 后端 | 说明 |
|---|---|---|
| `pip install -U qwen-asr` | **Transformers** | 基础后端，支持批推理、时间戳 |
| `pip install -U qwen-asr[vllm]` | **vLLM** | 推荐；支持批推理、异步服务、**流式推理**、时间戳 |
| `git clone` + `pip install -e .` / `pip install -e ".[vllm]"` | 源码 | 开发用 |

环境建议：`conda create -n qwen3-asr python=3.12 -y`。

##### Transformers 后端示例

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32,
    max_new_tokens=256,
)
results = model.transcribe(audio="https://.../asr_en.wav", language=None)
print(results[0].language, results[0].text)
```

##### vLLM 后端示例

```python
from qwen_asr import Qwen3ASRModel

if __name__ == '__main__':
    model = Qwen3ASRModel.LLM(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.7,
        max_inference_batch_size=128,
        max_new_tokens=4096,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )
    results = model.transcribe(audio=[...], language=["Chinese"], return_time_stamps=True)
```

#### 2. 原生 vLLM（day-0 官方支持）

```bash
uv venv && source .venv/bin/activate
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
uv pip install "vllm[audio]"
```

启动：`vllm serve Qwen/Qwen3-ASR-1.7B`

**同时支持两套 OpenAI 兼容接口**：
- `/v1/chat/completions`（messages 中 `{"type":"audio_url","audio_url":{"url":"..."}}`）
- `/v1/audio/transcriptions`（`client.audio.transcriptions.create`）
- 离线：`LLM(model=...)` + `llm.chat(...)`

#### 3. ForcedAligner 在 vLLM 中的用法（pooling runner）

vLLM 官方示例（`examples/pooling/token_classify`）提供离线与在线两种 forced alignment 脚本，关键启动参数：

```bash
vllm serve Qwen/Qwen3-ForcedAligner-0.6B \
  --runner pooling --enforce-eager --trust-request-chat-template \
  --hf-overrides '{"architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]}'
```

config 中暴露 `timestamp_token_id` 与 `timestamp_segment_time`；预测索引 × `timestamp_segment_time` = 毫秒时间戳。

> 注意：论文 Table 2 的 FA 基准**全部用 Transformers 跑**（NAR 模型下 vLLM 收益有限）；vLLM pooling 路径是后续补充的 serving 方案。

#### 4. 官方 CLI / Demo

| 命令 | 用途 |
|---|---|
| `qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000` | 启动服务 |
| `qwen-asr-demo --asr-checkpoint ... --backend transformers/vllm [--aligner-checkpoint ...]` | Gradio Web Demo |
| `qwen-asr-demo-streaming --asr-model-path ... --gpu-memory-utilization 0.9` | Flask 流式麦克风 Demo |

#### 5. Docker

```bash
docker run --gpus all --name qwen3-asr -p 8000:80 \
  --mount type=bind,source=$LOCAL_WORKDIR,target=/data/shared/Qwen3-ASR \
  --shm-size=4gb -it qwenllm/qwen3-asr:latest
```
需 NVIDIA Container Toolkit。

#### 6. 第三方生态（非官方，需自行评估）

- `groxaxo/Qwen3-ASR-FastAPI`：OpenAI 兼容 FastAPI 封装，支持 `/v1/audio/transcriptions`、SRT 字幕、量化（4bit/AWQ/GPTQ，vLLM 0.14 下 4bit 支持有限会自动回落 BF16）、Docker Compose、Open-WebUI 集成。**第三方，未经官方背书。**


### 29. 资源需求

#### 官方明示

| 项 | 要求 | 来源 |
|---|---|---|
| Python | 建议 3.12（官方示例用 conda 隔离环境） | 官方 README |
| 精度 | `torch.bfloat16`（官方所有示例与评测默认） | 官方 README / 论文 §2.4 |
| FlashAttention 2 | **强烈推荐**；仅在 `float16`/`bfloat16` 加载时可用；需硬件兼容（Ampere+）；安装时 RAM<96GB 用 `MAX_JOBS=4` | 官方 README |
| CUDA | vLLM nightly 走 cu129 索引 | 官方 README |
| vLLM 显存 | `gpu_memory_utilization` 示例 0.7（Python）/ 0.8（serve）/ 0.9（流式 Demo） | 官方 README |
| Docker | `--gpus all`、`--shm-size=4gb` | 官方 README |
| 采样率 | 16,000 Hz | 流式 Demo 说明 |

#### 官方未明示（待核实）

| 项 | 状态 |
|---|---|
| **推理显存（VRAM）表格** | **官方未提供**。第三方技术博客给出权重级估算：0.6B ≈ 1 GB / 1.7B ≈ 1.6 GB（**非官方，且未说明是否含 KV cache 与激活值，仅供参考**） |
| **最低 GPU 型号 / 算力要求** | 未给出 |
| **CPU-only 推理可行性** | 未说明；第三方博客称 0.6B 可在 CPU 运行（**待核实**，且性能未知） |
| **量化方案（GPTQ/AWQ/INT8/INT4）** | **官方未提供量化权重或量化指南**；仅第三方 FastAPI 封装尝试接入 bitsandbytes / AWQ / GPTQ，并注明 vLLM 0.14 下 4bit 支持有限、会自动回落 BF16 |
| **吞吐测试所用 GPU 型号** | 论文只写 "single typical computing resource" → **待核实** |
| **磁盘占用** | 未给出（可自行按 bf16 权重 × 参数量估算：1.7B ≈ 3.4 GB，0.6B ≈ 1.2 GB，含编码器后略高） |

#### 工程建议

1. 上线前**必须实测显存**：以 `gpu_memory_utilization=0.8` 起步，观察 vLLM 的 preallocated 显存与 KV cache 可用量。
2. 长音频（接近 1200s）会显著抬高 KV cache 占用——12.5Hz × 1200s = 15,000 音频帧 token，**务必验证最长音频场景下的 OOM 边界**。
3. 并发规划参考 latency_throughput 字段：0.6B 在并发 128 已触及单卡吞吐天花板。


### 30. 服务化栈

#### 部署路径总览

```
                    ┌─ Transformers 后端（pip install qwen-asr）── 批推理 / 时间戳 / Gradio Demo
Qwen3-ASR 权重 ─────┤
                    ├─ vLLM 后端（pip install qwen-asr[vllm]）── 批推理 / 异步 / 流式 / 时间戳
                    ├─ 原生 vLLM（vllm serve）── /v1/chat/completions + /v1/audio/transcriptions
                    ├─ Docker（qwenllm/qwen3-asr:latest）
                    └─ DashScope API（闭源 Qwen3-ASR-Flash-1208）
```

#### 1. 官方 serve

```bash
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```

#### 2. 客户端调用

**方式 A：`/v1/chat/completions`**
```python
import requests
data = {"messages":[{"role":"user","content":[
    {"type":"audio_url","audio_url":{"url":"https://.../asr_en.wav"}}]}]}
r = requests.post("http://localhost:8000/v1/chat/completions", json=data, timeout=300)
from qwen_asr import parse_asr_output
language, text = parse_asr_output(r.json()['choices'][0]['message']['content'])
```

**方式 B：OpenAI SDK（chat / audio 两种都行）**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(model="Qwen/Qwen3-ASR-1.7B", messages=[...])
# 或
resp = client.audio.transcriptions.create(model="Qwen/Qwen3-ASR-1.7B", file=f)
```

**方式 C：cURL**
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-ASR-1.7B","messages":[{"role":"user","content":[{"type":"audio_url","audio_url":{"url":"..."}}]}]}'
```

#### 3. 能力矩阵（官方 README）

| 能力 | Transformers 后端 | vLLM 后端 |
|---|---|---|
| 批推理 | ✅ | ✅ |
| 异步服务 | - | ✅ |
| 流式推理 | ❌ | ✅ |
| 时间戳预测（需 ForcedAligner） | ✅ | ✅（但**流式模式下不可返回时间戳**） |

#### 4. 时间戳服务的部署形态

- 需在启动/加载时指定 aligner：`forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B"`（Python）或 `--aligner-checkpoint`（Demo CLI）
- **不加载 aligner 时，Demo 的时间戳 UI 自动隐藏**
- ForcedAligner 在 vLLM 中走 **pooling runner**（`Qwen3ASRForcedAlignerForTokenClassification` 架构覆写），需 `--enforce-eager`
- 论文 Table 2 显示 FA 仅支持 **offline batch inference in PyTorch**

#### 5. 运维注意事项

| 项 | 说明 |
|---|---|
| 设备选择 | **vLLM 不遵循 `cuda:0` 风格**，需 `CUDA_VISIBLE_DEVICES` 或 Demo 的 `--cuda-visible-devices` |
| 绑定地址 | 容器/服务必须绑定 `0.0.0.0`，不能是 `127.0.0.1` |
| HTTPS | 远程或现代浏览器下麦克风权限需 HTTPS；可用 `--ssl-certfile/--ssl-keyfile`（自签证书会有告警，生产用真实证书） |
| 共享内存 | Docker 需 `--shm-size=4gb` |
| 版本 | vLLM 需 nightly（cu129 索引），论文数据基于 vLLM v0.14.0 |

#### 6. 缺失能力（待核实）

- 无官方 Kubernetes / Helm 部署清单
- 无官方 Triton / TensorRT-LLM 后端
- 无官方自动扩缩容（HPA）方案
- 无官方 streaming WebSocket 服务协议（流式 Demo 为 Flask 自定义实现）


### 31. 硬件生态

#### 已验证 / 官方支持的硬件路径

| 层次 | 支持情况 | 说明 |
|---|---|---|
| **NVIDIA GPU（CUDA）** | ✅ 主路径 | 官方所有示例、Docker、vLLM 均基于此 |
| **FlashAttention 2 兼容硬件** | ✅ 推荐 | 官方明示「需与 FA2 兼容的硬件」，即 NVIDIA Ampere 架构及以后（A100/RTX 30xx 及更新）。**Blackwell（RTX 50 系）需配套 CUDA 12.x 与正确的 flash-attn 构建** |
| **vLLM 生态** | ✅ day-0 | vLLM v0.14.0（论文基准）；nightly cu129 wheel |
| **Docker + NVIDIA Container Toolkit** | ✅ | 官方镜像 `qwenllm/qwen3-asr:latest` |
| **ModelScope / HuggingFace 权重分发** | ✅ | 中国大陆推荐 ModelScope |

#### 官方未覆盖 / 待核实

| 硬件 / 后端 | 状态 |
|---|---|
| **AMD GPU（ROCm）** | **官方未提及** → 待核实。（注：同系列 Qwen3-TTS 的 vLLM-Omni 文档列出了 ROCm 依赖 `onnxruntime-rocm` + `sox`，但 ASR 侧无对应说明） |
| **Apple Silicon（MPS）** | 官方未提及 → 待核实 |
| **国产 NPU（昇腾 / 寒武纪等）** | 官方未提及 → 待核实 |
| **CPU-only 推理** | 官方未说明；第三方博客称 0.6B 可在 CPU 跑（**待核实**，性能未知） |
| **端侧 / 移动端部署**（NCNN、MNN、CoreML、TFLite） | **官方未提供任何方案**。第三方评测文章明确将「推出更小参数量模型（如 0.1B）以适配手机/智能手表」列为改进方向，说明当前 0.6B 对端侧仍偏大 |
| **量化推理**（INT8 / INT4 / AWQ / GPTQ / FP8） | **官方未发布量化权重或量化指南** |

#### 与同系列 Qwen3-TTS 的硬件依赖差异

| | Qwen3-ASR | Qwen3-TTS |
|---|---|---|
| FlashAttention 2 | 推荐（省显存、长输入/大批次加速） | **强依赖**（社区实测：无 FA2 时 RTX 5090 上仅约 0.3x 实时） |
| 显存门槛 | 较低（0.6B 权重约 1GB 量级） | 较高（0.6B 约 2GB，1.7B 约 6GB，社区实测 1.7B 推理约 4.0–7.8 GB） |
| vLLM 路径 | vLLM 主线（day-0） | vLLM-Omni（**当前仅离线推理**） |

#### 工程建议

1. **优先 NVIDIA Ampere+**：这是唯一被官方完整验证的路径。
2. **必须安装 flash-attn**：尤其在长音频（接近 20 分钟）与大批次场景。
3. **边缘/端侧**：当前开源尺寸（0.6B LLM + 180M 编码器）对手机仍偏重；若必须端侧，需自行量化或蒸馏，**官方无支持**。


### 32. 集成范式

#### 三种官方集成形态

| 形态 | 适用场景 | 延迟 | 运维成本 |
|---|---|---|---|
| **Transformers 库内嵌** | 离线批处理、研究、低并发内部工具 | 中 | 低 |
| **vLLM 服务化（OpenAI 兼容）** | 生产环境、高并发、流式 | 低 | 中 |
| **DashScope API** | 无 GPU、快速验证、弹性用量 | 取决于网络 | 极低 |

#### 1. 库内嵌（Python）

```python
from qwen_asr import Qwen3ASRModel
model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16, device_map="cuda:0",
    max_inference_batch_size=32, max_new_tokens=256)
results = model.transcribe(audio=["a.wav","b.wav"], language=None)
```

#### 2. OpenAI 兼容服务（推荐生产形态）

- 端点：`/v1/chat/completions` 与 `/v1/audio/transcriptions`
- 好处：**现有 OpenAI SDK / Open-WebUI / LangChain 等生态可直接接入**，只需改 `base_url`
- 解析：必须用 `parse_asr_output()` 拆出 `language` 与 `text`，因为原始输出是 `language English<asr_text>...` 模板

#### 3. 流式集成（实时字幕 / 语音助手）

- 仅 vLLM 后端；浏览器端捕获麦克风 → 重采样 16kHz → 持续推送 PCM chunk
- 参考实现：`examples/example_qwen3_asr_vllm_streaming.py`
- 约束：不支持批处理、不支持时间戳

#### 4. 时间戳 / 字幕流水线

```
音频 → Qwen3-ASR（转写）→ Qwen3-ForcedAligner（对齐）→ 词/字级时间戳 → SRT/字幕文件
```
- 两个模型都要加载；FA 独占一份显存
- FA 最长 300 秒；超过需先分段
- 时间戳粒度可自定义（在任意词/字后插入槽位）

#### 5. 垂直领域适配（无需微调）

利用 **context biasing**：把领域术语、人名、产品名写入 system prompt。官方明示这是 SFT 阶段专门训练的能力，比微调成本低得多。

#### 6. 与 LLM 应用组合

典型语音助手链路：
```
麦克风/音频流 → Qwen3-ASR（vLLM 流式）→ 文本 → 通用 LLM（Qwen3 等）→ 文本 → Qwen3-TTS → 语音
```
Qwen3-ASR 与 Qwen3-TTS 共享 Qwen 生态（同一 tokenizer 体系、同一 vLLM 生态、同一 Apache 2.0 许可），**是同栈组合的天然选择**。

#### 7. 已验证的第三方集成（非官方）

- **Open-WebUI**：通过 `groxaxo/Qwen3-ASR-FastAPI` 作为 OpenAI speech endpoint 接入，无需 API key
- **Docker Compose**：同上仓库提供 GPU 容器与健康检查

#### 集成注意事项

1. **不要对 Qwen3-ASR 发自然语言指令**——它被刻意训练为 ASR-only，不遵循指令（防 prompt injection）。
2. **语种参数**：`language=None`（自动 LID）或显式指定；批量时可传 list。
3. **输出解析**：`parse_asr_output()` 是官方解析器，不要自己写字符串切分（输出模板含「无语音」分支）。
4. **超时**：官方 cURL/requests 示例用 `timeout=300`，长音频场景需相应调大。


### 33. 技术栈组件

| 层次 | 组件 | 说明 |
|---|---|---|
| **模型权重** | Qwen3-ASR-1.7B / 0.6B / ForcedAligner-0.6B | Apache 2.0，HF + ModelScope 分发 |
| **基座** | Qwen3-Omni（arXiv:2509.17765） | 3T token 音视频文多模态预训练 |
| **声学编码器** | AuT（AED 框架，4,000 万小时伪标签预训练） | 128 维 Fbank → 8x 下采样 → 12.5Hz |
| **LLM 骨干** | Qwen3-1.7B / Qwen3-0.6B | 自回归解码，LID + ASR 一体 |
| **对齐模块** | Projector | 编码器表示 → LLM embedding 空间 |
| **时间戳模块** | `[time]` 特殊 token + Timestamp Prediction Linear Layer（3,750 类） | 仅 ForcedAligner |
| **推理框架** | `qwen-asr`（Transformers / vLLM 双后端） | 官方 Python 包 |
| **推理引擎** | vLLM v0.14.0（day-0 支持，CUDA Graph + bf16） | 批推理 / 异步 / 流式 |
| **服务接口** | OpenAI 兼容：`/v1/chat/completions`、`/v1/audio/transcriptions` | 官方 + `qwen-asr-serve` |
| **Web UI** | Gradio Demo（`qwen-asr-demo`）、Flask 流式 Demo（`qwen-asr-demo-streaming`） | 官方 |
| **容器** | `qwenllm/qwen3-asr:latest` + NVIDIA Container Toolkit | 官方 |
| **加速库** | FlashAttention 2（`flash-attn`）、`torch.compile`（社区实践） | FA2 官方推荐 |
| **微调** | `finetuning/qwen3_asr_sft.py` + `torchrun` | 全参数；LoRA 待核实 |
| **数据工具** | `datasets`（官方微调依赖）、ModelScope / `huggingface_hub[cli]`（权重下载） | |
| **音频 I/O** | `torchaudio` / `soundfile` / `librosa`（社区实践）；输入统一 16kHz | |
| **伪标签工具** | Montreal Forced Aligner (MFA) | **仅 ForcedAligner 训练阶段使用**，推理不需要 |
| **云 API** | 阿里云百炼 DashScope（`qwen3-asr-flash` / `-realtime` / `-filetrans`） | 闭源，按秒计费 |
| **评测指标** | WER / CER（识别）、Accuracy（LID）、AAS（时间戳，ms） | 论文定义 |


### 34. 评测工具链

#### 评测协议（arXiv:2601.21337 §4.1）

论文采用**四部分评测协议**，核心动机是：「当前 ASR 模型在若干开源测试集上已触及标注误差上限，开源集分数差异很小，但在真实场景差异显著」——因此**自建内部基准是本文方法论的重点**。

| 部分 | 内容 |
|---|---|
| 1. 公开基准（英/中） | LibriSpeech、GigaSpeech、CommonVoice、Fleurs、MLS、Tedlium、VoxPopuli、WenetSpeech、AISHELL-2、SpeechIO、KeSpeech、WenetSpeech-Yue、WenetSpeech-Chuan |
| 2. 内部鲁棒性套件 | 16 组英语口音、22 种中国方言、老人与儿童语音、极低 SNR、非流畅与绕口令式重复语音、多说话人中文对话 |
| 3. 多语言评测 | CommonVoice、Fleurs、MLS、MLC-SLM + 内部 15 语种新闻测试集；并做 LID 评测 |
| 4. 歌唱识别 | M4Singer、MIR-1k-vocal、Opencpop、Popcs + 内部「整首歌作为单次输入」测试集 |

#### 指标定义

| 指标 | 定义 | 适用 |
|---|---|---|
| **WER** | 词错误率 ↓ | 词分隔语言（英、德、法等） |
| **CER** | 字错误率 ↓ | 字符类语言（普通话、粤语、韩语等） |
| **Macro-average** | 跨语言/方言的无加权均值 | 多语言聚合 |
| **LID Accuracy** | 语种识别准确率 ↑ | 语种识别 |
| **AAS** | Accumulated Average Shift = (1/N)Σ|n̂ᵢ−nᵢ|，ms ↓ | 时间戳精度 |

#### 基线系统

- 闭源 API：**GPT-4o-Transcribe**、**Gemini-2.5-Pro**、**Doubao-ASR**（对于无公开成绩的商业 API 与 Whisper-large-v3，作者**自行在测试集上跑推理**得到结果）
- 开源基线：**Whisper-large-v3**、**FunASR-MLT-Nano**（引自其官方技术报告）、**GLM-ASR-Nano-2512**

#### 统一推理配置（官方 README）

`dtype=bfloat16`、`max_new_tokens=1024`、vLLM、greedy 解码、**不指定语言参数**（即让模型自动 LID）。

#### 官方未提供的评测工具（待核实）

- **无开源的内部基准数据集**（内部测试集未发布）
- **无官方 eval 脚本 / benchmark harness**（仓库 `finetuning/` 有 `--eval_file` 参数，但输出指标未文档化）
- **无公开的 AAS 计算脚本**（论文给出公式，代码未随包发布）
- 第三方评测者需自行用 `jiwer` / `wenet` 等工具实现 WER/CER，AAS 需按论文公式自实现

#### 自建评测建议

1. **不要只看 LibriSpeech / AISHELL**：Qwen3-ASR 在这类「干净朗读」集上与 Whisper 差距最小，其优势集中在噪声、方言、歌唱、长音频。
2. **务必自建内部集**：论文的方法论表明，真实场景差异在开源集上体现不出来。
3. **长音频务必测 300s 以上的时间戳**：基线方法（MFA/WhisperX/NFA）在长音频上会崩，Qwen3-ForcedAligner 不会。
4. **流式场景单独测**：流式相对离线劣化 +0.64（1.7B）/ +0.92（0.6B）。


### 35. 已知坑与限制

#### 官方明示的功能限制

1. **流式推理三限制**（官方 README）：
   - **仅 vLLM 后端支持**（Transformers 后端不支持流式）；
   - **不支持批推理**；
   - **不支持返回时间戳**。
2. **ForcedAligner 范围限制**：
   - 仅支持 **11 种语言**（少于 ASR 的 52 种）；
   - 最长 **300 秒**语音；
   - 需**额外加载一个模型**（约 0.6B 参数），增加部署复杂度与显存。
3. **FlashAttention 2 限制**：仅 `float16`/`bfloat16` 加载时可用，需硬件兼容。
4. **时间戳依赖 aligner 参数**：不带 `--aligner-checkpoint` / `forced_aligner=` 时，Demo 的时间戳 UI 自动隐藏。
5. **HTTPS 要求**：远程/现代浏览器下麦克风权限需 HTTPS，自签证书会告警。
6. **端口绑定**：容器/服务必须绑定 `0.0.0.0`。
7. **vLLM 设备选择**：不遵循 `cuda:0` 风格，需 `CUDA_VISIBLE_DEVICES`。

#### 官方承认的性能短板（论文原文）

1. **Fleurs 全 30 语言子集**：Qwen3-ASR-1.7B **12.60** vs Whisper-large-v3 **8.16** —— 论文自述「语言多样性增加时长尾语言处理仍有改进空间」。
2. **WenetSpeech-Chuan hard**：Qwen3-ASR-1.7B 21.63 vs Doubao-ASR **20.20**。
3. **EntireSongs-en**：Qwen3-ASR-1.7B 14.60 vs Gemini-2.5-Pro **12.18**。
4. **Opencpop**：Qwen3-ASR-1.7B 3.08 vs Fun-ASR-MLT-Nano **2.98**。
5. **LID 剩余错误**：主要来自**马来语（ms）与印尼语（id）**的混淆。

#### 能力缺失（论文未列为已支持，第三方整理）

1. **无说话人分离（diarization）**——第三方评测文章将其列为「改进方向」，说明当前不具备。
2. **情绪识别**——开源权重不具备；仅在 DashScope 云端 API 提供（**第三方来源，待核实**）。
3. **不支持自然语言指令控制**——这是**刻意设计**（防 prompt injection），非缺陷，但会限制用法。
4. **无官方量化权重 / 端侧部署方案**——0.6B 对手机/手表仍偏大（第三方评测观点）。

#### 部署/工程侧坑点（社区 + 官方文档综合）

1. **vLLM 需要 nightly 版本**（cu129 索引），stable 版可能不支持该架构。
2. **vLLM 4-bit 量化支持有限**（vLLM 0.14 下会自动回落 BF16，第三方 FastAPI 封装有明确日志记录）。
3. **服务必须绑定 0.0.0.0**，否则容器内端口转发失败。
4. **长音频 KV cache 压力**：1200s 音频 ≈ 15,000 帧 token，需实测 OOM 边界。
5. **并发与 TTFT 强相关**：0.6B 并发 1 时 92ms，并发 128 时 3210ms；实时场景需控制并发 < 8。
6. **显存需求官方未公布**——第三方「0.6B 1GB / 1.7B 1.6GB」是权重级估算，**不可直接当作部署显存**。

#### 未见报告的问题（即尚未观察到，但需注意）

- 未见官方 issue 中报告「微调后语速/精度异常」这类问题（**对比：Qwen3-TTS 的官方微调脚本存在已知的 double label-shift bug，见 `qwen3_tts.json`**）。ASR 侧微调脚本目前**未见同类报道**。


### 36. 社区与文档

#### 一手权威来源

| 类型 | 链接 |
|---|---|
| **技术报告（论文）** | arXiv:2601.21337《Qwen3-ASR Technical Report》 https://arxiv.org/abs/2601.21337 （v1 2026-01-29，v2 2026-01-30） |
| **论文 HTML 全文** | https://arxiv.org/html/2601.21337v2 |
| **GitHub 仓库** | https://github.com/QwenLM/Qwen3-ASR |
| **HuggingFace Collection** | https://huggingface.co/collections/Qwen/qwen3-asr |
| **ModelScope Collection** | https://modelscope.cn/collections/Qwen/Qwen3-ASR |
| **微调文档** | https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md |
| **Docker 镜像** | `qwenllm/qwen3-asr:latest` |

#### 模型权重

```bash
# ModelScope（中国大陆推荐）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen3-ASR-1.7B
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ForcedAligner-0.6B --local_dir ./Qwen3-ForcedAligner-0.6B

# Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B
```

#### 云 API 文档（DashScope）

- 实时 API 与 FileTrans API：官方 README 给出中国站与国际站的阿里云 Model Studio 文档入口（`help.aliyun.com/zh/model-studio/...`）
- 定价：https://help.aliyun.com/zh/model-studio/billing-for-model-studio

#### 关联论文（引用链）

| 论文 | arXiv | 与 Qwen3-ASR 的关系 |
|---|---|---|
| Qwen3-Omni Technical Report | 2509.17765 | **基座模型** |
| GSPO（Group Sequence Policy Optimization） | 2507.18071 | **RL 算法** |
| LLM-ForcedAligner | 2601.18220 | ForcedAligner 的模型设计细节（Mu et al. 2026b） |
| Fun-ASR Technical Report | 2509.12508 | 对比基线 |
| Seed-ASR | 2407.04675 | Doubao-ASR 的方法来源 |

#### 生态活跃度（定性）

- 论文已被引用 27 次（Google Scholar 快照，2026-08 检索）
- 有第三方 OpenAI 兼容封装（`groxaxo/Qwen3-ASR-FastAPI`）、Open-WebUI 集成、vLLM 官方示例
- **未获取到 GitHub star 数、issue 数、commit 频率等量化活跃度指标 → 待核实**

#### 引用格式

```bibtex
@article{Qwen3-ASR,
  title={Qwen3-ASR Technical Report},
  author={Xian Shi and Xiong Wang and Zhifang Guo and Yongqi Wang and Pei Zhang and
          Xinyu Zhang and Zishan Guo and Hongkun Hao and Yu Xi and Baosong Yang and
          Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.21337},
  year={2026}
}
```


### 37. 适用场景与选型建议

#### 强匹配场景（推荐优先选用）

| 场景 | 理由 | 推荐变体 |
|---|---|---|
| **中文 / 方言语音识别** | 中文公开集全面领先：WenetSpeech 4.97/5.88（Whisper-v3 9.86/19.11）；KeSpeech 方言 5.10（Whisper-v3 28.79） | 1.7B（精度）/ 0.6B（成本） |
| **高并发离线批处理**（会议转写、呼叫中心录音） | 0.6B 并发 128 达 2000x 吞吐，RTF 0.064 | **0.6B + vLLM** |
| **实时字幕 / 语音助手**（低并发） | 0.6B 并发 1–8 时 TTFT 92–228ms | **0.6B + vLLM 流式** |
| **噪声 / 复杂声学环境** | ExtremeNoise 16.17（Whisper-v3 63.17） | 1.7B |
| **歌唱 / 带 BGM 歌词转写** | 唯一能处理「整首歌 + BGM」的开源方案（Whisper/Fun-ASR 在该场景退化为 N/A） | 1.7B |
| **字幕生成 / 打标流水线** | ForcedAligner AAS 32.4ms（human-labeled），比 MFA/NFA/WhisperX 低 67–77%，且长音频不退化 | ASR + FA 组合 |
| **数据合规 / 私有化部署** | Apache 2.0，可完全内网部署 | 任意变体 |
| **垂直领域术语适配（不想微调）** | 内建 context biasing，system prompt 注入即可 | 任意变体 |
| **多语种内容平台（主流 20 语种内）** | Fleurs 12 语种 4.90、20 语种 6.62 均最优 | 1.7B |

#### 需要谨慎评估的场景

| 场景 | 风险 | 建议 |
|---|---|---|
| **长尾语言**（匈牙利语、波斯语、芬兰语、希腊语、罗马尼亚语等） | Fleurs 全 30 语言子集上 WER 高于 Whisper-large-v3（12.60 vs 8.16） | 务必自测；或选 Whisper-large-v3 |
| **东南亚语系（马来语/印尼语）** | LID 会混淆 ms 与 id | 显式传 `language` 参数 |
| **超高并发实时（>64 路并发且 TTFT 敏感）** | 0.6B 并发 64 时 TTFT 已达 1631ms | 横向扩容，或降低并发预期 |
| **端侧 / 移动端 / 嵌入式** | 无官方量化与端侧方案，0.6B 仍偏大 | 需自行量化/蒸馏；或选 Fun-ASR-Nano 等更小模型 |
| **需要说话人分离** | 官方不支持 | 需外接 diarization（pyannote 等） |
| **需要情绪识别** | 开源权重不支持 | 用 DashScope API（待核实） |
| **需要指令式控制**（如「把转写结果翻译成英文」） | 被刻意设计为 ASR-only | 后接通用 LLM 处理 |

#### 不推荐场景

- **纯英文单一语种、无并发压力**：Whisper-large-v3 在 LibriSpeech 上仍具竞争力（1.51/3.97），且生态更成熟、量化方案更多。Qwen3-ASR 的优势在中文、方言、噪声、长音频与吞吐。
- **无 GPU 环境**：官方未提供 CPU 优化路径。

#### 选型速查

```
需要中文/方言/歌唱/噪声鲁棒  ── 是 ──> Qwen3-ASR
         │                                    │
         否                          吞吐优先? ── 是 ──> 0.6B + vLLM
         │                                    │
   纯英文 + 端侧 ──> Whisper 系          否 ──> 1.7B + vLLM
                                              │
                                    需要时间戳? ── 是 ──> + ForcedAligner-0.6B
```

#### 与 Qwen3-TTS 的同栈组合价值

Qwen3-ASR + Qwen3-TTS 共享 Qwen 生态（同一 tokenizer 体系、同一 vLLM 生态、同为 Apache 2.0），是构建**全栈开源语音交互系统（ASR → LLM → TTS）**时的一致性最高的选型：**Qwen3-TTS 论文的长语音评测即用 Qwen3-ASR 作为转写器**，说明两者已在官方内部形成了评测闭环。


### 38. 参考来源

#### 一手权威来源

1. **Qwen3-ASR Technical Report**（主来源，本文件中标注 arXiv:2601.21337 的所有数据与结论）
   - 摘要页：https://arxiv.org/abs/2601.21337
   - HTML 全文 v2（含全部表格）：https://arxiv.org/html/2601.21337v2
   - PDF：https://arxiv.org/pdf/2601.21337

2. **GitHub 官方仓库**（安装、用法、流式、微调、Docker、基准表）
   - https://github.com/QwenLM/Qwen3-ASR
   - 微调文档：https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md

3. **HuggingFace 模型集合**：https://huggingface.co/collections/Qwen/qwen3-asr
   - `Qwen/Qwen3-ASR-1.7B-hf`、`Qwen/Qwen3-ASR-0.6B-hf`、`Qwen/Qwen3-ForcedAligner-0.6B-hf`

4. **ModelScope 模型集合**（中国大陆推荐）：https://modelscope.cn/collections/Qwen/Qwen3-ASR

5. **vLLM 官方文档**
   - vLLM 主页：https://docs.vllm.ai/
   - Forced Alignment 离线/在线示例（`examples/pooling/token_classify`）：https://vllm.readthedocs.io/en/latest/examples/pooling/token_classify

6. **阿里云百炼定价页**（ASR/TTS API 价格，一手）：https://help.aliyun.com/zh/model-studio/billing-for-model-studio

#### 关联论文

7. **Qwen3-Omni Technical Report**（基座模型）：https://arxiv.org/abs/2509.17765
8. **GSPO**（RL 算法）：Zheng et al., *Group Sequence Policy Optimization*, https://arxiv.org/abs/2507.18071
9. **LLM-ForcedAligner**（ForcedAligner 设计细节）：Mu et al. 2026b, https://arxiv.org/abs/2601.18220
10. **Qwen3-TTS Technical Report**（同系列，长语音评测中用 Qwen3-ASR 做转写器）：https://arxiv.org/abs/2601.15621

#### 对比基线论文

11. Whisper（Radford et al. 2023）：https://arxiv.org/abs/2212.04356
12. Fun-ASR Technical Report：https://arxiv.org/abs/2509.12508
13. Seed-ASR（Doubao-ASR 方法来源）：https://arxiv.org/abs/2407.04675
14. WenetSpeech（中文基准）：https://arxiv.org/abs/2110.13900（对应 ICASSP 2022）
15. MFA（Montreal Forced Aligner，ForcedAligner 伪标签来源）：McAuliffe et al., Interspeech 2017

#### 二手/社区来源（**仅用于交叉印证，本文件中已逐条标注「待核实」**）

16. besthub.dev《Qwen3-ASR: Open-Source Speech Recognition Supporting 52 Languages and Dialects》：https://besthub.dev/articles/qwen3-asr-...
17. qwen-ai.com《Qwen3-ASR — Open-Source Speech Recognition (52 Languages)》：http://qwen-ai.com/qwen-3-asr/
18. 第三方 OpenAI 兼容封装 `groxaxo/Qwen3-ASR-FastAPI`：http://github.de/groxaxo/Qwen3-ASR-FastAPI
19. CSDN 技术博客《Qwen3-ASR-1.7B-hf —— 让AI"听懂"52种语言和方言》（提供层配置细节，**与论文冲突，待核实**）
20. abeiai.cn《Qwen3-ASR:通义千问开源的多语言语音识别套件》（提及 VAD/SRT 工具链，**待核实**）

> 说明：本文件所有**数值均优先采用 arXiv:2601.21337 与 GitHub 官方 README**。二手来源仅用于补充官方未覆盖的工程细节，且已在对应字段明确标注为「待核实」。


> **本章定位**：Qwen3-ASR 是本次 ASR 侧的唯一核心样本，也是三款中文/方言能力最强的模型。


---

<!-- item: qwen3_tts -->

## Qwen3-TTS

> **分类**：核心模型-TTS　|　**已填字段**：38 / 38　|　**⭐ 标记为本轮重点的架构与训练字段**


### 1. 基本信息与版本演进

- **名称**：Qwen3-TTS（Qwen 系列第一个 TTS 模型）
- **团队**：阿里巴巴通义千问 Qwen Team（核心作者 Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, ... Jin Xu†, Junyang Lin†）
- **开源时间**：2026 年 1 月（官方 README 标注 **2026.1.22**）；技术报告《Qwen3-TTS Technical Report》**arXiv:2601.15621**
- **定位**：多语言（10 语种）、可控（自然语言指令）、鲁棒、**流式**的文本转语音模型家族
- **核心能力**：
  - **3 秒零样本声音克隆**（voice cloning）
  - **基于描述的声音设计**（VoiceDesign：用自然语言凭空创造新音色）
  - **9 个预设高质量音色 + 指令式风格控制**（CustomVoice）
  - **97ms 端到端首包延迟**（0.6B 变体）/ 101ms（1.7B 变体）
  - **10 分钟以上长语音稳定合成**（无分块拼接痕迹）
- **训练规模**：**超过 500 万小时**语音数据，覆盖 **10 种语言**
- **已开源内容**：`Qwen3-TTS-Tokenizer-12Hz` + 5 个 12Hz 模型（0.6B/1.7B × Base/CustomVoice，1.7B-VoiceDesign）。**25Hz 系列目前仅出现在论文评测表中，官方 README 明示「技术报告中提到的其他模型将在近期发布」，权重尚未开放。**
- **基座**：Qwen3 LM 家族（Qwen3-0.6B / Qwen3-1.7B）
- **代码与权重**：GitHub `QwenLM/Qwen3-TTS`、HuggingFace `Qwen/Qwen3-TTS-*`、ModelScope `Qwen/Qwen3-TTS-*`
- **官方一句话定位**：在 Seed-TTS 零样本克隆基准上建立新 SOTA（英文 WER 1.24），10 语种说话人相似度全面超过 MiniMax 与 ElevenLabs 等商业基线


### 2. 开源协议与商用边界

- **开源权重**：Apache License 2.0 —— 论文摘要原文 "we release both tokenizers and models under the Apache 2.0 license"
  - 已开源：`Qwen3-TTS-Tokenizer-12Hz`、`Qwen3-TTS-12Hz-1.7B-Base` / `-CustomVoice` / `-VoiceDesign`、`Qwen3-TTS-12Hz-0.6B-Base` / `-CustomVoice`
- **代码库**：Apache 2.0（含推理框架与 `finetuning/` 微调脚本）
- **闭源部分**：DashScope 云端 API（`qwen3-tts-flash` / `qwen3-tts-vc` / `qwen3-tts-vd` / `qwen3-tts-instruct-flash` 及其 `-realtime` 版本），按输入字符数计费，权重未开放
- **注意**：`flash-attn` 为独立第三方库，许可证与本项目无关；仅在 `torch.float16` / `torch.bfloat16` 加载时可用
- **合规提示（官方 README 未明示，但技术性质决定）**：3 秒声音克隆属高敏感能力，商业化部署需自行落实声音授权与 AI 生成内容标识义务


### 3. 模型变体与权重获取

#### 已发布（可下载）模型

| 模型 | 尺寸 | 能力 | 流式 | 指令控制 | 状态 |
|---|---|---|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | 按自然语言描述**设计全新音色** | ✅ | ✅ | 已发布 |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | 9 个预设音色 + 指令控制风格/情感 | ✅ | ✅ | 已发布 |
| `Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | **3 秒声音克隆**；可用于微调（FT） | ✅ | ❌ | 已发布 |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | 9 个预设音色 | ✅ | ❌ | 已发布 |
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | **3 秒声音克隆**；可用于微调 | ✅ | ❌ | 已发布 |
| `Qwen3-TTS-Tokenizer-12Hz` | — | 语音 ↔ 离散 codes 编解码 | — | — | 已发布 |

#### 论文中出现但**尚未发布**的模型（Table 1）

| 模型 | 状态 |
|---|---|
| `Qwen3-TTS-25Hz-1.7B-Base` | 仅评测数据 |
| `Qwen3-TTS-25Hz-1.7B-VoiceEditing` | 仅评测数据（**具备 Voice Editing 能力，开源系列中无对应变体**） |
| `Qwen3-TTS-25Hz-1.7B-CustomVoice` | 仅评测数据 |
| `Qwen3-TTS-25Hz-0.6B-Base` / `-CustomVoice` | 仅评测数据 |

> 官方 README 原文："Other models mentioned in the technical report will be released in the near future."

#### 完整型号空间

```
尺寸：      0.6B  |  1.7B
Tokenizer： 12Hz（已发布） | 25Hz（未发布）
变体：      Base（克隆 + 可微调） | CustomVoice（预设音色 + 指令） | VoiceDesign（设计音色） | VoiceEditing（仅 25Hz 论文版）
```

#### 参数与配置（Qwen3-TTS-Tokenizer-12Hz，官方 README）

| 项 | 值 |
|---|---|
| NQ（量化层数） | **16** |
| Codebook Size | **2048** |
| FPS | **12.5** |

#### 9 个预设音色（CustomVoice，官方 README + `get_supported_speakers()`）

| Speaker | 描述 | 母语 |
|---|---|---|
| Vivian | Bright, slightly edgy young female | Chinese |
| Serena | Warm, gentle young female | Chinese |
| Uncle_Fu | Seasoned male, low mellow | Chinese |
| Dylan | Youthful Beijing male | Chinese (Beijing) |
| Eric | Lively Chengdu male | Chinese (Sichuan) |
| Ryan | Dynamic male, rhythmic | English |
| Aiden | Sunny American male | English |
| Ono_Anna | Playful Japanese female | Japanese |
| Sohee | Warm Korean female | Korean |

> 官方建议：**每个 speaker 尽量用其母语合成以获得最佳质量**（跨语言质量可能略降）。


### 4. 架构范式 ⭐

#### 范式：离散多码本自回归 LM + Dual-Track 流式（**端到端，无 LM+DiT 级联**）

Qwen3-TTS 的架构主张在论文 Introduction 中表达得非常明确，这是它区别于 CosyVoice / FireRedTTS 等方案的核心：

> **「Universal End-to-End Architecture：离散 multi-codebook LM 架构，全信息端到端语音建模，绕过传统 LM+DiT 的信息瓶颈与级联错误（bypassing information bottlenecks and cascading errors inherent in traditional LM+DiT schemes）」**

##### 1. 为什么是离散 token 而不是连续特征

- 离散 speech tokenization + 自回归 LM 建模离散单元，能**在保持高自然度的同时提升稳定性**（引用 VALL-E、CosyVoice、Moshi 等工作）
- 离散表示便于与 LLM 无缝集成（论文明确将「与 LLM 无缝集成」和「极低首包延迟」并列为架构目标）

##### 2. 两条 Tokenizer 路线（本家族最重要的设计权衡）

论文对两条路线的取舍给出了直接的工程判断，非常有参考价值：

| | **Qwen-TTS-Tokenizer-25Hz** | **Qwen-TTS-Tokenizer-12Hz** |
|---|---|---|
| 表示 | **单码本**、25 Hz、codebook 32768 | **多码本（16 层 RVQ）**、12.5 Hz、codebook 2048 |
| 语义/声学 | 语义与声学**融合**在单个码本 | 第 1 层 = 语义；后 15 层 = 声学细节（**语义-声学解耦**） |
| 骨架 | 基于 **Qwen2-Audio** 继续预训练（ASR 任务） | 参考 **Mimi** 架构，GAN 训练，WavLM 作语义教师 |
| 波形重建 | 需要 **块式 DiT + Flow Matching + BigVGAN** | **仅需轻量因果 ConvNet**（纯左上下文） |
| 流式代价 | 需 look-ahead：LM 生成 **16 个 token** 后 DiT 才能出第一个 8-token mel chunk；BigVGAN 再引入 130ms 右上下文 | 无需 look-ahead，token 到位即可出波形 |
| 首包延迟 | 138–523 ms | **97–333 ms** |
| 长语音稳定性 | **更优**（long-zh WER 1.517 vs 12Hz 的 2.356） | 略逊 |
| 零样本克隆 WER | 1.10/1.49（0.6B/1.7B） | **0.92/1.32** |
| 官方判断 | 「单码本设计限制了它在**超低延迟**应用与通用语音合成中的适用性」 | 「实现极端码率压缩与超低延迟流式」 |

**选型结论**：12Hz 是**延迟优先**路线（开源主力），25Hz 是**长时稳定性 / 语义保真**路线（论文自述在长语音任务上更优，但权重未发布）。

##### 3. Dual-Track（双轨）流式架构——97ms 首包的关键

论文 §3.1：

> 「For real-time synthesis, Qwen3-TTS employs a **dual-track representation by concatenating textual and acoustic tokens along the channel axis**. Upon receiving a textual token, the model immediately predicts the corresponding acoustic tokens, which are then converted into waveforms by the Code2Wav module.」

要点拆解：
- **沿 channel 轴拼接文本 token 与声学 token**，形成双轨序列（而非时间轴上的顺序拼接）；
- 收到**单个文本 token** 后立即预测对应声学 token——官方 README 表述为「输入单个字符后即可输出首个音频包」；
- 声学 token 由 **Code2Wav 模块**转波形；
- **同一模型同时支持 streaming 与 non-streaming**（单模型双模）。

##### 4. 分层预测 + MTP（12Hz 系列）

- backbone 接收聚合后的 codebook 特征 → 预测**第 0 个码本**（zeroth codebook，即语义层）；
- **MTP（Multi-Token Prediction）模块**生成**全部残差码本**（后 15 层声学细节）；
- 收益：「捕获精细声学细节，显著提升音色一致性与表现力，同时通过**单帧即时生成**最小化延迟」。

##### 5. 说话人建模

「we **jointly train a learnable speaker encoder** with the backbone」——说话人编码器与骨干**联合训练**（而非外挂 x-vector 提取器），用于保持精确的身份控制。


### 5. 模块拆解 ⭐

#### 1. 整体链路（论文 Figure 3）

```
文本 ──[Qwen Tokenizer]──> 文本 token ──┐
                                        ├──> [Dual-Track 拼接（沿 channel 轴）]
参考音频 ──[Speaker Encoder]──> speaker emb ──┘
                                        │
                                        ▼
                              [Qwen3 LM Backbone (0.6B / 1.7B)]
                                        │
                      12Hz: backbone 预测 codebook-0（语义）
                            + [MTP 模块] 预测残差 codebook 1..15（声学细节）
                      25Hz: backbone + linear head 预测单码本 token
                                        │
                                        ▼
                              [Code2Wav 模块]
                       12Hz: 轻量因果 ConvNet（直接 waveform）
                       25Hz: 块式 DiT (Flow Matching) → mel → BigVGAN → waveform
                                        │
                                        ▼
                                    音频波形
```

#### 2. 组件清单

| 组件 | 12Hz 系列 | 25Hz 系列（未发布） |
|---|---|---|
| **文本前端** | 标准 Qwen tokenizer | 同左 |
| **语音 Tokenizer** | `Qwen3-TTS-Tokenizer-12Hz`：12.5Hz、16 层 RVQ、codebook 2048 | `Qwen-TTS-Tokenizer-25Hz`：25Hz、单码本、codebook 32768 |
| **Speaker Encoder** | 可学习，与 backbone **联合训练** | 同左 |
| **LM Backbone** | Qwen3-0.6B / Qwen3-1.7B | 同左 |
| **残差码本预测** | **MTP（Multi-Token Prediction）模块** | 不需要 |
| **Token → 波形** | 轻量**因果 ConvNet** | **块式 DiT（Flow Matching）→ mel → 改进版 BigVGAN** |
| **流式机制** | Dual-Track 双轨（channel 轴拼接） | 同左，但受 DiT/BigVGAN 的 look-ahead 拖累 |

#### 3. Qwen3-TTS-12Hz 的分层预测细节

```
backbone 输入：aggregated codebook features（聚合的码本特征）
   ↓
backbone 输出：zeroth codebook（第 0 层 = 语义层）
   ↓
MTP 模块：一次性生成所有 residual codebooks（第 1..15 层 = 声学细节）
   ↓
"single-frame instant generation" —— 单帧即时生成，最小化延迟
```

> 工程含义：语义层由主 LM 自回归生成（决定"说什么、什么语气"），声学细节由 MTP 并行补齐（决定"音色质感"）。这是「低延迟 + 高保真」能同时成立的结构性原因。

#### 4. Qwen3-TTS-25Hz 的滑动窗口块注意力（Streaming Detokenizer）

- 采用 **滑动窗口块注意力（sliding-window block attention）**限制每个 token 的上下文；
- **DiT + Flow Matching**（Lipman et al.）把 code 序列映射为 mel-spectrogram；
- **改进的 BigVGAN** 从 mel 重建波形；
- 流式解码：把相邻 code 分组为**固定长度块**，构造对应 attention mask；
- **DiT 感受野限制为 4 个块**：当前块 + 3 块回看 + 1 块前瞻；
- BigVGAN 感受野固定，采用同样的分块流程。
- 论文引用 **StreamFlow**（Guo et al. 2025, arXiv:2506.23986）作为块式 guided attention mask 的方法来源。

#### 5. 12Hz Tokenizer 内部（§2.2）

- **语义路径**：以 **WavLM**（Chen et al. 2022）为教师，引导第 1 层语义码本对齐语义特征；
- **声学路径**：**15 层残差向量量化（RVQ）**逐层细化语义码本未捕获的细节；
- **训练框架**：GAN —— 生成器直接对**原始波形**操作以提取并量化两种表示；判别器提升重建语音的自然度与保真度；
- 额外损失：**多尺度 mel-spectrogram 重建损失**，强化时频一致性；
- **流式设计**：**全因果（fully causal）**特征编码器与解码器——编码器顺序处理帧、无前瞻地以 12.5Hz 输出语义与声学 token；解码器从 token **增量式**重建音频。

#### 6. 可控性机制（§3.3）

- 所有控制通过**在输入序列前拼接用户提供的指令（instructions）**实现；
- 数据统一格式化为 **ChatML**，把语音控制变成语言建模任务；
- **Voice Design** 额外引入：**概率激活的 "thinking pattern"**（probabilistically activated thinking pattern），专门提升对复杂描述的指令遵循。


### 6. 音频编码器 ⭐

#### 1. 语音编码器：Qwen3-TTS-Tokenizer-12Hz（开源主力）

| 属性 | 值 | 来源 |
|---|---|---|
| 架构 | 参考 **Mimi**（Défossez et al. 2024, Moshi）的语义-声学解耦量化 | arXiv:2601.15621 §2.2 |
| 帧率（FPS） | **12.5 Hz** | 官方 README / 论文 Table 4 |
| 量化层数 NQ | **16**（1 层语义 + 15 层声学 RVQ） | 官方 README / 论文 Table 4 |
| 码本大小 | **2048** | 官方 README / 论文 Table 4 |
| 因果性 | **全因果（fully causal）**编码器与解码器，无 look-ahead | 论文 §2.2 |
| 训练框架 | **GAN**（生成器直接对 raw waveform 操作；判别器提升自然度与保真度） | 论文 §2.2 |
| 语义教师 | **WavLM**（Chen et al. 2022）引导第 1 层语义码本 | 论文 §2.2 |
| 声学路径 | **15 层 RVQ** 逐层细化细节 | 论文 §2.2 |
| 重建损失 | 多尺度 mel-spectrogram 重建损失（时频一致性） | 论文 §2.2 |
| 解码器 | **轻量因果 ConvNet**（非 DiT） | 论文 §1 |

**关键设计收益（论文原文表述）**：容量提升后，「波形重建仅需一个轻量因果 ConvNet，**无需 speaker vector 提取或复杂扩散模型**」。这是 12Hz 系列首包延迟只有 ~4ms（tokenizer decode TPP）的直接原因。

#### 2. 语音编码器：Qwen-TTS-Tokenizer-25Hz（权重未发布）

| 属性 | 值 |
|---|---|
| 架构 | 基于 **Qwen2-Audio** 的 25Hz **单码本** tokenizer |
| 帧率 | 25 Hz |
| 码本大小 | **32768** |
| 训练 | **两阶段**：Stage 1 继续预训练 Qwen2-Audio（ASR 任务），在音频编码器中间位置插入 resampling 层 + VQ 层；Stage 2 加入基于卷积的 mel-spectrogram 解码器，全模型微调，把声学信息显式注入 token 表示 |
| ASR 表现 | Stage 1 在 C.V. EN 7.51 / C.V. CN 10.73 / Fleurs EN 3.07 / Fleurs CN 4.23（优于 S3 Tokenizer 系列）；Stage 2 略降（10.40 / 14.99 / 4.14 / 4.67）——**论文明确说明这是刻意的语义保真 vs 声学丰富度权衡** |

#### 3. Speaker Encoder

- 「we **jointly train a learnable speaker encoder** with the backbone」（论文 §3.1）
- **与骨干联合训练**，而非外挂预训练的 speaker verification 模型；
- 用途：保持精确的身份控制（precise identity control）；
- 论文未披露其结构、维度、训练目标 → **待核实**。

#### 4. Tokenizer 独立可用性

`Qwen3-TTS-Tokenizer-12Hz` 已单独开源，可脱离 TTS 模型使用：

```python
import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0")
enc = tokenizer.encode("https://.../tokenizer_demo_1.wav")
wavs, sr = tokenizer.decode(enc)
sf.write("decode_output.wav", wavs[0], sr)
```

> 工程注意：微调流程的 `prepare_data.py` 需要该 tokenizer 先抽出 `audio_codes`。社区实测 **音频必须是 24kHz**，否则会在训练深处崩溃（见 known_issues）。


### 7. 语音 Tokenizer 与编解码器 ⭐

#### 1. 语音 Codec：Qwen3-TTS-Tokenizer-12Hz

| 项 | 值 |
|---|---|
| 类型 | 语义-声学解耦的多码本 codec（Mimi 路线） |
| NQ（量化层数） | **16** = 1 层语义码本 + 15 层声学 RVQ |
| Codebook Size | **2048**（每层） |
| FPS | **12.5 Hz** |
| 每 token 对应音频 | **80 ms** |
| 编码器 / 解码器 | 全因果；解码器为轻量因果 ConvNet |
| 训练 | GAN（raw waveform）+ 多尺度 mel 重建损失；WavLM 语义蒸馏 |
| 是否需 speaker vector / 扩散模型 | **否** |

##### 重建质量（LibriSpeech test-clean，2,620 条；arXiv:2601.15621 Table 4）

| 模型 | NQ | Codebook | FPS | PESQ_WB ↑ | PESQ_NB ↑ | STOI ↑ | UTMOS ↑ | SIM ↑ |
|---|---|---|---|---|---|---|---|---|
| SpeechTokenizer | 8 | 1024 | 50 | 2.60 | 3.05 | 0.92 | 3.90 | 0.85 |
| X-codec | 2 | 1024 | 50 | 2.68 | 3.27 | 0.86 | 4.11 | 0.84 |
| X-codec 2 | 1 | 65536 | 50 | 2.43 | 3.04 | 0.92 | 4.13 | 0.82 |
| XY-Tokenizer | 8 | 1024 | 12.5 | 2.41 | 3.00 | 0.91 | 3.98 | 0.83 |
| **Mimi** | 16 | 2048 | 12.5 | 2.88 | 3.42 | 0.94 | 3.87 | 0.87 |
| FireredTTS 2 Tokenizer | 16 | 2048 | 12.5 | 2.73 | 3.28 | 0.94 | 3.88 | 0.87 |
| **Qwen-TTS-Tokenizer-12Hz** | **16** | **2048** | **12.5** | **3.21** | **3.68** | **0.96** | **4.16** | **0.95** |

**结论**：在与 Mimi、FireredTTS2 完全相同的 NQ / codebook / FPS 配置下，五项指标全面 SOTA，SIM 提升最显著（0.87 → 0.95）。

##### 25Hz Tokenizer 的语义质量（ASR 代理任务，Table 3，WER ↓）

| 模型 | Codebook | FPS | C.V. EN | C.V. CN | Fleurs EN | Fleurs CN |
|---|---|---|---|---|---|---|
| S3 Tokenizer (VQ) | 4096 | 50 | 12.06 | 15.38 | - | - |
| S3 Tokenizer (VQ) | 4096 | 25 | 11.56 | 18.26 | 7.65 | 5.03 |
| S3 Tokenizer (FSQ) | 6561 | 25 | 10.67 | 7.29 | 6.58 | 4.43 |
| **Qwen-TTS-Tokenizer-25Hz (Stage 1)** | **32768** | 25 | **7.51** | 10.73 | **3.07** | **4.23** |
| Qwen-TTS-Tokenizer-25Hz (Stage 2) | 32768 | 25 | 10.40 | 14.99 | 4.14 | 4.67 |

> Stage 2 的 ASR 退化是**刻意的**：注入更多声学细节会削弱纯语义判别性，但有利于下游 TTS / 波形重建。

#### 2. 文本 Tokenizer

标准 **Qwen tokenizer**（Qwen3 系列）。所有数据以 **ChatML** 格式组织，把语音控制统一为语言建模任务。

#### 3. 流式分包定义（影响延迟的关键参数，论文 §3.4）

| 路线 | 分包规则 | 单包音频时长 |
|---|---|---|
| **12Hz** | 1 个 speech packet = **4 个 token**（12.5Hz → 每个 token 80ms） | **320 ms** |
| **25Hz** | LM 需先生成 **16 个 token**，DiT 才能产出第一个 8-token mel chunk（25Hz → 每 token 40ms → 320ms mel）；BigVGAN 额外 130ms 右上下文 look-ahead | 首包约 **190 ms** 音频 |

> 12Hz 的选择理由（论文原文）：「为避免过小的数据包导致过多调度开销，我们把一个 speech packet 定义为 4 个 token（320ms 语音）。该设计显著降低解码时间、获得更低首包延迟，同时因为 codec decoder 轻量且对批处理友好，在高并发下也能保持低 RTF。」


### 8. LLM 主干 ⭐

#### 骨干：Qwen3-0.6B / Qwen3-1.7B

| 属性 | 值 | 来源 |
|---|---|---|
| 模型 | **Qwen3 LM 家族**（Qwen3-0.6B / Qwen3-1.7B） | 论文 §3.1 |
| 选择理由 | 「leverages the Qwen3 LM family to achieve **high concurrency and low-latency inference**」 | 论文 §3.1 |
| 输入 | 文本 token（Qwen tokenizer）+ 声学 token，**Dual-Track 沿 channel 轴拼接** | 论文 §3.1 |
| 输出（12Hz） | **zeroth codebook**（第 0 层语义码本）；残差码本由 MTP 模块生成 | 论文 §3.1 |
| 输出（25Hz） | 单码本 speech token，经 linear head 预测 | 论文 §3.1 |
| Speaker Encoder | 可学习，**与 backbone 联合训练** | 论文 §3.1 |
| 上下文长度 | 预训练 S3 阶段从 **8,192 提升到 32,768** | 论文 §3.2 |

#### 关键设计解读

##### 1. 为什么用通用 Qwen3 LM 而不是专用 TTS 架构

- **高并发 + 低延迟**是首要目标，Qwen3 系列经过充分工程优化（可复用 vLLM、FA2、torch.compile、CUDA Graph 等生态）；
- **继承强文本理解能力**——论文 §3.3 明确说 VoiceDesign 是「built upon the Qwen3 text model foundation, Qwen3-TTS inherits robust text comprehension capabilities」。这是「用自然语言设计音色」能成立的前提；
- 对复杂文本（数字、符号、混排）的鲁棒性也来自文本基座。

##### 2. 12Hz vs 25Hz 的预测头差异

| | 12Hz | 25Hz |
|---|---|---|
| 预测目标 | 第 0 层码本（语义） | 单码本 token |
| 残差层处理 | **MTP 模块**一次性生成全部 15 层残差 | 无 |
| 论文评价 | 「捕获精细声学细节，显著提升音色一致性与表现力，同时通过单帧即时生成最小化延迟」 | 「backbone 融合文本特征与前序语音 token，通过 linear head 预测当前语音 token」 |

##### 3. 上下文长度演进（S3 阶段）

论文 §3.2 长上下文阶段：「我们把最大 token 长度从 **8,192 提升到 32,768**，并对训练数据中的长语音做上采样。实验表明这些调整增强了模型处理扩展与复杂输入、生成上下文适宜语音响应的能力。」

> 工程含义：**32,768 token 的上下文是「10 分钟以上长语音稳定生成」的容量基础**。按 12.5Hz 计算，32,768 token 可覆盖约 43 分钟的音频帧（扣除文本与 prompt 后实际更短）。

##### 4. 未披露项（待核实）

- 骨干的确切层数 / hidden size / 注意力头数（应等同 Qwen3-0.6B / Qwen3-1.7B 公开配置，但论文未复述）
- MTP 模块的结构与层数
- Speaker Encoder 的结构与维度
- 12Hz 系列是否为「分-talker / 子-talker」双模块结构 —— 社区微调讨论中出现了 `model.talker` 与「sub-talker loss」的提法，说明内部确有 talker 子模块，**但官方论文未描述 → 待核实**


### 9. 声学解码器 ⭐

#### 1. 12Hz 系列：轻量因果 ConvNet（Code2Wav）

- 架构：**轻量因果卷积网络（lightweight causal ConvNet）**
- 输入：16 层 RVQ 的离散 codes
- 输出：**直接输出波形**（论文表述为「waveform reconstruction using only a lightweight causal ConvNet」）
- **无需**：speaker vector 提取、扩散模型 / DiT
- 因果性：**纯左上下文（pure left-context）**——论文 §3.4 原文「Qwen-TTS-Tokenizer-12Hz uses a pure left-context streaming codec decoder, enabling waveform emission immediately after the required tokens are available, **without waiting for future context**」
- **代价极小**：论文 Table 2 中 12Hz 的 Tokenizer Decode TPP 仅 **4–5 ms**（对比 25Hz 的 25–147 ms）

> 这是 12Hz 系列首包延迟能压到 97ms 的最直接原因：codec 解码几乎不占时间，瓶颈全在 LM 侧。

#### 2. 25Hz 系列：块式 DiT + Flow Matching + BigVGAN（三段级联，权重未发布）

```
codes ──> [DiT + Flow Matching] ──> mel-spectrogram ──> [改进版 BigVGAN] ──> waveform
```

| 组件 | 说明 |
|---|---|
| **DiT（Diffusion Transformer）** | 用 **Flow Matching**（Lipman et al., ICLR 2023）训练；把 code 序列映射为 mel-spectrogram |
| **滑动窗口块注意力** | 每个 token 被限制在有限上下文内；相邻 code 分组为**固定长度块**，构造对应 attention mask（方法来源：**StreamFlow**, Guo et al. 2025, arXiv:2506.23986） |
| **DiT 感受野** | 限制为 **4 个块** = 当前块 + 3 块回看 + 1 块前瞻 |
| **BigVGAN** | 改进版（Lee et al., ICLR 2023），感受野固定；同样采用分块流程以支持流式波形合成 |
| **额外延迟** | BigVGAN 引入 **130 ms 右上下文 look-ahead** |

#### 3. 两条路线的解码延迟对比（arXiv:2601.15621 Table 2）

| 模型 | 并发 | LM TTFP | Tokenizer Decode TPP | 首包延迟 | LM TPP | RTF |
|---|---|---|---|---|---|---|
| Qwen3-TTS-25Hz-1.7B | 1 | 125 ms | **25 ms** | 150 ms | 56 ms | 0.253 |
| Qwen3-TTS-25Hz-1.7B | 3 | 222 ms | **62 ms** | 284 ms | 64 ms | 0.394 |
| Qwen3-TTS-25Hz-1.7B | 6 | 376 ms | **147 ms** | 523 ms | 85 ms | 0.725 |
| Qwen3-TTS-25Hz-0.6B | 1 | 113 ms | 25 ms | 138 ms | 50 ms | 0.234 |
| Qwen3-TTS-25Hz-0.6B | 3 | 198 ms | 62 ms | 260 ms | 59 ms | 0.378 |
| Qwen3-TTS-25Hz-0.6B | 6 | 334 ms | 147 ms | 481 ms | 80 ms | 0.709 |
| **Qwen3-TTS-12Hz-1.7B** | 1 | 97 ms | **4 ms** | **101 ms** | 21 ms | 0.313 |
| Qwen3-TTS-12Hz-1.7B | 3 | 190 ms | 5 ms | 195 ms | 24 ms | 0.363 |
| Qwen3-TTS-12Hz-1.7B | 6 | 328 ms | 5 ms | 333 ms | 32 ms | **0.463** |
| **Qwen3-TTS-12Hz-0.6B** | 1 | 93 ms | **4 ms** | **97 ms** | 19 ms | 0.288 |
| Qwen3-TTS-12Hz-0.6B | 3 | 174 ms | 5 ms | 179 ms | 22 ms | 0.338 |
| Qwen3-TTS-12Hz-0.6B | 6 | 294 ms | 5 ms | 299 ms | 30 ms | **0.434** |

**测试条件**（论文 §3.4）：内部 vLLM 引擎（**vLLM V0 backend**）、**单个 typical computational resource**（卡型未指明 → 待核实）、对 tokenizer 解码阶段应用 **torch.compile + CUDA Graph** 加速；所有数字为**端到端实测**；稳态成本按流式生成中每个 speech packet 测量。首包延迟 = LM TTFP + Tokenizer Decode TPP。

**关键读数**：12Hz 的 codec 解码耗时在高并发下**几乎不变**（4 → 5 ms），而 25Hz 从 25ms 涨到 147ms。这意味着 **12Hz 在高并发场景下的延迟优势会被放大**。


### 10. 声码器

#### 12Hz 系列（已开源）：无独立 vocoder

12Hz 路线的 **Code2Wav 模块是一个轻量因果 ConvNet，直接从离散 codes 生成波形**，论文明确说明它「eliminating the need for speaker vector extraction or complex diffusion models」。

> 即：**12Hz 系列没有 mel 中间表示，也没有独立 vocoder 阶段**。Tokenizer 的 ConvNet 解码器本身就是波形生成器。这也解释了为何官方 README 称其为「轻量 non-DiT 架构实现高保真重建」。

#### 25Hz 系列（权重未发布）：BigVGAN

| 组件 | 说明 |
|---|---|
| **声码器** | **改进版 BigVGAN**（Lee et al., ICLR 2023） |
| 输入 | DiT + Flow Matching 生成的 mel-spectrogram |
| 输出 | 波形 |
| 流式处理 | 感受野固定，采用与 DiT 相同的分块流程 |
| **额外延迟** | **130 ms 右上下文 look-ahead** |

25Hz 的级联：`codes → [DiT + Flow Matching] → mel → [BigVGAN] → waveform`

#### 两条路线 vocoder 对比

| | 12Hz（已开源） | 25Hz（未发布） |
|---|---|---|
| 是否有 mel 中间表示 | **否** | 是 |
| 波形生成器 | 轻量因果 ConvNet（tokenizer 内置） | 改进版 BigVGAN |
| 是否需扩散/Flow Matching | **否** | 是（DiT + Flow Matching） |
| 解码耗时（TPP） | **4–5 ms** | 25–147 ms |
| 引入的 look-ahead | **0** | 130 ms（BigVGAN）+ DiT 的 1 块前瞻 |

#### 未披露项（待核实）

- 12Hz ConvNet 解码器的层数、通道配置、上采样倍率
- 输出采样率（社区实测微调数据需 **24kHz**，与官方 API 的 `PCM_24000HZ_MONO_16BIT` 一致 → 推断为 **24 kHz**，但论文未明确写明 → **待核实**）
- BigVGAN 的具体改进点


### 11. 上下文长度与 I/O 约束

#### 上下文长度

| 项 | 值 | 来源 |
|---|---|---|
| 预训练 S1/S2 阶段最大 token 长度 | **8,192** | 论文 §3.2 |
| 预训练 S3 阶段（长上下文阶段）最大 token 长度 | **32,768** | 论文 §3.2 |
| 长语音生成能力 | 「capable of synthesizing **over 10 minutes** of natural and fluent speech」 | 论文 §1 / §5 |
| 长语音评测集 | 内部数据集，中英各 **100 篇文本，长度 200–2000 词** | 论文 §4.2.6 |

#### 输入

| 项 | 支持情况 |
|---|---|
| 文本 | 通过 `generate_custom_voice` / `generate_voice_design` / `generate_voice_clone` 传入；**支持 batch**（list 形式，可混合语言与不同 instruct） |
| `text` 参数 | 字符串或字符串列表 |
| `language` 参数 | 显式指定（如 `"Chinese"` / `"English"`）；Seed-TTS / InstructTTS 评测中用 `language="auto"` |
| `instruct` 参数 | 自然语言风格/情感指令（CustomVoice 与 VoiceDesign 支持；**Base 模型不支持**） |
| `speaker` 参数 | 9 个预设音色名，或微调后注册的 `--speaker_name`（如 `speaker_test`） |
| 参考音频 `ref_audio` | 本地路径 / URL / base64 / `(numpy_array, sample_rate)` |
| 参考文本 `ref_text` | 参考音频的转写（ICL 克隆时需要；`x_vector_only_mode=True` 时不需要） |
| 生成参数 | 所有 `generate_*` 可透传 HF Transformers `generate` 参数：`max_new_tokens`、`top_p`、`top_k`、`temperature` 等 |

#### 输出

- `wavs, sr = model.generate_*(...)`；用 `soundfile.write(path, wavs[0], sr)` 落盘
- **输出采样率推断为 24 kHz**（DashScope API 使用 `AudioFormat.PCM_24000HZ_MONO_16BIT`；社区微调要求训练音频为 24kHz）—— **论文未明确写明，待核实**

#### Prompt 复用（重要性能优化）

```python
prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=False)
wavs, sr = model.generate_voice_clone(
    text=["Sentence A.", "Sentence B."],
    language=["English", "English"],
    voice_clone_prompt=prompt_items)   # 复用，避免重复编码
```

> 工程价值：批量生成同一音色时，复用 `voice_clone_prompt` 可省掉重复的参考音频编码，并**提升跨批次音色一致性**。

#### `x_vector_only_mode` 权衡

| 模式 | 是否需要 `ref_text` | 质量 |
|---|---|---|
| `x_vector_only_mode=False`（默认，ICL） | **需要** | 高（更好保留韵律） |
| `x_vector_only_mode=True` | 不需要（只用 speaker embedding） | **「cloning quality may be reduced」**（官方 README 明示质量下降） |

> 论文 §3.3 解释了两种克隆路径：(i) 通过 speaker embedding 从参考语音克隆 → 支持**实时克隆**；(ii) 通过 text–speech 对的 **in-context learning** → **更好保留韵律**。

#### 未披露项（待核实）

- 单次请求的最大文本长度（字符数/token 上限）
- DashScope API 的输入上限为 **512 tokens**（第三方来源，待核实）
- 批处理的最大 batch size 建议值


### 12. 训练数据 ⭐

#### 规模与语种

| 项 | 值 | 来源 |
|---|---|---|
| **总训练数据量** | **超过 500 万小时（over 5 million hours）**语音数据 | 论文 Abstract / §1 / §3.2 |
| **语种数** | **10 种**（另有表述「trained across more than 10 languages」） | 论文 §1 |
| **具体语种** | Chinese（含北京话、四川话方言音色）、English、Japanese、Korean、German、French、Russian、Portuguese、Spanish、Italian | 官方 README |
| **数据格式** | 全部以 **ChatML** 格式化，统一输入并支持可控语音生成 | 论文 §3.2 |

#### 数据分层策略

论文 §3.2 描述了预训练三阶段各自的数据使用方式：

| 阶段 | 数据策略 |
|---|---|
| **S1 General Stage** | 使用**全部** 500 万+ 小时多语种语音数据，建立「多语种文本表示 → 语音」的单调映射，构建通用能力 |
| **S2 High-Quality Stage** | 用**专门的质量分层流水线（dedicated pipeline）**对数据进行质量分层，用**高质量数据做持续预训练（CPT）** |
| **S3 Long-Context Stage** | 提升 max token length（8,192 → 32,768），并对训练数据中的**长语音做上采样（upsample long speech）** |

#### S2 阶段的动机（重要工程洞察）

论文原文：「This stage **alleviates hallucinations caused by noisy data** in the initial stage and significantly improves the quality of generated speech.」

> 即：**先用海量但含噪的数据建能力，再用高质量子集做 CPT 去幻觉**。这与 Qwen3-ASR 的「大规模预训练 + 小规模高质量 SFT」是同一套方法论。

#### 后训练数据

| 阶段 | 数据 |
|---|---|
| **DPO** | 基于人类反馈构造的**多语种语音样本的偏好对（preference pairs）** |
| **GSPO** | **rule-based rewards（基于规则的奖励）** |
| **Speaker Fine-tuning** | 针对 base model 的**轻量说话人微调** |

#### 未披露项（待核实）

- 500 万小时中各语种的分布
- 质量分层流水线的具体判据（DNSMOS？SNR？人工标注？）
- TTS 训练数据是否为 TTS 专用录制 / 是否含 ASR 数据
- 克隆训练所用的说话人数量
- 是否使用了合成数据 / 数据增强
- 「thinking pattern」数据的构造与激活概率
- 训练集群规模、卡型、训练时长


### 13. 训练阶段 ⭐

论文 §3.2 把训练明确分为 **Pre-training（3 阶段）+ Post-training（3 阶段）**，共 6 个子阶段。所有数据统一为 ChatML 格式。

---

#### 一、预训练（Pre-training）

##### S1 — General Stage（通用阶段）

- **数据**：500 万+ 小时多语种语音数据
- **目标**：
  1. 建立「多语种文本表示 → 语音」的**单调映射（monotonic mapping）**
  2. 构建 Qwen3-TTS 的通用能力
- **上下文长度**：8,192

##### S2 — High-Quality Stage（高质量阶段）

- **方法**：用专门流水线对数据做**质量分层（stratify data quality）**，用高质量数据做**持续预训练（CPT）**
- **目标**：「**缓解**第一阶段因含噪数据导致的**幻觉（hallucinations）**，并显著提升生成语音质量」
- **上下文长度**：8,192

##### S3 — Long-Context Stage（长上下文阶段）

- **方法**：
  1. 最大 token 长度 **8,192 → 32,768**
  2. 对训练数据中的**长语音上采样（upsample long speech）**
- **目标**：「增强模型处理**扩展与复杂输入**、生成上下文适宜语音响应的能力」
- **实证**：支撑「10 分钟以上流畅语音生成」与长语音 WER 的 SOTA

---

#### 二、后训练（Post-training）

论文原文：「The post-training phase comprises three stages, enabling Qwen3-TTS to generate human-like speech and remain stable across tasks.」

##### P1 — DPO（Direct Preference Optimization）

- **算法**：DPO（Rafailov et al., NeurIPS 2023）
- **数据**：基于**人类反馈**构造的**多语种语音样本偏好对**
- **目标**：把模型输出与人类偏好对齐（align model outputs with human preferences）

##### P2 — GSPO（强化学习）

- **算法**：**GSPO**（Group Sequence Policy Optimization，Zheng et al. 2025, arXiv:2507.18071）
- **奖励**：**rule-based rewards（基于规则的奖励）**
- **目标**：「全面增强模型跨任务的能力与**稳定性（stability）**」
- **注意**：与 Qwen3-ASR 的 GSPO 阶段相比，**TTS 侧明确了是 rule-based rewards**；ASR 侧论文未说明奖励设计（待核实）

##### P3 — Lightweight Speaker Fine-tuning（轻量说话人微调）

- **对象**：**base model**（0.6B / 1.7B Base 变体）
- **方法**：轻量（lightweight）说话人微调
- **目标**：「使 Qwen3-TTS 采用特定音色（adopt specific voices），同时进一步提升语音响应的**自然度、表现力与可控性**」
- **对应能力**：这就是 CustomVoice 的 9 个预设音色与论文 Table 9 中「Aiden Voice」说话人微调实验的来源
- **工程含义**：官方的「CustomVoice 预设音色」与「社区 LoRA 微调定制音色」走的是**同一条技术路径**，只是规模与数据不同

---

#### 三、Tokenizer 的两阶段训练（Qwen-TTS-Tokenizer-25Hz，§2.1）

| 阶段 | 内容 |
|---|---|
| **Stage 1** | 在 **ASR 任务**上继续预训练 **Qwen2-Audio**，在音频编码器中间位置插入一个 **resampling 层**与一个 **VQ 层** |
| **Stage 2** | 加入**基于卷积的 mel-spectrogram 解码器**（从 audio token 重建 mel），**全模型微调**，把必要的声学信息显式注入 token 表示 |

（12Hz tokenizer 的训练在 §2.2 描述为 GAN 框架 + WavLM 语义蒸馏 + 多尺度 mel 重建损失，未再分阶段。）

---

#### 四、可控性训练技巧（§3.3）

- **指令前置**：把含细粒度控制信号的**用户指令拼接到输入序列前部**
- **ChatML 格式化**：把语音控制变成语言建模任务
- **概率激活的 thinking pattern**：「we introduce a **probabilistically activated thinking pattern** during training to improve instruction following, especially for **complex descriptions**」
  - 这是在 VoiceDesign 场景下取得 InstructTTSEval SOTA 的关键（论文 §5 结论中再次点名）
  - **具体激活概率、thinking 模板未披露 → 待核实**

---

#### 五、训练阶段与能力的因果对应（便于二次开发定位）

| 想要的能力 | 主要来自 | 二次开发启示 |
|---|---|---|
| 基础多语种合成 | S1 | 新语种需 S1 级别数据量，微调难以补齐 |
| 抑制幻觉 / 提升音质 | S2 CPT | 微调数据质量 >> 数量 |
| 10 分钟长语音稳定 | S3 长上下文 | 微调时需同步拉长上下文，否则长文本会失稳 |
| 拟人度 / 偏好对齐 | DPO | 需偏好对数据，成本高 |
| 跨任务稳定性 | GSPO | 需 rule-based reward 设计 |
| 特定音色 + 自然度 | P3 轻量说话人微调 | **官方已给出 `finetuning/` 脚本，这是唯一官方支持的微调路径** |


### 14. 后训练与强化学习 ⭐

#### 后训练三段式总览（论文 §3.2）

```
Pre-trained (S1→S2→S3)
      ↓
P1: DPO        —— 人类偏好对齐（preference pairs, 多语种）
      ↓
P2: GSPO       —— 规则奖励强化学习（rule-based rewards）→ 全面能力与稳定性
      ↓
P3: 轻量说话人微调 —— 特定音色 + 自然度/表现力/可控性
```

#### P1 — DPO（Direct Preference Optimization）

| 项 | 内容 |
|---|---|
| 算法 | **DPO**（Rafailov et al., NeurIPS 2023） |
| 数据 | 基于**人类反馈**为**多语种语音样本**构造的**偏好对（preference pairs）** |
| 目标 | 把模型输出与人类偏好对齐 |
| 论文表述 | 「we construct preference pairs for multilingual speech samples based on human feedback and then perform DPO on Qwen3-TTS」 |
| 超参 / 数据量 | **未披露 → 待核实** |
| 偏好判据（什么算"更好"） | **未披露 → 待核实** |

#### P2 — GSPO（强化学习）

| 项 | 内容 |
|---|---|
| 算法 | **GSPO**（Group Sequence Policy Optimization），Zheng et al. 2025，arXiv:2507.18071 |
| 奖励 | **rule-based rewards（基于规则的奖励）** —— 论文明确写出 |
| 数据规模 | **未披露 → 待核实** |
| 目标 | 「comprehensively enhance the model's **capabilities and stability across tasks**」 |
| 规则的具体构成 | **未披露 → 待核实**（合理推测为 WER/内容一致性、说话人相似度、时长/语速合理性等可自动计算的指标，但论文未确认，**不应作为事实陈述**） |

> **与 Qwen3-ASR 的对比**：两者都用 GSPO，但**只有 TTS 侧明确写了 rule-based rewards**。ASR 侧论文仅说「提升噪声鲁棒性、转写稳定性、难例分析能力」，未说明奖励设计。

#### P3 — 轻量说话人微调（Lightweight Speaker Fine-tuning）

| 项 | 内容 |
|---|---|
| 对象 | **base model**（Qwen3-TTS-12Hz-1.7B/0.6B-Base） |
| 方法 | 轻量（lightweight）说话人微调 |
| 目标 | 「enabling Qwen3-TTS to **adopt specific voices** while further improving the **naturalness, expressiveness, and controllability**」 |
| 官方开放程度 | ✅ **这是官方唯一开放的微调路径**，见 `finetuning/README.md` |
| 当前限制 | **仅支持单说话人微调**（single-speaker fine-tuning）；多说话人微调与高级微调特性「将在未来版本支持」 |

##### 官方说话人微调的实证（Table 9）

论文把 Qwen3-TTS 在 **Aiden Voice** 上做目标说话人微调，与 **GPT-4o-Audio-Preview（Ballad Voice）** 对比（WER ↓）：

| 语言 | 25Hz-0.6B-CV | 25Hz-1.7B-CV | 12Hz-0.6B-CV | 12Hz-1.7B-CV | GPT-4o-Audio Preview |
|---|---|---|---|---|---|
| Chinese | 0.874 | **0.708** | 0.944 | 0.903 | 3.519 |
| English | 1.332 | 0.936 | 1.188 | **0.899** | 2.197 |
| German | 0.990 | **0.634** | 2.722 | 1.057 | 1.161 |
| Italian | 1.861 | 1.271 | 2.545 | 1.362 | **1.194** |
| Portuguese | 1.728 | 1.854 | 3.219 | 2.681 | **1.504** |
| Spanish | 1.309 | **1.284** | 1.154 | 1.330 | 4.000 |
| Japanese | 3.875 | 4.518 | 6.877 | **4.924** | 5.001 |
| Korean | 2.202 | 2.274 | 3.053 | **1.741** | 2.763 |
| French | 3.865 | **3.080** | 3.841 | 3.781 | **3.605** |
| Russian | 6.529 | **4.444** | 5.809 | 4.734 | 5.250 |

**关键结论**：
- Qwen3-TTS 在 **10 语言中 7 种优于 GPT-4o-Audio-Preview**（中、英、德、西、日、韩、俄）
- **尽管只在单语数据上微调**，音色与韵律仍能迁移到全部 10 种语言而**不降低稳定性**（跨语言泛化极强）
- 日语 4.924 vs 5.001、韩语 1.741 vs 2.763 的提升尤为显著

#### 未披露项汇总（待核实）

- DPO 的偏好对数量、β 参数、训练轮数
- GSPO 的 group size、KL 系数、奖励规则的具体项与权重
- 后训练各阶段的数据规模
- 后训练硬件与耗时
- 说话人微调所需的最少数据量（官方微调示例用 `--num_epochs 10`，未给数据量下限）


### 15. 数据配方 ⭐

#### 官方数据配方（从论文可还原的部分）

##### 1. 预训练数据分层流水线（S2 阶段）

论文 §3.2 仅说明「We **stratify data quality with a dedicated pipeline**」，**未披露分层判据**（如 DNSMOS / SNR / ASR 置信度 / 人工评分）→ **待核实**。

分层后的用途：
- S1：全量 500 万+ 小时建能力
- S2：高质量子集做 CPT 去幻觉
- S3：对**长语音上采样**，配合 32,768 上下文

##### 2. 数据格式：ChatML

「All data is formatted in **ChatML** to standardize inputs and support controllable speech generation.」

含义：所有任务（克隆、设计、控制）统一为 ChatML 对话格式，使「语音控制」变成纯语言建模任务，从而复用 LLM 的标准训练/推理栈。

##### 3. 可控性数据的构造

- **指令前置**：把含细粒度控制信号的用户指令**拼接到输入序列前部**
- **thinking pattern**：以**概率激活**的方式在训练中引入，专门提升对**复杂描述**的指令遵循

##### 4. 后训练数据

| 阶段 | 数据 |
|---|---|
| DPO | 基于人类反馈的**多语种语音样本偏好对** |
| GSPO | rule-based rewards（数据量未披露） |
| Speaker FT | 目标说话人的音频-转写对 |

##### 5. 微调数据格式（官方 `finetuning/`）

```jsonl
{"audio":"./data/utt0001.wav","text":"<转写>","ref_audio":"./data/ref.wav"}
```

关键要求：
- **`ref_audio` 全数据集保持一致**（官方强烈建议，提升音色一致性与生成稳定性）
- 音频需先经 `prepare_data.py` 用 `Qwen3-TTS-Tokenizer-12Hz` 抽出 `audio_codes`
- **社区实测：音频必须是 24 kHz**（否则训练深处崩溃）

#### 社区补充的数据实践（**第三方，非官方**）

来源：instavar.com LoRA 微调文章

1. **重采样到 24kHz mono**（脚本 `scripts/resample_to_24k.sh`）—— 硬性前置
2. **每条训练音频末尾追加 1 秒静音**
3. **剥离转写文本中的标注标签**（如情绪标签、音效标注）
4. **使用同一条参考音频**贯穿整个数据集
5. 训练数据规模：该社区实践用 **5k 样本 / 8 GPU / bs 16 / ga 16**

#### 未披露项（待核实）

- 500 万小时中各语种 / 各性别 / 各风格的分布
- 质量分层的具体阈值与模型
- 数据清洗规则（时长过滤、响度归一化、去重、静音切除）
- 转写文本的归一化规则（数字、缩写、标点）
- 是否使用数据增强（速度扰动、噪声叠加、混响）
- 克隆训练所用的说话人数量与每人平均时长
- DPO 偏好对的构造流程


### 16. 微调支持

#### 官方支持范围

| 项 | 状态 |
|---|---|
| 可微调模型 | **Qwen3-TTS-12Hz-1.7B-Base / Qwen3-TTS-12Hz-0.6B-Base** |
| 微调类型 | **单说话人微调（single-speaker fine-tuning）** |
| 多说话人微调 | ❌ **暂不支持**（官方：「will be supported in future releases」） |
| LoRA | ❌ **官方脚本不支持**（`finetuning/sft_12hz.py` 为全参数 SFT）；社区需自行接入 peft |
| CustomVoice / VoiceDesign 微调 | ❌ 官方脚本仅面向 Base 模型 |
| 官方示例学习率 | **文档步骤示例 `2e-6`；一键脚本示例 `2e-5`（README 内部不一致，见 known_issues）** |

#### 完整流程（官方 `finetuning/README.md`）

##### Step 0：环境

```bash
pip install -U qwen-tts
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

##### Step 1：准备 JSONL

每行一个 JSON，三个必填字段：`audio`（目标训练音频 wav）、`text`（对应转写）、`ref_audio`（参考说话人音频 wav）

```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

> **官方强烈建议：全数据集使用同一个 `ref_audio`** —— 「Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.」

##### Step 2：抽取 audio_codes

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

##### Step 3：SFT

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test
```

产出：`output/checkpoint-epoch-0`、`checkpoint-epoch-1`、...

##### Step 4：推理验证

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2", device_map="cuda:0",
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.", speaker="speaker_test")
sf.write("output.wav", wavs[0], sr)
```

> 注意：微调后用 `generate_custom_voice` + 注册进来的 **speaker 名**来调用，而不是 `generate_voice_clone`。

#### 官方超参

| 参数 | 文档步骤示例 | 一键脚本示例 |
|---|---|---|
| `--batch_size` | 32 | 2 |
| `--lr` | **2e-6** | **2e-5**（⚠️ 见 known_issues） |
| `--num_epochs` | 10 | 3 |
| `--speaker_name` | speaker_test | speaker_1 |

#### 上游修复时间线

| 日期 | 事件 |
|---|---|
| 2026-01-22 | 初始开源（`prepare_data.py` 于 01-23 有「fix and update」提交） |
| 2026-02-04 | `README.md` 提交「fix: adjust finetune script for stable training」 |
| 2026-02 起 | 社区 issue #179 / PR #178 报告并修复 double label-shift bug |
| **2026-03-17** | `sft_12hz.py` 提交「**fix finetuning bug**」——官方已修复语速异常问题 |

> **结论**：2026-03-17 之后的官方脚本已修复历史 bug。但社区实践仍建议 `lr=2e-6`（而非默认的 2e-5），并做 checkpoint 听觉筛选。

#### 社区验证的实践要点（**第三方来源，非官方，需自行验证**）

来源：instavar.com 的 LoRA 微调深度文章 + 配套仓库 `cheeweijie/qwen3-tts-lora-finetuning`

| 项 | 社区建议值 | 理由 |
|---|---|---|
| 学习率 | **2e-6**（而非 2e-5） | 2e-5 会导致纯噪声输出、EOS 丢失、无限生成 |
| 音频采样率 | **必须 24 kHz**，否则训练深处崩溃且无早期告警 | codec 流水线内部断言 |
| 训练 epoch | 单说话人 **约 10 epoch** 后开始过拟合 | 验证损失在 8–12 轮趋平 |
| LoRA scale（若用 LoRA） | **0.25–0.35**（不是 1.0） | scale 1.0 会「过度操控」，声音发假 |
| temperature | 0.8（默认 0.9） | 更稳定 |
| top_p | 0.85（默认 1.0） | 减少音色漂移 |
| top_k | 30（默认 50） | 收紧采样 |
| repetition_penalty | 1.05 | 多 chunk 一致性关键 |
| `max_new_tokens` | 显式设置（如 2048） | ~0.5% 的 base 推理会遇到 EOS 失败（无限生成） |
| `eos_token_id` | `[2150, 2157, 151670, 151673, 151645, 151643]` | 兜底 EOS |
| 数据准备 | 每条训练音频**末尾加 1 秒静音**；剥离转写中的标注标签 | 稳定性 |
| 长文本分 chunk | 每个 chunk 前**固定随机种子**；**复用同一次提取的 speaker embedding** | 防止跨 chunk 音色漂移 |

#### 未披露 / 缺失项（待核实）

- **最少需要多少训练数据**：官方未给出下限
- **微调是否保留原有的 9 个预设音色**：官方未说明
- **微调后是否仍支持流式**：官方未说明
- **多卡训练**：官方脚本未提供 `torchrun` 示例（对比 Qwen3-ASR 有）
- **微调后的显存需求**：未给出（社区实测 24GB RTX 3090 Ti 可跑 1.7B LoRA）
- **微调是否影响 VoiceDesign / 指令控制能力**：未说明


### 17. 训练硬件门槛

#### 官方未公开（待核实）

- **预训练集群规模**（GPU 卡型 / 数量 / 训练时长）：500 万小时 + 3 阶段预训练 + 3 阶段后训练，论文**完全未披露**硬件信息
- **后训练（DPO / GSPO / Speaker FT）硬件**：未披露
- **官方微调脚本的多卡支持**：`finetuning/` 下**未提供 torchrun 示例**（对比 Qwen3-ASR 的微调脚本明确支持多卡 DDP）→ 推测为单卡设计，**待核实**

#### 官方明示的运行环境

| 项 | 要求 |
|---|---|
| Python | 建议 3.12（官方示例用 conda） |
| 精度 | `torch.bfloat16`（所有官方示例与评测） |
| FlashAttention 2 | **强依赖**，见下 |
| 注意力后端 | `attn_implementation="flash_attention_2"`（官方所有示例）；社区建议 `sdpa` 作为消费级 GPU 的稳定回退 |
| 设备 | `device_map="cuda:0"` |

#### ⚠️ FlashAttention 2 是强依赖（不是可选优化）

官方 README：「FlashAttention 2 ... 用于减少 GPU 显存使用、加速」。但**第三方技术分析给出了远更强的结论**：

> 「**Critical dependency: FlashAttention 2** — 97ms 延迟是理论最优值，**实践中需要 FlashAttention 2**。技术分析表明，无 FA2 支持时推理速度会急剧下降。**即便在 RTX 5090 上，缺少该优化时模型也会变得迟滞（约 0.3x 实时因子）**，因为解码器中的并行处理否则会成为瓶颈。」

**来源性质**：该 0.3x 数字来自第三方技术分析站点（ai-rockstars.com），**非 Qwen 官方文档或论文**，标注为**待核实**，但方向性结论（FA2 对 Qwen3-TTS 是强依赖）与官方 README 的强调强度一致。

同理，该来源给出的 VRAM 估算：
- **1.7B：6–8 GB VRAM**（RTX 3060/4060 级别）
- **0.6B：4–6 GB VRAM**（可支持边缘部署）

> 用户提供的「0.6B 约 2GB、1.7B 约 6GB」与该来源的下限吻合。

#### 社区实测显存（**第三方，非官方**）

| 来源 | 配置 | 峰值显存 |
|---|---|---|
| PyPI `qwen3-tts-triton` 项目自测 | RTX 5090，bf16，batch=1，1.7B 推理 | **4.03–4.33 GB** |
| CSDN 实测（Qwen3-TTS-12Hz-1.7B-Base） | RTX 3090 / 4090 / 5090 | **7.8 / 7.6 / 7.5 GB** |
| instavar LoRA 微调实践 | 24GB RTX 3090 Ti，1.7B LoRA 微调 | 可跑（未给具体峰值） |

> 三个来源差异较大（4.0 vs 7.5 GB），可能与是否启用 FA2、batch size、序列长度有关。**建议以实测为准。**

#### 工程建议

1. **必须安装 flash-attn**，并检查其在你的 GPU 架构上真正生效（而非静默回退到 sdpa/eager）。
2. **消费级 GPU 上如遇 FA2 编译/兼容问题**，社区建议改用 `attn_implementation="sdpa"`（速度较慢但稳定）。
3. **Blackwell（RTX 50 系）需配套 CUDA 12.x**；社区有专门针对 RTX 5090 的 fork 验证可用。
4. **微调显存**：官方未给 batch_size 对应的卡型，`--batch_size 32` 在单卡 24GB 上很可能 OOM，需按卡型下调。


### 18. 语种与方言覆盖

#### 支持语种：10 种

| 语种 | 备注 |
|---|---|
| **Chinese** | 含方言音色：Dylan（北京话）、Eric（成都话/四川话） |
| **English** | 含多国/地区口音（Ryan / Aiden 为英语母语音色） |
| **Japanese** | Ono_Anna |
| **Korean** | Sohee |
| **German** | |
| **French** | |
| **Russian** | |
| **Portuguese** | |
| **Spanish** | |
| **Italian** | |

论文 §1 表述为「The model is trained across **more than 10 languages** and supports **speaker-consistent multilingual generation**」，官方 README 明确列出上述 10 种。

#### 关键能力：说话人一致的多语种生成（Speaker-Consistent Multilingual）

同一音色可跨 10 种语言合成而保持音色一致。官方建议：**每个预设 speaker 尽量使用其母语**以获得最佳质量（跨语言质量可能略降）。

#### 多语种性能（arXiv:2601.15621 Table 6，WER ↓ / SIM ↑）

| 语言 | 25Hz-0.6B | 25Hz-1.7B | 12Hz-0.6B | 12Hz-1.7B | MiniMax | ElevenLabs |
|---|---|---|---|---|---|---|
| **内容一致性（WER ↓）** | | | | | | |
| Chinese | 1.108 | **0.777** | 1.145 | 0.928 | 2.252 | 16.026 |
| English | 1.048 | 1.014 | **0.836** | 0.934 | 2.164 | 2.339 |
| German | 1.501 | **0.960** | 1.089 | 1.235 | 1.906 | 0.572 |
| Italian | 1.169 | 1.105 | 1.534 | **0.948** | 1.543 | 1.743 |
| Portuguese | 2.046 | 1.778 | 2.254 | **1.526** | 1.877 | 1.331 |
| Spanish | 2.031 | 1.491 | 1.491 | 1.126 | **1.029** | 1.084 |
| Japanese | 4.189 | 5.121 | 6.404 | 3.823 | **3.519** | 10.646 |
| Korean | 2.458 | 2.695 | **1.741** | 1.755 | 1.747 | 1.865 |
| French | 2.852 | **2.631** | 2.931 | 2.858 | 4.099 | 5.216 |
| Russian | 5.957 | 4.535 | 4.458 | **3.212** | 4.281 | 3.878 |
| **说话人相似度（SIM ↑）** | | | | | | |
| Chinese | 0.797 | 0.796 | **0.811** | 0.799 | 0.780 | 0.677 |
| English | 0.811 | 0.815 | **0.829** | 0.775 | 0.756 | 0.613 |
| German | 0.749 | 0.737 | 0.769 | **0.775** | 0.733 | 0.614 |
| Italian | 0.722 | 0.718 | 0.792 | **0.817** | 0.699 | 0.579 |
| Portuguese | 0.790 | 0.783 | 0.794 | **0.817** | 0.805 | 0.711 |
| Spanish | 0.732 | 0.731 | 0.812 | **0.814** | 0.762 | 0.615 |
| Japanese | **0.810** | 0.807 | 0.798 | 0.788 | 0.776 | 0.738 |
| Korean | **0.824** | 0.814 | 0.812 | 0.799 | 0.776 | 0.700 |
| French | 0.698 | 0.703 | 0.700 | **0.714** | 0.628 | 0.535 |
| Russian | 0.734 | 0.744 | 0.781 | **0.792** | 0.761 | 0.676 |

**核心结论**：
- **WER**：Qwen3-TTS 在 **10 种语言中的 6 种**（中、英、意、法、韩、俄）取得最低 WER，显著超越 MiniMax 与 ElevenLabs；德、葡、西、日 4 种保持高度竞争力（日/西/葡/德上 ElevenLabs 或 MiniMax 略优）
- **SIM**：**Qwen3-TTS 在全部 10 种语言上 SIM 均最高**——论文表述为「dominant performance in voice cloning fidelity」
- **注意 ElevenLabs 在中文上的灾难性表现（WER 16.026）**，说明中文场景 Qwen3-TTS 优势极大

#### 跨语种语音生成（Table 7，Mixed Error Rate ↓）

| 任务 | 25Hz-1.7B-Base | **12Hz-1.7B-Base** | CosyVoice3 | CosyVoice2 |
|---|---|---|---|---|
| en→zh | 5.66 | **4.77** | 5.09 | 13.5 |
| ja→zh | 3.92 | **3.43** | **3.05** | 48.1 |
| ko→zh | 1.14 | 1.08 | **1.06** | 7.70 |
| zh→en | 2.91 | **2.77** | 2.98 | 6.47 |
| ja→en | 3.95 | **3.04** | 4.20 | 17.1 |
| ko→en | 3.48 | **3.09** | 4.19 | 11.2 |
| zh→ja | 9.29 | 8.40 | **7.08** | 13.1 |
| en→ja | 7.74 | 7.21 | **6.80** | 14.9 |
| ko→ja | 4.17 | **3.67** | 3.93 | 5.86 |
| **zh→ko** | 8.12 | **4.82** | 14.4 | 24.8 |
| en→ko | 6.83 | **5.14** | 5.87 | 21.9 |
| ja→ko | 6.86 | **5.59** | 7.92 | 21.5 |

**最强结果**：**zh→ko 从 CosyVoice3 的 14.4 降到 4.82（错误率降低约 66%）**。12Hz-1.7B-Base 在「目标为英语与韩语」的全部场景建立新 SOTA。

#### 已知语言短板

1. **日语、俄语的 WER 明显偏高**（3.2–6.4 区间，远高于中英的 0.8–1.2）
2. **zh→ja / en→ja 方向上 CosyVoice3 更优**
3. 论文 Future Work：「further scale our multilingual coverage **beyond the current 10 languages**」——官方承认 10 语种不够


### 19. 流式能力

#### 架构：Dual-Track 混合流式生成

官方 README 四大架构特性之三：

> 「**Extreme Low-Latency Streaming Generation** — Dual-Track 混合流式生成架构；**单模型同时支持 streaming / non-streaming**；**输入单个字符后即可输出首个音频包**；端到端合成延迟低至 **97ms**」

论文 §3.1 的实现说明：

> 「For real-time synthesis, Qwen3-TTS employs a **dual-track representation by concatenating textual and acoustic tokens along the channel axis**. Upon receiving a textual token, the model **immediately** predicts the corresponding acoustic tokens, which are then converted into waveforms by the Code2Wav module.」

#### 分包与延迟参数（论文 §3.4）

| 路线 | 分包规则 | 每包音频 | Look-ahead |
|---|---|---|---|
| **12Hz** | 1 packet = **4 tokens**（12.5Hz，每 token 80ms） | **320 ms** | **0**（纯左上下文） |
| **25Hz** | LM 需生成 **16 tokens** → DiT 产出首个 8-token mel chunk（25Hz，每 token 40ms → 320ms mel）；BigVGAN +130ms 右上下文 | 首包约 **190 ms** | DiT 1 块前瞻 + BigVGAN 130ms |

#### 实测首包延迟与 RTF（Table 2）

**测试条件**：内部 vLLM 引擎（vLLM V0 backend）、单个 typical computational resource（**卡型未指明 → 待核实**）、对 tokenizer 解码阶段应用 torch.compile + CUDA Graph；端到端实测。

| 模型 | 并发 | LM TTFP | Tokenizer Decode TPP | **首包延迟** | LM TPP | **RTF** |
|---|---|---|---|---|---|---|
| **12Hz-0.6B** | 1 | 93 ms | 4 ms | **97 ms** | 19 ms | 0.288 |
| 12Hz-0.6B | 3 | 174 ms | 5 ms | 179 ms | 22 ms | 0.338 |
| 12Hz-0.6B | 6 | 294 ms | 5 ms | **299 ms** | 30 ms | **0.434** |
| **12Hz-1.7B** | 1 | 97 ms | 4 ms | **101 ms** | 21 ms | 0.313 |
| 12Hz-1.7B | 3 | 190 ms | 5 ms | 195 ms | 24 ms | 0.363 |
| 12Hz-1.7B | 6 | 328 ms | 5 ms | **333 ms** | 32 ms | **0.463** |
| 25Hz-0.6B | 1 | 113 ms | 25 ms | 138 ms | 50 ms | 0.234 |
| 25Hz-0.6B | 3 | 198 ms | 62 ms | 260 ms | 59 ms | 0.378 |
| 25Hz-0.6B | 6 | 334 ms | 147 ms | **481 ms** | 80 ms | **0.709** |
| 25Hz-1.7B | 1 | 125 ms | 25 ms | 150 ms | 56 ms | 0.253 |
| 25Hz-1.7B | 3 | 222 ms | 62 ms | 284 ms | 64 ms | 0.394 |
| 25Hz-1.7B | 6 | 376 ms | 147 ms | **523 ms** | 85 ms | **0.725** |

**关键读数**：

1. **低并发（1）下 25Hz 的 RTF 反而更低**（0.234/0.253 vs 0.288/0.313）——因为 25Hz 单码本序列更短、LM 步数更少。
2. **高并发（6）下 12Hz 完胜**（RTF 0.434/0.463 vs 0.709/0.725）——25Hz 的 DiT+BigVGAN 解码在高并发下急剧劣化（25ms → 147ms）。
3. **12Hz 的 codec 解码几乎不随并发增长**（4 → 5 ms），这是纯因果 ConvNet 的红利。
4. **首包延迟 12Hz 全面领先**：97–333 ms vs 138–523 ms。

#### 工程注意事项

1. **在官方 `qwen-tts` 包中，流式与非流式由同一个模型支持**（单模型双模），无需切换权重。
2. **vLLM-Omni 当前仅支持离线推理，不支持真正的流式音频输出**（官方 README 明示：「Now only offline inference is supported. Online serving will be supported later」）。
3. **社区流式 fork**：`dffdeeq/Qwen3-TTS-streaming`（声称约 6x 加速）、`andimarafioti/faster-qwen3-tts`（CUDA Graphs，RTX 4060/4090 上 5.8–9.8x 吞吐）——**均为第三方，非官方**。
4. 官方提示：streaming fork 能改善**用户感知延迟**（更早出声），但**不等于**开箱即用的 voice-agent-ready；仍需实测首音频延迟、RTF、chunk 边界稳定性、重复请求下的显存行为。

#### 未披露项（待核实）

- 官方 `qwen-tts` 包中流式推理的具体 API（论文与 README 均未给出 streaming 的 Python 调用示例）
- 流式模式下的 chunk 大小配置参数
- 流式与离线两种模式的质量差异（论文未给出对应的 WER 对比）


### 20. VAD 与后处理

#### 官方提供的能力

| 能力 | 支持情况 | 说明 |
|---|---|---|
| 音频 I/O | ✅ | `ref_audio` 支持本地路径 / URL / base64 / `(numpy_array, sample_rate)` |
| 批量生成 | ✅ | 所有 `generate_*` 方法的 `text` / `language` / `speaker` / `instruct` **均可传 list** |
| 音色 prompt 复用 | ✅ | `create_voice_clone_prompt()` 提取一次、多次复用 |
| 生成参数透传 | ✅ | 可传 HF `generate` 参数（`max_new_tokens`、`top_p`、`top_k`、`temperature` 等） |
| **VAD（语音活动检测）** | ❌ **未提供** | 官方 README 无 VAD 相关模块 |
| **长文本自动分段** | ❌ **未提供** | 需调用方自行分 chunk |
| **音频后处理**（响度归一化、去噪） | ❌ **未提供** | |
| **标点/数字归一化（TN）** | ❌ 未见文档 | 论文提到对**含噪输入文本**有鲁棒性（「intelligently handling complex text with automatic adaptation of tone and fluency」），但非独立模块 |

#### 长文本处理：必须自行分 chunk，且需注意音色漂移

这是 TTS 工程化最容易踩的坑。社区实践（**第三方，非官方**）给出的对策：

1. **每个 chunk 前固定随机种子** —— 防止跨 chunk 采样差异导致音色漂移
2. **复用同一次提取的 speaker embedding** —— 每 chunk 重新编码参考音频会引入可变性
3. **尽量用更大的 chunk**（减少边界数量）
4. **设置 `repetition_penalty=1.05`** —— 多 chunk 一致性的关键参数
5. 在**语义边界**处分段，保留上下文

#### 已知的后处理相关问题

| 问题 | 现象 | 对策 |
|---|---|---|
| **冷启动解码失真** | 新进程首次推理时，speech tokenizer decoder 无 warm-up 上下文，前几帧音频重建错误（GitHub issue #219） | 前置静音 codec token 作为 warm-up，再从输出中裁掉对应样本。**对以数字开头的短语音影响最明显** |
| **音频在约 24 秒处被截断** | `chunked_decode` 过度裁剪样本（PR #259，未合并） | 关注上游 PR 状态 |
| **EOS 失败（无限生成）** | 约 **0.5%** 的 base 模型推理不会停止 | 显式设置 `max_new_tokens`；必要时指定 `eos_token_id=[2150, 2157, 151670, 151673, 151645, 151643]` |
| **纯噪声输出** | 学习率过高导致 | 用 `lr=2e-6` |

> 上述四项均来自**社区实践**（instavar.com 与配套 GitHub 仓库），**非官方文档**。

#### DashScope API 侧（云端）

- 输出格式可选：`wav` / `mp3` / `pcm` / `flac` / `aac` / `opus`（NVIDIA Dynamo 文档的 vLLM-Omni 接口描述）
- DashScope Realtime API 输出 `PCM_24000HZ_MONO_16BIT`（官方示例）

#### 未披露项（待核实）

- 输出音频的响度标准（LUFS）
- 是否内建音频裁剪/静音切除
- 官方是否提供文本前端（TN）模块


### 21. 音色克隆

#### 核心能力：3 秒零样本声音克隆

官方 README 与论文 §1 均明确：**3-second voice cloning**（从 3 秒参考音频克隆目标音色）。

#### 两条克隆路径（论文 §3.3）

| 路径 | 机制 | 特点 |
|---|---|---|
| **(i) Speaker Embedding 路径** | 通过 speaker embedding 从参考语音克隆 | **支持实时克隆**（real-time cloning）；`x_vector_only_mode=True` |
| **(ii) In-Context Learning（ICL）路径** | 通过 text–speech 对做上下文学习 | **更好保留韵律（better preserves prosody）**；需要 `ref_text` |

#### API 用法

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0", dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")

wavs, sr = model.generate_voice_clone(
    text="I am solving the equation: ...",
    language="English",
    ref_audio="https://.../clone.wav",
    ref_text="Okay. Yeah. I resent you. ...",
)
sf.write("output_voice_clone.wav", wavs[0], sr)
```

**Prompt 复用（批量生成必用）**：

```python
prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=False)
wavs, sr = model.generate_voice_clone(
    text=["Sentence A.", "Sentence B."],
    language=["English", "English"],
    voice_clone_prompt=prompt_items)
```

**`ref_audio` 输入形式**：本地路径 / URL / base64 / `(numpy_array, sample_rate)`

#### `x_vector_only_mode` 权衡

| 模式 | 需要 `ref_text` | 质量 |
|---|---|---|
| `False`（默认，ICL） | ✅ | 高（韵律保留更好） |
| `True` | ❌ | **官方明示「cloning quality may be reduced」** |

#### 零样本克隆性能（Seed-TTS test set，WER ↓，Table 5）

| 模型 | test-zh | test-en |
|---|---|---|
| Seed-TTS | 1.12 | 2.25 |
| MaskGCT | 2.27 | 2.62 |
| E2 TTS | 1.97 | 2.19 |
| F5-TTS | 1.56 | 1.83 |
| Spark TTS | 1.20 | 1.98 |
| Llasa-8B | 1.59 | 2.97 |
| KALL-E | 0.96 | 1.94 |
| FireRedTTS 2 | 1.14 | 1.95 |
| CosyVoice 3 | **0.71** | 1.45 |
| MiniMax-Speech | 0.83 | 1.65 |
| Qwen3-TTS-25Hz-0.6B-Base | 1.18 | 1.64 |
| Qwen3-TTS-25Hz-1.7B-Base | 1.10 | 1.49 |
| Qwen3-TTS-12Hz-0.6B-Base | 0.92 | 1.32 |
| **Qwen3-TTS-12Hz-1.7B-Base** | **0.77** | **1.24** |

**结论**：
- **test-en 1.24 为该基准 SOTA**（超越 CosyVoice 3 的 1.45、MiniMax-Speech 的 1.65、Seed-TTS 的 2.25）
- test-zh 0.77 仅次于 CosyVoice 3 的 0.71（**中文内容一致性上 CosyVoice 3 仍略优**）
- **12Hz 变体在内容准确率上一致优于 25Hz 变体** —— 论文解释：「12.5Hz tokenizer 更粗的时间分辨率使自回归模型能更好地建模长程依赖，从而生成更稳定的语音」
- 0.6B → 1.7B 有稳定增益

#### 说话人相似度（SIM）

见 `languages` 字段 Table 6：**Qwen3-TTS 在全部 10 种语言的 SIM 上均高于 MiniMax-Speech 与 ElevenLabs Multilingual v2**。

#### 说话人微调（超越零样本）

若需**固定音色**的批量生产，官方推荐路径是**说话人微调**（`finetuning/`）而非每次 ICL 克隆。论文 Table 9 显示：在 Aiden Voice 上微调后，**10 语言中 7 种 WER 优于 GPT-4o-Audio-Preview**，且**只用单语数据微调即可把音色迁移到全部 10 语言**。

#### 已知限制

1. **官方微调仅支持单说话人**——多说话人微调需等上游支持
2. **`x_vector_only_mode=True` 质量下降**
3. **音色漂移**：长文本分 chunk 生成时，每 chunk 独立采样会导致音色变化（社区对策见 `vad_and_postprocess`）
4. **口音迁移弱**：社区反馈「预训练的口音偏见很强」，口音难以通过克隆迁移（**第三方观察，待核实**）

#### 合规提示

3 秒克隆属高敏感能力。社区 fork 的 README 均附有伦理声明（仅克隆获得同意的声音、不得用于欺诈、披露 AI 生成音频）。**官方 README 未包含该声明，但部署方需自行履行相应义务。**


### 22. 情感与风格控制

#### 三种可控生成模式

| 模式 | 模型 | 控制方式 | 指令支持 |
|---|---|---|---|
| **VoiceDesign** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 用自然语言描述**凭空设计新音色** | ✅ |
| **CustomVoice** | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9 个预设音色 + 指令控制风格/情感/韵律 | ✅ |
| **CustomVoice** | `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 9 个预设音色 | ❌ |
| **Base** | `Qwen3-TTS-12Hz-1.7B/0.6B-Base` | 3 秒克隆 | ❌ **不支持风格指令** |

#### 1. VoiceDesign：自然语言设计音色

```python
wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显...",
)
```

**技术支撑（论文 §3.3）**：
- 基于 Qwen3 文本基座，**继承强文本理解能力**
- 训练中引入**概率激活的 "thinking pattern"**，专门提升对**复杂描述**的指令遵循
- 论文 §5 结论：正是 thinking pattern 让 Qwen3-TTS 在 voice design 场景取得 SOTA

**「先设计再克隆」工作流**（官方 README 给出完整示例）：
1. 用 VoiceDesign 模型生成一段参考短音频
2. 用 `create_voice_clone_prompt()` 建立可复用 prompt
3. 用 `generate_voice_clone(voice_clone_prompt=...)` 批量生成同音色的新内容

#### 2. CustomVoice：预设音色 + 指令风格控制

```python
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese",
    speaker="Vivian",
    instruct="用特别愤怒的语气说",
)
```

支持 batch（可混合语言、不同 speaker、不同 instruct）。

#### 3. InstructTTSEval 可控性评测（Table 8）

指标：**APS**（Attribute Perception and Synthesis accuracy，↑）、**DSD**（Description–Speech Consistency，↑）、**RP**（Response Precision，↑）

##### Target Speaker（编辑/控制已有说话人）

| 模型 | APS-ZH | DSD-ZH | RP-ZH | APS-EN | DSD-EN | RP-EN |
|---|---|---|---|---|---|---|
| Gemini-flash | 88.2 | 90.9 | 77.3 | **92.3** | **93.8** | **80.1** |
| Gemini-pro | **89.0** | 90.1 | 75.5 | 87.6 | 86.0 | 67.2 |
| Qwen3TTS-25Hz-1.7B-CustomVoice | 83.1 | 75.0 | 63.0 | 79.0 | 82.8 | 69.3 |
| Qwen3TTS-12Hz-1.7B-CustomVoice | 83.0 | 77.8 | 61.2 | 77.3 | 77.1 | 63.7 |
| GPT-4o-mini-tts | 54.9 | 52.3 | 46.0 | 76.4 | 74.3 | 54.8 |

**结论**：CustomVoice **显著优于 GPT-4o-mini-tts**（论文称中文 APS 提升 +28%），但 **Gemini 系列仍是上界（strong upper bound）**。

##### Voice Design（创造新音色）

| 模型 | APS-ZH | DSD-ZH | RP-ZH | APS-EN | DSD-EN | RP-EN |
|---|---|---|---|---|---|---|
| **Qwen3TTS-12Hz-1.7B-VD** | **85.2** | **81.1** | **65.1** | 82.9 | 82.4 | **68.4** |
| Mimo-Audio-7B-Instruct | 75.7 | 74.3 | 61.5 | 80.6 | 77.6 | 59.5 |
| VoiceSculptor | 75.7 | 64.7 | 61.5 | - | - | - |
| Hume | - | - | - | **83.0** | 75.3 | 54.3 |
| VoxInstruct | 47.5 | 52.3 | 42.6 | 54.9 | 57.0 | 39.3 |
| Parler-tts-mini / large | - | - | - | 63.4 / 60.0 | 48.7 / 45.9 | 28.6 / 31.2 |
| PromptTTS | - | - | - | 64.3 | 47.2 | 31.4 |
| PromptStyle | - | - | - | 57.4 | 46.4 | 30.9 |

**结论**：**Qwen3TTS-12Hz-1.7B-VD 在开源模型中建立新 SOTA**，中文三项全优，英文 RP 最优；超越 Mimo-Audio、VoiceSculptor、ParlerTTS、PromptTTS 等，并优于商业系统 Hume（DSD/RP）与 GPT-4o-mini-tts。

#### 4. 控制的技术机制

- **指令前置**：「we prepend user-provided instructions containing fine-grained control signals to the input sequences」
- **ChatML 格式化**：把语音控制变成语言建模任务
- 可控制的维度：**音色（timbre）、情感（emotion）、韵律（prosody）**及其他多维声学属性

#### 5. 已知限制

1. **Base 模型完全不支持指令控制**（官方 README 表格中 Instruction Following 列为 ❌）
2. **0.6B-CustomVoice 不支持指令控制**（只有 1.7B 版本支持）
3. **口音控制能力弱**：社区反馈「预训练口音偏见很强，口音难以通过指令迁移」（**第三方观察，待核实**）
4. **情感标签需逐音色实测**：社区建议把情感标签当作「prompt 控制」而非精确指令，对**大方向**可靠、对**精确表现**不可靠；若音色本身不稳定，情感标签会**降低**克隆相似度
5. **微调会使表现力趋于平坦**：社区观察「Fine-tuning flattens expressiveness」——如需保留表现力，建议用 ICL 克隆而非微调（**第三方，待核实**）

#### 6. 未披露项（待核实）

- 支持的情感/风格标签的官方词表（官方示例仅给出「用特别愤怒的语气说」这类自由文本）
- thinking pattern 的激活概率与触发条件
- 指令长度上限


### 23. 鲁棒性

#### 1. 长语音稳定性（论文 §4.2.6，最核心的鲁棒性主张）

**测试设置**：内部数据集，中英各 **100 篇文本，长度 200–2000 词**；用 **Qwen3-ASR** 转写后按 Seed-TTS Eval 方法计算 WER。

| 模型 | long-zh（WER ↓） | long-en（WER ↓） |
|---|---|---|
| Higgs-Audio-v2（chunk 分块方案） | 5.505 | 6.917 |
| VibeVoice | 22.619 | 1.780 |
| VoxCPM | 4.835 | 7.474 |
| **Qwen3-TTS-25Hz-1.7B-CustomVoice** | **1.517** | **1.225** |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 2.356 | 2.812 |

**结论与归因（论文原文）**：
- Qwen3-TTS-25Hz-1.7B 在双语上均取得最低 WER
- **VibeVoice 在中文上严重退化（WER > 22）**
- **对比 Higgs-Audio-v2 等 chunk-based 系统存在边界伪影（boundary artifacts），Qwen3-TTS 生成无缝音频、全程韵律一致**
- **25Hz 优于 12Hz**（1.517/1.225 vs 2.356/2.812）——论文推测「语义 token 可能更有利于在长序列上维持稳定性」
- 结论章：可生成「**超过 10 分钟**的流畅语音，无 chunk-based 系统的典型伪影」

> ⚠️ **重要工程提示**：长语音场景的最优解（25Hz）**权重尚未开源**。已开源的 12Hz 系列在 >10 分钟长语音上的 WER 为 2.356/2.812，虽不如 25Hz，但仍远优于 Higgs-Audio-v2 / VibeVoice / VoxCPM。

#### 2. 抗幻觉（S2 高质量阶段的直接目标）

论文 §3.2：S2 阶段「**alleviates hallucinations caused by noisy data** in the initial stage and significantly improves the quality of generated speech」。

这是论文唯一明确量化归因的鲁棒性来源，但未给出消融数据 → **具体幅度待核实**。

#### 3. 跨任务稳定性（GSPO 阶段目标）

P2 阶段用 rule-based rewards 的 GSPO 「comprehensively enhance the model's **capabilities and stability across tasks**」。

#### 4. 复杂文本鲁棒性（官方 README 定性描述）

> 「Strong Robustness：对含噪输入文本（noisy input text）展现出明显改善的鲁棒性，能智能处理复杂文本，自动适配语气与流畅度（automatic adaptation of tone and fluency）」

**无量化数据 → 待核实。**

#### 5. 跨语种鲁棒性

见 `languages` 字段 Table 7：12Hz-1.7B-Base 在 12 个跨语种方向中的 9 个取得最优，zh→ko 错误率相对 CosyVoice3 降低约 66%。**CosyVoice2 在多个方向表现不稳定（13.5–48.1），而 Qwen3-TTS 在所有方向保持一致低错误率**。

#### 6. 说话人微调的跨语言泛化（Table 9）

**只在单语数据上微调，音色与韵律即可迁移到全部 10 种语言而不降低稳定性**——这是鲁棒性的一项强证据。详见 `post_training_rl` 字段。

#### 7. 已知脆弱点

| 脆弱点 | 现象 | 来源 |
|---|---|---|
| 长文本分 chunk 音色漂移 | 每 chunk 独立采样导致音色变化 | 社区（第三方） |
| 冷启动解码失真 | 首次推理前几帧重建错误（issue #219） | 社区（第三方） |
| 微调后表现力平坦化 | 情感/韵律表现力下降 | 社区（第三方，待核实） |
| 高学习率导致崩溃 | 纯噪声 / EOS 丢失 / 无限生成 | 社区（第三方） |
| 微调后语速逐轮加快 | double label-shift bug（**官方已修**） | GitHub issue #179 / PR #178 |
| 数字开头短语音质量 | 冷启动失真在此场景最明显 | 社区（第三方） |

#### 8. 未评测项（论文未覆盖，需自行验证）

- 极端情感/风格指令下的稳定性
- SSR（speaker similarity under noise reference audio）—— 参考音频带噪时的克隆鲁棒性
- 长语音（>10 分钟）的 SIM 衰减曲线
- 对抗性/越狱文本输入的鲁棒性


### 24. ASR 指标

**N/A — TTS 模型不产出 ASR 指标。**

Qwen3-TTS 是 Text→Speech 方向的模型，不输出转写文本，因此没有 WER/CER/语种识别准确率等 ASR 指标。

需要 ASR 指标请见 `qwen3_asr.json` 的 `asr_metrics` 字段。

#### 但本模型的论文中**使用**了 ASR 指标（易混淆，特别说明）

Qwen3-TTS 论文在以下三处用 **ASR 转写 + WER** 作为**内容一致性（content consistency）**的度量手段，此时 WER 是 **TTS 的质量指标**（衡量"合成的语音是否说对了文本"），不是 ASR 能力指标：

| 用途 | 转写器 | 指标 |
|---|---|---|
| Table 3：25Hz Tokenizer 的语义质量 | 未指明（用 ASR 任务评测） | WER ↓ |
| Table 5/6/7/9/10：语音生成的内容一致性 | 表 10 明确使用 **Qwen3-ASR** | WER/CER ↓ |
| Table 6：说话人相似度 | WavLM-based speaker verification | SIM ↑ |

**这构成了一个官方内部的评测闭环：Qwen3-ASR ←→ Qwen3-TTS 互为评测工具。** 若你在自建 TTS 评测流水线时想复用该做法，用 Qwen3-ASR 作转写器是官方验证过的选择（论文明确说明选它是因为「its high accuracy in long-form recognition」）。


### 25. TTS 指标

**统一评测配置**（官方 README）：`bfloat16`、`max_new_tokens=2048`；Seed-TTS test set 与 InstructTTS 用 `language="auto"`，其余显式指定语言。

---

#### 1. 零样本语音克隆（Seed-TTS test set，WER ↓，Table 5）

| 模型 | test-zh | test-en |
|---|---|---|
| Seed-TTS | 1.12 | 2.25 |
| MaskGCT | 2.27 | 2.62 |
| E2 TTS | 1.97 | 2.19 |
| F5-TTS | 1.56 | 1.83 |
| Spark TTS | 1.20 | 1.98 |
| Llasa-8B | 1.59 | 2.97 |
| KALL-E | 0.96 | 1.94 |
| FireRedTTS 2 | 1.14 | 1.95 |
| CosyVoice 3 | **0.71** | 1.45 |
| MiniMax-Speech | 0.83 | 1.65 |
| Qwen3-TTS-25Hz-0.6B-Base | 1.18 | 1.64 |
| Qwen3-TTS-25Hz-1.7B-Base | 1.10 | 1.49 |
| Qwen3-TTS-12Hz-0.6B-Base | 0.92 | 1.32 |
| **Qwen3-TTS-12Hz-1.7B-Base** | **0.77** | **1.24** |

→ **test-en 1.24 为 SOTA**；12Hz 一致优于 25Hz。

#### 2. 多语种（WER ↓ / SIM ↑，Table 6）

见 `languages` 字段完整表格。核心结论：**10 语种中 6 种 WER 最优；10 语种 SIM 全部最优**（超越 MiniMax-Speech 与 ElevenLabs Multilingual v2）。

#### 3. 跨语种（Mixed Error Rate ↓，Table 7）

见 `languages` 字段完整表格。核心结论：**12Hz-1.7B-Base 在目标为英语/韩语的全部方向 SOTA；zh→ko 4.82 vs CosyVoice3 14.4（-66%）**。

#### 4. 可控生成（InstructTTSEval，Table 8）

见 `emotion_and_control` 字段完整表格。核心结论：
- **VoiceDesign：Qwen3TTS-12Hz-1.7B-VD 开源 SOTA**，中文 APS 85.2 / DSD 81.1 / RP 65.1 全项最优
- **Target Speaker（CustomVoice）：显著优于 GPT-4o-mini-tts**（中文 APS +28%），但 Gemini 系列仍是上界

#### 5. 目标说话人微调后多语种（WER ↓，Table 9）

见 `post_training_rl` 字段完整表格。核心结论：**10 语言中 7 种优于 GPT-4o-Audio-Preview**；单语微调即可跨 10 语迁移。

#### 6. 长语音生成（WER ↓，Table 10）

| 模型 | long-zh | long-en |
|---|---|---|
| Higgs-Audio-v2（chunk） | 5.505 | 6.917 |
| VibeVoice | 22.619 | 1.780 |
| VoxCPM | 4.835 | 7.474 |
| **Qwen3-TTS-25Hz-1.7B-CustomVoice** | **1.517** | **1.225** |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 2.356 | 2.812 |

#### 7. Codec / Tokenizer 重建质量

##### 12Hz（LibriSpeech test-clean，2,620 条，Table 4）

| 模型 | NQ | Codebook | FPS | PESQ_WB ↑ | PESQ_NB ↑ | STOI ↑ | UTMOS ↑ | SIM ↑ |
|---|---|---|---|---|---|---|---|---|
| SpeechTokenizer | 8 | 1024 | 50 | 2.60 | 3.05 | 0.92 | 3.90 | 0.85 |
| X-codec | 2 | 1024 | 50 | 2.68 | 3.27 | 0.86 | 4.11 | 0.84 |
| X-codec 2 | 1 | 65536 | 50 | 2.43 | 3.04 | 0.92 | 4.13 | 0.82 |
| XY-Tokenizer | 8 | 1024 | 12.5 | 2.41 | 3.00 | 0.91 | 3.98 | 0.83 |
| Mimi | 16 | 2048 | 12.5 | 2.88 | 3.42 | 0.94 | 3.87 | 0.87 |
| FireredTTS 2 Tokenizer | 16 | 2048 | 12.5 | 2.73 | 3.28 | 0.94 | 3.88 | 0.87 |
| **Qwen-TTS-Tokenizer-12Hz** | **16** | **2048** | **12.5** | **3.21** | **3.68** | **0.96** | **4.16** | **0.95** |

##### 25Hz（ASR 代理任务，WER ↓，Table 3）

| 模型 | Codebook | FPS | C.V. EN | C.V. CN | Fleurs EN | Fleurs CN |
|---|---|---|---|---|---|---|
| S3 Tokenizer (VQ) | 4096 | 50 | 12.06 | 15.38 | - | - |
| S3 Tokenizer (FSQ) | 6561 | 25 | 10.67 | 7.29 | 6.58 | 4.43 |
| **Qwen-TTS-Tokenizer-25Hz (Stage 1)** | **32768** | 25 | **7.51** | 10.73 | **3.07** | **4.23** |
| Qwen-TTS-Tokenizer-25Hz (Stage 2) | 32768 | 25 | 10.40 | 14.99 | 4.14 | 4.67 |

---

#### 指标口径说明（供自建评测复用）

| 指标 | 含义 | 方向 |
|---|---|---|
| **WER / CER** | 合成音频经 ASR 转写后与输入文本的差异 = **内容一致性** | ↓ |
| **SIM** | WavLM-based speaker verification 的余弦相似度 = **音色相似度** | ↑ |
| **PESQ_WB / PESQ_NB** | 感知语音质量（宽带/窄带） | ↑ |
| **STOI** | 短时客观可懂度 | ↑ |
| **UTMOS** | 无参考语音自然度 MOS 预测 | ↑ |
| **APS / DSD / RP** | InstructTTSEval：属性感知与合成准确率 / 描述-语音一致性 / 响应精确率 | ↑ |
| **RTF** | Real-Time Factor = 生成耗时 / 音频时长 | ↓（<1 为快于实时） |
| **首包延迟** | LM TTFP + Tokenizer Decode TPP | ↓ |

#### 缺失的官方指标（待核实）

- **无主观 MOS / CMOS 人工听感评分**（论文 Table 中仅有 UTMOS 等客观代理指标）
- 无 speaker similarity 在长语音下的衰减数据
- 无情感分类准确率
- 无 12Hz vs 25Hz 的完整音质 A/B（UTMOS 等仅对比了 12Hz 与其他 codec）


### 26. 延迟与吞吐

#### 官方流式效率基准（arXiv:2601.15621 Table 2）

**测试条件**：
- 引擎：**内部 vLLM 引擎（vLLM V0 backend）**
- 硬件：**single typical computational resource**（**GPU 型号未指明 → 待核实**）
- 优化：对 **tokenizer 解码阶段**应用 **torch.compile + CUDA Graph**
- 所有数字为**端到端实测**；稳态成本（LM TPP）按流式生成中每个 speech packet 测量
- **首包延迟 = LM TTFP + Tokenizer Decode TPP**

| 模型 | 并发 | LM TTFP | Tokenizer Decode TPP | **首包延迟** | LM TPP | **RTF** |
|---|---|---|---|---|---|---|
| **12Hz-0.6B** | 1 | 93 ms | 4 ms | **97 ms** | 19 ms | 0.288 |
| 12Hz-0.6B | 3 | 174 ms | 5 ms | 179 ms | 22 ms | 0.338 |
| 12Hz-0.6B | 6 | 294 ms | 5 ms | **299 ms** | 30 ms | **0.434** |
| **12Hz-1.7B** | 1 | 97 ms | 4 ms | **101 ms** | 21 ms | 0.313 |
| 12Hz-1.7B | 3 | 190 ms | 5 ms | 195 ms | 24 ms | 0.363 |
| 12Hz-1.7B | 6 | 328 ms | 5 ms | **333 ms** | 32 ms | **0.463** |
| 25Hz-0.6B | 1 | 113 ms | 25 ms | 138 ms | 50 ms | **0.234** |
| 25Hz-0.6B | 3 | 198 ms | 62 ms | 260 ms | 59 ms | 0.378 |
| 25Hz-0.6B | 6 | 334 ms | 147 ms | 481 ms | 80 ms | 0.709 |
| 25Hz-1.7B | 1 | 125 ms | 25 ms | 150 ms | 56 ms | **0.253** |
| 25Hz-1.7B | 3 | 222 ms | 62 ms | 284 ms | 64 ms | 0.394 |
| 25Hz-1.7B | 6 | 376 ms | 147 ms | 523 ms | 85 ms | 0.725 |

**术语**：TTFP = Time-To-First-Packet（LM 产生首个语音包的 token 组所需时间）；TPP = Time-Per-Packet（稳态下每产生一个 packet 的 token 所需时间）；RTF = Real-Time Factor。

#### 三条核心工程结论

1. **低并发看 RTF，选 25Hz 更快；高并发看 RTF，选 12Hz 更快。**
   并发 1：25Hz-0.6B RTF 0.234 < 12Hz-0.6B 0.288（25Hz 单码本序列短、LM 步数少）
   并发 6：12Hz-0.6B RTF 0.434 << 25Hz-0.6B 0.709（25Hz 的 DiT+BigVGAN 解码在高并发下劣化 6 倍：25ms → 147ms）

2. **首包延迟 12Hz 全面领先**：97–333 ms vs 138–523 ms。差距主要来自 codec 解码（4–5ms vs 25–147ms）。

3. **12Hz 的 codec 解码几乎与并发无关**（4 → 5 ms）——纯因果 ConvNet、无 look-ahead 的红利。**这意味着 12Hz 在高并发服务化场景下的优势会随并发继续放大**（论文未测并发 >6 的数据 → 外推部分**待核实**）。

#### 分包定义对延迟的影响

| 路线 | 分包 | 每包音频 | 首包需等待 |
|---|---|---|---|
| 12Hz | 4 tokens/packet | 320 ms | 4 tokens（LM 出 4 个即可解码） |
| 25Hz | 8 tokens/chunk（DiT）+ 16 tokens 预生成 | 320 ms mel（≈190 ms 音频） | **16 tokens** + BigVGAN 130ms look-ahead |

论文原文解释 12Hz 的分包选择：「为避免过小的数据包导致过多调度开销，我们把一个 speech packet 定义为 4 个 token（320ms 语音）。该设计显著降低解码时间、获得更低首包延迟，同时因为 codec decoder 轻量且对批处理友好，在高并发下也能保持低 RTF。」

#### 社区实测延迟（**第三方，非官方，待核实**）

| 来源 | 配置 | 结果 |
|---|---|---|
| CSDN 实测（12Hz-1.7B-Base，35 字中文） | RTX 3090 / 4090 / 5090 | 首包 **97.2 / 96.8 / 96.5 ms**；端到端 1420 / 1280 / 1150 ms |
| 同上，文本长度扫描 | RTX 3090 | 首包恒定 ~97ms（5 字 / 35 字 / 100 字相同）；端到端 1120 / 1420 / 2150 ms（约每字 +10ms） |
| PyPI `qwen3-tts-triton` 自测 | RTX 5090，bf16，1.7B | Base（PyTorch）RTF 0.88x；Triton 优化 1.00x；**"Faster" 方案 3.49x**；**Hybrid（Faster+Triton）4.20–4.27x** |
| ai-rockstars 技术分析 | **无 FA2，RTX 5090** | **约 0.3x 实时**（即 RTF ≈ 3.3，远慢于实时） |

**交叉印证**：CSDN 的首包 96.5–97.2ms 与论文 97ms 高度吻合，可作为官方数据的独立验证。端到端延迟随文本长度近似线性增长（每字约 +10ms）也是一个有用的工程换算依据。

#### 吞吐（QPS）

- **论文未提供 QPS / requests-per-second 数据 → 待核实**
- 第三方估算（qwen3-tts.app 硬件指南，**可靠性存疑，仅作量级参考**）：
  - RTX 5090：0.6B 约 85 req/s，1.7B 约 58 req/s（短文本 20 词）
  - RTX 4090：1.7B RTF 0.65
  - H100：1.7B RTF 0.35–0.50
- **官方 README 明确：vLLM-Omni 当前仅支持离线推理，在线 serving 后续支持** —— 因此**官方路径下目前没有 serving 级吞吐数据**

#### vLLM-Omni 加速

官方 README：「vLLM-Omni will continue to offer support and optimization for Qwen3-TTS in areas such as **inference speed** and streaming capabilities」—— 即 vLLM-Omni 的加速收益**尚未量化公布 → 待核实**。


### 27. 成本模型

#### 路径 A：自部署（Apache 2.0 开源权重）

- **许可成本**：0
- **成本构成**：GPU 算力 + 存储 + 运维 + 微调/音色工程人力
- **吞吐换算**（基于论文 Table 2，12Hz-0.6B，并发 6）：RTF 0.434 → 单卡理论可支撑约 **2.3 倍实时**的音频产出，即 1 小时 GPU 时间产出约 2.3 小时音频
- **并发 1 时 RTF 0.288** → 单路实时对话场景 GPU 占用率约 29%，**单卡理论可承载 3 路并发实时对话**（未计调度开销）
- ⚠️ 上述为**理想值**，未计请求排队、参考音频编码、I/O、冗余。工程上建议按 **50%~70%** 折算
- **单位 GPU 成本**：官方未提供任何自建成本基准 → **待核实**

##### 显存与硬件门槛（决定可选 GPU 型号）

| 来源 | 0.6B | 1.7B | 性质 |
|---|---|---|---|
| 用户提供 | 约 2 GB | 约 6 GB | 用户提供 |
| ai-rockstars 技术分析 | 4–6 GB | 6–8 GB | 第三方 |
| PyPI qwen3-tts-triton 自测 | — | 4.03–4.33 GB（RTX 5090，bs=1） | 第三方 |
| CSDN 实测 | — | 7.5–7.8 GB（RTX 3090/4090/5090） | 第三方 |

> 差异较大，**必须实测**。保守规划按 1.7B 需 8GB+ 显存。

#### 路径 B：阿里云百炼 DashScope API（官方定价页，一手）

**计费规则：按输入文本字符数计费，输出（音频）不计费。**

**字符计算规则（官方）**：
- 一个汉字（简体/繁体/日文汉字/韩文汉字）计为 **2 个字符**
- 其他字符（英文字母、数字、标点、空格、日文假名、韩文字母）计为 **1 个字符**
- 使用 SSML 时，标签本身不计入，仅统计待合成文本

> 示例：「你好」= 4 字符；「中A文123」= 8 字符；「中文。」= 5 字符

##### 华北 2（北京）

| 服务 | 模型 ID | 输入单价 | 免费额度 |
|---|---|---|---|
| **千问3-TTS-Instruct-Flash** | `qwen3-tts-instruct-flash`（≡ `-2026-01-26`） | **0.8 元/万字符** | 1 万字符 |
| **千问3-TTS-VD**（声音设计） | `qwen3-tts-vd-2026-01-26` | **0.8 元/万字符** | 1 万字符 |
| **千问3-TTS-VC**（声音复刻） | `qwen3-tts-vc-2026-01-22` | **0.8 元/万字符** | 1 万字符 |
| **千问3-TTS-Flash** | `qwen3-tts-flash`（≡ `-2025-11-27`） | **0.8 元/万字符** | 1 万字符 |
| **千问3-TTS-Instruct-Flash-Realtime** | `qwen3-tts-instruct-flash-realtime`（≡ `-2026-01-22`） | **1 元/万字符** | 1 万字符 |
| **千问3-TTS-VD-Realtime** | `qwen3-tts-vd-realtime-2026-01-15` | **1 元/万字符** | 1 万字符 |
| **千问3-TTS-VC-Realtime** | `qwen3-tts-vc-realtime-2026-01-15` | **1 元/万字符** | 1 万字符 |
| **千问3-TTS-Flash-Realtime** | `qwen3-tts-flash-realtime`（≡ `-2025-11-27`） | **1 元/万字符** | 1 万字符 |

> 免费额度有效期：自开通百炼 / 模型发布 / 申请通过之日起 **90 天内**（以较晚者为准）

##### 国际（新加坡）

| 服务 | 输入单价 |
|---|---|
| `qwen3-tts-instruct-flash` / `qwen3-tts-vd-2026-01-26` / `qwen3-tts-vc-2026-01-22` | **0.8 元/万字符** |
| `qwen3-tts-flash`（≡ 2025-11-27） | **0.733924 元/万字符** |
| `qwen3-tts-instruct-flash-realtime` | **1 元/万字符** |
| `qwen3-tts-vd-realtime` / `qwen3-tts-vc-realtime` / `qwen3-tts-flash-realtime` | **0.954101 元/万字符** |

##### 音色创建的一次性费用（全球统一）

| 服务 | 模型 ID | 单价 | 免费额度 |
|---|---|---|---|
| **声音复刻（Voice Enrollment）** | `qwen-voice-enrollment` | **0.01 元/个音色** | 1000 个音色/账号 |
| **声音设计（Voice Design）** | `qwen-voice-design` | **0.2 元/个音色** | 10 个音色/账号 |

> 第三方来源（glbgpt.com，可靠性存疑）给出的国际区美元报价：复刻音色 US$0.01/次、设计音色 US$0.20/个；非实时 Flash US$0.10/万字符、Instruct/VC/VD US$0.115/万字符；实时 Flash/VC US$0.13/万字符。**与官方人民币报价量级一致，可作交叉参考，但以官方为准。**

##### 数量级换算（按北京非实时 0.8 元/万字符）

- 中文 1 万字 ≈ 2 万字符 → **约 1.6 元**
- 英文 1 万词 ≈ 6 万字符 → **约 4.8 元**
- 10 万字中文小说 ≈ 20 万字符 → **约 16 元**（加音色复刻 0.01 元）

#### 成本决策建议

| 场景 | 建议 | 理由 |
|---|---|---|
| 用量小（< 百万字符/月）、无合规限制 | **DashScope API** | 免 GPU、免运维；0.8 元/万字符对中小量极具性价比 |
| 大规模有声书 / 批量配音 | **自部署** | 按 0.8 元/万字符，1 亿字符 = 8,000 元；单卡 GPU 月成本通常低于此 |
| 实时语音助手（QPS 高、延迟敏感） | **自部署** | API 的网络 RTT 不可控；本地首包 97ms 是可控的 |
| 数据/音色不出内网 | **自部署（必须）** | 克隆涉及声纹，合规敏感 |
| 需要定制音色（微调） | **自部署（必须）** | API 侧的 `qwen-voice-enrollment` 是复刻服务，不是开放微调 |
| 需要 VoiceDesign 快速试音色 | **API 优先** | 0.2 元/个设计音色，试错成本极低；满意后再本地化 |

> ⚠️ 阿里云定价随时调整，且页面明示「本文仅展示原价，活动优惠请前往百炼控制台查看」。上线前请以控制台实际价格为准。


### 28. 推理框架支持

#### 1. 官方 Python 包 `qwen-tts`

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install -U qwen-tts
# 开发用源码安装：git clone https://github.com/QwenLM/Qwen3-TTS.git && pip install -e .
# FlashAttention 2（强烈建议）：
pip install -U flash-attn --no-build-isolation   # RAM<96GB 时 MAX_JOBS=4
```

三类生成 API：

| 方法 | 适用模型 | 关键参数 |
|---|---|---|
| `generate_custom_voice()` | 12Hz-1.7B/0.6B-CustomVoice（及微调后的 Base） | `text`, `language`, `speaker`, `instruct`（1.7B 支持） |
| `generate_voice_design()` | 12Hz-1.7B-VoiceDesign | `text`, `language`, `instruct`（必需） |
| `generate_voice_clone()` | 12Hz-1.7B/0.6B-Base | `text`, `language`, `ref_audio`, `ref_text`, `x_vector_only_mode`, `voice_clone_prompt` |

辅助：`create_voice_clone_prompt()`、`get_supported_speakers()`、`get_supported_languages()`

**Tokenizer 独立使用**：`from qwen_tts import Qwen3TTSTokenizer`（`encode` / `decode`）

#### 2. vLLM-Omni（day-0 官方支持，**仅离线推理**）

官方 README 原文：「vLLM officially provides day-0 support for Qwen3-TTS! ... **Now only offline inference is supported. Online serving will be supported later**, and vLLM-Omni will continue to offer support and optimization for Qwen3-TTS in areas such as inference speed and streaming capabilities.」

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni/examples/offline_inference/qwen3_tts

python end2end.py --query-type CustomVoice
python end2end.py --query-type CustomVoice --use-batch-sample
python end2end.py --query-type VoiceDesign
python end2end.py --query-type VoiceDesign --use-batch-sample
python end2end.py --query-type Base --mode-tag icl
```

- 输出默认落在 `output_audio/`
- 需要按硬件配置调整 stage configuration 的内存分配
- **ROCm 依赖**：`pip uninstall onnxruntime && pip install onnxruntime-rocm sox`
- 文档：https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/#installation
- 示例页：https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen3_tts

#### 3. NVIDIA Dynamo（vLLM-Omni worker，第三方集成）

NVIDIA Dynamo 文档把 Qwen3-TTS 列为 **Text-to-Audio** 的 tested models：

- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`（默认模型，预设 speakers）
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`（通过 instructions 描述音色）

启动：`bash examples/backends/vllm/launch/agg_omni_audio.sh`，通过 `/v1/audio/speech` 端点调用（OpenAI TTS 兼容格式）。

**支持的参数**：`input`、`model`、`voice`、`response_format`（wav/mp3/pcm/flac/aac/opus）、`speed`（0.25–4.0）、`task_type`（CustomVoice/VoiceDesign/Base）、`language`、`instructions`、`ref_audio`、`ref_text`、`max_new_tokens`（默认 2048，范围 1–4096）

**限制（Dynamo 文档明示）**：**audio streaming（`stream: true`）与 Base 任务（voice cloning）尚不支持**

#### 4. 本地 Web UI

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign  --ip 0.0.0.0 --port 8000
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base         --ip 0.0.0.0 --port 8000
```

Base 模型需 HTTPS（浏览器麦克风权限）：
```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000 \
  --ssl-certfile cert.pem --ssl-keyfile key.pem --no-ssl-verify
```

#### 5. DashScope API（闭源，云端）

| 服务 | 中国站文档 | 国际站文档 |
|---|---|---|
| CustomVoice 实时 | https://help.aliyun.com/zh/model-studio/qwen-tts-realtime | https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime |
| Voice Clone 实时 | https://help.aliyun.com/zh/model-studio/qwen-tts-voice-cloning | 同路径英文版 |
| Voice Design 实时 | https://help.aliyun.com/zh/model-studio/qwen-tts-voice-design | 同路径英文版 |

#### 6. 社区加速 / 工具（**第三方，非官方**）

| 项目 | 内容 |
|---|---|
| `andimarafioti/faster-qwen3-tts` | CUDA Graphs 实现，RTX 4060/4090 上 **5.8–9.8x 吞吐** |
| `dffdeeq/Qwen3-TTS-streaming` | 流式 fork，约 **6x** 加速 |
| `rekuenkdr/ComfyUI-Qwen3-TTS` | 训练 fork，含 LR 修复、梯度累积、断点续训、音频缓存 |
| `bc-dunia/qwen3-TTS-studio` | UI，暴露全部生成参数（含 `subtalker_temperature` / `subtalker_top_k`） |
| `MonOttawa/Qwen3-TTS-Openai-Fastapi` | OpenAI 兼容 FastAPI 封装，支持 vLLM-Omni 后端切换 |
| `cheeweijie/qwen3-tts-lora-finetuning` | LoRA 微调补丁集（PR #178 / #188 等以 patch 形式提供） |
| PyPI `qwen3-tts-triton` | Triton kernel 优化，RTX 5090 上 Hybrid 方案 **4.2–4.27x** |

#### 7. 框架支持矩阵

| 能力 | `qwen-tts`（Transformers） | vLLM-Omni | NVIDIA Dynamo | DashScope API |
|---|---|---|---|---|
| 离线推理 | ✅ | ✅ | ✅ | ✅ |
| 流式输出 | ✅（单模型双模） | ❌ 仅离线 | ❌（`stream:true` 不支持） | ✅ Realtime |
| 在线 serving | 需自行封装 | ❌ 后续支持 | ✅ `/v1/audio/speech` | ✅ |
| 声音克隆（Base） | ✅ | ✅（`--mode-tag icl`） | ❌ | ✅ VC 服务 |
| VoiceDesign | ✅ | ✅ | ✅ | ✅ VD 服务 |
| 微调 | ✅ `finetuning/` | — | — | ❌ |


### 29. 资源需求

#### 官方明示

| 项 | 要求 |
|---|---|
| Python | 建议 3.12 |
| 精度 | `torch.bfloat16`（所有官方示例与评测） |
| 注意力后端 | `attn_implementation="flash_attention_2"`（**强依赖**，见下） |
| 设备 | `device_map="cuda:0"` |
| 生成参数 | `max_new_tokens=2048`（官方评测配置） |
| ROCm（若用 AMD） | `onnxruntime-rocm` + `sox`（vLLM-Omni 文档） |

#### ⚠️ FlashAttention 2 是强依赖

官方 README 表述为「减少 GPU 显存使用、加速」。第三方技术分析给出更严重的结论：

> 「**Critical dependency: FlashAttention 2** —— 97ms 延迟是理论最优值，实践中需要 FA2。**即便在 RTX 5090 上，缺少该优化时推理速度也仅约 0.3x 实时因子**。」

**该 0.3x 数字来自第三方站点（ai-rockstars.com），非官方 → 待核实**，但方向性结论可信（官方 README 的强度异常高）。

**回退方案**：社区建议在消费级 GPU 上如遇 FA2 兼容问题，改用 `attn_implementation="sdpa"`（速度较慢但稳定）。

#### 显存需求（**官方未提供 → 全部来自第三方，需实测**）

| 来源 | 0.6B | 1.7B | 配置 |
|---|---|---|---|
| 用户提供 | ~2 GB | ~6 GB | — |
| ai-rockstars 技术分析 | 4–6 GB | **6–8 GB**（RTX 3060/4060 级） | — |
| PyPI `qwen3-tts-triton` 自测 | — | **4.03–4.33 GB** | RTX 5090，bf16，batch=1，2 条文本（ko+en） |
| CSDN 实测 | — | **7.5 / 7.6 / 7.8 GB** | RTX 5090 / 4090 / 3090，35 字中文 |
| instavar LoRA 实践 | — | 24GB RTX 3090 Ti 可跑 1.7B **LoRA 微调** | 微调（非推理） |

> 三个来源差异显著（4.0 vs 7.5 GB），**必须以实测为准**。保守规划：推理按 1.7B 需 **8 GB+**、0.6B 需 **6 GB+**；微调需 **24 GB** 量级。

#### 最小可用配置推断（**非官方，为工程规划用的保守估计**）

| 场景 | 建议配置 |
|---|---|
| 0.6B 推理 | 6–8 GB VRAM，NVIDIA Ampere+，CUDA 12.x |
| 1.7B 推理 | 8–12 GB VRAM，NVIDIA Ampere+，CUDA 12.x |
| 1.7B 微调 | 24 GB VRAM（社区验证），官方未给建议 |
| Blackwell（RTX 50 系） | 需 CUDA 12.x + 兼容的 flash-attn 构建；社区有专门 fork 验证可用 |

#### 官方未覆盖（待核实）

| 项 | 状态 |
|---|---|
| 官方显存表格 | ❌ 未提供 |
| 最低 GPU 型号 / 算力要求 | ❌ 未说明 |
| CPU-only 推理 | ❌ 未说明 |
| 官方量化方案（INT8/INT4/FP8/AWQ/GPTQ） | ❌ **未提供量化权重或指南** |
| 端侧 / 移动端部署（NCNN/MNN/CoreML） | ❌ 未提供；0.6B 对手机仍偏重 |
| 磁盘占用 | 可按 bf16 估算：0.6B ≈ 1.2 GB，1.7B ≈ 3.4 GB（不含 tokenizer） |
| 吞吐测试的 GPU 型号 | ❌ 论文只写 "single typical computational resource" |

#### 工程检查清单

1. **验证 FA2 真正生效**（而非静默回退）——这是最大的性能变量
2. **固定随机种子**做可复现的 A/B 对比
3. **长文本分 chunk 时监控显存**（32,768 上下文 + 长序列 KV cache）
4. **首次推理有冷启动失真**（issue #219），生产服务需在启动时 warm-up
5. **显式设置 `max_new_tokens`** 防御 EOS 失败导致的无限生成


### 30. 服务化栈

#### 官方 serving 现状：**目前没有官方在线 serving 方案**

这是一个重要的选型事实，必须在架构设计阶段就考虑：

| 路径 | 在线 serving | 状态 |
|---|---|---|
| `qwen-tts`（Transformers） | 需自行封装 | 官方只提供 Gradio Demo |
| **vLLM-Omni** | ❌ **「Now only offline inference is supported. Online serving will be supported later」** | 官方明示 |
| **NVIDIA Dynamo** | ✅ `/v1/audio/speech`（OpenAI TTS 兼容） | **第三方集成**，且 **不支持 Base 克隆任务与 `stream:true`** |
| **DashScope API** | ✅ | 闭源，按字符计费 |

> **结论**：若要自建 Qwen3-TTS 的在线服务，**目前必须自行基于 `qwen-tts` 包封装**（FastAPI / gRPC），或采用 NVIDIA Dynamo，或等待 vLLM-Omni 的 online serving 支持。

#### 1. 自行封装（当前主流做法）

社区参考实现 `MonOttawa/Qwen3-TTS-Openai-Fastapi`：

```bash
export TTS_BACKEND=vllm_omni                       # 或官方 transformers 后端
export TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
export TTS_WARMUP_ON_START=true                    # 减少首请求延迟
export HOST=0.0.0.0; export PORT=8880; export WORKERS=1
python -m api.main
```

客户端（OpenAI SDK 直连）：
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
resp = client.audio.speech.create(model="qwen3-tts", voice="Vivian", input="Hello!")
```

其文档要点：
- vLLM-Omni 后端提供 day-0 支持，**但当前只支持离线推理（非真正的音频流式）**
- 内置**异步锁**做并发处理（线程安全）
- 支持**启动预热**降低首请求延迟
- **推荐 0.6B 模型**用于生产（速度/质量最佳权衡）

#### 2. NVIDIA Dynamo 的 `/v1/audio/speech`

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello, how are you?","voice":"vivian","language":"English"}' \
  --output output.wav
```

支持参数：`input`、`model`、`voice`、`response_format`（wav/mp3/pcm/flac/aac/opus）、`speed`（0.25–4.0）、`task_type`（CustomVoice/VoiceDesign/Base）、`language`、`instructions`、`ref_audio`、`ref_text`、`max_new_tokens`（默认 2048，1–4096）

**限制**：**audio streaming（`stream: true`）与 Base 任务（voice cloning）尚不支持**。可用的 voices 与 languages 在启动时从模型 `config.json` 动态加载。

#### 3. DashScope Realtime（WebSocket）

```python
from dashscope.audio.qwen_tts_realtime import QwenTtsRealtime, AudioFormat
qwen_tts_realtime = QwenTtsRealtime(model='qwen3-tts-flash-realtime', callback=cb)
qwen_tts_realtime.connect()
qwen_tts_realtime.update_session(voice='Cherry',
    response_format=AudioFormat.PCM_24000HZ_MONO_16BIT, mode='server_commit')
qwen_tts_realtime.append_text("...")
```

**第三方来源（glbgpt.com，待核实）**补充的限制：
- Realtime WebSocket **输入上限 512 tokens**
- 完整音频 URL **24 小时后过期**，生产系统必须转存到持久存储
- 非实时（HTTP）与实时（WebSocket）的 model ID 与日期快照不同，需从当前文档复制而非类推

#### 4. 服务化工程要点

| 项 | 建议 |
|---|---|
| **预热** | 必须做。冷启动解码器会产出失真前几帧（issue #219）；预热还能消除首次 CUDA 初始化开销 |
| **音色 prompt 缓存** | 对同一音色的批量请求，缓存 `create_voice_clone_prompt()` 结果 |
| **并发** | 官方未提供 QPS 数据；按论文 RTF 0.288（并发 1，0.6B）估算单路占用约 29% GPU |
| **超时与截断** | 显式 `max_new_tokens`（默认 2048），防御 ~0.5% 的 EOS 失败 |
| **固定种子** | 每个请求固定 seed 以保证同文本同音色可复现 |
| **显存** | vLLM-Omni 需按硬件调整 stage configuration 的内存分配 |
| **ROCm** | 需 `onnxruntime-rocm` + `sox` |

#### 5. 缺失能力（待核实）

- 官方 Kubernetes / Helm 部署清单
- 官方 gRPC / WebSocket 流式协议
- 官方 Triton / TensorRT-LLM 后端
- 官方请求队列 / 优先级 / 限流组件
- 官方 Prometheus metrics exporter


### 31. 硬件生态

#### 已验证 / 官方支持的硬件路径

| 层次 | 支持情况 | 说明 |
|---|---|---|
| **NVIDIA GPU（CUDA）** | ✅ 主路径 | 官方所有示例、vLLM-Omni、NVIDIA Dynamo 均基于此 |
| **FlashAttention 2 兼容硬件** | ✅ **强依赖** | NVIDIA Ampere 及以后（A100 / RTX 30xx+）。**无 FA2 时性能崩塌**（第三方称 RTX 5090 上仅 0.3x 实时，待核实） |
| **Blackwell（RTX 50 系）** | ✅ 社区验证 | 需 CUDA 12.x + 兼容 flash-attn 构建；有专门 fork 验证可用（如 `hiroki-abe-58/Qwen3-TTS-JP` 在 RTX 5090 上开发与测试） |
| **vLLM-Omni** | ✅ day-0 | 内部 vLLM V0 backend（论文基准）；`torch.compile` + CUDA Graph 用于 tokenizer 解码阶段 |
| **NVIDIA Dynamo** | ✅ | 官方集成，提供 `/v1/audio/speech` |
| **AMD GPU（ROCm）** | ⚠️ 部分 | vLLM-Omni 文档列出 ROCm 依赖 `onnxruntime-rocm` + `sox`，说明有 ROCm 路径；但**无官方性能数据 → 待核实** |
| **ModelScope / HuggingFace** | ✅ | 中国大陆推荐 ModelScope |

#### 官方未覆盖 / 待核实

| 硬件 / 后端 | 状态 |
|---|---|
| **Apple Silicon（MPS）** | 官方未提及 → 待核实 |
| **国产 NPU（昇腾 / 寒武纪等）** | 官方未提及 → 待核实 |
| **CPU-only 推理** | 官方未说明；按 1.7B 规模与自回归解码特性，CPU 实时性极差 → 不建议 |
| **端侧 / 移动端**（NCNN / MNN / CoreML / TFLite） | **官方未提供任何方案**；0.6B 对手机仍偏重 |
| **官方量化**（INT8 / INT4 / FP8 / AWQ / GPTQ） | **未提供量化权重或指南** |

#### 社区硬件优化生态（**第三方，非官方**）

| 项目 | 加速手段 | 报告收益 |
|---|---|---|
| `andimarafioti/faster-qwen3-tts` | CUDA Graphs | RTX 4060/4090 上 **5.8–9.8x 吞吐** |
| `dffdeeq/Qwen3-TTS-streaming` | 流式优化 | 约 **6x** |
| PyPI `qwen3-tts-triton` | Triton kernel（RMSNorm 5.51x、M-RoPE 9.87x、Fused Norm+Residual 4.49x）+ Faster 组合 | RTX 5090 上 **Hybrid 4.20–4.27x**（RTF 0.88x → 4.2x） |
| `rekuenkdr/ComfyUI-Qwen3-TTS` | 训练 fork（LR 修复、梯度累积、断点续训、音频缓存） | 训练可用性改进 |

> `qwen3-tts-triton` 的自测还报告了质量回归验证（UTMOS / CER / Speaker Sim），其中 `base+tq` 与 `triton+tq` 的 CER delta 超过阈值被判 FAIL —— **说明激进的 kernel 优化可能损害质量，采用前必须做质量回归**。

#### 与 Qwen3-ASR 的硬件依赖对比

| | Qwen3-ASR | Qwen3-TTS |
|---|---|---|
| FlashAttention 2 | 推荐 | **强依赖**（无 FA2 时可能慢到不可用） |
| vLLM 路径 | vLLM 主线（day-0，含 online serving） | **vLLM-Omni（仅离线）** |
| 显存门槛 | 较低 | 较高（1.7B 需 8GB+ 保守估计） |
| 实时性要求 | 吞吐优先 | **延迟优先**（首包 97ms 是核心卖点） |
| AMD ROCm | 无说明 | vLLM-Omni 有 ROCm 依赖说明 |

#### 工程建议

1. **必须 NVIDIA Ampere+ 且装好 FA2**，这是唯一的官方完整验证路径。
2. **生产服务务必做 warm-up**（冷启动解码失真 + CUDA 初始化）。
3. ** Blackwell 用户**关注 flash-attn 的 CUDA 12.x 构建，或采用社区 fork。
4. **关注 vLLM-Omni 的 online serving 进展** —— 这是官方 serving 能力的关键缺口。
5. **采用社区加速方案前做质量回归**（UTMOS / CER / SIM）。


### 32. 集成范式

#### 四种集成形态

| 形态 | 适用场景 | 首包延迟 | 运维成本 | 备注 |
|---|---|---|---|---|
| **库内嵌**（`qwen-tts`） | 离线批处理、桌面工具、研究 | 97ms（本地） | 低 | 唯一支持完整能力（含流式）的官方路径 |
| **自建服务**（FastAPI + `qwen-tts`） | 生产在线服务 | 97ms + 网络 | 中 | **当前唯一能获得在线服务的自建方式**（vLLM-Omni 无 online serving） |
| **vLLM-Omni 离线批** | 大规模离线批量合成 | — | 中 | 吞吐优化，无流式 |
| **NVIDIA Dynamo** | 已有 Dynamo 栈的团队 | — | 中 | 有 `/v1/audio/speech`，但**不支持 Base 克隆与 stream** |
| **DashScope API** | 无 GPU、快速验证 | 网络 RTT | 极低 | 闭源，按字符计费 |

#### 1. 自定义音色生产流水线（最典型的工程形态）

```
① 采集目标说话人 20–60 分钟音频（24kHz mono）
       ↓
② ASR 转写（推荐 Qwen3-ASR）+ 人工校对
       ↓
③ 数据清洗（末尾补 1s 静音、剥离标注标签、统一 ref_audio）
       ↓
④ prepare_data.py 抽取 audio_codes
       ↓
⑤ sft_12hz.py 微调（lr=2e-6，约 10 epoch，保存每轮 checkpoint）
       ↓
⑥ 逐 checkpoint 听觉筛选（语速、音色相似度、稳定性）
       ↓
⑦ 部署：generate_custom_voice(speaker="<注册的音色名>")
```

**关键**：微调后通过 `speaker` 参数调用（而非 `generate_voice_clone`），音色名由 `--speaker_name` 注册。

#### 2. 零样本克隆 + 批量生成（无微调）

```python
prompt_items = model.create_voice_clone_prompt(ref_audio=ra, ref_text=rt, x_vector_only_mode=False)
for batch in batches:
    wavs, sr = model.generate_voice_clone(
        text=batch, language=["Chinese"]*len(batch), voice_clone_prompt=prompt_items)
```

> **复用 prompt 是性能与一致性的双重关键**：省去重复编码，并保证跨批次音色一致。

#### 3. VoiceDesign → Clone 工作流（官方推荐）

用自然语言设计出满意音色后，固化成可复用的 clone prompt：
```
VoiceDesign 生成参考短音频 → create_voice_clone_prompt → 批量 generate_voice_clone
```
适合游戏 NPC、虚拟角色等需要"原创音色"的场景。

#### 4. 全栈语音助手（与 Qwen3-ASR 组合）

```
麦克风/音频流 → Qwen3-ASR（vLLM 流式，TTFT 92ms）→ 文本
      → 通用 LLM（Qwen3 系列）→ 回复文本
      → Qwen3-TTS（Dual-Track 流式，首包 97ms）→ 语音输出
```

**端到端理论最低延迟 ≈ 92ms（ASR TTFT）+ LLM TTFT + 97ms（TTS 首包）**

同栈组合的优势：
- 同一 Qwen 生态（tokenizer 体系、vLLM 生态、Apache 2.0）
- **官方内部已形成评测闭环**：Qwen3-TTS 论文用 Qwen3-ASR 做长语音转写评测，说明两者已在官方生产链路中配合验证
- Qwen3-ASR 的 vLLM 生态更成熟（有 online serving），可作为链路中的稳定环节

#### 5. 长文本合成（有声书 / 配音）

必须自行分 chunk，且：
1. 在**语义边界**分段，保留上下文
2. 每个 chunk **固定随机种子**
3. **复用同一次提取的 speaker embedding**
4. 设置 `repetition_penalty=1.05`
5. 对数字做归一化，并在**每个连接点做听感检查**

> 社区提醒：「有效的片段在跨调用时若语速或能量发生变化，听起来仍会像分开录制的。**编辑时间与重复调用往往才是真正的生产成本。**」

#### 6. OpenAI 兼容接入

若自建 FastAPI 封装（`MonOttawa/Qwen3-TTS-Openai-Fastapi` 等），可直接用标准 OpenAI SDK：
```python
client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")
resp = client.audio.speech.create(model="qwen3-tts", voice="Vivian", input="...")
```
好处：现有基于 OpenAI TTS 的应用可低成本迁移到自建 Qwen3-TTS。

#### 7. 集成注意事项

1. **输出采样率推断为 24kHz**（API 侧为 `PCM_24000HZ_MONO_16BIT`）——下游播放器/转码需匹配
2. **Base 模型不支持 `instruct`**，需要风格控制必须换 CustomVoice(1.7B) 或 VoiceDesign
3. **每个 speaker 尽量用其母语**合成
4. **API 的音频 URL 24 小时过期**（第三方来源，待核实），生产必须转存
5. **3 秒克隆是高风险能力**，需在产品层落实授权与标识


### 33. 技术栈组件

| 层次 | 组件 | 说明 |
|---|---|---|
| **模型权重** | 12Hz-1.7B/0.6B × {Base, CustomVoice, VoiceDesign} | Apache 2.0 |
| **语音 Codec** | `Qwen3-TTS-Tokenizer-12Hz`（16 层 RVQ / codebook 2048 / 12.5Hz / 全因果） | Apache 2.0，可独立使用 |
| —（未发布） | `Qwen-TTS-Tokenizer-25Hz`（单码本 / codebook 32768 / 25Hz / Qwen2-Audio  backbone） | 权重未开放 |
| **文本前端** | 标准 Qwen tokenizer + ChatML 格式化 | |
| **LM 骨干** | Qwen3-1.7B / Qwen3-0.6B | |
| **说话人编码器** | 可学习 speaker encoder，**与 backbone 联合训练** | |
| **残差码本预测** | **MTP（Multi-Token Prediction）模块**（12Hz） | |
| **波形生成（12Hz）** | **轻量因果 ConvNet**（tokenizer 内置，纯左上下文） | 无 mel / 无 vocoder |
| **波形生成（25Hz，未发布）** | 块式 **DiT + Flow Matching** → mel → **改进版 BigVGAN** | |
| **流式机制** | **Dual-Track**（文本/声学 token 沿 channel 轴拼接） | 单模型同时支持流式与非流式 |
| **推理框架** | `qwen-tts`（Transformers 后端） | 官方 Python 包 |
| **推理引擎** | **vLLM-Omni**（day-0，**仅离线**）；内部 vLLM V0 backend（论文基准） | |
| **加速** | FlashAttention 2（**强依赖**）、torch.compile、CUDA Graph（tokenizer 解码阶段） | |
| **服务化** | 自建 FastAPI（社区方案）/ **NVIDIA Dynamo** `/v1/audio/speech` / **DashScope API** | 官方无 online serving |
| **微调** | `finetuning/prepare_data.py` + `sft_12hz.py` | 单说话人全参数 SFT |
| **Web UI** | `qwen-tts-demo`（Gradio） | |
| **云 API** | DashScope：`qwen3-tts-flash` / `-vc` / `-vd` / `-instruct-flash`（含 `-realtime`） | 按字符计费 |
| **评测指标** | WER/CER（内容一致性）、SIM（WavLM）、PESQ/STOI/UTMOS（音质）、APS/DSD/RP（InstructTTSEval）、RTF、首包延迟 | |
| **评测搭档** | **Qwen3-ASR**（长语音 WER 评测的转写器） | 官方内部闭环 |
| **参考开源工作** | Mimi（codec 架构）、WavLM（语义教师）、BigVGAN（vocoder）、StreamFlow（块注意力）、Flow Matching、DPO、GSPO | |


### 34. 评测工具链

#### 官方评测协议（arXiv:2601.15621 §4）

评测分两大类：**语音 tokenizer** 与 **语音生成**。

##### A. 语音 Tokenizer 评测

| 对象 | 数据 | 指标 |
|---|---|---|
| **25Hz** | CommonVoice 与 Fleurs 的英/中子集，以 **ASR 任务**做代理评测 | WER ↓（对比 S3 Tokenizer VQ/FSQ 系列） |
| **12Hz** | **LibriSpeech test-clean，2,620 条** | PESQ_WB、PESQ_NB、STOI、UTMOS、**SIM（WavLM-based speaker verification）**；对比 SpeechTokenizer、X-codec、X-codec 2、XY-Tokenizer、Mimi、FireredTTS 2 |

##### B. 语音生成评测（六个方向）

| 方向 | 基准 | 指标 |
|---|---|---|
| 零样本语音生成 | **Seed-TTS test set** | WER ↓（内容一致性） |
| 多语种语音生成 | 多语种测试集（Zhang et al. 2025a，即 MiniMax-Speech 的测试集） | WER ↓ + **SIM ↑** |
| 跨语种语音生成 | **CV3-Eval** | Mixed Error Rate ↓（英用 WER，其余用 CER） |
| 可控语音生成 | **InstructTTSEval** | **APS / DSD / RP ↑** |
| 目标说话人生成 | 多语种测试集（在 **Aiden Voice** 上微调） | WER ↓ |
| 长语音生成 | **内部数据集：中英各 100 篇，200–2000 词** | WER ↓（**用 Qwen3-ASR 转写**，按 Seed-TTS Eval 方法） |

##### 商业基线

- **MiniMax-Speech**（多语种、零样本）
- **ElevenLabs Multilingual v2**（多语种）
- **GPT-4o-mini-tts**（InstructTTSEval Target Speaker）
- **GPT-4o-Audio-Preview（Ballad Voice）**（目标说话人微调对比）
- **Hume**（InstructTTSEval Voice Design EN）

##### 开源基线

Seed-TTS、MaskGCT、E2 TTS、F5-TTS、Spark TTS、Llasa-8B、KALL-E、FireRedTTS 2、CosyVoice 3 / 2、VibeVoice、Higgs-Audio-v2、VoxCPM、Mimo-Audio-7B-Instruct、VoiceSculptor、VoxInstruct、Parler-tts、PromptTTS、PromptStyle

#### 统一推理配置（官方 README）

`dtype=bfloat16`、`max_new_tokens=2048`；**Seed-TTS test set 与 InstructTTS 用 `language="auto"`**，其余显式指定语言。

#### 效率评测方法（论文 §3.4，值得复用）

- **首包延迟 = LM TTFP + Tokenizer Decode TPP**
- **LM TPP**：稳态下 LM 生成一个 packet 的 token 所需时间
- **Tokenizer Decode TPP**：tokenizer 每 packet 的解码时间
- **RTF**：Real-Time Factor
- 优化手段：**torch.compile + CUDA Graph 应用于 tokenizer 解码阶段**
- 分包定义：12Hz = 4 tokens/packet（320ms）；25Hz = 8 tokens/chunk（320ms mel ≈ 190ms 音频）

#### 官方未提供的评测工具（待核实）

- **无开源评测脚本 / benchmark harness**
- **无主观 MOS / CMOS 听测数据**（论文全部使用客观代理指标）
- **无长语音测试集**（内部数据集未发布）
- **无 SIM 在长语音下的衰减测试**

#### 自建评测建议

##### 必测三项

1. **内容一致性（WER/CER）**：用 Qwen3-ASR 转写合成音频，与输入文本比对（官方验证过的做法）
2. **音色相似度（SIM）**：WavLM-based speaker verification 的余弦相似度
3. **延迟**：首包延迟 + RTF（按你的并发目标测，不要只看并发 1）

##### 补测项（官方未覆盖但工程必需）

| 项 | 方法 |
|---|---|
| **长文本音色漂移** | 分 10 段生成，逐段算 SIM，观察衰减曲线 |
| **数字/符号/中英混排** | 构造含数字、日期、货币、单位、英文单词的中文文本 |
| **情感指令有效性** | 逐音色做听测（社区建议：先锁定中性音色，再在短/中/多 chunk 三类文本上测情感） |
| **冷启动失真** | 新进程首次推理的前几帧单独听测 |
| **EOS 稳定性** | 大批量生成中统计无限生成/截断的发生率（社区报告约 0.5%） |
| **并发下的延迟退化** | 在目标并发下测首包延迟（论文只测到并发 6，外推需实测） |


### 35. 已知坑与限制

#### 一、官方微调脚本的历史 Bug（**已由官方修复，但需确认本地版本**）

##### Bug 1（最严重）：double label-shift 导致语速逐轮加快

| 项 | 内容 |
|---|---|
| **现象** | 每多训练一个 epoch，生成语音语速越快，直到无法听清 |
| **根因** | `sft_12hz.py` 在 **L114–125 手动做了一次 label shift**，然后把 labels 传给 HF 的 `ForCausalLMLoss`，**后者内部又 shift 了一次**，造成训练-推理不一致 |
| **影响范围** | **所有使用官方脚本的微调运行** |
| **证据** | GitHub issue #179（rekuenkdr，2026-02-06）；社区复现：8 GPU + bs16 + ga16 + lr1e-6 + 1.7B + 5k 样本 + epoch 4 |
| **量化影响** | 修复前：pre-SFT loss 22 → post-SFT loss 13，**音频 very fast**；修复后：8.3 → 7.8，**音频 correct** |
| **修复** | **PR #178**（提供代码 diff）；**官方 `finetuning/sft_12hz.py` 于 2026-03-17 提交「fix finetuning bug」** |
| **修复要点** | ① `sft_12hz.py` 不再手动 shift，直接传 `input_embeds` / `labels`；② `modeling_qwen3_tts.py` L1243 把 `self.loss_function(...)` 换成 `F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)`；③ sub-talker loss 同样处理 |
| **行动项** | **确认你的本地副本 ≥ commit 680d4e9；如未包含 PR #178 的修改，必须手动打补丁** |

##### Bug 2：缺少 `text_projection` 调用

| 项 | 内容 |
|---|---|
| **现象** | 0.6B 上**直接崩溃**；1.7B 上**静默使用错误的 embedding**（不报错但结果错误） |
| **位置** | `sft_12hz.py` line 93 |
| **修复** | **PR #188**；**commit 680d4e9 已在官方仓库合入** |
| **危险等级** | 高（1.7B 上的静默错误极难发现） |

##### Bug 3：默认学习率过高

| 项 | 内容 |
|---|---|
| **官方脚本默认值** | **2e-5**（一键脚本示例）；文档步骤示例又写 **2e-6** —— **README 内部不一致** |
| **现象** | lr=2e-5 会导致**纯噪声输出**与**无限生成（模型遗忘 EOS token）** |
| **社区共识值** | **2e-6** |
| **来源** | GitHub issue #39；instavar LoRA 文章与配套仓库 |

##### Bug 4：音频采样率必须为 24 kHz

- 非 24kHz 音频会在**训练深处崩溃，且没有早期告警**
- 对策：先跑 `scripts/resample_to_24k.sh` 把所有 WAV 重采样到 **24 kHz mono**
- 修复 PR：#233

---

#### 二、推理侧的已知问题（**社区报告，非官方文档**）

| # | 问题 | 现象 | 对策 | 来源 |
|---|---|---|---|---|
| 5 | **EOS 失败（无限生成）** | 约 **0.5%** 的 base 模型推理不停止，挂起 | 显式设置 `max_new_tokens`；必要时 `eos_token_id=[2150, 2157, 151670, 151673, 151645, 151643]` | GitHub 社区 |
| 6 | **冷启动解码失真** | 新进程首次推理时 tokenizer decoder 无 warm-up 上下文，前几帧音频重建错误。**以数字开头的短语音最明显** | 前置静音 codec token 做 warm-up，再从输出裁掉对应样本；生产服务启动时预热 | issue #219 |
| 7 | **跨 chunk 音色漂移** | 长文本分 chunk 生成时音色逐段变化 | 每 chunk 固定随机种子；复用同一次提取的 speaker embedding；加大 chunk 尺寸；`repetition_penalty=1.05` | 社区 |
| 8 | **音频在约 24 秒处被截断** | `chunked_decode` 过度裁剪样本 | 关注 PR #259（**尚未合并**） | PR #259 |
| 9 | **LoRA scale 1.0 过度操控** | 声音发假、不自然 | 用 **0.25–0.35**（建议起点 0.3）；在 0.2/0.3/0.35/0.5 上各生成 5 条做筛选 | instavar |
| 10 | **小验证集上 eval 崩溃** | `RuntimeError: zero-dimensional tensor` | 对空 loss tensor 加保护 | instavar |
| 11 | **推理段错误（segfault）** | checkpoint 扫描过程中崩溃 | 批量推理时做防御性处理 | instavar |
| 12 | **微调后表现力平坦化** | 情感/韵律表现力下降 | 如需保留表现力，考虑用 ICL 克隆而非微调 | 社区观察（待核实） |
| 13 | **口音难以迁移** | 预训练的口音偏见很强 | 无解，需接受 | 社区观察（待核实） |

---

#### 三、能力/部署限制（官方明示）

| # | 限制 |
|---|---|
| 14 | **vLLM-Omni 仅支持离线推理**，online serving 后续支持 |
| 15 | **NVIDIA Dynamo 不支持 `stream:true` 与 Base（voice cloning）任务** |
| 16 | **Base 模型不支持指令控制**（风格/情感） |
| 17 | **0.6B-CustomVoice 不支持指令控制**（仅 1.7B 版本支持） |
| 18 | **`x_vector_only_mode=True` 会降低克隆质量** |
| 19 | **微调仅支持单说话人**；多说话人与高级微调特性待后续版本 |
| 20 | **25Hz 系列权重尚未发布**（含长语音最优的 25Hz-1.7B-CustomVoice 与具备 VoiceEditing 能力的变体） |
| 21 | **FlashAttention 2 仅在 float16/bfloat16 加载时可用**，需硬件兼容 |
| 22 | **FlashAttention 2 是强依赖**：无 FA2 时性能崩塌（第三方称 RTX 5090 上仅 0.3x 实时，**待核实**） |
| 23 | Base 模型 Web Demo 需 HTTPS（浏览器麦克风权限） |

---

#### 四、官方承认的性能短板

| # | 短板 | 数据 |
|---|---|---|
| 24 | **长语音上 12Hz 弱于 25Hz** | long-zh 2.356 vs 1.517；long-en 2.812 vs 1.225。**而 25Hz 权重未发布** |
| 25 | **日语/俄语 WER 明显偏高** | 日 3.823–6.404；俄 3.212–5.957（对比中英 0.78–1.15） |
| 26 | **zh→ja / en→ja 上 CosyVoice3 更优** | 7.08 vs 8.40；6.80 vs 7.21 |
| 27 | **中文零样本克隆 WER 略逊 CosyVoice 3** | 0.77 vs 0.71 |
| 28 | **Target Speaker 控制上 Gemini 系列仍是上界** | Gemini-flash APS-ZH 88.2 vs Qwen3-TTS 83.0 |
| 29 | **语种覆盖仅 10 种** | 论文 Future Work 明确要扩展 |

---

#### 五、论文/文档未披露造成的工程不确定性（待核实）

- 输出采样率（推断 24 kHz，未明确写明）
- 单次请求的最大文本长度
- 官方微调所需的最少数据量
- 微调后是否保留 9 个预设音色、是否仍支持流式
- 官方微调脚本是否支持多卡
- 吞吐（QPS）数据
- vLLM-Omni 的实际加速收益


### 36. 社区与文档

#### 一手权威来源

| 类型 | 链接 |
|---|---|
| **技术报告（论文）** | arXiv:2601.15621《Qwen3-TTS Technical Report》 https://arxiv.org/abs/2601.15621 |
| **论文 HTML 全文** | https://arxiv.org/html/2601.15621v1 |
| **GitHub 仓库** | https://github.com/QwenLM/Qwen3-TTS |
| **微调文档** | https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md |
| **HuggingFace Collection** | https://huggingface.co/collections/Qwen/qwen3-tts |
| **ModelScope Collection** | https://modelscope.cn/collections/Qwen/Qwen3-TTS |
| **官方 Demo（HF Space）** | https://huggingface.co/spaces/Qwen/Qwen3-TTS |
| **官方 Demo（ModelScope）** | https://modelscope.cn/studios/Qwen/Qwen3-TTS |
| **官方 Blog** | https://qwen.ai/blog?id=qwen3tts-0115 |
| **vLLM-Omni 文档** | https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/#installation |
| **vLLM-Omni Qwen3-TTS 示例** | https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen3_tts |
| **DashScope API 文档（中国站）** | https://help.aliyun.com/zh/model-studio/qwen-tts-realtime （另有 `-voice-cloning` / `-voice-design`） |
| **DashScope API 文档（国际站）** | https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime |
| **定价页** | https://help.aliyun.com/zh/model-studio/billing-for-model-studio |

#### 权重下载

```bash
# ModelScope（中国大陆推荐）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz --local_dir ./Qwen3-TTS-Tokenizer-12Hz
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base
# ... 其余同理（CustomVoice / VoiceDesign / 0.6B 系列）

# Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
```

#### 关键社区资源（**第三方，非官方，需自行评估**）

| 资源 | 内容 |
|---|---|
| **GitHub issue #179** | double label-shift bug 的完整根因分析与修复验证（含 loss 对比数据） https://github.com/QwenLM/Qwen3-TTS/issues/179 |
| **PR #178 / #188 / #233 / #259** | 微调脚本的各类修复 |
| **instavar.com 博客** | 《LoRA Finetuning Qwen3-TTS for Custom Voices》—— 完整的失败模式分类、验证过的推理参数、checkpoint 选择流程 |
| **`cheeweijie/qwen3-tts-lora-finetuning`** | LoRA 微调补丁仓库（patch 形式，不改上游） |
| **NVIDIA Dynamo 文档** | https://docs.dynamo.nvidia.com/dynamo/dev/diffusion/text-to-audio （Qwen3-TTS 的 `/v1/audio/speech` 集成） |
| **PyPI `qwen3-tts-triton`** | Triton kernel 优化 **附带质量回归验证**（UTMOS/CER/SIM），是「加速方案必须做质量回归」的示范 |
| **社区加速 fork** | `andimarafioti/faster-qwen3-tts`（CUDA Graphs，5.8–9.8x）、`dffdeeq/Qwen3-TTS-streaming`（~6x） |
| **训练 fork** | `rekuenkdr/ComfyUI-Qwen3-TTS`（LR 修复 + 梯度累积 + 断点续训 + 音频缓存） |
| **UI 工具** | `bc-dunia/qwen3-TTS-studio`（暴露 `subtalker_temperature` / `subtalker_top_k` 等全部参数） |
| **RTX 5090 fork** | `hiroki-abe-58/Qwen3-TTS-JP`（集成 faster-whisper 自动转写参考音频） |

#### 关联论文

| 论文 | arXiv | 关系 |
|---|---|---|
| **Qwen3-ASR Technical Report** | 2601.21337 | 长语音评测的转写器；同栈组合 |
| **Qwen3-Omni Technical Report** | 2509.17765 | 同系列多模态基座 |
| **GSPO** | 2507.18071 | RL 算法 |
| **MiniMax-Speech** | 2505.07916 | 主要对比基线 |
| **CosyVoice 3** | 2505.17589 | 主要对比基线 |
| **Mimi / Moshi** | 2410.00037 | 12Hz tokenizer 架构参考 |
| **StreamFlow** | 2506.23986 | 25Hz 块式 guided attention mask |
| **InstructTTSEval** | 2506.16381 | 可控性评测基准 |
| **VibeVoice** | 2508.19205 | 长语音对比基线 |
| **VoxCPM** | 2509.24650 | 长语音对比基线 |
| **Flow Matching** | ICLR 2023 | 25Hz DiT 训练目标 |
| **BigVGAN** | ICLR 2023 | 25Hz 声码器 |
| **WavLM** | IEEE JSTSP | 12Hz 语义教师 / SIM 度量模型 |
| **DPO** | NeurIPS 2023 | 后训练 P1 阶段算法 |

#### 引用格式（官方 README 提供）

```bibtex
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and
          Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and
          Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```

#### 生态活跃度

**未获取到 GitHub star 数、issue 数、commit 频率、HuggingFace 下载量等量化指标 → 待核实。**

定性观察：社区在**微调修复**（多个 PR/issue）与**推理加速**（CUDA Graphs / Triton / 流式 fork）两条线上非常活跃，说明生产落地需求强烈，但也侧面反映官方在 serving 与微调稳定性上仍有明显缺口。


### 37. 适用场景与选型建议

#### 强匹配场景（推荐优先选用）

| 场景 | 理由 | 推荐变体 |
|---|---|---|
| **实时语音助手 / 对话式 AI** | **97ms 首包延迟**（0.6B）是开源 TTS 中最强；单模型流式/非流式双模 | **12Hz-0.6B-Base** 或 `-CustomVoice` |
| **3 秒零样本声音克隆** | Seed-TTS test-en WER **1.24（SOTA）**；10 语种 SIM 全部超越 MiniMax 与 ElevenLabs | **12Hz-1.7B-Base** |
| **中文 / 中英混排配音** | 中文 WER 0.777（vs MiniMax 2.252、ElevenLabs **16.026**） | 1.7B |
| **跨语种配音（中→英/韩等）** | zh→ko 4.82（CosyVoice3 14.4，-66%）；9/12 方向 SOTA | **12Hz-1.7B-Base** |
| **原创音色设计（游戏 NPC / 虚拟角色）** | VoiceDesign 在 InstructTTSEval 中文 APS 85.2 / DSD 81.1 / RP 65.1 全项 SOTA（含 thinking pattern） | **12Hz-1.7B-VoiceDesign** |
| **带情感/风格指令的配音** | CustomVoice 显著优于 GPT-4o-mini-tts（中文 APS +28%） | **12Hz-1.7B-CustomVoice**（0.6B 不支持指令） |
| **固定音色批量生产（有声书/课件/客服播报）** | 官方说话人微调路径成熟；单语微调即可跨 10 语；10 语中 7 语优于 GPT-4o-Audio-Preview | **12Hz-1.7B/0.6B-Base + 微调** |
| **高并发合成服务** | 并发 6 时 RTF 0.434（0.6B）/ 0.463（1.7B），远优于 25Hz 的 0.709/0.725 | **12Hz 系列** |
| **数据合规 / 私有机房** | Apache 2.0，可完全内网部署，声纹不出内网 | 任意变体 |
| **与 Qwen3-ASR 组全栈** | 同生态、同 vLLM 栈、官方内部已形成评测闭环 | ASR + TTS 组合 |

#### 需要谨慎评估的场景

| 场景 | 风险 | 建议 |
|---|---|---|
| **超长语音（>10 分钟）有声书** | **最优解是 25Hz-1.7B（long-zh 1.517），但权重未发布**；开源 12Hz 为 2.356/2.812 | 若 12Hz 的长时稳定性不达标，需等待 25Hz 或改用 VibeVoice（long-en 1.780 尚可，但 long-zh 22.619 崩溃） |
| **自建在线 serving** | **vLLM-Omni 仅离线**；官方无 online serving；Dynamo 不支持 Base 克隆与 stream | 必须自建 FastAPI 封装，或等 vLLM-Omni 更新 |
| **日语 / 俄语为主** | WER 3.2–6.4，显著差于中英 | 务必实测；可对比 ElevenLabs（日语 3.519） |
| **需要 VoiceEditing** | 仅论文中 25Hz-1.7B-VoiceEditing 具备，**权重未发布** | 暂无开源方案 |
| **端侧 / 移动端** | 无官方量化与端侧方案；0.6B 仍偏重 | 需自行量化/蒸馏 |
| **无 GPU 环境** | 无 CPU 优化路径 | 用 DashScope API |
| **需要多说话人微调** | 官方仅支持单说话人 | 需等上游或自行改造 |
| **对音色一致性要求极高的长文本** | 分 chunk 会漂移（官方无自动分段） | 自行实现种子固定 + speaker embedding 复用 + 边界听测 |

#### 不推荐场景

- **纯离线、非实时、超长语音**：25Hz 才是最优（长语音 WER 更低），但拿不到权重；若可接受 12Hz 的 2.356/2.812 则可用。
- **只需单一固定音色且无合规要求、量小**：直接用 DashScope `qwen-voice-enrollment`（0.01 元/音色）+ 合成（0.8 元/万字符），成本远低于自建 GPU。
- **需要生产级开箱服务**：当前官方 serving 能力缺口明显（vLLM-Omni 仅离线、无官方 online serving）。

#### 选型速查

```
需要自定义音色？ ── 否 ──> 12Hz-1.7B/0.6B-CustomVoice（9 预设音色）
      │
      是
      ↓
有训练数据（≥数十分钟）？ ── 是 ──> 12Hz-1.7B-Base + 官方微调（lr=2e-6）
      │                              └─ 注意：确认脚本 ≥ commit 680d4e9
      否（只有 3 秒参考音频）
      ↓
12Hz-1.7B-Base + generate_voice_clone（ICL，需 ref_text）
      │
需要"原创"音色（非克隆）？ ── 是 ──> 12Hz-1.7B-VoiceDesign
      │
需要指令控制情感/风格？ ── 是 ──> 必须用 1.7B（0.6B 与 Base 不支持 instruct）
      │
延迟敏感（实时对话）？ ── 是 ──> 0.6B（首包 97ms，RTF 0.434@并发6）
      │
超长语音（>10 分钟）？ ── 是 ──> 需 25Hz（未发布）；12Hz 需充分验证
```

#### 与 Qwen3-ASR 的组合价值

Qwen3-ASR + Qwen3-TTS 构成**完整的开源语音交互栈**：

| 环节 | 模型 | 关键指标 |
|---|---|---|
| 语音 → 文本 | Qwen3-ASR-0.6B | 并发 128 达 2000x 吞吐；TTFT 92ms |
| 文本 → 文本 | Qwen3 系列 LLM | — |
| 文本 → 语音 | Qwen3-TTS-12Hz-0.6B | 首包 97ms；RTF 0.434（并发 6） |

同栈优势：Apache 2.0、同一 Qwen tokenizer 体系、同一 vLLM 生态、**官方内部已用 Qwen3-ASR 评测 Qwen3-TTS**（论文 §4.2.6 明确说明选它是因为"high accuracy in long-form recognition"）。这是目前一致性最高、商业风险最低的开源语音栈选型。


### 38. 参考来源

#### 一手权威来源

1. **Qwen3-TTS Technical Report**（主来源，本文件中标注 arXiv:2601.15621 的所有数据与结论）
   - 摘要页：https://arxiv.org/abs/2601.15621
   - HTML 全文 v1：https://arxiv.org/html/2601.15621v1

2. **GitHub 官方仓库**：https://github.com/QwenLM/Qwen3-TTS
   - 微调文档：https://github.com/QwenLM/Qwen3-TTS/blob/main/finetuning/README.md
   - 已知问题（double label-shift）：https://github.com/QwenLM/Qwen3-TTS/issues/179
   - 修复 PR：#178（double label-shift）、#188（text_projection）、#233（24kHz 重采样）、#259（chunked_decode 截断）

3. **HuggingFace 模型集合**：https://huggingface.co/collections/Qwen/qwen3-tts

4. **ModelScope 模型集合**（中国大陆推荐）：https://modelscope.cn/collections/Qwen/Qwen3-TTS

5. **官方 Blog**：https://qwen.ai/blog?id=qwen3tts-0115

6. **vLLM-Omni 官方文档**
   - Quickstart：https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/#installation
   - Qwen3-TTS 离线推理示例：https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen3_tts

7. **NVIDIA Dynamo 文档（Text-to-Audio）**：https://docs.dynamo.nvidia.com/dynamo/dev/diffusion/text-to-audio

8. **阿里云百炼 DashScope 文档**
   - 实时 API（CustomVoice）：https://help.aliyun.com/zh/model-studio/qwen-tts-realtime
   - 声音复刻 API：https://help.aliyun.com/zh/model-studio/qwen-tts-voice-cloning
   - 声音设计 API：https://help.aliyun.com/zh/model-studio/qwen-tts-voice-design
   - 国际站：https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime

9. **阿里云百炼定价页**（API 价格，一手）：https://help.aliyun.com/zh/model-studio/billing-for-model-studio
   - 英文版：https://help.aliyun.com/document_detail/2586397.html

#### 关联论文

10. **Qwen3-ASR Technical Report**：https://arxiv.org/abs/2601.21337（长语音评测的转写器；同栈组合）
11. **Qwen3-Omni Technical Report**：https://arxiv.org/abs/2509.17765
12. **GSPO**：Zheng et al., https://arxiv.org/abs/2507.18071
13. **DPO**：Rafailov et al., NeurIPS 2023

#### 架构与方法来源论文

14. **Mimi / Moshi**（12Hz tokenizer 架构参考）：Défossez et al., https://arxiv.org/abs/2410.00037
15. **StreamFlow**（25Hz 块式 guided attention mask）：Guo et al., https://arxiv.org/abs/2506.23986
16. **Flow Matching**（25Hz DiT 训练目标）：Lipman et al., ICLR 2023, https://arxiv.org/abs/2210.02747
17. **BigVGAN**（25Hz 声码器）：Lee et al., ICLR 2023, https://arxiv.org/abs/2206.04658
18. **WavLM**（语义教师 / SIM 度量）：Chen et al., https://arxiv.org/abs/2110.13900
19. **SpeechTokenizer**：Zhang et al., https://arxiv.org/abs/2308.16692

#### 对比基线论文

20. **MiniMax-Speech**：https://arxiv.org/abs/2505.07916
21. **CosyVoice 3**：https://arxiv.org/abs/2505.17589
22. **CosyVoice 2**：https://arxiv.org/abs/2412.10117
23. **VibeVoice**：https://arxiv.org/abs/2508.19205
24. **VoxCPM**：https://arxiv.org/abs/2509.24650
25. **FireRedTTS 2**：https://arxiv.org/abs/2509.02020
26. **InstructTTSEval**（可控性基准）：Huang et al., https://arxiv.org/abs/2506.16381
27. **Seed-TTS**（零样本基准）：Anastassiou et al., https://arxiv.org/abs/2406.02430
28. **Mimo-Audio**：https://arxiv.org/abs/2512.23808
29. **VoiceSculptor**：https://arxiv.org/abs/2601.10629

#### 二手 / 社区来源（**仅用于交叉印证，本文件中已逐条标注「待核实」**）

30. **instavar.com**《LoRA Finetuning Qwen3-TTS for Custom Voices》 https://instavar.com/blog/LoRA_Finetuning_Qwen3_TTS_Custom_Voices
    —— 失败模式分类表（13 项）、验证过的推理参数、checkpoint 选择流程
31. **`cheeweijie/qwen3-tts-lora-finetuning`** 配套仓库 https://github.com/cheeweijie/qwen3-tts-lora-finetuning
    —— 以 patch 形式提供 PR #178/#188/#233 的修复与 LoRA 训练脚本
32. **ai-rockstars.com**《Qwen 3 TTS Release》 —— **「无 FA2 时 RTX 5090 仅 0.3x 实时」与 VRAM 估算的来源**
33. **PyPI `qwen3-tts-triton`** https://pypi.org/project/qwen3-tts-triton/
    —— Triton/Torch kernel 微基准、E2E RTF、以及**带质量回归验证**（UTMOS/CER/SIM）的优化评测
34. **glbgpt.com**《Qwen3-TTS Review (2026)》 —— API 限制（512 tokens 输入、24 小时 URL 过期）与国际区美元定价
35. **CSDN 实测**（RTX 3090/4090/5090 三卡对比）—— 首包延迟 96.5–97.2ms、显存 7.5–7.8GB
36. **NVIDIA Dynamo 文档**（`/v1/audio/speech` 参数与限制）
37. **社区 fork**：`andimarafioti/faster-qwen3-tts`、`dffdeeq/Qwen3-TTS-streaming`、`rekuenkdr/ComfyUI-Qwen3-TTS`、`bc-dunia/qwen3-TTS-studio`、`MonOttawa/Qwen3-TTS-Openai-Fastapi`

> 说明：本文件所有**数值均优先采用 arXiv:2601.15621 与 GitHub 官方 README**。二手来源仅用于补充官方未覆盖的工程细节（显存、社区 bug、API 限制、加速收益），且已在对应字段明确标注为「待核实」。


> **本章定位**：Qwen3-TTS 以双 Tokenizer 路线同时覆盖低延迟与长文本稳定性，是 TTS 侧的首选。


---

<!-- item: cosyvoice3 -->

## CosyVoice3

> **分类**：核心模型-TTS　|　**已填字段**：38 / 38　|　**⭐ 标记为本轮重点的架构与训练字段**


### 1. 基本信息与版本演进

#### 基本画像

- **名称**：CosyVoice 3 / Fun-CosyVoice3-0.5B-2512（仓库名 CosyVoice，三代模型的官方品牌为 "Fun-CosyVoice 3.0"）
- **开发方**：阿里通义实验室语音团队 FunAudioLLM（作者：Zhihao Du、Changfeng Gao、Yuxuan Wang 等，arXiv:2505.17589）
- **定位**：面向 **in-the-wild（真实开放场景）** 的零样本多语言语音合成系统；在**内容一致性（CER/WER）、说话人相似度（SS/SIM）、韵律自然度**三个维度全面超越 CosyVoice 2
- **论文**：*CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training*，arXiv:2505.17589，v1 提交于 2025-05-23，v2 修订于 2025-05-27
- **开源与权重时间线**（来自官方 GitHub README Roadmap）：
  - 2024/07：CosyVoice 1.0（arXiv:2407.05407，300M，监督语义 token + Flow Matching 训练支持、FastAPI 服务）
  - 2024/08：RAS（Repetition Aware Sampling）推理 + 流式推理（KV cache、SDPA）
  - 2024/09：CosyVoice-300M 25Hz base + 变声（voice conversion）
  - 2024/12：CosyVoice 2.0（arXiv:2412.10117，25Hz，0.5B，双向流式 150ms，FSQ 替换 VQ、chunk-aware flow matching）
  - 2025/05：CosyVoice 3 论文发布；CosyVoice2-0.5B 支持 vLLM
  - 2025/07：发布 Fun-CosyVoice 3.0 评测集（CV3-Eval）
  - 2025/08：NVIDIA Yuekai Zhang 贡献 **Triton + TensorRT-LLM runtime** 与 **CosyVoice2 GRPO 训练**支持
  - **2025/12**：发布 **Fun-CosyVoice3-0.5B-2512 base 模型 + RL 模型 + 训练/推理脚本**，以及 ModelScope Gradio 空间（即当前默认推荐模型）
- **注**：论文（2025-05）与权重发布（2025-12，`-2512` 后缀指 2025 年 12 月）间隔约半年，README 中 "2025/12 release Fun-CosyVoice3-0.5B-2512 base model, rl model and its training/inference script" 为一手来源
- **核心升级点（论文摘要原文四点）**：
  1. 监督多任务训练的新型语音 tokenizer（ASR / SER / LID / AED / SA）
  2. 新的可微奖励模型用于后训练（DiffRO），可推广到其他 LLM-based TTS
  3. **数据规模扩展**：1 万小时 → **100 万小时**，9 语言 + 18 种中国方言
  4. **模型规模扩展**：0.5B → **1.5B**

#### 官方 Key Features（HF 模型卡 / README 原文）

- **Language Coverage**：9 种常见语言（中、英、日、韩、德、西、法、意、俄）+ 18+ 种中国方言/口音（广东、闽南、四川、东北、山西/陕西、上海、天津、山东、宁夏、甘肃等），支持多语言与跨语言零样本克隆
- **Content Consistency & Naturalness**：SOTA 级内容一致性 / 说话人相似度 / 韵律自然度
- **Pronunciation Inpainting**：支持中文拼音与英文 CMU 音素的「发音修补」
- **Text Normalization**：无需传统前端模块即可读数字、特殊符号与各种文本格式
- **Bi-Streaming**：text-in 与 audio-out 双向流式，最低 **150ms** 延迟
- **Instruct Support**：支持语言、方言、情感、语速、音量等指令控制

#### 同生态位坐标

阿里 FunAudioLLM 家族：FunASR（工业 ASR）、Fun-ASR-Nano（31 语言端到端 LLM ASR）、SenseVoice（快速 ASR + 情感 + 事件检测）、FunClip（视频剪辑）。CosyVoice 代码大量复用 FunASR / FunCodec / Matcha-TTS / AcademiCodec / WeNet。


### 2. 开源协议与商用边界

- **许可证**：**Apache License 2.0**（HF 模型卡 `license: apache-2.0`；GitHub 仓库 LICENSE）
- **商用边界**：**可商用**，无收入/规模阈值、无「不得用于改进其他模型」类条款，是本文调研的同级别开源 TTS 中**商用最干净**的几家之一（与 VoxCPM、GLM-TTS、Kokoro、VibeVoice 同级；对比 IndexTTS-2 的 bilibili 协议、Fish Audio 的研究许可、F5-TTS 权重的 CC-BY-NC）
- **版权与免责**：README 底部有 Disclaimer——「内容仅供学术用途，用于演示技术能力，部分样例来自互联网，如侵犯权益请联系删除」。声音克隆本身涉及**声音权/肖像权/人格权**合规，Apache-2.0 不覆盖被克隆者的授权问题，生产环境需自行取得说话人授权
- **代码与权重同协议**：代码与模型权重均为 Apache-2.0，无 Fish-Speech / Higgs-Audio 那种「代码宽松 + 权重受限」的错配问题
- **第三方依赖的许可注意**：
  - `CosyVoice-ttsfrd` 资源（可选，用于更好的文本归一化）为独立下载项，**其许可需单独确认（待核实）**；不安装时默认回退到 `WeTextProcessing`（wetext）
  - `ttsfrd` 提供的是 cp310 的 Linux whl（`ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl`），进一步锁定 Python 3.10 + Linux
- **社区衍生权重**：MLX 量化变体（如 `aufklarer/CosyVoice3-0.5B-MLX-4bit`、`aitytech/CosyVoice3-0.5B-MLX-4bit`）在模型卡中声明 Apache 2.0「matching the upstream CosyVoice 3 release」


### 3. 模型变体与权重获取

| 模型 ID | 参数量 | 采样率 / token 率 | 状态 | 用途 | 下载地址 |
|---|---|---|---|---|---|
| **Fun-CosyVoice3-0.5B-2512** | 0.5B | 24kHz / 25Hz | 🟢 官方主推 | 零样本多语言 TTS + DiffRO 后训练前 | [HF](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) / [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) |
| **Fun-CosyVoice3-0.5B-2512_RL** | 0.5B | 24kHz / 25Hz | 🟢 官方主推 | DiffRO 后训练版，**CER 更低、SS 略降** | 同上仓库 |
| CosyVoice 3-1.5B（论文） | 1.5B | 24kHz / 25Hz | 🟡 论文报告，**权重未开源** | 论文中验证 model-size scaling；多语言 CV3-Eval 上优于 0.5B | 论文 Table 4/5 |
| CosyVoice 3-1.5B RL（论文） | 1.5B | 24kHz / 25Hz | 🟡 论文报告 | 论文中最佳组合（zh CER 0.71 / en WER 1.45） | 论文 Table 4 |
| CosyVoice2-0.5B | 0.5B | 24kHz / 25Hz | 🟢 稳定 | 上一代，含 vLLM / Triton TRT-LLM / GRPO 支持 | [HF](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) / [ModelScope](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) |
| CosyVoice-300M / -SFT / -Instruct | 0.3B | 22050Hz（v1）/ 25Hz | ⚪ Legacy | v1 系列；SFT 用于微调说话人，Instruct 用于指令 | ModelScope `iic/CosyVoice-300M` 等 |
| CosyVoice-ttsfrd | — | — | 可选资源 | 文本归一化资源包 | ModelScope `iic/CosyVoice-ttsfrd` |

**注（重要选型信息）**：开源权重只有 **0.5B** 一档。论文中的 1.5B 与 1.5B-RL 结果**目前无法直接复现/部署**，因为阿里未放出 1.5B 权重（至少在 2025-12 的发布中未放出）。若你的选型依赖 1.5B 的指标，请注意指标与可获取权重不一致。

**MLX 社区量化变体（Apple Silicon，来自 HF 模型卡）**：

| 变体 | LLM 精度 | 体积 | 说明 |
|---|---|---|---|
| CosyVoice3-0.5B-MLX-4bit | 4-bit（group_size=64，量化 attention proj / MLP / speech head） | ~1.2 GB | LLM 467MB(4bit) + DiT 634MB(fp16) + HiFi-GAN 79MB(fp16) |
| CosyVoice3-0.5B-MLX-8bit | 8-bit | ~1.4 GB | 待核实具体分层 |
| CosyVoice3-0.5B-MLX-8bit-full | 8-bit 全量化 | ~1.6 GB | 待核实 |
| CosyVoice3-0.5B-MLX-bf16 | bf16 | ~2.1 GB | 未量化 |

转换说明（模型卡原文）：Flow Matching 对量化敏感，故 **DiT 保持 fp16**；HiFi-GAN 用 fp16 并折叠 weight normalization（`w = g * v / ||v||`）；Conv1d 权重从 PyTorch `[out, in, kernel]` 转置为 MLX `[out, kernel, in]`。M2 Max 上 RTF 约 0.5（来源：用户提供的社区实测，**官方未公布，待进一步核实**）。


### 4. 架构范式 ⭐

#### 总范式：**级联式（cascade）「LLM 自回归 → 非自回归流匹配 → 神经声码器」三段流水线**

CosyVoice 3 延续并强化了 CosyVoice 2 确立的两阶段混合范式（论文 §1 背景段描述）：

> 第一阶段用 **LLM 建立粗粒度语义 token 序列**（负责「说什么」+ 韵律轮廓 + 内容一致性）；
> 第二阶段用**非自回归的生成模型（diffusion / flow matching）渲染声学细节**（负责「谁在说」+ 音色与音质）；
> 最后经 **vocoder** 还原波形。

这一范式的核心工程价值在于：**把「可控性/内容准确性」与「音质/实时性」解耦到两个可独立优化、独立加速的模块**。

##### 三代范式演进

| 维度 | CosyVoice 1.0（2024-07） | CosyVoice 2.0（2024-12） | **CosyVoice 3.0（2025-12 权重）** |
|---|---|---|---|
| 语音 tokenizer | **VQ**（监督语义 token，SenseVoice-Large ASR 监督） | **FSQ**（有限标量量化，解决 VQ 码本坍塌） | **FSQ + 多任务监督**（ASR/SER/LID/AED/SA），基座换成 MinMo |
| Token 率 | 25Hz（2024/09 起；初版 50Hz） | 25Hz | **25Hz** |
| 码本利用率 | ~23%（VQ 坍塌） | 接近满利用率 | 接近满利用率（$Q=(2K+1)^D-1$，连续化构造天然无坍塌） |
| LLM 主干 | Transformer Encoder（Conformer text encoder + Transformer LM） | **Qwen2 系列文本 LLM 初始化** | **Qwen2.5-0.5B / 1.5B** |
| 声学模型 | Conditional CFM + U-Net(ConditionalDecoder) | chunk-aware Flow Matching | **DiT 作为 CFM 骨干**（100M → 300M 参数） |
| 文本编码器 / 长度调节器 | 需要 | 需要 | **移除**（DiT 足够强，改用简单插值解决 25Hz token 与 mel 帧率不匹配） |
| 流式 | 非流式 / 后期加流式 | 双向流式 150ms | 双向流式 150ms（README） |
| 后训练 | 无 | 无 / 后期加 GRPO | **DiffRO（可微奖励优化）** |
| 数据 | 数万小时 | ~1 万小时 | **~100 万小时** |

##### 为什么是「AR + FM」而不是纯 NAR 或纯 AR

- **纯 NAR（如 F5-TTS、E2-TTS）**：一次前向出全部 mel，快，但**时长必须预先给定/由长度回归决定**，跨语言与长句稳定性差，且难以做 token 级时长控制
- **纯 AR（早期 VALL-E 类）**：可自然建模时长，但 acoustic token 序列太长，推理慢，且高码本下采样质量不稳
- **CosyVoice 3 的折中**：LLM 只自回归生成 **25Hz 的低帧率语义 token**（比 mel 的 ~86Hz 少 3 倍以上），把高维度的声学细节交给 DiT 一次性渲染。这带来两个直接收益：
  1. **首包延迟可控**——LLM 每 40ms 出一个 token，配合 chunked FM 可做到 150ms 首包
  2. **时长自然涌现**——不需要显式 duration predictor，LLM 自己决定何时出 EOS

##### 训练范式：四阶段（论文 Figure 2(b)）

```
1. 大规模预训练 (large-scale pretraining)
        ↓
2. 后训练 (post-training, DiffRO)  ← 突破训练数据性能上限的关键
        ↓
3. 持续预训练 (continual pretraining) ← 把 zero-shot 的指令可控 / 多语能力迁移到 SFT 模型
        ↓
4. 多说话人微调 (multi-speaker fine-tuning, SFT)
```

**注意**：论文 §7 明确未公布 batch size / learning rate / GPU 小时等优化超参，二次开发时只能从 `examples/libritts` 的 yaml 里取 CosyVoice 2 时代的参考值（见 `training_stages` 字段）。

##### 与「无 tokenizer」派的分野

VoxCPM 系列是 **tokenizer-free**（直接在 AudioVAE 连续 latent 上做 diffusion autoregressive），而 CosyVoice 3 仍然坚持**离散 FSQ token**。取舍：

- 离散 token 的好处：LLM 可直接复用文本 LLM 的工程栈（KV cache、vLLM、speculative decoding、RAS 采样），流式与批处理优化成熟；DiffRO 这类「在 token logits 上做可微奖励优化」的方法也只在离散 token 上成立
- 离散 token 的代价：量化上限决定音质天花板（CosyVoice 3 输出 24kHz，VoxCPM2 已到 48kHz）

##### 架构图（文字版，对应论文 Figure 2）

```
参考音频 ──> [Voice Encoder₁ (12×Transformer + RoPE)] ──> H ──> [FSQ] ──> 语音 token μ (25Hz, 词表 6561)
                                                            │
                                                            └─> [Voice Encoder₂ + MinMo LLM] ──> 多任务监督头
                                                                  (ASR / SER / LID / AED / SA)
参考音频 ──> [CAM++ Speaker Encoder] ──> 192-d embedding ──> normalize + affine ──┐
                                                                                  ↓
文本 + 指令 ──> [Qwen2.5 Tokenizer] ──> text token ──> 【LLM: Qwen2.5-0.5B/1.5B】 ──(AR, FSQ token 25Hz)──> μ̂
                                                                                  ↓
                                            【DiT Flow Matching (22 层)】 ←── mel(80 band) + spk conditioning
                                                    (Euler 10 步, CFG 0.7)
                                                                                  ↓
                                                        【HiFi-GAN/HiFT 声码器】 ──> 24kHz 波形
```


### 5. 模块拆解 ⭐

#### 组件清单与数据形状

| # | 组件 | 实现 | 输入 → 输出 | 参数量 | 来源 |
|---|---|---|---|---|---|
| 1 | **语音 Tokenizer（S3 Tokenizer v3）** | MinMo voice encoder₁（12×Transformer + RoPE）→ FSQ（低秩投影 + 有界 ROUND）→ voice encoder₂ + MinMo LLM（多任务头） | 音频波形 → **25Hz 离散 token**，词表 6561 | MinMo 为 140 万小时预训练的多模态理解模型，模块自身参数量**未公布（待核实）** | 论文 §2.1 / Table 3 |
| 2 | **文本前端** | Qwen2.5 tokenizer（BPE）+ 内置 TN（无需独立前端）；可选 `ttsfrd` / `WeTextProcessing` | 原始文本（含数字、符号）→ text token | — | README |
| 3 | **说话人编码器** | **CAM++**（`spk_embed_dim: 192`），L2 归一化后经 `Linear(192 → llm_input_size)` 仿射投影，以 1 个 token 的形式拼在 `sos_emb` 之后、text 之前 | 参考音频 → 192-d speaker embedding | 小模型（几十 M 量级，待核实） | 代码 `cosyvoice/llm/llm.py`；配置文件 `spk_embed_dim: 192` |
| 4 | **主 LLM（Text→Speech Token）** | **Qwen2.5-0.5B**（24 层，hidden 896，14 query 头 / 2 KV 头 GQA）；后接 `nn.Linear(llm_output_size, speech_token_size + 200, bias=False)` | [sos, spk_emb, text_token, task_id, prompt_speech_token] → 自回归 FSQ token | 0.5B（1.5B 变体见论文） | MLX 模型卡 + 代码 `CosyVoice3LM` |
| 5 | **声学解码器（Token→Mel）** | **22 层 DiT**，dim 1024，16 头，AdaLN 条件化；Conditional Flow Matching，Euler solver，10 步（NFE=10），CFG 0.7 | 25Hz FSQ token + speaker cond → **80 band mel** | 论文称 CFM 从 100M 增至 300M | 论文 §4.2（层数/dim/头/步数/CFG 来自 MLX 模型卡与用户提供的代码级核实，**论文本身未给，需以代码为准**） |
| 6 | **声码器（Mel→Waveform）** | **HiFT / HiFi-GAN**： NSF（8 谐波）+ F0 预测器（ConvRNNF0Predictor）+ ISTFT；`upsample_rates` 使总上采样 480×；`istft_params: n_fft=16, hop_len=4` | 80 mel → **24kHz 波形**（v1 yaml 为 22050Hz，v2/v3 为 24000Hz） | 小（MLX fp16 版 79MB） | 代码 `cosyvoice/hifigan/generator.py`；配置 `hift:` 段 |
| 7 | **流式调度** | chunk-aware flow matching + KV cache + `mix_ratio=[5, 15]`（bistream 模式：每 5 个 text token 对应 15 个 speech token） | — | — | 代码 `cosyvoice/llm/llm.py` |
| 8 | **采样器** | `ras_sampling`：`top_p=0.8, top_k=25, win_size=10, tau_r=0.1`（RAS = Repetition Aware Sampling） | logits → token | — | 配置文件 `cosyvoice.yaml` 的 `sampling:` 段 |

#### 关键张量维度（代码/配置级核实）

- `sample_rate`：24000（CosyVoice 2/3）；v1 的 libritts 示例 yaml 为 22050
- `llm_input_size` / `llm_output_size`：1024（v1 配置；Qwen2.5 变体由模型自身 hidden=896 决定，**两者来自不同代际配置，注意区分**）
- `spk_embed_dim`：**192**（CAM++ 输出，跨代一致）
- mel：`num_mels=80`, `n_fft=1024`, `hop_size=256`, `win_size=1024`, `fmin=0`, `fmax=8000`
- speech token：**25Hz**；FSQ 码本 6561 = $(2K+1)^D$ 的具体取值（当 $D=8, K=1$ 时 $3^8=6561$；论文未给 $D$、$K$，**6561 来自代码/模型卡，待论文确认**）
- `stop_token_ids`：`[speech_token_size + i for i in range(200)]`（CosyVoice3LM），解码头输出维度 `speech_token_size + 200`
- 特殊 token（CosyVoice3LM）：`sos = size+0`、`eos = size+1`、`task_id = size+2`、`fill_token = size+3`；另有硬编码 **`<|endofprompt|> = 151646`**，代码中 assert 其必须出现在 `text` 里

#### 关于「特殊 fl_* 控制 token」的澄清

用户资料中提到的 `<|fl_speaker_clone|>`、`<|fl_speaker_instruct|>`、`<|fl_speaker_instruct2|>`、`<|fl_save_speaker|>` 等命名，**未在 `cosyvoice/llm/llm.py` 中出现**。当前代码使用的特殊 token 是另一套基于 `speech_token_size` 偏移的内部编号（见上）。`fl_*` 体系更可能来自 CosyVoice 的更高层 tokenizer/prompt 构造逻辑、其他分支文件、或早期/内部版本。**建议二次开发时以代码中的 `<|endofprompt|>` + task_id 体系为准，并标注此字段待官方确认。**


### 6. 音频编码器 ⭐

#### 两个「音频编码器」，职责完全不同

##### 1. 语音理解编码器（Tokenzier 侧）：MinMo Voice Encoder

- **基座**：**MinMo**（论文引用 [33]），一个在 **>140 万小时**语音上预训练的多模态语音理解大模型
- **结构**（论文 §2.1 明确）：
  - `Voice Encoder₁`：**12 个 Transformer block，带 RoPE（旋转位置编码）**
  - 输入语音 $X$ → `Voice Encoder₁` → 中间表示 $H$
- **对比前代**：CosyVoice 2 的语义 token 监督来自 **SenseVoice-Large**（纯 ASR）；CosyVoice 3 换成 MinMo，监督信号从单一 ASR 扩展到 **5 个任务**（见 `tokenizer_codec`）
- **参数量**：论文未公布（**待核实**）
- **训练梯度**：FSQ 与 `Voice Encoder₁` 的梯度用 **straight-through estimation** 近似

##### 2. 说话人编码器（TTS 侧）：CAM++

- **模型**：**CAM++**（`$spk_embed_dim = 192$`），工业级说话人确认模型，在 FunASR 生态中广泛使用
- **处理流程**（代码 `cosyvoice/llm/llm.py` 中的 `TransformerLM.forward` / `inference`）：
  ```python
  embedding = F.normalize(embedding, dim=1)            # L2 归一化
  embedding = self.spk_embed_affine_layer(embedding)    # Linear(192 → llm_input_size)
  embedding = embedding.unsqueeze(1)                    # (B, 1, D)
  # 拼接顺序：lm_input = [sos_emb, embedding, text_token, task_id_emb, speech_token, ...]
  ```
- **注意**：在 **Qwen2LM / CosyVoice3LM 的 `inference` 路径中，speaker embedding 被移除**——`lm_input = [sos_emb, text_emb, task_id_emb, prompt_speech_token_emb]`。也就是说，**三代推理主要靠 prompt 音频的 speech token（in-context learning）而非 CAM++ 嵌入来传递音色**；CAM++ 分支（`use_spk_embedding`）更多用于 SFT / 特定模式
- **DiT 侧条件化**：speaker embedding 经 `Linear(192 → 80)` 投影后作为 DiT 的全局条件（`spk_emb_dim: 80`，与 80 band mel 对齐），配合 AdaLN 注入

##### 3. 评估用的说话人模型（非训练组件，易混淆）

论文 §4.3 Evaluation Settings 明确：
- 用 **ERes2Net**（引用 [47]）从生成语音提取 speaker embedding 算 SS（表中括号内数值）
- 与 Seed-TTS 比较时用 **WavLM-based** 说话人确认模型（引用 [49]）算 SS（表中括号外数值）
- **论文中完全没有使用 / 提及 CAM++**（CAM++ 只存在于推理代码里）

**选型提示**：同一模型的 SS 数值会因评估模型不同而产生 5~8 个点的差异（例如 CosyVoice3-0.5B test-zh SS：WavLM 口径 **0.780**，ERes2Net 口径 **0.840**）。**跨模型对比 SS 时务必确认是同一口径**，否则会得出错误结论。


### 7. 语音 Tokenizer 与编解码器 ⭐

#### 核心创新 1：基于 MinMo 的监督多任务语音 Tokenizer（论文 §2.1）

##### 结构

```
Audio X
   │
   ▼
[Voice Encoder₁]  12 × Transformer block + RoPE      ← 可训练，梯度用 straight-through 近似
   │
   ▼  H  (连续中间表示，帧率 25Hz)
   │
   ▼
[FSQ Module]  ─ Proj_down(H) → D 维低秩空间
              ─ bounded ROUND: 每维量化到 [-K, K]
              ─ H̄ = ROUND(Proj_down(H))         ← 离散码
              ─ Ĥ = Proj_up(H̄)                   ← 重建，残差连接回主干
   │
   ▼  μ = Σ_{j=0}^{D-1} h̄_{i,j} · (2K+1)^j      ← 把 D 维整数向量折叠成单一 token id
   │
   ▼
[Voice Encoder₂ + MinMo LLM] ──> 多任务监督头（ASR / SER / LID / AED / SA）
```

##### FSQ（Finite Scalar Quantization）为什么取代 VQ

| 问题 | VQ（CosyVoice 1.0 / 2.0 之前的方案） | FSQ（CosyVoice 2/3） |
|---|---|---|
| 码本坍塌 | 存在，**码本利用率约 23%**（用户提供，来自前代论文/社区实测） | **接近满利用率**：码本由 $(2K+1)^D$ 的笛卡尔积隐式定义，不存在「未使用码字」 |
| 码本学习方式 | 需要 EMA / commitment loss 维护码本 | **无可学习码本**，投影 + 有界取整即可 |
| 词表大小 | $Q$ 为超参，坍塌后有效容量远小于 $Q$ | $Q = (2K+1)^D - 1$，容量可精确计算 |
| 梯度 | straight-through | straight-through |

**论文明确给出的公式**：
- 低秩投影：$\bar{H} = \text{ROUND}(\text{Proj}_{down}(H))$，重建 $\hat{H} = \text{Proj}_{up}(\bar{H})$
- Token id：$\mu_i = \sum_{j=0}^{D-1} \bar{h}_{i,j}(2K+1)^j$
- 码本大小：$Q = (2K+1)^D - 1$
- **$D$ 和 $K$ 的具体数值论文未给出**（只以符号出现）。社区/模型卡给出的 **6561** 对应 $D=8, K=1$（$3^8 = 6561$），**该对应关系为推断，待官方确认**

##### 多任务监督（这是 CosyVoice 3 与 2 的最大区别之一）

| 监督任务 | 缩写 | 训练小时数 | 作用 |
|---|---|---|---|
| 自动语音识别 | ASR | **365K h**（中英日韩俄法德 7 语） | 让 token 承载语义内容 → 直接决定 TTS 的 CER/WER |
| 语言识别 | LID | **85K h** | 让 token 区分语言 → 支撑 9 语言 + 跨语言克隆 |
| 语音情感识别 | SER | **48K h** | 让 token 携带情感 → 支撑情感克隆与 DiffRO-EMO |
| 音频事件检测 | AED | **21K h** | 让 token 感知非语音事件（笑声、呼吸等） |
| 说话人分析 | SA | **11K h** | 让 token 分离说话人身份 → 支撑 zero-shot 音色克隆 |
| **合计** | — | **530K h**（Table 3） | — |

**设计意图**：传统语音 tokenizer 只用 ASR 监督，token 里只有「说了什么」，情感/口音/音色信息被视为噪声丢掉。CosyVoice 3 用 **5 个任务同时监督**，让同一串 token 同时编码语义 + 情感 + 语言 + 事件 + 说话人，**韵律自然度的提升主要来源于此**。

**对二次开发的直接含义**：如果你要替换/微调 tokenizer，只保留 ASR 监督会显著掉韵律质量；SER 和 AED 数据是最难补齐的部分。

##### Token 率

- **25 Hz**（论文 §2.1 末尾明确）：每秒 25 个 speech token
- 相比 mel 的 ~86.1 Hz（24000 / 256 hop ≈ 93.75，cosyvoice 配置 hop_size=256 → 约 93.75Hz；实际常用 86Hz 口径），**压缩了约 3.5 倍**，这是 LLM 能做实时自回归的前提
- 25Hz ↔ mel 帧率不匹配，CosyVoice 3 用**简单插值操作**解决（论文 §4.2），**不再需要 length regulator**（v1 的 `InterpolateRegulator` 被移除）


### 8. LLM 主干 ⭐

#### 主 LLM：Qwen2.5 系列（Text → Speech Token 的自回归生成器）

##### 配置

| 项目 | 数值 | 来源 |
|---|---|---|
| 基座 | **Qwen2.5-0.5B**（1.5B 变体见论文） | MLX 模型卡（`Qwen2.5-0.5B (24L, 896d, 14Q/2KV heads)`） |
| 层数 | **24 层** | MLX 模型卡 |
| hidden dim | **896** | MLX 模型卡 |
| 注意力 | **14 query 头 / 2 KV 头（GQA）** | MLX 模型卡 |
| 输入/输出 size | `llm_input_size / llm_output_size`（v1 yaml 为 1024；Qwen 变体由模型自身 hidden=896 决定） | `cosyvoice.yaml` / 代码 |
| 解码头 | `nn.Linear(llm_output_size, speech_token_size + 200, bias=False)` | 代码 `CosyVoice3LM` |
| 参数量 | **0.5B**（论文中 scale 到 1.5B 的版本多语言 CV3-Eval 更优，但**权重未开源**） | README / 论文 §4.2 |

**重要提示**：`cosyvoice.yaml`（`examples/libritts/cosyvoice/conf/cosyvoice.yaml`）里的 `llm:` 段描述的是 **v1 的 `TransformerLM`**（ConformerEncoder 6 层 + TransformerEncoder 14 层 16 头）。**CosyVoice 3 实际使用 `CosyVoice3LM`（Qwen2.5 基座）**，配置在预训练权重的 yaml 里。二次开发时不要照抄 examples 下的 yaml 作为三代的配置。

##### 输入序列组织

```
lm_input = [ sos_emb,
             (可选) spk_embedding,        # CAM++ 192-d，经 affine 投影成 1 个 token
             text_token...,               # Qwen2.5 BPE，必须以 <|endofprompt|>(151646) 结尾
             task_id_emb,                 # 单 token
             prompt_speech_token_emb...,  # 参考音频的 FSQ token（in-context）
             speech_token... ]            # 待生成的目标
```

- `<|endofprompt|>`（id **151646**）是**必须**的：代码中有 `assert 151646 in text, '<|endofprompt|> not detected ...'`
- `task_id` 单 token 区分任务类型（zero-shot / SFT / instruct / cross-lingual / bistream 等）
- **音色传递机制**：三代推理路径（`CosyVoice3LM.inference`）**不使用** CAM++ embedding，而是把参考音频的 FSQ token 作为 prompt 拼进上下文（in-context learning）。这与 Fish-Speech / VALL-E 的思路一致，好处是同时传递音色 + 韵律 + 录音环境

##### 采样策略（配置文件 `cosyvoice.yaml` 的 `sampling:` 段，已核实）

```yaml
sampling: !name:cosyvoice.utils.common.ras_sampling
    top_p: 0.8
    top_k: 25
    win_size: 10
    tau_r: 0.1
```

- **top_k = 25**，**top_p = 0.8**
- **RAS（Repetition Aware Sampling）**：`win_size = 10`，`tau_r = 0.1`
  - 原理：检测滑动窗口内是否出现重复 token 模式，若检测到则临时**降低 temperature / 抑制重复**，解决自回归 TTS 经典的「吞字、复读、死循环」问题
  - 2024/08 作为独立 feature 加入（README Roadmap），是 CosyVoice 稳定性的关键工程 trick
- vLLM 路径：`SamplingParams(top_k=sampling, stop_token_ids=..., min_tokens=min_len, max_tokens=max_len)`，`sampling` 默认 **25**

##### 长度控制

- `min_len = (text_len - prompt_text_len) * min_token_text_ratio`（默认 **2**）
- `max_len = (text_len - prompt_text_len) * max_token_text_ratio`（默认 **20**）
- 用 token 数区间约束，而非强制精确时长。**精确 token 级时长控制是 IndexTTS-2 的差异化能力，CosyVoice 3 不提供**

##### 规模效果（论文 Table 4）

| 模型 | test-zh CER | test-zh SS | test-en WER | test-en SS | test-hard CER | test-hard SS |
|---|---|---|---|---|---|---|
| CosyVoice 3-0.5B | 1.16 | 0.780 | 2.02 | 0.718 | 6.08 | 0.758 |
| CosyVoice 3-1.5B | 1.12 | 0.781 | 2.21 | 0.720 | **5.83** | 0.758 |

**结论**：1.5B 相比 0.5B 在 test-zh CER 上只从 1.16 → 1.12，提升有限；**多语言 CV3-Eval（Table 5）上差距更明显**（如 ja: 10.4 → 7.57，ko: 12.8 → 5.69，it: 6.68 → 10.5 反而变差）。**选型建议：中文/英文主场景用 0.5B 性价比最高；多语言重场景才需要 1.5B（但权重未开源）**。


### 9. 声学解码器 ⭐

#### 声学解码器：22 层 DiT + Conditional Flow Matching（Token → Mel）

##### 配置

| 项目 | 数值 | 来源 |
|---|---|---|
| 骨干 | **Diffusion Transformer (DiT)** | 论文 §4.2「we adopt the recent diffusion transformer (DiT) as the backbone」 |
| 层数 | **22 层** | MLX 模型卡（`22-layer DiT (1024d, 16 heads, 10 ODE steps)`） |
| hidden dim | **1024** | MLX 模型卡 |
| 注意力头 | **16 头** | MLX 模型卡 |
| 条件化 | **AdaLN**（Adaptive Layer Norm，由 timestep + speaker 条件调制） | 用户提供 + 代码结构；**论文未明确写 AdaLN，待代码确认** |
| ODE solver | **Euler**，**10 步（NFE=10）** | MLX 模型卡 / 用户提供 |
| CFG | **0.7**（推理时 `inference_cfg_rate: 0.7`；训练时 `training_cfg_rate: 0.2`） | v1 `cosyvoice.yaml` 的 `cfm_params`；三代同参数待确认 |
| 参数量 | **100M → 300M**（论文 §4.2：「increasing the number of parameters from 100M to 300M」） | 论文 |
| 输出 | **80 band mel**，帧率通过**简单插值**从 25Hz token 上采样得到 | 论文 §4.2 |

##### Flow Matching 配方（v1 `cosyvoice.yaml` 的 `cfm_params`，供参考）

```yaml
cfm_params:
    sigma_min: 1e-06
    solver: 'euler'
    t_scheduler: 'cosine'
    training_cfg_rate: 0.2
    inference_cfg_rate: 0.7
    reg_loss_type: 'l1'
```

##### 关键架构简化（论文 §4.2 原文要点）

1. **移除了 text encoder**：DiT 足够强，CFM 不再需要额外的文本编码器
2. **移除了 length regularization module**：改用简单插值解决 25Hz token 与 mel 帧率不匹配
3. 对比 v1 的 `ConditionalDecoder`（U-Net 结构：`n_blocks: 4`, `num_mid_blocks: 12`, `num_heads: 8`, `channels: [256, 256]`），三代全面换成 DiT

**工程含义**：DiT 相对于 U-Net 的好处是**结构与标准 Transformer 同构**，可以直接吃 CUDA Graph、FlashAttention、序列并行等 LLM 侧的优化；代价是失去了 U-Net 的多尺度归纳偏置，需要更多数据（CosyVoice 3 正好有 100 万小时）。

##### 与 Sway Sampling 的对比（F5-TTS 的差异化）

F5-TTS 提出 **Sway Sampling**（推理时非均匀地分配 flow step，在轨迹两端多分配步数），可在不增加 NFE 的前提下提升质量。CosyVoice 3 论文**未采用 / 未提及 Sway Sampling**，使用的是标准 cosine t-scheduler + Euler。若要在 CosyVoice 3 上复现 F5 的质量增益，Sway Sampling 是一个低成本的推理侧可选项（**未在官方代码验证，待核实**）。

##### 量化敏感性（实测经验，来自 MLX 模型卡）

> 「Flow: fp16 (flow matching is sensitive to quantization)」

即：**DiT 部分对量化敏感，4bit/8bit 量化会明显掉质量**。MLX 变体的做法是 **LLM 4bit/8bit 量化 + DiT 保持 fp16**。这一条对所有基于 flow matching 的 TTS 通用（F5-TTS、VoxCPM 的 LocDiT 同理），是显存优化时的重要边界。


### 10. 声码器

#### 声码器：HiFT / HiFi-GAN（Mel → 24kHz 波形）

##### 结构（`cosyvoice/hifigan/generator.py` 的 `HiFTGenerator`，配置来自 `cosyvoice.yaml` 的 `hift:` 段）

```yaml
hift: !new:cosyvoice.hifigan.generator.HiFTGenerator
    in_channels: 80            # 80 band mel
    base_channels: 512
    nb_harmonics: 8            # NSF 谐波数 = 8
    sampling_rate: 24000       # v1 yaml 为 22050；v2/v3 为 24000
    nsf_alpha: 0.1
    nsf_sigma: 0.003
    nsf_voiced_threshold: 10
    upsample_rates: [8, 8]     # 配合 ISTFT 实现总 480× 上采样
    upsample_kernel_sizes: [16, 16]
    istft_params:
        n_fft: 16
        hop_len: 4
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
    source_resblock_kernel_sizes: [7, 11]
    source_resblock_dilation_sizes: [[1,3,5], [1,3,5]]
    lrelu_slope: 0.1
    audio_limit: 0.99
    f0_predictor: ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
```

##### 技术要点

- **NSF（Neural Source Filter）**：8 个谐波的正弦激励源 + 滤波，显式建模 F0，对**音高准确性与歌唱/高情感语音**友好；`nsf_voiced_threshold: 10` 控制浊音判定
- **F0 预测器**：`ConvRNNF0Predictor`，从 mel 预测 F0，作为 NSF 的激励条件
- **ISTFT 上采样**：`n_fft=16, hop_len=4` 的极短窗 ISTFT，配合 `[8, 8]` 的两级上采样实现 **480× 上采样**（从 mel 帧率到 24kHz）。极短窗 ISTFT 是 HiFT 相比原始 HiFi-GAN 的关键提速手段
- **判别器**（训练期）：`MultipleDiscriminator` = `MultiPeriodDiscriminator (MPD)` + `MultiResSpecDiscriminator (MRD)`
- **Mel 变换**（训练/判别）：`n_fft=1024, num_mels=80, hop_size=256, win_size=1024, fmin=0, fmax=null(训练判别)/8000(特征提取)`

##### 质量基线（论文 Table 4 的 Vocoder Resyn. 行）

| 系统 | test-zh CER |
|---|---|
| Human | 1.26 |
| **Vocoder Resyn.（声码器重合成）** | **1.27** |

即：把真实录音过一遍声码器，CER 只从 1.26 → 1.27，**声码器引入的内容损失几乎可以忽略**。这说明 HiFT 声码器不是 CosyVoice 3 的质量瓶颈。

##### 体积与量化

- MLX fp16 版 **79 MB**（weight normalization 已折叠：`w = g * v / ||v||`）
- **可量化**：与 DiT 不同，声码器对 fp16 量化不敏感

##### 采样率说明（易踩坑）

- **v1（CosyVoice-300M）**：22050 Hz
- **v2 / v3（CosyVoice2-0.5B / Fun-CosyVoice3-0.5B）**：**24000 Hz**
- 对比：VoxCPM2 已到 **48kHz**，VoxCPM1.5 为 44.1kHz，VoxCPM-0.5B 为 16kHz。**若产品对音质有「录音室级」要求，24kHz 是 CosyVoice 3 的硬上限**

##### 其他可选项

v1 代码支持 **BigVGAN** 作为替代声码器（需 `git submodule update --init --recursive`）；F5-TTS 生态中 Vocos 也是常见选择。CosyVoice 3 官方默认 HiFT，**未依赖 BigVGAN submodule**。


### 11. 上下文长度与 I/O 约束

#### 上下文与 I/O 契约

##### 文本输入

- **分词**：Qwen2.5 BPE tokenizer（`text_token_size` 依配置，v1 为 51866 / 60515）
- **必须包含 `<|endofprompt|>`**（id **151646**），代码中硬编码 assert
- **无需独立文本前端**：TN（文本归一化）能力内化进 LLM——数字、日期、货币、特殊符号、中英混排直接读（README Key Features「Text Normalization」）
- 可选加速：`ttsfrd`（阿里内部 TN 资源，cp310 Linux whl）或 `WeTextProcessing`（wetext，默认回退）

##### 发音修补（Pronunciation Inpainting，论文 §2.3）

- **中文**：在文本中把多音字/生僻字替换为**拼音**
- **英文**：把单词替换为 **CMU 音素**
- 训练时构造辅助集：单音字 → 拼音，单词 → CMU 音素
- **效果（论文 Table 13，修正率）**：

| 方法 | zh 修正率 | en 修正率 |
|---|---|---|
| **RepMono + MixPhn** | **100%**（15 errors → 15 corrections） | **100%**（9 → 9） |
| RepMono + CatPhn | 86.7% | 100% |
| RepAll + MixPhn | 69.2% | 72.7% |

> `RepMono` = 只替换多音字（polyphone）；`RepAll` = 全部替换；`MixPhn` = 音素与文本混合输入；`CatPhn` = 音素拼接。**生产建议用 RepMono + MixPhn 策略**。

##### 参考音频（Prompt）

- 形式：**音频 + 对应文本**（both prompt speech token 与 prompt text）
- 最短时长：README 未明确给出数值。社区经验值通常 3~10 秒（**官方数值待核实**）
- 跨语言克隆支持：中文参考音 → 英文输出，反之亦可

##### 音频输出

- **24kHz 单声道 wav**（流式时为 chunk 序列）
- 流式：`mix_ratio = [5, 15]`，即 bistream 模式下每消费 5 个 text token 产出 15 个 speech token

##### 上下文长度

- 由 Qwen2.5 的位置编码决定（0.5B 通常 32K 上下文，**CosyVoice 3 的具体有效上下文长度未公布，待核实**）
- 长音频生成：论文未宣称超长能力，**不是 VibeVoice（90 分钟 / 4 说话人）的对手场景**

##### I/O 模式总览

| 模式 | 触发方式 | 说明 |
|---|---|---|
| zero-shot 克隆 | prompt 音频 + prompt 文本 + 目标文本 | 默认模式 |
| 跨语言克隆 | prompt 语言 ≠ 目标语言 | 内置支持 |
| instruct（指令） | 文本指令（情感/方言/语速/音量/角色） | 需 instruct 系列权重 / CV3 支持 |
| SFT | 微调后的 speaker id | v1 的 `CosyVoice-300M-SFT`；三代 SFT 见 README「Advanced Usage」 |
| voice conversion（变声） | `convert` 系列 API | v1 时代 feature |
| bistream（双向流式） | text-in 流式 + audio-out 流式 | `mix_ratio=[5,15]` |

##### 长度约束（代码级）

- `min_token_text_ratio` 默认 **2**，`max_token_text_ratio` 默认 **20**
- 数据过滤：`filter: max_length: 40960, min_length: 0, token_max_length: 200, token_min_length: 1`
- 训练 `max_frames_in_batch: 2000`（dynamic batch，V100 16G 上 gan 训练建议降到 1400）


### 12. 训练数据 ⭐

#### 训练数据：从 1 万小时 → 约 100 万小时

##### 总规模（论文 Abstract / §1 / §4.2 反复确认）

> "Training data is expanded from ten thousand hours to **one million hours**, encompassing **9 languages** and **18 Chinese dialects** across various domains and text formats."

| 项目 | CosyVoice 2 | **CosyVoice 3** | 倍数 |
|---|---|---|---|
| 主 TTS 训练数据 | ~1 万小时 | **~100 万小时** | **100×** |
| 指令跟随数据 | ~1,500 小时 | **5,000 小时**（>100 种控制指令类型） | 3.3× |
| Tokenizer 监督数据 | SenseVoice-Large ASR | **530K 小时**（5 任务，见下） | — |
| 语言 | 中英日韩 + 少数方言 | **9 语言 + 18/19 种中国方言** | — |

##### 语言与方言覆盖

- **9 种语言**：中文、英文、日文、韩文、德文、西班牙文、法文、意大利文、俄文
- **中国方言/口音**：论文 Abstract 写 **18 Chinese dialects**；§4.2 写 **19 common accents or dialects**（Figure 3b 标题）。列举包括：anhui（安徽）、cantonese（粤语）、chongqing（重庆）、hebei（河北）、shandong（山东）、shanghai（上海）、sichuan（四川）、tianjin（天津）、xi'an（西安）、zhengzhou（郑州）、dongbei（东北）、minnan（闽南）等
  > **注意 18 / 19 两个数不一致，论文内部存在表述差异，引用时建议写「18+ 种」**

##### Tokenizer 训练数据明细（论文 Table 3）

| 任务 | 小时数 | 说明 |
|---|---|---|
| **ASR**（多语言，中英日韩俄法德 7 语） | **365K h** | 最大头，决定 CER/WER |
| **LID**（语言识别） | **85K h** | |
| **SER**（语音情感识别） | **48K h** | 支撑情感表达 |
| **AED**（音频事件检测） | **21K h** | 支撑笑声/呼吸等副语言 |
| **SA**（说话人分析） | **11K h** | 支撑音色克隆 |
| **合计** | **530K h** | |

##### 领域 / 风格 / 文本格式多样性（§4.2）

- **领域（domain）**：电商、导航、金融、教育等
- **风格（style）**：对话、演讲、唱歌等（论文 Limitations 承认**歌唱生成能力有限**）
- **文本格式**：通过 **TN（text normalization）** 和 **ITN（inverse text normalization）** 构造不同书写格式的样本；用 **self-training** 制造大量 rare cases（罕见字、绕口令等）

##### 指令数据（§2.5）

- **5,000 小时**高质量 instruction-following 数据（从 1,500 小时扩展而来）
- 覆盖 **>100 种**说话风格类型（Table 1 列出 top-100 高频风格）：emotions、speed、voice tones、dialects、accents、role-playing

##### 多语言数据流水线（论文 §3，六步，工程可复现性高）

1. **语音检测与切分**：切成长度 **< 30s** 的片段
2. **降噪**：MossFormer2
3. **ASR 转录**：Faster-Whisper / NeMo / Seamless，**WER < 15% 的片段才保留**
4. **标点调整**：用 MFA（Montreal Forced Aligner）对齐；停顿 ≥ 300ms 加逗号，≤ 50ms 去掉停顿标点
5. **音量归一化**：除以 max × 0.6
6. **异常过滤**：按音频-文本长度比，去掉**最小 1% 和最大 5%**

##### 单语说话人 → 多语说话人（§2.6.1）

构造辅助集：指定 **speaker ID + language ID**，让只会单语的说话人「会说」其他语言，从而扩充多语言数据。论文 §5.8 验证效果：

| 语言 | CER/WER |
|---|---|
| 中、英、德、西、法、意、俄 | **全部 < 4%** |
| 日语 | **CER 9%**（汉字→假名转换 + 多音导致错误） |
| 韩语 | **CER ~6%**（数据量与质量有限） |

##### 发音修补数据（§2.3）

- 中文：单音字 → 拼音
- 英文：单词 → CMU 音素
- 见 `context_and_io` 字段的 Table 13

##### TN 自训练数据（§2.4）

三种方式构造：① 规则 TN；② Qwen-Max 做 TN；③ Qwen-Max 做 ITN

##### 数据来源声明

论文**未逐项列出数据集名称**（不像 F5-TTS 明确说用 Emilia / WenetSpeech4TTS / LibriTTS）。考虑到阿里同时维护 WenetSpeech4TTS，部分数据可能来自自建工业数据 + 公开集混合，**具体构成待核实**。这对**商用合规性尽职调查**是一个需要注意的空白点（对比：F5-TTS 明确声明 Emilia 是 CC-BY-NC，所以权重非商用；CosyVoice 3 声明 Apache-2.0，说明其数据链路上不存在 NC 约束的数据集）。


### 13. 训练阶段 ⭐

#### 四阶段训练流水线（论文 Figure 2(b) + Introduction 末）

```
┌─ Stage 1: 大规模预训练 (Large-scale Pretraining) ────────────────────────┐
│  数据：~100 万小时，9 语言 + 18/19 方言，多领域多文本格式                    │
│  目标：LLM 学会 text → FSQ speech token；DiT 学会 token+spk → mel         │
│  产出：base 模型（Fun-CosyVoice3-0.5B-2512）                              │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─ Stage 2: 后训练 (Post-training, DiffRO) ───────────────────────────────┐
│  方法：可微奖励优化（Gumbel-Softmax 采样 + 直接反传，非 RL loop）          │
│  奖励：ASR Token2Text / SER / MOS / AED 多任务奖励                        │
│  产出：RL 模型（Fun-CosyVoice3-0.5B-2512_RL）                            │
│  作用：突破训练数据性能上限，zh CER 1.21 → 0.81                            │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─ Stage 3: 持续预训练 (Continual Pretraining) ───────────────────────────┐
│  目的：把 zero-shot 模型的「指令可控」与「多语能力」迁移到 SFT 模型          │
│  这是 CosyVoice 3 相对 2 新增的阶段                                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─ Stage 4: 多说话人微调 (Multi-speaker Fine-tuning, SFT) ─────────────────┐
│  产出：特定说话人 / 特定场景的 SFT 模型                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

##### ⚠️ 关键缺口：优化超参未公开

论文**只给了模型规模与数据规模，没有给 batch size、learning rate、GPU 小时、训练步数**。这意味着**从零复现 CosyVoice 3 的预训练在论文层面是不可能的**。二次开发时可用的参考值来自 `examples/libritts/cosyvoice/conf/cosyvoice.yaml`（**注意：这是 v1 时代的配置**）：

```yaml
# llm + flow 训练
train_conf:
    optim: adam
    optim_conf:
        lr: 0.001            # SFT 时改成 1e-5
    scheduler: warmuplr      # SFT 时改成 constantlr
    scheduler_conf:
        warmup_steps: 2500
    max_epoch: 200
    grad_clip: 5
    accum_grad: 2
    log_interval: 100

# GAN（声码器）训练
train_conf_gan:
    optim: adam
    optim_conf:
        lr: 0.0002
    scheduler: constantlr
    optim_d: adam
    optim_conf_d:
        lr: 0.0002
    scheduler_d: constantlr
    max_epoch: 200
    grad_clip: 5
    accum_grad: 1            # GAN 训练必须为 1
    log_interval: 100
```

##### 数据处理流水线（配置级，`data_pipeline`）

```yaml
data_pipeline: [parquet_opener, tokenize, filter, resample,
                compute_fbank, parse_embedding, shuffle, sort, batch, padding]
data_pipeline_gan: [parquet_opener, tokenize, filter, resample, truncate,
                    compute_fbank, compute_f0, parse_embedding,
                    shuffle, sort, batch, padding]
```

- 数据格式：**parquet**（`parquet_opener`）
- `shuffle_size: 1000`，`sort_size: 500`（sort_size 必须 < shuffle_size）
- `batch_type: 'dynamic'`，`max_frames_in_batch: 2000`（V100 16G 上 GAN 训练建议 1400）
- `padding: use_spk_embedding: False`（**SFT 时改成 True**）
- 随机种子固定：`random.seed(1986)` / `numpy.random.seed(1986)` / `torch.manual_seed(1986)` / `torch.cuda.manual_seed_all(1986)`

##### 官方提供的训练入口

- `examples/libritts/`：完整的训练 + 推理脚本（README「Advanced Usage」指向这里）
- CosyVoice2 额外支持 **GRPO 训练**（2025/08 由 NVIDIA Yuekai Zhang 贡献），路径在 `examples/` 下
- vLLM 示例：`vllm_example.py`
- 分布式：基于 PyTorch 生态（具体 DeepSpeed/FSDP 配置**待核实**）

##### 微调（SFT）实践要点

- `lr` 从 0.001 降到 **1e-5**
- `scheduler` 从 `warmuplr` 换成 `constantlr`
- `padding.use_spk_embedding` 改成 **True**
- v1 时代的 `CosyVoice-300M-SFT` 表明 SFT 是官方支持路径；三代的 SFT 脚本在 `examples/` 下


### 14. 后训练与强化学习 ⭐

#### 核心创新 2：DiffRO（Differentiable Reward Optimization，论文 §2.2）

##### 要解决的问题

传统对 TTS 做 RL（PPO/GRPO）的三大痛点：
1. **计算昂贵**：合成音频必须经过 CFM + vocoder 才能评估，一次 rollout 成本极高
2. **输出难以区分正负样本**：相似音频间的质量差异很难被奖励模型稳定捕捉
3. **在线采样成本高**：每次迭代都要重新生成大量音频

##### DiffRO 的核心洞察

> **不要对合成音频做 RL，直接对 speech token 做可微优化。**

因为 speech token 是**离散但可微近似**的，可以用 **Gumbel-Softmax** 松弛后直接反向传播，把「奖励模型的梯度」端到端传回 LLM。这样：
- 不需要 RL 的 on-policy 采样循环
- 不需要经过 CFM + vocoder
- **计算成本接近一次普通的前向 + 反向**

##### 算法（论文公式 3~7）

**Step 1**：用 Gumbel-Softmax 从 LLM 的 token 分布中采样（保持可微）

$$\tilde{\mu}_t = \text{GumbelSoftmax}\ P_{\pi_\theta}(\mu_t | \mu_{1:t-1}; Y)$$

**Step 2**：训练一个 **ASR-like Token2Text 模型**（在 ASR 数据上训练），用其后验概率作为奖励

$$R_{ASR}(Y) = \log P_{ASR}(\tilde{Y}_n = Y_n | Y_{1:n-1}; \tilde{\mu}_{1:T})$$

即：把采样的 speech token 序列喂给 Token2Text 模型，看它能否正确「读回」目标文本 $Y$。读得越准，奖励越高 → **直接优化内容一致性（CER/WER）**。

**Step 3**：优化目标（带 KL 约束，防止退化）

$$\pi_\theta^* = \max_{\pi_\theta}\ \mathbb{E}[R(Y)] - \beta\ D_{KL}[\pi_\theta(\mu|Y) \| \pi_{ref}(\mu|Y)]$$

**KL 在 output token-level logits 上计算**（不是 sequence-level，这是关键设计）：

$$D_{KL} = \sum_{t=1}^{T} \sum_{k=0}^{Q} P_{\pi_\theta}(\mu_t = k) \log \frac{P_{\pi_\theta}(\mu_t = k)}{P_{\pi_{ref}}(\mu_t = k)}$$

其中 $Q = (2K+1)^D - 1$ 为 FSQ 码本大小。

**Step 4**：多任务奖励 MTR（Multi-Task Reward）

除 ASR 外，把其他音频理解任务的后验也作为奖励：

$$R_{MTR}(Y, \{A_i\}) = \sum_i \log P_{task_i}(\tilde{A}_i = A_i | \tilde{\mu})$$

| 奖励模型 | 控制属性 | 备注 |
|---|---|---|
| **Token2Text（ASR）** | 内容准确性（CER/WER） | 主奖励 |
| **SER**（语音情感识别） | 情感表达 | 即 **DiffRO-EMO** 变体（Table 9） |
| **MOS 分数预测** | 整体音质 | |
| **AED**（音频事件检测） | 副语言事件（笑声、呼吸等） | |

##### 效果（论文 §5.4）

| 指标 | 提升幅度 |
|---|---|
| **总体 WER** | **相对提升 20% ~ 50%** |
| **低资源语言 / 跨语言场景** | **相对提升 > 50%** |
| **韩语（CosyVoice 3-0.5B）** | **相对提升 68.7%** |
| 0.5B vs 1.5B 的差距 | RL 训练后显著缩小 |

##### 官方指标对比（README 表格，CV3-Eval / Seed-TTS 测试集）

| 模型 | test-zh CER | test-zh SS | test-en WER | test-en SS | test-hard CER | test-hard SS |
|---|---|---|---|---|---|---|
| Fun-CosyVoice3-0.5B-2512 | 1.21 | 78.0 | 2.24 | 71.8 | 6.71 | 75.8 |
| **Fun-CosyVoice3-0.5B-2512_RL** | **0.81** | 77.4 | **1.68** | 69.5 | **5.44** | 75.0 |

**规律**：DiffRO 让 **CER/WER 大幅下降**（zh: 1.21 → 0.81，-33%；en: 2.24 → 1.68，-25%；hard: 6.71 → 5.44，-19%），但 **SS 小幅下降**（zh: 78.0 → 77.4；en: 71.8 → 69.5，-2.3）。

##### 已知问题（论文自己承认）

1. **Reward Hacking**：说话人相似度（SS）会略降——模型学会了「牺牲音色换发音准确」
2. **EMO 奖励可能损害发音**：DiffRO-EMO 变体在强化情感时可能伤害 CER
3. **对 hard samples 提升较小**：罕见词、绕口令这类样本，DiffRO 的收益明显低于常规样本（因为奖励模型本身在这些样本上也不准）

##### 与同类后训练方法的横向对比

| 方法 | 代表模型 | 机制 | 成本 | 优点 | 缺点 |
|---|---|---|---|---|---|
| **DiffRO** | **CosyVoice 3** | Gumbel-Softmax + 可微奖励直接反传 | **低**（≈1 次前反向） | 无需 rollout，收敛快，可多任务奖励 | 只能优化 token 级目标；reward hacking 风险 |
| **GRPO** | GLM-TTS、CosyVoice2（NVIDIA 贡献）、Fish S2 Pro | 组内相对策略优化，需要 on-policy 采样 | **中高** | 更通用，可优化序列级目标 | 采样成本高，需 reward model 校准 |
| **DPO 类** | 部分社区方案 | 偏好对 | 中 | 简单 | TTS 上偏好标注难 |
| **RLHF** | Fish S2（早期） | 人类反馈 | 极高 | 对齐人类偏好 | 标注成本 |

**选型含义**：DiffRO 是 CosyVoice 3 相对所有竞品的**独有能力**。如果你的场景对**发音准确率（CER/WER）**极度敏感（如有声书、教育、播报），`_RL` 权重是明确的优选项；如果你的场景对**音色相似度**更敏感（如数字人、角色配音），则 base 权重（SS 更高）可能更合适。**建议两个权重都测**。

##### 补充：CosyVoice2 的 GRPO 路径

2025/08 NVIDIA Yuekai Zhang 贡献了 CosyVoice2 的 **GRPO 训练**支持（README Roadmap），与 DiffRO 是两条并行路径。三代主推 DiffRO。


### 15. 数据配方 ⭐

#### 数据配方（可直接复现的工程清单）

##### A. 多语言数据流水线（论文 §3，六步）

| 步骤 | 工具/方法 | 关键阈值 |
|---|---|---|
| 1. 语音检测与切分 | VAD + 切分 | 片段 **< 30s** |
| 2. 降噪 | **MossFormer2** | — |
| 3. ASR 转录 | Faster-Whisper / NeMo / Seamless | **保留 WER < 15% 的片段** |
| 4. 标点调整 | **MFA**（Montreal Forced Aligner） | 停顿 ≥ **300ms** → 加逗号；≤ **50ms** → 去掉停顿标点 |
| 5. 音量归一化 | 除以 max × 0.6 | — |
| 6. 异常过滤 | 按音频-文本长度比 | 去掉**最小 1% + 最大 5%** |

##### B. 多样性构造（§4.2）

- **领域（domain）**：电商、导航、金融、教育等
- **风格（style）**：对话、演讲、唱歌（歌唱能力有限，见 Limitations）
- **文本格式**：TN + ITN 双向构造不同书写形式；self-training 制造 rare cases

##### C. 任务特化数据集构造

| 目标能力 | 构造方法 | 数据规模/效果来源 |
|---|---|---|
| **指令跟随** | 高质量 instruction 数据，>100 种风格类型（emotion / speed / voice tone / dialect / accent / role-play） | **5,000 小时**（从 1,500 扩到 5,000） |
| **发音修补** | 中文单音字 → 拼音；英文单词 → CMU 音素。两种策略：`RepMono`（只替多音字）/ `RepAll`（全替）；两种注入：`MixPhn`（混合）/ `CatPhn`（拼接） | Table 13：**RepMono + MixPhn 达 100% 修正率** |
| **文本归一化** | 三种方式：① 规则 TN ② Qwen-Max 做 TN ③ Qwen-Max 做 ITN | §2.4 |
| **单语 → 多语** | 指定 speaker ID + language ID，让单语说话人「会说」其他语言 | §2.6.1；验证见 §5.8（7 语 CER/WER < 4%，ja 9%，ko ~6%） |
| **多任务 tokenizer 监督** | ASR / SER / LID / AED / SA 五任务标注 | Table 3：365K / 48K / 85K / 21K / 11K = **530K h** |

##### D. 训练侧数据管线（配置级）

```yaml
# 过滤
filter:  max_length: 40960, min_length: 0, token_max_length: 200, token_min_length: 1
# 特征提取
feat_extractor: n_fft: 1024, num_mels: 80, hop_size: 256, win_size: 1024, fmin: 0, fmax: 8000, center: False
# 重采样
resample: resample_rate: 24000   # v1 yaml 为 22050
# 截断（GAN 训练）
truncate: truncate_length: 24576   # 必须是 hop_size 的整数倍
# 说话人嵌入
parse_embedding: normalize: True
# 动态 batch
batch: batch_type: 'dynamic', max_frames_in_batch: 2000
sort:  sort_size: 500
shuffle: shuffle_size: 1000
# padding
padding: use_spk_embedding: False   # SFT 时改 True
```

##### E. 已知的数据侧局限（论文 Limitations）

- **歌唱生成能力有限**（singing）
- **日语 CER 偏高（9%）**：汉字→假名转换 + 多音字导致
- **韩语 CER 偏高（~6%）**：数据量与质量有限
- **hard samples（罕见词、绕口令）**：DiffRO 收益明显低于常规样本

##### F. 数据合规提示

- 论文**未逐项列出数据集名称**，数据构成**待核实**
- 但 CosyVoice 3 以 **Apache-2.0** 发布，说明其数据链路上不存在 CC-BY-NC 类约束数据集（对比 F5-TTS 因 Emilia 的 NC 约束，权重只能 CC-BY-NC）
- **商用尽职调查建议**：若要在强合规场景（金融、政务）使用，建议向阿里确认训练数据构成，或自行做数据血缘审计


### 16. 微调支持

#### 微调支持

##### 官方支持路径

| 方式 | 支持情况 | 入口 |
|---|---|---|
| **SFT（全参数微调）** | ✅ 官方支持，README Roadmap 明确「release ... its training/inference script」 | `examples/libritts/` |
| **LoRA / PEFT** | ⚠️ **官方未明确提供**（**待核实**）；社区有第三方实现 | — |
| **SFT 说话人（v1 时代）** | ✅ `iic/CosyVoice-300M-SFT` 权重可直接用 | ModelScope |
| **Instruct 微调（v1 时代）** | ✅ `iic/CosyVoice-300M-Instruct` | ModelScope |
| **GRPO 后训练** | ✅ CosyVoice2 支持（2025/08，NVIDIA Yuekai Zhang 贡献） | `examples/` 下 |
| **DiffRO 后训练** | ✅ 三代训练脚本随 2025/12 权重一同发布 | 官方训练脚本 |

##### SFT 关键超参切换（配置级）

```yaml
# 预训练 → SFT 需要改三处
train_conf:
    optim_conf:
        lr: 0.001  →  1e-5
    scheduler: warmuplr  →  constantlr
    # 以及
padding:
    use_spk_embedding: False  →  True
```

##### 数据量预期

- 官方未给出 SFT 最小数据量。**参考同类模型**：VoxCPM 宣称「5~10 分钟音频即可适配特定说话人/语言/领域」；GLM-TTS 提供 LoRA 定制路径。CosyVoice 3 的具体数值**待核实**
- 实践中 zero-shot（3~10 秒参考音频）已能满足多数音色克隆需求，**SFT 主要解决的是「特定领域发音/风格固化」而非「音色」**

##### 微调的硬件门槛

- 0.5B 全参数 SFT：参考 `max_frames_in_batch: 2000`（dynamic batch），V100 16G 上 GAN 训练建议降到 1400。**推荐 A100 80G 起**
- 具体显存需求官方未给（**待核实**）

##### 微调注意事项（工程经验）

1. **不要照抄 `examples/libritts/cosyvoice/conf/cosyvoice.yaml` 作为三代配置**——那是 v1 的 `TransformerLM` 配置。三代用 `CosyVoice3LM`（Qwen2.5 基座），配置在预训练权重包内
2. **DiT 对量化敏感**，微调后若要量化部署，只对 LLM 做量化、DiT 保持 fp16
3. **SFT 后需做持续预训练才能保留指令/多语能力**（论文 Stage 3 的存在就是为了解决「SFT 后遗忘指令能力」这一问题）——这是一个容易被忽略的点

##### 不支持 / 弱支持

- **精确时长控制**：不提供 token 级时长指定（这是 IndexTTS-2 的差异化能力）
- **歌唱 / 歌声合成**：论文明确能力有限
- **多说话人长对话**：不是 CosyVoice 3 的设计目标（对比 VibeVoice 的 90 分钟 / 4 说话人）


### 17. 训练硬件门槛

#### 训练硬件

##### ⚠️ 官方未公开（待核实）

论文 §7 **未公布训练所用的 GPU 型号、卡数、GPU 小时、batch size、learning rate**。README 与 ModelScope 模型卡也未给训练硬件配置。

##### 可推断的量级（**以下为估算，非官方数值，仅供容量规划参考**）

| 阶段 | 数据规模 | 模型规模 | 粗估算力需求 | 依据 |
|---|---|---|---|---|
| Tokenizer 多任务监督训练 | 530K h | MinMo 基座 + 12 层 encoder | **大**（MinMo 本身是 140 万小时预训练的多模态模型） | 论文 §2.1 |
| Stage 1 预训练 | ~1,000K h | 0.5B LLM + 300M DiT | **极大** | 论文 §4.2 |
| Stage 2 DiffRO 后训练 | 中等 | 0.5B LLM（冻结 DiT / vocoder） | **中**（只需 LLM 前反向 + 奖励模型） | 论文 §2.2 的成本论证 |
| Stage 3 持续预训练 | 中等 | 0.5B | 中 | — |
| Stage 4 SFT | 小（分钟~小时级） | 0.5B | **小**（单卡 A100 80G 可跑） | 配置 `max_frames_in_batch: 2000` |
| GAN 声码器训练 | — | HiFT（小） | 小 | `max_frames_in_batch: 1400` @ V100 16G |

##### 微调/推理侧的硬件参考（有据可查的部分）

- **推理**：无官方数值。社区 MLX 变体在 **M2 Max** 上 RTF ≈ 0.5（4bit ~1.2GB；**来源为社区实测，官方未公布**）
- **训练数据切分上限**：音频片段 < 30s；`filter.max_length: 40960`
- **动态 batch**：`max_frames_in_batch: 2000`（V100 16G GAN 训练建议 1400）

##### 横向参照系（用于估算，来自同类开源模型的公开数字）

| 模型 | 公开的训练/微调硬件信息 |
|---|---|
| **Kokoro-82M** | 总计 **1000 A100-80G GPU 小时，约 $1000**（v0.19: 500h @ $0.8/h；v1.0: 500h @ $1.2/h） |
| **F5-TTS** | 提供完整 finetune 脚本，可在单卡消费级 GPU 上微调（社区常见 24GB） |
| **VoxCPM** | SFT/LoRA 微调，5~10 分钟音频即可；未公布训练硬件 |
| **CosyVoice 3** | **全部待核实** |

##### 结论

**CosyVoice 3 的训练侧信息透明度较低**，这对其「可二次训练性」是不利因素。若你的团队计划做**大规模预训练级别的二次开发**，CosyVoice 3 的复现难度高；若只做 **SFT / DiffRO 后训练**，官方脚本 + 单卡/多卡 A100 是可行的。


### 18. 语种与方言覆盖

#### 语言与方言覆盖

##### 9 种主要语言（README / HF 模型卡 `language:` 字段）

| 语言 | code | CV3-Eval CER/WER（CosyVoice3-0.5B） | + DiffRO 后 |
|---|---|---|---|
| 中文 | zh | **3.89**（CER） | **2.89** |
| 英文 | en | **5.24**（WER） | **3.68** |
| 日文 | ja | **10.4**（CER） | **5.15** |
| 韩文 | ko | **12.8**（CER） | **4.02** |
| 德文 | de | **7.41**（WER） | **4.51** |
| 西班牙文 | es | **4.25**（WER） | **2.99** |
| 法文 | fr | **12.9**（WER） | **8.56** |
| 意大利文 | it | **6.68**（WER） | **2.94** |
| 俄文 | ru | **6.77**（WER） | **3.79** |

> 来源：论文 Table 5（CV3-Eval 多语言测试集）。1.5B 变体：zh 3.91 / en 4.99 / ja 7.57 / ko 5.69 / de 6.43 / es 4.47 / fr 11.8 / it 10.5 / ru 6.64

**重要观察（论文强调）**：CosyVoice 3 是**唯一覆盖全部 9 种语言**的系统；F5-TTS / Spark-TTS / GPT-SoVITS 在许多语言上是 `–`（不支持）。

**DiffRO 的增益在低资源语言上极其显著**：ko 12.8 → 4.02（-69%），ja 10.4 → 5.15（-50%），it 6.68 → 2.94（-56%）。这与论文 §5.4「低资源语言相对提升 > 50%」一致。

##### 18+ 种中国方言/口音

**论文内部数字不一致**：Abstract 写 **18 Chinese dialects**，§4.2 写 **19 common accents or dialects**（Figure 3b 标题）。列举包括：

- anhui（安徽）、cantonese（粤语）、chongqing（重庆）、hebei（河北）、shandong（山东）、shanghai（上海）、sichuan（四川）、tianjin（天津）、xi'an（西安）、zhengzhou（郑州）
- Table 1 中另有 dongbei（东北）、minnan（闽南）等
- HF 模型卡列举：Guangdong（广东）、Minnan（闽南）、Sichuan（四川）、Dongbei（东北）、Shan3xi（山西）、Shan1xi（陕西）、Shanghai（上海）、Tianjin（天津）、Shandong（山东）、Ningxia（宁夏）、Gansu（甘肃）等

##### 跨语言与语种转换能力

| 场景 | 数据 | 来源 |
|---|---|---|
| **跨语言克隆**（cross-lingual zero-shot） | 支持。Table 7 中 en→ja 达 **1.03** WER（0.5B+DiffRO） | 论文 Table 7 |
| **zh2en** | WER **4.99**，SS 67.8，MOS 3.75（0.5B） | 论文 Table 8 |
| **en2zh** | WER **8.48**，SS 67.4，MOS 3.82（0.5B） | 论文 Table 8 |
| **单语说话人 → 多语** | 中英德西法意俄 CER/WER **全部 < 4%**；日语 CER **9%**；韩语 CER **~6%** | 论文 §5.8 |

##### 语言覆盖的横向位置

| 模型 | 语言数 | 备注 |
|---|---|---|
| Fish Audio S2 Pro | **80+** | Tier1: ja/en/zh；Tier2: ko/es/pt/ar/ru/fr/de |
| VoxCPM2 | **30** | + 9 种中国方言 |
| **CosyVoice 3** | **9 + 18/19 方言** | 深度优化中文与方言 |
| GLM-TTS | 中英为主 + 四川话/东北话 | 英文数据少于中文的一半 |
| IndexTTS-2.5 | 中英日西阿 | — |
| VibeVoice | 中英 | 其他语言会出意外输出 |
| F5-TTS | 中英（+ 社区多语种扩展） | Emilia 为主 |

##### 已知语言短板（论文自述）

1. **日语 CER 9%**：汉字→假名（kanji→kana）转换 + 多音字导致错误
2. **韩语 CER ~6%**：数据量与质量有限
3. **法文 WER 12.9**（DiffRO 后 8.56）：9 语言中最差

##### 生产建议

- **中文及方言场景**：CosyVoice 3 是当前开源最强选项之一
- **英法德意俄西**：覆盖但质量弱于中文，且法语明显偏弱
- **80 语言规模的多语言需求**：应转向 Fish Audio S2 Pro（但商用受限）或 VoxCPM2（Apache-2.0）
- **日韩场景**：需要 DiffRO 后的 RL 权重才有可用质量


### 19. 流式能力

#### 流式（Bi-Streaming）

##### 官方指标

| 指标 | 数值 | 来源 |
|---|---|---|
| **首包延迟（首包/首帧延迟）** | **150 ms** | README Key Features「achieves latency as low as 150ms while maintaining high-quality audio output」；HF 模型卡同 |
| 流式模式 | **双向流式**：text-in 流式 + audio-out 流式 | README；代码 `inference_bistream` |
| 质量损失 | 官方称「maintaining high-quality audio output」（流式与非流式同质量，这是 CosyVoice 2 的核心卖点，三代延续） | README / 论文 §1 |

##### 实现机制（代码级）

1. **LLM 侧**：`self.llm.forward_one_step(..., cache=cache)` 逐 token 自回归 + KV cache，每次只 forward 一个新 token
2. **FM 侧**：**chunk-aware flow matching**（CosyVoice 2 引入），按 chunk 增量渲染 mel，不需要等全部 token 生成完
3. **Bistream 调度**：`mix_ratio = [5, 15]`—— **每消费 5 个 text token，产出 15 个 speech token**。这是一个**比例调度**而非固定 chunk size
4. **旧路径**（v1 `TransformerLM`）：`forward_chunk(...)` + `att_cache/cnn_cache` + `required_cache_size=-1`
5. **vLLM 路径**：`SamplingParams(top_k=25, stop_token_ids=..., min_tokens=min_len, max_tokens=max_len)`，逐 token yield

##### 为什么 150ms 是可达的

- 25Hz token 率 → **每个 token 40ms**
- LLM 首 token 延迟 + FM chunk 缓冲 + vocoder 上采样延迟 ≈ 3~4 个 token 的时间 → ~150ms
- 相比之下，纯 NAR 模型（F5-TTS）需要等**整个文本**处理完才能开始出声，长文本首包延迟会随文本长度线性增长（F5-TTS 实测 253ms @ concurrency 2，但那是整句延迟而非首包）

##### 流式部署入口

| 组件 | 路径 |
|---|---|
| FastAPI 服务 | `runtime/python/fastapi/server.py` + `client.py` |
| gRPC 服务 | `runtime/python/grpc/server.py` + `client.py`（支持 `--max_conc 4`） |
| Docker | `runtime/python/Dockerfile` |
| vLLM | `vllm_example.py`（vLLM 0.11.x+ V1 engine 或 0.9.0 legacy） |
| Triton + TensorRT-LLM | `runtime/triton_trtllm/`（`docker compose up -d`） |

##### 与其他模型的流式能力对比

| 模型 | 流式 | 首包/首帧延迟 | 备注 |
|---|---|---|---|
| **CosyVoice 3** | ✅ 双向流式 | **150 ms**（官方） | — |
| VibeVoice-Realtime-0.5B | ✅ | **~300 ms** | 支持流式文本输入，长音频 ~10 分钟 |
| Fish Audio S2 Pro | ✅ | **~100 ms TTFA**（H200 + SGLang） | RTF 0.195，3000+ tokens/s |
| VoxCPM2 | ✅ | 未公布 | RTF ~0.30（4090）/ ~0.13（nano-vllm） |
| F5-TTS | ⚠️ 有限 | 253 ms 平均延迟（concurrency 2，L20，16 NFE） | 非 AR 架构，流式不自然 |
| IndexTTS-2 | ⚠️ 未明确 | 未公布 | AR 架构，可做但非重点 |

##### 流式注意事项

1. **RAS 采样在流式下更重要**：自回归流式生成一旦进入重复循环，用户会立刻听到明显的「卡带」，比非流式更致命
2. **mix_ratio [5, 15] 是可调的**：实时性优先时可以提高每轮产出的 speech token 数，但要防止「文本还没到，音频已生成完」的断流
3. **vLLM 版本约束严格**：README 明确「vLLM 0.11.x+ (V1 engine) 或 0.9.0 legacy」；**0.10.x 未测试**；< 0.9.0 不支持。建议**单独建 conda 环境**（`conda create -n cosyvoice_vllm --clone cosyvoice`）避免污染主环境


### 20. VAD 与后处理

#### VAD 与后处理

##### VAD（语音活动检测）

- **CosyVoice 3 本身不是 VAD 模型**，不提供独立的 VAD 端点
- **数据流水线中使用 VAD**：论文 §3 多语言数据流水线第 1 步为「语音检测与分割」，把音频切成 < 30s 的片段（**具体 VAD 模型未指明，待核实**；阿里生态常用 FSMN-VAD，来自 FunASR）
- **参考音频预处理**：若参考音频含静音/噪声，建议先用 VAD 裁剪——CosyVoice 的 zero-shot 克隆质量对参考音频质量敏感
- **生态方案**：生产链路通常搭配 **FunASR 的 FSMN-VAD**（流式 VAD，官方开源，Apache-2.0）或 Silero VAD

##### 内置后处理（作为 TTS 的一部分）

| 能力 | 说明 | 来源 |
|---|---|---|
| **文本归一化（TN）** | 「Supports reading of numbers, special symbols and various text formats **without a traditional frontend module**」——TN 能力内化进 LLM，不需要独立前端 | README Key Features |
| TN 后端可选项 | ① `ttsfrd`（阿里内部资源，需额外下载安装，cp310 Linux whl）② `WeTextProcessing`（wetext，默认回退） | README Install |
| **降噪（训练侧）** | MossFormer2（数据流水线第 2 步） | 论文 §3 |
| **音量归一化（训练侧）** | 除以 max × 0.6（数据流水线第 5 步） | 论文 §3 |
| **声码器内置限幅** | `audio_limit: 0.99`（HiFTGenerator 配置），防止波形削波 | `cosyvoice.yaml` |

##### 推理侧后处理（需自行实现的部分）

CosyVoice 3 **不内置**以下后处理，需在生产链路中自行补齐：

1. **音频格式转换**：输出为 24kHz wav，转 mp3/opus 需外部工具（ffmpeg / sox）
2. **响度归一化**：如需符合 EBU R128 / 平台响度标准（-16 LUFS 等），需外部处理（pyloudnorm / ffmpeg loudnorm）
3. **静音裁剪**：首尾静音需外部 VAD 处理
4. **流式 chunk 拼接**：bistream 模式下的 chunk 边界需要处理（尤其用 gRPC/FastAPI 时）

##### 参考音频预处理建议（工程实践）

```
参考音频 → [VAD 裁剪静音] → [降噪（可选）] → [音量归一化] → [3~10s 截断] → CosyVoice 3
```

- 参考音频建议 **3~10 秒**（官方最短时长**待核实**）
- 参考音频需配对应的**准确文本**（prompt_text）——文本错误会直接损害克隆质量
- 参考音频应无背景音乐、无混响

##### 评测/生产可用的后处理工具链（见 `eval_toolchain`）

CV3-Eval 提供 DNSMOS、MOS 预测、SS 计算等；生产侧常用 FunASR（ASR 回环校验）+ SpeechMOS。


### 21. 音色克隆

#### 零样本语音克隆

##### 机制

**In-Context Learning（上下文学习）**，而非 speaker embedding 注入：

```
lm_input = [sos_emb, text_token..., <|endofprompt|>, task_id_emb, prompt_speech_token_emb..., speech_token...]
                                                                  ↑
                                              参考音频的 FSQ token（25Hz）直接拼进上下文
```

- **三代推理路径（`CosyVoice3LM.inference`）不使用 CAM++ 的 192-d speaker embedding**（见 `audio_encoder` 字段）——这一点与前代不同，是重要的架构变更
- CAM++ 分支（`use_spk_embedding`）保留，主要用于 SFT 模式与 DiT 侧条件化
- **优势**：同时传递音色 + 韵律 + 语速 + 录音环境（因为传的是完整 token 序列而非压缩的 embedding）
- **代价**：上下文变长，长参考音频会增加 prefill 成本

##### 三种克隆模式

| 模式 | 输入 | 特点 |
|---|---|---|
| **zero-shot 克隆** | prompt 音频 + prompt 文本 + 目标文本 | 默认模式，无需微调 |
| **跨语言克隆** | prompt 语言 ≠ 目标语言 | 内置支持；en→ja WER 1.03（0.5B+DiffRO，Table 7） |
| **SFT 克隆** | 微调权重 / speaker id | 音色最稳，需训练 |

##### 关键指标（Speaker Similarity，SS）

| 测试集 | Fun-CosyVoice3-0.5B-2512 | _RL 版 | 说明 |
|---|---|---|---|
| test-zh SS | **78.0** | 77.4 | WavLM 口径（README 表） |
| test-en SS | **71.8** | 69.5 | WavLM 口径 |
| test-hard SS | **75.8** | 75.0 | WavLM 口径 |
| test-zh SS（论文 Table 4，0.5B） | 0.780（WavLM）/ **0.840**（ERes2Net） | 0.774 / 0.836 | 论文口径 |

> **口径警告**：同一模型 WavLM 与 ERes2Net 两个口径差 5~8 个点。跨模型比较必须统一口径。

**参照系（README 表，WavLM 口径）**：Human 79.6（test-zh SS 79.6? 实际 README 表中 Human 为 75.5）/ Seed-TTS 79.6 / MiniMax-Speech 78.3 / VoxCPM 77.2 / Index-TTS2 76.5 / GLM-TTS 76.1 / CosyVoice2 75.7 / F5-TTS 74.1 / Spark-TTS 66.0

**CosyVoice 3 的 test-zh SS 78.0 在所有开源模型中排名第一**（仅低于闭源的 Seed-TTS 79.6 与 MiniMax-Speech 78.3）。

##### 参考音频要求

| 项目 | 值 | 来源 |
|---|---|---|
| 最短参考音频 | **官方未公布**（社区经验 3~10 秒） | ⚠️ **待核实** |
| 是否需要 prompt 文本 | **是**（zero-shot 模式下强烈建议提供） | 代码 / README |
| 采样率要求 | 无强制（内部 resample 到 24kHz） | 配置 `resample_rate` |
| 跨语言 | 支持 | README |

##### 克隆质量短板

1. **DiffRO 会略降 SS**（zh: 78.0 → 77.4；en: 71.8 → 69.5）——reward hacking 的代价
2. **日语/韩语克隆质量弱**：ja CER 9%、ko CER ~6%
3. **hard samples 克隆稳定性差**：罕见词/绕口令

##### 合规红线

- Apache-2.0 **不覆盖被克隆者的声音权/人格权**。生产环境必须取得说话人书面授权
- README Disclaimer：「内容仅供学术用途...部分样例来自互联网，如侵犯权益请联系删除」
- 国内《民法典》第 1023 条明确保护声音权；生成式 AI 服务需遵守标识要求


### 22. 情感与风格控制

#### 情感与控制能力

##### 1. 指令控制（Instruct）

README Key Features：「**Instruct Support**: Supports various instructions such as languages, dialects, emotions, speed, volume, etc.」

- 指令数据量：**5,000 小时**（从 1,500 扩展而来），覆盖 **>100 种**说话风格类型（Table 1 列出 top-100）
- 类型：emotions（情感）、speed（语速）、voice tones（音调）、dialects（方言）、accents（口音）、role-playing（角色扮演）

**指令生成效果（论文 Table 14）**：

| 模型 | Expresso WER | Expresso SIM | Expresso MOS | Internal WER | Internal SIM | Internal MOS |
|---|---|---|---|---|---|---|
| CosyVoice 2 | 9.42 | 60.98 | 3.54 | 7.75 | 72.99 | 3.53 |
| **CosyVoice 3-1.5B** | 13.43 | **68.25** | 3.56 | **7.31** | **81.06** | 3.51 |

风格相似度（SIM）相对提升约 **11%**。**注意 Expresso 集上 WER 反而变差（9.42 → 13.43）**——情感指令会损害发音稳定性。

##### 2. 情感克隆准确率（论文 Table 9，emo2vec-large-plus 作分类器）

| 模型 | text-related happy/sad/angry | text-unrelated happy/sad/angry |
|---|---|---|
| **CosyVoice3-1.5B + DiffRO-EMO** | **0.98 / 0.68 / 0.84** | **0.98 / 0.50 / 0.68** |

**关键规律（论文明确指出）**：
- **happy 最容易**（0.98，几乎不受文本影响）
- **sad / angry 在 text-unrelated 场景下显著下降**（0.68 → 0.50，0.84 → 0.68）——即当文本本身不含情感线索时，模型难以凭空生成悲伤/愤怒
- 这是**全行业的共同短板**，不是 CosyVoice 独有

##### 3. 发音控制：Pronunciation Inpainting

- 中文拼音 + 英文 CMU 音素
- **RepMono + MixPhn 策略达 100% 修正率**（Table 13，详见 `context_and_io`）

##### 4. 时长控制

- ❌ **不提供精确时长控制**（token 级时长指定是 IndexTTS-2 的差异化能力）
- CosyVoice 3 只有 `min_token_text_ratio=2` / `max_token_text_ratio=20` 的软性区间约束
- 语速可通过指令（"speak faster"）间接控制

##### 5. 特殊 Token 控制

- 官方 Key Features 与控制能力通过 **instruct 文本指令**实现，**非**通过 `<|fl_speaker_clone|>` 这类硬编码 token
- 代码中的特殊 token：`sos / task_id / eos / fill_token`（基于 `speech_token_size` 偏移）+ 必须出现的 `<|endofprompt|> = 151646`
- 用户资料中的 `<|fl_speaker_clone|>` / `<|fl_speaker_instruct|>` / `<|fl_speaker_instruct2|>` / `<|fl_save_speaker|>` **未在当前代码的 `llm.py` 中找到**，可能位于 tokenizer/prompt 构造的其他文件、其他分支或早期版本（**待官方确认**）

##### 6. DiffRO-EMO：情感的可微奖励强化

- 把 **SER（语音情感识别）模型**作为奖励，直接强化情感表达
- 效果：Table 9 中 `+ DiffRO-EMO` 达到 text-unrelated happy 0.98 / sad 0.50 / angry 0.68
- **副作用（论文承认）**：EMO 奖励**可能损害发音**（reward hacking）

##### 7. 与其他模型的情感能力对比

| 模型 | 情感控制方式 | 特点 |
|---|---|---|
| **CosyVoice 3** | 文本指令 + DiffRO-EMO 奖励 | 5,000h 指令数据 + >100 种风格；sad/angry 在 text-unrelated 下偏弱 |
| **IndexTTS-2** | **音色-情感解耦**（GRL 梯度反转层）+ Qwen3 软指令 + 情感参考音频 | 可分别指定「音色 prompt」与「情感 prompt」，情感控制最灵活 |
| **Fish Audio S2 Pro** | **15,000+ 自然语言 tag**（`[whisper]`、`[excited]`、`[angry]`...），**sub-word 级细粒度** | 最细粒度；支持 `[pause]`、`[inhale]`、`[chuckle]` 等副语言 |
| VoxCPM2 | Voice Design（自然语言描述生成全新音色）+ style guidance | 可无参考音频造音色 |
| GLM-TTS | GRPO 多奖励（CER/SIM/Emotion/Laughter） | 显式优化笑声与呼吸 |
| HiggsAudio-v2 | 情感识别 + 多角色对话 | 社区评价情感表达最强（非官方评测） |
| F5-TTS | ❌ 基本无 | — |
| Kokoro-82M | ❌ 无 | 固定音色与风格 |

**选型结论**：若情感/副语言控制是核心需求，**IndexTTS-2（解耦）或 Fish S2 Pro（tag 粒度）优于 CosyVoice 3**；若只是需要「高兴/难过/语速/方言」这类中粒度指令，CosyVoice 3 足够且商用无风险。


### 23. 鲁棒性

#### 鲁棒性

##### 1. In-the-wild 场景（这是 CosyVoice 3 的核心卖点）

论文标题 *Towards In-the-wild Speech Generation* 直接点明目标。相比前代在数据、领域、文本格式三方面的扩展（论文 §1 明确列出 CosyVoice 2 的五大局限）：

| CosyVoice 2 的局限 | CosyVoice 3 的对策 |
|---|---|
| language coverage（语言覆盖窄） | 9 语言 + 18/19 方言 |
| domain diversity（领域单一） | 电商、导航、金融、教育等 + 对话/演讲/唱歌 |
| data volume（数据量小） | 1 万 h → 100 万 h |
| text formats（文本格式单一） | TN + ITN 构造多格式；self-training 造 rare cases |
| post-training techniques（无后训练） | **DiffRO** |

##### 2. 硬性质量指标：hard 测试集（绕口令、罕见词）

| 模型 | test-hard CER | test-hard SS |
|---|---|---|
| Human | — | — |
| Seed-TTS（闭源） | 7.59 | 77.6 |
| **Fun-CosyVoice3-0.5B-2512** | **6.71** | 75.8 |
| **Fun-CosyVoice3-0.5B-2512_RL** | **5.44** | 75.0 |
| CosyVoice2 | 6.83 | 72.4 |
| Index-TTS2 | 7.12 | 75.5 |
| VibeVoice-1.5B | — | — |
| F5-TTS | 8.67 | 71.3 |
| VoxCPM | 8.87 | 73.0 |
| Spark-TTS | — | — |

**CosyVoice 3 的 test-hard CER 6.71（RL 版 5.44）优于所有列出的开源与闭源模型**（最好的闭源 Seed-TTS 为 7.59）。**这是 CosyVoice 3 最强的单项能力。**

论文 Table 6（hard 样本细节）：

| 模型 | hard-zh WER | hard-zh SS | DNSMOS | hard-en WER | SS | DNSMOS |
|---|---|---|---|---|---|---|
| CosyVoice3-0.5B+DiffRO | **8.26** | 77.8 | 3.80 | **7.60** | 73.9 | 3.95 |
| CosyVoice3-1.5B+DiffRO | 9.06 | 78.2 | 3.81 | 7.56 | 74.6 | 3.95 |

##### 3. 音频质量（DNSMOS）

DNSMOS **3.80 ~ 3.95**（hard 测试集，论文 Table 6）。对比：真实录音 DNSMOS 通常 3.9~4.1，**CosyVoice 3 的合成音质已接近真实录音水平**。

##### 4. 主观 MOS（论文 Figure 4 文字描述）

- **中文**：CosyVoice 2 / 3-0.5B / 3-1.5B 三者相近，但**都低于人类**；三者 MOS **均 > 4.45**
- **英文**：CosyVoice 2 低于人类；**3-0.5B 与人类持平**；**3-1.5B 高于人类**
- 排序：**1.5B > 0.5B > 2.0**

##### 5. 已知鲁棒性短板（论文 Limitations 自述）

| 短板 | 具体表现 |
|---|---|
| **歌唱能力有限** | singing 生成能力不足 |
| **日语 CER 9%** | kanji→kana 转换 + 多音字 |
| **韩语 CER ~6%** | 数据量与质量有限 |
| **法文 WER 12.9** | 9 语言中最差 |
| **DiffRO 对 hard samples 提升小** | 罕见词、绕口令上收益明显低于常规样本（因为奖励模型本身不准） |
| **DiffRO reward hacking** | SS 略降；EMO 奖励可能损害发音 |
| **情感在 text-unrelated 下退化** | sad 0.68 → 0.50，angry 0.84 → 0.68 |

##### 6. 相对前代的提升幅度（论文 §5，相对 CosyVoice 2）

| 测试集 | 相对提升 |
|---|---|
| test-zh | **44%** |
| test-en | **51%** |
| test-hard | **26%** |

##### 7. 工程鲁棒性（开源社区反馈）

- **RAS 采样**（2024/08 引入）专门解决自回归 TTS 的「吞字、复读、死循环」稳定性问题，是生产可用性的关键
- 依赖环境脆弱（见 `known_issues`）——这是**工程鲁棒性的主要风险点**，而非模型质量


### 24. ASR 指标

#### N/A（说明）

**CosyVoice 3 是 TTS 模型，不是 ASR 模型**，因此不存在标准意义上的「ASR 指标」。但本条目中与 ASR 相关的能力与数值有三处，一并记录：

##### 1. 语音 Tokenizer 的 ASR 监督任务（训练侧）

- ASR 是 tokenizer 五大监督任务之一，**365K 小时**（中英日韩俄法德 7 语），占 tokenizer 训练数据的 69%
- 基座 **MinMo** 是 140 万小时预训练的多模态语音理解模型
- 作用：让 FSQ token 承载语义内容，**直接决定 TTS 的 CER/WER 表现**

##### 2. DiffRO 的奖励模型是 ASR-like 模型（训练侧）

- 训练一个 **Token2Text（ASR 架构）模型**，用其后验概率 $\log P_{ASR}(Y|\tilde{\mu})$ 作为可微奖励
- 这不是一个对外可用的 ASR，但可以视为「内嵌的 ASR 校验器」

##### 3. TTS 内容一致性的「ASR 评测」（评测侧）

TTS 领域的 CER/WER 本质是**用 ASR 模型回环识别合成音频，再与原文对比**：

| 测试集 | 指标 | 使用的 ASR 模型 | CosyVoice 3-0.5B 数值 |
|---|---|---|---|
| Seed-TTS test-zh | **CER** | 未指明具体模型（**待核实**） | 1.21%（README）/ 1.16%（论文 Table 4） |
| Seed-TTS test-en | **WER** | 同上 | 2.24%（README）/ 2.02%（论文 Table 4） |
| Seed-TTS test-hard | **CER** | 同上 | 6.71%（README）/ 6.08%（论文 Table 4） |
| CV3-Eval 多语言 | CER/WER | 未指明 | 见 `languages` 字段 |
| 内部 30 语基准（VoxCPM2 报告） | WER/CER | Gemini 3.1 Flash Lite API | 未包含 CosyVoice 3 |

> ⚠️ **README 表格（1.21 / 2.24 / 6.71）与论文 Table 4（1.16 / 2.02 / 6.08）数值不同**，推测是不同批次/不同 checkpoint 或不同评测口径。**引用时务必注明来源**。

##### 4. 若需要配对的 ASR 能力

CosyVoice 3 不含 ASR 端点。同生态的 ASR 方案：

| 模型 | 说明 | 许可证 |
|---|---|---|
| **FunASR** | 阿里工业 ASR，50+ 语言，说话人分离，流式 | Apache-2.0 |
| **Fun-ASR-Nano** | 端到端 LLM ASR，31 语言，热词，vLLM 流式 | 待核实 |
| **SenseVoice** | 超快 ASR + 情感 + 音频事件检测 | Apache-2.0 |
| **VibeVoice-ASR** | 长音频 60 分钟单遍，50+ 语言，Who/When/What 结构化 | MIT |

##### 5. 常见的 ASR 回环校验工具链

生产环境做 TTS 质量监控时，通常用独立 ASR 回环检查「合成音频是否读对原文」：

- **FunASR**（Paraformer / SenseVoice）— 中文场景首选
- **faster-whisper** / **WhisperX** — 多语言
- **NeMo ASR**、**Seamless** — 训练数据流水线中使用
- 对齐工具：**ctc-forced-aligner**、**MFA**（Montreal Forced Aligner，CosyVoice 3 数据流水线中使用）
- 评分：**SpeechMOS**（UTMOS）、**DNSMOS**、**emo2vec**（情感分类）


### 25. TTS 指标

#### TTS 核心指标

##### A. Seed-TTS-Eval（WavLM 口径，来自官方 README，2025-12 版）

| Model | Open | Size | test-zh CER↓ | test-zh SS↑ | test-en WER↓ | test-en SS↑ | test-hard CER↓ | test-hard SS↑ |
|---|---|---|---|---|---|---|---|---|
| Human | - | - | 1.26 | 75.5 | 2.14 | 73.4 | - | - |
| Seed-TTS | ❌ | - | 1.12 | 79.6 | 2.25 | 76.2 | 7.59 | 77.6 |
| MiniMax-Speech | ❌ | - | 0.83 | 78.3 | 1.65 | 69.2 | - | - |
| F5-TTS | ✅ | 0.3B | 1.52 | 74.1 | 2.00 | 64.7 | 8.67 | 71.3 |
| Spark TTS | ✅ | 0.5B | 1.2 | 66.0 | 1.98 | 57.3 | - | - |
| CosyVoice2 | ✅ | 0.5B | 1.45 | 75.7 | 2.57 | 65.9 | 6.83 | 72.4 |
| FireRedTTS2 | ✅ | 1.5B | 1.14 | 73.2 | 1.95 | 66.5 | - | - |
| Index-TTS2 | ✅ | 1.5B | 1.03 | 76.5 | 2.23 | 70.6 | 7.12 | 75.5 |
| VibeVoice-1.5B | ✅ | 1.5B | 1.16 | 74.4 | 3.04 | 68.9 | - | - |
| VibeVoice-Realtime | ✅ | 0.5B | - | - | 2.05 | 63.3 | - | - |
| HiggsAudio-v2 | ✅ | 3B | 1.50 | 74.0 | 2.44 | 67.7 | - | - |
| VoxCPM | ✅ | 0.5B | 0.93 | 77.2 | 1.85 | 72.9 | 8.87 | 73.0 |
| GLM-TTS | ✅ | 1.5B | 1.03 | 76.1 | - | - | - | - |
| GLM-TTS RL | ✅ | 1.5B | 0.89 | 76.4 | - | - | - | - |
| **Fun-CosyVoice3-0.5B-2512** | ✅ | 0.5B | **1.21** | **78.0** | 2.24 | 71.8 | **6.71** | 75.8 |
| **Fun-CosyVoice3-0.5B-2512_RL** | ✅ | 0.5B | **0.81** | 77.4 | **1.68** | 69.5 | **5.44** | 75.0 |

##### B. 论文 Table 4（WavLM 口径，括号内为 ERes2Net 口径）

| Model | test-zh CER | test-zh SS | test-en WER | test-en SS | test-hard CER | test-hard SS |
|---|---|---|---|---|---|---|
| Human | 1.26 | 0.755 (0.775) | 2.14 | 0.734 (0.742) | - | - |
| CosyVoice 2 | 1.45 | 0.748 (0.806) | 2.57 | 0.652 (0.736) | 6.83 | 0.724 (0.776) |
| **CosyVoice 3-0.5B** | 1.16 | 0.780 (0.840) | 2.02 | 0.718 (0.790) | 6.08 | 0.758 (0.815) |
| **CosyVoice 3-0.5B RL** | **0.75** | 0.774 (0.836) | 1.76 | 0.695 (0.783) | **5.09** | 0.750 (0.809) |
| CosyVoice 3-1.5B | 1.12 | 0.781 (0.837) | 2.21 | 0.720 (0.789) | 5.83 | 0.758 (0.816) |
| CosyVoice 3-1.5B RL | **0.71** | 0.775 (0.836) | **1.45** | 0.695 (0.784) | 5.66 | 0.750 (0.810) |

> ⚠️ **README 与论文的数值不同**（如 zh CER 1.21 vs 1.16，hard CER 6.71 vs 6.08），引用时必须注明来源。差异可能来自不同 checkpoint 批次或评测脚本版本。

**相对 CosyVoice 2 的提升**：test-zh **44%**，test-en **51%**，test-hard **26%**

##### C. CV3-Eval 多语言（论文 Table 5）

| Model | zh | en | ja | ko | de | es | fr | it | ru |
|---|---|---|---|---|---|---|---|---|---|
| CosyVoice2 | 4.08 | 6.32 | 9.13 | 19.7 | – | – | – | – | – |
| +DiffRO | 3.00 | 4.72 | 6.36 | 5.14 | – | – | – | – | – |
| **CosyVoice3-0.5B** | 3.89 | 5.24 | 10.4 | 12.8 | 7.41 | 4.25 | 12.9 | 6.68 | 6.77 |
| **+DiffRO** | **2.89** | **3.68** | **5.15** | **4.02** | **4.51** | **2.99** | **8.56** | **2.94** | **3.79** |
| CosyVoice3-1.5B | 3.91 | 4.99 | 7.57 | 5.69 | 6.43 | 4.47 | 11.8 | 10.5 | 6.64 |
| +DiffRO | 3.01 | 3.71 | 5.27 | 4.01 | 3.93 | 3.26 | 8.09 | 2.72 | 4.11 |

**CosyVoice 3 是唯一覆盖全部 9 种语言的系统**（F5-TTS / Spark-TTS / GPT-SoVITS 多语言为 `–`）。

##### D. Hard 样本（论文 Table 6）

| Model | hard-zh WER | hard-zh SS | DNSMOS | hard-en WER | hard-en SS | DNSMOS |
|---|---|---|---|---|---|---|
| CosyVoice3-0.5B+DiffRO | 8.26 | 77.8 | 3.80 | 7.60 | 73.9 | 3.95 |
| CosyVoice3-1.5B+DiffRO | 9.06 | 78.2 | 3.81 | 7.56 | 74.6 | 3.95 |

##### E. 跨语言（论文 Table 7，WER%）

- en→ja：**1.03**（0.5B+DiffRO）/ 1.06（1.5B+DiffRO）
- to-zh 最低 5.09（1.5B+DiffRO）
- 完整 12 个交叉条件数值见原表

##### F. zh2en / en2zh（论文 Table 8）

| Model | en2zh WER | en2zh SS | en2zh MOS | zh2en WER | zh2en SS | zh2en MOS |
|---|---|---|---|---|---|---|
| CosyVoice3-0.5B | 8.48 | 67.4 | 3.82 | 4.99 | 67.8 | 3.75 |
| CosyVoice3-1.5B | 8.01 | 66.9 | 3.83 | 4.32 | 66.4 | 3.77 |

##### G. 主观 MOS（论文 Figure 4 文字描述）

- **中文**：CosyVoice 2 / 3-0.5B / 3-1.5B 三者接近，**都低于人类**，三者均 **> 4.45**
- **英文**：CosyVoice 2 低于人类；**3-0.5B 与人类持平**；**3-1.5B 高于人类**
- 整体排序：**1.5B > 0.5B > 2.0**

##### H. 指令生成（论文 Table 14）

| Model | Expresso WER | Expresso SIM | Expresso MOS | Internal WER | Internal SIM | Internal MOS |
|---|---|---|---|---|---|---|
| CosyVoice 2 | 9.42 | 60.98 | 3.54 | 7.75 | 72.99 | 3.53 |
| CosyVoice 3-1.5B | 13.43 | 68.25 | 3.56 | 7.31 | **81.06** | 3.51 |

##### I. 发音修补修正率（论文 Table 13）

| 方法 | zh 修正率 | en 修正率 |
|---|---|---|
| **RepMono + MixPhn** | **100%** | **100%** |
| RepMono + CatPhn | 86.7% | 100% |
| RepAll + MixPhn | 69.2% | 72.7% |

##### J. 声码器质量基线（论文 Table 4）

| 系统 | test-zh CER |
|---|---|
| Human | 1.26 |
| **Vocoder Resyn.** | **1.27** |

→ 声码器引入的内容损失几乎可忽略（+0.01 CER）

##### K. 第三方交叉验证（VoxCPM2 报告，arXiv:2606.06928 的 Seed-TTS-eval 表）

| Model | Parameters | test-EN WER | test-ZH CER | test-ZH SIM | test-Hard CER | test-Hard SIM |
|---|---|---|---|---|---|---|
| CosyVoice3 | 0.5B | 2.02 | 1.16 | 78.0 | 6.08 | 75.8 |
| CosyVoice3 | 1.5B | 2.22 | 1.12 | 78.1 | 5.83 | 75.8 |

> 注：VoxCPM2 报告把 CosyVoice3 标为 ❌（非开源），这与阿里实际开源 0.5B 权重不符，属于该报告的标注口径问题（可能指 1.5B 未开源）。**en SIM 一栏 VoxCPM2 报告未给 CosyVoice3 数值。**

##### L. CosyVoice 3 在同表中的位置总结

| 维度 | CosyVoice 3 表现 | 排名 |
|---|---|---|
| **test-hard CER（最难）** | **5.44（RL）** | 🥇 **第一**（优于闭源 Seed-TTS 7.59） |
| **test-zh SS（音色相似度）** | 78.0 | 🥇 **开源第一**（仅低于 Seed-TTS 79.6 / MiniMax 78.3） |
| test-zh CER | 0.81（RL） | 🥈 仅次于 MiniMax 0.83，优于 Seed-TTS 1.12 |
| test-en WER | 1.68（RL） | 🥉 落后 MiniMax 1.65、F5-TTS...（F5 为 2.00，故 RL 版 1.68 更好） |
| test-en SS | 69.5（RL） | 中等（VoxCPM 72.9、Seed-TTS 76.2 更高） |
| 多语言覆盖 | 9 语言全覆盖 | 🥇 唯一全覆盖（1.5B/0.5B） |


### 26. 延迟与吞吐

#### 延迟与吞吐

##### ⚠️ 重要：官方指标的边界

| 指标 | 数值 | 来源 | 可信度 |
|---|---|---|---|
| **首包延迟** | **150 ms** | README Key Features / HF 模型卡 | ✅ 官方，但**未注明硬件与测试条件** |
| 双向流式 | text-in + audio-out | README | ✅ |
| **RTF** | **官方未公布** | — | ❌ **待核实** |
| **吞吐（tokens/s、并发数）** | **官方未公布** | — | ❌ **待核实** |
| **首 token 延迟（LLM）** | **未公布** | — | ❌ **待核实** |
| 显存占用 | **未公布** | — | ❌ **待核实** |

> 论文全文**没有报告 CosyVoice 3 的流式延迟（毫秒级）、chunk 大小、首包延迟、RTF 等任何数字**（只给了 token 率 25Hz）。150ms 这一数字仅出现在 README 与 HF 模型卡。

##### 可从架构推算的量

- **25 Hz token 率** → 每个 speech token **40 ms**
- **DiT 10 步 Euler**（NFE=10），CFG 0.7 —— 注意 **CFG 0.7 < 1.0**，这是一个「弱化条件引导」的设置，说明官方在质量与速度间取了速度侧
- `mix_ratio = [5, 15]`：bistream 模式每 5 个 text token 对应 15 个 speech token
- 声码器：HiFT 用极短窗 ISTFT（`n_fft=16, hop_len=4`）+ `[8,8]` 上采样实现 480×，是针对速度优化的 design

##### 社区实测（非官方，仅供参照）

| 平台 | 变体 | RTF | 来源 |
|---|---|---|---|
| Apple **M2 Max**（MLX） | CosyVoice3-0.5B 量化版 | **~0.5** | 用户提供的社区数据（**官方未确认，待核实**） |
| NVIDIA GPU（PyTorch） | 官方 0.5B | **待核实** | — |
| NVIDIA GPU（vLLM 0.11.x） | 官方 0.5B | **待核实** | — |
| NVIDIA GPU（Triton + TRT-LLM） | CosyVoice2 LLM | **4× 加速**（相对 HF transformers） | README「could give 4x acceleration comparing with huggingface transformers implementation」 |

##### 横向对比（有其他来源的数字）

| 模型 | 延迟/RTF | 测试条件 | 来源 |
|---|---|---|---|
| **CosyVoice 3** | 首包 **150 ms** | 未注明 | README |
| **Fish Audio S2 Pro** | TTFA **~100 ms**，RTF **0.195**，**3000+ tokens/s**（RTF<0.5 下） | 单卡 **H200** + SGLang | fish-speech README |
| **VibeVoice-Realtime-0.5B** | 首 audible 延迟 **~300 ms** | 未注明 | VibeVoice README |
| **VoxCPM2** | RTF **~0.30**（PyTorch）/ **~0.13**（Nano-vLLM） | **RTX 4090** | VoxCPM README |
| VoxCPM1.5 | RTF ~0.15 / ~0.08 | RTX 4090 | VoxCPM README |
| VoxCPM-0.5B | RTF ~0.17 / ~0.10 | RTX 4090 | VoxCPM README |
| **F5-TTS Base (Vocos)** | 平均延迟 **253 ms**，RTF **0.0394**（Client-Server）/ **0.0402**（Offline TRT-LLM）/ **0.1467**（Offline PyTorch） | 单卡 **L20**，26 组 prompt×text，**16 NFE**，concurrency 2/1 | F5-TTS README |
| Kokoro-82M | ~32× 实时（T4）；2~3× 实时（4 核 CPU） | T4 / CPU | 第三方评测（**非官方，待核实**） |

##### 加速路径（官方提供）

1. **vLLM**（0.11.x+ V1 engine 或 0.9.0 legacy）：PagedAttention、continuous batching、prefix caching
   - README 明确：< 0.9.0 **不支持**；0.10.x **未测试**
   - 建议独立环境：`conda create -n cosyvoice_vllm --clone cosyvoice`
2. **Triton + TensorRT-LLM**（`runtime/triton_trtllm/`，2025-08 NVIDIA Yuekai Zhang 贡献）：LLM 部分 **4× 加速**（相对 HF transformers）
3. **Docker + gRPC/FastAPI**：`--max_conc 4` 控制并发
4. **MLX 量化**（Apple Silicon）：4bit ~1.2GB / 8bit ~1.4GB / 8bit-full ~1.6GB / bf16 ~2.1GB
5. **HiFT 声码器**本身已针对速度优化（短窗 ISTFT）

##### 吞吐优化的工程建议（基于架构推断，非官方）

1. **LLM 与 DiT 分别部署**：LLM 是 AR（memory-bound，适合 vLLM/continuous batching）；DiT 是 10 步 ODE（compute-bound，适合 TensorRT / CUDA Graph / batch 推理）。**两段的最优 serving 策略不同，拆开部署收益最大**
2. **DiT 保持 fp16**，只对 LLM 做量化
3. **Prefix caching**：同一参考音频的 prompt speech token 可被多请求复用（vLLM 的 RadixAttention 天然支持）
4. **RAS 采样在批处理下的行为需单独验证**（`win_size=10` 的重复检测在 batch 内是否串扰，建议实测）

##### 与 F5-TTS 的范式差异（影响延迟选型）

- **F5-TTS（NAR）**：整句一次前向，RTF 0.04（TRT-LLM）极低，但**首包延迟随文本长度增长**，无法做真正的流式
- **CosyVoice 3（AR+FM）**：RTF 更高（更慢），但**首包延迟恒定 150ms**，可做真流式
- **选型判据**：离线批量合成（有声书、视频配音）→ F5-TTS 这类 NAR 吞吐碾压；实时对话 / 语音助手 → CosyVoice 3 这类 AR 流式


### 27. 成本模型

#### 成本模型

##### ⚠️ 官方未给出任何单位成本数据

阿里未公布 CosyVoice 3 的 RTF、显存、吞吐，因此**无法从官方数据推导单位成本**。下面给出**估算框架**与**可参照的第三方锚点**。

##### 自部署成本构成

```
总成本 = 模型权重存储 + GPU 算力（推理）+ 常驻显存 + 工程运维 + 数据/微调成本
```

| 成本项 | CosyVoice 3 的情况 | 量级（估算，待实测） |
|---|---|---|
| **权重存储** | 0.5B，bf16 约 **1~2 GB**（LLM + DiT + 声码器）；MLX 4bit 整包 ~1.2GB | 极低 |
| **常驻显存** | 官方未公布。参考同类 0.5B LLM+DiT 架构，推理约 **4~8 GB**（fp16） | ⚠️ **待实测** |
| **训练/微调** | SFT 单卡 A100 80G 可跑；预训练不可复现（超参未公开） | 高 |
| **依赖运维** | **较高**——Python 3.10 强约束、pynini 易失败、submodule 需 recursive、sox 系统依赖 | 中等 |
| **商用授权** | **$0**（Apache-2.0，无收入阈值、无规模阈值） | **零** |

##### 关键对比：授权成本

| 模型 | 商用授权成本 | 阈值 |
|---|---|---|
| **CosyVoice 3** | **$0** | 无 |
| VoxCPM2 / VoxCPM1.5 / VoxCPM-0.5B | $0 | 无 |
| GLM-TTS | $0 | 无 |
| VibeVoice | $0 | 无 |
| Kokoro-82M | $0 | 无 |
| **IndexTTS-2** | $0 **但有阈值**：月活 > 1 亿 或 年收入 > **10 亿人民币** 需单独书面授权 | 有 |
| **Fish Audio S2 / S2 Pro** | **需商业授权**（商用 = 单独书面协议，联系 business@fish.audio） | 所有商用 |
| **F5-TTS** | **不可商用**（权重 CC-BY-NC，因 Emilia 数据） | — |
| Spark-TTS | **不可商用**（CC BY-NC-SA） | — |
| OpenAudio S1-mini | **不可商用**（CC-BY-NC-SA-4.0） | — |

##### 第三方 API 价格锚点（用于「自建 vs 买 API」决策）

| 模型 | 计价方式 | 价格 | 来源 |
|---|---|---|---|
| Fish-Speech-1.5（SiliconFlow） | 按输入字节 | **$15 / 1M UTF-8 bytes** | SiliconFlow 模型页 |
| IndexTTS-2（SiliconFlow） | 按输入字节 | **$7.15 / 1M UTF-8 bytes** | SiliconFlow |
| Kokoro-82M（Together AI） | 按字符 | **$4 / 1M characters**（另一处标 $1/M） | Together AI 模型页 |
| Kokoro-82M（DeepInfra） | 按字符 | **$0.06 / 1 小时音频** ≈ $1/M 字符 | DeepInfra |
| ElevenLabs | 订阅 | $5~99/月 | 第三方评测 |
| OpenAI TTS | 按字符 | $15 / 1M 字符 | 第三方评测 |

**中文换算提示**：1 个汉字 = 3 个 UTF-8 字节。以 Fish-Speech 的 $15/M bytes 计，1M 汉字 ≈ $45。**中文场景下按字节计价对中文极不友好**，自建部署的性价比优势在中文场景更明显。

##### 自建 vs API 的粗略盈亏平衡

由于缺少官方 RTF，**无法给出精确的盈亏平衡点**。定性判断：

| 场景 | 建议 |
|---|---|
| 月合成量 < 100 万字符，且有多语言需求 | **买 API**（工程成本远低于自建） |
| 中文为主，月合成量 > 1000 万字符 | **自建 CosyVoice 3 / VoxCPM2**（Apache-2.0，授权零成本） |
| 强合规（数据不出域） | **必须自建**，CosyVoice 3 / VoxCPM2 / GLM-TTS 是唯三可行的高性能选项 |
| 需要 48kHz 录音室音质 | VoxCPM2（CosyVoice 3 上限 24kHz） |

##### 隐性成本（易低估）

1. **工程集成成本**：CosyVoice 的依赖环境脆弱（Python 3.10 + pynini + submodule），首次部署常需要 0.5~2 人日
2. **质量监控成本**：需自建 ASR 回环校验（CER）+ SS 评分 + DNSMOS 的监控链路
3. **音频后处理成本**：响度归一化、格式转换、静音裁剪
4. **合规成本**：说话人授权管理、AI 生成标识
5. **模型迭代成本**：CosyVoice 版本演进快（1.0→2.0→3.0 用了一年半），需预留升级窗口


### 28. 推理框架支持

#### 推理框架支持

##### 官方支持矩阵

| 框架/运行时 | 支持状态 | 版本约束 | 入口 |
|---|---|---|---|
| **PyTorch / HuggingFace-style 原生** | ✅ 默认 | — | `python example.py`、`webui.py` |
| **vLLM** | ✅ 官方支持 | **vLLM 0.11.x+（V1 engine）** 或 **0.9.0（legacy）**；**< 0.9.0 不支持**；**0.10.x 未测试** | `vllm_example.py` |
| **NVIDIA Triton + TensorRT-LLM** | ✅ 官方支持（2025-08，NVIDIA Yuekai Zhang 贡献） | `docker compose up -d` | `runtime/triton_trtllm/` |
| **FastAPI** | ✅ 官方支持 | — | `runtime/python/fastapi/{server,client}.py` |
| **gRPC** | ✅ 官方支持 | 支持 `--max_conc 4` | `runtime/python/grpc/{server,client}.py` |
| **Docker** | ✅ 官方支持 | `docker build -t cosyvoice:v1.0 .` | `runtime/python/Dockerfile` |
| **Gradio WebUI** | ✅ 官方支持 | `python3 webui.py --port 50000 --model_dir <dir>` | `webui.py` |
| **Apple MLX** | ⚠️ 社区支持 | 见 `model_variants` | 第三方 HF 仓库（`aufklarer/`、`aitytech/`） |
| **ModelScope Gradio Space** | ✅ 官方（2025/12 上线） | — | Fun-CosyVoice3-0.5B ModelScope 空间 |
| **llama.cpp / GGUF** | ❌ 未见官方支持 | — | ⚠️ **待核实**（VoxCPM2 有 llama.cpp-omni 支持） |
| **ONNX** | ❌ 未见官方支持 | — | ⚠️ **待核实**（F5-TTS 有 F5-TTS-ONNX；VoxCPM 有 VoxCPM-ONNX） |

##### vLLM 集成细节

```bash
conda create -n cosyvoice_vllm --clone cosyvoice
conda activate cosyvoice_vllm
# for vllm==0.9.0
pip install vllm==v0.9.0 transformers==4.51.3 numpy==1.26.4
# for vllm>=0.11.0
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4
python vllm_example.py
```

**注意（README 原文）**：「Older vllm version(<0.9.0) do not support CosyVoice inference, and versions in between (e.g., 0.10.x) are not tested.」「Notice that `vllm` has a lot of specific requirements. You can create a new env to in case your hardward do not support vllm and old env is corrupted.」

**vLLM 路径下的采样**：`SamplingParams(top_k=sampling, stop_token_ids=..., min_tokens=min_len, max_tokens=max_len)`，`sampling` 默认 25。**注意 vLLM 路径只传了 top_k，RAS 的重复检测逻辑在 vLLM 路径下可能不生效**（`win_size`/`tau_r` 未传给 SamplingParams）——这是一个需要实测确认的点。

##### Triton + TensorRT-LLM

- 加速对象：**CosyVoice2 的 LLM 部分**
- 收益：**4× 加速**（相对 HF transformers 实现）
- 启动：`cd runtime/triton_trtllm && docker compose up -d`
- 说明：官方 README 的部署说明与示例仍以 CosyVoice-300M / CosyVoice2 为主，**CosyVoice 3 的 TRT-LLM 支持程度待核实**

##### MLX（Apple Silicon）

- 变体：`CosyVoice3-0.5B-MLX-{4bit, 8bit, 8bit-full, bf16}`
- 转换要点：LLM 量化（4bit group_size=64 量化 attention proj / MLP / speech head）；**DiT 保持 fp16**（flow matching 对量化敏感）；HiFi-GAN fp16 + weight norm 折叠；Conv1d 权重从 `[out, in, kernel]` 转置为 `[out, kernel, in]`
- 生态：`speech-swift`（Apple SDK/CLI runtime）、`soniqo.audio`
- 性能：M2 Max RTF ~0.5（**社区数据，待核实**）

##### 生态对比（部署成熟度）

| 模型 | vLLM | TRT-LLM/Triton | FastAPI/gRPC | Docker | MLX/Apple | llama.cpp/ONNX | 文档 |
|---|---|---|---|---|---|---|---|
| **CosyVoice 3** | ✅（版本严格） | ✅（NVIDIA 贡献） | ✅ 两者都有 | ✅ | ⚠️ 社区 | ❌ | ✅ README + 中文社区 |
| **F5-TTS** | ⚠️ 社区 | ✅（Yuekai Zhang） | ⚠️ Gradio/CLI | ✅ ghcr.io | ✅ f5-tts-mlx | ✅ F5-TTS-ONNX | ✅ README |
| **VoxCPM2** | ✅ vLLM-Omni 官方 | ❌ | ✅（Nano-vLLM 自带 FastAPI） | ⚠️ | ⚠️ VoxCPMANE (ANE) | ✅ llama.cpp-omni / VoxCPM.cpp / VoxCPM-ONNX / voxcpm_rs | ✅ readthedocs 完整文档 |
| **Fish Audio S2 Pro** | ✅ vLLM-Omni recipe | ❌ | ✅ SGLang-Omni | ✅ | ❌ | ❌ | ✅ speech.fish.audio |
| **IndexTTS-2** | ✅ 仅 2.5 有 recipe | ❌ | ⚠️ WebUI + Python API | ❌ | ❌ | ❌ | ⚠️ README 为主 |
| **GLM-TTS** | ⚠️ 待核实 | ❌ | ⚠️ Gradio + CLI | ❌ | ❌ | ❌ | ⚠️ README |
| **VibeVoice** | ✅ 仅 ASR 侧 | ❌ | ⚠️（TTS 代码已下架） | ❌ | ❌ | ❌ | ⚠️ |

**结论**：CosyVoice 3 的**服务端部署生态是同级别中最完整的之一**（vLLM + Triton/TRT-LLM + FastAPI + gRPC + Docker 全都有，且由 NVIDIA 官方贡献过代码）。**端侧（移动端/嵌入式）生态是明显短板**——没有官方 llama.cpp/GGUF/ONNX 支持，对比 VoxCPM2 的 llama.cpp-omni / VoxCPM.cpp / ONNX / Apple ANE / Rust 全栈差距明显。


### 29. 资源需求

#### 资源需求

##### ⚠️ 核心问题：官方未公布显存与 RTF

CosyVoice 3 的 README、论文、ModelScope 模型卡**均未给出显存占用与推理速度数值**。以下数据来自架构推算、社区实测与同类模型参照，**标注了可信度**。

##### 权重体积（可信度高，来自 MLX 模型卡的组件拆分）

| 组件 | bf16/fp16 体积 | 4-bit 体积 | 量化建议 |
|---|---|---|---|
| LLM（Qwen2.5-0.5B，24L/896d/14Q-2KV） | ~1.0 GB（推算） | **467 MB** | ✅ 可量化（4bit/8bit） |
| DiT（22 层，1024d，16 头） | **634 MB**（fp16，MLX 卡实测） | — | ❌ **不建议量化**（flow matching 对量化敏感） |
| HiFi-GAN / HiFT 声码器 | **79 MB**（fp16，weight norm 折叠） | — | ✅ 可 fp16 |
| **合计（fp16/bf16）** | **~1.7~2.1 GB** | ~1.2 GB（4bit LLM） | — |

MLX 官方变体体积：4bit **~1.2 GB** / 8bit **~1.4 GB** / 8bit-full **~1.6 GB** / bf16 **~2.1 GB**

##### 显存需求（**推算，待实测**）

| 场景 | 估算显存 | 说明 |
|---|---|---|
| fp16 推理（batch=1，短句） | **~3~5 GB** | 权重 2GB + KV cache + mel/中间激活 |
| fp16 推理（流式） | **~3~5 GB** | 同上 |
| 4bit LLM + fp16 DiT | **~2~3 GB** | MLX 4bit 整包 1.2GB 的体积对应 |
| SFT 微调（0.5B 全参） | **~24~40 GB**（估算） | 参考同类 0.5B 模型 SFT；**强烈建议 A100 80G** |
| GAN 声码器训练 | **~16 GB**（V100 可跑） | `max_frames_in_batch: 1400` @ V100 16G |

> ⚠️ 上述推理显存为架构推算，**务必实测确认**。作为参照：VoxCPM-0.5B 官方 VRAM ~5 GB、VoxCPM1.5 ~6 GB、VoxCPM2 ~8 GB。

##### 系统依赖

| 项目 | 要求 | 备注 |
|---|---|---|
| **Python** | **3.10**（README 明确 `conda create -n cosyvoice -y python=3.10`） | 强约束；`ttsfrd` whl 为 cp310 |
| PyTorch | 由 requirements.txt 锁定（**版本较老**） | 见 `known_issues` |
| **pynini** | `conda install -y -c conda-forge pynini==2.1.5` | ⚠️ pip 安装易失败 |
| **sox** | `apt-get install sox libsox-dev` / `yum install sox sox-devel` | 系统级依赖 |
| **git submodule** | `git clone --recursive` 或 `git submodule update --init --recursive` | 网络失败需重试至成功 |
| CUDA | 需 NVIDIA GPU（Docker 用 `--runtime=nvidia`）；CUDA 版本由 torch 版本决定 | — |
| 磁盘 | 模型 ~2GB + 环境 ~10GB（含 conda + torch） | — |

##### 硬件生态适配

| 硬件 | 支持情况 | 说明 |
|---|---|---|
| **NVIDIA GPU（数据中心）** | ✅ 最佳 | vLLM + Triton/TRT-LLM 全支持 |
| **NVIDIA GPU（消费级，如 4090）** | ✅ 可行 | 显存估算 3~5GB，4090 24GB 绰绰有余 |
| **Apple Silicon (M 系列)** | ⚠️ 社区 MLX | 4bit ~1.2GB；M2 Max RTF ~0.5（待核实） |
| **AMD GPU (ROCm)** | ⚠️ **待核实** | 无官方说明 |
| **Intel GPU (XPU)** | ⚠️ **待核实** | 无官方说明 |
| **CPU 推理** | ⚠️ 理论上可行但极慢 | 无官方优化；**不推荐** |
| **移动端 / 嵌入式** | ❌ 无官方方案 | 对比 VoxCPM2 有 llama.cpp-omni / GGUF |

##### 与同类模型的资源对比

| 模型 | 参数量 | 官方 VRAM | 输出采样率 | 备注 |
|---|---|---|---|---|
| **CosyVoice 3-0.5B** | 0.5B | **待核实**（估算 3~5GB） | **24 kHz** | — |
| VoxCPM-0.5B | 0.6B | **~5 GB** | 16 kHz | — |
| VoxCPM1.5 | 0.8B | **~6 GB** | 44.1 kHz | — |
| VoxCPM2 | 2B | **~8 GB** | 48 kHz | — |
| F5-TTS Base | 0.3B | 未公布 | 24 kHz | L20 上 RTF 0.0394 (TRT-LLM) |
| Fish Audio S2 Pro | 4B | 未公布（SGLang on H200） | — | RTF 0.195 @ H200 |
| IndexTTS-2 | 1.5B | FP16 约 **7.8 GB**（用户提供，**待核实**） | — | — |
| Kokoro-82M | 82M | ~400 MB（内存，非显存） | 24 kHz | CPU 可跑 |
| HiggsAudio-v2 | 3B | 未公布 | 24 kHz | — |
| Spark-TTS | 0.5B | 未公布 | 16 kHz | — |

##### 容量规划建议

1. **先实测再规划**：官方数据的缺失意味着必须自建 benchmark（见 `eval_toolchain`）
2. **LLM 与 DiT 分开部署**：两段资源特性不同（AR 是 memory-bound + KV cache 增长；DiT 是 compute-bound + 固定开销），分开部署可以更精细地分配资源与扩缩容
3. **量化策略**：LLM 4bit/8bit + DiT fp16 + vocoder fp16，是 MLX 变体验证过的方案，可推广到 GPU（**GPU 上的具体收益待实测**）
4. **预留 2 倍余量**：自回归 TTS 的显存峰值出现在长文本 + 大 batch 时，KV cache 会随序列长度线性增长


### 30. 服务化栈

#### 服务化栈

##### 官方提供的服务组件

```
┌──────────────────────────────────────────────────────────────┐
│  客户端                                                        │
│  ├─ Python SDK（example.py / vllm_example.py）                 │
│  ├─ FastAPI client（runtime/python/fastapi/client.py）         │
│  ├─ gRPC client（runtime/python/grpc/client.py）               │
│  └─ Gradio WebUI（webui.py）                                   │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│  服务层（可选其一或组合）                                       │
│  ├─ FastAPI server   →  HTTP / JSON，适合 Web 集成             │
│  ├─ gRPC server      →  --max_conc 4，适合低延迟内部调用        │
│  ├─ vLLM server      →  PagedAttention + continuous batching  │
│  └─ Triton + TRT-LLM →  生产级 GPU 服务（NVIDIA 贡献）          │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│  模型层                                                        │
│  ├─ S3 Tokenizer（FSQ，25Hz）                                  │
│  ├─ CAM++ Speaker Encoder（192-d）                             │
│  ├─ LLM：Qwen2.5-0.5B（AR，RAS sampling）                      │
│  ├─ DiT Flow Matching（22 层，Euler 10 步，CFG 0.7）           │
│  └─ HiFT / HiFi-GAN 声码器（24kHz）                            │
└──────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│  容器化：Docker（runtime/python/Dockerfile）                   │
│  编排：docker compose（triton_trtllm）                          │
└──────────────────────────────────────────────────────────────┘
```

##### 部署命令（官方）

```bash
# 1) Docker 构建
cd runtime/python && docker build -t cosyvoice:v1.0 .

# 2) gRPC 服务（max_conc 控制并发）
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c   "cd /opt/CosyVoice/CosyVoice/runtime/python/grpc &&    python3 server.py --port 50000 --max_conc 4 --model_dir iic/CosyVoice-300M && sleep infinity"
cd grpc && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>

# 3) FastAPI 服务
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 /bin/bash -c   "cd /opt/CosyVoice/CosyVoice/runtime/python/fastapi &&    python3 server.py --port 50000 --model_dir iic/CosyVoice-300M && sleep infinity"
cd fastapi && python3 client.py --port 50000 --mode <sft|zero_shot|cross_lingual|instruct>

# 4) Triton + TensorRT-LLM
cd runtime/triton_trtllm && docker compose up -d

# 5) vLLM
conda create -n cosyvoice_vllm --clone cosyvoice && conda activate cosyvoice_vllm
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4
python vllm_example.py

# 6) WebUI
python3 webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M
```

##### 推理模式（client `--mode` 参数）

| mode | 说明 |
|---|---|
| `sft` | 使用 SFT 微调说话人 |
| `zero_shot` | 零样本克隆（默认） |
| `cross_lingual` | 跨语言克隆 |
| `instruct` | 指令控制 |

##### 生产化需要自行补齐的部分

| 缺口 | 说明 | 建议 |
|---|---|---|
| **鉴权 / 限流** | 官方 server 无任何认证 | 前置 API Gateway（Kong / APISIX / Nginx） |
| **OpenAI 兼容接口** | 官方不提供 `/v1/audio/speech` | 自行封装（VoxCPM2 通过 vLLM-Omni 原生提供，Fish S2 Pro 通过 SGLang-Omni 提供） |
| **多实例负载均衡** | 无 | K8s HPA / Triton 的 model ensemble |
| **Prometheus 监控** | 无 | 自行埋点（RTF、首包延迟、CER、错误率） |
| **请求队列 / 优先级** | 无 | 自行实现 |
| **音频格式转换** | 输出 wav | ffmpeg / 后置服务 |
| **水平扩缩容** | 无官方 K8s manifest | 自行编写 |
| **模型热更新** | 无 | 自行实现 |

##### 推荐的三种生产架构

###### 架构 A：中小规模 + 快速上线
```
FastAPI server（官方）× N → Nginx 负载均衡 → 客户端
```
- 优点：改动最小，直接复用官方代码
- 缺点：无 continuous batching，GPU 利用率低

###### 架构 B：大规模 + 高吞吐（推荐）
```
              ┌─→ vLLM (LLM 部分，PagedAttention + continuous batching)
客户端 → API GW┤
              └─→ DiT + vocoder 服务（Triton/TRT-LLM 或 TorchServe）
```
- 优点：两段分别按最优策略优化；prefix caching 可复用参考音频的 prompt token
- 缺点：需要自行拆分模型与实现跨服务调度，**官方未提供开箱即用的拆分方案**

###### 架构 C：NVIDIA 全栈（已有 Triton 经验时）
```
Triton Inference Server + TensorRT-LLM backend（官方 docker compose）
```
- 优点：LLM 部分 4× 加速；Triton 天然支持 dynamic batching、模型 ensemble、监控
- 缺点：TRT-LLM 需要编译 engine，模型更新成本高；官方示例以 CosyVoice2 为主，**CosyVoice 3 支持度待核实**

##### 与竞品 serving 生态对比

| 模型 | 原生 OpenAI 兼容 | 生产级 serving | 容器化 | 监控 |
|---|---|---|---|---|
| **CosyVoice 3** | ❌ 需自封 | ✅ Triton + TRT-LLM | ✅ Docker | ❌ |
| **VoxCPM2** | ✅ `vllm serve openbmb/VoxCPM2 --omni` | ✅ vLLM-Omni + Nano-vLLM | ⚠️ | ❌ |
| **Fish S2 Pro** | ✅ SGLang-Omni | ✅ SGLang + vLLM-Omni | ✅ | ❌ |
| IndexTTS-2 / 2.5 | ❌ | ⚠️ vLLM recipe（仅 2.5） | ❌ | ❌ |
| F5-TTS | ❌ | ✅ Triton + TRT-LLM | ✅ ghcr.io | ❌ |

**结论**：CosyVoice 3 的 serving 栈**组件齐全但缺少「开箱即用的生产层」**（鉴权、OpenAI 兼容、监控、编排）。VoxCPM2 与 Fish S2 Pro 通过 vLLM-Omni / SGLang-Omni 提供了更现代的开箱体验。选择 CosyVoice 3 意味着**需要投入更多工程工作在生产化封装上**。


### 31. 硬件生态

#### 硬件生态

##### 支持的硬件平台

| 平台 | 支持度 | 细节 | 来源 |
|---|---|---|---|
| **NVIDIA GPU（数据中心：A100/H100/L20 等）** | ✅ 最佳 | vLLM + Triton + TensorRT-LLM 全支持；Triton TRT-LLM 由 NVIDIA Yuekai Zhang 官方贡献 | README |
| **NVIDIA GPU（消费级：RTX 3090/4090）** | ✅ 可行 | 估算显存 3~5GB，4090 24GB 完全够；**具体 RTF 待实测** | 推算 |
| **Apple Silicon（M1~M4）** | ⚠️ 社区支持 | MLX 量化变体（4bit ~1.2GB / 8bit ~1.4GB / 8bit-full ~1.6GB / bf16 ~2.1GB）；生态含 `speech-swift`（Apple SDK/CLI）、`soniqo.audio` | HF 第三方仓库 |
| **AMD GPU (ROCm)** | ⚠️ **待核实** | 无官方说明（F5-TTS README 明确列出 ROCm 支持与版本矩阵） | — |
| **Intel GPU (XPU / IPEX)** | ⚠️ **待核实** | 无官方说明 | — |
| **CPU（x86）** | ⚠️ 理论可行 | 无官方优化，速度不可接受；**不推荐** | — |
| **移动端 / 嵌入式** | ❌ 无方案 | 无 GGUF / llama.cpp / ONNX / CoreML 官方导出 | — |
| **国产 AI 芯片（昇腾等）** | ⚠️ **待核实** | 无官方说明 | — |

##### 关键硬件优化点

1. **Triton + TensorRT-LLM（NVIDIA 官方贡献，2025-08）**
   - 加速对象：CosyVoice2 的 LLM 部分
   - 收益：**4× 加速**（相对 HF transformers）
   - 路径：`runtime/triton_trtllm/`，`docker compose up -d`

2. **vLLM（0.11.x+ V1 engine）**
   - 特性：PagedAttention、continuous batching、CUDA Graph、prefix caching
   - 严格版本约束：0.11.x+ 或 0.9.0；< 0.9.0 不支持；0.10.x 未测试

3. **MLX 量化（Apple Silicon）**
   - 分层策略：LLM 4bit（group_size=64，量化 attention projections / MLP / speech head）；**DiT 保持 fp16**；HiFi-GAN fp16 + weight norm 折叠
   - 理由（模型卡原文）：「Flow: fp16 (flow matching is sensitive to quantization)」
   - Conv1d 权重需从 PyTorch `[out, in, kernel]` 转置为 MLX `[out, kernel, in]`

4. **声码器的速度优化设计**
   - HiFT 用极短窗 ISTFT（`n_fft=16, hop_len=4`）+ `[8,8]` 上采样实现 **480×** 上采样
   - 相比逐样本上采样的 HiFi-GAN，计算量大幅下降

##### 与竞品的硬件生态对比

| 模型 | NVIDIA | Apple Silicon | AMD | Intel | CPU/端侧 | 说明 |
|---|---|---|---|---|---|---|
| **CosyVoice 3** | ✅✅✅ | ⚠️ MLX 社区 | ❓ | ❓ | ❌ | 服务端强，端侧弱 |
| **VoxCPM2** | ✅✅ | ✅ ANE (VoxCPMANE) | ✅ (vLLM-Omni) | ✅ (vLLM-Omni) | ✅✅ llama.cpp-omni / VoxCPM.cpp / ONNX / Rust | **全平台覆盖最广** |
| **F5-TTS** | ✅✅ Triton/TRT-LLM | ✅ f5-tts-mlx | ✅ ROCm（README 有版本矩阵） | ✅ XPU/IPEX | ✅ F5-TTS-ONNX | 多平台文档最清晰 |
| **Fish S2 Pro** | ✅✅ SGLang/vLLM-Omni | ❌ | ❓ | ❓ | ❌ | H200 上 RTF 0.195 |
| IndexTTS-2 | ✅ | ❌ | ❌ | ❌ | ❌ | 仅 NVIDIA |
| VibeVoice | ⚠️（TTS 代码已下架） | ❌ | ❌ | ❌ | ⚠️ ASR 侧有 VibeASR.cpp (CPU) | — |

##### 选型含义

- **服务端（云 / 机房）部署**：CosyVoice 3 硬件生态成熟，NVIDIA 全栈支持到位
- **端侧 / 边缘 / 端云一体**：CosyVoice 3 **明显不如 VoxCPM2**（后者有 llama.cpp-omni GGUF、VoxCPM.cpp、ONNX、Apple ANE、Rust 实现）与 F5-TTS（ONNX + MLX）
- **Apple Silicon 本地开发**：CosyVoice 3 有社区 MLX 方案可用（4bit 1.2GB），但**不是官方维护**，升级滞后风险高（VoxCPM2 同样有 VoxCPMANE 覆盖 ANE）
- **国产化 / 信创**：CosyVoice 3 无明确适配信息；F5-TTS 与 VoxCPM2 通过 vLLM-Omni 有更多后端可能性


### 32. 集成范式

#### 集成模式

##### 模式 1：嵌入 Python 应用（最简单）

```python
from cosyvoice.cli.cosyvoice import CosyVoice  # 具体类名以代码为准
# 参考 example.py
```
- 适用：批处理脚本、研究、原型验证
- 缺点：无法并发，进程内耦合

##### 模式 2：HTTP 微服务（FastAPI，官方）

```
业务服务 ──HTTP POST──> CosyVoice FastAPI server ──> wav bytes
```
- 适用：Web 应用、多语言技术栈集成
- 官方客户端：`runtime/python/fastapi/client.py`
- 需自行补齐：鉴权、限流、超时、重试

##### 模式 3：gRPC 微服务（官方，低延迟）

```
业务服务 ──gRPC──> CosyVoice gRPC server（--max_conc 4）──> 流式 audio chunks
```
- 适用：内部服务间调用、流式场景、对延迟敏感
- 官方客户端：`runtime/python/grpc/client.py --mode <sft|zero_shot|cross_lingual|instruct>`

##### 模式 4：vLLM 服务（高吞吐）

```
业务服务 ──> vLLM server（PagedAttention + continuous batching）──> token stream ──> DiT + vocoder
```
- 适用：高并发、多租户
- 入口：`vllm_example.py`
- 注意：vLLM 只加速 LLM 部分，DiT 与 vocoder 需另起服务

##### 模式 5：Triton + TensorRT-LLM（NVIDIA 生产栈）

```
业务服务 ──HTTP/gRPC──> Triton Inference Server ──TRT-LLM backend──> LLM (4× 加速)
```
- 适用：已有 Triton 基础设施的团队
- 启动：`cd runtime/triton_trtllm && docker compose up -d`

##### 模式 6：Docker 一体化（官方）

```bash
docker run -d --runtime=nvidia -p 50000:50000 cosyvoice:v1.0 ...
```
- 适用：快速验证、单机部署
- 缺点：不好水平扩展

##### 推荐的语音 Agent 集成架构（结合 ASR + LLM + TTS）

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────────┐
│ VAD      │──>│ ASR       │──>│ LLM      │──>│ TTS          │
│(FSMN-VAD)│   │(FunASR /  │   │(业务 LLM) │   │(CosyVoice3)  │
│          │   │ SenseVoice)│   │          │   │ 流式 150ms   │
└──────────┘   └───────────┘   └──────────┘   └──────┬───────┘
                                                      │
                                              流式返回 audio chunks
```

- **全链路延迟预算参考**：VAD ~50ms + ASR ~200~300ms + LLM TTFT ~300~800ms + TTS 首包 **150ms**（CosyVoice 3 官方）
- CosyVoice 3 的 150ms 在全链路中占比不高，**瓶颈通常在 ASR 与 LLM**

##### 与 ASR 生态的同栈优势

CosyVoice 与 FunASR / SenseVoice / Fun-ASR-Nano 同属阿里 FunAudioLLM 家族，**共用大量基础设施（FunASR / FunCodec / WeNet）**：
- 依赖复用：同一套 conda 环境可同时跑 ASR 与 TTS
- 数据格式复用：都基于 parquet + kaldi-style 数据处理
- 中文场景的 G2P / TN 处理一致

**这是选择 CosyVoice 3 而非 VoxCPM2 / Fish 的一个实际工程理由**：如果你的 ASR 已经在用 FunASR，TTS 选 CosyVoice 3 可以显著降低运维复杂度。

##### 集成时的关键注意点

1. **`<|endofprompt|>`（id 151646）必须出现在输入文本中**——代码里有 assert，漏了会直接报错
2. **prompt 文本要准确**：zero-shot 模式下 prompt_text 的错误会直接损害克隆质量
3. **参考音频预处理**：VAD 裁剪 + 降噪 + 3~10s 截断
4. **流式 chunk 拼接**：gRPC/FastAPI 流式返回时需处理 chunk 边界
5. **输出是 24kHz wav**：转其他格式需外部处理
6. **并发控制**：官方 gRPC 用 `--max_conc 4`，默认值偏低，需按实测调整
7. **RAS 在 batch 场景下的行为需实测**：`win_size=10` 的重复检测逻辑在批处理/流式下的表现未经验证


### 33. 技术栈组件

#### 技术栈组件清单（选型时可逐项对照）

##### 模型与算法层

| 组件 | 选型 | 备注 |
|---|---|---|
| 语音离散化 | **FSQ**（有限标量量化），25Hz，码本 6561 | 取代 VQ，解决码本坍塌 |
| 语音 tokenizer 基座 | **MinMo**（140 万小时预训练多模态理解模型） | 12 层 Transformer + RoPE + FSQ |
| Tokenizer 监督任务 | ASR / SER / LID / AED / SA（530K h） | 5 任务联合 |
| 文本 tokenizer | **Qwen2.5 BPE** | 必须含 `<|endofprompt|>` = 151646 |
| 主 LLM | **Qwen2.5-0.5B**（24L / 896d / 14Q-2KV GQA） | 1.5B 变体仅论文 |
| 声学模型 | **22 层 DiT + Conditional Flow Matching**（1024d / 16 头 / Euler 10 步 / CFG 0.7） | 100M → 300M |
| 声码器 | **HiFT / HiFi-GAN**（NSF 8 谐波 + F0 预测器 + ISTFT n_fft=16/hop=4） | 480× 上采样 |
| 说话人编码器 | **CAM++**（192-d） | 三代推理主路径**未使用**，靠 in-context |
| 采样策略 | **RAS**（top_k=25, top_p=0.8, win_size=10, tau_r=0.1） | 防重复/死循环 |
| 后训练 | **DiffRO**（Gumbel-Softmax + 可微奖励 + token 级 KL） | 多任务奖励：ASR/SER/MOS/AED |

##### 数据与训练层

| 组件 | 选型 |
|---|---|
| 数据格式 | **parquet** |
| 特征提取 | mel-spectrogram（n_fft=1024, num_mels=80, hop_size=256, win_size=1024） |
| F0 提取 | `compute_f0`（hop_size=256） |
| 批处理 | dynamic batch（`max_frames_in_batch: 2000`） |
| 优化器 | Adam（lr 0.001 预训练 / 1e-5 SFT） |
| 调度器 | warmuplr（预训练）/ constantlr（SFT） |
| 分布式 | 基于 PyTorch 生态（**具体 DeepSpeed/FSDP 配置待核实**） |
| GAN 训练（声码器） | MPD + MRD 多判别器，lr 2e-4，constantlr |
| 数据流水线 | VAD 切分(<30s) → MossFormer2 降噪 → ASR 转录(WER<15%) → MFA 标点调整 → 音量归一化 → 异常过滤 |

##### 推理与服务层

| 组件 | 选型 | 版本 |
|---|---|---|
| 深度学习框架 | **PyTorch** | requirements.txt 锁定（较老） |
| LLM 推理引擎 | **HuggingFace transformers** / **vLLM** | vLLM 0.11.x+ 或 0.9.0 |
| 生产 GPU 服务 | **NVIDIA Triton + TensorRT-LLM** | — |
| API 框架 | **FastAPI** / **gRPC** | — |
| 容器化 | **Docker** + docker compose | — |
| 前端 Demo | **Gradio** | — |
| Apple Silicon | **MLX**（社区） | — |
| 文本归一化 | **WeTextProcessing**（默认）/ **ttsfrd**（可选） | ttsfrd 限 cp310 Linux |
| G2P / 音素 | 内置（中文拼音 / 英文 CMU 音素，用于发音修补） | — |

##### 评测层（见 `eval_toolchain`）

##### 语言与运行时

| 项目 | 值 |
|---|---|
| 语言 | Python 3.10（强约束） |
| 关键依赖 | pynini==2.1.5（conda-forge）、sox、git submodule（--recursive） |
| 配置系统 | **OmegaConf**（`!new:` / `!ref <...>` 语法的 yaml） |
| 代码来源 | 大量复用 FunASR / FunCodec / Matcha-TTS / AcademiCodec / WeNet |

##### 可替换/可插拔点（二次开发友好度）

| 组件 | 可替换性 | 难度 |
|---|---|---|
| 声码器 | ✅ 高（v1 代码支持 BigVGAN；接口清晰） | 低 |
| 采样策略 | ✅ 高（`sampling: Callable` 外部注入） | 低 |
| 文本前端 / TN | ✅ 高（ttsfrd ↔ wetext 可切换） | 低 |
| DiT 步数 / CFG | ✅ 高（推理参数） | 低 |
| 说话人编码器 | ⚠️ 中（三代推理主路径已不用，改动涉及训练） | 中 |
| 语音 tokenizer | ❌ 低（需重训 LLM 与 DiT） | 高 |
| LLM 主干 | ❌ 低（需完整重训） | 高 |

**结论**：CosyVoice 3 的**推理侧可插拔性良好**（采样、声码器、步数、TN 都可换），但**模型核心（tokenizer + LLM）几乎不可替换**。若你的场景需要自定义 tokenizer 或更换 LLM 主干，成本极高。


### 34. 评测工具链

#### 评测工具链

##### 官方评测集：CV3-Eval

- **仓库**：https://github.com/FunAudioLLM/CV3-Eval
- **发布时间**：2025/07（README Roadmap：「release Fun-CosyVoice 3.0 eval set」）
- **用途**：CosyVoice 3 官方的多维度评测基准，包括多语言 CER/WER、hard 样本、跨语言、情感克隆、发音修补、指令生成等
- **覆盖维度**（对应论文 Table 4~14）：
  - Seed-TTS-Eval（test-zh / test-en / test-hard）
  - 多语言 CV3-Eval（9 语言）
  - hard 样本（hard-zh / hard-en + DNSMOS）
  - 跨语言（to-zh / to-en / to-ja / to-ko 交叉，12 个条件）
  - zh2en / en2zh（WER / SS / MOS）
  - 情感克隆准确率（text-related / text-unrelated）
  - 发音修补修正率
  - 指令生成（Expresso / Internal 集）

##### 论文使用的评测模型

| 指标 | 工具/模型 | 引用 |
|---|---|---|
| **SS（说话人相似度）** | **WavLM-based** 说话人确认模型（表外数值）/ **ERes2Net**（表内括号数值） | [47] ERes2Net, [49] WavLM |
| **CER / WER** | ASR 模型（**论文未指明具体模型**，待核实） | — |
| **情感分类** | **emo2vec-large-plus** | Table 9 |
| **音频质量** | **DNSMOS** | Table 6 |
| **主观质量** | **MOS** 人工评分（Figure 4） | — |
| **音高 / 对齐（训练数据处理）** | **MFA**（Montreal Forced Aligner） | §3 |

##### 社区常用评测工具（生产自建 benchmark 时可用）

| 工具 | 用途 | 备注 |
|---|---|---|
| **FunASR**（Paraformer / SenseVoice） | 中文 ASR 回环 CER | Apache-2.0，与 CosyVoice 同生态 |
| **faster-whisper / WhisperX** | 多语言 ASR 回环 WER | CosyVoice 数据流水线中用了 Faster-Whisper |
| **NeMo ASR / Seamless** | 多语言 ASR | 同上 |
| **ctc-forced-aligner** | 强制对齐（时长/韵律分析） | F5-TTS 生态使用 |
| **SpeechMOS (UTMOS)** | 自动 MOS 预测 | F5-TTS 致谢列表 |
| **DNSMOS** | 音频质量（噪声/MOS 分） | CosyVoice 3 论文使用 |
| **emo2vec** | 情感分类 | CosyVoice 3 论文使用 |
| **WavLM / ERes2Net / CAM++** | 说话人相似度 | SS 计算的三种口径 |
| **pyloudnorm** | 响度归一化校验 | 生产后处理 |

##### ⚠️ 评测口径的三个坑（跨模型对比时必看）

1. **SS 口径差异最大**：同一模型 WavLM 与 ERes2Net 差 **5~8 个点**（如 CosyVoice3-0.5B test-zh SS：0.780 vs 0.840）。**必须统一口径**
2. **README 与论文数值不一致**：test-zh CER 1.21（README）vs 1.16（论文 Table 4）；test-hard CER 6.71 vs 6.08。推测为不同 checkpoint 或评测脚本版本
3. **第三方报告的标注错误**：VoxCPM2 报告把 CosyVoice3 标为「❌ 非开源」（实际 0.5B 已开源），且 en SIM 一栏缺值。**引用第三方表时需回溯原始来源**

##### 自建 benchmark 建议清单

| 维度 | 指标 | 工具 | 建议样本量 |
|---|---|---|---|
| 内容准确性 | CER（中）/ WER（英） | FunASR / Whisper | ≥ 500 句/语言 |
| 音色相似度 | SS（余弦相似度） | WavLM 或 ERes2Net（**固定一个**） | ≥ 200 句 × 20 说话人 |
| 音频质量 | DNSMOS / UTMOS | SpeechMOS | 全量 |
| 韵律自然度 | MOS 人工评分 | 众包 | ≥ 100 句 × 20 评测者 |
| 稳定性 | 重复/吞字率 | ASR 回环 + 文本比对 | ≥ 1000 句 |
| 延迟 | 首包延迟、RTF | 自埋点 | ≥ 100 次 |
| 吞吐 | QPS、并发数 | 压测工具 | — |
| 鲁棒性 | hard 集（绕口令、生僻字、中英混排、数字） | 自建 | ≥ 200 句 |
| 情感准确率 | 情感分类准确率 | emo2vec | ≥ 300 句 |

##### 复现官方结果的路径

```
1. git clone https://github.com/FunAudioLLM/CV3-Eval
2. 下载 Fun-CosyVoice3-0.5B-2512（base 与 RL 两个 checkpoint）
3. 按 CV3-Eval 脚本跑 test-zh / test-en / test-hard
4. 用 WavLM 或 ERes2Net 算 SS（注意与官方口径对齐）
5. 用 FunASR / whisper 算 CER / WER（**注意：官方未指明 ASR 模型，结果会有偏差**）
```


### 35. 已知坑与限制

#### 已知问题与坑

##### A. 环境安装（**发生频率最高**）

| 问题 | 表现 | 解决方案 |
|---|---|---|
| **Python 版本强约束** | README 明确 `conda create -n cosyvoice -y python=3.10` | 严格用 **3.10**；`ttsfrd` 的 whl 是 `cp310`，进一步锁死 |
| **pynini 安装失败** | pip 安装 WeTextProcessing 时 pynini 编译失败（需 C++ 编译） | `conda install -y -c conda-forge pynini==2.1.5` **（必须用 conda，不要用 pip）** |
| **git submodule 拉取失败** | Matcha-TTS 等第三方代码缺失 | `git clone --recursive`；失败则反复执行 `git submodule update --init --recursive` 直到成功 |
| **sox 兼容性问题** | 音频处理报错 | Ubuntu: `apt-get install sox libsox-dev`；CentOS: `yum install sox sox-devel` |
| **requirements.txt 依赖版本较老** | 与新显卡/新 CUDA 不兼容 | 需在 3.10 环境内安装；升级依赖可能破坏其他组件 |
| **vLLM 版本冲突** | vLLM 会强改 transformers/numpy 版本，污染主环境 | README 建议：`conda create -n cosyvoice_vllm --clone cosyvoice` **单独建环境** |
| **vLLM 版本窗口极窄** | < 0.9.0 不支持；0.10.x 未测试 | 用 **0.11.x+** 或 **0.9.0** |
| **ttsfrd 安装（可选）** | 需手动解压 + 装两个 whl | ```cd pretrained_models/CosyVoice-ttsfrd/ && unzip resource.zip -d . && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl``` |

##### B. 模型/算法层面的已知局限（来自论文）

| 局限 | 说明 |
|---|---|
| **歌唱能力有限** | 论文 Limitations 明确 |
| **日语 CER 9%** | kanji→kana 转换 + 多音字 |
| **韩语 CER ~6%** | 数据量与质量有限 |
| **法文 WER 12.9** | 9 语言中最差（DiffRO 后 8.56） |
| **DiffRO reward hacking** | SS 略降（zh 78.0→77.4，en 71.8→69.5）；EMO 奖励可能损害发音 |
| **DiffRO 对 hard samples 提升小** | 罕见词、绕口令收益明显低于常规样本 |
| **情感在 text-unrelated 下退化** | sad 0.68→0.50，angry 0.84→0.68 |
| **1.5B 权重未开源** | 论文中的 1.5B / 1.5B-RL 结果无法复现 |
| **训练超参未公开** | batch size / lr / GPU 小时全无，预训练级复现不可能 |

##### C. 文档与代码不一致（**二次开发高风险区**）

| 不一致 | 说明 |
|---|---|
| **`examples/libritts/cosyvoice/conf/cosyvoice.yaml` 是 v1 配置** | 该文件描述 `TransformerLM`（ConformerEncoder 6 层 + TransformerEncoder 14 层 16 头），**不是 CosyVoice 3 的 `CosyVoice3LM`（Qwen2.5）配置**。照抄会出错 |
| **README 与论文指标数值不同** | test-zh CER 1.21（README）vs 1.16（论文）；test-hard 6.71 vs 6.08 |
| **`fl_*` 特殊 token 未在代码中找到** | 用户资料中的 `<|fl_speaker_clone|>` 等未出现在 `cosyvoice/llm/llm.py`；代码用的是 `sos/task_id/eos/fill_token` 偏移体系 + `<|endofprompt|>=151646` |
| **方言数量 18 vs 19** | 论文 Abstract 写 18，§4.2 写 19 |
| **部署文档仍以旧模型为例** | README 的 Docker/TRT-LLM 示例多用 `iic/CosyVoice-300M`，CosyVoice 3 的支持度需自行验证 |

##### D. 代码级的硬约束（会直接报错）

1. **`<|endofprompt|>`（151646）必须存在**：
   ```python
   # NOTE temporary hardcode, 151646 is <|endofprompt|> token
   assert 151646 in text, '<|endofprompt|> not detected ...'
   ```
   缺失会直接 assert 失败
2. **`sort_size` 必须 < `shuffle_size`**：配置 `sort_size: 500` < `shuffle_size: 1000`
3. **GAN 训练的 `accum_grad` 必须为 1**：配置注释「in gan training, accum_grad must be 1」
4. **`truncate_length: 24576` 必须是 hop_size 的整数倍**

##### E. 工程/运维层面的坑

| 问题 | 说明 |
|---|---|
| **官方未公布 RTF、显存、吞吐** | 容量规划必须自建 benchmark |
| **无 OpenAI 兼容接口** | 需自行封装 `/v1/audio/speech` |
| **官方 server 无鉴权/限流/监控** | 生产化需前置 API Gateway + 自埋点 |
| **gRPC 默认 `--max_conc 4` 偏低** | 需按实测调整 |
| **vLLM 路径下 RAS 可能不生效** | `SamplingParams` 只传了 `top_k`，`win_size`/`tau_r` 未传入（**需实测确认**） |
| **DiT 对量化敏感** | 量化会明显掉质量；**只量化 LLM，DiT 保持 fp16** |
| **无端侧部署方案** | 无 GGUF / llama.cpp / ONNX 官方支持 |
| **版本迭代快** | 1.0(2024-07) → 2.0(2024-12) → 3.0(2025-12)，约每半年一代，需预留升级窗口 |

##### F. 合规风险

| 风险 | 说明 |
|---|---|
| **声音权 / 人格权** | Apache-2.0 不覆盖被克隆者的授权。国内《民法典》第 1023 条保护声音权 |
| **AI 生成标识** | 生成式 AI 服务需遵守内容标识要求 |
| **训练数据构成不透明** | 论文未列出数据集名称（虽 Apache-2.0 说明无 NC 约束数据，但强合规场景建议做数据血缘审计） |
| **`ttsfrd` 资源许可未明** | 该资源的许可证需单独确认 |
| **README Disclaimer** | 「内容仅供学术用途...部分样例来自互联网」 |

##### G. 社区反馈的高频问题（待进一步核实）

- 参考音频质量对克隆效果影响极大（噪声、混响、背景音乐都会显著劣化）
- 长文本生成的稳定性依赖 RAS，极端情况下仍可能出现重复
- 中英混读场景下偶发音素错误（可用发音修补解决）


### 36. 社区与文档

#### 社区与文档

##### 官方渠道

| 渠道 | 链接 | 说明 |
|---|---|---|
| **GitHub** | https://github.com/FunAudioLLM/CosyVoice | 主仓库；Sub-license: Apache-2.0 |
| **arXiv 论文** | https://arxiv.org/abs/2505.17589 | v1 2025-05-23，v2 2025-05-27 |
| **CV3-Eval 评测集** | https://github.com/FunAudioLLM/CV3-Eval | 官方评测基准 |
| **Demo 页** | https://funaudiollm.github.io/cosyvoice3/ | 试听样例 |
| **Hugging Face** | https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | 权重 |
| **ModelScope** | https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | 权重 + Gradio 空间（2025/12 上线） |
| **钉钉交流群** | README 中的二维码 | 官方中文社区 |
| **GitHub Issues** | https://github.com/FunAudioLLM/CosyVoice/issues | 官方问题反馈 |

##### 生态位（FunAudioLLM 家族）

| 项目 | 定位 |
|---|---|
| **FunASR** | 工业 ASR，50+ 语言，说话人分离，流式 |
| **Fun-ASR-Nano** | 端到端 LLM ASR，31 语言，热词，vLLM 流式 |
| **SenseVoice** | 超快 ASR + 情感 + 音频事件检测 |
| **FunClip** | AI 视频剪辑 |
| **CosyVoice** | TTS（本条目） |

##### 文档质量评估

| 维度 | 评价 |
|---|---|
| **快速上手** | ✅ 良好（README 的 Install + Basic Usage 清晰） |
| **API 文档** | ⚠️ 一般（主要靠 `example.py` 与 `vllm_example.py` 示例，无完整 API reference） |
| **训练文档** | ⚠️ 一般（指向 `examples/libritts`，但配置是 v1 的） |
| **部署文档** | ✅ 较好（Docker / FastAPI / gRPC / TRT-LLM 都有命令示例） |
| **架构文档** | ⚠️ 依赖论文（README 无架构说明，代码注释较少） |
| **中文资料** | ✅ 丰富（ModelScope 社区、钉钉群、大量中文技术博客） |
| **配置文档** | ❌ 缺失（yaml 配置项无逐项说明） |

**对比**：VoxCPM2 有完整的 **readthedocs 文档站**（quickstart / usage guide / cookbook / finetuning / FAQ / models/version_history），文档完整度明显优于 CosyVoice 3。

##### 社区活跃度信号

- **NVIDIA 官方贡献**（2025/08，Yuekai Zhang 贡献 Triton TRT-LLM runtime + GRPO 训练）——第三方硬件厂商主动贡献是生态健康度的强信号
- **社区 MLX 量化变体**：`aufklarer/CosyVoice3-0.5B-MLX-{4bit,8bit,bf16}`、`aitytech/CosyVoice3-0.5B-MLX-4bit` 等多个第三方仓库
- **S3Tokenizer 的 PyTorch 重实现**：`xingchensong/S3Tokenizer`（社区复现）
- **被第三方引用**：VoxCPM2 官方致谢「CosyVoice for the Flow Matching-based LocDiT implementation」——说明 CosyVoice 的 flow matching 实现已成为行业参考实现
- **VibeVoice 继承自 Qwen2.5-1.5B**，与 CosyVoice 3 同属 Qwen 系 TTS

##### 维护节奏

| 时间 | 事件 |
|---|---|
| 2024/07 | v1.0 |
| 2024/08 | RAS + 流式 |
| 2024/09 | 300M-25Hz + 变声 |
| 2024/12 | v2.0 |
| 2025/05 | v3 论文 + vLLM 支持 |
| 2025/07 | CV3-Eval |
| 2025/08 | Triton TRT-LLM + GRPO |
| **2025/12** | **v3 权重 + 训练/推理脚本 + ModelScope 空间** |

**约每 6 个月一个重大版本**——迭代快，但**也意味着需要预留升级窗口**。

##### 引用方式（官方 bibtex）

```bibtex
@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu
          and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}

@inproceedings{lyu2025build,
  title={Build LLM-Based Zero-Shot Streaming TTS System with Cosyvoice},
  author={Lyu, Xiang and Wang, Yuxuan and Zhao, Tianyu and Wang, Hao and Liu, Huadai and Du, Zhihao},
  booktitle={ICASSP 2025},
  pages={1--2},
  year={2025},
  organization={IEEE}
}
```

##### 二次开发建议的社区入口顺序

1. GitHub Issues（搜索已有问题，命中率较高）
2. ModelScope 模型页评论区（中文问题响应更好）
3. 钉钉官方群（README 二维码）
4. `examples/libritts/` 下的脚本（**注意配置是 v1 的**）


### 37. 适用场景与选型建议

#### 最适合的场景

##### ✅ 强烈推荐

| 场景 | 理由 |
|---|---|
| **中文 + 中国方言的 TTS** | 9 语言 + 18/19 方言，test-zh SS 78.0 为开源第一；粤语/四川话/东北话/上海话等覆盖最全 |
| **对发音准确率极度敏感的场景**（新闻播报、有声书、教育、金融播报） | test-hard CER **5.44（RL 版）优于所有开源与闭源模型**；DiffRO 专门优化 CER/WER；**优先用 `_RL` 权重** |
| **需要零样本克隆 + 商用授权干净** | Apache-2.0，无收入/规模阈值；对比 IndexTTS-2（bilibili 协议）、Fish（需商业授权）、F5-TTS（不可商用） |
| **实时语音助手 / 对话式 AI** | 双向流式 **150ms** 首包；25Hz token 率 + chunk-aware FM |
| **已使用 FunASR / SenseVoice 的技术栈** | 同属 FunAudioLLM 生态，依赖/数据格式/中文处理一致，运维复杂度低 |
| **强合规 / 数据不出域** | Apache-2.0 + 可完全私有化，是少数几个高性能且商用无风险的选择 |
| **需要发音精细控制（多音字、中英混读、专业术语）** | 拼音 + CMU 音素发音修补，**RepMono+MixPhn 达 100% 修正率** |
| **对稳定性要求高（避免吞字/复读）** | RAS 采样的重复感知机制是同类中少见的工程保障 |

##### ⚠️ 可用但有更好选择

| 场景 | 问题 | 更好的选择 |
|---|---|---|
| **需要精确时长控制（视频配音、口型同步）** | CosyVoice 3 只有 token 比例的软约束 | **IndexTTS-2**（token 级精确时长 + 可控/非可控双模式） |
| **音色-情感需要独立控制** | CosyVoice 3 靠文本指令，text-unrelated 情感下 sad/angry 明显退化 | **IndexTTS-2**（GRL 解耦，音色 prompt 与情感 prompt 分离） |
| **需要副语言/细粒度情感（笑声、呼吸、耳语）** | CosyVoice 3 无此能力 | **Fish Audio S2 Pro**（15,000+ tag，sub-word 级）；**GLM-TTS**（GRPO 显式优化笑声/呼吸） |
| **需要 48kHz 录音室音质** | CosyVoice 3 上限 24kHz | **VoxCPM2**（48kHz，Apache-2.0） |
| **需要 30~80 语言** | CosyVoice 3 只有 9 语言 | **VoxCPM2**（30 语言）/ **Fish Audio S2 Pro**（80+ 语言） |
| **长音频 / 多说话人对话（播客、有声剧）** | 无长上下文优化 | **VibeVoice**（90 分钟 / 4 说话人，但 TTS 代码已下架） |
| **端侧 / 边缘 / 移动端部署** | 无 GGUF/llama.cpp/ONNX 官方方案 | **VoxCPM2**（llama.cpp-omni / VoxCPM.cpp / ONNX / ANE / Rust）/ **Kokoro-82M**（CPU 实时） |
| **离线批量合成（吞吐优先）** | AR 架构吞吐不如 NAR | **F5-TTS**（NAR，RTF 0.04 @ TRT-LLM）——但权重不可商用 |
| **低资源/小显存** | 显存/RTF 未公开，容量规划困难 | **Kokoro-82M**（400MB，CPU 可跑） |
| **需要 LoRA 轻量化定制** | 官方未明确支持 | **VoxCPM**（官方 SFT + LoRA，5~10 分钟音频）/ **GLM-TTS**（LoRA） |

##### ❌ 不推荐

| 场景 | 理由 |
|---|---|
| **歌唱 / 歌声合成** | 论文明确能力有限 |
| **日语 / 韩语为主的产品** | ja CER 9%、ko CER ~6%，明显偏弱 |
| **法语为主的产品** | fr WER 12.9（DiffRO 后 8.56），9 语言中最差 |
| **需要完整复现论文、做预训练级二次开发** | 训练超参（batch/lr/GPU 小时）完全未公开；1.5B 权重未开源 |
| **想直接照抄 examples 配置微调三代模型** | examples 下的 yaml 是 v1 配置，会踩坑 |
| **无 GPU 环境** | 无 CPU 优化方案 |

##### 一句话选型结论

> **CosyVoice 3 是当前「中文 + 商用合规 + 高发音准确率 + 流式」四象限交叉点上的最优开源选择。**
> 它的核心护城河是 **DiffRO 后训练在 hard 集上的统治力**（CER 5.44，优于所有闭源）与 **Apache-2.0 的零商用门槛**；
> 主要短板是 **端侧生态缺失**、**官方性能指标（RTF/显存）不透明**、以及 **1.5B 权重未开源**。
>
> **如果需要精确时长 / 情感解耦 → 转向 IndexTTS-2（注意 bilibili 协议）；如果需要 48kHz / 30 语言 / 端侧 → 转向 VoxCPM2（Apache-2.0）；如果需要副语言 tag 级控制 → Fish Audio S2 Pro（需商业授权）。**

##### 落地检查清单

- [ ] 确认是否需要精确时长控制 → 是则换 IndexTTS-2
- [ ] 确认是否需要 >9 语言 → 是则换 VoxCPM2 / Fish S2
- [ ] 确认输出采样率要求 → 需要 48kHz 则换 VoxCPM2
- [ ] **同时测 base 与 `_RL` 两个权重**（CER vs SS 的权衡）
- [ ] 用 WavLM **或** ERes2Net 之一作为 SS 口径，全程不变
- [ ] 用统一 ASR（FunASR 或 whisper）做 CER/WER 回环
- [ ] 自建 RTF / 显存 / 并发 benchmark（官方数据缺失）
- [ ] Python 3.10 环境 + conda 装 pynini + `--recursive` clone
- [ ] 生产化补齐：鉴权、限流、监控、OpenAI 兼容封装
- [ ] 说话人授权与 AI 生成标识的合规流程


### 38. 参考来源

#### 权威来源

##### 一手来源（论文 / 官方仓库 / 官方模型卡）

1. **CosyVoice 3 技术报告**（arXiv:2505.17589，v1 2025-05-23，v2 2025-05-27）
   - 标题：*CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training*
   - 作者：Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, Jieping Ye
   - https://arxiv.org/abs/2505.17589 ｜ PDF: https://arxiv.org/pdf/2505.17589 ｜ HTML: https://arxiv.org/html/2505.17589v2
   - DOI: https://doi.org/10.48550/arXiv.2505.17589
   - **用于**：tokenizer 结构、多任务监督数据（Table 3）、DiffRO 全部公式、训练数据规模、Table 4~14 全部实验数值、数据流水线六步、Limitations

2. **官方 GitHub 仓库**
   - https://github.com/FunAudioLLM/CosyVoice
   - **用于**：Roadmap 时间线、Evaluation 表、Install 说明、部署命令、vLLM 版本约束、Triton TRT-LLM 说明、Ecosystem、Citation
   - 配置文件：https://github.com/FunAudioLLM/CosyVoice/blob/main/examples/libritts/cosyvoice/conf/cosyvoice.yaml
   - LLM 源码：https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/llm/llm.py

3. **Hugging Face 模型卡（Fun-CosyVoice3-0.5B-2512）**
   - https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
   - **用于**：Apache-2.0 许可、9 语言列表、方言列表、Key Features（pronunciation inpainting / TN / bi-streaming 150ms / instruct support）

4. **ModelScope 模型页**
   - https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

5. **CV3-Eval 官方评测集**
   - https://github.com/FunAudioLLM/CV3-Eval

6. **Demo 页**
   - https://funaudiollm.github.io/cosyvoice3/

##### 前代论文（版本演进参照）

7. **CosyVoice 1.0**（arXiv:2407.05407）— *Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens* ｜ https://arxiv.org/abs/2407.05407
8. **CosyVoice 2.0**（arXiv:2412.10117）— *CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models* ｜ https://arxiv.org/abs/2412.10117
9. **ICASSP 2025 工程论文** — *Build LLM-Based Zero-Shot Streaming TTS System with Cosyvoice*（Lyu et al.）

##### MLX 社区变体（用于组件体积与架构佐证）

10. **CosyVoice3-0.5B-MLX-4bit**（aitytech）— https://huggingface.co/aitytech/CosyVoice3-0.5B-MLX-4bit
    - **用于**：LLM 467MB(4bit) / DiT 634MB(fp16) / HiFi-GAN 79MB(fp16) 的组件体积拆分；Qwen2.5-0.5B (24L, 896d, 14Q/2KV heads)、22-layer DiT (1024d, 16 heads, 10 ODE steps) 的架构参数；「flow matching is sensitive to quantization」的量化建议
11. **CosyVoice3-0.5B-MLX-8bit** — https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-8bit
12. **CosyVoice3-0.5B-MLX-bf16** — https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-bf16
13. **S3Tokenizer 的 PyTorch 重实现**（社区）— https://huggingface.co/xingchensong/S3Tokenizer

##### 竞品对照来源（用于横向定位）

14. **VoxCPM2 技术报告**（arXiv:2606.06928）— https://arxiv.org/abs/2606.06928 ｜ 仓库：https://github.com/OpenBMB/VoxCPM
    - **用于**：Seed-TTS-eval 的第三方交叉验证表（含 CosyVoice3 0.5B/1.5B 数值）、CV3-eval 多语言表、VoxCPM2 的 RTF/VRAM/语言/许可
15. **IndexTTS2 论文**（arXiv:2506.21619）— https://arxiv.org/abs/2506.21619 ｜ 仓库：https://github.com/index-tts/index-tts
    - **用于**：IndexTTS-2 摘要（时长控制、情感解耦、GPT latent、三阶段训练、Qwen3 软指令）
16. **F5-TTS 论文**（arXiv:2410.06885）— https://arxiv.org/abs/2410.06885 ｜ 仓库：https://github.com/SWivid/F5-TTS
    - **用于**：F5-TTS 架构、许可证（代码 MIT / 权重 CC-BY-NC）、L20 benchmark（253ms / RTF 0.0394 / 0.1467）
17. **Fish Audio S2 技术报告**（arXiv:2603.08823）— https://arxiv.org/abs/2603.08823 ｜ 仓库：https://github.com/fishaudio/fish-speech
    - **用于**：S2 Pro 的 4B Dual-AR 架构、80+ 语言、H200 RTF 0.195 / TTFA ~100ms、GRPO、FISH AUDIO RESEARCH LICENSE
18. **Fish-Speech v1.4 论文**（arXiv:2411.01156）— https://arxiv.org/abs/2411.01156
19. **GLM-TTS 技术报告**（arXiv:2512.14291）— https://arxiv.org/abs/2512.14291
    - **用于**：GLM-TTS 1.5B 的 CER/SIM、tokenizer 25Hz/32k、GRPO 四奖励、Phoneme-in、2D-Vocos
20. **Microsoft VibeVoice** — https://github.com/microsoft/VibeVoice
    - **用于**：7.5Hz 连续 tokenizer、next-token diffusion、90 分钟/4 说话人、Realtime-0.5B ~300ms、**TTS 代码于 2025-09-05 下架**、MIT 许可
21. **Kokoro-82M** — https://github.com/hexgrad/kokoro ｜ https://huggingface.co/hexgrad/Kokoro-82M
    - **用于**：82M 参数、StyleTTS 2 + ISTFTNet、Apache-2.0、8 语言 54 音色、1000 A100 小时/$1000 训练成本

##### 生态与工具

22. **FunASR** — https://github.com/modelscope/FunASR
23. **Fun-ASR-Nano** — https://github.com/FunAudioLLM/Fun-ASR
24. **SenseVoice** — https://github.com/FunAudioLLM/SenseVoice
25. **FunClip** — https://github.com/modelscope/FunClip
26. **Triton + TensorRT-LLM runtime**（NVIDIA Yuekai Zhang 贡献）— https://github.com/FunAudioLLM/CosyVoice/tree/main/runtime/triton_trtllm
27. **Matcha-TTS**（CosyVoice 借鉴的 CFM 实现）— https://github.com/shivammehta25/Matcha-TTS
28. **AcademiCodec** — https://github.com/yangdongchao/AcademiCodec
29. **FunCodec** — https://github.com/modelscope/FunCodec

##### 数据/评测相关引用（来自论文参考文献）

- [33] **MinMo**（语音理解基座）
- [34] **FSQ**（Finite Scalar Quantization）
- [36] **RoPE**（Rotary Position Embedding）
- [44][25] **DiT**（Diffusion Transformer）
- [47] **ERes2Net**（说话人确认，用于 SS 评估）
- [49] **WavLM**（说话人确认，用于 SS 评估）


> **本章定位**：CosyVoice3 代表另一条技术路线：监督多任务 Tokenizer + 可微奖励后训练 DiffRO。


---


## 第四部分 · 横向对比

### 4.0 三款核心模型横向对比

| 维度 | **Qwen3-ASR** | **Qwen3-TTS** | **CosyVoice3** |
|---|---|---|---|
| 任务 | ASR（+强制对齐） | TTS | TTS |
| 发布 | 2026-01-29 | 2026-01-22 | 论文 2025-05 / 权重 2025-12 |
| 协议 | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| 参数量 | 1.7B / 0.6B（+AuT 编码器 300M / 180M） | 0.6B / 1.7B | 0.5B（论文另有 1.5B，权重未开源） |
| 架构范式 | LALM：AuT 编码器 → Projector → Qwen3 LM（AR） | 双 Tokenizer + 双轨 LM：Talker → Code Predictor(MTP) → Codec | LLM（AR 生成 FSQ token）→ DiT 流匹配 → HiFi-GAN/HiFT |
| 语音表示 | 128 维 Fbank → 8× 下采样 → **12.5 Hz** 连续隐序列 | **12.5 Hz 16 层 RVQ**（码本 2048）或 25 Hz 单码本 | **25 Hz FSQ**，词表 6561 |
| LLM 基座 | Qwen3-0.6B / 1.7B | Qwen3-0.6B / 1.7B | **Qwen2.5-0.5B / 1.5B** |
| 声学解码 | 无（直接出文本） | 12Hz：轻量因果 ConvNet；25Hz：块式 DiT + Flow Matching | **22 层 DiT 流匹配**，Euler 10 步，CFG 0.7 |
| 声码器 | 无 | 内建于 Codec | HiFi-GAN / HiFT，24 kHz |
| 训练数据 | AuT 预训练 4000 万小时 + Omni 3T token | **500 万小时** / 10 语言 | MinMo 140 万小时基座 + **100 万小时** / 9 语言 + 18 方言 |
| 后训练 | GSPO（约 5 万条 utterance） | DPO / GSPO / 说话人微调 | **DiffRO**（可微奖励优化） |
| 语种 | 30 语言 + 22 中国方言 | 10 语言 | 9 语言 + 18 方言 |
| 首包延迟 | TTFT **92 ms**（0.6B，并发 1） | **97 ms**（12Hz-0.6B，并发 1） | **150 ms**（官方 README） |
| 关键指标 | AISHELL-2 CER **2.71** | Seed-TTS test-en WER **1.24** | Seed-TTS test-zh CER **0.81**（RL 版） |
| 最长输入/输出 | 20 分钟音频 | 10 分钟+ 稳定长文本 | 双向流式 |
| 微调 | 官方 reproducible fine-tuning recipe | 官方脚本（**有语速异常 bug，2026-03-17 修复**） | 官方训练/推理脚本齐备 |
| 推理引擎 | vLLM day-0 | vLLM-Omni day-0 | vLLM / Triton+TRT-LLM / gRPC / FastAPI |

### 4.1 选型决策树

```
你需要 TTS 还是 ASR？
├─ ASR
│   ├─ 中文 / 方言 / 长音频 / 歌唱 → Qwen3-ASR-1.7B
│   ├─ 高并发 / 成本敏感 / 端侧     → Qwen3-ASR-0.6B
│   ├─ CPU / 无 GPU / 边缘          → SenseVoiceSmall-GGUF（llama.cpp，q8 约 254MB）
│   └─ 需要情感识别 + 事件检测       → SenseVoiceSmall（ASR+LID+SER+AED 富文本）
└─ TTS
    ├─ 实时交互（<200ms 首包）      → Qwen3-TTS-12Hz-0.6B-Base
    ├─ 中文内容一致性优先           → CosyVoice3-0.5B-2512_RL（CER 0.81）
    ├─ 克隆音质 / 跨语种配音        → Qwen3-TTS-12Hz-1.7B-Base
    ├─ 方言 / 发音修补              → CosyVoice3（18 方言 + 拼音/CMU inpainting）
    ├─ 无参考音频、纯描述造音色      → Qwen3-TTS-12Hz-1.7B-VoiceDesign
    ├─ 视频配音（需精确时长控制）    → IndexTTS-2（⚠️ bilibili 协议，非 Apache）
    └─ 无 GPU / 快速验证            → Kokoro-82M（Apache-2.0，CPU 可跑）
```

---


---


## 开源 TTS 竞品对比（Fish-Speech / IndexTTS-2 / F5-TTS 等）

> **分类**：技术栈-竞品对比　|　**已填字段**：17 / 38　|　**⭐ 标记为本轮重点的架构与训练字段**


### 1. 基本信息与版本演进

#### 调研范围与时间基准

- **调研日期**：2026-08-28
- **覆盖对象**（12 个代表性开源/开放权重 TTS 系统）：
  | 梯队 | 模型 |
  |---|---|
  | **旗舰/最新** | Fish Audio S2 Pro（4B）、VoxCPM2（2B）、IndexTTS-2.5（0.8B） |
  | **主流成熟** | **CosyVoice 3-0.5B**（参照基准）、**IndexTTS-2（1.5B）**、**F5-TTS（0.3B）**、GLM-TTS（1.5B）、VibeVoice-1.5B、HiggsAudio-v2（3B） |
  | **轻量/专用** | **Spark-TTS（0.5B）**、VoxCPM1.5（0.8B）、VoxCPM-0.5B（0.6B）、**Kokoro-82M**、GPT-SoVITS |
  | **已弃用** | **Fish-Speech 1.5**（2026-02-28 官方弃用，被 S2 取代） |

#### ⚠️ 三条最关键的「过时信息纠正」（用户资料与实际不符）

##### 1. Fish Audio 已于 2026-03-07 把许可证从 Apache-2.0 改为 **FISH AUDIO RESEARCH LICENSE**

- 用户资料：「Fish-Speech 1.5，Apache 2.0 但 S2 Pro 商用需另授权」
- **实际情况**：`fishaudio/fish-speech` 仓库 **main 分支的 LICENSE 已整体替换为 FISH AUDIO RESEARCH LICENSE AGREEMENT（Last Updated: March 7, 2026）**。**代码与权重统一为研究许可，任何商用都需要 Fish Audio 的单独书面授权**（联系 business@fish.audio）。不再存在「代码 Apache-2.0 + 权重受限」的分离，而是**整体受限**。
- 另外：**Fish-Speech 1.5 已于 2026-02-28 官方弃用**（deprecated），被 Fish Audio S2 取代。
- 中间产物 **OpenAudio S1-mini（0.5B）** 的权重是 **CC-BY-NC-SA-4.0**（非商用），代码仓库仍是 Apache-2.0。

##### 2. IndexTTS-2 的许可证是 **bilibili Model Use License Agreement，不是 Apache-2.0**

- 用户资料：「IndexTTS-2，Apache 2.0」
- **实际情况**：直接抓取 `https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE`，内容为 **bilibili Model Use License Agreement**，第 1.4 条明确定义「Model」为「the artificial-intelligence model named **"bilibili indextts2"**」。
- 关键条款：
  - **§2.2 规模阈值**：月活 > **1 亿** 或 年收入 > **10 亿人民币** → 需单独书面授权
  - **§3.4(c) 反向限制**：**不得用于改进任何 AI 模型**，除非是 indextts2 本身、其衍生作品、或非商业 AI 模型（**这条对「用 IndexTTS-2 生成数据训练自己的模型」是硬性阻断**）
  - **§4.2 禁止高风险场景**：医疗诊断、自动驾驶、军事、关键基础设施、大规模生物识别监控、自动化决策
  - **§6 管辖**：中国法律 + 上海仲裁委员会
- **信息冲突提示**：大量中文技术博客（2026 年）仍写「Apache-2.0 可商用」，这是**沿用了 IndexTTS 1.0/1.5 的旧信息**。**选型时必须以仓库 LICENSE 原文为准。**

##### 3. F5-TTS 的权重是 **CC-BY-NC（非商用）**，不是 MIT

- 用户资料：「F5-TTS，MIT 协议」
- **实际情况**（F5-TTS README 原文）：「Our code is released under **MIT License**. The pre-trained models are licensed under the **CC-BY-NC** license due to the training data **Emilia**, which is an in-the-wild dataset.」
- **根因**：**Emilia**（101.7k 小时 in-the-wild 语音，CC-BY-NC-4.0）→ 非商用条款**传导到所有用它训练的权重**。同病相怜的还有 **MaskGCT**（CC-BY-NC-4.0）、**Llasa**（1B/3B/8B 全部 cc-by-nc-4.0，理由是伦理与隐私）。
- **出路**：Emilia-YODAS 子集是 CC-BY-4.0（113.9k 小时）；Emilia-Large 合并后 215.6k 小时。**只用 YODAS 半边训练的模型可以发宽松权重**。

#### 一句话总览

> **2026 年的开源 TTS 格局已从「拼指标」转向「拼授权 + 拼部署生态」。**
> 质量上第一梯队（Seed-TTS-eval zh CER < 1.2）已被 Fish S2 Pro、VoxCPM2、Qwen3-TTS、GLM-TTS-RL、CosyVoice3-RL、MiniMax 挤满，差距在噪声级别；
> **真正的分水岭是「能不能商用」和「能不能部署到你的硬件上」**——这两项上 CosyVoice 3 与 VoxCPM2 是仅有的「高质量 + Apache-2.0 + 服务端生态完整」的双优解。


### 2. 开源协议与商用边界

#### 许可证与商用边界（**本次调研最关键的产出**）

##### A. 汇总表

| 模型 | 代码许可 | 权重许可 | **商用？** | 阈值/限制 |
|---|---|---|---|---|
| **CosyVoice 3** | Apache-2.0 | **Apache-2.0** | ✅ **完全可商用** | 无收入/规模阈值 |
| **VoxCPM2 / 1.5 / 0.5B** | Apache-2.0 | **Apache-2.0** | ✅ **完全可商用** | 无（README 明确 "free for commercial use"） |
| **GLM-TTS** | **Apache-2.0** | **MIT** | ✅ 可商用 | 无 |
| **VibeVoice**（TTS） | MIT | MIT | ⚠️ 法律可，官方不建议 | README：「Not for commercial/real-world use without further testing」；**TTS 代码已下架** |
| **Kokoro-82M** | Apache-2.0 | **Apache-2.0** | ✅ **完全可商用** | 无；训练数据全部为 permissive/IPA |
| **IndexTTS-2 / 2.5** | **bilibili 模型使用许可协议** | 同左 | ⚠️ **有条件商用** | ① 月活 > **1 亿** 或 年收入 > **10 亿人民币** → 需单独书面授权<br>② **不得用于改进其他 AI 模型**（除自身/衍生/非商业模型）<br>③ 禁止高风险场景（医疗/自动驾驶/军事/关键基础设施/大规模生物识别/自动化决策）<br>④ 中国法律 + 上海仲裁委 |
| **Fish Audio S2 / S2 Pro**<br>（`fishaudio/fish-speech` 仓库整体） | **FISH AUDIO RESEARCH LICENSE**<br>（2026-03-07 起） | 同左 | ❌ **需商业授权** | 研究 + 非商业免费；**任何 Commercial Purpose 都需要 Fish Audio 单独书面协议**（business@fish.audio）<br>「Commercial Purpose」定义极宽，包含**企业内部运营**<br>另：不得用于创建/改进任何基础生成式 AI 模型 |
| **OpenAudio S1-mini**（0.5B） | Apache-2.0 | **CC-BY-NC-SA-4.0** | ❌ **非商用** | SA 传染性：衍生作品需同协议 |
| **F5-TTS** | **MIT** | **CC-BY-NC-4.0** | ❌ **非商用** | 根因：训练数据 **Emilia** 为 CC-BY-NC-4.0 |
| **MaskGCT** | Amphion（宽松） | **CC-BY-NC-4.0** | ❌ 非商用 | 同 Emilia 传导 |
| **Llasa（1B/3B/8B）** | 宽松 | **CC-BY-NC-4.0** | ❌ 非商用 | 理由为伦理与隐私 |
| **Spark-TTS** | Apache-2.0（推理代码） | **CC-BY-NC-SA-4.0** | ❌ **非商用** | HF 模型卡 `license: cc-by-nc-sa-4.0`；**已从 Apache-2.0 改为 CC-BY-NC-SA** |
| **HiggsAudio-v2**（3B） | Apache-2.0（仓库 badge） | HF 标 **"other"**，checkpoint 内有单独 LICENSE | ⚠️ **需逐文件确认** | v3 系列改用 Boson research & non-commercial license |
| **GPT-SoVITS** | 仓库 Apache-2.0 | **界定模糊** | ⚠️ **高风险** | 训练/微调数据常含未授权音频；社区共识为「商用有版权风险」 |

##### B. 「代码许可 ≠ 权重许可」——最常见的踩坑模式

> **README 上的 badge 通常只说代码许可，不说权重许可。**

| 案例 | 代码 | 权重 | 后果 |
|---|---|---|---|
| F5-TTS | MIT ✅ | CC-BY-NC ❌ | 被大量文章误传为「MIT 可商用」 |
| Fish Audio | Apache-2.0（旧）→ 研究许可（2026-03） | 同左（已统一） | 曾长期被当作 Apache-2.0 |
| OpenAudio S1-mini | Apache-2.0 ✅ | CC-BY-NC-SA ❌ | 商用部署踩雷 |
| HiggsAudio-v2 | Apache-2.0 ✅ | "other"（需读 LICENSE 文件） | 需要法务逐文件确认 |
| Spark-TTS | Apache-2.0 ✅ | CC-BY-NC-SA ❌ | 权重许可被静默更新过 |
| IndexTTS-2 | **bilibili 协议**（非 Apache） | 同左 | 中文社区普遍误传为 Apache-2.0 |

**操作建议**：
1. **永远读仓库根目录的 LICENSE 原文，不读 badge、不读二手文章**
2. **权重许可看 HuggingFace 模型卡右上角或 ModelScope 的「许可协议」字段**
3. 每季度复查一次——Fish Audio 在 2026-03 整体换过协议，Spark-TTS 也从 Apache-2.0 改成了 CC-BY-NC-SA

##### C. 端到端合规清单（与模型许可无关的通用风险）

| 风险 | 说明 | 缓解 |
|---|---|---|
| **声音权 / 人格权** | 任何许可都不覆盖被克隆者的授权。中国《民法典》第 1023 条保护声音权 | 取得说话人**书面**授权 |
| **AI 生成标识** | 生成式 AI 服务需遵守内容标识要求 | 加水印 / 片头提示 |
| **训练数据血缘** | 部分模型未公开训练集构成 | 强合规场景做数据血缘审计或选 Apache-2.0 且数据公开的模型 |
| **跨境与出口管制** | 部分许可含 Trade Control Laws 条款（如 Fish Audio §IV.b） | 法务评审 |
| **下游传染** | CC-BY-NC-**SA** 具有传染性（OpenAudio S1-mini、Spark-TTS） | 避免 SA 协议权重进入商业产品 |

##### D. 决策树：按商用需求快速筛

```
需要商用？
├─ 否（研究/个人/原型）→ 所有模型都可用
└─ 是
   ├─ 公司规模：月活 >1亿 或 年收入 >10亿人民币？
   │   └─ 是 → ❌ 排除 IndexTTS-2（需单独授权）
   ├─ 需要用模型生成的数据去训练/改进自有模型？
   │   └─ 是 → ❌ 排除 IndexTTS-2（§3.4c）、Fish Audio（§IV.b）
   ├─ 部署在高风险场景（医疗/自动驾驶/军事/关键基础设施/大规模生物识别/自动化决策）？
   │   └─ 是 → ❌ 排除 IndexTTS-2（§4.2）
   ├─ 预算为零（不接受商业授权谈判）？
   │   └─ 是 → ❌ 排除 Fish Audio S2/S2 Pro
   └─ 剩余可选项（高质量 + 商用干净）：
       ✅ CosyVoice 3（Apache-2.0）
       ✅ VoxCPM2（Apache-2.0，48kHz，30 语言）
       ✅ GLM-TTS（代码 Apache-2.0 / 权重 MIT）
       ✅ Kokoro-82M（轻量，Apache-2.0）
       ⚠️ VibeVoice（MIT，但 TTS 代码已下架，官方不建议商用）
```


### 3. 模型变体与权重获取

#### 各竞品的模型变体矩阵

##### Fish Audio（fish-speech 仓库）

| 变体 | 参数量 | 状态 | 许可证 | 说明 |
|---|---|---|---|---|
| **S2 Pro** | **4B**（slow AR）+ 400M（fast AR） | 🟢 当前旗舰 | **FISH AUDIO RESEARCH LICENSE** | Dual-AR，10M+ 小时，80+ 语言，GRPO RL |
| OpenAudio S1 | ~4B | 🟡 托管为主 | 同仓库许可 | S1 系列 |
| **OpenAudio S1-mini** | 0.5B | 🟢 开放权重 | 代码 Apache-2.0 / **权重 CC-BY-NC-SA-4.0** | S1 的 0.5B 蒸馏版 |
| **Fish-Speech 1.5** | ~0.5B | 🔴 **已弃用（2026-02-28）** | 跟随仓库（现为研究许可） | 30 万小时中英 + 10 万小时日语；TTS Arena ELO 1339；英文 WER 3.5% / CER 1.2%；中文 CER 1.3% |

> **注**：用户资料中 Fish-Speech 1.5 的「30 万小时中英 + 10 万小时日语、ELO 1339、英文 WER 3.5% / CER 1.2-1.3%」数值**来自 SiliconFlow 模型页（创建于 2024-11-29）与二手评测**，非 fishaudio 官方 README。该模型**已弃用**，仅作历史参照。

##### IndexTTS（B 站 Index 团队）

| 版本 | 发布时间 | 参数量 | 核心升级 | 语言 |
|---|---|---|---|---|
| IndexTTS 1.0 | 2025-03 | — | 基础零样本克隆 | 中英 |
| IndexTTS 1.5 | 2025-05 | — | 稳定性提升、英文改善 | 中英 |
| **IndexTTS-2** | **2025-09-08** | **1.5B** | **精确时长控制 + 情感解耦 + GPT latent + Qwen3 软指令** | 中英 |
| **IndexTTS-2.5** | 2026-01 | **0.8B** | 多语言扩展 + 推理加速 + `duration_factor` 时长比例 | **中英日西阿** |

- 论文：arXiv:2506.21619（v1 2025-06-23，v2 2025-09-03）
- 权重：HF `IndexTeam/IndexTTS-2`（`IndexTeam/Index-TTS-2`）、ModelScope `IndexTeam/IndexTTS-2`
- Demo：https://index-tts.github.io/index-tts2.github.io/
- 训练：55,000 小时（30K 中 + 25K 英）+ 135 小时情感数据（361 说话人）；**8×A100 80GB，AdamW lr 2e-4，训练 3 周**
- 精度：IndexTTS-2 用 **FP16**，2.5 用 **BF16**

##### VoxCPM（OpenBMB / ModelBest + THUHCSI）

| 变体 | 骨干参数 | 采样率 | LM token 率 | 语言 | 显存 | RTF (4090) | RTF (Nano-vLLM) | 状态 |
|---|---|---|---|---|---|---|---|---|
| **VoxCPM2** | **2B** | **48 kHz** | 6.25 Hz | **30** | **~8 GB** | ~0.30 | **~0.13** | 🟢 最新（2026-04） |
| VoxCPM1.5 | 0.8B | 44.1 kHz | 6.25 Hz | 2（中英） | ~6 GB | ~0.15 | ~0.08 | 稳定（2025-12） |
| VoxCPM-0.5B | 0.6B | 16 kHz | 12.5 Hz | 2（中英） | ~5 GB | ~0.17 | ~0.10 | Legacy（2025-09） |

- 全部 **Apache-2.0**；VoxCPM-0.5B 论文 ICLR 2026；VoxCPM2 技术报告 arXiv:2606.06928
- VoxCPM2 基座：**MiniCPM-4**；架构：tokenizer-free diffusion AR，**AudioVAE V2** latent，四阶段 `LocEnc → TSLM → RALM → LocDiT`

##### F5-TTS（上海交大 X-LANCE / SII / PCL）

| 变体 | 参数量 | 说明 |
|---|---|---|
| **F5-TTS v1 Base** | ~0.3B | DiT + ConvNeXt V2，2025-03-12 发布，训练与推理更优 |
| F5-TTS（v0，2024-10） | ~0.3B | 初版 |
| **E2 TTS** | ~0.3B | Flat-UNet Transformer，论文 arXiv:2406.18009 的最接近复现 |

- 论文：arXiv:2410.06885；代码 **MIT** / 权重 **CC-BY-NC**
- 社区多语种扩展（zh/ja 等）、f5-tts-mlx（Apple）、F5-TTS-ONNX
- 声码器可切换 **Vocos** / **BigVGAN**

##### VibeVoice（Microsoft）

| 变体 | 参数量 | 说明 | 状态 |
|---|---|---|---|
| **VibeVoice-TTS-1.5B** | 1.5B | 90 分钟 / 4 说话人长音频 | ⚠️ **TTS 代码 2025-09-05 下架**，权重在 HF（2.7B safetensors, BF16），Demo 已 Disabled |
| **VibeVoice-Realtime-0.5B** | 0.5B | 流式文本输入，~300ms 首 audible 延迟，~10 分钟长生成 | 🟢 可用 |
| VibeVoice-ASR-7B | 7B | 60 分钟单遍 ASR，50+ 语言 | 🟢 |
| VibeVoice-ASR-BitNet | — | CPU 边缘推理（4.62GB → 1.58GB，RTF<1 @ 3+ 线程） | 🟢 |
| VibeVoice-ASR-7B 论文 | — | ICLR 2026 Oral | — |

- 许可证：**MIT**；基座 Qwen2.5-1.5B

##### GLM-TTS（智谱 AI / zai-org）

| 变体 | 参数量 | 指标（Seed-TTS-eval zh） |
|---|---|---|
| **GLM-TTS** | **1.5B** | CER **1.03** / SIM **76.1**；en WER 2.23 / SIM 67.2 |
| **GLM-TTS_RL** | 1.5B | CER **0.89** / SIM **76.4**；en WER 1.91 / SIM 68.1 |

- 论文：arXiv:2512.14291；代码 **Apache-2.0** / 权重 **MIT**
- 训练数据 100k+ 小时；**英文数据少于中文的一半**（官方说明）

##### 其他

| 模型 | 开发方 | 参数量 | 采样率 | 许可证 |
|---|---|---|---|---|
| **Spark-TTS** | SparkAudio（港科大） | 0.5B | **16 kHz** | 代码 Apache-2.0 / 权重 **CC-BY-NC-SA-4.0** |
| **HiggsAudio-v2** | Boson AI | 3B | 24 kHz | 代码 Apache-2.0 / 权重 **"other"（需读 LICENSE）** |
| **Kokoro-82M** | hexgrad | **82M** | 24 kHz | **Apache-2.0**（代码+权重） |
| **GPT-SoVITS** | RVC-Boss 社区 | ~1B 级（GPT+SoVITS） | 32/48 kHz（版本相关） | 仓库 Apache-2.0，**权重/数据商用界定模糊** |
| Qwen3-TTS | 阿里 | **1.7B** | — | ✅ 开源（VoxCPM2 报告标为 ✅） |
| MOSS-TTS | 复旦 MOSI | — | — | ✅ |
| LongCat-Audio-DiT | 美团 | 3.5B | — | ✅ |
| MegaTTS3 / DiTAR | 字节 | 0.5B / 0.6B | — | ❌ 闭源 |
| FireRedTTS-2 | 小红书 | 1.5B | — | ✅ |

#### 参照基准：CosyVoice 3

见 `cosyvoice3.json`。开源权重：0.5B base + 0.5B RL（1.5B 仅论文）。


### 4. 架构范式 ⭐

#### 一、范式地图：2026 年开源 TTS 的四条技术路线

```
                        ┌─────────────────────────────────────────────┐
                        │  路线 A：LLM-AR + Flow Matching（主流）      │
                        │  CosyVoice 3 / IndexTTS-2 / GLM-TTS / F5-TTS│
                        └─────────────────────────────────────────────┘
                        ┌─────────────────────────────────────────────┐
                        │  路线 B：Dual-AR（双自回归，RVQ 残差码本）    │
                        │  Fish Audio S2 Pro                          │
                        └─────────────────────────────────────────────┘
                        ┌─────────────────────────────────────────────┐
                        │  路线 C：Tokenizer-Free 扩散自回归           │
                        │  VoxCPM 系列                                │
                        └─────────────────────────────────────────────┘
                        ┌─────────────────────────────────────────────┐
                        │  路线 D：纯 NAR Flow Matching / 扩散         │
                        │  F5-TTS / E2-TTS / MaskGCT / VibeVoice      │
                        └─────────────────────────────────────────────┘
```

##### 路线 A：LLM-AR（离散 token）+ Flow Matching（mel）

```
文本 → [LLM 自回归] → 离散语音 token（低帧率）→ [Flow Matching / DiT] → mel → [声码器] → 波形
                          ↑                                                    ↑
                    语义 / 韵律 / 时长                                    音色 / 音质
```

| 模型 | LLM | token 率 | 声学模型 | 声码器 |
|---|---|---|---|---|
| **CosyVoice 3** | Qwen2.5-0.5B（24L/896d/14Q-2KV） | **25 Hz**（FSQ，码本 6561） | 22 层 DiT，Euler 10 步，CFG 0.7 | HiFT/HiFi-GAN，24 kHz |
| **IndexTTS-2** | GPT-2 架构，**1280 维 / 24 层 / 20 头** | 离散 token | **S2M: Conformer**（semantic→mel） | **BigVGAN2**，24 kHz |
| **GLM-TTS** | Llama-based AR | **25 Hz**（32k 词表，从 12.5Hz/16k 升级） | Flow Matching **DiT** | **2D-Vocos**（论文称优于 1D） |
| **F5-TTS** | ❌ 无 AR（纯 NAR） | — | **DiT + ConvNeXt V2**，Sway Sampling | **Vocos** / BigVGAN，24 kHz |

**特点**：
- 优点：可控性强（离散 token 便于做 RAS、DiffRO、token 级时长控制）；LLM 可直接复用 vLLM 生态
- 缺点：量化上限决定音质天花板；多一次级联误差

##### 路线 B：Dual-AR（Fish Audio S2 Pro）

```
                    ┌─ Slow AR（4B）：沿时间轴预测主语义码本 ────┐
文本 + [tag] 指令 → ┤                                            ├→ RVQ 10 码本（~21 Hz）→ 声码器 → 波形
                    └─ Fast AR（400M）：每步生成剩余 9 个残差码本 ─┘
```

- **非对称设计**：大模型管语义，小模型管声学细节 → 音质与速度兼顾
- **结构性优势**：「Dual-AR 架构与标准 LLM 同构」→ **原生支持 SGLang 的全部加速特性**（Continuous Batching、Paged KV Cache、CUDA Graph、RadixAttention Prefix Caching）
- **实测**（单卡 H200 + SGLang）：RTF **0.195**，TTFA **~100 ms**，**3000+ acoustic tokens/s**（RTF<0.5 下）
- **RVQ 10 码本 ~21 Hz** —— 对比 CosyVoice 3 的单码本 25Hz FSQ，信息量大得多

##### 路线 C：Tokenizer-Free 扩散自回归（VoxCPM 系列）

```
文本 → [LocEnc] → [TSLM] → [RALM] → [LocDiT] → AudioVAE V2 解码 → 48kHz 波形
        局部编码   文本语义  残差 acoustic  局部扩散
                   LM       LM
```

- **核心主张**：**放弃离散 tokenization**，在 AudioVAE V2 的**连续 latent** 上直接做 diffusion autoregressive
- **好处**：无量化损失 → 可以做 **48kHz**；AudioVAE V2 的非对称 encode/decode 让 16kHz 参考音频直接产出 48kHz（内置超分）
- **代价**：无法使用纯 LLM 的离散工程栈（RAS、DiffRO 这类 token 级 trick 不适用）；LM token 率低至 **6.25 Hz**（VoxCPM1.5/2），序列短但每步计算重
- **开源致谢**：VoxCPM2 明确致谢 **CosyVoice 的 Flow Matching-based LocDiT 实现**、DiTAR、MiniCPM-4、DAC

##### 路线 D：纯 NAR（F5-TTS / E2-TTS / MaskGCT）

```
文本 + 参考音频 → [DiT，Flow Matching，一次前向] → mel → 声码器 → 波形
```

- **优点**：**吞吐极高**（F5-TTS 在 L20 上 RTF **0.0394** @ TRT-LLM Client-Server，比 AR 方案快一个数量级）
- **缺点**：
  - **无法真流式**：需等全部文本处理完才开始出声
  - **时长必须预先给定**（由参考音频长度或长度回归决定）
  - 长句/跨语言稳定性弱于 AR
- **F5-TTS 的 Sway Sampling**：推理时**非均匀分配 flow step**（轨迹两端多分配步数），不增 NFE 而提质量——这是所有 flow matching 类模型都可复用的低成本推理 trick

##### 各路线的能力边界

| 能力 | A（AR+FM） | B（Dual-AR） | C（Tokenizer-Free） | D（纯 NAR） |
|---|---|---|---|---|
| 真流式首包延迟 | ✅ 好（150ms 级） | ✅ 最好（~100ms） | ✅ 好 | ❌ 差/无 |
| 吞吐（offline） | 中 | 中高 | 中 | ✅ **最高** |
| 精确时长控制 | ⚠️ 需额外设计（IndexTTS-2 有） | ⚠️ 弱 | ⚠️ 弱 | ❌ 天然不具备 |
| 音质上限 | 受量化限制（24kHz） | 好 | ✅ **最好**（48kHz，无量化） | 受量化限制 |
| LLM 生态复用 | ✅ 完全 | ✅ 完全 | ⚠️ 部分 | ❌ 无 |
| 情感/副语言控制 | ✅ 好（指令/解耦） | ✅ **最好**（15k tag） | ✅ 好（Voice Design） | ❌ 弱 |

#### 二、横向对比总表（**核心产出**）

> 所有数值均标注测试条件与来源。**「待核实」表示未能从一手来源确认**。

##### 2.1 基础属性与授权

| 模型 | 参数量 | 许可证（代码/权重） | **商用** | 输出采样率 | 显存需求 | 训练数据 | 语言数 |
|---|---|---|---|---|---|---|---|
| **CosyVoice 3-0.5B** | 0.5B | Apache-2.0 / Apache-2.0 | ✅ | **24 kHz** | **待核实**（估算 3~5GB） | **~1,000K h** | **9 + 18/19 方言** |
| **Fish Audio S2 Pro** | **4B** + 400M | 研究许可 / 研究许可 | ❌ 需授权 | 待核实 | 待核实（H200 部署） | **10M+ h** | **80+** |
| Fish-Speech 1.5 | ~0.5B | 已弃用（2026-02-28） | ❌ | 24 kHz | ~6 GB（第三方评测） | 300K h（中英）+100K h（日） | 多语 |
| **IndexTTS-2** | **1.5B** | **bilibili 协议** | ⚠️ 有条件 | **24 kHz** | **FP16 ~7.8 GB** | **55K h**（30K 中+25K 英）+135h 情感 | 中英 |
| IndexTTS-2.5 | 0.8B | bilibili 协议 | ⚠️ 有条件 | 24 kHz | 待核实（BF16） | — | **中英日西阿** |
| **F5-TTS Base** | ~0.3B | MIT / **CC-BY-NC** | ❌ | 24 kHz | 待核实 | Emilia 系 | 中英（社区多语扩展） |
| **VoxCPM2** | **2B** | Apache-2.0 / Apache-2.0 | ✅ | **48 kHz** | **~8 GB** | **2M+ h** | **30**（+9 种中国方言） |
| VoxCPM1.5 | 0.8B | Apache-2.0 | ✅ | 44.1 kHz | ~6 GB | — | 2 |
| VoxCPM-0.5B | 0.6B | Apache-2.0 | ✅ | 16 kHz | ~5 GB | — | 2 |
| **VibeVoice-1.5B** | 1.5B | MIT / MIT | ⚠️ 官方不建议 | 24 kHz | 待核实（第三方称 ≥8GB） | 英文为主，中文少 | 中英 |
| VibeVoice-Realtime | 0.5B | MIT | ⚠️ | 24 kHz | 待核实 | — | 9 语言实验音色 |
| **GLM-TTS** | **1.5B** | Apache-2.0 / **MIT** | ✅ | 待核实 | 待核实 | **100K+ h** | 中英为主 + 方言 |
| GLM-TTS_RL | 1.5B | Apache-2.0 / MIT | ✅ | 待核实 | 待核实 | 同上 | 同上 |
| **HiggsAudio-v2** | 3B | Apache-2.0 / **"other"** | ⚠️ 需确认 | 24 kHz | 待核实（第三方称 ≥8GB） | 10M+ h（第三方称） | 多语 |
| **Spark-TTS** | 0.5B | Apache-2.0 / **CC-BY-NC-SA** | ❌ | **16 kHz** | **1~2 GB**（第三方评测） | 待核实 | 中英 |
| **Kokoro-82M** | **82M** | Apache-2.0 / Apache-2.0 | ✅ | 24 kHz | ~400 MB（内存） | **几百小时** | **8 语言 / 54 音色** |
| **GPT-SoVITS** | ~1B 级 | Apache-2.0 / **界定模糊** | ⚠️ 高风险 | 32/48 kHz | 待核实 | 社区数据 | 中英日等多语 |

##### 2.2 能力维度

| 模型 | 零样本最短参考音频 | **情感控制** | **时长控制** | 流式 | 首包/首帧延迟 | SFT/LoRA |
|---|---|---|---|---|---|---|
| **CosyVoice 3** | 待核实（社区 3~10s） | 文本指令（>100 风格）+ DiffRO-EMO；**text-unrelated 下 sad/angry 弱** | ❌ 无（仅 token 比例软约束） | ✅ 双向 | **150 ms**（官方） | ✅ SFT；LoRA **待核实** |
| **Fish Audio S2 Pro** | **10~30 s**（官方） | ✅✅ **最强**：**15,000+ 自然语言 tag**，`[whisper]`/`[excited]`/`[angry]`/`[pause]`/`[inhale]`/`[chuckle]`…，支持 sub-word 级 | ⚠️ 弱 | ✅ | **~100 ms TTFA**（H200+SGLang） | 待核实 |
| Fish-Speech 1.5 | **5 s**（第三方） | tag 控制 | ❌ | ✅ | ~150 ms（第三方） | 待核实 |
| **IndexTTS-2** | **5~15 s**（官方/社区） | ✅✅ **音色-情感解耦**（GRL）；**4 种控制方式**：情感参考音频 / 8 维情感向量 / 文本描述（T2E，Qwen3-1.7B 蒸馏自 DeepSeek-R1）/ 跨说话人情感迁移；7 种基础情感 | ✅✅ **唯一**：**token 级精确时长**，0.75x~1.25x，**误差 < 0.02%~0.07%** | ⚠️ 未明确 | 待核实 | 待核实 |
| **F5-TTS** | **3 s**（第三方） | ❌ 基本无 | ❌ 天然不具备 | ⚠️ 有限 | 253 ms 平均延迟（L20, 16 NFE, conc 2） | ✅ finetune 脚本 |
| **VoxCPM2** | 待核实 | ✅ **Voice Design**（自然语言描述造全新音色，无需参考音频）+ Controllable Cloning（style guidance 调节情感/语速）+ Ultimate Cloning（参考音频+文本续写） | ⚠️ 弱 | ✅ | 待核实 | ✅ **SFT + LoRA**（5~10 分钟音频） |
| **VibeVoice-1.5B** | 待核实 | ⚠️ 部分（表达性对话） | ❌ | ⚠️ | — | 待核实 |
| VibeVoice-Realtime | 预设音色 | ❌ | ❌ | ✅ 流式文本输入 | **~300 ms** | 待核实 |
| **GLM-TTS** | **~3 s** | ✅ GRPO 四奖励（CER/SIM/**Emotion**/**Laughter**）；Phoneme-in 发音控制 | ⚠️ 弱 | ⚠️ 类流式 | 400 ms（第三方，vs ElevenLabs 75ms） | ✅ **LoRA**（官方定位为成本优化） |
| **HiggsAudio-v2** | **5 s**（第三方） | ✅ 情感识别 + 多角色对话；**第三方评为情感表达最强** | ❌ | 待核实 | 待核实 | 待核实 |
| **Spark-TTS** | 短样本即可 | ✅ 性别 / 音调 / 语速（虚拟说话人创建） | ❌ | 待核实 | 876 ms（L20, conc 1） | ❌ 训练代码未发布 |
| **Kokoro-82M** | ❌ **不支持克隆**（固定 54 音色） | ❌ 无 | ❌ 无 | ⚠️ | ~10 ms（第三方） | ❌ |
| **GPT-SoVITS** | **5 s**（零样本）；**1 分钟微调**大幅提升 | ⚠️ 有限 | ❌ | ⚠️ | 待核实 | ✅ 少样本微调（核心卖点） |

##### 2.3 质量指标（Seed-TTS-eval，WavLM SIM 口径）

| 模型 | 参数量 | 开源 | test-zh CER↓ | test-zh SIM↑ | test-en WER↓ | test-en SIM↑ | test-hard CER↓ | test-hard SIM↑ |
|---|---|---|---|---|---|---|---|---|
| Human | — | — | 1.26 | 75.5 | 2.14 | 73.4 | — | — |
| MiniMax-Speech | — | ❌ | **0.83** | 78.3 | **1.65** | 69.2 | — | — |
| Seed-TTS | — | ❌ | 1.12 | **79.6** | 2.25 | **76.2** | 7.59 | 77.6 |
| **Qwen3-TTS** | 1.7B | ✅ | 1.22 | 77.0 | **1.23** | 71.7 | **6.76** | 74.8 |
| **LongCat-Audio-DiT** | 3.5B | ✅ | 1.09 | **81.8** | 1.50 | 78.6 | 6.04 | **79.7** |
| **FishAudio S2** | 4B | ✅ | **0.54** | — | **0.99** | — | **5.99** | — |
| **VoxCPM2** | 2B | ✅ | **0.97** | **79.5** | 1.84 | 75.3 | 8.13 | 75.3 |
| **GLM-TTS_RL** | 1.5B | ✅ | **0.89** | 76.4 | 1.91 | 68.1 | — | — |
| **GLM-TTS** | 1.5B | ✅ | 1.03 | 76.1 | 2.23 | 67.2 | — | — |
| **IndexTTS2** | 1.5B | ✅ | **1.03** | 76.5 | 2.23 | 70.6 | 7.12 | 75.5 |
| **CosyVoice3-0.5B_RL** | 0.5B | ✅ | **0.81** | 77.4 | **1.68** | 69.5 | **5.44** | 75.0 |
| **CosyVoice3-0.5B** | 0.5B | ✅ | 1.21 | **78.0** | 2.24 | 71.8 | **6.71** | 75.8 |
| **VoxCPM-0.5B** | 0.6B | ✅ | 0.93 | 77.2 | 1.85 | **72.9** | 8.87 | 73.0 |
| VibeVoice-1.5B | 1.5B | ✅ | 1.16 | 74.4 | 3.04 | 68.9 | — | — |
| **F5-TTS** | 0.3B | ✅ | 1.52 | 74.1 | 2.00 | 64.7~67.0 | 8.67 | 71.3 |
| HiggsAudio-v2 | 3B | ✅ | 1.50 | 74.0 | 2.44 | 67.7 | 55.07 | 65.6 |
| SparkTTS | 0.5B | ✅ | 1.2~1.54 | 66.0 | 1.98~3.14 | 57.3 | — | — |
| CosyVoice2 | 0.5B | ✅ | 1.45 | 75.7 | 2.57 | 65.9 | 6.83 | 72.4 |
| FireRedTTS-2 | 1.5B | ✅ | 1.14~1.51 | 73.2~73.6 | 1.95 | 66.5 | — | — |
| MOSS-TTS | — | ✅ | 1.20 | **78.8** | 1.85 | 73.4 | — | — |

> ⚠️ **F5-TTS / SparkTTS / FireRedTTS-2 的 SIM 在不同来源中差异较大**（如 F5-TTS test-en SIM：CosyVoice README 给 64.7，VoxCPM2 报告给 67.0；SparkTTS test-zh CER：1.2 vs 1.54）。原因可能是不同评测脚本/ASR 模型/checkpoint。**跨模型比较时只可比同一张表内的数字。**

##### 2.4 部署生态成熟度

| 模型 | vLLM | SGLang | Triton/TRT-LLM | FastAPI/gRPC | Docker | OpenAI 兼容 | MLX/Apple | 端侧（llama.cpp/ONNX/GGUF） | 官方文档 |
|---|---|---|---|---|---|---|---|---|---|
| **CosyVoice 3** | ✅（0.11.x+/0.9.0） | ❌ | ✅（NVIDIA 贡献，4× 加速） | ✅ **两者都有** | ✅ | ❌ 需自封 | ⚠️ 社区 | ❌ | README（中） |
| **Fish Audio S2 Pro** | ✅ vLLM-Omni recipe | ✅ **SGLang-Omni**（原生） | ❌ | ✅ | ✅ | ✅（SGLang/vLLM-Omni） | ❌ | ❌ | ✅ speech.fish.audio（完整） |
| **VoxCPM2** | ✅ **vLLM-Omni 官方** | ❌ | ❌ | ✅（Nano-vLLM 自带） | ⚠️ | ✅ **`/v1/audio/speech`** | ✅ VoxCPMANE (ANE) | ✅✅ **最全**（llama.cpp-omni / VoxCPM.cpp / ONNX / voxcpm_rs） | ✅✅ **readthedocs 完整文档站** |
| **F5-TTS** | ⚠️ 社区 | ❌ | ✅（Yuekai Zhang） | ⚠️ Gradio/CLI | ✅ ghcr.io | ❌ | ✅ f5-tts-mlx | ✅ F5-TTS-ONNX | ✅ README（含 ROCm/XPU 矩阵） |
| **IndexTTS-2** | ⚠️ 仅 2.5 有 recipe | ❌ | ❌ | ⚠️ WebUI + Python API | ❌ | ❌ | ❌ | ❌ | ⚠️ README |
| IndexTTS-2.5 | ✅ vLLM recipe | ❌ | ❌ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ⚠️ README |
| **VibeVoice** | ✅ 仅 ASR 侧 | ❌ | ❌ | ⚠️（TTS 代码已下架） | ❌ | ❌ | ❌ | ⚠️ VibeASR.cpp（仅 ASR，CPU） | ⚠️ |
| **GLM-TTS** | 待核实 | ❌ | ❌ | ⚠️ Gradio + CLI | ❌ | ❌ | ❌ | ❌ | ⚠️ README |
| **Spark-TTS** | ❌ | ❌ | ✅（Triton + TRT-LLM，L20 实测） | ❌ WebUI + CLI | ❌ | ❌ | ❌ | ❌ | ⚠️ README |
| **Kokoro-82M** | ❌ | ❌ | ❌ | ⚠️ pip 包 / 第三方 API | ❌ | ✅（多家 API 厂商） | ❌ | ✅（多种 CPU 部署） | ⚠️ HF 模型卡 |
| GPT-SoVITS | ❌ | ❌ | ❌ | ⚠️ WebUI + API | ⚠️ 社区 | ❌ | ❌ | ❌ | ⚠️ 中文社区为主 |

**成熟度排序（服务端）**：VoxCPM2 ≈ Fish S2 Pro > **CosyVoice 3** > F5-TTS > Spark-TTS > IndexTTS-2 > GLM-TTS > VibeVoice
**成熟度排序（端侧）**：VoxCPM2 >> Kokoro-82M > F5-TTS >> CosyVoice 3（无）


### 5. 模块拆解 ⭐

#### 各竞品的组件拆解对照

| 模型 | 文本前端 | 主生成模型 | 声学/声码 | 说话人条件化 | 采样/推理 trick |
|---|---|---|---|---|---|
| **CosyVoice 3** | Qwen2.5 BPE + 内置 TN（可选 ttsfrd/wetext） | **Qwen2.5-0.5B**（24L/896d/14Q-2KV GQA），FSQ token 25Hz | **22 层 DiT**（1024d/16 头，Euler 10 步，CFG 0.7）+ HiFT/HiFi-GAN | **CAM++ 192-d**（推理主路径改为 in-context prompt token） | **RAS**（top_k 25 / top_p 0.8 / win 10 / tau 0.1） |
| **Fish Audio S2 Pro** | 无音素预处理（80+ 语言直接输入） | **Slow AR 4B + Fast AR 400M**（Dual-AR） | **RVQ 10 码本 ~21 Hz** + Firefly-GAN 系神经声码器 | `<\|speaker:i\|>` token，单次推理支持多说话人 | SGLang 全套加速；**GRPO RL**（多维度奖励：语义准确性/指令遵循/声学偏好/音色相似度） |
| **IndexTTS-2** | 支持**汉字+拼音混合建模**（`pinyin.vocab`） | **T2S**: GPT-2 架构，**1280 维 / 24 层 / 20 头** 自回归 | **S2M**: Conformer（semantic→mel）+ **BigVGAN2** 声码器，24 kHz | 音色 prompt 与**情感 prompt 分离**；**GRL 梯度反转层**解耦 | **时间编码向量**做 token 级时长；**T2E**（Qwen3-1.7B，蒸馏自 DeepSeek-R1） |
| **IndexTTS-2.5** | 同上 | 同上（0.8B） | 同上 | 同上 | 新增 `duration_factor`（0.5x~2.0x）；BF16 |
| **F5-TTS** | 字符/音素（社区多语扩展） | ❌ 无 AR | **DiT + ConvNeXt V2**，Flow Matching，**16 NFE** | 参考音频 mel 拼接 | **Sway Sampling**（非均匀 flow step 分配）；Vocos/BigVGAN 可切换 |
| **VoxCPM2** | MiniCPM-4 分词 | **LocEnc → TSLM → RALM → LocDiT**（tokenizer-free diffusion AR，6.25Hz） | **AudioVAE V2**（非对称 encode/decode + 内置超分） | Reference / Continuation 双克隆模式 | `cfg_value=2.0`，`inference_timesteps=10` |
| **VibeVoice-1.5B** | Qwen2.5-1.5B tokenizer | LLM（Qwen2.5-1.5B）+ **diffusion head**（next-token diffusion） | **连续 tokenizer**（Acoustic + Semantic），**7.5 Hz** | 说话人 prompt | 超低帧率 7.5Hz → 支撑 90 分钟长上下文 |
| **GLM-TTS** | **Phoneme-in**（音素+文本混合输入） | Llama-based AR LLM | Flow Matching **DiT** + **2D-Vocos** 声码器 | Speaker Embedding + Speech Tokenizer | **GRPO**（4 奖励）+ DAPO 的 Clip-Higher（ε_high 0.3 / ε_low 0.2）+ Dynamic Sampling |
| **Spark-TTS** | Qwen2.5 BPE | **Qwen2.5-0.5B 单流解耦语音 token** | ❌ **无 flow matching**——直接从 LLM 预测的 code 重构音频 | 参考音频 / 参数化虚拟说话人 | 单流架构，直接音频重构 |
| **HiggsAudio-v2** | 统一音频分词器 | 3B LLM | **残差向量量化（RVQ）**，2 kbps 压缩率保 24kHz | 参考音频 | 多角色对话（单次推理多音色） |
| **Kokoro-82M** | **misaki G2P**（IPA 音素） | ❌ 无 AR、无 diffusion、无 encoder | **StyleTTS 2 + ISTFTNet** 纯 decoder | ❌ 固定 54 音色，不可克隆 | 极简 decoder-only |
| **GPT-SoVITS** | 中文 G2P + BERT | GPT（语义 token AR） | **SoVITS**（VITS 系） | 参考音频 + 微调 | 少样本微调 |

#### 关键设计差异的工程含义

| 设计选择 | 代表 | 含义 |
|---|---|---|
| **有 / 无声学中间模型** | Spark-TTS **无 flow matching**（LLM 直接重构音频） | 架构最简、部署最容易；但音质上限受限于 code 分辨率（**16 kHz**，是所有对比模型中最低的） |
| **离散 vs 连续 token** | CosyVoice 3（FSQ 离散）/ VoxCPM2（连续 latent） | 离散 → 可用 LLM 全套工程栈（RAS、DiffRO、vLLM）；连续 → 无量化损失，可做 48kHz |
| **单码本 vs RVQ 残差** | CosyVoice 3（单 FSQ 6561）/ Fish S2 Pro（RVQ 10 码本）/ HiggsAudio-v2（RVQ 2kbps） | RVQ 信息量大、音质好；但 AR 需要 Dual-AR 或多层预测，工程复杂 |
| **token 帧率** | VibeVoice **7.5 Hz** / VoxCPM2 **6.25 Hz** / CosyVoice 3 **25 Hz** / Fish S2 **~21 Hz** | 帧率越低 → 序列越短 → 越适合超长音频；但每 token 需承载更多信息，重建难度更高 |
| **声码器选型** | HiFT（CosyVoice 3）/ BigVGAN2（IndexTTS-2）/ Vocos（F5-TTS）/ 2D-Vocos（GLM-TTS）/ ISTFTNet（Kokoro） | BigVGAN 系音质最好但最重；Vocos 轻量；ISTFTNet 适合小模型 |
| **说话人条件化方式** | CAM++ embedding（注入）vs in-context prompt token（拼接） | **in-context 能传递韵律+录音环境**，但占用上下文；embedding 更省但不传韵律。**CosyVoice 3 三代已转向 in-context**


### 6. 语音 Tokenizer 与编解码器 ⭐

#### 语音 Tokenizer / Codec 对照

| 模型 | 量化方式 | 帧率 | 码本/词表 | 监督方式 | 特点 |
|---|---|---|---|---|---|
| **CosyVoice 3** | **FSQ**（有限标量量化） | **25 Hz** | **6561**（=$(2K+1)^D$，$D$=8,$K$=1 推断） | **ASR + SER + LID + AED + SA**（530K h，MinMo 基座 12 层 Transformer + RoPE） | 无码本坍塌（利用率接近 100%）；多任务监督带来韵律优势；论文未公布 $D$/$K$ |
| **CosyVoice 2** | FSQ | 25 Hz | — | SenseVoice-Large **ASR** 单一监督 | 前代 |
| **CosyVoice 1.0** | **VQ** | 25/50 Hz | 4096（`speech_token_size`） | SenseVoice ASR | **码本坍塌，利用率约 23%** |
| **Fish Audio S2 Pro** | **RVQ（残差向量量化）** | **~21 Hz** | **10 个码本**（1 语义 + 9 残差） | — | Dual-AR 分别预测：slow AR 出主码本，fast AR 出残差 |
| **IndexTTS-2** | 离散 token（**具体量化方式待核实**） | 待核实 | 待核实 | — | 支持汉字+拼音混合 vocab（`pinyin.vocab`） |
| **F5-TTS** | ❌ **无离散 tokenizer** | — | — | — | 纯连续：直接在 mel 空间做 Flow Matching |
| **VoxCPM2** | ❌ **Tokenizer-Free** | **6.25 Hz**（LM token 率） | — | — | **AudioVAE V2** 连续 latent；非对称 encode/decode（16kHz 输入 → 48kHz 输出，内置超分） |
| **VibeVoice** | ❌ **连续 tokenizer**（Acoustic + Semantic 双 tokenizer） | **7.5 Hz** | — | — | 超低帧率是支撑 90 分钟长音频的关键 |
| **GLM-TTS** | 离散（类型待核实） | **25 Hz**（v2；前代 12.5 Hz） | **32k**（前代 16k） | 新增 **pitch-estimation 约束** | 升级 tokenizer 带来 SIM 75.2→76.1、CER 1.44→1.03（官方消融） |
| **HiggsAudio-v2** | **统一音频分词器 + RVQ** | 待核实 | — | — | **2 kbps 压缩率保持 24kHz 音质** |
| **Spark-TTS** | **单流解耦语音 token** | 待核实 | 待核实 | — | Qwen2.5 直接预测 code，无独立声学模型 |
| **Kokoro-82M** | ❌ 无（直接用 IPA 音素 + StyleTTS 2 连续表示） | — | — | — | 极简 |

#### FSQ vs VQ vs RVQ vs 无量化 —— 选型视角

| 方案 | 码本利用率 | 信息量 | 工程复杂度 | 音质上限 | 适用 |
|---|---|---|---|---|---|
| **VQ** | ⚠️ 差（CosyVoice 1.0 约 **23%**） | 中 | 低（但需 EMA 维护码本 + commitment loss） | 中 | 已基本被淘汰 |
| **FSQ** | ✅ **接近 100%**（笛卡尔积隐式码本，无未使用码字） | 中 | **最低**（无可学习码本，投影+有界取整） | 中 | **CosyVoice 3 的选择**：工程最简单 + 无坍塌 |
| **RVQ** | ✅ 好 | ✅ **最大**（多层残差） | 高（需 Dual-AR 或多层预测头） | ✅ **最好** | **Fish S2 Pro / HiggsAudio-v2**：音质优先 |
| **连续 / 无量化** | N/A | ✅ 无信息损失 | 高（需 VAE / 连续扩散） | ✅✅ **最高**（VoxCPM2 可到 48kHz） | **VoxCPM2 / VibeVoice**：音质与长音频优先 |
| **纯 mel（无 tokenizer）** | N/A | 中 | **最低** | 中 | **F5-TTS**：部署最简单 |

#### 「多任务监督 tokenizer」是 CosyVoice 3 的独门

只有 CosyVoice 3 明确用 5 个任务（ASR/SER/LID/AED/SA）联合监督 tokenizer。这带来：
- ✅ **韵律自然度提升**（情感、事件信息被显式保留在 token 里）
- ✅ **DiffRO 得以成立**（可微奖励优化只在离散 token 上可做）
- ❌ **复现门槛高**：SER（48K h）与 AED（21K h）数据最难补齐

**其他模型多数只用 ASR 监督或直接无监督**，这是 CosyVoice 3 在韵律/情感上的结构性优势来源。

#### 数据驱动的 tokenizer 升级案例（GLM-TTS 的消融，很有参考价值）

GLM-TTS 论文报告：tokenizer 从 12.5Hz/16k 升级到 **25Hz/32k** + 引入 **pitch-estimation 约束**，带来：

| 指标 | 升级前 | 升级后 |
|---|---|---|
| SIM | 75.2 | **76.1** |
| CER | 1.44 | **1.03** |

**含义**：tokenizer 是 TTS 质量的关键杠杆，**很多 TTS 栈在 tokenizer 层就已经失败，下游建模再努力也没用**。二次开发时优先检查 tokenizer。


### 7. 语种与方言覆盖

#### 语言覆盖对照

| 模型 | 语言数 | 具体语言 | 方言/口音 |
|---|---|---|---|
| **Fish Audio S2 Pro** | **80+** | **Tier 1**：日、英、中<br>**Tier 2**：韩、西、葡、阿、俄、法、德<br>**Global**：sv/it/tr/no/nl/cy/eu/ca/da/gl/ta/hu/fi/pl/et/hi/la/ur/th/vi/jw/bn/yo/…（共 80+） | ❌ |
| **VoxCPM2** | **30** | 阿拉伯语、缅甸语、中、丹麦、荷兰、英、芬兰、法、德、希腊、希伯来、印地、印尼、意、日、高棉、韩、老挝、马来、挪威、波兰、葡、俄、西、斯瓦希里、瑞典、他加禄、泰、土耳其、越南 | ✅ **9 种中国方言**：四川话、粤语、吴语、东北话、河南话、陕西话、山东话、天津话、闽南话 |
| **CosyVoice 3** | **9 + 18/19 方言** | 中、英、日、韩、德、西、法、意、俄 | ✅✅ **18/19 种**（安徽、粤语、重庆、河北、山东、上海、四川、天津、西安、郑州、东北、闽南、山西、陕西、宁夏、甘肃…）——**方言覆盖最广** |
| IndexTTS-2 | 中英 | 中、英 | ❌ |
| **IndexTTS-2.5** | **5** | 中、英、日、西、阿拉伯 | ❌ |
| **F5-TTS** | 中英 + 社区扩展 | 中、英（社区有 zh/ja 等多语版本） | ❌ |
| VibeVoice-1.5B | **2** | **英（为主）、中（数据显著更少）** | ❌ |
| VibeVoice-Realtime | 实验性 | 9 语言实验音色（DE/FR/IT/JP/KR/NL/PL/PT/ES）+ 11 种英语风格音色 | ❌ |
| **GLM-TTS** | 中英为主 | 中（优秀）、英（次要）；**英文训练数据 < 中文的一半** | ✅ 四川话、东北话 |
| HiggsAudio-v2 | 多语（具体待核实） | 待核实 | ❌ |
| Spark-TTS | **2** | 中、英（支持 code-switching） | ❌ |
| **Kokoro-82M** | **8** | 英、法、日、韩、中（普通话 + 粤语）等 | ⚠️ 含粤语 |
| GPT-SoVITS | 多语 | 中、英、日等 | ⚠️ |

#### 多语言实测质量（CV3-Eval，CER/WER ↓，越低越好）

| Model | zh | en | ja | ko | de | es | fr | it | ru |
|---|---|---|---|---|---|---|---|---|---|
| CosyVoice2 | 4.08 | 6.32 | 9.13 | 19.7 | – | – | – | – | – |
| **CosyVoice3-0.5B** | 3.89 | 5.24 | 10.4 | 12.8 | 7.41 | 4.25 | 12.9 | 6.68 | 6.77 |
| **CosyVoice3-0.5B + DiffRO** | **2.89** | **3.68** | 5.15 | 4.02 | **4.51** | **2.99** | 8.56 | **2.94** | 3.79 |
| CosyVoice3-1.5B | 3.91 | 4.99 | 7.57 | 5.69 | 6.43 | 4.47 | 11.8 | 10.5 | 6.64 |
| **Fish Audio S2** | **2.65** | **2.43** | **3.96** | **2.76** | **2.22** | **2.00** | **6.26** | **2.04** | **2.78** |
| **VoxCPM2** | 3.65 | 5.00 | 5.96 | 5.69 | 4.77 | 3.80 | 9.85 | 4.25 | 5.21 |

> 来源：VoxCPM2 技术报告（arXiv:2606.06928）的 CV3-eval 表。
> **Fish Audio S2 在 9 语言中全部第一**，这是 1000 万小时数据的直接体现。
> **CosyVoice3-0.5B + DiffRO 接近甚至超过 VoxCPM2**（it 2.94 vs 4.25，es 2.99 vs 3.80，ko 4.02 vs 5.69）。

#### MiniMax-MLS-test 多语言 WER（VoxCPM2 报告，WER ↓）

| Language | Minimax | ElevenLabs | Qwen3-TTS | FishAudio S2 | VoxCPM2 |
|---|---|---|---|---|---|
| Chinese | 2.252 | 16.026 | 0.928 | **0.730** | 1.136 |
| English | 2.164 | 2.339 | **0.934** | 1.620 | 2.289 |
| Japanese | 3.519 | 10.646 | 3.823 | **2.760** | 4.628 |
| Korean | 1.747 | 1.865 | 1.755 | **1.180** | 1.962 |
| German | 1.906 | 0.572 | 1.235 | **0.550** | 0.679 |
| French | 4.099 | 5.216 | **2.858** | 3.050 | 4.534 |
| Spanish | 1.029 | 1.084 | 1.126 | **0.910** | 1.438 |
| Russian | 4.281 | 3.878 | 3.212 | **2.400** | 3.634 |
| Arabic | **1.665** | 1.666 | – | 3.500 | 13.046 |
| Hindi | 6.962 | **5.827** | – | 14.640 | 19.699 |
| Vietnamese | **0.88** | 73.415 | – | 7.410 | 3.307 |

> ⚠️ **注意 Arabic / Hindi / Czech / Romanian / Ukrainian 上 VoxCPM2 的 WER 极高**（13~24），说明「30 语言」中很多语言质量不达标。**宣传的语言数 ≠ 可用质量的语言数**，选型时务必逐语言实测。

#### 内部 30 语言基准（VoxCPM2 报告，Gemini 3.1 Flash Lite API 转写）

| 指标 | VoxCPM2 | Fish S2-Pro |
|---|---|---|
| 30 语言平均 | **1.68%** | 未给平均 |
| zh CER | 0.92% | 1.02% |
| en WER | 0.42% | 1.03% |
| ja CER | 2.40% | 1.82% |
| ko CER | 0.95% | 0.29% |
| **km（高棉）CER** | **2.05%** | **75.15%** |
| **lo（老挝）CER** | **1.90%** | **87.40%** |
| **my（缅甸）CER** | **1.42%** | **85.27%** |

> 有意思的反转：**Fish S2-Pro 在高棉/老挝/缅甸上灾难性失败（75~87% CER）**，说明其「80+ 语言」中长尾语言质量极不稳定。**VoxCPM2 的 30 语言平均 1.68% 更可靠。**

#### 选型结论

| 需求 | 推荐 |
|---|---|
| **中文 + 中国方言** | 🥇 **CosyVoice 3**（18/19 方言）；🥈 **VoxCPM2**（9 种方言，30 语言） |
| **30 语言且质量均衡** | 🥇 **VoxCPM2**（Apache-2.0） |
| **80+ 语言长尾覆盖** | 🥇 Fish Audio S2 Pro（但需商业授权，且长尾语言质量不稳） |
| **中英双语** | IndexTTS-2 / GLM-TTS / F5-TTS 都够用 |
| **日英中三语** | Fish Audio S2 Pro（Tier 1）或 IndexTTS-2.5 |
| **轻量多语** | Kokoro-82M（8 语言，含粤语） |

**⚠️ 通用提醒**：所有厂商标称的「支持 N 语言」都需要逐语言实测验证。**标称语言数是营销数字，不是质量承诺。**


### 8. 音色克隆

#### 零样本语音克隆对照

| 模型 | 最短参考音频 | 是否需要参考文本 | 克隆模式 | SS/SIM 表现 | 跨语言克隆 |
|---|---|---|---|---|---|
| **CosyVoice 3** | **待核实**（社区 3~10 s） | **是**（强烈建议） | **in-context**：参考音频的 FSQ token 直接拼进上下文（三代推理主路径**不用 CAM++ 192-d embedding**） | test-zh SS **78.0**（开源第一） | ✅（en→ja WER 1.03） |
| **Fish Audio S2 Pro** | **10~30 s**（官方） | 待核实 | 参考音频 + `<\|speaker:i\|>` token；**单次推理支持多说话人**（无需为每个说话人单独上传参考音频） | 未公布 SIM | ✅（80+ 语言） |
| Fish-Speech 1.5 | **5 s**（第三方） | 待核实 | 参考音频 | 待核实 | ✅ |
| **IndexTTS-2** | **5~15 s** | 待核实 | **音色 prompt 与情感 prompt 分离**（GRL 解耦）；`emo_alpha` 控制情感强度 0.0~1.0 | test-zh SIM **76.5**；LibriSpeech SS **0.887**（第三方） | ✅ 中英 |
| **F5-TTS** | **3 s**（第三方） | 可留空（会用 ASR 自动转录，但占额外显存） | 参考音频 mel 拼接 | test-zh SIM 74.1，test-en SIM 64.7（**偏低**） | ⚠️ 弱 |
| **VoxCPM2** | 待核实 | Ultimate Cloning 需要 | **三种模式**：① Controllable Cloning（参考音频 + style guidance）② Ultimate Cloning（参考音频 + 文本续写，保留全部细节）③ **Voice Design**（纯自然语言描述，无需参考音频） | test-zh SIM **79.5**（最高）；MiniMax-MLS 多语言 SIM **17/24 语言第一** | ✅（30 语言） |
| VoxCPM1.5 / 0.5B | 待核实 | 需要 | **仅 Continuation 模式** | VoxCPM-0.5B test-zh SIM 77.2 | 中英 |
| **VibeVoice-1.5B** | 待核实 | 待核实 | ⚠️ 主要为长音频多说话人设计 | test-zh SIM 74.4 | ⚠️ |
| **GLM-TTS** | **~3 s** | 待核实 | Speaker Embedding + Speech Tokenizer | test-zh SIM 76.1（RL 版 76.4） | 中英 |
| **HiggsAudio-v2** | **5 s**（第三方） | 待核实 | 参考音频 + **多角色对话**（单次推理多音色） | test-zh SIM 74.0 | 待核实 |
| **Spark-TTS** | 短样本即可 | 需要（`--prompt_text`） | 参考音频 / 参数化虚拟说话人（性别/音调/语速） | test-zh SIM **66.0**（**对比模型中最低**） | ✅ 中英 code-switching |
| **Kokoro-82M** | ❌ **不支持克隆** | — | 固定 **54 个预设音色**（v1.0，8 语言） | N/A | N/A |
| **GPT-SoVITS** | **5 s**（零样本）；**1 分钟微调**大幅提升 | 需要 | 参考音频 + 少样本微调（**核心卖点**） | 社区口碑好，无公开统一 benchmark | 多语 |

#### 关键发现

1. **最短参考音频的「标称值」不可直接比较**：各家的测试条件（采样率、噪声、说话人数量）完全不同。**3s（F5-TTS/GLM-TTS）vs 10~30s（Fish S2 Pro）** 的差距更多是保守程度不同，而非真实能力差距
2. **CosyVoice 3 三代换了克隆机制**：从 CAM++ 192-d embedding 注入改为 **in-context prompt token 拼接**。这能传递音色 + 韵律 + 录音环境，是 SS 78.0（开源第一）的重要原因
3. **Spark-TTS 的克隆质量是明显短板**：test-zh SIM **66.0**，比第二名（VibeVoice 74.4）低 8 个点以上；test-en SIM 57.3 更是全场最低。**不要因为「0.5B 轻量 + 参数可控」就选它做克隆场景**
4. **F5-TTS 的英文克隆同样偏弱**：test-en SIM 64.7（CosyVoice README 口径）或 67.0（VoxCPM2 口径），明显低于同级别
5. **VoxCPM2 的 Voice Design 是独有能力**：不需要参考音频，纯自然语言描述（性别/年龄/音色/情感/语速）直接生成全新音色。适合数字人、游戏 NPC 批量造声
6. **Fish S2 Pro 的多说话人单次生成**是独有能力：`<|speaker:i|>` token，一段参考音频含多个说话人即可复用

#### 合规红线（所有模型通用）

- **所有开源许可都不覆盖被克隆者的声音权/人格权**
- 中国《民法典》第 1023 条明确保护声音权
- 生成式 AI 服务需遵守内容标识要求
- IndexTTS-2 的 bilibili 协议 §4.1(c)(d) 明确禁止用于生成虚假信息、歧视性内容、侵犯隐私的内容

#### 克隆质量提升的工程建议

```
参考音频预处理链：
原始音频 → [VAD 裁剪静音] → [降噪] → [音量归一化] → [3~10s 截断] → [准确转录] → TTS
```
- **参考文本准确性对克隆质量影响极大**（尤其 in-context 类模型）
- 无背景音乐、无混响
- 同一说话人提供多段参考音频时，选 SNR 最高的
- 需要极致相似度时优先考虑**微调路线**：GPT-SoVITS（1 分钟）、VoxCPM2（SFT/LoRA，5~10 分钟）


### 9. 情感与风格控制

#### 情感与控制能力对照（**本轮差异最大的维度**）

##### 总表

| 模型 | 情感控制方式 | 粒度 | 时长控制 | 发音控制 | 其他控制 |
|---|---|---|---|---|---|
| **Fish Audio S2 Pro** | ✅✅ **15,000+ 自然语言 tag**，`[whisper]`/`[excited]`/`[angry]`/`[pause]`/`[emphasis]`/`[laughing]`/`[inhale]`/`[chuckle]`/`[tsk]`/`[singing]`/`[sigh]`/`[screaming]`/`[surprised]`/`[exhale]`/`[panting]`/`[audience laughter]`/`[with strong accent]`/`[volume up|down]`/`[clearing throat]`/`[echo]`/`[pitch up]`… 支持**自由文本描述** | ✅✅ **sub-word 级**（文本任意位置嵌入） | ❌ | ❌ | 多说话人 `<\|speaker:i\|>`、多轮生成、语速/音量 tag |
| **IndexTTS-2** | ✅✅ **音色-情感解耦**（GRL 梯度反转层）。**4 种输入方式**：① 情感参考音频（`emo_audio_prompt`）② **8 维情感向量** `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` ③ 文本描述（**T2E 模块**，Qwen3-1.7B 蒸馏自 DeepSeek-R1，7 种基础情感）④ 跨说话人情感迁移。`emo_alpha` 0.0~1.0 调强度 | ✅  utterance 级 + 强度连续可调 | ✅✅ **token 级精确时长**（0.75x~1.25x，误差 <0.02%~0.07%） | ✅ **汉字+拼音混合**（`pinyin.vocab`） | `use_random` 引入随机性 |
| **VoxCPM2** | ✅ **Voice Design**：`(A young woman, gentle and sweet voice)` 前缀描述直接造全新音色；Controllable Cloning 中用 style guidance 调节情感/语速 | ✅ 描述级 | ❌ | ❌ | 语音设计（造音色）、上下文感知韵律 |
| **GLM-TTS** | ✅ **GRPO 四奖励**：CER / SIM / **Emotion** / **Laughter**。显式优化情感与笑声/呼吸等副语言 | ⚠️ utterance 级 | ❌ | ✅✅ **Phoneme-in**（音素+文本混合，PER 从 13.23 → **5.14**，官方消融） | 笑声、呼吸 |
| **CosyVoice 3** | ⚠️ 文本指令（>100 种风格，5,000h 指令数据）+ **DiffRO-EMO**（SER 作为可微奖励） | ⚠️ utterance 级 | ❌（仅 token 比例软约束） | ✅✅ **拼音 + CMU 音素**（RepMono+MixPhn **100% 修正率**） | 方言、语速、音量、角色扮演 |
| **HiggsAudio-v2** | ✅ 情感识别 + 多角色对话（第三方评为情感表达最强） | ⚠️ | ❌ | ❌ | 多角色音色切换 |
| **Spark-TTS** | ⚠️ 参数化虚拟说话人：**性别 / 音调 / 语速**（非情感） | ⚠️ | ❌ | ❌ | 性别、pitch、speed |
| **F5-TTS** | ❌ 基本无 | — | ❌ | ⚠️ 音素输入（社区） | 多风格/多说话人（Gradio） |
| **VibeVoice** | ⚠️ 对话动态与情感细微（长音频场景） | ⚠️ | ❌ | ❌ | 4 说话人切换、自发唱歌（emergent，非可控） |
| **Kokoro-82M** | ❌ 无 | — | ❌ | ⚠️ speed（0.25~4） | 语速 |

##### 情感克隆准确率实测（CosyVoice 3 论文 Table 9，emo2vec-large-plus）

| 模型 | text-related happy/sad/angry | **text-unrelated** happy/sad/angry |
|---|---|---|
| CosyVoice3-1.5B + DiffRO-EMO | 0.98 / 0.68 / 0.84 | **0.98 / 0.50 / 0.68** |

**全行业共同短板（论文明确指出）**：
- **happy 最容易**（0.98，几乎不受文本影响）
- **sad / angry 在 text-unrelated 下显著退化**（0.68 → 0.50；0.84 → 0.68）——文本本身不含情感线索时，模型难以凭空生成悲伤/愤怒

##### IndexTTS-2 的情感指标（第三方汇总）

| 指标 | 数值 |
|---|---|
| 情感相似度（ES） | **0.887**（专属情感测试集） |
| 情感 MOS | **4.22 ± 0.12** |
| 支持基础情感 | **7 种**：angry / happy / fear / disgust / sad / surprise / calm（+neutral） |
| 8 维情感向量 | `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]` |

##### GLM-TTS 的 Phoneme-in 消融（很有参考价值）

| 配置 | PER（音素错误率） |
|---|---|
| 无 Phoneme-in | **13.23** |
| **有 Phoneme-in** | **5.14** |

**含义**：中文多音字/生僻字的发音问题，**用「音素+文本混合输入」可以把 PER 降低 61%**。这是在有声书、教育、朗读场景中最实用的一项能力。CosyVoice 3 的拼音/CMU 发音修补（100% 修正率）是同一思路的另一种实现。

##### 时长控制：只有 IndexTTS-2 是认真的

| 模型 | 时长控制 | 精度 |
|---|---|---|
| **IndexTTS-2** | ✅✅ **Specified Duration Mode**（显式指定 token 数）+ **Natural Duration Mode**（自由生成，忠实复刻 prompt 韵律）；比例范围 **0.75x~1.25x** | **误差 < 0.02%（原始时长）/ < 0.03%（0.875x~1.125x）/ 最大 0.067%**；10 秒目标偏差 **≤ 7 ms** |
| IndexTTS-2.5 | `duration_factor`（0.5x ~ 2.0x） | 待核实 |
| CosyVoice 3 | ❌ 仅 `min_token_text_ratio=2` / `max_token_text_ratio=20` 软约束 | — |
| F5-TTS | ❌ 由参考音频长度或长度回归决定 | — |
| 其他 | ❌ | — |

**⚠️ 重要提示**：IndexTTS README 的 News 里有一条「*This functionality is not yet enabled in this release.*」（精确时长控制在该 release 中尚未启用）。但随后的 2.5 版本提供了 `duration_factor`。**实际可用性需在目标版本上实测确认。**

##### 选型建议

| 需求 | 首选 | 次选 |
|---|---|---|
| **副语言 / 细粒度情感**（笑声、呼吸、耳语、停顿） | 🥇 **Fish S2 Pro**（15k tag，需商业授权） | 🥈 **GLM-TTS**（GRPO 显式优化笑声/呼吸，Apache-2.0） |
| **音色-情感独立控制**（同音色换情绪 / 同情绪换音色） | 🥇 **IndexTTS-2**（GRL 解耦） | — |
| **视频配音 / 口型同步（精确时长）** | 🥇 **IndexTTS-2**（唯一） | — |
| **发音精确控制**（多音字、生僻字、术语） | 🥇 **CosyVoice 3**（100% 修正率，Apache-2.0） | 🥈 **GLM-TTS**（Phoneme-in，PER 13.23→5.14） |
| **参数化虚拟说话人**（性别/音调/语速） | 🥇 **Spark-TTS**（非商用） | — |
| **纯描述造全新音色**（无参考音频） | 🥇 **VoxCPM2 Voice Design** | 🥈 Fish S2 Pro |
| **中粒度指令**（高兴/难过/方言/语速） | 🥇 **CosyVoice 3**（>100 风格，Apache-2.0） | 🥈 GLM-TTS |


### 10. TTS 指标

#### TTS 质量指标汇总

##### A. Seed-TTS-eval（最通用的横向基准）

> ⚠️ **口径警告**：不同来源的数值有出入。**只可比同一张表内的数字。**

**表 A-1：CosyVoice 官方 README（WavLM SIM 口径，2025-12）**

| Model | Open | Size | test-zh CER↓ | test-zh SS↑ | test-en WER↓ | test-en SS↑ | test-hard CER↓ | test-hard SS↑ |
|---|---|---|---|---|---|---|---|---|
| Human | - | - | 1.26 | 75.5 | 2.14 | 73.4 | - | - |
| Seed-TTS | ❌ | - | 1.12 | 79.6 | 2.25 | 76.2 | 7.59 | 77.6 |
| MiniMax-Speech | ❌ | - | 0.83 | 78.3 | 1.65 | 69.2 | - | - |
| F5-TTS | ✅ | 0.3B | 1.52 | 74.1 | 2.00 | 64.7 | 8.67 | 71.3 |
| Spark TTS | ✅ | 0.5B | 1.2 | 66.0 | 1.98 | 57.3 | - | - |
| CosyVoice2 | ✅ | 0.5B | 1.45 | 75.7 | 2.57 | 65.9 | 6.83 | 72.4 |
| FireRedTTS2 | ✅ | 1.5B | 1.14 | 73.2 | 1.95 | 66.5 | - | - |
| Index-TTS2 | ✅ | 1.5B | 1.03 | 76.5 | 2.23 | 70.6 | 7.12 | 75.5 |
| VibeVoice-1.5B | ✅ | 1.5B | 1.16 | 74.4 | 3.04 | 68.9 | - | - |
| VibeVoice-Realtime | ✅ | 0.5B | - | - | 2.05 | 63.3 | - | - |
| HiggsAudio-v2 | ✅ | 3B | 1.50 | 74.0 | 2.44 | 67.7 | - | - |
| VoxCPM | ✅ | 0.5B | 0.93 | 77.2 | 1.85 | 72.9 | 8.87 | 73.0 |
| GLM-TTS | ✅ | 1.5B | 1.03 | 76.1 | - | - | - | - |
| GLM-TTS RL | ✅ | 1.5B | 0.89 | 76.4 | - | - | - | - |
| **Fun-CosyVoice3-0.5B-2512** | ✅ | 0.5B | 1.21 | **78.0** | 2.24 | 71.8 | **6.71** | 75.8 |
| **Fun-CosyVoice3-0.5B-2512_RL** | ✅ | 0.5B | **0.81** | 77.4 | **1.68** | 69.5 | **5.44** | 75.0 |

**表 A-2：VoxCPM2 技术报告（arXiv:2606.06928，第三方交叉验证）**

| Model | Parameters | Open | test-EN WER↓ | test-EN SIM↑ | test-ZH CER↓ | test-ZH SIM↑ | test-Hard CER↓ | test-Hard SIM↑ |
|---|---|---|---|---|---|---|---|---|
| MegaTTS3 | 0.5B | ❌ | 2.79 | 77.1 | 1.52 | 79.0 | - | - |
| DiTAR | 0.6B | ❌ | 1.69 | 73.5 | 1.02 | 75.3 | - | - |
| **CosyVoice3** | 0.5B | ❌* | 2.02 | 71.8 | 1.16 | 78.0 | 6.08 | 75.8 |
| **CosyVoice3** | 1.5B | ❌* | 2.22 | 72.0 | 1.12 | 78.1 | 5.83 | 75.8 |
| Seed-TTS | - | ❌ | 2.25 | 76.2 | 1.12 | 79.6 | 7.59 | 77.6 |
| MiniMax-Speech | - | ❌ | 1.65 | 69.2 | 0.83 | 78.3 | - | - |
| **F5-TTS** | 0.3B | ✅ | 2.00 | **67.0** | 1.53 | 76.0 | 8.67 | 71.3 |
| MaskGCT | 1B | ✅ | 2.62 | 71.7 | 2.27 | 77.4 | - | - |
| CosyVoice | 0.3B | ✅ | 4.29 | 60.9 | 3.63 | 72.3 | 11.75 | 70.9 |
| CosyVoice2 | 0.5B | ✅ | 3.09 | 65.9 | 1.38 | 75.7 | 6.83 | 72.4 |
| **SparkTTS** | 0.5B | ✅ | 3.14 | 57.3 | 1.54 | 66.0 | - | - |
| FireRedTTS | 0.5B | ✅ | 3.82 | 46.0 | 1.51 | 63.5 | 17.45 | 62.1 |
| FireRedTTS-2 | 1.5B | ✅ | 1.95 | 66.5 | 1.14 | 73.6 | - | - |
| Qwen2.5-Omni | 7B | ✅ | 2.72 | 63.2 | 1.70 | 75.2 | 7.97 | 74.7 |
| Qwen3-Omni | 30B-A3B | ✅ | 1.39 | - | 1.07 | - | - | - |
| OpenAudio-s1-mini | 0.5B | ✅ | 1.94 | 55.0 | 1.18 | 68.5 | 23.37 | 64.3 |
| **IndexTTS2** | 1.5B | ✅ | 2.23 | 70.6 | 1.03 | 76.5 | 7.12 | 75.5 |
| **VibeVoice** | 1.5B | ✅ | 3.04 | 68.9 | 1.16 | 74.4 | - | - |
| **HiggsAudio-v2** | 3B | ✅ | 2.44 | 67.7 | 1.50 | 74.0 | **55.07** | 65.6 |
| **VoxCPM-0.5B** | 0.6B | ✅ | 1.85 | 72.9 | 0.93 | 77.2 | 8.87 | 73.0 |
| VoxCPM1.5 | 0.8B | ✅ | 2.12 | 71.4 | 1.18 | 77.0 | 7.74 | 73.1 |
| MOSS-TTS | - | ✅ | 1.85 | 73.4 | 1.20 | **78.8** | - | - |
| **Qwen3-TTS** | 1.7B | ✅ | 1.23 | 71.7 | 1.22 | 77.0 | 6.76 | 74.8 |
| **FishAudio S2** | 4B | ✅ | **0.99** | - | **0.54** | - | 5.99 | - |
| **LongCat-Audio-DiT** | 3.5B | ✅ | 1.50 | **78.6** | 1.09 | **81.8** | 6.04 | **79.7** |
| **VoxCPM2** | 2B | ✅ | 1.84 | 75.3 | 0.97 | 79.5 | 8.13 | 75.3 |

> \* VoxCPM2 报告把 CosyVoice3 标为 ❌（非开源），**这是标注口径问题**（0.5B 权重实际已开源，1.5B 未开源）。引用时需注意。

##### B. GLM-TTS 论文（arXiv:2512.14291）中的基线表

| Model | Parameters | test-zh CER↓ | test-zh SIM↑ | test-en WER↓ | test-en SIM↑ |
|---|---|---|---|---|---|
| IndexTTS 2 | 1.5B | 1.03 | 76.5 | 2.23 | 70.6 |
| VibeVoice | 1.5B | 1.16 | 74.4 | 3.04 | 68.9 |
| HiggsAudio-v2 | 3B | 1.50 | 74.0 | 2.44 | 67.7 |
| VoxCPM | 0.5B | 0.93 | 77.2 | 1.85 | 72.9 |
| **GLM-TTS** | 1.5B | 1.03 | 76.1 | 2.23 | 67.2 |
| **GLM-TTS_RL** | 1.5B | **0.89** | 76.4 | 1.91 | 68.1 |

##### C. 差异与陷阱（**必读**）

| 陷阱 | 说明 |
|---|---|
| **SS/SIM 口径不一致** | WavLM vs ERes2Net vs CAM++ 三种口径可差 **5~8 个点**。CosyVoice3-0.5B test-zh SS：WavLM 0.780 / ERes2Net 0.840 |
| **同一模型不同来源数值不同** | F5-TTS test-en SIM：CosyVoice README 64.7 vs VoxCPM2 报告 67.0；SparkTTS test-zh CER：1.2 vs 1.54；IndexTTS2 test-en WER：2.23（两表一致）✅ |
| **README 与论文数值不同** | CosyVoice3-0.5B test-zh CER：README 1.21 vs 论文 Table 4 1.16；test-hard CER：6.71 vs 6.08 |
| **HiggsAudio-v2 的 hard 集灾难** | test-hard CER **55.07**（VoxCPM2 报告）——远超所有其他模型（5.44~8.87）。说明它在绕口令/罕见词上极不稳定。**第三方「情感最强」的评价不能掩盖这个硬伤** |
| **FireRedTTS（1代）的 hard 集** | test-hard CER 17.45，同样很差 |
| **OpenAudio-s1-mini 的 hard 集** | test-hard CER 23.37 |
| **语言数 ≠ 质量** | Fish S2-Pro 在高棉/老挝/缅甸 CER 75~87%；VoxCPM2 在阿拉伯语 13.05、印地语 19.70 |

##### D. 主观 MOS（第三方评测，非官方，仅供参考）

| 模型 | 中文自然度 | 英文自然度 | 情感表达 | 克隆相似度 |
|---|---|---|---|---|
| Higgs Audio V2 | 4.6 | 4.5 | **4.7** | 4.4 |
| CosyVoice 2 | **4.7** | 4.1 | — | — |
| VibeVoice-1.5B | 4.2 | 4.4 | — | — |
| Kokoro-82M | 3.9 | 4.3 | — | — |

> 来源：声网 2026-05 评测（**第三方，非官方，测试条件未公开，可信度中等**）

##### E. 各模型的质量短板总结

| 模型 | 最强项 | 最弱项 |
|---|---|---|
| **CosyVoice 3** | **test-hard CER 5.44（RL 版）全场第一**；test-zh SS 78.0 开源第一 | test-en SS 69.5（RL 版）偏低；日语 CER 9%、韩语 ~6%、法语 12.9 |
| **FishAudio S2** | **test-zh CER 0.54 / test-en WER 0.99 全场第一**（含闭源） | 长尾语言（高棉/老挝/缅甸）CER 75~87%；未公布 SIM |
| **VoxCPM2** | test-zh SIM **79.5** 全场第一；48kHz；30 语言 | test-hard CER 8.13 偏弱；阿拉伯语/印地语极差 |
| **IndexTTS-2** | **时长控制唯一**；情感解耦唯一；test-zh CER 1.03 | 仅中英；en WER 2.23 中等 |
| **GLM-TTS** | GRPO 四奖励；Phoneme-in；test-zh CER 0.89（RL） | test-en SIM 67.2 偏低（英文数据 < 中文一半） |
| **F5-TTS** | 部署最简单；吞吐最高 | test-en SIM 64.7 偏低；无情感控制；**权重不可商用** |
| **Spark-TTS** | 轻量、参数可控 | **test-zh SIM 66.0 / test-en SIM 57.3 全场最低**；16kHz；不可商用 |
| **HiggsAudio-v2** | 情感表达（第三方评价） | **test-hard CER 55.07**（灾难级） |
| **VibeVoice-1.5B** | 90 分钟 / 4 说话人长音频 | test-en WER 3.04 偏弱；**TTS 代码已下架** |
| **Kokoro-82M** | 极致轻量、CPU 可跑 | **不支持克隆**；中文自然度 3.9 偏低 |


### 11. 延迟与吞吐

#### 延迟与吞吐

##### A. 有官方/半官方数据的模型

| 模型 | 指标 | 数值 | **测试条件** | 来源 |
|---|---|---|---|---|
| **Fish Audio S2 Pro** | RTF | **0.195** | 单卡 **NVIDIA H200** + SGLang | fish-speech README |
| | TTFA（首包） | **~100 ms** | 同上 | 同上 |
| | 吞吐 | **3000+ acoustic tokens/s**（RTF < 0.5 下） | 同上 | 同上 |
| **VoxCPM2** | RTF | **~0.30**（PyTorch）/ **~0.13**（Nano-vLLM） | **RTX 4090** | VoxCPM README |
| VoxCPM1.5 | RTF | ~0.15 / ~0.08 | RTX 4090 | 同上 |
| VoxCPM-0.5B | RTF | ~0.17 / ~0.10 | RTX 4090 | 同上 |
| **F5-TTS Base (Vocos)** | 平均延迟 | **253 ms** | 单卡 **L20**，26 组 prompt×text，**16 NFE**，**concurrency 2**，Client-Server | F5-TTS README |
| | RTF | **0.0394** | 同上（Client-Server） | 同上 |
| | RTF | **0.0402** | L20，**batch_size 1**，Offline **TRT-LLM** | 同上 |
| | RTF | **0.1467** | L20，batch_size 1，Offline **PyTorch** | 同上 |
| **Spark-TTS-0.5B** | 平均延迟 / RTF | **876.24 ms / 0.1362** | 单卡 **L20**，26 组（共 169 秒音频），**concurrency 1**，Triton+TRT-LLM | Spark-TTS GitHub |
| | | **920.97 ms / 0.0737** | L20，**concurrency 2** | 同上 |
| | | **1611.51 ms / 0.0704** | L20，**concurrency 4** | 同上 |
| **CosyVoice 3** | 首包延迟 | **150 ms** | 未注明 | README / HF 模型卡 |
| | RTF | **待核实**（官方未公布） | — | — |
| **VibeVoice-Realtime-0.5B** | 首 audible 延迟 | **~300 ms** | 未注明 | VibeVoice README |
| **Kokoro-82M** | 实时率 | **~32×**（T4）；**2~3×**（4 核 CPU） | T4 / CPU | 第三方评测（非官方） |
| **GLM-TTS** | 首包延迟 | 400 ms（vs ElevenLabs 75 ms） | 第三方对比 | glm-tts.com（非官方） |

##### B. 加速倍数（有对比条件的）

| 模型 | 加速手段 | 收益 |
|---|---|---|
| **F5-TTS** | TRT-LLM vs PyTorch（offline, bs=1） | **0.0402 vs 0.1467 → 3.65×** |
| **CosyVoice 2/3** | Triton + TRT-LLM（LLM 部分）vs HF transformers | **4×**（README 原文） |
| **VoxCPM2** | Nano-vLLM vs PyTorch | **0.13 vs 0.30 → 2.3×** |
| VoxCPM1.5 | Nano-vLLM vs PyTorch | 0.08 vs 0.15 → 1.9× |
| VoxCPM-0.5B | Nano-vLLM vs PyTorch | 0.10 vs 0.17 → 1.7× |
| **Spark-TTS** | concurrency 1 → 4 | RTF 0.1362 → 0.0704 → **1.9×**（但延迟从 876ms 涨到 1611ms） |

##### C. ⚠️ 横向比较的困难

**几乎没有两个模型在相同硬件、相同条件、相同文本长度下测过。** 已知的可比点：

1. **F5-TTS vs Spark-TTS 在同一 L20 上、同样 26 组 prompt**：
   - F5-TTS：253 ms / RTF 0.0394 @ conc 2
   - Spark-TTS：876 ms / RTF 0.0737 @ conc 2
   - → **F5-TTS 约快 1.9 倍**（RTF 口径），延迟低 **3.5 倍**
   - 但注意 F5-TTS 用 16 NFE，Spark-TTS 是 AR 架构，两者范式不同

2. **首包延迟**（有限可比）：
   - Fish S2 Pro ~100 ms（H200）< CosyVoice 3 150 ms（未注明）< VibeVoice-Realtime ~300 ms（未注明）
   - ⚠️ CosyVoice 3 未注明硬件，Fish S2 Pro 用 H200（顶级卡），**不公平**

##### D. 各模型未公开的性能数据（**容量规划必须自建 benchmark**）

| 模型 | 缺失项 |
|---|---|
| **CosyVoice 3** | RTF、显存、吞吐、首 token 延迟 —— **全部未公布** |
| **IndexTTS-2** | 首包延迟、RTF、吞吐（只有 FP16 显存 ~7.8GB 的社区数据） |
| **GLM-TTS** | RTF、显存、吞吐 |
| **VibeVoice-1.5B** | RTF、显存（第三方称 ≥8GB） |
| **HiggsAudio-v2** | 全部 |
| **Fish Audio S2 Pro** | 显存（只给了 H200 上的 RTF） |
| **F5-TTS** | 显存 |

##### E. 范式对延迟/吞吐的影响（选型的关键判据）

| 场景 | 最优范式 | 代表 | 理由 |
|---|---|---|---|
| **实时对话 / 语音助手**（首包延迟敏感） | **AR（A/B/C 路线）** | CosyVoice 3、Fish S2 Pro、VoxCPM2 | 首包延迟恒定（150ms 级），不随文本长度增长 |
| **离线批量合成**（吞吐敏感） | **纯 NAR（D 路线）** | **F5-TTS** | RTF **0.0394** @ TRT-LLM，比 AR 方案快一个数量级 |
| **长音频 / 播客**（10~90 分钟） | **超低帧率连续 tokenizer** | **VibeVoice**（7.5 Hz，90 分钟） | 帧率越低序列越短，才可能塞进上下文 |
| **高并发多租户** | **有 continuous batching 的 AR** | Fish S2 Pro（SGLang）、VoxCPM2（vLLM-Omni） | PagedAttention + prefix caching 天然适配 |
| **边缘 / CPU** | **极轻量** | **Kokoro-82M**（2~3× 实时 @ 4 核 CPU） | 82M 参数，400MB 内存 |

##### F. 自建 benchmark 的标准化建议

```
固定条件清单（缺一不可）：
  硬件型号 + 驱动/CUDA 版本
  精度（fp16 / bf16 / fp32 / 量化位宽）
  batch size / concurrency
  文本长度分布（短句 <20 字 / 中句 20~100 字 / 长句 >100 字）
  参考音频长度
  是否流式
  测量项：首包延迟（TTFA）、尾包延迟、RTF、吞吐（chars/s 或 tokens/s）、峰值显存
  重复次数 ≥ 100，取 P50 / P95 / P99（不要只报均值）
```


### 12. 推理框架支持

#### 推理框架支持矩阵

| 模型 | PyTorch 原生 | vLLM | SGLang | Triton/TRT-LLM | FastAPI/gRPC | Docker | Gradio/WebUI | MLX/Apple | 端侧（GGUF/ONNX/cpp） |
|---|---|---|---|---|---|---|---|---|---|
| **CosyVoice 3** | ✅ | ✅ 0.11.x+/0.9.0 | ❌ | ✅（NVIDIA 贡献） | ✅ 两者都有 | ✅ | ✅ | ⚠️ 社区 | ❌ |
| **Fish S2 Pro** | ✅ | ✅ vLLM-Omni recipe | ✅ **SGLang-Omni（原生）** | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **VoxCPM2** | ✅ | ✅ **vLLM-Omni 官方** | ❌ | ❌ | ✅（Nano-vLLM 自带） | ⚠️ | ✅ `app.py` | ✅ VoxCPMANE (ANE) | ✅✅ **llama.cpp-omni / VoxCPM.cpp / ONNX / voxcpm_rs** |
| **F5-TTS** | ✅ | ⚠️ 社区 | ❌ | ✅（Yuekai Zhang） | ⚠️（Gradio/CLI） | ✅ ghcr.io | ✅ | ✅ f5-tts-mlx | ✅ F5-TTS-ONNX |
| **IndexTTS-2** | ✅ | ❌（仅 2.5 有 recipe） | ❌ | ❌ | ⚠️ WebUI + Python API | ❌ | ✅ | ❌ | ❌ |
| **IndexTTS-2.5** | ✅ | ✅ vLLM recipe | ❌ | ❌ | ⚠️ | ❌ | ✅ | ❌ | ❌ |
| **VibeVoice** | ✅ | ✅ 仅 ASR 侧 | ❌ | ❌ | ⚠️ | ❌ | ✅（TTS 已 Disabled） | ❌ | ⚠️ VibeASR.cpp（仅 ASR） |
| **GLM-TTS** | ✅ | 待核实 | ❌ | ❌ | ⚠️ Gradio + CLI | ❌ | ✅ | ❌ | ❌ |
| **Spark-TTS** | ✅ | ❌ | ❌ | ✅（L20 实测） | ❌ | ❌ | ✅ `webui.py` | ❌ | ❌ |
| **Kokoro-82M** | ✅ | ❌ | ❌ | ❌ | ⚠️（pip 包） | ❌ | ✅ | ❌ | ✅（多种 CPU 部署） |
| **GPT-SoVITS** | ✅ | ❌ | ❌ | ❌ | ⚠️ 社区 API | ⚠️ 社区 | ✅ | ❌ | ❌ |

##### 关键集成细节

###### CosyVoice 3
```bash
# vLLM（版本约束极严）
pip install vllm==v0.11.0 transformers==4.57.1 numpy==1.26.4   # 或 0.9.0 + transformers 4.51.3
python vllm_example.py
# 注意：< 0.9.0 不支持；0.10.x 未测试
# Triton + TRT-LLM
cd runtime/triton_trtllm && docker compose up -d     # LLM 部分 4× 加速
```

###### Fish Audio S2 Pro
```bash
# SGLang-Omni（原生推荐）
# 见 https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md
# vLLM-Omni
# 见 https://github.com/vllm-project/vllm-omni/blob/main/recipes/fishaudio/Fish-Speech-S2-Pro.md
# 官方文档：https://speech.fish.audio/server/
```

###### VoxCPM2
```bash
# 官方 vLLM-Omni（OpenAI 兼容）
vllm serve openbmb/VoxCPM2 --omni --port 8000
curl http://localhost:8000/v1/audio/speech   -H "Content-Type: application/json"   -d '{"model":"openbmb/VoxCPM2","input":"Hello","voice":"default"}' --output out.wav

# Nano-vLLM（高吞吐）
pip install nano-vllm-voxcpm

# llama.cpp-omni（端侧，CPU/Metal/CUDA/Vulkan）
./build/bin/voxcpm2-cli -t "Hello" -o output.wav     VoxCPM2-BaseLM-Q8_0.gguf VoxCPM2-Acoustic-F16.gguf
# RTF ~1.76 (Q8_0) on Apple M4 Pro / Metal
```

###### F5-TTS
```bash
pip install f5-tts                    # 纯推理
# 或
pip install -e .                      # 含训练/微调
f5-tts_infer-cli --model F5TTS_v1_Base --ref_audio x.wav --ref_text "..." --gen_text "..."
f5-tts_infer-gradio --port 7860 --host 0.0.0.0
# Triton + TRT-LLM：见 src/f5_tts/runtime/triton_trtllm/README.md
```

###### IndexTTS-2 / 2.5
```bash
uv sync --all-extras                  # 必须用 uv，不支持 pip/conda 直接安装
uv run webui.py --version 2 --model_dir ./checkpoints_2
uv run webui.py --fp16 --deepspeed    # FP16 + DeepSpeed
# Python API
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints_2/config.yaml", model_dir="checkpoints_2", use_fp16=False)
# vLLM recipe 仅 2.5：https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5
```

##### 部署成熟度分层

**Tier 1（开箱即用的生产栈）**：
- **VoxCPM2** — vLLM-Omni 提供 OpenAI 兼容 `/v1/audio/speech`；readthedocs 完整文档；端侧全平台
- **Fish S2 Pro** — SGLang-Omni 原生支持；speech.fish.audio 完整文档

**Tier 2（组件齐全，需自行封装生产层）**：
- **CosyVoice 3** — vLLM + Triton/TRT-LLM + FastAPI + gRPC + Docker 全有，但**无鉴权、无限流、无监控、无 OpenAI 兼容**
- **F5-TTS** — Triton/TRT-LLM + Docker（ghcr.io）+ ONNX + MLX

**Tier 3（研究级，生产化需大量工作）**：
- **IndexTTS-2 / 2.5** — 只有 WebUI + Python API；依赖 `uv`（不支持 pip/conda）
- **GLM-TTS** — 只有 Gradio + CLI
- **VibeVoice** — **TTS 代码已下架**，只剩权重
- **Spark-TTS / GPT-SoVITS** — WebUI 为主；Spark-TTS 有 Triton 参考实现但无训练代码

##### 「OpenAI 兼容接口」是 2026 年的新分水岭

| 模型 | 原生 `/v1/audio/speech` | 说明 |
|---|---|---|
| **VoxCPM2** | ✅ | `vllm serve --omni`，可直接用任何 OpenAI 客户端 |
| **Fish S2 Pro** | ✅ | SGLang-Omni / vLLM-Omni |
| Kokoro-82M | ✅（第三方托管） | DeepInfra / Together AI / 等 |
| **CosyVoice 3** | ❌ | 需自行封装 |
| IndexTTS-2 / GLM-TTS / F5-TTS / Spark-TTS / VibeVoice | ❌ | 需自行封装 |

**选型含义**：如果你的系统已经按 OpenAI 音频 API 设计，VoxCPM2 / Fish S2 Pro 可以**零改造接入**；CosyVoice 3 等需要额外写一层适配（工作量约 1~3 人日）。


### 13. 资源需求

#### 资源需求对照

##### A. 显存（VRAM）

| 模型 | 参数量 | 官方 VRAM | 精度 | 来源 |
|---|---|---|---|---|
| **VoxCPM-0.5B** | 0.6B | **~5 GB** | — | VoxCPM README（官方） |
| **VoxCPM1.5** | 0.8B | **~6 GB** | — | 同上 |
| **VoxCPM2** | 2B | **~8 GB** | — | 同上 |
| **IndexTTS-2** | 1.5B | **~7.8 GB** | **FP16** | 社区实测（多来源一致）；另一来源称 GPU 显存 **≥16GB（推荐 24GB 以上）** |
| **Spark-TTS** | 0.5B | **1~2 GB** | — | 第三方评测 |
| **Kokoro-82M** | 82M | **~400 MB**（内存，非显存） | — | 第三方评测 |
| **CosyVoice 3** | 0.5B | **待核实**（架构估算 3~5 GB） | — | 官方未公布 |
| **F5-TTS** | 0.3B | 待核实 | — | 官方未公布 |
| **VibeVoice-1.5B** | 1.5B | 待核实（第三方称 ≥8 GB） | BF16（2.7B safetensors） | — |
| **GLM-TTS** | 1.5B | 待核实 | — | — |
| **HiggsAudio-v2** | 3B | 待核实（第三方称 ≥8 GB） | — | — |
| **Fish Audio S2 Pro** | 4B + 400M | 待核实 | — | 官方只给了 H200 上的 RTF |
| **GPT-SoVITS** | ~1B 级 | 待核实 | — | — |

##### B. 权重体积

| 模型 | 体积 | 备注 |
|---|---|---|
| **Kokoro-82M** | 量化后 **86~92 MB** | 可在树莓派级别设备跑 |
| **CosyVoice 3-0.5B** | bf16 **~2.1 GB**；MLX 4bit **~1.2 GB** / 8bit ~1.4 GB / 8bit-full ~1.6 GB | MLX 组件拆分：LLM 467MB(4bit) + DiT 634MB(fp16) + HiFi-GAN 79MB(fp16) |
| VibeVoice-1.5B | 2.7B params / **BF16** safetensors | HF 上 opencsg 标 2.7B |
| VibeVoice-ASR-BitNet | 4.62 GB → **1.58 GB**（I8_S + I2_S 异构量化） | CPU 推理，RTF<1 @ 3+ 线程 |
| 其他 | 待核实 | — |

##### C. 训练硬件（已知部分）

| 模型 | 训练配置 | 来源 |
|---|---|---|
| **IndexTTS-2** | **8×NVIDIA A100 80GB**，AdamW，**lr 2e-4**，训练 **3 周** | 第三方汇总（tts.wiki，源自论文） |
| **Kokoro-82M** | **1000 A100-80GB GPU 小时，约 $1000**<br>（v0.19: 500h @ $0.8/h；v1.0: 500h @ $1.2/h） | HF 模型卡（**唯一公开完整训练成本的模型**） |
| **Fish Audio S2 Pro** | 10M+ 小时数据，GRPO RL | 未公布硬件 |
| **VoxCPM2** | 2M+ 小时数据 | 未公布硬件 |
| **CosyVoice 3** | ~1M 小时数据 | **完全未公布**（超参、GPU 型号、卡数全无） |
| **F5-TTS** | Emilia 系数据；提供完整 finetune 脚本 | 社区常见单卡 24GB 微调 |
| **VoxCPM 微调** | SFT/LoRA，**5~10 分钟音频**即可适配 | VoxCPM 官方文档 |

##### D. 微调资源

| 模型 | SFT | LoRA | 最小数据 | 硬件建议 |
|---|---|---|---|---|
| **CosyVoice 3** | ✅（官方脚本） | ⚠️ **待核实** | 待核实 | A100 80G 起 |
| **VoxCPM2 / 1.5** | ✅ | ✅ | **5~10 分钟音频** | 未公布 |
| **GLM-TTS** | 待核实 | ✅（官方定位为成本优化手段） | 待核实 | 未公布 |
| **F5-TTS** | ✅ | ✅（社区） | 未公布 | 单卡 24GB（社区经验） |
| **Spark-TTS** | ❌ **训练代码未发布** | ❌ | — | — |
| **GPT-SoVITS** | ✅ **核心卖点** | ⚠️ | **1 分钟音频** | 消费级 GPU |
| IndexTTS-2 | ⚠️ 待核实 | ⚠️ 待核实 | — | — |
| Kokoro-82M | ❌ | ❌ | — | — |

##### E. 运行环境约束（**部署踩坑高发区**）

| 模型 | Python | 包管理器 | 特殊依赖 | 备注 |
|---|---|---|---|---|
| **CosyVoice 3** | **3.10**（强约束） | conda/pip | **pynini==2.1.5（必须 conda 装）**、sox、`git clone --recursive` | `ttsfrd` whl 是 cp310，锁死 3.10 |
| **IndexTTS-2** | 待核实 | **uv（必须）** | CUDA Toolkit **12.8+**；Windows 上 DeepSpeed 难装 | README 明确「uv sync --all-extras」，**不支持 pip/conda 直接安装** |
| **F5-TTS** | **≥3.10**（推荐 3.11） | conda/pip | ffmpeg；支持 CUDA / ROCm / XPU / Apple Silicon | 多平台文档最清晰（含 ROCm 版本矩阵） |
| **Fish Audio** | **3.12**（用户资料） | — | — | 待核实 |
| **Spark-TTS** | **3.12** | conda/pip | — | README: `conda create -n sparktts -y python=3.12` |
| **VoxCPM2** | **≥3.10 (<3.13)** | pip | PyTorch ≥2.5.0，CUDA ≥12.0 | 支持 cuda / mps / cpu / cuda:N |
| **Kokoro-82M** | 宽松 | pip | `espeak-ng`（apt） | 最易装 |

##### F. 硬件平台覆盖

| 模型 | NVIDIA | AMD(ROCm) | Intel(XPU) | Apple Silicon | CPU | 移动端/边缘 |
|---|---|---|---|---|---|---|
| **VoxCPM2** | ✅ | ✅（vLLM-Omni） | ✅（vLLM-Omni） | ✅（ANE） | ✅ | ✅✅ **最全** |
| **F5-TTS** | ✅ | ✅（README 有版本矩阵） | ✅（IPEX） | ✅（f5-tts-mlx） | ⚠️ | ✅（ONNX） |
| **CosyVoice 3** | ✅✅ | ❓ | ❓ | ⚠️（社区 MLX） | ⚠️ | ❌ |
| Fish S2 Pro | ✅✅（H200） | ❓ | ❓ | ❌ | ❌ | ❌ |
| IndexTTS-2 | ✅ | ❌ | ❌ | ❌ | ⚠️（理论） | ❌ |
| Kokoro-82M | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅✅ | ✅ |
| Spark-TTS | ✅ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| VibeVoice | ⚠️ | ❌ | ❌ | ❌ | ⚠️（ASR 侧 VibeASR.cpp） | ⚠️ |

**结论**：**VoxCPM2 是硬件覆盖最广的**（NVIDIA / AMD / Intel / Apple ANE / CPU / 端侧全栈）；**F5-TTS 的多平台文档最清晰**；**CosyVoice 3 服务端强但端侧为零**。

##### G. 量化友好度（**flow matching 类模型的共同约束**）

| 组件 | 可否量化 | 依据 |
|---|---|---|
| **LLM（AR 部分）** | ✅ **可 4bit/8bit** | CosyVoice 3 MLX 变体：LLM 4bit（group_size=64） |
| **DiT / Flow Matching** | ❌ **不建议** | MLX 模型卡原文：「Flow: fp16 (**flow matching is sensitive to quantization**)」 |
| **声码器** | ✅ 可 fp16 | CosyVoice 3 MLX：HiFi-GAN fp16 + weight norm 折叠；79MB |

**这一条对所有 flow matching 类 TTS 通用**（CosyVoice 3、F5-TTS、VoxCPM2 的 LocDiT、GLM-TTS）。**显存优化时只量化 LLM，声学模型保持 fp16。**


### 14. 服务化栈

#### 服务化栈对照

##### 各模型的官方 serving 组件

| 模型 | 官方 serving 方案 | 并发控制 | 流式 | OpenAI 兼容 | 监控 |
|---|---|---|---|---|---|
| **CosyVoice 3** | FastAPI + gRPC + Docker + vLLM + Triton/TRT-LLM | `--max_conc 4` | ✅ bistream | ❌ | ❌ |
| **Fish S2 Pro** | **SGLang-Omni** + vLLM-Omni + Docker | SGLang continuous batching | ✅ | ✅ | ❌ |
| **VoxCPM2** | **vLLM-Omni**（官方）+ Nano-vLLM（含 FastAPI）+ Gradio + llama.cpp-omni | PagedAttention + continuous batching | ✅ `generate_streaming` | ✅ `/v1/audio/speech` | ❌ |
| **F5-TTS** | Gradio + CLI + Docker（ghcr.io）+ Triton/TRT-LLM | Triton dynamic batching | ⚠️ chunk inference | ❌ | ❌ |
| **IndexTTS-2** | WebUI（Gradio）+ Python API + DeepSpeed | ❌ | ⚠️ | ❌ | ❌ |
| **IndexTTS-2.5** | 同上 + **vLLM recipe** | vLLM | ⚠️ | ❌ | ❌ |
| **VibeVoice** | Gradio（TTS Demo 已 Disabled） | ❌ | ✅ Realtime-0.5B | ❌ | ❌ |
| **GLM-TTS** | Gradio（`glmtts_inference.py`）+ CLI | ❌ | ⚠️ | ❌ | ❌ |
| **Spark-TTS** | WebUI + CLI + Triton/TRT-LLM 参考实现 | Triton | ❌ | ❌ | ❌ |
| **Kokoro-82M** | pip 包（`KPipeline`）+ 第三方托管 API | — | ⚠️ | ✅（第三方） | ❌ |

##### 生产化缺口（**所有开源 TTS 共同的短板**）

| 缺口 | 说明 |
|---|---|
| **鉴权 / 限流** | 所有官方 server 均无认证机制 |
| **监控指标** | 无一个模型自带 Prometheus / OpenTelemetry 埋点 |
| **请求队列 / 优先级** | 需自行实现 |
| **K8s 编排 manifest** | 只有 CosyVoice 3 / F5-TTS / VoxCPM2 有 Docker，无人提供官方 Helm/K8s |
| **模型热更新** | 均需自行实现 |
| **音频后处理** | 响度归一化、格式转换、静音裁剪均需外部处理 |
| **多实例负载均衡** | 需自行搭建 |

##### 三种典型生产架构

###### 架构 A：单机 / 小规模（最简）
```
客户端 → API Gateway（鉴权+限流）→ 官方 server（FastAPI/Gradio）→ GPU
```
- 适用：内部工具、< 100 QPS
- 代表：IndexTTS-2、GLM-TTS、Spark-TTS、F5-TTS 的默认路径

###### 架构 B：高并发服务端（推荐用于 CosyVoice 3 / IndexTTS-2）
```
                    ┌─→ vLLM / SGLang（LLM 部分，continuous batching + prefix caching）
客户端 → API GW ────┤
                    └─→ 声学模型服务（Triton / TorchServe / 自研 gRPC）
```
- 两段分别按最优策略优化；前缀缓存可复用参考音频的 prompt token
- 缺点：需自行拆分模型与实现跨服务调度

###### 架构 C：现代 Omni 服务栈（VoxCPM2 / Fish S2 Pro 的原生路径）
```
vllm serve openbmb/VoxCPM2 --omni --port 8000
        ↓
/v1/audio/speech（OpenAI 兼容，PagedAttention，continuous batching，streaming chunk 交付，多 GPU 开箱即用）
```
- **改造成本最低**：任何 OpenAI 客户端可直接调用
- 缺点：Omni 生态较新（vLLM-Omni / SGLang-Omni 仍在快速演进），需跟版本

##### 生态集成盘点

**VoxCPM2 的生态（最丰富，值得单独列出）**：
| 项目 | 定位 |
|---|---|
| vLLM-Omni | 官方 vLLM 多模态扩展，OpenAI 兼容 + PagedAttention + 多 GPU |
| Nano-vLLM-VoxCPM | 高吞吐，异步 API + FastAPI HTTP server |
| llama.cpp-omni | CPU / Metal / CUDA / Vulkan，GGUF，无 Python |
| VoxCPM.cpp | GGML/GGUF：CPU、CUDA、Vulkan |
| audio.cpp | ggml 统一 C++ 框架，CLI + server |
| VoxCPM-ONNX | ONNX CPU 推理 |
| VoxCPMANE | Apple Neural Engine 后端 |
| voxcpm_rs | Rust 重实现 |
| ComfyUI-VoxCPM / ComfyUI_RH_VoxCPM / ComfyUI-VoxCPMTTS | ComfyUI 节点 |
| TTS WebUI 扩展 | 浏览器端 |
| Video Dubbing Translator | 本地视频配音全链路（vocal 分离 → WhisperX ASR → 翻译 → VoxCPM2 克隆 → lip-sync） |

**F5-TTS 的生态**：
- F5-TTS-ONNX（ONNX Runtime）、f5-tts-mlx（Apple）、Triton+TRT-LLM、ghcr.io Docker、TTS WebUI 集成

**CosyVoice 3 的生态**：
- Triton + TRT-LLM（NVIDIA 官方贡献）、vLLM、FastAPI、gRPC、Docker、ModelScope Gradio Space、社区 MLX 变体、speech-swift（Apple SDK）

##### 选型含义

| 需求 | 推荐 |
|---|---|
| **已有 OpenAI 音频 API 技术栈** | 🥇 **VoxCPM2**（vLLM-Omni 原生 `/v1/audio/speech`） |
| **已有 Triton 基础设施** | 🥇 **CosyVoice 3**（NVIDIA 官方贡献的 TRT-LLM backend，4× 加速）或 **F5-TTS** |
| **要最高吞吐（离线批量）** | 🥇 **F5-TTS**（NAR，RTF 0.0394 @ TRT-LLM） |
| **要端云一体** | 🥇 **VoxCPM2**（llama.cpp / ONNX / ANE / Rust 全覆盖） |
| **要多说话人长音频** | 🥇 VibeVoice（但代码已下架，只剩权重，需自行实现推理） |
| **快速验证原型** | 🥇 Kokoro-82M（pip install 即用）或 F5-TTS（pip install f5-tts） |


### 15. 已知坑与限制

#### 已知问题与坑（按模型）

##### Fish Audio（`fishaudio/fish-speech`）

| 问题 | 说明 |
|---|---|
| 🔴 **许可证整体变更** | 2026-03-07 起，仓库 LICENSE 从 Apache-2.0 变为 **FISH AUDIO RESEARCH LICENSE**。**所有商用（含企业内部运营）都需要单独书面授权** |
| 🔴 **Fish-Speech 1.5 已弃用** | **2026-02-28 官方 deprecated**，被 S2 取代。社区评测的 ELO 1033 vs 1339 差异（不同榜单口径） |
| ⚠️ **Python 3.12 强约束**（用户资料，待核实） | 部署前需确认 |
| ⚠️ **长尾语言质量灾难** | 高棉 CER 75.15%、老挝 87.40%、缅甸 85.27%（VoxCPM2 报告的 30 语基准）。**「80+ 语言」不等于 80 个可用语言** |
| ⚠️ **命名混乱** | Fish Speech / fish-speech 1.5 / OpenAudio S1 / S2 / S2 Pro 混用；文档与第三方教程常指错 |
| ⚠️ **权重与代码许可错配（历史遗留）** | OpenAudio S1-mini：代码 Apache-2.0，权重 **CC-BY-NC-SA-4.0** |
| ⚠️ **未公布 SIM 指标** | Seed-TTS-eval 只给了 WER/CER，无说话人相似度 |
| ⚠️ **未公布显存** | 只给了 H200 上的 RTF |
| ⚠️ **「不得用于改进任何基础生成式 AI 模型」**（§IV.b） | 用 Fish 的输出训练自有模型会违约 |
| ⚠️ **Story Studio 界面难用 / tag 偶尔被读出来**（社区反馈） | 长内容场景需谨慎 |

##### IndexTTS-2

| 问题 | 说明 |
|---|---|
| 🔴 **许可证是 bilibili 协议，不是 Apache-2.0** | 中文博客普遍误传。**必须读仓库 LICENSE 原文** |
| 🔴 **§3.4(c) 禁止用于改进其他 AI 模型** | 「用 IndexTTS-2 生成数据训练自有模型」是违约行为 |
| 🔴 **§2.2 规模阈值** | 月活 >1 亿 或 年收入 >10 亿人民币 → 需单独授权 |
| 🔴 **§4.2 禁止高风险场景** | 医疗诊断、自动驾驶、军事、关键基础设施、大规模生物识别、自动化决策 |
| ⚠️ **精确时长控制在 release 中「尚未启用」** | README News 原文："*This functionality is not yet enabled in this release.*"（2.5 版本提供 `duration_factor`，需实测） |
| ⚠️ **必须用 `uv`** | `uv sync --all-extras`；**不支持 pip/conda 直接安装** |
| ⚠️ **CUDA Toolkit 12.8+** | 版本要求高 |
| ⚠️ **Windows 上 DeepSpeed 难装** | README 提示 |
| ⚠️ **显存需求说法不一** | 一说 FP16 ~7.8GB（RTX 3060 8GB 可跑），一说 ≥16GB（推荐 24GB+） |
| ⚠️ **无 vLLM 支持（2.0）** | 仅 2.5 有 vLLM recipe |
| ⚠️ **无 Docker** | 部署需自行容器化 |
| ⚠️ **`use_random=True` 降低克隆保真度** | 官方 note |
| ⚠️ **长文本生成可能不稳定** | 社区反馈 |
| ⚠️ **仅中英双语**（2.0） | 2.5 才扩展到 5 语言 |
| ⚠️ **训练数据主要来自 Emilia** | Emilia 是 CC-BY-NC-4.0；**数据链路的合规影响需法务评估** |
| ⚠️ **中文博客的「Apache-2.0」信息已过期** | 引用二手资料会踩坑 |

##### F5-TTS

| 问题 | 说明 |
|---|---|
| 🔴 **权重 CC-BY-NC，不可商用** | 根因是训练数据 **Emilia**（101.7k h，CC-BY-NC-4.0）。README 原文明确 |
| ⚠️ **纯 NAR，无法真流式** | 需等全部文本处理完；首包延迟随文本长度线性增长 |
| ⚠️ **无时长控制** | 时长由参考音频长度或长度回归决定 |
| ⚠️ **test-en SIM 偏低**（64.7~67.0） | 英文克隆质量是明显短板 |
| ⚠️ **无情感控制** | — |
| ⚠️ **ROCm 版本矩阵坑** | RDNA 3.5/4（gfx1151/gfx1201，如 RX 9060/9070、Radeon 8050S/8060S）**不在 ROCm 6.x 支持列表**，用 6.x 会报 `HIP error: invalid device function`；需 ROCm 7.x |
| ⚠️ **参考文本留空会用 ASR 自动转录** | 占用额外显存，且转录错误会传播 |
| ⚠️ **长音频需 chunk inference** | 长句/长文本需手动切块 |
| ⚠️ **未公布显存** | — |

##### VoxCPM2

| 问题 | 说明 |
|---|---|
| ⚠️ **长尾语言质量不达标** | 阿拉伯语 WER 13.05、印地语 19.70、捷克语 24.13、罗马尼亚语 21.58、乌克兰语 6.32（MiniMax-MLS 测试集）。**30 语言中真正可用的约 20 个** |
| ⚠️ **test-hard CER 8.13 偏弱** | 明显弱于 CosyVoice3-RL 的 5.44 与 FishAudio S2 的 5.99 |
| ⚠️ **Controllable Generation 稳定性** | 官方自述：Voice Design 与 Controllable Cloning 结果在多次运行间有波动，**建议生成 1~3 次取满意结果** |
| ⚠️ **Python ≥3.10 (<3.13)** | 版本窗口 |
| ⚠️ **PyTorch ≥2.5.0，CUDA ≥12.0** | — |
| ⚠️ **vLLM-Omni 需从源码装** | `uv pip install vllm==0.19.0` + `git clone vllm-omni && uv pip install -e .`；生态快速演进，需跟版本 |
| ⚠️ **VoxCPM-0.5B 仅 16kHz** | 老版本采样率低 |
| ⚠️ **VoxCPM1.5 / 0.5B 只有 Continuation 克隆模式** | 只有 VoxCPM2 支持 Isolated Reference |
| ⚠️ **无 Triton/TRT-LLM** | — |

##### VibeVoice

| 问题 | 说明 |
|---|---|
| 🔴 **TTS 代码已下架** | README News（2025-09-05）：「we have removed the VibeVoice-TTS code from this repository」——因被用于与声明意图不符的用途。**只剩权重，Demo 已 Disabled** |
| 🔴 **官方不建议商用** | README：「We do not recommend using VibeVoice in commercial or real-world applications without further testing and development. This model is intended for research and development purposes only.」 |
| ⚠️ **语言限制** | 仅英（数据为主）、中（数据显著更少）；其他语言会出「unexpected audio outputs」 |
| ⚠️ **中文发音错误** | 尤其中文特殊字符；中文稳定性低于英文 |
| ⚠️ **不支持非语音音频** | 不生成背景噪声、音乐、音效 |
| ⚠️ **不支持重叠语音** | 对话中的重叠段不建模 |
| ⚠️ **继承 Qwen2.5-1.5B 的偏见** | 官方声明 |
| ⚠️ **HF 上标 2.7B params，但仓库称 1.5B** | 数值口径需确认 |

##### GLM-TTS

| 问题 | 说明 |
|---|---|
| ⚠️ **英文数据 < 中文的一半** | 官方说明；test-en SIM 67.2 偏低 |
| ⚠️ **RL 权重与 2D-Vocos 尚未完全发布** | 第三方（instavar，2026-02-14 核查）：「README 仍把 RL 优化权重与 2D Vocos 更新标为 'coming soon'」——**需区分论文声明与当前可运行能力** |
| ⚠️ **多数能力仅内部评测** | Phoneme-in 的 PER 13.23→5.14、Vocos2D 质量提升均来自内部消融，**未经独立验证** |
| ⚠️ **部署生态弱** | 只有 Gradio + CLI，无 Docker / vLLM / Triton |
| ⚠️ **无 test-hard 数据** | 只报了 test-zh / test-en |

##### Spark-TTS

| 问题 | 说明 |
|---|---|
| 🔴 **权重 CC-BY-NC-SA-4.0，不可商用** | **已从 Apache-2.0 静默改为 CC-BY-NC-SA**。HF 模型卡 `license: cc-by-nc-sa-4.0` |
| 🔴 **克隆质量全场最低** | test-zh SIM **66.0**、test-en SIM **57.3** |
| ⚠️ **16 kHz 采样率** | 所有对比模型中最低 |
| ⚠️ **训练代码未发布** | 官方 FAQ 称「即将发布完整训练代码」 |
| ⚠️ **无流式** | — |
| ⚠️ **延迟偏高** | 876 ms @ L20 concurrency 1 |

##### HiggsAudio-v2

| 问题 | 说明 |
|---|---|
| 🔴 **test-hard CER 55.07（灾难级）** | 绕口令/罕见词上完全不可用（VoxCPM2 报告） |
| ⚠️ **许可不清** | 仓库 badge 是 Apache-2.0，但 HF 上 v2 checkpoint 标 **"other"** 并有单独 LICENSE 文件；**v3 系列改用 Boson research & non-commercial license**。必须逐文件确认 |
| ⚠️ **参数量说法不一** | 有说 3B，有说「未公开」 |
| ⚠️ **部署生态弱** | — |
| ⚠️ **第三方「情感最强」的评价未经严格验证** | 与 test-hard 55.07 的硬伤并存 |

##### GPT-SoVITS

| 问题 | 说明 |
|---|---|
| 🔴 **商用界定模糊，版权风险高** | 仓库 Apache-2.0，但训练/微调数据常含未授权音频；社区共识为「商用有风险」 |
| ⚠️ **无统一 benchmark** | 社区口碑好但无 Seed-TTS-eval 类公开数据 |
| ⚠️ **工程化程度低** | WebUI 为主，无生产级 serving |
| ⚠️ **版本碎片化** | 多个分支、多个版本并行 |

##### Kokoro-82M

| 问题 | 说明 |
|---|---|
| 🔴 **不支持零样本克隆** | 固定 54 个预设音色 |
| ⚠️ **中文自然度偏低** | 第三方 MOS 3.9（vs 英文 4.3） |
| ⚠️ **训练数据仅几百小时** | 质量天花板明显 |
| ⚠️ **无情感控制** | — |
| ⚠️ **需 `espeak-ng`** | 系统依赖 |

##### 通用坑（所有模型）

1. **README 的 badge 说代码许可，不说权重许可** — 必读 LICENSE 原文
2. **「支持 N 语言」是营销数字** — 必须逐语言实测
3. **SS/SIM 口径不统一**（WavLM / ERes2Net / CAM++）— 跨模型比较会得出错误结论
4. **flow matching 部分对量化敏感** — 只量化 LLM
5. **参考音频质量对克隆效果影响极大** — 需 VAD 裁剪 + 降噪 + 音量归一化
6. **参考文本准确性** — in-context 类模型的转录错误会直接损害质量
7. **许可证会被静默更新** — Fish Audio（2026-03）、Spark-TTS（从 Apache-2.0 改 CC-BY-NC-SA）都发生过；需定期复查
8. **声音权/人格权不由开源许可覆盖** — 必须取得说话人书面授权
9. **没有模型自带生产级鉴权/限流/监控** — 均需自行补齐


### 16. 适用场景与选型建议

#### 开源 TTS 选型决策指南

##### 一、完整横向对比表（**核心产出**）

###### 1.1 主表：参数量 / 许可 / 资源 / 能力

| 模型 | 参数量 | 许可证（代码/权重） | **商用** | 显存 | 最短参考音频 | 情感控制 | 时长控制 | 语言数 | 首包延迟 | 部署生态 |
|---|---|---|---|---|---|---|---|---|---|---|
| **CosyVoice 3-0.5B** | 0.5B | Apache-2.0 / Apache-2.0 | ✅ **完全** | 待核实（估 3~5GB） | 待核实（3~10s） | 中（>100 指令 + DiffRO-EMO） | ❌ | **9 + 18/19 方言** | **150 ms** | ⭐⭐⭐⭐ |
| **CosyVoice3-0.5B_RL** | 0.5B | Apache-2.0 | ✅ **完全** | 同上 | 同上 | 同上 | ❌ | 同上 | 150 ms | ⭐⭐⭐⭐ |
| **Fish Audio S2 Pro** | **4B**+400M | 研究许可 / 研究许可 | ❌ **需授权** | 待核实 | **10~30 s** | ✅✅ **最强**（15k tag，sub-word 级） | ❌ | **80+** | **~100 ms**（H200） | ⭐⭐⭐⭐⭐ |
| Fish-Speech 1.5 | ~0.5B | 🔴 **已弃用**（2026-02-28） | ❌ | ~6GB（第三方） | 5 s（第三方） | 中（tag） | ❌ | 多语 | ~150 ms | ⭐⭐ |
| **IndexTTS-2** | **1.5B** | **bilibili 协议** | ⚠️ **有条件** | **FP16 ~7.8 GB** | **5~15 s** | ✅✅ **音色-情感解耦**（GRL，4 种输入） | ✅✅ **唯一**（token 级，误差<0.02%） | **2**（中英） | 待核实 | ⭐⭐ |
| **IndexTTS-2.5** | 0.8B | bilibili 协议 | ⚠️ 有条件 | 待核实 | 5~15 s | ✅ 同上 + `duration_factor` | ✅（0.5x~2.0x） | **5** | 待核实 | ⭐⭐⭐ |
| **F5-TTS Base** | ~0.3B | MIT / **CC-BY-NC** | ❌ **不可** | 待核实 | **3 s** | ❌ 无 | ❌ | 2（+社区扩展） | 253 ms（整句） | ⭐⭐⭐ |
| **VoxCPM2** | **2B** | Apache-2.0 / Apache-2.0 | ✅ **完全** | **~8 GB** | 待核实 | ✅ **Voice Design** + style guidance | ❌ | **30**（+9 方言） | 待核实 | ⭐⭐⭐⭐⭐ |
| VoxCPM1.5 | 0.8B | Apache-2.0 | ✅ | ~6 GB | 待核实 | 中 | ❌ | 2 | 待核实 | ⭐⭐⭐⭐⭐ |
| VoxCPM-0.5B | 0.6B | Apache-2.0 | ✅ | ~5 GB | 待核实 | 中 | ❌ | 2 | 待核实 | ⭐⭐⭐⭐⭐ |
| **VibeVoice-1.5B** | 1.5B | MIT / MIT | ⚠️ 官方不建议 | 待核实（≥8GB？） | 待核实 | ⚠️ 部分 | ❌ | 2 | — | ⭐（**代码已下架**） |
| VibeVoice-Realtime | 0.5B | MIT | ⚠️ | 待核实 | 预设音色 | ❌ | ❌ | 9（实验） | **~300 ms** | ⭐⭐ |
| **GLM-TTS** | **1.5B** | Apache-2.0 / **MIT** | ✅ | 待核实 | **~3 s** | ✅ GRPO 四奖励（含笑声） | ❌ | 2（中英）+方言 | 400 ms（第三方） | ⭐⭐ |
| GLM-TTS_RL | 1.5B | Apache-2.0 / MIT | ✅ | 待核实 | ~3 s | ✅ 同上 | ❌ | 同上 | 待核实 | ⭐⭐ |
| **HiggsAudio-v2** | 3B | Apache-2.0 / **"other"** | ⚠️ **需确认** | 待核实（≥8GB？） | 5 s | ✅ 情感识别 + 多角色 | ❌ | 多语 | 待核实 | ⭐⭐ |
| **Spark-TTS** | 0.5B | Apache-2.0 / **CC-BY-NC-SA** | ❌ **不可** | **1~2 GB** | 短样本 | ⚠️ 性别/音调/语速 | ❌ | 2 | 876 ms（L20） | ⭐⭐ |
| **Kokoro-82M** | **82M** | Apache-2.0 / Apache-2.0 | ✅ **完全** | ~400 MB（内存） | ❌ **不支持克隆** | ❌ 无 | ❌ | **8**（54 音色） | ~10 ms（第三方） | ⭐⭐⭐ |
| **GPT-SoVITS** | ~1B 级 | Apache-2.0 / **界定模糊** | ⚠️ **高风险** | 待核实 | **5 s**（1 min 微调） | ⚠️ 有限 | ❌ | 多语 | 待核实 | ⭐⭐ |

###### 1.2 质量指标表（Seed-TTS-eval）

| 模型 | test-zh CER↓ | test-zh SIM↑ | test-en WER↓ | test-en SIM↑ | test-hard CER↓ | test-hard SIM↑ | RTF |
|---|---|---|---|---|---|---|---|
| Human | 1.26 | 75.5 | 2.14 | 73.4 | — | — | — |
| MiniMax-Speech（闭源） | **0.83** | 78.3 | **1.65** | 69.2 | — | — | — |
| Seed-TTS（闭源） | 1.12 | **79.6** | 2.25 | **76.2** | 7.59 | 77.6 | — |
| **FishAudio S2** | **0.54** 🥇 | — | **0.99** 🥇 | — | **5.99** | — | **0.195**（H200） |
| **LongCat-Audio-DiT** | 1.09 | **81.8** 🥇 | 1.50 | 78.6 | 6.04 | **79.7** 🥇 | — |
| **VoxCPM2** | 0.97 | 79.5 | 1.84 | 75.3 | 8.13 | 75.3 | **~0.13**（4090+Nano-vLLM） |
| **Qwen3-TTS** | 1.22 | 77.0 | 1.23 | 71.7 | 6.76 | 74.8 | — |
| **GLM-TTS_RL** | **0.89** | 76.4 | 1.91 | 68.1 | — | — | — |
| **GLM-TTS** | 1.03 | 76.1 | 2.23 | 67.2 | — | — | — |
| **IndexTTS2** | 1.03 | 76.5 | 2.23 | 70.6 | 7.12 | 75.5 | — |
| **CosyVoice3-0.5B_RL** | **0.81** | 77.4 | **1.68** | 69.5 | **5.44** 🥇 | 75.0 | 待核实 |
| **CosyVoice3-0.5B** | 1.21 | **78.0** | 2.24 | 71.8 | **6.71** | 75.8 | 待核实 |
| **VoxCPM-0.5B** | 0.93 | 77.2 | 1.85 | **72.9** | 8.87 | 73.0 | ~0.10（4090+Nano-vLLM） |
| VibeVoice-1.5B | 1.16 | 74.4 | 3.04 | 68.9 | — | — | — |
| **F5-TTS** | 1.52 | 74.1 | 2.00 | 64.7~67.0 | 8.67 | 71.3 | **0.0394**（L20+TRT-LLM）🥇 |
| HiggsAudio-v2 | 1.50 | 74.0 | 2.44 | 67.7 | **55.07** 🔴 | 65.6 | — |
| SparkTTS | 1.2~1.54 | **66.0** 🔴 | 1.98~3.14 | **57.3** 🔴 | — | — | 0.0704（L20 conc4） |
| CosyVoice2 | 1.45 | 75.7 | 2.57 | 65.9 | 6.83 | 72.4 | — |

---

##### 二、场景化推荐

###### 决策树

```
开始
 ├─ 需要商用？
 │   ├─ 否（研究/原型/个人）→ 任选，优先 Fish S2 Pro（质量最高）
 │   └─ 是
 │       ├─ 月活 >1亿 或 年收入 >10亿人民币？
 │       │   └─ 是 → ❌ 排除 IndexTTS-2
 │       ├─ 需要用生成数据训练自有模型？
 │       │   └─ 是 → ❌ 排除 IndexTTS-2、Fish Audio
 │       ├─ 高风险场景（医疗/自动驾驶/军事/关键基础设施/大规模生物识别/自动化决策）？
 │       │   └─ 是 → ❌ 排除 IndexTTS-2
 │       └─ 继续 ↓
 │
 ├─ 核心需求是什么？
 │   ├─ **中文 + 中国方言 + 高发音准确率** → 🥇 **CosyVoice 3-0.5B_RL**
 │   ├─ **精确时长控制（视频配音/口型同步）** → 🥇 **IndexTTS-2**（合规过关前提下）
 │   ├─ **音色-情感解耦** → 🥇 **IndexTTS-2**
 │   ├─ **副语言/细粒度情感（笑声/呼吸/耳语）** → 🥇 Fish S2 Pro（商用）／ 🥈 **GLM-TTS**（开源）
 │   ├─ **48kHz 录音室音质** → 🥇 **VoxCPM2**
 │   ├─ **30 语言** → 🥇 **VoxCPM2**
 │   ├─ **80+ 语言** → 🥇 Fish Audio S2 Pro（商业授权 + 长尾实测）
 │   ├─ **长音频 / 多说话人（播客/有声剧）** → 🥇 VibeVoice（代码已下架，需自研推理）
 │   ├─ **离线批量 / 最高吞吐** → 🥇 **F5-TTS**（但权重不可商用）→ 开源替代：**VoxCPM2**
 │   ├─ **端侧 / 边缘 / 移动端** → 🥇 **VoxCPM2**（llama.cpp/ONNX/ANE）／ 🥈 **Kokoro-82M**
 │   ├─ **CPU 实时** → 🥇 **Kokoro-82M**
 │   ├─ **少样本微调造专属音色** → 🥇 **GPT-SoVITS**（有版权风险）／ 🥈 **VoxCPM2**（SFT/LoRA，5~10 min）
 │   ├─ **实时对话 / 语音助手（低首包延迟）** → 🥇 Fish S2 Pro（100ms）／ 🥈 **CosyVoice 3**（150ms）
 │   └─ **快速原型验证** → 🥇 **Kokoro-82M**（pip install）／ 🥈 **F5-TTS**
 │
 └─ 技术栈约束？
     ├─ 已用 OpenAI 音频 API → 🥇 **VoxCPM2**（vLLM-Omni 原生兼容）
     ├─ 已有 Triton 基础设施 → 🥇 **CosyVoice 3** / **F5-TTS**
     ├─ 已用 FunASR（阿里系）→ 🥇 **CosyVoice 3**（同生态，依赖复用）
     └─ 已用 CUDA 12.8+ / uv → **IndexTTS-2** 可行
```

###### 分场景推荐表

| 场景 | 🥇 首选 | 🥈 次选 | 关键理由 |
|---|---|---|---|
| **中文播报 / 有声书 / 教育** | **CosyVoice 3-0.5B_RL** | GLM-TTS_RL | test-hard CER **5.44** 全场第一；Apache-2.0；拼音/CMU 发音修补 100% 修正率 |
| **中文 + 方言（粤语/四川话等）** | **CosyVoice 3** | VoxCPM2（9 方言） | 18/19 种方言覆盖最广 |
| **视频配音 / 口型同步** | **IndexTTS-2** | — | 唯一 token 级精确时长（误差 <0.02%，10s 偏差 ≤7ms） |
| **虚拟主播 / 情感对话** | Fish S2 Pro | **GLM-TTS** | 15k tag sub-word 级 vs GRPO 四奖励（含笑声） |
| **数字人 / 游戏 NPC 批量造声** | **VoxCPM2**（Voice Design） | Fish S2 Pro | 纯描述造全新音色，无需参考音频 |
| **多语言客服 / 出海产品** | **VoxCPM2**（30 语） | Fish S2 Pro（80+，需授权） | Apache-2.0 + 30 语言；但需逐语言实测 |
| **播客 / 有声剧（长音频多角色）** | VibeVoice | — | 90 分钟 / 4 说话人（**代码已下架**） |
| **离线批量合成（十万句级）** | **F5-TTS**（若可接受非商用） | **VoxCPM2**（商用） | F5-TTS RTF 0.0394 @ TRT-LLM 快一个数量级 |
| **实时语音助手** | Fish S2 Pro（100ms） | **CosyVoice 3**（150ms） | 首包延迟是关键 |
| **端侧 / 移动端 / IoT** | **VoxCPM2** | Kokoro-82M | llama.cpp-omni / ONNX / ANE / Rust 全覆盖 |
| **CPU 服务器 / 无 GPU** | **Kokoro-82M** | — | 4 核 CPU 上 2~3× 实时 |
| **少样本定制音色（1 分钟音频）** | GPT-SoVITS（有风险） | **VoxCPM2**（SFT/LoRA） | — |
| **强合规 / 数据不出域** | **CosyVoice 3** / **VoxCPM2** | GLM-TTS | 仅这三个既有高性能又有干净商用授权 |
| **预算敏感 / 高并发** | **VoxCPM2** | Kokoro-82M | Apache-2.0 零授权成本 + RTF 0.13 |
| **快速原型** | **Kokoro-82M** | **F5-TTS** | `pip install kokoro` / `pip install f5-tts` |

###### ❌ 明确不推荐

| 组合 | 理由 |
|---|---|
| **Spark-TTS 用于任何商用或克隆场景** | 权重 CC-BY-NC-SA 不可商用；克隆 SIM 66.0/57.3 全场最低；16kHz |
| **F5-TTS 用于商用** | 权重 CC-BY-NC（Emilia 传导） |
| **HiggsAudio-v2 用于绕口令/专业术语/生僻字** | test-hard CER **55.07** |
| **VibeVoice-TTS 用于生产** | 代码已下架 + 官方明确不建议商用 |
| **Fish-Speech 1.5 用于任何新项目** | 已官方弃用（2026-02-28）+ 许可证已变更 |
| **Kokoro-82M 用于声音克隆** | 根本不支持克隆 |
| **GPT-SoVITS 用于商用产品** | 版权风险高，商用界定模糊 |
| **IndexTTS-2 用于高风险场景 / 训练自有模型** | bilibili 协议 §3.4(c) 与 §4.2 明确禁止 |

---

##### 三、给工程师的落地清单

###### 选型阶段
- [ ] **逐字读 LICENSE 原文**（不是 badge、不是二手文章），重点确认**权重**许可
- [ ] 确认公司规模是否触及 IndexTTS-2 的阈值（月活 1 亿 / 年收入 10 亿人民币）
- [ ] 确认是否需要用生成数据训练自有模型（→ 排除 IndexTTS-2、Fish Audio）
- [ ] 确认是否属高风险场景（→ 排除 IndexTTS-2）
- [ ] **逐语言实测 WER/CER**，不要信「支持 N 语言」
- [ ] **统一 SS/SIM 口径**（WavLM 或 ERes2Net，二选一全程不变）
- [ ] 用统一 ASR（FunASR 或 whisper）做 CER/WER 回环

###### POC 阶段
- [ ] **同时测 base 与 RL 权重**（CER vs SS 的权衡是真实存在的）
- [ ] 自建 RTF / 显存 / 并发 benchmark（**多数模型官方数据缺失**）
- [ ] 测 hard 集（绕口令、生僻字、中英混排、数字、长句）
- [ ] 测参考音频质量的敏感度（噪声、混响、时长）
- [ ] 量化策略：**LLM 量化 + 声学模型保持 fp16**

###### 生产阶段
- [ ] 补齐鉴权 / 限流 / 监控（**所有开源模型都没有**）
- [ ] 音频后处理：响度归一化（EBU R128）、格式转换、静音裁剪
- [ ] 说话人书面授权流程 + AI 生成标识
- [ ] 模型版本锁定 + 许可变更监控（Fish Audio 与 Spark-TTS 都静默改过协议）
- [ ] 预留升级窗口（TTS 领域约每 6 个月一代）

###### 最后一句话

> **2026 年做开源 TTS 选型，第一决策点是「许可证」而不是「指标」。**
> 指标上第一梯队的差距已在噪声级别（zh CER 0.54~1.22），
> 但**授权（Apache-2.0 / bilibili / 研究许可 / CC-BY-NC）与部署生态（OpenAI 兼容 / 端侧覆盖）的差距是数量级的**。
>
> **默认推荐：中文+方言+合规 → CosyVoice 3；48kHz+30 语言+端云一体 → VoxCPM2；需要时长/情感解耦且合规过关 → IndexTTS-2；只需 CPU 轻量 → Kokoro-82M。**


### 17. 参考来源

#### 权威来源

##### A. 一手来源：官方仓库 / 模型卡 / 许可证原文

**CosyVoice 3（参照基准）**
1. 论文：*CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training*，arXiv:2505.17589 — https://arxiv.org/abs/2505.17589 ｜ HTML: https://arxiv.org/html/2505.17589v2
2. GitHub：https://github.com/FunAudioLLM/CosyVoice
3. HF 模型卡：https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
4. ModelScope：https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
5. CV3-Eval 评测集：https://github.com/FunAudioLLM/CV3-Eval

**Fish Audio / Fish-Speech**
6. GitHub（含 **LICENSE 原文**）：https://github.com/fishaudio/fish-speech ｜ LICENSE: https://raw.githubusercontent.com/fishaudio/fish-speech/main/LICENSE
7. **Fish Audio S2 技术报告**：*Fish Audio S2 Technical Report*，arXiv:2603.08823 — https://arxiv.org/abs/2603.08823
8. Fish-Speech v1.4 论文：arXiv:2411.01156 — https://arxiv.org/abs/2411.01156
9. Fish Audio 博客（S2 开源公告）：https://fish.audio/blog/fish-audio-open-sources-s2/
10. 官方文档：https://speech.fish.audio/ （install / inference / server / docker）
11. HF 模型：https://huggingface.co/fishaudio/s2-pro
12. SGLang-Omni 集成：https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md
13. vLLM-Omni 集成：https://github.com/vllm-project/vllm-omni/blob/main/recipes/fishaudio/Fish-Speech-S2-Pro.md

**IndexTTS**
14. 论文：*IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech*，arXiv:2506.21619 — https://arxiv.org/abs/2506.21619
15. GitHub：https://github.com/index-tts/index-tts
16. **LICENSE 原文（bilibili 模型使用许可协议）**：https://raw.githubusercontent.com/index-tts/index-tts/main/LICENSE
17. HF 模型：https://huggingface.co/IndexTeam/IndexTTS-2 ｜ ModelScope: https://modelscope.cn/models/IndexTeam/IndexTTS-2
18. Demo：https://index-tts.github.io/index-tts2.github.io/
19. IndexTTS-2.5 vLLM recipe：https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5

**F5-TTS**
20. 论文：*F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching*，arXiv:2410.06885 — https://arxiv.org/abs/2410.06885
21. GitHub：https://github.com/SWivid/F5-TTS
22. E2-TTS 论文：arXiv:2406.18009 — https://arxiv.org/abs/2406.18009
23. Triton + TRT-LLM：https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/runtime/triton_trtllm/README.md
24. MLX 实现：https://github.com/lucasnewman/f5-tts-mlx ｜ ONNX: https://github.com/DakeQQ/F5-TTS-ONNX

**VoxCPM**
25. GitHub：https://github.com/OpenBMB/VoxCPM
26. VoxCPM2 技术报告：arXiv:2606.06928 — https://arxiv.org/abs/2606.06928
27. VoxCPM 论文（ICLR 2026）：arXiv:2509.24650 — https://arxiv.org/abs/2509.24650
28. 官方文档：https://voxcpm.readthedocs.io/en/latest/
29. HF：https://huggingface.co/openbmb/VoxCPM2 ｜ ModelScope: https://modelscope.cn/models/OpenBMB/VoxCPM2
30. Nano-vLLM-VoxCPM：https://github.com/a710128/nanovllm-voxcpm
31. vLLM-Omni：https://github.com/vllm-project/vllm-omni
32. llama.cpp-omni：https://github.com/tc-mb/llama.cpp-omni

**VibeVoice（Microsoft）**
33. GitHub：https://github.com/microsoft/VibeVoice
34. HF Collection：https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f
35. TTS 报告：https://openreview.net/forum?id=FihSkzyxdv （ICLR 2026 Oral）｜ PDF: https://openreview.net/pdf?id=FihSkzyxdv
36. ASR 报告：arXiv:2601.18184 — https://arxiv.org/abs/2601.18184
37. VibeASR.cpp（CPU 边缘）：https://github.com/microsoft/VibeASR.cpp
38. VibeVoice-1.5B：https://huggingface.co/microsoft/VibeVoice-1.5B ｜ Realtime: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B

**GLM-TTS（智谱 AI）**
39. 论文：arXiv:2512.14291 — https://arxiv.org/abs/2512.14291 ｜ HTML: https://arxiv.org/html/2512.14291v1
40. GitHub：`zai-org/GLM-TTS`（HF `zai-org/GLM-TTS`、ModelScope 均有镜像）
41. 社区整理站：https://glmtts.net/ ｜ http://glm-tts.com/

**Spark-TTS**
42. 论文：*Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens*，arXiv:2503.01710 — https://arxiv.org/abs/2503.01710
43. GitHub：https://github.com/SparkAudio/Spark-TTS
44. HF 模型卡（**含 `license: cc-by-nc-sa-4.0`**）：https://huggingface.co/SparkAudio/Spark-TTS-0.5B

**HiggsAudio**
45. GitHub：https://github.com/boson-ai/higgs-audio
46. HF：`bosonai/higgs-audio-v2-generation-3B-base`（许可标为 **"other"**，checkpoint 内有独立 LICENSE 文件）

**Kokoro-82M**
47. GitHub：https://github.com/hexgrad/kokoro
48. HF 模型卡（含训练成本 $1000 / 1000 A100 小时）：https://huggingface.co/hexgrad/Kokoro-82M
49. misaki G2P：https://github.com/hexgrad/misaki
50. StyleTTS 2（架构来源）：https://github.com/yl4579/StyleTTS2 ｜ 论文 arXiv:2306.07691
51. ISTFTNet：arXiv:2203.02395

**GPT-SoVITS**
52. GitHub：https://github.com/RVC-Boss/GPT-SoVITS

##### B. 评测与交叉验证

53. **VoxCPM2 技术报告**（arXiv:2606.06928）中的 Seed-TTS-eval 与 CV3-eval 表 — **本文多模型指标的主要交叉验证来源**
54. **GLM-TTS 论文**（arXiv:2512.14291）Table 2/3 的 19 个基线对比
55. **CosyVoice 3 官方 README** 的 Evaluation 表 — https://github.com/FunAudioLLM/CosyVoice
56. SiliconFlow 模型库（Fish-Speech-1.5 / IndexTTS-2 的定价与规格）：https://www.siliconflow.com/
57. Together AI / DeepInfra（Kokoro-82M 定价与规格）：https://www.together.ai/models/kokoro-82m ｜ https://deepinfra.com/hexgrad/Kokoro-82M

##### C. 许可证与合规分析（二手，但分析质量高）

58. **Open-Weight Text to Speech Models in 2026: The XTTS Successors**（builderai.tools）— https://builderai.tools/blog/open-weight-text-to-speech-models-2026
    - **最重要的二手来源**：系统梳理了「代码许可 vs 权重许可」的错配问题，指出 **Emilia（101.7k h, CC-BY-NC-4.0）是大多 NC 条款的根源**，并说明 Emilia-YODAS（CC-BY-4.0）是可以规避的路径
59. Instavar 对 GLM-TTS 技术报告的独立核查（2026-02-14 状态核查）— https://instavar.com/blog/GLM_TTS_Technical_Report_Production_Zero_Shot_TTS
60. Hivebook 关于 Fish Speech / OpenAudio 许可的分析 — https://hivebook.wiki/wiki/fish-speech-openaudio-multilingual-tts-and-voice-cloning

##### D. 引用 bibtex

```bibtex
@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and others},
  journal={arXiv preprint arXiv:2505.17589}, year={2025}
}

@article{liao2026fishaudios2technical,
  title={Fish Audio S2 Technical Report},
  author={Liao, Shijia and Wang, Yuxuan and Liu, Songting and others},
  journal={arXiv preprint arXiv:2603.08823}, year={2026}
}

@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Zhou, Siyi and Zhou, Yiquan and He, Yi and others},
  journal={arXiv preprint arXiv:2506.21619}, year={2025}
}

@article{chen-etal-2024-f5tts,
  title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
  author={Chen, Yushen and Niu, Zhikang and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2410.06885}, year={2024}
}

@article{zhou2026voxcpm2,
  title={VoxCPM2 Technical Report},
  author={Zhou, Yixuan and Zeng, Guoyang and Liu, Xin and others},
  journal={arXiv preprint arXiv:2606.06928}, year={2026}
}

@article{wang2025sparktts,
  title={Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens},
  journal={arXiv preprint arXiv:2503.01710}, year={2025}
}
```

##### E. 数据/评测工具

61. **Emilia** 数据集（CC-BY-NC-4.0，F5-TTS / MaskGCT / 诸多模型的训练语料，也是 NC 条款的根源）— arXiv:2407.05361
62. Emilia-YODAS（CC-BY-4.0 子集，规避 NC 的路径）
63. WenetSpeech4TTS — arXiv:2406.05763
64. MiniMax-MLS-test 多语言测试集（VoxCPM2 报告中使用）
65. InstructTTSEval（Voice Design 评测，VoxCPM2 报告中使用）
66. SpeechMOS (UTMOS) — https://github.com/tarepan/SpeechMOS
67. ctc-forced-aligner — https://github.com/MahmoudAshraf97/ctc-forced-aligner
68. MFA (Montreal Forced Aligner) — https://montreal-forced-aligner.readthedocs.io/
69. emo2vec（情感分类，CosyVoice 3 论文使用）
70. FunASR（中文 ASR 回环校验）— https://github.com/modelscope/FunASR


---


## 第五部分 · 工程落地细节

> 本部分为技术栈维度的字段级明细：推理引擎、ASR 生态、生产加速与部署路径。

---


---

<!-- item: inference_engines -->

## 推理引擎：vLLM / vLLM-Omni / SGLang

> **分类**：技术栈-推理引擎　|　**已填字段**：33 / 38　|　**⭐ 标记为本轮重点的架构与训练字段**


### 1. 基本信息与版本演进

本条目调研的是**让 Qwen3-ASR / Qwen3-TTS / CosyVoice3 跑起来并上生产的推理与服务引擎层**，不是模型本身。

- **vLLM**：vLLM 主线（github.com/vllm-project/vllm）。Qwen3-ASR 为 day-0 支持（`vllm serve Qwen/Qwen3-ASR-1.7B` 直接可用）。当前主线版本节奏很快：FunASR 官方部署中心 2026-08-13 验证的组合是 **vLLM 0.27.1+cu129 / Torch 2.13.0+cu129**（来源：funasr.com/en/deploy/vllm.html）。
- **vLLM-Omni**：vLLM 官方子项目（github.com/vllm-project/vllm-omni），2025-11 正式发布，把 vLLM 从「自回归文本引擎」扩展到全模态（含 DiT / 扩散、TTS、多阶段流水线）。**与主线强绑定**：自 0.14.0 起每个偶数号上游 vLLM minor 版本对应一个 vLLM-Omni 稳定版（0.16.0 / 0.18.0 / 0.20.0 / 0.22.0 / 0.24.0 / 0.26.0），仓库主干 2026-08-27 刚完成 "Rebase to vllm 0.28.0"。论文：arXiv:2602.02204。
- **SGLang-Omni**：sgl-project/sglang-omni，多阶段 omni 服务运行时，Apache-2.0。**v0.1.2 已在 PyPI**（2026/08，`uv pip install "sglang-omni==0.1.2"`）。支持 Qwen3-TTS-12Hz-Base（0.6B/1.7B，含 voice cloning + streaming）与 Qwen3-ASR-1.7B。
- **其余相关引擎**：NVIDIA TensorRT Edge-LLM（v0.6.0 起 day-0 支持 Qwen3-ASR 与 Qwen3-TTS，FP16）、NVIDIA Dynamo（以 vLLM-Omni 为 worker 暴露 `/v1/audio/speech`）、NVIDIA Triton + TensorRT-LLM（CosyVoice 官方 runtime 目录）。

**重要更正（相对常见二手资料）**：网上大量文章仍写「vLLM-Omni 对 Qwen3-TTS 仅支持离线推理，在线 serving 后续支持」。这条在 2026-08 已**过时**——vLLM-Omni 主线 `examples/online_serving/text_to_speech/` 已提供 Qwen3-TTS 的 `vllm serve ... --omni` 在线服务，支持 `/v1/audio/speech`、流式 PCM（chunked HTTP）、WebSocket 增量流式（`/v1/audio/speech/stream`）、`/v1/audio/voices` 音色上传与预计算、FastRTC/Gradio 演示。选型时请以仓库文档为准。


### 2. 开源协议与商用边界

- **vLLM**：Apache-2.0（github.com/vllm-project/vllm）。
- **vLLM-Omni**：Apache-2.0（仓库 LICENSE 文件，README 明确声明）。论文 arXiv:2602.02204。
- **SGLang-Omni**：Apache-2.0（README + 仓库 LICENSE 使用 verbatim Apache-2.0 文本，2026-05-31 提交确认）。
- **TensorRT-LLM / TensorRT Edge-LLM**：NVIDIA 自有许可（非 Apache），商用需遵循 NVIDIA 软件许可协议。
- **NVIDIA Triton Inference Server**：BSD-3-Clause 风格开源许可；生产使用时注意 GPU 驱动 / CUDA / TensorRT 的版本绑定。
- 被服务的三款模型（Qwen3-ASR / Qwen3-TTS / CosyVoice3）均为 **Apache-2.0**，可商用、可修改、可私有化，引擎层协议不构成额外限制。


### 3. 模型变体与权重获取

本条目是引擎层，无自有权重；下表是**引擎 × 模型**的支持矩阵（截至 2026-08-28 核实）：

| 模型 | vLLM 主线 | vLLM-Omni | SGLang-Omni | TensorRT Edge-LLM | Triton+TRT-LLM |
|---|---|---|---|---|---|
| Qwen3-ASR-0.6B / 1.7B | ✅ day-0，`vllm serve` 直接起 | ✅（走 `/v1/audio/transcriptions`） | ✅（30 language hints） | ✅ FP16（0.6B / 1.7B） | 待核实 |
| Qwen3-TTS-12Hz-{0.6B,1.7B}-{CustomVoice,VoiceDesign,Base} | ❌ | ✅ 在线 + 离线，PCM/WebSocket 流式，voice cloning | ✅ Base（voice cloning + streaming，0.6B/1.7B） | ✅ CustomVoice（0.6B/1.7B，FP16） | 待核实 |
| CosyVoice3 (`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`) | ❌ | ✅ 2-stage（talker + flow-matching code2wav），24 kHz，仅 voice cloning（无内置音色预设） | ❌ 官方支持列表未含 | 待核实 | ✅ 官方 `runtime/triton_trtllm`（NVIDIA Yuekai Zhang 贡献） |

SGLang-Omni 同时覆盖 Higgs Audio v3、Fish S2-Pro、Voxtral TTS、MOSS-TTS、Ming-Omni-TTS、dots.tts、ZONOS2、MiniMax Music 3；ASR 侧另有 Fun-ASR-Nano-2512-hf、ARK-ASR-3B、MOSS-Transcribe-Diarize、whisper-large-v3（实验）。


### 4. 架构范式 ⭐

#### 1. vLLM 主线：单引擎自回归 + PagedAttention
vLLM 原生的执行范式是**单阶段自回归解码**：PagedAttention 管理 KV cache、continuous batching 聚合请求、CUDA Graph 降低单步开销。ASR 模型（Qwen3-ASR）在这种范式下天然适配——音频 encoder 把波形切成 audio token（Qwen3-ASR 为 **25 tok/s**，社区整理数据；对比 Whisper 50 tok/s 但**固定 30 s 上下文窗口**、Voxtral 12.5 tok/s），LLM 部分自回归出文本，整个过程只有一个 decode 循环，continuous batching 直接生效。
**结论：Qwen3-ASR 用 vLLM 主线即可，不需要 vLLM-Omni。**

#### 2. vLLM-Omni：异构多阶段流水线（multi-stage pipeline）
TTS / Omni 模型无法塞进单一 AR 循环：Qwen3-TTS 是 **Talker（每帧第 1 个码本 token）→ Code Predictor（剩余 15 个码本 token）→ Codec/Code2Wav（转波形）** 三段，计算形态完全不同（AR 计算密集 / AR 但结构规整 / 波形解码时延敏感）。vLLM-Omni 的做法：

- **异构流水线抽象**：把整图拆成 stage，每个 stage 有自己的 engine args（`gpu_memory_utilization`、`max_num_seqs`、`max_model_len`、`enforce_eager`、`tensor_parallel_size`…）和自己的 device 绑定。
- **流水线 stage 重叠执行**（pipelined stage execution overlapping）：stage 间用窗口（window）触发下游，而非等上游全完成。
- **OmniConnector 全解耦**：stage 之间可跨进程 / 跨 GPU / 跨节点，动态资源分配；PD disaggregation 为实验特性。
- **非自回归纳入同一框架**：DiT / 扩散（CosyVoice3 的 flow-matching code2wav）作为 generation stage 与 AR stage 共存。
- 论文见 arXiv:2602.02204《vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models》。

#### 3. SGLang-Omni：计算形态centric 的 stage 专用调度
SGLang-Omni 的关键差异是**每个 stage 跑一个针对自身瓶颈的独立 scheduler**（compute-bound thinker / memory-bound talker / latency-sensitive codec），stage 间通过 inbox/outbox 抽象 + **零拷贝共享内存**传 tensor，control plane 协调请求、relay data plane 走 shared-memory / NCCL / NIXL / Mooncake 后端。AR backbone 仍复用 SGLang 的 RadixAttention + continuous batching + CUDA Graph。它同时提供 `sglang_omni_router` 做多 worker 的 OpenAI 兼容前门。

#### 4. Nari Labs 的关键洞察（对三个引擎都成立）
Qwen3-TTS 真正的瓶颈不在模型而在**调度**：把 Talker / Code Predictor / Codec 暴露为**三个可独立调度的任务**放到同一个调度面上（受 M* 启发），并做 **deadline-aware 优先级**——未产出首包音频的请求高优先级；已在播放的流只在接近下一次播放 deadline 时才变紧急。这样既能让紧急请求插队，又能拿紧急请求当 anchor、用兼容任务填满 batch 保住批处理效率。注意一个反直觉点：**把 Talker 与 Code Predictor 合并反而更差**，因为合并后变成一个不可抢占的大工作单元，会阻塞更紧急的 Code Predictor / Codec 任务。


### 5. 模块拆解 ⭐

#### vLLM-Omni 运行时组件（源码路径可核）
- **API 层**：`vllm_omni/entrypoints/openai/serving_speech.py`（`OmniOpenAIServingSpeech`，处理 `/v1/audio/speech`、batch 端点与音色管理）、`serving_speech_stream.py`（`OmniStreamingSpeechHandler`，WebSocket 生命周期 + 句边界检测）、`protocol/audio.py`（请求协议：`task_type` = CustomVoice|VoiceDesign|Base、`ref_audio` 支持 URL/base64/file URI、`ref_text`、`instructions`、`language`、`max_new_tokens` 1–4096、`speed` 0.25–4.0、`response_format` wav|mp3|pcm|flac|aac|opus）、`audio_utils_mixin.py`（`AudioMixin`：立体声转单声道、重采样、`speed != 1.0` 时用 `torchaudio.transforms.TimeStretch` 相位声码器变速不变调）、`speech_usage.py`（`SpeechOutputTokenCounter`，OpenAI 兼容 usage）。
- **TTS adapter 注册表**：用 `TTSModelAdapter` 子类（如 `MingTTSAdapter`）避免主流程 if/else；Qwen3-TTS 的 adapter 实现了**动态 initial chunk size 覆盖以优化 TTFA**。
- **模型执行层**：`vllm_omni/model_executor/models/qwen3_tts/`（`qwen3_tts_talker.py`、`qwen3_tts_code2wav.py`、`cuda_graph_decoder_wrapper.py`、`qwen3_tts_code_predictor_vllm.py`、`tokenizer_12hz/`）。
- **stage 输入处理器**：`vllm_omni/model_executor/stage_input_processors/qwen3_tts.py`，根据 `async_chunk` 开关在 chunked 与端到端 codec 处理器之间自动分派。
- **平台抽象**：`vllm_omni/platforms/{cuda,rocm,npu,xpu,musa}/`，`NPUARWorker` / `XPUARWorker` 等；NPU 侧对 `qwen3_tts_code2wav`、`qwen3_tts_tokenizer_v2` 打 patch 以适配昇腾算子。

#### Qwen3-TTS 三阶段的计算画像（决定怎么分配资源）
| Stage | 职责 | 计算画像 | 生命周期（实测） |
|---|---|---|---|
| Talker（stage 0） | 自回归预测每帧第 1 个码本 token | AR、KV cache 密集 | 随文本长度增长 |
| Code Predictor | 生成剩余 15 个码本 token | AR 但**结构极规整**（每帧固定 15 步） | 随音频帧数增长 |
| Code2Wav（stage 1） | codec token → 波形（流式分块） | 时延敏感、可批量 | Base 克隆约 **3 s/请求**；CustomVoice/VoiceDesign 仅 **50–200 ms** |

#### SGLang-Omni 组件
`sglang_omni/serve/speech_service.py`、`preprocessing/resource_connector.py`、Coordinator + 各 stage Scheduler、relay data plane（shared-memory / NCCL / NIXL / Mooncake）、`sglang_omni_router`。


### 6. 音频编码器 ⭐

从引擎视角看，音频 encoder 的位置与代价各不相同：

- **Qwen3-ASR（vLLM 主线）**：audio frontend + encoder + adaptor 仍在 PyTorch 里跑，只有 LLM decode 部分交给 vLLM。FunASR 的 vLLM 引擎文档明确写了这个分工——"音频 frontend、encoder、adaptor 和可选 CTC 时间戳解码器仍在 PyTorch 中运行；LLM 解码部分通过 prompt embedding 交给 vLLM"。因此 **encoder 的设备由 `device=` 参数单独指定**（如 `device="cuda:0"`），与 vLLM 的 `gpu_memory_utilization` 是两套预算。
- **audio token 帧率**：Qwen3-ASR **25 tok/s**（社区整理数据，来源 lattifai README，非 vLLM 官方文档，**标注为待一手核实**）；Whisper 50 tok/s 但强制 30 s 窗口；Voxtral 12.5 tok/s。帧率直接决定长音频的 prompt 长度与 KV cache 占用。
- **Qwen3-TTS（vLLM-Omni）**：输入侧没有"音频 encoder"，只有文本 embedding（官方 TensorRT Edge-LLM 文档明确："Unlike Qwen3-Omni, Qwen3-TTS has no Thinker or visual encoder. The text embedding is self-contained in the Talker, exported as `text_embedding.safetensors`"）。参考音频侧走 **speech tokenizer / codec encoder**（12 Hz 帧率），只在 voice cloning（Base / CosyVoice3）时执行。
- **CosyVoice3（vLLM-Omni）**：需要 **S3-Tokenizer-v3** 参考音频编码器，且必须提供 `ref_audio + ref_text`。


### 7. 语音 Tokenizer 与编解码器 ⭐

- **Qwen3-TTS codec**：12 Hz 多码本 RVQ/FSQ 结构，Talker 出第 1 个码本、Code Predictor 出剩余 15 个（共 16 码本）；输出 **24 kHz**。vLLM-Omni 的 code2wav 分块参数 `codec_chunk_frames`（默认 25 帧）直接决定首包延迟与解码开销的平衡。
- **CosyVoice3 codec**：Qwen2.5-0.5B LLM 生成 **FSQ 6561** 语音 token（**25 Hz**），DiT 里上采样 25 Hz → 50 Hz 后出 80-band mel，HiFi-GAN 出 24 kHz。参考音频编码器为 **S3-Tokenizer-v3**（12 层 Conformer + FSMN + FSQ，242M 参数，bf16 约 462 MB）。
- **TensorRT Edge-LLM 的 codec 导出**：`tensorrt-edgellm-export` 会把 Qwen3-TTS 拆成三个 ONNX 目录：`llm/`（Talker，含 `embedding.safetensors` codec embedding + `text_embedding.safetensors` + `text_projection.safetensors`）、`code_predictor/`（含 `codec_embeddings.safetensors`、`lm_heads.safetensors`、`small_to_mtp_projection.safetensors`）、`code2wav/`（vocoder ONNX）。
- **流式分块与音质的耦合**：Code2Wav 分块后**块与块之间没有 left-context overlap**，因此不同 `chunkFrames` 下 WAV **不是字节一致**（块边界 PCM 会损失少量上下文）；但 RVQ codes 与分块数无关、是 bit-exact 的。


### 8. LLM 主干 ⭐

- **Qwen3-ASR**：Qwen3 系列 AR 解码器（0.6B / 1.7B）。vLLM 侧支持 tensor parallel 多卡；FunASR vLLM 引擎首次运行会从 `model.pt` 抽取 LLM 权重到 vLLM 兼容目录（如 `Qwen3-0.6B-vllm`），后续复用。
- **Qwen3-TTS Talker / Code Predictor**：同为 Qwen3 系 AR transformer，在 vLLM-Omni 里作为两个 stage 分别由 vLLM 引擎托管（PagedAttention + continuous batching + CUDA Graph）。
- **CosyVoice3**：**Qwen2.5-0.5B**（24 层、896 hidden、14 查询头 / 2 KV 头 GQA）做语音 token 生成器，输入形如 `[sos, text_tokens..., task_id, speech_tokens...]`，带 KV cache 自回归。
- **采样策略**：vLLM-Omni 的默认 `default_sampling_params` 可按 stage 配置（示例 YAML 中见过 `temperature: 0.0 / top_p: 1.0 / top_k: -1 / repetition_penalty: 1.1 / seed: 42`）。CosyVoice 系另有 **RAS（Repetition Aware Sampling）** 提升长文本稳定性（CosyVoice roadmap 2024/08）。


### 9. 声学解码器 ⭐

- **Qwen3-TTS**：Code2Wav 为**因果 codec 解码器**，vLLM-Omni 支持按 chunk 增量解码并即时下发 PCM（`async_chunk: true` 时走 chunked 处理器，`--no-async-chunk` 切回端到端同步路径）。
- **CosyVoice3**：**22 层 DiT flow matching**（1024 hidden、16 heads、**10 步 Euler ODE**、CFG rate 0.7、AdaLN 时间步条件），把 speech token 转成 80-band mel，token 上采样 **25 Hz → 50 Hz**（线性插值）。MLX 侧还实现了 **chunk-aware 因果掩码**（25 token/块 ≈ 1 s 音频），目标首块约 **150 ms**。
- Triton+TRT-LLM 路径里 CosyVoice 另有 **DiT-based Token2Wav**（借用 Step-Audio2 的 DiT）可选，见 `runtime/triton_trtllm/README.DIT.md`。


### 10. 声码器

- **Qwen3-TTS**：code2wav 内置波形合成，vLLM-Omni 输出 **24 kHz**（流式 PCM 播放命令用 `play -t raw -r 24000 -e signed -b 16 -c 1 -`）。对比：Fish Speech S2 Pro 44.1 kHz、VoxCPM2 48 kHz。
- **CosyVoice3**：**HiFi-GAN**（Neural Source Filter，8 次谐波 + F0 预测 + 3 级上采样 [8,5,3]=120x + ISTFT(hop=4)=480x，Snake 激活，n_fft=16），输出 **24 kHz**。MLX 导出为 fp32、weight-norm 已折叠（`w = g * v / ||v||`），体积 79 MB。
- 引擎侧共性：**vocoder 是 streaming 的最后一个瓶颈**。vLLM-Omni 提供 `active_stream_window: 2`（deploy YAML 顶层）用于限制同时活跃的 stage-1 流，在 vocoder-bound 过载（目标并发下单流 `rtf_p99 ≥ 1`）时用 TTFP 换取流式连续性。


### 11. 上下文长度与 I/O 约束

- **Qwen3-ASR（vLLM）**：`/v1/audio/transcriptions` 接收 multipart 音频文件；`/v1/chat/completions` 用 `{"type":"audio_url","audio_url":{"url":...}}`。QwenLM 官方 `qwen-asr` 包推荐参数：`max_inference_batch_size=128`、`max_new_tokens=4096`（长音频调大）、`gpu_memory_utilization=0.7`。昇腾 vllm-ascend 示例用 `--max-model-len 32768 --block-size 128`。
- **Qwen3-TTS（vLLM-Omni）**：`max_new_tokens` 默认 2048、范围 1–4096；`speed` 0.25–4.0（**流式时不支持 speed**）；`response_format` wav|mp3|pcm|flac|aac|opus。音频采样率 24 kHz。
- **WebSocket 流式协议（vLLM-Omni `/v1/audio/speech/stream`）**：`input.done` flush 但不关闭连接，可在一条连接上连续合成多句。
- **音频时长上限**：Qwen3-ASR 侧社区经验值「最长 20 分钟」（**待核实**）；SenseVoice 原生 C++ server 用 `--max-audio-seconds 300` 显式限制。
- **音频格式统一**：工程上建议统一重采样到 **16 kHz 单声道 PCM16** 再送 ASR（见 production_deployment.json 的传输章节）。


### 12. 微调支持

引擎层不提供微调能力，但影响微调后的部署路径：

- **vLLM**：Qwen3-ASR 微调后 checkpoint 可直接 `vllm serve ./checkpoints/xxx`（社区实测单卡 A100 40G：`--tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-num-seqs 256 --max-model-len 480000 --enforce-eager --dtype bfloat16`，并强调 `--max-model-len` 必须与训练时 `--max_length` 一致，否则长音频被截断；`--enforce-eager` 用于规避动态音频长度导致的 CUDA Graph 编译失败）。**注意：该组参数与数值来自社区博客，待一手核实。**
- **量化微调模型**：Qwen3-ASR-FastAPI 社区项目给出 4bit / AWQ / GPTQ 三种模式并自动回退 BF16；并明确指出 **vLLM 0.14.0 原生 4-bit 支持有限**，会自动 fallback 到 BF16/FP16 并打日志（"⚠ Model loaded in fallback mode"），属预期行为。
- **RL 训练侧**：vLLM-Omni 通过 VeRL-Omni v0.2.0 支持 diffusion RL（request-level/step-wise batching + FA3）。


### 13. 语种与方言覆盖

引擎层不做语言处理，但暴露语言相关参数：

- **Qwen3-ASR**：模型侧支持 52 语种/方言（官方）；vLLM `/v1/audio/transcriptions` 支持 `language` 参数（社区实测显式传 `language=zh` 更稳）。SGLang-Omni 文档标注 Qwen3-ASR-1.7B 支持 **30 个 language hints**。
- **Qwen3-TTS**：vLLM-Omni 请求体 `language` 参数，可选值**启动时从模型 config.json 动态加载**（非硬编码），非法值会被校验拒绝；SGLang-Omni 标注 10 语种。
- **CosyVoice3**：9 语种（中/英/日/韩/德/西/法/意/俄）。
- 引擎侧的工程提示：**语种提示（language hint）应显式传**，比让模型自动检测更省时延、也更可控。


### 14. 流式能力

#### vLLM 主线（ASR）
- Qwen3-ASR 的流式推理**仅 vLLM backend 支持**（`qwen-asr` 官方包说明：streaming inference 目前只在 vLLM 后端可用，**不支持 batch 推理、也不返回时间戳**）。另有 `qwen-asr-demo-streaming` 可直接起 Web demo（`--asr-model-path Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8000`）。
- API 模式三选一（社区整理，来源 lattifai README）：
  | Mode | Endpoint | 适用 |
  |---|---|---|
  | `transcriptions`（默认） | `/v1/audio/transcriptions` | 专用 ASR 模型（Qwen3-ASR、Whisper、GLM-ASR） |
  | `chat` | `/v1/chat/completions` | 通用 LLM（Gemma-3n 等），非 ASR 模型自动切换 |
  | `realtime` | `/v1/realtime`（WebSocket） | Voxtral Realtime（仅 1 并发） |
- FunASR 的 vLLM 引擎提供 **流式 SDK**（`FunASRNanoStreamingVLLM`，720 ms chunk 重编码 + rollback 8 字符，返回 fixed/unfixed 区域）与 **生产级 WebSocket 实时服务**（`serve_realtime_ws.py`，含动态 VAD、partial 预览、幻觉重复清理、热词、语种提示、说话人分离）。

#### vLLM-Omni（TTS）—— 在线流式已可用
- **启动**：`vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8091`（或 `./qwen3_tts/run_server.sh [CustomVoice|VoiceDesign|Base]`）。
- **HTTP 流式 PCM**（需 deploy config 中 `async_chunk: true`，`qwen3_tts.yaml` 默认开启）：
  ```
  curl -X POST http://localhost:8091/v1/audio/speech -H "Content-Type: application/json" \
    -d '{"input":"Hello","voice":"vivian","language":"English",
         "stream":true,"stream_format":"audio","response_format":"pcm"}' --no-buffer
  ```
  流式时不支持 `speed`。
- **WebSocket 增量流式**：`/v1/audio/speech/stream`，`python qwen3_tts/streaming_speech_client.py --text "..." [--simulate-stt --stt-delay 0.1]`，可一条连接多句（`input.done` 只 flush 不断连）。
- **离线批量流式**：`python end2end.py --query-type CustomVoice --streaming --output-dir /tmp/out_stream`（要求 stage config 里 `async_chunk: true`）。
- **分块调优旋钮**：`codec_chunk_frames`（默认 25 帧）、`initial_codec_chunk_frames`（可按请求覆盖，用于压 TTFA）。

#### SGLang-Omni
drop-in 支持 `/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/audio/translations`、`/v1/chat/completions`，均带 real-time streaming；另有 `/v1/realtime` WebSocket（含服务端 VAD 与 **server-owned barge-in**，2026-08-12 合并）。

#### 流式质量的通用坑（跨引擎）
1. **前导静音**：模型返回的首段 PCM 常含数十毫秒静音，Nari Labs 实测这部分约 **80 ms**，需要**动态 trim**（短 RMS 窗检测持续语音，丢掉 onset 之前的样本）才是真实"可听 TTFA"。
2. **帧累积（frame accumulation）**：初始 chunk 小 → TTFA 低但播放余量小、解码更频繁；后续 chunk 应逐步放大。**推荐"小起大续"**的 ramp 配置。
3. **underrun（播放卡顿）**：比 TTFA 更重要却常被忽略。Nari Labs 实测 vLLM-Omni 默认配置下 **100% 请求出现 underrun**。调引擎时必须同时压 TTFA 和 underrun 率。
4. **单客户端基准没有意义**：同一份配置 1 RPS 与 6 RPS 的排名会**反转**（详见 latency_throughput）。


### 15. VAD 与后处理

引擎层本身不做 VAD/标点，但决定它们挂在哪一层：

- **vLLM 主线（Qwen3-ASR）**：服务端**不内置 VAD**。生产上两种挂法——(a) 客户端/网关侧用 Silero VAD 或 WebRTC VAD 切句后逐段 POST `/v1/audio/transcriptions`；(b) 用 FunASR 的 `serve_realtime_ws.py`，服务端集成 **动态 VAD**（按累计语音时长自动收紧静音阈值）后再送 vLLM 解码。
  FunASR 动态 VAD 默认阈值表（累计语音时长 → 静音阈值）：≤5 s → 2.0 s；5–10 s → 1.5 s；10–15 s → 1.0 s；15–30 s → 0.8 s；30–45 s → 0.4 s；>45 s → 0.1 s。可用 `silence_schedule=[(5000,1000),(15000,500),(inf,200)]` 自定义，`dynamic_silence=False` 关闭。
- **标点 / ITN**：Qwen3-ASR 与 SenseVoice 自带标点输出；Paraformer 需外接 `ct-punc`。vLLM 侧对应 FunASR `generate(itn=True)` 的 ITN 开关。
- **时间戳**：Qwen3-ASR 需额外加载 **`Qwen/Qwen3-ForcedAligner-0.6B`**（官方建议同时装 FlashAttention 加速 forced aligner）；**流式推理不支持返回时间戳**。
- **说话人分离**：CAM++ 作为独立模型挂在服务层（`spk_model="cam++"`），不是在 vLLM 里。
- **CosyVoice3 / Qwen3-TTS（TTS 侧）**：无 VAD；参考音频的语音活动由调用方保证。


### 16. 音色克隆

- **Qwen3-TTS Base**：vLLM-Omni 支持。请求体 `task_type=Base` + `ref_audio`（本地路径 / HTTP URL / base64 data URL）+ `ref_text`；`--mode-tag icl`（默认，标准 ICL 模式）或 `xvec_only`（`x_vector_only_mode`）。
- **预计算音色（强烈推荐生产使用）**：`python qwen3_tts/precompute_custom_voice.py --model ... --voice-name alice --ref-audio ... --ref-text "..." --mode icl --output-dir /path/to/custom_voices`，然后在 deploy YAML 加 `custom_voice_dir: /path/to/custom_voices`，之后只传 `voice="alice"` 即可，省掉每次请求的参考音频编码。`--mode icl` 存 speaker_embedding + ref_code，`--mode xvec` 只存 speaker embedding。
- **运行时上传音色**：`POST /v1/audio/voices`（multipart：`audio_sample` / `consent` / `name` / `ref_text` / `speaker_description`），`GET /v1/audio/voices` 列出。
- **CosyVoice3**：vLLM-Omni 上**只有 voice cloning**，每个请求都必须给 `ref_audio + ref_text`，**没有内置音色预设**；3 秒参考音频即可零样本克隆。
- **SGLang-Omni**：Qwen3-TTS-12Hz-Base 支持 voice cloning + streaming。
- **说话人嵌入缓存（跨引擎的通用加速点）**：CosyVoice Triton runtime 的 `use_spk2info_cache=True` 在固定说话人场景下把并发 1 的首块延迟从 220.43 ms 降到 189.88 ms、RTF 从 0.1237 降到 0.1155（单 L20 GPU，seed_tts_cosy2）——**固定音色务必开缓存**。


### 17. 情感与风格控制

- **Qwen3-TTS CustomVoice**：`instructions` 字段做风格/情感自然语言指令（如"用开心的语气说"）；`voice` 指定预设说话人（vivian / ryan 等，可用值启动时从模型 config.json 动态加载）。
- **Qwen3-TTS VoiceDesign**：`task_type=VoiceDesign` + `instructions` 描述式音色设计。
- **语速**：vLLM-Omni `speed` 参数 0.25–4.0，用 `torchaudio.transforms.TimeStretch`（相位声码器）变速不变调；**流式模式不支持**。
- **说话人嵌入插值**：`qwen3_tts/speaker_embedding_interpolation.py` 可混合两个预设说话人的嵌入生成中间音色。
- **NVIDIA Dynamo 侧的现状**：其 `/v1/audio/speech`（vLLM-Omni worker）目前**不支持 audio streaming 与 Base（voice cloning）task**，只覆盖 CustomVoice 与 VoiceDesign——选型时若需要克隆，不要用 Dynamo 这条路径。


### 18. 鲁棒性

- **过载保护**：vLLM-Omni 提供 `active_stream_window: 2`（deploy YAML 顶层），在 vocoder-bound 过载（目标并发下单流 `rtf_p99 ≥ 1`）时限流，用 TTFP 换流式连续性（PR #3592）。
- **并发过冲会丢请求**：SGLang-Omni 官方文档记录 Qwen3-ASR 在并发 64 时**丢失 1–4% 请求（HTTP 500）**，根因是 `max_running_requests=32` + `request_build_max_pending=16` 的 build backlog 溢出。**必须做准入队列与背压设计**。
- **默认配置的 underrun**：vLLM-Omni 默认 `async_chunk` 流式在 1 RPS 下 100% 请求出现播放卡顿，必须调 `codec_chunk_frames` / `initial_codec_chunk_frames` 才能上生产。
- **TensorRT plan 不可移植**：SenseVoice/CosyVoice 的 TensorRT engine 与 GPU 架构 + TensorRT 版本强绑定，**换卡必须重建并重验精度**（FunASR 官方部署契约明确写出）。
- **ROCm 特殊依赖**：vLLM-Omni 在 ROCm 上需要 `pip uninstall onnxruntime` 后装 `onnxruntime-rocm` 与 `sox`，否则 Qwen3-TTS 跑不起来。
- **长音频**：Qwen3-ASR 长音频需调大 `max_new_tokens` / `max_model_len`，否则截断；`--max-model-len` 必须与微调时的 `--max_length` 一致。


### 19. ASR 指标

引擎层不改模型精度，但基准数据可用于交叉验证部署是否退化：

- **Qwen3-ASR-1.7B**（SGLang-Omni 官方并发剖面，H100 80GB，bf16，SeedTTS EN 1088 clips / ZH 2020 clips，3 次重复均值）：
  - EN 语料 WER 稳定在 **0.0122**（并发 1/8/16/32），并发 64 时 0.0125（**上浮仅因被拒绝请求改变了分母，不是识别退化**）。
  - ZH 语料 CER 稳定在 **0.0062**，并发 1/8/16/32/64 全部一致。
  - **这说明并发压到饱和点之前，WER/CER 不退化；如果你的压测里精度掉点，先查是不是有请求被 shed 或超时重试。**
- **FunASR vLLM 引擎**官方性能表（Fun-ASR-Nano）：
  | 场景 | PyTorch baseline | vLLM | 加速 |
  |---|---|---|---|
  | 离线，5.6 s 音频 | 0.89 s | 0.30 s | **3x** |
  | 离线，2-GPU tensor parallel | 0.89 s | ~0.20 s | **4.5x** |
  | 批量 16 条 | 约 16x 串行成本 | 约 4x | **4x** |
  | 批量 32 条 | 约 32x 串行成本 | 约 5x | **6x** |
  | WebSocket RTF | 0.156 | 0.078 | **2x** |
- 其他 ASR 横向数据见 funasr_sensevoice.json 的 asr_metrics 字段。


### 20. TTS 指标

引擎层对 TTS 指标的影响主要在**流式 vs 非流式的质量落差**与**首包延迟**：

- **流式与非流式的等价性（TensorRT Edge-LLM 官方结论，可推广）**：Qwen3-TTS 流式模式下 RVQ codes 与同输入同 seed 的非流式路径 **bit-exact**；差异只发生在 Code2Wav 块边界（块间无 left-context overlap），因此**不同 chunkFrames 下 WAV 不字节一致**，块边界 PCM 会损失少量上下文。→ 评测流式输出时应固定 chunk 配置再横向比。
- **Nari Labs 用 Deepgram STT 回听**评估合成音频的可懂度，作为"非畸形输出"的自动校验（他们把"non-malformed output"列为实时 TTS 的四要素之一：低可听 TTFA、零 underrun、有容量、输出可懂）。
- CosyVoice3 / Qwen3-TTS 的 WER/CER/SIM/UTMOS 具体数值见另两个模型的条目；本条目只记录**引擎相关的度量方法学**（见 eval_toolchain）。


### 21. 延迟与吞吐

#### A. Qwen3-TTS 首包延迟（TTFA）—— Nari Labs 2026-08-19 公开基准
测试条件：**单张 H100 SXM，Qwen3-TTS-12Hz-1.7B-CustomVoice，Poisson 开环流量，每种配置跑 5 分钟**，用**可听 TTFA**（从请求发出到第一个人耳能听到的采样，已做前导静音 trim）而非原始 TTFA。

| 引擎 | 默认 p95 可听 TTFA（1 RPS） | 调优后 p95（1 RPS） | 调优后 p95（6 RPS） | 默认 underrun 率 |
|---|---|---|---|---|
| vLLM-Omni | 277.9 ms | **56.8 ms** | **93.5 ms** | **100%** |
| SGLang-Omni△ | 1140.7 ms | 120.9 ms | 273.7 ms | — |
| VoxServe | 315.1 ms | **49.3 ms** | 363.2 ms | 0% |
| M-star（匿名研究系统） | 1160.0 ms | 104.0 ms | 179.5 ms | — |
| Nari Labs 自研 | — | **sub-50 ms（到 10 RPS）** | **<100 ms（20 RPS）** | 0% |

（△ SGLang-Omni 在测试时对 Qwen3-TTS 的上游支持尚不完整；该行数值应视为**时点快照**，SGLang-Omni 后续已持续补完 Qwen3-TTS 支持，见 inference_frameworks。）

关键读数：
1. **排名会反转**：1 RPS 下 VoxServe 最快（49.3 ms），6 RPS 下最慢（363.2 ms，7.4x 退化）；vLLM-Omni 只退化 1.6x。**单客户端基准无法预测生产 p95。**
2. **前导静音约 80 ms**，动态 trim 是纯收益（不改模型推理）。
3. Nari 自研栈在 10 RPS 时约 **630 chars/s**，20 RPS 仍 <100 ms。

#### B. Qwen3-ASR 并发剖面（SGLang-Omni 官方文档，H100 80GB，bf16，DP=1）
默认参数：`max_running_requests=32`、`request_build_max_workers=2`、`request_build_max_pending=16`、async decode 开启。

**SeedTTS EN（1088 clips）**
| 并发 | req/s | RTFx | 延迟 mean | 延迟 p95 | WER | 完成数 |
|---|---|---|---|---|---|---|
| 1 | 11.6 | 54.8 | 0.088 s | 0.129 s | 0.0122 | 3264/3264 |
| 8 | 27.5 | 130.4 | 0.289 s | 0.458 s（0.437） | 0.0122 | 3264/3264 |
| 16 | 53.0 | 251.2 | 0.304 s | 0.458 s | 0.0122 | 3264/3264 |
| 32 | 108.0 | 511.4 | 0.295 s | 0.399 s | 0.0122 | 3264/3264 |
| 64 | 103.0 | 486.8 | 0.610 s | 0.741 s | 0.0125 | 3165/3264 |

**SeedTTS ZH（2020 clips）**
| 并发 | req/s | RTFx | 延迟 mean | 延迟 p95 | CER | 完成数 |
|---|---|---|---|---|---|---|
| 1 | 14.3 | 66.9 | 0.070 s | 0.088 s | 0.0062 | 6060/6060 |
| 8 | 64.2 | 300.8 | 0.124 s | 0.175 s | 0.0064 | 6060/6060 |
| 16 | 96.5 | 451.9 | 0.165 s | 0.231 s | 0.0062 | 6060/6060 |
| 32 | 126.4 | 591.6 | 0.252 s | 0.353 s | 0.0062 | 6060/6060 |
| 64 | 129.2 | 604.3 | 0.489 s | 0.613 s | 0.0062 | 5993/6060 |

**瓶颈分解（EN，profiled passes，ms）**
| 并发 | build | build→queued | queued→scheduled | first forward（encoder+prefill） | decode tail | total |
|---|---|---|---|---|---|---|
| 1 | 5.6 | 0.7 | 1.1 | 28.0 | 55.3 | 91 |
| 8 | 6.0 | 16.5 | 2.0 | 29.5 | 131.8 | 196 |
| 16 | 6.1 | 20.0 | 1.9 | 30.1 | 185.6 | 259 |
| 32 | 6.5 | 23.1 | 5.9 | 46.2 | 357.7 | 472 |
| 64 | 6.3 | 30.1 | **412.8** | 33.3 | 392.6 | **910** |

结论：**GPU 利用率全程只有 33%–54%，GPU 根本不是瓶颈**；拐点来自并发 64 的准入队列（413 ms 等待）与 host 侧 decode 逐 step 派发开销。async decode（PR #1326）相比之前把饱和吞吐提升约 **10%**。

#### C. 其他实测点
- **Qwen3-TTS 0.6B-Base 在 RTX 3090 24GB（vLLM-Omni 默认配置，未量化）**：首包 TTFB ~130 ms，稳态显存 ~13 GB（含 KV cache），输出 PCM 24 kHz 单声道（**社区实测，待一手核实**）。
- **vLLM-Omni 官方 Qwen3-TTS Base 调优数据**：stage 1 `max_num_seqs: 10`（默认，针对 Base 克隆）在 **1×H100、1.7B-Base、seed-tts、c=4/c=8** 下带来约 **2× TTFA p95 改善**，代价是 **8–12% 音频吞吐损失**。
- **Qwen3-ASR 社区实测（A100 40G，社区博客数据，待一手核实）**：单并发 RTF 0.012；128 并发 RTF 0.068（约 1980× 实时）；TTFT 平均 112 ms。
- **CosyVoice3（vLLM-Omni + TensorRT 优化，社区博客数据，待一手核实）**：TTFP ~2800 ms → ~200 ms（14×），RTF 0.66 → 0.12（5.5×），吞吐 0.21 → 1.08 req/s（5.1×）。优化点：speaker embedding 从 CPU ONNX-Runtime 搬到 GPU TensorRT、LLM 开 CUDA Graph + 向量化 RAS 采样、flow-matching 解码器迁到 TensorRT FP16、去掉每次请求重建 tokenizer 的开销（原 452 ms）。


### 22. 成本模型

#### 已核实（Nari Labs 2026-08，Apache-2.0 实现与基准已开源）
- **Qwen3-TTS-12Hz-1.7B-CustomVoice，1× H100 SXM** @ $4.29/小时：
  - 10 RPS、p95 可听 TTFA sub-50 ms；
  - 约 **630 chars/s @ 10 RPS**；
  - 满利用率折算约 **$2 / 1M 字符**（**算力成本下限**，不含网络、GPU 空转与运维）。
  - 对比：ElevenLabs V3 **$100 / 1M**，Cartesia Sonic 3.5 **$49 / 1M**（且 TTFA 更高）。

#### 待一手核实（用户提供，未能独立验证到 Baseten 原始博客）
- Baseten 成本模型（单 H100 副本）：25 并发 594 chars/s、RTF p50/p90 = 0.752/0.834、TTFA 271/431 ms、**$3.04/1M 字符**；1 并发 110 chars/s、**$16.41/1M**；32 并发 511 chars/s 但 p90 RTF 1.69 已超实时、**$3.53/1M**。
- 闭源对比 $30–50/1M 字符；原生 Qwen3-TTS 约 $4.80/1M。
→ **上述两组数值在文档中应标注"二手/待核实"。**

#### 成本结构要点（跨引擎通用）
1. **并发度是成本的第一变量，不是模型大小**：1 并发到 25 并发，单字符成本可差 5 倍以上。低并发跑 GPU 是最贵的用法。
2. **超过实时拐点后加并发不降成本**：32 并发时 p90 RTF 1.69（>1）意味着已无法实时，继续压并发只是把延迟转嫁给用户。
3. **算力成本 ≠ 交付成本**：需叠加网络出口、GPU 空转（低峰期）、多副本高可用冗余、运维。
4. 闭源 API 与自建的盈亏平衡点，取决于**你的实际并发曲线**——若业务是尖峰型（白天忙、夜里闲），自建的 GPU 空转成本会显著抬高单字符成本。


### 23. 推理框架支持

#### 1. vLLM 主线（Qwen3-ASR 首选）
**安装（官方 recipes，CUDA 12.9 nightly wheel）**：
```
uv venv && source .venv/bin/activate
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
uv pip install "vllm[audio]"     # 音频额外依赖，不装会导致音频请求 HTTP 400
```
**启动**：
```
vllm serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8081
# 或用官方 wrapper（qwen-asr 包提供，参数完全透传给 vllm serve）
qwen-asr-serve Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000
```
**Python 直连（官方 `qwen-asr` 包，vLLM 后端）**：
```
model = Qwen3ASRModel.LLM(model="Qwen/Qwen3-ASR-1.7B", gpu_memory_utilization=0.7,
    max_inference_batch_size=128, max_new_tokens=4096,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"))
```
注意：官方要求把代码包在 `if __name__ == '__main__':` 下，规避 vLLM 的 spawn 问题；时间戳场景建议 `pip install -U flash-attn --no-build-isolation` 加速 forced aligner。

**API 模式**（社区整理，来源 lattifai README；vLLM 官方文档确认了 `/v1/audio/transcriptions` 与 `/v1/chat/completions` 两条）：
- `transcriptions`（默认）→ `/v1/audio/transcriptions`，专用 ASR 模型；
- `chat` → `/v1/chat/completions`，通用 LLM 自动切换；
- `realtime` → `/v1/realtime`（WebSocket），Voxtral Realtime，仅支持 1 并发。

#### 2. vLLM-Omni（Qwen3-TTS / CosyVoice3 首选）
**与 vLLM 主线的关系（关键）**：
- 独立仓库 `vllm-project/vllm-omni`，**不是 pip 插件**，需要源码安装；
- **版本强绑定**：自 0.14.0 起，每个偶数号上游 vLLM minor 对应一个 vLLM-Omni 稳定版（0.16.0 / 0.18.0 / 0.20.0 / 0.22.0 / 0.24.0 / 0.26.0），仓库主干 2026-08-27 完成 "Rebase to vllm 0.28.0"。**社区实践明确要求 vLLM 与 vLLM-Omni 大版本号一致，否则会出现各种奇怪问题**；
- 环境要求：**Linux only**（Mac/Windows 不可用），**Python 3.12**。

**安装**：
```
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install vllm==0.22.0 --torch-backend=auto          # CUDA
# ROCm: uv pip install vllm==0.22.0+rocm721 --extra-index-url https://wheels.vllm.ai/rocm/0.22.0/rocm721
git clone https://github.com/vllm-project/vllm-omni.git && cd vllm-omni
uv pip install -e .[cuda]     # 或 .[rocm] / .[npu]
# ROCm 额外：pip uninstall onnxruntime && pip install onnxruntime-rocm sox
```

**Qwen3-TTS 在线服务（2026-08 已支持，勿信旧文）**：
```
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8091
./qwen3_tts/run_server.sh [CustomVoice|VoiceDesign|Base]
```
支持的 TTS 模型矩阵（vLLM-Omni 官方 `examples/online_serving/text_to_speech`）：

| 模型 | HF repo | 音色克隆 | 流式 | 音色预设/上传 | Gradio |
|---|---|---|---|---|---|
| **Qwen3-TTS** | `Qwen/Qwen3-TTS-12Hz-1.7B-{CustomVoice,VoiceDesign,Base}` | ✅ (Base) | ✅ PCM + WebSocket | ✅ 预设 + `/v1/audio/voices` | ✅ 标准 + FastRTC |
| **CosyVoice3** | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | ✅（必须） | ✅ PCM | ❌ 无预设 | ❌ |
| Fish Speech S2 Pro | `fishaudio/s2-pro` | ✅ | ✅ PCM（44.1 kHz） | ❌ | ✅ |
| higgs-audio v2 | `bosonai/higgs-audio-v2-generation-3B-base` | ✅ | ✅ codec_streaming | ❌ | ❌ |
| GLM-TTS | `zai-org/GLM-TTS` | ✅（必需） | ✅ PCM | ❌ | ✅ |
| VoxCPM2 | `openbmb/VoxCPM2` | ✅ | ✅ AudioWorklet（48 kHz） | ❌ | ✅ |
| Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | ✅ | ✅ | ✅ 预设 | ✅ |
| OmniVoice | `k2-fsa/OmniVoice` | — | 仅离线 | — | — |

**Qwen3-TTS 离线推理**：
```
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni/examples/offline_inference/qwen3_tts
python end2end.py --query-type CustomVoice                    # 单样本
python end2end.py --query-type VoiceDesign --use-batch-sample  # 批量
python end2end.py --query-type Base --mode-tag icl             # ICL 克隆（默认）
python end2end.py --query-type Base --mode-tag xvec_only       # 仅 x-vector
python end2end.py --query-type CustomVoice --streaming --output-dir /tmp/out_stream
```

#### 3. SGLang-Omni
- 安装：`uv pip install "sglang-omni==0.1.2"`（v0.1.2，2026/08；CLI 已集成进 `sglang serve`，PR #1531）。Docker 为推荐方式。
- 端点：`/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/audio/translations`、`/v1/chat/completions`，均支持 real-time streaming；另有带 VAD 与 **server-owned barge-in** 的 `/v1/realtime` WebSocket。
- 模型覆盖：TTS 侧 Higgs Audio v3 / Fish S2-Pro / Voxtral / **Qwen3-TTS-12Hz-Base（0.6B & 1.7B，10 语种）** / MOSS-TTS / Ming-Omni-TTS / dots.tts / ZONOS2；ASR 侧 **Qwen3-ASR-1.7B（30 language hints）** / Fun-ASR-Nano-2512-hf / ARK-ASR-3B / MOSS-Transcribe-Diarize / whisper-large-v3（实验）。**CosyVoice3 不在其支持列表**。
- 硬件：NVIDIA CUDA（默认，全覆盖）、Intel XPU（实验，Qwen3-ASR / Qwen3-TTS / Qwen3-Omni 端到端可跑，Omni thinker 走多 XPU TP）、AMD ROCm（README hardware table）。
- 附加：`sglang_omni_router`（多 worker OpenAI 兼容前门，健康检查 / 就绪 / 生命周期 / 能力发现）。

#### 4. NVIDIA TensorRT-LLM 家族
- **TensorRT Edge-LLM v0.6.0（2026-03-16 发布）**：官方 supported-models 页明确列出 **Qwen3-ASR-0.6B / 1.7B（FP16）** 与 **Qwen3-TTS-12Hz-1.7B-CustomVoice（FP16）**，精度矩阵里这两族目前只有 FP16 打勾（FP8 / INT4 AWQ / INT4 GPTQ / NVFP4 均为 "—"）。GTC 2026 旗舰 demo 为 Jetson AGX Thor 上同时跑 ASR + LLM + TTS。
- **Qwen3-TTS 边缘部署流程**（TensorRT Edge-LLM v0.9.0 文档）：
  1. x86 host 上 ONNX 导出：`tensorrt-edgellm-export Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice $ONNX_OUTPUT_DIR`，产出 `llm/`（Talker）、`code_predictor/`、`code2wav/` 三个目录；
  2. 设备端建三个引擎：`llm_build --maxInputLen 4096 --maxKVCacheCapacity 4096 --maxBatchSize 1`（Talker、Code Predictor 各一次）+ `audio_build`（Code2Wav）。**文档明确要求当前 Qwen3-TTS runtime 用 `--maxBatchSize 1`**；
  3. 推理：`qwen3_tts_inference --talkerEngineDir ... --code2wavEngineDir ... --streaming --chunkFrames=10`，每请求有独立 `onChunkReady` 回调即时声码化。
- **数据中心侧 TensorRT-LLM**：社区有「Qwen3-ASR-1.7B + TensorRT-LLM 提速 3.2x」的实践文章（**CSDN 社区数据，未能核实一手来源，标注为待核实**）。
- **NVIDIA Dynamo**：以 vLLM-Omni 为 worker（`--output-modalities audio`）暴露 `/v1/audio/speech`，`examples/backends/vllm/launch/agg_omni_audio.sh`；已测 Qwen3-TTS CustomVoice / VoiceDesign；**当前不支持 audio streaming 与 Base（voice cloning）**。

#### 5. Triton Inference Server
- **FunASR 官方 runtime**：SenseVoice TensorRT/Triton（生产验证），详见 funasr_sensevoice.json。
- **CosyVoice 官方 runtime**：`runtime/triton_trtllm`（Triton + TensorRT-LLM，NVIDIA Yuekai Zhang 2025-08 贡献）+ `README.DIT.md`（DiT Token2Wav 变体），详见 production_deployment.json。

#### 6. Transformers / 其他
- Qwen3-ASR 与 Qwen3-TTS 均支持原生 Transformers 推理（`Qwen3ASRModel.from_pretrained`、官方 Gradio demo），功能最全但**无 continuous batching**，只适合验证与离线批处理。
- **Ollama 不适用于这三款模型**（Ollama 只跑 GGUF、且不支持音频输入/语音输出）——若看到"用 Ollama 部署 Qwen3-TTS"的说法，是错的。


### 24. 资源需求

#### GPU 显存
- **vLLM-Omni 多阶段显存分配（最常见的坑）**：默认配置会为**每个 stage 独立申请 `gpu_memory_utilization` 比例的显存**，导致 GPU 被独占。规则（vLLM-Omni 官方 configuration 文档）：
  - `requested_memory = total_gpu_memory × gpu_memory_utilization`；系统会校验 `free_memory ≥ requested_memory`，不满足则该 stage 初始化失败并报错。
  - **共享同一张 GPU 的多个 stage，其 `gpu_memory_utilization` 之和必须 ≤ 1.0**（示例：stage 0 = 0.6、stage 1 = 0.3，留 10% buffer）。
  - **不同 GPU 上的 stage 各自可用到 1.0**。
  - 经验值：AR 阶段（Talker）权重 + KV cache + activation 都需要，给 **0.4–0.7**；扩散/生成阶段（code2wav）给 **0.1–0.3**。
  - 社区给出的 Qwen3-TTS 双阶段示例：`code_predictor: 0.4`、`code2wav: 0.3`。
  - 多 GPU 拆分：`CUDA_VISIBLE_DEVICES` 把不同 stage 放到不同卡；或用 `--stage-overrides '{"1": {"gpu_memory_utilization": 0.5}}'`。
- **实测显存占用**：Qwen3-TTS-12Hz-0.6B-Base 在 RTX 3090 24GB 上、vLLM-Omni 默认配置、未量化 → **稳态约 13 GB**（社区实测，待一手核实）。
- **CosyVoice3**：0.5B，bf16 全量约 2.1 GB（MLX bf16 bundle），4-bit 约 1.1 GB，属"轻量"档。
- **SenseVoice 侧**：详见 funasr_sensevoice.json（Triton FP16 TensorRT 需按 batch/frames 设定 optimization profile）。

#### 精度与量化
- **TensorRT Edge-LLM 精度矩阵（官方）**：Qwen3-ASR-0.6B/1.7B 与 Qwen3-TTS-12Hz-1.7B-CustomVoice **仅 FP16**（FP8/INT4/NVFP4 均未标注支持）。
- **vLLM 侧量化**：Qwen3-ASR-FastAPI 社区项目支持 4bit / AWQ / GPTQ 并自动回退 BF16；**vLLM 0.14.0 原生 4-bit 支持有限，会自动 fallback 到 BF16/FP16 并打日志**。
- **CosyVoice3 MLX 量化档**：bf16 (~2.1 GB) / 8-bit (int8 LLM + bf16 DiT，~1.4 GB) / 8-bit-full (int8 LLM + int8 DiT，~1.6 GB) / 4-bit (int4 LLM + int4 DiT，~1.1 GB)。官方建议：长文本上 8-bit 音质更干净、文本漂移更少。

#### 批处理 / 并发调优参数全表
**vLLM-Omni（三层优先级：显式 CLI > deploy YAML > parser 默认值）**
| 参数 | 作用 | 推荐值 / 说明 |
|---|---|---|
| `gpu_memory_utilization` | 每 stage 显存占比 | 共享 GPU 时各 stage 之和 ≤ 1.0 |
| `max_num_seqs`（每 stage） | 该 stage 并发序列上限 | **Base 克隆 stage-1 用 10**（生命周期 ~3 s/req）；**CustomVoice / VoiceDesign stage-1 用 1**（生命周期 50–200 ms，TTFA 最优） |
| `async_chunk` | 是否 chunked 流式 | `true`（`qwen3_tts.yaml` 默认）；`--no-async-chunk` 切回同步 |
| `codec_chunk_frames` | Code2Wav 每块帧数 | 默认 25 帧；调小降 TTFA、调大保连续播放 |
| `initial_codec_chunk_frames` | 首块帧数（可按请求覆盖） | 专用于压 TTFA |
| `active_stream_window` | 限制同时活跃 stage-1 流 | 过载（单流 `rtf_p99 ≥ 1`）时设 2，用 TTFP 换连续性 |
| `enable_prefix_caching` / `no-enable-prefix-caching` | 前缀缓存 | YAML 默认 off，可按 stage 开 |
| `enforce_eager` | 禁用 CUDA Graph | 调试用；动态音频长度导致编译失败时也需要 |
| `--stage-overrides '{...}'` | 按 stage 覆盖任意 engine 参数 | 多 stage 共享 GPU 时**不要用全局显存参数** |
| `--deploy-config <path>` | 覆盖默认 deploy YAML | 按 HF `model_type` 自动加载 |

**批量解码的硬约束（vLLM-Omni 离线）**：
```
python end2end.py --query-type CustomVoice \
  --txt-prompts benchmark_prompts.txt --batch-size 4 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts_batch.yaml
```
- `--batch-size` **必须匹配 CUDA Graph 捕获尺寸（1/2/4/8/16…）**，因为 Talker 的 code predictor KV cache 按 `max_num_seqs` 分配，CUDA Graph 会把 batch 补齐到下一个捕获尺寸；
- **两个阶段都要设 `max_num_seqs >= batch_size`** —— 只在 stage 1 调高无效，stage 1 只能批量处理"同时在跑"的请求的块，前提是 stage 0 也在并发处理多个请求。

**SGLang-Omni（Qwen3-ASR）**
| 参数 | 默认 | 说明 |
|---|---|---|
| `max_running_requests` | 32 | 并发 64 时会溢出 → 1–4% 请求被拒（HTTP 500） |
| `request_build_max_workers` | 2 | build 线程数 |
| `request_build_max_pending` | 16 | build backlog 上限，是并发 64 掉请求的主因 |
| async decode | on | 相对基线提升饱和吞吐约 10%（PR #1326） |

**vLLM 主线（Qwen3-ASR）**：`--gpu-memory-utilization 0.8`、`--max-model-len`（长音频调大、须与微调 `--max_length` 一致）、`--max-num-seqs`、`--tensor-parallel-size`、`--enforce-eager`（规避动态长度 CUDA Graph 编译失败）、`--dtype bfloat16`。

#### CPU / 边缘
- vLLM-Omni **仅 Linux + Python 3.12**，无法在 macOS/Windows 跑；边缘走 TensorRT Edge-LLM（Jetson Thor / Orin）。
- ASR 的 CPU / 边缘路径见 funasr_sensevoice.json（llama.cpp / GGUF、SenseVoice 原生 C++ server）。


### 25. 服务化栈

#### API 面
| 引擎 | 端点 | 说明 |
|---|---|---|
| vLLM 主线 | `/v1/audio/transcriptions` | OpenAI 兼容转写（Qwen3-ASR 默认） |
| vLLM 主线 | `/v1/chat/completions` | `audio_url` content part；通用 LLM 自动切换 |
| vLLM 主线 | `/v1/realtime`（WS） | Voxtral Realtime，仅 1 并发 |
| vLLM-Omni | `/v1/audio/speech` | TTS；`stream=true` + `stream_format=audio` + `response_format=pcm` 出流式 PCM |
| vLLM-Omni | `/v1/audio/speech/stream`（WS） | 增量文本输入 → 流式音频，`input.done` flush |
| vLLM-Omni | `/v1/audio/voices` | GET 列出 / POST 上传自定义音色 |
| SGLang-Omni | `/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/audio/translations`、`/v1/chat/completions`、`/v1/realtime`（WS，带 VAD + barge-in） | 全部支持 real-time streaming |
| Triton | gRPC (8001) / HTTP (8000) / `/v2/health/ready` | CosyVoice / SenseVoice 官方 runtime |

#### 编排与部署形态
- **vLLM-Omni 三种启动形态**：
  1. **统一进程（uniproc executor）**——单 GPU 服务现在的默认（更低 IPC 开销，针对 Base 克隆场景，issue #2603 / PR #2604）；
  2. **stage-based CLI**——每个实例只起一个 stage（`--stage-id N --headless --omni-master-address ... --omni-master-port ...`），适合把阶段拆到不同 GPU/节点；
  3. `--stage-overrides` 做细粒度覆盖，YAML 太长时写成继承默认配置的 overlay（`base_config: /path/to/vllm_omni/deploy/qwen3_omni_moe.yaml`）。
- **SGLang-Omni Router**：多 worker 前门，统一健康/就绪/生命周期/能力发现。
- **NVIDIA Dynamo**：`examples/backends/vllm/launch/agg_omni_audio.sh` 起 vLLM-Omni worker（`--output-modalities audio`）。
- **昇腾**：`quay.io/ascend/vllm-omni:v0.14.0` 镜像 + docker-compose 一键起 Qwen3-TTS 服务（端口 18005 → 容器 8000，调 `/v1/audio/speech`）。
- **Gateway 职责（引擎不管，必须自己补）**：TLS、认证、租户配额、限流、审计、上传大小/时长限制、监控、灰度回滚。FunASR 官方部署契约反复强调"**runtime 负责推理，platform 负责暴露方式**"，这条对三个引擎都成立。

#### 健康检查三段式（FunASR 官方建议，可直接套用）
```
curl -fsS http://localhost:8000/health          # 存活
curl -fsS http://localhost:8000/v1/models       # 模型身份
curl -fsS http://localhost:8000/v1/audio/transcriptions -F file=@known.wav -F model=xxx  # 真实转写
```
**健康检查 ≠ 模型就绪 ≠ 能出正确结果，三个必须分开验。**


### 26. 硬件生态

| 硬件 | vLLM 主线 | vLLM-Omni | SGLang-Omni | TensorRT Edge-LLM |
|---|---|---|---|---|
| NVIDIA CUDA | ✅（cu129 nightly wheel） | ✅ 默认后端，全模型覆盖 | ✅ 默认后端 | ✅（Jetson Thor / Orin，SM80–SM120） |
| AMD ROCm | ✅ | ✅（需 `onnxruntime-rocm` + `sox`） | ✅ README hardware table | ❌ |
| 华为昇腾 NPU | ✅ 经 vllm-ascend（`quay.io/ascend/vllm-ascend:v0.22.1rc1`） | ✅（`quay.io/ascend/vllm-omni:v0.14.0`，Atlas A2/A3，对 `qwen3_tts_code2wav`/`tokenizer_v2` 打 patch） | 待核实 | ❌ |
| Intel XPU | ✅ | ✅（`docker/` 有 XPU 镜像） | ✅ 实验（Arc；Qwen3-ASR/TTS/Omni 端到端可跑） | ❌ |
| Moore Threads MUSA | ✅ | ✅（`platforms/musa/`） | 待核实 | ❌ |
| 海光 DCU | ✅ 定制 wheel（`vllm-0.15.1+das.opt1.alpha.dtk2604`） | ✅ vLLM 0.18.1 + vLLM-Omni 0.18.0 组合实测可行 | 待核实 | ❌ |
| Apple Silicon | ❌ | ❌（Linux only） | ❌ | ❌（走 MLX，见 production_deployment.json） |

**版本锁定是国产卡路径的第一要务**：海光路径的官方组合是 DTK 26.04 + Python 3.10.12 + torch 2.9.0+das.opt1.dtk2604 + transformers 4.57.6 + vllm 0.15.1+das.opt1.alpha；昇腾路径是 CANN 8.5.1 + torch_npu 2.9.0 + vLLM/vLLM-Ascend 0.17.0+。**任意一项版本漂移都可能直接跑不起来。**


### 27. 集成范式

#### 模式 A：文件/批量转写（最简单）
客户端 → 网关 → `POST /v1/audio/transcriptions`（multipart）→ vLLM(Qwen3-ASR)。
适合离线质检、字幕生成、批量归档。**用 OpenAI SDK 即可零改造接入**。

#### 模式 B：实时转写
浏览器/端上采集 → WebRTC(Opus) 或 WebSocket(PCM16) → 网关 → VAD 切句 → vLLM `/v1/audio/transcriptions`；
或直接用 FunASR `serve_realtime_ws.py` / `funasr-server`，服务端集成动态 VAD + partial 预览 + 说话人分离。

#### 模式 C：实时 TTS
`POST /v1/audio/speech`（`stream=true` + `response_format=pcm`）→ chunked HTTP 边收边播；
或 WebSocket `/v1/audio/speech/stream` 增量送文本、`input.done` 触发合成（适合 LLM token 流 → TTS 的边生成边合成）。
**工程要点**：按句/短语边界切分 LLM 输出流，每句立刻推 TTS 队列——这一个决策对感知延迟的贡献大于任何模型替换。

#### 模式 D：ASR → LLM → TTS 全链路
1. VAD 端点检测 → 2. ASR 出终稿 → 3. LLM 流式出 token → 4. **按短语边界（`.!?;:`）切成句** → 5. TTS 流式合成 → 6. 客户端播放。
- **barge-in（打断）**：播放期间仍对用户音频跑 VAD；一旦确认说话，立刻停播放、取消 TTS 请求、取消 LLM 生成、开始缓存新的用户语音。
- **前导静音 trim**：TTS 首段 PCM 的 ~80 ms 静音要在客户端/网关侧动态裁掉，否则 TTFA 数字好看、体感仍然慢。
- **统一采样率**：ASR 侧统一 16 kHz 单声道 PCM16；TTS 输出 24 kHz（Qwen3-TTS / CosyVoice3）或 44.1/48 kHz（Fish / VoxCPM2），播放前按模型采样率对齐。

#### 模式 E：多模型混合（推荐的生产形态）
- **ASR**：Qwen3-ASR（vLLM）负责高精度多语种/方言；SenseVoice / Paraformer（FunASR）负责高吞吐、边缘与情感/事件检测。两者通过同一个 OpenAI 兼容 `/v1/audio/transcriptions` 契约对外，网关按语种/成本/时延路由。
- **TTS**：Qwen3-TTS（vLLM-Omni）做通用与 VoiceDesign；CosyVoice3（vLLM-Omni 或 Triton+TRT-LLM）做方言与超低克隆门槛场景。
- vLLM-Omni 的 `/v1/audio/speech` 对这两种 TTS 是**同一套 API**，换模型只改参数，业务代码零改动。


### 28. 技术栈组件

下面是一份**可直接照抄的分层清单**，覆盖「音频采集 → 传输 → VAD → 特征提取 → 模型推理 → 服务化 → 编排 → 监控 → 评测」全链路。每层给出推荐组件与选型理由。

---

#### L0 音频采集（Capture）

| 场景 | 推荐组件 | 理由 |
|---|---|---|
| 浏览器 | **Web Audio API + AudioWorklet** + `getUserMedia({echoCancellation:true, noiseSuppression:true, autoGainControl:true})` | AudioWorklet 在独立音频线程运行，不受主线程 GC/jank 影响；浏览器原生 AEC/AGC/NS。采集后 `AudioContext` 重采样到 16 kHz 再编码 |
| 浏览器（快速原型） | **MediaRecorder + ScriptProcessor**（已废弃，仅原型） | 简单但有主线程抖动，不建议生产 |
| Python 服务端/桌面 | **sounddevice**（PortAudio 绑定）或 **PyAudio** | sounddevice API 更现代、NumPy 友好；PyAudio 生态老但资料多。**推荐 sounddevice** |
| 文件/流转码 | **ffmpeg**（`ffmpeg -i in.mp3 -ar 16000 -ac 1 -f s16le -`） | 事实标准；统一解码 MP3/M4A/FLAC/OGG/Opus → 16 kHz 单声道 PCM16。Qwen3-ASR-FastAPI 就是这么做的（"WAV/MP3/M4A/FLAC/OGG-Opus 自动转 16 kHz mono"） |
| 文件读写（Python） | **soundfile**（libsndfile） | 比 scipy.io.wavfile 稳，支持 flac/ogg |
| 移动端 | 平台原生（iOS AVAudioEngine / Android AudioRecord）+ Opus 编码 | 走平台最优路径 |

---

#### L1 传输（Transport）

| 场景 | 推荐协议 | 参数建议 | 理由 |
|---|---|---|---|
| 实时双向语音 | **WebRTC（Opus over RTP/SRTP）** | Opus 16–64 kbps，20 ms 帧；jitter buffer **20–30 ms**（远低于 VoIP 的 60–100 ms）；生产用 SFU 而非 P2P | 内建 FEC/PLC 抗丢包、NAT 穿透、拥塞控制；SFU 屏蔽最后一公里网络抖动，并让 SFU 节点（500 连接）与 GPU 节点（10 连接）独立扩容 |
| 服务端到服务端流式 | **WebSocket（PCM16 二进制帧）** | 16 kHz mono PCM16；客户端按 100–200 ms 分块发送 | 实现简单、无编解码开销；适合内网/可控网络 |
| 单向下发（LLM/TTS 结果） | **SSE（Server-Sent Events）** | — | 比 WebSocket 轻，天然兼容 HTTP 基础设施与重连 |
| 非实时/文件 | **HTTP multipart POST** | 加 size/duration 上限 | OpenAI 兼容契约，SDK 直接可用 |
| 中转/批处理 | **gRPC（Triton 8001）** | 配合 Triton dynamic batching | Triton 生态首选 |

**统一工程规范（强烈建议写进团队规范）**：
1. **所有模型输入统一 16 kHz、单声道、PCM16（int16 LE）**——ASR 模型几乎全部按 16 kHz 训练，重采样放在网关层做一次，而不是每个模型各做一次。
2. **降采样必须先低通滤波**，否则 8 kHz 以上分量会混叠进可听带，直接污染 ASR 输入。**推荐 `soxr`**（Centedge 工程实践明确点名），并按 SSRC/会话**每个流一个 resampler 实例**，避免滤波器状态串扰。
3. **立体声 → 单声道**用均值下混（`stereo.mean(axis=1)`）。
4. WebRTC 侧：**Opus 解码器按 SSRC 每路一个**（Opus 有状态），异常时重建解码器 + 返回一帧静音 + 跳过后续 3 包让流重新同步，避免一个坏包产出数秒噪音。

---

#### L2 VAD（Voice Activity Detection）

| 组件 | 部署位置 | 推荐参数 | 理由 |
|---|---|---|---|
| **Silero VAD**（首选） | 服务端 CPU / 浏览器 WASM | 输入**严格 512 样本 @16 kHz（32 ms）**；`positiveSpeechThreshold=0.3`、`negativeSpeechThreshold=0.25`、`silenceDuration=1400ms`、`minSpeechDuration=400ms`、`preSpeechPadDuration=800ms` | 轻量神经网络、CPU 上极快、精度好；ONNX 可导出，跨语言可用。工程实践：2 个连续语音块确认起始（~64 ms，够快且能滤掉咳嗽），20 个连续静音块（~640 ms）结束一句 |
| **FSMN-VAD**（FunASR） | 服务端（与 FunASR 链路配套） | `max_single_segment_time=30000`（ms）；开启动态 VAD | 阿里自研，与 Paraformer/SenseVoice/Qwen3-ASR 同栈；**动态 VAD 按累计时长自动收紧静音阈值**（≤5 s → 2.0 s，>45 s → 0.1 s），长句不会被切碎 |
| **WebRTC VAD** | 极资源受限场景 | 8/16/32 kHz，mode 0–3 | 传统 GMM，零依赖、极省 CPU；嘈杂环境鲁棒性差于神经网络方案 |
| **Ten VAD / 语义 VAD** | 需要"是否说完一句话"语义判断时 | 配合 LLM 判断 | 单纯能量/神经网络 VAD 无法区分"停顿思考"与"说完了" |

**关键工程点**：VAD 的尾点（endpointing）策略直接决定 ASR 延迟体感。**短句等长静音、长句快切**的动态阈值是被验证有效的模式。

---

#### L3 特征提取（Feature Extraction）

| 组件 | 用途 | 理由 |
|---|---|---|
| **torchaudio** | GPU 侧 mel-spectrogram / 重采样（`torchaudio.transforms.Resample`、`transforms.TimeStretch`） | 与 PyTorch 同栈、可 GPU 加速；vLLM-Omni 的 `speed` 变速就用的 `TimeStretch`（相位声码器，变速不变调） |
| **librosa** | 离线分析、数据准备、Mel/STFT 实验 | 生态最全、文档最好；生产推理路径一般不走它 |
| **kaldi fbank（C++）** | 边缘/GGUF 路径 | llama.cpp/GGUF runtime 内置 kaldi 兼容 80-mel fbank + LFR 前端，纯 C++ 无 Python 依赖 |
| **soxr** | 重采样（尤其 48 kHz → 16 kHz） | 抗混叠滤波质量最好，工程实践首选 |
| **numpy / scipy.signal** | 轻量后处理、RMS 窗检测（前导静音 trim） | Nari Labs 的动态 trim 就是短 RMS 窗检测持续语音起始点 |

**实践建议**：**特征提取尽量留在推理引擎内部**。vLLM / vLLM-Omni / llama.cpp 都已内置前端，业务层重复做一遍只会引入不一致。业务层只负责「统一采样率 + 声道 + 位深」。

---

#### L4 模型推理（Inference）

| 任务 | 首选引擎 | 备选 | 理由 |
|---|---|---|---|
| Qwen3-ASR（GPU 高并发） | **vLLM 主线**（`vllm[audio]`，cu129 nightly） | SGLang-Omni、vLLM-Ascend（昇腾） | day-0 支持、continuous batching、tensor parallel；128 并发可到 ~2000× 实时 |
| Qwen3-ASR（CPU/边缘） | **llama.cpp / GGUF**（Fun-ASR-Nano）或 SenseVoice GGUF | ONNX Runtime | 无 GPU/无 Python 场景；q8 编码器体积减半、CER 反而略降 |
| Qwen3-TTS | **vLLM-Omni**（在线 `/v1/audio/speech` + 离线 `end2end.py`） | SGLang-Omni、TensorRT Edge-LLM（Jetson） | 唯一同时覆盖在线流式 + 三种 query type + 音色管理的路径 |
| CosyVoice3 | **vLLM-Omni**（在线，流式默认开） | Triton + TensorRT-LLM（批量高吞吐）、MLX（Apple Silicon） | vLLM-Omni 一条命令起；Triton 路径在 L20 上 batch=16 RTF 0.0386 |
| SenseVoice（GPU 批量） | **Triton + TensorRT FP16** | FunASR AutoModel（PyTorch） | TensorRT 在 V100 上相对 CPU 基线有 526× 加速（官方 README 口径） |
| Apple Silicon | **MLX**（CosyVoice3 4/8-bit bundle） | — | 4-bit 仅 1.1 GB，MacBook Air 可跑 |

---

#### L5 服务化（Serving）

| 层 | 推荐组件 | 理由 |
|---|---|---|
| API 框架 | **FastAPI + Uvicorn**（自研封装）或直接吃引擎自带的 OpenAI 兼容服务 | 引擎已提供 OpenAI 兼容端点时**不要再造一层**；只在需要鉴权/配额/队列/多模型路由时才加 FastAPI 网关 |
| 容器 | **Docker + Docker Compose**（验证）→ **Kubernetes**（生产，ClusterIP + 探针 + 持久模型缓存） | FunASR 官方推荐路径；注意 CPU 示例镜像**不是** CUDA 生产镜像 |
| 引擎内建服务 | `vllm serve --omni`（vLLM-Omni）、`sgl-omni serve`（SGLang-Omni）、`tritonserver`（Triton）、`funasr-server`（FunASR）、`sensevoice-server`（C++ 单二进制） | 优先用引擎自带服务，少一层就少一跳延迟 |
| 音色/说话人管理 | vLLM-Omni `/v1/audio/voices` + `precompute_custom_voice.py` | 预计算音色可省掉每次请求的参考音频编码 |
| 批处理 | vLLM continuous batching；Triton dynamic batching；离线批量用 `end2end.py --use-batch-sample` + `qwen3_tts_batch.yaml` | 注意 `--batch-size` 必须匹配 CUDA Graph 捕获尺寸 |

---

#### L6 编排与排队（Orchestration）

| 组件 | 用途 |
|---|---|
| **asyncio + asyncio.Queue**（Python 单进程） | PCM 包入队、VAD 状态机驱动 utterance 切分；把音频包流转成"一段话" |
| **Celery / RQ / Arq + Redis** | 离线批处理任务队列、重试、结果持久化 |
| **NATS / Kafka / Redis Streams** | 多服务解耦（ASR 服务 ↔ LLM 服务 ↔ TTS 服务） |
| **Ray Serve / Triton Ensemble / vLLM-Omni stage pipeline** | 多阶段流水线编排（TTS 三阶段、ASR→LLM→TTS 全链路） |
| **SGLang-Omni Router / Nginx / Envoy / Kong** | 多副本负载均衡、健康检查、能力发现 |
| **背压与准入控制** | 必须显式实现：SGLang-Omni 在并发 64 时因 build backlog 溢出丢 1–4% 请求（HTTP 500）。推荐**有界队列 + 快速失败（429/503）+ 客户端指数退避**，而不是无限排队 |

---

#### L7 监控与可观测（Observability）

| 层 | 推荐组件 | 关键指标 |
|---|---|---|
| 指标 | **Prometheus + Grafana**；vLLM / SGLang / Triton 均自带 `/metrics` | TTFA p50/p95/p99、RTF、underrun 率、chars/s、req/s、排队时长、GPU 利用率、KV cache 命中率 |
| 链路追踪 | **OpenTelemetry**（每个阶段一个 span：VAD / ASR / LLM / TTS / 播放） | 语音链路的阶段耗时分解，定位瓶颈必备 |
| 日志 | 结构化 JSON 日志，每请求记录 **音频时长 / 模型 / 设备 / 延迟 / response_format / 错误类型** | FunASR 官方 readiness checklist 明确要求 |
| 成本 | 每会话 token/字符计数器 + 熔断器（LLM 异常时降级为预置话术） | Centedge 工程实践 |
| 语音链路专项 | （a）**underrun 率**（比 TTFA 更影响体感）；（b）**可听 TTFA**（trim 掉前导静音后的 TTFA）；（c）**barge-in 成功率** | Nari Labs 提的实时 TTS 四要素：低可听 TTFA / 零 underrun / 有容量 / 输出可懂 |

---

#### L8 评测（Evaluation）—— 详见 eval_toolchain 字段

---

#### 一条推荐的默认栈（可直接落地）

```
[浏览器] Web Audio + AudioWorklet → Opus 20ms 帧
   ↓ WebRTC (SFU: LiveKit / mediasoup / Pion)
[网关] Opus 解码 → soxr 48k→16k 低通重采样 → 下混单声道
   ↓
[VAD] Silero VAD (512样本/32ms, 2帧起 / 20帧止)
   ↓
[ASR] vLLM(Qwen3-ASR-1.7B)  /v1/audio/transcriptions
   ↓
[LLM] 自建 LLM 服务，流式 token
   ↓ 按 .!?;: 切句
[TTS] vLLM-Omni(Qwen3-TTS-1.7B) /v1/audio/speech stream=true pcm
   ↓ 客户端侧动态 trim 前导静音 → 播放
[编排] asyncio + Redis Streams；有界队列 + 429 快速失败
[监控] OTel span/阶段 + Prometheus(TTFA/RTF/underrun) + 每会话成本计数
[评测] jiwer(WER/CER) + WavLM/ECAPA(SIM) + UTMOSv2 + 人工 MOS 校准
```


### 29. 评测工具链

引擎层评测关注**延迟/吞吐/成本**三项，模型质量评测见另两个条目。这里给出可直接套用的方法学。

#### 1. 实时 TTS 的服务质量四要素（Nari Labs 定义，强烈推荐采纳）
1. **低可听 TTFA**（audible TTFA）：从请求发出到第一个**人耳能听到**的采样。必须是 trim 掉前导静音之后的测法，否则"吐 80 ms 静音再说话"会和"立刻说话"得分相同。
2. **零 underrun**：播放开始后客户端不能把缓冲放空。这比 TTFA 更能决定体感，却最常被漏测。
3. **容量（capacity）**：1 和 2 必须**随 RPS 上升仍成立**。只测 1 RPS 的基准没有意义。
4. **非畸形输出**：语音可懂（Nari 用 Deepgram STT 回听自动校验）。

#### 2. 压测方法学
- **流量模型用 Poisson 开环**（open-loop），而不是闭环固定并发——开环才会惩罚排队行为。（Nari Labs 遵循 Fireworks AI 的 LLM 压测方法；SGLang-Omni 用 closed-loop client，两者结论不完全可比，报告时必须注明。）
- **每种配置跑 5 分钟**（Nari 做法），SGLang-Omni 用「1 次丢弃的 warmup + 每并发级别 3 次重复」。
- **先 warm 再测**：vLLM 首次启动需初始化 KV cache 与 CUDA Graph，FunASR 可能还要从 `model.pt` 抽取权重，**首次启动通常 60–90 秒**；`/health` 返回 ok 之后必须再 warm 一轮才测容量。
- **并发扫 1 / 8 / 16 / 32 / 64**，观察拐点（SGLang-Omni 实测在 32 饱和、64 开始丢请求）。
- **把环境指纹写进结果**：GPU 型号、驱动版本、torch、引擎版本、模型 snapshot commit、依赖 freeze（`--fingerprint`）。
- **REST / SSE / WebSocket 分别压**，三者的连接模型与瓶颈不同。

#### 3. 指标定义
- **RTF（Real-Time Factor）= 处理时长 / 音频时长**，< 1 表示比实时快。
- **RTFx** = 1/RTF（SGLang-Omni 用它表示吞吐倍率，如 ZH 并发 64 时 RTFx 604.3）。
- **TTFA / TTFP / TTFB**：首包音频延迟，注意区分"原始"、"可听"。
- **chars/s**：TTS 的成本核算主指标（Nari 10 RPS 时约 630 chars/s）。
- **underrun 率**：出现播放断流的请求占比。

#### 4. 引擎无关的陷阱
1. **换 ASR 监听模型 = 换了一把尺子**：和 Seed-TTS 官方表对比时，绝不能换监听模型（EN 用 Whisper、ZH 用 Paraformer）。
2. **SIM 不能混用 embedding 网络**：WavLM 与 ECAPA-TDNN 的 speaker embedding 余弦相似度数值不可比，同一份 SIM 报告必须统一。
3. **MOS 预测器在高分区饱和**：UTMOS(v1) 在 4.5 分以上会压缩，且存在 **zoomed-in bias**（听测者倾向用满 5 级尺，中档系统被系统性压低）。模型已到 4.3+ 时，换 **UTMOSv2**（domain embedding + stacking 缓解）或改走 **MUSHRA / A-Test**。
4. **只看平均分看不出分布漂移**：UTMOSv2 均分不变但 TTSDS2 掉了，说明韵律多样性下降/口音塌成单模式。TTSDS2 是**分布级**指标（Frechet 距离），需两个目录各 20+ 条（论文建议 100+）wav。
5. **TTS Arena（主观 ELO）与客观分经常不一致**，例如 Kokoro 82M 主观排名很高。客观分用于 CI 回归，主观分用于对外宣称。

#### 5. 工具清单（引擎层相关部分）
| 工具 | 用途 |
|---|---|
| 自研 harness + Poisson 开环 | TTFA / underrun / RTF / chars/s（Nari Labs 的 harness 已 Apache-2.0 开源，可直接抄） |
| `vllm bench serve` / `benchmarks/eval/benchmark_asr_seedtts.py`（SGLang-Omni） | 引擎侧吞吐/延迟基准 |
| `batch_speech_client.py`（vLLM-Omni `examples/online_serving/qwen3_tts/`） | TTS 并发吞吐测量 |
| `client_grpc.py`（CosyVoice `runtime/triton_trtllm/`） | Triton gRPC 并发基准 |
| Prometheus + Grafana | 生产指标 |
| Deepgram STT / 任意 ASR | 合成音频可懂度自动回听校验 |

（模型质量侧的 WER/CER、SIM、UTMOS、TTSDS2、jiwer、Seed-TTS-eval、FLEURS 等，见 production_deployment.json 的 eval_toolchain 字段。）


### 30. 已知坑与限制

1. **vLLM-Omni 默认配置独占 GPU**（最高频坑）：默认给每个 stage 独立分配显存导致 GPU 被占满。解法：deploy YAML 里按 stage 设 `gpu_memory_utilization`（共享同一卡时**各 stage 之和 ≤ 1.0**），或用 `CUDA_VISIBLE_DEVICES` / `--stage-id --headless` 拆到不同卡。社区双 GPU 高并发配置 `qwen3_tts_high_concurrency.yaml`，据称优化后高并发音频吞吐 **+61.5%**、端到端 P99 延迟 **−49.4%**（**该组数字来自中文社区文章，未找到一手来源，标注待核实**）。
2. **Qwen3-TTS 流式默认 underrun 率 100%**：vLLM-Omni 默认 `async_chunk` 配置在 1 RPS 下就 100% 出现播放卡顿。必须调 `codec_chunk_frames` / `initial_codec_chunk_frames` 并做前导静音 trim 才能上生产。
3. **单客户端基准不可信**：1 RPS 与 6 RPS 的引擎排名会反转（VoxServe 49.3 ms → 363.2 ms）。**必须用 Poisson 开环流量压到目标 RPS。**
4. **vLLM / vLLM-Omni 版本必须对齐**：大版本号不一致会出现各种奇怪问题。vLLM-Omni 只跟偶数号上游 minor（0.22 / 0.24 / 0.26 / 0.28…）。
5. **vLLM-Omni 仅 Linux + Python 3.12**：macOS / Windows 不可用，也意味着 Apple Silicon 完全走不了这条路（改走 MLX）。
6. **ROCm 依赖冲突**：必须先 `pip uninstall onnxruntime` 才能装 `onnxruntime-rocm`（Qwen3-TTS 需要），另需 `sox`。
7. **批量解码的 CUDA Graph 尺寸约束**：`--batch-size` 必须是 1/2/4/8/16 等捕获尺寸；**且 stage 0 与 stage 1 都要设 `max_num_seqs >= batch_size`**，只调 stage 1 无效。
8. **并发过冲丢请求**：SGLang-Omni 默认 `max_running_requests=32` + `request_build_max_pending=16`，并发 64 时丢 1–4% 请求（HTTP 500）。必须在网关侧做准入控制。
9. **GPU 不是瓶颈，别盲目加卡**：SGLang-Omni 实测 Qwen3-ASR 全程 GPU 利用率仅 33%–54%，瓶颈在准入队列与 host 侧 decode 派发。加卡不如调 `max_running_requests` / build worker。
10. **Qwen3-ASR 流式不支持 batch 与时间戳**（官方明确），且时间戳要额外加载 `Qwen3-ForcedAligner-0.6B`。
11. **vLLM 4-bit 支持有限**：Qwen3-ASR-FastAPI 社区项目在 vLLM 0.14.0 上会自动从 4bit 回退到 BF16/FP16 并打日志。
12. **TensorRT plan 不可跨 GPU 架构 / TensorRT 版本移植**，换卡必须重建并重验精度、显存、容量。
13. **前导静音 ~80 ms**：不做动态 trim 的话，报出来的 TTFA 会显著好于用户实际体感。
14. **NVIDIA Dynamo 路径不支持 streaming 与 Base（voice cloning）**，需要克隆能力时别选这条。
15. **Ollama 跑不了这三款模型**（只支持 GGUF 且不支持音频 I/O），遇到相关说法请忽略。
16. **长音频截断**：`--max-model-len` / `max_new_tokens` 必须与训练/微调时的 `--max_length` 对齐，否则静默截断；动态音频长度导致 CUDA Graph 编译失败时需 `--enforce-eager`。


### 31. 社区与文档

| 项目 | 许可 | 活跃度 | 文档入口 | 备注 |
|---|---|---|---|---|
| vLLM | Apache-2.0 | 极高（事实标准） | docs.vllm.ai；recipes 仓 `vllm-project/recipes`（含 Qwen3-ASR 专页） | Qwen3-ASR day-0 |
| vLLM-Omni | Apache-2.0 | 高（2,741 commits，26 tags，184 branches；2026-08-28 仍有提交） | vllm-omni.readthedocs.io；Slack `#sig-omni` @ slack.vllm.ai；每周二社区例会 | 论文 arXiv:2602.02204 |
| SGLang-Omni | Apache-2.0 | 高（约 933 stars / 371 forks，2026-08 累计 732 commits；**118 open issues，早期采用者需预期不稳定**） | sgl-project.github.io/sglang-omni/ | v0.1.2 在 PyPI |
| TensorRT Edge-LLM | NVIDIA 许可 | 中 | nvidia.github.io/TensorRT-Edge-LLM/ | v0.9.0 |
| NVIDIA Dynamo | 开源 | 中 | docs.nvidia.com/dynamo/ | 以 vLLM-Omni 为 TTS worker |

**文档质量提示**：vLLM-Omni 文档在快速完善中，部分边角功能（尤其是 stage config 的字段语义）需要直接读 `vllm_omni/deploy/*.yaml` 与源码。选型决策前建议对照仓库 README 的 release notes 确认当前版本对齐的 vLLM 版本。


### 32. 适用场景与选型建议

#### 决策树

**Q1：你要跑 ASR 还是 TTS？**
- **ASR（Qwen3-ASR）** → **vLLM 主线**。理由：day-0 支持、continuous batching、tensor parallel、OpenAI 兼容三端点齐全、国产卡（昇腾 vllm-ascend / 海光 DTK wheel）都有官方或半官方镜像。**不需要 vLLM-Omni。**
  - 例外 A：需要 CPU / 边缘 / 无 Python → 走 FunASR + llama.cpp/GGUF（见 funasr_sensevoice.json）。
  - 例外 B：已深度使用 SGLang 生态、或想要 Router + barge-in 的一体化体验 → SGLang-Omni（但注意其 Qwen3-ASR 吞吐目前显著低于自家的固定 baseline，见 latency_throughput B 段）。
  - 例外 C：Jetson / 车载 → TensorRT Edge-LLM（FP16，Qwen3-ASR-0.6B/1.7B 官方支持）。

**Q2：TTS 要在线服务还是离线批处理？**
- **在线（绝大多数生产场景）** → **vLLM-Omni**。它现在是唯一同时提供 `/v1/audio/speech`、流式 PCM、WebSocket 增量流式、`/v1/audio/voices` 音色管理、预计算音色、Gradio/FastRTC 演示的组合，并且 Qwen3-TTS 与 CosyVoice3 走同一套 API。
- **离线批处理（有声书、批量配音）** → vLLM-Omni `end2end.py --use-batch-sample` + `qwen3_tts_batch.yaml`（注意 CUDA Graph 尺寸约束）；或 CosyVoice 的 Triton+TRT-LLM runtime（L20 上 batch=16 RTF 0.0386，吞吐最高）。

**Q3：硬件是什么？**
- NVIDIA 数据中心 GPU（H100/A100/L20/L40S）→ vLLM-Omni（或 Triton+TRT-LLM 做极致批量吞吐）。
- 昇腾 910B → `quay.io/ascend/vllm-omni:v0.14.0`，docker-compose 一键起。
- 海光 DCU K100_AI → vLLM 0.18.1 + vLLM-Omni 0.18.0 组合（官方镜像已验证）。
- Jetson Thor / Orin → TensorRT Edge-LLM（Qwen3-TTS CustomVoice，FP16，`--maxBatchSize 1`，`--chunkFrames=10` 流式）。
- Apple Silicon → **MLX**（不在这三个引擎的能力范围内，见 production_deployment.json）。
- AMD ROCm / Intel XPU / MUSA → vLLM-Omni（需额外依赖；XPU 在 SGLang-Omni 上是实验状态）。

**Q4：延迟预算是多少？**
- 需要 **sub-50 ms p95 可听 TTFA 且 10 RPS 以上**：现成引擎目前做不到，需参考 Nari Labs 的做法自建调度（Talker / Code Predictor / Codec 三任务统一调度 + deadline-aware 优先级 + 前导静音 trim + 帧累积 ramp），其实现与基准已 Apache-2.0 开源。
- **p95 50–100 ms、6 RPS 以内**：vLLM-Omni 调优后可达（56.8 / 93.5 ms）。
- **150–300 ms 可接受**：vLLM-Omni 默认配置 + 前导静音 trim 即可，或用 SGLang-Omni。

**Q5：要不要上 Triton？**
- 团队**已经在运营 Triton / GPU serving**，且场景是**批量、固定 GPU 型号、固定音色** → 上（CosyVoice Triton+TRT-LLM 有完整的 build/serve/bench 脚本与 `use_spk2info_cache` 优化）。
- 追求快速上线、需要流式、需要频繁换模型 → **不要上 Triton**，vLLM-Omni 的迭代速度和 API 完备度更优。

#### 一句话结论
- **Qwen3-ASR → vLLM 主线 + `vllm[audio]`（cu129）；国产卡走 vllm-ascend / 海光 DTK wheel；边缘走 FunASR GGUF。**
- **Qwen3-TTS → vLLM-Omni 在线服务（`/v1/audio/speech` + 流式 PCM/WebSocket）；低延迟极致场景参考 Nari Labs 自建调度；Jetson 走 TensorRT Edge-LLM。**
- **CosyVoice3 → vLLM-Omni 在线流式；批量高吞吐走官方 Triton + TensorRT-LLM runtime；Apple Silicon 走 MLX。**
- **SGLang-Omni 作为第二选项/对冲**：模型覆盖广、Router 与 barge-in 好用、Apache-2.0 且迭代快，但项目仍早期（118 open issues），且当前 Qwen3-ASR 吞吐落后于其对照基线 1.4–3.3 倍。


### 33. 参考来源

**vLLM / Qwen3-ASR**
- vLLM Recipes — Qwen3-ASR 使用指南（官方）：https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3-ASR.md
- vLLM 文档（中文镜像）：https://docs.vllm.com.cn/projects/recipes/en/latest/Qwen/Qwen3-ASR.html
- QwenLM/Qwen3-ASR 官方仓库（vLLM backend、streaming、ForcedAligner、Docker）：https://github.com/QwenLM/Qwen3-ASR
- vLLM 0.27.1 release（cu129 wheel）：https://github.com/vllm-project/vllm/releases/tag/v0.27.1

**vLLM-Omni**
- 仓库：https://github.com/vllm-project/vllm-omni
- 文档：https://vllm-omni.readthedocs.io/en/latest/
- 在线 TTS（含 Qwen3-TTS 全部用法）：https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/examples/online_serving/text_to_speech.md
- 离线 Qwen3-TTS：https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts
- GPU 显存计算与配置：https://docs.vllm.tw/projects/vllm-omni/en/latest/configuration/gpu_memory_utilization
- 论文：vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models, arXiv:2602.02204 — https://arxiv.org/abs/2602.02204
- 社区 Slack：#sig-omni @ slack.vllm.ai

**SGLang-Omni**
- 文档首页（支持模型矩阵）：https://sgl-project.github.io/sglang-omni/
- 仓库：https://github.com/sgl-project/sglang-omni
- Qwen3-ASR 高并发基准与瓶颈剖面（H100 80GB 全套数据）：https://sgl-project.github.io/sglang-omni/developer_reference/qwen3_asr_concurrency_profile.html

**NVIDIA**
- TensorRT Edge-LLM supported models（Qwen3-ASR / Qwen3-TTS FP16）：https://nvidia.github.io/TensorRT-Edge-LLM/0.6.0/user_guide/getting_started/supported-models.html
- TensorRT Edge-LLM — TTS（Qwen3-TTS 导出/建引擎/流式推理）：https://nvidia.github.io/TensorRT-Edge-LLM/0.9.0/user_guide/examples/tts.html
- NVIDIA Dynamo — Text-to-Audio（vLLM-Omni worker + Qwen3-TTS）：https://docs.nvidia.com/dynamo/diffusion/text-to-audio
- Getting Started with TensorRT Edge-LLM on Jetson Thor：https://www.hackster.io/shahizat/getting-started-with-nvidia-tensorrt-edge-llm-on-jetson-thor-14735e

**基准与调优**
- Nari Labs — How We Made a Text-to-Speech Model Respond in Sub-50 ms（2026-08-19，含四引擎对比与 5 项优化）：https://www.worldprogramming.org/posts/how-we-made-a-text-to-speech-model-respond-in-sub-50-ms-8ttfmg
- 同上（镜像/解读）：https://aibacon.net/posts/nari-labs-qwen3-tts-serving-stack-sub-50ms-ttfa-single-h100.html
- ExplainX 解读（含成本模型）：https://explainx.ai/blog/nari-labs-qwen3-tts-speed-cost-frontier-august-2026

**API 模式与参数（社区整理，需以官方为准）**
- lattifai README（audio tok/s 表 + 三种 API mode + 启动命令）：https://pypi.org/project/lattifai/
- Qwen3-ASR-FastAPI（OpenAI 兼容封装、量化模式、性能档位）：https://github.com/groxaxo/Qwen3-ASR-FastAPI


---

<!-- item: funasr_sensevoice -->

## ASR 生态：FunASR / SenseVoice

> **分类**：技术栈-ASR生态　|　**已填字段**：29 / 38　|　**⭐ 标记为本轮重点的架构与训练字段**


### 1. 基本信息与版本演进

**FunASR** 是阿里巴巴通义实验室（原 DAMO 语音实验室）开源的语音识别**基础设施/工具包**，不是一个模型。定位（官方首页原话）："Private-deployment speech infrastructure for high-throughput GPU inference, realtime streaming, OpenAI-compatible APIs, and standalone CPU or edge use."

- **协议**：Apache-2.0；**API**：OpenAI 兼容；**验证日期**：2026-08-13（官方部署中心标注）。
- **作用**：把 ASR、VAD、标点恢复、说话人分离/验证、情感识别、音频事件检测、热词、时间戳统一到一个 `AutoModel` 接口下，并提供从 Colab → Python API → 服务化 → 容器/K8s → 边缘二进制的完整部署阶梯。
- **模型家族**（同属 FunAudioLLM 组织，但各有独立仓库）：
  - **Paraformer**（非自回归，中文生产级，`paraformer-zh` / `paraformer-zh-streaming` / `paraformer-large`）
  - **SenseVoice**（`iic/SenseVoiceSmall`，非自回归多任务：ASR + LID + SER + AED）
  - **Fun-ASR / Fun-ASR-Nano**（`FunAudioLLM/Fun-ASR-Nano-2512`，LLM-ASR，31 语种；另有 `Fun-ASR-MLT-Nano-2512` 多语 checkpoint）
  - **Qwen3-ASR**（`Qwen/Qwen3-ASR-1.7B`，52 语种/方言，FunASR 1.3.x 起可通过 `AutoModel` 直接调用）
  - **辅助模型**：`fsmn-vad`（VAD）、`ct-punc`（中英标点）、`cam++`（说话人验证/分离）、`emotion2vec_plus_large`（专用情感模型）
  - **FunClip**：基于 FunASR 的智能视频剪辑工具。
- **SenseVoiceSmall 架构**：**SAN-M encoder（70 层）+ CTC head**，没有 LLM、没有自回归。特征为 kaldi 80 维 mel fbank + LFR（低帧率拼接）→ `[T, 560]`，前面拼 4 个 query token（`lang` / `event` / `emotion` / `itn`）→ SAN-M encoder → `[4+T, 512]` → CTC head（Linear 512→25055）→ greedy CTC（argmax + 去重 + 去 blank）→ SentencePiece detok。
- **Qwen3-ASR 与 FunASR 的关系**：Qwen3-ASR 是独立模型（QwenLM 仓库、`qwen-asr` pip 包），但 FunASR 1.3.x 起把它作为一个可选项纳入 `AutoModel`，形成「FunASR 是调度层 + 模型可插拔」的格局。二者不是替代关系。


### 2. 开源协议与商用边界

- **FunASR**：**Apache-2.0**（官方首页明确标注），可商用、可修改、可私有化部署。
- **SenseVoiceSmall**：Apache-2.0（`FunAudioLLM/SenseVoice` 仓库）。
- **Paraformer / Fun-ASR-Nano**：Apache-2.0。
- **Qwen3-ASR**：Apache-2.0。
- **GGUF 生态**：FunASR 的 GGUF 运行时（llama.cpp / audio.cpp）与权重同为开源可商用（官方原话"全家桶开源、可商用"）。
- **权重获取渠道**：ModelScope（国内默认，快）/ HuggingFace（`hub="hf"` 切换）。生产建议 `disable_update=True` + 本地路径，避免每次启动校验。
- **注意**：FunASR 工具链里部分第三方依赖（如 WeTextProcessing / pynini、部分 ONNX 量化工具）有各自的许可，私有化打包前应做一次依赖审计。


### 3. 模型变体与权重获取

#### 官方模型选择表（FunASR 教程原文）
| 模型 | 适用场景 | 支持语言 | 速度 | 标点 |
|---|---|---|---|---|
| **Paraformer** | 中文生产级识别 | 中文、英文 | 快 | 需要 `punc_model` |
| **Fun-ASR-Nano** | 多语言、方言、歌词 | 31 种语言 | 中等 | **内置 ✓** |
| **SenseVoice** | 情感 + 音频事件 + 识别 | 5 种语言 | 极快（10 s 音频仅需 70 ms） | **内置 ✓** |
| **Qwen3-ASR** | 最高精度，上下文理解 | 52 种语言 | 较慢（大模型） | **内置 ✓** |
| **Paraformer-Streaming** | 实时转写 | 中文 | 实时 | 需要 `punc_model` |

**官方快速推荐**：
- 中文会议/通话 → `paraformer-zh` + `fsmn-vad` + `ct-punc` + `cam++`
- 多语言 → `Fun-ASR-Nano`
- 需要情感/音频事件检测 → `SenseVoice`
- 实时字幕 → `paraformer-zh-streaming`
- 最高质量、不在乎延迟 → `Qwen3-ASR`

#### 各模型的 ModelScope / HF ID
- `paraformer-zh`、`paraformer-zh-streaming`、`paraformer`（large）
- `fsmn-vad`、`ct-punc`、`cam++`
- `iic/SenseVoiceSmall`、`iic/emotion2vec_plus_large`
- `FunAudioLLM/Fun-ASR-Nano-2512`（需 `trust_remote_code=True, remote_code="./model.py"`）
- `Qwen/Qwen3-ASR-1.7B`（需先 `pip install qwen-asr`）
- GGUF：`FunAudioLLM/funasr`（f16 / f32 / q8 + 嵌入 vocab）；SenseVoice 原生 server 用 `sensevoice-small-q8.gguf` + `fsmn-vad.gguf`
- vLLM 社区转换 checkpoint：`allendou/Fun-ASR-Nano-2512-vllm@e718b36e`

#### 导出格式
- **ONNX**：`model.export(type="onnx", quantize=False)`，opset 14，dynamic axes；运行时 `funasr-onnx`。
- **Libtorch**：`model.export(type="torch")`，运行时 `funasr-torch`。
- **GGUF**：`export_sensevoice_gguf.py`（f32 ~936 MB / f16 ~470 MB / q8 ~254 MB）。


### 4. 架构范式 ⭐

#### 范式分野：非自回归（NAR）vs LLM 自回归（AR）

这是整个 ASR 生态最重要的一条分界线，也直接决定引擎选型。

| | **NAR 路线**（SenseVoice / Paraformer） | **AR 路线**（Qwen3-ASR / Fun-ASR-Nano） |
|---|---|---|
| 解码方式 | 一次前向并行出全部 token（CTC / 非自回归 transformer） | 逐 token 自回归解码 |
| 延迟特征 | **确定性物理时延**，与文本长度几乎无关 | 随输出长度线性增长 |
| 吞吐 | 极高（10 s 音频 70 ms） | 靠 continuous batching 与并发换吞吐 |
| **能否用 vLLM** | **不能**（不走 KV-cache 解码路径） | **能**（这是 vLLM 加速的前提） |
| 显存 | 小（几百 MB 级） | 大（GB 级 + KV cache） |
| CPU / 边缘 | **可**（ONNX / GGUF / C++） | 极受限（1.7B 不推荐 CPU 解码） |
| 上下文理解 / 热词 | 靠 WFST 热词（外部偏置） | 靠 prompt（上下文偏置，天然支持） |
| 语种/方言广度 | 窄（SenseVoice 公开 ckpt 仅 5 语种） | 广（Qwen3-ASR 52 语种/22 方言） |

**FunASR 官方 vLLM 文档的适用模型表**（这是最权威的一条判定标准）：

| 模型系列 | vLLM 支持 | 说明 |
|---|---|---|
| **FunASRNano** | ✅ | Audio encoder + adaptor + Qwen3-0.6B LLM |
| **LLMASR / LLMASRNAR** | ✅ | Whisper 风格音频编码器 + Qwen / Vicuna / LLaMA 解码器 |
| **GLMASR** | ✅ | GLM-ASR-Nano 使用自回归 LLM 解码 |
| **QwenAudioWarp** | ✅ | LLM-based audio generation path |
| **Paraformer / SenseVoice / Conformer / Transformer** | ❌ | 非 LLM 自回归解码路径，请用标准 `AutoModel` |

→ **结论：SenseVoice 与 Paraformer 永远不要试图用 vLLM 加速，那是架构层面的错配。**

#### SenseVoice 的多任务范式
SenseVoice 的创新在于**把 LID（语种识别）、SER（情感识别）、AED（音频事件检测）、ITN（逆文本正则化）全部编码成 prompt 里的 query token**，与 ASR 共享一次 encoder 前向：
- 输入序列 = `[lang_query, event_query, emotion_query, itn_query] + 音频特征`
- 输出形如：`<|zh|><|HAPPY|><|Speech|><|withitn|>今天真是太开心了。`
- 用 `rich_transcription_postprocess()` 提取干净文本。
- **代价**：公开 checkpoint 只覆盖 **5 个语种**（中/粤/英/日/韩），与论文中"40 万小时、50+ 语言"的训练描述不是一回事——**选型时以 checkpoint 实际能力为准**。


### 5. 模块拆解 ⭐

#### SenseVoiceSmall 结构（llama.cpp runtime 文档可核）
```
audio.wav (16k mono)
   │ kaldi 80-mel fbank + LFR (C++)
   ▼
features [T, 560]
   │ prepend 4 query tokens [lang, event, emotion, itn]
   ▼
[4 + T, 560]
   │ SAN-M encoder (ggml) ── sensevoice-small.gguf
   ▼
encoder out [4+T, 512]
   │ CTC head (Linear 512→25055) → greedy CTC (argmax, dedup, drop blank)
   ▼
token ids
   │ SentencePiece detok (detok.py)
   ▼
transcription
```
- **参数量约 234M**（与 Whisper-Small 相当）；**无 CMVN**（推理时不接 CMVN，这是 llama.cpp 移植时特别点出的 gotcha）。
- CTC 词表 25055；tokenizer 为 `chn_jpn_yue_eng_ko_spectok.bpe.model`（SenseVoice Triton 路径需手工下载并校验 SHA-256 `aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8`）。
- SAN-M encoder 与 Fun-ASR-Nano 的 encoder 同架构，因此 ggml forward 在两个 runtime 间共享。

#### Fun-ASR-Nano 结构
Audio encoder（同 SAN-M）+ adaptor + **Qwen3-0.6B LLM** 解码器 → 所以它能吃 vLLM 的 PagedAttention / continuous batching / tensor parallel。

#### FunASR 工具链模块清单
| 模块 | 模型 ID | 职责 | 输出 |
|---|---|---|---|
| VAD | `fsmn-vad` | 语音切分 | `[[start_ms, end_ms], ...]` |
| ASR | paraformer / SenseVoice / Fun-ASR-Nano / Qwen3-ASR | 转写 | `text` |
| 标点 | `ct-punc` | 中英双语标点恢复 | 带标点文本 |
| 说话人 | `cam++` | 说话人验证/分离 | `sentence_info[].spk` |
| 情感 | SenseVoice 内置 / `emotion2vec_plus_large` | 情感标签 + 分数 | `['angry','happy','neutral','sad',...]` |
| ITN | 模型内置 / `fst_itn_zh` | 数字、日期、单位规整 | 规范化文本 |

**一个关键组合约束**：Paraformer 需要 `punc_model` 才能做句子切分（进而才能上 `spk_model`）；而 **Fun-ASR-Nano、SenseVoice、Qwen3-ASR 自带标点输出，不需要 `punc_model`**——但说话人分离仍然需要 `spk_model="cam++"`。


### 6. 音频编码器 ⭐

- **SenseVoice / Fun-ASR-Nano / Paraformer**：kaldi 风格 **80 维 mel fbank + LFR**（低帧率拼接后每帧 560 维）。三者的前端基本一致，这也是 GGUF runtime 能共享 encoder forward 的原因。
- **Qwen3-ASR**：Whisper 风格的 audio encoder + adaptor，输出 audio token 后交给 LLM 解码。**audio token rate 约 25 tok/s**（社区整理数据，来源 lattifai README，**待一手核实**），且**没有 Whisper 那种固定 30 s 窗口限制**——更长的音频反而识别更准（社区反馈最长支持约 20 分钟，待核实）。
- **输入约定**：全部模型统一 **16 kHz 单声道**。FunASR 直接吃 numpy array（`model.generate(input=audio)`），不需要先落盘。
- **前端归属**：在 FunASR 的 vLLM 引擎里，**frontend / encoder / adaptor / CTC 时间戳解码器仍在 PyTorch 中运行，只有 LLM 解码部分通过 prompt embedding 交给 vLLM**。这意味着 encoder 的设备（`device="cuda:0"`）与 vLLM 的 `gpu_memory_utilization` 是**两套独立预算**，配置时要分别留。


### 7. 语音 Tokenizer 与编解码器 ⭐

- **SenseVoice**：SentencePiece BPE（`chn_jpn_yue_eng_ko_spectok.bpe.model`），CTC 词表 **25055**。无神经 codec——它是纯 encoder + CTC，不做音频离散化重建。
- **Paraformer**：中文字符/子词词表（`vocab8404` 版本常见）。
- **Fun-ASR-Nano / Qwen3-ASR**：沿用 Qwen 系列 BPE 词表，与文本 LLM 共享——这也是它们能直接接 vLLM 的原因之一。
- **工程提示**：SenseVoice 的 Triton / GGUF 部署路径**必须单独下载并校验 tokenizer 模型的 SHA-256**，路径或版本错了会静默产出乱码（不是报错）。


### 8. LLM 主干 ⭐

- **SenseVoice / Paraformer**：**没有 LLM backbone**（纯 encoder + CTC / NAR decoder）。这是它们快、小、可上 CPU 的根本原因，也是它们**无法享受 vLLM continuous batching** 的根本原因。
- **Fun-ASR-Nano**：**Qwen3-0.6B** 自回归解码器 + 音频 encoder/adaptor。首次运行时 FunASR 会从 `model.pt` 抽取 LLM 权重到 vLLM 兼容目录（如 `Qwen3-0.6B-vllm`），后续复用。
- **Qwen3-ASR**：Qwen3 系列（0.6B / 1.7B）自回归解码器，音频 encoder 提供 prompt embedding。
- **采样**：`temperature=0.0`（默认贪心）、`repetition_penalty`、`max_new_tokens` 默认 512。


### 9. 上下文长度与 I/O 约束

- **输入**：文件路径、URL、`numpy array`、`torch.Tensor`、`wav.scp`、JSONL（FunASR `generate()` 全支持）；音频/视频文件均可（Qwen3-ASR 社区封装支持视频容器抽轨）。
- **采样率**：统一 **16 kHz**。非 16 kHz 由 FunASR/pyffmpeg 自动重采样，但生产建议**在网关层统一做好**（见 production_deployment.json）。
- **最大音频时长**：由 VAD 的 `max_single_segment_time` 控制（默认 30000 ms；Paraformer 场景常用 60000 ms）。SenseVoice 原生 C++ server 提供 `--max-audio-seconds 300` 硬限制。
- **批量控制**：`batch_size_s`（按**时间**动态分批，推荐）与 `batch_size`（按**条数**）；OOM 三板斧是「减小 `batch_size_s`（300→60）、减小 `max_single_segment_time`（30000→15000）、加 `batch_size_threshold_s=30`」。
- **输出格式**：OpenAI 兼容 API 支持 `json` / `text` / `srt` / `vtt` / `verbose_json`（`verbose_json` 返回分段、时间戳、说话人；FunASR 1.3.27 起在 `verbose_json.language` 返回检测语种）。
- **长音频**：Paraformer 走 VAD 切分 + 逐段识别 + 标点；Qwen3-ASR 整体送入（但受 `max_new_tokens` / `max_model_len` 限制）。


### 10. 微调支持

- FunASR 提供官方训练入口（教程站 training 页）与 `examples/industrial_data_pretraining/fun_asr_nano` 等工业数据示例目录。
- **热词/偏置优先于微调**：生产上"专业术语识别不准"的第一解法是**热词**，不是微调——热词零成本、可在线更新、可回滚。只有在热词不够（例如领域口音、特定句式）时才考虑微调。
- 微调后部署路径：`AutoModel(model="/path/to/local/model", disable_update=True)`，或直接 `vllm serve ./checkpoints/xxx`（仅 AR 模型）。


### 11. 语种与方言覆盖

| 模型 | 语种覆盖 | 备注 |
|---|---|---|
| **SenseVoiceSmall** | **5 种**（中 / 粤 / 英 / 日 / 韩） | 公开 checkpoint 的实际能力；**论文描述的 50+ 语言不等于可用能力** |
| **Paraformer** | 中文、英文 | 中文生产级；另有 large 版 |
| **Fun-ASR-Nano-2512** | **31 种** | 方言 + 歌词识别是强项；多语种 MLT-Nano 是独立 checkpoint |
| **Qwen3-ASR** | **52 种**（30 语言 + 22 中文方言） | 覆盖最广；SGLang-Omni 文档标注"30 language hints" |

**语种相关能力矩阵（社区整理，含 FunASR 加持能力）**：
| 功能 | Paraformer | SenseVoice-Small | Fun-ASR-Nano | FireRedASR v2 | Qwen3-ASR |
|---|---|---|---|---|---|
| 中文识别 | ✓ | ✓ | ✓ | ✓ | ✓ |
| **流式识别** | ✓ | ✗ | ✓ | ✗ | ✓ |
| **情感识别** | ✗ | **✓** | ✗ | ✗ | ✗ |
| **音频事件检测** | ✗ | **✓** | ✗ | ✗ | ✗ |
| 语种识别 | ✗ | ✓ | ✗ | ✓ | ✓ |
| 时间戳 | ✓ | ✓ | ✗ | ✓ | ✓ |
| 歌词识别 | ✗ | ✗ | ✓ | ✓ | ✓（歌唱 + BGM 强项） |
| 标点恢复 | ✗ | ✓ | ✓ | ✓ | ✓ |

**注意**：Paraformer / SenseVoiceSmall / Fun-ASR-Nano 三者都可以通过 FunASR 工具包**外挂** VAD（fsmn-vad）、标点（ct-punc）、说话人分离（cam++）、热词、情感（emotion2vec）来补齐上表中打 ✗ 的格子——表格反映的是"模型自身能力"，不是"工具链能力"。


### 12. 流式能力

#### 1. 传统双引擎打法（FunASR 原生）
**在线模型出实时结果 + 离线模型出精修终稿**：
```
paraformer-zh-streaming（每 600 ms 出增量）
   ↓ VAD 尾点检测
paraformer-zh（离线大模型） + ct-punc（标点） → 终稿
```
流式调用要点（`paraformer-zh-streaming`）：
```
model = AutoModel(model="paraformer-zh-streaming")
chunk_size = [0, 10, 5]                  # [未使用, 显示粒度(×60ms), 前瞻(×60ms)] → 600ms 显示 + 300ms 前瞻
chunk_stride = chunk_size[1] * 960       # 每块 9600 采样点
cache = {}                               # 必须在所有 chunk 间保持，不能重建
for i, chunk in enumerate(chunks):
    res = model.generate(input=chunk, cache=cache, is_final=(i==last),
                         chunk_size=chunk_size,
                         encoder_chunk_look_back=4, decoder_chunk_look_back=1)
```
- **最后一个 chunk 必须 `is_final=True`**，否则尾部缓存文字不吐出。
- FunASR 另提供 `run_server_2pass.sh` 的 **two-pass 服务**（`--model-dir` 离线 + `--online-model-dir` 在线 + `--punc-dir` + `--itn-dir`），是这套打法的开箱即用版本。

#### 2. 单模型统一流式打法（新一代）
**Qwen3-ASR / FireRedASR2-S** 是一个模型同时支持流式与离线，不需要双引擎。
- Qwen3-ASR 的流式**仅 vLLM 后端支持**（官方明确），且**不支持 batch 推理、不返回时间戳**。
- 官方流式 demo：`qwen-asr-demo-streaming --asr-model-path Qwen/Qwen3-ASR-1.7B --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8000`。
- **代价**：Uni-ASR 论文（arXiv:2603.11123）指出 Qwen3-ASR 的流式是"迭代重复的非流式解码"，相比原生流式架构计算开销显著更高——**流式 QPS 敏感场景要据此评估成本**。

#### 3. FunASR 的 vLLM 流式（两种入口）
- **流式 SDK**：`FunASRNanoStreamingVLLM`，`chunk_ms=720`、`rollback_chars=8`；把音频切成 720 ms 块累积重编码，各 chunk prompt 批量送 vLLM，返回 **fixed / unfixed 文本区域**。行为细节：Stage 1 前 10 个 chunk 不用 `prev_text`（用于找稳定前缀）；Stage 2 起用稳定前缀做 assistant context；最后 `rollback_chars` 个字符保持未确认等待修正；**前 1.5–3 秒可能为空或不稳定（模型特性，不是 bug）**。
- **生产 WebSocket 实时服务**：`serve_realtime_ws.py`（推荐）。集成动态 VAD 分句、vLLM 分段解码、**partial 预览**（`--decode-interval` 默认 0.48 s）、幻觉重复清理、热词、语种提示、说话人分离。协议：
  ```
  客户端 → 服务端: "START" / "HOTWORDS:词1,词2" / "LANGUAGE:中文" / [PCM16 16kHz 单声道二进制] / "STOP"
  服务端 → 客户端: {"event":"started"} / {"sentences":[...], "partial":"...", "is_final":false} / {"event":"stopped"}
  ```
  启动：`CUDA_VISIBLE_DEVICES=0,1 python serve_realtime_ws.py --port 10095 --tensor-parallel-size 2 --language 中文`

#### 4. 动态 VAD（FunASR 1.3.x 新增，强烈推荐）
**短句等长静音、长句快切**，避免把话切碎或让 ASR 输入过长：

| 累计语音时长 | 流式封装默认阈值 | fsmn-vad 原生默认阈值 |
|---|---|---|
| ≤ 5 s | 2.0 s | 800 ms |
| 5–10 s | 1.5 s | 600 ms |
| 10–15 s | 1.0 s | 500 ms |
| 15–30 s | 0.8 s | 400 ms |
| 30–45 s | 0.4 s | 300 ms |
| > 45 s | 0.1 s（强制切分） | — |
自定义：`silence_schedule=[(5000,1000),(15000,500),(float("inf"),200)]`；关闭：`dynamic_silence=False`；独立使用：`DynamicStreamingVAD(AutoModel(model="fsmn-vad"))` + `.feed(chunk)` / `.finalize()`。


### 13. VAD 与后处理

#### VAD（fsmn-vad）
```
# 离线
model = AutoModel(model="fsmn-vad")
res = model.generate(input="audio.wav")
# res[0]["value"] = [[610, 5530], [7200, 12400], ...] 每对 [起始ms, 结束ms]

# 流式（200 ms 每块）
chunk_stride = int(200 * sr / 1000)
res = model.generate(input=chunk, cache=cache, is_final=..., chunk_size=200)
# [[610, -1]] 语音在 610ms 开始；[[-1, 5530]] 语音在 5530ms 结束；[[610, 5530]] 完整段
```
关键参数：`vad_kwargs={"max_single_segment_time": 30000}`（默认 30000 ms；Paraformer 长音频常用 60000）。

#### 标点恢复（ct-punc）
```
model = AutoModel(model="ct-punc")
res = model.generate(input="那今天的会就到这里吧 happy new year 明年见")
# → "那今天的会就到这里吧，happy new year，明年见。"
```
**何时需要**：只有 **Paraformer** 需要（它输出无标点纯文本）。**Fun-ASR-Nano / SenseVoice / Qwen3-ASR 自带标点输出**。

#### 说话人分离（cam++）
```
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad",
                  punc_model="ct-punc", spk_model="cam++")
res = model.generate(input="meeting.wav", batch_size_s=300)
for sent in res[0]["sentence_info"]:
    print(f"[说话人 {sent['spk']}] [{sent['start']}-{sent['end']}ms] {sent['text']}")
```
支持 Paraformer / Fun-ASR-Nano / SenseVoice 三者；后两者**不需要 `punc_model`**。

#### 情感检测
- **SenseVoice 一次推理同时出识别 + 情感**：原始输出 `<|zh|><|HAPPY|><|Speech|>你好`，用 `rich_transcription_postprocess()` 提取干净文本。
- **专用情感模型**：`iic/emotion2vec_plus_large`，`granularity="utterance"` 或帧级，输出 labels + scores（如 neutral 0.89）。

#### ITN（逆文本正则化）
`use_itn=True`（SenseVoice）/ `itn=True`（vLLM 引擎）；把"二零二六年八月"规整为"2026年8月"。FunASR runtime 里对应 `--itn-dir thuduj12/fst_itn_zh`。

#### 时间戳
- **Paraformer**：`res[0]["timestamp"]` = `[[880,1120],[1120,1360],...]`（每字毫秒）。
- **Fun-ASR-Nano**：`res[0]["timestamps"]` = `[{"token":"开","start_time":0.42,"end_time":0.48}, ...]`。
- **Qwen3-ASR**：需额外加载 `Qwen/Qwen3-ForcedAligner-0.6B`（`forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0")`），建议同时装 FlashAttention 加速；**流式不支持时间戳**。
- **FunASR 1.3.29（2026-07）**：恢复了 SenseVoice 的 VAD 分段时间戳。**FunASR 1.3.27（2026-07）**：`verbose_json.language` 返回检测语种。


### 14. 情感与风格控制

- **SenseVoice 是这套生态里唯一的"一次前向出情感 + 事件 + 语种 + 文本"的模型**，这是它相对 Paraformer / Whisper 的最大差异化。
- 情感标签：`<|HAPPY|>` / `<|SAD|>` / `<|ANGRY|>` / `<|NEUTRAL|>` 等。
- 音频事件标签：`<|Speech|>` / `<|BGM|>` / `<|Applause|>` / `<|Laughter|>` / `<|Cry|>` 等。
- 语种标签：`<|zh|>` / `<|en|>` / `<|yue|>` / `<|ja|>` / `<|ko|>`。
- ITN 标签：`<|withitn|>` / `<|woitn|>`。
- **控制手段**：ASR 侧没有"情感/风格指令"这种 TTS 式控制；可控性是**热词偏置 + 语种提示 + ITN 开关 + VAD 参数**。
- 若需要更细粒度情感（不只是 4 类），用专用模型 `iic/emotion2vec_plus_large`。


### 15. 鲁棒性

- **长音频 OOM**（最常见）：三板斧——减小 `batch_size_s`（300→60）、减小 `max_single_segment_time`（30000→15000）、加 `batch_size_threshold_s=30` 强制长段走 batch=1。
- **模型缓存校验误判**：每次启动打印 "Downloading Model..." **通常不是真的在重新下载**，只是校验本地缓存；要完全跳过用 `AutoModel(model="/local/path", disable_update=True)`。
- **"ModelName is not registered"**：几乎总是 pip 版本太旧，须 `pip install git+https://github.com/modelscope/FunASR.git`。
- **GPU 没用上**：显式 `device="cuda:0"`；先确认 `torch.cuda.is_available()` 为 True。
- **vLLM 引擎首次启动慢 60–90 秒**（KV cache + CUDA graph 初始化，加上可能从 `model.pt` 抽取权重）——健康检查要等 `/health` + warmup 双重确认，不要在启动后立刻压测。
- **vLLM CUDA OOM**：降 `gpu_memory_utilization`、降 `max_model_len`、或加 `tensor_parallel_size`。
- **音频 HTTP 400**：vLLM 侧确认装了 `vllm[audio]`。
- **浏览器麦克风**：Chrome 要求 HTTPS 或 localhost；远程 GPU 服务器要先 `ssh -L 10095:localhost:10095 <server>` 再从本机打开。
- **幻觉重复**：FunASR 的 vLLM WebSocket 服务内置了"幻觉重复清理"，自研链路时要自己加（NAR 模型 + 静音段尤其容易出重复）。
- **安全边界（官方反复强调）**：示例服务**不含**生产认证、租户配额、完整可观测；默认绑 `127.0.0.1` 或内网，TLS/认证/限流/上传大小限制全部放在网关层。


### 16. ASR 指标

#### GGUF CPU 精度（FunASR 官方 llama.cpp 页，同一 184 集中文测试集，micro-avg CER）
| 模型 | CER ↓ | 说明 |
|---|---|---|
| FunASR SenseVoice | **8.01%** | 多语种 + 情感/事件 |
| FunASR Paraformer | 9.85% | 非自回归，工业级中文 |
| FunASR Fun-ASR-Nano | 8.30% | LLM-ASR，中英日 + 中文方言/口音 |
| whisper.cpp small | 22.12% | |
| whisper.cpp large-v3-turbo | 23.15% | |
| whisper.cpp base | 31.33% | |

→ **FunASR 三个模型的中文 CER 约为 whisper.cpp 的 1/3。**

#### 量化档位（含体积/精度/速度，FunASR 官方）
| 模型 | 档 | 体积 | CER ↓ | 速度 |
|---|---|---|---|---|
| SenseVoice encoder | f16 | 470 MB | 8.01% | 23× |
| **SenseVoice encoder** | **q8** | **254 MB** | **7.99%** | **27×** |
| Paraformer encoder | f16 | 435 MB | 9.85% | 22× |
| Paraformer encoder | q8 | 237 MB | 9.78% | 27× |
| Fun-ASR-Nano LLM* | q8_0 | 805 MB | 8.30% | 6.0× |
| Fun-ASR-Nano LLM* | q5_K_M | 551 MB | 8.25% | 5.7× |
| Fun-ASR-Nano LLM* | q4_K_M | 484 MB | 8.35% | 6.1× |

（注：Fun-ASR-Nano 另需 encoder，f16 470 MB。）
**关键结论**：**q8 编码器体积减半、CER 反而略降**（8.01% → 7.99%）；Nano LLM 三档精度几乎无差——**推荐 q4_K_M（最小）或 q5_K_M（最佳）**。

#### 公开基准 CER/WER（来自 Uni-ASR 论文 arXiv:2603.11123 汇总，越低越好）
| 测试集 | GLM-ASR-nano | Whisper-large-v2 | Seed-ASR | FireRedASR-AED | Fun-ASR-nano | Qwen3-ASR-1.7B |
|---|---|---|---|---|---|---|
| AISHELL-1 | 1.81 | — | 0.68 | 0.55 | 1.80 | — |
| LibriSpeech test-clean | 2.00 | 2.5 | 1.58 | 1.93 | 1.76 | **1.63** |
| LibriSpeech test-other | 4.19 | 4.9 | 2.84 | 4.44 | 4.33 | **3.38** |
| FLEURS_zh | — | 4.7 | — | — | 2.56 | **2.41** |
| FLEURS_en | — | 4.2 | 3.43 | — | 5.96 | 3.35 |
| WeNetSpeech Test_Meeting | 6.73 | — | 5.69 | 4.76 | 6.60 | **5.88** |
| WeNetSpeech Test_Net | — | — | 4.66 | 4.88 | 6.01 | **4.97** |

（注：该表格由第三方论文汇总，各列实验条件不完全一致，**横向对比需谨慎**。）

#### Fun-ASR-MLT-Nano-2512 官方多语 WER（HF 模型卡，部分）
| 测试集 | 数值 |
|---|---|
| Fleurs-zh | 2.53 |
| Fleurs-en | 4.74 |
| Librispeech-clean | 1.51 |
| Librispeech-other | 3.03 |
| WenetSpeech Meeting | 6.17 |
| WenetSpeech Net | 5.46 |

#### 效率基准（用户提供，未能独立核实一手来源，**标注待核实**）
- SenseVoice-Small：GPU 约 **170×** 实时、CPU 约 **17×**；
- Paraformer-Large：约 **120× / 15×**；
- Whisper-large-v3：GPU 仅 **13×**。
- 官方口径可核实的一条：**SenseVoiceSmall 处理 10 秒音频仅需 70 ms**（FunASR 教程"模型选择"表 + README），比 Whisper-Large 快 15 倍、参数量与 Whisper-Small 相当时快 5 倍以上。

#### SGLang-Omni 官方实测（Qwen3-ASR-1.7B，H100 80GB）
- SeedTTS EN 语料 WER **0.0122**（并发 1–32 稳定）；SeedTTS ZH 语料 CER **0.0062**。
- **并发到饱和点之前精度不退化**，见 inference_engines.json。


### 17. 延迟与吞吐

#### FunASR vLLM 引擎官方性能表（Fun-ASR-Nano）
| 场景 | PyTorch baseline | vLLM | 加速 |
|---|---|---|---|
| 离线，5.6 s 音频 | 0.89 s | 0.30 s | **3×** |
| 离线，2-GPU tensor parallel | 0.89 s | ~0.20 s | **4.5×** |
| 批量 16 条 | 约 16× 串行成本 | 约 4× | **4×** |
| 批量 32 条 | 约 32× 串行成本 | 约 5× | **6×** |
| WebSocket RTF | 0.156 | 0.078 | **2×** |

#### GGUF / CPU 吞吐（FunASR 官方，速度列即实时倍率）
- SenseVoice encoder f16：23×；**q8：27×**
- Paraformer encoder f16：22×；q8：27×
- Fun-ASR-Nano LLM：q8_0 6.0× / q5_K_M 5.7× / q4_K_M 6.1×（另加 encoder 前向）
- SenseVoice 单条参考：44 s 音频 CPU 编码约 **1.32 s**（llama.cpp runtime README）

#### 效率参考（用户提供，待一手核实）
- SenseVoice-Small：GPU 约 170× 实时、CPU 约 17×；
- Paraformer-Large：约 120× / 15×；
- Whisper-large-v3：GPU 仅 13×。
- Qwen3-ASR-0.6B RTF 0.00923（单并发）、1.7B RTF 0.01482（单并发）；Qwen3-ASR 高并发（128 并发）约 2000× 吞吐。

#### SenseVoice 原生 C++ server 的容量变量（官方）
音频时长、语种、通道数、VAD 分段分布；并发数、排队时间、warmup、模型缓存状态；确切的硬件/驱动/运行时/线程配置。**官方明确要求 REST / SSE / WebSocket 分别压测**，因为三者的连接模型不同。

#### 首字/首包延迟参考（用户提供，待一手核实）
- FunASR（SenseVoice-Small）端到端直接并行，TTFT 约 **70 ms**；
- Qwen3-ASR 约 **92 ms**。


### 18. 成本模型

- **FunASR / SenseVoice 的核心成本优势是"非自回归 + 可 CPU/边缘"**：NAR 模型的吞吐与并发几乎线性，且不需要 KV cache，单张消费级 GPU 甚至纯 CPU 就能跑出很高的实时倍率。对比 Qwen3-ASR 需要 GB 级显存与 AR 解码，**单位音频时长的成本低一个量级**。
- **社区测算（来源：gitcode 研究文章，**二手数据待核实**）**：
  - Qwen3-ASR-Flash 云 API：约 **$0.000035 / 音频秒**（≈ $0.0021/分钟）；
  - 在 Vast.ai 自建 A100 跑 FunASR：可低至 **$0.20 / 1000 小时**（≈ $0.0000033/分钟）——**私有化自建在超大规模离线质检场景下比云 API 便宜约 600 倍**。
- **选型推论**：
  - 海量离线质检 / 批量字幕 / 呼叫中心整轨转写 → **FunASR（SenseVoice / Paraformer）自建**，成本优势压倒性；
  - 高精度、多语种/方言、需要上下文理解 → **Qwen3-ASR**，为精度付费；
  - 边缘/终端/数据不出域 → **FunASR GGUF（llama.cpp / audio.cpp / sensevoice-server）**，零 GPU 成本。
- **注意**：以上成本数字均为社区测算，**未核实一手来源**；做预算前必须用自家音频与目标硬件复测。


### 19. 推理框架支持

#### 官方部署矩阵（funasr.com 部署中心，2026-08-13 快照，共 10 条路径）

| 路径 | 工作负载 | 硬件 | 接口 | 成熟度 | 验证日期 |
|---|---|---|---|---|---|
| **Native FunASR on vLLM** | batch / private-api | nvidia-gpu / kubernetes | OpenAI-compatible HTTP | **社区验证** | 2026-08-13 |
| **SenseVoice TensorRT / Triton** | batch / private-api | nvidia-gpu / kubernetes | Triton gRPC/HTTP + TensorRT plan | **生产验证** | 2026-08-04（索引页一处显示 2026-08-27，以 contract 页为准） |
| **llama.cpp / GGUF standalone** | edge / batch | cpu / desktop-edge-gpu | CLI / local HTTP server | **生产验证** | 2026-08-13 |
| **SenseVoice native OpenAI realtime server** | private-api / realtime / edge | cpu / nvidia-gpu | OpenAI HTTP + Realtime WS + SSE + SRT/VTT | **生产验证** | 2026-08-13 |
| **audio.cpp native Fun-ASR-Nano 和 SenseVoice** | edge / batch / private-api | cpu / nvidia-gpu / desktop-edge-gpu | CLI / OpenAI-compatible HTTP | **社区验证** | 2026-08-13 |
| **OpenAI-compatible private API** | private-api / batch | cpu / nvidia-gpu / kubernetes | OpenAI HTTP / OpenAPI | **生产验证** | 2026-08-17 |
| **Realtime streaming and captions** | realtime | cpu / nvidia-gpu / kubernetes | WebSocket / Python client | **社区验证** | 2026-07-26 |
| **Containers and Kubernetes** | private-api / batch / realtime | kubernetes / cpu / nvidia-gpu | Docker Compose / K8s ClusterIP | **社区验证** | 2026-07-26 |
| **ONNX / C++ CPU runtime** | batch / realtime / edge | cpu | C++ / ONNX Runtime | **社区验证** | 2026-07-26 |
| **Production rollout checklist** | 全部 | 全部 | HTTP / WebSocket | **生产验证** | 2026-07-26 |

##### ① Native FunASR on vLLM（社区验证，2026-08-13）
- 运行时：**vLLM 0.27.1+cu129 / Torch 2.13.0+cu129**；FunASR `main`；模型 **Fun-ASR-Nano-2512 社区 vLLM 转换**。
- 安装：
  ```
  curl -fL https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl -o vllm-0.27.1+cu129-...whl
  echo "bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b  vllm-0.27.1+cu129-...whl" | sha256sum -c -
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python --torch-backend=auto "vllm[audio] @ file://$PWD/vllm-0.27.1+cu129-...whl"
  ```
- 启动：
  ```
  CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve allendou/Fun-ASR-Nano-2512-vllm \
    --revision e718b36e2578203ec893e9b488239225f8d668e2 \
    --served-model-name fun-asr-nano --host 127.0.0.1 --port 8899 \
    --dtype float32 --gpu-memory-utilization 0.40 --enforce-eager
  ```
- 冒烟：
  ```
  curl -fsS http://127.0.0.1:8899/v1/audio/transcriptions -F file=@zh.mp3 -F model=fun-asr-nano -F language=zh -F response_format=json
  curl -fsS http://127.0.0.1:8899/v1/audio/transcriptions -F file=@zh.mp3 -F model=fun-asr-nano -F language=zh \
       -F 'hotwords=开放时间,开放时间,开放时间' -F response_format=json
  ```
- **官方限制（必须知道）**：依赖**社区转换 checkpoint**，不接受"只认官方权重"的场景；**证据仅覆盖一张 H100**，换 GPU / 音频 / 语种 / 热词 / 流量都必须复测。

##### ② SenseVoice TensorRT / Triton（生产验证）
- 运行时：**TensorRT 10.0.1 / Triton 24.05**，FunASR `main@6408aaa9`。
- 步骤：
  ```
  pip install -e . "onnx>=1.16" soundfile "tritonclient[grpc]"
  # 导出 ONNX（注意 quantize=False，model_quant.onnx 不支持）
  model = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0")
  model.export(type="onnx", quantize=False, device="cuda:0",
               output_dir="./sensevoice_onnx", max_seq_len=4096)
  # 下载并校验 tokenizer
  curl -fL https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/main/chn_jpn_yue_eng_ko_spectok.bpe.model \
    -o runtime/triton_gpu/model_repo_sense_voice_small/scoring/chn_jpn_yue_eng_ko_spectok.bpe.model
  echo "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8  .../chn_jpn_yue_eng_ko_spectok.bpe.model" | sha256sum -c -
  # 建 TensorRT FP16 引擎（optimization profile 必须贴合生产 batch/时长分布）
  python runtime/triton_gpu/scripts/build_sensevoice_tensorrt.py \
      ./sensevoice_onnx/model.onnx \
      runtime/triton_gpu/model_repo_sense_voice_small/encoder/1/model.plan \
      --precision fp16 \
      --min-batch 1 --opt-batch 8 --max-batch 16 \
      --min-frames 1 --opt-frames 512 --max-frames 4096 \
      --workspace-gb 8
  cp .../encoder/config.pbtxt.tensorrt .../encoder/config.pbtxt
  cd runtime/triton_gpu && tritonserver --model-repository ./model_repo_sense_voice_small \
      --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000
  ```
- 健康检查：`curl -fsS http://localhost:8000/v2/health/ready`、`curl -fsS http://localhost:8000/v2/models/sensevoice/ready`。
- gRPC 冒烟（输入张量 `WAV` / `WAV_LENS` / `LANGUAGE` / `TEXT_NORM`，输出 `TRANSCRIPTS`）。
- **限制**：TensorRT plan 不可跨 GPU 架构 / TensorRT 版本移植，换卡必须重建并重验精度、显存、容量；不支持 CPU / macOS / Windows / 通用边缘。

##### ③ llama.cpp / GGUF standalone（生产验证，2026-08-13，v0.2.1 九个发布包）
- 覆盖 Linux(x64/arm64)、macOS(arm64)、Windows(x64)，静态自包含无依赖；支持 **CPU / Vulkan / CUDA**。
- 三行跑通：
  ```
  bash download-funasr-model.sh sensevoice ./gguf
  llama-funasr-sensevoice -m ./gguf/SenseVoiceSmall-f16.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav
  # -> 欢迎大家来体验达摩院推出的语音识别模型
  ```
- 换模型：`download-funasr-model.sh paraformer` 配 `llama-funasr-paraformer`；`nano` 配 `llama-funasr-cli`。
- 六个二进制：`llama-funasr-{cli,encoder,embd,sensevoice,paraformer,vad}`；内置 FSMN-VAD（`--vad`）、二进制内 detok 直接出文字、kaldi 兼容 fbank 前端。
- **限制**：v0.2.1 修好了集成显卡选择问题但仍需按硬件复测；AMD Radeon 780M 待复测；RX 9070 XT 的 0xC0000005 初始化崩溃未解决；Android/Mali 不是预编译或验证目标。

##### ④ SenseVoice 原生 OpenAI 实时 server（生产验证，2026-08-13）
- **一个 C++ 二进制，无 Python 运行时**。FunASR/SenseVoice `main@b054623c`，运行时 `sensevoice-server@558bd67c`。
- 接口：OpenAI 兼容 HTTP、OpenAI Realtime WebSocket（`WS /v1/realtime?intent=transcription`）、SSE、SRT/VTT；模型 `SenseVoiceSmall-GGUF` + `FSMN-VAD-GGUF`；硬件 cpu / nvidia-gpu；**Linux**。
- 安装与启动：
  ```
  git clone https://github.com/QwenAudio/SenseVoice.git && cd SenseVoice
  git checkout b054623cca8f015b73ec471dce4f473ac47413da
  python3 -m pip install -U huggingface_hub
  bash runtime/llama.cpp/download-funasr-model.sh sensevoice model
  cmake -S runtime/llama.cpp -B runtime/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build runtime/llama.cpp/build --target sensevoice-server -j

  runtime/llama.cpp/build/bin/sensevoice-server \
    -m model/sensevoice-small-q8.gguf -vad model/fsmn-vad.gguf \
    --threads 8 --max-connections 4 --max-audio-seconds 300 127.0.0.1 8040
  ```
  模型文件**必须**命名为 `sensevoice-small-q8.gguf` 与 `fsmn-vad.gguf`。
- 冒烟四连：
  ```
  curl -fsS http://127.0.0.1:8040/health
  curl -fsS http://127.0.0.1:8040/v1/models
  curl -fsS http://127.0.0.1:8040/v1/audio/transcriptions -F file=@speech.wav
  curl -fsS http://127.0.0.1:8040/v1/audio/transcriptions -F file=@speech.wav -F response_format=vtt | tee transcript.vtt && grep -q WEBVTT transcript.vtt
  curl -fsSN http://127.0.0.1:8040/v1/audio/transcriptions -F file=@speech.wav -F stream=true | grep -q transcript.text.done
  python3 runtime/llama.cpp/tests/stream_client.py 127.0.0.1 8040 speech_16k.wav 200
  ```
- **限制**：官方只验证 Linux 源码构建；**不含**生产认证、TLS、租户配额、持久队列——必须在网关/基础设施层补。

##### ⑤ audio.cpp（社区验证，2026-08-13）
Fun-ASR-Nano-2512 + SenseVoice-Small Q8，原生 C++ + GGML，CPU / NVIDIA GPU / 桌面边缘 GPU；CLI + OpenAI 兼容 HTTP + SenseVoice 缓冲流式。
**限制**：SenseVoice 已合入 `main@979e070f` 但**尚未进 tagged release（需 pin commit）**；两条路径都**不提供词级时间戳**。

##### ⑥–⑩ 其余路径
- **OpenAI 兼容私有 API**：`funasr-server --device cuda --port 8000`（`pip install funasr fastapi uvicorn python-multipart`），`/v1/audio/transcriptions`，`model="sensevoice"`。
- **实时流式与字幕**：在线或 two-pass WebSocket 协议，模型 `Paraformer-online` + `FSMN-VAD`。
- **容器与 K8s**：Docker Compose 验证 → K8s 模板 + 持久模型缓存 + 探针 + 私有 ClusterIP（**CPU 示例镜像不是 CUDA 生产镜像**）。
- **ONNX / C++ CPU runtime**：Paraformer / Paraformer-online 的纯 CPU 部署。
- **MCP server**：`examples/mcp_server/funasr_mcp.py`，供 Claude Code / Cursor 等 MCP 客户端调用。
- **字幕生成**：`examples/subtitle/generate_subtitle.py video.mp4 [--spk] [--format vtt]`。

#### 框架支持度一句话总结
- **vLLM**：**只支持 AR 模型**（Fun-ASR-Nano / LLMASR / GLMASR / Qwen3-ASR）。**Paraformer 与 SenseVoice 永远不要用 vLLM。**
- **ONNX Runtime**：Paraformer / SenseVoice 的主力 CPU 部署方式（`funasr-onnx`）。
- **Libtorch**：C++ 环境（`funasr-torch`，`export(type="torch")`）。
- **TensorRT / Triton**：SenseVoice 有官方生产验证路径。
- **llama.cpp / GGUF / audio.cpp**：边缘与 CPU 的唯一现实选择。
- **TensorRT-LLM**：SenseVoice / Paraformer 走不通（NAR 架构），无需考虑。


### 20. 资源需求

#### GGUF 体积（官方）
| 模型 | 档 | 体积 | CER | 速度 |
|---|---|---|---|---|
| SenseVoice encoder | f16 | 470 MB | 8.01% | 23× |
| SenseVoice encoder | q8 | **254 MB** | **7.99%** | 27× |
| Paraformer encoder | f16 | 435 MB | 9.85% | 22× |
| Paraformer encoder | q8 | 237 MB | 9.78% | 27× |
| Fun-ASR-Nano LLM（+encoder f16 470MB） | q8_0 | 805 MB | 8.30% | 6.0× |
| Fun-ASR-Nano LLM | q5_K_M | 551 MB | 8.25% | 5.7× |
| Fun-ASR-Nano LLM | q4_K_M | **484 MB** | 8.35% | 6.1× |
| SenseVoice f32（未量化导出） | f32 | ~936 MB | — | — |

#### vLLM 引擎资源要求（FunASR 官方）
| 资源 | 最低 | 推荐 |
|---|---|---|
| GPU 显存 | 8 GB | **16 GB 以上**（给 KV cache 留充足空间） |
| CUDA | 11.8 | **12.x** |
| GPU 数量 | 1 | tensor parallel 时 2 张或更多 |

#### SenseVoice TensorRT 资源
- 建引擎时 `--workspace-gb 8`；optimization profile 需按生产分布设定（`--min-batch/--opt-batch/--max-batch`、`--min-frames/--opt-frames/--max-frames`）。
- Triton 启动参数：`--pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000`。
- **必须留存 ONNX、engine、build log、SHA-256、Triton 配置以便回滚。**

#### CPU / 边缘
- SenseVoice q8 **254 MB** 是最常用的边缘档位；配合 `sensevoice-server` 的 `--threads 8 --max-connections 4 --max-audio-seconds 300` 即可在一台 CPU 机器上提供 OpenAI 兼容服务。
- 量化收益（社区实测，**待一手核实**）：INT8 相对 FP32 体积 −74%、内存 −71%、加载时间 −62%、CPU 推理提速 33–36%、CER 平均 +1.1%。

#### 精度代价
- **GGUF q8 基本无损**（SenseVoice CER 8.01% → 7.99%，Paraformer 9.85% → 9.78%，**甚至略有改善**）；Nano LLM q4/q5/q8 三档差异 < 0.1%。
- **ONNX 动态 INT8**（QUInt8 权重、MatMul 为主、输出节点与 bias 不量化）：社区实测 CER +1.1%（清晰 +0.5%，嘈杂 +1.2%，方言 +0.9%，多语混合 +1.5%），**待一手核实**。


### 21. 服务化栈

#### FunASR 原生服务
| 入口 | 命令 | 用途 |
|---|---|---|
| `funasr-server` | `funasr-server --device cuda --port 8000` | OpenAI 兼容本地转写服务（`pip install funasr fastapi uvicorn python-multipart`） |
| `serve_realtime_ws.py` | `CUDA_VISIBLE_DEVICES=0 python serve_realtime_ws.py --port 10095 --language 中文` | vLLM 生产级 WebSocket 实时服务（动态 VAD + partial + 热词 + 说话人） |
| `run_server.sh` | FunASR `runtime/` 下 | 离线文件转写服务（wss，端口 10095） |
| `run_server_2pass.sh` | 同上 | two-pass（在线 + 离线精修）服务 |
| `sensevoice-server` | C++ 单二进制 | 无 Python 运行时，REST + SSE + SRT/VTT + OpenAI Realtime WS |
| Gradio demo / Postman collection / OpenAPI spec | — | 联调与网关对接 |

#### 完整启动时参数示例（FunASR runtime，生产常用）
```
nohup bash run_server.sh \
  --download-model-dir /workspace/models \
  --vad-dir damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
  --model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx \
  --punc-dir damo/punc_ct-transformer_cn-en-common-vocab471067-large-onnx \
  --lm-dir damo/speech_ngram_lm_zh-cn-ai-wesp-fst \
  --itn-dir thuduj12/fst_itn_zh \
  --hotword /workspace/models/hotwords.txt \
  --certfile 0 > server.log 2>&1 &
```
- `--certfile 0` 关闭 SSL。
- 需要**时间戳**用 `--model-dir damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx`；
- 需要**神经网络热词**用 `--model-dir damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404-onnx`。

#### WebSocket 协议与端口
- FunASR runtime 服务默认端口 **10095**（wss；`--certfile 0` 时可用 ws）。
- 客户端：`funasr_wss_client.py --host 127.0.0.1 --port 10095 --mode offline --audio_in ../audio/asr_example.wav`；`--mode online` / `2pass`。
- SenseVoice 原生 server 端口示例 **8040**，WebSocket 客户端发 **16 kHz 单声道 PCM16 的 base64 分块**。

#### Docker 部署（可直接抄）
```
# CPU 离线服务镜像
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.4.7
# 在线（流式）服务镜像
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-zh:0.1.6
# 带 VAD+PUNC 的在线服务镜像
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.5
```
Docker Compose 生产模板要点（社区验证版）：
```yaml
services:
  funasr:
    image: registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.5
    container_name: funasr-service
    restart: unless-stopped
    ports: ["10095:10095"]
    volumes:
      - ./models:/workspace/models
      - ./config/hotwords.txt:/workspace/models/hotwords.txt   # 热词文件外部挂载
      - ./logs:/var/log/funasr
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:10095 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
    command: ["bash","-c","cd FunASR/runtime && nohup bash run_server_2pass.sh ... > /var/log/funasr/service.log 2>&1 & ..."]
```
OpenAI API 的 Compose 版在 `examples/openai_api/`（`cp .env.example .env && docker compose up --build`），**默认 CPU，要用 CUDA 需自行换镜像**。

#### 网关职责（引擎不管）
TLS、认证、租户配额、限流、审计、上传策略（大小/时长/声道/采样率/并发）、监控、灰度回滚。


### 22. 硬件生态

| 硬件 | FunASR（PyTorch/ONNX） | FunASR vLLM 引擎 | GGUF / llama.cpp | SenseVoice 原生 C++ server | TensorRT / Triton |
|---|---|---|---|---|---|
| NVIDIA GPU | ✅ | ✅（≥8 GB，推荐 16 GB+，CUDA 12.x） | ✅（CUDA 包） | ✅（可选 CUDA） | ✅（**唯一支持**） |
| CPU（x86 / ARM） | ✅（ONNX 主导） | ❌ | ✅（**主场景**，Vulkan 可加速） | ✅（**主场景**） | ❌ |
| Apple Silicon | ✅（ONNX / PyTorch MPS） | ❌ | ✅（macOS arm64 发布包） | ❌（官方仅验 Linux） | ❌ |
| Windows | ✅ | ❌ | ✅（x64 发布包，含 Vulkan / CUDA 包） | ❌ | ❌ |
| 昇腾 / 海光等国产卡 | 待核实 | 待核实 | 待核实 | 待核实 | 待核实 |
| Android / Mali | 待核实 | ❌ | ❌（非预编译/非验证目标） | ❌ | ❌ |

**边缘实测参考（社区数据，待一手核实）**：INT8 量化 SenseVoice 处理 10 s 音频 —— Intel i7-11800H **55 ms**、Raspberry Pi 4B **420 ms**、NVIDIA T4 **30 ms**、Jetson Xavier NX **85 ms**。


### 23. 集成范式

#### 模式 A：Python 内嵌（最快上手）
```
from funasr import AutoModel
model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
res = model.generate(input="meeting.wav", batch_size_s=300, hotword="达摩院 语音识别")
```
适合离线任务、内部工具、数据流水线。**注意：`cache={}` 必须在所有 chunk 间保持。**

#### 模式 B：OpenAI 兼容私有 API（推荐作为团队统一契约）
```
funasr-server --device cuda --port 8000
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
result = client.audio.transcriptions.create(model="sensevoice",
    file=open("meeting.wav","rb"), response_format="verbose_json")
```
**价值**：任何支持 OpenAI audio API 的 SDK / 工作流引擎（Dify、n8n、LangChain、AutoGen）零改造接入；Qwen3-ASR 与 SenseVoice 用**同一套契约**对外，网关可按语种/成本/时延路由。

#### 模式 C：实时流式（WebSocket）
- 前端：`client_mic.html`（浏览器麦克风 + 文件上传 + 热词 + 说话人标签）；`client_python.py --server ws://localhost:10095 --mic`。
- 浏览器限制：Chrome 要求 HTTPS 或 localhost；远程 GPU 服务器先 `ssh -L 10095:localhost:10095 <server>`。
- 协议细节见 streaming 字段。

#### 模式 D：双引擎（two-pass）精修
在线模型出实时结果 → VAD 尾点 → 离线大模型 + 标点模型出终稿。FunASR 用 `run_server_2pass.sh` 一条命令起。

#### 模式 E：与 Agent / MCP 集成
`examples/mcp_server/funasr_mcp.py` 把转写能力暴露成本地 tool，供 Claude Code / Cursor 直接调用。

#### 模式 F：混合路由（推荐生产形态）
```
                    ┌─ 高吞吐 / 中文 / 边缘 ──→ FunASR(SenseVoice / Paraformer)
网关 /v1/audio/transcriptions ─┤
                    └─ 高精度 / 多语种方言 / 需上下文 ──→ Qwen3-ASR (vLLM)
```
判定维度建议：语种数、是否需要情感/事件、是否要求上下文偏置、时延预算、单卡吞吐目标、是否边缘。

#### 智能分段（Qwen3-ASR 社区封装的工程经验，可参考）
基于 VAD 的**贪心合并算法**自动切分长音频，避免片段中包含过长静音；流式 ASR 叠加**智能远场过滤**（自动滤掉远场声音与环境音，减少误触发）。


### 24. 技术栈组件

下面给出 **ASR 生态专用**的分层清单（通用部分见 inference_engines.json 的 tech_stack_components 字段，两份配合使用）。

---

#### L0 音频采集与预处理
| 组件 | 用途 | 备注 |
|---|---|---|
| **ffmpeg** | 统一解码 MP3/M4A/FLAC/OGG/Opus/视频容器 → **16 kHz 单声道 PCM16** | 事实标准。Qwen3-ASR-FastAPI 就是这么做的 |
| **soundfile**（libsndfile） | Python 内读写 wav/flac/ogg | FunASR `generate()` 可直接吃 numpy array，不落盘 |
| **sounddevice / PyAudio** | 实时麦克风采集 | 优先 sounddevice |
| **Web Audio API + AudioWorklet** | 浏览器采集 + 原生 AEC/NS/AGC | 见 inference_engines.json |
| **FunASR `AutoModel` 内置前端** | 自动重采样到 16 kHz | 生产仍建议在网关统一做一次 |

#### L1 传输
| 场景 | 协议 |
|---|---|
| 文件/批量 | HTTP multipart `POST /v1/audio/transcriptions` |
| 实时流式 | **WebSocket**（FunASR runtime 10095；SenseVoice server 8040；`serve_realtime_ws.py` 10095） |
| 单向结果下发 | **SSE**（SenseVoice 原生 server 支持 `stream=true`） |
| GPU 批量服务 | **gRPC**（Triton 8001） |

#### L2 VAD
| 组件 | 场景 | 参数 |
|---|---|---|
| **fsmn-vad** | 与 FunASR 全链路配套；支持**动态 VAD** | `max_single_segment_time=30000`（Paraformer 长音频 60000）；流式 `chunk_size=200`（200 ms/块） |
| **Silero VAD** | 自建实时链路、网关侧切句 | 512 样本 @16 kHz（32 ms）；`positiveSpeechThreshold=0.3`、`negativeSpeechThreshold=0.25`、`silenceDuration=1400ms`、`minSpeechDuration=400ms`、`preSpeechPadDuration=800ms`；实践值「2 帧起（~64 ms）/ 20 帧止（~640 ms）」 |
| **WebRTC VAD** | 极资源受限 | 8/16/32 kHz，mode 0–3；嘈杂环境弱于神经网络 |

#### L3 特征提取
| 组件 | 场景 |
|---|---|
| **引擎内置前端** | FunASR / llama.cpp 内置 kaldi 80-mel fbank + LFR；**不要业务层重复实现** |
| **torchaudio** | GPU 侧重采样/特征；Qwen3-ASR 的 encoder 在 PyTorch 侧 |
| **librosa** | 离线分析、数据准备 |
| **soxr** | 48 kHz → 16 kHz 抗混叠重采样（WebRTC 入口必备） |

#### L4 模型推理（按场景选）
| 场景 | 选型 |
|---|---|
| 中文高吞吐 / 生产稳定 | **Paraformer**（+ fsmn-vad + ct-punc + cam++） |
| 情感 + 音频事件 + LID | **SenseVoiceSmall** |
| 多语言（31 语种）/ 方言 / 歌词 | **Fun-ASR-Nano-2512** |
| 最高精度 / 52 语种方言 / 上下文偏置 | **Qwen3-ASR-1.7B**（vLLM） |
| 实时字幕（中文） | **paraformer-zh-streaming**（600 ms 增量）或 **Qwen3-ASR 流式**（vLLM） |
| GPU 批量极限吞吐 | **SenseVoice + Triton/TensorRT FP16** |
| CPU / 边缘 / 无 Python | **GGUF + llama.cpp** 或 **sensevoice-server**（C++ 单二进制） |

#### L5 后处理链
| 组件 | 作用 | 何时需要 |
|---|---|---|
| `ct-punc` | 中英双语标点 | **仅 Paraformer 需要** |
| ITN（`use_itn=True` / `fst_itn_zh`） | 数字、日期、单位规整 | 输出面向人阅读时 |
| `cam++` | 说话人分离/验证（192 维 embedding） | 会议、通话质检 |
| ForcedAligner（`Qwen3-ForcedAligner-0.6B`） | 词/字级时间戳 | Qwen3-ASR 要时间戳时 |
| `rich_transcription_postprocess()` | 剥离 SenseVoice 的 `<|zh|><|HAPPY|>` 等标签 | SenseVoice 必用 |
| 幻觉重复清理 | 剔除静音段产生的重复文本 | 自研流式链路必加 |

#### L6 热词 / 上下文偏置（两条完全不同的路子）
| 路线 | 机制 | 适用模型 | 用法 |
|---|---|---|---|
| **WFST 热词**（FunASR runtime） | 把热词编成带权重的有限状态机，与语言模型 WFST 组合，提升目标词匹配优先级 | **Paraformer**（需 contextual 模型） | `hotwords.txt` 每行「热词 权重」，权重 1–100 默认 10；服务端 `--hotword /path/hotwords.txt` 全局生效，客户端 API 参数本次生效；支持**在线更新无需重启** |
| **Prompt 上下文偏置**（新一代） | 热词写入 ASR prompt，靠 LLM 的上下文理解生效 | **Qwen3-ASR / Fun-ASR-Nano** | `hotwords=["张三","北京"]` 列表；vLLM 侧用逗号分隔字符串 `-F 'hotwords=开放时间,开放时间,开放时间'` |

**热词权重经验值**（社区）：关键专业术语 75–85；高频业务词 65–75；普通优化词 50–65。另有一组更保守的建议：专业术语 60–80、高频常用词 40–60、普通词 10–30。**过拟合风险**：权重过高会压低普通词准确率，FunASR 用动态权重调整 + 混淆集过滤（如同音词"阿里巴巴"vs"阿里爸爸"）缓解。**热词库容量建议单实例不超过 10 万条**，超出需走分布式方案。

#### L7 服务化与编排
- `funasr-server` / `serve_realtime_ws.py` / `run_server.sh` / `run_server_2pass.sh` / `sensevoice-server`（见 serving_stack）。
- 容器：Docker Compose 验证 → Kubernetes（私有 ClusterIP + 探针 + 持久模型缓存）。
- 队列：Celery/RQ + Redis（离线批）；asyncio + asyncio.Queue（实时）。
- **背压**：有界队列 + 429/503 快速失败 + 客户端指数退避。

#### L8 监控
- 指标：RTF、TTFT、排队时长、VAD 分段分布、热词命中率、语种分布、错误类型。
- 日志：每请求记录**音频时长 / 模型 / 设备 / 延迟 / response_format / 错误类型**（FunASR 官方 readiness checklist 明确要求）。
- 追踪：OpenTelemetry，按 VAD / ASR / PUNC / SPK 分段打 span。

#### L9 评测
见 eval_toolchain 字段。

---

#### 两条推荐默认栈

**A. 中文高吞吐离线质检栈（成本优先）**
```
ffmpeg 16k mono → FunASR(paraformer-zh + fsmn-vad + ct-punc + cam++)
  → 批任务队列(Celery+Redis) → 结果入库
  → 热词(hotwords.txt WFST)  → 监控(RTF/热词命中率)
```

**B. 高精度多语种实时栈（质量优先）**
```
WebRTC(Opus) → soxr 48k→16k → Silero VAD
  → vLLM(Qwen3-ASR-1.7B) /v1/audio/transcriptions
  → [可选] Qwen3-ForcedAligner-0.6B 出词级时间戳
  → 热词走 prompt 偏置 → SSE 下发 → OTel 全链路
```


### 25. 评测工具链

#### ASR 侧（本条目主责）
| 工具 | 用途 | 备注 |
|---|---|---|
| **jiwer** | WER / CER 计算（Python，含 CLI） | 事实标准。**中文必须先分词**（推荐 jieba），否则 CER/WER 语义不对 |
| **sclite**（NIST） | WER/SER + 错误对齐 | 学术对照常用 |
| **ASR-Evaluation** | 多指标集成评测 | |
| **NeMo Speech Data Explorer** | 多模型/多语言可视化对比 + 错误分析 | 输入 JSON manifest（音频路径 / 参考 / 预测） |
| **`rich_transcription_postprocess`** | 剥离 SenseVoice 标签后再评测 | **不剥离会把标签算进 CER，必须做** |

**评测流程要点**：
1. **参考与预测必须走完全相同的预处理**（大小写、标点、非语音标签、数字规整）。不一致会人为抬高 WER。
2. 中文用 **CER**（字错误率），英文用 **WER**（词错误率）；报告时必须注明。
3. 多人对话音频要先按说话人切分再算，否则 WER 虚高。
4. 除 WER 外要看**错误类型分布**（S 替换 / D 删除 / I 插入），定位是声学问题还是语言模型问题。
5. **只盯 WER 不够**：下游任务（意图识别、实体抽取）准确率才是业务指标。

**常用开源测试集**：AISHELL-1/2（中文普通话）、LibriSpeech（英文多口音）、WenetSpeech（中文会议/网络）、FLEURS（多语言）、Common Voice（60+ 语言）、CHIME（噪声/远场）。

#### TTS 侧（跨条目引用，详见 production_deployment.json）
| 层 | 工具 |
|---|---|
| 可懂度 | WER（英文，Whisper）/ CER（中文，Paraformer-zh） |
| 说话人相似度 | WavLM-large、**ECAPA-TDNN**、**wespeaker**、X-Vector、RawNet3；`k2-fsa/TTS_eval_models`（ECAPA-TDNN + WavLM 组合） |
| 自然度 | **UTMOS / UTMOSv2**、NISQA、DNSMOS、SQUIM-MOS |
| 综合基准 | **Seed-TTS-eval**（零样本）、**TTSDS / TTSDS2**（分布级，pip `ttsds`，Spearman ρ=0.67）、**TTS Arena**（主观 ELO）、**TTSD-eval**（对话 TTS，MMS-FA 对齐 + wespeaker） |
| 数据集 | `k2-fsa/TTS_eval_datasets`；LibriSpeech-PC；FLEURS；Seed-TTS test-zh / test-en |

#### 测试集分级（推荐）
- **L1 快速回归**：Seed-TTS 各 50 条（EN 50 + ZH 50），CI 每次跑。
- **L2 完整零-shot 标准**：完整 Seed-TTS-eval + LibriSpeech-PC。
- **L3 多语言扩展**：FLEURS / MiniMax Multilingual。
- **L4 对话场景**：100 组多轮对话（30 s–720 s），来自 TTSD-eval。

#### 六阶段客观评测流水线
1. **测试集与 manifest**：`test.tsv` 含 `utt_id`、参考文本、参考音频路径、speaker ID、固定采样率（16 k 或 24 k）；
2. **批量合成**：统一超参（seed / temperature / top_p / streaming flag），输出到 `results/{testset}/{utt_id}.wav`；
3–5. **指标计算**：SIM（speaker embedding 余弦）、WER/CER（hubert / Seed-TTS 后端）、UTMOS；
6. **汇总与排名**：JSON/CSV 落盘（dataset, model_id, WER, SIM, UTMOS, date, git_hash），可选算 TTSDS2 综合分并结合 TTS Arena 主观结果。

#### 六个必须遵守的陷阱规避规则
1. **对比 Seed-TTS 官方表格时，绝不能更换 ASR 监听模型**——换模型 = 换尺子。
2. **同一份 SIM 报告不能混用不同网络（WavLM vs ECAPA-TDNN）的 speaker embedding**。
3. **MOS 预测器在 4.5 分以上会饱和**；模型已到 4.3+ 时用 **UTMOSv2** 或走 **MUSHRA / A-Test**，并注意 **zoomed-in bias**（听测者用满 5 级尺会系统性压低中档系统）。
4. **TTSDS2 是分布级指标**，需要两个目录各 20+ 条（论文建议 100+）wav，样本太少估不出稳定分布。
5. **流式场景必须额外报** TTFA 与 streaming RTF。
6. 报 WER/CER 时**同时报 UTMOS**，单一指标会误导。


### 26. 已知坑与限制

1. **最常见的架构误用：给 SenseVoice / Paraformer 上 vLLM**。这两个是 NAR 模型，不走 KV-cache 解码，vLLM **完全帮不上忙**（FunASR 官方 FAQ 直接写明"Paraformer 能用 vLLM 吗？不能"）。
2. **论文语种 ≠ checkpoint 语种**：SenseVoice 论文描述 40 万小时 / 50+ 语言，但公开 checkpoint **只支持 5 个语种**（中/粤/英/日/韩）。选型以 checkpoint 为准。
3. **每次启动都打印 "Downloading Model..."**：多数情况只是校验本地缓存，不是重新下载。要彻底跳过用 `disable_update=True` + 本地路径。
4. **长音频 OOM**：减 `batch_size_s`（300→60）、减 `max_single_segment_time`（30000→15000）、加 `batch_size_threshold_s=30`。
5. **"ModelName is not registered"**：pip 版本太旧，`pip install git+https://github.com/modelscope/FunASR.git`。
6. **流式 `cache` 被重建**：`cache={}` 必须在所有 chunk 间共享；最后一个 chunk 忘记 `is_final=True` 会丢尾部文字。
7. **流式前 1.5–3 秒输出为空/不稳定**：FunASR vLLM 流式引擎的**模型特性**，不是 bug（前 10 个 chunk 用于找稳定前缀）。
8. **Qwen3-ASR 流式不支持 batch 与时间戳**；时间戳要额外加载 `Qwen3-ForcedAligner-0.6B`。
9. **SenseVoice 推理不接 CMVN**（llama.cpp runtime 明确点名的移植 gotcha），自己复现时要对齐。
10. **Triton 路径 tokenizer 必须单独下载并校验 SHA-256**（`aa87f860...`），路径/版本错了会静默产出乱码而非报错。
11. **TensorRT plan 不可移植**：换 GPU 架构或 TensorRT 版本必须重建并重验精度、显存、容量；`model_quant.onnx`（动态量化）**不被 SenseVoice TensorRT 路径支持**，必须用 `quantize=False` 导出。
12. **vLLM 引擎首次启动 60–90 秒**，健康检查要等 `/health` + warmup 双确认，别一启动就压测。
13. **官方示例服务不含认证 / TLS / 租户配额 / 完整可观测**，必须在网关层补齐；默认绑 `127.0.0.1`。
14. **热词过拟合**：权重过高会压低普通词准确率；容量超 10 万条需分布式方案。
15. **audio.cpp 的 SenseVoice 尚未进 tagged release**（`main@979e070f`），必须 pin commit；且无词级时间戳。
16. **llama.cpp 边缘包硬件兼容有未解决问题**：AMD Radeon 780M 待复测、RX 9070 XT 的 0xC0000005 崩溃未解、Android/Mali 非验证目标。
17. **浏览器麦克风限制**：Chrome 要求 HTTPS 或 localhost，远程服务器需 SSH 端口转发。
18. **Native FunASR on vLLM 依赖社区转换 checkpoint**，且证据仅覆盖一张 H100——不接受"只认官方权重"的场景，换卡必须复测。
19. **社区 vLLM checkpoint 需 `--dtype float32 --enforce-eager`**（官方验证命令即如此），不要想当然套 BF16。


### 27. 社区与文档

| 项目 | 许可 | 定位 | 入口 |
|---|---|---|---|
| **FunASR** | Apache-2.0 | 工具包 + 部署基础设施 | github.com/modelscope/FunASR；文档 modelscope.github.io/FunASR；部署中心 funasr.com（中/英双语） |
| **Fun-ASR** | Apache-2.0 | 端到端 ASR 模型家族（旗舰 + 31 语种 MLT） | github.com/FunAudioLLM/Fun-ASR |
| **SenseVoice** | Apache-2.0 | 语种 + 情感 + 事件 | github.com/FunAudioLLM/SenseVoice |
| **FunClip** | 开源 | 视频理解与智能剪辑 | github.com/modelscope/FunClip |

**文档质量（本生态最大的差异化优势之一）**：funasr.com 部署中心为**每条路径**都写了「Deployment contract」——成熟度、验证版本/commit、验证日期、适用边界（Good fit / Not a fit）、可复制的安装-启动-健康检查-冒烟命令、运维与容量要点、安全边界、**已知限制**、证据链接（验证记录 / PR / issue）。这在开源语音项目里相当罕见，**选型时优先采信 contract 页而不是任何二手文章**。

**社区活跃**：FunASR 与 SenseVoice 均为高 star 项目、Apache-2.0、持续迭代（SenseVoice 的 llama.cpp runtime 于 2026-06 通过 PR #292 合入；部署中心验证日期普遍在 2026-07/08）。商业支持通道：阿里云百炼 / DashScope（`qwen3-asr-flash-realtime` 等）。


### 28. 适用场景与选型建议

#### 决策表：什么时候用 FunASR，什么时候用 Qwen3-ASR

| 维度 | **选 FunASR（SenseVoice / Paraformer）** | **选 Qwen3-ASR** |
|---|---|---|
| **首要目标** | 吞吐性价比、确定性低时延 | 最高精度、上下文理解 |
| **架构** | 非自回归（NAR），延迟与文本长度无关 | 自回归（AR），靠并发换吞吐 |
| **语种** | SenseVoice 5 语种；Fun-ASR-Nano 31 语种 | **52 语种 / 22 中文方言** |
| **时延** | 端到端 70 ms 级，TTFT ~70 ms | 单并发 RTF 0.0148（1.7B），TTFT ~92 ms |
| **吞吐** | GPU 170× 实时（SenseVoice-Small，待核实）；128 并发下 NAR 优势明显 | 高并发（128 并发）约 2000× 实时 |
| **硬件** | CPU / 边缘 / ONNX / GGUF / C++ 均可 | 需中高端 GPU（A100 / 4090 级），**CPU 解码极慢不推荐** |
| **情感 / 音频事件** | **SenseVoice 一次前向同时出**（唯一选择） | 不具备（情感仅云 API 有） |
| **歌词 / 歌唱 + BGM** | Fun-ASR-Nano 支持 | **Qwen3-ASR 强项**（官方主打） |
| **上下文偏置** | WFST 热词（外部偏置，权重需调） | **Prompt 上下文偏置**（天然、更灵活） |
| **时间戳** | Paraformer 原生；SenseVoice 需 1.3.29+ | 需 ForcedAligner-0.6B；**流式不支持** |
| **流式** | paraformer-online 每 600 ms（双引擎精修） | 单模型统一流式离线（**仅 vLLM 后端**） |
| **显存** | 几百 MB 级 | GB 级 + KV cache |
| **成本** | 极低（可 CPU / 边缘） | 高（需 GPU） |

#### 场景化推荐

| 场景 | 推荐 | 理由 |
|---|---|---|
| 海量呼叫中心整轨质检 | **SenseVoice / Paraformer + FunASR** | NAR 高吞吐 + 低于 100 ms 的确定性时延；成本可低到自建 CPU |
| 大规模短视频批量字幕 | **Paraformer + FunClip** | 吞吐优先，时延不敏感 |
| 实时字幕 / 直播 | **paraformer-zh-streaming**（双引擎精修） | 600 ms 增量，成熟稳定 |
| 智能客服（需情绪识别） | **SenseVoice** | 一次前向同时出情感/事件/语种/文本 |
| 会议转写 + 说话人分离 | **Paraformer + fsmn-vad + ct-punc + cam++** | 官方推荐组合，最成熟 |
| 多语言出海（31 语种内） | **Fun-ASR-Nano-2512** | 31 语种 + 方言 + 歌词；内置标点 |
| 多语言出海（需 52 语种/22 方言） | **Qwen3-ASR-1.7B + vLLM** | 覆盖最广，官方称 LibriSpeech / WenetSpeech 优于 Whisper-large-v3 与商业 API |
| 歌唱 / BGM 场景转写 | **Qwen3-ASR** | 官方主打能力 |
| 专业术语密集（医疗/法律/金融） | **Qwen3-ASR（prompt 偏置）** 或 **Paraformer contextual + WFST 热词** | 两条路都行；热词零成本可先试 |
| 中文方言（粤语/吴语/闽南语等） | **Qwen3-ASR**（22 方言） | Fun-ASR-Nano 覆盖部分方言 |
| 边缘 / 端侧 / 数据不出域 | **SenseVoice q8 GGUF（254 MB）+ sensevoice-server / llama-funasr-sensevoice** | 无 Python、无 GPU、单二进制；q8 精度不降反升 |
| 已有 Triton / GPU serving 平台 | **SenseVoice + Triton/TensorRT FP16** | 官方生产验证路径，批量吞吐最高 |
| 需要词级时间戳做字幕/对齐 | Paraformer 原生 或 Qwen3-ASR + ForcedAligner | 注意 Qwen3-ASR 流式不支持时间戳 |
| 边缘 + 需要 LLM 级理解 | Fun-ASR-Nano（Qwen3-0.6B）GGUF q4_K_M / q5_K_M | 0.6B 是 AR 路线里唯一能上边缘的 |

#### 一句话决策
- **先把 FunASR 当"调度层 + 工具箱"**，它同时能调 Paraformer、SenseVoice、Fun-ASR-Nano，甚至 Qwen3-ASR——**你不必二选一**。
- **精度/语种广度/上下文理解优先 → Qwen3-ASR（vLLM）**；**吞吐/时延/成本/边缘优先 → SenseVoice 或 Paraformer（FunASR）**。
- **需要情感与音频事件 → SenseVoice，没有替代品**。
- **同一套 OpenAI 兼容 `/v1/audio/transcriptions` 契约 + 网关路由**，是同时吃两家红利的最优工程形态。


### 29. 参考来源

**官方站点与文档**
- FunASR 官网（部署选择器 + 完整矩阵）：https://funasr.com/en ｜ https://www.funasr.com
- FunASR 部署中心（每条路径的 Deployment contract）：https://funasr.com/en/deploy/
- FunASR 使用教程（模型选择表、pipeline 组装、流式、ONNX 导出、FAQ）：https://modelscope.github.io/FunASR/zh/tutorial.html
- FunASR vLLM 推理引擎文档（适用模型表、三种入口、性能表、API 参考、FAQ）：https://modelscope.github.io/FunASR/zh/vllm.html
- FunASR 部署矩阵（ModelScope 文档）：https://modelscope.github.io/FunASR/deployment-matrix.html

**部署契约页（一手验证记录，优先采信）**
- Native FunASR on vLLM：https://funasr.com/en/deploy/vllm.html
- SenseVoice TensorRT / Triton：https://funasr.com/en/deploy/sensevoice-tensorrt.html
- llama.cpp / GGUF standalone：https://funasr.com/en/deploy/llama-cpp.html
- SenseVoice native OpenAI realtime server：https://funasr.com/en/deploy/sensevoice-native-server.html
- audio.cpp native Fun-ASR-Nano and SenseVoice：https://funasr.com/en/deploy/audio-cpp.html
- OpenAI-compatible private API：https://funasr.com/en/deploy/openai-api.html
- Realtime streaming and captions：https://funasr.com/en/deploy/realtime.html
- Containers and Kubernetes：https://funasr.com/en/deploy/containers.html
- ONNX / C++ CPU runtime：https://funasr.com/en/deploy/cpu-runtime.html
- Production rollout checklist：https://funasr.com/en/deploy/production.html

**GitHub**
- FunASR：https://github.com/modelscope/FunASR
- SenseVoice：https://github.com/FunAudioLLM/SenseVoice
- Fun-ASR：https://github.com/FunAudioLLM/Fun-ASR
- FunClip：https://github.com/modelscope/FunClip
- FunASR Triton GPU runtime（TensorRT 部署指南）：https://github.com/modelscope/FunASR/blob/main/runtime/triton_gpu/README.md
- SenseVoice llama.cpp runtime（DESIGN / README / BENCHMARKS）：https://github.com/FunAudioLLM/SenseVoice/tree/main/runtime/llama.cpp
- FunASR llama.cpp 运行时总览（体积/精度/速度对照）：https://www.funasr.com/llama-cpp.html

**模型权重**
- iic/SenseVoiceSmall（ModelScope）：https://www.modelscope.cn/models/iic/SenseVoiceSmall
- FunAudioLLM/Fun-ASR-Nano-2512：https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512
- FunAudioLLM/Fun-ASR-MLT-Nano-2512（多语 WER 表）：https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512
- FunAudioLLM/funasr（GGUF 权重）：https://huggingface.co/FunAudioLLM/funasr
- allendou/Fun-ASR-Nano-2512-vllm（社区 vLLM 转换）：https://huggingface.co/allendou/Fun-ASR-Nano-2512-vllm

**第三方交叉验证（二手，需谨慎）**
- Uni-ASR 论文（含多模型 CER/WER 汇总表，arXiv:2603.11123）：https://arxiv.org/pdf/2603.11123.pdf
- FunASR 与 QwenASR 对比研究报告（成本测算）：https://gitcode.csdn.net/6a22ac7c662f9a54cb7a0ab6.html
- 中文语音识别模型横评（速度/功能对比）：https://developer.cloud.tencent.com/article/2642961
- SenseVoice ONNX 量化实测：https://blog.csdn.net/weixin_42351520/article/details/160126743


---

<!-- item: production_deployment -->

## 生产加速与部署路径：TensorRT-LLM / ONNX Runtime / Triton

> **分类**：技术栈-生产部署　|　**已填字段**：33 / 38　|　**⭐ 标记为本轮重点的架构与训练字段**


### 1. 基本信息与版本演进

本条目覆盖「让 Qwen3-ASR / Qwen3-TTS / CosyVoice3 上生产」的**加速与部署路径**，包括四类：

1. **GPU 极致加速：NVIDIA Triton Inference Server + TensorRT / TensorRT-LLM**
   - **CosyVoice**：官方 `runtime/triton_trtllm` 目录，2025-08 由 **NVIDIA Yuekai Zhang** 贡献（CosyVoice roadmap 原文："2025/08 Thanks to the contribution from NVIDIA Yuekai Zhang, add triton trtllm runtime support and cosyvoice2 grpo training support"）。另有 DiT Token2Wav 变体（`README.DIT.md`，借用 Step-Audio2 的 DiT 解码器）。
   - **SenseVoice**：FunASR 官方 `runtime/triton_gpu`，**生产验证 2026-08-04**，TensorRT 10.0.1 / Triton 24.05。
2. **CPU / 跨平台：ONNX Runtime**
   - SenseVoice、Paraformer 导出 ONNX（opset 14）+ 动态 INT8 量化；`funasr-onnx` 运行时；Libtorch 走 `funasr-torch`。
3. **边缘 / 单二进制：llama.cpp / GGUF**
   - FunASR 官方 v0.2.1 九个发布包；SenseVoice q8 仅 **254 MB**；`sensevoice-server` 无 Python 运行时的 C++ 服务。
4. **Apple Silicon：MLX**
   - CosyVoice3 的 4bit / 8bit / 8bit-full / bf16 四档 bundle；CAM++ 走 CoreML Neural Engine（~14 MB）。

另覆盖：**国产卡适配**（海光 DCU、华为昇腾）、**服务化与传输**（Docker/Compose、FastAPI、gRPC、WebSocket、SSE、OpenAI Realtime WS、WebRTC/Opus、16 kHz 统一重采样）、**完整技术栈组件清单**、**评测工具链**。


### 2. 开源协议与商用边界

| 组件 | 许可 | 商用边界 |
|---|---|---|
| Triton Inference Server | 开源（BSD-3 系） | 可商用；注意与 GPU 驱动 / CUDA / TensorRT 的版本绑定 |
| TensorRT / TensorRT-LLM / TensorRT Edge-LLM | **NVIDIA 自有许可**（非 Apache） | 需遵循 NVIDIA 软件许可协议；再分发受限 |
| ONNX / ONNX Runtime | MIT / Apache-2.0 系 | 无限制 |
| llama.cpp / ggml | MIT | 无限制；FunASR GGUF 全家桶官方声明"开源、可商用" |
| MLX / mlx-examples | MIT | 无限制 |
| FunASR / SenseVoice / CosyVoice / Qwen3-ASR / Qwen3-TTS | **Apache-2.0** | 无限制，可私有化 |
| vllm-ascend（昇腾插件） | Apache-2.0 | 可商用 |

**打包提醒**：NVIDIA 组件（TensorRT / CUDA runtime）在容器再分发时需遵守对应许可；私有化交付给客户前建议做一次依赖许可审计。


### 3. 模型变体与权重获取

本条目是部署路径层，不产出自有权重；下表是**路径 × 模型**矩阵（截至 2026-08-28）：

| 路径 | Qwen3-ASR | Qwen3-TTS | CosyVoice3 | SenseVoice | Paraformer |
|---|---|---|---|---|---|
| **Triton + TensorRT(-LLM)** | 待核实 | 待核实 | ✅ 官方 runtime（2025-08） | ✅ 官方生产验证（2026-08-04） | 待核实 |
| **TensorRT Edge-LLM** | ✅ FP16（0.6B/1.7B） | ✅ FP16（0.6B/1.7B CustomVoice） | 待核实 | ❌ | ❌ |
| **ONNX Runtime（CPU）** | ❌（AR） | 待核实 | ✅ 社区 14 个 ONNX 文件 ~3.8 GB | ✅ 官方 | ✅ 官方 |
| **Libtorch（C++）** | ❌ | 待核实 | 待核实 | ✅ 官方 | ✅ 官方 |
| **llama.cpp / GGUF** | ❌ | 待核实（CosyVoice3 有社区 GGUF） | 待核实 | ✅ q8 254 MB | ✅ q8 237 MB |
| **MLX（Apple Silicon）** | ❌ | ❌ | ✅ 4/8/bf16 四档 | ❌ | ❌ |

**CosyVoice3 ONNX 文件清单（社区导出，~3.8 GB，14 个文件）**：
`campplus.onnx`（28 MB，说话人嵌入）、`speech_tokenizer_v3.onnx`（969 MB）、`llm_backbone_initial/decode_fp16.onnx`（717 MB × 2）、`flow.decoder.estimator.fp16.onnx`（664 MB）、`hift_decoder_fp32.onnx`（70 MB）。依赖仅 `onnxruntime==1.18.0` + numpy / soundfile / librosa / transformers，纯 CPU 可推理。


### 4. 架构范式 ⭐

#### 1. Triton 的「模型仓库 + 调度器」范式
Triton 的核心不是算子加速，而是**把推理变成可运维的服务单元**：
- **model repository**：每个模型一个目录 + `config.pbtxt`（实例数、batching 策略、输入输出张量名与 dtype、dynamic batching 参数）。
- **Dynamic batching**：服务端聚合请求成批，对 NAR 模型（SenseVoice）是纯收益；对 AR 模型（CosyVoice LLM）要配合 `max_queue_delay`。
- **Decoupled mode（解耦模式）**：`Decoupled=True` 时一个请求可以返回**多个响应**——这是**流式 TTS 的关键**。CosyVoice 官方 runtime 明确：流式设 `Decoupled=True`，离线设 `Decoupled=False`，**切换模式后必须重跑 stage 2（重建 Triton model repository）**。
- **Ensemble / BLS**：把多阶段（LLM → token2wav）串成一个对外模型，客户端只看到一次调用。
- **多实例并发**：`instance_group` 配置实例数，匹配并发需求。

#### 2. TensorRT 的「离线编译 + 精度校准」范式
- **不是运行时优化，是离线 AOT 编译**：ONNX → TensorRT engine（`.plan`），编译期做层融合（Conv+BN+ReLU）、kernel auto-tuning、精度校准、显存优化。
- **Optimization profile 必须贴合生产分布**：min/opt/max shape 决定 kernel 选择。SenseVoice 官方给的 profile 是 `batch 1/8/16`、`frames 1/512/4096`。**profile 与实际流量不匹配会显著掉速。**
- **不可移植**：`.plan` 与 GPU 架构 + TensorRT 版本强绑定。这是最大的运维成本。

#### 3. ONNX Runtime 的「图优化 + 动态量化」范式
- 导出期只做格式转换（`torch.onnx.export`，opset 14，dynamic axes）；**量化发生在加载时**（`quantize=True` 触发 ORT 内建动态量化：权重 QUInt8、主要作用于 MatMul、输出节点与 bias 不量化）。
- 运行时可调：`ORT_ENABLE_ALL`（常数折叠 + 算子融合）、`intra_op_num_threads`（建议 = 物理核数）、`arena_extend_strategy` / `cpu_mem_limit` / `gpu_mem_limit`。
- **ORT 的价值不是峰值性能，而是"无 PyTorch、跨平台、可预测"**——部署包体积与依赖复杂度降一个量级。

#### 4. GGUF / llama.cpp 的「无依赖单二进制」范式
- 量化在**转换期**离线完成（f16 / q8 / q5_K_M / q4_K_M），运行时只是 ggml 前向。
- 内置前端（kaldi fbank + LFR）、内置 VAD（FSMN）、内置 detok → **一个二进制、无 .so 依赖、无 Python**。
- 适用：笔记本、边缘盒子、嵌入式 C/C++ 应用、数据不出域场景。

#### 5. MLX 的「统一内存 + 量化 bundle」范式
- Apple Silicon 上 CPU/GPU 共享内存，无需显式拷贝；MLX 的 `Linear → QuantizedLinear` 在加载时按 `.scales` 是否存在做逐路径替换。
- **分档 bundle**：量化配置写进 `config.json`，一个下载包对应一种精度/体积权衡。
- 说话人编码器（CAM++）走 **CoreML + Neural Engine**（~14 MB FP16），把最轻但每次克隆都要跑的一步放到 ANE 上。


### 5. 模块拆解 ⭐

#### CosyVoice Triton + TensorRT-LLM runtime 的目录结构（官方 `runtime/triton_trtllm`）
```
runtime/triton_trtllm/
├── model_repo/                    # Triton 模型仓库
├── scripts/                       # 转换与构建脚本
├── Dockerfile.server              # soar97/triton-cosyvoice:25.06
├── docker-compose.yml             # 单卡聚合部署
├── docker-compose.dit.yml         # DiT Token2Wav 变体
├── run.sh                         # 0–6 阶段编排
├── run_stepaudio2_dit_token2wav.sh# -1–8 阶段编排（DiT，含 disaggregated）
├── client_grpc.py / client_http.py
├── offline_inference.py / streaming_inference.py
├── token2wav.py / token2wav_dit.py
├── README.md / README.DIT.md
└── requirements.txt
```

**`run.sh` 的阶段编排（这是整套 runtime 最值得抄的设计）**：
| Stage | 内容 |
|---|---|
| 0 | 从 HuggingFace 下载 cosyvoice-2 0.5B 模型 |
| 1 | HF checkpoint → TensorRT-LLM 格式转换 + 构建 TensorRT 引擎 |
| 2 | 生成 Triton model repository，并按 `Decoupled=True`（流式）/ `False`（离线）配置 |
| 3 | 启动 Triton Inference Server |
| 4 | 单句 HTTP 客户端测试（离线模式） |
| 5 | gRPC 基准客户端（`streaming` / `offline`） |
| 6 | 离线推理基准测试 |

用法：`bash run.sh <start> <end>`，例如 `bash run.sh 0 3` 跑完"下载→转换→建仓→启动"。

**DiT 变体（`run_stepaudio2_dit_token2wav.sh`）额外阶段**：
| Stage | 内容 |
|---|---|
| -1 | 克隆 Step-Audio2 与 CosyVoice 仓库 |
| 0 | 下载 cosyvoice2_llm、CosyVoice2-0.5B、Step-Audio-2-mini |
| 1 | LLM → TensorRT-LLM 引擎 |
| 2 | 建仓（含 cosyvoice2_dit 与 token2wav_dit 配置） |
| 3 | 启动 Triton（Token2Wav）+ `trtllm-serve`（CosyVoice2 LLM） |
| 4 | gRPC 基准 |
| 5 | 离线 TTS 基准 |
| 6 | Step-Audio2-mini DiT Token2Wav 单独推理 |
| **7** | **分离式部署（disaggregated）：LLM 在 GPU 0，Token2Wav 在 GPU 1–3** |
| 8 | 分离式部署的基准客户端 |

#### SenseVoice Triton 模型仓库（`runtime/triton_gpu/model_repo_sense_voice_small`）
- `encoder/1/model.plan`（TensorRT 引擎）+ `encoder/config.pbtxt`（从 `config.pbtxt.tensorrt` 复制）
- `scoring/chn_jpn_yue_eng_ko_spectok.bpe.model`（**需手工下载并校验 SHA-256**）
- 输入张量：`WAV [B, T]`、`WAV_LENS [B,1]`（int32）、`LANGUAGE [B,1]`（int32）、`TEXT_NORM [B,1]`（int32）
- 输出张量：`TRANSCRIPTS`（bytes）

#### CosyVoice3 在 MLX 上的组件（speech-swift 文档可核）
| 组件 | 架构 | 精度 | 体积 |
|---|---|---|---|
| **LLM** | Qwen2.5-0.5B，24 层，896 hidden，14 查询头 / 2 KV 头（GQA） | int4 / int8 / bf16 | 388 MB (4bit) / 640 MiB (8bit) |
| **DiT**（Flow Matching） | 22 层 DiT，1024 hidden，16 heads，**10 步 Euler ODE**，CFG 0.7，AdaLN | int4 / int8 / bf16 | 186 MB (4bit) / 634 MiB (8bit, bf16 DiT 变体) |
| **HiFi-GAN vocoder** | NSF（8 谐波）+ F0 预测 + 上采样 [8,5,3] + ISTFT(hop=4, n_fft=16)，Snake 激活 | **fp32**（weight-norm 已折叠） | 79 MB |
| **S3-Tokenizer-v3** | 12 层 Conformer + FSMN + FSQ（242M 参数），25 Hz | **bf16** | 462 MB |
| **CAM++**（说话人） | CoreML，Neural Engine，192 维 → 仿射投影 192→80 条件化 DiT | FP16 | **~14 MB**（`aufklarer/CamPlusPlus-Speaker-CoreML`） |


### 6. 音频编码器 ⭐

- **SenseVoice / Paraformer**：kaldi 80-mel fbank + LFR → `[T, 560]`。ONNX 导出时 feats 的 dynamic axes 通常设为 `batch_size=2, seq_len=dynamic`。
- **CosyVoice3**：**S3-Tokenizer-v3**（12 层 Conformer + FSMN + FSQ，242M 参数）做参考音频编码，bf16 约 462 MB，**每个 voice profile 只跑一次**（所以保持 bf16 而非量化，精度优先于体积）。
- **Qwen3-ASR**：Whisper 风格 audio encoder + adaptor，输出 prompt embedding 给 Qwen3 LLM；**在 FunASR vLLM 引擎里 encoder 仍在 PyTorch，不在 vLLM**。
- **部署侧的工程含义**：encoder 与 decoder 常常跨框架（PyTorch encoder + TRT-LLM decoder），因此**显存预算要分两套算**，且 encoder 的 device 需要显式指定。


### 7. 语音 Tokenizer 与编解码器 ⭐

- **CosyVoice3**：**FSQ，词表 6561，25 Hz**（LLM 侧）；DiT 内部上采样 25 Hz → 50 Hz 出 80-band mel；HiFi-GAN 出 24 kHz。
- **Qwen3-TTS**：12 Hz 多码本（Talker 出第 1 个码本，Code Predictor 出剩余 15 个），24 kHz 输出；TensorRT Edge-LLM 导出时 codec embedding 单独存 `embedding.safetensors`，`lm_heads.safetensors` 在 code_predictor 目录。
- **SenseVoice**：无神经 codec；CTC 词表 25055，tokenizer `chn_jpn_yue_eng_ko_spectok.bpe.model`。
- **部署陷阱**：Triton / GGUF 路径下 tokenizer 往往是**单独下载的文件**，路径或版本错了**不报错、只产出乱码**。SenseVoice Triton 路径官方给了 SHA-256 校验步骤，**生产必须照做**。


### 8. LLM 主干 ⭐

- **CosyVoice3**：**Qwen2.5-0.5B**（24 层、896 hidden、14Q/2KV 头）。走 vLLM / TensorRT-LLM 时保留 speech token embedding 与 speech head（`export_cosyvoice2_vllm()` 做模型手术：保留原组件 → 用 CosyVoice 的 speech token embedding 替换 Qwen2 的 text embedding → 把 speech token 预测头设为输出层）。
- **CosyVoice2/3 的采样稳定性**：**RAS（Repetition Aware Sampling）**，2024-08 引入，防止长文本自回归重复。vLLM-Omni 的 TensorRT 优化里做了**向量化 RAS 采样**。
- **Fun-ASR-Nano**：Qwen3-0.6B；**Qwen3-ASR**：Qwen3 0.6B/1.7B。
- **量化策略**：MLX bundle 对 LLM 的 attention projections、MLP、speech head 全部做 group quantization（group_size=64）。


### 9. 声学解码器 ⭐

- **CosyVoice3 DiT Flow Matching**：22 层 / 1024 hidden / 16 heads / **10 步 Euler ODE** / CFG rate 0.7 / AdaLN 时间步条件；token 上采样 25 Hz → 50 Hz。
- **加速路径对比**：
  | 路径 | 做法 | 效果 |
  |---|---|---|
  | CosyVoice 官方 `export_onnx.py` | 把 flow.decoder.estimator 单独导出 ONNX（opset 18，dynamic axes，10 次随机 shape 一致性测试，rtol=1e-2 / atol=1e-4） | 便于 ORT 推理 |
  | CosyVoice 官方 `convert_onnx_to_trt()` | ONNX → TensorRT engine（min/opt/max shape = batch 1/1/4，seq 16/256/512） | flow decoder **约 4×**（官方口径，deepwiki 汇总） |
  | Triton+TRT-LLM 官方 runtime | 整链路（LLM + token2wav）上 TRT-LLM | batch=16 RTF **0.0386** |
  | DiT 变体（Step-Audio2 Token2Wav） | 用 DiT 替换原 flow matcher，并可 disaggregated 到 GPU 1–3 | 见 README.DIT.md |
- **MLX 侧的流式**：**chunk-aware 因果掩码**，25 token/块（≈1 s 音频），目标首块 **~150 ms**。
- **延迟/质量权衡**：ODE 步数是最直接的旋钮（10 步是默认）；生产如需更低延迟可尝试减步数，但要重测音质。


### 10. 声码器

- **CosyVoice3 HiFi-GAN**：**Neural Source Filter（NSF），8 次谐波 + F0 预测 + 3 级上采样 [8,5,3]=120× + ISTFT(hop=4)=480×**，Snake 激活，n_fft=16，输出 **24 kHz**。MLX 导出为 fp32、weight-norm 已折叠（`w = g * v / ||v||`），79 MB。ONNX 版为 `hift_decoder_fp32.onnx`（70 MB）。
- **Qwen3-TTS**：code2wav 内置波形合成（因果 codec），24 kHz。
- **部署含义**：vocoder 是**流式链路的最后一个瓶颈**。vLLM-Omni 提供 `active_stream_window: 2` 在 vocoder-bound 过载时限流；CosyVoice Triton 侧则靠 BLS 实例数与 `Decoupled=True` 保连续播放。
- **批处理提示**：Triton 路径下有 `FlashCosyVoice`（`yuekaizhang/FlashCosyVoice`，`-b trt` 分支）专门做 **token2wav batching**，离线批量场景应装上。


### 11. 上下文长度与 I/O 约束

- **SenseVoice Triton**：输入张量固定为 `WAV` / `WAV_LENS` / `LANGUAGE` / `TEXT_NORM`，输出 `TRANSCRIPTS`；optimization profile 覆盖 batch 1–16、frames 1–4096（`max_seq_len=4096` 导出）。
- **SenseVoice 原生 C++ server**：`--max-audio-seconds 300`、`--max-connections 4`、`--threads 8`；WebSocket 客户端必须送 **16 kHz 单声道 PCM16** 的 base64 分块。
- **CosyVoice3**：24 kHz 输出；Triton 路径需 `ref_audio` + `ref_text`（零样本克隆）。
- **Qwen3-TTS / CosyVoice3（vLLM-Omni）**：`max_new_tokens` 1–4096（默认 2048）；`speed` 0.25–4.0（流式不支持）；`response_format` wav|mp3|pcm|flac|aac|opus。
- **统一约定（强烈建议）**：ASR 输入 **16 kHz / 单声道 / PCM16**；TTS 输出按模型采样率（Qwen3-TTS & CosyVoice3 = 24 kHz）。


### 12. 微调支持

部署路径层不提供微调，但有几条与微调/再训练直接相关的工程事实：

- **CosyVoice 官方在 2025/08 同时合入了 triton trtllm runtime 与 **cosyvoice2 GRPO 训练支持**——训练与部署是同一批贡献，说明这条 runtime 是为「训练后可直接上线」设计的。
- **微调后换引擎 = 重建一切**：TensorRT engine 是从 checkpoint 编译出来的，**checkpoint 变了必须重跑 stage 1（重建引擎）**，并重验精度、显存、容量。这是 TRT 路径最大的迭代成本。
- **ONNX 导出也要重做**：`model.export()` 产出绑定当前权重，微调后必须重新导出 + 重新量化 + 重跑一致性测试。
- **GGUF 同理**：`export_sensevoice_gguf.py` 从 `model.pt` 转换，微调后重跑。
- **对比之下 vLLM 路径最轻**：checkpoint 换了直接 `vllm serve` 指过去即可（AR 模型）。**如果迭代频繁，优先选 vLLM 而不是 TRT。**


### 13. 语种与方言覆盖

| 模型 | 语种覆盖 | 部署侧注意事项 |
|---|---|---|
| **CosyVoice3** | **9 语种**（中 / 英 / 日 / 韩 / 德 / 西 / 法 / 意 / 俄）+ **18+ 中国方言**（粤语、四川话、东北话、上海话、闽南语等） | MLX bundle 保持 bf16 的 S3-Tokenizer 以保克隆质量；vLLM-Omni 上**只有 voice cloning，无内置音色预设** |
| **Qwen3-TTS** | 10 语种（SGLang-Omni 标注）；vLLM-Omni 的 `language` 可选值**启动时从模型 config.json 动态加载** | 非法 language 会被校验拒绝 |
| **SenseVoice** | 5 语种（中/粤/英/日/韩） | Triton 输入用 `LANGUAGE` 张量传 int32（0 = auto/zh） |
| **Qwen3-ASR** | 52 语种 / 22 方言 | 昇腾 vllm-ascend 与海光 DCU 均已实测 |

**跨语种克隆**：CosyVoice3 支持（MLX 文档明确 zero-shot reference conditioning）；CosyVoice 系在方言场景基本没有替代品。


### 14. 流式能力

#### Triton 侧：Decoupled mode 是流式的前提
- **流式 TTS**：`Decoupled=True`，一个请求返回多个响应（多段音频块）。
- **离线 TTS**：`Decoupled=False`。
- **切换模式必须重跑 stage 2**（重建 model repository），这是最容易漏的一步。
- CosyVoice 官方实测（单 L20 GPU，`yuekai/seed_tts_cosy2` 数据集，26 组 prompt/目标文本对，约 170 s 音频）：

**流式 TTS（首块延迟）**
| 模式 | 并发 | 平均延迟 (ms) | P50 (ms) | RTF |
|---|---|---|---|---|
| Streaming, `use_spk2info_cache=False` | 1 | 220.43 | 218.07 | 0.1237 |
| Streaming, `use_spk2info_cache=False` | 2 | 476.97 | 369.25 | 0.1022 |
| Streaming, `use_spk2info_cache=False` | 4 | 1107.34 | 1243.75 | 0.0922 |
| **Streaming, `use_spk2info_cache=True`** | **1** | **189.88** | **184.81** | **0.1155** |
| Streaming, `use_spk2info_cache=True` | 2 | 323.04 | 316.83 | 0.0905 |
| Streaming, `use_spk2info_cache=True` | 4 | 977.68 | 903.68 | 0.0733 |

→ **固定说话人务必开 `use_spk2info_cache=True`**：并发 1 时延迟降 13.8%（220.43 → 189.88 ms）、RTF 降 6.6%。

**离线 TTS（整句延迟）**
| 模式 | 并发 | 平均延迟 (ms) | P50 (ms) |
|---|---|---|---|
| Offline, `Decoupled=False`, `use_spk2info_cache=False` | 1 | 758.04 | 615.79 |
| Offline, `Decoupled=False`, `use_spk2info_cache=False` | 2 | 1025.93 | 901.68 |
| Offline, `Decoupled=False`, `use_spk2info_cache=False` | 4 | 1914.13 | 1783.58 |

#### TensorRT Edge-LLM 侧：chunk-level 流式
`qwen3_tts_inference --streaming --chunkFrames=10`：Code2Wav 不再等完整序列，而是 Talker 边生成边声码化，每个请求有独立 `onChunkReady` 回调。
- **RVQ codes 与分块数无关、bit-exact**；
- **WAV 在不同 chunkFrames 下不字节一致**（块间无 left-context overlap，块边界 PCM 损失少量上下文）。
- 当前 runtime 要求 `--maxBatchSize 1`。

#### vLLM-Omni 侧（对照）
CosyVoice3 在 vLLM-Omni 上 `async_chunk: true` **默认开启**（deploy YAML），`--no-async-chunk` 才走同步路径。

#### 通用流式工程要点（三条）
1. **前导静音 trim**（~80 ms）——首段 PCM 常含静音，不裁掉报出来的 TTFA 会显著好于体感。
2. **帧累积 ramp**——首块小（降 TTFA）+ 后续块逐步放大（保连续播放）。
3. **underrun 率**必须和 TTFA 一起测——vLLM-Omni 默认配置下 Qwen3-TTS 1 RPS 就 100% underrun。


### 15. VAD 与后处理

- **VAD 挂在哪一层**：生产上 VAD 几乎总在**网关/服务层**，不在推理引擎里。例外是 FunASR 的 `serve_realtime_ws.py`（动态 VAD 集成在服务端）与 `sensevoice-server`（C++ 内置 FSMN-VAD）。
- **FSMN-VAD 的 GGUF 化**：`fsmn-vad.gguf` 与 ASR 模型一起分发，`sensevoice-server` 启动时 `-vad model/fsmn-vad.gguf`；llama.cpp 二进制用 `--vad ./gguf/fsmn-vad.gguf`。
- **标点 / ITN**：Triton 与 GGUF 路径都在模型输出侧（SenseVoice 自带 `<|withitn|>` 标签）；Paraformer 需外接 `ct-punc`。
- **说话人**：`campplus.onnx`（28 MB，CosyVoice ONNX 版）或 CAM++ CoreML（~14 MB，MLX 版，走 Neural Engine）。
- **时间戳**：SenseVoice 需 FunASR ≥ 1.3.29（恢复 VAD 分段时间戳）；Qwen3-ASR 需 ForcedAligner-0.6B；audio.cpp 路径**无词级时间戳**。
- **语种返回**：FunASR ≥ 1.3.27 的 `verbose_json.language` 返回检测语种。


### 16. 音色克隆

| 路径 | 克隆方式 | 工程要点 |
|---|---|---|
| **CosyVoice3 + vLLM-Omni** | `ref_audio`（本地路径 / HTTP URL / base64 data URL）+ `ref_text`，**无内置音色预设** | 3 秒参考音频即可；每次请求都要带 ref |
| **CosyVoice3 + Triton** | 同上 | **固定说话人务必 `use_spk2info_cache=True`**（延迟 −13.8%）；说话人缓存 `spk2info.pt` 约 200 MB/人 |
| **CosyVoice3 + MLX** | S3-Tokenizer-v3 抽取 voice profile（`extractVoiceProfile`），**必须提供参考音频的文本转录**才能拿到最佳克隆质量 | CAM++ 只需要加载一次，多说话人对话场景复用 |
| **Qwen3-TTS + vLLM-Omni** | `task_type=Base` + `ref_audio` + `ref_text`；`--mode-tag icl`（默认）或 `xvec_only` | **预计算音色**（`precompute_custom_voice.py`）是生产首选；或运行时 `POST /v1/audio/voices` 上传 |
| **Qwen3-TTS + TensorRT Edge-LLM** | CustomVoice（预设说话人），**不支持 Base 克隆** | 边缘场景只能走预设音色 |
| **Qwen3-TTS + NVIDIA Dynamo** | 仅 CustomVoice / VoiceDesign，**Base 不支持** | — |

**通用工程建议**：把"参考音频 → 音色 embedding / voice profile"这一步**离线预计算并缓存**，不要每次请求重跑。这在所有路径上都是最大的单点收益（Triton 13.8%、vLLM-Omni 免去参考编码）。


### 17. 情感与风格控制

- **CosyVoice3 指令控制**：MLX runtime 内置 **8 个情感/风格标签**（happy/excited、sad、angry、whispers/whispering、laughs/laughing、calm、surprised、serious），未知标签作为自由指令透传；文本格式 `{instruction}(token 151646){text_to_synthesize}`；`--cosy-instruct` 设全局默认指令。
  **重要**：零样本克隆时，若使用非默认指令，走 CosyVoice 的 **instruct2 布局**——LLM 接收带框指令 + 目标文本，参考音频仍作为 voice anchor 挂在 Flow 模型上；未加样式的路径则继续使用参考转录 + speech-token 上下文，以最大化说话人保真度。
- **多说话人对话**：`DialogueParser` 解析 `[S1] text [S2] text` 标签 → `DialogueSynthesizer` 逐段合成 + 每段独立 embedding；`--turn-gap`（默认 0.2 s）与 `--crossfade` 可调。
- **长文本克隆**：按句切分，**每段前对 MLX 播种**（`--seed`）保证可复现；段间用边缘淡入淡出拼接，避免爆音。prompt 区域在 **mel 域**切掉（与上游一致），不在音频域裁剪。
- **Qwen3-TTS**：`instructions` 自然语言指令（CustomVoice）/ 描述式音色设计（VoiceDesign）；vLLM-Omni `speed` 0.25–4.0 用相位声码器变速不变调（流式不支持）。
- **社区反馈**：CosyVoice3 的指令控制属"基础级"（只能说"用开心的语气 / 用四川话"这类粗粒度指令），细粒度情感标签能力弱于 Fish Speech；ChatTTS 的对话自然度更高（但 CC BY-NC 不可商用）。


### 18. 鲁棒性

- **TensorRT plan 不可移植**（反复强调）：换 GPU 架构或 TensorRT 版本必须重建并重验精度、显存、容量。**这是 TRT 路径最大的运维风险**，建议在 CI 里绑定「构建机 GPU 型号 == 生产 GPU 型号」。
- **引擎构建失败排查**：先确认导出用的是 `quantize=False`（`model_quant.onnx` 不被 SenseVoice TRT 路径支持）；再跑 ONNX checker 验张量名与 dtype；Triton 加载失败时确认 `model.plan` 与 `config.pbtxt.tensorrt` 在同一 encoder 版本目录。
- **容量变量（官方列出，压测时必须覆盖）**：音频时长、语种、通道数、VAD 分段分布；并发、排队时间、warmup、模型缓存状态；确切的硬件/驱动/运行时/线程配置。
- **热词库容量**：单实例 WFST 热词建议 ≤ 10 万条，超出需走分布式/专用方案。
- **WebRTC 抗弱网**：Opus 有状态，**每个 SSRC 一个解码器**；异常时重建解码器 + 返回一帧静音 + 跳过后续 3 包让流重新同步，避免一个坏包产出数秒噪音；jitter buffer 设 20–30 ms（远低于 VoIP 的 60–100 ms）。
- **重采样混叠**：48 kHz → 16 kHz **必须先低通**（soxr），否则 8 kHz 以上分量折回可听带污染 ASR；每个流一个 resampler 实例，避免滤波器状态串扰。
- **barge-in（打断）**：播放期间持续对用户音频跑 VAD；确认说话后立刻停播放 + 取消 TTS + 取消 LLM 生成 + 缓存新语音。这是"专业语音助手"与"难受的语音助手"的分水岭。
- **安全边界**：认证、TLS、限流、审计、上传大小/时长/声道/采样率/并发限制**全部放在网关层**；运行时只负责推理。只加载来源可验、带 digest 的 ONNX / TRT plan / tokenizer / model repository。


### 19. ASR 指标

本条目是部署路径层，**量化/加速对精度的影响**才是关键指标：

#### GGUF 量化精度（FunASR 官方，184 集中文测试集，micro-avg CER）
| 模型 | 档 | 体积 | CER | 速度 |
|---|---|---|---|---|
| SenseVoice encoder | f16 | 470 MB | 8.01% | 23× |
| **SenseVoice encoder** | **q8** | **254 MB** | **7.99%** | **27×** |
| Paraformer encoder | f16 | 435 MB | 9.85% | 22× |
| Paraformer encoder | q8 | 237 MB | 9.78% | 27× |
| Fun-ASR-Nano LLM | q8_0 | 805 MB | 8.30% | 6.0× |
| Fun-ASR-Nano LLM | q5_K_M | 551 MB | 8.25% | 5.7× |
| Fun-ASR-Nano LLM | q4_K_M | 484 MB | 8.35% | 6.1× |

→ **q8 编码器体积减半、CER 反而略降**；Nano LLM 三档差异 < 0.1%。**q8 是性价比最优点。**

#### ONNX 动态 INT8（社区实测，待一手核实）
SenseVoice 量化前后：模型 890 MB → 230 MB（−74%）、内存 1.2 GB → 350 MB（−71%）、加载 2.1 s → 0.8 s（−62%）；CER 平均 **+1.1%**（清晰 +0.5% / 嘈杂 +1.2% / 方言 +0.9% / 多语混合 +1.5%）。

#### 精度对齐（国产卡）
- 海光 DCU：官方/社区实测 **DCU 与 GPU 推理精度完全一致**（Qwen3-ASR），换平台无需重新校准（**社区口径，待一手核实**）。
- 昇腾：社区实测昇腾版与 CUDA 版的 WER 差异 **< 0.8%**（**社区口径，待一手核实**）。

#### 绝对精度基准
见 funasr_sensevoice.json 的 asr_metrics 字段（含 AISHELL / LibriSpeech / FLEURS / WenetSpeech 的 CER/WER 汇总）。


### 20. TTS 指标

- **CosyVoice3（Seed-TTS-eval 口径）**：中文 CER **1.16%**（比 CosyVoice2 降 44%）、英文 WER **2.02%**、中文 SIM **78.0%**（**社区汇总数据，需以官方表为准**）。
- **Qwen3-TTS 1.7B（Seed-TTS-eval）**：EN WER **1.5%**、ZH CER **1.33%**。
- **MOSS-TTSD v1.0（TTSD-eval）**：ZH ACC 95.87%、ZH WER 4.85%。
- **量化对 TTS 质量的影响**：MLX 文档明确 **8-bit（int8 LLM + bf16 DiT）比 4-bit 在长文本上音质更干净、文本漂移更少**；4-bit 是体积优先档（~1.1 GB）。
- **部署侧要额外报的指标**：**流式场景必须报 TTFA 与 streaming RTF**；固定 chunk 配置后再横向比（不同 chunkFrames 的 WAV 不字节一致）。
- **评测陷阱**（详见 eval_toolchain）：换监听模型 = 换尺子；SIM 不能混用 embedding 网络；MOS 预测器高分区饱和。


### 21. 延迟与吞吐

#### A. CosyVoice Triton + TensorRT-LLM（官方 README，单 L20 GPU，26 组 prompt/目标文本对，约 170 s 音频）

**离线推理模式：HuggingFace backend vs TensorRT-LLM**
| Batch | HF llm_time (s) | HF total (s) | HF RTF | TRT-LLM llm_time (s) | TRT-LLM total (s) | TRT-LLM RTF |
|---|---|---|---|---|---|---|
| 1 | 39.26 | 44.31 | 0.2494 | **12.46** | **17.31** | **0.0987** |
| 2 | 30.54 | 35.62 | 0.2064 | 7.64 | 12.65 | 0.0739 |
| 4 | 18.63 | 23.90 | 0.1421 | 4.89 | 9.38 | 0.0539 |
| 8 | 11.22 | 16.45 | 0.0947 | 2.92 | 7.23 | 0.0418 |
| **16** | 8.42 | 13.78 | 0.0821 | **2.01** | **6.63** | **0.0386** |

→ **batch=1 时 RTF 从 0.2494 降到 0.0987（2.5×）；batch=16 时 0.0821 → 0.0386（2.1×）；LLM 部分加速 3.2–4.2×。**

**客户端-服务端模式**（见 streaming 字段的流式表与离线表）：
- 流式首块：并发 1 **189.88 ms**（开说话人缓存）/ 220.43 ms（不开）；并发 4 时劣化到 977.68–1107.34 ms。
- 离线整句：并发 1 **758.04 ms**；并发 4 时 1914.13 ms。

**综合 RTF 排序（官方 benchmark 汇总）**：TRT-LLM batch=16 **0.0386** < TRT-LLM batch=1 0.0987 < 离线模式 0.0891 < 流式+缓存 0.1155 < 流式无缓存 0.1237 < HF batch=16 0.0821。

#### B. CosyVoice3 各部署路径横向（社区汇总，部分待一手核实）
| 路径 | 硬件 | RTF | 备注 |
|---|---|---|---|
| **vLLM-Omni + TensorRT 优化** | H100/A100/L40S | **0.12** | TTFP ~2800 ms → **~200 ms（14×）**；吞吐 0.21 → 1.08 req/s（5.1×）。优化点：speaker embedding 从 CPU ONNX-RT 搬到 GPU TRT、LLM 开 CUDA Graph + 向量化 RAS、flow-matching 解码器迁 TRT FP16、去掉每次请求重建 tokenizer（原 452 ms） |
| **Triton + TRT-LLM（batch 16）** | L20 | **0.0386** | 官方数据，吞吐最高 |
| **llama-cpp GGUF** | T4 (FP16) | **0.45** | PyTorch 原版 1.17 → GGUF 0.45（2.6×） |
| **Docker（neosun/cosyvoice）** | L40S | **0.72–0.82** | TTFB ~1.2 s；二次调用因 embedding 缓存快 53% |
| **MLX 8-bit** | Apple Silicon | **0.40** | 6.16 s 音频 / 2.44 s 生成（speech-swift debug CLI，**官方标注为兼容性 sanity check，非完整基准**） |

#### C. SenseVoice 加速
- **Triton + TensorRT FP16（V100）**：相对 CPU 基线约 **526× 加速**（官方 README 口径，含 kernel 融合 + dynamic batching + CUDA 显存优化）；**FP16 是官方推荐档，FP8 当时仍在开发中**。
- **ONNX Runtime CPU（10 s 音频，社区实测，待一手核实）**：
  | 硬件 | FP32 | INT8 量化 | INT8 + ORT 调优 |
  |---|---|---|---|
  | Intel i7-11800H | 210 ms | 70 ms | **55 ms** |
  | Raspberry Pi 4B | — | — | 420 ms |
  | NVIDIA T4 | — | — | 30 ms |
  | Jetson Xavier NX | — | — | 85 ms |

#### D. 国产卡实测
| 平台 | 配置 | 模型 | 实测 |
|---|---|---|---|
| **海光 DCU K100_AI 64GB** | vLLM 0.15.1 + DTK 26.04 镜像（Python 3.10） | Qwen3-ASR-1.7B | **约 200 ms 完成识别** |
| **海光 DCU K100_AI 64GB** | vLLM 0.18.1 + vLLM-Omni 0.18.0 + DTK 26.04 | Qwen3-TTS-12Hz-1.7B-Base | **平均延迟 < 97 ms** |
| **海光 DCU K100_AI 64GB** | 同上 | Qwen3-TTS-12Hz-1.7B-VoiceDesign | 平均语音合成延迟 **< 500 ms** |
| **海光 DCU K100_AI 64GB** | vLLM 0.21 + DTK 26.04（与 ASR/TTS 同卡） | Qwen3.5-9B | 约 **31 token/s** |
| **昇腾 910B** | CANN 8.5.1 / torch_npu 2.9.0 / vllm-ascend | Qwen3-ASR-1.7B（BF16） | 单卡 910B 64GB 或 Atlas 300I DUO 48GB 可部署 |
| **昇腾 910B** | `quay.io/ascend/vllm-omni:v0.14.0` | Qwen3-TTS-12Hz-0.6B-Base | docker-compose 一键起，`/v1/audio/speech`（端口 18005） |

（**注**：海光与昇腾的以上延迟数字来自 CSDN 社区实践文章，**未提供压测方法与并发条件，标注为待一手核实**；昇腾另有社区实测"10 s 音频约 2.1 s，RTF≈0.21，显存 12 GB"，同样待核实。）

#### E. Apple Silicon / MLX
- **CosyVoice3-0.5B-MLX-8bit 实测**：6.16 s 音频 / 2.44 s 生成 → **RTF 0.40**（speech-swift debug CLI，Apple Silicon；**官方明确标注"numbers are sanity checks for runtime compatibility, not a final benchmark suite"**）。
- **MLX 阶段延迟目标**：chunk-aware 因果 DiT，25 token 块（≈1 s 音频），**目标首块 ~150 ms**。
- **用户提供但未能核实的数值**（标注待核实）：M2 Max 上 RTF 约 0.5；各阶段延迟 LLM ~13 ms/token、DiT 370–520 ms、HiFi-GAN 50–170 ms。


### 22. 成本模型

- **Triton + TRT-LLM 的成本逻辑**：用**批量**换成本。CosyVoice 官方数据显示 batch 从 1 加到 16，RTF 从 0.0987 降到 0.0386（**单卡吞吐提升 2.6×**），意味着同样一张 L20 能服务的请求量翻 2.6 倍。**适合离线/批量场景（有声书、批量配音），不适合低延迟交互。**
- **vLLM-Omni 的成本逻辑**：用**调度**换成本。CosyVoice3 经 TRT 优化后吞吐 0.21 → 1.08 req/s（5.1×），单卡成本降到 1/5（**社区数据，待一手核实**）。
- **GGUF / CPU 的成本逻辑**：用**硬件降级**换成本。SenseVoice q8（254 MB）纯 CPU 就有 27× 实时，意味着**可以完全不买 GPU**——对于边缘、私有化、数据不出域场景，这是决定性的。
- **MLX 的成本逻辑**：用**零增量硬件**换成本。Mac 本地跑（4-bit bundle 仅 1.1 GB），适合开发、内部工具、小规模个性化场景。
- **TTS 侧的整体成本水位**（交叉引用 inference_engines.json）：Nari Labs 在 1× H100 SXM 上做到 ~630 chars/s @ 10 RPS，满利用率折算约 **$2 / 1M 字符**（算力下限）；对比 ElevenLabs V3 $100 / 1M、Cartesia Sonic 3.5 $49 / 1M。
- **TRT 路径的隐性成本**（必须计入）：引擎构建时间（数分钟到数十分钟）、构建机与生产机 GPU 型号必须一致、每次模型/驱动/TensorRT 升级都要重建 + 重验。**迭代频繁的项目，这些成本会吃掉 TRT 的性能收益。**
- **决策建议**：先用 vLLM / vLLM-Omni 上线跑通并采集真实流量分布，**再用真实分布去定 Triton 的 optimization profile 与 batch 策略**——顺序反了会导致 TRT 引擎 profile 与实际流量不匹配，性能反而不如 vLLM。


### 23. 推理框架支持

#### 1. NVIDIA Triton + TensorRT / TensorRT-LLM

**CosyVoice（官方 runtime，2025-08，NVIDIA Yuekai Zhang 贡献）**
```
# 最快起步
docker compose up                                   # 单卡聚合
docker compose -f docker-compose.dit.yml up         # DiT Token2Wav 变体

# 自行构建
docker build . -f Dockerfile.server -t soar97/triton-cosyvoice:25.06
docker run -it --name cosyvoice-server --gpus all --net host \
  -v /mnt:/mnt --shm-size=2g soar97/triton-cosyvoice:25.06

# 容器内跑阶段 0→3（下载 → 转换建引擎 → 建仓 → 起服务）
bash run.sh 0 3
# 单句 HTTP 测试
bash run.sh 4 4
# gRPC 基准
bash run.sh 5 5                                     # 或 python3 client_grpc.py --num-tasks 2 \
                                                    #   --huggingface-dataset yuekai/seed_tts_cosy2 \
                                                    #   --split-name test_zh --mode [streaming|offline]
# 离线基准（需先装 FlashCosyVoice 做 token2wav batching）
git clone https://github.com/yuekaizhang/FlashCosyVoice.git /workspace/FlashCosyVoice -b trt
cd /workspace/FlashCosyVoice && pip install -e . && cd -
bash run.sh 6 6
```
- **OpenAI 兼容桥接**：`git clone https://github.com/yuekaizhang/Triton-OpenAI-Speech` → `python3 tts_server.py --url http://localhost:8000 --ref_audios_dir ./ref_audios/ --port 10086 --default_sample_rate 24000`。**注意：目前只有离线 TTS 模式与 OpenAI 兼容 server 兼容。**
- **切换 HF 后端**：`backend=hf`（调试用）。

**SenseVoice（FunASR 官方 `runtime/triton_gpu`，生产验证）**
见 funasr_sensevoice.json 的 inference_frameworks ② 段（完整命令）。核心三步：ONNX 导出（`quantize=False`）→ `build_sensevoice_tensorrt.py --precision fp16 --min-batch 1 --opt-batch 8 --max-batch 16 --min-frames 1 --opt-frames 512 --max-frames 4096 --workspace-gb 8` → `tritonserver --model-repository ... --pinned-memory-pool-byte-size=512000000 --cuda-memory-pool-byte-size=0:1024000000`。

**CosyVoice 内部的 TensorRT 加速（非 Triton 路径，也可单独用）**
```
# 1) flow decoder estimator → ONNX（opset 18，dynamic axes，10 次随机 shape 一致性测试）
python cosyvoice/bin/export_onnx.py
# 2) ONNX → TensorRT engine（min/opt/max = batch 1/1/4，seq 16/256/512）
convert_onnx_to_trt()   # cosyvoice/utils/file_utils.py
# 3) 加载（自动，当 load_trt=True）
CosyVoice3(..., load_trt=True)   # 用 TrtContextWrapper 管理执行上下文
```
官方口径：flow decoder estimator **约 4×**（相对 PyTorch）。另有 `export_jit.py`（TorchScript JIT：常数折叠、死代码消除、算子融合、类型特化）。

#### 2. ONNX Runtime
**导出与量化（SenseVoice / Paraformer，官方）**
```
from funasr import AutoModel
model = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0")
model.export(type="onnx", quantize=False)                 # 先导出 FP32 ONNX
# 量化：导出期或加载期
export_utils.export(model=rebuilt_model, quantize=True, **kwargs)   # 产出 model_quant.onnx
```
**推理（funasr-onnx）**
```
from funasr_onnx import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
model = SenseVoiceSmall("iic/SenseVoiceSmall", batch_size=10, quantize=True)
res = model(["audio.wav"], language="auto", textnorm="withitn")
res = [rich_transcription_postprocess(i) for i in res]
```
- 量化类型：**动态量化**；权重 **QUInt8**；目标算子以 **MatMul** 为主；**输出节点与 bias 不量化**；opset 14。
- 会话调优：`ORT_ENABLE_ALL`、`intra_op_num_threads`（= 物理核数）、`inter_op_num_threads`、`arena_extend_strategy=kSameAsRequested` + `cpu_mem_limit`（CPU 受限）、`gpu_mem_limit`/`device_id`（GPU 受限）。
- **Libtorch**：`model.export(type="torch")` → `funasr_torch.SenseVoiceSmall(model_dir, batch_size=10, device="cuda:0")`。
- **CosyVoice3 ONNX 版**（社区）：14 个文件 ~3.8 GB，依赖 `onnxruntime==1.18.0` + numpy/soundfile/librosa/transformers，纯 CPU 可跑；前端（speech tokenizer + speaker embedding）用 ORT 图优化（`ORT_ENABLE_ALL` + `intra_op_num_threads=1`）。

#### 3. llama.cpp / GGUF / audio.cpp（单二进制边缘）
```
bash download-funasr-model.sh sensevoice ./gguf      # 自动下 GGUF + VAD
llama-funasr-sensevoice -m ./gguf/SenseVoiceSmall-f16.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav
```
- 六/七个二进制：`llama-funasr-{cli,encoder,embd,sensevoice,paraformer,vad}`；v0.2.1 九个平台包（Linux x64/arm64、macOS arm64、Windows x64；含 Vulkan / CUDA 加速包）。
- **sensevoice-server**（C++，无 Python）：REST + SSE + SRT/VTT + OpenAI Realtime WS；`--threads 8 --max-connections 4 --max-audio-seconds 300`。
- **CosyVoice3 GGUF**（社区）：`Ferraronp/CosyVoice3-qwen2.5-0.5b-speech-gguf`，`CosyVoice3(model_dir=..., load_llama_cpp=True, gguf_model_path=...)`；T4 上 RTF 1.17 → 0.45（2.6×）。

#### 4. MLX（Apple Silicon）
```
# Swift
let model = try await CosyVoiceTTSModel.fromPretrained(modelId: "aufklarer/CosyVoice3-0.5B-MLX-8bit")
let audio = model.synthesize(text: "Welcome to the demo.", language: "english")
# CLI（speech-swift）
speech speak "Welcome to the demo." --engine cosyvoice --cosyvoice-variant 8bit --output out.wav
speech speak "..." --engine cosyvoice --voice-sample ref.wav   --cosy-reference-transcript "Transcript of ref.wav..." --output cloned.wav
```
Bundle 档位：`CosyVoice3-0.5B-MLX-{bf16, 8bit, 8bit-full, 4bit}`。转换要点：Conv1d 权重从 PyTorch `[out, in, kernel]` 转置到 MLX `[out, kernel, in]`；HiFi-GAN 折叠 weight-norm。

#### 5. 国产卡
**海光 DCU（ROCm 生态）**
```
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.15.1-ubuntu22.04-dtk26.04-0130-py3.10-20260220
docker run -it --shm-size 60g --network=host --name qwen3-asr --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root \
  -v /opt/hyhal/:/opt/hyhal/:ro -v /path/your_code_data/:/path/your_code_data/ \
  <image> bash
# 容器内
pip uninstall vllm
pip install vllm-0.15.1+das.opt1.alpha.dtk2604-cp310-cp310-linux_x86_64.whl
pip install numpy==1.25.0 && pip install -r requirements.txt
export VLLM_USE_FUSED_RMS_ROPE=0
vllm serve Qwen/Qwen3-ASR-1.7B --trust-remote-code --limit-mm-per-prompt '{"audio": 1}'
```
版本组合：**DTK 26.04 / Python 3.10.12 / torch 2.9.0+das.opt1.dtk2604 / transformers 4.57.6 / vllm 0.15.1+das.opt1.alpha**。
TTS 侧：vLLM 0.18.1 + vLLM-Omni 0.18.0 镜像（`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-nemotron-20260422`）。

**华为昇腾（CANN 生态）**
```
# Qwen3-ASR（vllm-ascend）
export IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1          # Atlas A2
# export IMAGE=quay.io/ascend/vllm-ascend:v0.22.1rc1-310p   # Atlas 300I DUO
docker run --rm --name vllm-ascend --shm-size=1g --net host \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info -v /root/.cache:/root/.cache -it -d $IMAGE bash

export ASCEND_RT_VISIBLE_DEVICES=3; export OMP_PROC_BIND=false; export OMP_NUM_THREADS=16
export VLLM_USE_V1=1; export CPU_AFFINITY_CONF=1; export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True; export MIES_USE_MB_SWAPPER=1
vllm serve $MODEL_PATH --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.9 \
  --max-model-len 32768 --block-size 128 --allowed-local-media-path / \
  --async-scheduling --enable-prefix-caching --prefix-caching-hash-algo xxhash \
  --served-model-name qw3-asr --mm_processor_cache_type="shm" \
  --compilation-config '{"cudagraph_mode": "FULL"}'
```
环境：**CANN 8.5.1 / torch 2.9.0+cpu / torch_npu 2.9.0 / vllm 0.17.0 / vllm-ascend 0.17.0rc1**（Atlas A2 910B4 64G，驱动 25.2.3）。Atlas 300I DUO 源码安装时需 `pip uninstall -y triton-ascend triton`。
TTS 侧：`quay.io/ascend/vllm-omni:v0.14.0` + docker-compose，端口 18005 → 容器 8000，调 `/v1/audio/speech`。vLLM-Omni 的 NPU 平台对 `qwen3_tts_code2wav`、`qwen3_tts_tokenizer_v2` 打了 patch 以适配昇腾算子。


### 24. 资源需求

#### CosyVoice3 各路径的体积 / 显存
| 路径 | 体积 | 精度配置 |
|---|---|---|
| **MLX bf16** | ~2.1 GB | LLM bf16 / DiT bf16 / tokenizer bf16 / vocoder fp32 |
| **MLX 8-bit** | ~1.4 GB（llm 640 MiB + flow 634 MiB + hifigan 79 MiB + tokenizer 462 MiB） | LLM int8(group=64) / **DiT bf16** / vocoder fp32 |
| **MLX 8-bit-full** | ~1.6 GB | LLM int8 / DiT int8 |
| **MLX 4-bit** | **~1.1 GB**（llm 388 MB + dit 186 MB + hifigan 79 MB + tokenizer 462 MB） | LLM int4(group=64) / DiT int4 |
| **ONNX（社区）** | ~3.8 GB（14 文件） | fp16 LLM / fp16 flow / fp32 hift |
| **原生 PyTorch** | 0.5B 参数 | bf16 |

#### SenseVoice GGUF / ONNX
| 档位 | 体积 | CER | 速度 |
|---|---|---|---|
| SenseVoice GGUF f16 | 470 MB | 8.01% | 23× |
| **SenseVoice GGUF q8** | **254 MB** | **7.99%** | 27× |
| Paraformer GGUF f16 | 435 MB | 9.85% | 22× |
| Paraformer GGUF q8 | 237 MB | 9.78% | 27× |
| Fun-ASR-Nano q4_K_M（+ encoder f16 470 MB） | 484 MB | 8.35% | 6.1× |
| SenseVoice ONNX FP32 / INT8 | 890 MB → 230 MB | +1.1% | CPU +33~36% |

#### GPU / 引擎构建资源
- **CosyVoice Triton**：`--workspace-gb 8`（TensorRT 构建工作区）；整链路需要足够的 GPU 显存放 LLM 引擎 + token2wav 引擎；disaggregated 模式可把 LLM 放 GPU 0、Token2Wav 放 GPU 1–3。
- **SenseVoice Triton**：workspace 8 GB；`--pinned-memory-pool-byte-size=512000000`、`--cuda-memory-pool-byte-size=0:1024000000`；optimization profile batch 1/8/16、frames 1/512/4096。
- **vLLM-Omni 多阶段**：共享 GPU 时各 stage 的 `gpu_memory_utilization` 之和 ≤ 1.0（详见 inference_engines.json）。
- **昇腾**：Qwen3-ASR-1.7B BF16 → 单张 910B 64GB 或 Atlas 300I DUO 48GB。
- **海光**：Qwen3-ASR-1.7B 单张 K100_AI 64GB。

#### CPU / 边缘
- SenseVoice q8（254 MB）或 Paraformer q8（237 MB）可在树莓派 / Jetson / 工控机跑；`sensevoice-server --threads N` 控制并发。
- CosyVoice3 MLX 4-bit（1.1 GB）可在 MacBook Air 级别设备运行。


### 25. 服务化栈

#### 传输协议选型
| 场景 | 协议 | 参数 | 理由 |
|---|---|---|---|
| **实时双向语音**（首选） | **WebRTC / Opus over RTP** | Opus **16–64 kbps**、20 ms 帧；jitter buffer **20–30 ms**（VoIP 是 60–100 ms）；生产用 **SFU** 不用 P2P | 内建 FEC/PLC 抗丢包、NAT 穿透（ICE/STUN/TURN）、拥塞控制；SFU 屏蔽最后一公里抖动，并让 SFU 节点（约 500 连接）与 GPU 节点（约 10 连接）**独立扩容** |
| 服务端↔服务端流式 | **WebSocket（PCM16 二进制）** | 16 kHz mono PCM16，按 100–200 ms 分块 | 实现简单、无编解码开销，适合内网/可控网络 |
| 单向结果下发 | **SSE** | — | 比 WS 轻，天然兼容 HTTP 基础设施与重连 |
| OpenAI 兼容实时 | **OpenAI Realtime WebSocket**（`WS /v1/realtime?intent=transcription`，vLLM /v1/realtime，SGLang-Omni /v1/realtime 带 VAD + barge-in） | — | 生态兼容，SDK 直接可用 |
| 字幕下发 | **SRT / VTT** | SenseVoice 原生 server 直接支持 `response_format=vtt` | 字幕场景零改造 |
| 文件/批量 | **HTTP multipart POST** | 加 size/duration 上限 | OpenAI 契约 |
| GPU 批量服务 | **gRPC**（Triton 8001） | 配合 dynamic batching | Triton 生态首选 |

#### 采样率工程规范（必做）
1. **ASR 输入统一 16 kHz / 单声道 / PCM16（int16 LE）**——几乎所有 ASR 模型按 16 kHz 训练。重采样在**网关层做一次**，不要每个模型各做一次。
2. **降采样必须先低通**（推荐 **soxr**），否则 8 kHz 以上分量混叠进可听带，直接污染 ASR。**每个流一个 resampler 实例**（按 SSRC），避免滤波器状态串扰。
3. **立体声→单声道**用均值下混：`stereo.mean(axis=1).astype(np.int16)`。
4. **TTS 输出按模型采样率对齐**：Qwen3-TTS / CosyVoice3 / higgs-audio / GLM-TTS = **24 kHz**；Fish Speech S2 Pro = **44.1 kHz**；VoxCPM2 = **48 kHz**。流式 PCM 播放命令示例：`play -t raw -r 24000 -e signed -b 16 -c 1 -`。
5. **WebRTC 侧**：Opus 解码器**按 SSRC 每路一个**（有状态）；异常时重建解码器 + 返回一帧静音 + 跳过后续 3 包重新同步。

#### 服务化组件
| 层 | 推荐 | 备注 |
|---|---|---|
| API 框架 | **FastAPI + Uvicorn**（仅在需要网关能力时） | 引擎已提供 OpenAI 兼容端点时**不要再造一层** |
| 容器 | **Docker + Docker Compose**（验证）→ **Kubernetes**（生产：ClusterIP + 探针 + 持久模型缓存 + NetworkPolicy） | CPU 示例镜像不是 CUDA 生产镜像 |
| Triton | `tritonserver --model-repository <dir>` + health `/v2/health/ready` + `/v2/models/<name>/ready` | 健康检查与模型就绪是两件事 |
| FunASR | `funasr-server` / `serve_realtime_ws.py` / `run_server.sh` / `run_server_2pass.sh` | 端口 10095 |
| SenseVoice C++ | `sensevoice-server` | 端口自定义（示例 8040） |
| 健康检查三段式 | `/health`（存活）+ `/v1/models`（模型身份）+ 真实音频转写（能出正确结果） | **三个必须分开验** |

#### 编排与排队
- 单进程：asyncio + asyncio.Queue（PCM 包 → VAD 状态机 → utterance）。
- 多服务：Celery/RQ + Redis（离线批）；NATS/Kafka/Redis Streams（服务间解耦）。
- 多阶段：Triton Ensemble / BLS、Ray Serve、vLLM-Omni stage pipeline、SGLang-Omni Coordinator。
- **背压必备**：有界队列 + 429/503 快速失败 + 客户端指数退避（SGLang-Omni 并发 64 丢 1–4% 请求就是没有背压的典型后果）。


### 26. 硬件生态

| 硬件 | Triton + TRT(-LLM) | TensorRT Edge-LLM | ONNX Runtime | GGUF / llama.cpp | MLX | vLLM / vLLM-Omni |
|---|---|---|---|---|---|---|
| **NVIDIA 数据中心**（A100/H100/L20/L40S/V100/T4） | ✅ **主场景** | — | ✅ | ✅（T4 实测） | — | ✅ |
| **NVIDIA Jetson**（Orin / Thor） | 部分 | ✅ **主场景**（Qwen3-ASR/TTS FP16，Jetson AGX Thor 128 GB） | ✅ | ✅（Vulkan/CUDA 包） | — | ❌（Linux x86 为主） |
| **x86 CPU** | ❌ | ❌ | ✅ **主场景** | ✅ **主场景** | — | ❌ |
| **Apple Silicon** | ❌ | ❌ | ✅ | ✅（macOS arm64 包） | ✅ **主场景** | ❌（vLLM-Omni Linux-only） |
| **华为昇腾 910B / Atlas 300I DUO** | 待核实 | ❌ | 待核实 | 待核实 | — | ✅ vllm-ascend / vllm-omni NPU 镜像 |
| **海光 DCU K100_AI** | 待核实 | ❌ | 待核实 | 待核实 | — | ✅ DTK 定制 wheel（vLLM 0.15.1 / 0.18.1 + Omni 0.18.0） |
| **AMD ROCm（MI300X 等）** | 部分 | ❌ | ✅ | ⚠️ Vulkan 有未解问题 | — | ✅（需 `onnxruntime-rocm` + `sox`） |
| **Intel XPU** | 待核实 | ❌ | ✅ | 待核实 | — | ✅（vLLM-Omni docker 有 XPU 镜像；SGLang-Omni 实验） |
| **Moore Threads MUSA** | 待核实 | ❌ | 待核实 | 待核实 | — | ✅（`platforms/musa/`） |
| **Windows** | ❌ | ❌ | ✅ | ✅（x64 Vulkan/CUDA 包） | — | ❌ |

**国产卡路径的核心风险是版本锁定**：海光（DTK 26.04 + torch 2.9.0+das.opt1.dtk2604 + vllm 0.15.1+das.opt1.alpha）与昇腾（CANN 8.5.1 + torch_npu 2.9.0 + vllm-ascend 0.17.0+）的任一组件版本漂移都可能导致无法启动。**建议把完整版本组合写进镜像 tag 并做镜像级交付，不要在生产机上 pip install。**


### 27. 集成范式

#### 模式 A：批量/离线配音（吞吐优先）
```
文本清单 → 队列(Celery+Redis)
   → Triton(CosyVoice3 + TRT-LLM, Decoupled=False, batch=16)
   → FlashCosyVoice 做 token2wav batching
   → 音频落盘/对象存储
```
关键：`use_spk2info_cache=True`（固定音色）、batch 对齐 optimization profile、engine 与 GPU 型号绑定。

#### 模式 B：实时语音助手（延迟优先）
```
[浏览器] Web Audio + AudioWorklet → Opus 20 ms
   ↓ WebRTC (SFU: LiveKit / mediasoup / Pion)
[网关] Opus 解码 → soxr 48k→16k 低通 → 下混单声道
   ↓ Silero VAD（2 帧起 / 20 帧止）
[ASR] vLLM(Qwen3-ASR) /v1/audio/transcriptions
   ↓
[LLM] 流式 token，按 .!?;: 短语边界切句
   ↓
[TTS] vLLM-Omni(Qwen3-TTS) /v1/audio/speech stream=true response_format=pcm
   ↓ 客户端动态 trim 前导静音（~80 ms）→ 播放
```
**三个决定体感的工程点（按贡献排序）**：
1. **LLM 输出按短语边界切句、边生成边送 TTS**——这一项的收益大于任何模型替换；
2. **barge-in 处理**——用户说话时立刻停播放 + 取消 TTS + 取消 LLM；
3. **前导静音动态 trim**。

#### 模式 C：边缘 / 数据不出域
```
设备端采集 → sensevoice-server（C++ 单二进制，内置 FSMN-VAD，REST/SSE/VTT/Realtime WS）
          → 文本留在本机 / 仅上送文本
```
或 `llama-funasr-sensevoice -m sensevoice-small-q8.gguf --vad fsmn-vad.gguf -a audio.wav`（纯 CLI 批处理）。

#### 模式 D：Mac 本地 / 桌面应用
```
speech-swift CLI / Swift SDK
  → CosyVoice3-0.5B-MLX-{4bit|8bit}
  → CAM++ CoreML (Neural Engine, ~14 MB) 抽 speaker embedding
  → 长文本按句切分 + 每段播种 + 边缘淡入淡出拼接
  → 多说话人对话：[S1]/[S2] 标签 + --turn-gap 0.2s
```

#### 模式 E：K8s 上的混合多模型服务
```
Ingress(TLS/认证/限流)
   ├─ asr-deploy   (vLLM Qwen3-ASR, GPU)
   ├─ asr-cpu-deploy (funasr-server SenseVoice, CPU) ← 低优先级/降级路径
   ├─ tts-deploy   (vLLM-Omni Qwen3-TTS, GPU)
   └─ tts-batch    (Triton CosyVoice3 + TRT-LLM, GPU) ← 离线批量
   + 持久模型缓存 PVC + readiness/liveness 探针 + HPA
```
**降级设计**：GPU 池打满时把流量切到 CPU 上的 SenseVoice（精度略降但可用），是这套栈最实用的韧性手段——因为两条路径**共用同一套 OpenAI 兼容契约**。

#### 一体化健康检查（三段式，所有路径通用）
```
curl -fsS $BASE/health                       # 1 存活
curl -fsS $BASE/v1/models                    # 2 模型身份
curl -fsS $BASE/v1/audio/transcriptions -F file=@known.wav -F model=xxx   # 3 真实转写
# Triton 用 /v2/health/ready + /v2/models/<name>/ready
```


### 28. 技术栈组件

一份**可直接照抄**的完整分层清单，从音频采集一路到评测。通用引擎层见 inference_engines.json，ASR 生态专用见 funasr_sensevoice.json，本份从**生产部署视角**补全并统一。

---

#### L0 音频采集（Capture）

| 场景 | 组件 | 关键参数 / 理由 |
|---|---|---|
| **浏览器（生产首选）** | **Web Audio API + AudioWorklet**，`getUserMedia({echoCancellation:true, noiseSuppression:true, autoGainControl:true})` | AudioWorklet 跑在独立音频渲染线程，**不受主线程 GC/jank 影响**（ScriptProcessorNode 已废弃且在主线程，生产禁用） |
| 浏览器原型 | MediaRecorder（Opus/WebM） | 简单，但有容器化开销与主线程抖动 |
| Python 服务端 / 桌面 | **sounddevice**（PortAudio） | NumPy 友好、API 现代；PyAudio 作为备选 |
| **文件 / 流转码（必备）** | **ffmpeg** `ffmpeg -i in.* -ar 16000 -ac 1 -f s16le -` | 事实标准，统一把 MP3/M4A/FLAC/OGG/Opus/视频容器转成 **16 kHz 单声道 PCM16** |
| 文件读写（Python） | **soundfile**（libsndfile） | 比 scipy.io.wavfile 稳，支持 flac/ogg |
| 移动端 | iOS AVAudioEngine / Android AudioRecord + Opus | 走平台最优路径 |
| macOS 本地 | MLX 生态 + AVFoundation | 见 L4 |

#### L1 传输（Transport）

| 场景 | 协议 / 组件 | 参数 | 理由 |
|---|---|---|---|
| **实时双向语音** | **WebRTC**（Opus over RTP/SRTP）+ **SFU**（LiveKit / mediasoup / Pion / werift） | Opus **16–64 kbps**、20 ms 帧；**jitter buffer 20–30 ms**；ICE/STUN/TURN 必配 | 内建 **FEC + PLC**（丢包隐藏）、拥塞控制、NAT 穿透。SFU 让「连接数」与「GPU 算力」独立扩容（SFU 节点 ~500 连接 vs GPU 节点 ~10 连接） |
| 服务端↔服务端流式 | **WebSocket** 二进制帧 | 16 kHz mono PCM16，100–200 ms 分块 | 零编解码开销，内网首选 |
| 单向下发 | **SSE** | — | 比 WS 轻，天然重连，兼容 HTTP 基础设施 |
| OpenAI 生态实时 | **OpenAI Realtime WebSocket** | `/v1/realtime?intent=transcription` | SDK 零改造 |
| 字幕 | **SRT / VTT** | SenseVoice 原生 server 直接支持 | 字幕场景零改造 |
| 文件 / 批量 | HTTP **multipart** POST | 加 size/duration 上限 | OpenAI 契约 |
| GPU 批量 | **gRPC**（Triton :8001） | + dynamic batching | Triton 生态 |

**四条硬性工程规范（建议写进团队规范）**：
1. **ASR 输入统一 16 kHz / 单声道 / PCM16（int16 LE）**，在**网关层重采样一次**，不要每个模型各做一次。
2. **降采样必须先低通**——用 **soxr**。不做低通的话 >8 kHz 分量会混叠回来，等于主动给 ASR 喂噪声。
3. **每路流一个 resampler / 一个 Opus 解码器实例**（按 SSRC）。Opus 与重采样滤波器都是有状态的，共享实例会让不同说话人的状态互相污染。
4. **Opus 解码异常恢复**：重建解码器 + 返回一帧静音 + 跳过后续 3 包让流重新同步。否则一个坏包会产出数秒噪音。

#### L2 VAD（Voice Activity Detection）

| 组件 | 部署位置 | 参数 | 何时用 |
|---|---|---|---|
| **Silero VAD**（通用首选） | 服务端 CPU / 浏览器 WASM / 端上 | 输入**严格 512 样本 @16 kHz（32 ms）**；`positiveSpeechThreshold=0.3`、`negativeSpeechThreshold=0.25`、`silenceDuration=1400ms`、`minSpeechDuration=400ms`、`preSpeechPadDuration=800ms`；实践值「2 连续语音帧起（~64 ms）/ 20 连续静音帧止（~640 ms）」 | 自建实时链路；起爆够快（64 ms）且能滤掉咳嗽 |
| **FSMN-VAD**（FunASR 栈） | 服务端（与 FunASR 配套） | `max_single_segment_time=30000`；**开启动态 VAD**（≤5 s→2.0 s 静音，5–10 s→1.5 s，10–15 s→1.0 s，15–30 s→0.8 s，30–45 s→0.4 s，>45 s→0.1 s） | FunASR 全链路；**长句不会被切碎** |
| **WebRTC VAD** | 极资源受限 | 8/16/32 kHz，mode 0–3 | 零依赖、极省 CPU；嘈杂环境鲁棒性弱于神经网络 |
| 语义 VAD / Ten VAD | 需判断"是否说完一句话" | 配合 LLM | 能量/神经网络 VAD 分不清"停顿思考"与"说完了" |

#### L3 特征提取（Feature Extraction）

| 组件 | 用途 | 备注 |
|---|---|---|
| **引擎内置前端**（优先） | FunASR / llama.cpp 内置 kaldi 80-mel fbank + LFR；vLLM 内置 audio processor | **不要业务层重复实现**，重复做只会引入与训练不一致 |
| **torchaudio** | GPU 侧重采样、`transforms.TimeStretch`（相位声码器变速不变调） | vLLM-Omni 的 `speed` 就用的它 |
| **librosa** | 离线分析、数据准备、实验 | 生产推理路径一般不走 |
| **soxr** | 高质量重采样（48 k→16 k） | **抗混叠最好，WebRTC 入口必备** |
| **numpy / scipy.signal** | RMS 窗检测（前导静音 trim）、轻量后处理 | Nari Labs 的动态 trim 就是短 RMS 窗检测持续语音起点 |

#### L4 模型推理（Inference）—— 按场景选型

| 场景 | 首选 | 备选 | 理由 |
|---|---|---|---|
| **Qwen3-ASR，GPU 高并发** | **vLLM 主线**（`vllm[audio]`，cu129 nightly） | SGLang-Omni、vllm-ascend（昇腾）、海光 DTK wheel | day-0、continuous batching、tensor parallel |
| **Qwen3-ASR，边缘/Jetson** | **TensorRT Edge-LLM**（FP16，0.6B/1.7B） | — | 官方 day-0 支持 |
| **SenseVoice/Paraformer，GPU 批量** | **Triton + TensorRT FP16** | FunASR AutoModel | 官方生产验证；V100 上相对 CPU 基线 ~526×（官方口径） |
| **SenseVoice/Paraformer，CPU/边缘** | **ONNX Runtime**（INT8 动态量化） | **GGUF + llama.cpp**（q8 254 MB）、`sensevoice-server`（C++） | ONNX 依赖轻、跨平台；GGUF 无 Python 依赖，可单二进制 |
| **Qwen3-TTS** | **vLLM-Omni 在线**（`/v1/audio/speech` + PCM/WebSocket 流式 + `/v1/audio/voices`） | SGLang-Omni、TensorRT Edge-LLM（Jetson，仅 CustomVoice） | 唯一覆盖在线流式 + 三 query type + 音色管理 |
| **CosyVoice3，在线流式** | **vLLM-Omni**（`async_chunk: true` 默认开） | — | 一条命令起 |
| **CosyVoice3，批量高吞吐** | **Triton + TRT-LLM**（batch=16 RTF 0.0386 @ L20） | DiT 变体 + disaggregated（LLM GPU0 / Token2Wav GPU1-3） | 吞吐最高 |
| **CosyVoice3，Apple Silicon** | **MLX**（4bit 1.1 GB / 8bit 1.4 GB / bf16 2.1 GB） | — | 零增量硬件成本 |
| **CosyVoice3，低显存 GPU（T4）** | **GGUF + llama.cpp**（RTF 1.17→0.45） | ONNX Runtime | 老卡/消费卡的现实选择 |

#### L5 后处理（Post-processing）

| 组件 | 作用 | 何时需要 |
|---|---|---|
| `ct-punc` | 中英双语标点 | **仅 Paraformer 需要**（SenseVoice / Fun-ASR-Nano / Qwen3-ASR 自带） |
| ITN（`fst_itn_zh` / `use_itn=True`） | 数字、日期、单位规整 | 输出面向人阅读时 |
| `cam++`（192 维） | 说话人验证 / 分离 | 会议、通话质检；CoreML 版 ~14 MB 走 Neural Engine |
| `Qwen3-ForcedAligner-0.6B` | 词/字级时间戳 | Qwen3-ASR 要时间戳时（**流式不支持**） |
| `rich_transcription_postprocess()` | 剥离 `<|zh|><|HAPPY|>` 等标签 | SenseVoice 必用 |
| 幻觉重复清理 | 剔除静音段产生的重复文本 | 自研流式链路必加 |
| **前导静音动态 trim** | 裁掉首段 PCM 的 ~80 ms 静音 | **TTS 流式必加**，否则 TTFA 数字好看体感慢 |

#### L6 服务化（Serving）

| 层 | 组件 | 备注 |
|---|---|---|
| API 框架 | **FastAPI + Uvicorn**（仅在需要网关能力时） | 引擎已有 OpenAI 兼容端点时**不要再造一层** |
| 容器 | **Docker + Compose**（验证）→ **Kubernetes**（ClusterIP + 探针 + 持久模型缓存 PVC + NetworkPolicy + GPU 调度） | CPU 示例镜像 ≠ CUDA 生产镜像 |
| 引擎服务 | `vllm serve --omni` / `sgl-omni serve` / `tritonserver` / `funasr-server` / `serve_realtime_ws.py` / `sensevoice-server` | 优先用引擎自带服务 |
| 音色管理 | vLLM-Omni `/v1/audio/voices` + `precompute_custom_voice.py`；Triton 侧 `use_spk2info_cache=True` | **固定音色必开缓存**（Triton 延迟 −13.8%） |
| 批处理 | vLLM continuous batching / Triton dynamic batching / `end2end.py --use-batch-sample` + `qwen3_tts_batch.yaml` | vLLM-Omni 的 `--batch-size` 必须匹配 CUDA Graph 捕获尺寸 |
| **健康检查三段式** | `/health`（存活）+ `/v1/models`（身份）+ 真实音频转写（结果正确） | Triton：`/v2/health/ready` + `/v2/models/<name>/ready` |

#### L7 编排与排队（Orchestration）

| 组件 | 用途 |
|---|---|
| **asyncio + asyncio.Queue** | 单进程内 PCM 包→VAD 状态机→utterance |
| **Celery / RQ / Arq + Redis** | 离线批任务队列、重试、结果持久化 |
| **NATS / Kafka / Redis Streams** | 服务间解耦（ASR ↔ LLM ↔ TTS） |
| **Triton Ensemble / BLS、Ray Serve、vLLM-Omni stage pipeline、SGLang-Omni Coordinator** | 多阶段流水线编排 |
| **SGLang-Omni Router / Nginx / Envoy / Kong** | 多副本 LB、健康检查、能力发现 |
| **背压与准入控制（必备）** | 有界队列 + 429/503 快速失败 + 客户端指数退避。**不做背压的后果**：SGLang-Omni 并发 64 时丢 1–4% 请求（HTTP 500） |

#### L8 监控与可观测（Observability）

| 层 | 组件 | 关键指标 |
|---|---|---|
| 指标 | **Prometheus + Grafana**（vLLM / SGLang / Triton 均自带 `/metrics`） | TTFA p50/p95/p99、RTF、**underrun 率**、chars/s、req/s、排队时长、GPU 利用率、KV cache 命中率 |
| 链路追踪 | **OpenTelemetry**（VAD / ASR / LLM / TTS / 播放各一个 span） | 语音链路阶段耗时分解，定位瓶颈必备 |
| 日志 | 结构化 JSON，每请求记录**音频时长 / 模型 / 设备 / 延迟 / response_format / 错误类型** | FunASR 官方 readiness checklist 明确要求 |
| 成本 | 每会话 token/字符计数 + 熔断器（LLM 异常降级为预置话术） | |
| **语音专项** | （a）**可听 TTFA**（trim 后）；（b）**underrun 率**；（c）**barge-in 成功率** | Nari Labs 的实时 TTS 四要素：低可听 TTFA / 零 underrun / 有容量 / 输出可懂 |

#### L9 评测（Evaluation）

| 侧 | 工具 |
|---|---|
| ASR 精度 | **jiwer**（中文先 jieba 分词算 CER，英文算 WER）、sclite、NeMo Speech Data Explorer |
| TTS 可懂度 | WER（英文 Whisper）/ CER（中文 Paraformer-zh） |
| TTS 说话人相似度 | **WavLM-large**、**ECAPA-TDNN**、**wespeaker**、X-Vector、RawNet3；`k2-fsa/TTS_eval_models`（ECAPA-TDNN + WavLM 组合） |
| TTS 自然度 | **UTMOS / UTMOSv2**、NISQA、DNSMOS、SQUIM-MOS |
| TTS 综合基准 | **Seed-TTS-eval**、**TTSDS / TTSDS2**（`pip install ttsds`）、**TTS Arena**（主观 ELO）、**TTSD-eval**（对话 TTS：MMS-FA 对齐 + wespeaker） |
| 数据集 | `k2-fsa/TTS_eval_datasets`；LibriSpeech-PC；FLEURS；AISHELL；WenetSpeech；Common Voice；CHIME |
| 服务侧压测 | 自研 harness（**Poisson 开环**）+ `vllm bench serve` + `client_grpc.py`（Triton）+ `batch_speech_client.py`（vLLM-Omni） |

详见 eval_toolchain 字段。

---

#### 三套可直接落地的默认栈

**A. 云端实时语音助手（延迟优先）**
```
Web Audio+AudioWorklet → WebRTC/Opus(SFU) → soxr 48k→16k → Silero VAD
  → vLLM(Qwen3-ASR-1.7B) /v1/audio/transcriptions
  → LLM 流式 token，按 .!?;: 切句
  → vLLM-Omni(Qwen3-TTS-1.7B) /v1/audio/speech (stream=true, pcm)
  → 客户端 trim 前导静音 → 播放；barge-in 全程接管
  → OTel 分段 span + Prometheus(TTFA/RTF/underrun) + 有界队列 429
```

**B. 企业离线批量（成本/吞吐优先）**
```
音频/文本清单 → Celery+Redis 队列
  ASR: FunASR(paraformer-zh + fsmn-vad + ct-punc + cam++) 或 SenseVoice ONNX/INT8（CPU）
  TTS: Triton(CosyVoice3 + TRT-LLM, Decoupled=False, batch=16) + FlashCosyVoice
  → 热词(hotwords.txt WFST) → 结果入库 → Grafana(RTF/热词命中率)
```

**C. 边缘 / 数据不出域**
```
设备采集 → sensevoice-server(C++ 单二进制, 内置 FSMN-VAD, REST/SSE/VTT/Realtime WS, q8 254MB)
        或 llama-funasr-sensevoice CLI 批处理
  → 文本留在本机；仅上送文本
  → 可选：CosyVoice3 MLX 4bit(1.1GB) 在本机做 TTS 回放
```


### 29. 评测工具链

#### 一、ASR 侧工具链

| 工具 | 用途 | 关键注意 |
|---|---|---|
| **jiwer** | WER / CER（Python + CLI） | **中文必须先分词**（jieba），否则指标语义不对；参考与预测必须走**完全相同**的预处理（大小写、标点、非语音标签、数字规整） |
| **sclite**（NIST） | WER/SER + 错误对齐 | 学术对照 |
| **NeMo Speech Data Explorer** | 多模型/多语言可视化对比 + 错误分析 | 输入 JSON manifest（音频路径/参考/预测），输出交互表 |
| `rich_transcription_postprocess()` | 剥离 SenseVoice 的 `<|zh|><|HAPPY|>` 标签 | **不剥离会把标签算进 CER** |

**测试集推荐**：AISHELL-1/2（中文普通话）、LibriSpeech（英文多口音）、WenetSpeech（中文会议/网络）、FLEURS（多语言）、Common Voice（60+ 语言）、CHIME（噪声/远场）。**自建域内测试集**是必须的——公开集好不代表你的业务音频好。

**评测流程**：① 统一预处理 → ② 中文 CER / 英文 WER（报告时注明）→ ③ 多人对话先按说话人切分 → ④ 看 S/D/I 错误类型分布定位问题 → ⑤ **补下游任务指标**（意图识别、实体抽取准确率）。

#### 二、TTS 侧工具链

| 维度 | 工具 | 说明 |
|---|---|---|
| 可懂度 | WER（Whisper，英文）/ CER（Paraformer-zh，中文） | **Seed-TTS 官方用 Whisper(EN) + Paraformer(ZH)** |
| 说话人相似度 SIM | **WavLM-large**、**ECAPA-TDNN**、**wespeaker**、X-Vector、RawNet3 | `k2-fsa/TTS_eval_models` 提供 ECAPA-TDNN + WavLM 组合；cosine 相似度 |
| 自然度 | **UTMOS**、**UTMOSv2**、NISQA、DNSMOS、SQUIM-MOS | 神经网络预测 MOS，适合 CI 回归 |
| 综合基准 | **Seed-TTS-eval**（字节定义，零样本） | 提供 EN WER（Whisper）、ZH CER（Paraformer）、SIM |
| | **TTSDS / TTSDS2**（`pip install ttsds`，MIT） | **分布级**指标：TTS 输出集 vs 真人参考集在 speaker/prosody/content/environment 多维 embedding 空间的 **Frechet 距离**（语音版 FID）。20 TTS 系统 × 14 语言，测试集每季度自动更新防污染 |
| | **TTS Arena** | 盲听 A/B 的主观 ELO 排名（如 Kokoro 82M 排名很高，常与客观分不符） |
| | **TTSD-eval**（OpenMOSS） | 对话 TTS：ACC / SIM / WER，用 MMS-FA 对齐 + wespeaker embedding |
| 信号级 | PESQ、STOI、MCD | 需匹配参考，相关性很低（见下表），**只用做辅助** |

**TTSDS2 论文（arXiv:2506.19441）的 Spearman 相关性实测**（200 名母语听者、11,846 条 MOS/CMOS/SMOS 评分、20 个 TTS 系统、4 个数据集）：

| 指标 | Clean | Noisy | Wild | Kids | 平均 |
|---|---|---|---|---|---|
| **TTSDS2** | 0.75 | 0.69 | 0.73 | 0.59 | **0.67** |
| TTSDS（原版） | 0.60 | 0.62 | 0.52 | 0.49 | ~0.60 |
| **UTMOS** | 0.51 | 0.30 | 0.31 | 0.47 | — |
| **UTMOSv2** | 0.39 | 0.25 | 0.09 | 0.34 | — |
| DNSMOS | 0.41 | 0.37 | 0.22 | 0.57 | — |
| SQUIM | 0.68 | 0.46 | 0.37 | 0.48 | — |
| NISQA | 0.05 | 0.00 | 0.06 | 0.05 | — |
| ECAPA-TDNN (SIM) | 0.36 | 0.29 | 0.47 | 0.29 | — |
| X-Vector (SIM) | 0.46 | 0.42 | 0.56 | 0.40 | — |
| RawNet3 (SIM) | 0.36 | 0.26 | 0.52 | 0.44 | — |
| Audiobox AE-CE / AE-CU / AE-PQ | 0.60 / 0.49 / 0.49 | 0.46 / 0.37 / 0.33 | 0.32 / 0.30 / 0.21 | 0.58 / 0.60 / 0.55 | — |
| FAD (CLAP) | −0.22 | 0.06 | −0.01 | 0.45 | — |
| STOI / PESQ | −0.11 / 0.01 | 0.01 / −0.16 | 0.02 / 0.27 | −0.06 / −0.34 | — |
| MCD | −0.46 | −0.37 | −0.27 | −0.45 | — |
| WER | −0.19 | −0.18 | −0.17 | −0.11 | — |

→ **TTSDS2 是唯一在所有数据集与主观指标上都 ρ > 0.5 的指标**；单看 UTMOS / SIM / PESQ 都不可靠。

#### 三、测试集分级（推荐）
- **L1 快速回归**：Seed-TTS 各 50 条（EN 50 + ZH 50）——每次 CI 跑。
- **L2 完整零-shot 标准**：完整 Seed-TTS-eval + LibriSpeech-PC。
- **L3 多语言扩展**：FLEURS / MiniMax Multilingual。
- **L4 对话场景**：100 组多轮对话（30 s–720 s），来自 TTSD-eval。

#### 四、六阶段客观评测流水线
1. **测试集与 manifest**：`test.tsv` 含 `utt_id`、参考文本、参考音频路径、speaker ID、**固定采样率（16 k 或 24 k）**；
2. **批量合成**：统一超参（seed / temperature / top_p / **streaming flag**），输出到 `results/{testset}/{utt_id}.wav`；
3–5. **指标计算**：SIM（speaker embedding 余弦）、WER/CER（hubert 或 Seed-TTS 后端）、UTMOS；
6. **汇总与排名**：JSON/CSV 落盘（dataset, model_id, WER, SIM, UTMOS, date, git_hash），可选算 TTSDS2 综合分并结合 TTS Arena 主观结果。

#### 五、六个必须遵守的陷阱规避规则
1. **对比 Seed-TTS 官方表格时，绝不能更换 ASR 监听模型**——换模型 = 换尺子，数值不可比。
2. **同一份 SIM 报告不能混用不同网络（WavLM vs ECAPA-TDNN）的 speaker embedding**。
3. **MOS 预测器在 4.5 分以上会饱和**；且存在 **zoomed-in bias**（听测者倾向用满 5 级尺，导致中档系统被系统性压低）。模型已到 **4.3+** 时：换 **UTMOSv2**（domain embedding + stacking 缓解），或改走 **MUSHRA / A-Test**；**不要拿 UTMOS v1 的 raw 分跨 paper 横向比**。
4. **只看平均分看不出分布漂移**：UTMOSv2 均分不变但 TTSDS2 掉了，说明韵律多样性下降 / 口音塌成单模式。**TTSDS2 需两个目录各 20+ 条（论文建议 100+）wav**，样本太少估不出稳定分布。
5. **流式场景必须额外报 TTFA 与 streaming RTF**；且**固定 chunk 配置后再横向比**（不同 chunkFrames 的 WAV 不字节一致）。
6. **报 WER/CER 时同时报 UTMOS**——单一指标会误导。客观分用于 CI 回归，主观分（TTS Arena / 人工 MOS）用于对外宣称。

#### 六、服务侧压测工具链
| 工具 | 用途 |
|---|---|
| 自研 harness（**Poisson 开环**，每配置 ≥5 分钟） | TTFA / underrun 率 / RTF / chars/s |
| `vllm bench serve` | vLLM 侧吞吐延迟 |
| `benchmarks/eval/benchmark_asr_seedtts.py`（SGLang-Omni） | ASR 并发基准 |
| `client_grpc.py`（CosyVoice `runtime/triton_trtllm`） | Triton gRPC 并发基准（可设 `--num-tasks`、`--huggingface-dataset`、`--split-name`、`--mode`） |
| `batch_speech_client.py`（vLLM-Omni `examples/online_serving/qwen3_tts`） | TTS 并发吞吐 |
| Deepgram STT / 任意 ASR | TTS 合成音频可懂度自动回听校验 |
| Prometheus + Grafana | 生产持续监控 |


### 30. 已知坑与限制

1. **TensorRT plan 不可跨 GPU 架构 / TensorRT 版本移植**（最大的运维风险）。构建机 GPU 型号与驱动必须与生产一致；每次模型/驱动/TensorRT 升级都要重建 + 重验精度、显存、容量。**建议在 CI 中锁定构建环境。**
2. **SenseVoice TRT 路径不支持 `model_quant.onnx`**：必须用 `quantize=False` 导出。用错不会报清楚的错，只会构建失败或精度异常。
3. **Triton tokenizer 必须单独下载并校验 SHA-256**（`aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8`），路径/版本错了**静默产出乱码**。
4. **切换 Decoupled 模式必须重跑 stage 2**（重建 model repository），只改 config 不够。
5. **Optimization profile 与实际流量不匹配会显著掉速**：profile 要按真实 batch 与音频时长分布来定，不要照抄默认值。
6. **TensorRT 迭代成本被严重低估**：引擎构建数分钟到数十分钟，模型更新就要重来。**迭代频繁的项目优先 vLLM。**
7. **ONNX 导出一致性必须自测**：CosyVoice 官方 `export_onnx.py` 会跑 10 次随机 shape（16–512 frames）比对 PyTorch 与 ONNX 输出（rtol=1e-2 / atol=1e-4）——**自己加模型时也要做这一步**。
8. **ROCm 依赖冲突**：vLLM-Omni 在 ROCm 上必须先 `pip uninstall onnxruntime` 才能装 `onnxruntime-rocm`（Qwen3-TTS 需要），另需 `sox`。
9. **llama.cpp 边缘硬件兼容未解问题**：AMD Radeon 780M 待复测；RX 9070 XT 的 0xC0000005 初始化崩溃未解决；**Android/Mali 不是预编译或验证目标**。
10. **audio.cpp 的 SenseVoice 尚未进 tagged release**（`main@979e070f`），必须 pin commit；且无词级时间戳。
11. **SenseVoice 推理不接 CMVN**——自己复现前端时要对齐，否则精度对不上。
12. **昇腾 Atlas 300I DUO 源码安装需先卸载 triton / triton-ascend**，否则冲突。
13. **海光 DCU 需要定制 vLLM wheel**（`vllm-0.15.1+das.opt1.alpha.dtk2604`），且与 numpy==1.25.0 绑定；`VLLM_USE_FUSED_RMS_ROPE=0` 是必需的环境变量。
14. **vLLM-Omni 仅 Linux + Python 3.12**，Apple Silicon 完全走不了（改走 MLX）。
15. **vLLM / vLLM-Omni 大版本号必须对齐**，否则出现各种奇怪问题。
16. **前导静音 ~80 ms**：不做动态 trim，报出的 TTFA 会显著好于体感。
17. **重采样不做低通会混叠**：48 k→16 k 必须用 soxr 之类带抗混叠滤波的实现。
18. **Opus 解码器状态污染**：必须按 SSRC 每路一个实例；一个坏包会污染解码状态产出数秒噪音。
19. **TTS 流式不同 chunk 配置的输出不字节一致**（Code2Wav 块间无 left-context overlap），横向评测必须固定 chunk 配置。
20. **并发过冲无背压会丢请求**（SGLang-Omni 并发 64 丢 1–4%）。
21. **NVIDIA Dynamo 路径不支持 audio streaming 与 Base 克隆**。
22. **Qwen3-TTS 在 TensorRT Edge-LLM 只支持 CustomVoice**（0.6B/1.7B），不支持 Base 克隆；且当前 runtime 要求 `--maxBatchSize 1`。
23. **国产卡延迟数字缺乏压测条件**：海光 / 昇腾的多数公开延迟数据来自社区实践文章，未说明并发与音频时长，**不可直接用于容量规划**。


### 31. 社区与文档

| 项目 / 组件 | 许可 | 活跃度与文档 |
|---|---|---|
| **CosyVoice**（FunAudioLLM） | Apache-2.0 | 高。roadmap 明确记录 2025/08 合入 NVIDIA Yuekai Zhang 的 triton trtllm runtime；runtime 目录含 README.md + README.DIT.md + 完整 run.sh 阶段编排 + 基准表 |
| **FunASR** | Apache-2.0 | 高。funasr.com 部署中心为每条路径写 Deployment contract（成熟度/验证 commit/验证日期/适用边界/可复制命令/已知限制/证据链接）——**开源语音项目里文档质量的天花板** |
| **SenseVoice** | Apache-2.0 | 高。llama.cpp runtime 由社区 PR #292（2026-06）贡献，含 DESIGN.md / README.md / BENCHMARKS.md |
| **NVIDIA Triton / TensorRT-LLM** | 开源 / NVIDIA 许可 | 极高。docs.nvidia.com；CosyVoice 侧由 NVIDIA CISI 项目孵化（mair-hub） |
| **TensorRT Edge-LLM** | NVIDIA 许可 | 中。v0.6.0（2026-03-16）新增 Qwen3-TTS / Qwen3-ASR day-0 支持；nvidia.github.io/TensorRT-Edge-LLM/ |
| **speech-swift（MLX）** | 开源 | 低-中。soniqo/speech-swift + `aufklarer/*` 系列 MLX bundle；文档在 github.com/soniqo/speech-swift/blob/main/docs/models/cosyvoice-tts.md |
| **vllm-ascend** | Apache-2.0 | 高。docs.vllm.ai/projects/ascend/ + 每个模型的 tutorial 页（含 Qwen3-ASR-1.7B 专页） |
| **海光 DCU** | — | 中。developer.sourcefind.cn 提供 modelzoo 与定制 wheel；文档主要在社区文章 |

**文档可信度分级（建议）**：
- **一手可信**：funasr.com/deploy/* 的 contract 页、各仓库 README / runtime README、NVIDIA 官方文档、vLLM recipes。
- **需交叉验证**：中文社区博客（CSDN / 掘金 / 头条）里的性能与延迟数字，几乎都不给并发与音频时长，**只能当线索不能当依据**。本文件中已逐条标注「待一手核实」。
- **已过时**：任何写「vLLM-Omni 只支持 Qwen3-TTS 离线推理」的文章（2026-08 起在线 serving 已可用）。


### 32. 适用场景与选型建议

#### 决策树

**Q1：你的首要优化目标是什么？**

| 目标 | 推荐路径 |
|---|---|
| **单卡吞吐 / 批量成本** | **Triton + TensorRT-LLM**（CosyVoice batch=16 RTF 0.0386 @ L20；SenseVoice TRT 相对 CPU 基线 526× @ V100） |
| **在线低延迟 / 快速迭代** | **vLLM-Omni**（`/v1/audio/speech` 流式 PCM/WebSocket；改模型只需换参数） |
| **零 GPU / 边缘 / 数据不出域** | **GGUF + llama.cpp** 或 **`sensevoice-server`**（C++ 单二进制，q8 254 MB） |
| **无 PyTorch / 依赖最小化** | **ONNX Runtime**（`onnxruntime==1.18.0` + numpy/soundfile/librosa 即可） |
| **Mac 本地 / 桌面应用** | **MLX**（CosyVoice3 4-bit 1.1 GB） |
| **Jetson / 车载** | **TensorRT Edge-LLM**（Qwen3-ASR / Qwen3-TTS FP16，Jetson AGX Thor） |
| **国产卡合规** | 昇腾：vllm-ascend / `quay.io/ascend/vllm-omni`；海光：DTK 定制 wheel |

**Q2：模型迭代频率高吗？**
- **高（周更/日更）** → **不要上 TensorRT**。引擎构建 + 重验的成本会吃掉全部性能收益。用 vLLM / vLLM-Omni。
- **低（季度/半年）** → 可以上 Triton + TRT，把构建的 engine 与配置做镜像级交付。

**Q3：流量形态？**
- **稳态高并发批量** → Triton（dynamic batching + optimization profile 能吃到最大红利）。
- **尖峰突发 / 长尾** → vLLM（continuous batching 对不均匀请求更友好）。

**Q4：说话人是否固定？**
- **固定（客服播报、固定 IP 音色）** → **务必开缓存**：Triton `use_spk2info_cache=True`（延迟 −13.8%）；vLLM-Omni `precompute_custom_voice.py` + `custom_voice_dir`。
- **频繁变化（UGC 克隆）** → 缓存收益有限，优先优化参考音频编码这一步（vLLM-Omni 的 `/v1/audio/voices` 上传 + 复用）。

#### 场景 → 方案对照表

| 场景 | ASR | TTS |
|---|---|---|
| 实时语音助手 / 客服机器人 | Qwen3-ASR + vLLM（或 SenseVoice + Triton 做高吞吐） | Qwen3-TTS + vLLM-Omni 流式 PCM |
| 呼叫中心整轨质检 | SenseVoice / Paraformer + FunASR（ONNX/CPU 亦可） | — |
| 有声书 / 批量配音 | — | **CosyVoice3 + Triton + TRT-LLM（batch=16）** |
| 方言内容生产 | Qwen3-ASR（22 方言） | **CosyVoice3（18+ 方言，无替代品）** |
| 视频字幕 | Paraformer + FunClip / SenseVoice + VTT | — |
| 边缘设备（工控机 / 树莓派 / Jetson） | SenseVoice q8 GGUF（llama.cpp / sensevoice-server） | CosyVoice3 GGUF（T4 RTF 0.45）或 TensorRT Edge-LLM（Jetson） |
| Mac 桌面应用 | SenseVoice GGUF（macOS arm64） | CosyVoice3 MLX 4bit/8bit |
| 国产化合规（政务 / 金融） | Qwen3-ASR + vllm-ascend / 海光 DTK | Qwen3-TTS + `quay.io/ascend/vllm-omni` / 海光 vLLM-Omni 0.18.0 |
| 车载 / 机器人（Jetson Thor） | Qwen3-ASR + TensorRT Edge-LLM FP16 | Qwen3-TTS CustomVoice + TensorRT Edge-LLM（`--chunkFrames=10` 流式） |

#### 三条最重要的工程建议
1. **先用 vLLM / vLLM-Omni 上线并采集真实流量分布，再用真实分布去定 Triton 的 optimization profile 与 batch 策略。** 顺序反了，TRT 引擎会因为 profile 不匹配而跑不过 vLLM。
2. **把「固定音色/说话人缓存」当成必做项**，不是优化项。这是所有 TTS 部署路径上 ROI 最高的一步（Triton 延迟 −13.8%，vLLM-Omni 免去每次请求的参考音频编码）。
3. **ASR 与 TTS 共用同一套 OpenAI 兼容契约（`/v1/audio/transcriptions` + `/v1/audio/speech`）**，让网关可以按成本/时延/语种路由，也让 GPU 池打满时可以降级到 CPU 上的 SenseVoice——**契约统一是这套技术栈最大的杠杆**。

#### 一句话结论
- **加速**：批量吞吐上 **Triton + TensorRT-LLM**；在线低延迟与快速迭代上 **vLLM-Omni**；边缘上 **GGUF / TensorRT Edge-LLM**；Mac 上 **MLX**。
- **部署**：Docker Compose 验证 → Kubernetes 生产；传输用 **WebRTC/Opus**（实时）或 **WebSocket/PCM16**（内网）；**ASR 输入统一 16 kHz 单声道 PCM16，降采样必须低通**。
- **国产化**：昇腾走 vllm-ascend + `quay.io/ascend/vllm-omni`；海光走 DTK 定制 wheel（vLLM 0.15.1 / 0.18.1 + Omni 0.18.0）。**版本锁定压倒一切。**
- **评测**：ASR 用 jiwer（中文分词算 CER）；TTS 用 WER/CER + SIM + UTMOS 三件套，配 **TTSDS2**（唯一全域 ρ>0.5 的指标）做分布漂移预警；**绝不能换监听模型、绝不能混用 embedding 网络、MOS 高分必须人工校准**。


### 33. 参考来源

**CosyVoice Triton + TensorRT-LLM**
- FunAudioLLM/CosyVoice 仓库（roadmap 记录 2025/08 由 NVIDIA Yuekai Zhang 贡献 triton trtllm runtime）：https://github.com/FunAudioLLM/CosyVoice
- runtime/triton_trtllm README（含完整基准表）：https://github.com/FunAudioLLM/CosyVoice/blob/main/runtime/triton_trtllm/README.md
- README.DIT.md（DiT Token2Wav + disaggregated 部署）：https://github.com/FunAudioLLM/CosyVoice/blob/main/runtime/triton_trtllm/README.DIT.md
- Triton-OpenAI-Speech（OpenAI 兼容桥接）：https://github.com/yuekaizhang/Triton-OpenAI-Speech
- FlashCosyVoice（token2wav batching，`-b trt` 分支）：https://github.com/yuekaizhang/FlashCosyVoice
- CosyVoice 优化后端（ONNX/TensorRT/vLLM 导出与性能，DeepWiki 汇总）：https://deepwiki.com/FunAudioLLM/CosyVoice/3.3-optimization-backends
- CosyVoice3 性能基准汇总（DeepWiki）：https://deepwiki.com/wehos/CosyVoice-v3/7.5-performance-optimization

**SenseVoice 加速**
- FunASR 部署契约 — SenseVoice TensorRT / Triton：https://funasr.com/en/deploy/sensevoice-tensorrt.html
- FunASR Triton GPU runtime README：https://github.com/modelscope/FunASR/blob/main/runtime/triton_gpu/README.md
- 构建脚本：https://github.com/modelscope/FunASR/blob/main/runtime/triton_gpu/scripts/build_sensevoice_tensorrt.py
- 原生 TensorRT 集成 PR：https://github.com/modelscope/FunASR/pull/3463
- 部署请求与验证记录 issue：https://github.com/modelscope/FunASR/issues/3453

**ONNX / ONNX Runtime**
- SenseVoice ONNX 导出（DeepWiki）：https://deepwiki.com/FunAudioLLM/SenseVoice/4.1-onnx-export
- SenseVoice 量化与优化（含 Triton 526× 口径）：https://deepwiki.com/astordu/SenseVoice/7.4-quantization-and-optimization
- FunASR ONNX / C++ CPU runtime 部署契约：https://funasr.com/en/deploy/cpu-runtime.html
- SenseVoice ONNX 量化实测（社区，待核实）：https://blog.csdn.net/weixin_42351520/article/details/160126743

**llama.cpp / GGUF**
- FunASR llama.cpp 总览（体积/精度/速度对照）：https://www.funasr.com/llama-cpp.html
- FunASR 部署契约 — llama.cpp / GGUF standalone：https://funasr.com/en/deploy/llama-cpp.html
- FunASR 部署契约 — SenseVoice native OpenAI realtime server：https://funasr.com/en/deploy/sensevoice-native-server.html
- SenseVoice llama.cpp runtime（DESIGN/README/BENCHMARKS）：https://github.com/FunAudioLLM/SenseVoice/tree/main/runtime/llama.cpp
- FunASR 部署契约 — audio.cpp：https://funasr.com/en/deploy/audio-cpp.html

**Apple Silicon / MLX**
- aufklarer/CosyVoice3-0.5B-MLX-4bit：https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-4bit
- aufklarer/CosyVoice3-0.5B-MLX-8bit（含 RTF 0.40 实测）：https://huggingface.co/aufklarer/CosyVoice3-0.5B-MLX-8bit
- speech-swift — CosyVoice3 TTS Architecture（组件/量化档位/情感标签/流式）：https://github.com/soniqo/speech-swift/blob/main/docs/models/cosyvoice-tts.md

**NVIDIA 边缘 / 引擎**
- TensorRT Edge-LLM supported models（Qwen3-ASR / Qwen3-TTS FP16）：https://nvidia.github.io/TensorRT-Edge-LLM/0.6.0/user_guide/getting_started/supported-models.html
- TensorRT Edge-LLM — TTS（Qwen3-TTS 导出/建引擎/流式）：https://nvidia.github.io/TensorRT-Edge-LLM/0.9.0/user_guide/examples/tts.html
- Getting Started with TensorRT Edge-LLM on Jetson Thor：https://www.hackster.io/shahizat/getting-started-with-nvidia-tensorrt-edge-llm-on-jetson-thor-14735e
- NVIDIA Dynamo — Text-to-Audio：https://docs.nvidia.com/dynamo/diffusion/text-to-audio

**国产卡**
- 昇腾 Qwen3-ASR 部署实践（Ascend-SACT）：https://gitcode.com/Ascend-SACT/Qwen3-ASR_vllm-ascend
- vLLM Ascend 文档 — Qwen3-ASR-1.7B tutorial：https://vllm-ascend.readthedocs.io/en/latest/tutorials/models/Qwen3-ASR-1.7B.html
- vLLM Ascend v0.18.0 Release Notes（CANN 8.5.1 / torch_npu 2.9.0）：https://docs.vllm.ai/projects/ascend/en/v0.18.0/user_guide/release_notes.html
- 昇腾 Qwen3-TTS-12Hz-0.6B-Base（gitcode，vllm-omni:v0.14.0 镜像）：https://ai.atomgit.com/atomgit-ascend/qwen--qwen3-tts-12hz-0.6b-base
- 海光 DCU K100_AI 单卡全离线同声传译系统（vLLM 0.15.1 / 0.18.1 + Omni 0.18.0 实测）：https://aiot.csdn.net/6a82d0d610ee7a33f29c19ff.html
- 海光 DCU 环境下的轻量级语音克隆系统（<97 ms 实测）：https://deepseek.csdn.net/6a5b2965662f9a54cb908b52.html

**评测工具链**
- TTSDS2 论文（arXiv:2506.19441，含全套 Spearman 相关性表）：https://arxiv.org/abs/2506.19441 ｜ http://ar5iv.labs.arxiv.org/html/2506.19441
- 开源 TTS 评测框架指南（三指标铁三角 + 六阶段流水线 + 陷阱）：https://www.besthub.dev/articles/building-an-open-source-tts-evaluation-framework-with-zipvoice-omnivoice-latest-benchmarks-d571124e43c9
- Seed-TTS Eval 综述（Emergent Mind）：https://www.emergentmind.com/topics/seed-tts-eval
- 音频 MOS 演进（zoomed-in bias、UTMOSv2、TTSDS2 定位）：https://blog.csdn.net/qq_34941290/article/details/162661768
- ASR 评测工程实践（jiwer / 数据集 / 陷阱）：https://forums.developer.nvidia.com/t/how-to-benchmark-asr-models-with-word-error-rate-wer/346066/1

**实时传输工程**
- WebRTC 实时语音 Agent（Opus vs PCM16、Silero VAD、barge-in）：https://callsphere.ai/blog/agentic-ai-webrtc-real-time-voice-interfaces-development
- WebRTC 音频管线（RTP→jitter buffer→Opus 解码→重采样→VAD，soxr 抗混叠）：https://centedge.io/category/webrtc/
- WebRTC 双向语音在 AI Agent 中的应用（SFU / jitter buffer 20-30 ms）：https://www.getstream.io/blog/webrtc-ai-voice-video/
- realtime-audio-sdk（Silero VAD 参数默认值参考）：https://github.com/realtime-ai/realtime-audio-sdk




---

## 第六部分 · 风险清单与待核实项

### 6.1 选型风险

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| **许可证漂移** | 高 | Fish Audio 2026-03-07 整体换协议、Spark-TTS 从 Apache 改 CC-BY-NC-SA。每季度复查 LICENSE 原文，不读 badge |
| **FlashAttention 2 强依赖** | 高 | 无 FA2 时 Qwen3-TTS 仅 0.3× 实时。CI 中加入 FA2 可用性断言 |
| **vLLM-Omni 显存独占** | 中 | 默认每 stage 独立预留显存，需显式配置 `gpu_memory_utilization` |
| **官方延迟数字不可直接对标** | 中 | Qwen3-TTS 97 ms 未注明 GPU 型号；CosyVoice3 150 ms 未注明硬件。必须按目标硬件复测 |
| **微调代码 bug** | 中 | Qwen3-TTS 微调曾存在 double label-shift 导致语速异常，2026-03-17 修复。锁定 commit |
| **Base 模型不支持风格指令** | 中 | Qwen3-TTS 的 Base 变体不支持 instruct 控制，需要风格控制必须用 CustomVoice / VoiceDesign |
| **25Hz 系列权重未发布** | 中 | Qwen3-TTS 论文中的 25Hz 模型（长文本 WER 1.517 更优）目前无权重，不要按论文指标做容量规划 |
| **CosyVoice3-1.5B 权重未开源** | 中 | 论文最佳组合（zh CER 0.71 / en WER 1.45）是 1.5B RL 版，开源只有 0.5B |
| **依赖版本冲突** | 低 | CosyVoice 需 Python 3.10 + pynini（用 conda-forge）；Fish-Speech 强约束 Python 3.12 |
| **MOS 预测器饱和** | 低 | UTMOS > 4.5 后无法区分前沿模型，需人工 MOS/CMOS 校准 |

### 6.2 明确「待核实」的关键数据（**不要用于容量规划**）

**Qwen3-ASR**
- 预训练/微调集群规模与卡型（论文完全未披露，吞吐测试只写 "single typical computing resource"）
- 官方显存表（官方无，第三方数据互相冲突）
- 编码器层数/头数（第三方博客 24/18 层与 32 层说法冲突 → 以 `config.json` 为准）
- GSPO 奖励函数设计（论文未说明）
- 内建 VAD / SRT 能力（仅见第三方转述）

**Qwen3-TTS**
- 输出采样率（推断 24 kHz，论文未写明）
- QPS 吞吐与 vLLM-Omni 加速收益（无官方数据）
- 官方量化与端侧方案（无）
- 25Hz 系列权重（未发布）

**CosyVoice3**
- RTF、显存、吞吐（官方均未公布）
- 训练超参（batch / lr / GPU 小时，论文 §7 明确未公布）
- 最短参考音频时长（官方未给，社区说法 3~10 s）
- FSQ 的 D / K 取值（6561 为代码推断）

**技术栈**
- Baseten 成本模型全部数值（未找到原始博客，二手）
- vLLM-Omni `qwen3_tts_high_concurrency.yaml` 的 +61.5% 吞吐 / −49.4% P99（仅中文社区文章）
- MLX 上 M2 Max RTF ≈ 0.5 及各阶段延迟（未核实）
- 海光 DCU 200 ms / <97 ms、昇腾 RTF ≈ 0.21（社区实践，无并发与音频时长条件）
- FunASR 效率基准 170×/17×、120×/15×、13×（社区数据）

### 6.3 本次调研修正的常见误传

| 误传 | 事实 |
|---|---|
| 「IndexTTS-2 是 Apache-2.0」 | 实为 **bilibili 模型使用许可协议**，含月活 1 亿 / 年收入 10 亿阈值，且禁止用于改进其他 AI 模型 |
| 「F5-TTS 是 MIT 可商用」 | 代码 MIT，但**权重 CC-BY-NC-4.0**（训练数据 Emilia 传导） |
| 「Fish-Speech 是 Apache-2.0」 | 2026-03-07 起仓库 LICENSE 整体换成 **FISH AUDIO RESEARCH LICENSE**；Fish-Speech 1.5 已于 2026-02-28 官方弃用 |
| 「vLLM-Omni 只支持离线推理」 | 已过时——主线 `examples/online_serving/text_to_speech/` 现提供 `vllm serve --omni`，支持 `/v1/audio/speech`、流式 PCM、WebSocket 增量流式 |
| 「CosyVoice3 有 1.5B 开源版」 | 1.5B 仅存在于论文，开源权重只有 0.5B |

---

## 附录 A · 字段定义

字段框架共 38 个，分 8 大类。**⭐ 标记的 10 个「架构与训练」字段为本轮调研的重点**，在正文中以 ⭐ 标出。

| 分类 | 字段 |
|---|---|
| 基本信息 | basic_info、license、model_variants |
| **架构与模型** | architecture_paradigm ⭐、component_breakdown ⭐、audio_encoder ⭐、tokenizer_codec ⭐、llm_backbone ⭐、acoustic_decoder ⭐、vocoder、context_and_io |
| **训练与微调** | training_data ⭐、training_stages ⭐、post_training_rl ⭐、data_recipe ⭐、finetune_support、train_hardware |
| 核心能力 | languages、streaming、vad_and_postprocess、voice_clone、emotion_and_control、robustness |
| 性能指标 | asr_metrics、tts_metrics、latency_throughput、cost_model |
| 工程部署 | inference_frameworks、resource_requirements、serving_stack、hardware_ecosystem、integration_pattern |
| 技术栈组件 | tech_stack_components、eval_toolchain |
| 选型风险 | known_issues、community_and_docs、best_fit、references |

---

## 附录 B · 调研对象与产物

| # | 调研对象 | 分类 | 结果文件 |
|---|---|---|---|
| 1 | Qwen3-ASR | 核心模型-ASR | `results/qwen3_asr.json` |
| 2 | Qwen3-TTS | 核心模型-TTS | `results/qwen3_tts.json` |
| 3 | CosyVoice3 | 核心模型-TTS | `results/cosyvoice3.json` |
| 4 | vLLM / vLLM-Omni / SGLang | 技术栈-推理引擎 | `results/inference_engines.json` |
| 5 | FunASR / SenseVoice | 技术栈-ASR生态 | `results/funasr_sensevoice.json` |
| 6 | Fish-Speech / IndexTTS-2 / F5-TTS 等 | 技术栈-竞品对比 | `results/tts_competitors.json` |
| 7 | TensorRT-LLM / ONNX Runtime / Triton | 技术栈-生产部署 | `results/production_deployment.json` |

调研配置：`outline.yaml`（items + execution）、`fields.yaml`（字段定义）。
本文件由 `generate_report.py` 自动生成，可重复执行。
