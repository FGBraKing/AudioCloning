#!/usr/bin/env python3
"""
TTS + ASR 回环交叉验证测试
流程: Text → CosyVoice3 合成 → WAV → Qwen3-ASR 识别 → Text → 比对
"""

import sys
import os
import re
import time
import logging

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_roundtrip")


# ─── 文本比对工具 ─────────────────────────────────────────────

def strip_punctuation(text: str) -> str:
    """去除标点符号和空格，仅保留有效字符"""
    return re.sub(r"[^\w]", "", text)


def lcs_length(a: str, b: str) -> int:
    """计算两个字符串的最长公共子序列长度"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # 空间优化: 只保留两行
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def similarity_ratio(original: str, recognized: str) -> float:
    """计算识别文本与原始文本的相似度 (LCS / 原文长度)"""
    a = strip_punctuation(original)
    b = strip_punctuation(recognized)
    if not a:
        return 0.0
    lcs_len = lcs_length(a, b)
    return lcs_len / len(a)


# ─── 主测试流程 ─────────────────────────────────────────────

def main():
    from src.config import ASSET_DIR, OUTPUT_DIR
    from src.tts import load_model, TTSEngine
    from src.asr import ASREngine

    # 测试文本
    test_text = "你好，欢迎使用个人声音克隆系统。这是一段测试语音。"
    ref_audio = str(ASSET_DIR / "zero_shot_prompt.wav")
    ref_text = "希望你以后能够做的比我还好呦。"
    output_wav = str(OUTPUT_DIR / "test_roundtrip_tts.wav")

    print("=" * 60)
    print("  TTS + ASR 回环交叉验证")
    print("=" * 60)
    print(f"\n  原始文本: {test_text}")
    print(f"  参考音频: {ref_audio}")

    # ─── Step 1: 加载 TTS 模型 ───────────────────────────────
    print("\n[1/5] 加载 CosyVoice3 TTS 模型...")
    t0 = time.time()
    mlx_model = load_model()
    tts_engine = TTSEngine(mlx_model)
    print(f"  TTS 模型就绪，耗时 {time.time() - t0:.1f}s，采样率 {tts_engine.sample_rate}Hz")

    # ─── Step 2: TTS 合成 ────────────────────────────────────
    print("\n[2/5] TTS 合成音频...")
    t0 = time.time()
    try:
        result_path = tts_engine.synthesize_to_file(
            mode="zero_shot",
            text=test_text,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            filename="test_roundtrip_tts.wav",
        )
        tts_elapsed = time.time() - t0
        print(f"  合成完成: {result_path}")
        print(f"  TTS 耗时: {tts_elapsed:.2f}s")
    except Exception as e:
        print(f"\n[ERROR] TTS 合成失败: {e}")
        sys.exit(1)

    # ─── Step 3: 启动 ASR 引擎 ───────────────────────────────
    print("\n[3/5] 启动 ASR 引擎 (llama-server)...")
    asr_engine = ASREngine()
    if not asr_engine.is_model_available:
        print("\n[ERROR] ASR 模型不可用，请先运行: python download_asr_model.py")
        sys.exit(1)

    t0 = time.time()
    if not asr_engine.start_server(timeout=90.0):
        print("\n[ERROR] ASR 服务启动失败")
        sys.exit(1)
    print(f"  ASR 服务就绪，耗时 {time.time() - t0:.1f}s")

    # ─── Step 4: ASR 识别 ────────────────────────────────────
    print(f"\n[4/5] ASR 识别合成音频...")
    try:
        t0 = time.time()
        recognized_text = asr_engine.transcribe(result_path)
        asr_elapsed = time.time() - t0
        print(f"  识别结果: {recognized_text}")
        print(f"  ASR 耗时: {asr_elapsed:.2f}s")
    except Exception as e:
        print(f"\n[ERROR] ASR 识别失败: {e}")
        asr_engine.stop_server()
        sys.exit(1)

    # ─── Step 5: 文字比对 ────────────────────────────────────
    print("\n[5/5] 文字交叉比对...")
    ratio = similarity_ratio(test_text, recognized_text)
    orig_clean = strip_punctuation(test_text)
    recog_clean = strip_punctuation(recognized_text)

    print(f"\n  原文 (去标点): {orig_clean}")
    print(f"  识别 (去标点): {recog_clean}")
    print(f"  LCS 匹配率:   {ratio * 100:.1f}%")

    # 停止 ASR 服务
    asr_engine.stop_server()

    # ─── 结果判定 ────────────────────────────────────────────
    print("\n" + "=" * 60)
    if ratio >= 0.8:
        print(f"  回环验证通过! 匹配率 {ratio * 100:.1f}% >= 80%")
    else:
        print(f"  回环验证未达标: 匹配率 {ratio * 100:.1f}% < 80%")
        print("  (TTS 合成质量或 ASR 识别精度可能需要检查)")
    print("=" * 60)

    # 返回退出码
    sys.exit(0 if ratio >= 0.8 else 1)


if __name__ == "__main__":
    main()
