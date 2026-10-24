"""TTS + ASR 回环交叉验证 (需要本地模型；ASR 走 sherpa-onnx)。

流程: Text → Qwen3-TTS 合成 → WAV → Qwen3-ASR 识别 → 文本比对 (LCS ≥ 80%)。
"""

from __future__ import annotations

import re

import pytest

from audioclone.config import ASSET_DIR
from audioclone.core import TTSService, create_tts_backend

TEST_TEXT = "你好，欢迎使用个人声音克隆系统。这是一段测试语音。"
REF_TEXT = "希望你以后能够做的比我还好呦。"


def _strip_punctuation(text: str) -> str:
    """去除标点符号和空格，仅保留有效字符。"""
    return re.sub(r"[^\w]", "", text)


def _lcs_length(a: str, b: str) -> int:
    """计算两个字符串的最长公共子序列长度 (空间优化: 两行 DP)。"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], prev[j - 1], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _similarity_ratio(original: str, recognized: str) -> float:
    """识别文本与原文的相似度 (LCS / 原文有效字符数)。"""
    a = _strip_punctuation(original)
    b = _strip_punctuation(recognized)
    if not a:
        return 0.0
    return _lcs_length(a, b) / len(a)


def test_lcs_basic():
    """LCS 度量正确性单元测试。"""
    assert _similarity_ratio("你好世界", "你好世界") == 1.0
    assert _similarity_ratio("你好世界", "") == 0.0
    assert _similarity_ratio("你好世界", "世界你好") < 1.0


@pytest.mark.integration
def test_roundtrip(tts_model_dir: str, asr_ready: bool):
    if not asr_ready:
        pytest.skip("ASR 模型文件不完整，请先运行: uv run audioclone-download-asr")

    ref_audio = ASSET_DIR / "zero_shot_prompt.wav"
    if not ref_audio.exists():
        pytest.skip(f"参考音频不存在: {ref_audio}")

    # 1) TTS 合成（显式指向本地 pretrained_models，避免默认走 HF 缓存/网络）
    engine = TTSService(create_tts_backend(model_dir=tts_model_dir))
    synth_path = engine.synthesize_to_file(
        mode="zero_shot",
        text=TEST_TEXT,
        ref_audio_path=str(ref_audio),
        ref_text=REF_TEXT,
        filename="test_roundtrip_tts.wav",
    )

    # 2) ASR 识别（sherpa-onnx；模型未就位则跳过）
    from audioclone.core import create_service

    asr_service = create_service("asr", "sherpa-asr-onnx")
    if not asr_service.is_model_available:
        pytest.skip("sherpa-onnx ASR 模型未就位（uv sync --extra onnx 并下载模型后可跑）")
    recognized = asr_service.transcribe(synth_path)

    # 3) 比对
    ratio = _similarity_ratio(TEST_TEXT, recognized)
    assert ratio >= 0.8, (
        f"回环验证未达标: 匹配率 {ratio * 100:.1f}% < 80% (识别: {recognized})"
    )