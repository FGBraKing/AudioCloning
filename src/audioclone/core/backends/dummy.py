"""Dummy 后端 — 伪模型，用于框架级测试 / 快速预览（无需真实模型与 GPU）。"""

from __future__ import annotations

import numpy as np

from audioclone.core.capabilities import (
    AudioClassificationCapability,
    AudioToAudioCapability,
    ModelBackend,
    SpeechRecognitionCapability,
    TextToAudioCapability,
    TextToSpeechCapability,
    VoiceActivityCapability,
)

# 生成的静音样本默认时长（秒）
DEFAULT_SILENCE_SEC = 0.5


class DummyTTSBackend(ModelBackend, TextToSpeechCapability):
    """返回确定性静音片段的假 TTS 后端，用于验证业务链路。"""

    name = "dummy"

    def __init__(self, sample_rate: int = 24000, duration_sec: float = DEFAULT_SILENCE_SEC) -> None:
        self.sample_rate = sample_rate
        self._duration = duration_sec

    def synthesize(self, mode, text, ref_audio_path, ref_text="", instruct_text="", speed=1.0):
        text = text or " "
        n = int(self.sample_rate * self._duration)
        # 用文本长度影响幅度，便于断言“调用确实发生了”
        amp = (len(text) % 3 + 1) * 0.1
        return (np.ones(n, dtype=np.float32) * amp).astype(np.float32)


class DummyASRBackend(ModelBackend, SpeechRecognitionCapability):
    """返回固定占位文本的假 ASR 后端。"""

    name = "dummy"
    RESULT = "这是一段模拟识别结果。"

    def transcribe(self, audio_path: str) -> str:
        return self.RESULT


class DummyOmnibackend(
    ModelBackend,
    TextToSpeechCapability,
    TextToAudioCapability,
    SpeechRecognitionCapability,
    AudioToAudioCapability,
    AudioClassificationCapability,
    VoiceActivityCapability,
):
    """全能力假后端 — 演示「一个模型实现全部六种音频任务」。"""

    name = "dummy-omni"
    ASR_RESULT = "这是一段模拟识别结果。"

    def __init__(self, sample_rate: int = 24000, duration_sec: float = DEFAULT_SILENCE_SEC) -> None:
        self.sample_rate = sample_rate
        self._duration = duration_sec

    def synthesize(self, mode, text, ref_audio_path, ref_text="", instruct_text="", speed=1.0):
        n = int(self.sample_rate * self._duration)
        return np.ones(n, dtype=np.float32).astype(np.float32)

    def text_to_audio(self, text, duration_sec=None, **kwargs):
        n = int(self.sample_rate * (duration_sec or self._duration))
        return np.ones(n, dtype=np.float32).astype(np.float32)

    def transform(self, audio_path, instruct="", speed=1.0, **kwargs):
        n = int(self.sample_rate * self._duration)
        return np.ones(n, dtype=np.float32).astype(np.float32)

    def transcribe(self, audio_path: str) -> str:
        return self.ASR_RESULT

    def classify(self, audio_path, top_k=5, **kwargs):
        return [("speech", 0.9), ("music", 0.08), ("noise", 0.02)][:top_k]

    def vad(self, audio_path, **kwargs):
        return [(0.0, 1.0)]


__all__ = ["DummyTTSBackend", "DummyASRBackend", "DummyOmnibackend"]