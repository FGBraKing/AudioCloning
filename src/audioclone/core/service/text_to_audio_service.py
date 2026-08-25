"""文本转音频（Text-to-Audio）服务 — 由文本提示生成音乐 / 音效。"""

from __future__ import annotations

import numpy as np

from audioclone.core.capabilities import TextToAudioCapability
from audioclone.core.exceptions import CapabilityError, InvalidAudioInput


class TextToAudioService:
    """T2A 业务服务。需传入一个提供 ``TextToAudioCapability`` 的后端。"""

    def __init__(self, backend: TextToAudioCapability) -> None:
        if not isinstance(backend, TextToAudioCapability):
            raise CapabilityError(
                f"后端 {getattr(backend, 'name', backend)} 不支持文本转音频能力 "
                "(TextToAudioCapability)"
            )
        self.backend = backend
        self.sample_rate = getattr(backend, "sample_rate", None)

    def generate(self, text: str, duration_sec: float | None = None, **kwargs) -> np.ndarray:
        """根据文本提示生成音频。

        Args:
            text: 文本提示（描述目标声音/音乐/音效）
            duration_sec: 期望时长（秒），None 由后端决定
            **kwargs: 后端扩展参数

        Returns:
            float32 音频数组
        """
        if not text or not text.strip():
            raise InvalidAudioInput("文本提示为空")
        return self.backend.text_to_audio(text, duration_sec=duration_sec, **kwargs)