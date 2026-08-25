"""文本转音频（Text-to-Audio）能力 — 由文本提示生成音乐 / 音效。"""

from __future__ import annotations

import numpy as np

from audioclone.core.capabilities.base import Capability


class TextToAudioCapability(Capability):
    """T2A 能力：根据文本提示生成音频。

    与 TTS 的区别：此处面向「提示描述的声音/音乐/音效」，而非逐字朗读的人声。
    """

    sample_rate: int

    def text_to_audio(
        self,
        text: str,
        duration_sec: float | None = None,
        **kwargs,
    ) -> np.ndarray:
        """由文本提示生成音频。

        Args:
            text: 文本提示（描述目标声音/音乐/音效）
            duration_sec: 期望时长（秒），None 由后端决定
            **kwargs: 后端扩展参数

        Returns:
            float32 音频数组，采样率见 ``sample_rate``
        """
        raise NotImplementedError