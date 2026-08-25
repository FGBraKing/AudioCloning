"""音频到音频（Audio-to-Audio）能力 — 变声、音色转换、音质增强等。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from audioclone.core.capabilities.base import Capability


class AudioToAudioCapability(Capability, ABC):
    """A2A 能力：输入一段参考音频，转换/增强后输出另一段音频。"""

    sample_rate: int

    @abstractmethod
    def transform(
        self,
        audio_path: str,
        instruct: str = "",
        speed: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """对音频执行转换（变声 / 增强 / 风格迁移等）。

        Args:
            audio_path: 输入音频路径
            instruct: 转换指令（可选，如「改成男声」「增强人声」）
            speed: 语速/变速
            **kwargs: 后端扩展参数

        Returns:
            float32 音频数组，采样率见 ``sample_rate``
        """
        raise NotImplementedError