"""语音活动检测（Voice Activity Detection）能力。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from audioclone.core.capabilities.base import Capability


class VoiceActivityCapability(Capability, ABC):
    """VAD 能力：检测音频中的说话片段（起止时刻）。"""

    @abstractmethod
    def vad(
        self,
        audio_path: str,
        **kwargs,
    ) -> list[tuple[float, float]]:
        """检测语音活动片段。

        Args:
            audio_path: 音频文件路径
            **kwargs: 后端扩展参数（如阈值、最短片段时长）

        Returns:
            按时间升序的 ``[(start_sec, end_sec), ...]`` 语音片段
        """
        raise NotImplementedError