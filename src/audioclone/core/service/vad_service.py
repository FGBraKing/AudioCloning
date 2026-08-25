"""语音活动检测（Voice Activity Detection）服务。"""

from __future__ import annotations

from pathlib import Path

from audioclone.core.capabilities import VoiceActivityCapability


class VoiceActivityService:
    """VAD 业务服务。需传入一个提供 ``VoiceActivityCapability`` 的后端。"""

    def __init__(self, backend: VoiceActivityCapability) -> None:
        if not isinstance(backend, VoiceActivityCapability):
            raise ValueError(
                f"后端 {getattr(backend, 'name', backend)} 不支持语音活动检测能力 "
                "(VoiceActivityCapability)"
            )
        self.backend = backend

    def vad(self, audio_path: str, **kwargs) -> list[tuple[float, float]]:
        """检测语音活动片段。

        Args:
            audio_path: 音频文件路径
            **kwargs: 后端扩展参数（如阈值、最短片段时长）

        Returns:
            按时间升序的 ``[(start_sec, end_sec), ...]`` 语音片段
        """
        if not Path(audio_path).exists():
            raise ValueError(f"音频文件不存在: {audio_path}")
        return self.backend.vad(audio_path, **kwargs)