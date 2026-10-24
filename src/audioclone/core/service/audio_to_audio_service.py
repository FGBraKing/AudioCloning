"""音频到音频（Audio-to-Audio）服务 — 变声、音色转换、音质增强等。"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from audioclone.core.capabilities import AudioToAudioCapability
from audioclone.core.exceptions import CapabilityError, InvalidAudioInput


class AudioToAudioService:
    """A2A 业务服务。需传入一个提供 ``AudioToAudioCapability`` 的后端。"""

    def __init__(self, backend: AudioToAudioCapability) -> None:
        if not isinstance(backend, AudioToAudioCapability):
            raise CapabilityError(
                f"后端 {getattr(backend, 'name', backend)} 不支持音频到音频能力 "
                "(AudioToAudioCapability)"
            )
        self.backend = backend
        # 推理互斥锁：后端模型非线程安全，API 线程池并发下保护
        self._lock = threading.RLock()
        self.sample_rate = getattr(backend, "sample_rate", None)

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
            float32 音频数组
        """
        if not Path(audio_path).exists():
            raise InvalidAudioInput(f"音频文件不存在: {audio_path}")
        with self._lock:
            return self.backend.transform(audio_path, instruct=instruct, speed=speed, **kwargs)