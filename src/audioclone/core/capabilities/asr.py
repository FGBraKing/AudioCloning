"""语音识别 / 语音理解（Speech-to-Text）能力。"""

from __future__ import annotations

from abc import abstractmethod

from audioclone.core.capabilities.base import Capability


class SpeechRecognitionCapability(Capability):
    """ASR / 语音理解能力：对音频文件返回识别文本。

    同时承载 ASR 服务生命周期（模型/服务就绪、启停）的约定。
    """

    @property
    def is_model_available(self) -> bool:
        """模型/服务是否就绪（默认 True）。"""
        return True

    def start(self, timeout: float = 60.0) -> bool:
        """启动后端服务（默认 True）。"""
        return True

    def stop(self) -> None:
        """停止后端服务（默认空操作）。"""
        ...

    def is_ready(self) -> bool:
        """服务是否就绪（默认 True）。"""
        return True

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """对音频文件进行语音识别。

        Args:
            audio_path: 音频文件路径

        Returns:
            识别出的文本
        """