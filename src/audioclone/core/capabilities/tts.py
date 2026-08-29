"""文本转语音（Text-to-Speech）能力。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from audioclone.core.capabilities.base import Capability

# TTS 支持的推理模式
TTS_MODES: tuple[str, ...] = ("zero_shot", "cross_lingual", "instruct")


class TextToSpeechCapability(Capability, ABC):
    """TTS 能力：接收单段文本 + 参考音频，返回该段的音频数组。"""

    sample_rate: int

    #: 该后端支持的推理模式（``TTS_MODES`` 的子集）。
    #: 并非每个模型都支持全部模式（如 Qwen3-TTS CustomVoice 仅支持 instruct），
    #: 服务层据此在编排前 fail-fast，而不是把不兼容的调用丢给模型。
    supported_modes: tuple[str, ...] = TTS_MODES

    @abstractmethod
    def synthesize(
        self,
        mode: str,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        instruct_text: str = "",
        speed: float = 1.0,
    ) -> np.ndarray:
        """按单段文本合成。

        Args:
            mode: zero_shot / cross_lingual / instruct
            text: 合成文本
            ref_audio_path: 参考音频路径
            ref_text: 参考文本（可选）
            instruct_text: 指令文本（instruct 模式）
            speed: 语速 (0.5~2.0)

        Returns:
            float32 音频数组，采样率见 ``sample_rate``
        """