"""音频分类（Audio Classification）能力。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from audioclone.core.capabilities.base import Capability


class AudioClassificationCapability(Capability, ABC):
    """对音频进行分类（如语音/音乐/环境声 / 语种 / 事件识别）。"""

    @abstractmethod
    def classify(
        self,
        audio_path: str,
        top_k: int = 5,
        **kwargs,
    ) -> list[tuple[str, float]]:
        """对音频分类，返回 ``[(标签, 置信度)]`` 列表。

        Args:
            audio_path: 音频文件路径
            top_k: 返回置信度最高的前 k 个类别
            **kwargs: 后端扩展参数

        Returns:
            按置信度降序的 (标签, 分数) 列表
        """
        raise NotImplementedError