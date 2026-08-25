"""音频分类（Audio Classification）服务。"""

from __future__ import annotations

from pathlib import Path

from audioclone.core.capabilities import AudioClassificationCapability


class AudioClassificationService:
    """音频分类业务服务。需传入一个提供 ``AudioClassificationCapability`` 的后端。"""

    def __init__(self, backend: AudioClassificationCapability) -> None:
        if not isinstance(backend, AudioClassificationCapability):
            raise ValueError(
                f"后端 {getattr(backend, 'name', backend)} 不支持音频分类能力 "
                "(AudioClassificationCapability)"
            )
        self.backend = backend

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
        if not Path(audio_path).exists():
            raise ValueError(f"音频文件不存在: {audio_path}")
        return self.backend.classify(audio_path, top_k=top_k, **kwargs)