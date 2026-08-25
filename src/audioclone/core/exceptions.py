"""领域异常体系 — 提供精确的错误分类，供上层统一处理。"""

from __future__ import annotations


class AudioError(Exception):
    """AudioCloning 领域异常基类。"""


class InvalidAudioInput(AudioError):
    """输入非法（文本为空、格式不支持、时长越界、文件不存在等）。"""


class CapabilityError(AudioError):
    """后端不具备完成任务所需的能力。"""


class BackendUnavailable(AudioError):
    """后端/模型服务未就绪（模型未加载、子进程未启动）。"""


class ModelLoadError(AudioError):
    """模型加载失败。"""


__all__ = [
    "AudioError",
    "BackendUnavailable",
    "CapabilityError",
    "InvalidAudioInput",
    "ModelLoadError",
]