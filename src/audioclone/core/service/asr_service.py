"""ASR 服务 — 业务编排：格式 / 时长校验。

只依赖 ``ASRBackend`` 契约，具体模型协议（子进程管理、HTTP 调用等）封装在后端实现中。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from audioclone.config import ASR_AUDIO_MAX_SEC, ASR_SUPPORTED_FORMATS
from audioclone.core.capabilities import SpeechRecognitionCapability
from audioclone.core.exceptions import BackendUnavailable, CapabilityError, InvalidAudioInput
from audioclone.utils.audio import get_audio_duration

logger = logging.getLogger(__name__)


class ASRService:
    """ASR 业务服务。需传入一个提供 ``SpeechRecognitionCapability`` 的后端。"""

    def __init__(self, backend: SpeechRecognitionCapability) -> None:
        if not isinstance(backend, SpeechRecognitionCapability):
            raise CapabilityError(
                f"后端 {getattr(backend, 'name', backend)} 不支持语音识别能力 "
                "(SpeechRecognitionCapability)"
            )
        self.backend = backend
        # 推理互斥锁：后端模型非线程安全，API 线程池并发下保护
        self._lock = threading.RLock()

    # ─── 模型能力查询 ────────────────────────────────────────
    @property
    def is_model_available(self) -> bool:
        return self.backend.is_model_available

    def is_ready(self) -> bool:
        return self.backend.is_ready()

    def get_status(self) -> dict[str, Any]:
        """组合后端默认信息与自身业务信息。"""
        get_status = getattr(self.backend, "get_status", None)
        status: dict[str, Any] = get_status() if callable(get_status) else {}
        status["backend"] = getattr(self.backend, "name", type(self.backend).__name__)
        return status

    # ─── 生命周期 ────────────────────────────────────────────
    def start_server(self, timeout: float = 60.0) -> bool:
        return self.backend.start(timeout=timeout)

    def stop_server(self) -> None:
        self.backend.stop()

    # ─── 识别 ────────────────────────────────────────────────
    def transcribe(self, audio_path: str) -> str:
        """对音频进行识别，先做业务层校验。"""
        suffix = Path(audio_path).suffix.lower()
        if suffix not in ASR_SUPPORTED_FORMATS:
            raise InvalidAudioInput(
                f"不支持的音频格式: {suffix}，支持: {', '.join(ASR_SUPPORTED_FORMATS)}"
            )

        duration = get_audio_duration(audio_path)
        if duration <= 0:
            raise InvalidAudioInput("无法读取音频文件")
        if duration > ASR_AUDIO_MAX_SEC:
            raise InvalidAudioInput(f"音频时长过长 ({duration:.1f}s > {ASR_AUDIO_MAX_SEC}s)")

        if not self.backend.is_ready():
            raise BackendUnavailable("ASR 服务未就绪，请等待模型加载完成")

        with self._lock:
            return self.backend.transcribe(audio_path)

    # ─── 上下文管理器 ────────────────────────────────────────
    def __enter__(self) -> ASRService:
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_server()
        return None