"""Gradio 事件处理逻辑。"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gradio as gr

from audioclone.config import MAX_TEXT_LENGTH
from audioclone.core import ASRService, TTSService
from audioclone.utils.audio import assert_valid_ref_audio

logger = logging.getLogger(__name__)


def gradio_error(action_label: str, message: str) -> Callable[[Callable], Callable]:
    """Gradio handler 错误装饰器：记录日志并把异常转成 ``gr.Error``。

    业务抛出的 ``gr.Error`` 会原样透传；其余异常记录后转为 ``gr.Error``。
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except gr.Error:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("[WebUI] %s 失败: %s", action_label, e, exc_info=True)
                raise gr.Error(f"{message}: {e}") from e

        return wrapper

    return decorator


def _require_ref_audio(prompt_wav: str) -> None:
    assert_valid_ref_audio(prompt_wav)  # 不合法抛 InvalidAudioInput，由装饰器转 gr.Error


def make_zero_shot_handler(engine: TTSService):
    """创建零样本克隆处理函数。"""

    @gradio_error("零样本克隆", "合成失败")
    def handle(tts_text, prompt_wav_upload, prompt_wav_record, prompt_text, speed):
        prompt_wav = prompt_wav_upload or prompt_wav_record
        if not prompt_wav:
            raise gr.Error("请上传或录制参考音频")
        if not tts_text or not tts_text.strip():
            raise gr.Error("请输入合成文本")

        _require_ref_audio(prompt_wav)

        if len(tts_text) > MAX_TEXT_LENGTH:
            raise gr.Error(f"文本过长 ({len(tts_text)} > {MAX_TEXT_LENGTH} 字符)")

        speed = float(speed) if speed else 1.0

        logger.info("[WebUI] 零样本克隆")
        start = time.time()
        audio = engine.infer_zero_shot(
            tts_text, prompt_wav, prompt_text or "", speed=speed
        )
        elapsed = time.time() - start
        logger.info("[WebUI] 零样本克隆完成，耗时 %.2fs", elapsed)
        return engine.sample_rate, audio

    return handle


def make_cross_lingual_handler(engine: TTSService):
    """创建跨语言克隆处理函数。"""

    @gradio_error("跨语言克隆", "合成失败")
    def handle(tts_text, prompt_wav, speed):
        if not prompt_wav:
            raise gr.Error("请上传参考音频")
        if not tts_text or not tts_text.strip():
            raise gr.Error("请输入合成文本")

        _require_ref_audio(prompt_wav)

        speed = float(speed) if speed else 1.0

        start = time.time()
        audio = engine.infer_cross_lingual(tts_text, prompt_wav, speed=speed)
        elapsed = time.time() - start
        logger.info("[WebUI] 跨语言克隆完成，耗时 %.2fs", elapsed)
        return engine.sample_rate, audio

    return handle


def make_instruct_handler(engine: TTSService):
    """创建指令控制合成处理函数。"""

    @gradio_error("指令合成", "合成失败")
    def handle(tts_text, prompt_wav, instruct_text, speed):
        if not prompt_wav:
            raise gr.Error("请上传参考音频")
        if not tts_text or not tts_text.strip():
            raise gr.Error("请输入合成文本")
        if not instruct_text or not instruct_text.strip():
            raise gr.Error("请输入自然语言指令")

        _require_ref_audio(prompt_wav)

        speed = float(speed) if speed else 1.0

        start = time.time()
        audio = engine.infer_instruct(tts_text, instruct_text, prompt_wav, speed=speed)
        elapsed = time.time() - start
        logger.info("[WebUI] 指令合成完成，耗时 %.2fs", elapsed)
        return engine.sample_rate, audio

    return handle


def make_asr_handler(asr_engine: ASRService):
    """创建 ASR 语音识别处理函数。"""
    from audioclone.config import ASR_AUDIO_MAX_SEC, ASR_SUPPORTED_FORMATS
    from audioclone.utils.audio import get_audio_duration

    @gradio_error("ASR 识别", "识别失败")
    def handle(audio_upload, audio_record):
        audio_path = audio_upload or audio_record
        if not audio_path:
            raise gr.Error("请上传或录制音频")

        suffix = Path(audio_path).suffix.lower()
        if suffix not in ASR_SUPPORTED_FORMATS:
            raise gr.Error(
                f"不支持的音频格式: {suffix}，支持: {', '.join(ASR_SUPPORTED_FORMATS)}"
            )

        duration = get_audio_duration(audio_path)
        if duration <= 0:
            raise gr.Error("无法读取音频文件")
        if duration > ASR_AUDIO_MAX_SEC:
            raise gr.Error(f"音频时长过长 ({duration:.1f}s > {ASR_AUDIO_MAX_SEC}s)")

        if not asr_engine.is_ready():
            raise gr.Error("ASR 服务未就绪，请等待模型加载完成")

        logger.info("[WebUI] ASR 识别: %s (%.1fs)", audio_path, duration)
        start = time.time()
        text = asr_engine.transcribe(audio_path)
        elapsed = time.time() - start
        info = f"识别完成 | 音频时长: {duration:.1f}s | 耗时: {elapsed:.2f}s"
        logger.info("[WebUI] ASR 识别完成，耗时 %.2fs", elapsed)

        return text, info

    return handle