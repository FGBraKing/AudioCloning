"""Gradio 事件处理逻辑。"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import gradio as gr

from audioclone.config import MAX_TEXT_LENGTH
from audioclone.core import ASRService, TTSService
from audioclone.utils.audio import validate_ref_audio

logger = logging.getLogger(__name__)


def make_zero_shot_handler(engine: TTSService):
    """创建零样本克隆处理函数。"""

    def handle(tts_text, prompt_wav_upload, prompt_wav_record, prompt_text, speed):
        prompt_wav = prompt_wav_upload or prompt_wav_record
        if not prompt_wav:
            raise gr.Error("请上传或录制参考音频")
        if not tts_text or not tts_text.strip():
            raise gr.Error("请输入合成文本")

        valid, msg = validate_ref_audio(prompt_wav)
        if not valid:
            raise gr.Error(f"参考音频不合法: {msg}")

        if len(tts_text) > MAX_TEXT_LENGTH:
            raise gr.Error(f"文本过长 ({len(tts_text)} > {MAX_TEXT_LENGTH} 字符)")

        speed = float(speed) if speed else 1.0

        logger.info("[WebUI] 零样本克隆")
        start = time.time()
        try:
            audio = engine.infer_zero_shot(
                tts_text, prompt_wav, prompt_text or "", speed=speed
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[WebUI] 零样本克隆失败: %s", e)
            raise gr.Error(f"合成失败: {e}") from e

        elapsed = time.time() - start
        logger.info("[WebUI] 零样本克隆完成，耗时 %.2fs", elapsed)
        return engine.sample_rate, audio

    return handle


def make_cross_lingual_handler(engine: TTSService):
    """创建跨语言克隆处理函数。"""

    def handle(tts_text, prompt_wav, speed):
        if not prompt_wav:
            raise gr.Error("请上传参考音频")
        if not tts_text or not tts_text.strip():
            raise gr.Error("请输入合成文本")

        valid, msg = validate_ref_audio(prompt_wav)
        if not valid:
            raise gr.Error(f"参考音频不合法: {msg}")

        speed = float(speed) if speed else 1.0

        start = time.time()
        try:
            audio = engine.infer_cross_lingual(tts_text, prompt_wav, speed=speed)
        except Exception as e:  # noqa: BLE001
            logger.error("[WebUI] 跨语言克隆失败: %s", e)
            raise gr.Error(f"合成失败: {e}") from e

        elapsed = time.time() - start
        logger.info("[WebUI] 跨语言克隆完成，耗时 %.2fs", elapsed)
        return engine.sample_rate, audio

    return handle


def make_instruct_handler(engine: TTSService):
    """创建指令控制合成处理函数。"""

    def handle(tts_text, prompt_wav, instruct_text, speed):
        if not prompt_wav:
            raise gr.Error("请上传参考音频")
        if not tts_text or not tts_text.strip():
            raise gr.Error("请输入合成文本")
        if not instruct_text or not instruct_text.strip():
            raise gr.Error("请输入自然语言指令")

        valid, msg = validate_ref_audio(prompt_wav)
        if not valid:
            raise gr.Error(f"参考音频不合法: {msg}")

        speed = float(speed) if speed else 1.0

        start = time.time()
        try:
            audio = engine.infer_instruct(tts_text, instruct_text, prompt_wav, speed=speed)
        except Exception as e:  # noqa: BLE001
            logger.error("[WebUI] 指令合成失败: %s", e)
            raise gr.Error(f"合成失败: {e}") from e

        elapsed = time.time() - start
        logger.info("[WebUI] 指令合成完成，耗时 %.2fs", elapsed)
        return engine.sample_rate, audio

    return handle


def make_asr_handler(asr_engine: ASRService):
    """创建 ASR 语音识别处理函数。"""
    from audioclone.config import ASR_AUDIO_MAX_SEC, ASR_SUPPORTED_FORMATS
    from audioclone.utils.audio import get_audio_duration

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
        try:
            text = asr_engine.transcribe(audio_path)
        except Exception as e:  # noqa: BLE001
            logger.error("[WebUI] ASR 识别失败: %s", e)
            raise gr.Error(f"识别失败: {e}") from e

        elapsed = time.time() - start
        info = f"识别完成 | 音频时长: {duration:.1f}s | 耗时: {elapsed:.2f}s"
        logger.info("[WebUI] ASR 识别完成，耗时 %.2fs", elapsed)

        return text, info

    return handle