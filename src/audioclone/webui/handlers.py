"""Gradio 事件处理逻辑。"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import gradio as gr

from audioclone.config import MAX_TEXT_LENGTH
from audioclone.core import (
    ASRService,
    AudioClassificationService,
    AudioToAudioService,
    ModelBackend,
    TextToAudioService,
    TTSService,
    VoiceActivityService,
)
from audioclone.core.capabilities import (
    AudioClassificationCapability,
    AudioToAudioCapability,
    TextToAudioCapability,
    VoiceActivityCapability,
)
from audioclone.core.exceptions import AudioError
from audioclone.utils.audio import assert_valid_ref_audio
from audioclone.utils.logging_utils import set_trace_id
from audioclone.utils.validation import assert_tts_text

logger = logging.getLogger(__name__)


def gradio_error(action_label: str, message: str) -> Callable[[Callable], Callable]:
    """Gradio handler 错误装饰器：记录日志并把异常转成 ``gr.Error``。

    - ``gr.Error``：原样透传；
    - ``AudioError``（领域异常）：直接展示其语义化消息；
    - 其余异常：记录日志后转 ``gr.Error``（附带 ``message`` 前缀）。
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            set_trace_id(uuid.uuid4().hex[:12])  # 每次调用注入 trace_id，服务层日志可关联
            try:
                return fn(*args, **kwargs)
            except gr.Error:
                raise
            except AudioError as e:
                raise gr.Error(str(e)) from e
            except Exception as e:  # noqa: BLE001
                logger.error("[WebUI] %s 失败: %s", action_label, e, exc_info=True)
                raise gr.Error(f"{message}: {e}") from e
            finally:
                set_trace_id(None)

        return wrapper

    return decorator


def _require_ref_audio(prompt_wav: str) -> None:
    assert_valid_ref_audio(prompt_wav)  # 不合法抛 InvalidAudioInput，由装饰器转 gr.Error


def make_zero_shot_handler(engine: TTSService) -> Callable[..., Any]:
    """创建零样本克隆处理函数。"""

    @gradio_error("零样本克隆", "合成失败")
    def handle(tts_text: Any, prompt_wav_upload: Any, prompt_wav_record: Any, prompt_text: Any, speed: Any) -> Any:
        prompt_wav = prompt_wav_upload or prompt_wav_record
        if not prompt_wav:
            raise gr.Error("请上传或录制参考音频")

        tts_text = assert_tts_text(tts_text, MAX_TEXT_LENGTH)
        _require_ref_audio(prompt_wav)

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


def make_cross_lingual_handler(engine: TTSService) -> Callable[..., Any]:
    """创建跨语言克隆处理函数。"""

    @gradio_error("跨语言克隆", "合成失败")
    def handle(tts_text: Any, prompt_wav: Any, speed: Any) -> Any:
        if not prompt_wav:
            raise gr.Error("请上传参考音频")

        tts_text = assert_tts_text(tts_text, MAX_TEXT_LENGTH)
        _require_ref_audio(prompt_wav)

        speed = float(speed) if speed else 1.0

        start = time.time()
        audio = engine.infer_cross_lingual(tts_text, prompt_wav, speed=speed)
        elapsed = time.time() - start
        logger.info("[WebUI] 跨语言克隆完成，耗时 %.2fs", elapsed)
        return engine.sample_rate, audio

    return handle


def make_instruct_handler(engine: TTSService) -> Callable[..., Any]:
    """创建指令控制合成处理函数。"""

    @gradio_error("指令合成", "合成失败")
    def handle(tts_text: Any, prompt_wav: Any, instruct_text: Any, speed: Any) -> Any:
        if not prompt_wav:
            raise gr.Error("请上传参考音频")
        if not instruct_text or not instruct_text.strip():
            raise gr.Error("请输入自然语言指令")

        tts_text = assert_tts_text(tts_text, MAX_TEXT_LENGTH)
        _require_ref_audio(prompt_wav)

        speed = float(speed) if speed else 1.0

        start = time.time()
        audio = engine.infer_instruct(tts_text, instruct_text, prompt_wav, speed=speed)
        elapsed = time.time() - start
        logger.info("[WebUI] 指令合成完成，耗时 %.2fs", elapsed)
        return engine.sample_rate, audio

    return handle


def make_asr_handler(asr_engine: ASRService) -> Callable[..., Any]:
    """创建 ASR 语音识别处理函数。"""
    from audioclone.config import ASR_AUDIO_MAX_SEC, ASR_SUPPORTED_FORMATS
    from audioclone.utils.audio import get_audio_duration

    @gradio_error("ASR 识别", "识别失败")
    def handle(audio_upload: Any, audio_record: Any) -> Any:
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


def make_multitask_handlers(backend: ModelBackend) -> dict[str, Callable[..., Any]]:
    """创建多任务实验台 handler（T2A / A2A / 分类 / VAD），按后端能力动态可用。"""
    sample_rate = getattr(backend, "sample_rate", 24000)

    def _guard(capability: type) -> None:
        if not backend.supports(capability):
            raise gr.Error(f"当前后端 ({backend.name}) 不支持该能力")

    @gradio_error("T2A 生成", "生成失败")
    def t2a(text: Any, duration_sec: Any) -> Any:
        _guard(TextToAudioCapability)
        if not text or not text.strip():
            raise gr.Error("请输入文本提示")
        audio = TextToAudioService(cast(TextToAudioCapability, backend)).generate(
            text, duration_sec=float(duration_sec) if duration_sec else None
        )
        return sample_rate, audio

    @gradio_error("A2A 转换", "转换失败")
    def a2a(audio_path: Any, instruct: Any) -> Any:
        _guard(AudioToAudioCapability)
        if not audio_path:
            raise gr.Error("请上传音频")
        _require_ref_audio(audio_path)  # 复用统一音频校验（格式/时长/采样率）
        audio = AudioToAudioService(cast(AudioToAudioCapability, backend)).transform(
            audio_path, instruct=instruct or ""
        )
        return sample_rate, audio

    @gradio_error("音频分类", "分类失败")
    def classify(audio_path: Any, top_k: Any = None) -> str:
        _guard(AudioClassificationCapability)
        if not audio_path:
            raise gr.Error("请上传音频")
        _require_ref_audio(audio_path)
        labels = AudioClassificationService(
            cast(AudioClassificationCapability, backend)
        ).classify(audio_path, top_k=int(top_k or 5))
        return "\n".join(f"{name}: {score:.3f}" for name, score in labels)

    @gradio_error("VAD 检测", "检测失败")
    def vad(audio_path: Any) -> str:
        _guard(VoiceActivityCapability)
        if not audio_path:
            raise gr.Error("请上传音频")
        _require_ref_audio(audio_path)
        segments = VoiceActivityService(cast(VoiceActivityCapability, backend)).vad(audio_path)
        return "\n".join(f"[{s:.2f}s - {e:.2f}s]" for s, e in segments)

    return {"t2a": t2a, "a2a": a2a, "classify": classify, "vad": vad}