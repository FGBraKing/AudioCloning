"""Qwen3-TTS (MLX) 后端 — 覆盖 Base / CustomVoice / VoiceDesign 三种模型变体。

三个变体是**同一架构的不同能力分支**，由 ``mlx_audio`` 的 ``qwen3_tts`` 模块承载，
各自暴露不同的生成入口：

===================  ==================================================  ==============
变体                 生成入口                                            支持的模式
===================  ==================================================  ==============
``base``             ``generate(text, ref_audio, ref_text)``             zero_shot / cross_lingual
``custom_voice``     ``generate_custom_voice(text, speaker, instruct)``  instruct
``voice_design``     ``generate_voice_design(text, instruct)``           instruct
===================  ==================================================  ==============

- Base 走参考音频克隆：给 ``ref_text`` 为 ICL（上下文学习）克隆，不给则退化为
  speaker-encoder 模式（对应「跨语言」：不依赖参考音频转录文本）。
- CustomVoice / VoiceDesign 不消费参考音频，音色由 ``speaker`` 或自然语言描述决定，
  因此只映射 ``instruct`` 模式。

依赖说明
--------
``qwen3_tts`` 架构自 ``mlx-audio >= 0.3.0`` 起提供；若当前 ``mlx_audio`` 不含该模块，
:meth:`Qwen3TTSMLXBackend.load` 会抛出带修复指引的 :class:`ModelLoadError`，
而不是在深层堆栈里报一个难以理解的 ImportError。
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

from audioclone.config import SAMPLE_RATE
from audioclone.core.capabilities import ModelBackend, TextToSpeechCapability
from audioclone.core.exceptions import InvalidAudioInput, ModelLoadError
from audioclone.core.model_slot import ModelSlot

logger = logging.getLogger(__name__)

#: mlx_audio 中承载 Qwen3-TTS 的架构模块名
ARCH = "qwen3_tts"

#: 依赖缺失时的修复指引（日志与异常共用）
_HINT = (
    "当前 mlx_audio 未提供 Qwen3-TTS 架构（mlx_audio.tts.models.qwen3_tts）。\n"
    "  - 本项目依赖上游 mlx-audio（mlx_audio.tts.models.qwen3_tts 自 0.3.0 起提供）；\n"
    "  - 请检查是否误装了 mlx-audio-plus（CosyVoice 分支，基线 mlx-audio 0.2.9，不含该模块），\n"
    "    若是请执行: uv remove mlx-audio-plus && uv add 'mlx-audio[tts]>=0.5.0'；\n"
    "  - 或直接重装: uv sync --reinstall-package mlx-audio"
)


def has_qwen3_tts_support() -> bool:
    """当前环境的 ``mlx_audio`` 是否提供 Qwen3-TTS 架构。"""
    try:
        return importlib.util.find_spec(f"mlx_audio.tts.models.{ARCH}") is not None
    except (ImportError, ValueError):  # 父包缺失 / 包名不合法
        return False


class Qwen3TTSMLXBackend(ModelBackend, TextToSpeechCapability):
    """Qwen3-TTS MLX 后端基类：按 ``variant`` 分派到不同的生成入口。"""

    #: 模型变体（base / custom_voice / voice_design），由子类声明
    variant: str = ""
    capabilities = (TextToSpeechCapability,)
    slot = ModelSlot.TTS

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        speaker: str = "Vivian",
        language: str = "auto",
    ) -> None:
        from audioclone.config import QWEN3_TTS_MODEL_DIR

        self.model_id = model_id or ""
        # 目录名连写为 qwen3_tts_<变体>：避免被 mlx_audio 按 "-" 切分后误判架构
        default_dir = QWEN3_TTS_MODEL_DIR / f"{ARCH}_{self.variant}"
        self.model_dir = Path(model_dir) if model_dir else default_dir
        self.speaker = speaker          # custom_voice：预设音色名
        self.language = language        # 语言代码（auto / Chinese / English ...）
        self._model: Any = None
        self.sample_rate: int = SAMPLE_RATE

    # ─── 生命周期 ────────────────────────────────────────────
    def _load(self) -> None:
        if self._model is not None:
            return
        if not has_qwen3_tts_support():
            raise ModelLoadError(_HINT)

        from mlx_audio.tts.utils import load_model as mlx_load_model

        logger.info("[TTS] 加载 Qwen3-TTS (%s): %s", self.variant, self.model_dir)
        self._model = self._load_model(mlx_load_model)
        self.sample_rate = int(getattr(self._model, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        logger.info("[TTS] Qwen3-TTS %s 加载完成，采样率: %s", self.variant, self.sample_rate)

    def _load_model(self, loader: Any) -> Any:
        """调用 ``mlx_audio`` 加载器，并显式锁定架构。

        ``mlx_audio`` 默认会按**目录名片段**猜测架构，而 ``Qwen3-TTS-...`` 会被切成
        ``['qwen3', 'tts', ...]``，其中 ``qwen3`` 命中另一个（VyvoTTS）模块，
        导致用错误的架构去加载权重。这里显式传入 ``model_type`` / ``model_name_parts``
        锁定 ``qwen3_tts``；旧版加载器不支持这些关键字时回退为默认推断。
        """
        try:
            return loader(self.model_dir, model_type=ARCH, model_name_parts=[ARCH])
        except TypeError:
            logger.warning(
                "[TTS] 当前 mlx_audio 不支持显式 model_type，回退为按目录名推断架构；"
                "若加载报维度/权重不匹配，请将模型目录重命名为以 '%s' 开头的名称",
                ARCH,
            )
            return loader(self.model_dir)

    def _close(self) -> None:
        self._model = None

    # ─── 合成 ────────────────────────────────────────────────
    def synthesize(
        self,
        mode: str,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        instruct_text: str = "",
        speed: float = 1.0,
    ) -> np.ndarray:
        self.load()
        if mode not in self.supported_modes:
            raise InvalidAudioInput(
                f"后端 {self.name}（{self.variant}）不支持推理模式 '{mode}'，"
                f"仅支持: {', '.join(self.supported_modes)}"
            )

        logger.info("[TTS] Qwen3-TTS mode=%s variant=%s | 文本: %s...", mode, self.variant, text[:50])
        if mode == "instruct":
            audio = self._synth_instruct(text, instruct_text)
        else:
            audio = self._synth_clone(text, ref_audio_path, ref_text if mode == "zero_shot" else "")
        return np.asarray(audio, dtype=np.float32)

    def _synth_clone(self, text: str, ref_audio_path: str, ref_text: str) -> Any:
        """Base 变体：参考音频克隆（给 ref_text 走 ICL，不给走 speaker-encoder）。"""
        return self._collect(
            self._model.generate(
                text=text,
                ref_audio=ref_audio_path,
                ref_text=ref_text or None,
                lang_code=self.language,
                verbose=False,
            )
        )

    def _synth_instruct(self, text: str, instruct_text: str) -> Any:
        """CustomVoice / VoiceDesign 变体：按指令合成（子类实现）。"""
        raise NotImplementedError

    @staticmethod
    def _collect(results: Any) -> np.ndarray:
        """把生成器的多个音频片段拼成单段数组。"""
        chunks = [np.asarray(getattr(r, "audio", r), dtype=np.float32) for r in results]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            raise ModelLoadError("Qwen3-TTS 未生成任何音频")
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    # ─── 状态 ────────────────────────────────────────────────
    def get_supported_speakers(self) -> list[str]:
        """返回可用的预设音色（仅 CustomVoice 有意义）。"""
        getter = getattr(self._model, "get_supported_speakers", None)
        return list(getter()) if callable(getter) else []


class Qwen3TTSBaseMLXBackend(Qwen3TTSMLXBackend):
    """Qwen3-TTS Base — 参考音频零样本克隆 / 跨语言克隆。"""

    name = "qwen3-tts-base-mlx"
    variant = "base"
    supported_modes = ("zero_shot", "cross_lingual")


class Qwen3TTSCustomVoiceMLXBackend(Qwen3TTSMLXBackend):
    """Qwen3-TTS CustomVoice — 预设音色 + 情绪/风格指令控制。"""

    name = "qwen3-tts-customvoice-mlx"
    variant = "custom_voice"
    supported_modes = ("instruct",)

    def _synth_instruct(self, text: str, instruct_text: str) -> Any:
        return self._collect(
            self._model.generate_custom_voice(
                text=text,
                speaker=self.speaker,
                language=self.language,
                instruct=instruct_text or None,
                verbose=False,
            )
        )


class Qwen3TTSVoiceDesignMLXBackend(Qwen3TTSMLXBackend):
    """Qwen3-TTS VoiceDesign — 用自然语言描述音色（无需参考音频）。"""

    name = "qwen3-tts-voicedesign-mlx"
    variant = "voice_design"
    supported_modes = ("instruct",)

    def _synth_instruct(self, text: str, instruct_text: str) -> Any:
        if not instruct_text.strip():
            raise InvalidAudioInput("VoiceDesign 后端需要 instruct_text 描述期望音色")
        return self._collect(
            self._model.generate_voice_design(
                text=text,
                instruct=instruct_text,
                language=self.language,
                verbose=False,
            )
        )


__all__ = [
    "Qwen3TTSBaseMLXBackend",
    "Qwen3TTSCustomVoiceMLXBackend",
    "Qwen3TTSMLXBackend",
    "Qwen3TTSVoiceDesignMLXBackend",
    "has_qwen3_tts_support",
]
