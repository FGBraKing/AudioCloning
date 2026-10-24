"""Qwen3-TTS (TensorRT Edge-LLM) 后端 — 封装 NVIDIA 官方 TTS 推理步。

NVIDIA 平台的默认 TTS：TensorRT Edge-LLM **官方支持 Qwen3-TTS 全三变体**
（Base/CustomVoice/VoiceDesign，共享 Talker + CodePredictor + Code2Wav 流水线），
与本项目 MLX 侧的 TTS 默认模型同族同能力。

工作流（本后端只封装第 3 步；权重由用户在 x86 主机自行导出，**本项目不下载**）：

1. **导出**（x86 主机）：``tensorrt-edgellm-export`` 产出三组件 ONNX
   （Talker / CodePredictor / Code2Wav；Base 变体额外含 ``clone_encoders/``）
2. **构建**（目标设备）：``llm_build`` / ``audio_build`` 生成 TensorRT 引擎
3. **推理**（本后端封装）：``qwen3_tts_inference`` 接收请求 JSON，输出 WAV

本机无 NVIDIA 硬件时的行为：``_load()`` 校验引擎目录与推理工具的存在性并给出
可操作指引，**不运行、不下载**；单元测试以假目录/假推理工具覆盖调用契约。
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from audioclone.config import QWEN3_TTS_MODEL_IDS
from audioclone.core.capabilities import ModelBackend, TextToSpeechCapability
from audioclone.core.exceptions import InvalidAudioInput, ModelLoadError
from audioclone.core.model_slot import ModelSlot

logger = logging.getLogger(__name__)

#: Edge-LLM 官方推理工具名（构建后随框架提供，需在 PATH 或显式给路径）
INFERENCE_BIN = "qwen3_tts_inference"


def _find_inference_bin(inference_bin: str) -> Path | None:
    """定位 ``qwen3_tts_inference``：显式路径或 PATH 查找。"""
    if "/" in inference_bin or "\\" in inference_bin:
        path = Path(inference_bin).expanduser()
        return path if path.is_file() else None
    which = shutil.which(inference_bin)
    return Path(which) if which else None


class Qwen3TTSTensorRTBackend(ModelBackend, TextToSpeechCapability):
    """Qwen3-TTS TensorRT Edge-LLM 后端基类：按 ``variant`` 分派请求字段。"""

    #: 模型变体（base / custom_voice / voice_design），由子类声明
    variant: str = ""
    #: 对应的 HuggingFace 源仓库（仅作导出参考；权重不由本项目下载）
    default_model_id: str = ""
    capabilities = (TextToSpeechCapability,)
    slot = ModelSlot.TTS

    def __init__(
        self,
        model_id: str | None = None,
        model_dir: str | Path | None = None,
        *,
        inference_bin: str = INFERENCE_BIN,
        speaker: str = "vivian",
        language: str = "auto",
    ) -> None:
        self.model_id = model_id or self.default_model_id
        # model_dir = Edge-LLM 构建产物目录（TensorRT 引擎），非 HF 权重目录
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.inference_bin = inference_bin
        self.speaker = speaker          # custom_voice：预设音色名（如 vivian/ryan/serena）
        self.language = language
        self._ready = False
        self.sample_rate = 24000        # 以推理输出的 WAV 头为准

    # ─── 生命周期（校验式加载：不下载、不构建） ───────────────
    def _load(self) -> None:
        if self._ready:
            return
        if not self.model_dir:
            raise ModelLoadError(
                f"{self.name} 需提供 TensorRT 引擎目录（model_dir / --model_dir / "
                "AUDIOCLONE_TTS_MODEL_DIR）。引擎由 TensorRT Edge-LLM 生成：\n"
                "  1. x86 主机: tensorrt-edgellm-export（导出 Qwen3-TTS 三组件 ONNX）\n"
                "  2. 目标设备: llm_build / audio_build（构建引擎）\n"
                "本项目不下载权重、不执行构建。"
            )
        if not self.model_dir.exists():
            raise ModelLoadError(f"TensorRT 引擎目录不存在: {self.model_dir}")
        if not any(self.model_dir.iterdir()):
            raise ModelLoadError(
                f"TensorRT 引擎目录为空: {self.model_dir}（请先完成 Edge-LLM export + build）"
            )
        self._inference_path = _find_inference_bin(self.inference_bin)
        if self._inference_path is None:
            raise ModelLoadError(
                f"未找到推理工具 '{self.inference_bin}'（Edge-LLM 构建产物，"
                "请将其所在目录加入 PATH 或以 model_dir 同级的显式路径传入）"
            )
        self._ready = True
        logger.info("[TTS] TensorRT Edge-LLM 引擎就绪 (%s): %s", self.variant, self.model_dir)

    def _close(self) -> None:
        self._ready = False

    # ─── 合成（子进程调用官方推理工具） ───────────────────────
    def synthesize(
        self,
        mode: str,
        text: str,
        ref_audio_path: str = "",
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

        request = self._build_request(mode, text, ref_audio_path, ref_text, instruct_text, speed)
        audio = self._run_inference(request)
        return np.asarray(audio, dtype=np.float32)

    def _build_request(
        self, mode: str, text: str, ref_audio: str, ref_text: str, instruct: str, speed: float
    ) -> dict[str, Any]:
        """构造 ``qwen3_tts_inference`` 请求 JSON（字段对应 Edge-LLM 变体能力）。"""
        request: dict[str, Any] = {
            "tts_model_type": self.variant,
            "text": text,
            "language": self.language,
            "speed": speed,
        }
        if self.variant == "base":
            # 零样本克隆：给 ref_text 走 ICL（timbre+prosody）；cross_lingual 不依赖
            # 转录文本（timbre-only，与 MLX 侧语义一致）
            request["ref_audio"] = ref_audio
            request["ref_text"] = ref_text if mode == "zero_shot" else None
        elif self.variant == "custom_voice":
            request["speaker"] = self.speaker
            request["instruct"] = instruct or None
        else:  # voice_design
            if not instruct or not instruct.strip():
                raise InvalidAudioInput("VoiceDesign 后端需要 instruct_text 描述期望音色")
            request["instruct"] = instruct
        return request

    def _run_inference(self, request: dict[str, Any]) -> np.ndarray:
        """调用 ``qwen3_tts_inference`` 子进程并读取输出 WAV。"""
        with tempfile.TemporaryDirectory(prefix="qwen3_tts_trt_") as tmp:
            work = Path(tmp)
            input_json = work / "input.json"
            output_dir = work / "output"
            output_dir.mkdir()
            input_json.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")

            cmd = [
                str(self._inference_path),
                "--input_json", str(input_json),
                "--output_dir", str(output_dir),
            ]
            logger.info("[TTS] TensorRT Edge-LLM 推理 (%s): %s...", self.variant, text_preview(request))
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600, check=False
                )
            except FileNotFoundError as e:
                raise ModelLoadError(f"推理工具调用失败: {e}") from e
            except subprocess.TimeoutExpired as e:
                raise ModelLoadError("TensorRT 推理超时（600s）") from e
            if proc.returncode != 0:
                raise ModelLoadError(
                    f"TensorRT 推理失败（exit={proc.returncode}）: {proc.stderr.strip()[:300]}"
                )
            wavs = sorted(output_dir.glob("*.wav"))
            if not wavs:
                raise ModelLoadError(
                    "TensorRT 推理未产出 WAV（检查 Edge-LLM 版本与请求字段；"
                    f"工具输出: {proc.stdout.strip()[:200]}"
                )
            chunks = [sf.read(w, dtype="float32")[0] for w in wavs]
            audio = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            logger.info(
                "[TTS] TensorRT 推理完成，耗时 %.2fs，产出 %d 段", time.monotonic() - start, len(chunks)
            )
            return audio

    # ─── 状态 ────────────────────────────────────────────────
    def get_status(self) -> dict[str, object]:
        engine_ready = bool(
            self.model_dir and self.model_dir.exists() and any(self.model_dir.iterdir())
        )
        return {
            "model_loaded": self._ready,
            "model_id": self.model_id,
            "engine_dir": str(self.model_dir) if self.model_dir else None,
            "engine_dir_ready": engine_ready,
            "inference_bin_found": _find_inference_bin(self.inference_bin) is not None,
            "sample_rate": self.sample_rate,
        }


def text_preview(request: dict[str, Any], limit: int = 50) -> str:
    """日志用请求预览。"""
    return str(request.get("text", ""))[:limit]


class Qwen3TTSBaseTensorRTBackend(Qwen3TTSTensorRTBackend):
    """TensorRT 版 Base 变体 — 参考音频零样本克隆 / 跨语言。"""

    name = "qwen3-tts-base-tensorrt"
    variant = "base"
    default_model_id = QWEN3_TTS_MODEL_IDS["base"]
    supported_modes = ("zero_shot", "cross_lingual")


class Qwen3TTSCustomVoiceTensorRTBackend(Qwen3TTSTensorRTBackend):
    """TensorRT 版 CustomVoice 变体 — 预设音色 + 情绪/风格指令。"""

    name = "qwen3-tts-customvoice-tensorrt"
    variant = "custom_voice"
    default_model_id = QWEN3_TTS_MODEL_IDS["custom_voice"]
    supported_modes = ("instruct",)


class Qwen3TTSVoiceDesignTensorRTBackend(Qwen3TTSTensorRTBackend):
    """TensorRT 版 VoiceDesign 变体 — 自然语言描述造音色。"""

    name = "qwen3-tts-voicedesign-tensorrt"
    variant = "voice_design"
    default_model_id = QWEN3_TTS_MODEL_IDS["voice_design"]
    supported_modes = ("instruct",)


__all__ = [
    "Qwen3TTSBaseTensorRTBackend",
    "Qwen3TTSCustomVoiceTensorRTBackend",
    "Qwen3TTSVoiceDesignTensorRTBackend",
    "Qwen3TTSTensorRTBackend",
]
