"""Qwen3-ASR (llama.cpp) 后端 — 基于 llama-server + Qwen3-ASR-0.6B-GGUF。

通过 OpenAI 兼容 HTTP API 实现语音识别；负责子进程生命周期管理。
"""

from __future__ import annotations

import base64
import io
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from audioclone.config import (
    ASR_CTX_SIZE,
    ASR_MODEL_DIR,
    ASR_N_GPU_LAYERS,
    ASR_SAMPLE_RATE,
    ASR_SERVER_HOST,
    ASR_SERVER_PORT,
)
from audioclone.core.capabilities import ModelBackend, SpeechRecognitionCapability

logger = logging.getLogger(__name__)


class Qwen3AsrLlamaBackend(ModelBackend, SpeechRecognitionCapability):
    """管理 llama-server 子进程并通过 HTTP API 进行语音识别。"""

    name = "qwen3-asr-llama"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        host: str = ASR_SERVER_HOST,
        port: int = ASR_SERVER_PORT,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else ASR_MODEL_DIR
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"

        self._process: subprocess.Popen[str] | None = None
        self._model_path: Path | None = None
        self._mmproj_path: Path | None = None

        # 扫描模型文件
        self._locate_model_files()

    # ─── 模型文件定位 ─────────────────────────────────────────
    def _locate_model_files(self) -> None:
        if not self.model_dir.exists():
            logger.warning("[ASR] 模型目录不存在: %s", self.model_dir)
            return

        gguf_files = list(self.model_dir.glob("*.gguf"))
        if not gguf_files:
            logger.warning("[ASR] 模型目录中未找到 .gguf 文件: %s", self.model_dir)
            return

        def prefer_q8(files: list[Path]) -> Path:
            q8 = [f for f in files if "q8" in f.name.lower()]
            return q8[0] if q8 else files[0]

        mmproj_files = sorted(
            (f for f in gguf_files if "mmproj" in f.name.lower()),
            key=lambda f: f.stat().st_size,
        )
        model_files = sorted(
            (f for f in gguf_files if "mmproj" not in f.name.lower()),
            key=lambda f: f.stat().st_size,
        )

        if mmproj_files:
            self._mmproj_path = prefer_q8(list(mmproj_files))
            logger.info("[ASR] mmproj 文件: %s", self._mmproj_path.name)
        if model_files:
            self._model_path = prefer_q8(list(model_files))
            logger.info("[ASR] 主模型文件: %s", self._model_path.name)

    @property
    def is_model_available(self) -> bool:
        return self._model_path is not None and self._mmproj_path is not None

    # ─── 生命周期 ────────────────────────────────────────────
    def start(self, timeout: float = 60.0) -> bool:
        if not self.is_model_available:
            logger.error("[ASR] 模型文件不完整，无法启动服务")
            return False

        if self._process is not None and self._process.poll() is None:
            logger.info("[ASR] llama-server 已在运行")
            return True

        server_bin = shutil.which("llama-server")
        if not server_bin:
            logger.error("[ASR] 未找到 llama-server，请确认 llama.cpp 已安装")
            return False

        cmd = [
            server_bin,
            "-m", str(self._model_path),
            "--mmproj", str(self._mmproj_path),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(ASR_CTX_SIZE),
            "-ngl", str(ASR_N_GPU_LAYERS),
        ]

        logger.info("[ASR] 启动 llama-server: %s", " ".join(cmd))
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[ASR] 启动 llama-server 失败: %s", e)
            return False

        logger.info("[ASR] 等待服务就绪 (超时 %.0fs)...", timeout)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._process.poll() is not None:
                stderr_output = self._process.stderr.read() if self._process.stderr else ""
                logger.error("[ASR] llama-server 意外退出: %s", stderr_output[:500])
                self._process = None
                return False
            if self.is_ready():
                logger.info("[ASR] llama-server 就绪，耗时 %.1fs", time.time() - start_time)
                return True
            time.sleep(1.0)

        logger.error("[ASR] 等待服务就绪超时 (%.0fs)", timeout)
        self.stop()
        return False

    def stop(self) -> None:
        if self._process is not None:
            logger.info("[ASR] 正在停止 llama-server...")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
            logger.info("[ASR] llama-server 已停止")

    def close(self) -> None:
        self.stop()

    def is_ready(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=3.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # ─── 识别 ────────────────────────────────────────────────
    def transcribe(self, audio_path: str) -> str:
        if not self.is_ready():
            raise RuntimeError("ASR 服务未就绪，请确认 llama-server 已启动")

        audio_b64 = self._prepare_audio(audio_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                    {"type": "text", "text": "请转录这段音频的内容。"},
                ],
            }
        ]
        payload = {
            "model": "qwen3-asr",
            "messages": messages,
            "max_tokens": ASR_CTX_SIZE // 2,
            "temperature": 0.0,
        }

        logger.info("[ASR] 开始识别: %s", audio_path)
        start_time = time.time()

        try:
            resp = httpx.post(
                f"{self.base_url}/v1/chat/completions", json=payload, timeout=120.0
            )
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise RuntimeError("ASR 识别超时") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"ASR 请求失败: {e.response.status_code} - {e.response.text[:200]}"
            ) from e

        result = resp.json()
        raw_text = result["choices"][0]["message"]["content"].strip()
        text = self._parse_asr_output(raw_text)

        logger.info("[ASR] 识别完成，耗时 %.2fs，结果: %s...", time.time() - start_time, text[:80])
        return text

    def _prepare_audio(self, audio_path: str) -> str:
        """读取音频并转为 16kHz mono WAV 的 base64 编码。"""
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != ASR_SAMPLE_RATE:
            from scipy.signal import resample

            num_samples = int(len(data) * ASR_SAMPLE_RATE / sr)
            data = resample(data, num_samples).astype(np.float32)

        buffer = io.BytesIO()
        sf.write(buffer, data, ASR_SAMPLE_RATE, format="WAV")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _parse_asr_output(raw: str) -> str:
        match = re.search(r"<asr_text>(.*)", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        cleaned = re.sub(r"^language\s+\w+\s*", "", raw)
        return cleaned.strip() or raw

    # ─── 状态 ────────────────────────────────────────────────
    def get_status(self) -> dict[str, object]:
        running = self._process is not None and self._process.poll() is None
        return {
            "model_available": self.is_model_available,
            "model_path": str(self._model_path) if self._model_path else None,
            "mmproj_path": str(self._mmproj_path) if self._mmproj_path else None,
            "server_running": running,
            "server_ready": self.is_ready(),
            "server_url": self.base_url,
        }