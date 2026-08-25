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
    ASR_CHUNK_OVERLAP_SEC,
    ASR_CTX_SIZE,
    ASR_HTTP_TIMEOUT,
    ASR_MAX_SEGMENT_SEC,
    ASR_MODEL_DIR,
    ASR_MODEL_ID,
    ASR_N_GPU_LAYERS,
    ASR_READY_TIMEOUT,
    ASR_SAMPLE_RATE,
    ASR_SERVER_HOST,
    ASR_SERVER_PORT,
    LOGS_DIR,
    ensure_logs_dir,
)
from audioclone.core.capabilities import ModelBackend, SpeechRecognitionCapability
from audioclone.utils.audio import resample_audio

logger = logging.getLogger(__name__)


class Qwen3AsrLlamaBackend(ModelBackend, SpeechRecognitionCapability):
    """管理 llama-server 子进程并通过 HTTP API 进行语音识别。"""

    name = "qwen3-asr-llama"
    capabilities = (SpeechRecognitionCapability,)

    def __init__(
        self,
        model_dir: str | Path | None = None,
        model_id: str | None = None,
        host: str = ASR_SERVER_HOST,
        port: int = ASR_SERVER_PORT,
        max_segment_sec: float = ASR_MAX_SEGMENT_SEC,
        overlap_sec: float = ASR_CHUNK_OVERLAP_SEC,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else ASR_MODEL_DIR
        self.model_id = model_id or ASR_MODEL_ID
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.max_segment_sec = max_segment_sec  # 长音频分片阈值（秒），避免整段 base64 膨胀
        self.overlap_sec = overlap_sec          # 分片间重叠（秒），避免切词导致识别质量下降

        self._process: subprocess.Popen[str] | None = None
        self._model_path: Path | None = None
        self._mmproj_path: Path | None = None
        self._log_path: Path | None = None
        self._log_handle: io.TextIOWrapper | None = None

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
    def start(self, timeout: float = ASR_READY_TIMEOUT) -> bool:
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

        # 输出重定向到日志文件，避免 PIPE 缓冲填满阻塞子进程
        ensure_logs_dir()
        self._log_path = LOGS_DIR / "asr_llama_server.log"
        self._log_handle = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115

        logger.info("[ASR] 启动 llama-server: %s", " ".join(cmd))
        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=self._log_handle, text=True
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[ASR] 启动 llama-server 失败: %s", e)
            self._close_log()
            return False

        logger.info("[ASR] 等待服务就绪 (超时 %.0fs)...", timeout)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._process.poll() is not None:
                logger.error(
                    "[ASR] llama-server 意外退出，日志: %s",
                    self._read_log_tail(),
                )
                self._process = None
                return False
            if self.is_ready():
                logger.info("[ASR] llama-server 就绪，耗时 %.1fs", time.time() - start_time)
                return True
            time.sleep(1.0)

        logger.error("[ASR] 等待服务就绪超时 (%.0fs)", timeout)
        self.stop()
        return False

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _read_log_tail(self, limit: int = 1024) -> str:
        try:
            if self._log_path is None:
                return ""
            data = self._log_path.read_text(encoding="utf-8", errors="replace")
            return data[-limit:] if data else ""
        except OSError:
            return ""

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
        self._close_log()
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

        data = self._load_audio_array(audio_path)
        chunks = self._chunk_audio(data)

        logger.info("[ASR] 开始识别: %s（%d 段）", audio_path, len(chunks))
        start_time = time.time()

        texts: list[str] = []
        for i, chunk in enumerate(chunks):
            seg_text = self._transcribe_chunk(chunk)
            if seg_text:
                texts.append(seg_text)
            logger.info("[ASR] 段 %d/%d 完成", i + 1, len(chunks))

        text = " ".join(texts)
        logger.info("[ASR] 识别完成，耗时 %.2fs，结果: %s...", time.time() - start_time, text[:80])
        return text

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """对单个 16kHz 音频片段发起一次识别请求。"""
        audio_b64 = self._encode_wav_base64(audio)

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

        try:
            resp = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=ASR_HTTP_TIMEOUT,
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
        return self._parse_asr_output(raw_text)

    def _load_audio_array(self, audio_path: str) -> np.ndarray:
        """读取音频为 16kHz mono float32 数组（重采样）。"""
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return resample_audio(data, int(sr), ASR_SAMPLE_RATE)

    def _chunk_audio(self, data: np.ndarray) -> list[np.ndarray]:
        """按 ``max_segment_sec`` 分片，段间保留 ``overlap_sec`` 重叠并去重尾部。

        重叠避免切在字/词中间导致识别质量下降；尾部片段若几乎被前一段覆盖则并入。
        """
        max_len = max(1, int(self.max_segment_sec * ASR_SAMPLE_RATE))
        if len(data) <= max_len:
            return [data]

        overlap = max(0, int(self.overlap_sec * ASR_SAMPLE_RATE))
        step = max(1, max_len - overlap)
        chunks = [data[i : i + max_len] for i in range(0, len(data), step)]

        # 尾部去重：最后一段长度 <= 重叠量，说明其几乎被前一段覆盖 → 前一段扩展到数据末尾
        if len(chunks) >= 2 and len(chunks[-1]) <= overlap:
            start = (len(chunks) - 2) * step
            chunks[-2] = data[start:]
            chunks.pop()
        return chunks

    @staticmethod
    def _encode_wav_base64(data: np.ndarray) -> str:
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