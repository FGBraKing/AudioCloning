"""
ASR 推理引擎 — 基于 llama.cpp (llama-server) + Qwen3-ASR-0.6B-GGUF
通过 OpenAI 兼容 HTTP API 实现语音识别。
"""

import base64
import io
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import soundfile as sf

from ..config import (
    ASR_CTX_SIZE,
    ASR_MODEL_DIR,
    ASR_N_GPU_LAYERS,
    ASR_SERVER_HOST,
    ASR_SERVER_PORT,
)

logger = logging.getLogger(__name__)

# ASR 模型要求的采样率
ASR_SAMPLE_RATE = 16000


class ASREngine:
    """
    ASR 引擎：管理 llama-server 子进程并通过 HTTP API 进行语音识别。

    Usage:
        engine = ASREngine()
        engine.start_server()
        text = engine.transcribe("audio.wav")
        engine.stop_server()

    或使用上下文管理器:
        with ASREngine() as engine:
            text = engine.transcribe("audio.wav")
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        初始化 ASR 引擎。

        Args:
            model_dir: 模型目录路径，默认使用 config.ASR_MODEL_DIR
        """
        self.model_dir = Path(model_dir) if model_dir else ASR_MODEL_DIR
        self.host = ASR_SERVER_HOST
        self.port = ASR_SERVER_PORT
        self.base_url = f"http://{self.host}:{self.port}"

        self._process: Optional[subprocess.Popen] = None
        self._model_path: Optional[Path] = None
        self._mmproj_path: Optional[Path] = None

        # 扫描模型文件
        self._locate_model_files()

    def _locate_model_files(self):
        """在模型目录中定位主模型和 mmproj 文件（优先 Q8_0 量化）"""
        if not self.model_dir.exists():
            logger.warning(f"[ASR] 模型目录不存在: {self.model_dir}")
            return

        gguf_files = list(self.model_dir.glob("*.gguf"))
        if not gguf_files:
            logger.warning(f"[ASR] 模型目录中未找到 .gguf 文件: {self.model_dir}")
            return

        # 分离 mmproj 和主模型
        mmproj_files = sorted(
            [f for f in gguf_files if "mmproj" in f.name.lower()],
            key=lambda f: f.stat().st_size,
        )
        model_files = sorted(
            [f for f in gguf_files if "mmproj" not in f.name.lower()],
            key=lambda f: f.stat().st_size,
        )

        # 优先选择 Q8_0（体积小、Metal 推理快），否则选最小的
        def prefer_q8(files):
            q8 = [f for f in files if "q8" in f.name.lower()]
            return q8[0] if q8 else files[0]

        if mmproj_files:
            self._mmproj_path = prefer_q8(mmproj_files)
            logger.info(f"[ASR] mmproj 文件: {self._mmproj_path.name}")
        else:
            logger.warning("[ASR] 未找到 mmproj 文件（音频编码器投影）")

        if model_files:
            self._model_path = prefer_q8(model_files)
            logger.info(f"[ASR] 主模型文件: {self._model_path.name}")
        else:
            logger.warning("[ASR] 未找到主模型 GGUF 文件")

    @property
    def is_model_available(self) -> bool:
        """检查模型文件是否就绪"""
        return self._model_path is not None and self._mmproj_path is not None

    def start_server(self, timeout: float = 60.0) -> bool:
        """
        启动 llama-server 子进程。

        Args:
            timeout: 等待服务就绪的超时时间（秒）

        Returns:
            是否成功启动
        """
        if not self.is_model_available:
            logger.error("[ASR] 模型文件不完整，无法启动服务")
            return False

        if self._process is not None and self._process.poll() is None:
            logger.info("[ASR] llama-server 已在运行")
            return True

        # 查找 llama-server 二进制
        server_bin = shutil.which("llama-server")
        if not server_bin:
            logger.error("[ASR] 未找到 llama-server，请确认 llama.cpp 已安装")
            return False

        # 构建启动命令
        cmd = [
            server_bin,
            "-m", str(self._model_path),
            "--mmproj", str(self._mmproj_path),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(ASR_CTX_SIZE),
            "-ngl", str(ASR_N_GPU_LAYERS),
        ]

        logger.info(f"[ASR] 启动 llama-server: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            logger.error(f"[ASR] 启动 llama-server 失败: {e}")
            return False

        # 等待服务就绪
        logger.info(f"[ASR] 等待服务就绪 (超时 {timeout}s)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            # 检查进程是否意外退出
            if self._process.poll() is not None:
                stderr_output = self._process.stderr.read() if self._process.stderr else ""
                logger.error(f"[ASR] llama-server 意外退出: {stderr_output[:500]}")
                self._process = None
                return False

            if self.is_ready():
                elapsed = time.time() - start_time
                logger.info(f"[ASR] llama-server 就绪，耗时 {elapsed:.1f}s")
                return True

            time.sleep(1.0)

        logger.error(f"[ASR] 等待服务就绪超时 ({timeout}s)")
        self.stop_server()
        return False

    def stop_server(self):
        """终止 llama-server 子进程"""
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

    def is_ready(self) -> bool:
        """检查 llama-server 是否就绪"""
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=3.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def _prepare_audio(self, audio_path: str) -> str:
        """
        读取音频文件并转为 16kHz mono WAV 的 base64 编码。

        Args:
            audio_path: 音频文件路径

        Returns:
            base64 编码的 WAV 数据
        """
        # 读取音频
        data, sr = sf.read(audio_path, dtype="float32")

        # 多声道转单声道
        if data.ndim > 1:
            data = data.mean(axis=1)

        # 重采样到 16kHz
        if sr != ASR_SAMPLE_RATE:
            from scipy.signal import resample
            num_samples = int(len(data) * ASR_SAMPLE_RATE / sr)
            data = resample(data, num_samples).astype(np.float32)

        # 编码为 WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, data, ASR_SAMPLE_RATE, format="WAV")
        wav_bytes = buffer.getvalue()

        # Base64 编码
        return base64.b64encode(wav_bytes).decode("utf-8")

    def transcribe(self, audio_path: str) -> str:
        """
        对音频文件进行语音识别。

        Args:
            audio_path: 音频文件路径

        Returns:
            识别出的文本

        Raises:
            RuntimeError: 服务未就绪或识别失败
        """
        if not self.is_ready():
            raise RuntimeError("ASR 服务未就绪，请确认 llama-server 已启动")

        # 准备音频数据
        audio_b64 = self._prepare_audio(audio_path)

        # 构建请求
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                    {
                        "type": "text",
                        "text": "请转录这段音频的内容。",
                    },
                ],
            }
        ]

        payload = {
            "model": "qwen3-asr",
            "messages": messages,
            "max_tokens": ASR_CTX_SIZE // 2,
            "temperature": 0.0,
        }

        logger.info(f"[ASR] 开始识别: {audio_path}")
        start_time = time.time()

        try:
            resp = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise RuntimeError("ASR 识别超时")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"ASR 请求失败: {e.response.status_code} - {e.response.text[:200]}")

        # 解析响应
        result = resp.json()
        raw_text = result["choices"][0]["message"]["content"].strip()

        # 清理 Qwen3-ASR 输出格式: "language Chinese<asr_text>实际文本"
        text = self._parse_asr_output(raw_text)

        elapsed = time.time() - start_time
        logger.info(f"[ASR] 识别完成，耗时 {elapsed:.2f}s，结果: {text[:80]}...")

        return text

    @staticmethod
    def _parse_asr_output(raw: str) -> str:
        """
        解析 Qwen3-ASR 模型输出格式。
        典型格式: 'language Chinese<asr_text>实际识别文本'
        """
        import re
        # 尝试提取 <asr_text> 后的内容
        match = re.search(r"<asr_text>(.*)", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 如果没有标签，尝试去除 language 前缀
        match = re.sub(r"^language\s+\w+\s*", "", raw)
        return match.strip() if match.strip() else raw

    def get_status(self) -> dict:
        """获取 ASR 引擎状态信息"""
        return {
            "model_available": self.is_model_available,
            "model_path": str(self._model_path) if self._model_path else None,
            "mmproj_path": str(self._mmproj_path) if self._mmproj_path else None,
            "server_running": self._process is not None and self._process.poll() is None,
            "server_ready": self.is_ready(),
            "server_url": self.base_url,
        }

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()
        return False
