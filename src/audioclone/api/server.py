"""FastAPI REST API 服务器 (命令行: ``uv run audioclone-api``)。

提供程序化声音克隆接口。要点：
- 引擎经依赖注入（``app.state`` + ``Depends``），无模块级可变单例。
- 合成端点为普通 ``def``，FastAPI 自动调度到线程池，避免阻塞事件循环。
- 上传做大小/时长/格式限流；统一领域异常映射。
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response

# 本地回环不受系统代理影响（在 httpx/gradio 等网络库导入前设置）
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

from audioclone.api.metrics import metrics
from audioclone.config import (
    ALLOWED_UPLOAD_EXT,
    LOG_JSON,
    MAX_TEXT_LENGTH,
    MAX_UPLOAD_BYTES,
    MODEL_ID,
)
from audioclone.core import TTSService, create_tts_backend
from audioclone.core.capabilities import TextToSpeechCapability
from audioclone.core.exceptions import InvalidAudioInput, ModelLoadError
from audioclone.utils.audio import assert_valid_ref_audio
from audioclone.utils.logging_utils import configure_logging
from audioclone.utils.validation import assert_tts_text, assert_valid_mode

logger = logging.getLogger(__name__)

VALID_MODES = ("zero_shot", "cross_lingual", "instruct")

MAX_UPLOAD_DISPLAY = f"{MAX_UPLOAD_BYTES / (1024 * 1024):.0f}MB"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动创建并加载 TTS 后端，退出统一释放。"""
    logger.info("正在加载 TTS 服务（后端由配置决定）...")
    opts: dict = {}
    model_dir = os.environ.get("COSYVOICE_MODEL_DIR", None)
    if model_dir:
        opts["model_dir"] = model_dir
    backend = create_tts_backend(**opts)
    try:
        backend.load()
    except Exception as e:  # noqa: BLE001
        backend.close()
        raise ModelLoadError(f"TTS 后端加载失败: {e}") from e
    if not isinstance(backend, TextToSpeechCapability):
        backend.close()
        raise ModelLoadError(f"TTS 后端 {backend.name} 不具备文本转语音能力")
    app.state.engine = TTSService(backend)
    app.state.backend = backend
    logger.info("TTS 服务就绪 (采样率 %s)", app.state.engine.sample_rate)
    try:
        yield
    finally:
        logger.info("关闭 TTS 后端...")
        backend.close()


app = FastAPI(
    title="AudioCloning API",
    description="音频模型体验工具箱 REST API (MLX)",
    version="0.2.1",
    lifespan=lifespan,
)


@app.middleware("http")
async def _request_log(request: Request, call_next):
    """请求日志：记录方法、路径、状态码与耗时；注入 trace_id 并采集指标。"""
    trace_id = uuid.uuid4().hex[:12]
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Request-ID"] = trace_id
    metrics.inc_request(request.url.path, response.status_code)
    logger.info(
        "HTTP %s %s -> %s (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed * 1000,
        extra={"trace_id": trace_id},
    )
    return response


def _get_engine(request: Request) -> TTSService:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise InvalidAudioInput("TTS 服务未就绪（模型未加载）")
    return engine


@app.exception_handler(InvalidAudioInput)
async def _invalid_input_handler(request: Request, exc: InvalidAudioInput):
    return await _json_error(400, str(exc))


@app.exception_handler(ModelLoadError)
async def _model_error_handler(request: Request, exc: ModelLoadError):
    return await _json_error(503, str(exc))


async def _json_error(status: int, detail: str):
    return JSONResponse(status_code=status, content=jsonable_encoder({"detail": detail}))


@app.get("/api/health")
async def health(request: Request) -> dict[str, Any]:
    """健康检查。"""
    engine = getattr(request.app.state, "engine", None)
    return {
        "status": "ok",
        "framework": "MLX",
        "model": MODEL_ID,
        "sample_rate": engine.sample_rate if engine else None,
        "model_loaded": engine is not None,
    }


@app.get("/api/metrics")
def metrics_endpoint() -> Response:
    """Prometheus 风格进程内指标（请求计数 / 合成耗时 / 运行时长）。"""
    return Response(content=metrics.render(), media_type="text/plain")


def _save_upload(upload: UploadFile) -> str:
    """将上传音频安全落盘为临时文件；流式限制大小、校验扩展名。"""
    filename = upload.filename or "upload.wav"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXT:
        raise InvalidAudioInput(
            f"不支持的音频格式: {suffix}，支持: {', '.join(ALLOWED_UPLOAD_EXT)}"
        )

    size = getattr(upload, "size", None)
    if size is not None and size > MAX_UPLOAD_BYTES:
        raise InvalidAudioInput(
            f"上传文件过大 ({size} 字节)，上限 {MAX_UPLOAD_DISPLAY}"
        )

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        written = 0
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise InvalidAudioInput(
                    f"上传文件过大，上限 {MAX_UPLOAD_DISPLAY}"
                )
            tmp.write(chunk)
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise
    tmp.close()
    return tmp.name


@app.post("/api/clone")
def clone_voice(
    text: str = Form(..., description="合成文本"),
    ref_audio: UploadFile = File(..., description="参考音频文件"),
    ref_text: str = Form("", description="参考音频对应文本"),
    speed: float = Form(1.0, description="语速 0.5~2.0"),
    mode: str = Form("zero_shot", description="推理模式: zero_shot/cross_lingual/instruct"),
    instruct_text: str = Form("", description="指令文本 (instruct 模式)"),
    engine: TTSService = Depends(_get_engine),
) -> FileResponse:
    """声音克隆接口（普通 def，由线程池执行，不阻塞事件循环）。"""
    assert_tts_text(text, MAX_TEXT_LENGTH)
    assert_valid_mode(mode, VALID_MODES)
    if mode == "instruct" and not instruct_text.strip():
        raise InvalidAudioInput("instruct 模式需提供 instruct_text")

    # 上传限流落盘
    tmp_path = _save_upload(ref_audio)

    try:
        # 时长 / 采样率校验（不合法抛 InvalidAudioInput → 400）
        assert_valid_ref_audio(tmp_path)

        start = time.time()
        output_path = engine.synthesize_to_file(
            mode=mode,
            text=text,
            ref_audio_path=tmp_path,
            ref_text=ref_text,
            instruct_text=instruct_text,
            speed=speed,
        )
        elapsed = time.time() - start
        metrics.record_synthesis(elapsed)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=Path(output_path).name,
            headers={"X-Synthesis-Time": f"{elapsed:.2f}s"},
        )
    except InvalidAudioInput:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("合成失败: %s", e, exc_info=True)
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"合成失败: {e}") from e
    finally:
        os.unlink(tmp_path)


def main() -> None:
    configure_logging(json=LOG_JSON)
    parser = argparse.ArgumentParser(description="AudioCloning API Server", prog="audioclone-api")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址（默认仅本机）")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # log_config=None：不覆盖上方 configure_logging 已配置的处理器
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()