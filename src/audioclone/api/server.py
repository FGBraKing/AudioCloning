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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response

# 本地回环不受系统代理影响（在 httpx 等网络库导入前设置）
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

from audioclone import __version__
from audioclone.api.metrics import metrics
from audioclone.config import (
    ALLOWED_UPLOAD_EXT,
    LOG_JSON,
    MAX_INLINE_UPLOAD_BYTES,
    MAX_TEXT_LENGTH,
    MAX_UPLOAD_BYTES,
    MODEL_ID,
    OUTPUT_DIR,
)
from audioclone.core import TTSService, create_tts_backend, model_slots
from audioclone.core.capabilities import TextToSpeechCapability
from audioclone.core.exceptions import (
    BackendUnavailable,
    InvalidAudioInput,
    ModelLoadError,
    ModelSlotError,
)
from audioclone.utils.audio import assert_valid_ref_audio, get_audio_duration
from audioclone.utils.logging_utils import configure_logging, set_trace_id
from audioclone.utils.validation import assert_tts_text, assert_valid_mode

logger = logging.getLogger(__name__)

VALID_MODES = ("zero_shot", "cross_lingual", "instruct")

MAX_UPLOAD_DISPLAY = f"{MAX_UPLOAD_BYTES / (1024 * 1024):.0f}MB"
MAX_OUTPUTS_LIMIT = 200  # /api/outputs 单次返回上限（防滥用）


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
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def _request_log(request: Request, call_next):
    """请求日志：记录方法、路径、状态码与耗时；注入 trace_id 并采集指标。

    trace_id 经 contextvars 透传至线程池中的服务层日志（请求级可追踪）。
    """
    trace_id = uuid.uuid4().hex[:12]
    set_trace_id(trace_id)
    start = time.time()
    try:
        response = await call_next(request)
    finally:
        set_trace_id(None)
    elapsed = time.time() - start
    response.headers["X-Request-ID"] = trace_id
    if request.url.path != "/api/metrics":  # 指标端点自身不计入请求计数
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
        raise BackendUnavailable("TTS 服务未就绪（模型未加载）")
    return engine


@app.exception_handler(InvalidAudioInput)
async def _invalid_input_handler(request: Request, exc: InvalidAudioInput):
    return await _json_error(400, str(exc))


@app.exception_handler(ModelLoadError)
async def _model_error_handler(request: Request, exc: ModelLoadError):
    return await _json_error(503, str(exc))


@app.exception_handler(BackendUnavailable)
async def _backend_unavailable_handler(request: Request, exc: BackendUnavailable):
    return await _json_error(503, str(exc))


@app.exception_handler(ModelSlotError)
async def _model_slot_handler(request: Request, exc: ModelSlotError):
    """槽位冲突（试图加载第二个同类模型）→ 409。"""
    return await _json_error(409, str(exc))


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
        # 槽位占用情况：保证运行时每种任务只有一个已加载模型
        "model_slots": model_slots.snapshot(),
    }


@app.get("/api/metrics")
def metrics_endpoint() -> Response:
    """Prometheus 风格进程内指标（请求计数 / 合成耗时 / 运行时长）。"""
    return Response(content=metrics.render(), media_type="text/plain")


@app.get("/api/outputs")
def list_outputs(limit: int = 50, include_duration: bool = True) -> dict[str, Any]:
    """列出历史输出音频（含 ``segments/`` 子目录），按修改时间倒序。

    Args:
        limit: 最多返回条数（上限 200，防滥用）。
        include_duration: 是否计算每文件时长（读头，大目录可置 False 加速）。
    """
    limit = max(0, min(limit, MAX_OUTPUTS_LIMIT))
    if not OUTPUT_DIR.exists():
        return {"count": 0, "files": []}

    files: list[dict[str, Any]] = []
    for p in sorted(OUTPUT_DIR.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(files) >= limit:
            break
        if not p.is_file() or p.suffix.lower() not in ALLOWED_UPLOAD_EXT:
            continue
        try:
            stat = p.stat()
            entry: dict[str, Any] = {
                "name": p.name,
                "path": str(p.relative_to(OUTPUT_DIR)),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            }
            if include_duration:
                entry["duration_sec"] = round(get_audio_duration(str(p)), 2)
            files.append(entry)
        except OSError:
            continue
    return {"count": len(files), "files": files}


def _save_upload(upload: UploadFile) -> str:
    """将上传音频安全落盘为临时文件；限制大小、校验扩展名。

    内存安全：不信任客户端声明的 ``size``，按 1MB 分块读取；内容 ≤ ``AUDIOCLONE_MAX_INLINE_UPLOAD_BYTES``
    时在内存暂存后单次落盘（小文件快路径），超过阈值即转流式写盘（内存始终有界）。
    """
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
    buffer = bytearray()  # 仅在小文件阈值内暂存
    try:
        written = 0
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise InvalidAudioInput(f"上传文件过大，上限 {MAX_UPLOAD_DISPLAY}")
            if written <= MAX_INLINE_UPLOAD_BYTES:
                buffer += chunk  # 小文件：内存暂存
            else:
                if buffer:  # 超过阈值：先落盘内存部分，再转流式
                    tmp.write(bytes(buffer))
                    buffer.clear()
                tmp.write(chunk)
        if buffer:
            tmp.write(bytes(buffer))
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