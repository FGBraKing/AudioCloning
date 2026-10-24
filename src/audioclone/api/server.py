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
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

# 本地回环不受系统代理影响（在 httpx 等网络库导入前设置）
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
# 同 webui：SOCKS 代理会让 httpx 客户端创建阶段直接失败
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

from audioclone import __version__
from audioclone.api.metrics import metrics
from audioclone.config import (
    ALLOWED_UPLOAD_EXT,
    ASR_AUDIO_MAX_SEC,
    LOG_JSON,
    MAX_INLINE_UPLOAD_BYTES,
    MAX_TEXT_LENGTH,
    MAX_UPLOAD_BYTES,
    OUTPUT_DIR,
)
from audioclone.core import (
    ModelBackend,
    TTSService,
    create_service,
    create_tts_backend,
    model_slots,
)
from audioclone.core.capabilities import TextToSpeechCapability
from audioclone.core.exceptions import (
    BackendUnavailable,
    InvalidAudioInput,
    ModelLoadError,
    ModelSlotError,
)
from audioclone.core.model_manager import download_model, model_status
from audioclone.core.runtime import detect_runtime
from audioclone.utils.audio import assert_audio_duration, assert_valid_ref_audio, get_audio_duration
from audioclone.utils.logging_utils import configure_logging, set_trace_id
from audioclone.utils.validation import assert_tts_text, assert_valid_mode

logger = logging.getLogger(__name__)

# plain=纯文本直读（sherpa-onnx Kokoro 等，无需参考音频）
VALID_MODES = ("zero_shot", "cross_lingual", "instruct", "plain")

MAX_UPLOAD_DISPLAY = f"{MAX_UPLOAD_BYTES / (1024 * 1024):.0f}MB"
MAX_OUTPUTS_LIMIT = 200  # /api/outputs 单次返回上限（防滥用）


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动创建并尝试加载 TTS 后端，退出统一释放。

    模型未下载/加载失败**不阻止服务启动**：engine 置空，接口按既有语义返回 503，
    并在日志中给出主动下载指引（``POST /api/models/download``）。
    """
    logger.info("正在加载 TTS 服务（后端由配置决定）...")
    # 模型路径：默认 HF 系统缓存（AUDIOCLONE_TTS_MODEL_DIR 可覆盖，经 Settings 流入后端）
    backend = create_tts_backend()
    try:
        backend.load()
    except Exception as e:  # noqa: BLE001
        logger.error(
            "TTS 后端 %s 加载失败，服务将以未就绪状态启动（推理接口返回 503）: %s\n"
            "首次运行请主动下载模型: uv run audioclone-download-tts 或 POST /api/models/download",
            backend.name,
            e,
        )
        backend.close()
        app.state.engine = None
        app.state.backend = backend
        yield
        return
    if not isinstance(backend, TextToSpeechCapability):
        backend.close()
        raise ModelLoadError(f"TTS 后端 {backend.name} 不具备文本转语音能力")
    app.state.engine = TTSService(backend)
    app.state.backend = backend
    logger.info("TTS 服务就绪 (采样率 %s)", app.state.engine.sample_rate)

    # 多任务服务：T2A / A2A / 分类 / VAD 各绑专属后端（构造轻量，首次调用惰性加载）
    task_services: dict[str, Any] = {}
    task_backends: list[ModelBackend] = []
    for kind in ("t2a", "a2a", "classification", "vad"):
        try:
            svc = create_service(kind)
            task_services[kind] = svc
            task_backends.append(svc.backend)
            logger.info("多任务后端 [%s]: %s", kind, svc.backend.name)
        except Exception as e:  # noqa: BLE001 - 单个任务创建失败不影响其余功能
            logger.error("任务 %s 服务创建失败: %s", kind, e)
    app.state.task_services = task_services

    try:
        yield
    finally:
        for b in task_backends:
            b.close()
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
    backend = getattr(request.app.state, "backend", None)
    return {
        "status": "ok",
        "framework": detect_runtime().framework,
        "backend": getattr(backend, "name", None),
        "model": getattr(backend, "model_id", None),
        "supported_modes": list(getattr(backend, "supported_modes", []) or []),
        "sample_rate": engine.sample_rate if engine else None,
        "model_loaded": engine is not None,
        # 槽位占用情况：保证运行时每种任务只有一个已加载模型
        "model_slots": model_slots.snapshot(),
        # 运行环境与推理框架（后端自动选择的依据）
        "runtime": detect_runtime().as_dict(),
    }


# ─── 模型管理：状态查询 + 主动下载（首次运行不自动下载） ──────
_download_state: dict[str, Any] = {
    "running": False, "task": None, "backend": None,
    "result": None, "error": None, "started_at": None, "finished_at": None,
}
_download_lock = threading.Lock()


class ModelDownloadRequest(BaseModel):
    """主动下载请求体。"""

    task: Literal["tts", "asr", "t2a", "a2a", "classification", "vad"]  # 任务类型
    backend: str | None = None                     # 后端名（缺省用当前配置默认）
    local_dir: str | None = None                   # 自定义落盘目录（缺省进 HF 系统缓存）
    source: Literal["auto", "hf", "modelscope"] = "auto"  # auto=HF 失败自动回退 ModelScope


def _download_worker(req: ModelDownloadRequest) -> None:
    """后台线程：执行下载并维护状态（失败不抛出，记录到 state）。"""
    try:
        result = download_model(req.task, req.backend, req.local_dir, req.source)
        _download_state["result"] = result
        logger.info("模型下载完成: %s", result)
    except Exception as e:  # noqa: BLE001 - 后台线程内自捕获，经 state 暴露
        _download_state["error"] = str(e)
        logger.error("模型下载失败: %s", e)
    finally:
        _download_state["running"] = False
        _download_state["finished_at"] = datetime.now(tz=UTC).isoformat()
        _MODELS_CACHE["data"] = None  # 模型就位后立即刷新状态缓存


#: /api/models 状态 TTL 缓存：health 高频轮询时避免反复扫描模型缓存目录
_MODELS_CACHE: dict[str, Any] = {"at": 0.0, "data": None}
_MODELS_CACHE_TTL = 30.0  # 秒


def _models_overview_cached() -> dict[str, Any]:
    now = time.monotonic()
    if _MODELS_CACHE["data"] is None or now - _MODELS_CACHE["at"] > _MODELS_CACHE_TTL:
        _MODELS_CACHE["data"] = {
            task: model_status(task)
            for task in ("tts", "asr", "t2a", "a2a", "classification", "vad")
        }
        _MODELS_CACHE["at"] = now
    return _MODELS_CACHE["data"]


@app.get("/api/models")
async def models_overview() -> dict[str, Any]:
    """各任务模型的就绪状态（离线查询 + 30s TTL 缓存）。

    ``cached=false`` 表示模型未下载，需经 ``POST /api/models/download`` 主动下载。
    """
    return _models_overview_cached()


@app.post("/api/models/download")
async def start_model_download(req: ModelDownloadRequest) -> JSONResponse:
    """主动下载模型权重（后台执行，立即返回 202；进度经 ``GET /api/models/download`` 查询）。

    ``source``：``auto``（默认）= 先 HuggingFace（支持 ``AUDIOCLONE_HF_ENDPOINT`` 自定义
    镜像）失败自动回退 ModelScope；``hf`` / ``modelscope`` 强制指定。
    ``local_dir`` 可指定自定义目录。同一时刻仅允许一个下载任务（并发请求返回 409）。
    """
    try:
        model_status(req.task, req.backend)  # 校验任务/后端名（fail-fast）
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    with _download_lock:
        if _download_state["running"]:
            raise HTTPException(
                status_code=409,
                detail=f"已有下载任务进行中: task={_download_state['task']}，"
                f"请先经 GET /api/models/download 查询进度",
            )
        _download_state.update(
            running=True, task=req.task, backend=req.backend,
            result=None, error=None,
            started_at=datetime.now(tz=UTC).isoformat(), finished_at=None,
        )
    threading.Thread(target=_download_worker, args=(req,), daemon=True).start()
    return JSONResponse(
        status_code=202,
        content=jsonable_encoder({
            "status": "downloading", "task": req.task, "backend": req.backend,
            "message": "下载已在后台启动，经 GET /api/models/download 查询进度",
        }),
    )


@app.get("/api/models/download")
async def model_download_status() -> dict[str, Any]:
    """查询当前/最近一次模型下载任务的状态。"""
    return dict(_download_state)


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
    ref_audio: UploadFile | None = File(None, description="参考音频文件（plain 模式可省略）"),
    ref_text: str = Form("", description="参考音频对应文本"),
    speed: float = Form(1.0, description="语速 0.5~2.0"),
    mode: str = Form("zero_shot", description="推理模式: zero_shot/cross_lingual/instruct/plain"),
    instruct_text: str = Form("", description="指令文本 (instruct 模式)"),
    engine: TTSService = Depends(_get_engine),
) -> FileResponse:
    """声音克隆/合成接口（普通 def，由线程池执行，不阻塞事件循环）。

    ``plain`` 模式为纯文本直读（无需参考音频），其余模式均需 ``ref_audio``。
    """
    assert_tts_text(text, MAX_TEXT_LENGTH)
    assert_valid_mode(mode, VALID_MODES)
    if mode == "instruct" and not instruct_text.strip():
        raise InvalidAudioInput("instruct 模式需提供 instruct_text")
    if mode != "plain" and (ref_audio is None or not ref_audio.filename):
        raise InvalidAudioInput(f"{mode} 模式需提供参考音频 ref_audio")

    # 上传限流落盘（plain 模式无参考音频，直接合成）
    tmp_path = _save_upload(ref_audio) if ref_audio is not None and ref_audio.filename else None

    try:
        if tmp_path:
            # 时长 / 采样率校验（不合法抛 InvalidAudioInput → 400）
            assert_valid_ref_audio(tmp_path)

        start = time.time()
        output_path = engine.synthesize_to_file(
            mode=mode,
            text=text,
            ref_audio_path=tmp_path or "",
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
        if tmp_path:
            os.unlink(tmp_path)


# ─── 多任务端点：T2A / A2A / 分类 / VAD ──────────────────────
def _get_task_service(request: Request, kind: str) -> Any:
    svc = getattr(request.app.state, "task_services", {}).get(kind)
    if svc is None:
        raise BackendUnavailable(f"任务 {kind} 服务不可用")
    return svc


def _save_wav_result(audio: np.ndarray, sample_rate: int, prefix: str) -> Path:
    """把生成音频落盘 outputs/ 并返回路径（复用输出目录管理）。"""
    from audioclone.config import OUTPUT_DIR, OUTPUT_KEEP_MAX
    from audioclone.utils.audio import prune_outputs, save_audio

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.wav"
    save_audio(audio, str(out), sample_rate)
    prune_outputs(OUTPUT_DIR, OUTPUT_KEEP_MAX)  # 多任务输出同样受保留上限约束
    return out


def _load_upload_audio(upload: UploadFile) -> str:
    """上传音频 → 临时落盘 + 格式/时长校验，返回路径（调用方负责删除）。"""
    tmp_path = _save_upload(upload)
    suffix = Path(tmp_path).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXT:
        os.unlink(tmp_path)
        raise InvalidAudioInput(
            f"不支持的音频格式: {suffix}，支持: {', '.join(ALLOWED_UPLOAD_EXT)}"
        )
    return tmp_path


@app.post("/api/t2a")
def text_to_audio(
    text: str = Form(..., description="文本提示（描述目标音乐/音效）"),
    duration_sec: float | None = Form(None, description="期望时长（秒），缺省由模型决定"),
    lyrics: str = Form("", description="歌词（可选，音乐生成用）"),
    request: Request = None,  # type: ignore[assignment]
) -> FileResponse:
    """文本转音频接口（音乐生成；模型在首次调用时惰性加载）。"""
    svc = _get_task_service(request, "t2a")
    if not text.strip():
        raise InvalidAudioInput("文本提示为空")

    start = time.time()
    try:
        audio = svc.generate(text, duration_sec=duration_sec, lyrics=lyrics or None)
    except Exception as e:  # noqa: BLE001
        logger.error("T2A 生成失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成失败: {e}") from e
    elapsed = time.time() - start
    metrics.record_synthesis(elapsed)

    out = _save_wav_result(audio, svc.backend.sample_rate, "t2a")
    return FileResponse(
        out, media_type="audio/wav", filename=out.name,
        headers={"X-Generation-Time": f"{elapsed:.2f}s"},
    )


@app.post("/api/transform")
def transform_audio(
    audio: UploadFile = File(..., description="输入音频文件"),
    instruct: str = Form("", description="转换指令（可选，当前增强后端忽略）"),
    request: Request = None,  # type: ignore[assignment]
) -> FileResponse:
    """音频到音频接口（降噪增强；模型在首次调用时惰性加载）。"""
    svc = _get_task_service(request, "a2a")
    tmp_path = _load_upload_audio(audio)
    try:
        assert_audio_duration(tmp_path, ASR_AUDIO_MAX_SEC)
        start = time.time()
        try:
            audio_data = svc.transform(tmp_path, instruct=instruct or "")
        except Exception as e:  # noqa: BLE001
            logger.error("A2A 转换失败: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"转换失败: {e}") from e
        elapsed = time.time() - start
        metrics.record_synthesis(elapsed)
        out = _save_wav_result(audio_data, svc.backend.sample_rate, "a2a")
        return FileResponse(
            out, media_type="audio/wav", filename=out.name,
            headers={"X-Transform-Time": f"{elapsed:.2f}s"},
        )
    finally:
        os.unlink(tmp_path)


@app.post("/api/classify")
def classify_audio(
    audio: UploadFile = File(..., description="输入音频文件"),
    top_k: int = Form(5, description="返回置信度最高的前 k 个类别"),
    request: Request = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """音频分类接口（情感 + 事件；模型在首次调用时惰性加载）。"""
    svc = _get_task_service(request, "classification")
    tmp_path = _load_upload_audio(audio)
    try:
        assert_audio_duration(tmp_path, ASR_AUDIO_MAX_SEC)
        start = time.time()
        try:
            labels = svc.classify(tmp_path, top_k=max(1, top_k))
        except Exception as e:  # noqa: BLE001
            logger.error("分类失败: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"分类失败: {e}") from e
        elapsed = time.time() - start
        return {
            "results": [{"label": name, "score": round(score, 4)} for name, score in labels],
            "elapsed_sec": round(elapsed, 2),
        }
    finally:
        os.unlink(tmp_path)


@app.post("/api/vad")
def detect_vad(
    audio: UploadFile = File(..., description="输入音频文件"),
    request: Request = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    """语音活动检测接口（返回语音片段起止时刻；模型在首次调用时惰性加载）。"""
    svc = _get_task_service(request, "vad")
    tmp_path = _load_upload_audio(audio)
    try:
        assert_audio_duration(tmp_path, ASR_AUDIO_MAX_SEC)
        start = time.time()
        try:
            segments = svc.vad(tmp_path)
        except Exception as e:  # noqa: BLE001
            logger.error("VAD 检测失败: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"检测失败: {e}") from e
        elapsed = time.time() - start
        return {
            "segments": [
                {"start": round(s, 3), "end": round(e, 3), "duration": round(e - s, 3)}
                for s, e in segments
            ],
            "speech_sec": round(sum(e - s for s, e in segments), 2),
            "elapsed_sec": round(elapsed, 2),
        }
    finally:
        if tmp_path:
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