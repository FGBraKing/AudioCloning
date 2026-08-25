"""FastAPI REST API 服务器 (命令行: ``uv run audioclone-api``)。

提供程序化声音克隆接口。
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from audioclone.config import MAX_TEXT_LENGTH, MODEL_ID
from audioclone.core import TTSService, create_tts_backend

logger = logging.getLogger(__name__)

engine: TTSService | None = None

VALID_MODES = ("zero_shot", "cross_lingual", "instruct")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用生命周期：启动时创建并加载 TTS 后端。"""
    global engine
    logger.info("正在加载 TTS 服务（后端由配置决定）...")
    opts = {}
    model_dir = os.environ.get("COSYVOICE_MODEL_DIR", None)
    if model_dir:
        opts["model_dir"] = model_dir
    backend = create_tts_backend(**opts)
    backend.load()
    engine = TTSService(backend)
    logger.info("TTS 服务就绪 (采样率 %s)", engine.sample_rate)
    yield


app = FastAPI(
    title="AudioCloning API",
    description="个人声音克隆 REST API (MLX)",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """健康检查。"""
    return {
        "status": "ok",
        "framework": "MLX",
        "model": MODEL_ID,
        "sample_rate": engine.sample_rate if engine else None,
        "model_loaded": engine is not None,
    }


@app.post("/api/clone")
async def clone_voice(
    text: str = Form(..., description="合成文本"),
    ref_audio: UploadFile = File(..., description="参考音频文件"),
    ref_text: str = Form("", description="参考音频对应文本"),
    speed: float = Form(1.0, description="语速 0.5~2.0"),
    mode: str = Form("zero_shot", description="推理模式: zero_shot/cross_lingual/instruct"),
    instruct_text: str = Form("", description="指令文本 (instruct 模式)"),
) -> FileResponse:
    """声音克隆接口。"""
    if not engine:
        raise HTTPException(status_code=503, detail="模型未加载")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="合成文本为空")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"文本过长 (最大 {MAX_TEXT_LENGTH} 字符)")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")
    if mode == "instruct" and not instruct_text.strip():
        raise HTTPException(status_code=400, detail="instruct 模式需提供 instruct_text")

    # 保存上传的参考音频到临时文件
    suffix = Path(ref_audio.filename).suffix if ref_audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(ref_audio.file, tmp)
        tmp_path = tmp.name

    try:
        start = time.time()
        assert engine is not None
        output_path = engine.synthesize_to_file(
            mode=mode,
            text=text,
            ref_audio_path=tmp_path,
            ref_text=ref_text,
            instruct_text=instruct_text,
            speed=speed,
        )
        elapsed = time.time() - start

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=Path(output_path).name,
            headers={"X-Synthesis-Time": f"{elapsed:.2f}s"},
        )
    except Exception as e:  # noqa: BLE001
        logger.error("合成失败: %s", e)
        raise HTTPException(status_code=500, detail=f"合成失败: {e}") from e
    finally:
        os.unlink(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="AudioCloning API Server", prog="audioclone-api")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()