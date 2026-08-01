"""
FastAPI REST API 服务器
提供程序化声音克隆接口
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path

# 绕过系统代理对本地回环地址的拦截
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AudioCloningAPI")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from typing import Optional
import tempfile
import shutil

from src.tts import TTSEngine, load_model
from src.config import MAX_TEXT_LENGTH, MODEL_ID

engine: Optional[TTSEngine] = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan: load model on startup."""
    global engine
    logger.info("正在加载 CosyVoice3 MLX 模型...")
    model_dir = os.environ.get("COSYVOICE_MODEL_DIR", None)
    mlx_model = load_model(model_dir)
    engine = TTSEngine(mlx_model)
    logger.info("模型加载完成")
    yield


app = FastAPI(
    title="AudioCloning API",
    description="个人声音克隆 REST API (MLX)",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
async def health():
    """健康检查"""
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
):
    """声音克隆接口"""
    if not engine:
        raise HTTPException(status_code=503, detail="模型未加载")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="合成文本为空")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"文本过长 (最大 {MAX_TEXT_LENGTH} 字符)")
    if mode not in ("zero_shot", "cross_lingual", "instruct"):
        raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}")

    # 保存上传的参考音频到临时文件
    suffix = Path(ref_audio.filename).suffix if ref_audio.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(ref_audio.file, tmp)
        tmp_path = tmp.name

    try:
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

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=Path(output_path).name,
            headers={"X-Synthesis-Time": f"{elapsed:.2f}s"},
        )
    except Exception as e:
        logger.error(f"合成失败: {e}")
        raise HTTPException(status_code=500, detail=f"合成失败: {e}")
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="AudioCloning API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
