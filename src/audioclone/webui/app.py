"""AudioCloning Gradio WebUI 入口 (命令行: ``uv run audioclone-webui``)。"""

from __future__ import annotations

import argparse
import logging
import os

# 本地回环不受系统代理影响（在 gradio/httpx 等网络库导入前设置）
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")

import gradio as gr

from audioclone.config import (
    ASR_SERVER_PORT,
    GRADIO_CONCURRENCY,
    GRADIO_HOST,
    GRADIO_MAX_QUEUE,
    GRADIO_PORT,
    MAX_TEXT_LENGTH,
    MODEL_ID,
)
from audioclone.core import (
    ASRService,
    ModelBackend,
    TTSService,
    create_asr_backend,
    create_tts_backend,
    list_backend_capabilities,
)
from audioclone.webui.components import (
    build_asr_tab,
    build_cross_lingual_tab,
    build_instruct_tab,
    build_system_status_panel,
    build_zero_shot_tab,
)
from audioclone.webui.handlers import (
    make_asr_handler,
    make_cross_lingual_handler,
    make_instruct_handler,
    make_zero_shot_handler,
)

logger = logging.getLogger(__name__)


def build_ui(engine: TTSService, asr_engine: ASRService) -> gr.Blocks:
    """构建完整 Gradio 界面。"""
    demo = gr.Blocks(title="AudioCloning - 个人声音克隆系统")

    with demo:
        gr.Markdown(
            """
            # AudioCloning - 音频模型体验工具箱 v0.1.1
            基于 Fun-CosyVoice3-0.5B (MLX) | 支持零样本克隆、跨语言合成、自然语言指令控制、语音识别
            """
        )

        # 系统状态面板
        status = build_system_status_panel()

        tts_status_str = f"✅ **TTS 模型:** {MODEL_ID}"
        asr_ready = asr_engine.is_ready()
        asr_status_str = (
            f"✅ **ASR 服务:** 运行中 (127.0.0.1:{ASR_SERVER_PORT})"
            if asr_ready
            else "⚠️ **ASR 服务:** 未就绪"
        )
        device_str = f"🍎 **框架:** Apple MLX | 采样率: {engine.sample_rate}Hz | Metal GPU 加速"
        server_str = f"🌐 **WebUI:** http://localhost:{GRADIO_PORT} | **API:** http://localhost:8000"

        demo.load(
            fn=lambda: (tts_status_str, asr_status_str, device_str, server_str),
            inputs=[],
            outputs=[
                status["tts_status"],
                status["asr_status"],
                status["device_info"],
                status["server_info"],
            ],
        )

        # 三个 Tab
        with gr.Tabs():
            with gr.Tab("声音克隆 (零样本)"):
                zs = build_zero_shot_tab()
                zs_handler = make_zero_shot_handler(engine)
                zs["generate_btn"].click(
                    fn=zs_handler,
                    inputs=[
                        zs["tts_text"],
                        zs["prompt_wav_upload"],
                        zs["prompt_wav_record"],
                        zs["prompt_text"],
                        zs["speed"],
                    ],
                    outputs=[zs["audio_output"]],
                )
                zs["clear_btn"].click(
                    fn=lambda: ("", None, None, "", None),
                    outputs=[
                        zs["tts_text"],
                        zs["prompt_wav_upload"],
                        zs["prompt_wav_record"],
                        zs["prompt_text"],
                        zs["audio_output"],
                    ],
                )

            with gr.Tab("跨语言合成"):
                cl = build_cross_lingual_tab()
                cl_handler = make_cross_lingual_handler(engine)
                cl["generate_btn"].click(
                    fn=cl_handler,
                    inputs=[cl["tts_text"], cl["prompt_wav"], cl["speed"]],
                    outputs=[cl["audio_output"]],
                )
                cl["clear_btn"].click(
                    fn=lambda: ("", None, None),
                    outputs=[cl["tts_text"], cl["prompt_wav"], cl["audio_output"]],
                )

            with gr.Tab("自然语言控制"):
                inst = build_instruct_tab()
                inst_handler = make_instruct_handler(engine)
                inst["generate_btn"].click(
                    fn=inst_handler,
                    inputs=[
                        inst["tts_text"],
                        inst["prompt_wav"],
                        inst["instruct_text"],
                        inst["speed"],
                    ],
                    outputs=[inst["audio_output"]],
                )
                inst["clear_btn"].click(
                    fn=lambda: ("", None, "", None),
                    outputs=[
                        inst["tts_text"],
                        inst["prompt_wav"],
                        inst["instruct_text"],
                        inst["audio_output"],
                    ],
                )

            with gr.Tab("语音识别 (ASR)"):
                asr = build_asr_tab()
                asr_handler = make_asr_handler(asr_engine)
                asr["generate_btn"].click(
                    fn=asr_handler,
                    inputs=[asr["audio_upload"], asr["audio_record"]],
                    outputs=[asr["text_output"], asr["info"]],
                )
                asr["clear_btn"].click(
                    fn=lambda: (None, None, "", ""),
                    outputs=[
                        asr["audio_upload"],
                        asr["audio_record"],
                        asr["text_output"],
                        asr["info"],
                    ],
                )

        gr.Markdown(
            "---\n"
            "### 📖 使用指南\n"
            "\n"
            "| 功能 Tab | 说明 |\n"
            "|----------|------|\n"
            "| **声音克隆 (零样本)** | 上传 3~10 秒参考音频 + 输入文本，即可克隆音色。提供参考文本可提升精度 |\n"
            "| **跨语言合成** | 使用源语言音色合成目标语言语音，无需提供文本转录 |\n"
            "| **自然语言控制** | 通过指令控制语气/情感/语速，如“请用开心的语气说” |\n"
            "| **语音识别 (ASR)** | 上传或录制音频，转为文字，支持最长 5 分钟 |\n"
            "\n"
            "### 🎙️ 参考音频规格\n"
            "\n"
            "- 时长: **3~10 秒**（硬性限制 1~30 秒）\n"
            "- 格式: wav / mp3 / flac\n"
            "- 采样率: ≥ 16kHz（系统自动重采样到 24kHz）\n"
            "- 内容: 清晰人声，避免背景噪音和音乐\n"
            "\n"
            "### ⚠️ 注意事项\n"
            "\n"
            f"- 合成文本上限 **{MAX_TEXT_LENGTH} 字符**，超出将拒绝合成\n"
            "- 首次加载 TTS 模型约需 **30~60 秒**，请等待状态面板显示 ✅ 后再操作\n"
            "- ASR 服务首次启动约需 10~20 秒（模型加载到 Metal GPU）\n"
            "- 若启动报 503/网络错误，请确认系统代理已绕过 localhost（项目已默认配置）\n"
            "- 若端口被占用，使用 `--port` 参数指定其他端口\n"
        )

    return demo


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="AudioCloning WebUI", prog="audioclone-webui")
    parser.add_argument("--port", type=int, default=GRADIO_PORT)
    parser.add_argument("--host", type=str, default=GRADIO_HOST)
    parser.add_argument("--model_dir", type=str, default=None, help="TTS 模型目录路径")
    parser.add_argument("--asr_model_dir", type=str, default=None, help="ASR 模型目录路径")
    parser.add_argument("--tts-backend", type=str, default=None, help="TTS 后端名称")
    parser.add_argument("--asr-backend", type=str, default=None, help="ASR 后端名称")
    parser.add_argument("--no_asr", action="store_true", help="禁用 ASR 功能")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  AudioCloning - 音频模型体验工具箱 v0.2.1")
    logger.info("=" * 60)

    tts_backend: ModelBackend | None = None
    asr_backend: ModelBackend | None = None
    try:
        # 加载 TTS 服务（后端经 model_loader 创建，默认 cosyvoice3-mlx）
        tts_opts: dict = {}
        if args.tts_backend is not None:
            tts_opts["name"] = args.tts_backend
        if args.model_dir is not None:
            tts_opts["model_dir"] = args.model_dir  # 显式传入时才覆盖，兼容 dummy 等后端
        tts_backend = create_tts_backend(**tts_opts)
        logger.info("正在加载 TTS 后端: %s (能力: %s)...", tts_backend.name, ", ".join(list_backend_capabilities(tts_backend)))
        tts_backend.load()
        engine = TTSService(tts_backend)
        logger.info("TTS 服务就绪 (采样率 %s)", engine.sample_rate)

        # 初始化 ASR 服务（后端经 model_loader 创建，默认 qwen3-asr-llama）
        asr_opts: dict = {}
        if args.asr_backend is not None:
            asr_opts["name"] = args.asr_backend
        if args.asr_model_dir is not None:
            asr_opts["model_dir"] = args.asr_model_dir
        asr_backend = create_asr_backend(**asr_opts)
        asr_engine = ASRService(asr_backend)
        if not args.no_asr:
            if asr_engine.is_model_available:
                logger.info("正在启动 ASR 服务 (%s)...", asr_backend.name)
                if asr_engine.start_server():
                    logger.info("ASR 服务已就绪")
                else:
                    logger.warning("ASR 服务启动失败，语音识别功能不可用")
            else:
                logger.warning("ASR 模型文件未找到，语音识别功能不可用")
                logger.warning("请运行: uv run audioclone-download-asr")
        else:
            logger.info("ASR 功能已禁用 (--no_asr)")

        # 构建 UI
        demo = build_ui(engine, asr_engine)
        demo.queue(max_size=GRADIO_MAX_QUEUE, default_concurrency_limit=GRADIO_CONCURRENCY)

        demo.launch(
            server_name=args.host,
            server_port=args.port,
            theme=gr.themes.Soft(),
        )
    finally:
        # 统一释放全部后端资源
        if asr_backend is not None:
            asr_backend.close()
        if tts_backend is not None:
            tts_backend.close()


if __name__ == "__main__":
    main()