"""AudioCloning NiceGUI WebUI 入口 (命令行: ``uv run audioclone-webui``)。

分层与 Gradio 版本一致：``components`` 出视图 → ``handlers`` 出逻辑 → 本模块装配事件与路由。
推理调用一律经 ``run.io_bound`` 放入线程池，避免阻塞 asyncio 事件循环。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from typing import Any, cast

# 本地回环不受系统代理影响（在 httpx 等网络库导入前设置）
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
# ALL_PROXY=socks5:// 会让 httpx 在创建客户端时就抛 ImportError（缺 socksio），
# 与目标主机是否命中 no_proxy 无关 —— 必须在建立任何 httpx 客户端前移除。
os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)

from nicegui import app, ui  # noqa: E402

from audioclone import __version__  # noqa: E402
from audioclone.config import (  # noqa: E402
    LOG_JSON,
    WEBUI_CONCURRENCY,
    WEBUI_HOST,
    WEBUI_PORT,
)
from audioclone.core import (  # noqa: E402
    ASRService,
    ModelBackend,
    TTSService,
    create_asr_backend,
    create_service,
    create_tts_backend,
    list_backend_capabilities,
)
from audioclone.core.capabilities import (  # noqa: E402
    SpeechRecognitionCapability,
    TextToSpeechCapability,
)
from audioclone.core.exceptions import ModelLoadError  # noqa: E402
from audioclone.core.runtime import detect_runtime  # noqa: E402
from audioclone.utils.logging_utils import configure_logging  # noqa: E402
from audioclone.webui import media, theme, widgets  # noqa: E402
from audioclone.webui.components import (  # noqa: E402
    Panel,
    SystemStatus,
    build_asr_panel,
    build_cross_lingual_panel,
    build_guide_panel,
    build_instruct_panel,
    build_multitask_panel,
    build_overview_panel,
    build_plain_panel,
    build_zero_shot_panel,
)
from audioclone.webui.handlers import (  # noqa: E402
    make_asr_handler,
    make_cross_lingual_handler,
    make_download_handler,
    make_instruct_handler,
    make_multitask_handlers,
    make_plain_handler,
    make_zero_shot_handler,
)

logger = logging.getLogger(__name__)

# 推理并发闸门：保护非线程安全的模型实现，避免多个请求同时进入后端
_INFER_SEM = asyncio.Semaphore(max(1, WEBUI_CONCURRENCY))


async def _infer(fn: Any, *args: Any) -> Any:
    """受限并发地在线程池执行阻塞推理。"""
    async with _INFER_SEM:
        return await widgets.io_bound(fn, *args)


def _tts_model_label(backend: ModelBackend) -> str:
    """展示用的 TTS 模型标识（取后端自带的 model_id）。"""
    return str(getattr(backend, "model_id", "") or "(未指定)")


# ─────────────────────────────────────────────────────────────
# 页面骨架
# ─────────────────────────────────────────────────────────────
def _build_index() -> None:
    """构建单页应用骨架：顶栏 + 侧边导航 + 面板切换。"""
    status: SystemStatus = app.state.webui_status
    engine: TTSService = app.state.engine
    asr_engine: ASRService = app.state.asr_engine
    task_services: dict[str, Any] = app.state.task_services

    dark = ui.dark_mode(app.state.webui_dark)
    nav_state = {"open": False}

    def toggle_mobile_nav() -> None:
        nav_state["open"] = not nav_state["open"]
        if nav_state["open"]:
            mobile_nav.classes(add="ac-mobile-nav--open")
        else:
            mobile_nav.classes(remove="ac-mobile-nav--open")

    # ── 顶栏 ──────────────────────────────────────────────
    with ui.header().classes("items-center px-4 lg:px-6").style(
        "background: var(--ac-surface); backdrop-filter: blur(16px);"
        "border-bottom: 1px solid var(--ac-border); color: var(--ac-text);"
    ):
        with ui.row().classes("w-full items-center gap-3 flex-wrap"):
            ui.button(icon="menu", color=None, on_click=toggle_mobile_nav).props(
                "flat dense round"
            ).classes("ac-under-lg").tooltip("功能导航")

            with ui.row().classes("items-center gap-2"):
                with ui.element("div").classes(
                    "w-9 h-9 rounded-xl flex items-center justify-center text-white"
                ).style("background-image: linear-gradient(135deg,#6366f1,#a855f7)"):
                    ui.icon("graphic_eq").classes("text-[20px]")
                ui.label("AudioCloning").classes("text-[17px] font-bold")
                theme.chip(f"v{status.version}", tone="accent")

            ui.element("div").classes("flex-1")

            with ui.row().classes("items-center gap-2 ac-lg-up"):
                theme.chip(f"TTS · {status.tts_backend}", tone="ok" if status.tts_ready else "err")
                theme.chip(
                    f"ASR · {status.asr_backend}",
                    tone="ok" if status.asr_ready else "warn",
                )

            dark_btn = ui.button(icon="light_mode", color=None).props("flat dense round").tooltip(
                "切换深浅色"
            )

    def toggle_dark() -> None:
        dark.toggle()
        dark_btn.props(f"icon={'light_mode' if dark.value else 'dark_mode'}")

    dark_btn.on_click(toggle_dark)

    # ── 主体：移动端导航 + 侧边导航 + 面板区 ────────────────
    with ui.element("div").classes("w-full max-w-[1440px] mx-auto px-4 lg:px-6 py-5"):
        mobile_nav = ui.element("div").classes("ac-mobile-nav flex-col gap-1 w-full")

        with ui.element("div").classes("flex w-full gap-5 items-start"):
            sidebar = ui.element("div").classes(
                "ac-lg-up flex-col gap-1 w-[220px] shrink-0 sticky top-[76px]"
            )
            with ui.column().classes("flex-1 min-w-0"):
                panels = [
                    build_overview_panel(status),
                    build_zero_shot_panel(status.tts_modes),
                    build_cross_lingual_panel(status.tts_modes),
                    build_instruct_panel(status.tts_modes),
                    build_plain_panel(status.tts_modes),
                    build_asr_panel(),
                    build_multitask_panel(task_services),
                    build_guide_panel(),
                ]

    _wire_navigation(panels, sidebar, mobile_nav, nav_state)
    _wire_handlers(panels, engine, asr_engine, task_services)
    _wire_download_buttons({p.key: p for p in panels})


def _wire_navigation(
    panels: list[Panel],
    sidebar: ui.element,
    mobile_nav: ui.element,
    nav_state: dict[str, bool],
) -> None:
    """创建导航项并绑定面板切换（桌面侧栏与移动端折叠菜单共用同一套按钮）。"""
    nav_slots: dict[str, list[ui.button]] = {}
    active = {"key": panels[0].key}

    def select(key: str) -> None:
        active["key"] = key
        for panel in panels:
            panel.set_visibility(panel.key == key)
        for panel_key, buttons in nav_slots.items():
            for button in buttons:
                if panel_key == key:
                    button.classes(add="ac-nav--active")
                else:
                    button.classes(remove="ac-nav--active")
        nav_state["open"] = False
        mobile_nav.classes(remove="ac-mobile-nav--open")

    for container in (sidebar, mobile_nav):
        with container:
            ui.label("功能导航").classes("ac-eyebrow px-3 pb-1")
            for panel in panels:
                button = (
                    ui.button(panel.title, icon=panel.icon, color=None)
                    .props("flat no-caps align=left")
                    .classes("ac-nav")
                    .on_click(lambda _=None, key=panel.key: select(key))
                )
                nav_slots.setdefault(panel.key, []).append(button)

    select(active["key"])


def _wire_handlers(
    panels: list[Panel],
    engine: TTSService,
    asr_engine: ASRService,
    task_services: dict[str, Any],
) -> None:
    """绑定各面板的业务事件（逻辑全部来自 handlers 层）。"""
    by_key = {panel.key: panel for panel in panels}
    zs, cl, inst = by_key["zero_shot"], by_key["cross_lingual"], by_key["instruct"]
    plain, asr_panel, mt = by_key["plain"], by_key["asr"], by_key["multitask"]

    zero_shot = make_zero_shot_handler(engine)
    cross_lingual = make_cross_lingual_handler(engine)
    instruct = make_instruct_handler(engine)
    plain_handler = make_plain_handler(engine)
    asr_handler = make_asr_handler(asr_engine)
    mt_handlers = make_multitask_handlers(task_services)

    # ── 声音克隆（零样本） ─────────────────────────────────
    async def on_zero_shot() -> None:
        w = zs.widgets
        out = await widgets.run_guarded(
            w["generate"],
            lambda: _infer(
                zero_shot,
                w["text"].value,
                w["audio"].value,
                None,
                w["prompt_text"].value,
                w["speed"].value,
            ),
        )
        if out:
            sample_rate, audio = out
            w["result"].show(audio, sample_rate, prefix="zero_shot")

    def clear_zero_shot() -> None:
        zs["text"].set_value("")
        zs["prompt_text"].set_value("")
        zs["audio"].clear()
        zs["speed"].reset()
        zs["result"].clear()

    zs["generate"].on_click(on_zero_shot)
    zs["clear"].on_click(clear_zero_shot)

    # ── 跨语言合成 ─────────────────────────────────────────
    async def on_cross_lingual() -> None:
        w = cl.widgets
        out = await widgets.run_guarded(
            w["generate"],
            lambda: _infer(cross_lingual, w["text"].value, w["audio"].value, w["speed"].value),
        )
        if out:
            sample_rate, audio = out
            w["result"].show(audio, sample_rate, prefix="cross_lingual")

    def clear_cross_lingual() -> None:
        cl["text"].set_value("")
        cl["audio"].clear()
        cl["speed"].reset()
        cl["result"].clear()

    cl["generate"].on_click(on_cross_lingual)
    cl["clear"].on_click(clear_cross_lingual)

    # ── 自然语言控制 ───────────────────────────────────────
    async def on_instruct() -> None:
        w = inst.widgets
        out = await widgets.run_guarded(
            w["generate"],
            lambda: _infer(
                instruct,
                w["text"].value,
                w["audio"].value,
                w["instruct"].value,
                w["speed"].value,
            ),
        )
        if out:
            sample_rate, audio = out
            w["result"].show(audio, sample_rate, prefix="instruct")

    def clear_instruct() -> None:
        inst["text"].set_value("")
        inst["instruct"].set_value("")
        inst["audio"].clear()
        inst["speed"].reset()
        inst["result"].clear()

    inst["generate"].on_click(on_instruct)
    inst["clear"].on_click(clear_instruct)

    # ── 纯文本合成 (plain) ─────────────────────────────────
    async def on_plain() -> None:
        w = plain.widgets
        out = await widgets.run_guarded(
            w["generate"], lambda: _infer(plain_handler, w["text"].value, w["speed"].value)
        )
        if out:
            sample_rate, audio = out
            w["result"].show(audio, sample_rate, prefix="plain")

    def clear_plain() -> None:
        plain["text"].set_value("")
        plain["speed"].reset()
        plain["result"].clear()

    plain["generate"].on_click(on_plain)
    plain["clear"].on_click(clear_plain)

    # ── 语音识别 (ASR) ─────────────────────────────────────
    async def on_asr() -> None:
        w = asr_panel.widgets
        out = await widgets.run_guarded(
            w["generate"], lambda: _infer(asr_handler, w["audio"].value, None)
        )
        if out:
            text, info = out
            w["result"].show(text, note=info)

    def clear_asr() -> None:
        asr_panel["audio"].clear()
        asr_panel["result"].clear()

    asr_panel["generate"].on_click(on_asr)
    asr_panel["clear"].on_click(clear_asr)

    # ── 多任务实验台 ───────────────────────────────────────
    async def on_t2a() -> None:
        w = mt.widgets
        out = await widgets.run_guarded(
            w["t2a_generate"],
            lambda: _infer(mt_handlers["t2a"], w["t2a_text"].value, w["t2a_duration"].value),
        )
        if out:
            sample_rate, audio = out
            w["t2a_result"].show(audio, sample_rate, prefix="t2a")

    async def on_a2a() -> None:
        w = mt.widgets
        out = await widgets.run_guarded(
            w["a2a_generate"],
            lambda: _infer(mt_handlers["a2a"], w["a2a_audio"].value, w["a2a_instruct"].value),
        )
        if out:
            sample_rate, audio = out
            w["a2a_result"].show(audio, sample_rate, prefix="a2a")

    async def on_classify() -> None:
        w = mt.widgets
        out = await widgets.run_guarded(
            w["cls_generate"], lambda: _infer(mt_handlers["classify"], w["cls_audio"].value)
        )
        if out:
            w["cls_result"].show(out, note="按置信度降序")

    async def on_vad() -> None:
        w = mt.widgets
        out = await widgets.run_guarded(
            w["vad_generate"], lambda: _infer(mt_handlers["vad"], w["vad_audio"].value)
        )
        if out:
            w["vad_result"].show(out, note="检测到的语音片段")

    mt["t2a_generate"].on_click(on_t2a)
    mt["a2a_generate"].on_click(on_a2a)
    mt["cls_generate"].on_click(on_classify)
    mt["vad_generate"].on_click(on_vad)


def _wire_download_buttons(panels_by_key: dict[str, Panel]) -> None:
    """绑定总览面板的模型下载按钮：进程内下载，完成后刷新状态标签。"""
    from audioclone.core import model_manager

    overview = panels_by_key.get("overview")
    if overview is None:
        return

    for kind in ("tts", "asr", "t2a", "a2a", "classification", "vad"):
        btn = overview.widgets.get(f"dl_{kind}_btn")
        status_label = overview.widgets.get(f"dl_{kind}_status")
        if btn is None or status_label is None:
            continue

        handler = make_download_handler(kind)

        def _refresh(kind: str = kind, status_label: Any = status_label) -> None:
            st = model_manager.model_status(kind)
            if st.get("cached") is True:
                status_label.set_value("已就位 ✓")
            elif st.get("downloadable") is False:
                status_label.set_value("该后端无需模型")
            else:
                status_label.set_value("仍未下载（可手动重试）")

        async def on_download(
            kind: str = kind, btn: Any = btn, handler: Any = handler
        ) -> None:
            out = await widgets.run_guarded(btn, lambda: _infer(handler))
            if out is not None:
                widgets.notify_success(str(out))
                await widgets.io_bound(_refresh)
                _refresh()

        btn.on_click(on_download)


# ─────────────────────────────────────────────────────────────
# 装配入口
# ─────────────────────────────────────────────────────────────
def build_pages() -> None:
    """注册页面路由（后端加载完成后调用一次）。"""
    theme.apply_theme()
    widgets.register_global_widgets()
    app.on_shutdown(media.cleanup)

    @ui.page("/")
    def index() -> None:
        _build_index()


def main() -> None:
    configure_logging(json=LOG_JSON)
    parser = argparse.ArgumentParser(description="AudioCloning WebUI", prog="audioclone-webui")
    parser.add_argument("--port", type=int, default=WEBUI_PORT)
    parser.add_argument("--host", type=str, default=WEBUI_HOST)
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="TTS 模型目录（默认使用 HuggingFace 系统缓存）",
    )
    parser.add_argument(
        "--asr_model_dir", type=str, default=None,
        help="ASR 模型目录（默认使用 HuggingFace 系统缓存）",
    )
    parser.add_argument("--tts-backend", type=str, default=None, help="TTS 后端名称")
    parser.add_argument("--asr-backend", type=str, default=None, help="ASR 后端名称")
    parser.add_argument("--t2a-backend", type=str, default=None, help="T2A（音乐生成）后端名称")
    parser.add_argument("--a2a-backend", type=str, default=None, help="A2A（音频转换）后端名称")
    parser.add_argument("--classification-backend", type=str, default=None, help="音频分类后端名称")
    parser.add_argument("--vad-backend", type=str, default=None, help="VAD 后端名称")
    parser.add_argument("--no_asr", action="store_true", help="禁用 ASR 功能")
    parser.add_argument(
        "--theme", choices=("dark", "light"), default="dark", help="界面配色（默认 dark）"
    )
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  AudioCloning - 音频模型体验工具箱 v%s", __version__)
    logger.info("=" * 60)

    # 运行环境探测（决定 auto 后端选择的推理框架）：macOS→MLX / NVIDIA→TensorRT-LLM /
    # AMD 独显→vLLM / AMD 核显与无 GPU→onnx（sherpa-onnx）；显存/内存不足自动降级 onnx。
    # 探测结果（含 mem/vram 容量与降级原因）由 runtime 模块自行输出日志，
    # 并经 /api/health 的 runtime 字段暴露；显式 --tts-backend/--asr-backend 时不受影响
    detect_runtime()

    tts_backend: ModelBackend | None = None
    asr_backend: ModelBackend | None = None
    task_backends: list[ModelBackend] = []
    try:
        # 加载 TTS 服务（后端经 model_loader 创建，默认 qwen3-tts-base-mlx）
        tts_opts: dict = {}
        if args.tts_backend is not None:
            tts_opts["name"] = args.tts_backend
        if args.model_dir is not None:
            tts_opts["model_dir"] = args.model_dir  # 显式传入时才覆盖，兼容 dummy 等后端
        tts_backend = create_tts_backend(**tts_opts)
        logger.info(
            "正在加载 TTS 后端: %s (能力: %s)...",
            tts_backend.name,
            ", ".join(list_backend_capabilities(tts_backend)),
        )
        load_start = time.time()
        try:
            tts_backend.load()
        except ModelLoadError as e:
            # 依赖缺失 / 模型缺失等可预期失败：给出可读原因而非原始堆栈
            logger.error("TTS 后端 %s 加载失败: %s", tts_backend.name, e)
            raise SystemExit(1) from None
        tts_load_sec = time.time() - load_start
        engine = TTSService(cast(TextToSpeechCapability, tts_backend))
        logger.info("TTS 服务就绪 (采样率 %s, 加载耗时 %.1fs)", engine.sample_rate, tts_load_sec)

        # 初始化 ASR 服务（后端经 model_loader 创建，默认按环境自动选择）
        asr_opts: dict = {}
        if args.asr_backend is not None:
            asr_opts["name"] = args.asr_backend
        if args.asr_model_dir is not None:
            asr_opts["model_dir"] = args.asr_model_dir
        asr_backend = create_asr_backend(**asr_opts)
        asr_engine = ASRService(cast(SpeechRecognitionCapability, asr_backend))
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

        # 多任务实验台：四个任务各绑专属服务（构造轻量不加载模型；首次调用时惰性加载）
        task_services: dict[str, Any] = {}
        task_backends = []
        for kind, cli_name in (
            ("t2a", args.t2a_backend),
            ("a2a", args.a2a_backend),
            ("classification", args.classification_backend),
            ("vad", args.vad_backend),
        ):
            svc = create_service(kind, cli_name)
            task_services[kind] = svc
            task_backends.append(svc.backend)
            logger.info("多任务后端 [%s]: %s", kind, svc.backend.name)

        app.state.engine = engine
        app.state.asr_engine = asr_engine
        app.state.task_services = task_services
        app.state.backend = cast(ModelBackend, engine.backend)
        app.state.webui_dark = args.theme == "dark"
        app.state.webui_status = SystemStatus(
            version=__version__,
            tts_backend=tts_backend.name,
            tts_model=_tts_model_label(tts_backend),
            tts_ready=True,
            tts_modes=tuple(getattr(tts_backend, "supported_modes", ()) or ()),
            asr_backend=asr_backend.name,
            asr_ready=asr_engine.is_ready(),
            sample_rate=engine.sample_rate,
            load_sec=tts_load_sec,
            webui_url=f"http://localhost:{args.port}",
            api_url="http://localhost:8000",
            capabilities=tuple(list_backend_capabilities(tts_backend)),
        )

        build_pages()
        logger.info("WebUI 启动中: http://%s:%s", args.host, args.port)
        ui.run(
            host=args.host,
            port=args.port,
            title=f"AudioCloning · 音频模型体验工具箱 v{__version__}",
            dark=args.theme == "dark",
            show=not args.no_browser,
            reload=False,
            uvicorn_logging_level="warning",
        )
    finally:
        # 统一释放全部后端资源
        for backend in task_backends:
            backend.close()
        if asr_backend is not None:
            asr_backend.close()
        if tts_backend is not None:
            tts_backend.close()


if __name__ == "__main__":
    main()
