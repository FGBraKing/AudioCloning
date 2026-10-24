"""NiceGUI 面板布局 — 总览 / 三个合成面板 / ASR / 多任务实验台 / 使用指南。

本模块只负责「视图」：创建控件并组织排版，对外暴露控件字典供 ``app.py`` 绑定事件，
业务逻辑一律位于 ``handlers.py``，两侧通过控件 key 解耦。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nicegui import ui

from audioclone.core import model_manager  # noqa: E402
from audioclone.core.capabilities import (
    TTS_MODES,
)
from audioclone.webui import theme
from audioclone.webui.widgets import (
    AudioInput,
    AudioResult,
    BusyButton,
    CharCounterTextarea,
    SpeedControl,
    TextResult,
)


# ─────────────────────────────────────────────────────────────
# 面板容器
# ─────────────────────────────────────────────────────────────
@dataclass
class Panel:
    """一个导航面板：``container`` 为根容器（切换可见性），``widgets`` 暴露内部控件。"""

    key: str
    title: str
    icon: str
    container: ui.column
    widgets: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.widgets[key]

    def set_visibility(self, visible: bool) -> None:
        self.container.set_visibility(visible)


def grid(classes: str) -> ui.element:
    """响应式栅格容器（Tailwind 断点控制列数）。"""
    return ui.element("div").classes(f"grid w-full gap-4 {classes}")


def info_row(label: str, value: str, *, tone: str = "") -> None:
    """「键 — 值」信息行。"""
    with ui.row().classes("w-full items-center justify-between gap-4 py-1"):
        ui.label(label).classes("text-[13px] shrink-0").style("color: var(--ac-text-muted)")
        if tone:
            theme.chip(value, tone=tone)
        else:
            ui.label(value).classes("text-[13px] font-medium text-right")


def backends_supporting(mode: str) -> list[str]:
    """列出支持指定推理模式的已登记 TTS 后端名（用于给出可切换的指引）。"""
    from audioclone.config import TTS_BACKENDS
    from audioclone.core.backends.registry import registry

    names: list[str] = []
    for name in TTS_BACKENDS:
        factory = registry.get_factory(name)
        if factory is not None and isinstance(factory, type):
            if mode in getattr(factory, "supported_modes", ()):
                names.append(name)
    return names


def mode_guard(mode: str, supported_modes: tuple[str, ...]) -> bool:
    """按当前后端能力守卫一个合成面板：支持则静默，不支持则渲染说明条。

    返回当前后端是否支持该模式；调用方据此决定是否禁用提交按钮，
    让用户在点击前就明白「为什么不能用」，而不是点击后才收到报错。
    """
    if mode in supported_modes:
        return True

    with ui.row().classes("w-full items-start gap-2 p-3 rounded-2xl").style(
        "background: rgba(245,158,11,.10); border: 1px solid rgba(245,158,11,.35)"
    ):
        ui.icon("info").classes("text-[18px]").style("color: var(--ac-warning, #f59e0b)")
        with ui.column().classes("gap-1"):
            ui.label(f"当前 TTS 后端不支持「{MODE_LABELS.get(mode, mode)}」模式，本面板已停用").classes(
                "text-[13px] font-semibold"
            )
            candidates = backends_supporting(mode)
            hint = "可切换到: " + " / ".join(candidates) if candidates else "当前没有后端支持该模式"
            ui.label(hint).classes("ac-hint")
    return False


def apply_mode_state(widget: BusyButton, supported: bool) -> None:
    """按模式守卫结果同步提交按钮的可用态。"""
    if supported:
        return
    widget.set_enabled(False)
    widget.button.tooltip("当前 TTS 后端不支持该模式")


# ─────────────────────────────────────────────────────────────
# 系统状态（由 app.py 组装后传入）
# ─────────────────────────────────────────────────────────────
@dataclass
class SystemStatus:
    """构建界面所需的系统快照。"""

    version: str
    tts_backend: str
    tts_model: str
    tts_ready: bool
    tts_modes: tuple[str, ...]
    asr_backend: str
    asr_ready: bool
    sample_rate: int
    load_sec: float | None
    webui_url: str
    api_url: str
    capabilities: tuple[str, ...]


#: 推理模式 → 中文展示名（部分后端只支持子集，如 Qwen3-TTS CustomVoice 仅 instruct）
MODE_LABELS: dict[str, str] = {
    "zero_shot": "零样本克隆",
    "cross_lingual": "跨语言",
    "instruct": "指令控制",
}


# ─────────────────────────────────────────────────────────────
# 面板 1：总览 / 系统状态
# ─────────────────────────────────────────────────────────────


#: 下载卡片中的任务显示名
_TASK_DISPLAY_NAMES = {
    "tts": "语音合成 (TTS)",
    "asr": "语音识别 (ASR)",
    "t2a": "音乐生成 (T2A)",
    "a2a": "语音增强 (A2A)",
    "classification": "音频分类",
    "vad": "语音活动检测 (VAD)",
}

def build_overview_panel(status: SystemStatus) -> Panel:
    with ui.column().classes("w-full gap-5") as container:
        # Hero
        with theme.card():
            with ui.column().classes("w-full gap-2"):
                theme.eyebrow("AudioCloning · 音频模型体验工具箱")
                ui.label("把文字变成你的声音").classes("text-3xl font-bold ac-gradient-text")
                ui.label(
                    "Qwen3-TTS 声音克隆、语音识别、音乐生成、语音增强、情感分类与 VAD，"
                    "按运行环境自动选择推理框架（MLX / TensorRT Edge-LLM / sherpa-onnx），全部本地完成。"
                ).classes("ac-hint max-w-3xl")
                with ui.row().classes("flex-wrap gap-2 mt-1"):
                    theme.chip(f"v{status.version}", tone="accent")
                    theme.chip(f"后端 {status.tts_backend}")
                    theme.chip(f"Apple MLX · {status.sample_rate}Hz")

        with grid("grid-cols-1 xl:grid-cols-2"):
            # 系统状态
            with theme.card():
                with theme.section("系统状态", icon="monitor_heart"):
                    info_row(
                        "TTS 模型",
                        f"{status.tts_model}",
                    )
                    info_row(
                        "TTS 状态",
                        "已加载" if status.tts_ready else "未就绪",
                        tone="ok" if status.tts_ready else "err",
                    )
                    info_row(
                        "支持模式",
                        "、".join(MODE_LABELS.get(m, m) for m in status.tts_modes) or "（无）",
                    )
                    info_row(
                        "ASR 服务",
                        f"{status.asr_backend} · {'运行中' if status.asr_ready else '未就绪'}",
                        tone="ok" if status.asr_ready else "warn",
                    )
                    info_row("推理框架", f"Apple MLX · Metal GPU · {status.sample_rate}Hz")
                    if status.load_sec is not None:
                        info_row("模型加载耗时", f"{status.load_sec:.1f}s")
                    theme.divider()
                    with ui.row().classes("flex-wrap gap-2"):
                        theme.chip(f"WebUI {status.webui_url}", tone="accent")
                        theme.chip(f"API {status.api_url}")

            # 后端能力
            with theme.card():
                with theme.section("当前后端能力", icon="tune"):
                    ui.label(
                        "同一后端可聚合多种能力；不支持的任务会在执行时给出明确提示。"
                    ).classes("ac-hint")
                    with ui.row().classes("flex-wrap gap-2"):
                        for name in status.capabilities:
                            theme.chip(name, tone="ok")

        # 模型下载（六任务状态 + 一键下载）
        with theme.card():
            with theme.section("模型下载", icon="cloud_download"):
                ui.label(
                    "各任务模型默认不预置，点击下载（HuggingFace 优先，失败自动回退 "
                    "ModelScope；镜像端点 AUDIOCLONE_HF_ENDPOINT）。"
                ).classes("ac-hint")
                dl_widgets: dict[str, Any] = {}
                for kind in ("tts", "asr", "t2a", "a2a", "classification", "vad"):
                    st = model_manager.model_status(kind)
                    backend_name = st.get("backend", kind)
                    downloadable = st.get("downloadable") is True
                    cached = st.get("cached") is True
                    with grid("grid-cols-1 md:grid-cols-12 items-center gap-2"):
                        with ui.row().classes("md:col-span-8 items-center gap-2"):
                            ui.label(_TASK_DISPLAY_NAMES[kind]).classes("text-[14px] font-semibold")
                            theme.chip(backend_name,
                                       tone="warn" if backend_name.startswith("dummy") else "ok")
                            if not downloadable:
                                theme.chip("无需模型", tone="accent")
                            elif cached:
                                theme.chip("已就位", tone="ok")
                            else:
                                theme.chip("未下载", tone="warn")
                        with ui.row().classes("md:col-span-4 justify-end"):
                            btn = BusyButton("下载模型", icon="cloud_download")
                            btn.set_enabled(downloadable and not cached)
                            status_label = ui.label("").classes("ac-hint")
                        dl_widgets[f"dl_{kind}_btn"] = btn
                        dl_widgets[f"dl_{kind}_status"] = status_label

        # 快速上手
        with theme.card():
            with theme.section("快速上手", icon="rocket_launch"):
                steps = [
                    ("1", "准备参考音频", "3~10 秒清晰人声，wav / mp3 / flac 均可"),
                    ("2", "填写合成文本", "单次最多 300 字符，长文本自动分段拼接"),
                    ("3", "开始合成", "等待进度结束后即可试听与下载"),
                ]
                with grid("grid-cols-1 md:grid-cols-3"):
                    for no, title, desc in steps:
                        with ui.column().classes("gap-1 p-3 rounded-2xl").style(
                            "background: var(--ac-surface-soft); border: 1px solid var(--ac-border)"
                        ):
                            ui.label(no).classes("text-lg font-bold ac-gradient-text")
                            ui.label(title).classes("text-[14px] font-semibold")
                            ui.label(desc).classes("ac-hint")

    return Panel(key="overview", title="总览", icon="space_dashboard",
                 container=container, widgets=dl_widgets)


# ─────────────────────────────────────────────────────────────
# 面板 2：声音克隆（零样本）
# ─────────────────────────────────────────────────────────────
def build_zero_shot_panel(supported_modes: tuple[str, ...] = TTS_MODES) -> Panel:
    mode_ok = mode_guard("zero_shot", supported_modes)
    with ui.column().classes("w-full gap-5") as container:
        with theme.card(dense=True):
            with ui.column().classes("w-full gap-1"):
                ui.label("声音克隆 · 零样本").classes("text-xl font-bold")
                ui.label(
                    "上传或录制一段 3~10 秒的参考音频，即可用该音色合成任意文本；"
                    "提供参考音频对应的文字可进一步克隆精度。"
                ).classes("ac-hint")

        with grid("grid-cols-1 lg:grid-cols-12"):
            with theme.card(classes="lg:col-span-5"):
                with theme.section("参考音频", icon="graphic_eq"):
                    audio = AudioInput(
                        title="上传或录制参考音频",
                        upload_label="拖放参考音频，或点击选择文件",
                        hint_text="支持 wav / mp3 / flac，时长 1~30 秒，建议 3~10 秒清晰人声",
                        allow_record=True,
                    )
                    prompt_text = (
                        ui.input(
                            label="参考文本（可选，提升克隆精度）",
                            placeholder="请输入参考音频对应的文字内容…",
                        )
                        .classes("w-full ac-field")
                        .props("outlined rounded")
                    )

            with theme.card(classes="lg:col-span-7"):
                with theme.section("合成文本", icon="edit_note"):
                    text = CharCounterTextarea(
                        label="输入合成文本",
                        value="你好，欢迎使用个人声音克隆系统，我是由AI生成的语音助手。",
                        placeholder="请输入需要合成的文本…",
                        rows=4,
                    )
                    speed = SpeedControl()
                    theme.divider()
                    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                        generate = BusyButton("开始合成", icon="auto_awesome", classes="ac-btn-primary px-5")
                        clear = (
                            ui.button("清空", icon="restart_alt")
                            .props("flat no-caps")
                            .classes("ac-btn-ghost")
                        )
                        apply_mode_state(generate, mode_ok)

        result = AudioResult(title="合成结果")

    return Panel(
        key="zero_shot",
        title="声音克隆（零样本）",
        icon="record_voice_over",
        container=container,
        widgets={
            "audio": audio,
            "prompt_text": prompt_text,
            "text": text,
            "speed": speed,
            "generate": generate,
            "clear": clear,
            "result": result,
        },
    )


# ─────────────────────────────────────────────────────────────
# 面板 3：跨语言合成
# ─────────────────────────────────────────────────────────────
def build_cross_lingual_panel(supported_modes: tuple[str, ...] = TTS_MODES) -> Panel:
    mode_ok = mode_guard("cross_lingual", supported_modes)
    with ui.column().classes("w-full gap-5") as container:
        with theme.card(dense=True):
            with ui.column().classes("w-full gap-1"):
                ui.label("跨语言合成").classes("text-xl font-bold")
                ui.label(
                    "使用源语言的音色合成目标语言语音，无需提供参考音频的文本转录。"
                ).classes("ac-hint")

        with grid("grid-cols-1 lg:grid-cols-12"):
            with theme.card(classes="lg:col-span-5"):
                with theme.section("参考音频（源语言音色）", icon="graphic_eq"):
                    audio = AudioInput(
                        title="上传参考音频",
                        upload_label="拖放参考音频，或点击选择文件",
                        hint_text="支持 wav / mp3 / flac，时长 1~30 秒",
                    )

            with theme.card(classes="lg:col-span-7"):
                with theme.section("合成文本（目标语言）", icon="translate"):
                    text = CharCounterTextarea(
                        label="输入目标语言文本",
                        value="Hello, this is a cross-lingual voice cloning demonstration.",
                        placeholder="Enter the text in target language…",
                        rows=4,
                    )
                    speed = SpeedControl()
                    theme.divider()
                    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                        generate = BusyButton("开始合成", icon="auto_awesome", classes="ac-btn-primary px-5")
                        clear = ui.button("清空", icon="restart_alt").props("flat no-caps").classes(
                            "ac-btn-ghost"
                        )
                        apply_mode_state(generate, mode_ok)

        result = AudioResult(title="合成结果")

    return Panel(
        key="cross_lingual",
        title="跨语言合成",
        icon="translate",
        container=container,
        widgets={
            "audio": audio,
            "text": text,
            "speed": speed,
            "generate": generate,
            "clear": clear,
            "result": result,
        },
    )


# ─────────────────────────────────────────────────────────────
# 面板 4：自然语言控制
# ─────────────────────────────────────────────────────────────
def build_instruct_panel(supported_modes: tuple[str, ...] = TTS_MODES) -> Panel:
    mode_ok = mode_guard("instruct", supported_modes)
    with ui.column().classes("w-full gap-5") as container:
        with theme.card(dense=True):
            with ui.column().classes("w-full gap-1"):
                ui.label("自然语言控制").classes("text-xl font-bold")
                ui.label(
                    "用一句话描述语气、情感、方言或语速，模型会据此调整合成表现。"
                ).classes("ac-hint")

        with grid("grid-cols-1 lg:grid-cols-12"):
            with theme.card(classes="lg:col-span-5"):
                with theme.section("参考音频（音色来源）", icon="graphic_eq"):
                    audio = AudioInput(
                        title="上传参考音频",
                        upload_label="拖放参考音频，或点击选择文件",
                        hint_text="支持 wav / mp3 / flac，时长 1~30 秒",
                    )
                with theme.section("自然语言指令", icon="chat"):
                    instruct = (
                        ui.textarea(
                            label="指令内容",
                            value="请用开心的语气说",
                            placeholder="例如：请用开心的语气说 / 请用广东话表达 / 请尽可能快地说",
                        )
                        .classes("w-full ac-field")
                        .props("outlined rounded autogrow rows=2")
                    )

            with theme.card(classes="lg:col-span-7"):
                with theme.section("合成文本", icon="edit_note"):
                    text = CharCounterTextarea(
                        label="输入合成文本",
                        value="今天天气真好，适合出去走走，感受一下阳光和微风。",
                        placeholder="请输入需要合成的文本…",
                        rows=4,
                    )
                    speed = SpeedControl()
                    theme.divider()
                    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                        generate = BusyButton("开始合成", icon="auto_awesome", classes="ac-btn-primary px-5")
                        clear = ui.button("清空", icon="restart_alt").props("flat no-caps").classes(
                            "ac-btn-ghost"
                        )
                        apply_mode_state(generate, mode_ok)

        result = AudioResult(title="合成结果")

    return Panel(
        key="instruct",
        title="自然语言控制",
        icon="psychology",
        container=container,
        widgets={
            "audio": audio,
            "instruct": instruct,
            "text": text,
            "speed": speed,
            "generate": generate,
            "clear": clear,
            "result": result,
        },
    )


# ─────────────────────────────────────────────────────────────
# 面板 5：语音识别 (ASR)
# ─────────────────────────────────────────────────────────────
def build_plain_panel(supported_modes: tuple[str, ...] = TTS_MODES) -> Panel:
    """纯文本直读合成面板（plain 模式；sherpa-onnx Kokoro 等，无需参考音频）。"""
    mode_ok = mode_guard("plain", supported_modes)
    with ui.column().classes("w-full gap-5") as container:
        with theme.card(dense=True):
            with ui.column().classes("w-full gap-1"):
                ui.label("纯文本合成").classes("text-xl font-bold")
                ui.label(
                    "无需参考音频，直接由预置音色朗读文本（plain 模式；"
                    "适合 sherpa-onnx Kokoro 等 CPU 推理后端）。"
                ).classes("ac-hint")
            if not mode_ok:
                ui.label("当前后端不支持 plain 模式，请切换后端（如 sherpa-tts-onnx）。").classes(
                    "ac-hint text-orange-400"
                )

        with theme.card():
            with theme.section("合成文本", icon="edit_note"):
                text = CharCounterTextarea(
                    label="输入合成文本",
                    value="你好，这是纯文本直读合成的示例。",
                    placeholder="请输入需要合成的文本…",
                    rows=4,
                )
                speed = SpeedControl()
                theme.divider()
                with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                    generate = BusyButton("开始合成", icon="record_voice_over", classes="ac-btn-primary px-5")
                    clear = (
                        ui.button("清空", icon="restart_alt")
                        .props("flat no-caps")
                        .classes("ac-btn-ghost")
                    )
            with ui.row().classes("w-full"):
                result = AudioResult(title="合成结果")

    return Panel(
        key="plain",
        title="纯文本合成",
        icon="record_voice_over",
        container=container,
        widgets={
            "text": text,
            "speed": speed,
            "generate": generate,
            "clear": clear,
            "result": result,
        },
    )


def build_asr_panel() -> Panel:
    with ui.column().classes("w-full gap-5") as container:
        with theme.card(dense=True):
            with ui.column().classes("w-full gap-1"):
                ui.label("语音识别 (ASR)").classes("text-xl font-bold")
                ui.label("上传或录制音频，本地转为文字，支持最长 5 分钟。").classes("ac-hint")

        with grid("grid-cols-1 lg:grid-cols-12"):
            with theme.card(classes="lg:col-span-5"):
                with theme.section("音频输入", icon="audio_file"):
                    audio = AudioInput(
                        title="上传或录制音频",
                        upload_label="拖放音频文件，或点击选择文件",
                        hint_text="支持 wav / mp3 / flac / ogg / m4a，最长 5 分钟",
                        allow_record=True,
                    )

            with theme.card(classes="lg:col-span-7"):
                with theme.section("识别结果", icon="subtitles"):
                    result = TextResult(
                        title="识别文本", placeholder="点击「开始识别」后在此显示结果…"
                    )
                    theme.divider()
                    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                        generate = BusyButton("开始识别", icon="hearing", classes="ac-btn-primary px-5")
                        clear = ui.button("清空", icon="restart_alt").props("flat no-caps").classes(
                            "ac-btn-ghost"
                        )

    return Panel(
        key="asr",
        title="语音识别 (ASR)",
        icon="hearing",
        container=container,
        widgets={
            "audio": audio,
            "generate": generate,
            "clear": clear,
            "result": result,
        },
    )


# ─────────────────────────────────────────────────────────────
# 面板 6：多任务实验台
# ─────────────────────────────────────────────────────────────
def _backend_badge(services: dict[str, Any], kind: str) -> None:
    """任务卡片徽标：显示该任务实际使用的后端名（dummy 以警示色区分）。"""
    backend = getattr(services.get(kind), "backend", None)
    name = getattr(backend, "name", "未接入")
    theme.chip(name, tone="warn" if name.startswith("dummy") else "ok")


def build_multitask_panel(task_services: dict[str, Any]) -> Panel:
    """多任务实验台：T2A / A2A / 分类 / VAD 各绑定专属后端。

    ``task_services`` 为 ``{kind: service}``（``create_service(kind)`` 创建，
    服务构造期已按能力校验后端），各卡片徽标显示实际后端名。
    """
    with ui.column().classes("w-full gap-5") as container:
        with theme.card(dense=True):
            with ui.column().classes("w-full gap-1"):
                ui.label("多任务实验台").classes("text-xl font-bold")
                ui.label(
                    "文本转音频 / 音频到音频 / 音频分类 / 语音活动检测。"
                    "各任务使用独立后端（徽标显示后端名；模型未下载时点击会提示下载方式）。"
                ).classes("ac-hint")

        with grid("grid-cols-1 xl:grid-cols-2"):
            # T2A
            with theme.card():
                with ui.row().classes("w-full items-center justify-between"):
                    theme.section_heading("文本转音频 (T2A)", icon="music_note")
                    _backend_badge(task_services, "t2a")
                t2a_text = (
                    ui.textarea(label="文本提示（描述声音/音乐/音效）", value="轻柔的钢琴背景音乐")
                    .classes("w-full ac-field")
                    .props("outlined rounded autogrow rows=2")
                )
                t2a_duration = ui.number(label="时长（秒）", value=2.0, min=0.5, max=30, step=0.5).classes(
                    "w-40 ac-field"
                )
                t2a_generate = BusyButton("生成", icon="music_note", classes="ac-btn-primary px-4")
                t2a_result = AudioResult(title="生成结果", autoplay=False)

            # A2A
            with theme.card():
                with ui.row().classes("w-full items-center justify-between"):
                    theme.section_heading("音频到音频 (A2A)", icon="transform")
                    _backend_badge(task_services, "a2a")
                a2a_audio = AudioInput(
                    title="上传音频",
                    upload_label="拖放音频文件，或点击选择文件",
                    hint_text="时长上限 300 秒",
                )
                a2a_instruct = (
                    ui.input(label="转换指令", value="改成男声", placeholder="例如：改成男声")
                    .classes("w-full ac-field")
                    .props("outlined rounded")
                )
                a2a_generate = BusyButton("转换", icon="transform", classes="ac-btn-primary px-4")
                a2a_result = AudioResult(title="转换结果", autoplay=False)

            # 音频分类
            with theme.card():
                with ui.row().classes("w-full items-center justify-between"):
                    theme.section_heading("音频分类", icon="category")
                    _backend_badge(task_services, "classification")
                cls_audio = AudioInput(
                    title="上传音频",
                    upload_label="拖放音频文件，或点击选择文件",
                    hint_text="时长上限 300 秒",
                )
                cls_generate = BusyButton("分类", icon="category", classes="ac-btn-primary px-4")
                cls_result = TextResult(title="分类结果（标签: 置信度）")

            # VAD
            with theme.card():
                with ui.row().classes("w-full items-center justify-between"):
                    theme.section_heading("语音活动检测 (VAD)", icon="graphic_eq")
                    _backend_badge(task_services, "vad")
                vad_audio = AudioInput(
                    title="上传音频",
                    upload_label="拖放音频文件，或点击选择文件",
                    hint_text="时长上限 300 秒",
                )
                vad_generate = BusyButton("检测", icon="graphic_eq", classes="ac-btn-primary px-4")
                vad_result = TextResult(title="语音片段（起止时间）")

    return Panel(
        key="multitask",
        title="多任务实验台",
        icon="science",
        container=container,
        widgets={
            "t2a_text": t2a_text,
            "t2a_duration": t2a_duration,
            "t2a_generate": t2a_generate,
            "t2a_result": t2a_result,
            "a2a_audio": a2a_audio,
            "a2a_instruct": a2a_instruct,
            "a2a_generate": a2a_generate,
            "a2a_result": a2a_result,
            "cls_audio": cls_audio,
            "cls_generate": cls_generate,
            "cls_result": cls_result,
            "vad_audio": vad_audio,
            "vad_generate": vad_generate,
            "vad_result": vad_result,
        },
    )


# ─────────────────────────────────────────────────────────────
# 面板 7：使用指南
# ─────────────────────────────────────────────────────────────
def build_guide_panel() -> Panel:
    features = [
        ("record_voice_over", "声音克隆（零样本）", "上传 3~10 秒参考音频 + 输入文本即可克隆音色；提供参考文本可提升精度"),
        ("translate", "跨语言合成", "使用源语言音色合成目标语言语音，无需提供文本转录"),
        ("psychology", "自然语言控制", "通过指令控制语气/情感/方言/语速，如「请用开心的语气说」"),
        ("hearing", "语音识别 (ASR)", "上传或录制音频转为文字，支持最长 5 分钟"),
        ("science", "多任务实验台", "T2A 生成 / A2A 变声 / 音频分类 / VAD，按后端能力可用"),
    ]
    specs = [
        ("时长", "3~10 秒（硬性限制 1~30 秒）"),
        ("格式", "wav / mp3 / flac"),
        ("采样率", "≥ 16kHz（自动重采样到 24kHz）"),
        ("内容", "清晰人声，避免背景噪音和音乐"),
    ]
    notices = [
        "合成文本上限 300 字符，超出将拒绝合成",
        "首次加载 TTS 模型约需 30~60 秒，请等待状态面板显示就绪后再操作",
        "ASR 服务首次启动约需 10~20 秒（模型加载到 Metal GPU）",
        "若启动报 503/网络错误，请确认系统代理已绕过 localhost（项目已默认配置）",
        "若端口被占用，使用 --port 参数指定其他端口",
    ]

    with ui.column().classes("w-full gap-5") as container:
        with theme.card(dense=True):
            with ui.column().classes("w-full gap-1"):
                ui.label("使用指南").classes("text-xl font-bold")
                ui.label("功能速览、参考音频规格与常见注意事项。").classes("ac-hint")

        with theme.card():
            with theme.section("功能速览", icon="apps"):
                with grid("grid-cols-1 md:grid-cols-2 xl:grid-cols-3"):
                    for icon, title, desc in features:
                        with ui.column().classes("gap-1 p-4 rounded-2xl").style(
                            "background: var(--ac-surface-soft); border: 1px solid var(--ac-border)"
                        ):
                            ui.icon(icon).classes("text-[22px] text-primary")
                            ui.label(title).classes("text-[14px] font-semibold")
                            ui.label(desc).classes("ac-hint")

        with grid("grid-cols-1 lg:grid-cols-2"):
            with theme.card():
                with theme.section("参考音频规格", icon="mic"):
                    with grid("grid-cols-2"):
                        for label, value in specs:
                            with ui.column().classes("gap-1 p-3 rounded-2xl").style(
                                "background: var(--ac-surface-soft); border: 1px solid var(--ac-border)"
                            ):
                                ui.label(label).classes("ac-eyebrow")
                                ui.label(value).classes("text-[13px] font-medium")

            with theme.card():
                with theme.section("注意事项", icon="error_outline"):
                    with ui.column().classes("w-full gap-2"):
                        for text in notices:
                            with ui.row().classes("items-start gap-2"):
                                ui.icon("fiber_manual_record").classes("text-[8px] mt-[6px] text-primary")
                                ui.label(text).classes("text-[13px] flex-1").style("line-height: 1.6")

            with theme.card(classes="lg:col-span-2"):
                with theme.section("输出说明", icon="folder_open"):
                    ui.label("合成音频保存于 outputs/ 目录").classes("text-[13px]")
                    ui.label(
                        "可通过环境变量 AUDIOCLONE_OUTPUT_KEEP_MAX 设置保留文件数上限"
                        "（>0 自动清理旧输出，0 不清理）。"
                    ).classes("ac-hint")

    return Panel(key="guide", title="使用指南", icon="menu_book", container=container)


__all__ = [
    "Panel",
    "SystemStatus",
    "build_asr_panel",
    "build_cross_lingual_panel",
    "build_guide_panel",
    "build_instruct_panel",
    "build_multitask_panel",
    "build_plain_panel",
    "build_overview_panel",
    "build_zero_shot_panel",
]
