"""NiceGUI 主题层 — 设计令牌、全局样式与可复用样式助手。

界面视觉规范集中在本模块，业务面板（``components.py``）只消费语义化类名，
不散落魔法样式字符串，便于统一调整与后续换肤。

设计语言
- 深色为默认：深空蓝底 + 双色径向光晕，玻璃拟态卡片（半透明 + 模糊 + 细描边）。
- 强调色：靛蓝 → 紫罗兰渐变（主操作），青绿（辅助/成功态）。
- 圆角 18px、留白宽松、字重分层（正文 400 / 小标题 600）。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from nicegui import app, ui

# ─── 设计令牌（与 CSS 变量一一对应；供 Python 侧着色使用） ───
PRIMARY = "#6366f1"
PRIMARY_DEEP = "#4f46e5"
ACCENT = "#8b5cf6"
TEAL = "#14b8a6"
DARK_PAGE = "#080b14"
DARK_SURFACE = "#151a2b"

FONT_STACK = (
    '"Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", '
    '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif'
)

MONO_STACK = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'

CUSTOM_CSS = """
:root {
  --ac-font: __FONT__;
  --ac-mono: __MONO__;
  --ac-surface: rgba(255, 255, 255, 0.72);
  --ac-surface-strong: rgba(255, 255, 255, 0.94);
  --ac-surface-soft: rgba(148, 163, 184, 0.08);
  --ac-border: rgba(15, 23, 42, 0.09);
  --ac-border-strong: rgba(15, 23, 42, 0.18);
  --ac-text: #0f172a;
  --ac-text-muted: #64748b;
  --ac-chip-bg: rgba(99, 102, 241, 0.10);
}

html body.body--dark {
  --ac-surface: rgba(21, 26, 43, 0.66);
  --ac-surface-strong: rgba(19, 23, 38, 0.92);
  --ac-surface-soft: rgba(148, 163, 184, 0.10);
  --ac-border: rgba(148, 163, 184, 0.14);
  --ac-border-strong: rgba(148, 163, 184, 0.30);
  --ac-text: #e7ecf7;
  --ac-text-muted: #94a3b8;
  --ac-chip-bg: rgba(129, 140, 248, 0.16);
}

html body {
  font-family: var(--ac-font);
  color: var(--ac-text);
  background-attachment: fixed;
  background-image:
    radial-gradient(900px 520px at 10% -10%, rgba(99, 102, 241, 0.16), transparent 62%),
    radial-gradient(760px 460px at 92% 2%, rgba(20, 184, 166, 0.13), transparent 60%),
    radial-gradient(1000px 620px at 50% 115%, rgba(139, 92, 246, 0.12), transparent 70%);
  -webkit-font-smoothing: antialiased;
}

html body.body--dark {
  background-attachment: fixed;
  background-image:
    radial-gradient(900px 520px at 10% -10%, rgba(99, 102, 241, 0.26), transparent 62%),
    radial-gradient(760px 460px at 92% 2%, rgba(20, 184, 166, 0.16), transparent 60%),
    radial-gradient(1000px 620px at 50% 115%, rgba(139, 92, 246, 0.20), transparent 70%);
}

/* ─── 卡片：玻璃拟态容器 ─────────────────────────────────── */
.ac-card {
  background: var(--ac-surface) !important;
  border: 1px solid var(--ac-border) !important;
  border-radius: 18px !important;
  box-shadow: 0 12px 34px -20px rgba(15, 23, 42, 0.45) !important;
  backdrop-filter: blur(16px) saturate(140%);
  -webkit-backdrop-filter: blur(16px) saturate(140%);
  transition: border-color .2s ease, box-shadow .2s ease;
}
.ac-card--flat {
  box-shadow: none !important;
  background: var(--ac-surface-soft) !important;
}

/* ─── 排版 ───────────────────────────────────────────────── */
.ac-eyebrow {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: var(--ac-text-muted);
}
.ac-section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: .01em;
}
.ac-hint {
  font-size: 12px;
  line-height: 1.6;
  color: var(--ac-text-muted);
}
.ac-gradient-text {
  background: linear-gradient(100deg, #6366f1 0%, #a855f7 48%, #14b8a6 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

/* ─── 状态芯片 ───────────────────────────────────────────── */
.ac-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  line-height: 1.4;
  white-space: nowrap;
  border: 1px solid var(--ac-border);
  background: var(--ac-chip-bg);
  color: var(--ac-text-muted);
}
.ac-chip--ok { background: rgba(16, 185, 129, .14); border-color: rgba(16, 185, 129, .32); color: #10b981; }
.ac-chip--warn { background: rgba(245, 158, 11, .14); border-color: rgba(245, 158, 11, .32); color: #f59e0b; }
.ac-chip--err { background: rgba(244, 63, 94, .14); border-color: rgba(244, 63, 94, .32); color: #f43f5e; }
.ac-chip--accent { background: rgba(99, 102, 241, .16); border-color: rgba(99, 102, 241, .38); color: #818cf8; }

/* ─── 侧边导航 ───────────────────────────────────────────── */
.ac-nav {
  width: 100%;
  justify-content: flex-start !important;
  align-items: center;
  gap: 10px;
  padding: 9px 13px !important;
  border-radius: 12px;
  font-size: 13.5px;
  font-weight: 500;
  letter-spacing: .01em;
  color: var(--ac-text-muted);
  border: 1px solid transparent;
  transition: background .18s ease, color .18s ease, border-color .18s ease;
}
.ac-nav:hover { background: var(--ac-chip-bg); color: var(--ac-text); }
.ac-nav--active {
  background: linear-gradient(135deg, rgba(99, 102, 241, .22), rgba(168, 85, 247, .12));
  border-color: rgba(99, 102, 241, .38);
  color: var(--ac-text);
  font-weight: 600;
}
.ac-nav-icon { font-size: 19px; }

/* ─── 按钮 ───────────────────────────────────────────────── */
.ac-btn-primary {
  background-image: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #a855f7 100%) !important;
  color: #fff !important;
  font-weight: 600 !important;
  letter-spacing: .01em;
  border-radius: 12px;
  box-shadow: 0 10px 22px -12px rgba(99, 102, 241, .9) !important;
  transition: filter .18s ease, transform .18s ease;
}
.ac-btn-primary:hover { filter: brightness(1.08); }
.ac-btn-primary:active { transform: translateY(1px); }
.ac-btn-ghost {
  border-radius: 12px;
  font-weight: 500;
  border: 1px solid var(--ac-border-strong);
  color: var(--ac-text-muted);
}
.ac-btn-ghost:hover { color: var(--ac-text); border-color: var(--ac-border-strong); }

/* ─── 表单控件 ───────────────────────────────────────────── */
.ac-field .q-field__control {
  border-radius: 12px;
  background: var(--ac-surface-strong);
}
.ac-field .q-field__native,
.ac-field .q-field__label { color: var(--ac-text); }
.ac-field .q-field__marginal { color: var(--ac-text-muted); }

/* ─── 上传区（Quasar QUploader 皮肤） ─────────────────────── */
.ac-upload .q-uploader {
  width: 100%;
  border-radius: 14px;
  box-shadow: none;
  background: var(--ac-surface-soft);
  border: 1px dashed var(--ac-border-strong);
}
.ac-upload .q-uploader__header {
  background: transparent;
  border-bottom: 1px solid var(--ac-border);
  padding: 8px 12px;
  min-height: 46px;
}
.ac-upload .q-uploader__header-content { flex-wrap: nowrap; }
.ac-upload .q-uploader__title { font-size: 13px; font-weight: 600; color: var(--ac-text); }
.ac-upload .q-uploader__subtitle { font-size: 11px; color: var(--ac-text-muted); }
.ac-upload .q-uploader__list {
  padding: 8px;
  min-height: 84px;
  scrollbar-width: none;
}
.ac-upload .q-uploader__list::-webkit-scrollbar { display: none; }
.ac-upload .q-uploader__file {
  border-radius: 10px;
  background: var(--ac-surface-strong);
  border: 1px solid var(--ac-border);
  color: var(--ac-text);
}
.ac-upload .q-uploader__dnd {
  background: rgba(99, 102, 241, .12);
  border: 2px dashed rgba(99, 102, 241, .55);
}

/* ─── 结果区 ─────────────────────────────────────────────── */
.ac-result-text {
  font-family: var(--ac-mono);
  font-size: 13px;
  line-height: 1.85;
  white-space: pre-wrap;
  word-break: break-word;
  padding: 14px 16px;
  border-radius: 14px;
  background: var(--ac-surface-soft);
  border: 1px solid var(--ac-border);
  color: var(--ac-text);
}
.ac-audio-player { width: 100%; border-radius: 12px; }
.ac-audio-player audio { width: 100%; height: 40px; outline: none; }

/* ─── 响应式显隐（断点 1024px） ─────────────────────────────
   Tailwind 的 display 变体与 hidden 组合时顺序不可靠（lg:flex 压不过 hidden），
   故自建断点工具类；本段为非 layered 样式，优先级高于框架 utilities。        */
.ac-lg-up { display: none; }        /* 仅桌面端显示 */
.ac-under-lg { display: flex; }     /* 仅移动端显示 */
.ac-mobile-nav { display: none; }
.ac-mobile-nav--open { display: flex; }
@media (min-width: 1024px) {
  .ac-lg-up { display: flex; }
  .ac-under-lg { display: none; }
  .ac-mobile-nav, .ac-mobile-nav--open { display: none; }
}

/* ─── 杂项 ───────────────────────────────────────────────── */
.ac-divider { height: 1px; background: var(--ac-border); margin: 2px 0; }
.ac-scroll::-webkit-scrollbar { width: 8px; height: 8px; }
.ac-scroll::-webkit-scrollbar-thumb {
  background: var(--ac-border-strong);
  border-radius: 999px;
}
.ac-scroll::-webkit-scrollbar-track { background: transparent; }

@media (max-width: 640px) {
  .ac-card { border-radius: 14px !important; padding: 14px !important; }
}
"""


def apply_theme() -> None:
    """注册全局配色与样式（进程内调用一次即可）。

    用 ``app.colors``（应用级）而非 ``ui.colors``（页面级组件），
    以便在任何页面渲染前完成配色注册。
    """
    app.colors(
        primary=PRIMARY,
        secondary=TEAL,
        accent=ACCENT,
        positive="#10b981",
        negative="#f43f5e",
        warning="#f59e0b",
        info="#38bdf8",
        dark=DARK_SURFACE,
        dark_page=DARK_PAGE,
    )
    ui.add_css(CUSTOM_CSS.replace("__FONT__", FONT_STACK).replace("__MONO__", MONO_STACK), shared=True)


@contextmanager
def card(*, classes: str = "", dense: bool = False) -> Iterator[ui.card]:
    """统一风格的内容卡片（内边距用内联样式，避免被 Quasar 默认内边距覆盖）。"""
    padding = "14px" if dense else "20px"
    with ui.card().classes(f"ac-card w-full gap-3 {classes}".strip()) as c:
        c.style(f"padding: {padding}")
        yield c


def section_heading(title: str, *, icon: str = "") -> None:
    """小节标题（无容器，便于与右侧徽章等元素同排）。"""
    with ui.row().classes("items-center gap-2"):
        if icon:
            ui.icon(icon).classes("text-[19px] text-primary")
        ui.label(title).classes("ac-section-title")


@contextmanager
def section(title: str, *, icon: str = "", hint: str = "") -> Iterator[ui.column]:
    """带标题/说明的小节容器。"""
    with ui.column().classes("w-full gap-3") as container:
        section_heading(title, icon=icon)
        if hint:
            ui.label(hint).classes("ac-hint")
        yield container


def chip(text: str, *, tone: str = "", icon: str = "") -> ui.label:
    """状态芯片。tone: ok / warn / err / accent / ''。"""
    tone_cls = f" ac-chip--{tone}" if tone else ""
    label = ui.label(text).classes(f"ac-chip{tone_cls}")
    if icon:
        label.classes("items-center")
    return label


def eyebrow(text: str) -> ui.label:
    """小号全大写引导标签。"""
    return ui.label(text).classes("ac-eyebrow")


def divider() -> ui.element:
    """细分隔线。"""
    return ui.element("div").classes("ac-divider w-full")


__all__ = [
    "ACCENT",
    "DARK_PAGE",
    "DARK_SURFACE",
    "PRIMARY",
    "PRIMARY_DEEP",
    "TEAL",
    "apply_theme",
    "card",
    "chip",
    "divider",
    "eyebrow",
    "section",
    "section_heading",
]
