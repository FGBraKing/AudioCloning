"""NiceGUI 复合控件 — 音频输入（上传/录音）、结果展示、受控按钮等。

本模块封装界面中反复出现的交互模式，面板层（``components.py``）只做组装：
- :class:`AudioInput`：文件上传 + 浏览器录音 + 试听（对应 Gradio 的 upload/microphone 双入口）；
- :class:`AudioResult` / :class:`TextResult`：结果区（自动落盘、自动播放、下载）；
- :class:`CharCounterTextarea`：带实时字数校验的文本输入；
- :class:`SpeedControl`：语速调节（滑块 + 数值联动）；
- :class:`BusyButton` / :func:`run_guarded`：异步任务的忙碌态与异常提示。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Request
from nicegui import app, run, ui

from audioclone.config import MAX_TEXT_LENGTH, MAX_UPLOAD_BYTES
from audioclone.utils.audio import get_audio_duration
from audioclone.webui.errors import UIError
from audioclone.webui.media import is_upload_path, save_result, save_upload

logger = logging.getLogger(__name__)

# 录音上传端点与前端事件名（JS ↔ Python 的约定）
RECORDING_ROUTE = "/webui/api/recording"
RECORDING_EVENT = "webui_recording"

ACCEPT = "audio/*,.wav,.mp3,.flac,.ogg,.m4a"

# ─────────────────────────────────────────────────────────────
# 浏览器录音引擎（全局注入一次）：AudioWorklet 采集 PCM → 前端编码 WAV → 上传
# 以 WAV 上传可避免 webm/opus 在服务端的解码兼容问题。
# ─────────────────────────────────────────────────────────────
RECORDER_JS = """
window.ACRecorder = (function () {
  const WORKLET_SRC = `
class ACRecProc extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor('ac-rec-proc', ACRecProc);
`;
  const sessions = new Map();

  function writeAscii(view, offset, text) {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  }

  function encodeWav(chunks, sampleRate) {
    let total = 0;
    for (const chunk of chunks) total += chunk.length;
    const view = new DataView(new ArrayBuffer(44 + total * 2));
    writeAscii(view, 0, 'RIFF');
    view.setUint32(4, 36 + total * 2, true);
    writeAscii(view, 8, 'WAVE');
    writeAscii(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);          // PCM
    view.setUint16(22, 1, true);          // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeAscii(view, 36, 'data');
    view.setUint32(40, total * 2, true);
    let offset = 44;
    for (const chunk of chunks) {
      for (let i = 0; i < chunk.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      }
    }
    return new Blob([view], { type: 'audio/wav' });
  }

  async function start(id, endpoint) {
    if (sessions.has(id)) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true }
      });
      const Ctx = window.AudioContext || window.webkitAudioContext;
      const ctx = new Ctx();
      if (ctx.state === 'suspended') await ctx.resume();
      const source = ctx.createMediaStreamSource(stream);
      const mute = ctx.createGain();       // 零增益汇出：驱动处理链路且避免回声
      mute.gain.value = 0;
      const chunks = [];
      let useWorklet = false;
      try {
        const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: 'application/javascript' }));
        await ctx.audioWorklet.addModule(url);
        URL.revokeObjectURL(url);
        useWorklet = true;
      } catch (e) { useWorklet = false; }
      let node;
      if (useWorklet) {
        node = new AudioWorkletNode(ctx, 'ac-rec-proc');
        node.port.onmessage = (ev) => chunks.push(new Float32Array(ev.data));
      } else {
        node = ctx.createScriptProcessor(4096, 1, 1);
        node.onaudioprocess = (ev) => chunks.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
      }
      source.connect(node);
      node.connect(mute);
      mute.connect(ctx.destination);
      sessions.set(id, { stream, ctx, source, node, chunks, sampleRate: ctx.sampleRate, endpoint });
    } catch (err) {
      emitEvent('__EVENT__', { id: id, error: (err && err.message) || String(err) });
    }
  }

  async function stop(id) {
    const session = sessions.get(id);
    if (!session) return;
    sessions.delete(id);
    try {
      session.source.disconnect();
      session.node.disconnect();
      session.stream.getTracks().forEach((t) => t.stop());
      if (session.ctx.state !== 'closed') await session.ctx.close();
    } catch (e) { /* 关闭失败不影响已采集数据 */ }
    if (!session.chunks.length) {
      emitEvent('__EVENT__', { id: id, error: '未采集到音频，请检查麦克风权限' });
      return;
    }
    try {
      const form = new FormData();
      form.append('file', encodeWav(session.chunks, session.sampleRate), 'recording.wav');
      const resp = await fetch(session.endpoint, { method: 'POST', body: form });
      const data = await resp.json();
      emitEvent('__EVENT__', {
        id: id,
        path: (data && data.path) || '',
        name: (data && data.name) || '',
        duration: (data && data.duration) || 0,
        error: (data && data.error) || ''
      });
    } catch (err) {
      emitEvent('__EVENT__', { id: id, error: '录音上传失败: ' + ((err && err.message) || String(err)) });
    }
  }

  async function cancel(id) {
    const session = sessions.get(id);
    if (!session) return;
    sessions.delete(id);
    try {
      session.source.disconnect();
      session.node.disconnect();
      session.stream.getTracks().forEach((t) => t.stop());
      if (session.ctx.state !== 'closed') await session.ctx.close();
    } catch (e) { /* 忽略 */ }
  }

  return { start, stop, cancel, isActive: (id) => sessions.has(id) };
})();
"""


def register_global_widgets() -> None:
    """注入录音脚本与录音上传端点（进程内调用一次）。"""
    ui.add_head_html(f"<script>{RECORDER_JS.replace('__EVENT__', RECORDING_EVENT)}</script>", shared=True)

    if any(getattr(r, "path", None) == RECORDING_ROUTE for r in app.routes):
        return

    @app.post(RECORDING_ROUTE)  # type: ignore[misc]
    async def _save_recording(request: Request) -> dict[str, Any]:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            return {"error": f"录音文件过大（上限 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB）"}
        try:
            form = await request.form()
            upload = form["file"]
            data = await upload.read()  # type: ignore[union-attr]
            name = getattr(upload, "filename", None) or "recording.wav"
            path = save_upload(str(name), data)
            return {"path": path, "name": Path(path).name, "duration": get_audio_duration(path)}
        except Exception as e:  # noqa: BLE001 - 端点错误需回传前端提示
            logger.warning("[WebUI] 录音保存失败: %s", e)
            return {"error": f"录音保存失败: {e}"}


def notify_error(message: str) -> None:
    """统一的错误提示。"""
    ui.notify(message, type="negative", position="top", close_button="关闭", multi_line=True)


def notify_success(message: str) -> None:
    """统一的成功提示。"""
    ui.notify(message, type="positive", position="top", close_button="关闭")


# ─────────────────────────────────────────────────────────────
# 任务执行辅助
# ─────────────────────────────────────────────────────────────
class BusyButton:
    """带内置 spinner 与禁用态的主操作按钮。"""

    def __init__(
        self,
        text: str,
        *,
        icon: str | None = None,
        color: str | None = None,
        classes: str = "",
    ) -> None:
        self.button = ui.button(text, icon=icon, color=color).classes(f"{classes}".strip())
        with self.button:
            self._spinner = ui.spinner("dots", size="18px").classes("text-white")
            self._spinner.set_visibility(False)

    def on_click(self, handler: Callable[..., Any]) -> BusyButton:
        self.button.on_click(handler)
        return self

    def start(self) -> None:
        self.button.disable()
        self._spinner.set_visibility(True)

    def stop(self) -> None:
        self.button.enable()
        self._spinner.set_visibility(False)

    def set_enabled(self, enabled: bool) -> None:
        """启用/禁用按钮（用于面板能力不匹配时阻止无效提交）。"""
        if enabled:
            self.button.enable()
        else:
            self.button.disable()


async def run_guarded(button: BusyButton | None, work: Callable[[], Awaitable[Any]]) -> Any:
    """执行异步任务：忙碌态 + 统一异常提示。失败返回 ``None``。"""
    if button is not None:
        button.start()
        await asyncio.sleep(0)  # 让忙碌态先刷新到浏览器
    try:
        return await work()
    except UIError as e:
        notify_error(str(e))
    except Exception as e:  # noqa: BLE001 - 兜底提示，详细信息已入日志
        logger.error("[WebUI] 未预期的界面异常: %s", e, exc_info=True)
        notify_error(f"执行失败: {e}")
    finally:
        if button is not None:
            button.stop()
    return None


def io_bound(fn: Callable[..., Any], *args: Any) -> Awaitable[Any]:
    """在线程池执行阻塞的推理调用，避免卡住事件循环。"""
    return run.io_bound(fn, *args)


# ─────────────────────────────────────────────────────────────
# 输入类控件
# ─────────────────────────────────────────────────────────────
class AudioInput:
    """音频输入控件：文件上传 + 可选浏览器录音 + 试听 + 清除。"""

    def __init__(
        self,
        *,
        title: str,
        upload_label: str = "拖放音频到此处，或点击选择文件",
        hint_text: str = "",
        allow_record: bool = False,
    ) -> None:
        self.path: str | None = None
        self._uid = uuid.uuid4().hex[:12]
        self._elapsed = 0.0

        with ui.column().classes("w-full gap-2"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(title).classes("text-[13px] font-semibold")
                self._meta = ui.label("").classes("ac-hint")
            if hint_text:
                ui.label(hint_text).classes("ac-hint")

            self._upload = ui.upload(
                label=upload_label,
                auto_upload=True,
                max_file_size=MAX_UPLOAD_BYTES,
                on_upload=self._handle_upload,
            ).classes("w-full ac-upload").props(f'accept="{ACCEPT}" flat')
            self._upload.on_rejected(lambda: notify_error("文件被拒绝：格式或大小不符合要求"))

            if allow_record:
                with ui.row().classes("w-full items-center gap-2"):
                    self._record_btn = (
                        ui.button("录制", icon="mic", color=None)
                        .props("flat no-caps dense")
                        .classes("ac-btn-ghost")
                        .on_click(self._start_recording)
                    )
                    self._stop_btn = (
                        ui.button("停止", icon="stop", color=None)
                        .props("flat no-caps dense")
                        .classes("ac-btn-ghost")
                        .on_click(self._stop_recording)
                    )
                    self._stop_btn.set_visibility(False)
                    self._timer_label = ui.label("").classes("ac-hint")
                self._ticker = ui.timer(0.5, self._tick, active=False)

            self._preview = ui.column().classes("w-full gap-2")

        ui.on(RECORDING_EVENT, self._handle_recording_event)
        self._render_preview()

    # ── 状态 ──────────────────────────────────────────────
    @property
    def value(self) -> str | None:
        return self.path

    def clear(self) -> None:
        """清空当前音频。"""
        self._set_path(None)

    # ── 上传 ──────────────────────────────────────────────
    async def _handle_upload(self, e: Any) -> None:
        data = await e.file.read()
        path = save_upload(e.file.name, data)
        self._upload.reset()
        duration = get_audio_duration(path)
        if duration <= 0:
            notify_error(f"无法读取音频文件：{e.file.name}")
            Path(path).unlink(missing_ok=True)
            self._set_path(None)
            return
        self._set_path(path)

    # ── 录音 ──────────────────────────────────────────────
    def _start_recording(self) -> None:
        self._elapsed = 0.0
        self._record_btn.set_visibility(False)
        self._stop_btn.set_visibility(True)
        self._timer_label.set_text("● 录制中 0.0s")
        self._ticker.activate()
        ui.run_javascript(f"ACRecorder.start('{self._uid}', '{RECORDING_ROUTE}')")

    def _stop_recording(self) -> None:
        self._ticker.deactivate()
        self._stop_btn.set_visibility(False)
        self._record_btn.set_visibility(True)
        self._timer_label.set_text("正在处理录音…")
        ui.run_javascript(f"ACRecorder.stop('{self._uid}')")

    def _tick(self) -> None:
        self._elapsed += 0.5
        self._timer_label.set_text(f"● 录制中 {self._elapsed:.1f}s")

    def _handle_recording_event(self, e: Any) -> None:
        payload = e.args[0] if isinstance(e.args, list) and e.args else e.args
        if not isinstance(payload, dict) or payload.get("id") != self._uid:
            return
        self._ticker.deactivate()
        self._stop_btn.set_visibility(False)
        self._record_btn.set_visibility(True)

        error = payload.get("error") or ""
        if error:
            self._timer_label.set_text("录音失败")
            notify_error(f"录音失败：{error}")
            return

        path = str(payload.get("path") or "")
        if not path or not is_upload_path(path):
            self._timer_label.set_text("录音失败")
            notify_error("录音数据无效，请重试")
            return

        self._timer_label.set_text("录音完成")
        self._set_path(path)

    # ── 试听 ──────────────────────────────────────────────
    def _set_path(self, path: str | None) -> None:
        self.path = path
        self._render_preview()

    def _render_preview(self) -> None:
        self._preview.clear()
        with self._preview:
            if self.path:
                duration = get_audio_duration(self.path)
                self._meta.set_text(f"{Path(self.path).name} · {duration:.1f}s")
                with ui.row().classes("w-full items-center gap-3"):
                    ui.audio(self.path, controls=True).classes("ac-audio-player flex-1")
                    ui.button(icon="close", color=None).props("flat dense round").classes(
                        "ac-btn-ghost"
                    ).on_click(self.clear).tooltip("清除音频")
            else:
                self._meta.set_text("未选择音频")


class CharCounterTextarea:
    """带实时字数统计与超限提示的多行文本输入。"""

    def __init__(
        self,
        *,
        label: str,
        value: str = "",
        placeholder: str = "",
        max_length: int = MAX_TEXT_LENGTH,
        rows: int = 3,
    ) -> None:
        self.max_length = max_length
        with ui.column().classes("w-full gap-1"):
            self.textarea = (
                ui.textarea(label=label, value=value, placeholder=placeholder)
                .classes("w-full ac-field")
                .props(f"outlined rounded autogrow rows={rows}")
            )
            with ui.row().classes("w-full justify-end"):
                self._counter = ui.label().classes("ac-hint")
        self.textarea.on_value_change(self._update)
        self._update()

    @property
    def value(self) -> str:
        return self.textarea.value or ""

    def set_value(self, text: str) -> None:
        self.textarea.set_value(text)
        self._update()

    def _update(self) -> None:
        count = len(self.value)
        self._counter.set_text(f"{count} / {self.max_length} 字符")
        if count > self.max_length:
            self._counter.classes(add="text-negative font-semibold")
        else:
            self._counter.classes(remove="text-negative font-semibold")


class SpeedControl:
    """语速调节：滑块与数值双向联动。"""

    def __init__(self, *, value: float = 1.0, min_value: float = 0.5, max_value: float = 2.0) -> None:
        with ui.column().classes("w-full gap-1"):
            ui.label("语速").classes("text-[13px] font-semibold")
            with ui.row().classes("w-full items-center gap-3"):
                self.slider = ui.slider(min=min_value, max=max_value, step=0.1, value=value).classes("flex-1")
                self.number = ui.number(
                    value=value, min=min_value, max=max_value, step=0.1, format="%.1f"
                ).classes("w-24 ac-field")
        self.slider.bind_value(self.number, "value")

    @property
    def value(self) -> float:
        raw = self.number.value
        return float(raw) if raw else 1.0

    def reset(self) -> None:
        self.number.set_value(1.0)


# ─────────────────────────────────────────────────────────────
# 结果类控件
# ─────────────────────────────────────────────────────────────
class AudioResult:
    """音频结果区：落盘 → 播放器（自动播放）+ 元信息 + 下载。"""

    def __init__(self, *, title: str = "结果", autoplay: bool = True) -> None:
        self._autoplay = autoplay
        self._file: Path | None = None
        with ui.card().classes("ac-card w-full gap-2").style("padding: 16px"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(title).classes("ac-section-title")
                self._meta = ui.label("").classes("ac-hint")
            self._body = ui.column().classes("w-full gap-2")
        self._render_placeholder()

    def show(self, audio: np.ndarray, sample_rate: int, *, prefix: str = "result", note: str = "") -> None:
        """保存并展示音频。"""
        path = save_result(audio, sample_rate, prefix=prefix)
        self._file = path
        duration = len(audio) / sample_rate if sample_rate else 0.0
        self._meta.set_text(
            f"{path.name} · {duration:.1f}s · {sample_rate}Hz" + (f" · {note}" if note else "")
        )
        self._body.clear()
        with self._body:
            ui.audio(path, controls=True, autoplay=self._autoplay).classes("ac-audio-player")
            with ui.row().classes("w-full justify-end"):
                ui.button("下载 WAV", icon="download", color=None).props("flat no-caps dense").classes(
                    "ac-btn-ghost"
                ).on_click(lambda: ui.download(path))

    def clear(self) -> None:
        self._file = None
        self._meta.set_text("")
        self._render_placeholder()

    def _render_placeholder(self) -> None:
        self._body.clear()
        with self._body:
            ui.label("尚无结果，填写内容并点击执行按钮").classes("ac-hint")


class TextResult:
    """文本结果区：等宽展示文本/列表结果，支持复制。"""

    def __init__(self, *, title: str = "结果", placeholder: str = "尚无结果") -> None:
        self._placeholder = placeholder
        self._text = ""
        with ui.card().classes("ac-card w-full gap-2").style("padding: 16px"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(title).classes("ac-section-title")
                self._meta = ui.label("").classes("ac-hint")
            self._body = ui.column().classes("w-full")
        self._render_placeholder()

    def show(self, text: str, *, note: str = "") -> None:
        self._text = text or ""
        self._meta.set_text(note)
        self._body.clear()
        with self._body:
            ui.label(self._text or "（空）").classes("ac-result-text w-full")
            if self._text:
                with ui.row().classes("w-full justify-end"):
                    ui.button("复制", icon="content_copy", color=None).props("flat no-caps dense").classes(
                        "ac-btn-ghost"
                    ).on_click(self._copy)

    def clear(self) -> None:
        self._text = ""
        self._meta.set_text("")
        self._render_placeholder()

    def _copy(self) -> None:
        ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(self._text)})")
        notify_success("已复制到剪贴板")

    def _render_placeholder(self) -> None:
        self._body.clear()
        with self._body:
            ui.label(self._placeholder).classes("ac-hint")


__all__ = [
    "AudioInput",
    "AudioResult",
    "BusyButton",
    "CharCounterTextarea",
    "SpeedControl",
    "TextResult",
    "io_bound",
    "notify_error",
    "notify_success",
    "register_global_widgets",
    "run_guarded",
]
