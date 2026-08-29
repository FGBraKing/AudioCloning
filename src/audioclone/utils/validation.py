"""共享校验助手 — 供 API 与 WebUI handler 复用，避免校验逻辑重复。

约定：不合法统一抛 ``InvalidAudioInput``，由上层（FastAPI exception_handler /
``@ui_error`` 装饰器）转换为 400 或界面提示（``UIError``）。
"""

from __future__ import annotations

from audioclone.core.exceptions import InvalidAudioInput


def assert_tts_text(text: str, max_len: int) -> str:
    """校验合成文本：非空且不超长；不合法抛 ``InvalidAudioInput``，返回去除首尾空白的文本。"""
    if not text or not text.strip():
        raise InvalidAudioInput("合成文本为空")
    if len(text) > max_len:
        raise InvalidAudioInput(f"文本过长 ({len(text)} > {max_len} 字符)")
    return text.strip()


def assert_valid_mode(mode: str, modes: tuple[str, ...]) -> None:
    """校验推理模式合法；不合法抛 ``InvalidAudioInput``。"""
    if mode not in modes:
        raise InvalidAudioInput(f"不支持的模式: {mode}（可用: {', '.join(modes)}）")


__all__ = ["assert_tts_text", "assert_valid_mode"]
