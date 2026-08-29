"""WebUI 展示层异常。

与界面框架解耦：``handlers`` 只抛本模块定义的异常，由展示层（NiceGUI）决定如何
呈现（当前实现为 ``ui.notify`` 顶部通知）。这样业务逻辑不依赖任何 UI 框架，
便于单测与后续替换前端。
"""

from __future__ import annotations


class UIError(Exception):
    """可直接展示给终端用户的界面错误（语义等价于原 Gradio 的 ``gr.Error``）。

    领域异常（``AudioError`` 及其子类）会被 ``ui_error`` 装饰器转换为本异常，
    其消息原样展示，不带前缀。
    """


__all__ = ["UIError"]
