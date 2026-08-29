"""模型槽位守卫 — 保证运行时每种任务最多只有一个已加载的模型。

设计动机
--------
MLX 模型常驻统一内存，两个同任务模型同时驻留会直接翻倍内存占用并争用 Metal 上下文。
本项目始终「按需加载一个 TTS 模型 + 一个 ASR 模型」，:class:`ModelSlotRegistry`
把这条口头约定变成可执行的运行期约束。

职责边界
--------
- 本模块只登记/查询/释放槽位，不感知模型实现、不做任何 IO。
- 占槽与释放的时机统一由 :class:`~audioclone.core.capabilities.base.ModelBackend`
  的 ``load()`` / ``close()`` 模板方法驱动；后端只需实现 ``_load()`` / ``_close()``。
"""

from __future__ import annotations

import logging
import threading
from enum import StrEnum

from audioclone.core.exceptions import ModelSlotError

logger = logging.getLogger(__name__)


class ModelSlot(StrEnum):
    """模型槽位：同一槽位在任意时刻只允许一个已加载的后端。"""

    TTS = "tts"
    ASR = "asr"


class ModelSlotRegistry:
    """槽位占用登记表（线程安全，进程内单例由 :data:`model_slots` 持有）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._holders: dict[ModelSlot, str] = {}

    # ─── 占用 / 释放 ─────────────────────────────────────────
    def acquire(self, slot: ModelSlot, backend_name: str) -> None:
        """登记 ``backend_name`` 占用 ``slot``。

        同一后端重复调用是幂等的（允许 ``load()`` 被多次调用）；
        但槽位已被**另一个**后端占用时抛 :class:`ModelSlotError`。
        """
        with self._lock:
            holder = self._holders.get(slot)
            if holder is not None and holder != backend_name:
                raise ModelSlotError(
                    f"{slot.value.upper()} 槽位已被后端 {holder} 占用，无法同时加载 {backend_name}"
                    f"（运行时仅允许一个 {slot.value.upper()} 模型，请先 close() 原后端）"
                )
            if holder is None:
                self._holders[slot] = backend_name
                logger.info("[ModelSlot] %s 槽位由 %s 占用", slot.value, backend_name)

    def release(self, slot: ModelSlot, backend_name: str | None = None) -> None:
        """释放槽位；``backend_name`` 非空时仅在该后端持有者名下才释放。"""
        with self._lock:
            holder = self._holders.get(slot)
            if holder is None:
                return
            if backend_name is not None and holder != backend_name:
                return
            del self._holders[slot]
            logger.info("[ModelSlot] %s 槽位已释放（%s）", slot.value, holder)

    def release_all(self) -> None:
        """清空全部槽位（进程退出 / 测试夹具复位）。"""
        with self._lock:
            if self._holders:
                logger.info("[ModelSlot] 释放全部槽位: %s", dict(self._holders))
            self._holders.clear()

    # ─── 查询 ────────────────────────────────────────────────
    def active(self, slot: ModelSlot) -> str | None:
        """返回占用该槽位的后端名（未占用返回 None）。"""
        with self._lock:
            return self._holders.get(slot)

    def snapshot(self) -> dict[str, str | None]:
        """返回全部槽位快照，便于健康检查/状态面板展示。"""
        with self._lock:
            return {slot.value: self._holders.get(slot) for slot in ModelSlot}


#: 进程内单例
model_slots = ModelSlotRegistry()

__all__ = ["ModelSlot", "ModelSlotRegistry", "model_slots"]
