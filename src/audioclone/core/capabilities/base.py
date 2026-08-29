"""能力与模型后端基类。"""

from __future__ import annotations

from typing import ClassVar

from audioclone.core.model_slot import ModelSlot, model_slots


class Capability:
    """模型能力标记接口（非抽象，作为组合/查询的载体）。

    每个任务定义一种能力。一个后端继承多个能力接口 = 「一个模型多任务」。
    """


class ModelBackend:
    """模型后端的通用基类：统一身份标识、能力查询与生命周期模板。

    能力来源二选一：
    - 显式声明 ``capabilities`` 类属性（推荐，声明式）；
    - 未声明时回退到「真实继承关系」判断（``issubclass``）。

    生命周期：子类实现 :meth:`_load` / :meth:`_close`，不要覆盖 :meth:`load` / :meth:`close`；
    槽位占用（保证运行时同类模型只有一个）由基类统一处理，``_load()`` 抛异常时自动回滚。
    """

    name: str
    capabilities: tuple[type[Capability], ...] = ()

    #: 该后端所属的槽位；``None`` 表示无状态后端（如 dummy），不参与槽位约束
    slot: ClassVar[ModelSlot | None] = None

    # 实例未初始化时回落到类属性（子类 __init__ 不必调用 super().__init__()）
    _slot_held: bool = False

    @classmethod
    def supports(cls, capability: type[Capability]) -> bool:
        """当前后端是否实现了指定能力。"""
        if cls.capabilities:
            explicit = any(issubclass(c, capability) for c in cls.capabilities)
            inherited = issubclass(cls, capability)
            return explicit or inherited
        return issubclass(cls, capability)

    # ─── 生命周期模板方法 ────────────────────────────────────
    def load(self) -> None:
        """加载模型资源并占用槽位（模板方法；子类实现 :meth:`_load`）。"""
        if self._slot_held:
            return  # 已加载：幂等
        if self.slot is not None:
            model_slots.acquire(self.slot, self.name)
            self._slot_held = True
        try:
            self._load()
        except BaseException:
            self._release_slot()
            raise

    def close(self) -> None:
        """释放模型资源并归还槽位（模板方法；子类实现 :meth:`_close`）。"""
        try:
            self._close()
        finally:
            self._release_slot()

    def _load(self) -> None:
        """子类实现真正的资源加载（默认无动作）。"""

    def _close(self) -> None:
        """子类实现资源释放（默认无动作）。"""

    def _release_slot(self) -> None:
        """归还槽位（幂等，仅在确实持有时释放）。"""
        if self._slot_held and self.slot is not None:
            model_slots.release(self.slot, self.name)
            self._slot_held = False
