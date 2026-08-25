"""能力与模型后端基类。"""

from __future__ import annotations


class Capability:
    """模型能力标记接口（非抽象，作为组合/查询的载体）。

    每个任务定义一种能力。一个后端继承多个能力接口 = 「一个模型多任务」。
    """


class ModelBackend:
    """模型后端的通用基类：统一身份标识与能力查询。

    能力来源二选一：
    - 显式声明 ``capabilities`` 类属性（推荐，声明式）；
    - 未声明时回退到「真实继承关系」判断（``issubclass``）。
    """

    name: str
    capabilities: tuple[type[Capability], ...] = ()

    @classmethod
    def supports(cls, capability: type[Capability]) -> bool:
        """当前后端是否实现了指定能力。"""
        if cls.capabilities:
            explicit = any(issubclass(c, capability) for c in cls.capabilities)
            inherited = issubclass(cls, capability)
            return explicit or inherited
        return issubclass(cls, capability)

    def load(self) -> None:
        """加载模型资源（惰性，可覆盖）。"""
        ...

    def close(self) -> None:
        """释放模型资源（可覆盖）。"""
        ...