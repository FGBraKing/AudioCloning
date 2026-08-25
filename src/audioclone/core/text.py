"""文本处理模块 — 分段器接口、长文本分段、归一化。"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from audioclone.config import MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)

# 句末标点（用于分段切割点）
SENTENCE_ENDS = r"[。！？.!?；;\n]"
# 逗号/空格等次级切割点
SECONDARY_BREAKS = "，,"


class TextSegmenter(ABC):
    """长文本分段器接口。

    便于接入不同分句策略（标点 / NLP 分句 / 情感粒度等）。
    """

    @abstractmethod
    def split(self, text: str, max_len: int) -> list[str]:
        """将文本切分为不超过 ``max_len`` 的若干段。"""


class SentenceTextSegmenter(TextSegmenter):
    """基于句末标点的默认分段器。"""

    def split(self, text: str, max_len: int) -> list[str]:
        return split_long_text(text, max_len=max_len)


def split_long_text(text: str, max_len: int = MAX_TEXT_LENGTH) -> list[str]:
    """将长文本按句子边界分段，每段不超过 ``max_len`` 字符。

    优先在句末标点处切割，避免截断句子；无标点时退回到逗号/空格切割。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    segments: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_len:
            segments.append(remaining)
            break

        # 在 max_len 范围内找最后一个句末标点
        chunk = remaining[:max_len]
        matches = list(re.finditer(SENTENCE_ENDS, chunk))
        last_end = matches[-1].end() if matches else -1

        if last_end > 0:
            segments.append(remaining[:last_end])
            remaining = remaining[last_end:].lstrip()
        else:
            # 找不到句末标点，尝试在逗号、空格处切割
            split_pos = max(chunk.rfind(c) for c in SECONDARY_BREAKS)
            if split_pos > max_len // 2:
                segments.append(remaining[: split_pos + 1])
                remaining = remaining[split_pos + 1 :].lstrip()
            else:
                segments.append(chunk)
                remaining = remaining[max_len:].lstrip()

    logger.debug("文本分段: %d 段, 原文 %d 字", len(segments), len(text))
    return segments


def normalize_text(text: str) -> str:
    """基础文本归一化：去除多余空白。"""
    if not text:
        return ""
    return re.sub(r"[ \t]+", " ", text).strip()