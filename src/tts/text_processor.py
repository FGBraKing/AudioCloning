"""
文本处理模块 — 长文本分段、归一化
"""

import re
import logging
from typing import List

from ..config import MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)

# 句末标点（用于分段切割点）
SENTENCE_ENDS = r"[。！？.!?；;\n]"


def split_long_text(text: str, max_len: int = MAX_TEXT_LENGTH) -> List[str]:
    """
    将长文本按句子边界分段，每段不超过 max_len 字符。

    优先在句末标点处切割，避免截断句子。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    segments = []
    remaining = text

    while remaining:
        if len(remaining) <= max_len:
            segments.append(remaining)
            break

        # 在 max_len 范围内找最后一个句末标点
        chunk = remaining[:max_len]
        last_end = -1
        for m in re.finditer(SENTENCE_ENDS, chunk):
            last_end = m.end()

        if last_end > 0:
            segments.append(remaining[:last_end])
            remaining = remaining[last_end:].lstrip()
        else:
            # 找不到句末标点，尝试在逗号、空格处切割
            comma_pos = chunk.rfind("，")
            space_pos = chunk.rfind(",")
            split_pos = max(comma_pos, space_pos)
            if split_pos > max_len // 2:
                segments.append(remaining[:split_pos + 1])
                remaining = remaining[split_pos + 1:].lstrip()
            else:
                segments.append(chunk)
                remaining = remaining[max_len:].lstrip()

    logger.debug(f"文本分段: {len(segments)} 段, 原文 {len(text)} 字")
    return segments


def normalize_text(text: str) -> str:
    """
    基础文本归一化：去除多余空白。
    """
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text
