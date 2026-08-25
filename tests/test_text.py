"""纯逻辑单元测试（无需模型/GPU）。"""

from __future__ import annotations

import pytest

from audioclone.core.text import normalize_text, split_long_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" 你好  世界 ", "你好 世界"),
        ("", ""),
        ("A  B  C", "A B C"),
    ],
)
def test_normalize_text(text, expected):
    assert normalize_text(text) == expected


def test_split_short_text():
    assert split_long_text("你好，这是一段短文本。", max_len=300) == ["你好，这是一段短文本。"]


def test_split_empty_text():
    assert split_long_text("   ") == []


def test_split_long_text_by_sentence():
    text = "这是第一句话。" + "这是第二句话！" + "这是第三句话？"
    segments = split_long_text(text, max_len=10)
    assert segments == ["这是第一句话。", "这是第二句话！", "这是第三句话？"]


def test_split_long_text_fallback_comma():
    # 无句末标点时按逗号切割
    text = "短段，" * 20
    segments = split_long_text(text, max_len=6)
    assert all(len(seg) <= 6 for seg in segments[:-1]) or len(segments) > 1
    assert "".join(segments) == text


def test_split_preserves_contents():
    text = "句子A。" + "句子B。" + "句子C。"
    segments = split_long_text(text, max_len=5)
    assert "".join(segments) == text