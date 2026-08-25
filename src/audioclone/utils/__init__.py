"""公共工具模块。"""

from __future__ import annotations

from audioclone.utils.audio import (
    concat_segments,
    get_audio_duration,
    save_audio,
    validate_ref_audio,
)

__all__ = ["concat_segments", "get_audio_duration", "save_audio", "validate_ref_audio"]