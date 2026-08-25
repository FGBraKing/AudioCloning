"""ASR 端到端集成测试 — llama-server + Qwen3-ASR (需要本地模型与 llama.cpp)。"""

from __future__ import annotations

import logging

import pytest

from audioclone.config import ASSET_DIR
from audioclone.core import ASRService, create_asr_backend

logger = logging.getLogger(__name__)

TEST_AUDIO = ASSET_DIR / "zero_shot_prompt.wav"


def _make_chunk_backend(max_segment_sec: float, overlap_sec: float):
    from audioclone.core.backends.qwen3_asr_llama import Qwen3AsrLlamaBackend

    # 模型目录不存在时仅告警，不阻塞纯逻辑分片测试
    return Qwen3AsrLlamaBackend(
        model_dir="/nonexistent-model-dir",
        max_segment_sec=max_segment_sec,
        overlap_sec=overlap_sec,
    )


def test_chunk_audio_short_single():
    import numpy as np

    backend = _make_chunk_backend(max_segment_sec=60.0, overlap_sec=1.0)
    data = np.zeros(16000, dtype=np.float32)  # 1s << 60s 阈值 → 单段
    chunks = backend._chunk_audio(data)
    assert len(chunks) == 1
    assert chunks[0].shape == data.shape


def test_chunk_audio_overlap_and_tail_dedup():
    import numpy as np

    sr = 16000  # ASR_SAMPLE_RATE
    backend = _make_chunk_backend(max_segment_sec=2.0, overlap_sec=0.5)
    data = np.zeros(5 * sr, dtype=np.float32)  # 5s

    chunks = backend._chunk_audio(data)

    # 多段
    assert len(chunks) >= 2
    # 首段为完整段长（2s），步进为 1.5s → 段长 > 步进，即相邻段存在重叠
    assert len(chunks[0]) == 2 * sr
    assert len(chunks[0]) > int(1.5 * sr)
    # 覆盖完整长度（含重叠）
    assert sum(len(c) for c in chunks) >= len(data)


@pytest.mark.integration
def test_asr_roundtrip(asr_ready: bool):
    if not asr_ready:
        pytest.skip("ASR 模型文件不完整，请先运行: uv run audioclone-download-asr")
    if not TEST_AUDIO.exists():
        pytest.skip(f"测试音频不存在: {TEST_AUDIO}")

    with ASRService(create_asr_backend()) as engine:
        text = engine.transcribe(str(TEST_AUDIO))
    assert isinstance(text, str) and len(text) > 0, "ASR 识别结果为空"