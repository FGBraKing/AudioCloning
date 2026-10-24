"""Qwen3-TTS (TensorRT Edge-LLM) 后端单元测试 — 假引擎目录 + 假推理工具，离线运行。

本机无 NVIDIA 硬件：推理工具用可执行 shell 脚本模拟（读取 input.json → 产出 WAV），
验证加载校验、请求构造与子进程调用契约。
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioclone.core import model_slots
from audioclone.core.backends.qwen3_tts_tensorrt import (
    Qwen3TTSBaseTensorRTBackend,
    Qwen3TTSCustomVoiceTensorRTBackend,
    Qwen3TTSVoiceDesignTensorRTBackend,
)
from audioclone.core.exceptions import InvalidAudioInput, ModelLoadError
from audioclone.core.model_slot import ModelSlot


@pytest.fixture(autouse=True)
def _clean_slots():
    model_slots.release_all()
    yield
    model_slots.release_all()


@pytest.fixture
def fake_engine_dir(tmp_path) -> Path:
    """假 Edge-LLM 构建产物目录（非空即通过加载校验）。"""
    engine = tmp_path / "engines"
    engine.mkdir()
    (engine / "talker.engine").touch()
    return engine


@pytest.fixture
def fake_inference_bin(tmp_path) -> Path:
    """假推理工具：读 input.json → 在 output_dir 产出两段 WAV（验证拼接路径）。"""
    work = tmp_path / "bin"
    work.mkdir()
    script = work / "qwen3_tts_inference"
    script.write_text(
        "#!/bin/sh\n"
        '# 用法: qwen3_tts_inference --input_json X --output_dir Y\n'
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = "--input_json" ]; then IN="$2"; fi\n'
        '  if [ "$1" = "--output_dir" ]; then OUT="$2"; fi\n'
        "  shift\n"
        "done\n"
        'python3 -c "\n'
        "import json, sys\n"
        "import numpy as np\n"
        "import soundfile as sf\n"
        "req = json.load(open(sys.argv[1]))\n"
        "sf.write(sys.argv[2] + '/seg1.wav', np.ones(24000, dtype=np.float32), 24000)\n"
        "sf.write(sys.argv[2] + '/seg2.wav', np.ones(12000, dtype=np.float32), 24000)\n"
        '" "$IN" "$OUT"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


# ─── 身份与注册 ──────────────────────────────────────────────
def test_identity_and_modes():
    base = Qwen3TTSBaseTensorRTBackend()
    assert base.name == "qwen3-tts-base-tensorrt"
    assert base.variant == "base"
    assert base.slot is ModelSlot.TTS
    assert base.supported_modes == ("zero_shot", "cross_lingual")
    assert Qwen3TTSCustomVoiceTensorRTBackend().supported_modes == ("instruct",)
    assert Qwen3TTSVoiceDesignTensorRTBackend().supported_modes == ("instruct",)


def test_registered():
    from audioclone.core.backends.registry import registry

    assert {
        "qwen3-tts-base-tensorrt",
        "qwen3-tts-customvoice-tensorrt",
        "qwen3-tts-voicedesign-tensorrt",
    } <= set(registry.names())


# ─── 加载校验（不下载、不构建、可读报错） ────────────────────
def test_load_requires_engine_dir():
    with pytest.raises(ModelLoadError) as ei:
        Qwen3TTSBaseTensorRTBackend().load()
    msg = str(ei.value)
    assert "引擎目录" in msg
    assert "tensorrt-edgellm-export" in msg  # 给出可操作工作流指引
    assert model_slots.active(ModelSlot.TTS) is None


def test_load_rejects_empty_engine_dir(fake_engine_dir):
    for f in fake_engine_dir.iterdir():
        f.unlink()
    with pytest.raises(ModelLoadError) as ei:
        Qwen3TTSBaseTensorRTBackend(model_dir=fake_engine_dir).load()
    assert "为空" in str(ei.value)


def test_load_requires_inference_bin(fake_engine_dir, monkeypatch):
    monkeypatch.setattr(
        "audioclone.core.backends.qwen3_tts_tensorrt._find_inference_bin",
        lambda _bin: None,
    )
    with pytest.raises(ModelLoadError) as ei:
        Qwen3TTSBaseTensorRTBackend(model_dir=fake_engine_dir).load()
    assert "推理工具" in str(ei.value)


# ─── 合成：请求构造 + 子进程契约 ─────────────────────────────
def test_base_synthesize_builds_clone_request(
    fake_engine_dir, fake_inference_bin, monkeypatch, tmp_path
):
    """Base 变体：请求含 ref_audio/ref_text；两段输出拼接为单数组。"""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        request = json.loads(Path(cmd[cmd.index("--input_json") + 1]).read_text())
        assert request["tts_model_type"] == "base"
        assert request["ref_audio"] == "ref.wav"
        assert request["ref_text"] == "参考文本"
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        sf.write(out_dir / "seg.wav", np.ones(100, dtype=np.float32), 24000)
        return subprocess_result(0)

    monkeypatch.setattr(
        "audioclone.core.backends.qwen3_tts_tensorrt.subprocess.run", fake_run
    )
    monkeypatch.setattr(
        "audioclone.core.backends.qwen3_tts_tensorrt.sf.read",
        lambda path, dtype: (np.ones(100, dtype=np.float32), 24000),
    )
    backend = Qwen3TTSBaseTensorRTBackend(
        model_dir=fake_engine_dir, inference_bin=str(fake_inference_bin)
    )
    audio = backend.synthesize("zero_shot", "你好", "ref.wav", ref_text="参考文本")

    assert audio.dtype == np.float32
    assert "--input_json" in " ".join(captured["cmd"])
    assert model_slots.active(ModelSlot.TTS) == "qwen3-tts-base-tensorrt"


def test_base_cross_lingual_drops_ref_text(fake_engine_dir, fake_inference_bin, monkeypatch):
    """cross_lingual：ref_text 置 None（timbre-only，与 MLX 侧语义一致）。"""
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["request"] = json.loads(Path(cmd[cmd.index("--input_json") + 1]).read_text())
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        sf.write(out_dir / "seg.wav", np.ones(10, dtype=np.float32), 24000)
        return subprocess_result(0)

    monkeypatch.setattr("audioclone.core.backends.qwen3_tts_tensorrt.subprocess.run", fake_run)
    monkeypatch.setattr(
        "audioclone.core.backends.qwen3_tts_tensorrt.sf.read",
        lambda path, dtype: (np.ones(10, dtype=np.float32), 24000),
    )
    backend = Qwen3TTSBaseTensorRTBackend(
        model_dir=fake_engine_dir, inference_bin=str(fake_inference_bin)
    )
    backend.synthesize("cross_lingual", "Hello", "ref.wav", ref_text="应被忽略")
    assert seen["request"]["ref_text"] is None


def test_voice_design_requires_instruct(fake_engine_dir, fake_inference_bin):
    backend = Qwen3TTSVoiceDesignTensorRTBackend(
        model_dir=fake_engine_dir, inference_bin=str(fake_inference_bin)
    )
    with pytest.raises(InvalidAudioInput) as ei:
        backend.synthesize("instruct", "你好", instruct_text="   ")
    assert "instruct_text" in str(ei.value)


def test_base_rejects_instruct_mode(fake_engine_dir, fake_inference_bin):
    backend = Qwen3TTSBaseTensorRTBackend(
        model_dir=fake_engine_dir, inference_bin=str(fake_inference_bin)
    )
    with pytest.raises(InvalidAudioInput) as ei:
        backend.synthesize("instruct", "你好")
    assert "不支持推理模式" in str(ei.value)


def test_inference_failure_raises_readable_error(fake_engine_dir, fake_inference_bin, monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess_result(1, stderr="CUDA error out of memory")

    monkeypatch.setattr("audioclone.core.backends.qwen3_tts_tensorrt.subprocess.run", fake_run)
    backend = Qwen3TTSBaseTensorRTBackend(
        model_dir=fake_engine_dir, inference_bin=str(fake_inference_bin)
    )
    backend.load()
    with pytest.raises(ModelLoadError) as ei:
        backend.synthesize("zero_shot", "你好", "ref.wav")
    assert "CUDA error" in str(ei.value)


def test_no_wav_output_raises(fake_engine_dir, fake_inference_bin, monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess_result(0)

    monkeypatch.setattr("audioclone.core.backends.qwen3_tts_tensorrt.subprocess.run", fake_run)
    backend = Qwen3TTSBaseTensorRTBackend(
        model_dir=fake_engine_dir, inference_bin=str(fake_inference_bin)
    )
    backend.load()
    with pytest.raises(ModelLoadError) as ei:
        backend.synthesize("zero_shot", "你好", "ref.wav")
    assert "未产出 WAV" in str(ei.value)


# ─── 真实子进程端到端（假工具脚本） ──────────────────────────
def test_end_to_end_with_fake_tool(fake_engine_dir, fake_inference_bin, tmp_path):
    """真实子进程：假工具读取 input.json 产出 WAV → 后端拼接返回。"""
    backend = Qwen3TTSBaseTensorRTBackend(
        model_dir=fake_engine_dir, inference_bin=str(fake_inference_bin)
    )
    audio = backend.synthesize("zero_shot", "你好世界", "ref.wav", ref_text="参考文本")
    assert len(audio) == 36000  # 两段 24000 + 12000 拼接


# ─── 状态 ────────────────────────────────────────────────────
def test_get_status(fake_engine_dir):
    status = Qwen3TTSBaseTensorRTBackend(model_dir=fake_engine_dir).get_status()
    assert status["engine_dir_ready"] is True
    assert status["model_loaded"] is False
    assert status["model_id"].startswith("mlx-community/Qwen3-TTS")


def subprocess_result(code: int, stderr: str = ""):
    import subprocess

    out = subprocess.CompletedProcess(args=[], returncode=code)
    out.stdout = ""
    out.stderr = stderr
    return out
