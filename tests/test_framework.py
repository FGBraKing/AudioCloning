"""框架级架构测试 — 用 dummy 后端验证服务层与具体模型解耦（无需真实模型/GPU）。"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audioclone.core import (
    ASRService,
    AudioClassificationCapability,
    AudioClassificationService,
    AudioToAudioCapability,
    AudioToAudioService,
    ModelBackend,
    SpeechRecognitionCapability,
    TextToAudioCapability,
    TextToAudioService,
    TextToSpeechCapability,
    TTSService,
    VoiceActivityCapability,
    VoiceActivityService,
    create_asr_backend,
    create_tts_backend,
    list_backend_capabilities,
)
from audioclone.core.backends.dummy import (
    DummyASRBackend,
    DummyOmnibackend,
    DummyTTSBackend,
)
from audioclone.core.exceptions import AudioError


def _make_wav(path: Path, sr: int = 16000, seconds: float = 1.0) -> None:
    data = (np.ones(int(sr * seconds), dtype=np.float32) * 0.01).astype(np.float32)
    sf.write(str(path), data, sr)


# ─── 后端选择 / 工厂 ─────────────────────────────────────────
def test_create_dummy_backends():
    tts = create_tts_backend("dummy")
    assert tts.name == "dummy"
    assert isinstance(tts, DummyTTSBackend)

    asr = create_asr_backend("dummy")
    assert asr.name == "dummy"
    assert isinstance(asr, DummyASRBackend)


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        create_tts_backend("nope")


# ─── 服务层全流程（TTS，dummy 后端） ─────────────────────────
def test_tts_service_dummy(tmp_path, monkeypatch):
    monkeypatch.setattr("audioclone.core.service.tts_service.OUTPUT_DIR", tmp_path)

    svc = TTSService(DummyTTSBackend(sample_rate=24000))
    assert svc.sample_rate == 24000

    out = svc.synthesize_to_file(
        mode="zero_shot",
        text="请测试一下这段文本。",
        ref_audio_path="unused.wav",  # dummy 后端忽略参考音频
        filename="out.wav",
    )
    assert (tmp_path / "out.wav").exists()
    info = sf.info(out)
    assert info.duration > 0


# ─── 服务层全流程（ASR，dummy 后端） ─────────────────────────
def test_asr_service_dummy(tmp_path):
    wav = tmp_path / "input.wav"
    _make_wav(wav)

    svc = ASRService(DummyASRBackend())
    assert svc.is_model_available is True
    assert svc.is_ready() is True

    text = svc.transcribe(str(wav))
    assert text == DummyASRBackend.RESULT


def test_asr_service_rejects_bad_format(tmp_path):
    bogus = tmp_path / "input.mp4"  # 不在支持格式内
    bogus.write_bytes(b"nope")
    svc = ASRService(DummyASRBackend())
    with pytest.raises(AudioError):
        svc.transcribe(str(bogus))


# ─── 依赖方向校验：服务层不应依赖具体模型库 ───────────────────
@pytest.mark.parametrize(
    "module",
    [
        "audioclone.core.service.tts_service",
        "audioclone.core.service.asr_service",
        "audioclone.core.service.text_to_audio_service",
        "audioclone.core.service.audio_to_audio_service",
        "audioclone.core.service.audio_classification_service",
        "audioclone.core.service.vad_service",
    ],
)
def test_service_does_not_import_concrete_libs(module):
    """业务服务层源码不应引用 mlx / llama 等具体实现符号。"""
    import importlib

    src = importlib.import_module(module).__file__
    code = Path(src).read_text(encoding="utf-8")
    for forbidden in ("mlx", "llama-server", "llama_server"):
        assert forbidden not in code, f"{module} 不应出现具体模型依赖: {forbidden}"


def test_service_never_imports_backends():
    """契约：服务层不得 import 任何具体后端实现（保持依赖单向）。"""
    import importlib

    for module in [
        "audioclone.core.service.tts_service",
        "audioclone.core.service.asr_service",
    ]:
        src = importlib.import_module(module).__file__
        code = Path(src).read_text(encoding="utf-8")
        assert "core.backends.cosyvoice" not in code
        assert "core.backends.qwen3" not in code
        assert "core.backends.dummy" not in code


def test_capability_must_be_implemented():
    """契约：能力接口为抽象，未实现的方法在实例化时即失败（而非运行时）。"""
    class BrokenBackend(ModelBackend, TextToSpeechCapability):
        name = "broken"
        sample_rate = 24000
        # 未实现 synthesize → 应无法实例化

    with pytest.raises(TypeError):
        BrokenBackend()


# ─── 能力驱动：一模型多任务 ─────────────────────────────────
def test_capability_querying():
    """能力查询：supports() 反映具体后端实现了哪些能力。"""
    tts = DummyTTSBackend()
    assert tts.supports(TextToSpeechCapability)
    assert not tts.supports(SpeechRecognitionCapability)

    asr = DummyASRBackend()
    assert asr.supports(SpeechRecognitionCapability)
    assert not asr.supports(TextToSpeechCapability)


def test_multimodal_backend_aggregates_capabilities():
    """一个后端同时实现多个能力 = 一模型多任务。"""

    class DummyMultimodal(ModelBackend, TextToSpeechCapability, SpeechRecognitionCapability):
        name = "multimodal"
        sample_rate = 16000

        def synthesize(self, mode, text, ref_audio_path, ref_text="", instruct_text="", speed=1.0):
            return np.zeros(self.sample_rate, dtype=np.float32)

        def transcribe(self, audio_path: str) -> str:
            return "多模态识别"

    m = DummyMultimodal()
    assert m.supports(TextToSpeechCapability)
    assert m.supports(SpeechRecognitionCapability)
    # 一个后端可同时驱动两个任务的服务层
    tts_svc = TTSService(m)
    asr_svc = ASRService(m)
    assert tts_svc.sample_rate == 16000
    assert asr_svc.is_ready() is True


def test_capabilities_listing():
    """list_backend_capabilities 返回后端能力的人类可读名称。"""
    tts = DummyTTSBackend()
    caps = list_backend_capabilities(tts)
    assert "text-to-speech" in caps
    assert "speech-recognition" not in caps


def test_service_rejects_incapable_backend():
    """服务层拒绝不具备对应能力的后端，避免把错误模型接到任务上。"""
    with pytest.raises(AudioError):
        TTSService(DummyASRBackend())  # type: ignore[arg-type]
    with pytest.raises(AudioError):
        ASRService(DummyTTSBackend())  # type: ignore[arg-type]


# ─── 六种能力套件（全能力后端） ──────────────────────────────
ALL_CAPABILITIES = (
    TextToSpeechCapability,
    TextToAudioCapability,
    SpeechRecognitionCapability,
    AudioToAudioCapability,
    AudioClassificationCapability,
    VoiceActivityCapability,
)


def test_omnibackend_supports_all_capabilities():
    """一个后端同时支持全部六种音频任务（能力聚合）。"""
    m = DummyOmnibackend()
    for cap in ALL_CAPABILITIES:
        assert m.supports(cap), f"缺少能力: {cap}"


def test_omnibackend_capabilities_listing():
    """list_backend_capabilities 对全能力后端返回完整六项。"""
    caps = list_backend_capabilities(DummyOmnibackend())
    expected = {
        "text-to-speech",
        "text-to-audio",
        "speech-recognition",
        "audio-to-audio",
        "audio-classification",
        "voice-activity-detection",
    }
    assert set(caps) == expected


@pytest.mark.parametrize(
    ("method", "call"),
    [
        ("synthesize", lambda m: np.shape(m.synthesize("zero_shot", "你好", "r.wav", ref_text="x"))),
        ("text_to_audio", lambda m: np.shape(m.text_to_audio("雷雨声", duration_sec=0.2))),
        ("transcribe", lambda m: m.transcribe("a.wav")),
        ("transform", lambda m: np.shape(m.transform("a.wav", instruct="变男声"))),
        ("classify", lambda m: m.classify("a.wav", top_k=3)),
        ("vad", lambda m: m.vad("a.wav")),
    ],
)
def test_omnibackend_all_methods_invokable(method, call):
    """六种能力的核心方法均可调用并返回合理结果。"""
    m = DummyOmnibackend()
    assert callable(getattr(m, method))
    result = call(m)
    assert result is not None


def test_omnibackend_drives_tts_and_asr_services():
    """同一个全能力后端可同时驱动 TTS 与 ASR 两个任务的服务。"""
    m = DummyOmnibackend(sample_rate=16000)
    tts_svc = TTSService(m)  # 能力校验通过（具备 TextToSpeechCapability）
    ASRService(m)  # 能力校验通过（具备 SpeechRecognitionCapability），构造即校验
    assert tts_svc.sample_rate == 16000
    # 直接验证 ASR 识别方法（transcribe 的真实文件校验在 service 层另行覆盖）
    assert m.transcribe("unused.wav") == m.ASR_RESULT


def test_omnibackend_drives_all_six_services():
    """同一个全能力后端可驱动全部六种任务的 Service。"""
    m = DummyOmnibackend(sample_rate=16000)
    svcs = [
        TTSService(m),
        TextToAudioService(m),
        ASRService(m),
        AudioToAudioService(m),
        AudioClassificationService(m),
        VoiceActivityService(m),
    ]
    assert len(svcs) == 6


def _make_wav_file(tmp_path) -> str:
    wav = tmp_path / "input.wav"
    _make_wav(wav)
    return str(wav)


def test_t2a_service(tmp_path, monkeypatch):
    svc = TextToAudioService(DummyOmnibackend())
    assert svc.sample_rate is not None
    audio = svc.generate("雷雨声", duration_sec=0.2)
    assert audio is not None and len(audio) > 0
    with pytest.raises(AudioError):
        svc.generate("   ")


def test_a2a_service(tmp_path):
    svc = AudioToAudioService(DummyOmnibackend())
    audio = svc.transform(_make_wav_file(tmp_path), instruct="变男声")
    assert audio is not None and len(audio) > 0
    with pytest.raises(AudioError):
        svc.transform(str(tmp_path / "missing.wav"))


def test_classification_service(tmp_path):
    svc = AudioClassificationService(DummyOmnibackend())
    labels = svc.classify(_make_wav_file(tmp_path), top_k=3)
    assert labels and labels[0][0] == "speech"
    with pytest.raises(AudioError):
        svc.classify(str(tmp_path / "missing.wav"))


def test_vad_service(tmp_path):
    svc = VoiceActivityService(DummyOmnibackend())
    segments = svc.vad(_make_wav_file(tmp_path))
    assert segments == [(0.0, 1.0)]
    with pytest.raises(AudioError):
        svc.vad(str(tmp_path / "missing.wav"))


@pytest.mark.parametrize(
    ("service", "backend"),
    [
        (TextToAudioService, DummyTTSBackend),
        (AudioToAudioService, DummyASRBackend),
        (AudioClassificationService, DummyTTSBackend),
        (VoiceActivityService, DummyASRBackend),
    ],
)
def test_new_services_reject_incapable_backend(service, backend):
    """新 Service 同样拒绝不具备对应能力的后端。"""
    with pytest.raises(AudioError):
        service(backend())  # type: ignore[arg-type]


# ─── registry 按能力聚合查询 ─────────────────────────────────
def test_registry_list_backends_for():
    from audioclone.core.backends.registry import registry

    tts = set(registry.list_backends_for(TextToSpeechCapability))
    assert {
        "dummy-tts",
        "dummy-omni",
        "qwen3-tts-base-mlx",
        "qwen3-tts-customvoice-mlx",
        "qwen3-tts-voicedesign-mlx",
    } <= tts

    asr = set(registry.list_backends_for(SpeechRecognitionCapability))
    assert {"dummy-asr", "dummy-omni", "qwen3-asr-llama"} <= asr
    # Qwen3-TTS 三个变体只做 TTS，不应出现在 ASR 能力清单里
    assert not {"qwen3-tts-base-mlx", "qwen3-tts-customvoice-mlx"} & asr

    vad = registry.list_backends_for(VoiceActivityCapability)
    assert "dummy-omni" in vad
    assert not {"dummy-tts", "dummy-asr"} & set(vad)  # 纯 TTS/ASR dummy 不具备 VAD 能力


def test_dummy_alias_resolves_to_task_backends():
    """CLI/配置常用名 ``dummy`` 按任务解析到对应 dummy-tts / dummy-asr。"""
    tts = create_tts_backend("dummy")
    assert isinstance(tts, DummyTTSBackend)
    asr = create_asr_backend("dummy")
    assert isinstance(asr, DummyASRBackend)


def test_registry_create_checks_capability():
    """registry.create 在创建期按能力 fail-fast（P2）。"""
    from audioclone.core.backends.registry import registry
    from audioclone.core.exceptions import CapabilityError

    with pytest.raises(CapabilityError):
        registry.create("dummy-tts", capability=SpeechRecognitionCapability)

    backend = registry.create("dummy-tts", capability=TextToSpeechCapability)
    assert isinstance(backend, DummyTTSBackend)


# ─── RLock 串行化（并发安全） ────────────────────────────────
def test_rlock_serializes_synthesis():
    class SlowBackend(DummyTTSBackend):
        """记录最大并发合成数，验证 RLock 串行。"""

        def __init__(self):
            super().__init__(sample_rate=24000)
            self.active = 0
            self.max_active = 0
            self._guard = threading.Lock()

        def synthesize(self, *a, **k):
            with self._guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.01)
            with self._guard:
                self.active -= 1
            return super().synthesize(*a, **k)

    svc = TTSService(SlowBackend())
    errors: list[Exception] = []

    def worker() -> None:
        try:
            svc.infer_zero_shot("你好，并发测试。", "unused.wav")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert svc.backend.max_active == 1  # 串行化：无并发合成


# ─── 长文本：段间静音 + 进度回调 + 逐段落盘 ──────────────────
def test_tts_segment_gap_progress_and_keep(tmp_path, monkeypatch):
    monkeypatch.setattr("audioclone.core.service.tts_service.OUTPUT_DIR", tmp_path)
    svc = TTSService(DummyTTSBackend(sample_rate=24000))

    calls: list[tuple[int, int]] = []
    text = "这是用于分段测试的句子。" * 40  # > 300 字符，强制多段
    out = svc.synthesize_to_file(
        mode="zero_shot",
        text=text,
        ref_audio_path="unused.wav",
        filename="gap.wav",
        progress_cb=lambda i, n: calls.append((i, n)),
        keep_segments=True,
    )

    assert len(calls) >= 2  # 多段
    assert calls[-1] == (len(calls), len(calls))  # 进度覆盖到末段

    seg_dir = tmp_path / "segments"
    seg_files = sorted(seg_dir.glob("gap_seg_*.wav"))
    assert len(seg_files) == len(calls)  # 逐段落盘

    # 段间插入了静音 → 总时长大于各段时长之和
    info = sf.info(out)
    total_seg_sec = len(calls) * 0.5
    assert info.duration > total_seg_sec