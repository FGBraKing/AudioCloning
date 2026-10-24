"""model_manager 单元测试 — 状态查询与主动下载（全部离线，stub 下载实现）。"""

from __future__ import annotations

import pytest

from audioclone.core import model_manager
from audioclone.utils import model_path


# ─── model_status：离线状态查询 ──────────────────────────────
def test_status_dummy_backend_not_downloadable():
    status = model_manager.model_status("tts", "dummy")
    assert status["downloadable"] is False
    assert status["cached"] is None


def test_status_unknown_backend_raises():
    with pytest.raises(KeyError):
        model_manager.model_status("tts", "nope")
    with pytest.raises(KeyError):
        model_manager.model_status("wat", "dummy")


def test_status_real_backend_reports_cache_miss(monkeypatch):
    """未缓存时 cached=False，且不触发任何网络（find_cached_model 已离线）。"""
    monkeypatch.setattr(model_manager, "find_cached_model", lambda model_id: None)
    monkeypatch.setattr(model_manager, "find_modelscope_model", lambda model_id: None)
    status = model_manager.model_status("asr")
    assert status["backend"] == "qwen3-asr-mlx"  # mac 默认 MLX ASR
    assert status["downloadable"] is True
    assert status["cached"] is False


def test_status_asr_requires_gguf_in_snapshot(monkeypatch, tmp_path):
    """ASR 快照缺 .gguf 时视为未就绪。"""
    monkeypatch.setattr(model_manager, "find_cached_model", lambda model_id: tmp_path)
    assert model_manager.model_status("asr")["cached"] is False
    (tmp_path / "model-q8_0.gguf").touch()
    assert model_manager.model_status("asr")["cached"] is True


# ─── download_model：主动下载（stub 真实下载器） ─────────────
def test_download_skipped_for_dummy():
    result = model_manager.download_model("tts", "dummy")
    assert result["status"] == "skipped"


def test_download_hf_into_system_cache(monkeypatch, tmp_path):
    """默认来源 hf：不传 local_dir → 进入（stub 的）HF 系统缓存路径。"""
    captured = {}

    def fake_snapshot(model_id, local_dir=None, **kwargs):
        captured["model_id"] = model_id
        captured["local_dir"] = local_dir
        return str(tmp_path / "hf_snapshot")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    result = model_manager.download_model("tts", "qwen3-tts-base-mlx")
    assert result["status"] == "completed"
    assert captured["model_id"].startswith("mlx-community/Qwen3-TTS")
    assert captured["local_dir"] is None
    assert result["path"] == str(tmp_path / "hf_snapshot")


def test_download_hf_with_local_dir(monkeypatch, tmp_path):
    import huggingface_hub

    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda model_id, local_dir=None, **kw: seen.update(
            model_id=model_id, local_dir=local_dir
        ) or str(local_dir),
    )
    target = tmp_path / "custom"
    result = model_manager.download_model("asr", local_dir=str(target))
    assert result["status"] == "completed"
    assert seen["local_dir"] == str(target)
    assert target.exists()


def test_download_modelscope_falls_back_to_pretrained(monkeypatch, tmp_path):
    """ModelScope 来源未给 local_dir → 落 pretrained_models/<模型名>。"""
    import modelscope

    seen = {}
    monkeypatch.setattr(
        modelscope, "snapshot_download",
        lambda model_id, local_dir=None: seen.update(model_id=model_id, local_dir=local_dir),
    )
    monkeypatch.setattr(model_manager, "PROJECT_ROOT", tmp_path)
    result = model_manager.download_model("asr", source="modelscope")
    assert result["status"] == "completed"
    assert result["source"] == "modelscope"
    assert seen["local_dir"] == str(tmp_path / "pretrained_models" / "Qwen3-ASR-0.6B-8bit")


def test_download_auto_falls_back_to_modelscope_on_hf_failure(monkeypatch, tmp_path):
    """source=auto（默认）：HF 失败自动回退 ModelScope，结果标记 fallback。"""
    import huggingface_hub
    import modelscope

    def hf_boom(*args, **kwargs):
        raise ConnectionError("hf unreachable")

    ms_seen = {}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", hf_boom)
    monkeypatch.setattr(
        modelscope, "snapshot_download",
        lambda model_id, local_dir=None: ms_seen.update(model_id=model_id, local_dir=local_dir),
    )
    monkeypatch.setattr(model_manager, "PROJECT_ROOT", tmp_path)
    result = model_manager.download_model("asr")  # 默认 auto
    assert result["status"] == "completed"
    assert result["source"] == "modelscope"
    assert result["fallback"] is True
    assert ms_seen["local_dir"] == str(tmp_path / "pretrained_models" / "Qwen3-ASR-0.6B-8bit")


def test_download_hf_forced_source_does_not_fallback(monkeypatch):
    """source=hf 强制指定：失败原样上抛，不回退。"""
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("hf down")),
    )
    with pytest.raises(ConnectionError):
        model_manager.download_model("asr", source="hf")


def test_download_unknown_source_raises():
    with pytest.raises(ValueError):
        model_manager.download_model("tts", "dummy", source="ftp")


# ─── HuggingFace 镜像端点 ────────────────────────────────────
def test_hf_endpoint_precedence(monkeypatch):
    """Settings（AUDIOCLONE_HF_ENDPOINT）> 标准环境变量 HF_ENDPOINT > None。"""
    from audioclone.utils import model_path

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setattr(model_path.settings, "hf_endpoint", None)
    assert model_path.hf_endpoint() is None

    monkeypatch.setenv("HF_ENDPOINT", "https://env-mirror.example")
    assert model_path.hf_endpoint() == "https://env-mirror.example"

    monkeypatch.setattr(model_path.settings, "hf_endpoint", "https://settings-mirror.example")
    assert model_path.hf_endpoint() == "https://settings-mirror.example"  # Settings 优先


def test_download_passes_custom_endpoint(monkeypatch, tmp_path):
    """自定义镜像端点透传给 huggingface_hub.snapshot_download。"""
    import huggingface_hub

    from audioclone.utils import model_path

    monkeypatch.setattr(model_path.settings, "hf_endpoint", "https://hf-mirror.com")
    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda model_id, local_dir=None, endpoint=None: seen.update(
            model_id=model_id, endpoint=endpoint
        ) or str(tmp_path),
    )
    result = model_manager.download_model("tts", "qwen3-tts-base-mlx")
    assert result["status"] == "completed"
    assert seen["endpoint"] == "https://hf-mirror.com"


def test_download_endpoint_none_by_default(monkeypatch, tmp_path):
    """未配置镜像时不传端点（endpoint=None，走官方源）。"""
    import huggingface_hub

    from audioclone.utils import model_path

    monkeypatch.setattr(model_path.settings, "hf_endpoint", None)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda model_id, local_dir=None, endpoint=None: seen.update(endpoint=endpoint)
        or str(tmp_path),
    )
    model_manager.download_model("tts", "qwen3-tts-base-mlx")
    assert seen["endpoint"] is None


# ─── 加载路径：ModelScope 本地目录回退查找 ───────────────────
def test_resolve_model_dir_finds_modelscope_dir(monkeypatch, tmp_path):
    """HF 缓存未命中时，回退查找 pretrained_models/<模型名>（ModelScope 落盘位置）。"""
    import audioclone.config as config_mod
    from audioclone.utils import model_path

    ms_dir = tmp_path / "pretrained_models" / "Some-Model"
    ms_dir.mkdir(parents=True)
    monkeypatch.setattr(model_path, "find_cached_model", lambda model_id: None)
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    # resolve_model_dir 内部 `from audioclone.config import PROJECT_ROOT` 取 config 模块属性
    assert model_path.resolve_model_dir("org/Some-Model", None) == ms_dir


# ─── resolve_model_dir：未缓存不下载、报可读错误 ─────────────
def test_resolve_model_dir_does_not_download(monkeypatch):
    """未缓存时直接报错并给下载指引，绝不触发网络下载。"""
    monkeypatch.setattr(model_path, "find_cached_model", lambda model_id: None)
    from audioclone.core.exceptions import ModelLoadError

    with pytest.raises(ModelLoadError) as ei:
        model_path.resolve_model_dir("some/model", None)
    msg = str(ei.value)
    assert "未下载" in msg
    assert "audioclone-download" in msg  # CLI 指引
    assert "/api/models/download" in msg  # API 指引


def test_find_cached_model_returns_none_on_error(monkeypatch):
    import huggingface_hub

    def boom(*args, **kwargs):
        raise FileNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    assert model_path.find_cached_model("some/model") is None


def test_find_cached_model_hits_cache(monkeypatch, tmp_path):
    import huggingface_hub

    seen = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda model_id, **kw: seen.update(model_id=model_id, kw=kw)
        or str(tmp_path),
    )
    assert model_path.find_cached_model("some/model") == tmp_path
    assert seen["kw"].get("local_files_only") is True  # 离线查找，不触网
