"""模型路径解析 — HuggingFace 系统缓存查找，可选自定义本地路径。

默认策略：模型权重统一存放于 HF 系统缓存（``~/.cache/huggingface/hub``，可用
``HF_HOME`` / ``HF_HUB_CACHE`` 重定向）。**本模块只做离线查找，绝不触发下载**——
未缓存时抛出带下载指引的 :class:`ModelLoadError`，由用户主动下载（CLI 命令或
``POST /api/models/download``）。显式给定本地目录（CLI ``--model_dir`` /
env ``AUDIOCLONE_*_MODEL_DIR``）时优先使用。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from audioclone.config import settings
from audioclone.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


def hf_endpoint() -> str | None:
    """生效的 HuggingFace 端点：Settings（``AUDIOCLONE_HF_ENDPOINT``）优先，
    其次标准 ``HF_ENDPOINT`` 环境变量；均未设置返回 None（官方源）。"""
    return settings.hf_endpoint or os.environ.get("HF_ENDPOINT") or None


def find_cached_model(model_id: str) -> Path | None:
    """在 HF 系统缓存中查找模型快照（``local_files_only``，不访问网络）。

    命中返回快照目录；未缓存 / huggingface_hub 不可用返回 None。
    """
    try:
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 - 未缓存(LocalEntryNotFoundError)等一律视为未命中
        return None
    return Path(snapshot)


def find_modelscope_model(model_id: str) -> Path | None:
    """查找 ModelScope 来源的本地模型目录（``pretrained_models/<模型名>``）。

    与 ``model_manager.download_model`` 的 ModelScope 默认落盘位置对应，
    使"HF 失败回退 ModelScope 下载"的模型在默认加载路径下也能被找到。
    """
    from audioclone.config import PROJECT_ROOT

    path = PROJECT_ROOT / "pretrained_models" / model_id.split("/")[-1]
    return path if path.exists() else None


def _github_local_dir(spec: dict) -> Path:
    """GitHub Release 分发模型的默认本地目录：``pretrained_models/<资产主名>/``。"""
    from audioclone.config import PROJECT_ROOT

    gh = spec["github_release"]
    stem = Path(gh["asset"]).stem  # gtcrn_simple.onnx -> gtcrn_simple
    return Path(spec["model_dir"]).expanduser() if spec.get("model_dir") else (
        PROJECT_ROOT / "pretrained_models" / stem
    )


def find_github_model(spec: dict) -> Path | None:
    """查找 GitHub Release 分发的本地模型目录（存在即返回）。"""
    if "github_release" not in spec:
        return None
    path = _github_local_dir(spec)
    return path if path.exists() and any(path.iterdir()) else None


def download_github_release_asset(spec: dict, local_dir: str | Path | None = None) -> Path:
    """从 GitHub Release 下载单个资产文件（httpx 流式），返回模型目录。

    资产 URL：``https://github.com/<repo>/releases/download/<tag>/<asset>``；
    落盘 ``pretrained_models/<资产主名>/<asset>``（或 spec/参数指定的 model_dir）。
    """
    import httpx

    gh = spec["github_release"]
    url = f"https://github.com/{gh['repo']}/releases/download/{gh['tag']}/{gh['asset']}"
    base = _github_local_dir(spec)
    local_dir_path = Path(local_dir).expanduser() if local_dir else base
    local_dir = local_dir_path
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / gh["asset"]

    logger.info("正在从 GitHub 下载 %s ...", url)
    try:
        with httpx.stream("GET", url, timeout=300, follow_redirects=True) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(1 << 20):
                    f.write(chunk)
    except httpx.HTTPError as e:
        raise ModelLoadError(
            f"从 GitHub 下载 {gh['asset']} 失败: {e}\n"
            "  - 检查网络或配置代理；也可手动下载后放入 " + str(local_dir)
        ) from e
    logger.info("下载完成: %s", dest)
    return local_dir


def resolve_model_dir(model_id: str, model_dir: str | Path | None) -> Path:
    """解析模型目录：显式路径优先 → HF 系统缓存 → ModelScope 本地目录（不下载）。

    - ``model_dir`` 非空 → 直接使用该本地目录（不存在则报错，不猜路径）；
    - 否则依次查找 HF 缓存与 ModelScope 默认目录；均未命中抛 :class:`ModelLoadError`，
      提示主动下载。

    :raises ModelLoadError: 自定义目录不存在，或模型未下载。
    """
    if model_dir:
        path = Path(model_dir).expanduser()
        if not path.exists():
            raise ModelLoadError(f"自定义模型目录不存在: {path}")
        logger.info("使用自定义模型目录: %s", path)
        return path

    if not model_id:
        raise ModelLoadError("未指定 model_id 且未提供自定义 model_dir，无法定位模型")

    cached = find_cached_model(model_id)
    if cached is not None:
        logger.info("模型已缓存: %s (%s)", model_id, cached)
        return cached

    ms_dir = find_modelscope_model(model_id)
    if ms_dir is not None:
        logger.info("使用 ModelScope 本地目录: %s (%s)", model_id, ms_dir)
        return ms_dir

    endpoint = hf_endpoint()
    mirror_hint = f"  - 镜像: 当前端点 {endpoint}\n" if endpoint else (
        "  - 镜像: export AUDIOCLONE_HF_ENDPOINT=https://hf-mirror.com\n"
    )
    raise ModelLoadError(
        f"模型 {model_id} 未下载（HuggingFace 缓存与本地 pretrained_models 均不存在）。"
        "请主动下载后再启动：\n"
        "  - CLI: uv run audioclone-download-tts / audioclone-download-asr\n"
        '  - API: POST /api/models/download {"task": "tts" | "asr"}\n'
        "  - 或用本地目录启动: --model_dir / --asr_model_dir 指定含权重文件的目录\n"
        f"{mirror_hint}"
        "  - 下载源默认 HF，失败自动回退 ModelScope；也可 --source modelscope 强制"
    )
