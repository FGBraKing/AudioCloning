"""模型下载管理 — CLI 命令与 REST API 共用的主动下载 / 状态查询入口。

加载路径（``utils/model_path.py``）只做离线查找、不下载；下载必须经本模块**主动**
触发：``uv run audioclone-download-tts/asr``（CLI）或 ``POST /api/models/download``（API）。

下载源策略：默认 **HuggingFace**（支持 ``AUDIOCLONE_HF_ENDPOINT`` / ``HF_ENDPOINT``
自定义镜像）；``source="auto"``（默认）时 HF 失败**自动回退 ModelScope**（落
``pretrained_models/<模型名>``，加载路径会回退查找该目录，无需改启动参数）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audioclone.config import PROJECT_ROOT
from audioclone.core.service.model_loader import KIND_BACKENDS, resolve_backend_name
from audioclone.utils.model_path import (
    download_github_release_asset,
    find_cached_model,
    find_github_model,
    find_modelscope_model,
    hf_endpoint,
)

logger = logging.getLogger(__name__)

#: 任务类型 → 后端覆盖参数表（单源 = model_loader.KIND_BACKENDS）
TASK_BACKENDS: dict[str, dict[str, dict[str, Any]]] = KIND_BACKENDS

#: 合法下载来源：auto=按后端分发方式自动选择；hf/modelscope/github 强制指定
DOWNLOAD_SOURCES = ("auto", "hf", "modelscope", "github")


def _model_spec(task: str, backend: str | None) -> tuple[str, dict[str, Any]]:
    """解析任务/后端 → (后端名, 覆盖参数)。未知任务或后端抛 KeyError；auto 按环境解析。"""
    if task not in TASK_BACKENDS:
        raise KeyError(f"未知任务类型: {task}（可选: tts / asr）")
    backends = TASK_BACKENDS[task]
    name = resolve_backend_name(task, backend)
    if name not in backends:
        raise KeyError(f"未知的 {task} 后端: {name}（可用: {', '.join(backends)}）")
    return name, backends[name]


def model_status(task: str, backend: str | None = None) -> dict[str, Any]:
    """查询某任务后端的模型就绪状态（离线，不访问网络）。

    返回 ``task / backend / model_id / downloadable / cached / model_path / hf_endpoint``；
    ``cached`` 仅对可下载后端有意义（ASR 额外要求快照内含 ``.gguf``）。
    """
    name, spec = _model_spec(task, backend)
    model_id = spec.get("model_id")
    status: dict[str, Any] = {
        "task": task,
        "backend": name,
        "model_id": model_id,
        "downloadable": model_id is not None or spec.get("github_release") is not None,
        "cached": None,
        "model_path": spec.get("model_dir"),
        "hf_endpoint": hf_endpoint(),
    }
    if model_id:
        cached = find_cached_model(model_id)
        if cached is not None and task == "asr" and not any(cached.glob("*.gguf")):
            cached = None  # 快照不完整（缺 GGUF）视为未就绪
        if cached is None:
            cached = find_modelscope_model(model_id)
            if cached is not None and task == "asr" and not any(cached.glob("*.gguf")):
                cached = None
        status["cached"] = cached is not None
        if cached is not None:
            status["model_path"] = str(cached)
    elif spec.get("github_release"):
        # GitHub Release 分发（如 GTCRN 增强模型）
        local = find_github_model(spec)
        status["cached"] = local is not None
        if local is not None:
            status["model_path"] = str(local)
    return status


def _download_hf(model_id: str, local_dir: str | None) -> Path:
    """从 HuggingFace 下载（支持自定义镜像端点）；默认进系统缓存。"""
    from huggingface_hub import snapshot_download

    endpoint = hf_endpoint()
    if local_dir:
        path = Path(local_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "正在从 HuggingFace%s 下载 %s 到自定义目录 %s ...",
            f"（镜像 {endpoint}）" if endpoint else "", model_id, path,
        )
        snapshot_download(model_id, local_dir=str(path), endpoint=endpoint)
    else:
        logger.info(
            "正在从 HuggingFace%s 下载 %s 到系统缓存 (~/.cache/huggingface/hub) ...",
            f"（镜像 {endpoint}）" if endpoint else "", model_id,
        )
        path = Path(snapshot_download(model_id, endpoint=endpoint))
    return path


def _download_modelscope(model_id: str, local_dir: str | None) -> Path:
    """从 ModelScope 下载，落 ``pretrained_models/<模型名>`` 或自定义目录。"""
    from modelscope import snapshot_download

    path = (
        Path(local_dir).expanduser()
        if local_dir
        else PROJECT_ROOT / "pretrained_models" / model_id.split("/")[-1]
    )
    path.mkdir(parents=True, exist_ok=True)
    logger.info("正在从 ModelScope 下载 %s 到 %s ...", model_id, path)
    snapshot_download(model_id, local_dir=str(path))
    return path


def download_model(
    task: str, backend: str | None = None, local_dir: str | None = None, source: str = "auto"
) -> dict[str, Any]:
    """主动下载指定任务后端的模型权重（阻塞直至完成，调用方负责放后台线程）。

    - ``source="auto"``（默认）：按后端分发方式自动选择——HF 仓库走 HuggingFace
      （支持自定义镜像，失败自动回退 ModelScope）；GitHub Release 分发的后端直接走
      GitHub。结果 ``source`` 字段为实际生效来源，``fallback=True`` 标记发生过回退；
    - ``source="hf"`` / ``"modelscope"``：强制指定来源，失败原样上抛不回退。

    :raises KeyError: 未知任务/后端；:raises ValueError: 未知来源。
    :raises Exception: 下载失败（网络/仓库不存在），原样上抛供调用方记录。
    """
    name, spec = _model_spec(task, backend)
    if source not in DOWNLOAD_SOURCES:
        raise ValueError(f"未知下载来源: {source}（可选: {' / '.join(DOWNLOAD_SOURCES)}）")
    model_id = spec.get("model_id")
    gh = spec.get("github_release")

    if not model_id and not gh:
        logger.info("后端 (%s) 无需下载模型", name)
        return {
            "task": task, "backend": name, "model_id": None,
            "source": source, "path": None, "status": "skipped",
        }

    if gh:  # GitHub Release 分发（单资产文件）
        if source in ("hf", "modelscope"):
            raise ValueError(
                f"后端 {name} 经 GitHub Release 分发（{gh['asset']}），"
                "不支持 --source hf/modelscope；请使用默认来源（auto）或 --source github"
            )
        eff_spec = {**spec, "model_dir": local_dir} if local_dir else spec
        path = download_github_release_asset(eff_spec).parent
        logger.info("下载完成（来源 github）: %s", path)
        return {
            "task": task, "backend": name, "model_id": None,
            "source": "github", "fallback": False, "path": str(path),
            "status": "completed",
        }

    assert model_id is not None  # gh 分支已排除（model_id 与 github_release 二选一）
    fallback = False
    if source in ("auto", "hf"):
        try:
            path = _download_hf(model_id, local_dir)
            used = "hf"
        except Exception as e:  # noqa: BLE001
            if source == "hf":
                raise
            fallback = True
            logger.warning(
                "HuggingFace 下载 %s 失败（%s），自动回退 ModelScope ...", model_id, e
            )
            path = _download_modelscope(model_id, local_dir)
            used = "modelscope"
    else:
        path = _download_modelscope(model_id, local_dir)
        used = "modelscope"

    logger.info("下载完成（来源 %s%s）: %s", used, "，经 ModelScope 回退" if fallback else "", path)
    return {
        "task": task, "backend": name, "model_id": model_id,
        "source": used, "fallback": fallback, "path": str(path), "status": "completed",
    }
