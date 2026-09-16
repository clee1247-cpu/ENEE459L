from __future__ import annotations

import json
import re
from typing import Any
import sys

from env import Env, ModuleNotAvailable, getattr_path, read_text, unknown, major_minor

# The NVIDIA-built PyTorch wheels for Jetson carry a local version segment —
# the part after "+" — that names the NVIDIA container release. A wheel from
# plain PyPI has no such segment. This is a hint, not a proof, which is why the
# probe reports the tag itself alongside the interpretation.
_NV_LOCAL_TAG = re.compile(r"(?:^|\.)nv\d", re.IGNORECASE)

# `# R36 (release), REVISION: 5.0, GCID: ...`
_L4T_RELEASE = re.compile(r"R(\d+)\s*\(release\)", re.IGNORECASE)
_L4T_REVISION = re.compile(r"REVISION:\s*([\d.]+)")

# hepler function
def _split_local_version(raw: str) -> dict[str, Any]:
    if not raw:
        return {"raw": raw, "public": None, "local": None, "nvidia_build": False}
    public, sep, local = raw.partition("+")
    local = local if sep else None
    return {
        "raw": raw,
        "public": public or None,
        "local": local,
        "nvidia_build": bool(local and _NV_LOCAL_TAG.search(local)),
    }

# ---------------------------------------------------------------------------
# The probes.
# ---------------------------------------------------------------------------

def probe_torch(env: Env) -> dict[str, Any]:
    src = "import torch"
    try:
        torch = env.importer("torch")
    except ModuleNotAvailable as exc:
        return unknown(src, f"torch is not installed: {exc}")

    raw_version = str(getattr_path(torch, "__version__", ""))
    version = _split_local_version(raw_version)
    cuda = getattr_path(torch, "cuda")
    is_available = getattr_path(cuda, "is_available")
    if not callable(is_available):
        return unknown(src, "torch.cuda.is_available is unavailable")

    try:
        cuda_available = bool(is_available())
    except Exception as exc:
        return unknown(src, f"torch CUDA availability check failed: {exc}")

    cuda_version = getattr_path(getattr_path(torch, "version"), "cuda")
    device_name = None
    if cuda_available:
        get_device_name = getattr_path(cuda, "get_device_name")
        if not callable(get_device_name):
            return unknown(src, "torch.cuda.get_device_name is unavailable")
        try:
            device_name = str(get_device_name(0))
        except Exception as exc:
            return unknown(src, f"torch CUDA device name check failed: {exc}")

    if cuda_available:
        diagnosis = "torch is installed and sees the GPU"
    elif not version["local"]:
        diagnosis = "torch is installed but appears to be a stock PyPI wheel without CUDA support"
    else:
        diagnosis = "torch is installed but CUDA is unavailable"

    return {
        "value": raw_version or None,
        "source": src,
        "status": "ok",
        "version": version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "device_name": device_name,
        "diagnosis": diagnosis,
    }


def probe_cuda(env: Env) -> dict[str, Any]:
    src = "/usr/local/cuda/version.json"
    raw = read_text(env.root, src)
    if not raw:
        return unknown(src, "CUDA toolkit manifest is missing or unreadable")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        return unknown(src, f"CUDA toolkit manifest is not valid JSON: {exc.msg}")

    version = getattr_path(getattr_path(manifest, "cuda"), "version")
    if not isinstance(version, str) or not version:
        try:
            version = manifest["cuda"]["version"]
        except (KeyError, TypeError):
            return unknown(src, "CUDA version field is missing from the manifest")
    line = major_minor(version)
    if line is None:
        return unknown(src, "CUDA version has an unexpected format")
    return {"value": version, "source": src, "status": "ok", "line": line}


def probe_opencv(env: Env) -> dict[str, Any]:
    src = "import cv2"
    try:
        cv2 = env.importer("cv2")
    except ModuleNotAvailable as exc:
        return unknown(src, f"OpenCV is not installed: {exc}")

    raw_version = getattr_path(cv2, "__version__")
    cuda = getattr_path(cv2, "cuda")
    count_devices = getattr_path(cuda, "getCudaEnabledDeviceCount")
    if not isinstance(raw_version, str) or not raw_version:
        return unknown(src, "OpenCV version is unavailable")
    if not callable(count_devices):
        return unknown(src, "cv2.cuda.getCudaEnabledDeviceCount is unavailable")
    try:
        cuda_devices = int(count_devices())
    except Exception as exc:
        return unknown(src, f"OpenCV CUDA device check failed: {exc}")

    cuda_enabled = cuda_devices > 0
    detail = (
        "OpenCV reports CUDA-enabled devices"
        if cuda_enabled
        else "the cv2.cuda namespace exists but reports no devices — this is a non-CUDA build"
    )
    return {
        "value": raw_version,
        "source": src,
        "status": "ok",
        "cuda_devices": cuda_devices,
        "cuda_enabled": cuda_enabled,
        "detail": detail,
    }


def probe_tensorrt(env: Env) -> dict[str, Any]:
    src = "import tensorrt"
    try:
        tensorrt = env.importer("tensorrt")
    except ModuleNotAvailable as exc:
        if env.python.prefix != env.python.base_prefix:
            detail = "TensorRT is unavailable; this virtual environment may not include system site packages"
        else:
            detail = f"TensorRT is not installed: {exc}"
        return unknown(src, detail)

    raw_version = getattr_path(tensorrt, "__version__")
    if not isinstance(raw_version, str) or not raw_version:
        return unknown(src, "TensorRT version is unavailable")
    line = major_minor(raw_version)
    if line is None:
        return unknown(src, "TensorRT version has an unexpected format")
    return {"value": raw_version, "source": src, "status": "ok", "line": line}


def probe_l4t(env: Env) -> dict[str, Any]:
    src = "/etc/nv_tegra_release"
    raw = read_text(env.root, src)
    if not raw:
        return unknown(src, "L4T release file is missing or unreadable")
    release = _L4T_RELEASE.search(raw)
    revision = _L4T_REVISION.search(raw)
    if not release or not revision:
        return unknown(src, "L4T release or revision could not be parsed")

    value = f"{release.group(1)}.{revision.group(1)}"
    line = major_minor(value)
    if line is None:
        return unknown(src, "L4T version has an unexpected format")
    return {
        "value": value,
        "source": src,
        "status": "ok",
        "line": line,
        "raw": raw,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Env.real()
    # out = probe_l4t(env)
#     print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Env.real()

    # testing probes
    report = {
        "probe_torch": probe_torch(env),
        "probe_cuda": probe_cuda(env),
        "probe_opencv": probe_opencv(env),
        "probe_tensorrt": probe_tensorrt(env),
        "probe_l4t": probe_l4t(env),
    }
    
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

