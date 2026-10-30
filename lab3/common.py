from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


PROJECT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(PROJECT / ".cache" / "torch"))

import numpy as np
import torch
import torchvision
import sklearn
import PIL

CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
RUNS = (("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict | list) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def code_provenance() -> dict:

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                            capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT,
                            capture_output=True, text=True)
    files = sorted([*PROJECT.glob("*.py"), PROJECT / "pyproject.toml", PROJECT / "uv.lock"])
    hashes = {p.name: sha256(p) for p in files if p.is_file()}
    source_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "git_status": "available" if commit.returncode == 0 else "not_a_git_repository",
            "source_sha256": source_hash, "source_files_sha256": hashes}


def environment(device: torch.device) -> dict:
    name = str(device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        name = (result.stdout.strip() or platform.machine()) + " / Metal Performance Shaders"
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "pillow": PIL.__version__,
            "platform": platform.platform(), "device": str(device), "device_name": name,
            "torch_num_threads": torch.get_num_threads()}


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; use --device cpu or allow access to the Mac GPU")
    return torch.device(requested)


def macro_f1(y: np.ndarray, logits: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, logits.argmax(axis=1), labels=range(6), average="macro", zero_division=0))
