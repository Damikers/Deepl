from __future__ import annotations

import csv
import hashlib
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SPLIT_CONTROL = {"train": (1200, 11993084), "test": (1502, 33351444)}
CALIBRATION_COUNTS = dict(zip(CLASSES, (40, 35, 36, 33, 29, 27), strict=True))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path) -> tuple[torch.nn.Module, dict]:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"state_dict", "regime", "seed", "classes", "img_size"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Model checkpoint must contain state_dict, regime, seed, classes, img_size")
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("Checkpoint class order does not match LR3")
    if checkpoint["img_size"] != IMG_SIZE:
        raise ValueError(f"Expected img_size={IMG_SIZE}, received {checkpoint['img_size']}")
    if checkpoint["regime"] not in ("scratch", "frozen", "unfreeze"):
        raise ValueError("Unknown LR3 model regime")
    if not isinstance(checkpoint["seed"], int):
        raise ValueError("Checkpoint seed must be an integer")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    metadata = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "regime": checkpoint["regime"],
        "seed": checkpoint["seed"],
        "classes": list(CLASSES),
        "img_size": int(checkpoint["img_size"]),
    }
    return model, metadata


def read_split(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["file", "label", "part"]:
            raise ValueError("split.csv must have file,label,part columns in this order")
        rows = list(reader)
    for row in rows:
        filename = row["file"]
        if (set(row) != {"file", "label", "part"}
                or not isinstance(filename, str)
                or Path(filename).name != filename
                or not Path(filename).stem.isdecimal()
                or Path(filename).suffix != ".jpg"
                or row["label"] not in CLASSES
                or row["part"] not in ("train", "validation", "test")):
            raise ValueError(f"Invalid split row: {row}")
    names = [row["file"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate filename or overlap between split parts")
    return rows


def summarize(rows: list[dict[str, str]]) -> dict:
    counts = Counter(row["label"] for row in rows)
    return {
        "count": len(rows),
        "filename_sum": sum(int(Path(row["file"]).stem) for row in rows),
        "class_counts": {label: counts[label] for label in CLASSES},
    }


def validate_split(rows: list[dict[str, str]]) -> dict:
    controls = {}
    for part, expected in SPLIT_CONTROL.items():
        actual = summarize([row for row in rows if row["part"] == part])
        if (actual["count"], actual["filename_sum"]) != expected:
            raise ValueError(f"{part} control mismatch: {actual}; expected {expected}")
        controls[part] = actual
    return controls


def calibration_selection(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    train = sorted((row for row in rows if row["part"] == "train"), key=lambda row: row["file"])
    if len(train) != 1200:
        raise ValueError("Calibration selection requires exactly 1200 train rows")
    indices = np.random.default_rng(0).permutation(1200)[:200]
    selected = [train[int(index)] for index in indices]
    control = summarize(selected)
    if (control["count"] != 200 or control["filename_sum"] != 1921470
            or control["class_counts"] != CALIBRATION_COUNTS):
        raise ValueError(f"Calibration control mismatch: {control}")
    control.update({
        "selection": "train sorted by filename strings, default_rng(0).permutation(1200)[:200]",
        "indices": indices.tolist(),
        "rows": [dict(row) for row in selected],
    })
    return selected, control


def source_directory(root: Path, source: str) -> Path:
    root = Path(root)
    for candidate in (root / source / source, root / source):
        if all((candidate / label).is_dir() for label in CLASSES):
            return candidate
    raise FileNotFoundError(f"Expected {source}/{source}/<class> or {source}/<class> under {root}")


def make_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ImageDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, str]], img_size: int = IMG_SIZE):
        if not rows or len({row["part"] for row in rows}) != 1:
            raise ValueError("An image dataset must contain one nonempty split part")
        part = rows[0]["part"]
        if part not in ("train", "test"):
            raise ValueError("LR4 uses only train calibration and test images")
        self.rows = [dict(row) for row in rows]
        self.directory = source_directory(root, "seg_train" if part == "train" else "seg_test")
        self.labels = np.array([CLASSES.index(row["label"]) for row in rows], dtype=np.int64)
        self.transform = make_transform(img_size)
        for index in range(len(self.rows)):
            if not self.image_path(index).is_file():
                raise FileNotFoundError(self.image_path(index))

    def __len__(self) -> int:
        return len(self.rows)

    def image_path(self, index: int) -> Path:
        row = self.rows[index]
        return self.directory / row["label"] / row["file"]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_path(index)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.labels[index])


def prepare_data(
    root: Path, split: Path, img_size: int = IMG_SIZE
) -> tuple[ImageDataset, ImageDataset, dict]:
    split = Path(split)
    rows = read_split(split)
    controls = validate_split(rows)
    calibration_rows, calibration_control = calibration_selection(rows)
    test_rows = [row for row in rows if row["part"] == "test"]
    test = ImageDataset(root, test_rows, img_size)
    calibration = ImageDataset(root, calibration_rows, img_size)
    metadata = {
        "split_path": str(split.resolve()),
        "split_sha256": sha256(split),
        "data_root": str(Path(root).resolve()),
        "split_control": controls,
        "calibration": calibration_control,
        "test_order": "test rows in split.csv order",
        "transform": {
            "resize": [img_size, img_size],
            "interpolation": "bilinear",
            "antialias": True,
            "rgb": True,
            "mean": list(MEAN),
            "std": list(STD),
            "dtype": "float32",
        },
    }
    return test, calibration, metadata


def make_test_loader(dataset: ImageDataset) -> DataLoader:
    return DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, num_workers=0)


def calibration_batches(dataset: ImageDataset) -> Iterator[np.ndarray]:
    for index in range(len(dataset)):
        tensor, _ = dataset[index]
        yield np.ascontiguousarray(tensor.unsqueeze(0).numpy(), dtype=np.float32)
