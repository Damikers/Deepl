from __future__ import annotations

import csv
from pathlib import Path

from common import CLASSES, IMG_SIZE, MEAN, STD
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CONTROL = {
    "train": [(200, 1996761), (200, 2032625), (200, 2183923), (200, 1950345), (200, 1987613), (200, 1841817)],
    "validation": [(218, 4837323), (237, 5276962), (276, 6135652), (262, 5789535), (255, 5655068), (250, 5541845)],
    "test": [(219, 4837559), (237, 5291326), (277, 6157250), (263, 5849765), (255, 5668011), (251, 5547533)],
}


def source_directory(root: Path, source: str) -> Path:
    for candidate in (root / source / source, root / source):
        if all((candidate / label).is_dir() for label in CLASSES):
            return candidate
    raise FileNotFoundError(f"Expected {source}/{source}/<class> or {source}/<class> under {root}")


def create_split(root: Path, destination: Path) -> None:
    rows = []
    for source in ("seg_train", "seg_test"):
        directory = source_directory(root, source)
        rng = np.random.default_rng(0)
        total = 0
        for label in CLASSES:
            files = sorted(p.name for p in (directory / label).glob("*.jpg"))
            total += len(files)
            indices = np.arange(len(files))
            rng.shuffle(indices)
            if source == "seg_train":
                parts = (("train", indices[:200]),)
            else:
                cut = len(files) // 2
                parts = (("validation", indices[:cut]), ("test", indices[cut:]))
            for part, selected in parts:
                rows.extend({"file": files[int(i)], "label": label, "part": part} for i in selected)
        expected = 14034 if source == "seg_train" else 3000
        if total != expected:
            raise ValueError(f"{source}: {total} images, expected {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "label", "part"])
        writer.writeheader()
        writer.writerows(rows)
    validate_split(destination)


def read_split(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["file", "label", "part"]:
            raise ValueError("split.csv must have file,label,part columns")
        return list(reader)


def validate_split(path: Path) -> list[dict]:
    rows = read_split(path)
    files = [r["file"] for r in rows]
    if len(set(files)) != len(files):
        raise ValueError("Duplicate file or overlap between parts")
    if any(r["label"] not in CLASSES or r["part"] not in CONTROL or
           Path(r["file"]).name != r["file"] or not Path(r["file"]).stem.isdecimal() or
           Path(r["file"]).suffix != ".jpg" for r in rows):
        raise ValueError("Invalid row in split.csv")
    table = []
    for i, label in enumerate(CLASSES):
        for part in CONTROL:
            names = [r["file"] for r in rows if r["label"] == label and r["part"] == part]
            actual = (len(names), sum(int(Path(f).stem) for f in names))
            expected = CONTROL[part][i]
            record = {"label": label, "part": part, "count": actual[0], "sum": actual[1],
                      "expected_count": expected[0], "expected_sum": expected[1], "passed": actual == expected}
            table.append(record)
    if not all(row["passed"] for row in table):
        raise ValueError(f"Split control mismatch: {[r for r in table if not r['passed']]}")
    return table


def make_transform(training: bool) -> transforms.Compose:
    spatial = [transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
               transforms.RandomHorizontalFlip()] if training else [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    return transforms.Compose([*spatial, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class SplitDataset(Dataset):

    def __init__(self, root: Path, split: Path, part: str):
        if part not in CONTROL:
            raise ValueError(part)
        self.rows = [r for r in read_split(split) if r["part"] == part]
        self.directory = source_directory(root, "seg_train" if part == "train" else "seg_test")
        self.transform = make_transform(part == "train")
        self.labels = np.array([CLASSES.index(r["label"]) for r in self.rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def image_path(self, index: int) -> Path:
        row = self.rows[index]
        return self.directory / row["label"] / row["file"]

    def __getitem__(self, index: int):
        with Image.open(self.image_path(index)) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.labels[index])


def make_loader(dataset: SplitDataset, training: bool) -> DataLoader:

    return DataLoader(dataset, batch_size=32 if training else 64, shuffle=training,
                      drop_last=False, num_workers=0)
