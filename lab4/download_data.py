from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from data import CLASSES, calibration_selection, read_split, sha256, validate_split


PROJECT = Path(__file__).resolve().parent
URL = "https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification"
ARCHIVE_SHA256 = "3921405607c3e9d67a3459b6c629894c3a723aa1647f9310b6793da306e89337"


def source_directory(root: Path, source: str) -> Path:
    candidates = (root / source / source, root / source)
    for candidate in candidates:
        if all((candidate / label).is_dir() for label in CLASSES):
            return candidate
    for candidate in candidates:
        if any((candidate / label).is_dir() for label in CLASSES):
            return candidate
    return candidates[0]


def safe_destination(root: Path, directory: Path, row: dict[str, str]) -> Path:
    destination = (directory / row["label"] / row["file"]).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Image path leaves the data directory: {destination}")
    return destination


def download_archive(archive: Path) -> None:
    if archive.is_file():
        if sha256(archive) == ARCHIVE_SHA256:
            return
        print("Cached dataset archive has a different SHA-256; downloading a verified replacement.", flush=True)
    certificate = Path("/etc/ssl/cert.pem")
    context = ssl.create_default_context(
        cafile=str(certificate) if sys.platform == "darwin" and certificate.is_file() else None
    )
    temporary = None
    try:
        print("Downloading the official Intel Image Classification archive...", flush=True)
        request = urllib.request.Request(URL, headers={"User-Agent": "sgn-lab4/1.0"})
        with urllib.request.urlopen(request, context=context, timeout=120) as response:
            if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                raise ValueError("The dataset download redirected to a non-HTTPS address")
            digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=archive.name + ".", suffix=".part", dir=archive.parent, delete=False
            ) as target:
                temporary = Path(target.name)
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    target.write(block)
            actual = digest.hexdigest()
            if actual != ARCHIVE_SHA256:
                raise ValueError(
                    f"Dataset archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, received {actual}. "
                    "The download was not installed or extracted."
                )
            if not zipfile.is_zipfile(temporary):
                raise ValueError("The downloaded dataset is not a ZIP archive")
            os.replace(temporary, archive)
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(
            f"Could not download the Intel dataset from {URL}: {error}. "
            "Check the network connection or point the program to an existing local dataset."
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_selected(archive: Path, root: Path, selected: list[tuple[str, dict, Path]]) -> int:
    extracted = 0
    with zipfile.ZipFile(archive) as source:
        members = {}
        for info in source.infolist():
            members.setdefault(info.filename, []).append(info)
        for part, row, destination in selected:
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            candidates = (
                f"{part}/{part}/{row['label']}/{row['file']}",
                f"{part}/{row['label']}/{row['file']}",
            )
            matching = [info for name in candidates for info in members.get(name, [])]
            if len(matching) != 1:
                raise ValueError(f"Expected exactly one archive member for {row['label']}/{row['file']}")
            member = matching[0]
            if member.is_dir() or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Expected a regular image file in the archive: {member.filename}")
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Unsafe archive destination: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with source.open(member) as image, tempfile.NamedTemporaryFile(
                    mode="wb", prefix=destination.name + ".", suffix=".part",
                    dir=destination.parent, delete=False
                ) as target:
                    temporary = Path(target.name)
                    shutil.copyfileobj(image, target)
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            extracted += 1
    return extracted


def ensure_data(root: Path, split: Path) -> None:
    root = Path(root).resolve()
    rows = read_split(Path(split))
    validate_split(rows)
    calibration, _ = calibration_selection(rows)
    test = [row for row in rows if row["part"] == "test"]
    selected = []
    for part, part_rows in (("seg_train", calibration), ("seg_test", test)):
        directory = source_directory(root, part)
        selected.extend(
            (part, row, safe_destination(root, directory, row)) for row in part_rows
        )
    if len(selected) != 1702:
        raise ValueError("Expected 200 calibration images and 1502 test images")
    if all(path.is_file() and path.stat().st_size > 0 for _, _, path in selected):
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "intel-image-classification.zip"
    download_archive(archive)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Dataset archive SHA-256 verification failed before extraction")
    print("Extracting the 200 calibration and 1502 test images...", flush=True)
    extracted = extract_selected(archive, root, selected)
    if not all(path.is_file() and path.stat().st_size > 0 for _, _, path in selected):
        raise ValueError("Some required images are missing after dataset extraction")
    metadata = {
        "url": URL,
        "archive_sha256": ARCHIVE_SHA256,
        "selected_images": len(selected),
        "calibration_images": len(calibration),
        "test_images": len(test),
        "extracted_members": extracted,
        "excluded": ["seg_pred", "validation", "unused_train"],
    }
    (root / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the LR4 calibration and test images.")
    parser.add_argument("--out", type=Path, default=PROJECT / "data")
    parser.add_argument("--split", type=Path, default=PROJECT / "split.csv")
    arguments = parser.parse_args()
    ensure_data(arguments.out, arguments.split)


if __name__ == "__main__":
    main()
