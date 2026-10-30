from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

URL = "https://www.kaggle.com/api/v1/datasets/download/puneet6060/intel-image-classification"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "intel-image-classification.zip"
    if not archive.exists():
        print("Downloading official Kaggle archive...", flush=True)
        with urllib.request.urlopen(URL, timeout=180) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a ZIP archive: {archive}")
    with zipfile.ZipFile(archive) as source:
        members = [m for m in source.infolist()
                   if m.filename.startswith(("seg_train/", "seg_test/"))]
        for member in members:
            destination = (args.out / member.filename).resolve()
            if not destination.is_relative_to(args.out.resolve()):
                raise ValueError("Unsafe archive path")
            source.extract(member, args.out)
    metadata = {"url": URL, "archive_sha256": hashlib.file_digest(archive.open("rb"), "sha256").hexdigest(),
                "extracted_members": len(members), "excluded": "seg_pred"}
    (args.out / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
