from __future__ import annotations

from pathlib import Path

from common import CLASSES, RUNS, macro_f1, read_json, sha256
from data import read_split, validate_split
import numpy as np


def effect_summary(records: list[dict]) -> dict:
    values = {regime: [r["history"][-1]["validation_macro_f1"] for r in records if r["regime"] == regime]
              for regime in ("scratch", "frozen", "unfreeze")}
    means = {r: float(np.mean(v)) for r, v in values.items()}
    ranges = {r: float(np.ptp(v)) for r, v in values.items() if r != "scratch"}
    noise = max(ranges.values())
    pre = means["frozen"] - means["scratch"]
    unf = means["unfreeze"] - means["frozen"]
    return {"means": means, "ranges": ranges, "R": noise, "delta_pre": pre, "delta_unf": unf,
            "total_gain": pre + unf,
            "pre_exceeds_noise": abs(pre) > noise, "unf_exceeds_noise": abs(unf) > noise}


def select_run(records: list[dict]) -> dict:

    summary = effect_summary(records)
    regime = max(("scratch", "frozen", "unfreeze"), key=lambda r: summary["means"][r])
    candidates = sorted((r for r in records if r["regime"] == regime), key=lambda r: r["seed"])
    selected = max(candidates, key=lambda r: r["history"][-1]["validation_macro_f1"])
    return {"regime": regime, "seed": selected["seed"],
            "validation_macro_f1": selected["history"][-1]["validation_macro_f1"],
            "regime_mean": summary["means"][regime], "selection_uses": "validation_epoch8_only"}


def load_records(experiment: Path) -> list[dict]:
    split = experiment / "split.csv"
    validate_split(split)
    digest = sha256(split)
    rows = [r for r in read_split(split) if r["part"] == "validation"]
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    expected_files = np.array([r["file"] for r in rows])
    directories = {p.name for p in (experiment / "runs").iterdir() if p.is_dir()}
    if directories != {f"{regime}_seed{seed}" for regime, seed in RUNS}:
        raise ValueError("Expected exactly the seven prescribed runs")
    records = []
    for regime, seed in RUNS:
        directory = experiment / "runs" / f"{regime}_seed{seed}"
        record = read_json(directory / "run.json")
        if record["split_sha256"] != digest:
            raise ValueError(f"split.csv SHA-256 mismatch: {directory.name}; retrain all seven runs")
        if record["regime"] != regime or record["seed"] != seed:
            raise ValueError(f"Run identity mismatch: {directory.name}")
        history = record["history"]
        if len(history) != 8 or [h["epoch"] for h in history] != list(range(1, 9)) or any(
                h["optimizer_steps"] != 38 or h["images"] != 1200 or h["last_batch_size"] != 16 for h in history):
            raise ValueError("Invalid training budget")
        for key in ("train_loss", "validation_macro_f1"):
            if not np.isfinite([h[key] for h in history]).all():
                raise ValueError("Non-finite history")
        logits_path = directory / "val_logits.npy"
        logits = np.load(logits_path, allow_pickle=False)
        if logits.shape != (1498, 6) or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError(f"Invalid validation logits: {directory.name}")
        if sha256(logits_path) != record["val_logits_sha256"]:
            raise ValueError("Validation logits checksum mismatch")
        if not np.array_equal(np.load(directory / "val_files.npy", allow_pickle=False), expected_files):
            raise ValueError("Validation order differs from split.csv")
        if macro_f1(y, logits) != history[-1]["validation_macro_f1"]:
            raise ValueError("Saved validation metric differs from logits")
        records.append(record)
    return records
