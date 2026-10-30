from __future__ import annotations

import argparse
from pathlib import Path
import time

from common import (CLASSES, IMG_SIZE, MEAN, STD, RUNS, choose_device, code_provenance,
                    environment, macro_f1, sha256, synchronize, write_json)
from checks import run_checks
from data import SplitDataset, create_split, make_loader, make_transform
from models import build_model, describe_model, predict, set_train_mode
from protocol import load_records, select_run
import numpy as np
import torch


def settings(model, optimizer, regime: str) -> dict:
    return {
        "classes": list(CLASSES), "img_size": IMG_SIZE, "dtype": "float32",
        "architecture": "torchvision.models.resnet18; fc=Linear(512,6,bias=True)",
        "initialization_order": ["torch.manual_seed(seed)", "np.random.seed(seed)",
                                 "resnet18(weights=W)", "fc=Linear(512,6)",
                                 "requires_grad", "to(device,float32)", "AdamW"],
        "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
        "linear_initialization": "torch.nn.Linear default reset_parameters",
        "resnet_defaults": {"zero_init_residual": False, "groups": 1, "width_per_group": 64,
                            "replace_stride_with_dilation": [False, False, False], "norm_layer": "BatchNorm2d"},
        "batchnorm": {"momentum": 0.1, "eps": 1e-5, "affine": True, "track_running_stats": True},
        "transforms": {"train_repr": repr(make_transform(True)), "eval_repr": repr(make_transform(False)),
                       "open": "Pillow.Image.open(...).convert('RGB')", "mean": MEAN, "std": STD,
                       "random_resized_crop": {"size": [160, 160], "scale": [0.7, 1.0],
                                               "ratio": [0.75, 4 / 3], "interpolation": "BILINEAR", "antialias": True},
                       "random_horizontal_flip_p": 0.5,
                       "eval_resize": {"size": [160, 160], "interpolation": "BILINEAR", "max_size": None, "antialias": True},
                       "to_tensor": "uint8 [0,255] -> float32 [0,1]", "normalize_inplace": False},
        "loaders": {"train": {"batch_size": 32, "shuffle": True, "drop_last": False, "num_workers": 0},
                    "validation_test": {"batch_size": 64, "shuffle": False, "drop_last": False, "num_workers": 0},
                    "generator": None, "sampler": "default", "pin_memory": False, "persistent_workers": False,
                    "timeout": 0, "worker_init_fn": None, "prefetch_factor": None, "in_order": True},
        "epochs": 8, "steps_per_epoch": 38, "total_optimizer_steps": 304,
        "criterion": {"name": "CrossEntropyLoss", "weight": None, "ignore_index": -100,
                      "reduction": "mean", "label_smoothing": 0.0},
        "optimizer": {"name": "AdamW", "defaults": optimizer.defaults,
                      "groups": describe_model(model, optimizer, regime)["optimizer_groups"]},
        "zero_grad_set_to_none": True, "validation_metric": "macro-F1; labels=0..5; zero_division=0",
        "epoch_loss": "sum(batch_mean_loss * actual_batch_size) / 1200; before optimizer.step",
        "scheduler": None, "gradient_clipping": None, "mixed_precision": False,
        "class_weighting": False, "early_stopping": False, "checkpoint_epoch": 8,
        "test_time_augmentation": False, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def train_run(root: Path, experiment: Path, regime: str, seed: int, device: torch.device, provenance: dict) -> None:
    directory = experiment / "runs" / f"{regime}_seed{seed}"
    directory.mkdir(parents=True)
    model, optimizer = build_model(regime, seed, device)
    train_dataset = SplitDataset(root, experiment / "split.csv", "train")
    val_dataset = SplitDataset(root, experiment / "split.csv", "validation")
    train_loader = make_loader(train_dataset, True)
    val_loader = make_loader(val_dataset, False)
    record = {"regime": regime, "seed": seed, "split_sha256": sha256(experiment / "split.csv"),
              **provenance, "environment": environment(device), "settings": settings(model, optimizer, regime),
              "history": [], "timing_scope": "8 training epochs plus validation after every epoch; excludes setup/checks/serialization"}
    criterion = torch.nn.CrossEntropyLoss()
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, 9):
        set_train_mode(model, regime)
        loss_sum = 0.0
        seen = steps = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss_sum += float(loss.detach().cpu()) * len(y)
            loss.backward()
            optimizer.step()
            seen += len(y)
            steps += 1
            last_batch_size = len(y)
        if (seen, steps, last_batch_size) != (1200, 38, 16):
            raise AssertionError("Incorrect epoch budget")
        logits = predict(model, val_loader, device)
        metrics = {"epoch": epoch, "train_loss": loss_sum / seen,
                   "validation_macro_f1": macro_f1(val_dataset.labels, logits),
                   "optimizer_steps": steps, "images": seen, "last_batch_size": last_batch_size}
        record["history"].append(metrics)
        print(f"{regime:8} seed={seed} epoch={epoch}/8 loss={metrics['train_loss']:.6f} "
              f"val_F1={metrics['validation_macro_f1']:.6f}", flush=True)
    synchronize(device)
    record["training_seconds"] = time.perf_counter() - started
    np.save(directory / "val_logits.npy", logits, allow_pickle=False)
    np.save(directory / "val_files.npy", np.array([r["file"] for r in val_dataset.rows]), allow_pickle=False)
    record["val_logits_sha256"] = sha256(directory / "val_logits.npy")
    checkpoints = experiment / ".checkpoints"
    checkpoints.mkdir(exist_ok=True)
    checkpoint_path = checkpoints / f"{regime}_seed{seed}.pt"
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "regime": regime, "seed": seed, "classes": list(CLASSES), "img_size": IMG_SIZE}
    torch.save(checkpoint, checkpoint_path)
    record["checkpoint_sha256"] = sha256(checkpoint_path)
    write_json(directory / "run.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Intel Image Classification root")
    parser.add_argument("--out", type=Path, default=Path("."), help="New/empty experiment directory")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true", help="Continue only missing complete protocol runs")
    args = parser.parse_args()
    experiment = args.out.resolve()
    experiment.mkdir(parents=True, exist_ok=True)
    existing = experiment / "split.csv"
    if existing.exists() and not args.resume:
        raise FileExistsError("Experiment exists; use a new --out or --resume")
    if args.resume:
        if not existing.exists():
            raise FileNotFoundError("No existing split to resume")

        candidate = experiment / ".split_candidate.csv"
        create_split(args.data, candidate)
        same = candidate.read_bytes() == existing.read_bytes()
        candidate.unlink()
        if not same:
            raise ValueError("Dataset differs from the existing experiment")
    else:
        create_split(args.data, existing)
    device = choose_device(args.device)
    print(f"Training device: {device}", flush=True)
    provenance = code_provenance()

    run_checks(args.data, existing, experiment / "checks", device)
    for regime, seed in RUNS:
        directory = experiment / "runs" / f"{regime}_seed{seed}"
        if directory.exists():
            if not args.resume or not (directory / "run.json").exists():
                raise FileExistsError(f"Incomplete/existing run: {directory}; move it aside before retrying")
            from common import read_json
            saved = read_json(directory / "run.json")
            checkpoint = experiment / ".checkpoints" / f"{regime}_seed{seed}.pt"
            if saved["split_sha256"] != sha256(existing) or not checkpoint.exists() or sha256(checkpoint) != saved["checkpoint_sha256"]:
                raise ValueError("Cannot resume: split or local checkpoint mismatch")
            print(f"Preserving completed run {directory.name}", flush=True)
            continue
        train_run(args.data, experiment, regime, seed, device, provenance)
    records = load_records(experiment)
    selected = select_run(records)
    checkpoint = experiment / ".checkpoints" / f"{selected['regime']}_seed{selected['seed']}.pt"
    destination = experiment / "lab3_model.pt"
    destination.write_bytes(checkpoint.read_bytes())
    write_json(experiment / "selection.json", {**selected, "model_sha256": sha256(destination),
                                              "split_sha256": sha256(existing)})
    print(f"Selected by validation: {selected}", flush=True)
    print("Training complete. Test has not been loaded.", flush=True)


if __name__ == "__main__":
    main()
