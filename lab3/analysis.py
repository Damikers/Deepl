from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import CLASSES, IMG_SIZE, environment, macro_f1, read_json, sha256, write_json
from data import SplitDataset, make_loader
from protocol import effect_summary, load_records, select_run
from models import predict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torchvision.models import resnet18


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(records: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    colors = {"scratch": "#333333", "frozen": "#147d92", "unfreeze": "#c16322"}
    styles = {1: "-", 2: "--", 3: ":"}
    for record in records:
        label = f"{record['regime']} / seed {record['seed']}"
        for ax, field in zip(axes, ("train_loss", "validation_macro_f1")):
            ax.plot(range(1, 9), [row[field] for row in record["history"]],
                    label=label, color=colors[record["regime"]], linestyle=styles[record["seed"]],
                    linewidth=1.8, marker="o", markersize=3)
    for ax, title, ylabel in zip(axes, ("Training loss", "Validation macro-F1"), ("Cross-entropy", "Macro-F1")):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel, xticks=range(1, 9))
        ax.grid(alpha=0.2)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_confusion(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    shown = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="True class", title="Test confusion matrix (CPU)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(shown, ax=ax, shrink=0.8, label="Images")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def error_examples(files: list[str], y: np.ndarray, logits: np.ndarray) -> list[dict]:

    prediction = logits.argmax(axis=1)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    matrix = confusion_matrix(y, prediction, labels=range(6))
    pairs = sorted(((i, j) for i in range(6) for j in range(6) if i != j and matrix[i, j] > 0),
                   key=lambda pair: (-int(matrix[pair]), pair[0], pair[1]))
    groups = {pair: sorted(np.flatnonzero((y == pair[0]) & (prediction == pair[1])),
                           key=lambda i: files[int(i)]) for pair in pairs}
    selected = []
    depth = 0
    target = min(5, int(np.sum(y != prediction)))
    while len(selected) < target:
        for pair in pairs:
            if depth < len(groups[pair]):
                index = int(groups[pair][depth])
                selected.append({"file": files[index], "true_class": CLASSES[pair[0]],
                                 "predicted_class": CLASSES[pair[1]],
                                 "predicted_probability": float(probabilities[index, pair[1]]),
                                 "pair_count": int(matrix[pair]), "test_index": index})
                if len(selected) == target:
                    break
        depth += 1
    return selected


def plot_errors(dataset: SplitDataset, examples: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, max(1, len(examples)), figsize=(16, 4), layout="constrained", squeeze=False)
    if not examples:
        axes[0, 0].text(0.5, 0.5, "No test errors", ha="center")
        axes[0, 0].axis("off")
    for ax, example in zip(axes.flat, examples):
        with Image.open(dataset.image_path(example["test_index"])) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{example['file']}\n{example['true_class']} → {example['predicted_class']}\n"
                     f"p={example['predicted_probability']:.3f}; pair count={example['pair_count']}", fontsize=10)
        ax.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(root: Path, experiment: Path, out: Path) -> dict:
    experiment, out = experiment.resolve(), out.resolve()
    protected = [experiment / name for name in ("runs", "checks", "figures", ".checkpoints", "data", ".venv", ".cache")]
    if out == experiment or any(out == p or out.is_relative_to(p) for p in protected):
        raise ValueError("Analysis output must be separate from source artifacts")

    records = load_records(experiment)
    effects = effect_summary(records)
    selected = select_run(records)
    saved_selection = read_json(experiment / "selection.json")
    if any(saved_selection.get(k) != v for k, v in selected.items()):
        raise ValueError("selection.json violates the validation selection rule")
    checkpoint_path = experiment / "lab3_model.pt"
    selected_record = next(r for r in records if (r["regime"], r["seed"]) == (selected["regime"], selected["seed"]))
    model_digest = sha256(checkpoint_path)
    if (model_digest != selected_record["checkpoint_sha256"] or model_digest != saved_selection["model_sha256"] or
        saved_selection["split_sha256"] != sha256(experiment / "split.csv")):
        raise ValueError("Selected model checkpoint/hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint["regime"] != selected["regime"] or checkpoint["seed"] != selected["seed"] or
        checkpoint["classes"] != list(CLASSES) or checkpoint["img_size"] != IMG_SIZE):
        raise ValueError("Invalid checkpoint metadata")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device="cpu", dtype=torch.float32)
    out.mkdir(parents=True, exist_ok=True)
    table = [{"run": i, "regime": r["regime"], "seed": r["seed"],
              "train_loss_epoch8": r["history"][-1]["train_loss"],
              "validation_macro_f1_epoch8": r["history"][-1]["validation_macro_f1"],
              "training_seconds": r["training_seconds"]} for i, r in enumerate(records, 1)]
    write_csv(out / "runs_summary.csv", table)
    plot_curves(records, out / "curves.png")
    print("All split hashes, validation logits, selection and model checks PASSED", flush=True)
    print(f"Selected: {selected}", flush=True)
    print(f"Effects: {effects}", flush=True)
    print("Evaluating only the selected model on test, on CPU...", flush=True)
    dataset = SplitDataset(root, experiment / "split.csv", "test")
    logits = predict(model, make_loader(dataset, False), torch.device("cpu"))
    prediction = logits.argmax(axis=1)
    matrix = confusion_matrix(dataset.labels, prediction, labels=range(6))
    accuracy = float(accuracy_score(dataset.labels, prediction))
    test_f1 = macro_f1(dataset.labels, logits)
    examples = error_examples([r["file"] for r in dataset.rows], dataset.labels, logits)
    np.save(out / "test_logits.npy", logits, allow_pickle=False)
    np.save(out / "test_files.npy", np.array([r["file"] for r in dataset.rows]), allow_pickle=False)
    write_csv(out / "test_confusion.csv", [{"true_class": label, **dict(zip(CLASSES, map(int, matrix[i])))}
                                          for i, label in enumerate(CLASSES)])
    write_csv(out / "errors.csv", examples, ["file", "true_class", "predicted_class", "predicted_probability", "pair_count", "test_index"])
    plot_confusion(matrix, out / "confusion.png")
    plot_errors(dataset, examples, out / "errors.png")
    result = {"split_sha256": sha256(experiment / "split.csv"), "model_sha256": model_digest,
              "runs": table, "effects": effects, "selected": selected,
              "test": {"macro_f1": test_f1, "accuracy": accuracy, "n": len(dataset),
                       "correct": int(np.trace(matrix)), "errors": int(matrix.sum() - np.trace(matrix)),
                       "accuracy_standard_error": float(np.sqrt(accuracy * (1 - accuracy) / len(dataset))),
                       "test_minus_validation_macro_f1": test_f1 - selected["validation_macro_f1"],
                       "confusion_matrix": matrix.tolist(), "classes": list(CLASSES)},
              "error_examples": examples, "evaluation_environment": environment(torch.device("cpu"))}
    write_json(out / "metrics.json", result)
    print(f"Test macro-F1={test_f1:.12f}; accuracy={accuracy:.12f}; n={len(dataset)}", flush=True)
    print(matrix, flush=True)
    for example in examples:
        print(example, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("reproduced"))
    args = parser.parse_args()
    analyze(args.data, args.experiment, args.out)


if __name__ == "__main__":
    main()
