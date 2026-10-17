from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".uv-cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".uv-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from checks import run_checks
from thresholds import (
    candidate_table,
    count_at_threshold,
    decode_threshold,
    encode_threshold,
    select_threshold,
    validate_inputs,
)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g")


def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.dtype.str, array.shape)).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def draw_curves(y: np.ndarray, scores: np.ndarray, threshold: float, out: Path) -> float:
    precision, recall, _ = precision_recall_curve(y, scores)
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    auc = float(roc_auc_score(y, scores))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    colors = ["#d47b22", "#b6385b"]
    policies = [(0.5, "t = 0.5"), (threshold, "t* (мінімум вартості)")]

    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    ax.step(recall, precision, where="post", color="#216c91", linewidth=1.5, label="PR на validation")
    ax.axhline(float(y.mean()), color="#888888", linestyle="--", linewidth=1, label=f"Частка шахрайства = {y.mean():.4%}")
    for (t, name), color in zip(policies, colors, strict=True):
        c = count_at_threshold(y, scores, t)
        if c.precision is not None and c.recall is not None:
            ax.scatter(c.recall, c.precision, s=65, color=color, label=name, zorder=4)
    ax.set(xlabel="Повнота (recall)", ylabel="Точність (precision)", title="Precision–recall: рідкісний позитивний клас", xlim=(0, 1.01), ylim=(0, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", fontsize=9)
    fig.savefig(out / "validation_pr.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    ax.plot(fpr, tpr, color="#216c91", linewidth=1.6, label=f"ROC-AUC = {auc:.6f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="Випадкове ранжування")
    for (t, name), color in zip(policies, colors, strict=True):
        c = count_at_threshold(y, scores, t)
        ax.scatter(c.FP / (c.FP + c.TN), c.recall, s=65, color=color, label=name, zorder=4)
    ax.set(xlabel="Частка хибних тривог (FPR)", ylabel="Повнота (TPR)", title="ROC на validation", xlim=(-0.01, 1.01), ylim=(-0.01, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(out / "validation_roc.png", dpi=180)
    plt.close(fig)
    return auc


def error_examples(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, row_ids: np.ndarray, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    y, scores = validate_inputs(y, scores)
    predicted = scores >= threshold
    finite_threshold = bool(np.isfinite(threshold))
    examples = []
    summaries = []
    for kind, mask in (("FP", predicted & (y == 0)), ("FN", ~predicted & (y == 1))):
        positions = np.flatnonzero(mask)
        first = positions[np.argsort(row_ids[positions], kind="stable")][:5]
        for i in first:
            examples.append({
                "row_id": int(row_ids[i]),
                "error_type": kind,
                "label": int(y[i]),
                "score": float(scores[i]),
                "Amount": float(amounts[i]),
                "score_minus_threshold": float(scores[i]) - threshold if finite_threshold else None,
                "distance_to_threshold": abs(float(scores[i]) - threshold) if finite_threshold else None,
            })
        distances = np.abs(scores[positions] - threshold) if finite_threshold else np.array([])
        summaries.append({
            "error_type": kind,
            "count": len(positions),
            "amount_min": float(amounts[positions].min()) if positions.size else None,
            "amount_median": float(np.median(amounts[positions])) if positions.size else None,
            "amount_max": float(amounts[positions].max()) if positions.size else None,
            "distance_min": float(distances.min()) if distances.size else None,
            "distance_median": float(np.median(distances)) if distances.size else None,
            "distance_max": float(distances.max()) if distances.size else None,
        })
    columns = ["row_id", "error_type", "label", "score", "Amount", "score_minus_threshold", "distance_to_threshold"]
    return pd.DataFrame(examples, columns=columns), pd.DataFrame(summaries)


def analyze(saved: Path, out: Path) -> None:
    if not (saved / "predictions.npz").is_file():
        raise FileNotFoundError(f"Немає збережених прогнозів: {saved / 'predictions.npz'}")
    out.mkdir(parents=True, exist_ok=True)

    run_checks(out / "checks")
    with np.load(saved / "predictions.npz", allow_pickle=False) as archive:
        y_validation = archive["validation_y"]
        s_validation = archive["validation_scores"]
        validation_ids = archive["validation_row_ids"]
    validate_inputs(y_validation, s_validation)
    candidates = candidate_table(y_validation, s_validation)
    threshold = select_threshold(candidates)
    save_csv(candidates, out / "validation_candidates.csv")
    validation_summary = pd.DataFrame([
        {"rule": name, "threshold": t, **count_at_threshold(y_validation, s_validation, t).as_dict()}
        for name, t in (("standard", 0.5), ("minimum_cost", threshold), ("all_negative", np.inf))
    ])
    save_csv(validation_summary, out / "validation_summary.csv")
    auc = draw_curves(y_validation, s_validation, threshold, out)

    frozen = {
        "threshold": encode_threshold(threshold),
        "selection_split": "validation",
        "prediction_rule": "score >= threshold",
        "cost": {"FN": 10, "FP": 1},
        "tie_break": "largest threshold among minimum-cost candidates",
        "candidate_count": len(candidates),
        "validation_fingerprint_sha256": array_fingerprint(y_validation, s_validation, validation_ids),
        "checks_passed": ["A", "B", "C"],
    }
    write_json(out / "threshold.json", frozen)
    threshold = decode_threshold(json.loads((out / "threshold.json").read_text())["threshold"])

    with np.load(saved / "predictions.npz", allow_pickle=False) as archive:
        y_test = archive["test_y"]
        s_test = archive["test_scores"]
        test_ids = archive["test_row_ids"]
        amounts = archive["test_amount"]
        split_rows = []
        for name in ("train", "validation", "test"):
            labels = archive[f"{name}_y"]
            split_rows.append({"split": name, "n": len(labels), "positives": int(labels.sum()), "positive_fraction": float(labels.mean())})
    validate_inputs(y_test, s_test)
    split_summary = pd.DataFrame(split_rows)
    test = count_at_threshold(y_test, s_test, threshold)
    baseline = count_at_threshold(y_test, s_test, 0.5)
    test_summary = pd.DataFrame([
        {"rule": name, "threshold": t, **metrics.as_dict()}
        for name, t, metrics in (("standard", 0.5, baseline), ("validation_threshold", threshold, test))
    ])
    save_csv(test_summary, out / "test_summary.csv")
    save_csv(split_summary, out / "split_summary.csv")
    confusion = pd.DataFrame({"actual": [0, 1], "predicted_0": [test.TN, test.FN], "predicted_1": [test.FP, test.TP]})
    save_csv(confusion, out / "test_confusion.csv")
    examples, error_summary = error_examples(y_test, s_test, amounts, test_ids, threshold)
    save_csv(examples, out / "errors.csv")
    save_csv(error_summary, out / "error_summary.csv")
    metrics = {
        "threshold": encode_threshold(threshold),
        "validation_roc_auc": auc,
        "candidate_count": len(candidates),
        "validation": count_at_threshold(y_validation, s_validation, threshold).as_dict(),
        "test": test.as_dict(),
        "test_baseline_0_5": baseline.as_dict(),
        "predictions_sha256": hashlib.sha256((saved / "predictions.npz").read_bytes()).hexdigest(),
    }
    write_json(out / "metrics.json", metrics)
    print("\nValidation:")
    print(validation_summary.to_string(index=False))
    print(f"\nROC-AUC = {auc:.12f}; кандидатів = {len(candidates)}; t* = {threshold!r}")
    print("\nTest (поріг уже зафіксовано):")
    print(test_summary.to_string(index=False))
    print(f"\nРезультати аналізу: {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Пункти 2–4: аналіз збережених прогнозів без CSV і повторного навчання.")
    parser.add_argument("--saved", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("reproduced"), help="Тека для результатів аналізу (типово reproduced).")
    args = parser.parse_args()
    analyze(args.saved, args.out)


if __name__ == "__main__":
    main()
