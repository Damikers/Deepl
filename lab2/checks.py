"""Незалежний повільний еталон на малих наборах A, B, C з умови."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thresholds import COLUMNS, candidate_table, select_threshold

CASES = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0], 0.4),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0], np.inf),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 11, 0.8),
}


def brute_force(y: np.ndarray, scores: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Не використовує таблицю, групування чи лічильники швидкого алгоритму."""
    candidates = [float("inf")] + sorted({float(s) for s in scores}, reverse=True)
    rows = []
    for threshold in candidates:
        tp = fp = fn = tn = 0
        for label, score in zip(y, scores, strict=True):
            if score >= threshold:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
            elif label == 1:
                fn += 1
            else:
                tn += 1
        rows.append({"threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp})
    best = min(rows, key=lambda row: (row["C"], -row["threshold"]))
    return pd.DataFrame(rows, columns=COLUMNS), float(best["threshold"])


def run_checks(out: Path | None = None) -> dict[str, pd.DataFrame]:
    tables = {}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    for name, (scores, labels, expected) in CASES.items():
        y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
        fast = candidate_table(y, s)
        slow, slow_threshold = brute_force(y, s)
        # Порівнюємо кожен поріг і кожен цілий лічильник без допусків.
        np.testing.assert_array_equal(fast["threshold"].to_numpy(), slow["threshold"].to_numpy())
        np.testing.assert_array_equal(fast[COLUMNS[1:]].to_numpy(), slow[COLUMNS[1:]].to_numpy())
        fast_threshold = select_threshold(fast)
        if fast_threshold != slow_threshold or fast_threshold != expected:
            raise AssertionError(f"Набір {name}: неправильний поріг або правило нічиєї.")
        # Незалежність результату від початкового порядку, зокрема всередині ties.
        reversed_table = candidate_table(y[::-1], s[::-1])
        np.testing.assert_array_equal(fast.to_numpy(), reversed_table.to_numpy())
        comparison = fast.copy()
        for column in COLUMNS[1:]:
            comparison[f"direct_{column}"] = slow[column]
        comparison["exact_match"] = True
        if out is not None:
            comparison.to_csv(out / f"check_{name}.csv", index=False, float_format="%.17g")
        print(f"Перевірка {name}: усі {len(fast)} кандидатів збігаються точно; t*={fast_threshold!r} — PASS")
        tables[name] = comparison
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/checks"))
    run_checks(parser.parse_args().out)
