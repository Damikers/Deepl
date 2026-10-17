"""Ручні лічильники та точний пошук порога за O(n log n)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

FN_COST = 10
FP_COST = 1
COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


@dataclass(frozen=True)
class Confusion:
    TP: int
    FP: int
    FN: int
    TN: int

    @property
    def C(self) -> int:
        return FN_COST * self.FN + FP_COST * self.FP

    @property
    def precision(self) -> float | None:
        denominator = self.TP + self.FP
        return self.TP / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.TP + self.FN
        return self.TP / denominator if denominator else None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            **asdict(self),
            "C": self.C,
            "precision": self.precision,
            "recall": self.recall,
        }


def validate_inputs(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Не округлюємо оцінки: float32 -> float64 зберігає їх точно."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.ndim != 1 or y.shape != scores.shape or y.size == 0:
        raise ValueError("Потрібні непорожні одновимірні y та scores однакової довжини.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Мітки мають дорівнювати 0 або 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Оцінки мають бути скінченними.")
    return y.astype(np.int64, copy=False), scores


def count_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float) -> Confusion:
    """Власний підрахунок TP/FP/FN/TN за правилом scores >= threshold."""
    y, scores = validate_inputs(y, scores)
    if np.isnan(threshold):
        raise ValueError("Поріг не може бути NaN.")
    positive = y == 1
    predicted = scores >= threshold
    return Confusion(
        TP=int(np.count_nonzero(predicted & positive)),
        FP=int(np.count_nonzero(predicted & ~positive)),
        FN=int(np.count_nonzero(~predicted & positive)),
        TN=int(np.count_nonzero(~predicted & ~positive)),
    )


def candidate_table(y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Кандидати за спаданням: +inf, усі унікальні scores.

    На кінці групи рівних оцінок префікс уже містить усю групу.
    Тому жоден кандидат не розділяє об'єкти з однаковою оцінкою.
    Після сортування є лише лінійні векторні операції, без циклу
    за кандидатами з повторним переглядом усіх прикладів.
    """
    y, scores = validate_inputs(y, scores)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y[order]
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), y.size - 1]
    cumulative_tp = np.cumsum(sorted_y, dtype=np.int64)
    tp = np.r_[np.int64(0), cumulative_tp[group_ends]]
    fp = np.r_[np.int64(0), group_ends + 1 - cumulative_tp[group_ends]]
    positives = int(y.sum())
    fn = positives - tp
    tn = y.size - positives - fp
    return pd.DataFrame(
        {
            "threshold": np.r_[np.inf, sorted_scores[group_ends]],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "C": FN_COST * fn + FP_COST * fp,
        },
        columns=COLUMNS,
    )


def select_threshold(table: pd.DataFrame) -> float:
    """Серед мінімізаторів C явно вибираємо найбільший поріг."""
    minimum = table["C"].min()
    return float(table.loc[table["C"] == minimum, "threshold"].max())


def encode_threshold(threshold: float) -> float | str:
    """У JSON немає стандартного числового значення Infinity."""
    return "inf" if np.isposinf(threshold) else float(threshold)


def decode_threshold(value: float | str) -> float:
    return float(value)
