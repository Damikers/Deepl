from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class IrisSplit:

    X_train: FloatArray
    y_train: IntArray
    X_test: FloatArray
    y_test: IntArray
    mean: FloatArray
    std: FloatArray
    train_indices: IntArray
    test_indices: IntArray


def load_and_prepare_iris(seed: int = 0) -> IrisSplit:

    iris = load_iris()
    X = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_parts: list[IntArray] = []
    test_parts: list[IntArray] = []

    for class_id in (0, 1, 2):
        class_indices = np.flatnonzero(y == class_id).astype(np.int64)
        rng.shuffle(class_indices)
        train_parts.append(class_indices[:35])
        test_parts.append(class_indices[35:])

    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)

    X_train_raw = X[train_indices]
    X_test_raw = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0, ddof=0)
    if np.any(std == 0.0):
        raise ValueError("Стандартизація неможлива: знайдено ознаку з нульовим std")

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    split = IrisSplit(
        X_train=np.asarray(X_train, dtype=np.float64),
        y_train=y_train,
        X_test=np.asarray(X_test, dtype=np.float64),
        y_test=y_test,
        mean=np.asarray(mean, dtype=np.float64),
        std=np.asarray(std, dtype=np.float64),
        train_indices=train_indices,
        test_indices=test_indices,
    )
    validate_split(split)
    return split


def validate_split(split: IrisSplit) -> None:

    expected_shapes = {
        "X_train": ((105, 4), split.X_train.shape),
        "y_train": ((105,), split.y_train.shape),
        "X_test": ((45, 4), split.X_test.shape),
        "y_test": ((45,), split.y_test.shape),
    }
    for name, (expected, actual) in expected_shapes.items():
        if actual != expected:
            raise AssertionError(f"{name}: очікувалася форма {expected}, отримано {actual}")

    if split.X_train.dtype != np.float64 or split.X_test.dtype != np.float64:
        raise AssertionError("Ознаки мають обчислюватися у float64")

    train_counts = np.bincount(split.y_train, minlength=3)
    test_counts = np.bincount(split.y_test, minlength=3)
    if not np.array_equal(train_counts, np.array([35, 35, 35])):
        raise AssertionError(f"Неправильна стратифікація train: {train_counts}")
    if not np.array_equal(test_counts, np.array([15, 15, 15])):
        raise AssertionError(f"Неправильна стратифікація test: {test_counts}")

    if not np.all(np.isfinite(split.X_train)) or not np.all(np.isfinite(split.X_test)):
        raise AssertionError("Після стандартизації знайдено нескінченні значення або NaN")
