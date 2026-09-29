from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class Parameters:

    W1: FloatArray
    b1: FloatArray
    W2: FloatArray
    b2: FloatArray


@dataclass(frozen=True)
class ForwardCache:

    X: FloatArray
    y: IntArray
    Z1: FloatArray
    A1: FloatArray
    logits: FloatArray
    probabilities: FloatArray


@dataclass(frozen=True)
class Gradients:

    W1: FloatArray
    b1: FloatArray
    W2: FloatArray
    b2: FloatArray


def initialize_parameters(seed: int = 0) -> Parameters:

    rng = np.random.default_rng(seed)
    W1 = rng.normal(
        loc=0.0,
        scale=np.sqrt(2.0 / 4.0),
        size=(4, 8),
    ).astype(np.float64)
    b1 = np.zeros(8, dtype=np.float64)
    W2 = rng.normal(
        loc=0.0,
        scale=np.sqrt(2.0 / (8.0 + 3.0)),
        size=(8, 3),
    ).astype(np.float64)
    b2 = np.zeros(3, dtype=np.float64)

    parameters = Parameters(W1=W1, b1=b1, W2=W2, b2=b2)
    validate_parameter_shapes(parameters)
    return parameters


def validate_parameter_shapes(parameters: Parameters) -> None:

    expected_shapes = {
        "W1": (4, 8),
        "b1": (8,),
        "W2": (8, 3),
        "b2": (3,),
    }
    for name, expected_shape in expected_shapes.items():
        value = getattr(parameters, name)
        if value.shape != expected_shape:
            raise ValueError(
                f"{name}: очікувалася форма {expected_shape}, отримано {value.shape}"
            )
        if value.dtype != np.float64:
            raise TypeError(f"{name}: очікувався dtype float64, отримано {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name}: знайдено NaN або нескінченність")


def forward_loss(
    X: FloatArray,
    y: IntArray,
    parameters: Parameters,
) -> tuple[float, ForwardCache]:

    validate_parameter_shapes(parameters)
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"X має мати форму (N, 4), отримано {X.shape}")
    if y.shape != (X.shape[0],):
        raise ValueError(f"y має мати форму ({X.shape[0]},), отримано {y.shape}")
    if X.dtype != np.float64:
        raise TypeError(f"X має бути float64, отримано {X.dtype}")
    if np.any((y < 0) | (y >= 3)):
        raise ValueError("Мітки класів мають належати множині {0, 1, 2}")

    Z1 = X @ parameters.W1 + parameters.b1
    A1 = np.maximum(Z1, 0.0)
    logits = A1 @ parameters.W2 + parameters.b2

    # Для кожного об'єкта максимум має форму (N, 1), щоб broadcasting
    # відбувався лише вздовж осі класів.
    row_max = np.max(logits, axis=1, keepdims=True)
    shifted_logits = logits - row_max
    exp_shifted = np.exp(shifted_logits)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    log_sum_exp = row_max + np.log(sum_exp)
    log_probabilities = logits - log_sum_exp
    probabilities = exp_shifted / sum_exp

    sample_indices = np.arange(X.shape[0])
    loss = float(-np.mean(log_probabilities[sample_indices, y]))

    arrays_to_check = (Z1, A1, logits, log_probabilities, probabilities)
    if not np.isfinite(loss) or not all(np.all(np.isfinite(a)) for a in arrays_to_check):
        raise FloatingPointError("Forward повернув NaN або нескінченність")

    cache = ForwardCache(
        X=X,
        y=y,
        Z1=Z1,
        A1=A1,
        logits=logits,
        probabilities=probabilities,
    )
    return loss, cache


def backward(
    cache: ForwardCache,
    parameters: Parameters,
    *,
    divide_by_batch: bool = True,
) -> Gradients:

    sample_count = cache.X.shape[0]

    # dL/dZ2 = (softmax(Z2) - one_hot(y)) / N для mean cross-entropy.
    dZ2 = cache.probabilities.copy()
    dZ2[np.arange(sample_count), cache.y] -= 1.0
    if divide_by_batch:
        dZ2 /= sample_count

    dW2 = cache.A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ parameters.W2.T
    dZ1 = dA1 * (cache.Z1 > 0.0)

    dW1 = cache.X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    gradients = Gradients(W1=dW1, b1=db1, W2=dW2, b2=db2)
    validate_gradient_shapes(gradients, parameters)
    return gradients


def validate_gradient_shapes(gradients: Gradients, parameters: Parameters) -> None:

    for name in ("W1", "b1", "W2", "b2"):
        gradient = getattr(gradients, name)
        parameter = getattr(parameters, name)
        if gradient.shape != parameter.shape:
            raise AssertionError(
                f"Градієнт {name} має форму {gradient.shape}, "
                f"параметр — {parameter.shape}"
            )
        if gradient.dtype != np.float64:
            raise TypeError(f"Градієнт {name} має dtype {gradient.dtype}, а не float64")
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError(f"Градієнт {name} містить NaN або нескінченність")


def gradient_dict(gradients: Gradients) -> dict[str, FloatArray]:

    return {
        "W1": gradients.W1,
        "b1": gradients.b1,
        "W2": gradients.W2,
        "b2": gradients.b2,
    }
