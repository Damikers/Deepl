from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from numpy_network import (
    FloatArray,
    Gradients,
    Parameters,
    forward_loss,
    gradient_dict,
)
if TYPE_CHECKING:
    from torch_reference import TorchResult


@dataclass(frozen=True)
class ComparisonRow:

    quantity: str
    max_absolute_difference: float
    all_finite: bool
    passed: bool


@dataclass(frozen=True)
class TorchComparison:

    numpy_loss: float
    torch_loss: float
    rows: tuple[ComparisonRow, ...]

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)


@dataclass(frozen=True)
class NumericalGradientRow:

    parameter: str
    manual_gradient: float
    numerical_gradient: float
    absolute_difference: float
    all_finite: bool
    passed: bool


@dataclass(frozen=True)
class NumericalGradientCheck:

    epsilon: float
    rows: tuple[NumericalGradientRow, ...]

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)


def compare_with_torch(
    numpy_loss: float,
    numpy_gradients: Gradients,
    torch_result: TorchResult,
    *,
    tolerance: float = 1e-12,
) -> TorchComparison:

    rows: list[ComparisonRow] = []

    loss_finite = bool(np.isfinite(numpy_loss) and np.isfinite(torch_result.loss))
    loss_difference = abs(numpy_loss - torch_result.loss)
    rows.append(
        ComparisonRow(
            quantity="Втрата",
            max_absolute_difference=loss_difference,
            all_finite=loss_finite,
            passed=loss_finite and loss_difference <= tolerance,
        )
    )

    numpy_gradient_dict = gradient_dict(numpy_gradients)
    torch_gradient_dict = gradient_dict(torch_result.gradients)
    for name in ("W1", "b1", "W2", "b2"):
        numpy_value = numpy_gradient_dict[name]
        torch_value = torch_gradient_dict[name]
        all_finite = bool(
            np.all(np.isfinite(numpy_value)) and np.all(np.isfinite(torch_value))
        )
        difference = float(np.max(np.abs(numpy_value - torch_value)))
        rows.append(
            ComparisonRow(
                quantity=f"Градієнт {name}",
                max_absolute_difference=difference,
                all_finite=all_finite,
                passed=all_finite and difference <= tolerance,
            )
        )

    return TorchComparison(
        numpy_loss=numpy_loss,
        torch_loss=torch_result.loss,
        rows=tuple(rows),
    )


def check_selected_numerical_gradients(
    X: FloatArray,
    y: np.ndarray,
    parameters: Parameters,
    manual_gradients: Gradients,
    *,
    epsilon: float = 1e-6,
    tolerance: float = 1e-7,
) -> NumericalGradientCheck:

    selected: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("W1", (0, 0)),
        ("b1", (0,)),
        ("W2", (0, 0)),
        ("b2", (0,)),
    )
    rows: list[NumericalGradientRow] = []

    for parameter_name, index in selected:
        parameter = getattr(parameters, parameter_name)
        original_value = float(parameter[index])

        # finally гарантує відновлення навіть тоді, коли forward завершиться помилкою.
        try:
            parameter[index] = original_value + epsilon
            loss_plus, _ = forward_loss(X, y, parameters)

            parameter[index] = original_value - epsilon
            loss_minus, _ = forward_loss(X, y, parameters)
        finally:
            parameter[index] = original_value

        numerical_gradient = (loss_plus - loss_minus) / (2.0 * epsilon)
        manual_gradient = float(getattr(manual_gradients, parameter_name)[index])
        absolute_difference = abs(numerical_gradient - manual_gradient)
        all_finite = bool(
            np.isfinite(numerical_gradient)
            and np.isfinite(manual_gradient)
            and np.isfinite(absolute_difference)
        )
        pretty_index = ",".join(str(i) for i in index)
        rows.append(
            NumericalGradientRow(
                parameter=f"{parameter_name}[{pretty_index}]",
                manual_gradient=manual_gradient,
                numerical_gradient=numerical_gradient,
                absolute_difference=absolute_difference,
                all_finite=all_finite,
                passed=all_finite and absolute_difference <= tolerance,
            )
        )

    return NumericalGradientCheck(epsilon=epsilon, rows=tuple(rows))
