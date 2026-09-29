from __future__ import annotations

import argparse
import sys

import numpy as np

from checks import (
    NumericalGradientCheck,
    TorchComparison,
    check_selected_numerical_gradients,
    compare_with_torch,
)
from data import IrisSplit, load_and_prepare_iris
from numpy_network import (
    Gradients,
    Parameters,
    backward,
    forward_loss,
    gradient_dict,
    initialize_parameters,
)
from torch_reference import torch_reference


def yes_no(value: bool) -> str:
    return "так" if value else "ні"


def print_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
    widths: tuple[int, ...],
    alignments: tuple[str, ...],
) -> None:
    """Надрукувати таблицю з фіксованими ширинами колонок."""

    if not (len(headers) == len(widths) == len(alignments)):
        raise ValueError("Кількість заголовків, ширин і вирівнювань не збігається")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Кількість значень у рядку не збігається з заголовком")

    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_line = "| " + " | ".join(
        f"{header:^{width}}" for header, width in zip(headers, widths, strict=True)
    ) + " |"

    print(separator)
    print(header_line)
    print(separator)
    for row in rows:
        row_line = "| " + " | ".join(
            f"{value:{alignment}{width}}"
            for value, width, alignment in zip(
                row, widths, alignments, strict=True
            )
        ) + " |"
        print(row_line)
    print(separator)


def print_data_summary(split: IrisSplit) -> None:
    print("Дані")
    rows = [
        ("X_train", str(split.X_train.shape)),
        ("y_train", str(split.y_train.shape)),
        ("X_test", str(split.X_test.shape)),
        ("y_test", str(split.y_test.shape)),
        ("Класи train", str(np.bincount(split.y_train, minlength=3).tolist())),
        ("Класи test", str(np.bincount(split.y_test, minlength=3).tolist())),
        ("dtype ознак", str(split.X_train.dtype)),
    ]
    print_table(
        headers=("Величина", "Значення"),
        rows=rows,
        widths=(18, 22),
        alignments=("<", "^"),
    )
    print()


def print_parameter_summary(parameters: Parameters) -> None:
    print("Параметри")
    rows = [
        (name, str(getattr(parameters, name).shape))
        for name in ("W1", "b1", "W2", "b2")
    ]
    print_table(
        headers=("Параметр", "Форма"),
        rows=rows,
        widths=(12, 14),
        alignments=("<", "^"),
    )
    print()


def print_torch_comparison(comparison: TorchComparison) -> None:
    print("Звірка NumPy / PyTorch")
    print(f"- NumPy loss:   {comparison.numpy_loss:.17g}")
    print(f"- PyTorch loss: {comparison.torch_loss:.17g}")
    print()
    rows = [
        (
            row.quantity,
            f"{row.max_absolute_difference:.12e}",
            yes_no(row.all_finite),
            yes_no(row.passed),
        )
        for row in comparison.rows
    ]
    print_table(
        headers=(
            "Величина",
            "Максимальна абсолютна різниця",
            "Скінченні",
            "Пройдено",
        ),
        rows=rows,
        widths=(20, 32, 10, 10),
        alignments=("<", ">", "^", "^"),
    )
    print()


def print_numerical_check(check: NumericalGradientCheck) -> None:
    print("Чисельна перевірка")
    print(f"- epsilon: {check.epsilon:.1e}")
    print()
    rows = [
        (
            row.parameter,
            f"{row.manual_gradient:.12e}",
            f"{row.numerical_gradient:.12e}",
            f"{row.absolute_difference:.12e}",
            yes_no(row.passed),
        )
        for row in check.rows
    ]
    print_table(
        headers=(
            "Параметр",
            "Градієнт backward()",
            "Чисельна похідна",
            "Абсолютна різниця",
            "Пройдено",
        ),
        rows=rows,
        widths=(14, 22, 22, 22, 10),
        alignments=("<", ">", ">", ">", "^"),
    )
    print()


def print_bug_prediction(sample_count: int) -> None:
    print("Прогноз до запуску навмисної помилки")
    print("- Втрата не зміниться, бо змінюється лише backward.")
    print(
        f"- Усі ненульові ручні градієнти збільшаться у N={sample_count} разів."
    )
    print("- Перевірка втрати з PyTorch пройде.")
    print("- Звірка градієнтів із PyTorch не пройде.")
    print("- Чисельна перевірка градієнтів не пройде.")
    print()


def print_bug_scaling(
    buggy_gradients: Gradients,
    correct_gradients: Gradients,
    sample_count: int,
) -> tuple[float, ...]:
    print("Масштабування градієнтів у помилковій реалізації")
    print(f"Очікувана рівність: g_bug = {sample_count} * g_correct")
    print()
    residuals: list[float] = []
    buggy = gradient_dict(buggy_gradients)
    correct = gradient_dict(correct_gradients)
    rows: list[tuple[str, str]] = []
    for name in ("W1", "b1", "W2", "b2"):
        residual = float(np.max(np.abs(buggy[name] - sample_count * correct[name])))
        residuals.append(residual)
        rows.append((name, f"{residual:.12e}"))
    print_table(
        headers=("Градієнт", "max abs(g_bug - N * g_correct)"),
        rows=rows,
        widths=(14, 34),
        alignments=("<", ">"),
    )
    print()
    return tuple(residuals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ЛР1: ручний backprop та незалежні перевірки градієнтів"
    )
    parser.add_argument(
        "--buggy-backward",
        action="store_true",
        help="навмисно прибрати ділення dZ2 на кількість об'єктів",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = load_and_prepare_iris(seed=0)
    parameters = initialize_parameters(seed=0)

    print_data_summary(split)
    print_parameter_summary(parameters)

    numpy_loss, cache = forward_loss(split.X_train, split.y_train, parameters)
    correct_gradients = backward(cache, parameters, divide_by_batch=True)

    if args.buggy_backward:
        print_bug_prediction(sample_count=split.X_train.shape[0])
        tested_gradients = backward(cache, parameters, divide_by_batch=False)
    else:
        tested_gradients = correct_gradients

    torch_result = torch_reference(
        split.X_train,
        split.y_train,
        parameters,
    )
    torch_comparison = compare_with_torch(
        numpy_loss,
        tested_gradients,
        torch_result,
        tolerance=1e-12,
    )
    print_torch_comparison(torch_comparison)

    numerical_check = check_selected_numerical_gradients(
        split.X_train,
        split.y_train,
        parameters,
        tested_gradients,
        epsilon=1e-6,
        tolerance=1e-7,
    )
    print_numerical_check(numerical_check)

    if not args.buggy_backward:
        success = torch_comparison.passed and numerical_check.passed
        print("Підсумок")
        print(
            "Правильна реалізація пройшла всі перевірки."
            if success
            else "ПОМИЛКА: правильна реалізація не пройшла всі перевірки."
        )
        return 0 if success else 1

    residuals = print_bug_scaling(
        tested_gradients,
        correct_gradients,
        sample_count=split.X_train.shape[0],
    )
    loss_passed = torch_comparison.rows[0].passed
    gradients_failed = all(not row.passed for row in torch_comparison.rows[1:])
    numerical_failed = all(not row.passed for row in numerical_check.rows)
    scaling_confirmed = all(residual <= 1e-12 for residual in residuals)
    bug_detected = (
        loss_passed and gradients_failed and numerical_failed and scaling_confirmed
    )

    print("Підсумок досліду з помилкою")
    print(f"- Втрата залишилася правильною: {yes_no(loss_passed)}")
    print(f"- Усі PyTorch-перевірки градієнтів виявили помилку: {yes_no(gradients_failed)}")
    print(f"- Усі чисельні перевірки виявили помилку: {yes_no(numerical_failed)}")
    print(f"- Збільшення у 105 разів підтверджено: {yes_no(scaling_confirmed)}")
    print(
        "Правильний результат: навмисну помилку виявлено."
        if bug_detected
        else "ПОМИЛКА: експеримент не дав очікуваного результату."
    )
    return 0 if bug_detected else 1


if __name__ == "__main__":
    sys.exit(main())
