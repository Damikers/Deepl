from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from numpy_network import FloatArray, Gradients, Parameters


@dataclass(frozen=True)
class TorchResult:

    loss: float
    gradients: Gradients


def torch_reference(
    X: FloatArray,
    y: np.ndarray,
    parameters: Parameters,
) -> TorchResult:

    first = nn.Linear(4, 8, bias=True, dtype=torch.float64)
    second = nn.Linear(8, 3, bias=True, dtype=torch.float64)
    model = nn.Sequential(first, nn.ReLU(), second)

    # NumPy зберігає W як (in, out), а nn.Linear — як (out, in).
    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(parameters.W1.T.copy()))
        first.bias.copy_(torch.from_numpy(parameters.b1.copy()))
        second.weight.copy_(torch.from_numpy(parameters.W2.T.copy()))
        second.bias.copy_(torch.from_numpy(parameters.b2.copy()))

    X_tensor = torch.from_numpy(X.copy()).to(dtype=torch.float64)
    y_tensor = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(dtype=torch.int64)

    model.zero_grad(set_to_none=True)
    logits = model(X_tensor)
    loss_tensor = F.cross_entropy(logits, y_tensor, reduction="mean")
    loss_tensor.backward()

    if any(parameter.grad is None for parameter in model.parameters()):
        raise RuntimeError("PyTorch не створив один або кілька градієнтів")

    # Транспонуємо градієнти ваг назад до NumPy-конвенції (in, out).
    gradients = Gradients(
        W1=first.weight.grad.detach().cpu().numpy().T.copy(),  # type: ignore[union-attr]
        b1=first.bias.grad.detach().cpu().numpy().copy(),  # type: ignore[union-attr]
        W2=second.weight.grad.detach().cpu().numpy().T.copy(),  # type: ignore[union-attr]
        b2=second.bias.grad.detach().cpu().numpy().copy(),  # type: ignore[union-attr]
    )
    return TorchResult(loss=float(loss_tensor.detach().cpu().item()), gradients=gradients)

