from __future__ import annotations

from itertools import islice
import math
from pathlib import Path

from common import CLASSES, MEAN, STD, write_json
from data import SplitDataset, make_loader, validate_split
from models import MODULES, FROZEN_MODULES, build_model, describe_model, set_train_mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BN_PREDICTION = (
    "Якщо frozen викликати лише model.train(), параметри кістяка не зміняться, "
    "бо requires_grad=False, але running_mean, running_var і num_batches_tracked "
    "BatchNorm зміняться. Навчатиметься fc. Крива train loss не є перевіркою "
    "заморожування: втрата може спадати і за такої помилки."
)


def save_batch(x: torch.Tensor, y: torch.Tensor, path: Path) -> None:
    pixels = (x[:16].cpu() * torch.tensor(STD)[None, :, None, None] +
              torch.tensor(MEAN)[None, :, None, None]).clamp(0, 1)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), layout="constrained")
    for ax, image, label in zip(axes.flat, pixels, y[:16]):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[int(label)], fontsize=10)
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def five_steps(regime: str, dataset: SplitDataset, device: torch.device, buggy: bool = False) -> dict:

    model, optimizer = build_model(regime, 1, device)
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    loader = make_loader(dataset, True)
    model.train() if buggy else set_train_mode(model, regime)
    losses = []
    for x, y in islice(loader, 5):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        optimizer.step()
    after = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    rows = []
    for module in MODULES:
        params = [name for name, _ in model.named_parameters() if name.startswith(module + ".")]
        buffers = [name for name, _ in model.named_buffers()
                   if name.startswith(module + ".") and name.endswith(("running_mean", "running_var", "num_batches_tracked"))]
        parameter_changed = any(not torch.equal(before[n], after[n]) for n in params)
        buffer_changed = any(not torch.equal(before[n], after[n]) for n in buffers) if buffers else None
        expected_parameter = module not in FROZEN_MODULES[regime]
        expected_buffer = (buggy or expected_parameter) if buffers else None
        passed = parameter_changed == expected_parameter and buffer_changed == expected_buffer
        rows.append({"regime": regime, "buggy": buggy, "module": module,
                     "parameters_changed": parameter_changed, "bn_buffers_changed": buffer_changed,
                     "expected_parameters_changed": expected_parameter,
                     "expected_bn_buffers_changed": expected_buffer, "passed": passed})
    deltas = {key: float((after["layer1.0.bn1." + key] - before["layer1.0.bn1." + key]).abs().max())
              for key in ("running_mean", "running_var")}
    result = {"regime": regime, "buggy": buggy, "steps": 5, "rows": rows,
              "train_batch_losses": losses, "layer1.0.bn1_max_abs_change": deltas}
    if not all(row["passed"] for row in rows):
        raise AssertionError(f"State check failed: {result}")
    return result


def run_checks(root: Path, split: Path, out: Path, device: torch.device) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    figures = out.parent / "figures"
    figures.mkdir(exist_ok=True)
    control = validate_split(split)
    dataset = SplitDataset(root, split, "train")
    descriptions = []
    initial = []
    for regime in ("scratch", "frozen", "unfreeze"):
        model, optimizer = build_model(regime, 1, device)
        descriptions.append(describe_model(model, optimizer, regime))
        if regime != "unfreeze":
            set_train_mode(model, regime)
            x, y = next(iter(make_loader(dataset, True)))
            assert x.shape == (32, 3, 160, 160) and x.dtype == torch.float32
            with torch.no_grad():
                loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            initial.append({"regime": regime, "seed": 1, "loss": float(loss.cpu()), "ln6": math.log(6)})
            if regime == "scratch":
                save_batch(x, y, figures / "augmented_batch.png")
        del model, optimizer
    state_checks = []
    for regime in ("frozen", "unfreeze"):
        print(f"Five-step state check: {regime}", flush=True)
        state_checks.append(five_steps(regime, dataset, device))

    write_json(out / "bn_prediction.json", {"prediction_before_experiment": BN_PREDICTION})
    print("Five-step intentional BN bug", flush=True)
    buggy = five_steps("frozen", dataset, device, buggy=True)
    result = {"all_passed": True, "split_control": control, "no_overlap": True,
              "models": descriptions, "initial_losses": initial, "state_checks": state_checks,
              "bn_bug": {"prediction": BN_PREDICTION, "prediction_confirmed": True, **buggy}}
    write_json(out / "checks.json", result)
    print("All pre-training checks PASSED", flush=True)
    return result
