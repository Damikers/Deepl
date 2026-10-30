from __future__ import annotations

from common import CLASSES
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

FROZEN_MODULES = {
    "scratch": (),
    "frozen": ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"),
    "unfreeze": ("conv1", "bn1", "layer1", "layer2", "layer3"),
}
TRAINABLE = {"scratch": 11179590, "frozen": 3078, "unfreeze": 8396806}
MODULES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")


def build_model(regime: str, seed: int, device: torch.device):
    if regime not in FROZEN_MODULES:
        raise ValueError(regime)
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(regime == "scratch" or name.startswith("fc.") or
                                 (regime == "unfreeze" and name.startswith("layer4.")))
    model = model.to(device=device, dtype=torch.float32)
    if regime == "unfreeze":
        groups = [{"params": list(model.fc.parameters()), "lr": 3e-3, "name": "fc"},
                  {"params": list(model.layer4.parameters()), "lr": 3e-4, "name": "layer4"}]
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": 3e-3, "name": "all" if regime == "scratch" else "fc"}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    describe_model(model, optimizer, regime)
    return model, optimizer


def set_train_mode(model: torch.nn.Module, regime: str) -> None:

    model.train()
    for name in FROZEN_MODULES[regime]:
        getattr(model, name).eval()


def describe_model(model, optimizer, regime: str) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if (total != 11179590 or trainable != TRAINABLE[regime] or
        any(not p.requires_grad for p in parameters) or
        len({id(p) for p in parameters}) != len(parameters) or
        {id(p) for p in parameters} != trainable_ids):
        raise AssertionError("Parameter/optimizer control failed")
    return {"regime": regime, "total": total, "trainable": trainable,
            "optimizer_groups": [{"name": g["name"], "numel": sum(p.numel() for p in g["params"]),
                                  **{k: v for k, v in g.items() if k not in ("params", "name")}}
                                 for g in optimizer.param_groups],
            "eval_modules_during_training": list(FROZEN_MODULES[regime])}


@torch.no_grad()
def predict(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    batches = [model(x.to(device)).cpu().numpy() for x, _ in loader]
    logits = np.concatenate(batches)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    return logits
