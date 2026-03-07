from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import optim

from .eva_x import eva_x_tiny_patch16

CheckpointMetadata = dict[str, Any]


class Eva_X_Model(nn.Module):
    def __init__(self, pretrained_weights_path: str):
        super().__init__()
        self.eva = eva_x_tiny_patch16(pretrained=pretrained_weights_path)
        self.eva.head = nn.Linear(192, 1)

        for parameter in self.eva.parameters():
            parameter.requires_grad = False

        for name, parameter in self.eva.named_parameters():
            if (
                "head" in name
                or "blocks.11" in name
                or "norm" in name
                or "fc_norm" in name
            ):
                parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eva(x)


def _to_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _load_checkpoint(ckpt_path: str) -> dict[str, Any]:
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _checkpoint_metadata(checkpoint: dict[str, Any]) -> CheckpointMetadata:
    return {
        "epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "class_to_idx": checkpoint.get("class_to_idx"),
    }


def create_model(
    pretrained_weights_path: str, device: str | torch.device
) -> Eva_X_Model:
    model = Eva_X_Model(pretrained_weights_path)
    model.to(_to_device(device))
    return model


def load_model_for_inference(
    ckpt_path: str,
    pretrained_weights_path: str,
    device: str | torch.device,
) -> Eva_X_Model:
    model = create_model(pretrained_weights_path, device)
    checkpoint = _load_checkpoint(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resume_from_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> tuple[optim.Optimizer, CheckpointMetadata]:
    checkpoint = _load_checkpoint(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return optimizer, _checkpoint_metadata(checkpoint)
