from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from eva_x import create_eva_x_tiny

DEFAULT_CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


class EvaXBinaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eva = create_eva_x_tiny()
        self.eva.head = nn.Linear(192, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eva(x)


def resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    resolved = Path(checkpoint_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    resolved = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing 'model_state_dict'.")
    return checkpoint


def class_names_from_checkpoint(checkpoint: dict[str, Any]) -> list[str]:
    mapping = checkpoint.get("class_to_idx")
    if not isinstance(mapping, dict) or not mapping:
        return DEFAULT_CLASS_NAMES.copy()

    try:
        return [
            name for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))
        ]
    except Exception:
        return DEFAULT_CLASS_NAMES.copy()


def checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    raw_epoch = checkpoint.get("epoch")
    display_epoch = raw_epoch + 1 if isinstance(raw_epoch, int) else raw_epoch

    return {
        "epoch": display_epoch,
        "stored_epoch": raw_epoch,
        "best_val_loss": checkpoint.get("best_val_loss"),
        "class_to_idx": checkpoint.get("class_to_idx"),
    }


def load_eva_x_binary_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing 'model_state_dict'.")

    model = EvaXBinaryModel()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_eva_x_binary(
    checkpoint_path: str | Path,
    device: torch.device,
) -> nn.Module:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    return load_eva_x_binary_from_checkpoint(checkpoint, device=device)
