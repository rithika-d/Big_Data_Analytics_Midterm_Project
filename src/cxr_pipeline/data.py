from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from .transforms import get_eval_transforms, get_train_transforms


def load_datasets(data_dir: str):
    root = Path(data_dir)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    for split_dir in (train_dir, val_dir, test_dir):
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing dataset split directory: {split_dir}")

    train_dataset = datasets.ImageFolder(
        root=str(train_dir), transform=get_train_transforms()
    )
    val_dataset = datasets.ImageFolder(
        root=str(val_dir), transform=get_eval_transforms()
    )
    test_dataset = datasets.ImageFolder(
        root=str(test_dir), transform=get_eval_transforms()
    )
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int = 32,
    num_workers: int = 4,
):
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
