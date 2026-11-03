"""Dataset utilities for the distributed-training starter project."""
import os
from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def find_unsw_data(custom_path: Optional[str] = None) -> Optional[str]:
    """Find preprocessed UNSW-NB15 data.npz in common paths."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "../../output_preprocess/data.npz"),
        os.path.join(base_dir, "../output_preprocess/data.npz"),
        os.path.join(base_dir, "output_preprocess/data.npz"),
        "output_preprocess/data.npz",
        "../output_preprocess/data.npz",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def build_datasets(
    num_samples: int = 12000,
    input_dim: int = 32,
    num_classes: int = 4,
    seed: int = 42,
    train_split: float = 0.8,
    data_path: Optional[str] = None,
    force_synthetic: bool = False,
) -> Tuple[TensorDataset, TensorDataset, int, int]:
    """Create or load datasets. Returns (train_ds, val_ds, input_dim, num_classes)."""
    resolved_path = find_unsw_data(data_path) if not force_synthetic else None

    if resolved_path and os.path.isfile(resolved_path):
        data = np.load(resolved_path, allow_pickle=False)
        X_train = torch.from_numpy(data["X_train"].astype(np.float32))
        y_train = torch.from_numpy(data["y_train"].astype(np.int64))
        X_test = torch.from_numpy(data["X_test"].astype(np.float32))
        y_test = torch.from_numpy(data["y_test"].astype(np.int64))

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_test, y_test)
        actual_input_dim = X_train.shape[1]
        actual_num_classes = len(torch.unique(y_train))
        return train_ds, val_ds, actual_input_dim, actual_num_classes

    # Fallback to synthetic dataset
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2")
    if input_dim < 1 or num_classes < 2:
        raise ValueError("input_dim must be >= 1 and num_classes >= 2")

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, input_dim, generator=generator)
    weights = torch.randn(input_dim, num_classes, generator=generator)
    bias = torch.randn(num_classes, generator=generator) * 0.2
    logits = x @ weights + bias
    y = logits.argmax(dim=1)

    noise_mask = torch.rand(num_samples, generator=generator) < 0.03
    random_labels = torch.randint(num_classes, (num_samples,), generator=generator)
    y[noise_mask] = random_labels[noise_mask]

    permutation = torch.randperm(num_samples, generator=generator)
    x, y = x[permutation], y[permutation]
    split = int(num_samples * train_split)
    split = min(max(split, 1), num_samples - 1)

    train_ds = TensorDataset(x[:split], y[:split])
    val_ds = TensorDataset(x[split:], y[split:])
    return train_ds, val_ds, input_dim, num_classes


def make_loaders(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    distributed: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    """Build data loaders; use DistributedSampler for DDP training."""
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler
