"""Dataset loading and partitioning utilities for Federated Learning."""
import os
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def find_unsw_data(custom_path: Optional[str] = None) -> Optional[str]:
    """Find the path to preprocessed UNSW-NB15 data.npz."""
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
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return None


def partition_indices_iid(n_samples: int, num_clients: int, seed: int = 42) -> List[np.ndarray]:
    """Deterministic IID partitioning of sample indices."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    return np.array_split(perm, num_clients)


def partition_indices_dirichlet(y: np.ndarray, num_clients: int, alpha: float = 0.5, seed: int = 42) -> List[np.ndarray]:
    """Deterministic non-IID partitioning using Dirichlet distribution over classes."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        splits = np.split(cls_idx, cuts)
        for c_id, split in enumerate(splits):
            client_indices[c_id].extend(split.tolist())

    return [np.array(indices, dtype=np.int64) for indices in client_indices]


def load_federated_data(
    data_path: Optional[str] = None,
    client_id: Optional[int] = None,
    num_clients: int = 2,
    partition_type: str = "iid",
    alpha: float = 0.5,
    batch_size: int = 128,
    seed: int = 42,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], DataLoader, int, int]:
    """Load and partition dataset for Federated Learning.

    Returns:
        (train_loader, val_loader, test_loader, input_dim, num_classes)
        If client_id is None, train_loader and val_loader will be None (for server evaluation).
    """
    resolved_path = find_unsw_data(data_path)

    if resolved_path and os.path.isfile(resolved_path):
        # Load preprocessed UNSW-NB15 npz
        data = np.load(resolved_path, allow_pickle=False)
        X_train_raw = data["X_train"].astype(np.float32)
        y_train_raw = data["y_train"].astype(np.int64)
        X_test_raw = data["X_test"].astype(np.float32)
        y_test_raw = data["y_test"].astype(np.int64)

        input_dim = X_train_raw.shape[1]
        num_classes = len(np.unique(y_train_raw))

        # Server global test loader
        test_ds = TensorDataset(torch.from_numpy(X_test_raw), torch.from_numpy(y_test_raw))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        if client_id is None:
            return None, None, test_loader, input_dim, num_classes

        # Partition train set for the clients
        if partition_type == "dirichlet":
            partitions = partition_indices_dirichlet(y_train_raw, num_clients, alpha=alpha, seed=seed)
        else:
            partitions = partition_indices_iid(len(X_train_raw), num_clients, seed=seed)

        client_idx = partitions[client_id % num_clients]
        X_client = X_train_raw[client_idx]
        y_client = y_train_raw[client_idx]

        # Split client's partition into local train (80%) and local val (20%)
        rng = np.random.RandomState(seed + client_id)
        perm = rng.permutation(len(X_client))
        val_size = max(1, int(len(X_client) * 0.2))
        train_idx, val_idx = perm[val_size:], perm[:val_size]

        train_ds = TensorDataset(torch.from_numpy(X_client[train_idx]), torch.from_numpy(y_client[train_idx]))
        val_ds = TensorDataset(torch.from_numpy(X_client[val_idx]), torch.from_numpy(y_client[val_idx]))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, input_dim, num_classes

    else:
        # Fallback synthetic dataset
        input_dim = 32
        num_classes = 2
        n_samples = 10000

        g = torch.Generator().manual_seed(seed)
        X_all = torch.randn(n_samples, input_dim, generator=g)
        w = torch.randn(input_dim, num_classes, generator=g)
        b = torch.randn(num_classes, generator=g) * 0.2
        y_all = (X_all @ w + b).argmax(dim=1)

        test_size = int(n_samples * 0.2)
        train_size = n_samples - test_size

        X_train, y_train = X_all[:train_size], y_all[:train_size]
        X_test, y_test = X_all[train_size:], y_all[train_size:]

        test_ds = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        if client_id is None:
            return None, None, test_loader, input_dim, num_classes

        client_size = train_size // num_clients
        start = (client_id % num_clients) * client_size
        end = start + client_size

        X_client = X_train[start:end]
        y_client = y_train[start:end]

        c_val_size = max(1, int(len(X_client) * 0.2))
        train_ds = TensorDataset(X_client[c_val_size:], y_client[c_val_size:])
        val_ds = TensorDataset(X_client[:c_val_size], y_client[:c_val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader, input_dim, num_classes
