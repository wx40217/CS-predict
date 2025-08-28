from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split


class MapPickDataset(Dataset):
    def __init__(self, parquet_path: str) -> None:
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        # Convert columns to numpy arrays
        self.features = np.stack(self.df["features"].to_numpy())
        self.map_idx = self.df["map_idx"].to_numpy().astype(np.int64)
        self.win_targets = np.stack(self.df["win_targets"].to_numpy()).astype(np.float32)
        self.num_maps = self.win_targets.shape[1] if self.win_targets.ndim == 3 else 7

        # Infer player feature dimension
        total_feat = self.features.shape[1]
        # Assume 10 players
        assert total_feat % 10 == 0, "Feature length must be divisible by 10"
        self.player_feat_dim = total_feat // 10

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feats = self.features[idx]
        feats = feats.reshape(10, self.player_feat_dim)
        map_label = int(self.map_idx[idx])
        wt = self.win_targets[idx]  # shape [num_maps, 2] with -1 for masked
        supervise_mask = (wt[:, 0] >= 0).astype(np.float32)
        return {
            "players": torch.from_numpy(feats).float(),
            "map_label": torch.tensor(map_label, dtype=torch.long),
            "win_targets": torch.from_numpy(wt).float(),
            "supervise_mask": torch.from_numpy(supervise_mask).float(),
        }


def split_dataset(ds: MapPickDataset, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    total = len(ds)
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    train_size = total - val_size - test_size
    return random_split(ds, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))