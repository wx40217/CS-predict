from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlayerEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 10, F]
        return self.net(x)


class TeamAggregator(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 10, H]
        team_a = x[:, :5, :]  # [B, 5, H]
        team_b = x[:, 5:, :]  # [B, 5, H]
        a = self.proj(team_a).mean(dim=1)
        b = self.proj(team_b).mean(dim=1)
        return a, b


class MapPickWinModel(nn.Module):
    def __init__(self, player_in_dim: int, player_hidden: int, team_hidden: int, num_maps: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.player_encoder = PlayerEncoder(player_in_dim, player_hidden, dropout)
        self.aggregator = TeamAggregator(player_hidden)
        self.fuse = nn.Sequential(
            nn.Linear(player_hidden * 2, team_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Heads
        self.pick_head = nn.Linear(team_hidden, num_maps)
        self.win_head = nn.Linear(team_hidden, num_maps * 2)

    def forward(self, players: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # players: [B, 10, F]
        h = self.player_encoder(players)
        a, b = self.aggregator(h)
        ab = torch.cat([a, b], dim=-1)
        t = self.fuse(ab)
        pick_logits = self.pick_head(t)  # [B, M]
        win_logits = self.win_head(t).view(t.size(0), -1, 2)  # [B, M, 2]
        win_prob = torch.sigmoid(win_logits)
        return pick_logits, win_prob


def pick_loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target)


def win_loss_fn(pred_prob: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # pred_prob: [B, M, 2]; target: [B, M, 2] with -1 for masked; mask: [B, M]
    # Use BCE on available map row
    valid = mask > 0.5
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_prob.device)
    pred = pred_prob[valid]
    tgt = target[valid]
    loss = F.binary_cross_entropy(pred, tgt)
    return loss