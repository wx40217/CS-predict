from __future__ import annotations

import argparse
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import add_common_args, load_config
from src.data.dataset import MapPickDataset, split_dataset
from src.model import MapPickWinModel, pick_loss_fn, win_loss_fn
from src.utils import get_device, save_checkpoint, set_seed


def train_loop(cfg: Dict) -> None:
    set_seed(cfg.get("seed", 42))
    device = get_device()

    processed_dir = cfg["paths"]["processed_dir"]
    parquet_path = os.path.join(processed_dir, "samples.parquet")
    ds = MapPickDataset(parquet_path)
    train_ds, val_ds, test_ds = split_dataset(ds, val_ratio=0.1, test_ratio=0.1, seed=cfg.get("seed", 42))

    dl_train = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    num_maps = len(cfg["model"]["maps"])

    model = MapPickWinModel(
        player_in_dim=ds.player_feat_dim,
        player_hidden=cfg["model"]["player_encoder_dim"],
        team_hidden=cfg["model"]["team_encoder_dim"],
        num_maps=num_maps,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["train"].get("precision", "amp") == "amp" and device.type == "cuda"))

    best_val = float("inf")
    steps = 0

    for epoch in range(cfg["train"]["max_epochs"]):
        model.train()
        pbar = tqdm(dl_train, desc=f"epoch {epoch}")
        for batch in pbar:
            players = batch["players"].to(device)
            map_label = batch["map_label"].to(device)
            win_targets = batch["win_targets"].to(device)
            mask = batch["supervise_mask"].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pick_logits, win_prob = model(players)
                loss_pick = pick_loss_fn(pick_logits, map_label)
                loss_win = win_loss_fn(win_prob, win_targets, mask)
                loss = loss_pick + 0.5 * loss_win

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"]) 
            scaler.step(opt)
            scaler.update()

            steps += 1
            if steps % cfg["train"]["log_every_n_steps"] == 0:
                pbar.set_postfix({"loss": float(loss.item()), "pick": float(loss_pick.item()), "win": float(loss_win.item())})

        # validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in dl_val:
                players = batch["players"].to(device)
                map_label = batch["map_label"].to(device)
                win_targets = batch["win_targets"].to(device)
                mask = batch["supervise_mask"].to(device)
                pick_logits, win_prob = model(players)
                loss_pick = pick_loss_fn(pick_logits, map_label)
                loss_win = win_loss_fn(win_prob, win_targets, mask)
                val_loss += float((loss_pick + 0.5 * loss_win).item())
                val_count += 1
        val_avg = val_loss / max(1, val_count)

        if val_avg < best_val:
            best_val = val_avg
            ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], "best.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            save_checkpoint({"model_state": model.state_dict(), "cfg": cfg, "val_loss": best_val}, ckpt_path)

        print(f"epoch {epoch} val_loss={val_avg:.4f} best={best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    train_loop(cfg)