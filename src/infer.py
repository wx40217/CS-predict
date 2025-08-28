from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch

from src.data.preprocess import extract_player_features, _parse_birthdate, _compute_age, MAPS
from src.model import MapPickWinModel
from src.utils import get_device, load_checkpoint


def load_player_cache(raw_dir: str) -> Dict[str, Dict[str, Any]]:
    players_path = os.path.join(raw_dir, "players.jsonl")
    cache: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(players_path):
        return cache
    with open(players_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = str(obj.get("id") or obj.get("playerId") or obj.get("player_id") or "")
            name = str(obj.get("nickname") or obj.get("name") or "").lower()
            if pid:
                cache[pid] = obj
            if name:
                cache[name] = obj
    return cache


def features_for_players(player_names: List[str], cache: Dict[str, Dict[str, Any]], numeric_keys: List[str]) -> np.ndarray:
    ref_date = torch.datetime.date.today() if hasattr(torch, "datetime") else None
    if ref_date is None:
        import datetime as _dt
        ref_date = _dt.datetime.utcnow().date()

    feats: List[float] = []
    for name in player_names:
        key = name.lower()
        pobj = cache.get(key, {"nickname": name})
        vec = extract_player_features(pobj, numeric_keys=numeric_keys, ref_date=ref_date, include_age=True, include_age_squared=True)
        feats.extend(vec)
    return np.array(feats, dtype=np.float32)


def run(ckpt: str, players_file: str, raw_dir: str, player_feat_dim_hint: int = 32) -> None:
    with open(players_file, "r", encoding="utf-8") as f:
        query = json.load(f)

    team_a: List[str] = query["team_a"]
    team_b: List[str] = query["team_b"]
    assert len(team_a) == 5 and len(team_b) == 5, "Expect 5 players per team"

    cache = load_player_cache(raw_dir)
    numeric_keys = ["rating", "kd", "impact", "dpr", "kast"]

    a_vec = features_for_players(team_a, cache, numeric_keys)
    b_vec = features_for_players(team_b, cache, numeric_keys)

    # Age features add 2 dims if present; if cache lacks numeric fields, vectors are smaller; to be safe, pad/truncate to hint
    per_player = len(a_vec) // 5
    target_pp = player_feat_dim_hint
    def pad_or_trim(vec: np.ndarray) -> np.ndarray:
        if len(vec) < 5 * target_pp:
            pad = np.zeros(5 * target_pp - len(vec), dtype=np.float32)
            return np.concatenate([vec, pad], axis=0)
        if len(vec) > 5 * target_pp:
            return vec[: 5 * target_pp]
        return vec

    a_vec = pad_or_trim(a_vec)
    b_vec = pad_or_trim(b_vec)

    feats = np.concatenate([a_vec, b_vec], axis=0).reshape(1, 10, target_pp)

    device = get_device()
    state = load_checkpoint(ckpt, map_location=device)
    cfg = state.get("cfg", {})
    num_maps = len(cfg.get("model", {}).get("maps", MAPS))

    model = MapPickWinModel(
        player_in_dim=target_pp,
        player_hidden=cfg.get("model", {}).get("player_encoder_dim", 128),
        team_hidden=cfg.get("model", {}).get("team_encoder_dim", 256),
        num_maps=num_maps,
        dropout=cfg.get("model", {}).get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(state["model_state"], strict=False)
    model.eval()

    with torch.no_grad():
        players = torch.from_numpy(feats).to(device)
        pick_logits, win_prob = model(players)
        pick_prob = torch.softmax(pick_logits, dim=-1)[0].cpu().numpy()
        win_prob = win_prob[0].cpu().numpy()

    top_map_idx = int(pick_prob.argmax())
    maps = cfg.get("model", {}).get("maps", MAPS)
    top_map_name = maps[top_map_idx]

    out = {
        "predicted_pick": top_map_name,
        "pick_probabilities": {maps[i]: float(pick_prob[i]) for i in range(num_maps)},
        "win_probabilities": {maps[i]: {"team_a": float(win_prob[i, 0]), "team_b": float(win_prob[i, 1])} for i in range(num_maps)},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--players_file", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    args = parser.parse_args()
    run(ckpt=args.ckpt, players_file=args.players_file, raw_dir=args.raw_dir)