from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

MAPS = ["ancient", "anubis", "inferno", "mirage", "nuke", "overpass", "vertigo"]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _norm_map_name(name: str) -> Optional[str]:
    if not name:
        return None
    n = name.lower().strip()
    n = n.replace("de_", "")
    for m in MAPS:
        if m in n:
            return m
    return None


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_birthdate(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y.%m.%d"]:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _compute_age(birth: Optional[date], ref: date) -> Optional[float]:
    if not birth:
        return None
    return (ref - birth).days / 365.25


NUMERIC_KEYS_DEFAULT = ["rating", "kd", "impact", "dpr", "kast"]


def build_player_index(players: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for p in players:
        pid = str(p.get("id") or p.get("playerId") or p.get("player_id") or "")
        if not pid:
            # try to index by name as fallback
            name = str(p.get("nickname") or p.get("name") or p.get("playerName") or "").lower()
            if not name:
                continue
            idx[name] = p
            continue
        idx[pid] = p
    return idx


def extract_player_features(
    player_obj: Dict[str, Any],
    numeric_keys: List[str],
    ref_date: date,
    age_clip: Tuple[int, int] = (15, 40),
    include_age: bool = True,
    include_age_squared: bool = True,
) -> List[float]:
    feats: List[float] = []
    for k in numeric_keys:
        v = player_obj.get(k)
        try:
            feats.append(float(v))
        except (TypeError, ValueError):
            feats.append(0.0)

    if include_age:
        bdate = _parse_birthdate(
            player_obj.get("birthDate")
            or player_obj.get("birthday")
            or player_obj.get("born")
            or player_obj.get("dob")
        )
        age_val = _compute_age(bdate, ref_date)
        if age_val is None:
            age_val = 22.0
        age_val = _clip(age_val, age_clip[0], age_clip[1])
        feats.append(age_val)
        if include_age_squared:
            feats.append(age_val * age_val)

    return feats


def pick_rosters(match: Dict[str, Any]) -> Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    # Try multiple known shapes
    for a_key, b_key in [
        ("team1Players", "team2Players"),
        ("team_a_players", "team_b_players"),
        ("playersA", "playersB"),
        ("lineupA", "lineupB"),
    ]:
        a = match.get(a_key)
        b = match.get(b_key)
        if isinstance(a, list) and isinstance(b, list):
            return a, b

    # Some payloads store nested under team objects
    for tk in ["team1", "team2", "team_a", "team_b"]:
        t = match.get(tk)
        if isinstance(t, dict) and isinstance(t.get("players"), list):
            # fallback: cannot ensure both teams present here
            pass

    return None


def determine_played_map(match: Dict[str, Any]) -> Optional[str]:
    # Seek a field indicating the map actually played
    for key in ["map", "mapName", "played_map", "veto_map", "map_name"]:
        m = match.get(key)
        if isinstance(m, str):
            norm = _norm_map_name(m)
            if norm:
                return norm
    # Some matches have a list of maps; choose first completed
    for key in ["maps", "vetoes", "vetoesList"]:
        maps = match.get(key)
        if isinstance(maps, list):
            for item in maps:
                if isinstance(item, dict):
                    mn = _norm_map_name(str(item.get("name") or item.get("map") or item.get("picked") or ""))
                    if mn:
                        return mn
                elif isinstance(item, str):
                    mn = _norm_map_name(item)
                    if mn:
                        return mn
    return None


def determine_winner(match: Dict[str, Any]) -> Optional[int]:
    # Return 0 if team A, 1 if team B, else None
    for key in ["winner", "winnerTeam", "winner_team"]:
        w = match.get(key)
        if isinstance(w, int):
            if w in (0, 1):
                return w
        if isinstance(w, str):
            if w.lower() in ("a", "team_a", "team1"):
                return 0
            if w.lower() in ("b", "team_b", "team2"):
                return 1
        if isinstance(w, dict):
            # if score object exists, try parse
            pass
    # Try score
    for key in ["score", "result", "finalScore"]:
        s = match.get(key)
        if isinstance(s, dict):
            a = s.get("a") or s.get("team1") or s.get("team_a")
            b = s.get("b") or s.get("team2") or s.get("team_b")
            try:
                a = int(a)
                b = int(b)
                if a > b:
                    return 0
                if b > a:
                    return 1
            except Exception:
                pass
    return None


def run(raw_dir: str, out_dir: str, min_date: Optional[str], numeric_keys: Optional[List[str]] = None, include_age: bool = True, include_age_squared: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)

    matches = _read_jsonl(os.path.join(raw_dir, "matches.jsonl"))
    players = _read_jsonl(os.path.join(raw_dir, "players.jsonl"))
    player_index = build_player_index(players)

    if numeric_keys is None:
        numeric_keys = NUMERIC_KEYS_DEFAULT

    ref_min_date: Optional[date] = None
    if min_date:
        try:
            ref_min_date = datetime.strptime(min_date, "%Y-%m-%d").date()
        except ValueError:
            ref_min_date = None

    rows: List[Dict[str, Any]] = []

    for m in tqdm(matches, desc="preprocess matches"):
        # date filter if available
        if ref_min_date:
            for key in ["date", "time", "matchTime", "playedAt"]:
                val = m.get(key)
                if isinstance(val, str):
                    try:
                        md = datetime.fromisoformat(val.replace("Z", "+00:00")).date()
                        if md < ref_min_date:
                            continue
                    except Exception:
                        pass

        rosters = pick_rosters(m)
        if not rosters:
            continue
        team_a, team_b = rosters
        if len(team_a) < 5 or len(team_b) < 5:
            continue

        map_played = determine_played_map(m)
        if map_played is None or map_played not in MAPS:
            continue
        map_idx = MAPS.index(map_played)

        winner = determine_winner(m)
        if winner is None:
            continue

        # Build features for first 5 players of each team
        ref_date = datetime.utcnow().date()
        feats: List[float] = []
        players_used: List[str] = []

        def player_obj_from_roster(roster_item: Dict[str, Any]) -> Dict[str, Any]:
            pid = str(roster_item.get("id") or roster_item.get("playerId") or roster_item.get("player_id") or "")
            name = str(roster_item.get("nickname") or roster_item.get("name") or "").lower()
            if pid and pid in player_index:
                return player_index[pid]
            if name and name in player_index:
                return player_index[name]
            return roster_item

        for roster in [team_a[:5], team_b[:5]]:
            for r in roster:
                pobj = player_obj_from_roster(r)
                vec = extract_player_features(
                    pobj,
                    numeric_keys=numeric_keys,
                    ref_date=ref_date,
                    age_clip=(15, 40),
                    include_age=include_age,
                    include_age_squared=include_age_squared,
                )
                feats.extend(vec)
                nickname = str(pobj.get("nickname") or pobj.get("name") or "unknown")
                players_used.append(nickname)

        team_a_win = 1 if winner == 0 else 0
        team_b_win = 1 - team_a_win

        win_targets = np.full((len(MAPS), 2), -1.0, dtype=np.float32)
        win_targets[map_idx, 0] = float(team_a_win)
        win_targets[map_idx, 1] = float(team_b_win)

        row: Dict[str, Any] = {
            "features": np.array(feats, dtype=np.float32),
            "map_idx": map_idx,
            "win_targets": win_targets,
            "players": players_used,
        }
        rows.append(row)

    if not rows:
        print("No rows produced; please ensure raw data exists and schema mapping is correct.")
        return

    # Convert to DataFrame with fixed-size arrays
    feat_len = len(rows[0]["features"])  # 10 * F
    df = pd.DataFrame(
        {
            "features": [r["features"] for r in rows],
            "map_idx": [r["map_idx"] for r in rows],
            "win_targets": [r["win_targets"] for r in rows],
            "players": ["|".join(r["players"]) for r in rows],
        }
    )

    # Save as Parquet with pyarrow
    out_path = os.path.join(out_dir, "samples.parquet")
    df.to_parquet(out_path, engine="pyarrow")
    meta = {
        "num_rows": len(df),
        "feature_length": feat_len,
        "maps": MAPS,
        "note": "features are flattened 10 x F vectors; win_targets is len(maps) x 2 with -1 for masked maps",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--min_date", type=str, default=None)
    args = parser.parse_args()
    run(raw_dir=args.raw_dir, out_dir=args.out_dir, min_date=args.min_date)