from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Set

import requests
from tqdm import tqdm

BASE_URL = "https://hltv-api.vercel.app"


class CachedWriter:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.matches_path = os.path.join(out_dir, "matches.jsonl")
        self.players_path = os.path.join(out_dir, "players.jsonl")
        self.teams_path = os.path.join(out_dir, "teams.jsonl")
        self._open_files()

    def _open_files(self) -> None:
        self._matches = open(self.matches_path, "a", encoding="utf-8")
        self._players = open(self.players_path, "a", encoding="utf-8")
        self._teams = open(self.teams_path, "a", encoding="utf-8")

    def close(self) -> None:
        self._matches.close()
        self._players.close()
        self._teams.close()

    def write_match(self, obj: Dict[str, Any]) -> None:
        self._matches.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def write_player(self, obj: Dict[str, Any]) -> None:
        self._players.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def write_team(self, obj: Dict[str, Any]) -> None:
        self._teams.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get(session: requests.Session, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 3, backoff: float = 1.5) -> Optional[Dict[str, Any]]:
    url = f"{BASE_URL}{path}"
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                return resp.json()
            time.sleep(backoff * (attempt + 1))
        except requests.RequestException:
            time.sleep(backoff * (attempt + 1))
    return None


def discover_recent_match_ids(session: requests.Session, max_matches: int) -> List[str]:
    ids: List[str] = []
    # Try a few plausible endpoints; collect whatever works
    for path in ["/matches", "/api/matches", "/api/results", "/results"]:
        data = get(session, path)
        if isinstance(data, list):
            for item in data:
                mid = str(item.get("id") or item.get("matchId") or item.get("match_id") or "")
                if mid:
                    ids.append(mid)
            if ids:
                break
    # de-duplicate and truncate
    seen: Set[str] = set()
    unique_ids: List[str] = []
    for mid in ids:
        if mid in seen:
            continue
        unique_ids.append(mid)
        seen.add(mid)
        if len(unique_ids) >= max_matches:
            break
    return unique_ids


def fetch_match_detail(session: requests.Session, match_id: str) -> Optional[Dict[str, Any]]:
    for path_tpl in [
        "/match/{id}",
        "/api/match/{id}",
        "/matches/{id}",
        "/api/matches/{id}",
    ]:
        data = get(session, path_tpl.format(id=match_id))
        if data:
            data["_source_url"] = f"{BASE_URL}{path_tpl.format(id=match_id)}"
            return data
    return None


def fetch_player_detail(session: requests.Session, player_id: str) -> Optional[Dict[str, Any]]:
    for path_tpl in [
        "/player/{id}",
        "/api/player/{id}",
        "/players/{id}",
        "/api/players/{id}",
    ]:
        data = get(session, path_tpl.format(id=player_id))
        if data:
            return data
    return None


def fetch_team_detail(session: requests.Session, team_id: str) -> Optional[Dict[str, Any]]:
    for path_tpl in [
        "/team/{id}",
        "/api/team/{id}",
        "/teams/{id}",
        "/api/teams/{id}",
    ]:
        data = get(session, path_tpl.format(id=team_id))
        if data:
            return data
    return None


def run(out_dir: str, max_matches: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()
    writer = CachedWriter(out_dir)

    try:
        match_ids = discover_recent_match_ids(session, max_matches=max_matches)
        if not match_ids:
            print("No match ids discovered; try adjusting endpoints or check network.")
            return

        for mid in tqdm(match_ids, desc="fetch matches"):
            m = fetch_match_detail(session, mid)
            if not m:
                continue
            writer.write_match(m)

            # Collect likely team and player ids from the match payload
            team_ids: Set[str] = set()
            player_ids: Set[str] = set()

            for key in ["team1", "team2", "team_a", "team_b", "teams"]:
                if key in m and isinstance(m[key], dict):
                    tid = str(m[key].get("id") or m[key].get("teamId") or m[key].get("team_id") or "")
                    if tid:
                        team_ids.add(tid)
                elif key in m and isinstance(m[key], list):
                    for t in m[key]:
                        if isinstance(t, dict):
                            tid = str(t.get("id") or t.get("teamId") or t.get("team_id") or "")
                            if tid:
                                team_ids.add(tid)

            for roster_key in ["players", "lineup", "team1Players", "team2Players", "team_a_players", "team_b_players"]:
                roster = m.get(roster_key)
                if isinstance(roster, list):
                    for p in roster:
                        if isinstance(p, dict):
                            pid = str(p.get("id") or p.get("playerId") or p.get("player_id") or "")
                            if pid:
                                player_ids.add(pid)

            for tid in team_ids:
                t = fetch_team_detail(session, tid)
                if t:
                    writer.write_team(t)

            for pid in player_ids:
                p = fetch_player_detail(session, pid)
                if p:
                    writer.write_player(p)

    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_matches", type=int, default=2000)
    args = parser.parse_args()
    run(out_dir=args.out_dir, max_matches=args.max_matches)