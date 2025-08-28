from __future__ import annotations

import argparse
import copy
from typing import Any, Dict

import yaml


def _set_by_path(d: Dict[str, Any], key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "_" in value:
            # avoid numeric coercion for ids with underscores
            return value
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(default_path: str, overrides: list[str] | None = None) -> Dict[str, Any]:
    with open(default_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    if overrides:
        cfg = copy.deepcopy(cfg)
        for ov in overrides:
            if "=" not in ov:
                continue
            key, val = ov.split("=", 1)
            _set_by_path(cfg, key, _coerce(val))
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    return parser