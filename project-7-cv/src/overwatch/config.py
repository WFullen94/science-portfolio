"""Config loader for the overhead-detection project (same pattern as the rest of the portfolio)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parents[1]  # project-7-cv/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh)


def resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
