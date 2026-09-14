"""Configuration loading and repository path helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_feature_policy(path: Path | None = None) -> dict[str, Any]:
    """Load the declarative feature policy used by graph construction."""
    policy_path = path or repository_root() / "config" / "features.yaml"
    with policy_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
