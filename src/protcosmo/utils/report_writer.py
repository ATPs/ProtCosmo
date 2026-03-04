"""Output writing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=False)


def write_json(payload: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def write_warnings(warnings: Iterable[str], path: Path) -> None:
    ensure_dir(path.parent)
    lines = [str(w).rstrip() for w in warnings if str(w).strip()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
