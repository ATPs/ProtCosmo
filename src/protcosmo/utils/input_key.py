"""Helpers for deriving stable input-file keys used by SpecId partitioning."""

from __future__ import annotations

from pathlib import Path

from .mass_file_resolver import SUPPORTED_MASS_FILE_SUFFIXES


def extract_input_file_key(spec_id: str) -> str:
    """Extract input-file key from PIN/Percolator SpecId/PSMId."""

    text = str(spec_id).strip()
    if not text:
        return ""
    parts = text.rsplit("_", maxsplit=3)
    return parts[0] if parts else text


def derive_mass_file_key(mass_file_path: str) -> str:
    """Derive CometPlus-like input key from a mass-file path."""

    name = Path(str(mass_file_path)).name
    lower = name.lower()
    for suffix in sorted(SUPPORTED_MASS_FILE_SUFFIXES, key=len, reverse=True):
        if lower.endswith(suffix):
            return name[: len(name) - len(suffix)]
    return Path(name).stem
