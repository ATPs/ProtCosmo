"""ProtCosmo package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _fallback_version() -> str:
    version_path = Path(__file__).resolve().parents[2] / "VERSION"
    if version_path.exists():
        text = version_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return "0.0.0"


try:
    __version__ = version("protcosmo")
except PackageNotFoundError:
    __version__ = _fallback_version()
