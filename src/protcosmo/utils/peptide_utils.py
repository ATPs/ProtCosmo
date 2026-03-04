"""Peptide normalization helpers for summary outputs."""

from __future__ import annotations

import re
from typing import List


_BRACKET_RE = re.compile(r"\[[^\]]*\]")


def _dot_positions_outside_brackets(text: str) -> List[int]:
    positions: List[int] = []
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == "." and depth == 0:
            positions.append(idx)
    return positions


def strip_pin_flanks(peptide: str) -> str:
    """Strip flanking residues from PIN peptide format A.PEPTIDE.B."""

    text = str(peptide).strip()
    dots = _dot_positions_outside_brackets(text)
    if len(dots) < 2:
        return text
    left, right = dots[0], dots[1]
    if right <= left:
        return text
    return text[left + 1 : right]


def normalize_modified_peptide(peptide: str) -> str:
    """Canonical modified sequence used in modified-level summary."""

    return strip_pin_flanks(peptide).strip()


def collapse_to_unmodified(peptide: str) -> str:
    """Collapse modified peptide to unmodified amino-acid sequence."""

    core = normalize_modified_peptide(peptide)
    core = _BRACKET_RE.sub("", core)
    core = core.replace("-", "")
    if core.startswith("n"):
        core = core[1:]
    if core.endswith("c"):
        core = core[:-1]
    letters = [ch.upper() for ch in core if ch.isalpha()]
    return "".join(letters)
