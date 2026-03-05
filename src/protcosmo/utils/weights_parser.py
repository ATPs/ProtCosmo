"""Percolator weight-file parsing for static scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


_SPLIT_RE = re.compile(r"[\t ]+")


@dataclass
class LinearModel:
    """A linear scoring model: score = w^T x + b."""

    feature_names: List[str]
    weights: np.ndarray
    intercept: float
    numeric_row_index: int


def _tokenize(line: str) -> List[str]:
    return [token for token in _SPLIT_RE.split(line.strip()) if token]


def _is_float_token(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _select_model_rows(
    blocks: Sequence[tuple[List[str], Sequence[tuple[List[float], int]]]],
    source_path: Path,
) -> List[tuple[List[str], List[float], int]]:
    """Select three model rows, preferring raw rows in per-bin weight blocks."""

    # Percolator CV export commonly repeats:
    # header -> normalized row -> raw row
    # for each bin. In this layout we should use the raw row (second numeric row).
    selected_by_block: List[tuple[List[str], List[float], int]] = []
    for header, rows in blocks:
        if not rows:
            continue
        raw_row_pos = 1 if len(rows) >= 2 else 0
        values, row_index = rows[raw_row_pos]
        selected_by_block.append((header, values, row_index))
        if len(selected_by_block) == 3:
            return selected_by_block

    flat_rows: List[tuple[List[str], List[float], int]] = [
        (header, values, row_index)
        for header, rows in blocks
        for values, row_index in rows
    ]
    if len(flat_rows) >= 6:
        return [flat_rows[1], flat_rows[3], flat_rows[5]]
    if len(flat_rows) >= 5:
        return [flat_rows[0], flat_rows[2], flat_rows[4]]
    if len(flat_rows) >= 3:
        return flat_rows[:3]

    raise ValueError(
        f"Weights file has {len(flat_rows)} numeric rows, but 3 model rows are required: {source_path}"
    )


def parse_selected_models(weights_path: str | Path) -> List[LinearModel]:
    """Parse weights file and select three models, preferring raw rows (2/4/6)."""

    path = Path(weights_path)
    current_header: List[str] | None = None
    current_block_rows: List[tuple[List[float], int]] | None = None
    blocks: List[tuple[List[str], List[tuple[List[float], int]]]] = []
    numeric_count = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = _tokenize(line)
            if not tokens:
                continue
            if all(_is_float_token(token) for token in tokens):
                if current_header is None or current_block_rows is None:
                    raise ValueError(
                        f"Numeric row encountered before header in weights file: {path}"
                    )
                if len(tokens) != len(current_header):
                    raise ValueError(
                        f"Numeric row length {len(tokens)} != header length {len(current_header)} "
                        f"in weights file: {path}"
                    )
                numeric_count += 1
                current_block_rows.append(([float(x) for x in tokens], numeric_count))
            else:
                current_header = tokens
                current_block_rows = []
                blocks.append((current_header, current_block_rows))

    selected_rows = _select_model_rows(blocks, path)

    models: List[LinearModel] = []
    for header, values, row_index in selected_rows:
        if "m0" not in header:
            raise ValueError(f"Column 'm0' was not found in weights header for row {row_index}: {path}")
        m0_idx = header.index("m0")
        intercept = values[m0_idx]
        feature_names: List[str] = []
        weight_values: List[float] = []
        for idx, feature in enumerate(header):
            if idx == m0_idx:
                continue
            feature_names.append(feature)
            weight_values.append(values[idx])
        models.append(
            LinearModel(
                feature_names=feature_names,
                weights=np.asarray(weight_values, dtype=np.float64),
                intercept=float(intercept),
                numeric_row_index=row_index,
            )
        )
    return models


def validate_models_feature_alignment(models: Sequence[LinearModel]) -> None:
    """Validate that all selected models share the same feature ordering."""

    if not models:
        raise ValueError("No models were parsed.")
    ref = models[0].feature_names
    for idx, model in enumerate(models[1:], start=2):
        if model.feature_names != ref:
            raise ValueError(f"Selected model {idx} does not match feature order of model 1.")
