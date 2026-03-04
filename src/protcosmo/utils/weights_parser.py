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


def parse_selected_models(weights_path: str | Path) -> List[LinearModel]:
    """Parse weights file and select numeric rows 1/3/5."""

    path = Path(weights_path)
    current_header: List[str] | None = None
    numeric_rows: List[tuple[List[str], List[float], int]] = []
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
                if current_header is None:
                    raise ValueError(
                        f"Numeric row encountered before header in weights file: {path}"
                    )
                if len(tokens) != len(current_header):
                    raise ValueError(
                        f"Numeric row length {len(tokens)} != header length {len(current_header)} "
                        f"in weights file: {path}"
                    )
                numeric_count += 1
                numeric_rows.append((current_header, [float(x) for x in tokens], numeric_count))
            else:
                current_header = tokens

    required = (1, 3, 5)
    if len(numeric_rows) < max(required):
        raise ValueError(
            f"Weights file has {len(numeric_rows)} numeric rows, but rows 1/3/5 are required: {path}"
        )

    models: List[LinearModel] = []
    for target_row in required:
        header, values, row_index = numeric_rows[target_row - 1]
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
