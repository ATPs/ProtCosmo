"""Percolator reference table readers and score-based estimators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


SCORE_CANDIDATES = ("score", "mokapot_score", "mokapot score")
QVALUE_CANDIDATES = ("q-value", "q_value", "mokapot q-value", "mokapot_q_value")
PEP_CANDIDATES = (
    "posterior_error_prob",
    "posterior_error_probability",
    "pep",
    "PEP",
    "mokapot_posterior_error_prob",
)


def _find_column(columns: Iterable[str], candidates: Tuple[str, ...]) -> str:
    lowered = {str(col).lower(): str(col) for col in columns}
    for name in candidates:
        hit = lowered.get(name.lower())
        if hit is not None:
            return hit
    raise ValueError(f"Unable to find any of required columns: {candidates}")


def _read_table(path: Path) -> pd.DataFrame:
    lower = path.name.lower()
    if lower.endswith(".parquet") or lower.endswith(".parquet.gz"):
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", comment="#")


@dataclass
class ReferenceLookup:
    """Sorted score lookup table for nearest smaller score estimation."""

    source_path: Path
    scores: np.ndarray
    q_values: np.ndarray
    pep_values: np.ndarray

    def estimate_array(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = np.searchsorted(self.scores, scores, side="right") - 1
        fallback = idx < 0
        safe_idx = idx.copy()
        safe_idx[fallback] = 0
        est_q = self.q_values[safe_idx]
        est_pep = self.pep_values[safe_idx]
        matched = self.scores[safe_idx]
        est_q[fallback] = 1.0
        est_pep[fallback] = 1.0
        matched[fallback] = np.nan
        return est_q, est_pep, matched, fallback


def build_reference_lookup(path: str | Path) -> ReferenceLookup:
    table_path = Path(path)
    df = _read_table(table_path)
    score_col = _find_column(df.columns, SCORE_CANDIDATES)
    q_col = _find_column(df.columns, QVALUE_CANDIDATES)
    pep_col = _find_column(df.columns, PEP_CANDIDATES)

    normalized = pd.DataFrame(
        {
            "score": pd.to_numeric(df[score_col], errors="coerce"),
            "q_value": pd.to_numeric(df[q_col], errors="coerce"),
            "pep": pd.to_numeric(df[pep_col], errors="coerce"),
        }
    ).dropna()
    if normalized.empty:
        raise ValueError(f"Reference file has no usable rows: {table_path}")

    # Keep best row for duplicate scores to avoid unstable lookup.
    normalized = normalized.sort_values(by=["score", "q_value", "pep"], ascending=[True, True, True])
    normalized = normalized.drop_duplicates(subset=["score"], keep="first")
    scores = normalized["score"].to_numpy(dtype=np.float64, copy=False)
    q_values = normalized["q_value"].to_numpy(dtype=np.float64, copy=False)
    pep_values = normalized["pep"].to_numpy(dtype=np.float64, copy=False)
    return ReferenceLookup(
        source_path=table_path,
        scores=scores,
        q_values=q_values,
        pep_values=pep_values,
    )
