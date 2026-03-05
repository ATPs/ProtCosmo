"""Percolator reference table readers and score-based estimators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
PSMID_CANDIDATES = ("PSMId", "psm_id", "psmid", "PSM_ID")


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


@dataclass
class PartitionedReferenceLookup:
    """Reference lookup with optional per-input-file partitions (derived from PSMId prefix)."""

    source_path: Path
    global_lookup: ReferenceLookup
    normalized_table: pd.DataFrame
    _partition_cache: Dict[str, ReferenceLookup]

    def lookup_for_input_key(self, input_key: str) -> ReferenceLookup:
        key = str(input_key).strip()
        if not key or "input_file_key" not in self.normalized_table.columns:
            return self.global_lookup
        if key not in self._partition_cache:
            subset = self.normalized_table[self.normalized_table["input_file_key"] == key]
            if subset.empty:
                self._partition_cache[key] = self.global_lookup
            else:
                self._partition_cache[key] = _build_lookup_from_normalized(self.source_path, subset)
        return self._partition_cache[key]


def _normalize_reference_table(path: Path, table: pd.DataFrame) -> pd.DataFrame:
    score_col = _find_column(table.columns, SCORE_CANDIDATES)
    q_col = _find_column(table.columns, QVALUE_CANDIDATES)
    pep_col = _find_column(table.columns, PEP_CANDIDATES)

    normalized = pd.DataFrame(
        {
            "score": pd.to_numeric(table[score_col], errors="coerce"),
            "q_value": pd.to_numeric(table[q_col], errors="coerce"),
            "pep": pd.to_numeric(table[pep_col], errors="coerce"),
        }
    )

    try:
        psm_id_col = _find_column(table.columns, PSMID_CANDIDATES)
        psm_id_series = table[psm_id_col].astype(str)
        normalized["psm_id"] = psm_id_series
        normalized["input_file_key"] = psm_id_series.str.split("_", n=1).str[0]
    except ValueError:
        pass

    normalized = normalized.dropna(subset=["score", "q_value", "pep"]).copy()
    if normalized.empty:
        raise ValueError(f"Reference file has no usable rows: {path}")
    return normalized


def _build_lookup_from_normalized(path: Path, normalized: pd.DataFrame) -> ReferenceLookup:
    compact = normalized.loc[:, ["score", "q_value", "pep"]].copy()
    # Keep best row for duplicate scores to avoid unstable lookup.
    compact = compact.sort_values(by=["score", "q_value", "pep"], ascending=[True, True, True])
    compact = compact.drop_duplicates(subset=["score"], keep="first")
    scores = compact["score"].to_numpy(dtype=np.float64, copy=False)
    q_values = compact["q_value"].to_numpy(dtype=np.float64, copy=False)
    pep_values = compact["pep"].to_numpy(dtype=np.float64, copy=False)
    return ReferenceLookup(
        source_path=path,
        scores=scores,
        q_values=q_values,
        pep_values=pep_values,
    )


def build_reference_lookup(path: str | Path) -> ReferenceLookup:
    table_path = Path(path)
    normalized = _normalize_reference_table(table_path, _read_table(table_path))
    return _build_lookup_from_normalized(table_path, normalized)


def build_partitioned_reference_lookup(path: str | Path) -> PartitionedReferenceLookup:
    table_path = Path(path)
    normalized = _normalize_reference_table(table_path, _read_table(table_path))
    return PartitionedReferenceLookup(
        source_path=table_path,
        global_lookup=_build_lookup_from_normalized(table_path, normalized),
        normalized_table=normalized,
        _partition_cache={},
    )
