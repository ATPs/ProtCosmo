"""Scoring helpers for PIN candidates."""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .weights_parser import LinearModel

_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_CHARGE_ONEHOT_KEY_RE = re.compile(r"^charge([0-9]+)$")

# Normalized feature-name aliases used when matching model features to PIN columns.
_FEATURE_KEY_ALIASES: Dict[str, Sequence[str]] = {
    "chargen": (
        "charge",
        "charge_state",
        "chargestate",
        "precursor_charge",
        "precursorcharge",
        "z",
    ),
    "psmscore": (
        "score",
        "rawscore",
        "search_score",
        "searchscore",
    ),
}


def _normalize_name(name: str) -> str:
    return _NON_ALNUM_RE.sub("", str(name).strip().lower())


def _build_normalized_column_map(columns: Sequence[str]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for col in columns:
        normalized.setdefault(_normalize_name(col), []).append(col)
    return normalized


def _candidate_keys(feature_name: str) -> List[str]:
    base_key = _normalize_name(feature_name)
    keys = {base_key}
    for alias in _FEATURE_KEY_ALIASES.get(base_key, ()):
        keys.add(_normalize_name(alias))
    return [key for key in keys if key]


def _resolve_feature_column(
    feature_name: str,
    normalized_column_map: Dict[str, List[str]],
) -> str | None:
    candidate_columns = normalized_column_map.get(_normalize_name(feature_name), [])
    if len(candidate_columns) == 1:
        return candidate_columns[0]
    if len(candidate_columns) > 1 and feature_name in candidate_columns:
        return feature_name

    for key in _candidate_keys(feature_name):
        alias_columns = normalized_column_map.get(key, [])
        if len(alias_columns) == 1:
            return alias_columns[0]

    feature_key = _normalize_name(feature_name)
    if len(feature_key) >= 6:
        containment_matches = []
        for col_key, columns in normalized_column_map.items():
            if col_key.endswith(feature_key) or feature_key.endswith(col_key):
                containment_matches.extend(columns)
        unique_matches = sorted(set(containment_matches))
        if len(unique_matches) == 1:
            return unique_matches[0]

    return None


def _infer_charge_n_from_onehot(
    data: pd.DataFrame,
    normalized_column_map: Dict[str, List[str]],
) -> pd.Series | None:
    charge_columns: Dict[int, str] = {}
    for key, columns in normalized_column_map.items():
        match = _CHARGE_ONEHOT_KEY_RE.fullmatch(key)
        if match and len(columns) == 1:
            charge_columns[int(match.group(1))] = columns[0]
    if not charge_columns:
        return None

    states = sorted(charge_columns)
    matrix = data.loc[:, [charge_columns[state] for state in states]]
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    active_mask = matrix > 0.5
    has_active = active_mask.any(axis=1)
    first_active = active_mask.argmax(axis=1)
    charges = np.zeros(len(data), dtype=np.float64)
    charges[has_active] = np.asarray(states, dtype=np.float64)[first_active[has_active]]
    return pd.Series(charges, index=data.index)


def _derive_feature_from_charge(
    data: pd.DataFrame,
    feature_name: str,
    normalized_column_map: Dict[str, List[str]],
    charge_n_cache: Dict[str, pd.Series | None],
) -> pd.Series | None:
    feature_key = _normalize_name(feature_name)

    def get_charge_n_series() -> pd.Series | None:
        if "charge_n_series" not in charge_n_cache:
            charge_column = _resolve_feature_column("ChargeN", normalized_column_map)
            if charge_column is not None:
                charge_n_cache["charge_n_series"] = (
                    pd.to_numeric(data[charge_column], errors="coerce").fillna(0.0)
                )
            else:
                charge_n_cache["charge_n_series"] = _infer_charge_n_from_onehot(
                    data, normalized_column_map
                )
        return charge_n_cache["charge_n_series"]

    if feature_key == "chargen":
        return get_charge_n_series()

    match = _CHARGE_ONEHOT_KEY_RE.fullmatch(feature_key)
    if match:
        charge_state = int(match.group(1))
        charge_series = get_charge_n_series()
        if charge_series is None:
            return None
        return (charge_series.round().astype(int) == charge_state).astype(float)

    return None


def _resolve_model_feature_columns(
    data: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[List[str], List[str], Dict[str, List[str]]]:
    resolved_columns: List[str] = []
    missing_features: List[str] = []
    normalized_column_map = _build_normalized_column_map(list(data.columns))
    charge_n_cache: Dict[str, pd.Series | None] = {}

    for feature_name in feature_names:
        column_name = _resolve_feature_column(feature_name, normalized_column_map)
        if column_name is None:
            derived = _derive_feature_from_charge(
                data=data,
                feature_name=feature_name,
                normalized_column_map=normalized_column_map,
                charge_n_cache=charge_n_cache,
            )
            if derived is not None:
                data[feature_name] = derived
                feature_key = _normalize_name(feature_name)
                normalized_column_map.setdefault(feature_key, []).append(feature_name)
                column_name = feature_name

        if column_name is None:
            missing_features.append(feature_name)
        else:
            resolved_columns.append(column_name)

    return resolved_columns, missing_features, normalized_column_map


def _build_missing_message(
    missing_features: Sequence[str],
    normalized_column_map: Dict[str, List[str]],
) -> str:
    available_columns = [col for columns in normalized_column_map.values() for col in columns]
    available_keys = list(normalized_column_map.keys())
    suggestion_parts = []
    for feature in missing_features:
        key = _normalize_name(feature)
        close_keys = difflib.get_close_matches(key, available_keys, n=3, cutoff=0.72)
        suggestions = []
        for close_key in close_keys:
            suggestions.extend(normalized_column_map.get(close_key, []))
        if suggestions:
            unique_suggestions = ", ".join(sorted(set(suggestions))[:3])
            suggestion_parts.append(f"{feature} -> {unique_suggestions}")

    available_preview = ", ".join(sorted(set(available_columns))[:30])
    message = (
        "PIN is missing model features after flexible matching: "
        f"{list(missing_features)}. Available columns (first 30): [{available_preview}]"
    )
    if suggestion_parts:
        message += ". Closest matches: " + "; ".join(suggestion_parts)
    return message


def score_pin_candidates(pin_df: pd.DataFrame, models: Sequence[LinearModel]) -> pd.DataFrame:
    """Apply selected linear models and add final mean score columns."""

    if not models:
        raise ValueError("No linear models provided for scoring.")

    data = pin_df.copy()
    score_cols = []
    resolved_columns_cache: Dict[tuple[str, ...], List[str]] = {}
    for model_idx, model in enumerate(models, start=1):
        feature_key = tuple(model.feature_names)
        if feature_key not in resolved_columns_cache:
            resolved_columns, missing, normalized_column_map = _resolve_model_feature_columns(
                data=data,
                feature_names=model.feature_names,
            )
            if missing:
                raise ValueError(_build_missing_message(missing, normalized_column_map))
            resolved_columns_cache[feature_key] = resolved_columns
        resolved_columns = resolved_columns_cache[feature_key]

        matrix = (
            data.loc[:, resolved_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float64, copy=False)
        )
        scores = matrix @ model.weights + model.intercept
        score_col = f"model_score_{model_idx}"
        data[score_col] = scores
        score_cols.append(score_col)

    data["final_score"] = data.loc[:, score_cols].mean(axis=1)
    return data
