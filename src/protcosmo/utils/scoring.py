"""Scoring helpers for PIN candidates."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .weights_parser import LinearModel


def score_pin_candidates(pin_df: pd.DataFrame, models: Sequence[LinearModel]) -> pd.DataFrame:
    """Apply selected linear models and add final mean score columns."""

    if not models:
        raise ValueError("No linear models provided for scoring.")

    data = pin_df.copy()
    score_cols = []
    for model_idx, model in enumerate(models, start=1):
        missing = [name for name in model.feature_names if name not in data.columns]
        if missing:
            raise ValueError(f"PIN is missing model features: {missing}")

        matrix = (
            data.loc[:, model.feature_names]
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
