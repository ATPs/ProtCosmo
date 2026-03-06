"""PSM winner selection and novel classification."""

from __future__ import annotations

from typing import List

import pandas as pd

from .input_key import extract_input_file_key
from .pin_reader import extract_rank_index, extract_spectrum_id, split_proteins


NOVEL_PREFIX = "COMETPLUS_NOVEL_"
DECOY_PREFIX = "DECOY_"


def is_novel_protein_id(protein_id: str) -> bool:
    text = str(protein_id).strip()
    if not text:
        return False
    if text.startswith(DECOY_PREFIX):
        text = text[len(DECOY_PREFIX) :]
    return text.startswith(NOVEL_PREFIX)


def get_novel_protein_ids(proteins_text: str) -> List[str]:
    tokens = split_proteins(proteins_text)
    return [token for token in tokens if is_novel_protein_id(token)]


def classify_novel_only(proteins_text: str) -> bool:
    tokens = split_proteins(proteins_text)
    if not tokens:
        return False
    return all(is_novel_protein_id(token) for token in tokens)


def select_best_psm_per_spectrum(scored_df: pd.DataFrame, mass_file: str) -> pd.DataFrame:
    """Select best PSM per spectrum with tie-break preference to non-novel."""

    work = scored_df.copy()
    work["mass_file"] = str(mass_file)
    work["spectrum_id"] = work["SpecId"].astype(str).map(extract_spectrum_id)
    work["input_file_key"] = work["SpecId"].astype(str).map(extract_input_file_key)
    work["rank_index"] = work["SpecId"].astype(str).map(extract_rank_index)
    work["novel_only"] = work["Proteins"].astype(str).map(classify_novel_only)
    work["novel_protein_ids"] = work["Proteins"].astype(str).map(get_novel_protein_ids)

    ordered = work.sort_values(
        by=["mass_file", "spectrum_id", "final_score", "novel_only", "rank_index"],
        ascending=[True, True, False, True, True],
    )
    winners = (
        ordered.groupby(["mass_file", "spectrum_id"], as_index=False, sort=False)
        .first()
        .reset_index(drop=True)
    )
    return winners
