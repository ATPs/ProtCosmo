"""Helpers for novel-output remapping and report table construction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .cache_utils import lookup_cache_get
from .percolator_ref import build_partitioned_reference_lookup
from .pin_reader import split_proteins

NOVEL_PREFIX = "COMETPLUS_NOVEL_"
DECOY_PREFIX = "DECOY_"


def _join_unique_csv(tokens: Sequence[str]) -> str:
    seen = set()
    ordered: List[str] = []
    for token in tokens:
        text = str(token).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ",".join(ordered)


def _is_novel_token(token: str) -> bool:
    text = str(token).strip()
    if text.startswith(DECOY_PREFIX):
        text = text[len(DECOY_PREFIX) :]
    return text.startswith(NOVEL_PREFIX)


def load_internal_novel_protein_map(path: Path) -> Dict[str, List[str]]:
    """Load peptide_id -> protein_id list mapping from internal novel TSV."""

    table = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    if "peptide_id" not in table.columns or "protein_id" not in table.columns:
        raise ValueError(
            f"Internal novel peptide file is missing required columns in {path}: peptide_id, protein_id"
        )

    mapping: Dict[str, List[str]] = {}
    for peptide_id, protein_text in zip(table["peptide_id"], table["protein_id"]):
        key = "" if peptide_id is None else str(peptide_id).strip()
        if not key:
            continue
        tokens = split_proteins("" if protein_text is None else str(protein_text))
        if not tokens:
            continue
        existing = mapping.get(key, [])
        mapping[key] = [token for token in _join_unique_csv(existing + tokens).split(",") if token]
    return mapping


def protein_ids_csv_from_text(
    proteins_text: str,
    *,
    peptide_to_proteins: Optional[Dict[str, List[str]]] = None,
    missing_novel_ids: Optional[set[str]] = None,
) -> str:
    """Build output proteinIds CSV, applying COMETPLUS_NOVEL_* remap when provided."""

    tokens = split_proteins(proteins_text)
    if not tokens:
        return ""
    if not peptide_to_proteins:
        return _join_unique_csv(tokens)

    remapped: List[str] = []
    for token in tokens:
        text = str(token).strip()
        is_decoy = False
        core = text
        if core.startswith(DECOY_PREFIX):
            is_decoy = True
            core = core[len(DECOY_PREFIX) :]
        if core in peptide_to_proteins:
            mapped = peptide_to_proteins[core]
            if is_decoy:
                remapped.extend([f"{DECOY_PREFIX}{item}" for item in mapped])
            else:
                remapped.extend(mapped)
            continue
        if missing_novel_ids is not None and _is_novel_token(text):
            missing_novel_ids.add(core)
        remapped.append(text)
    return _join_unique_csv(remapped)


def make_psm_output_table(
    novel_psms: pd.DataFrame,
    *,
    peptide_to_proteins: Optional[Dict[str, List[str]]] = None,
    missing_novel_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Build reference-style novel PSM output table."""

    columns = ["PSMId", "score", "q-value", "posterior_error_prob", "peptide", "proteinIds"]
    if novel_psms.empty:
        return pd.DataFrame(columns=columns)

    output = pd.DataFrame(
        {
            "PSMId": novel_psms["SpecId"].astype(str),
            "score": pd.to_numeric(novel_psms["estimated_psm_matched_score"], errors="coerce"),
            "q-value": pd.to_numeric(novel_psms["estimated_psm_q_value"], errors="coerce"),
            "posterior_error_prob": pd.to_numeric(novel_psms["estimated_psm_pep"], errors="coerce"),
            "peptide": novel_psms["modified_peptide"].astype(str),
            "proteinIds": novel_psms["Proteins"].astype(str).map(
                lambda text: protein_ids_csv_from_text(
                    text,
                    peptide_to_proteins=peptide_to_proteins,
                    missing_novel_ids=missing_novel_ids,
                )
            ),
        }
    )
    output = output.sort_values(by=["score", "PSMId"], ascending=[False, True], na_position="last")
    return output.loc[:, columns].reset_index(drop=True)


def compute_peptide_estimates(novel_psms: pd.DataFrame, lookup_cache: Dict[str, object]) -> pd.DataFrame:
    """Estimate peptide q-value/PEP for novel winners via partition-aware lookup."""

    if novel_psms.empty:
        novel_psms["estimated_peptide_q_value"] = np.nan
        novel_psms["estimated_peptide_pep"] = np.nan
        novel_psms["estimated_peptide_matched_score"] = np.nan
        novel_psms["estimated_peptide_fallback"] = False
        return novel_psms

    novel_psms = novel_psms.copy()
    novel_psms["estimated_peptide_q_value"] = np.nan
    novel_psms["estimated_peptide_pep"] = np.nan
    novel_psms["estimated_peptide_matched_score"] = np.nan
    novel_psms["estimated_peptide_fallback"] = False

    grouped = novel_psms.groupby(["percolator_peptides_file", "input_file_key"], dropna=False).groups
    for (ref_path, input_file_key), index_values in grouped.items():
        partitioned_lookup = lookup_cache_get(lookup_cache, ref_path, build_partitioned_reference_lookup)
        lookup = partitioned_lookup.lookup_for_input_key(str(input_file_key))
        index_list = list(index_values)
        scores = novel_psms.loc[index_list, "final_score"].to_numpy(dtype=np.float64, copy=False)
        est_q, est_pep, matched, fallback = lookup.estimate_array(scores)
        novel_psms.loc[index_list, "estimated_peptide_q_value"] = est_q
        novel_psms.loc[index_list, "estimated_peptide_pep"] = est_pep
        novel_psms.loc[index_list, "estimated_peptide_matched_score"] = matched
        novel_psms.loc[index_list, "estimated_peptide_fallback"] = fallback
    return novel_psms


def make_modified_summary(
    novel_psms: pd.DataFrame,
    *,
    peptide_to_proteins: Optional[Dict[str, List[str]]] = None,
    missing_novel_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Build modified-peptide summary with one highest-score row per peptide."""

    columns = ["PSMId", "score", "q-value", "posterior_error_prob", "peptide", "proteinIds"]
    if novel_psms.empty:
        return pd.DataFrame(columns=columns)

    table = pd.DataFrame(
        {
            "PSMId": novel_psms["SpecId"].astype(str),
            "score": pd.to_numeric(novel_psms["estimated_peptide_matched_score"], errors="coerce"),
            "q-value": pd.to_numeric(novel_psms["estimated_peptide_q_value"], errors="coerce"),
            "posterior_error_prob": pd.to_numeric(novel_psms["estimated_peptide_pep"], errors="coerce"),
            "peptide": novel_psms["modified_peptide"].astype(str),
            "proteinIds": novel_psms["Proteins"].astype(str).map(
                lambda text: protein_ids_csv_from_text(
                    text,
                    peptide_to_proteins=peptide_to_proteins,
                    missing_novel_ids=missing_novel_ids,
                )
            ),
            "_final_score": pd.to_numeric(novel_psms["final_score"], errors="coerce"),
        }
    )

    ranked = table.sort_values(
        by=["peptide", "score", "_final_score", "PSMId"],
        ascending=[True, False, False, True],
        na_position="last",
    )
    best_per_peptide = ranked.groupby("peptide", as_index=False, sort=False).first()
    output = best_per_peptide.sort_values(by=["score", "PSMId"], ascending=[False, True], na_position="last")
    return output.loc[:, columns].reset_index(drop=True)


def make_unmodified_summary(novel_psms: pd.DataFrame) -> pd.DataFrame:
    """Build unmodified-peptide summary table."""

    if novel_psms.empty:
        return pd.DataFrame(
            columns=[
                "mass_file",
                "unmodified_peptide",
                "novel_psm_count",
                "modified_form_count",
                "best_final_score",
                "estimated_psm_q_value",
                "estimated_psm_pep",
                "estimated_peptide_q_value",
                "estimated_peptide_pep",
            ]
        )

    ranked = novel_psms.sort_values(
        ["mass_file", "unmodified_peptide", "final_score"],
        ascending=[True, True, False],
    )
    best = ranked.groupby(["mass_file", "unmodified_peptide"], as_index=False).first()
    counts = (
        novel_psms.groupby(["mass_file", "unmodified_peptide"], as_index=False)
        .size()
        .rename(columns={"size": "novel_psm_count"})
    )
    form_counts = (
        novel_psms.groupby(["mass_file", "unmodified_peptide"], as_index=False)["modified_peptide"]
        .nunique()
        .rename(columns={"modified_peptide": "modified_form_count"})
    )
    summary = best.merge(counts, on=["mass_file", "unmodified_peptide"], how="left")
    summary = summary.merge(form_counts, on=["mass_file", "unmodified_peptide"], how="left")
    summary = summary.rename(columns={"final_score": "best_final_score"})
    keep_cols = [
        "mass_file",
        "unmodified_peptide",
        "novel_psm_count",
        "modified_form_count",
        "best_final_score",
        "estimated_psm_q_value",
        "estimated_psm_pep",
        "estimated_peptide_q_value",
        "estimated_peptide_pep",
    ]
    return summary.loc[:, keep_cols]


def make_protein_summary(novel_psms: pd.DataFrame) -> pd.DataFrame:
    """Build novel-protein summary table."""

    if novel_psms.empty:
        return pd.DataFrame(
            columns=[
                "mass_file",
                "novel_protein_id",
                "novel_psm_count",
                "modified_peptide_count",
                "unmodified_peptide_count",
                "best_final_score",
                "estimated_psm_q_value",
                "estimated_psm_pep",
            ]
        )

    exploded = (
        novel_psms.assign(novel_protein_id=novel_psms["novel_protein_ids"])
        .explode("novel_protein_id")
        .dropna(subset=["novel_protein_id"])
    )
    ranked = exploded.sort_values(
        ["mass_file", "novel_protein_id", "final_score"],
        ascending=[True, True, False],
    )
    best = ranked.groupby(["mass_file", "novel_protein_id"], as_index=False).first()
    counts = (
        exploded.groupby(["mass_file", "novel_protein_id"], as_index=False)
        .size()
        .rename(columns={"size": "novel_psm_count"})
    )
    mod_count = (
        exploded.groupby(["mass_file", "novel_protein_id"], as_index=False)["modified_peptide"]
        .nunique()
        .rename(columns={"modified_peptide": "modified_peptide_count"})
    )
    unmod_count = (
        exploded.groupby(["mass_file", "novel_protein_id"], as_index=False)["unmodified_peptide"]
        .nunique()
        .rename(columns={"unmodified_peptide": "unmodified_peptide_count"})
    )
    summary = best.merge(counts, on=["mass_file", "novel_protein_id"], how="left")
    summary = summary.merge(mod_count, on=["mass_file", "novel_protein_id"], how="left")
    summary = summary.merge(unmod_count, on=["mass_file", "novel_protein_id"], how="left")
    summary = summary.rename(columns={"final_score": "best_final_score"})
    keep_cols = [
        "mass_file",
        "novel_protein_id",
        "novel_psm_count",
        "modified_peptide_count",
        "unmodified_peptide_count",
        "best_final_score",
        "estimated_psm_q_value",
        "estimated_psm_pep",
    ]
    return summary.loc[:, keep_cols]


def resolve_internal_novel_mapping_path(config, output_dir: Path, output_prefix: str) -> Path:
    """Resolve mapping-path source for COMETPLUS_NOVEL_* id remapping."""

    if config.internal_novel_peptide:
        candidate = Path(str(config.internal_novel_peptide)).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate
    return (output_dir / f"{output_prefix}.internal_novel_peptide.tsv").resolve()
