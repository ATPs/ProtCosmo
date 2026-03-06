"""Scoring batch helpers for run-level and TSV-group scoring."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .cache_utils import lookup_cache_get
from .input_key import extract_input_file_key
from .novel_reports import protein_ids_csv_from_text
from .percolator_ref import build_partitioned_reference_lookup
from .pin_reader import read_pin
from .scoring import score_pin_candidates
from .selection import select_best_psm_per_spectrum
from .weights_parser import parse_selected_models, validate_models_feature_alignment


def score_winner_rows_from_pin(
    run,
    pin_path: Path,
    model_cache: Dict[str, object],
    psm_lookup_cache: Dict[str, object],
    warnings: List[str],
) -> pd.DataFrame:
    """Read PIN and score winner rows for one run."""

    if run.init_weights is None or run.percolator_psms is None or run.percolator_peptides is None:
        raise RuntimeError(
            f"Run {run.run_index}: scoring references are missing "
            "(init-weights/percolator-psms/percolator-peptides)."
        )

    pin_df = read_pin(pin_path)
    return score_winner_rows_from_df(
        run=run,
        pin_df=pin_df,
        model_cache=model_cache,
        psm_lookup_cache=psm_lookup_cache,
        warnings=warnings,
    )


def score_winner_rows_from_df(
    run,
    pin_df: pd.DataFrame,
    model_cache: Dict[str, object],
    psm_lookup_cache: Dict[str, object],
    warnings: List[str],
    model_cache_lock=None,
    psm_lookup_cache_lock=None,
) -> pd.DataFrame:
    """Score winner rows from an in-memory PIN DataFrame."""

    if run.init_weights is None or run.percolator_psms is None or run.percolator_peptides is None:
        raise RuntimeError(
            f"Run {run.run_index}: scoring references are missing "
            "(init-weights/percolator-psms/percolator-peptides)."
        )

    models = lookup_cache_get(
        model_cache,
        run.init_weights,
        parse_selected_models,
        lock=model_cache_lock,
    )
    validate_models_feature_alignment(models)
    scored = score_pin_candidates(pin_df, models)
    winners = select_best_psm_per_spectrum(scored, run.mass_file)

    partitioned_lookup = lookup_cache_get(
        psm_lookup_cache,
        run.percolator_psms,
        build_partitioned_reference_lookup,
        lock=psm_lookup_cache_lock,
    )
    est_q = np.full(len(winners), np.nan, dtype=np.float64)
    est_pep = np.full(len(winners), np.nan, dtype=np.float64)
    matched = np.full(len(winners), np.nan, dtype=np.float64)
    fallback = np.zeros(len(winners), dtype=bool)

    for input_file_key, index_values in winners.groupby("input_file_key", dropna=False).groups.items():
        index_list = list(index_values)
        if psm_lookup_cache_lock is None:
            lookup = partitioned_lookup.lookup_for_input_key(str(input_file_key))
        else:
            with psm_lookup_cache_lock:
                lookup = partitioned_lookup.lookup_for_input_key(str(input_file_key))
        psm_scores = winners.loc[index_list, "final_score"].to_numpy(dtype=np.float64, copy=False)
        g_q, g_pep, g_matched, g_fallback = lookup.estimate_array(psm_scores)
        est_q[index_list] = g_q
        est_pep[index_list] = g_pep
        matched[index_list] = g_matched
        fallback[index_list] = g_fallback

    fallback_count = int(np.count_nonzero(fallback))
    if fallback_count > 0:
        warnings.append(
            f"Run {run.run_index}: {fallback_count} winner PSM(s) had no smaller score in "
            f"reference {run.percolator_psms}; assigned q-value=1 and PEP=1."
        )

    winners["estimated_psm_q_value"] = est_q
    winners["estimated_psm_pep"] = est_pep
    winners["estimated_psm_matched_score"] = matched
    winners["estimated_psm_fallback"] = fallback
    winners["run_index"] = run.run_index
    winners["row_index"] = run.row_index
    winners["params_file"] = run.params
    winners["database_file"] = run.database
    winners["init_weights_file"] = run.init_weights
    winners["percolator_psms_file"] = run.percolator_psms
    winners["percolator_peptides_file"] = run.percolator_peptides
    winners["protein_ids"] = winners["Proteins"].astype(str).map(protein_ids_csv_from_text)
    return winners


def score_winner_rows_for_tsv_groups(
    run,
    pin_path: Path,
    config,
    model_cache: Dict[str, object],
    psm_lookup_cache: Dict[str, object],
    warnings: List[str],
) -> List[pd.DataFrame]:
    """Split merged PIN by TSV scoring groups and score each subset."""

    pin_df = read_pin(pin_path).copy()
    pin_df["__input_file_key"] = pin_df["SpecId"].astype(str).map(extract_input_file_key)
    prepared_groups: List[tuple[int, SimpleNamespace, pd.DataFrame]] = []

    for group in config.scoring_groups:
        subset = pin_df[pin_df["__input_file_key"].isin(set(group.mass_file_keys))].drop(
            columns=["__input_file_key"]
        )
        if subset.empty:
            warnings.append(
                f"Run {run.run_index}.{group.group_index}: no PIN rows matched "
                f"input-file keys for init-weights {group.init_weights}."
            )
            continue

        run_proxy = SimpleNamespace(
            run_index=f"{run.run_index}.{group.group_index}",
            row_index=run.row_index,
            mass_file=",".join(group.mass_files),
            params=run.params,
            database=run.database,
            init_weights=group.init_weights,
            percolator_psms=group.percolator_psms,
            percolator_peptides=group.percolator_peptides,
        )
        prepared_groups.append((group.group_index, run_proxy, subset))

    if not prepared_groups:
        return []

    worker_count = resolve_tsv_group_worker_count(getattr(config, "thread", None), len(prepared_groups))
    winners_parts: List[pd.DataFrame] = []
    if worker_count == 1:
        for _, run_proxy, subset in prepared_groups:
            group_warnings: List[str] = []
            winners_parts.append(
                score_winner_rows_from_df(
                    run=run_proxy,
                    pin_df=subset,
                    model_cache=model_cache,
                    psm_lookup_cache=psm_lookup_cache,
                    warnings=group_warnings,
                )
            )
            warnings.extend(group_warnings)
        return winners_parts

    model_cache_lock = Lock()
    psm_lookup_cache_lock = Lock()
    results_by_group: Dict[int, tuple[pd.DataFrame, List[str]]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_group = {
            executor.submit(
                _score_single_tsv_group,
                group_index,
                run_proxy,
                subset,
                model_cache,
                psm_lookup_cache,
                model_cache_lock,
                psm_lookup_cache_lock,
            ): group_index
            for group_index, run_proxy, subset in prepared_groups
        }
        for future in as_completed(future_to_group):
            group_index, winners, group_warnings = future.result()
            results_by_group[group_index] = (winners, group_warnings)

    for group_index, _, _ in prepared_groups:
        winners, group_warnings = results_by_group[group_index]
        winners_parts.append(winners)
        warnings.extend(group_warnings)
    return winners_parts


def resolve_tsv_group_worker_count(thread_value: Optional[int], group_count: int) -> int:
    """Worker count for grouped TSV scoring."""

    if group_count <= 1:
        return 1
    if thread_value is None or int(thread_value) <= 1:
        return 1
    return min(int(thread_value), group_count)


def _score_single_tsv_group(
    group_index: int,
    run_proxy,
    subset: pd.DataFrame,
    model_cache: Dict[str, object],
    psm_lookup_cache: Dict[str, object],
    model_cache_lock,
    psm_lookup_cache_lock,
) -> tuple[int, pd.DataFrame, List[str]]:
    group_warnings: List[str] = []
    winners = score_winner_rows_from_df(
        run=run_proxy,
        pin_df=subset,
        model_cache=model_cache,
        psm_lookup_cache=psm_lookup_cache,
        warnings=group_warnings,
        model_cache_lock=model_cache_lock,
        psm_lookup_cache_lock=psm_lookup_cache_lock,
    )
    return group_index, winners, group_warnings
