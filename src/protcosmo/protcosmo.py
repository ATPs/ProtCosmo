#!/usr/bin/env python3
"""ProtCosmo CLI."""

from __future__ import annotations

import argparse
import datetime as dt
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    _THIS_DIR = Path(__file__).resolve().parent
    _SRC_DIR = _THIS_DIR.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from protcosmo import __version__  # type: ignore
    from protcosmo.utils.comet_runner import run_cometplus_search  # type: ignore
    from protcosmo.utils.config_loader import load_pipeline_config  # type: ignore
    from protcosmo.utils.peptide_utils import (  # type: ignore
        collapse_to_unmodified,
        normalize_modified_peptide,
    )
    from protcosmo.utils.percolator_ref import build_reference_lookup  # type: ignore
    from protcosmo.utils.pin_reader import join_proteins, read_pin, split_proteins  # type: ignore
    from protcosmo.utils.report_writer import (  # type: ignore
        ensure_dir,
        write_json,
        write_tsv,
        write_warnings,
    )
    from protcosmo.utils.scoring import score_pin_candidates  # type: ignore
    from protcosmo.utils.selection import (  # type: ignore
        get_novel_protein_ids,
        select_best_psm_per_spectrum,
    )
    from protcosmo.utils.weights_parser import (  # type: ignore
        parse_selected_models,
        validate_models_feature_alignment,
    )
else:
    from . import __version__
    from .utils.comet_runner import run_cometplus_search
    from .utils.config_loader import load_pipeline_config
    from .utils.peptide_utils import collapse_to_unmodified, normalize_modified_peptide
    from .utils.percolator_ref import build_reference_lookup
    from .utils.pin_reader import join_proteins, read_pin, split_proteins
    from .utils.report_writer import ensure_dir, write_json, write_tsv, write_warnings
    from .utils.scoring import score_pin_candidates
    from .utils.selection import get_novel_protein_ids, select_best_psm_per_spectrum
    from .utils.weights_parser import parse_selected_models, validate_models_feature_alignment


ESTIMATION_NOTE = (
    "q-value and PEP are estimated by score lookup from reference Percolator outputs. "
    "Real q-value/PEP are expected to be smaller than the estimated values."
)


FULL_HELP_TEXT = f"""
Detailed behavior:
1. ProtCosmo runs CometPlus search per input mass file.
2. It reads generated PIN candidates and applies static linear scoring using percolator weights.
3. It selects the best PSM per spectrum.
4. It marks novel-only winners by protein id prefix COMETPLUS_NOVEL_.
5. It estimates q-value/PEP from percolator target tables by closest-smaller score mapping.
6. It writes all winners and novel-only summaries.

Scoring:
1. Use numeric rows 1/3/5 from --init-weights.
2. For each row/model: score = w^T x + b.
3. Final candidate score = mean of three model scores.

Tie-break in winner selection:
1. Higher score wins.
2. On tie, non-novel winner is preferred over novel-only winner.
3. If still tied, lower rank in SpecId is preferred.

Estimation warning:
{ESTIMATION_NOTE}
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protcosmo",
        description=(
            "ProtCosmo: search novel proteins/peptides with CometPlus and estimate QC metrics "
            "from Percolator outputs."
        ),
        epilog=ESTIMATION_NOTE,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--help-full", action="store_true", help="print full help and exit")
    parser.add_argument("--cometplus", default="cometplus", help='path to cometplus executable')
    parser.add_argument(
        "--mass-file",
        dest="mass_file",
        help="single mass file or comma-joined list of files",
    )
    parser.add_argument("--params", help="single params file or comma-joined list")
    parser.add_argument("--database", help="single database file or comma-joined list")
    parser.add_argument("--novel_protein", help="novel protein FASTA")
    parser.add_argument("--novel_peptide", help="novel peptide file (FASTA or tokenized text)")
    parser.add_argument("--thread", type=int, help="override num_threads")
    parser.add_argument("--scan", help="scan filter file (single-file mode only)")
    parser.add_argument(
        "--scan_numbers",
        help="explicit scan list such as 1001,1002,1003 (single-file mode only)",
    )
    parser.add_argument("--first-scan", dest="first_scan", type=int, help="alias for -F<num>")
    parser.add_argument("--last-scan", dest="last_scan", type=int, help="alias for -L<num>")
    parser.add_argument(
        "--init-weights",
        dest="init_weights",
        help="single weights file or comma-joined list",
    )
    parser.add_argument(
        "--percolator-psms",
        dest="percolator_psms",
        help="target PSM reference file(s), tsv or parquet",
    )
    parser.add_argument(
        "--percolator-peptides",
        dest="percolator_peptides",
        help="target peptide reference file(s), tsv or parquet",
    )
    parser.add_argument("--input_tsv", help="optional run-input TSV")
    parser.add_argument("--output-dir", required=True, help="output directory")
    return parser


def _print_full_help(parser: argparse.ArgumentParser) -> None:
    parser.print_help()
    print("")
    print(FULL_HELP_TEXT)


def _command_to_shell(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(token) for token in command)


def _lookup_cache_get(cache: Dict[str, object], key: str, loader):
    if key not in cache:
        cache[key] = loader(key)
    return cache[key]


def _compute_peptide_estimates(novel_psms: pd.DataFrame, lookup_cache: Dict[str, object]) -> pd.DataFrame:
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

    grouped = novel_psms.groupby("percolator_peptides_file").groups
    for ref_path, index_values in grouped.items():
        lookup = _lookup_cache_get(lookup_cache, ref_path, build_reference_lookup)
        index_list = list(index_values)
        scores = novel_psms.loc[index_list, "final_score"].to_numpy(dtype=np.float64, copy=False)
        est_q, est_pep, matched, fallback = lookup.estimate_array(scores)
        novel_psms.loc[index_list, "estimated_peptide_q_value"] = est_q
        novel_psms.loc[index_list, "estimated_peptide_pep"] = est_pep
        novel_psms.loc[index_list, "estimated_peptide_matched_score"] = matched
        novel_psms.loc[index_list, "estimated_peptide_fallback"] = fallback
    return novel_psms


def _make_modified_summary(novel_psms: pd.DataFrame) -> pd.DataFrame:
    if novel_psms.empty:
        return pd.DataFrame(
            columns=[
                "mass_file",
                "modified_peptide",
                "unmodified_peptide",
                "novel_psm_count",
                "best_final_score",
                "estimated_psm_q_value",
                "estimated_psm_pep",
                "estimated_peptide_q_value",
                "estimated_peptide_pep",
                "novel_protein_count",
            ]
        )

    ranked = novel_psms.sort_values(["mass_file", "modified_peptide", "final_score"], ascending=[True, True, False])
    best = ranked.groupby(["mass_file", "modified_peptide"], as_index=False).first()
    counts = (
        novel_psms.groupby(["mass_file", "modified_peptide"], as_index=False)
        .size()
        .rename(columns={"size": "novel_psm_count"})
    )
    protein_counts = (
        novel_psms.assign(_novel_protein_ids=novel_psms["novel_protein_ids"])
        .explode("_novel_protein_ids")
        .dropna(subset=["_novel_protein_ids"])
        .groupby(["mass_file", "modified_peptide"], as_index=False)["_novel_protein_ids"]
        .nunique()
        .rename(columns={"_novel_protein_ids": "novel_protein_count"})
    )
    summary = best.merge(counts, on=["mass_file", "modified_peptide"], how="left")
    summary = summary.merge(protein_counts, on=["mass_file", "modified_peptide"], how="left")
    summary["novel_protein_count"] = summary["novel_protein_count"].fillna(0).astype(int)
    summary = summary.rename(columns={"final_score": "best_final_score"})
    keep_cols = [
        "mass_file",
        "modified_peptide",
        "unmodified_peptide",
        "novel_psm_count",
        "best_final_score",
        "estimated_psm_q_value",
        "estimated_psm_pep",
        "estimated_peptide_q_value",
        "estimated_peptide_pep",
        "novel_protein_count",
    ]
    return summary.loc[:, keep_cols]


def _make_unmodified_summary(novel_psms: pd.DataFrame) -> pd.DataFrame:
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


def _make_protein_summary(novel_psms: pd.DataFrame) -> pd.DataFrame:
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


def run_pipeline(args, passthrough_args: List[str]) -> Dict[str, str]:
    start_time = dt.datetime.now(tz=dt.timezone.utc)
    config = load_pipeline_config(args, passthrough_args)
    output_dir = ensure_dir(config.output_dir)
    comet_output_root = ensure_dir(output_dir / "comet_outputs")

    warnings: List[str] = list(config.warnings)
    model_cache: Dict[str, object] = {}
    psm_lookup_cache: Dict[str, object] = {}
    peptide_lookup_cache: Dict[str, object] = {}

    all_winners_parts: List[pd.DataFrame] = []
    command_records: List[dict] = []

    for run in config.runs:
        result = run_cometplus_search(run, config, comet_output_root)
        command_records.append(
            {
                "run_index": run.run_index,
                "row_index": run.row_index,
                "mass_file": run.mass_file,
                "command": result.command,
                "command_shell": _command_to_shell(result.command),
                "run_dir": str(result.run_dir),
                "pin_path": str(result.pin_path),
                "stdout_log": str(result.stdout_path),
                "stderr_log": str(result.stderr_path),
                "return_code": result.return_code,
            }
        )

        pin_df = read_pin(result.pin_path)

        models = _lookup_cache_get(model_cache, run.init_weights, parse_selected_models)
        validate_models_feature_alignment(models)
        scored = score_pin_candidates(pin_df, models)
        winners = select_best_psm_per_spectrum(scored, run.mass_file)

        psm_lookup = _lookup_cache_get(psm_lookup_cache, run.percolator_psms, build_reference_lookup)
        psm_scores = winners["final_score"].to_numpy(dtype=np.float64, copy=False)
        est_q, est_pep, matched, fallback = psm_lookup.estimate_array(psm_scores)
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
        winners["protein_ids"] = winners["Proteins"].astype(str).map(split_proteins).map(join_proteins)
        all_winners_parts.append(winners)

    all_winners = pd.concat(all_winners_parts, ignore_index=True) if all_winners_parts else pd.DataFrame()
    if all_winners.empty:
        raise RuntimeError("No winner PSMs were produced.")

    novel_psms = all_winners[all_winners["novel_only"]].copy()
    novel_psms["modified_peptide"] = novel_psms["Peptide"].astype(str).map(normalize_modified_peptide)
    novel_psms["unmodified_peptide"] = novel_psms["Peptide"].astype(str).map(collapse_to_unmodified)
    novel_psms["novel_protein_ids"] = novel_psms["Proteins"].astype(str).map(get_novel_protein_ids)
    novel_psms = _compute_peptide_estimates(novel_psms, peptide_lookup_cache)

    peptide_fallback_count = int(novel_psms["estimated_peptide_fallback"].fillna(False).astype(bool).sum())
    if peptide_fallback_count > 0:
        warnings.append(
            f"{peptide_fallback_count} novel PSM(s) had no smaller score in peptide reference; "
            "assigned peptide q-value=1 and PEP=1."
        )

    modified_summary = _make_modified_summary(novel_psms)
    unmodified_summary = _make_unmodified_summary(novel_psms)
    protein_summary = _make_protein_summary(novel_psms)

    all_winners_out = all_winners.copy()
    novel_psms_out = novel_psms.copy()
    if "novel_protein_ids" in novel_psms_out.columns:
        novel_psms_out["novel_protein_ids"] = novel_psms_out["novel_protein_ids"].map(
            lambda x: join_proteins(x) if isinstance(x, list) else ""
        )

    output_paths = {
        "all_winners": str(output_dir / "protcosmo.all_winners.tsv"),
        "novel_psms": str(output_dir / "protcosmo.novel_psms.tsv"),
        "novel_peptides_modified": str(output_dir / "protcosmo.novel_peptides.modified.tsv"),
        "novel_peptides_unmodified": str(output_dir / "protcosmo.novel_peptides.unmodified.tsv"),
        "novel_proteins": str(output_dir / "protcosmo.novel_proteins.tsv"),
        "run_metadata": str(output_dir / "protcosmo.run_metadata.json"),
        "warnings": str(output_dir / "protcosmo.warnings.log"),
    }
    write_tsv(all_winners_out, Path(output_paths["all_winners"]))
    write_tsv(novel_psms_out, Path(output_paths["novel_psms"]))
    write_tsv(modified_summary, Path(output_paths["novel_peptides_modified"]))
    write_tsv(unmodified_summary, Path(output_paths["novel_peptides_unmodified"]))
    write_tsv(protein_summary, Path(output_paths["novel_proteins"]))
    write_warnings(warnings, Path(output_paths["warnings"]))

    end_time = dt.datetime.now(tz=dt.timezone.utc)
    metadata = {
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "argv": sys.argv,
        "passthrough_args": passthrough_args,
        "output_dir": str(output_dir),
        "estimation_note": ESTIMATION_NOTE,
        "input_tsv": None if config.input_tsv is None else str(config.input_tsv),
        "use_scan_filters": config.use_scan_filters,
        "run_count": len(config.runs),
        "warnings_count": len(warnings),
        "warnings": warnings,
        "commands": command_records,
        "output_paths": output_paths,
        "all_winners_count": int(len(all_winners)),
        "novel_psm_count": int(len(novel_psms)),
        "novel_modified_peptide_count": int(len(modified_summary)),
        "novel_unmodified_peptide_count": int(len(unmodified_summary)),
        "novel_protein_count": int(len(protein_summary)),
    }
    write_json(metadata, Path(output_paths["run_metadata"]))
    return output_paths


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    parser = build_parser()
    if "--help-full" in argv:
        _print_full_help(parser)
        return 0

    args, passthrough_args = parser.parse_known_args(argv)
    if args.help_full:
        _print_full_help(parser)
        return 0
    try:
        outputs = run_pipeline(args, passthrough_args)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("ProtCosmo finished successfully.")
    for key, path in outputs.items():
        print(f"{key}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
