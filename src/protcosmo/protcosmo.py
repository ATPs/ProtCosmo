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


class ProtCosmoHelpFormatter(argparse.RawTextHelpFormatter):
    """Help formatter with wider layout for readable multiline option docs."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=36, width=118)


SHORT_DESCRIPTION = """
ProtCosmo runs CometPlus searches, statically re-scores PIN candidates with Percolator weights,
selects one winner PSM per spectrum, and reports novel-only findings.

Use --help-full for step-by-step workflow, file-format details, and examples.
""".strip()


SHORT_EPILOG = f"""
Quick format notes:
- --mass-file supports: single file, comma list, list file (one path per line), or directory.
- --params and --database each accept only one value.
- --novel_protein: FASTA input.
- --novel_peptide: FASTA or tokenized text (comma/space/tab/newline delimiters).
- --output_internal_novel_peptide: auto-enabled in novel mode as `--output-dir/protcosmo.internal_novel_peptide.tsv`.
- --internal_novel_peptide: reuse previously exported internal novel TSV.
- --stop-after-saving-novel-peptide: stop after TSV export, skip search/scoring in ProtCosmo.
- --scan/--scan_numbers/--first-scan/--last-scan: only applied when exactly one run is resolved.

{ESTIMATION_NOTE}
""".strip()


FULL_HELP_TEXT = f"""
Detailed workflow:

Step 1. Build run configuration
- Resolve --mass-file into one or more concrete spectrum files.
- Supported --mass-file styles:
  - single mass spectrum file
  - comma-separated list of files
  - text file, one mass-file path per line (blank lines and # comments ignored)
  - directory (all CometPlus-supported files in that directory are used)
- CometPlus-supported suffixes recognized by ProtCosmo include:
  .mgf, .mgf.gz, .mzML, .mzML.gz, .mzMLb, .mzXML, .mzXML.gz, .raw, .ms2, .cms2, .bms2
- Duplicate mass-file paths are de-duplicated while keeping order.
- --params and --database are required and each accepts only one value.
- --init-weights/--percolator-psms/--percolator-peptides:
  - 1 value => broadcast to all resolved mass files
  - N values => 1:1 mapping with resolved mass-file count
- Scan-filter gating:
  - --scan/--scan_numbers/--first-scan/--last-scan are applied only when final run count is 1
  - when run count > 1, scan filters are ignored and warning is logged

Step 2. Run CometPlus for each mass file
- ProtCosmo builds a CometPlus command with:
  --params, --database, --output_percolatorfile 1, --max_duplicate_proteins -1, --name <output-dir>/comet.run_xxxx
- Optional ProtCosmo-controlled options forwarded to CometPlus:
  --novel_protein, --novel_peptide, --output_internal_novel_peptide,
  --internal_novel_peptide, --stop-after-saving-novel-peptide,
  --thread, scan filters (single-run only)
- Unknown options are passed through to CometPlus unchanged.
- Each run writes logs to:
  <output-dir>/cometplus.run_xxxx.stdout.log
  <output-dir>/cometplus.run_xxxx.stderr.log
- PIN output is detected from generated .pin/.pin.gz/.pin.parquet(.gz)

Step 3. Static scoring from --init-weights
- --init-weights is a Percolator weights file used for re-scoring PIN candidates.
- ProtCosmo always parses numeric rows 1, 3, and 5.
- For each selected row/model: score_k = w_k^T x + b_k  (b_k is column m0)
- Final candidate score: final_score = mean(score_1, score_3, score_5)

Step 4. Select winner per spectrum
- Winner selection per spectrum:
  1) higher final_score wins
  2) on score tie, non-novel winner is preferred over novel-only winner
  3) if still tied, smaller SpecId rank wins
- Novel-only means every protein ID in winner starts with COMETPLUS_NOVEL_

Step 5. Estimate q-value/PEP by lookup
- PSM-level estimate uses --percolator-psms.
- Peptide-level estimate for novel winners uses --percolator-peptides.
- Reference format: TSV or Parquet.
- Required logical columns (name variants accepted):
  score, q-value, posterior_error_prob/pep
- Lookup rule: nearest smaller-or-equal score.
- Fallback when no smaller score exists: q-value=1 and PEP=1 (warning logged).

Step 6. Write outputs
- protcosmo.novel_psms.tsv
- protcosmo.novel_peptides.modified.tsv
- protcosmo.warnings.log
- protcosmo.run_metadata.json

Option details and format examples:

--novel_protein <file>
- FASTA input only.
- Example:
  >novel_protein_1
  MPEPTIDEKQLA

--novel_peptide <file>
- Two supported formats:
  1) FASTA (auto-detected if any non-empty trimmed line starts with '>')
  2) tokenized text (comma/space/tab/newline delimiters)
- Tokenized mode normalization in CometPlus:
  - keep alphabetic chars only
  - convert to uppercase
  - remove empty/duplicate tokens
- Tokenized examples (valid):
  PEPTIDEK,PEPTIDEL
  PEPTIDEK PEPTIDEL
  PEPTIDEK
  PEPTIDEL
  ACD[+57]EFG
  K.PEPTIDE.R
- Parsed examples become:
  PEPTIDEK, PEPTIDEL, ACDEFG, KPEPTIDER

--output_internal_novel_peptide <file>
- Forwarded directly to CometPlus.
- CometPlus writes an internal TSV with columns: peptide, peptide_id, protein_id.
- Default behavior in ProtCosmo:
  - if --novel_protein or --novel_peptide is provided and this option is not set,
    ProtCosmo auto-adds:
    --output_internal_novel_peptide <output-dir>/protcosmo.internal_novel_peptide.tsv
  - this path is shared across runs, so only one file version is kept.

--internal_novel_peptide <file>
- Forwarded directly to CometPlus.
- Reuse a previously exported internal TSV as novel peptide source.

--stop-after-saving-novel-peptide
- Forwarded directly to CometPlus.
- CometPlus exits after saving internal TSV.
- In this mode, ProtCosmo does not run PIN scoring and does not write novel_psms/novel_peptides reports.
- ProtCosmo still writes run metadata and warnings logs.

--scan <file>
- Text file of positive scan integers.
- Delimiters: comma/space/tab/newline.
- CometPlus only searches these scans.
- ProtCosmo applies this only when exactly one mass-file run is resolved.

--scan_numbers <list>
- Inline explicit scan list with same parser as --scan.
- Example: 1001,1002,1003

--first-scan <num>
- First/start scan number to search in CometPlus (same semantics as -F<num>).
- Sets lower scan bound; can be used alone or with --last-scan.
- Applies to novel peptide/protein searches and normal searches.
- If --scan/--scan_numbers are also set, effective scans are intersection.
- Applied only when exactly one mass-file run is resolved.

--last-scan <num>
- Last/end scan number to search in CometPlus (same semantics as -L<num>).
- Sets upper scan bound; can be used alone or with --first-scan.
- Applies to novel peptide/protein searches and normal searches.
- If --scan/--scan_numbers are also set, effective scans are intersection.
- Applied only when exactly one mass-file run is resolved.

--init-weights <file>
- Percolator weight file used to score Comet PIN candidates.
- Expected shape: header row with feature names (must include m0), then numeric rows.
- ProtCosmo uses numeric rows 1, 3, and 5 for the final score.

Estimation warning:
{ESTIMATION_NOTE}
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="protcosmo",
        formatter_class=ProtCosmoHelpFormatter,
        description=SHORT_DESCRIPTION,
        epilog=SHORT_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--help-full",
        action="store_true",
        help="print detailed help (workflow, file formats, and examples) and exit",
    )

    run_group = parser.add_argument_group("Core run inputs")
    run_group.add_argument(
        "--cometplus",
        default="cometplus",
        help="path to CometPlus executable (default: cometplus)",
    )
    run_group.add_argument(
        "--mass-file",
        dest="mass_file",
        required=True,
        help=(
            "mass-file input source.\n"
            "Supported forms:\n"
            "  1) one mass spectrum file (e.g. .mgf, .mzML, .mzMLb, .mgf.gz, .mzML.gz)\n"
            "  2) comma-separated files\n"
            "  3) text file, one mass-file path per line (# comment lines allowed)\n"
            "  4) directory; all supported files in that directory are used"
        ),
    )
    run_group.add_argument(
        "--params",
        required=True,
        help=(
            "Comet params file path (required, single value only)."
        ),
    )
    run_group.add_argument(
        "--database",
        required=True,
        help=(
            "known database file path for CometPlus --database (required, single value only)."
        ),
    )
    run_group.add_argument(
        "--thread",
        type=int,
        help="override CometPlus num_threads",
    )
    run_group.add_argument(
        "--output-dir",
        required=True,
        help="output directory for reports, metadata, and CometPlus outputs/logs",
    )

    novel_group = parser.add_argument_group("Novel and scan-subset inputs")
    novel_group.add_argument(
        "--novel_protein",
        help=(
            "novel protein FASTA file for CometPlus novel mode.\n"
            "These proteins are digested with active Comet settings to build novel peptide candidates."
        ),
    )
    novel_group.add_argument(
        "--novel_peptide",
        help=(
            "novel peptide file for CometPlus novel mode.\n"
            "Supported formats:\n"
            "  1) FASTA (auto-detected if any non-empty line starts with '>')\n"
            "  2) tokenized text (delimiters: comma/space/tab/newline)\n"
            "Tokenized examples:\n"
            "  PEPTIDEK,PEPTIDEL\n"
            "  PEPTIDEK PEPTIDEL\n"
            "  one peptide per line is also valid"
        ),
    )
    novel_group.add_argument(
        "--output_internal_novel_peptide",
        dest="output_internal_novel_peptide",
        help=(
            "forward to CometPlus: export internal novel TSV (columns: peptide, peptide_id, protein_id).\n"
            "Default (auto-enabled when --novel_protein or --novel_peptide is used):\n"
            "  <output-dir>/protcosmo.internal_novel_peptide.tsv"
        ),
    )
    novel_group.add_argument(
        "--internal_novel_peptide",
        dest="internal_novel_peptide",
        help="forward to CometPlus: reuse previously exported internal novel TSV input.",
    )
    novel_group.add_argument(
        "--stop-after-saving-novel-peptide",
        dest="stop_after_saving_novel_peptide",
        action="store_true",
        help=(
            "forward to CometPlus: stop after internal TSV export.\n"
            "ProtCosmo then skips PIN scoring and only writes run_metadata + warnings logs."
        ),
    )
    novel_group.add_argument(
        "--scan",
        help=(
            "scan filter file (positive integers; delimiters: comma/space/tab/newline).\n"
            "CometPlus searches only these scans.\n"
            "ProtCosmo applies scan filters only when exactly one mass-file run is resolved."
        ),
    )
    novel_group.add_argument(
        "--scan_numbers",
        help=(
            "inline explicit scan list, e.g. 1001,1002,1003 (same parser as --scan).\n"
            "ProtCosmo applies this only when exactly one mass-file run is resolved."
        ),
    )
    novel_group.add_argument(
        "--first-scan",
        dest="first_scan",
        type=int,
        help=(
            "first/start scan number for CometPlus search (same semantics as -F<num>).\n"
            "This is the lower scan bound for novel peptide/protein search and normal search.\n"
            "If --scan/--scan_numbers are provided, effective scans are intersection.\n"
            "Applied only when exactly one mass-file run is resolved."
        ),
    )
    novel_group.add_argument(
        "--last-scan",
        dest="last_scan",
        type=int,
        help=(
            "last/end scan number for CometPlus search (same semantics as -L<num>).\n"
            "This is the upper scan bound for novel peptide/protein search and normal search.\n"
            "If --scan/--scan_numbers are provided, effective scans are intersection.\n"
            "Applied only when exactly one mass-file run is resolved."
        ),
    )

    score_group = parser.add_argument_group("Scoring and reference inputs")
    score_group.add_argument(
        "--init-weights",
        dest="init_weights",
        help=(
            "Percolator weight file(s) used to score PIN candidates.\n"
            "ProtCosmo parses numeric rows 1/3/5, applies score=w^T x + b, and averages the three scores.\n"
            "One value may be broadcast, or provide comma-separated values matching --mass-file count."
        ),
    )
    score_group.add_argument(
        "--percolator-psms",
        dest="percolator_psms",
        help=(
            "target PSM reference table(s) for q-value/PEP estimation (TSV or Parquet).\n"
            "Table must contain score, q-value, and PEP columns (name variants accepted)."
        ),
    )
    score_group.add_argument(
        "--percolator-peptides",
        dest="percolator_peptides",
        help=(
            "target peptide reference table(s) for q-value/PEP estimation (TSV or Parquet).\n"
            "Table must contain score, q-value, and PEP columns (name variants accepted)."
        ),
    )

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

    warnings: List[str] = list(config.warnings)
    model_cache: Dict[str, object] = {}
    psm_lookup_cache: Dict[str, object] = {}
    peptide_lookup_cache: Dict[str, object] = {}

    all_winners_parts: List[pd.DataFrame] = []
    command_records: List[dict] = []

    for run in config.runs:
        result = run_cometplus_search(
            run,
            config,
            output_dir,
            require_pin_output=(not config.stop_after_saving_novel_peptide),
        )
        command_records.append(
            {
                "run_index": run.run_index,
                "row_index": run.row_index,
                "mass_file": run.mass_file,
                "command": result.command,
                "command_shell": _command_to_shell(result.command),
                "run_dir": str(result.run_dir),
                "pin_path": None if result.pin_path is None else str(result.pin_path),
                "stdout_log": str(result.stdout_path),
                "stderr_log": str(result.stderr_path),
                "return_code": result.return_code,
            }
        )

        if config.stop_after_saving_novel_peptide:
            continue
        if result.pin_path is None:
            raise RuntimeError(f"Run {run.run_index}: CometPlus PIN output path is missing.")
        if run.init_weights is None or run.percolator_psms is None or run.percolator_peptides is None:
            raise RuntimeError(
                f"Run {run.run_index}: scoring references are missing "
                "(init-weights/percolator-psms/percolator-peptides)."
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

    if config.stop_after_saving_novel_peptide:
        output_paths = {
            "run_metadata": str(output_dir / "protcosmo.run_metadata.json"),
            "warnings": str(output_dir / "protcosmo.warnings.log"),
        }
        write_warnings(warnings, Path(output_paths["warnings"]))
        end_time = dt.datetime.now(tz=dt.timezone.utc)
        metadata = {
            "start_time_utc": start_time.isoformat(),
            "end_time_utc": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "argv": sys.argv,
            "passthrough_args": passthrough_args,
            "output_dir": str(output_dir),
            "mode": "stop_after_saving_novel_peptide",
            "estimation_note": ESTIMATION_NOTE,
            "use_scan_filters": config.use_scan_filters,
            "run_count": len(config.runs),
            "warnings_count": len(warnings),
            "warnings": warnings,
            "commands": command_records,
            "output_paths": output_paths,
            "all_winners_count": 0,
            "novel_psm_count": 0,
            "novel_modified_peptide_count": 0,
            "novel_unmodified_peptide_count": 0,
            "novel_protein_count": 0,
        }
        write_json(metadata, Path(output_paths["run_metadata"]))
        return output_paths

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

    novel_psms_out = novel_psms.copy()
    if "novel_protein_ids" in novel_psms_out.columns:
        novel_psms_out["novel_protein_ids"] = novel_psms_out["novel_protein_ids"].map(
            lambda x: join_proteins(x) if isinstance(x, list) else ""
        )

    output_paths = {
        "novel_psms": str(output_dir / "protcosmo.novel_psms.tsv"),
        "novel_peptides_modified": str(output_dir / "protcosmo.novel_peptides.modified.tsv"),
        "run_metadata": str(output_dir / "protcosmo.run_metadata.json"),
        "warnings": str(output_dir / "protcosmo.warnings.log"),
    }
    write_tsv(novel_psms_out, Path(output_paths["novel_psms"]))
    write_tsv(modified_summary, Path(output_paths["novel_peptides_modified"]))
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
    if any(arg == "--input_tsv" or str(arg).startswith("--input_tsv=") for arg in passthrough_args):
        print(
            "ERROR: --input_tsv is no longer supported. "
            "Use --mass-file with one of: single file, comma list, list file, or directory.",
            file=sys.stderr,
        )
        return 1
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
