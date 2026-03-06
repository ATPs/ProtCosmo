#!/usr/bin/env python3
"""ProtCosmo CLI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

if __package__ in (None, ""):
    _THIS_DIR = Path(__file__).resolve().parent
    _SRC_DIR = _THIS_DIR.parent
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))
    from protcosmo import __version__  # type: ignore
    from protcosmo.utils.comet_runner import run_cometplus_search  # type: ignore
    from protcosmo.utils.config_loader import load_pipeline_config  # type: ignore
    from protcosmo.utils.help_text import (  # type: ignore
        FULL_HELP_TEXT,
        ProtCosmoHelpFormatter,
        SHORT_DESCRIPTION,
        SHORT_EPILOG,
        print_full_help as _print_full_help,
    )
    from protcosmo.utils.novel_reports import (  # type: ignore
        NOVEL_PREFIX,
        compute_peptide_estimates as _compute_peptide_estimates,
        load_internal_novel_protein_map as _load_internal_novel_protein_map,
        make_modified_summary as _make_modified_summary,
        make_protein_summary as _make_protein_summary,
        make_psm_output_table as _make_psm_output_table,
        make_unmodified_summary as _make_unmodified_summary,
        resolve_internal_novel_mapping_path as _resolve_internal_novel_mapping_path,
    )
    from protcosmo.utils.peptide_utils import (  # type: ignore
        collapse_to_unmodified,
        normalize_modified_peptide,
    )
    from protcosmo.utils.runtime_logging import PipelineLogger  # type: ignore
    from protcosmo.utils.report_writer import (  # type: ignore
        ensure_dir,
        write_tsv,
    )
    from protcosmo.utils import scoring_batches as _scoring_batches  # type: ignore
    from protcosmo.utils.selection import get_novel_protein_ids  # type: ignore
else:
    from . import __version__
    from .utils.comet_runner import run_cometplus_search
    from .utils.config_loader import load_pipeline_config
    from .utils.help_text import (
        FULL_HELP_TEXT,
        ProtCosmoHelpFormatter,
        SHORT_DESCRIPTION,
        SHORT_EPILOG,
        print_full_help as _print_full_help,
    )
    from .utils.novel_reports import (
        NOVEL_PREFIX,
        compute_peptide_estimates as _compute_peptide_estimates,
        load_internal_novel_protein_map as _load_internal_novel_protein_map,
        make_modified_summary as _make_modified_summary,
        make_protein_summary as _make_protein_summary,
        make_psm_output_table as _make_psm_output_table,
        make_unmodified_summary as _make_unmodified_summary,
        resolve_internal_novel_mapping_path as _resolve_internal_novel_mapping_path,
    )
    from .utils.peptide_utils import collapse_to_unmodified, normalize_modified_peptide
    from .utils.runtime_logging import PipelineLogger
    from .utils.report_writer import ensure_dir, write_tsv
    from .utils import scoring_batches as _scoring_batches
    from .utils.selection import get_novel_protein_ids


_score_winner_rows_from_pin = _scoring_batches.score_winner_rows_from_pin
_score_winner_rows_from_df = _scoring_batches.score_winner_rows_from_df
_resolve_tsv_group_worker_count = _scoring_batches.resolve_tsv_group_worker_count


def _score_winner_rows_for_tsv_groups(*args, **kwargs):
    original_executor = _scoring_batches.ThreadPoolExecutor
    original_score_df = _scoring_batches.score_winner_rows_from_df
    _scoring_batches.ThreadPoolExecutor = ThreadPoolExecutor
    _scoring_batches.score_winner_rows_from_df = _score_winner_rows_from_df
    try:
        return _scoring_batches.score_winner_rows_for_tsv_groups(*args, **kwargs)
    finally:
        _scoring_batches.ThreadPoolExecutor = original_executor
        _scoring_batches.score_winner_rows_from_df = original_score_df


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
        help=(
            "mass-file input source.\n"
            "Supported forms:\n"
            "  1) one mass spectrum file (e.g. .mgf, .mzML, .mzMLb, .mgf.gz, .mzML.gz)\n"
            "  2) comma-separated files\n"
            "  3) text file, one mass-file path per line (# comment lines allowed)\n"
            "  4) directory; all supported files in that directory are used\n"
            "Required unless --input-pin or --input_tsv is set."
        ),
    )
    run_group.add_argument(
        "--input_tsv",
        help=(
            "row-based TSV input table mode.\n"
            "Required column: mass-file.\n"
            "Optional columns: params, database, init-weights, percolator-psms, percolator-peptides.\n"
            "In this mode, --mass-file is not allowed."
        ),
    )
    run_group.add_argument(
        "--params",
        help=(
            "Comet params file path (single value only).\n"
            "Required unless --input-pin is set, or provided by --input_tsv rows."
        ),
    )
    run_group.add_argument(
        "--database",
        help=(
            "known database file path for CometPlus --database (single value only).\n"
            "Required unless --input-pin is set, or provided by --input_tsv rows."
        ),
    )
    run_group.add_argument(
        "--input-pin",
        dest="input_pin",
        help=(
            "skip CometPlus and directly score this PIN file.\n"
            "In this mode, --mass-file/--params/--database are not required."
        ),
    )
    run_group.add_argument(
        "--thread",
        type=int,
        help=(
            "override CometPlus num_threads.\n"
            "In --input_tsv mode with multiple init-weight groups, this also controls "
            "parallel scoring worker count."
        ),
    )
    run_group.add_argument(
        "--keep-tmp",
        dest="keep_tmp",
        action="store_true",
        help="forward to CometPlus: keep temporary files",
    )
    run_group.add_argument(
        "--run-comet-each",
        dest="run_comet_each",
        action="store_true",
        default=True,
        help="forward to CometPlus: run each mass-file input in CometPlus mode (default: enabled)",
    )
    run_group.add_argument(
        "--no-run-comet-each",
        dest="run_comet_each",
        action="store_false",
        help="disable forwarding --run-comet-each to CometPlus",
    )
    run_group.add_argument(
        "--output-dir",
        required=True,
        help=(
            "output directory for reports and CometPlus outputs/logs.\n"
            "Also forwarded to CometPlus as --output-folder."
        ),
    )
    run_group.add_argument(
        "--output-prefix",
        default="protcosmo",
        help="filename prefix for ProtCosmo outputs (default: protcosmo)",
    )
    run_group.add_argument(
        "--stop-after-cometplus",
        dest="stop_after_cometplus",
        action="store_true",
        help="run CometPlus and stop before scoring",
    )
    run_group.add_argument(
        "--force",
        action="store_true",
        help="in novel mode, rerun and overwrite existing <output-prefix>.cometplus.novel.pin",
    )
    run_group.add_argument(
        "--log",
        action="store_true",
        help="also write runtime logs/warnings to <output-dir>/<output-prefix>.log",
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
            "  <output-dir>/<output-prefix>.internal_novel_peptide.tsv"
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
            "ProtCosmo then skips PIN scoring."
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
            "Percolator weight file used to score PIN candidates (single CLI value).\n"
            "ProtCosmo prefers raw-weight rows (commonly 2/4/6), applies score=w^T x + b, "
            "and averages the three scores.\n"
            "For per-mass-file variation, use --input_tsv rows."
        ),
    )
    score_group.add_argument(
        "--percolator-psms",
        dest="percolator_psms",
        help=(
            "target PSM reference table for q-value/PEP estimation (TSV or Parquet, single CLI value).\n"
            "Table must contain score, q-value, and PEP columns (name variants accepted)."
        ),
    )
    score_group.add_argument(
        "--percolator-peptides",
        dest="percolator_peptides",
        help=(
            "target peptide reference table for q-value/PEP estimation (TSV or Parquet, single CLI value).\n"
            "Table must contain score, q-value, and PEP columns (name variants accepted)."
        ),
    )

    return parser


def run_pipeline(args, passthrough_args: List[str]) -> Dict[str, str]:
    config = load_pipeline_config(args, passthrough_args)
    output_prefix = config.output_prefix
    output_dir = ensure_dir(config.output_dir)
    log_path = (output_dir / f"{output_prefix}.log").resolve() if config.log else None
    logger = PipelineLogger(log_path)
    output_paths: Dict[str, str] = {}
    if log_path is not None:
        output_paths["log"] = str(log_path)

    try:
        warnings: List[str] = list(config.warnings)
        model_cache: Dict[str, object] = {}
        psm_lookup_cache: Dict[str, object] = {}
        peptide_lookup_cache: Dict[str, object] = {}
        all_winners_parts: List[pd.DataFrame] = []
        stop_after_any = config.stop_after_saving_novel_peptide or config.stop_after_cometplus

        if config.input_pin:
            for run in config.runs:
                winners = _score_winner_rows_from_pin(
                    run=run,
                    pin_path=Path(run.mass_file),
                    model_cache=model_cache,
                    psm_lookup_cache=psm_lookup_cache,
                    warnings=warnings,
                )
                all_winners_parts.append(winners)
        else:
            novel_pin_path = (output_dir / f"{output_prefix}.cometplus.novel.pin").resolve()
            for run in config.runs:
                result = run_cometplus_search(
                    run,
                    config,
                    output_dir,
                    require_pin_output=(not stop_after_any),
                )
                if result.skipped:
                    logger.info(
                        f"Run {run.run_index}: existing PIN found at {result.pin_path}; "
                        "skipping CometPlus. Use --force to rerun and overwrite."
                    )
                else:
                    if result.overwrote_existing_pin:
                        logger.info(
                            f"Run {run.run_index}: --force is set; overwriting existing PIN at {novel_pin_path}."
                        )
                    if result.stdout_text:
                        logger.info(result.stdout_text)
                    if result.stderr_text:
                        logger.stderr(result.stderr_text)

                if stop_after_any:
                    continue
                if result.pin_path is None:
                    raise RuntimeError(f"Run {run.run_index}: CometPlus PIN output path is missing.")
                if config.use_input_tsv and len(config.scoring_groups) > 1:
                    all_winners_parts.extend(
                        _score_winner_rows_for_tsv_groups(
                            run=run,
                            pin_path=result.pin_path,
                            config=config,
                            model_cache=model_cache,
                            psm_lookup_cache=psm_lookup_cache,
                            warnings=warnings,
                        )
                    )
                else:
                    winners = _score_winner_rows_from_pin(
                        run=run,
                        pin_path=result.pin_path,
                        model_cache=model_cache,
                        psm_lookup_cache=psm_lookup_cache,
                        warnings=warnings,
                    )
                    all_winners_parts.append(winners)

        if stop_after_any:
            for warning in warnings:
                text = str(warning).strip()
                if text:
                    logger.warning(text)
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

        peptide_to_proteins: Dict[str, List[str]] = {}
        missing_novel_ids: set[str] = set()
        contains_novel_tokens = (
            novel_psms["Proteins"].astype(str).str.contains(NOVEL_PREFIX, regex=False).any()
            if not novel_psms.empty
            else False
        )
        if contains_novel_tokens:
            mapping_path = _resolve_internal_novel_mapping_path(config, output_dir, output_prefix)
            if mapping_path.exists():
                peptide_to_proteins = _load_internal_novel_protein_map(mapping_path)
            else:
                warnings.append(
                    "Unable to remap COMETPLUS_NOVEL_* protein IDs because mapping file is missing: "
                    f"{mapping_path}"
                )

        modified_summary = _make_modified_summary(
            novel_psms,
            peptide_to_proteins=peptide_to_proteins,
            missing_novel_ids=missing_novel_ids,
        )
        novel_psms_out = _make_psm_output_table(
            novel_psms,
            peptide_to_proteins=peptide_to_proteins,
            missing_novel_ids=missing_novel_ids,
        )

        if missing_novel_ids:
            warnings.append(
                "Unmapped COMETPLUS_NOVEL peptide_id values were kept in output proteinIds: "
                + ",".join(sorted(missing_novel_ids))
            )

        for warning in warnings:
            text = str(warning).strip()
            if text:
                logger.warning(text)

        output_paths["novel_psms"] = str((output_dir / f"{output_prefix}.nove.psms.tsv").resolve())
        output_paths["novel_peptides_modified"] = str((output_dir / f"{output_prefix}.novel.peptides.tsv").resolve())
        write_tsv(novel_psms_out, Path(output_paths["novel_psms"]))
        write_tsv(modified_summary, Path(output_paths["novel_peptides_modified"]))
        return output_paths
    finally:
        logger.close()


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
