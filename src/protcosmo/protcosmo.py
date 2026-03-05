#!/usr/bin/env python3
"""ProtCosmo CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TextIO

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
    from protcosmo.utils.percolator_ref import build_partitioned_reference_lookup  # type: ignore
    from protcosmo.utils.pin_reader import read_pin, split_proteins  # type: ignore
    from protcosmo.utils.report_writer import (  # type: ignore
        ensure_dir,
        write_tsv,
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
    from .utils.percolator_ref import build_partitioned_reference_lookup
    from .utils.pin_reader import read_pin, split_proteins
    from .utils.report_writer import ensure_dir, write_tsv
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
- --input-pin: skip CometPlus and score directly from this PIN file.
- --mass-file supports: single file, comma list, list file (one path per line), or directory.
- --params and --database each accept only one value (required unless --input-pin is set).
- --output-dir is also forwarded to CometPlus as --output-folder.
- --output-prefix controls ProtCosmo-generated output filename prefix (default: protcosmo).
- --log: also write runtime logs/warnings to <output-dir>/<output-prefix>.log.
- --force: overwrite existing <output-prefix>.cometplus.novel.pin in novel mode.
- --run-comet-each is forwarded to CometPlus by default.
- --stop-after-cometplus: run CometPlus then stop (no scoring/reporting).
- --novel_protein: FASTA input.
- --novel_peptide: FASTA or tokenized text (comma/space/tab/newline delimiters).
- --output_internal_novel_peptide: auto-enabled in novel mode as `--output-dir/<output-prefix>.internal_novel_peptide.tsv`.
- --internal_novel_peptide: reuse previously exported internal novel TSV.
- --stop-after-saving-novel-peptide: stop after TSV export, skip search/scoring in ProtCosmo.
- --keep-tmp: keep CometPlus temporary files.
- --scan/--scan_numbers/--first-scan/--last-scan: only applied when exactly one run is resolved.

{ESTIMATION_NOTE}
""".strip()


FULL_HELP_TEXT = f"""
Detailed workflow:

Step 1. Build run configuration
- If --input-pin is provided:
  - ProtCosmo skips CometPlus and scores this PIN directly.
  - --mass-file/--params/--database are not required.
- Resolve --mass-file into one or more concrete spectrum files.
- Supported --mass-file styles:
  - single mass spectrum file
  - comma-separated list of files
  - text file, one mass-file path per line (blank lines and # comments ignored)
  - directory (all CometPlus-supported files in that directory are used)
- CometPlus-supported suffixes recognized by ProtCosmo include:
  .mgf, .mgf.gz, .mzML, .mzML.gz, .mzMLb, .mzXML, .mzXML.gz, .raw, .ms2, .cms2, .bms2
- Duplicate mass-file paths are de-duplicated while keeping order.
- --params and --database are required and each accepts only one value (unless --input-pin is set).
- --init-weights/--percolator-psms/--percolator-peptides:
  - 1 value => broadcast to all resolved mass files
  - N values => 1:1 mapping with resolved mass-file count
- In novel mode (--novel_protein/--novel_peptide/--internal_novel_peptide):
  - if multiple mass files are resolved, ProtCosmo calls CometPlus once with all inputs.
- Scan-filter gating:
  - --scan/--scan_numbers/--first-scan/--last-scan are applied only when final run count is 1
  - when run count > 1, scan filters are ignored and warning is logged

Step 2. Run CometPlus for each mass file
- ProtCosmo builds a CometPlus command with:
  --params, --database, --output-folder <output-dir>,
  --output_percolatorfile 1, --max_duplicate_proteins -1
- One run may include one or many spectrum input files.
- Optional ProtCosmo-controlled options forwarded to CometPlus:
  --novel_protein, --novel_peptide, --output_internal_novel_peptide,
  --internal_novel_peptide, --stop-after-saving-novel-peptide,
  --keep-tmp, --run-comet-each, --thread, scan filters (single-run only)
- Unknown options are passed through to CometPlus unchanged.
- CometPlus stdout/stderr are streamed to screen (and to <output-prefix>.log when --log is set).
- PIN output is detected from generated .pin/.pin.gz/.pin.parquet(.gz)
- In novel mode, if <output-prefix>.cometplus.novel.pin already exists:
  - default: skip CometPlus and reuse this PIN
  - with --force: rerun CometPlus and overwrite it
- With --stop-after-cometplus, ProtCosmo stops after this step.

Step 3. Static scoring from --init-weights
- --init-weights is a Percolator weights file used for re-scoring PIN candidates.
- For Percolator CV exports with repeated blocks
  (header -> normalized row -> raw row), ProtCosmo uses raw rows 2, 4, and 6.
- For each selected row/model: score_k = w_k^T x + b_k  (b_k is column m0)
- Final candidate score: final_score = mean(score_1, score_2, score_3)

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
- <output-prefix>.nove.psms.tsv
- <output-prefix>.novel.peptides.tsv
- optional: <output-prefix>.log (when --log is set)

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
    --output_internal_novel_peptide <output-dir>/<output-prefix>.internal_novel_peptide.tsv
  - this path is shared across runs, so only one file version is kept.

--internal_novel_peptide <file>
- Forwarded directly to CometPlus.
- Reuse a previously exported internal TSV as novel peptide source.

--stop-after-saving-novel-peptide
- Forwarded directly to CometPlus.
- CometPlus exits after saving internal TSV.
- In this mode, ProtCosmo does not run PIN scoring and does not write novel_psms/novel_peptides reports.
- ProtCosmo still prints runtime logs/warnings to screen.

--keep-tmp
- Forwarded directly to CometPlus.
- Keep CometPlus temporary/intermediate files.

--run-comet-each / --no-run-comet-each
- Forwarded directly to CometPlus as `--run-comet-each`.
- Default in ProtCosmo: enabled.
- Use `--no-run-comet-each` to disable forwarding this option.

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
- ProtCosmo prefers raw-weight rows (commonly numeric rows 2, 4, and 6).

--input-pin <file>
- Skip CometPlus and directly score this PIN.
- Requires: --init-weights, --percolator-psms, --percolator-peptides.
- --mass-file/--params/--database are not required in this mode.

--stop-after-cometplus
- Run CometPlus and stop before scoring.
- ProtCosmo prints runtime logs/warnings to screen and exits.

--force
- In novel mode, if `<output-prefix>.cometplus.novel.pin` already exists:
  - default behavior is skip/reuse;
  - with `--force`, ProtCosmo reruns CometPlus and overwrites that PIN.

--log
- Also write runtime logs/warnings to:
  `<output-dir>/<output-prefix>.log`

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
        help=(
            "mass-file input source.\n"
            "Supported forms:\n"
            "  1) one mass spectrum file (e.g. .mgf, .mzML, .mzMLb, .mgf.gz, .mzML.gz)\n"
            "  2) comma-separated files\n"
            "  3) text file, one mass-file path per line (# comment lines allowed)\n"
            "  4) directory; all supported files in that directory are used\n"
            "Required unless --input-pin is set."
        ),
    )
    run_group.add_argument(
        "--params",
        help=(
            "Comet params file path (single value only).\n"
            "Required unless --input-pin is set."
        ),
    )
    run_group.add_argument(
        "--database",
        help=(
            "known database file path for CometPlus --database (single value only).\n"
            "Required unless --input-pin is set."
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
        help="override CometPlus num_threads",
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
            "Percolator weight file(s) used to score PIN candidates.\n"
            "ProtCosmo prefers raw-weight rows (commonly 2/4/6), applies score=w^T x + b, "
            "and averages the three scores.\n"
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


class PipelineLogger:
    def __init__(self, log_path: Optional[Path]) -> None:
        self.log_path = log_path
        self._handle: Optional[TextIO] = None
        if log_path is not None:
            ensure_dir(log_path.parent)
            self._handle = log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _emit(self, text: str, stream: TextIO) -> None:
        if not text:
            return
        out = text if text.endswith("\n") else f"{text}\n"
        stream.write(out)
        stream.flush()
        if self._handle is not None:
            self._handle.write(out)
            self._handle.flush()

    def info(self, text: str) -> None:
        self._emit(text, sys.stdout)

    def stderr(self, text: str) -> None:
        self._emit(text, sys.stderr)

    def warning(self, text: str) -> None:
        self._emit(f"WARNING: {text}", sys.stderr)


def _lookup_cache_get(cache: Dict[str, object], key: str, loader):
    if key not in cache:
        cache[key] = loader(key)
    return cache[key]


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


def _protein_ids_csv_from_text(proteins_text: str) -> str:
    return _join_unique_csv(split_proteins(proteins_text))


def _make_psm_output_table(novel_psms: pd.DataFrame) -> pd.DataFrame:
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
            "proteinIds": novel_psms["Proteins"].astype(str).map(_protein_ids_csv_from_text),
        }
    )
    output = output.sort_values(by=["score", "PSMId"], ascending=[False, True], na_position="last")
    return output.loc[:, columns].reset_index(drop=True)


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

    grouped = novel_psms.groupby(["percolator_peptides_file", "input_file_key"], dropna=False).groups
    for (ref_path, input_file_key), index_values in grouped.items():
        partitioned_lookup = _lookup_cache_get(lookup_cache, ref_path, build_partitioned_reference_lookup)
        lookup = partitioned_lookup.lookup_for_input_key(str(input_file_key))
        index_list = list(index_values)
        scores = novel_psms.loc[index_list, "final_score"].to_numpy(dtype=np.float64, copy=False)
        est_q, est_pep, matched, fallback = lookup.estimate_array(scores)
        novel_psms.loc[index_list, "estimated_peptide_q_value"] = est_q
        novel_psms.loc[index_list, "estimated_peptide_pep"] = est_pep
        novel_psms.loc[index_list, "estimated_peptide_matched_score"] = matched
        novel_psms.loc[index_list, "estimated_peptide_fallback"] = fallback
    return novel_psms


def _make_modified_summary(novel_psms: pd.DataFrame) -> pd.DataFrame:
    """Build reference-style peptide table with one highest-score row per peptide."""

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
            "proteinIds": novel_psms["Proteins"].astype(str).map(_protein_ids_csv_from_text),
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


def _score_winner_rows_from_pin(
    run,
    pin_path: Path,
    model_cache: Dict[str, object],
    psm_lookup_cache: Dict[str, object],
    warnings: List[str],
) -> pd.DataFrame:
    if run.init_weights is None or run.percolator_psms is None or run.percolator_peptides is None:
        raise RuntimeError(
            f"Run {run.run_index}: scoring references are missing "
            "(init-weights/percolator-psms/percolator-peptides)."
        )

    pin_df = read_pin(pin_path)

    models = _lookup_cache_get(model_cache, run.init_weights, parse_selected_models)
    validate_models_feature_alignment(models)
    scored = score_pin_candidates(pin_df, models)
    winners = select_best_psm_per_spectrum(scored, run.mass_file)

    partitioned_lookup = _lookup_cache_get(
        psm_lookup_cache, run.percolator_psms, build_partitioned_reference_lookup
    )
    est_q = np.full(len(winners), np.nan, dtype=np.float64)
    est_pep = np.full(len(winners), np.nan, dtype=np.float64)
    matched = np.full(len(winners), np.nan, dtype=np.float64)
    fallback = np.zeros(len(winners), dtype=bool)

    for input_file_key, index_values in winners.groupby("input_file_key", dropna=False).groups.items():
        index_list = list(index_values)
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
    winners["protein_ids"] = winners["Proteins"].astype(str).map(_protein_ids_csv_from_text)
    return winners


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

        for warning in warnings:
            text = str(warning).strip()
            if text:
                logger.warning(text)

        modified_summary = _make_modified_summary(novel_psms)
        novel_psms_out = _make_psm_output_table(novel_psms)

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
