"""CLI help text and formatting helpers."""

from __future__ import annotations

import argparse


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
- --input_tsv: row-based input table mode (required column: mass-file).
- --mass-file supports: single file, comma list, list file (one path per line), or directory.
- --input_tsv and --mass-file cannot be used together.
- --input-pin takes precedence when both --input-pin and --input_tsv are provided.
- --params and --database each accept only one value (required unless --input-pin is set, or set via --input_tsv rows).
- --init-weights/--percolator-psms/--percolator-peptides each accept one CLI value.
- In --input_tsv mode, per-mass-file scoring variation is configured from TSV rows.
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
- --thread: forwarded to CometPlus; also controls parallel worker count when scoring multiple TSV init-weight groups.

{ESTIMATION_NOTE}
""".strip()


FULL_HELP_TEXT = f"""
Detailed workflow:

Step 1. Build run configuration
- If --input-pin is provided:
  - ProtCosmo skips CometPlus and scores this PIN directly.
  - --mass-file/--params/--database are not required.
- If --input_tsv is provided (and --input-pin is not):
  - --mass-file is not allowed.
  - TSV must have a header and required column: mass-file.
  - Optional TSV columns: params, database, init-weights, percolator-psms, percolator-peptides.
  - Header matching is case-insensitive; dash/underscore aliases are accepted.
  - Unknown columns are ignored.
  - mass-file cell must be one file path (not comma list, not list-file, not directory).
  - CLI scoring refs (--init-weights/--percolator-psms/--percolator-peptides) override TSV values globally.
  - Effective params/database must each resolve to one unique value across rows.
  - Each mass-file basename key must be unique across rows.
  - All TSV mass files are merged into one CometPlus run input.
- Else (no --input_tsv):
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
- CLI --init-weights/--percolator-psms/--percolator-peptides each accept only one value.
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
- In --input_tsv mode:
  - with one effective init-weights: score merged PIN normally;
  - with multiple init-weights: split merged PIN by SpecId key
    using `SpecId.rsplit('_', maxsplit=3)[0]` and score each group independently;
  - if --thread > 1, groups are scored in parallel with up to --thread workers;
  - each unique init-weights must map to exactly one percolator-psms and one percolator-peptides.

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
- Output proteinIds remap:
  - COMETPLUS_NOVEL_* IDs are mapped to real protein_id values from:
    1) --internal_novel_peptide when provided
    2) otherwise <output-dir>/<output-prefix>.internal_novel_peptide.tsv
  - Unmapped COMETPLUS_NOVEL_* tokens are kept and logged as warnings.

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
- Percolator weight file used to score Comet PIN candidates (single CLI value only).
- Expected shape: header row with feature names (must include m0), then numeric rows.
- ProtCosmo prefers raw-weight rows (commonly numeric rows 2, 4, and 6).

--input-pin <file>
- Skip CometPlus and directly score this PIN.
- Requires: --init-weights, --percolator-psms, --percolator-peptides.
- --mass-file/--params/--database are not required in this mode.

--input_tsv <file>
- Optional row table mode.
- Required column: mass-file.
- Optional columns: params, database, init-weights, percolator-psms, percolator-peptides.
- Header order does not matter; aliases are accepted (case-insensitive, dash/underscore).
- Unknown columns are ignored.
- In this mode, --mass-file is not allowed.
- CLI scoring refs override TSV scoring columns globally.

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


def print_full_help(parser: argparse.ArgumentParser) -> None:
    """Print parser help plus detailed workflow notes."""

    parser.print_help()
    print("")
    print(FULL_HELP_TEXT)
