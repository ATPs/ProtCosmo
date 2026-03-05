# ProtCosmo Overall Design

This document describes how ProtCosmo runs internally, step by step, based on the current implementation in:

- `src/protcosmo/protcosmo.py`
- `src/protcosmo/utils/*`

## 1. Goal

ProtCosmo is a CLI pipeline that:

1. Resolves one or more mass-spectrum inputs.
2. Runs CometPlus search.
3. Reads PIN candidates and applies static scoring from Percolator weights.
4. Selects one winner PSM per spectrum.
5. Estimates q-value/PEP from Percolator reference tables by score lookup.
6. Exports novel-focused reports, warnings, and run metadata.

## 2. Runtime Flow (Internal Steps)

## Step 0: CLI parsing and entry checks (`protcosmo.main`)

1. Build parser with fixed ProtCosmo arguments and help text.
2. Parse known args; unknown args are collected as `passthrough_args`.
3. Reject legacy `--input_tsv` if present in passthrough arguments.
4. Call `run_pipeline(args, passthrough_args)`.

## Step 1: Build normalized pipeline config (`utils.config_loader.load_pipeline_config`)

1. Validate `--output-prefix` is non-empty.
2. Resolve `--mass-file` to concrete files using `resolve_mass_files`.
3. Detect novel mode when any of these are set:
   - `--novel_protein`
   - `--novel_peptide`
   - `--internal_novel_peptide`
4. If novel mode + multiple mass files:
   - merge into one CometPlus run containing all mass files.
   - otherwise run one CometPlus call per mass file.
5. Enforce single-value-only for:
   - `--params`
   - `--database`
6. Map these fields to run count (broadcast 1 value or strict N:1):
   - `--init-weights`
   - `--percolator-psms`
   - `--percolator-peptides`
7. If multiple runs and any scan filter is set (`--scan`, `--scan_numbers`, `--first-scan`, `--last-scan`):
   - disable scan filters.
   - add warning.

Output of this step is `PipelineConfig` with `runs: List[RunConfig]`.

## Step 1.1: Mass-file resolver internals (`utils.mass_file_resolver`)

`--mass-file` supports:

1. Single file.
2. Comma-separated list.
3. Directory (collect supported suffixes).
4. List file (one path per line, `#` comments allowed, comma also allowed per line).

Internal behavior:

1. Resolve relative paths against current working directory (or list-file directory for list entries).
2. Validate existence.
3. For directories, include only supported mass formats.
4. De-duplicate resolved files while preserving order.
5. For non-supported single files, attempt list parsing first; if list has no usable entries, fallback to treating it as direct mass file path.

## Step 2: Initialize pipeline context (`protcosmo.run_pipeline`)

1. Record start timestamp (UTC).
2. Ensure output directory exists.
3. Initialize:
   - warnings list (seeded from config warnings),
   - model cache (`init_weights` -> parsed models),
   - PSM lookup cache (`percolator_psms` -> lookup table),
   - peptide lookup cache (`percolator_peptides` -> lookup table),
   - winner collection list,
   - Comet command record list.

## Step 3: Build and run CometPlus command per run (`utils.comet_runner`)

For each `RunConfig`:

1. Build command with required fixed options:
   - `--params <abs path>`
   - `--database <abs path>`
   - `--output-folder <abs output-dir>`
   - `--output_percolatorfile 1`
   - `--max_duplicate_proteins -1`
2. Append optional ProtCosmo-managed options when present:
   - novel inputs/outputs (`--novel_protein`, `--novel_peptide`, `--output_internal_novel_peptide`, `--internal_novel_peptide`)
   - `--stop-after-saving-novel-peptide`
   - `--keep-tmp`
   - `--run-comet-each` (enabled by default)
   - `--thread`
   - scan filters (only when enabled by Step 1 rules)
3. Append passthrough args unchanged.
4. Append run mass file(s).
5. Execute via `subprocess.run(..., capture_output=True)`.
6. Write logs:
   - `<prefix>.cometplus.run_XXXX.stdout.log`
   - `<prefix>.cometplus.run_XXXX.stderr.log`
7. If CometPlus returns non-zero:
   - raise runtime error with command and log paths.
8. If PIN output is required:
   - detect newest changed `*.pin*`.
   - in novel mode, normalize output name to `<prefix>.cometplus.novel.pin` (decompress/convert if needed).

Each run is recorded into metadata (`command`, shell-escaped command, logs, return code, PIN path).

## Step 4: Early-stop mode (`--stop-after-saving-novel-peptide`)

If enabled:

1. Skip all PIN scoring/selection logic.
2. Write only:
   - `<prefix>.warnings.log`
   - `<prefix>.run_metadata.json`
3. Return immediately.

## Step 5: Read PIN and parse static models

For each run (normal mode):

1. Read PIN table (`utils.pin_reader.read_pin`):
   - supports TSV-like text, `.gz`, and parquet.
   - canonicalizes known column aliases.
   - requires at least `SpecId` and `Peptide`.
   - ensures numeric conversion for known feature columns.
2. Parse models from `--init-weights` (`utils.weights_parser.parse_selected_models`):
   - parse header + numeric rows,
   - select numeric rows 1, 3, and 5 only,
   - require `m0` as intercept column,
   - output `LinearModel(feature_names, weights, intercept)`.
3. Validate all selected models share identical feature order.

## Step 6: Score PIN candidates (`utils.scoring.score_pin_candidates`)

For each selected model:

1. Resolve model feature names to PIN columns using normalized matching + alias rules.
2. If needed, derive charge-related features from:
   - direct `ChargeN`, or
   - one-hot `Charge1..Charge6`.
3. Build numeric matrix `X` and compute:
   - `model_score_k = X @ w + b`
4. After all 3 models:
   - `final_score = mean(model_score_1, model_score_2, model_score_3)`.

If required feature mapping fails, pipeline raises an explicit error with suggestions.

## Step 7: Winner selection per spectrum (`utils.selection.select_best_psm_per_spectrum`)

1. Derive:
   - `spectrum_id` from `SpecId` without trailing `_rank`,
   - `rank_index` from trailing numeric rank (else very large fallback),
   - `novel_only` from `Proteins` (all protein IDs must be `COMETPLUS_NOVEL_...`, ignoring optional `DECOY_` prefix).
2. Sort by:
   - `mass_file` ascending,
   - `spectrum_id` ascending,
   - `final_score` descending,
   - `novel_only` ascending (non-novel preferred on tie),
   - `rank_index` ascending.
3. Keep first row per (`mass_file`, `spectrum_id`).

## Step 8: PSM-level estimated q/PEP (`utils.percolator_ref`)

1. Load reference table from `--percolator-psms` (TSV/parquet).
2. Detect required logical columns by candidate names:
   - score,
   - q-value,
   - posterior error probability (PEP).
3. Normalize to numeric and drop invalid rows.
4. Sort by score and keep best row for duplicate score.
5. For each winner `final_score`, use nearest smaller-or-equal reference score (`searchsorted(..., side="right") - 1`).
6. If no smaller score exists:
   - assign `estimated_psm_q_value = 1`,
   - assign `estimated_psm_pep = 1`,
   - flag fallback and add warning.

## Step 9: Novel subset and peptide/protein summaries

1. Keep `novel_only` winners as `novel_psms`.
2. Derive peptide forms (`utils.peptide_utils`):
   - `modified_peptide`: remove flanking residues from `A.PEPTIDE.B` style.
   - `unmodified_peptide`: remove bracket mods and keep amino-acid letters only.
3. Extract `novel_protein_ids` from `Proteins`.
4. Estimate peptide-level q/PEP using `--percolator-peptides` with same lookup rule as Step 8.
5. Build in-memory summaries:
   - modified peptide summary,
   - unmodified peptide summary,
   - novel protein summary.

## Step 10: Write output files (`utils.report_writer`)

Normal mode writes:

1. `<prefix>.nove.psms.tsv` (exact current filename in code)
2. `<prefix>.novel.peptides.tsv` (modified-peptide summary)
3. `<prefix>.warnings.log`
4. `<prefix>.run_metadata.json`

Metadata includes:

1. Start/end time and duration.
2. Original argv and passthrough args.
3. Run count, warning count, warnings.
4. Per-run command records and output paths.
5. Counts for all winners, novel PSMs, novel modified peptides, novel unmodified peptides, and novel proteins.

## 3. Important Behavior Notes

1. Unknown CLI options are intentionally forwarded to CometPlus.
2. Scan filters are only effective when final run count is exactly one.
3. q-value/PEP are estimated by lookup, not re-trained/recomputed Percolator results.
4. Caches avoid re-loading repeated weights/reference files across runs.
5. In novel mode with multiple mass files, ProtCosmo intentionally merges them into one CometPlus invocation.
