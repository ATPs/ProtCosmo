# ProtCosmo Overall Design

This document describes the current runtime design implemented in:

- `src/protcosmo/protcosmo.py`
- `src/protcosmo/utils/*`

## 1. Goal

ProtCosmo is a CLI pipeline that supports two execution paths:

1. CometPlus path: resolve mass-spectrum input(s), run CometPlus, score PIN, select winners, estimate q/PEP, and export reports.
2. Direct PIN path (`--input-pin`): skip CometPlus, score the provided PIN directly, then continue with winner selection/estimation/reporting.

It also supports early-stop modes:

1. `--stop-after-saving-novel-peptide`
2. `--stop-after-cometplus`

Both stop before scoring and write metadata/warnings only.

## 2. Runtime Flow (Internal Steps)

## Step 0: CLI parsing and entry checks (`protcosmo.main`)

1. Build parser with ProtCosmo arguments.
2. Parse known args; unknown args become `passthrough_args`.
3. Reject legacy `--input_tsv` if present in passthrough args.
4. Call `run_pipeline(args, passthrough_args)`.

## Step 1: Build normalized pipeline config (`utils.config_loader.load_pipeline_config`)

1. Normalize and validate `--output-prefix` (must be non-empty).
2. Read control flags:
   - `--stop-after-saving-novel-peptide`
   - `--stop-after-cometplus`
   - `--input-pin`
3. Validate mode conflicts:
   - `--stop-after-saving-novel-peptide` and `--stop-after-cometplus` are mutually exclusive.
   - `--input-pin` cannot be combined with either stop-after flag.
4. If `--input-pin` is set:
   - resolve `--input-pin` path to absolute path;
   - require `--init-weights`, `--percolator-psms`, `--percolator-peptides`;
   - build exactly one `RunConfig` using the pin path as `mass_file`;
   - do not require `--mass-file`, `--params`, `--database`.
5. Else (CometPlus path):
   - require `--mass-file` and resolve it via `resolve_mass_files`;
   - detect novel mode if any of `--novel_protein`, `--novel_peptide`, `--internal_novel_peptide` is set;
   - if novel mode + multiple mass files, merge into one run; otherwise one run per mass file;
   - require single-value `--params` and `--database`.
6. Scoring reference requirements:
   - normal mode: `--init-weights`, `--percolator-psms`, `--percolator-peptides` are required;
   - stop-after modes: these are optional.
7. For multi-run CometPlus mode, if any scan filter is set (`--scan`, `--scan_numbers`, `--first-scan`, `--last-scan`), disable scan filters and append warning.

Output is `PipelineConfig` with normalized `runs`.

## Step 1.1: Mass-file resolver (`utils.mass_file_resolver`)

`--mass-file` supports:

1. Single file.
2. Comma-separated list.
3. Directory (collect supported suffixes).
4. List file (one path per line; `#` comments; comma allowed per line).

Resolver behavior:

1. Resolve relative paths against cwd (or list-file directory for list entries).
2. Validate existence.
3. For directories, keep only supported spectrum formats.
4. De-duplicate while preserving order.
5. For non-supported single files, attempt list parsing first; if empty, fallback to direct path.

## Step 2: Initialize runtime context (`protcosmo.run_pipeline`)

1. Record UTC start time.
2. Ensure output directory exists.
3. Initialize caches and collectors:
   - model cache,
   - PSM lookup cache,
   - peptide lookup cache,
   - warnings,
   - winner frame parts,
   - command records.
4. Compute early-stop flag:
   - `stop_after_any = stop_after_saving_novel_peptide or stop_after_cometplus`.

## Step 3: CometPlus execution (`utils.comet_runner`, CometPlus path only)

For each run:

1. Build command with fixed options:
   - `--params`, `--database`, `--output-folder`, `--output_percolatorfile 1`, `--max_duplicate_proteins -1`.
2. Append optional ProtCosmo-managed flags:
   - novel inputs/outputs (`--novel_protein`, `--novel_peptide`, `--output_internal_novel_peptide`, `--internal_novel_peptide`),
   - `--stop-after-saving-novel-peptide`,
   - `--keep-tmp`,
   - `--run-comet-each`,
   - `--thread`,
   - scan filters (only if enabled by config).
3. If in novel mode and `--output_internal_novel_peptide` is not provided, auto set:
   - `<output-prefix>.internal_novel_peptide.tsv` under output dir.
4. Append passthrough args and run mass files.
5. Execute by `subprocess.run(..., capture_output=True)`.
6. Write ProtCosmo run logs:
   - `<prefix>.cometplus.run_XXXX.stdout.log`
   - `<prefix>.cometplus.run_XXXX.stderr.log`
7. If CometPlus emits `command.stdout.log` / `command.stderr.log`, rename to prefixed names:
   - `<prefix>.command.stdout.log`
   - `<prefix>.command.stderr.log`
8. If exit code is non-zero, raise runtime error with command + log paths.
9. If PIN is required (`require_pin_output=True`):
   - detect newest changed `*.pin*`;
   - in novel mode normalize to `<prefix>.cometplus.novel.pin` (including `.pin.gz` decompression and parquet->tsv conversion).

Each run is recorded in metadata command records.

## Step 4: Score source selection

1. If `--input-pin` mode:
   - skip Step 3 entirely;
   - use the provided PIN path as scoring input.
2. Else:
   - use `pin_path` returned by Step 3 per run.

Scoring per run uses shared helper logic (`_score_winner_rows_from_pin`).

## Step 5: Per-run scoring and winner table build

For each run to be scored:

1. Validate scoring references are available (`init_weights`, `percolator_psms`, `percolator_peptides`).
2. Read PIN (`utils.pin_reader.read_pin`):
   - supports text/gz/parquet,
   - canonicalizes aliases,
   - enforces required logical columns.
3. Parse static models from `--init-weights` (`utils.weights_parser.parse_selected_models`):
   - for repeated Percolator CV blocks (`header -> normalized -> raw`), select raw rows (commonly numeric 2/4/6),
   - require `m0` intercept,
   - enforce aligned feature order.
4. Score every PIN candidate row (`utils.scoring.score_pin_candidates`):
   - for each selected model, compute `model_score_k = w_k^T x + b_k`,
   - resolve model feature names to PIN columns with flexible alias matching,
   - if needed, derive charge features (for example from `Charge1..ChargeN` one-hot),
   - coerce feature values to numeric and fill missing as `0.0`,
   - set `final_score = mean(model_score_1, model_score_2, model_score_3)` from the 3 selected raw models.
5. Select one winner PSM per spectrum (`utils.selection.select_best_psm_per_spectrum`):
   - sort by `final_score` descending,
   - tie-break 1: prefer non-`novel_only`,
   - tie-break 2: smaller `SpecId` rank suffix.
6. Estimate winner PSM q/PEP from `--percolator-psms` via nearest smaller-or-equal score lookup using winner `final_score`:
   - lookup partition key is input-file key parsed from `SpecId`/`PSMId` prefix (before first `_`),
   - score matching is performed within that partition when possible; fallback is full-table lookup.
7. Attach per-run metadata columns to winners and append to global winner list.

## Step 6: Early-stop handling

If `stop_after_any` is true:

1. Skip all downstream novel filtering/summaries.
2. Write only:
   - `<prefix>.warnings.log`
   - `<prefix>.run_metadata.json`
3. Set metadata `mode` to:
   - `stop_after_saving_novel_peptide` or
   - `stop_after_cometplus`.
4. Return immediately.

## Step 7: Novel subset and peptide/protein summaries

Normal mode only:

1. Concatenate all winners.
2. Keep `novel_only` winners as `novel_psms`.
3. Build peptide representations:
   - `modified_peptide`,
   - `unmodified_peptide`.
4. Extract `novel_protein_ids`.
5. Estimate peptide-level q/PEP from `--percolator-peptides`:
   - input score is still each winner row `final_score` (not a separate peptide-rescoring model),
   - same nearest smaller-or-equal lookup rule as PSM estimation,
   - lookup also uses input-file key partitions derived from `PSMId` prefix when available,
   - fallback when no smaller score exists: q-value=1 and PEP=1 (warning logged).
6. Build summaries from `novel_psms`:
   - modified peptide summary,
   - unmodified peptide summary,
   - novel protein summary.
7. Peptide lookup is done per input-file partition when available (`PSMId` prefix).
8. Final `novel.peptides.tsv` keeps one row per `peptide` by highest matched peptide `score`, so `peptide` is unique in output.

## Step 8: Output and metadata

Normal mode outputs:

1. `<prefix>.nove.psms.tsv` (exact current filename in code)
   - reference-style columns:
     `PSMId`, `score`, `q-value`, `posterior_error_prob`, `peptide`, `proteinIds`
   - `score` comes from matched reference score lookup (not raw `final_score`),
   - `proteinIds` are comma-joined,
   - rows sorted by `score` descending.
2. `<prefix>.novel.peptides.tsv`
   - reference-style columns only:
     `PSMId`, `score`, `q-value`, `posterior_error_prob`, `peptide`, `proteinIds`
   - one highest-score row kept per `peptide` (unique peptide values),
   - rows sorted by `score` descending.
3. `<prefix>.warnings.log`
4. `<prefix>.run_metadata.json`

Metadata includes:

1. start/end/duration,
2. argv and passthrough args,
3. output dir and output prefix,
4. `input_pin` (path or null),
5. warnings and counts,
6. per-run command records (empty in input-pin mode),
7. winner/novel counts.

## 3. Important Behavior Notes

1. Unknown CLI options are forwarded to CometPlus only when CometPlus path is used.
2. In `--input-pin` mode, CometPlus is not invoked and command records are empty.
3. Scan filters are effective only when final CometPlus run count is exactly one.
4. q-value/PEP are lookup estimates, not retrained Percolator outputs.
5. PSM and peptide estimates both use `final_score` from static model scoring; they differ by reference table (`--percolator-psms` vs `--percolator-peptides`) and input-file partition.
6. Peptide-level output rows are score-representative rows (highest matched peptide score), not global lowest-q aggregations.
7. Output score fields are matched reference scores.
8. Caches avoid re-loading duplicated models/references across runs.
9. In novel mode with multiple mass files, inputs are merged into one CometPlus invocation.
