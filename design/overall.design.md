# ProtCosmo Overall Design

This document describes the current runtime behavior implemented in:

- `src/protcosmo/protcosmo.py`
- `src/protcosmo/utils/*`

## 1. Goal

ProtCosmo is a CLI pipeline with two execution paths:

1. CometPlus path: resolve mass-spectrum input(s), run CometPlus, score PIN candidates, pick winner PSMs, estimate q/PEP, and export reports.
2. Direct PIN path (`--input-pin`): skip CometPlus and score an existing PIN directly.

It also supports early-stop modes:

1. `--stop-after-saving-novel-peptide`
2. `--stop-after-cometplus`

Early-stop modes stop before scoring outputs.

## 2. Runtime Flow

## Step 0: CLI parse and entry checks (`protcosmo.main`)

1. Build parser.
2. Parse known args; unknown args become `passthrough_args`.
3. Reject legacy `--input_tsv` passthrough usage.
4. Call `run_pipeline(args, passthrough_args)`.

## Step 1: Normalize runtime config (`utils.config_loader.load_pipeline_config`)

1. Normalize `--output-prefix` (non-empty).
2. Read control flags:
   - `--stop-after-saving-novel-peptide`
   - `--stop-after-cometplus`
   - `--force`
   - `--log`
   - `--input-pin`
3. Validate mode conflicts:
   - stop-after flags are mutually exclusive;
   - `--input-pin` cannot combine with stop-after flags.
4. If `--input-pin` is set:
   - resolve to absolute path;
   - require scoring references (`--init-weights`, `--percolator-psms`, `--percolator-peptides`);
   - create one `RunConfig` using this PIN.
5. Else (CometPlus path):
   - resolve `--mass-file` inputs;
   - detect novel mode if any of `--novel_protein`, `--novel_peptide`, `--internal_novel_peptide` is set;
   - merge multi-file novel inputs into one run;
   - require single-value `--params` and `--database`.
6. Scoring references are optional in stop-after modes, required otherwise.
7. If scan filters are provided with multi-run mode, disable scan filters and append a warning.

Output is `PipelineConfig` with normalized runs and runtime booleans (`force`, `log`).

## Step 1.1: Mass-file resolution (`utils.mass_file_resolver`)

`--mass-file` supports:

1. Single file.
2. Comma-separated list.
3. Directory scan for supported suffixes.
4. List file (one path per line; `#` comments allowed).

Resolver behavior:

1. Resolve relative paths.
2. Validate existence.
3. Keep supported spectrum formats for directory mode.
4. De-duplicate while preserving order.
5. For ambiguous text files, attempt list parsing then fallback to direct path.

## Step 2: Initialize run context (`protcosmo.run_pipeline`)

1. Ensure output directory exists.
2. Initialize optional log file path:
   - if `--log` is set: `<output-dir>/<output-prefix>.log`
   - otherwise: screen-only logging.
3. Initialize runtime caches and warning collector.
4. Compute `stop_after_any`.

## Step 3: CometPlus execution (`utils.comet_runner`, CometPlus path)

For each run:

1. Build command with fixed options:
   - `--params`, `--database`, `--output-folder`, `--output_percolatorfile 1`, `--max_duplicate_proteins -1`.
2. Append optional ProtCosmo-managed options:
   - novel options, scan filters (when enabled), stop/keep/thread/run-comet-each.
3. In novel mode, if `--output_internal_novel_peptide` is missing, auto-set default output path.
4. Append passthrough args and mass-file inputs.
5. PIN reuse/overwrite rule (PIN-required runs only):
   - target file is `<output-prefix>.cometplus.novel.pin` in output dir;
   - if target exists and `--force` is not set: skip CometPlus and reuse existing PIN;
   - if target exists and `--force` is set: delete target and rerun CometPlus.
6. Execute with captured stdout/stderr.
7. Rename CometPlus internal `command.stdout.log` / `command.stderr.log` to prefixed names when present.
8. Detect produced PIN (`*.pin*`), then normalize novel PIN to `<output-prefix>.cometplus.novel.pin`.
9. Return captured stdout/stderr text, skip state, overwrite state, and PIN path.

`protcosmo.run_pipeline` prints CometPlus stdout/stderr to screen and also writes them into `<output-prefix>.log` when `--log` is enabled.

## Step 4: Score input source

1. Input-pin mode: score provided PIN directly.
2. CometPlus mode: score returned PIN per run (unless stop-after mode).

## Step 5: Scoring and winner selection

Per run:

1. Read PIN (`.pin`, `.pin.gz`, `.parquet`, `.parquet.gz`).
2. Parse selected Percolator models from weights (raw rows like 2/4/6).
3. Score all candidates with linear models and average into `final_score`.
4. Select one winner PSM per spectrum.
5. Estimate PSM q/PEP by nearest smaller-or-equal lookup against `--percolator-psms` (partition-aware by input key).
6. Collect winner tables across runs.

## Step 6: Early-stop behavior

If `stop_after_any` is true:

1. Skip novel summary report generation.
2. Print collected warnings to screen (and log file when enabled).
3. Return outputs map (empty unless `--log` was set).

## Step 7: Novel subset and peptide/protein summaries

Normal mode only:

1. Concatenate winner tables.
2. Keep `novel_only` winners.
3. Build modified/unmodified peptide forms and novel protein IDs.
4. Estimate peptide q/PEP from `--percolator-peptides`.
5. Build:
   - PSM output table (reference-style columns),
   - modified peptide summary table.
6. Internal unmodified/protein summaries are computed for runtime consistency, but not exported as files.

## Step 8: Outputs

## Report files

Normal mode outputs:

1. `<output-prefix>.nove.psms.tsv`
2. `<output-prefix>.novel.peptides.tsv`

Stop-after modes output no report TSV.

## Runtime log output

1. Default: logs and warnings are shown on screen.
2. With `--log`: same logs/warnings are also written to `<output-prefix>.log`.

## Removed artifacts

ProtCosmo no longer writes these files:

1. `<output-prefix>.cometplus.run_XXXX.stdout.log`
2. `<output-prefix>.cometplus.run_XXXX.stderr.log`
3. `<output-prefix>.run_metadata.json`
4. `<output-prefix>.warnings.log`

## 3. Important Behavior Notes

1. Unknown CLI options are passed through to CometPlus only in CometPlus path.
2. `--force` only affects novel PIN overwrite behavior.
3. Novel PIN skip/reuse check uses exactly `<output-prefix>.cometplus.novel.pin`.
4. q-value/PEP values are lookup estimates (not retrained Percolator outputs).
5. Output score fields come from matched reference-score lookup values.
6. Caches are reused across runs to avoid duplicate model/reference loads.
