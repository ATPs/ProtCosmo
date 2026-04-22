# ProtCosmo Overall Design

This document describes the current runtime behavior implemented in:

- `src/protcosmo/protcosmo.py`
- `src/protcosmo/utils/*`

Runtime helper placement:

- CLI long help text/formatter: `src/protcosmo/utils/help_text.py`
- Runtime logger: `src/protcosmo/utils/runtime_logging.py`
- Shared cache get-or-load helper: `src/protcosmo/utils/cache_utils.py`
- Grouped scoring helpers: `src/protcosmo/utils/scoring_batches.py`
- Novel-output remap/report table helpers: `src/protcosmo/utils/novel_reports.py`
- `protcosmo.py` binds scoring helpers as direct aliases from `utils.scoring_batches` and keeps a thin grouped-scoring bridge for patch/test compatibility.

Documentation sync:

- `README.md` includes end-to-end usage guidance and detailed CLI examples for all three run modes.

## 1. Goal

ProtCosmo is a CLI pipeline with three configuration paths:

1. CometPlus path (`--mass-file`): resolve mass-spectrum input(s), run CometPlus, score PIN candidates, pick winner PSMs, estimate q/PEP, and export reports.
2. TSV path (`--input_tsv`): load row-based mass/scoring metadata, run one merged CometPlus search, then score with one or multiple init-weight groups.
3. Direct PIN path (`--input-pin`): skip CometPlus and score an existing PIN directly.

It also supports early-stop modes:

1. `--stop-after-saving-novel-peptide`
2. `--stop-after-cometplus`

Early-stop modes stop before scoring outputs.

## 2. Runtime Flow

## Step 0: CLI parse and entry checks (`protcosmo.main`)

1. Build parser.
2. Parse known args; unknown args become `passthrough_args`.
3. `--help-full` prints detailed help text.
4. Call `run_pipeline(args, passthrough_args)`.

## Step 1: Normalize runtime config (`utils.config_loader.load_pipeline_config`)

1. Normalize `--output-prefix` (non-empty).
2. Read control flags:
   - `--stop-after-saving-novel-peptide`
   - `--stop-after-cometplus`
   - `--force`
   - `--log`
   - `--input-pin`
   - `--input_tsv`
3. Validate mode conflicts:
   - stop-after flags are mutually exclusive;
   - `--input-pin` cannot combine with stop-after flags.
4. Resolve optional parquet fast-path inputs:
   - `--ms2-parquet` and `--mgf-parquet-dir` must be provided together;
   - both paths must exist;
   - parquet fast path is rejected with `--input-pin`;
   - parquet fast path requires novel-mode inputs;
   - when the user provides `--thread`, the parquet fast path forwards that exact value to both CometPlus passes; otherwise it leaves thread handling unchanged.
5. If `--input-pin` is set:
   - resolve to absolute path;
   - require scoring references (`--init-weights`, `--percolator-psms`, `--percolator-peptides`), each single CLI value;
   - create one `RunConfig` using this PIN.
   - If `--input_tsv` is also provided, input-pin mode wins and TSV rows are ignored.
6. Else if `--input_tsv` is set:
   - reject simultaneous `--mass-file`;
   - parse TSV with header required;
   - required column: `mass-file`;
   - optional columns: `params`, `database`, `init-weights`, `percolator-psms`, `percolator-peptides`;
   - header matching is case-insensitive and accepts dash/underscore aliases;
   - unknown columns are ignored.
7. TSV mode row handling:
   - each `mass-file` cell must be one file path (no comma/list-file/directory semantics);
   - resolve paths;
   - derive per-row mass key from basename without suffix;
   - fail on mass-key collisions;
   - apply CLI override for scoring refs (`--init-weights`, `--percolator-psms`, `--percolator-peptides`), each single CLI value;
   - resolve effective `params`/`database` and require one unique value each across rows;
   - scoring refs are required unless stop-after mode.
8. TSV scoring groups:
   - group rows by effective `init-weights`;
   - each unique init-weights must map to exactly one `percolator-psms` and one `percolator-peptides`;
   - emit one merged run containing all TSV mass files.
9. Else (standard `--mass-file` mode):
   - resolve mass files via resolver (single/comma/list-file/directory);
   - detect novel mode if any of `--novel_protein`, `--novel_peptide`, `--internal_novel_peptide` is set;
   - merge multi-file novel inputs into one run;
   - require single-value `--params`, `--database`;
   - require single-value scoring refs unless stop-after mode.

10. Scan-filter gating:
    - if final run count > 1 and scan args are present, disable scan filters and append warning.

Output is `PipelineConfig` with normalized runs and runtime booleans (`force`, `log`, `use_input_tsv`) plus fast-path metadata (`ms2_parquet`, `mgf_parquet_dir`, `fastpath_enabled`, `fastpath_thread`) and optional TSV scoring group metadata.

## Step 1.1: Shared input key extraction

`utils.input_key.extract_input_file_key(spec_id)` is used across pipeline components and follows:

- `str(spec_id).rsplit("_", maxsplit=3)[0]`

This keeps key derivation consistent for:

1. Winner row partition keys (`selection.py`).
2. Reference partition keys from `PSMId` (`percolator_ref.py`).
3. TSV grouped split of merged PIN rows.

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
   - novel options, `--known_peptide`, scan filters (when enabled), stop/keep/thread/run-comet-each.
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

## Step 3.1: Optional two-pass parquet fast path (`utils.parquet_fastpath`)

When `fastpath_enabled` is true, `protcosmo.run_pipeline` uses a two-pass search path instead of the standard one-pass mzML/mzMLb flow.

1. Reuse rule:
   - if `<output-prefix>.cometplus.novel.pin` already exists and `--force` is not set, skip both fast-path passes and reuse the existing PIN;
   - if `--force` is set, rerun pass 1, rebuild staged MGFs, and rerun pass 2.
2. Pass 1: internal novel export
   - run CometPlus on the original `mass_files`;
   - forward `--novel_peptide` or `--novel_protein`;
   - forward `--known_peptide` when present;
   - always request `<output-dir>/<output-prefix>.internal_novel_peptide.tsv` unless an explicit path was provided;
   - add `--stop-after-saving-novel-peptide`;
   - if `--thread` was provided, forward that exact value.
3. Internal TSV reuse:
   - if `--internal_novel_peptide` is provided, ProtCosmo reuses it directly;
   - else if the expected internal TSV already exists and `--force` is not set, ProtCosmo reuses that TSV and skips pass 1.
4. DuckDB subset staging:
   - read the detailed internal TSV and require `charge`, `mz_window_min`, and `mz_window_max`;
   - read charge policy from the run's `.params` file: `override_charge`, `precursor_charge`, and `isotope_error`;
   - build isotope-shifted mz windows from the same signed isotope-offset set that CometPlus uses for `isotope_error` (`0`, `+1`, `-1`, `+4`, etc., depending on the selected mode);
   - for each non-zero isotope offset, append a copy of the window table with `mz_window_min` and `mz_window_max` shifted by `offset_mass / charge`; this mirrors CometPlus's behaviour of subtracting isotope shifts from the experimental mass window, so raw precursor mz values land in the same effective search windows;
   - match scan windows against `--ms2-parquet` by `idn`, charge (per `override_charge` semantics), and precursor `mz` against the combined original + isotope-shifted windows;
   - write one staged `<basename>.mgf` file per matched input under `<output-prefix>.fastpath_subset_mgfs/`;
   - preserve the original numeric basenames so grouped scoring and `SpecId` partitioning stay unchanged;
   - write:
     - `<output-prefix>.fastpath.scan_matches.tsv`
     - `<output-prefix>.fastpath.mgf_manifest.tsv`
     - `<output-prefix>.fastpath.timing.json`
5. Pass 2: resumed search
   - run CometPlus on the staged subset MGFs;
   - use `--internal_novel_peptide <exported-or-reused TSV>`;
   - do not forward `--novel_peptide`, `--novel_protein`, or `--known_peptide`;
   - preserve `--run-comet-each`;
   - if `--thread` was provided, forward that exact value.
6. Cleanup:
   - staged subset MGFs are removed after the run unless `--keep-tmp` is set;
   - manifest/timing files remain in the output directory.

## Step 4: Score input source

1. Input-pin mode: score provided PIN directly.
2. CometPlus mode:
   - normal: score returned PIN directly;
   - TSV mode with one scoring group: same as normal;
   - TSV mode with multiple init-weight groups: split merged PIN by `input_file_key` and score each group independently;
   - grouped scoring parallelism: if `--thread > 1`, run groups in parallel with up to `min(--thread, matched_group_count)` workers.

## Step 5: Scoring and winner selection

Per scoring batch:

1. Read PIN (`.pin`, `.pin.gz`, `.parquet`, `.parquet.gz`) into DataFrame.
2. Parse selected Percolator models from weights (raw rows like 2/4/6).
3. Score all candidates with linear models and average into `final_score`.
4. Select one winner PSM per spectrum.
5. Estimate PSM q/PEP by nearest smaller-or-equal lookup against `--percolator-psms` (partition-aware by input key).
6. Collect winner tables across runs/groups.
7. For parallel grouped scoring, collect per-group warnings and append them in group index order for stable output.

## Step 6: Early-stop behavior

If `stop_after_any` is true:

1. Skip novel summary report generation.
2. In fast-path mode, `--stop-after-saving-novel-peptide` stops after pass 1 and does not build staged MGFs or run pass 2.
3. In fast-path mode, `--stop-after-cometplus` still runs pass 2 and then stops before PIN scoring/report generation.
4. Print collected warnings to screen (and log file when enabled).
5. Return outputs map (empty unless `--log` was set).

## Step 7: Novel subset, peptide estimation, and protein-id remap

Normal mode only:

1. Concatenate winner tables.
2. Keep `novel_only` winners.
3. Build modified/unmodified peptide forms and novel protein IDs.
4. Estimate peptide q/PEP from `--percolator-peptides`.
5. Build peptide-id to protein-id mapping source:
   - `--internal_novel_peptide` if provided;
   - otherwise `<output-dir>/<output-prefix>.internal_novel_peptide.tsv`.
6. Remap output protein IDs:
   - replace `COMETPLUS_NOVEL_*` tokens with mapped real `protein_id` value(s);
   - dedupe and join with comma;
   - if mapping file missing or peptide_id unmapped, keep original token and emit warning.
7. Build:
   - PSM output table (reference-style columns),
   - modified peptide summary table.

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

## 2.1 Local profiling workflow

`local_test/profile_peptide_runtime.py` is a local-test helper for timing analysis of one novel-peptide run on real PXD010154 data.

It is not part of the production CLI path; instead it reproduces the standard single-run flow with explicit timers:

1. Resolve one `idn` to:
   - mass file under `ms2mzMLb/`
   - group-specific weights / percolator references from `PXD010154.info.tsv`
2. Build the same ProtCosmo CLI argument set used by `python -m protcosmo.protcosmo`.
3. Call `load_pipeline_config(...)` to reuse normal path validation and argument normalization.
4. Call `run_cometplus_search(...)` directly and save raw CometPlus stdout/stderr text.
5. Parse CometPlus stderr/stdout timing lines such as:
   - `known peptide extraction done`
   - `novel candidate assembly done`
   - `novel candidate subtraction done`
   - `novel mass calculation done`
   - `internal novel peptide export done`
   - `scan prefilter ...`
   - stdout search lines like `- searching "<path>" ... 39 sec (...)`
   - any other `[...] <label> done (<sec>; total <sec>)` lines
6. Continue ProtCosmo scoring/report steps explicitly and time each major stage:
   - PIN read
   - weights parse
   - candidate scoring
   - winner selection
   - PSM reference lookup build
   - PSM q/PEP estimation
   - novel subset build
   - peptide reference lookup build
   - peptide q/PEP estimation
   - internal novel mapping load
   - report table construction
   - report TSV writes
7. Write timing outputs to the profiling output directory:
   - `timing.steps.tsv`
   - `timing.summary.json`
   - raw CometPlus stdout/stderr dumps
8. Profiling results can be summarized in `notes/` with:
   - the exact outer profiling command
   - the exact generated ProtCosmo command
   - measured stage timings
   - full-dataset extrapolation and comparison with prior large-scale runs

## 2.2 Real multi-input full-scale profiling workflow

For PRIDE-style cached datasets such as `PXD010154`, the real full-scale peptide workflow should follow the same shape as `protcosmo_PRIDE.py`:

1. Reuse the prepared `ms2duck/protcosmo.input.tsv`.
2. Run one repo-local ProtCosmo command with:
   - `python -m protcosmo.protcosmo`
   - `--input_tsv <.../protcosmo.input.tsv>`
   - `--novel_peptide <...>`
   - `--thread <N>`
   - `--log`
3. Do not wrap the run in GNU Parallel for the main execution step.
4. Let CometPlus handle its own internal parallelism:
   - `mzMLb` process-parallel scan prefilter workers
   - grouped `run-comet-each` shard searches

Important runtime behavior for this mode:

1. `run_cometplus_search(...)` uses `subprocess.run(..., capture_output=True)`.
2. Because of that, `<output-prefix>.log` stays empty while the CometPlus subprocess is still running.
3. After CometPlus exits, ProtCosmo prints and mirrors the captured stdout/stderr into `<output-prefix>.log`.
4. This means real-time progress is visible from process state and generated intermediate files, while the final timing breakdown becomes available only after the CometPlus subprocess returns.

Recommended places to read timing information after a full-scale run:

1. `/usr/bin/time` output file:
   - total wall time
   - user/sys CPU time
   - RSS
2. `<output-prefix>.log`:
   - cumulative CometPlus wall times from `... done (<sec>; total <sec>)`
   - per-input prefilter timings from `scan prefilter: ...`
   - grouped merge/search behavior from `run-comet-each grouped merge`, `- searching ...`, and `run-comet-each done`
3. Final output files:
   - `<output-prefix>.internal_novel_peptide.tsv`
   - `<output-prefix>.cometplus.novel.pin`
   - `<output-prefix>.nove.psms.tsv`
   - `<output-prefix>.novel.peptides.tsv`

## 2.3 PXD010154 parquet benchmark note generator

`local_test/benchmark_pxd010154_parquet_fastpath.py` is a repo-local analysis helper for the April 21, 2026 PXD010154 comparison note.

It is not part of the production CLI path. Its job is to either run or reread the completed baseline and fast-path output directories, optionally run a dedicated baseline `--keep-tmp --stop-after-cometplus` recovery pass, and then write `notes/20260421.pxd010154_parquet_fastpath.runtime.md` with a reproducible timing breakdown plus exact scan-set attribution.

It also supports `--execute-fastpath-only`, which reruns only the fast-path case and reuses existing baseline outputs without overwriting the baseline directory's `/usr/bin/time` sidecar.

When the benchmark is rerun into an existing output directory, ProtCosmo appends to `protcosmo.log`. The helper therefore parses the latest fast-path-enabled or fast-path-disabled run block instead of the first matching timing lines.

The helper records:

1. Outer `/usr/bin/time` metrics:
   - wall time
   - user/sys CPU time
   - RSS
2. Core CometPlus step timings parsed from `protcosmo.log`:
   - known peptide import
   - novel candidate assembly/subtraction
   - novel mass calculation
   - internal novel peptide export/import
   - scan prefilter
   - `run-comet-each`
   - fast-path pass 1 / subset-build / pass 2 totals
3. Fast-path staging metrics parsed from `protcosmo.fastpath.timing.json`:
   - DuckDB scan-match time
   - MGF materialization time
   - matched input count
   - matched scan count
   - staged spectrum count
4. Derived comparison metrics:
   - post-search scoring/reporting time by subtracting CometPlus or fast-path phase totals from outer wall time
   - retained-scan totals from `scan prefilter: ... -> <N> scans retained`
   - baseline-only and fastpath-only novel peptide counts
   - scan-universe deltas between the mzMLb baseline and parquet fast path
   - PSM-delta-aware conclusion wording: when fast-path final PSM rows are not lower than baseline, the note reports disagreement diagnostics (scan-selection/search-side/ranking counts) without a "lost-row percentage" denominator
5. Exact scan-set attribution from `local_test/benchmark_pxd010154_scan_attribution.py`:
   - recover exact baseline retained `(idn, scan_id)` keys by parsing retained keep-tmp filtered MGFs
   - compare them against `<output-prefix>.fastpath.scan_matches.tsv`
   - write sidecar TSVs under the keep-tmp recovery output directory for:
     - exact baseline retained scans
     - baseline-only scans and fastpath-only scans with left-joined `PXD010154_ms2.parquet` features so scans absent from the parquet universe still remain in the exact list
     - per-file overlap summary
     - PSM-loss and peptide-loss attribution summaries
     - m/z and RT bin summaries
   - classify each baseline final novel PSM row into:
     - direct scan mismatch (`scan` absent from fastpath matched set)
     - matched-scan search-side difference
     - matched-scan ranking/FDR difference

## 2.4 PXD010154 fast-path scan-mismatch note generator

`local_test/analyze_pxd010154_fastpath_scan_mismatch.py` is a second repo-local analysis helper that reuses the existing scan-attribution sidecars and completed fast-path artifacts to explain why baseline-only and fastpath-only scan sets diverge.

It is also outside the production CLI path. Its job is to write `notes/20260422.pxd010154_fastpath_scan_mismatch_analysis.md` without rerunning full ProtCosmo searches.

The helper:

1. Reads existing sidecars:
   - `baseline_only_scans.tsv`
   - `fastpath_only_scans.tsv`
   - the detailed internal novel TSV
   - completed fast-path log and scan-match TSV
2. Tests exact detailed-window membership for each scan under:
   - raw precursor mz
   - `mz - 1.003355 / charge`
   - `mz + 1.003355 / charge`
3. Fetches representative sample precursor rows back from per-file `*.mgf.parquet` files and computes neutral masses from the shifted mz values.
4. Summarizes PTM coverage directly from `peptide_with_mod` in the detailed internal TSV.
5. Writes a note that compares:
   - existing data artifacts
   - current DuckDB matching code in `src/protcosmo/utils/parquet_fastpath.py`
   - CometPlus novel prefilter behavior in `NovelModeUtilsPrefilter.cpp`
6. Inspects the checked-out `src/protcosmo/utils/parquet_fastpath.py` to report whether the current source uses upward (`+ offset / charge`) or downward (`- offset / charge`) isotope-window shifts.
7. Lists the targeted production-side isotope tests already present in `local_test/test_protcosmo_utils.py` so the note distinguishes old artifact behavior from current source state.

This helper is analysis-only. It does not change the production fast-path behavior.

## 3. Important Behavior Notes

1. Unknown CLI options are passed through to CometPlus only in CometPlus path.
2. `--force` only affects novel PIN overwrite behavior.
3. Novel PIN skip/reuse check uses exactly `<output-prefix>.cometplus.novel.pin`.
4. q-value/PEP values are lookup estimates (not retrained Percolator outputs).
5. Output score fields come from matched reference-score lookup values.
6. Caches are reused across runs/groups to avoid duplicate model/reference loads.
7. CLI scoring refs are single-value options; per-mass-file scoring variation uses `--input_tsv`.
8. `--thread` controls both CometPlus `num_threads` forwarding and parallel worker count for multi-group TSV scoring.
9. `--known_peptide` is a first-class global ProtCosmo CLI option that resolves to an absolute path before CometPlus execution.
10. `--output_known_peptide` is not modeled by ProtCosmo; users may still rely on raw passthrough for unmanaged CometPlus options.
11. README fast-path docs include a dedicated "advantages and recommended usage" block plus a benchmark-only `--execute-fastpath-only` command template for rerunning only the parquet case while reusing baseline outputs.
