# ProtCosmo

ProtCosmo is a CLI pipeline for CometPlus-based search plus static Percolator-style rescoring.
It is designed for workflows that need novel peptide/protein discovery while keeping scoring and reporting reproducible.

## What ProtCosmo does

1. Runs CometPlus search (unless `--input-pin` mode is used).
2. Re-scores PIN candidates with linear models from `--init-weights`.
3. Selects one winner PSM per spectrum.
4. Estimates q-value and PEP by lookup from reference PSM/peptide tables.
5. Writes novel-focused report TSV files.
6. Optionally uses a two-pass DuckDB + `mgf.parquet` fast path for novel runs when `--ms2-parquet` is provided and all spectrum inputs are `*.mgf.parquet`.

Supported run modes:

1. `--mass-file` mode: standard CometPlus path.
2. `--input_tsv` mode: row-based configuration for multiple mass files and scoring groups.
3. `--input-pin` mode: skip CometPlus and score an existing PIN directly.

## Installation

```bash
cd $FOLDER
git clone https://github.com/ATPs/ProtCosmo.git
cd ProtCosmo
python -m pip install -e .
```

Check CLI:

```bash
protcosmo --help
protcosmo --help-full
protcosmo --version
```

## Requirements

- Python `>=3.9`
- Runtime dependencies:
  - `numpy>=1.23`
  - `pandas>=1.5`
  - `pyarrow>=10.0`
  - `duckdb>=1.0` for the optional parquet fast path

## Core arguments

Always required:

- `--output-dir`

Mode-specific required arguments:

1. `--mass-file` mode:
   - `--mass-file`
   - `--params`
   - `--database`
   - `--init-weights`
   - `--percolator-psms`
   - `--percolator-peptides`
2. `--input_tsv` mode:
   - `--input_tsv`
   - `--output-dir`
   - scoring fields can come from TSV columns or CLI overrides
3. `--input-pin` mode:
   - `--input-pin`
   - `--init-weights`
   - `--percolator-psms`
   - `--percolator-peptides`

Notes:

- `--input_tsv` and `--mass-file` cannot be used together.
- `--input-pin` takes precedence when both `--input-pin` and `--input_tsv` are provided.
- CLI `--init-weights`, `--percolator-psms`, `--percolator-peptides` each accept only one value.
- `--known_peptide` is a global optional CometPlus cache-reuse input; ProtCosmo does not model `--output_known_peptide`.
- Parquet fast path activates when `--ms2-parquet` is provided and all spectrum inputs are `*.mgf.parquet`.
- `--mgf-parquet-dir` has been removed; provide `*.mgf.parquet` paths directly via `--mass-file` or `--input_tsv` `mass-file` cells.
- `*.mgf.parquet` inputs without `--ms2-parquet` are rejected.
- `--ms2-parquet` with non-`*.mgf.parquet` inputs is rejected.
- In the parquet fast path, ProtCosmo forwards the user's `--thread` value unchanged when one is provided; if `--thread` is omitted, CometPlus uses its normal default thread behavior.
- In the parquet fast path, `--known_peptide` is forwarded only in pass 1; pass 2 resumes with `--internal_novel_peptide`.
- In the parquet fast path, `isotope_error` matching follows CometPlus novel prefilter semantics, including the signed offset sets for modes `0..7`; raw precursor mz is matched against internal windows shifted by the same effective isotope offset in mz-space.

## Input formats

### `--mass-file`

`--mass-file` accepts:

1. One file path.
2. Comma-separated file paths.
3. A text file containing one file path per line.
4. A directory (all supported spectrum files inside are used).

### `--input_tsv`

Required header column:

- `mass-file`

Optional columns:

- `params`
- `database`
- `init-weights`
- `percolator-psms`
- `percolator-peptides`

Header matching is case-insensitive and accepts dash/underscore variants.

Example TSV:

```tsv
mass-file	params	database	init-weights	percolator-psms	percolator-peptides
/data/spec/a.mzMLb	/data/comet.params	/data/db.fasta	/data/model_A.weights	/data/ref_A.psms.tsv	/data/ref_A.peptides.tsv
/data/spec/b.mzMLb	/data/comet.params	/data/db.fasta	/data/model_B.weights	/data/ref_B.psms.tsv	/data/ref_B.peptides.tsv
```

### Reference tables (`--percolator-psms`, `--percolator-peptides`)

- File types: TSV or Parquet.
- Must contain logical columns for score, q-value, and PEP (name variants are accepted).

## Outputs

Normal mode writes:

1. `<output-prefix>.nove.psms.tsv`
2. `<output-prefix>.novel.peptides.tsv`

Optional runtime log:

- `<output-prefix>.log` when `--log` is set.

Important:

- q-value/PEP in output are lookup-based estimates from reference tables.
- When the parquet fast path is used, ProtCosmo also writes:
  - `<output-prefix>.fastpath.scan_matches.tsv`
  - `<output-prefix>.fastpath.mgf_manifest.tsv`
  - `<output-prefix>.fastpath.timing.json`
  - staged subset MGFs under `<output-prefix>.fastpath_subset_mgfs/` unless `--keep-tmp` is not set, in which case they are removed after the run.

## Detailed examples

### Example 1: Standard search with novel peptide/protein inputs

```bash
protcosmo \
  --cometplus /opt/cometplus/bin/cometplus \
  --mass-file /data/spec/run01.mzMLb \
  --params /data/config/comet.params \
  --database /data/db/known_plus_novel.fasta \
  --novel_protein /data/novel/novel_protein.fa \
  --novel_peptide /data/novel/novel_peptide.txt \
  --init-weights /data/models/selected.weights \
  --percolator-psms /data/ref/target.psms.tsv \
  --percolator-peptides /data/ref/target.peptides.tsv \
  --output-dir /data/out/protcosmo \
  --output-prefix run01 \
  --thread 8 \
  --log
```

What this does:

1. Runs CometPlus search for one mass file.
2. Re-scores PIN with `selected.weights`.
3. Selects winner PSM per spectrum.
4. Estimates q/PEP from reference tables.
5. Writes `run01.nove.psms.tsv`, `run01.novel.peptides.tsv`, and `run01.log`.

### Example 2: Multiple mass files from a list file

`mass_files.txt`:

```text
/data/spec/run01.mzMLb
/data/spec/run02.mzMLb
/data/spec/run03.mzMLb
```

Run:

```bash
protcosmo \
  --mass-file /data/spec/mass_files.txt \
  --params /data/config/comet.params \
  --database /data/db/known_plus_novel.fasta \
  --init-weights /data/models/selected.weights \
  --percolator-psms /data/ref/target.psms.tsv \
  --percolator-peptides /data/ref/target.peptides.tsv \
  --output-dir /data/out/protcosmo \
  --output-prefix multi_run
```

### Example 3: Reuse a CometPlus known-peptide cache

```bash
protcosmo \
  --mass-file /data/spec/run01.mzMLb \
  --params /data/config/comet.params \
  --database /data/db/known.idx \
  --known_peptide /data/out/known.txt \
  --novel_peptide /data/novel/novel_peptide.txt \
  --init-weights /data/models/selected.weights \
  --percolator-psms /data/ref/target.psms.tsv \
  --percolator-peptides /data/ref/target.peptides.tsv \
  --output-dir /data/out/protcosmo \
  --output-prefix run01_reuse
```

Use this when CometPlus already exported a compatible known-peptide cache and you want ProtCosmo to pass it back through via `--known_peptide`.

### Example 4: `--input_tsv` mode with multiple init-weight groups

```bash
protcosmo \
  --cometplus /opt/cometplus/bin/cometplus \
  --input_tsv /data/config/protcosmo.input.tsv \
  --output-dir /data/out/protcosmo \
  --output-prefix grouped \
  --thread 6 \
  --log
```

Behavior in this mode:

1. All TSV mass files are merged into one CometPlus run.
2. If TSV has multiple effective `init-weights`, ProtCosmo splits PIN rows by input key and scores each group independently.
3. Group scoring runs in parallel when `--thread > 1`.

### Example 5: Two-pass parquet fast path for a PRIDE-style multi-input novel-peptide run

```bash
protcosmo \
  --cometplus /data/p/comet/Comet/ProtCosmo/CometPlus/cometplus \
  --input_tsv /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2duck/protcosmo.input.fastpath.mgf_parquet.tsv \
  --novel_peptide /data/p/xiaolong/ProtCosmo/ProtCosmo/local_test/data/novel_peptides \
  --known_peptide /data2/pub/proteome/web/protinsight/comet/proteins/20260206/cmt_2026_01_input_with_decoy_HCD__protinsight_proteinseq.target.decoy.fasta.idx.known_peptide.txt \
  --ms2-parquet /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/ms2duck/PXD010154_ms2.parquet \
  --output-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/temp/protcosmo_fastpath \
  --thread 20 \
  --log
```

Behavior in this mode:

1. Pass 1 runs CometPlus on the original inputs, exports the detailed internal novel TSV, forwards `--known_peptide` when present, and stops before search.
2. ProtCosmo uses DuckDB against `--ms2-parquet` plus the provided `*.mgf.parquet` inputs to stage compact subset MGFs, with `isotope_error` handling matched to CometPlus novel prefiltering.
3. Pass 2 resumes CometPlus with `--internal_novel_peptide` on the staged subset MGFs.
4. Search/scoring/reporting after pass 2 are unchanged from the existing workflow.

Fast-path advantages and recommended usage:

1. Why this path is usually faster on large PRIDE-style runs:
   - It avoids full-spectrum prefiltering on all original mzML/mzMLb files.
   - It builds a narrow scan set from `ms2.parquet` + internal novel windows first, then searches only staged subset MGFs.
   - It reuses `--known_peptide` in pass 1 to speed novel-candidate preparation when a known-peptide cache exists.
2. When to use it:
   - Multi-file novel-peptide/novel-protein runs, especially when total input size is large.
   - Cases where `<project>_ms2.parquet` and per-file `*.mgf.parquet` are already available.
3. When not to use it:
   - If `--ms2-parquet` is unavailable.
   - If you are in `--input-pin` mode (CometPlus is skipped, so fast path is irrelevant).
4. Operational notes:
   - Fast path requires `--ms2-parquet` and `mass-file` inputs ending with `*.mgf.parquet`.
   - No separate parquet-directory flag is used; list the exact `*.mgf.parquet` files in `--mass-file` or `--input_tsv`.
   - Keep `--thread` explicit for reproducible benchmark comparisons.
   - In pass 2, ProtCosmo resumes with `--internal_novel_peptide` and does not forward `--known_peptide`.
5. Benchmark-only rerun pattern (reuse baseline outputs, rerun only fast path):

```bash
cd /data/p/xiaolong/ProtCosmo/ProtCosmo
export PATH=/data/p/bin:/data/p/anaconda3/bin:$PATH
export PYTHONPATH=src
/data/p/anaconda3/bin/python local_test/benchmark_pxd010154_parquet_fastpath.py \
  --thread 20 \
  --baseline-output-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/temp/protcosmo_20260421_novel_peptides_input_tsv_known_t20 \
  --baseline-keep-tmp-output-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/temp/protcosmo_20260422_novel_peptides_input_tsv_known_t20_keep_tmp_cometonly \
  --fastpath-output-dir /data2/pub/proteome/PRIDE/protinsight/2019/07/PXD010154/temp/protcosmo_20260422_novel_peptides_parquet_fastpath_isofix_known_t20 \
  --note-path /data/p/xiaolong/ProtCosmo/ProtCosmo/notes/20260421.pxd010154_parquet_fastpath.runtime.md \
  --execute-fastpath-only \
  --force
```

This mode keeps existing baseline outputs unchanged and updates only the fast-path run plus the runtime comparison note.

### Example 6: `--input-pin` mode (skip CometPlus)

```bash
protcosmo \
  --input-pin /data/pin/existing.pin \
  --init-weights /data/models/selected.weights \
  --percolator-psms /data/ref/target.psms.tsv \
  --percolator-peptides /data/ref/target.peptides.tsv \
  --output-dir /data/out/protcosmo \
  --output-prefix score_only
```

Use this when PIN has already been generated and only scoring/reporting is needed.

### Example 7: Stop after CometPlus

```bash
protcosmo \
  --mass-file /data/spec/run01.mzMLb \
  --params /data/config/comet.params \
  --database /data/db/known_plus_novel.fasta \
  --output-dir /data/out/protcosmo \
  --output-prefix run01 \
  --stop-after-cometplus
```

This runs CometPlus and exits before scoring/report generation.

### Example 8: Force overwrite existing novel PIN

```bash
protcosmo \
  --mass-file /data/spec/run01.mzMLb \
  --params /data/config/comet.params \
  --database /data/db/known_plus_novel.fasta \
  --init-weights /data/models/selected.weights \
  --percolator-psms /data/ref/target.psms.tsv \
  --percolator-peptides /data/ref/target.peptides.tsv \
  --output-dir /data/out/protcosmo \
  --output-prefix run01 \
  --force
```

Without `--force`, existing `<output-prefix>.cometplus.novel.pin` is reused when possible.

## Practical tips

1. Start with `--help-full` for complete option semantics and examples.
2. Keep `params` and `database` consistent across rows in `--input_tsv` mode.
3. Use `--log` in production runs for easier debugging and provenance.
4. Use `--input-pin` to iterate quickly on scoring/reference choices without rerunning CometPlus.
