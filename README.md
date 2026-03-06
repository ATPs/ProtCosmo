# ProtCosmo

ProtCosmo is a CLI pipeline for CometPlus-based search plus static Percolator-style rescoring.
It is designed for workflows that need novel peptide/protein discovery while keeping scoring and reporting reproducible.

## What ProtCosmo does

1. Runs CometPlus search (unless `--input-pin` mode is used).
2. Re-scores PIN candidates with linear models from `--init-weights`.
3. Selects one winner PSM per spectrum.
4. Estimates q-value and PEP by lookup from reference PSM/peptide tables.
5. Writes novel-focused report TSV files.

Supported run modes:

1. `--mass-file` mode: standard CometPlus path.
2. `--input_tsv` mode: row-based configuration for multiple mass files and scoring groups.
3. `--input-pin` mode: skip CometPlus and score an existing PIN directly.

## Installation

```bash
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

### Example 3: `--input_tsv` mode with multiple init-weight groups

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

### Example 4: `--input-pin` mode (skip CometPlus)

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

### Example 5: Stop after CometPlus

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

### Example 6: Force overwrite existing novel PIN

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
