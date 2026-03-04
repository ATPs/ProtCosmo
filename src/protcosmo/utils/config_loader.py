"""Configuration parsing and validation for ProtCosmo."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CONFIG_FIELDS = (
    "mass-file",
    "params",
    "database",
    "init-weights",
    "percolator-psms",
    "percolator-peptides",
)


@dataclass
class RunConfig:
    """Normalized per-mass-file run configuration."""

    run_index: int
    row_index: int
    mass_file: str
    params: str
    database: str
    init_weights: str
    percolator_psms: str
    percolator_peptides: str


@dataclass
class PipelineConfig:
    """Top-level runtime config for the whole invocation."""

    cometplus: str
    output_dir: Path
    input_tsv: Optional[Path]
    novel_protein: Optional[str]
    novel_peptide: Optional[str]
    thread: Optional[int]
    scan: Optional[str]
    scan_numbers: Optional[str]
    first_scan: Optional[int]
    last_scan: Optional[int]
    passthrough_args: List[str]
    use_scan_filters: bool
    warnings: List[str]
    runs: List[RunConfig]


def _split_csv(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _read_input_tsv(input_tsv: Path) -> List[Dict[str, str]]:
    lines: List[str] = []
    with input_tsv.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            lines.append(line)

    if not lines:
        return []
    reader = csv.DictReader(io.StringIO("".join(lines)), delimiter="\t")
    rows: List[Dict[str, str]] = []
    for row in reader:
        normalized = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items()}
        rows.append(normalized)
    return rows


def _resolve_row_value(row: Dict[str, str], field: str) -> Optional[str]:
    aliases = (field, field.replace("-", "_"))
    for key in aliases:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _build_rows(args) -> List[Dict[str, str]]:
    if args.input_tsv:
        input_tsv = Path(args.input_tsv)
        rows = _read_input_tsv(input_tsv)
        if not rows:
            raise ValueError(f"--input_tsv has no usable rows: {input_tsv}")
        return rows
    return [{}]


def _apply_cli_overrides(rows: List[Dict[str, str]], args) -> List[Dict[str, str]]:
    overrides = {
        "mass-file": args.mass_file,
        "params": args.params,
        "database": args.database,
        "init-weights": args.init_weights,
        "percolator-psms": args.percolator_psms,
        "percolator-peptides": args.percolator_peptides,
    }
    updated: List[Dict[str, str]] = []
    for row in rows:
        merged = dict(row)
        for key, value in overrides.items():
            if value is not None:
                merged[key] = str(value)
        updated.append(merged)
    return updated


def _map_field(field_name: str, value: Optional[str], count: int, row_index: int) -> List[str]:
    values = _split_csv(value)
    if not values:
        raise ValueError(
            f"Row {row_index}: missing required value for '{field_name}'."
        )
    if len(values) == 1:
        return values * count
    if len(values) == count:
        return values
    raise ValueError(
        f"Row {row_index}: '{field_name}' has {len(values)} values but mass-file has {count}. "
        "Allowed: 1 (broadcast) or N (1:1 mapping)."
    )


def _expand_runs(rows: Iterable[Dict[str, str]]) -> List[RunConfig]:
    expanded: List[RunConfig] = []
    run_index = 0
    for row_index, row in enumerate(rows, start=1):
        mass_text = _resolve_row_value(row, "mass-file")
        mass_files = _split_csv(mass_text)
        if not mass_files:
            raise ValueError(f"Row {row_index}: missing required 'mass-file'.")

        count = len(mass_files)
        params = _map_field("params", _resolve_row_value(row, "params"), count, row_index)
        database = _map_field("database", _resolve_row_value(row, "database"), count, row_index)
        init_weights = _map_field("init-weights", _resolve_row_value(row, "init-weights"), count, row_index)
        psm_refs = _map_field(
            "percolator-psms",
            _resolve_row_value(row, "percolator-psms"),
            count,
            row_index,
        )
        peptide_refs = _map_field(
            "percolator-peptides",
            _resolve_row_value(row, "percolator-peptides"),
            count,
            row_index,
        )

        for idx in range(count):
            run_index += 1
            expanded.append(
                RunConfig(
                    run_index=run_index,
                    row_index=row_index,
                    mass_file=mass_files[idx],
                    params=params[idx],
                    database=database[idx],
                    init_weights=init_weights[idx],
                    percolator_psms=psm_refs[idx],
                    percolator_peptides=peptide_refs[idx],
                )
            )
    return expanded


def load_pipeline_config(args, passthrough_args: List[str]) -> PipelineConfig:
    """Create the execution config from CLI and optional input TSV."""

    rows = _build_rows(args)
    rows = _apply_cli_overrides(rows, args)
    runs = _expand_runs(rows)
    warnings: List[str] = []

    use_scan_filters = True
    scan_args_present = any(
        x is not None and str(x).strip()
        for x in (args.scan, args.scan_numbers, args.first_scan, args.last_scan)
    )
    if len(runs) > 1 and scan_args_present:
        use_scan_filters = False
        warnings.append(
            "Scan filters (--scan/--scan_numbers/--first-scan/--last-scan) were ignored "
            "because more than one mass-file was provided."
        )

    return PipelineConfig(
        cometplus=args.cometplus,
        output_dir=Path(args.output_dir),
        input_tsv=None if args.input_tsv is None else Path(args.input_tsv),
        novel_protein=args.novel_protein,
        novel_peptide=args.novel_peptide,
        thread=args.thread,
        scan=args.scan,
        scan_numbers=args.scan_numbers,
        first_scan=args.first_scan,
        last_scan=args.last_scan,
        passthrough_args=list(passthrough_args),
        use_scan_filters=use_scan_filters,
        warnings=warnings,
        runs=runs,
    )
