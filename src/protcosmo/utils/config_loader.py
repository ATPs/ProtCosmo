"""Configuration parsing and validation for ProtCosmo."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .input_key import derive_mass_file_key
from .mass_file_resolver import resolve_mass_files


_FIELD_KEY_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class RunConfig:
    """Normalized per-run configuration."""

    run_index: int
    row_index: int
    mass_file: str
    mass_files: List[str]
    params: str
    database: str
    init_weights: Optional[str]
    percolator_psms: Optional[str]
    percolator_peptides: Optional[str]


@dataclass
class InputTsvRow:
    """Resolved one-row record from --input_tsv."""

    row_index: int
    mass_file: str
    mass_file_key: str
    params: str
    database: str
    init_weights: Optional[str]
    percolator_psms: Optional[str]
    percolator_peptides: Optional[str]


@dataclass
class ScoringGroupConfig:
    """Scoring group keyed by init-weights in --input_tsv mode."""

    group_index: int
    init_weights: str
    percolator_psms: str
    percolator_peptides: str
    mass_files: List[str]
    mass_file_keys: List[str]


@dataclass
class PipelineConfig:
    """Top-level runtime config for the whole invocation."""

    cometplus: str
    output_dir: Path
    output_prefix: str
    input_tsv: Optional[str]
    use_input_tsv: bool
    input_tsv_rows: List[InputTsvRow]
    scoring_groups: List[ScoringGroupConfig]
    novel_protein: Optional[str]
    novel_peptide: Optional[str]
    output_internal_novel_peptide: Optional[str]
    internal_novel_peptide: Optional[str]
    stop_after_saving_novel_peptide: bool
    stop_after_cometplus: bool
    force: bool
    log: bool
    input_pin: Optional[str]
    keep_tmp: bool
    run_comet_each: bool
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


def _require_single_value(flag: str, value: Optional[str]) -> str:
    values = _split_csv(value)
    if not values:
        raise ValueError(f"{flag} is required.")
    if len(values) != 1:
        raise ValueError(f"{flag} accepts only one value.")
    return values[0]


def _single_optional_value(flag: str, value: Optional[str]) -> Optional[str]:
    values = _split_csv(value)
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"{flag} accepts only one value.")
    return values[0]


def _resolve_path(raw: str, *, base_dir: Path) -> str:
    candidate = Path(str(raw).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return str(candidate)


def _normalize_header_key(name: str) -> str:
    return _FIELD_KEY_RE.sub("-", str(name).strip().lower()).strip("-")


def _canonical_header(name: str) -> Optional[str]:
    key = _normalize_header_key(name)
    if key in {"mass-file", "massfile"}:
        return "mass-file"
    if key in {"params", "param"}:
        return "params"
    if key in {"database", "db"}:
        return "database"
    if key in {"init-weights", "initweights"}:
        return "init-weights"
    if key in {"percolator-psms", "percolatorpsms"}:
        return "percolator-psms"
    if key in {"percolator-peptides", "percolatorpeptides"}:
        return "percolator-peptides"
    return None


def _read_input_tsv_rows(path: Path) -> List[Dict[str, str]]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            lines.append(line)

    if not lines:
        raise ValueError(f"--input_tsv has no usable rows: {path}")

    reader = csv.DictReader(io.StringIO("".join(lines)), delimiter="\t")
    if not reader.fieldnames:
        raise ValueError(f"--input_tsv is missing a header row: {path}")

    field_map: Dict[str, str] = {}
    for source in reader.fieldnames:
        canonical = _canonical_header(source)
        if canonical is None:
            continue
        if canonical not in field_map:
            field_map[canonical] = source

    if "mass-file" not in field_map:
        raise ValueError("--input_tsv requires a 'mass-file' column.")

    rows: List[Dict[str, str]] = []
    for row_idx, row in enumerate(reader, start=1):
        normalized: Dict[str, str] = {}
        for canonical, source in field_map.items():
            value = row.get(source)
            normalized[canonical] = "" if value is None else str(value).strip()
        if not any(normalized.values()):
            continue
        normalized["row_index"] = str(row_idx)
        rows.append(normalized)

    if not rows:
        raise ValueError(f"--input_tsv has no usable data rows: {path}")
    return rows


def _resolve_tsv_mass_file(raw_mass_file: str, *, row_index: int, tsv_path: Path) -> str:
    text = str(raw_mass_file).strip()
    if not text:
        raise ValueError(f"Row {row_index}: missing required 'mass-file'.")
    if "," in text:
        raise ValueError(
            f"Row {row_index}: 'mass-file' must be a single file path in --input_tsv mode; "
            "comma-separated values are not allowed."
        )

    resolved = Path(_resolve_path(text, base_dir=tsv_path.parent))
    if not resolved.exists():
        raise ValueError(f"Row {row_index}: mass-file path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Row {row_index}: mass-file must be a file path: {resolved}")
    return str(resolved)


def _build_input_tsv_rows(
    args,
    *,
    input_tsv_path: Path,
    score_required: bool,
) -> tuple[List[InputTsvRow], List[ScoringGroupConfig], str, str]:
    raw_rows = _read_input_tsv_rows(input_tsv_path)

    cli_params = _single_optional_value("--params", args.params)
    cli_database = _single_optional_value("--database", args.database)

    cli_init = _single_optional_value("--init-weights", args.init_weights)
    cli_psms = _single_optional_value("--percolator-psms", args.percolator_psms)
    cli_peptides = _single_optional_value("--percolator-peptides", args.percolator_peptides)

    tsv_rows: List[InputTsvRow] = []
    key_to_row: Dict[str, int] = {}
    key_to_mass_file: Dict[str, str] = {}

    for raw in raw_rows:
        row_index = int(raw["row_index"])
        mass_file = _resolve_tsv_mass_file(raw.get("mass-file", ""), row_index=row_index, tsv_path=input_tsv_path)
        mass_key = derive_mass_file_key(mass_file)
        if mass_key in key_to_row:
            previous_row = key_to_row[mass_key]
            previous_path = key_to_mass_file[mass_key]
            raise ValueError(
                f"Row {row_index}: mass-file key collision for '{mass_key}'. "
                f"Previously used by row {previous_row}: {previous_path}"
            )
        key_to_row[mass_key] = row_index
        key_to_mass_file[mass_key] = mass_file

        row_params = raw.get("params")
        if row_params:
            params = _resolve_path(row_params, base_dir=input_tsv_path.parent)
        elif cli_params is not None:
            params = _resolve_path(cli_params, base_dir=Path.cwd())
        else:
            raise ValueError(f"Row {row_index}: missing required 'params' (or provide --params).")
        row_database = raw.get("database")
        if row_database:
            database = _resolve_path(row_database, base_dir=input_tsv_path.parent)
        elif cli_database is not None:
            database = _resolve_path(cli_database, base_dir=Path.cwd())
        else:
            raise ValueError(f"Row {row_index}: missing required 'database' (or provide --database).")

        if cli_init is not None:
            init_weights = _resolve_path(cli_init, base_dir=Path.cwd())
        elif raw.get("init-weights"):
            init_weights = _resolve_path(raw["init-weights"], base_dir=input_tsv_path.parent)
        else:
            init_weights = None

        if cli_psms is not None:
            percolator_psms = _resolve_path(cli_psms, base_dir=Path.cwd())
        elif raw.get("percolator-psms"):
            percolator_psms = _resolve_path(raw["percolator-psms"], base_dir=input_tsv_path.parent)
        else:
            percolator_psms = None

        if cli_peptides is not None:
            percolator_peptides = _resolve_path(cli_peptides, base_dir=Path.cwd())
        elif raw.get("percolator-peptides"):
            percolator_peptides = _resolve_path(raw["percolator-peptides"], base_dir=input_tsv_path.parent)
        else:
            percolator_peptides = None

        if score_required:
            missing_fields: List[str] = []
            if init_weights is None:
                missing_fields.append("init-weights")
            if percolator_psms is None:
                missing_fields.append("percolator-psms")
            if percolator_peptides is None:
                missing_fields.append("percolator-peptides")
            if missing_fields:
                raise ValueError(
                    f"Row {row_index}: missing scoring fields in --input_tsv mode: {', '.join(missing_fields)}"
                )

        tsv_rows.append(
            InputTsvRow(
                row_index=row_index,
                mass_file=mass_file,
                mass_file_key=mass_key,
                params=params,
                database=database,
                init_weights=init_weights,
                percolator_psms=percolator_psms,
                percolator_peptides=percolator_peptides,
            )
        )

    if not tsv_rows:
        raise ValueError(f"--input_tsv has no usable rows: {input_tsv_path}")

    unique_params = {row.params for row in tsv_rows}
    if len(unique_params) != 1:
        raise ValueError("--input_tsv requires one unique effective 'params' value across all rows.")
    unique_database = {row.database for row in tsv_rows}
    if len(unique_database) != 1:
        raise ValueError("--input_tsv requires one unique effective 'database' value across all rows.")

    scoring_groups: List[ScoringGroupConfig] = []
    if score_required:
        grouped: Dict[str, Dict[str, object]] = {}
        ordered_keys: List[str] = []
        for row in tsv_rows:
            assert row.init_weights is not None
            assert row.percolator_psms is not None
            assert row.percolator_peptides is not None

            if row.init_weights not in grouped:
                grouped[row.init_weights] = {
                    "percolator_psms": set(),
                    "percolator_peptides": set(),
                    "mass_files": [],
                    "mass_file_keys": [],
                }
                ordered_keys.append(row.init_weights)
            bucket = grouped[row.init_weights]
            bucket["percolator_psms"].add(row.percolator_psms)
            bucket["percolator_peptides"].add(row.percolator_peptides)
            bucket["mass_files"].append(row.mass_file)
            bucket["mass_file_keys"].append(row.mass_file_key)

        for group_index, init_weights in enumerate(ordered_keys, start=1):
            bucket = grouped[init_weights]
            psms_values = sorted(bucket["percolator_psms"])
            peptides_values = sorted(bucket["percolator_peptides"])
            if len(psms_values) != 1 or len(peptides_values) != 1:
                raise ValueError(
                    "Each unique init-weights value must map to exactly one percolator-psms "
                    "and one percolator-peptides value in --input_tsv mode."
                )
            scoring_groups.append(
                ScoringGroupConfig(
                    group_index=group_index,
                    init_weights=init_weights,
                    percolator_psms=psms_values[0],
                    percolator_peptides=peptides_values[0],
                    mass_files=list(bucket["mass_files"]),
                    mass_file_keys=list(bucket["mass_file_keys"]),
                )
            )

    params = next(iter(unique_params))
    database = next(iter(unique_database))
    return tsv_rows, scoring_groups, params, database


def load_pipeline_config(args, passthrough_args: List[str]) -> PipelineConfig:
    """Create the execution config from CLI."""

    stop_after = bool(getattr(args, "stop_after_saving_novel_peptide", False))
    stop_after_cometplus = bool(getattr(args, "stop_after_cometplus", False))
    input_pin_value = getattr(args, "input_pin", None)
    input_pin = str(input_pin_value).strip() if input_pin_value is not None else ""
    if not input_pin:
        input_pin = None
    output_prefix = str(getattr(args, "output_prefix", "protcosmo")).strip()
    if not output_prefix:
        raise ValueError("--output-prefix cannot be empty.")

    if stop_after and stop_after_cometplus:
        raise ValueError("--stop-after-saving-novel-peptide and --stop-after-cometplus cannot be used together.")
    if input_pin and stop_after:
        raise ValueError("--stop-after-saving-novel-peptide cannot be used with --input-pin.")
    if input_pin and stop_after_cometplus:
        raise ValueError("--stop-after-cometplus cannot be used with --input-pin.")

    if input_pin:
        resolved_pin = str(Path(_require_single_value("--input-pin", input_pin)).expanduser().resolve())
        init_weights = _single_optional_value("--init-weights", args.init_weights)
        percolator_psms = _single_optional_value("--percolator-psms", args.percolator_psms)
        percolator_peptides = _single_optional_value("--percolator-peptides", args.percolator_peptides)
        if init_weights is None:
            raise ValueError("--init-weights is required.")
        if percolator_psms is None:
            raise ValueError("--percolator-psms is required.")
        if percolator_peptides is None:
            raise ValueError("--percolator-peptides is required.")
        runs = [
            RunConfig(
                run_index=1,
                row_index=1,
                mass_file=resolved_pin,
                mass_files=[resolved_pin],
                params="",
                database="",
                init_weights=init_weights,
                percolator_psms=percolator_psms,
                percolator_peptides=percolator_peptides,
            )
        ]
        return PipelineConfig(
            cometplus=args.cometplus,
            output_dir=Path(args.output_dir),
            output_prefix=output_prefix,
            input_tsv=None,
            use_input_tsv=False,
            input_tsv_rows=[],
            scoring_groups=[],
            novel_protein=args.novel_protein,
            novel_peptide=args.novel_peptide,
            output_internal_novel_peptide=getattr(args, "output_internal_novel_peptide", None),
            internal_novel_peptide=getattr(args, "internal_novel_peptide", None),
            stop_after_saving_novel_peptide=False,
            stop_after_cometplus=False,
            force=bool(getattr(args, "force", False)),
            log=bool(getattr(args, "log", False)),
            input_pin=resolved_pin,
            keep_tmp=bool(getattr(args, "keep_tmp", False)),
            run_comet_each=bool(getattr(args, "run_comet_each", True)),
            thread=args.thread,
            scan=args.scan,
            scan_numbers=args.scan_numbers,
            first_scan=args.first_scan,
            last_scan=args.last_scan,
            passthrough_args=list(passthrough_args),
            use_scan_filters=False,
            warnings=[],
            runs=runs,
        )

    score_required = (not stop_after) and (not stop_after_cometplus)

    input_tsv_arg = getattr(args, "input_tsv", None)
    input_tsv_text = str(input_tsv_arg).strip() if input_tsv_arg is not None else ""
    if input_tsv_text:
        if getattr(args, "mass_file", None) is not None and str(getattr(args, "mass_file", "")).strip():
            raise ValueError("--mass-file cannot be used together with --input_tsv.")

        input_tsv_path = Path(input_tsv_text).expanduser()
        if not input_tsv_path.is_absolute():
            input_tsv_path = (Path.cwd() / input_tsv_path).resolve()
        else:
            input_tsv_path = input_tsv_path.resolve()
        if not input_tsv_path.exists():
            raise ValueError(f"--input_tsv path does not exist: {input_tsv_path}")
        if not input_tsv_path.is_file():
            raise ValueError(f"--input_tsv must be a file: {input_tsv_path}")

        tsv_rows, scoring_groups, params, database = _build_input_tsv_rows(
            args,
            input_tsv_path=input_tsv_path,
            score_required=score_required,
        )

        mass_files = [row.mass_file for row in tsv_rows]

        first_group = scoring_groups[0] if scoring_groups else None
        run = RunConfig(
            run_index=1,
            row_index=1,
            mass_file=",".join(mass_files),
            mass_files=mass_files,
            params=params,
            database=database,
            init_weights=None if first_group is None else first_group.init_weights,
            percolator_psms=None if first_group is None else first_group.percolator_psms,
            percolator_peptides=None if first_group is None else first_group.percolator_peptides,
        )

        return PipelineConfig(
            cometplus=args.cometplus,
            output_dir=Path(args.output_dir),
            output_prefix=output_prefix,
            input_tsv=str(input_tsv_path),
            use_input_tsv=True,
            input_tsv_rows=tsv_rows,
            scoring_groups=scoring_groups,
            novel_protein=args.novel_protein,
            novel_peptide=args.novel_peptide,
            output_internal_novel_peptide=getattr(args, "output_internal_novel_peptide", None),
            internal_novel_peptide=getattr(args, "internal_novel_peptide", None),
            stop_after_saving_novel_peptide=stop_after,
            stop_after_cometplus=stop_after_cometplus,
            force=bool(getattr(args, "force", False)),
            log=bool(getattr(args, "log", False)),
            input_pin=None,
            keep_tmp=bool(getattr(args, "keep_tmp", False)),
            run_comet_each=bool(getattr(args, "run_comet_each", True)),
            thread=args.thread,
            scan=args.scan,
            scan_numbers=args.scan_numbers,
            first_scan=args.first_scan,
            last_scan=args.last_scan,
            passthrough_args=list(passthrough_args),
            use_scan_filters=True,
            warnings=[],
            runs=[run],
        )

    mass_file = getattr(args, "mass_file", None)
    if mass_file is None or not str(mass_file).strip():
        raise ValueError("--mass-file is required unless --input-pin is set.")
    mass_files = resolve_mass_files(mass_file)
    novel_mode = any(
        x is not None and str(x).strip()
        for x in (
            args.novel_protein,
            args.novel_peptide,
            getattr(args, "internal_novel_peptide", None),
        )
    )
    merge_multi_input_novel = novel_mode and len(mass_files) > 1
    run_input_groups: List[List[str]]
    if merge_multi_input_novel:
        run_input_groups = [mass_files]
    else:
        run_input_groups = [[mass_file] for mass_file in mass_files]

    params = _require_single_value("--params", args.params)
    database = _require_single_value("--database", args.database)

    init_weights = _single_optional_value("--init-weights", args.init_weights)
    percolator_psms = _single_optional_value("--percolator-psms", args.percolator_psms)
    percolator_peptides = _single_optional_value("--percolator-peptides", args.percolator_peptides)
    if score_required and init_weights is None:
        raise ValueError("--init-weights is required.")
    if score_required and percolator_psms is None:
        raise ValueError("--percolator-psms is required.")
    if score_required and percolator_peptides is None:
        raise ValueError("--percolator-peptides is required.")

    runs: List[RunConfig] = []
    for idx, mass_group in enumerate(run_input_groups):
        run_mass_file = mass_group[0] if len(mass_group) == 1 else ",".join(mass_group)
        runs.append(
            RunConfig(
                run_index=idx + 1,
                row_index=1,
                mass_file=run_mass_file,
                mass_files=list(mass_group),
                params=params,
                database=database,
                init_weights=init_weights,
                percolator_psms=percolator_psms,
                percolator_peptides=percolator_peptides,
            )
        )

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
            "because --mass-file resolved to more than one input file."
        )

    return PipelineConfig(
        cometplus=args.cometplus,
        output_dir=Path(args.output_dir),
        output_prefix=output_prefix,
        input_tsv=None,
        use_input_tsv=False,
        input_tsv_rows=[],
        scoring_groups=[],
        novel_protein=args.novel_protein,
        novel_peptide=args.novel_peptide,
        output_internal_novel_peptide=getattr(args, "output_internal_novel_peptide", None),
        internal_novel_peptide=getattr(args, "internal_novel_peptide", None),
        stop_after_saving_novel_peptide=stop_after,
        stop_after_cometplus=stop_after_cometplus,
        force=bool(getattr(args, "force", False)),
        log=bool(getattr(args, "log", False)),
        input_pin=None,
        keep_tmp=bool(getattr(args, "keep_tmp", False)),
        run_comet_each=bool(getattr(args, "run_comet_each", True)),
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
