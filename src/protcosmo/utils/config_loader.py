"""Configuration parsing and validation for ProtCosmo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .mass_file_resolver import resolve_mass_files


@dataclass
class RunConfig:
    """Normalized per-mass-file run configuration."""

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
class PipelineConfig:
    """Top-level runtime config for the whole invocation."""

    cometplus: str
    output_dir: Path
    output_prefix: str
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


def _map_optional_field(
    flag: str,
    value: Optional[str],
    count: int,
    *,
    required: bool,
) -> List[Optional[str]]:
    values = _split_csv(value)
    if not values:
        if required:
            raise ValueError(f"{flag} is required.")
        return [None] * count
    if len(values) == 1:
        return values * count
    if len(values) == count:
        return values
    raise ValueError(
        f"{flag} has {len(values)} values but --mass-file resolved to {count} files. "
        "Allowed: 1 (broadcast) or N (1:1 mapping)."
    )


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
        init_weights = _map_optional_field("--init-weights", args.init_weights, 1, required=True)
        percolator_psms = _map_optional_field("--percolator-psms", args.percolator_psms, 1, required=True)
        percolator_peptides = _map_optional_field(
            "--percolator-peptides",
            args.percolator_peptides,
            1,
            required=True,
        )
        runs = [
            RunConfig(
                run_index=1,
                row_index=1,
                mass_file=resolved_pin,
                mass_files=[resolved_pin],
                params="",
                database="",
                init_weights=init_weights[0],
                percolator_psms=percolator_psms[0],
                percolator_peptides=percolator_peptides[0],
            )
        ]
        return PipelineConfig(
            cometplus=args.cometplus,
            output_dir=Path(args.output_dir),
            output_prefix=output_prefix,
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
    count = len(run_input_groups)

    score_required = (not stop_after) and (not stop_after_cometplus)
    init_weights = _map_optional_field("--init-weights", args.init_weights, count, required=score_required)
    percolator_psms = _map_optional_field("--percolator-psms", args.percolator_psms, count, required=score_required)
    percolator_peptides = _map_optional_field(
        "--percolator-peptides",
        args.percolator_peptides,
        count,
        required=score_required,
    )

    runs: List[RunConfig] = []
    for idx, mass_group in enumerate(run_input_groups):
        mass_file = mass_group[0] if len(mass_group) == 1 else ",".join(mass_group)
        runs.append(
            RunConfig(
                run_index=idx + 1,
                row_index=1,
                mass_file=mass_file,
                mass_files=list(mass_group),
                params=params,
                database=database,
                init_weights=init_weights[idx],
                percolator_psms=percolator_psms[idx],
                percolator_peptides=percolator_peptides[idx],
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
