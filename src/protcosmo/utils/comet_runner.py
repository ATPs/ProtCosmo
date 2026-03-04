"""CometPlus execution helpers."""

from __future__ import annotations

import glob
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config_loader import PipelineConfig, RunConfig


DEFAULT_INTERNAL_NOVEL_PEPTIDE_FILENAME = "protcosmo.internal_novel_peptide.tsv"


@dataclass
class CometRunResult:
    """Execution artifacts for one CometPlus run."""

    run_dir: Path
    output_base: Path
    command: List[str]
    return_code: int
    stdout_path: Path
    stderr_path: Path
    pin_path: Optional[Path]


def _resolve_arg_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def _resolve_output_internal_novel_peptide_path(value: str) -> str:
    # Preserve basename-only semantics for CometPlus:
    # when no directory is present, CometPlus resolves it under output-folder/cwd.
    candidate = Path(value).expanduser()
    if str(candidate.parent) in ("", "."):
        return value
    return str(candidate.resolve())


def _resolve_output_internal_target(command: List[str], run_dir: Path) -> Optional[Path]:
    if "--output_internal_novel_peptide" not in command:
        return None
    idx = command.index("--output_internal_novel_peptide")
    if idx + 1 >= len(command):
        return None
    raw = command[idx + 1]
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (run_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _find_pin_output(run_dir: Path, output_base: Path) -> Path:
    candidates = [
        Path(f"{output_base}.pin.parquet"),
        Path(f"{output_base}.pin.parquet.gz"),
        Path(f"{output_base}.pin"),
        Path(f"{output_base}.pin.gz"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    glob_candidates = sorted(
        glob.glob(str(run_dir / "*.pin*")),
        key=lambda path: os.path.getmtime(path),
        reverse=True,
    )
    if glob_candidates:
        return Path(glob_candidates[0])
    raise FileNotFoundError(f"No PIN output was found in run directory: {run_dir}")


def build_comet_command(
    run: RunConfig,
    config: PipelineConfig,
    output_base: Path,
) -> List[str]:
    command: List[str] = [
        config.cometplus,
        "--params",
        _resolve_arg_path(run.params),
        "--database",
        _resolve_arg_path(run.database),
        "--output_percolatorfile",
        "1",
        "--max_duplicate_proteins",
        "-1",
        "--name",
        str(output_base),
    ]

    if config.novel_protein:
        command.extend(["--novel_protein", _resolve_arg_path(config.novel_protein)])
    if config.novel_peptide:
        command.extend(["--novel_peptide", _resolve_arg_path(config.novel_peptide)])
    output_internal_novel = config.output_internal_novel_peptide
    if output_internal_novel is None and (config.novel_protein or config.novel_peptide):
        output_internal_novel = str((config.output_dir / DEFAULT_INTERNAL_NOVEL_PEPTIDE_FILENAME).resolve())
    if output_internal_novel:
        command.extend(
            [
                "--output_internal_novel_peptide",
                _resolve_output_internal_novel_peptide_path(output_internal_novel),
            ]
        )
    if config.internal_novel_peptide:
        command.extend(["--internal_novel_peptide", _resolve_arg_path(config.internal_novel_peptide)])
    if config.stop_after_saving_novel_peptide:
        command.append("--stop-after-saving-novel-peptide")
    if config.thread is not None:
        command.extend(["--thread", str(config.thread)])

    if config.use_scan_filters:
        if config.scan:
            command.extend(["--scan", _resolve_arg_path(config.scan)])
        if config.scan_numbers:
            command.extend(["--scan_numbers", config.scan_numbers])
        if config.first_scan is not None:
            command.extend(["--first-scan", str(config.first_scan)])
        if config.last_scan is not None:
            command.extend(["--last-scan", str(config.last_scan)])

    command.extend(config.passthrough_args)
    command.append(_resolve_arg_path(run.mass_file))
    return command


def run_cometplus_search(
    run: RunConfig,
    config: PipelineConfig,
    comet_output_root: Path,
    *,
    require_pin_output: bool = True,
) -> CometRunResult:
    """Run CometPlus for one mass file and return the generated PIN path."""

    run_dir = comet_output_root / f"run_{run.run_index:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_base = run_dir / "comet"
    command = build_comet_command(run, config, output_base)
    output_internal_target = _resolve_output_internal_target(command, run_dir)
    if output_internal_target is not None and output_internal_target.exists():
        output_internal_target.unlink()

    proc = subprocess.run(
        command,
        cwd=run_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path = run_dir / "cometplus.stdout.log"
    stderr_path = run_dir / "cometplus.stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        pretty = " ".join(shlex.quote(token) for token in command)
        raise RuntimeError(
            f"CometPlus failed for mass file {run.mass_file} with exit code {proc.returncode}.\n"
            f"Command: {pretty}\n"
            f"See logs:\n  stdout: {stdout_path}\n  stderr: {stderr_path}"
        )

    pin_path = _find_pin_output(run_dir, output_base) if require_pin_output else None
    return CometRunResult(
        run_dir=run_dir,
        output_base=output_base,
        command=command,
        return_code=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        pin_path=pin_path,
    )
