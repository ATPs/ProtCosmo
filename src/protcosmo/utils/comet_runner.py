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


def _snapshot_pin_mtime(run_dir: Path) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for path in glob.glob(str(run_dir / "*.pin*")):
        try:
            snapshot[str(Path(path).resolve())] = os.path.getmtime(path)
        except OSError:
            continue
    return snapshot


def _find_pin_output(run_dir: Path, before_snapshot: dict[str, float]) -> Path:
    candidates = sorted(
        (Path(path).resolve() for path in glob.glob(str(run_dir / "*.pin*"))),
        key=lambda path: os.path.getmtime(path),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No PIN output was found in output directory: {run_dir}")

    changed: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        try:
            mtime = os.path.getmtime(candidate)
        except OSError:
            continue
        previous = before_snapshot.get(key)
        if previous is None or mtime != previous:
            changed.append(candidate)
    if changed:
        return changed[0]
    # Fallback to newest pin if mtime resolution prevented change detection.
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No PIN output was found in run directory: {run_dir}")


def build_comet_command(
    run: RunConfig,
    config: PipelineConfig,
) -> List[str]:
    command: List[str] = [
        config.cometplus,
        "--params",
        _resolve_arg_path(run.params),
        "--database",
        _resolve_arg_path(run.database),
        "--output-folder",
        str(Path(config.output_dir).expanduser().resolve()),
        "--output_percolatorfile",
        "1",
        "--max_duplicate_proteins",
        "-1",
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
    if config.keep_tmp:
        command.append("--keep-tmp")
    if config.run_comet_each:
        command.append("--run-comet-each")
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
    command.extend(_resolve_arg_path(mass_file) for mass_file in run.mass_files)
    return command


def run_cometplus_search(
    run: RunConfig,
    config: PipelineConfig,
    output_dir: Path,
    *,
    require_pin_output: bool = True,
) -> CometRunResult:
    """Run CometPlus for one mass file and return the generated PIN path."""

    run_dir = output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_comet_command(run, config)
    before_pin_snapshot = _snapshot_pin_mtime(run_dir)
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
    stdout_path = run_dir / f"cometplus.run_{run.run_index:04d}.stdout.log"
    stderr_path = run_dir / f"cometplus.run_{run.run_index:04d}.stderr.log"
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        pretty = " ".join(shlex.quote(token) for token in command)
        raise RuntimeError(
            f"CometPlus failed for mass file {run.mass_file} with exit code {proc.returncode}.\n"
            f"Command: {pretty}\n"
            f"See logs:\n  stdout: {stdout_path}\n  stderr: {stderr_path}"
        )

    pin_path = _find_pin_output(run_dir, before_pin_snapshot) if require_pin_output else None
    return CometRunResult(
        run_dir=run_dir,
        command=command,
        return_code=proc.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        pin_path=pin_path,
    )
