"""Resolve ProtCosmo --mass-file inputs into concrete mass-spectrum files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


SUPPORTED_MASS_FILE_SUFFIXES = (
    ".mzml",
    ".mzmlb",
    ".mzxml",
    ".mgf",
    ".raw",
    ".ms2",
    ".cms2",
    ".bms2",
    ".mzml.gz",
    ".mzxml.gz",
    ".mgf.gz",
)


def _split_comma_items(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def is_supported_mass_file(path: Path) -> bool:
    """Return True when path name looks like a CometPlus-supported mass file."""

    lower = path.name.lower()
    return any(lower.endswith(suffix) for suffix in SUPPORTED_MASS_FILE_SUFFIXES)


def _collect_from_directory(directory: Path) -> List[str]:
    files = [
        str(entry.resolve())
        for entry in sorted(directory.iterdir(), key=lambda p: p.name)
        if entry.is_file() and is_supported_mass_file(entry)
    ]
    if not files:
        raise ValueError(
            f"--mass-file directory has no CometPlus-supported files: {directory}"
        )
    return files


class _MassListParseError(ValueError):
    """Base error for mass-file list parsing."""


class _MassListInvalidEntryError(_MassListParseError):
    """List file had explicit entries, but at least one entry is invalid."""


class _MassListNoUsableEntryError(_MassListParseError):
    """List file had no usable non-comment entries."""


def _resolve_mass_path(raw_path: str, *, base_dir: Path) -> List[str]:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        raise ValueError(f"--mass-file path does not exist: {candidate}")

    if candidate.is_dir():
        return _collect_from_directory(candidate)

    if candidate.is_file():
        if is_supported_mass_file(candidate):
            return [str(candidate)]
        # Fallback strategy for robustness:
        # 1) Try to parse as list file.
        # 2) If it has no usable entries, treat it as a direct mass file path and
        #    let CometPlus decide whether it is a supported input type.
        # 3) If list entries exist but are invalid, keep the explicit parse error.
        try:
            return _read_mass_file_list(candidate)
        except _MassListNoUsableEntryError:
            return [str(candidate)]
        except _MassListInvalidEntryError:
            raise

    raise ValueError(f"--mass-file path is neither file nor directory: {candidate}")


def _read_mass_file_list(list_file: Path) -> List[str]:
    output: List[str] = []
    saw_candidate_line = False
    with list_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            saw_candidate_line = True
            items = _split_comma_items(text)
            if not items:
                continue
            for item in items:
                candidate = Path(item).expanduser()
                if not candidate.is_absolute():
                    candidate = (list_file.parent / candidate).resolve()
                else:
                    candidate = candidate.resolve()
                if not candidate.exists():
                    raise _MassListInvalidEntryError(
                        f"--mass-file list contains non-existing path at "
                        f"{list_file}:{line_no}: {candidate}"
                    )
                if not candidate.is_file() or not is_supported_mass_file(candidate):
                    raise _MassListInvalidEntryError(
                        f"--mass-file list contains unsupported mass file at "
                        f"{list_file}:{line_no}: {candidate}"
                    )
                output.append(str(candidate))
    if not output:
        if saw_candidate_line:
            raise _MassListInvalidEntryError(
                f"--mass-file list file contains lines, but no valid mass-file entries: {list_file}"
            )
        raise _MassListNoUsableEntryError(
            f"--mass-file list file has no usable entries: {list_file}"
        )
    return output


def resolve_mass_files(mass_file_spec: str | None) -> List[str]:
    """Resolve --mass-file value into one or more concrete spectrum file paths."""

    if mass_file_spec is None or not str(mass_file_spec).strip():
        raise ValueError("--mass-file is required.")

    items = _split_comma_items(str(mass_file_spec))
    if not items:
        raise ValueError("--mass-file is required.")

    resolved: List[str] = []
    cwd = Path.cwd()
    for item in items:
        resolved.extend(_resolve_mass_path(item, base_dir=cwd))

    resolved = _dedupe_keep_order(resolved)
    if not resolved:
        raise ValueError("--mass-file resolved to an empty set of input files.")
    return resolved
