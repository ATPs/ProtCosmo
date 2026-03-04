"""PIN loading utilities."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


PIN_FEATURE_COLUMNS: List[str] = [
    "lnrSp",
    "deltLCn",
    "deltCn",
    "lnExpect",
    "Xcorr",
    "Sp",
    "IonFrac",
    "Mass",
    "PepLen",
    "Charge1",
    "Charge2",
    "Charge3",
    "Charge4",
    "Charge5",
    "Charge6",
    "enzN",
    "enzC",
    "enzInt",
    "lnNumSP",
    "dM",
    "absdM",
]

PIN_REQUIRED_COLUMNS: List[str] = [
    "SpecId",
    "Peptide",
]

PIN_CANONICAL_COLUMNS: List[str] = [
    "SpecId",
    "ScanNr",
    "Peptide",
    "Proteins",
    "Label",
] + PIN_FEATURE_COLUMNS

PIN_COLUMN_ALIASES = {
    "SpecId": (
        "spec_id",
        "specid",
        "spectrumid",
        "spectrum_id",
        "psmid",
        "psm_id",
    ),
    "ScanNr": (
        "scan",
        "scan_nr",
        "scan_number",
        "scannumber",
        "scanid",
    ),
    "Peptide": (
        "peptide_sequence",
        "peptidesequence",
        "modified_peptide",
        "modifiedpeptide",
        "sequence",
    ),
    "Proteins": (
        "protein",
        "protein_id",
        "protein_ids",
        "proteinids",
        "accessions",
        "proteinaccessions",
    ),
    "Label": ("target_decoy_label",),
    "deltLCn": ("deltalcn", "delta_lcn"),
    "deltCn": ("deltacn", "delta_cn"),
    "IonFrac": ("ion_fraction", "ionfraction"),
    "PepLen": ("pep_len", "peptide_length", "peptidelength"),
    "lnNumSP": ("ln_num_sp", "lnnumsp"),
    "dM": ("delta_mass", "deltamass", "dmass", "mass_error"),
    "absdM": ("abs_delta_mass", "absdeltamass", "abs_dmass", "abs_mass_error"),
}

_PROTEIN_SPLIT_RE = re.compile(r"[,;\t ]+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _normalize_column_name(name: str) -> str:
    return _NON_ALNUM_RE.sub("", str(name).strip().lower())


def _candidate_column_keys(canonical_name: str) -> set[str]:
    aliases = set(PIN_COLUMN_ALIASES.get(canonical_name, ()))
    aliases.add(canonical_name)
    return {_normalize_column_name(value) for value in aliases if str(value).strip()}


def _find_matching_column(columns: Sequence[str], canonical_name: str) -> str | None:
    if canonical_name in columns:
        return canonical_name
    candidate_keys = _candidate_column_keys(canonical_name)
    for col in columns:
        if _normalize_column_name(col) in candidate_keys:
            return col
    return None


def _find_matching_index(columns: Sequence[str], canonical_name: str) -> int:
    matched = _find_matching_column(columns, canonical_name)
    if matched is None:
        return -1
    for idx, col in enumerate(columns):
        if col == matched:
            return idx
    return -1


def _rename_to_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    canonical_targets = PIN_CANONICAL_COLUMNS
    rename_map = {}
    used_sources = set()
    columns = list(df.columns)

    for canonical_name in canonical_targets:
        if canonical_name in df.columns:
            continue
        source_name = _find_matching_column(columns, canonical_name)
        if source_name is None or source_name in used_sources or source_name == canonical_name:
            continue
        rename_map[source_name] = canonical_name
        used_sources.add(source_name)

    if rename_map:
        df = df.rename(columns=rename_map)

    if "Proteins" not in df.columns:
        df["Proteins"] = ""
    return df


def _read_pin_text(path: Path) -> pd.DataFrame:
    with _open_text(path) as handle:
        header_line = ""
        for line in handle:
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            header_line = stripped
            break
        if not header_line:
            raise ValueError(f"PIN file is empty: {path}")

        header = header_line.split("\t")
        proteins_idx = _find_matching_index(header, "Proteins")
        records: List[dict] = []
        for line in handle:
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) < len(header):
                parts.extend([""] * (len(header) - len(parts)))
            elif len(parts) > len(header):
                if proteins_idx >= 0:
                    trailing = parts[len(header) :]
                    parts = parts[: len(header)]
                    if trailing:
                        protein_text = parts[proteins_idx]
                        extra_text = ",".join(token for token in trailing if token)
                        if protein_text and extra_text:
                            protein_text = f"{protein_text},{extra_text}"
                        elif extra_text:
                            protein_text = extra_text
                        parts[proteins_idx] = protein_text
                else:
                    parts = parts[: len(header)]

            row = {header[idx]: parts[idx] for idx in range(len(header))}
            if proteins_idx < 0:
                row["Proteins"] = ""
            records.append(row)
    return pd.DataFrame.from_records(records)


def _ensure_columns(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    df = _rename_to_canonical_columns(df)
    missing = [col for col in PIN_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"PIN data is missing required columns in {source_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    if "Proteins" not in df.columns:
        df["Proteins"] = ""
    if "ScanNr" not in df.columns:
        df["ScanNr"] = pd.Series(range(len(df)), index=df.index, dtype="float64")
    for col in PIN_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["ScanNr"] = pd.to_numeric(df["ScanNr"], errors="coerce").fillna(0.0)
    if "Label" in df.columns:
        df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    df["SpecId"] = df["SpecId"].astype(str)
    df["Peptide"] = df["Peptide"].astype(str)
    df["Proteins"] = df["Proteins"].astype(str)
    return df


def read_pin(path: str | Path) -> pd.DataFrame:
    """Read PIN data from text/gzip/parquet path."""

    pin_path = Path(path)
    lower = pin_path.name.lower()
    if lower.endswith(".parquet") or lower.endswith(".parquet.gz"):
        data = pd.read_parquet(pin_path)
    else:
        data = _read_pin_text(pin_path)
    return _ensure_columns(data, pin_path)


def extract_spectrum_id(spec_id: str) -> str:
    parts = str(spec_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return str(spec_id)


def extract_rank_index(spec_id: str) -> int:
    parts = str(spec_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 10**9


def split_proteins(value: str) -> List[str]:
    text = str(value).strip()
    if not text:
        return []
    return [token for token in _PROTEIN_SPLIT_RE.split(text) if token]


def join_proteins(tokens: Iterable[str]) -> str:
    return ";".join(sorted({token.strip() for token in tokens if token and token.strip()}))
