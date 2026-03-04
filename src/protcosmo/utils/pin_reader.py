"""PIN loading utilities."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterable, List

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
    "ScanNr",
    "Peptide",
    "Proteins",
] + PIN_FEATURE_COLUMNS

_PROTEIN_SPLIT_RE = re.compile(r"[,;\t ]+")


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


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
        proteins_idx = header.index("Proteins") if "Proteins" in header else -1
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
    missing = [col for col in PIN_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"PIN data is missing required columns in {source_path}: {missing}")
    for col in PIN_FEATURE_COLUMNS + ["ScanNr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
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
