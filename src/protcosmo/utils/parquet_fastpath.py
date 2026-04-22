"""Helpers for the optional DuckDB + mgf.parquet novel-search fast path."""

from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import IO, Iterable, List, Optional, Sequence

import pandas as pd

from .config_loader import PipelineConfig, RunConfig
from .input_key import derive_mass_file_key


DEFAULT_INTERNAL_NOVEL_PEPTIDE_FILENAME = "protcosmo.internal_novel_peptide.tsv"


@dataclass(frozen=True)
class ChargePolicy:
    """CometPlus charge-handling policy read from a .params file.

    override_charge meanings (identical to CometPlus documentation):
        0 = keep known precursor charge; unknown (NULL/0) → try 2+ and 3+
        1 = ignore known charges; always try the full precursor_charge range
        2 = only accept spectra whose stored charge is within precursor_charge range;
            unknown-charge spectra are skipped
        3 = keep known charge; unknown → try full precursor_charge range
            (intensity-ratio check cannot be performed here without peak data;
            unknown-charge scans are conservatively expanded to all window charges
            in that range — CometPlus pass-2 applies the precise ratio check)

    start_charge / end_charge come from the "precursor_charge" param.
    When precursor_charge = 0 0, start_charge == 0 means the range is disabled.

    isotope_error mirrors Comet's isotope_error param:
        0 = 0
        1 = 0, +1
        2 = 0, +1, +2
        3 = 0, +1, +2, +3
        4 = -1, 0, +1, +2, +3
        5 = -1, 0, +1
        6 = -3, -2, -1, 0, +1, +2, +3
        7 = -8, -4, 0, +4, +8 (stable-isotope labeling)
    """

    override_charge: int
    start_charge: int
    end_charge: int
    isotope_error: int = 0


def parse_comet_params(params_path: str) -> "dict[str, str]":
    """Parse key=value pairs from a Comet .params file. Returns empty dict on error."""
    result: "dict[str, str]" = {}
    if not params_path:
        return result
    try:
        with open(params_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, raw_value = line.partition("=")
                    result[key.strip()] = raw_value.split("#")[0].strip()
    except OSError:
        pass
    return result


def build_charge_policy(comet_params: "dict[str, str]") -> ChargePolicy:
    """Extract charge policy from parsed Comet params dict."""
    try:
        override_charge = int(comet_params.get("override_charge", "0"))
    except ValueError:
        override_charge = 0

    start_charge, end_charge = 1, 6
    raw = comet_params.get("precursor_charge", "").split()
    try:
        if len(raw) >= 2:
            start_charge, end_charge = int(raw[0]), int(raw[1])
        elif len(raw) == 1:
            start_charge = int(raw[0])
    except ValueError:
        pass

    try:
        isotope_error = int(comet_params.get("isotope_error", "0"))
    except ValueError:
        isotope_error = 0

    return ChargePolicy(
        override_charge=override_charge,
        start_charge=start_charge,
        end_charge=end_charge,
        isotope_error=isotope_error,
    )


@dataclass(frozen=True)
class FastPathEntry:
    """One original input and its staged subset-MGF result."""

    mass_file: str
    mass_file_key: str
    mgf_parquet: Optional[str]
    staged_mgf: Optional[str]
    matched_scan_count: int
    staged_spectrum_count: int


@dataclass(frozen=True)
class FastPathArtifacts:
    """Artifacts created while building subset MGF files."""

    internal_novel_peptide_path: Path
    staged_dir: Path
    staged_mass_files: List[str]
    matched_input_count: int
    matched_scan_count: int
    staged_spectrum_count: int
    scan_matches_path: Path
    manifest_path: Path
    timing_json_path: Path
    entries: List[FastPathEntry]


def build_isotope_mass_offsets(isotope_error: int) -> list[float]:
    """Return CometPlus-compatible isotope mass offsets in Da."""

    if isotope_error == 7:
        return [float(x) * 4.0070995 for x in range(-2, 3)]

    positive_max = 0
    if isotope_error == 1:
        positive_max = 1
    elif isotope_error == 2:
        positive_max = 2
    elif isotope_error in {3, 4, 6}:
        positive_max = 3
    elif isotope_error == 5:
        positive_max = 1

    offsets = [float(x) * 1.003355 for x in range(0, positive_max + 1)]

    if isotope_error in {4, 5, 6}:
        negative_max = 3 if isotope_error == 6 else 1
        for x in range(1, negative_max + 1):
            offsets.append(-float(x) * 1.003355)

    if not offsets:
        offsets.append(0.0)
    return offsets


def resolve_internal_novel_export_path(config: PipelineConfig, output_dir: Path) -> Path:
    """Resolve the internal novel peptide TSV path for the fast path."""

    if config.internal_novel_peptide:
        return Path(str(config.internal_novel_peptide)).expanduser().resolve()
    if config.output_internal_novel_peptide:
        candidate = Path(str(config.output_internal_novel_peptide)).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (output_dir / candidate).resolve()
    return (output_dir / f"{config.output_prefix}.internal_novel_peptide.tsv").resolve()


def prepare_fastpath_export_invocation(
    run: RunConfig,
    config: PipelineConfig,
    output_dir: Path,
) -> tuple[RunConfig, PipelineConfig, Path]:
    """Prepare the CometPlus pass that exports internal novel peptide rows and stops."""

    internal_path = resolve_internal_novel_export_path(config, output_dir)
    export_config = replace(
        config,
        output_internal_novel_peptide=str(internal_path),
        internal_novel_peptide=None,
        stop_after_saving_novel_peptide=True,
        thread=config.fastpath_thread,
    )
    return run, export_config, internal_path


def prepare_fastpath_resume_invocation(
    run: RunConfig,
    config: PipelineConfig,
    staged_mass_files: Sequence[str],
    internal_novel_path: Path,
) -> tuple[RunConfig, PipelineConfig]:
    """Prepare the CometPlus pass that resumes from a detailed internal TSV."""

    staged_mass_files_list = [str(Path(path).expanduser().resolve()) for path in staged_mass_files]
    if not staged_mass_files_list:
        raise ValueError("Fast path resume pass requires at least one staged MGF file.")

    resume_run = replace(
        run,
        mass_file=staged_mass_files_list[0]
        if len(staged_mass_files_list) == 1
        else ",".join(staged_mass_files_list),
        mass_files=staged_mass_files_list,
    )
    resume_config = replace(
        config,
        novel_protein=None,
        novel_peptide=None,
        known_peptide=None,
        output_internal_novel_peptide=None,
        internal_novel_peptide=str(internal_novel_path.resolve()),
        stop_after_saving_novel_peptide=False,
        thread=config.fastpath_thread,
    )
    return resume_run, resume_config


def cleanup_fastpath_staged_dir(artifacts: Optional[FastPathArtifacts]) -> None:
    """Remove staged subset-MGF files when keep_tmp is disabled."""

    if artifacts is None:
        return
    if artifacts.staged_dir.exists():
        shutil.rmtree(artifacts.staged_dir)


def build_fastpath_subset_mgfs(
    run: RunConfig,
    config: PipelineConfig,
    output_dir: Path,
    internal_novel_path: Path,
) -> FastPathArtifacts:
    """Build one staged subset MGF per matched input file using DuckDB.

    Workers run in parallel (up to fastpath_thread).  Each worker reads one
    mgf.parquet file and writes one subset MGF containing only the matched
    spectra.  All staged MGF files are passed to CometPlus pass 2.
    """

    if not config.fastpath_enabled:
        raise ValueError("Fast path subset building requires config.fastpath_enabled=True.")
    if config.ms2_parquet is None:
        raise ValueError("Fast path subset building requires --ms2-parquet.")
    non_parquet_inputs = [
        str(Path(path).resolve())
        for path in run.mass_files
        if not Path(str(path)).name.lower().endswith(".mgf.parquet")
    ]
    if non_parquet_inputs:
        raise ValueError(
            "Fast path subset building requires all run.mass_files to be .mgf.parquet inputs. "
            f"Unsupported input(s): {', '.join(non_parquet_inputs[:3])}"
        )

    output_dir = output_dir.resolve()
    staged_dir = (output_dir / f"{config.output_prefix}.fastpath_subset_mgfs").resolve()
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    scan_matches_path = (output_dir / f"{config.output_prefix}.fastpath.scan_matches.tsv").resolve()
    manifest_path = (output_dir / f"{config.output_prefix}.fastpath.mgf_manifest.tsv").resolve()
    timing_json_path = (output_dir / f"{config.output_prefix}.fastpath.timing.json").resolve()

    # Read charge policy from the run's params file so the DuckDB prefilter
    # mirrors CometPlus override_charge / precursor_charge behaviour.
    charge_policy = build_charge_policy(parse_comet_params(run.params or ""))

    scan_match_start = perf_counter()
    internal_df = _load_detailed_internal_novel_table(internal_novel_path)
    key_to_mass_file = {derive_mass_file_key(path): str(Path(path).resolve()) for path in run.mass_files}
    key_to_mgf_parquet = {derive_mass_file_key(path): str(Path(path).resolve()) for path in run.mass_files}
    ordered_keys = [derive_mass_file_key(path) for path in run.mass_files]
    numeric_ids = _resolve_numeric_mass_file_ids(ordered_keys)
    scan_matches = _query_scan_matches(
        ms2_parquet=Path(config.ms2_parquet),
        internal_df=internal_df,
        ordered_keys=ordered_keys,
        numeric_ids=numeric_ids,
        charge_policy=charge_policy,
    )
    scan_match_sec = perf_counter() - scan_match_start

    if scan_matches.empty:
        raise RuntimeError(
            "Fast path requested, but DuckDB found no scan matches between "
            f"{internal_novel_path} and {config.ms2_parquet}."
        )

    scan_matches["mass_file"] = scan_matches["mass_file_key"].map(key_to_mass_file)
    scan_matches.to_csv(scan_matches_path, sep="\t", index=False)

    # O(n) groupby instead of O(n*k) boolean filters for 1808 keys over 1.1M rows.
    scan_groups: dict[str, List[int]] = {
        k: list(g["scan_id"])
        for k, g in scan_matches.groupby("mass_file_key", sort=False)
    }

    write_start = perf_counter()
    n_workers = max(1, config.fastpath_thread or 1)

    # ProcessPoolExecutor bypasses the GIL, giving true parallelism for the
    # CPU-bound string formatting inside _write_spectra_to_handle.  Pass only
    # picklable primitives (str, list[int]) to avoid DataFrame serialization cost.
    futures: dict[str, Future[FastPathEntry]] = {}
    staged_dir_str = str(staged_dir)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for key in ordered_keys:
            futures[key] = pool.submit(
                _process_one_file,
                key,
                key_to_mass_file[key],
                key_to_mgf_parquet[key],
                scan_groups.get(key, []),
                staged_dir_str,
            )

    # Collect in original key order; re-raise any worker exception here.
    entries: List[FastPathEntry] = []
    staged_mass_files: List[str] = []
    staged_spectrum_total = 0
    for key in ordered_keys:
        entry = futures[key].result()
        entries.append(entry)
        if entry.staged_mgf is not None:
            staged_mass_files.append(entry.staged_mgf)
            staged_spectrum_total += entry.staged_spectrum_count

    if not staged_mass_files:
        raise RuntimeError("Fast path found scan matches, but no staged subset MGF files were created.")

    write_sec = perf_counter() - write_start
    _write_manifest(manifest_path, entries)
    timing_json_path.write_text(
        json.dumps(
            {
                "scan_match_seconds": round(scan_match_sec, 6),
                "mgf_write_seconds": round(write_sec, 6),
                "matched_input_count": sum(1 for entry in entries if entry.matched_scan_count > 0),
                "matched_scan_count": int(len(scan_matches)),
                "staged_spectrum_count": staged_spectrum_total,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return FastPathArtifacts(
        internal_novel_peptide_path=internal_novel_path.resolve(),
        staged_dir=staged_dir,
        staged_mass_files=staged_mass_files,
        matched_input_count=sum(1 for entry in entries if entry.matched_scan_count > 0),
        matched_scan_count=int(len(scan_matches)),
        staged_spectrum_count=staged_spectrum_total,
        scan_matches_path=scan_matches_path,
        manifest_path=manifest_path,
        timing_json_path=timing_json_path,
        entries=entries,
    )


def _load_detailed_internal_novel_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", comment="#")
    required_columns = {"charge", "mz_window_min", "mz_window_max"}
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise ValueError(
            f"Fast path requires a detailed internal novel peptide TSV with columns "
            f"{', '.join(sorted(required_columns))}. Missing in {path}: {', '.join(missing)}"
        )
    table = table.loc[:, ["charge", "mz_window_min", "mz_window_max"]].dropna().copy()
    if table.empty:
        raise ValueError(f"Detailed internal novel peptide TSV has no usable mass-window rows: {path}")
    table["charge"] = pd.to_numeric(table["charge"], errors="raise").astype(int)
    table["mz_window_min"] = pd.to_numeric(table["mz_window_min"], errors="raise").astype(float)
    table["mz_window_max"] = pd.to_numeric(table["mz_window_max"], errors="raise").astype(float)
    return table.drop_duplicates().reset_index(drop=True)


def _resolve_numeric_mass_file_ids(keys: Sequence[str]) -> dict[int, str]:
    numeric_ids: dict[int, str] = {}
    for key in keys:
        if not str(key).isdigit():
            raise ValueError(
                f"Fast path requires numeric mass-file basenames that match ms2.parquet idn values. "
                f"Unsupported basename: {key}"
            )
        numeric_ids[int(key)] = str(key)
    return numeric_ids


def _process_one_file(
    key: str,
    mass_file: str,
    mgf_parquet_path: str,
    scan_ids: List[int],
    staged_dir_str: str,
) -> FastPathEntry:
    """Query one mgf.parquet and write its subset MGF.

    Accepts only picklable primitives so it can run in a ProcessPoolExecutor
    worker, bypassing the GIL for CPU-bound string formatting.
    """

    matched_scan_count = len(scan_ids)
    if matched_scan_count == 0:
        return FastPathEntry(
            mass_file=mass_file,
            mass_file_key=key,
            mgf_parquet=None,
            staged_mgf=None,
            matched_scan_count=0,
            staged_spectrum_count=0,
        )

    staged_dir = Path(staged_dir_str)
    mgf_parquet = Path(mgf_parquet_path).resolve()
    if not mgf_parquet.is_file():
        raise FileNotFoundError(f"Fast path MGF parquet file does not exist: {mgf_parquet}")

    scan_group = pd.DataFrame({"scan_id": scan_ids})
    spectra_df = _query_mgf_spectra(mgf_parquet, scan_group)
    actual_scan_count = int(spectra_df["scan_number"].nunique()) if not spectra_df.empty else 0
    if actual_scan_count != matched_scan_count:
        raise RuntimeError(
            f"Fast path spectrum mismatch for {mgf_parquet}: expected {matched_scan_count} scans, "
            f"but extracted {actual_scan_count}."
        )

    staged_path = (staged_dir / f"{key}.mgf").resolve()
    written_count = _write_subset_mgf(staged_path, spectra_df, fallback_title=key)
    if written_count != matched_scan_count:
        raise RuntimeError(
            f"Fast path MGF write mismatch for {staged_path}: expected {matched_scan_count} spectra, "
            f"but wrote {written_count}."
        )
    return FastPathEntry(
        mass_file=mass_file,
        mass_file_key=key,
        mgf_parquet=str(mgf_parquet),
        staged_mgf=str(staged_path),
        matched_scan_count=matched_scan_count,
        staged_spectrum_count=written_count,
    )


def _build_charge_join_condition(policy: ChargePolicy) -> str:
    """Return the SQL fragment (AND-appended to the mz-window JOIN) for charge matching.

    Mirrors CometPlus override_charge semantics exactly:

    0  keep known charge; unknown charge → try 2+ and 3+
    1  ignore known charges; always try the full precursor_charge range
    2  only accept spectra whose stored charge is within precursor_charge range;
       unknown-charge spectra are skipped entirely
    3  keep known charge; unknown charge → try full precursor_charge range
       (we cannot evaluate the intensity-below-precursor ratio here because
       ms2.parquet has no peak arrays, so we conservatively expand all
       unknown-charge spectra; CometPlus pass-2 applies the precise ratio check)

    precursor_charge = 0 0 means the range is disabled (iStartCharge == 0),
    which makes override=1/3 fall back to matching against all window charges.
    """
    oc = policy.override_charge
    sc = policy.start_charge
    ec = policy.end_charge
    range_active = sc > 0

    if oc == 1:
        if range_active:
            return f"cast(p.charge as integer) BETWEEN {sc} AND {ec}"
        return "1=1"

    if oc == 2:
        known_charge = "s.charge IS NOT NULL AND cast(s.charge as integer) > 0"
        in_range = (
            f" AND cast(s.charge as integer) BETWEEN {sc} AND {ec}"
            if range_active
            else ""
        )
        return (
            f"({known_charge}{in_range}"
            f" AND cast(s.charge as integer) = cast(p.charge as integer))"
        )

    if oc == 3:
        known_clause = (
            "(COALESCE(cast(s.charge as integer), 0) > 0"
            " AND COALESCE(cast(s.charge as integer), 0) = cast(p.charge as integer))"
        )
        if range_active:
            unknown_clause = (
                f"(COALESCE(cast(s.charge as integer), 0) <= 0"
                f" AND cast(p.charge as integer) BETWEEN {sc} AND {ec})"
            )
        else:
            unknown_clause = "COALESCE(cast(s.charge as integer), 0) <= 0"
        return f"{known_clause} OR {unknown_clause}"

    # override_charge == 0 (default): unknown → try charge 2 and 3.
    return (
        "(COALESCE(cast(s.charge as integer), 0) > 0"
        " AND COALESCE(cast(s.charge as integer), 0) = cast(p.charge as integer))"
        " OR (COALESCE(cast(s.charge as integer), 0) <= 0"
        "     AND cast(p.charge as integer) IN (2, 3))"
    )


def _query_scan_matches(
    *,
    ms2_parquet: Path,
    internal_df: pd.DataFrame,
    ordered_keys: Sequence[str],
    numeric_ids: dict[int, str],
    charge_policy: Optional[ChargePolicy] = None,
) -> pd.DataFrame:
    if charge_policy is None:
        charge_policy = ChargePolicy(override_charge=0, start_charge=1, end_charge=6)
    charge_cond = _build_charge_join_condition(charge_policy)

    # Build isotope-shifted windows.  CometPlus subtracts isotope offsets from
    # the experimental mass window before matching novel masses; in mz-space,
    # raw observed mz should match windows shifted by the same signed offsets.
    isotope_offsets = build_isotope_mass_offsets(charge_policy.isotope_error)
    nonzero_offsets = [offset for offset in isotope_offsets if offset != 0.0]
    if nonzero_offsets:
        shifted_frames = [internal_df.copy()]
        for offset in nonzero_offsets:
            shifted = internal_df.copy()
            shift_mz = offset / shifted["charge"]
            shifted["mz_window_min"] = shifted["mz_window_min"] + shift_mz
            shifted["mz_window_max"] = shifted["mz_window_max"] + shift_mz
            shifted_frames.append(shifted)
        combined_df = (
            pd.concat(shifted_frames, ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        combined_df = internal_df

    duckdb = _import_duckdb()
    connection = duckdb.connect()
    try:
        connection.register("novel_windows", combined_df)
        id_list = ",".join(str(value) for value in sorted(numeric_ids))
        query = f"""
            select
                cast(s.idn as varchar) as mass_file_key,
                cast(s.scan_id as integer) as scan_id
            from read_parquet({_duckdb_literal(str(ms2_parquet))}) as s
            join novel_windows as p
              on cast(s.mz as double) between cast(p.mz_window_min as double) and cast(p.mz_window_max as double)
              and ({charge_cond})
            where cast(s.idn as integer) in ({id_list})
            group by 1, 2
            order by 1, 2
        """
        matches = connection.execute(query).fetch_df()
    finally:
        connection.close()
    if matches.empty:
        return pd.DataFrame(columns=["mass_file_key", "scan_id"])
    matches["mass_file_key"] = matches["mass_file_key"].astype(str)
    matches["scan_id"] = pd.to_numeric(matches["scan_id"], errors="raise").astype(int)
    matches["mass_file_key"] = pd.Categorical(matches["mass_file_key"], categories=list(ordered_keys), ordered=True)
    matches = matches.sort_values(["mass_file_key", "scan_id"]).reset_index(drop=True)
    matches["mass_file_key"] = matches["mass_file_key"].astype(str)
    return matches


def _query_mgf_spectra(mgf_parquet: Path, matched_scans: pd.DataFrame) -> pd.DataFrame:
    duckdb = _import_duckdb()
    connection = duckdb.connect()
    try:
        scan_table = matched_scans.copy()
        scan_table["scan_id"] = pd.to_numeric(scan_table["scan_id"], errors="raise").astype(int)
        connection.register("matched_scans", scan_table)
        query = f"""
            select
                cast(m.scan_number as integer) as scan_number,
                m.precursor_mz,
                m.precursor_intensity,
                cast(m.precursor_charge as integer) as precursor_charge,
                m.title,
                m.rt,
                m.rt_unit,
                m.rt_seconds,
                m.mz_array,
                m.intensity_array
            from read_parquet({_duckdb_literal(str(mgf_parquet))}) as m
            join matched_scans as s
              on cast(m.scan_number as integer) = cast(s.scan_id as integer)
            order by cast(m.scan_number as integer)
        """
        spectra = connection.execute(query).fetch_df()
    finally:
        connection.close()
    return spectra


def _write_subset_mgf(path: Path, spectra_df: pd.DataFrame, *, fallback_title: str) -> int:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        return _write_spectra_to_handle(handle, spectra_df, fallback_title=fallback_title)


def _write_spectra_to_handle(handle: IO[str], spectra_df: pd.DataFrame, *, fallback_title: str) -> int:
    """Write all spectra in spectra_df to an already-open text handle. Returns spectrum count."""

    count = 0
    lines: list[str] = []
    for row in spectra_df.itertuples(index=False):
        mz_values = _as_list(row.mz_array)
        intensity_values = _as_list(row.intensity_array)
        if len(mz_values) != len(intensity_values):
            raise RuntimeError(
                f"Fast path MGF row has mismatched mz/intensity array lengths: "
                f"{len(mz_values)} != {len(intensity_values)}"
            )
        count += 1
        title_raw = row.title
        title = str(title_raw).strip() if pd.notna(title_raw) and str(title_raw).strip() else fallback_title
        lines.append("BEGIN IONS\n")
        lines.append(f"TITLE={title}\n")
        lines.append(f"SCANS={int(row.scan_number)}\n")
        rt_seconds = _resolve_rt_seconds_tuple(row)
        if rt_seconds is not None:
            lines.append(f"RTINSECONDS={_format_float(rt_seconds)}\n")
        precursor_mz = row.precursor_mz
        precursor_intensity = row.precursor_intensity
        if pd.notna(precursor_mz):
            pepmass = _format_float(float(precursor_mz))
            if pd.notna(precursor_intensity):
                pepmass = f"{pepmass} {_format_float(float(precursor_intensity))}"
            lines.append(f"PEPMASS={pepmass}\n")
        precursor_charge = row.precursor_charge
        if pd.notna(precursor_charge):
            lines.append(f"CHARGE={int(precursor_charge)}+\n")
        lines.extend(
            f"{_format_float(float(mz))} {_format_float(float(intensity))}\n"
            for mz, intensity in zip(mz_values, intensity_values)
        )
        lines.append("END IONS\n\n")
    handle.writelines(lines)
    return count


def _write_manifest(path: Path, entries: Sequence[FastPathEntry]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "mass_file",
                "mass_file_key",
                "mgf_parquet",
                "staged_mgf",
                "matched_scan_count",
                "staged_spectrum_count",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "mass_file": entry.mass_file,
                    "mass_file_key": entry.mass_file_key,
                    "mgf_parquet": "" if entry.mgf_parquet is None else entry.mgf_parquet,
                    "staged_mgf": "" if entry.staged_mgf is None else entry.staged_mgf,
                    "matched_scan_count": entry.matched_scan_count,
                    "staged_spectrum_count": entry.staged_spectrum_count,
                }
            )


def _resolve_rt_seconds_tuple(row) -> Optional[float]:
    rt_seconds = row.rt_seconds
    if rt_seconds is not None and pd.notna(rt_seconds):
        return float(rt_seconds)
    rt_value = row.rt
    if rt_value is None or not pd.notna(rt_value):
        return None
    rt = float(rt_value)
    unit = str(row.rt_unit or "").strip().lower()
    if unit == "minute":
        return rt * 60.0
    return rt


def _format_float(value: float) -> str:
    return f"{float(value):.10g}"


def _as_list(value) -> Iterable[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _duckdb_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _import_duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "ProtCosmo parquet fast path requires the optional Python dependency 'duckdb'."
        ) from exc
    return duckdb
