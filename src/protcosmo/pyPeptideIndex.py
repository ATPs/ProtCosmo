#!/usr/bin/env python3
"""Generate Comet-like peptide index tables from FASTA + comet.params.

This script parses comet.params, digests FASTA sequences, enumerates variable
mods, and writes TSV tables that mirror the suggested relational layout in
study/01-db-creation-suggestion.md.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(__file__))

from utils.peptide_index_tables import build_tables
from utils.pyLoadParameters import parse_comet_params
from utils.unimod_matcher import resolve_params_unimod_ids


class CometHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_UNIMOD_PATH = os.path.join(_SCRIPT_DIR, "resource", "common_unimod.csv")


def _normalize_unimod_arg(unimod_path: Optional[str]) -> Optional[str]:
    if unimod_path is None:
        return None
    value = unimod_path.strip()
    if not value:
        return None
    if value.lower() == "none":
        return None
    return os.path.expanduser(value)


def _resolve_unimod_maps(
    params,
    unimod_path: Optional[str],
    unimod_ppm: float,
    strict: bool = True,
) -> Tuple[Optional[Dict[int, str]], Optional[Dict[str, str]]]:
    normalized_unimod_path = _normalize_unimod_arg(unimod_path)
    if normalized_unimod_path is None:
        print("cannot match unimod: --unimod is None; continue normally.", file=sys.stderr)
        return None, None
    if not os.path.isfile(normalized_unimod_path):
        print(
            f"cannot match unimod: file does not exist: {normalized_unimod_path}; continue normally.",
            file=sys.stderr,
        )
        return None, None

    resolution = resolve_params_unimod_ids(params, normalized_unimod_path, unimod_ppm)
    if resolution.missing:
        print(
            f"cannot match unimod: {len(resolution.missing)} active modification(s) were not matched:",
            file=sys.stderr,
        )
        for missing in resolution.missing:
            print(f"  - {missing.to_message()}", file=sys.stderr)
        if strict:
            raise SystemExit(1)
        return None, None

    return resolution.variable_mod_to_unimod, resolution.fixed_mod_to_unimod


def generate_peptide_index_tables(
    params_path: str,
    fasta_paths: Optional[List[str]] = None,
    protein_sequences: Optional[List[str]] = None,
    max_proteins: Optional[int] = None,
    progress: bool = False,
    use_protein_name: bool = False,
    threads: int = 1,
    unimod_path: Optional[str] = _DEFAULT_UNIMOD_PATH,
    unimod_ppm: float = 10.0,
    strict_unimod: bool = True,
) -> Tuple[Dict[str, List[Tuple]], Tuple]:
    params = parse_comet_params(params_path)
    unimod_variable_map, unimod_fixed_map = _resolve_unimod_maps(
        params,
        unimod_path,
        unimod_ppm,
        strict=strict_unimod,
    )
    tables = build_tables(
        params,
        fasta_paths=fasta_paths,
        protein_sequences=protein_sequences,
        max_proteins=max_proteins,
        progress=progress,
        use_protein_name=use_protein_name,
        threads=threads,
        unimod_variable_map=unimod_variable_map,
        unimod_fixed_map=unimod_fixed_map,
    )
    run_id = 1
    created_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    database_label = " ".join(fasta_paths or []) if fasta_paths else "<inline_sequences>"
    index_run_row = (
        run_id,
        params.version,
        params.params_path,
        database_label,
        float(params.parsed.get("digest_mass_range", (0.0, 0.0))[0]),
        float(params.parsed.get("digest_mass_range", (0.0, 0.0))[1]),
        int(params.parsed.get("peptide_length_range", (0, 0))[0]),
        int(params.parsed.get("peptide_length_range", (0, 0))[1]),
        created_at,
        params.to_json(),
    )
    return tables, index_run_row


def format_field(value: Optional[object], round_floats: bool = True) -> str:
    if value is None:
        return "\\N"
    if isinstance(value, float):
        if round_floats:
            return f"{value:.6f}"
        return str(value)
    return str(value)


def write_tsv(
    path: str,
    rows: List[Tuple],
    round_floats: bool = True,
    header: Optional[List[str]] = None,
) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        if header:
            handle.write("#" + "\t".join(header) + "\n")
        for row in rows:
            handle.write(
                "\t".join(format_field(value, round_floats=round_floats) for value in row) + "\n"
            )


def main() -> None:
    description = (
        "Generate TSV tables that mirror Comet's peptide index layout.\n"
        "The parser follows Comet.cpp::LoadParameters rules (comments at '#',\n"
        "variable_modNN requires 8 fields, and [COMET_ENZYME_INFO] is parsed at EOF).\n\n"
        "Key comet.params settings used here:\n"
        "  database_name             FASTA path(s) unless overridden by -D/--database\n"
        "  search_enzyme_number      selects enzyme row in [COMET_ENZYME_INFO]\n"
        "  search_enzyme2_number     optional second enzyme (0 disables)\n"
        "  num_enzyme_termini        1=semi, 2=fully (default), 8 N-term only, 9 C-term only\n"
        "  allowed_missed_cleavage   max internal cleavage sites (default 2)\n"
        "  peptide_length_range      peptide length filter (default 5..50)\n"
        "  digest_mass_range         MH+ mass filter (default 600..5000)\n"
        "  max_variable_mods_in_peptide  total var mods per peptide (default 5)\n"
        "  require_variable_mod      1=keep only modified variants\n"
        "  clip_nterm_methionine     1=allow peptides starting at position 2 if protein starts with M\n"
        "  mass_type_parent          1=mono (default), 0=average\n"
        "  add_* / add_Nterm_*        static mods\n"
        "  --unimod / --unimod-ppm    map active Comet mods to UniMod IDs\n"
        "  --thread                 worker process count for variant enumeration (0=all CPUs)\n"
        "  variable_modNN            <mass> <residues> <binary> <min|max> <term_dist> <which_term> <required> <neutral_loss>\n"
        "                           - binary=1 enforces at most one site per peptide\n"
        "                           - term_dist=-1 disables distance filtering\n"
        "                           - which_term: 0=protein N, 1=protein C, 2=peptide N, 3=peptide C\n\n"
        "Example:\n"
        "  python pyPeptideIndex.py -P comet.params -D db.fasta -N out/comet\n"
        "  python pyPeptideIndex.py -P comet.params --protein ACDEFGHIK -N out/comet\n"
        "  python pyPeptideIndex.py -P comet.params -D db.fasta -N out/comet --thread 8\n\n"
        "Outputs (TSV with a single '#' header row):\n"
        "  <prefix>.index_run.tsv\n"
        "  <prefix>.static_mod.tsv\n"
        "  <prefix>.variable_mod.tsv\n"
        "  <prefix>.protein.tsv\n"
        "  <prefix>.peptide_sequence.tsv\n"
        "  <prefix>.peptide_sequence_protein.tsv\n"
        "  <prefix>.peptide_protein_location.tsv\n"
        "  <prefix>.peptide_variant.tsv\n"
        "  <prefix>.peptide_variant_mod.tsv\n"
    )
    epilog = (
        "Examples:\n"
        "  python pyPeptideIndex.py --params comet.params\n"
        "  python pyPeptideIndex.py -P comet.params -D db.fasta -N out/comet\n"
        "  python pyPeptideIndex.py -D db1.fasta -D db2.fasta --prefix results/comet\n"
        "  python pyPeptideIndex.py -P comet.params -D db.fasta --thread 0\n"
    )
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=CometHelpFormatter,
    )
    parser.add_argument(
        "-P",
        "--params",
        default="comet.params",
        help=(
            "Comet params file. Used for database_name, enzyme rules, peptide_length_range,\n"
            "digest_mass_range, static/variable mods, mass_type_parent, and related settings."
        ),
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-D",
        "--database",
        action="append",
        nargs="+",
        help=(
            "FASTA file(s). Overrides database_name from the params file.\n"
            "Repeat -D or pass multiple paths in one -D to include multiple FASTA files."
        ),
    )
    input_group.add_argument(
        "--protein",
        action="append",
        nargs="+",
        help=(
            "Protein sequence(s) supplied directly on the command line.\n"
            "Repeat --protein or pass multiple sequences in one --protein.\n"
            "Synthetic headers are assigned as protein_1, protein_2, ... and file offsets are synthetic."
        ),
    )
    parser.add_argument(
        "-N",
        "--prefix",
        default="comet",
        help="Output basename (can include a directory).",
    )
    parser.add_argument(
        "--max-record",
        type=int,
        default=None,
        help="Maximum number of proteins to process (default: all).",
    )
    parser.add_argument(
        "--use-protein-name",
        action="store_true",
        help="Use the protein name (first token in FASTA header) as protein_id.",
    )
    parser.add_argument(
        "--thread",
        type=int,
        default=1,
        help=(
            "Worker process count for peptide variant enumeration.\n"
            "Use 1 to disable multiprocessing; 0 uses all detected CPUs."
        ),
    )
    parser.add_argument(
        "--unimod",
        default=_DEFAULT_UNIMOD_PATH,
        help=(
            "UniMod CSV mapping file. Use 'None' to disable UniMod matching.\n"
            "If missing/None, the run continues without UniMod IDs.\n"
            "If present but active mods are unmatched, the run exits before indexing."
        ),
    )
    parser.add_argument(
        "--unimod-ppm",
        type=float,
        default=10.0,
        help="PPM tolerance used to match Comet modifications to UniMod entries.",
    )
    args = parser.parse_args()
    if args.thread < 0:
        parser.error("--thread must be >= 0")
    if args.unimod_ppm < 0.0:
        parser.error("--unimod-ppm must be >= 0")

    params = parse_comet_params(args.params)
    unimod_variable_map, unimod_fixed_map = _resolve_unimod_maps(
        params,
        args.unimod,
        args.unimod_ppm,
        strict=True,
    )

    fasta_paths: List[str] = []
    protein_sequences: List[str] = []
    if args.database:
        for group in args.database:
            fasta_paths.extend(group)
    elif args.protein:
        for group in args.protein:
            protein_sequences.extend(group)
    else:
        db_name = params.parsed.get("database_name", "")
        if db_name:
            fasta_paths = db_name.split()

    if not fasta_paths and not protein_sequences:
        raise SystemExit(
            "No FASTA database or protein sequences specified; use --database, --protein, or set database_name in params."
        )

    output_dir = os.path.dirname(args.prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tables = build_tables(
        params,
        fasta_paths=fasta_paths,
        protein_sequences=protein_sequences,
        max_proteins=args.max_record,
        progress=True,
        use_protein_name=args.use_protein_name,
        threads=args.thread,
        unimod_variable_map=unimod_variable_map,
        unimod_fixed_map=unimod_fixed_map,
    )

    run_id = 1
    created_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    database_label = " ".join(fasta_paths) if fasta_paths else "<inline_sequences>"
    index_run_row = (
        run_id,
        params.version,
        params.params_path,
        database_label,
        float(params.parsed.get("digest_mass_range", (0.0, 0.0))[0]),
        float(params.parsed.get("digest_mass_range", (0.0, 0.0))[1]),
        int(params.parsed.get("peptide_length_range", (0, 0))[0]),
        int(params.parsed.get("peptide_length_range", (0, 0))[1]),
        created_at,
        params.to_json(),
    )

    write_tsv(
        f"{args.prefix}.index_run.tsv",
        [index_run_row],
        header=[
            "run_id",
            "comet_version",
            "params_path",
            "database_label",
            "digest_mass_min",
            "digest_mass_max",
            "peptide_len_min",
            "peptide_len_max",
            "created_at_utc",
            "params_json",
        ],
    )
    write_tsv(
        f"{args.prefix}.static_mod.tsv",
        [(run_id, *row) for row in tables["static_mod"]],
        header=["run_id", "mod_index", "residue", "delta_mass", "site", "unimod_id"],
    )
    write_tsv(
        f"{args.prefix}.variable_mod.tsv",
        [(run_id, *row) for row in tables["variable_mod"]],
        header=[
            "run_id",
            "mod_index",
            "residues",
            "delta_mass",
            "binary_mod",
            "min_per_pep",
            "max_per_pep",
            "term_distance",
            "which_term",
            "require_this_mod",
            "neutral_loss1",
            "neutral_loss2",
            "unimod_id",
        ],
    )
    write_tsv(
        f"{args.prefix}.protein.tsv",
        [(run_id, *row) for row in tables["protein"]],
        header=["run_id", "protein_id", "pr_seq", "fasta_offset", "header"],
    )
    write_tsv(
        f"{args.prefix}.peptide_sequence.tsv",
        [(run_id, *row) for row in tables["peptide_sequence"]],
        header=["run_id", "peptide_id", "pep_seq", "length", "primary_protein_id"],
    )
    write_tsv(
        f"{args.prefix}.peptide_sequence_protein.tsv",
        [(run_id, *row) for row in tables["peptide_sequence_protein"]],
        header=["run_id", "peptide_id", "protein_id", "pep_seq"],
    )
    write_tsv(
        f"{args.prefix}.peptide_protein_location.tsv",
        tables["peptide_protein_location"],
        header=["protein_id", "peptide_id", "pep_start", "pep_end", "pep_seq"],
    )
    write_tsv(
        f"{args.prefix}.peptide_variant.tsv",
        [(run_id, *row) for row in tables["peptide_variant"]],
        round_floats=False,
        header=[
            "run_id",
            "variant_id",
            "peptide_id",
            "pep_seq",
            "mh_plus",
            "prev_aa",
            "next_aa",
            "var_mod_sites",
            "var_mod_sites_unimod",
            "var_mod_count",
            "fixed_mod_sites",
            "fixed_mod_sites_unimod",
            "fixed_mod_count",
            "mass_bin10",
        ],
    )
    write_tsv(
        f"{args.prefix}.peptide_variant_mod.tsv",
        [(run_id, *row) for row in tables["peptide_variant_mod"]],
        header=["run_id", "variant_id", "position", "mod_index", "unimod_id"],
    )


if __name__ == "__main__":
    main()


__all__ = [
    "build_tables",
    "generate_peptide_index_tables",
    "write_tsv",
    "parse_comet_params",
]
