#!/usr/bin/env python3
"""Annotate peptide coding hits with gene/protein information."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from nt_sequence_annotation import (
    DEFAULT_DATA_DIR_NAME,
    annotate_rows,
    build_alias_map,
    derive_output_path,
    determine_schema,
    load_rows,
    merge_gff_indexes,
    parse_gbff,
    parse_gff3,
    parse_gpff,
    parse_gtf,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate peptide_coding_hits TSV files with nearby genes and protein metadata."
    )
    parser.add_argument("hits", help="Input peptide_coding_hits TSV file.")
    parser.add_argument(
        "--output",
        help="Optional output TSV path (defaults to <input>_annotated.tsv).",
    )
    parser.add_argument("--gff3", help="Primary GFF3 annotation file.")
    parser.add_argument(
        "--gff",
        action="append",
        default=[],
        help="Additional GFF files to merge.",
    )
    parser.add_argument("--gtf", help="GTF annotation file.")
    parser.add_argument("--gbff", help="GenBank (GBFF) file for gene features.")
    parser.add_argument("--gpff", help="Protein FASTA (GPFF) file for sequences.")
    return parser


def resolve_gff_sources(args: argparse.Namespace, data_dir: Path) -> Dict[str, object]:
    gff_index: Dict[str, object] = {}
    if args.gff3:
        merge_gff_indexes(gff_index, parse_gff3(Path(args.gff3)))
    else:
        candidate = data_dir / "GRCh38_genes.gff3"
        if candidate.is_file():
            merge_gff_indexes(gff_index, parse_gff3(candidate))
    for extra in args.gff:
        path = Path(extra)
        if path.is_file():
            merge_gff_indexes(gff_index, parse_gff3(path))
    if args.gtf:
        gtf_path = Path(args.gtf)
        if gtf_path.is_file():
            merge_gff_indexes(gff_index, parse_gtf(gtf_path))
    else:
        default_gtf = data_dir / "hg38.knownGene.gtf"
        if default_gtf.is_file():
            merge_gff_indexes(gff_index, parse_gtf(default_gtf))
    return gff_index


def resolve_gbff_sources(args: argparse.Namespace, data_dir: Path):
    gbff_genes: Dict[str, List] = {}
    if args.gbff:
        gbff_path = Path(args.gbff)
        if gbff_path.is_file():
            gbff_genes = parse_gbff(gbff_path)
    else:
        candidate = next((p for p in sorted(data_dir.glob("*.gbff")) if p.is_file()), None)
        if candidate:
            gbff_genes = parse_gbff(candidate)
    return gbff_genes


def resolve_gpff_sources(args: argparse.Namespace, data_dir: Path):
    protein_records = {}
    if args.gpff:
        path = Path(args.gpff)
        if path.is_file():
            protein_records = parse_gpff(path)
    else:
        candidate = next((p for p in sorted(data_dir.glob("*.gpff")) if p.is_file()), None)
        if candidate:
            protein_records = parse_gpff(candidate)
    return protein_records


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    hits_path = Path(args.hits)
    if not hits_path.is_file():
        parser.error(f"Input file '{hits_path}' does not exist.")
    rows, headers = load_rows(hits_path)
    schema = determine_schema(headers)
    if schema is None:
        parser.error("Unable to determine coordinate columns in the input file.")
    data_dir = REPO_ROOT / DEFAULT_DATA_DIR_NAME
    gff_index = resolve_gff_sources(args, data_dir)
    gff_alias = build_alias_map(gff_index.keys())
    gbff_genes = resolve_gbff_sources(args, data_dir)
    gbff_alias = build_alias_map(gbff_genes.keys())
    protein_records = resolve_gpff_sources(args, data_dir)
    annotated = annotate_rows(rows, schema, gff_index, gff_alias, gbff_genes, gbff_alias, protein_records)
    output_path = derive_output_path(hits_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(annotated[0].keys()) if annotated else list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(annotated or rows)
    print(f"Wrote {len(annotated)} annotated peptide hits to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
