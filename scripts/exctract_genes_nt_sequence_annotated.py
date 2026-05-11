#!/usr/bin/env python3
"""Extract unique gene names and summary info from processed nucleotide output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract unique gene names from the processed TSV output and write a summary list."
    )
    parser.add_argument("hits", help="Processed nucleotide hits TSV file.")
    parser.add_argument(
        "--output-prefix",
        default="gene_list",
        help="Prefix for the generated gene_list.tsv file (default: gene_list).",
    )
    parser.add_argument(
        "--output-name",
        default="default",
        help="Name used to create the output_<name> directory for the generated files (default: default).",
    )
    return parser


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return "\t" if "\t" in line else ","
    return "\t"


def parse_gene_field(value: str) -> List[str]:
    if not value or value.upper() == "NA":
        return []
    names: List[str] = []
    for chunk in value.split(";"):
        name = chunk.strip()
        if not name:
            continue
        if ":" in name:
            name = name.split(":", 1)[0].strip()
        if name and name.upper() != "NA":
            names.append(name)
    return names


def load_rows(path: Path) -> List[Dict[str, str]]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [row for row in reader]


def build_gene_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    genes: Dict[str, Dict[str, str]] = {}
    for row in rows:
        raw_gene_field = row.get("gene_names") or row.get("genes") or ""
        gene_names = parse_gene_field(raw_gene_field)
        if not gene_names:
            continue
        seq_id = row.get("sequence_id", "")
        strand = (row.get("strand") or "+")[:1]
        try:
            region_start = int(row.get("region_start") or row.get("window_start") or row.get("start") or 0)
        except ValueError:
            region_start = 0
        try:
            region_end = int(row.get("region_end") or row.get("window_end") or row.get("end") or region_start)
        except ValueError:
            region_end = region_start
        annotations = row.get("region_annotations") or row.get("region_annotation") or "NA"
        transcript_ids = row.get("transcript_ids") or "NA"
        transcript_names = row.get("transcript_names") or "NA"
        protein_ids = row.get("protein_ids") or "NA"
        protein_products = row.get("protein_products") or "NA"
        coding = row.get("coding") or "NA"
        for gene in gene_names:
            existing = genes.get(gene)
            if existing is None or region_start < int(existing["region_start"]):
                genes[gene] = {
                    "gene_name": gene,
                    "sequence_id": seq_id,
                    "strand": strand,
                    "region_start": str(region_start),
                    "region_end": str(region_end),
                    "region_annotations": annotations,
                    "transcript_ids": transcript_ids,
                    "transcript_names": transcript_names,
                    "protein_ids": protein_ids,
                    "protein_products": protein_products,
                    "coding": coding,
                }
    return genes


def write_outputs(gene_data: Dict[str, Dict[str, str]], output_dir: Path, output_prefix: str) -> None:
    sorted_names = sorted(gene_data)
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = [
        "gene_name",
        "sequence_id",
        "strand",
        "region_start",
        "region_end",
        "region_annotations",
        "transcript_ids",
        "transcript_names",
        "protein_ids",
        "protein_products",
        "coding",
    ]
    tsv_path = output_dir / f"{output_prefix}.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        for name in sorted_names:
            row = gene_data[name]
            writer.writerow([row.get(col, "NA") for col in headers])


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    hits_path = Path(args.hits)
    if not hits_path.is_file():
        parser.error(f"Input file '{hits_path}' does not exist.")
    rows = load_rows(hits_path)
    gene_dict = build_gene_index(rows)
    if not gene_dict:
        print("No gene entries found in the provided file.")
        return 0
    output_dir = Path(f"output_{args.output_name}")
    write_outputs(gene_dict, output_dir, args.output_prefix)
    print(f"Wrote {len(gene_dict)} genes to {output_dir / (args.output_prefix + '.tsv')}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
