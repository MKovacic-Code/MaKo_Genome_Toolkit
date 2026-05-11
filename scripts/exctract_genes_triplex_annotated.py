#!/usr/bin/env python3
"""
Extract unique gene names from annotated triplex hits, mirroring nucleotide gene extraction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract unique gene annotations from triplex hits TSV file."
    )
    parser.add_argument("hits", help="Annotated triplex hits TSV file.")
    parser.add_argument(
        "--output-prefix",
        default="triplex_gene_list",
        help="Base name for the gene_list TSV output (default: triplex_gene_list).",
    )
    parser.add_argument(
        "--output-name",
        default="default",
        help="Name for output_<name> directory (default: default).",
    )
    return parser


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "\t" in line:
                return "\t"
            if "," in line:
                return ","
    return "\t"


def load_rows(path: Path) -> List[Dict[str, str]]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [row for row in reader]


def parse_gene_field(value: str) -> List[str]:
    if not value or value.upper() == "NA":
        return []
    output: List[str] = []
    for chunk in value.split(";"):
        name = chunk.strip()
        if not name:
            continue
        if ":" in name:
            name = name.split(":", 1)[0].strip()
        if name and name.upper() != "NA":
            output.append(name)
    return output


def build_gene_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    genes: Dict[str, Dict[str, str]] = {}
    for row in rows:
        names = parse_gene_field(row.get("gene_names") or row.get("genes") or "")
        if not names:
            continue
        seq_id = row.get("sequence_id", row.get("chrom_or_seq_id", ""))
        strand = (row.get("strand") or "+")[:1]
        try:
            region_start = int(row.get("region_start") or row.get("start") or 0)
        except ValueError:
            region_start = 0
        try:
            region_end = int(row.get("region_end") or row.get("end") or region_start)
        except ValueError:
            region_end = region_start
        annotations = row.get("region_types") or row.get("region_annotations") or "NA"
        for gene in names:
            existing = genes.get(gene)
            if existing is None or region_start < int(existing["region_start"]):
                genes[gene] = {
                    "gene_name": gene,
                    "sequence_id": seq_id,
                    "strand": strand,
                    "region_start": str(region_start),
                    "region_end": str(region_end),
                    "region_annotations": annotations,
                }
    return genes


def write_outputs(genes: Dict[str, Dict[str, str]], output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_genes = sorted(genes)
    headers = [
        "gene_name",
        "sequence_id",
        "strand",
        "region_start",
        "region_end",
        "region_annotations",
    ]
    tsv_path = output_dir / f"{prefix}.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        for gene in sorted_genes:
            row = genes[gene]
            writer.writerow([row.get(h, "NA") for h in headers])


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    hits_path = Path(args.hits)
    if not hits_path.is_file():
        parser.error(f"Triplex hits file '{hits_path}' does not exist.")
    rows = load_rows(hits_path)
    gene_index = build_gene_index(rows)
    if not gene_index:
        print("No gene entries found in the provided file.")
        return 0
    output_dir = Path(f"output_{args.output_name}")
    write_outputs(gene_index, output_dir, args.output_prefix)
    print(f"Wrote {len(gene_index)} genes to {output_dir / (args.output_prefix + '.tsv')}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
