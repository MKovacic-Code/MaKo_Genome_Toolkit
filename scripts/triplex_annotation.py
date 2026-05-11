#!/usr/bin/env python3
"""
Annotate triplex hits with gene/transcript/protein information using GFF3/GFF, GBFF, and GPFF files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from dataclasses import dataclass

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR_NAME = "data_human_homo_sapiens"

from nt_sequence_annotation import (
    GeneFeature,
    GFFIndex,
    ProteinRecord,
    find_genes,
    parse_gbff,
    parse_gff3,
    parse_gtf,
    parse_gpff,
    summarize_gff_annotations,
    merge_gff_indexes,
)


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "\t" in line:
                return "\t"
            if "," in line:
                return ","
    return "\t"


def load_triplex_hits(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [row for row in reader]
        headers = reader.fieldnames or []
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, headers


@dataclass
class TriplexSchema:
    seq_field: str
    start_field: str
    end_field: str
    strand_field: str | None


def determine_schema(headers: Sequence[str]) -> TriplexSchema | None:
    lower = {h.lower(): h for h in headers}

    def find_field(candidates: Sequence[str]) -> str | None:
        for cand in candidates:
            if cand in headers:
                return cand
            lc = cand.lower()
            if lc in lower:
                return lower[lc]
        return None

    seq_field = find_field(["sequence_id", "seq_id", "chrom_or_seq_id", "chromosome", "chrom"])
    start_field = find_field(["region_start", "start", "window_start"])
    end_field = find_field(["region_end", "end", "window_end"])
    strand_field = find_field(["strand"])
    if not (seq_field and start_field and end_field):
        return None
    return TriplexSchema(seq_field, start_field, end_field, strand_field)


def derive_output_path(input_path: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    suffix = input_path.suffix or ".tsv"
    base = input_path.with_suffix("")
    return base.with_name(base.name + "_annotated").with_suffix(suffix or ".tsv")


def normalize_seq_id(name: str) -> str:
    cleaned = name.lower().strip()
    cleaned = cleaned.replace("chr", "")
    cleaned = cleaned.replace("_", "")
    if "." in cleaned:
        cleaned = cleaned.split(".", 1)[0]
    for prefix in ("nc", "cm"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    cleaned = cleaned.lstrip("0")
    return cleaned or name.lower()


def build_alias_map(keys: Iterable[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for key in keys:
        normalized = normalize_seq_id(key)
        mapping.setdefault(normalized, []).append(key)
    return mapping


def resolve_entry(
    storage: Dict[str, object],
    alias_map: Dict[str, List[str]],
    seq_id: str,
) -> object | None:
    if seq_id in storage:
        return storage[seq_id]
    normalized = normalize_seq_id(seq_id)
    for alt in alias_map.get(normalized, []):
        if alt in storage:
            return storage[alt]
    return None


def annotate_hits(
    hits: Sequence[Dict[str, str]],
    schema: TriplexSchema,
    gff_index: Dict[str, GFFIndex],
    gff_alias: Dict[str, List[str]],
    gbff_genes: Dict[str, List[GeneFeature]],
    gbff_alias: Dict[str, List[str]],
    protein_records: Dict[str, ProteinRecord],
) -> List[Dict[str, object]]:
    annotated: List[Dict[str, object]] = []
    for row in hits:
        seq_id = row.get(schema.seq_field) or row.get("sequence_id") or ""
        if not seq_id:
            continue
        try:
            start = int(row.get(schema.start_field) or 0)
            end = int(row.get(schema.end_field) or start)
        except ValueError:
            continue
        start_field_lower = schema.start_field.lower()
        region_start = start + 1 if start_field_lower in {"start", "window_start"} else start
        region_end = max(region_start, end)
        strand = (row.get(schema.strand_field or "", "+") or "+")[:1]
        score = row.get("score", row.get("total_score", "NA"))
        length = row.get("length") or str(region_end - region_start + 1)
        mismatches = row.get("mismatches", "NA")
        rna_start = row.get("rna_start", "0")
        rna_end = row.get("rna_end", "0")

        gff_data = resolve_entry(gff_index, gff_alias, seq_id)
        gff_summary = summarize_gff_annotations(
            gff_data,
            region_start,
            region_end,
            protein_records,
        )
        gbff_data = resolve_entry(gbff_genes, gbff_alias, seq_id) or []
        gbff_hits = find_genes(gbff_data, region_start, region_end)
        gbff_gene_names = sorted(
            {
                gene.gene
                or gene.locus_tag
                or f"{gene.seq_id}:{gene.start}-{gene.end}"
                for gene in gbff_hits
                if gene.gene or gene.locus_tag
            }
        )
        gbff_products = sorted(
            {gene.product for gene in gbff_hits if gene.product}
        )
        combined_gene_names: set[str] = set()
        if gff_summary["gene_names"] != "NA":
            combined_gene_names.update(
                name for name in gff_summary["gene_names"].split(";") if name
            )
        combined_gene_names.update(gbff_gene_names)
        if not combined_gene_names:
            combined_gene_names.update(gene.label for gene in gbff_hits if gene.label)
        gene_name_value = (
            ";".join(sorted(combined_gene_names)) if combined_gene_names else "NA"
        )

        annotated.append(
            {
                "sequence_id": seq_id,
                "strand": strand,
                "region_start": region_start,
                "region_end": region_end,
                "score": score,
                "length": length,
                "mismatches": mismatches,
                "rna_start": rna_start,
                "rna_end": rna_end,
                "region_types": gff_summary["region_types"],
                "gene_names": gene_name_value,
                "transcript_ids": gff_summary["transcript_ids"],
                "transcript_names": gff_summary["transcript_names"],
                "protein_ids": gff_summary["protein_ids"],
                "protein_products": gff_summary["protein_products"],
                "coding": gff_summary.get("coding", "NA"),
                "gbff_products": ";".join(gbff_products) if gbff_products else "NA",
            }
        )
    return annotated


def write_output(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    headers = [
        "sequence_id",
        "strand",
        "region_start",
        "region_end",
        "score",
        "length",
        "mismatches",
        "rna_start",
        "rna_end",
        "region_types",
        "gene_names",
        "transcript_ids",
        "transcript_names",
        "protein_ids",
        "protein_products",
        "coding",
        "gbff_products",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(field, "NA") for field in headers])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate triplex hits with gene information from GFF3/GBFF/GPFF files."
    )
    parser.add_argument("hits", help="Triplex hits TSV file (raw or previously analyzed).")
    parser.add_argument(
        "--gff3",
        help="GFF3/GFF file for genomic annotations.",
    )
    parser.add_argument(
        "--gff",
        action="append",
        default=[],
        help="Additional GFF files for genome annotations.",
    )
    parser.add_argument(
        "--gtf",
        help="GTF file providing transcript/gene annotations (defaults to data/*.gtf when present).",
    )
    parser.add_argument(
        "--gbff",
        help="GenBank (GBFF) file to extract gene names/products.",
    )
    parser.add_argument(
        "--gpff",
        help="Protein FASTA (GPFF) to supply protein sequences/descriptions.",
    )
    parser.add_argument(
        "--output",
        help="Output TSV path (default: <input>_annotated.tsv).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = REPO_ROOT / DEFAULT_DATA_DIR_NAME

    rows, headers = load_triplex_hits(Path(args.hits))
    schema = determine_schema(headers)
    if schema is None:
        parser.error("Unable to determine coordinate columns for the input file.")

    gff_index: Dict[str, GFFIndex] = {}
    gff_alias: Dict[str, List[str]] = {}
    gff_paths: List[Path] = []
    if args.gff3:
        gff_paths.append(Path(args.gff3))
    for extra in args.gff:
        gff_paths.append(Path(extra))
    if gff_paths:
        for gff_path in gff_paths:
            if not gff_path.is_file():
                parser.error(f"GFF file '{gff_path}' does not exist.")
            merge_gff_indexes(gff_index, parse_gff3(gff_path))
    gtf_paths: List[Path] = []
    if args.gtf:
        gtf_paths.append(Path(args.gtf))
    else:
        default_gtf = data_dir / "hg38.knownGene.gtf"
        if default_gtf.is_file():
            gtf_paths.append(default_gtf)
        elif data_dir.is_dir():
            fallback = next((p for p in sorted(data_dir.glob("*.gtf")) if p.is_file()), None)
            if fallback:
                gtf_paths.append(fallback)
    for gtf_path in gtf_paths:
        if not gtf_path.is_file():
            parser.error(f"GTF file '{gtf_path}' does not exist.")
        merge_gff_indexes(gff_index, parse_gtf(gtf_path))
    if gff_index:
        gff_alias = build_alias_map(gff_index.keys())

    gbff_genes: Dict[str, List[GeneFeature]] = {}
    gbff_alias: Dict[str, List[str]] = {}
    if args.gbff:
        gbff_path = Path(args.gbff)
        if not gbff_path.is_file():
            parser.error(f"GBFF file '{gbff_path}' does not exist.")
        gbff_genes = parse_gbff(gbff_path)
        gbff_alias = build_alias_map(gbff_genes.keys())

    protein_records: Dict[str, ProteinRecord] = {}
    if args.gpff:
        gpff_path = Path(args.gpff)
        if not gpff_path.is_file():
            parser.error(f"GPFF file '{gpff_path}' does not exist.")
        protein_records = parse_gpff(gpff_path)

    annotated = annotate_hits(rows, schema, gff_index, gff_alias, gbff_genes, gbff_alias, protein_records)
    if not annotated:
        parser.error("No valid triplex hits to annotate.")

    out_path = derive_output_path(Path(args.hits), args.output)
    write_output(out_path, annotated)
    print(f"Wrote {len(annotated)} annotated triplex hits to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
