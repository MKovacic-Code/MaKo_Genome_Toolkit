#!/usr/bin/env python3
"""Annotate nucleotide hits with gene/transcript/protein information."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR_NAME = "data_human_homo_sapiens"


# ---------------------------------------------------------------------------
# Parsed-index cache
#
# Parsing the genome annotation files (GFF3/GTF up to a few GB, GBFF >4 GB) is
# by far the slowest part of annotation. The parsed result for a given file is
# fully determined by the file's bytes, so we memoise it to disk: the first run
# parses and writes a pickle alongside the source (in a .annotation_cache/
# folder), and later runs load that pickle whenever the source is unchanged.
# The cache key embeds the file size, mtime and a format version, so replacing
# or editing a source file -- or upgrading the parser -- invalidates the cache
# automatically.
#
# Caching is strictly best-effort: any read/write/validation failure falls back
# to a normal parse, so a missing, corrupt or stale cache can never yield a
# wrong or partial annotation. Set MAKO_NO_ANNOTATION_CACHE=1 to disable it.
# ---------------------------------------------------------------------------

CACHE_DIRNAME = ".annotation_cache"
# Bump when the parsed data structure changes (e.g. new GFFIndex fields) so old
# pickles are rejected instead of silently loaded.
CACHE_FORMAT_VERSION = 1


def _cache_disabled() -> bool:
    return bool(os.environ.get("MAKO_NO_ANNOTATION_CACHE"))


def _cache_key(path: Path, tag: str) -> tuple:
    stat = path.stat()
    return (tag, CACHE_FORMAT_VERSION, stat.st_size, stat.st_mtime_ns)


def _cache_path_for(path: Path, tag: str) -> Path:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return path.parent / CACHE_DIRNAME / f"{path.name}.{tag}.{digest}.pkl"


def _cached_parse(path: Path, tag: str, parser: Callable[[Path], object]) -> object:
    """Return ``parser(path)``, transparently memoised to disk by file fingerprint."""
    if _cache_disabled():
        return parser(path)
    cache_file = _cache_path_for(path, tag)
    try:
        if cache_file.is_file():
            with cache_file.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict) and payload.get("key") == _cache_key(path, tag):
                print(f"[annotation-cache] reusing parsed {tag} for {path.name}")
                return payload["data"]
    except Exception:
        pass  # corrupt/unreadable cache -> fall through to a fresh parse
    data = parser(path)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(cache_file.name + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(
                {"key": _cache_key(path, tag), "data": data},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp.replace(cache_file)  # atomic: a partial write never becomes the cache
        print(f"[annotation-cache] cached {tag} index for {path.name}")
    except Exception:
        # Read-only data dir / disk full / pickling issue: keep going uncached.
        pass
    return data


@dataclass
class GeneFeature:
    seq_id: str
    start: int
    end: int
    locus_tag: str | None
    gene: str | None
    product: str | None

    @property
    def label(self) -> str:
        parts = [self.gene or self.locus_tag or "?"]
        if self.product:
            parts.append(self.product)
        return ": ".join(parts)


@dataclass
class GFF3Feature:
    seq_id: str
    start: int
    end: int
    feature_type: str
    attributes: Dict[str, str]


@dataclass
class GFFIndex:
    features: List[GFF3Feature]
    starts: List[int]
    # Prefix maximum of feature end coordinates (aligned with `features`/`starts`).
    # max_ends[i] == max(features[0..i].end). Used by _gff_overlaps to find
    # features that begin before a query window but extend into it, without
    # stopping early at the first non-overlapping feature.
    max_ends: List[int]


def _finalize_index(features: List[GFF3Feature]) -> GFFIndex:
    """Sort features by start and precompute the start/prefix-max-end arrays."""
    features.sort(key=lambda feat: feat.start)
    starts = [feat.start for feat in features]
    max_ends: List[int] = []
    running = 0
    for feat in features:
        if feat.end > running:
            running = feat.end
        max_ends.append(running)
    return GFFIndex(features=features, starts=starts, max_ends=max_ends)


@dataclass
class ProteinRecord:
    accession: str
    description: str | None
    sequence: str


@dataclass
class InputSchema:
    seq_field: str
    start_field: str
    end_field: str
    strand_field: str | None


ANNOTATION_FIELDS = [
    "region_annotations",
    "gene_names",
    "transcript_ids",
    "transcript_names",
    "protein_ids",
    "protein_products",
    "protein_sequences",
    "coding",
]


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "\t" in line:
                return "\t"
            if "," in line:
                return ","
    return "\t"


def load_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [row for row in reader]
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows, fieldnames


def parse_gff3_attributes(raw: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for chunk in raw.split(";"):
        if not chunk.strip():
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            attrs[key.strip()] = value.strip()
    return attrs


INCLUDE_FEATURE_TYPES = {
    "gene",
    "mrna",
    "transcript",
    "exon",
    "cds",
    "intron",
    "five_prime_utr",
    "three_prime_utr",
    "lnc_rna",
    "mirna",
    "pseudogene",
}


def _parse_tabular_annotation(
    path: Path, attribute_parser: Callable[[str], Dict[str, str]]
) -> Dict[str, GFFIndex]:
    """Parse a 9-column GFF3/GTF file into per-sequence interval indexes.

    GFF3 and GTF share the same column layout and differ only in how the 9th
    (attributes) column is encoded, so a single reader handles both via a
    pluggable attribute parser.
    """
    per_seq: Dict[str, List[GFF3Feature]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            seq_id, _source, feature_type, start, end, *_rest, attributes = parts
            kind = feature_type.lower()
            if kind not in INCLUDE_FEATURE_TYPES:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except ValueError:
                continue
            per_seq[seq_id].append(
                GFF3Feature(
                    seq_id=seq_id,
                    start=start_i,
                    end=end_i,
                    feature_type=kind,
                    attributes=attribute_parser(attributes),
                )
            )
    return {seq_id: _finalize_index(features) for seq_id, features in per_seq.items()}


def parse_gff3(path: Path) -> Dict[str, GFFIndex]:
    return _cached_parse(
        path, "gff3", lambda p: _parse_tabular_annotation(p, parse_gff3_attributes)
    )


def parse_gtf_attributes(raw: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for chunk in raw.strip().split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if " " in chunk:
            key, value = chunk.split(" ", 1)
            attrs[key.strip()] = value.strip().strip('"')
    return attrs


def parse_gtf(path: Path) -> Dict[str, GFFIndex]:
    return _cached_parse(
        path, "gtf", lambda p: _parse_tabular_annotation(p, parse_gtf_attributes)
    )


def merge_gff_indexes(target: Dict[str, GFFIndex], additions: Dict[str, GFFIndex]) -> None:
    for seq_id, idx in additions.items():
        existing = target.get(seq_id)
        if existing is None:
            target[seq_id] = idx
        else:
            existing.features.extend(idx.features)
            target[seq_id] = _finalize_index(existing.features)


def parse_location(location: str) -> Tuple[int, int]:
    import re

    numbers = [int(num) for num in re.findall(r"\d+", location)]
    if not numbers:
        return 0, 0
    return min(numbers), max(numbers)


def parse_gbff(path: Path) -> Dict[str, List[GeneFeature]]:
    return _cached_parse(path, "gbff", _parse_gbff_impl)


def _parse_gbff_impl(path: Path) -> Dict[str, List[GeneFeature]]:
    annotations: Dict[str, List[GeneFeature]] = {}
    current_seq: str | None = None
    current_feature: GeneFeature | None = None
    pending_seq: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("LOCUS"):
                current_seq = line.split()[1]
                annotations.setdefault(current_seq, [])
                continue
            if line.startswith("VERSION"):
                parts = line.split()
                if len(parts) >= 2:
                    new_id = parts[1]
                    if pending_seq and new_id != pending_seq and pending_seq in annotations:
                        annotations.setdefault(new_id, []).extend(annotations.pop(pending_seq))
                    else:
                        annotations.setdefault(new_id, [])
                    pending_seq = new_id
                    current_seq = new_id
                continue
            if line.startswith("FEATURES"):
                current_feature = None
                continue
            if line.startswith("ORIGIN"):
                continue
            if not line.strip():
                continue
            if line.startswith("     gene") or line.startswith("     CDS"):
                location = line[21:].strip()
                start, end = parse_location(location)
                current_feature = GeneFeature(
                    seq_id=pending_seq or current_seq or "unknown",
                    start=start,
                    end=end,
                    locus_tag=None,
                    gene=None,
                    product=None,
                )
                annotations[current_feature.seq_id].append(current_feature)
            elif line.startswith("                     /") and current_feature:
                qualifier = line.strip()[1:]
                if "=" in qualifier:
                    key, value = qualifier.split("=", 1)
                    value = value.strip().strip('"')
                    if key == "gene":
                        current_feature.gene = value
                    elif key == "locus_tag":
                        current_feature.locus_tag = value
                    elif key == "product":
                        current_feature.product = value
    return annotations


def parse_gpff(path: Path) -> Dict[str, ProteinRecord]:
    return _cached_parse(path, "gpff", _parse_gpff_impl)


def _parse_gpff_impl(path: Path) -> Dict[str, ProteinRecord]:
    proteins: Dict[str, ProteinRecord] = {}
    current_acc: str | None = None
    current_version: str | None = None
    current_def: str | None = None
    seq_parts: List[str] = []
    reading_seq = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("LOCUS"):
                current_acc = None
                current_version = None
                current_def = None
                seq_parts = []
                reading_seq = False
            elif line.startswith("DEFINITION"):
                current_def = line[12:].strip()
            elif line.startswith("ACCESSION"):
                parts = line.split()
                if len(parts) >= 2:
                    current_acc = parts[1]
            elif line.startswith("VERSION"):
                parts = line.split()
                if len(parts) >= 2:
                    current_version = parts[1]
            elif line.startswith("ORIGIN"):
                reading_seq = True
                seq_parts = []
            elif line.startswith("//"):
                if current_acc and seq_parts:
                    accession = current_version or current_acc
                    proteins[accession] = ProteinRecord(
                        accession=accession,
                        description=current_def,
                        sequence="".join(seq_parts).replace(" ", "").upper(),
                    )
                reading_seq = False
            elif reading_seq:
                seq_parts.append("".join(ch for ch in line if ch.isalpha()))
    return proteins


def build_alias_map(keys: Iterable[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for key in keys:
        normalized = normalize_seq_id(key)
        mapping.setdefault(normalized, []).append(key)
    return mapping


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


def _gff_overlaps(index: GFFIndex | None, start: int, end: int) -> List[GFF3Feature]:
    if index is None:
        return []
    from bisect import bisect_left

    overlaps: List[GFF3Feature] = []
    starts = index.starts
    features = index.features
    max_ends = index.max_ends
    idx = bisect_left(starts, start)
    # Features beginning before the query window that may still extend into it.
    # Features are sorted by start, so their *ends* are not monotonic; a small
    # upstream feature (e.g. an exon) ending before `start` must NOT terminate
    # the scan, because an even-earlier feature (e.g. the enclosing gene) can
    # still span the window. The prefix-max-end array lets us stop only once no
    # earlier feature can possibly reach `start`.
    i = idx - 1
    while i >= 0 and max_ends[i] >= start:
        if features[i].end >= start:
            overlaps.append(features[i])
        i -= 1
    # Features beginning inside [start, end] always overlap (their start >= start
    # and start <= end, so their end >= start as well).
    j = idx
    n = len(features)
    while j < n and features[j].start <= end:
        overlaps.append(features[j])
        j += 1
    return overlaps


def summarize_gff_annotations(
    index: GFFIndex | None,
    start: int,
    end: int,
    protein_records: Dict[str, ProteinRecord],
) -> Dict[str, str]:
    overlaps = _gff_overlaps(index, start, end)
    if not overlaps:
        return {
            "region_types": "NA",
            "gene_names": "NA",
            "transcript_ids": "NA",
            "transcript_names": "NA",
            "protein_ids": "NA",
            "protein_products": "NA",
            "protein_sequences": "NA",
            "coding": "no",
        }
    region_types = set()
    gene_names = set()
    transcript_ids = set()
    transcript_names = set()
    protein_ids = set()
    protein_products = set()
    protein_sequences = set()
    coding = False
    for feature in overlaps:
        attrs = feature.attributes
        label = {
            "gene": "gene",
            "transcript": "transcript",
            "five_prime_utr": "UTR5",
            "three_prime_utr": "UTR3",
            "exon": "exon",
            "intron": "intron",
            "cds": "CDS",
            "lnc_rna": "lncRNA",
            "mrna": "mRNA",
            "mirna": "miRNA",
            "pseudogene": "pseudogene",
        }.get(feature.feature_type)
        if label:
            region_types.add(label)
        # Gene symbol is carried under different attribute keys depending on the
        # source: RefSeq GFF3 uses `gene`/`Name`, Ensembl/RefSeq GTF use
        # `gene_name`/`gene_id`, and some records only expose `locus_tag`. Try
        # them in descending order of readability so we surface a real symbol
        # whenever one exists instead of dropping the hit to NA.
        gene_val = (
            attrs.get("gene")
            or attrs.get("gene_name")
            or attrs.get("Name")
            or attrs.get("gene_id")
            or attrs.get("locus_tag")
        )
        if gene_val:
            gene_names.add(gene_val)
        if feature.feature_type in {"mrna", "transcript"}:
            tid = attrs.get("ID") or attrs.get("transcript_id")
            if tid:
                transcript_ids.add(tid)
            tname = attrs.get("Name")
            if tname:
                transcript_names.add(tname)
            product = attrs.get("product")
            if product:
                protein_products.add(product)
        if feature.feature_type == "cds":
            coding = True
            pid = attrs.get("protein_id") or attrs.get("ID")
            if pid:
                protein_ids.add(pid)
                product = attrs.get("product")
                if product:
                    protein_products.add(product)
                record = (
                    protein_records.get(pid)
                    or protein_records.get(pid.split(".")[0] if "." in pid else pid)
                )
                if record:
                    protein_sequences.add(record.sequence)
                    if record.description:
                        protein_products.add(record.description)
    return {
        "region_types": ";".join(sorted(region_types)) if region_types else "NA",
        "gene_names": ";".join(sorted(gene_names)) if gene_names else "NA",
        "transcript_ids": ";".join(sorted(transcript_ids)) if transcript_ids else "NA",
        "transcript_names": ";".join(sorted(transcript_names)) if transcript_names else "NA",
        "protein_ids": ";".join(sorted(protein_ids)) if protein_ids else "NA",
        "protein_products": ";".join(sorted(protein_products)) if protein_products else "NA",
        "protein_sequences": ";".join(sorted(protein_sequences)) if protein_sequences else "NA",
        "coding": "yes" if coding or protein_ids else "no",
    }


def find_genes(
    features: Sequence[GeneFeature],
    start: int,
    end: int,
) -> List[GeneFeature]:
    hits = []
    for feature in features:
        if feature.end < start or feature.start > end:
            continue
        hits.append(feature)
    return hits


def determine_schema(headers: Sequence[str]) -> InputSchema | None:
    lower = {h.lower(): h for h in headers}

    def find_field(candidates: Sequence[str]) -> str | None:
        for cand in candidates:
            if cand in headers:
                return cand
            lc = cand.lower()
            if lc in lower:
                return lower[lc]
        return None

    seq_field = find_field(["sequence_id", "seq_id", "chrom", "chromosome", "chrom_or_seq_id"])
    start_field = find_field(["region_start", "window_start", "start", "nt_start"])
    end_field = find_field(["region_end", "window_end", "end", "nt_end"])
    strand_field = find_field(["strand"])
    if not (seq_field and start_field and end_field):
        return None
    return InputSchema(seq_field, start_field, end_field, strand_field)


def derive_output_path(input_path: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    suffix = input_path.suffix or ".tsv"
    base = input_path.with_suffix("")
    return base.with_name(base.name + "_annotated").with_suffix(suffix or ".tsv")


def annotate_rows(
    rows: List[Dict[str, str]],
    schema: InputSchema,
    gff_index: Dict[str, GFFIndex],
    gff_alias: Dict[str, List[str]],
    gbff_genes: Dict[str, List[GeneFeature]],
    gbff_alias: Dict[str, List[str]],
    protein_records: Dict[str, ProteinRecord],
) -> List[Dict[str, str]]:
    annotated: List[Dict[str, str]] = []
    for row in rows:
        seq_id = row.get(schema.seq_field, "").strip()
        if not seq_id:
            continue
        try:
            region_start = int(row.get(schema.start_field) or 0)
            region_end = int(row.get(schema.end_field) or region_start)
        except ValueError:
            continue
        strand_value = (row.get(schema.strand_field or "", "+") or "+")
        # For combined strand windows, use only the forward part for annotation
        # so we don't double-annotate across both strands
        if strand_value == "combined" and row.get("forward_start") and row.get("forward_end"):
            try:
                annotation_start = int(row["forward_start"])
                annotation_end = int(row["forward_end"])
            except (ValueError, TypeError):
                annotation_start = region_start
                annotation_end = region_end
        else:
            annotation_start = region_start
            annotation_end = region_end
        gff_data = resolve_entry(gff_index, gff_alias, seq_id)
        gff_summary = summarize_gff_annotations(gff_data, annotation_start, annotation_end, protein_records)
        gbff_data = resolve_entry(gbff_genes, gbff_alias, seq_id) or []
        gbff_hits = find_genes(gbff_data, annotation_start, annotation_end)
        gbff_gene_names = sorted(
            {
                gene.gene
                or gene.locus_tag
                or f"{gene.seq_id}:{gene.start}-{gene.end}"
                for gene in gbff_hits
                if gene.gene or gene.locus_tag
            }
        )
        combined_gene_names: set[str] = set()
        if gff_summary["gene_names"] != "NA":
            combined_gene_names.update(name for name in gff_summary["gene_names"].split(";") if name)
        combined_gene_names.update(gbff_gene_names)
        if not combined_gene_names:
            combined_gene_names.update(gene.label for gene in gbff_hits if gene.label)
        gene_name_value = (
            ";".join(sorted(combined_gene_names)) if combined_gene_names else "NA"
        )
        updated = dict(row)
        updated["sequence_id"] = seq_id
        updated["strand"] = strand_value
        updated["region_start"] = str(region_start)
        updated["region_end"] = str(region_end)
        updated["region_annotations"] = gff_summary["region_types"]
        updated["gene_names"] = gene_name_value
        updated["transcript_ids"] = gff_summary["transcript_ids"]
        updated["transcript_names"] = gff_summary["transcript_names"]
        updated["protein_ids"] = gff_summary["protein_ids"]
        updated["protein_products"] = gff_summary["protein_products"]
        updated["protein_sequences"] = gff_summary["protein_sequences"]
        updated["coding"] = gff_summary["coding"]
        annotated.append(updated)
    return annotated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate nt_sequence_hits or nt_sequence_hits_analyzed TSV files using genome annotations."
    )
    parser.add_argument("hits", help="Input TSV file containing nucleotide hits.")
    parser.add_argument(
        "--output",
        help="Optional output TSV path. Defaults to <input>_annotated.tsv.",
    )
    parser.add_argument(
        "--gff3",
        help="Primary GFF3 file for genomic annotations.",
    )
    parser.add_argument(
        "--gff",
        action="append",
        default=[],
        help="Additional GFF files to merge into the annotation index.",
    )
    parser.add_argument(
        "--gtf",
        help="GTF annotation file; defaults to data/*.gtf when available.",
    )
    parser.add_argument(
        "--gbff",
        help="GenBank (.gbff) annotation file for gene lookups.",
    )
    parser.add_argument(
        "--gpff",
        help="Protein FASTA (GPFF) file providing amino acid sequences for protein IDs.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    hits_path = Path(args.hits)
    if not hits_path.is_file():
        parser.error(f"Hits file '{hits_path}' does not exist.")
    rows, headers = load_rows(hits_path)
    schema = determine_schema(headers)
    if schema is None:
        parser.error("Unable to determine coordinate columns (need start/end/sequence_id).")

    data_dir = REPO_ROOT / DEFAULT_DATA_DIR_NAME
    gff_index: Dict[str, GFFIndex] = {}
    gff_alias: Dict[str, List[str]] = {}
    gff_paths: List[Path] = []
    if args.gff3:
        gff_paths.append(Path(args.gff3))
    else:
        candidate = data_dir / "GRCh38_genes.gff3"
        if candidate.is_file():
            gff_paths.append(candidate)
    for extra in args.gff:
        gff_paths.append(Path(extra))
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

    annotated_rows = annotate_rows(rows, schema, gff_index, gff_alias, gbff_genes, gbff_alias, protein_records)
    if not annotated_rows:
        parser.error("No rows could be annotated (check the input file contents).")

    base_header = [h for h in headers if h not in ANNOTATION_FIELDS]
    ensured_fields = []
    for field in ("sequence_id", "strand", "region_start", "region_end"):
        if field not in base_header:
            ensured_fields.append(field)
    header = base_header + ensured_fields
    for extra in ANNOTATION_FIELDS:
        if extra not in header:
            header.append(extra)

    output_path = derive_output_path(hits_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for row in annotated_rows:
            writer.writerow([row.get(col, "NA") for col in header])
    print(f"Wrote {len(annotated_rows)} annotated rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
