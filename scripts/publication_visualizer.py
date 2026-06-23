#!/usr/bin/env python3
"""Render chromosome-style tracks for genome scanner outputs using Matplotlib for publication."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

DATASET_COLORS = [
    "#ef476f", "#118ab2", "#06d6a0", "#ffd166", "#073b4c", 
    "#b5179e", "#4895ef", "#ffb703", "#219ebc"
]

CLASS_COLOR_MAP = {
    "lnc_rna": "#8e44ad",
    "mrna": "#e63946",
    "mirna": "#457b9d",
    "pseudogene": "#f77f00",
    "five_prime_utr": "#43aa8b",
    "three_prime_utr": "#577590",
}

CLASS_DISPLAY = {
    "lnc_rna": "lncRNA",
    "mrna": "mRNA",
    "mirna": "miRNA",
    "pseudogene": "Pseudogene",
    "five_prime_utr": "5' UTR",
    "three_prime_utr": "3' UTR",
}

CLASS_ALIASES = {
    "lncrna": "lnc_rna", "lnc_rna": "lnc_rna", "mrna": "mrna", 
    "messenger_rna": "mrna", "mirna": "mirna", "micro_rna": "mirna",
    "pseudogene": "pseudogene", "pseudo": "pseudogene",
    "five_prime_utr": "five_prime_utr", "utr5": "five_prime_utr", "5utr": "five_prime_utr", "five_primeutr": "five_prime_utr",
    "three_prime_utr": "three_prime_utr", "utr3": "three_prime_utr", "3utr": "three_prime_utr", "three_primeutr": "three_prime_utr",
}

ANNOTATION_FIELDS = ["region_types", "region_annotations", "region_type"]
DEFAULT_SINGLE_COLOR = "#ff7f50"

SEQUENCE_FIELDS = ["sequence_id", "chrom", "chromosome", "chrom_or_seq_id", "seq_id", "contig"]
GENE_FIELDS = ["gene_names", "genes", "gene_name", "gene"]

KIND_COLUMN_CHOICES = {
    "nucleotide_hits": ("window_start", "window_end"),
    "nt_sequence_hits": ("window_start", "window_end"),
    "processed_hits": ("region_start", "region_end"),
    "nt_sequence_hits_analyzed": ("region_start", "region_end"),
    "nt_sequence_hits_annotated": ("region_start", "region_end"),
    "triplex_hits": ("start", "end"),
    "annotated_triplex": ("region_start", "region_end"),
    "triplex_hits_annotated": ("region_start", "region_end"),
}

FALLBACK_COLUMN_ORDER = [
    ("region_start", "region_end"),
    ("window_start", "window_end"),
    ("start", "end"),
]

LABEL_MODE_CHOICES = ("none", "gene", "gene+coords")
CHROMOSOME_NUMBER_LABELS = {**{idx: str(idx) for idx in range(1, 23)}, 23: "X", 24: "Y"}

# Chromosome lengths (bp) for the two bundled human assemblies, keyed by
# chromosome number (1-22, X=23, Y=24). Used to draw each ideogram at its true
# proportional length instead of stopping at the furthest hit.
GRCH38_LENGTHS = {
    1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555, 5: 181538259,
    6: 170805979, 7: 159345973, 8: 145138636, 9: 138394717, 10: 133797422,
    11: 135086622, 12: 133275309, 13: 114364328, 14: 107043718, 15: 101991189,
    16: 90338345, 17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
    21: 46709983, 22: 50818468, 23: 156040895, 24: 57227415,
}
T2T_LENGTHS = {
    1: 248387328, 2: 242696752, 3: 201105948, 4: 193574945, 5: 182045439,
    6: 172126628, 7: 160567428, 8: 146259331, 9: 150617247, 10: 134758134,
    11: 135127769, 12: 133324548, 13: 113566686, 14: 101161492, 15: 99753195,
    16: 96330374, 17: 84276897, 18: 80542538, 19: 61707364, 20: 66210255,
    21: 45090682, 22: 51324926, 23: 154259566, 24: 62460029,
}
ASSEMBLY_LENGTHS = {"grch38": GRCH38_LENGTHS, "t2t": T2T_LENGTHS}
# Approximate centromere position as a fraction of chromosome length. Only used
# to place the centromere constriction in the ideogram (acrocentric 13/14/15/
# 21/22 have small p-arms, hence small fractions).
CENTROMERE_FRACTION = {
    1: 0.496, 2: 0.388, 3: 0.458, 4: 0.263, 5: 0.269, 6: 0.350, 7: 0.377,
    8: 0.311, 9: 0.311, 10: 0.297, 11: 0.395, 12: 0.266, 13: 0.155, 14: 0.161,
    15: 0.186, 16: 0.407, 17: 0.301, 18: 0.230, 19: 0.447, 20: 0.436, 21: 0.257,
    22: 0.295, 23: 0.391, 24: 0.182,
}


def chromosome_number(seq_id: str) -> int | None:
    """Map a sequence id to a human chromosome number (1-22, X=23, Y=24)."""
    token = (seq_id or "").strip()
    if not token:
        return None
    normalized = token.split()[0]
    match = re.match(r"NC_0*(\d+)", normalized, re.IGNORECASE)
    if match:
        n = int(match.group(1))
        if 1 <= n <= 24:            # GRCh38: NC_000001..NC_000024
            return n
        if 60925 <= n <= 60948:     # T2T-CHM13v2.0: NC_060925..NC_060948
            return n - 60924
    low = normalized.lower()
    if low.startswith("chr"):
        suffix = normalized[3:].strip().upper()
        if suffix == "X":
            return 23
        if suffix == "Y":
            return 24
        try:
            n = int(suffix)
            return n if 1 <= n <= 24 else None
        except ValueError:
            return None
    try:
        n = int(normalized)
        return n if 1 <= n <= 24 else None
    except ValueError:
        return None


def detect_assembly(seq_ids: Iterable[str]) -> str:
    """Auto-detect assembly from accession ranges; default to GRCh38."""
    for sid in seq_ids:
        match = re.match(r"NC_0*(\d+)", (sid or "").strip(), re.IGNORECASE)
        if match and 60925 <= int(match.group(1)) <= 60948:
            return "t2t"
    return "grch38"


def chromosome_length(seq_id: str, assembly: str, fallback: int) -> int:
    """True chromosome length for a sequence id, or `fallback` for scaffolds."""
    num = chromosome_number(seq_id)
    table = ASSEMBLY_LENGTHS.get(assembly, GRCH38_LENGTHS)
    if num is not None and num in table:
        return table[num]
    return max(fallback, 1)


@dataclass
class DatasetMeta:
    name: str
    color: str
    opacity: float = 0.75
    style: str = "solid"


@dataclass
class RegionSegment:
    seq_id: str
    start: int
    end: int
    label: str
    dataset_index: int
    dataset_name: str
    dataset_color: str
    category: str | None
    opacity: float
    line_style: str
    annotation: str | None
    gene: str | None = None


def detect_delimiter(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i > 50: break
                if "\t" in line: return "\t"
                if "," in line: return ","
    except: pass
    return "\t"


def normalize_seq_id(value: str | None) -> str:
    if not value: return "unknown"
    return str(value).strip() or "unknown"


def format_chromosome_label(seq_id: str) -> str:
    token = (seq_id or "").strip()
    if not token: return "Chromosome"
    normalized = token.split()[0]

    def map_number(text: str) -> str | None:
        try: number = int(text)
        except ValueError: return None
        symbol = CHROMOSOME_NUMBER_LABELS.get(number)
        if symbol: return f"Chromosome {symbol}"
        if number > 0: return f"Chromosome {number}"
        return None

    match = re.match(r"NC_0*(\d+)", normalized, re.IGNORECASE)
    if match:
        mapped = map_number(match.group(1))
        if mapped: return mapped
    if normalized.lower().startswith("chr"):
        suffix = normalized[3:].strip()
        if not suffix: return "Chromosome"
        if suffix.upper() in ("X", "Y"): return f"Chromosome {suffix.upper()}"
        mapped = map_number(suffix)
        if mapped: return mapped
    mapped = map_number(normalized)
    if mapped: return mapped
    return token


def infer_coordinate_columns(headers: Sequence[str], kind: str) -> Tuple[str, str]:
    lowered = {h.lower(): h for h in headers}
    if kind != "auto":
        if kind not in KIND_COLUMN_CHOICES:
             # Try to find common columns anyway if kind is unknown
             for start, end in FALLBACK_COLUMN_ORDER:
                if start in headers and end in headers: return start, end
                if start.lower() in lowered and end.lower() in lowered:
                    return lowered[start.lower()], lowered[end.lower()]
             raise ValueError(f"Unknown input kind '{kind}' and could not find coordinates.")
        start, end = KIND_COLUMN_CHOICES[kind]
        if start in headers and end in headers: return start, end
        if start.lower() in lowered and end.lower() in lowered:
            return lowered[start.lower()], lowered[end.lower()]
        raise ValueError(f"Input kind '{kind}' expects columns '{start}'/'{end}'.")
    for start, end in FALLBACK_COLUMN_ORDER:
        if start in headers and end in headers: return start, end
        if start.lower() in lowered and end.lower() in lowered:
            return lowered[start.lower()], lowered[end.lower()]
    raise ValueError("Could not determine start/end column names automatically.")


def parse_sequence_filters(values: Sequence[str]) -> set[str]:
    result = set()
    for value in values:
        if not value: continue
        result.add(normalize_seq_id(value.strip()))
    return result


def parse_region_filters(values: Sequence[str]) -> Dict[str, List[Tuple[int, int]]]:
    filters: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for spec in values:
        if not spec: continue
        try:
            if ":" in spec:
                seq_part, coords = spec.split(":", 1)
                start_text, end_text = coords.replace(",", "").split("-")
                start, end = int(start_text), int(end_text)
                seq_id = normalize_seq_id(seq_part)
                filters[seq_id].append((start, end))
            else:
                # Just a sequence ID
                pass 
        except Exception:
            raise ValueError(f"Region '{spec}' must look like SEQ:START-END.")
    return filters


def within_region(seq_id: str, start: int, end: int, filters: Dict[str, List[Tuple[int, int]]]) -> bool:
    if not filters: return True
    ranges = filters.get(seq_id)
    if not ranges: return False # If we have filters but this sequence isn't in them, it's filtered out? 
    # Actually, if the seq_id is in filters but has no ranges, maybe it's "include all"?
    # For now, if seq_id is in filters, we check ranges.
    for r_start, r_end in ranges:
        if end >= r_start and start <= r_end: return True
    return False


def parse_gene_filters(values: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for entry in values:
        if not entry: continue
        for token in re.split(r"[;,/| ]+", entry):
            token = token.strip().lower()
            if token: result.add(token)
    return result


def format_segment_label(seq_id: str, start: int, end: int, gene_value: str | None, label_mode: str) -> str:
    if label_mode == "none": return ""
    coords = f"{seq_id}:{start}-{end}"
    gene_clean = (gene_value or "").strip()
    has_gene = bool(gene_clean) and gene_clean.upper() != "NA"
    if label_mode == "gene": return gene_clean if has_gene else coords
    if label_mode == "gene+coords": return f"{gene_clean} ({coords})" if has_gene else coords
    return coords


def parse_int(value: str | None) -> int | None:
    if not value: return None
    try: return int(float(str(value).strip()))
    except ValueError: return None


def normalize_line_style(value: str | None) -> str:
    if not value: return "solid"
    cleaned = str(value).strip().lower()
    return cleaned if cleaned in {"solid", "dashed"} else "solid"


def select_field(row: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        val = row.get(name)
        if val: return str(val).strip()
    return ""


def normalize_category_token(token: str) -> str | None:
    key = token.strip().lower().replace(" ", "_")
    return CLASS_ALIASES.get(key) if key else None


def detect_category(row: Dict[str, str]) -> str | None:
    for field in ANNOTATION_FIELDS:
        raw = row.get(field)
        if not raw or raw.upper() == "NA": continue
        for token in re.split(r"[;,/| ]+", str(raw)):
            normalized = normalize_category_token(token)
            if normalized: return normalized
    region_type = row.get("feature_type")
    if region_type:
        normalized = normalize_category_token(str(region_type))
        if normalized: return normalized
    return None


def build_segment_from_row(
    row: Dict[str, str], start_col: str, end_col: str, show_labels: bool,
    label_mode: str, dataset_index: int, dataset_name: str, dataset_color: str,
    opacity: float, line_style: str, annotation_field: str | None,
    seq_filters: set[str] | None, region_filters: Dict[str, List[Tuple[int, int]]] | None,
    gene_filters: set[str] | None,
) -> RegionSegment | None:
    seq_id = normalize_seq_id(select_field(row, SEQUENCE_FIELDS))
    if seq_filters and seq_id not in seq_filters: return None
    start = parse_int(row.get(start_col))
    end = parse_int(row.get(end_col))
    if start is None or end is None: return None
    if end < start: start, end = end, start
    if region_filters and not within_region(seq_id, start, end, region_filters): return None
    
    gene_value = select_field(row, GENE_FIELDS)
    label = format_segment_label(seq_id, start, end, gene_value, label_mode) if show_labels else ""
    
    if gene_filters:
        names = {t.strip().lower() for t in re.split(r"[;,/| ]+", gene_value or "") if t.strip()}
        if not names or names.isdisjoint(gene_filters): return None
        
    annotation = None
    if annotation_field:
        raw = row.get(annotation_field)
        if raw and str(raw).upper() != "NA": annotation = str(raw)
    category = detect_category(row)
    
    gene_clean = (gene_value or "").strip()
    gene_name = gene_clean if gene_clean and gene_clean.upper() != "NA" else None

    return RegionSegment(
        seq_id=seq_id, start=start, end=end, label=label, dataset_index=dataset_index,
        dataset_name=dataset_name, dataset_color=dataset_color, category=category,
        opacity=opacity, line_style=line_style, annotation=annotation, gene=gene_name,
    )


def iter_file_segments(
    path: Path, dataset_index: int, dataset_name: str, dataset_color: str,
    show_labels: bool, label_mode: str, kind_hint: str, opacity: float,
    line_style: str, annotation_field: str | None, seq_filters: set[str] | None,
    region_filters: Dict[str, List[Tuple[int, int]]] | None, gene_filters: set[str] | None,
) -> Iterable[RegionSegment]:
    delimiter = detect_delimiter(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames: return
            start_col, end_col = infer_coordinate_columns(reader.fieldnames, kind_hint)
            annotation_col = None
            if annotation_field:
                lowered = {h.lower(): h for h in reader.fieldnames}
                annotation_col = annotation_field if annotation_field in reader.fieldnames else lowered.get(annotation_field.lower())
            for row in reader:
                segment = build_segment_from_row(
                    row, start_col, end_col, show_labels, label_mode, dataset_index,
                    dataset_name, dataset_color, opacity, line_style, annotation_col,
                    seq_filters, region_filters, gene_filters,
                )
                if segment is not None: yield segment
    except Exception as e:
        print(f"Warning: Failed to parse {path}: {e}")


def natural_key(value: str) -> Tuple:
    parts = re.findall(r"\d+|\D+", value)
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts)


def group_segments(segments: Sequence[RegionSegment], dataset_count: int, seq_filters: set[str] | None = None) -> Dict[str, Dict[int, List[RegionSegment]]]:
    grouped: Dict[str, Dict[int, List[RegionSegment]]] = {}
    
    # Initialize with filtered sequences if any
    if seq_filters:
        for s in seq_filters:
            grouped[s] = {idx: [] for idx in range(dataset_count)}
            
    for segment in segments:
        bucket = grouped.setdefault(segment.seq_id, {idx: [] for idx in range(dataset_count)})
        bucket[segment.dataset_index].append(segment)
        
    for seq_bucket in grouped.values():
        for dataset_segments in seq_bucket.values():
            dataset_segments.sort(key=lambda seg: seg.start)
    return grouped


def compute_density_map(chrom_length: int, segments: List[RegionSegment], bins: int = 200) -> List[float]:
    if chrom_length <= 0 or not segments: return [0.0] * bins
    density = [0.0] * bins
    for segment in segments:
        norm_start = max(0.0, (segment.start - 1) / chrom_length)
        norm_end = min(1.0, segment.end / chrom_length)
        start_bin = int(norm_start * bins)
        end_bin = int(norm_end * bins)
        for bin_idx in range(start_bin, min(end_bin + 1, bins)):
            if bin_idx >= bins: break
            overlap_start = max(norm_start, bin_idx / bins)
            overlap_end = min(norm_end, (bin_idx + 1) / bins)
            if overlap_end > overlap_start:
                density[bin_idx] += overlap_end - overlap_start
    max_value = max(density) if density else 0.0
    return [v / max_value for v in density] if max_value > 0 else density


def render_plot(
    grouped: Dict[str, Dict[int, List[RegionSegment]]],
    dataset_meta: Sequence[DatasetMeta],
    output_path: Path,
    args: argparse.Namespace,
    classification_mode: bool,
    category_colors: Dict[str, str],
    legend_entries: Sequence[Tuple[str, str]],
    legend_title: str | None,
    fallback_color: str,
) -> None:
    # Setup fonts
    plt.rcParams["font.family"] = args.font_family
    plt.rcParams["font.size"] = args.font_size
    
    seq_order = sorted(grouped, key=natural_key)
    length_map = {
        seq: max((seg.end for d_segs in seq_data.values() for seg in d_segs), default=0)
        for seq, seq_data in grouped.items()
    }
    
    dataset_count = len(dataset_meta)
    num_chroms = max(1, len(seq_order))
    
    fig_width = args.fig_width
    fig_height = args.fig_height
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=args.dpi)
    ax.set_xlim(0, num_chroms)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    
    if args.title:
        fig.suptitle(args.title, fontsize=args.font_size + 4, fontweight="bold")
    
    # Track width calculations
    chrom_width_ratio = 0.1 / num_chroms if dataset_count == 1 else 0.08 / num_chroms
    track_width = chrom_width_ratio / max(1, dataset_count)
    dataset_gap = track_width * 0.1
    
    max_length_global = max(length_map.values(), default=0)
    if max_length_global == 0: max_length_global = 1

    for idx, seq in enumerate(seq_order):
        seq_data = grouped[seq]
        x_center = idx + 0.5
        chrom_length = length_map.get(seq, 0)
        
        # Scaling factor for this chromosome
        if args.scale_mode == "relative" and chrom_length > 0:
            scale_len = chrom_length
        else:
            scale_len = max_length_global
            
        # Draw Chromosome Title
        ax.text(x_center, 1.02, format_chromosome_label(seq), ha="center", va="bottom", fontsize=args.font_size)
        
        for d_idx in range(dataset_count):
            x_left = x_center - (dataset_count * track_width + (dataset_count - 1) * dataset_gap) / 2
            dx = x_left + d_idx * (track_width + dataset_gap)
            
            # Draw Chromosome Background
            # If scaling is relative, background is 1.0. If absolute, it's relative to global max.
            bg_height = (chrom_length / max_length_global) if args.scale_mode == "absolute" else 1.0
            if bg_height <= 0: bg_height = 0.01 # Minimal line for visibility
            
            chrom_patch = patches.FancyBboxPatch(
                (dx, 1.0 - bg_height), track_width, bg_height,
                boxstyle=f"round,pad=0,rounding_size={min(track_width/2, bg_height/2)}",
                facecolor=args.bg_color, edgecolor="#444444", lw=0.5, alpha=args.bg_alpha
            )
            ax.add_patch(chrom_patch)
            
            segments = seq_data.get(d_idx, [])
            if args.density and segments:
                densities = compute_density_map(chrom_length, segments)
                bins = len(densities)
                bin_height = bg_height / bins
                for bin_idx, value in enumerate(densities):
                    if value > 0:
                        y0 = 1.0 - (bin_idx + 1) * bin_height
                        op = dataset_meta[d_idx].opacity * min(1.0, value)
                        rect = patches.Rectangle((dx, y0), track_width, bin_height, facecolor=dataset_meta[d_idx].color, alpha=op, edgecolor="none")
                        ax.add_patch(rect)
                        
            for segment in segments:
                if classification_mode and segment.category:
                    color = category_colors.get(segment.category, fallback_color)
                elif classification_mode:
                    color = fallback_color
                else:
                    color = dataset_meta[d_idx].color
                    
                start_ratio = max(0.0, segment.start / scale_len)
                end_ratio = min(1.0, segment.end / scale_len)
                
                y_top = 1.0 - start_ratio
                rect_height = end_ratio - start_ratio
                y_bottom = 1.0 - end_ratio
                
                # Minimum height for visibility (e.g. 1 pixel approx)
                if rect_height < 0.002:
                    rect_height = 0.002
                    y_bottom = y_top - rect_height
                    
                ls = "--" if segment.line_style == "dashed" else "-"
                alpha = max(0.05, min(1.0, segment.opacity))
                
                hit_patch = patches.Rectangle(
                    (dx, y_bottom), track_width, rect_height,
                    facecolor=color, edgecolor=color, alpha=alpha, lw=0.5, linestyle=ls
                )
                ax.add_patch(hit_patch)
                
                if args.annotate_column and segment.annotation:
                    ax.text(
                        dx + track_width / 2, y_bottom + rect_height / 2,
                        str(segment.annotation),
                        ha="center", va="center", fontsize=args.font_size - 4, color="black", rotation=90
                    )
                
                if not args.color_only and segment.label:
                    label_x = x_center + (dataset_count * track_width + (dataset_count - 1) * dataset_gap) / 2 + 0.02
                    mid_y = y_bottom + rect_height / 2
                    ax.plot([dx + track_width, label_x - 0.01], [mid_y, mid_y], color=color, lw=0.5, alpha=0.3)
                    ax.text(label_x, mid_y, segment.label, ha="left", va="center", fontsize=args.font_size - 2, color="black")

    if legend_entries:
        handles = [patches.Rectangle((0,0),1,1, facecolor=color, edgecolor="#333", lw=0.5) for _, color in legend_entries]
        labels = [label for label, _ in legend_entries]
        
        if args.legend_pos == "bottom":
            ax.legend(handles, labels, title=legend_title, loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, ncol=min(len(labels), 4))
        else:
            ax.legend(handles, labels, title=legend_title, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=args.output_format, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _unique_genes_for_seq(seq_data: Dict[int, List[RegionSegment]], limit: int = 8) -> Tuple[List[str], int]:
    """Ordered-by-position unique gene names hit on a chromosome, plus overflow count."""
    seen: set[str] = set()
    ordered: List[Tuple[int, str]] = []
    for segments in seq_data.values():
        for seg in segments:
            if not seg.gene:
                continue
            for name in re.split(r"[;,/| ]+", seg.gene):
                name = name.strip()
                key = name.lower()
                if name and key not in seen:
                    seen.add(key)
                    ordered.append((seg.start, name))
    ordered.sort(key=lambda item: item[0])
    names = [name for _, name in ordered]
    if len(names) <= limit:
        return names, 0
    return names[:limit], len(names) - limit


def render_karyogram(
    grouped: Dict[str, Dict[int, List[RegionSegment]]],
    dataset_meta: Sequence[DatasetMeta],
    output_path: Path,
    args: argparse.Namespace,
    classification_mode: bool,
    category_colors: Dict[str, str],
    legend_entries: Sequence[Tuple[str, str]],
    legend_title: str | None,
    fallback_color: str,
    assembly: str,
) -> None:
    """Draw horizontal human-chromosome ideograms with hits marked along them.

    Each chromosome is drawn at its true proportional length with a centromere
    constriction; hits are vertical ticks at their genomic position, coloured by
    dataset (or functional class for a single annotated dataset). Gene names hit
    on each chromosome are listed to the right.
    """
    plt.rcParams["font.family"] = args.font_family
    plt.rcParams["font.size"] = args.font_size

    seq_order = sorted(grouped, key=natural_key)
    num_chroms = max(1, len(seq_order))
    dataset_count = len(dataset_meta)
    show_labels = args.label_mode in ("gene", "gene+coords") and not args.color_only

    def seq_length(seq: str) -> int:
        far = max((seg.end for d in grouped[seq].values() for seg in d), default=0)
        return chromosome_length(seq, assembly, far)

    length_map = {seq: seq_length(seq) for seq in seq_order}
    global_max = max(length_map.values(), default=1) or 1

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), dpi=args.dpi)
    right_limit = 1.34 if show_labels else 1.06
    ax.set_xlim(-0.13, right_limit)
    ax.set_ylim(0, num_chroms)
    ax.axis("off")
    if args.title:
        fig.suptitle(args.title, fontsize=args.font_size + 4, fontweight="bold")

    # Ideogram thickness adapts to the available per-row height so it never
    # overflows neighbouring rows regardless of chromosome count / figure size.
    row_pts = (args.fig_height / num_chroms) * 72.0
    body_lw = max(5.0, min(24.0, row_pts * 0.40))
    half_h = (body_lw / 2.0) * num_chroms / (72.0 * args.fig_height)
    tick_lw = max(0.5, min(2.6, body_lw * 0.11))
    cen_ms = max(3.0, body_lw * 0.34)

    for idx, seq in enumerate(seq_order):
        seq_data = grouped[seq]
        y = num_chroms - idx - 0.5
        length = length_map[seq]
        width = 1.0 if args.scale_mode == "relative" else length / global_max

        ax.text(-0.016, y, format_chromosome_label(seq), ha="right", va="center",
                fontsize=args.font_size, fontweight="bold")

        num = chromosome_number(seq)
        cen_frac = CENTROMERE_FRACTION.get(num) if num is not None else None
        cx = width * cen_frac if (cen_frac is not None and 0.0 < cen_frac < 1.0) else None
        gap = min(width * 0.012, 0.008)
        arms = [(0.0, cx - gap), (cx + gap, width)] if cx is not None else [(0.0, width)]

        # Ideogram body: dark outline beneath a lighter fill, round caps = telomeres.
        for x0, x1 in arms:
            if x1 <= x0:
                continue
            ax.plot([x0, x1], [y, y], lw=body_lw + 2.2, solid_capstyle="round",
                    color="#3a3a3a", zorder=2, alpha=args.bg_alpha)
            ax.plot([x0, x1], [y, y], lw=body_lw, solid_capstyle="round",
                    color=args.bg_color, zorder=2.1, alpha=args.bg_alpha)

        # Hits.
        for d_idx in range(dataset_count):
            segments = seq_data.get(d_idx, [])
            if not segments:
                continue
            ds_color = dataset_meta[d_idx].color
            ds_opacity = dataset_meta[d_idx].opacity
            if args.density:
                bins = max(60, int(width * 280))
                densities = compute_density_map(length, segments, bins=bins)
                for b, value in enumerate(densities):
                    if value <= 0:
                        continue
                    bx = width * (b + 0.5) / bins
                    op = max(0.05, min(1.0, ds_opacity * value))
                    ax.plot([bx, bx], [y - half_h, y + half_h], color=ds_color,
                            lw=max(tick_lw, width / bins * 72 * args.fig_width),
                            alpha=op, zorder=3, solid_capstyle="butt")
                continue
            for seg in segments:
                if classification_mode:
                    color = category_colors.get(seg.category, fallback_color) if seg.category else fallback_color
                else:
                    color = ds_color
                pos = (seg.start + seg.end) / 2.0
                x = max(0.0, min(width, width * pos / length))
                ax.plot([x, x], [y - half_h, y + half_h], color=color,
                        lw=tick_lw, alpha=max(0.1, min(1.0, seg.opacity)),
                        zorder=3, solid_capstyle="butt")
                if args.annotate_column and seg.annotation:
                    ax.text(x, y + half_h + 0.04, str(seg.annotation), ha="center",
                            va="bottom", rotation=90, fontsize=max(5, args.font_size - 5),
                            color="#222222", zorder=5)

        # Centromere marker on top of the body.
        if cx is not None:
            ax.plot([cx], [y], marker="o", ms=cen_ms, color="#9b1320",
                    mec="#3a0008", mew=0.5, zorder=4)

        # Gene names hit on this chromosome, listed to the right.
        if show_labels:
            names, overflow = _unique_genes_for_seq(seq_data)
            if names:
                text = ", ".join(names)
                if overflow:
                    text += f"  +{overflow} more"
                ax.text(width + 0.015, y, text, ha="left", va="center",
                        fontsize=max(6, args.font_size - 2), color="#222222")

    if legend_entries:
        handles = [patches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#333", lw=0.5)
                   for _, color in legend_entries]
        labels = [label for label, _ in legend_entries]
        # Gene labels already occupy the right margin, so put the legend along the
        # bottom whenever they are shown to avoid overlapping them.
        if args.legend_pos == "bottom" or show_labels:
            ax.legend(handles, labels, title=legend_title, loc="upper center",
                      bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=min(len(labels), 5))
        else:
            ax.legend(handles, labels, title=legend_title, loc="upper left",
                      bbox_to_anchor=(1.005, 1.0), frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=args.output_format, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render publication-ready chromosome tracks.")
    parser.add_argument("inputs", nargs="+", help="TSV/CSV files to visualize.")
    parser.add_argument("--input-kind", dest="input_kinds", action="append")
    parser.add_argument("--input-label", dest="input_labels", action="append")
    parser.add_argument("--input-opacity", dest="input_opacities", action="append", type=float)
    parser.add_argument("--input-style", dest="input_styles", action="append")
    parser.add_argument("--input-color", dest="input_colors", action="append")
    parser.add_argument("--output", default="output_/chromosome_tracks.png")
    parser.add_argument("--output-format", default="png", choices=["png", "pdf", "svg", "tiff", "eps"])
    parser.add_argument("--fig-width", type=float, default=12.0)
    parser.add_argument("--fig-height", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--palette", default="default")
    parser.add_argument("--label-mode", choices=LABEL_MODE_CHOICES, default="gene+coords")
    parser.add_argument("--color-only", action="store_true")
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--annotate-column")
    parser.add_argument("--sequence-id", action="append", default=[])
    parser.add_argument("--region", action="append", metavar="SEQ:START-END", default=[])
    parser.add_argument("--gene-filter", action="append", default=[])
    parser.add_argument("--title")
    parser.add_argument("--scale-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--bg-color", default="#f8f8f8")
    parser.add_argument("--bg-alpha", type=float, default=1.0)
    parser.add_argument("--legend-pos", choices=["right", "bottom"], default="right")
    parser.add_argument(
        "--orientation", choices=["horizontal", "vertical"], default="horizontal",
        help="horizontal: human-chromosome ideograms with centromeres (default); "
             "vertical: legacy stacked bar tracks.",
    )
    parser.add_argument(
        "--assembly", choices=["auto", "grch38", "t2t"], default="auto",
        help="Reference assembly for true chromosome lengths (default: auto-detect).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    input_paths = [Path(path) for path in args.inputs]
    if not input_paths: parser.error("At least one input file must be provided.")
    for path in input_paths:
        if not path.is_file(): parser.error(f"Input file '{path}' does not exist.")
        
    dataset_count = len(input_paths)
    
    # Handle Palettes
    palette_colors = list(DATASET_COLORS)
    if args.palette and args.palette.lower() != "default":
        try:
            cmap = plt.get_cmap(args.palette)
            palette_colors = [mcolors.to_hex(cmap(i / max(1, dataset_count - 1))) for i in range(dataset_count)]
        except ValueError:
            pass # fallback to default
            
    dataset_colors = []
    input_colors = args.input_colors or []
    for idx in range(dataset_count):
        if idx < len(input_colors) and input_colors[idx]:
            dataset_colors.append(input_colors[idx])
        else:
            dataset_colors.append(palette_colors[idx % len(palette_colors)])
    
    dataset_labels = []
    input_labels = args.input_labels or []
    for idx, path in enumerate(input_paths):
        if idx < len(input_labels) and input_labels[idx]: dataset_labels.append(input_labels[idx])
        else: dataset_labels.append(path.stem)
        
    kind_list = args.input_kinds or []
    def select_kind(idx: int) -> str:
        if not kind_list: return "auto"
        if len(kind_list) == 1: return kind_list[0]
        if idx < len(kind_list) and kind_list[idx]: return kind_list[idx]
        return "auto"

    annotation_field = (args.annotate_column or "").strip() or None
    def select_setting(options: Sequence | None, idx: int):
        if not options: return None
        if len(options) == 1: return options[0]
        if idx < len(options): return options[idx]
        return options[-1]

    def select_opacity(idx: int) -> float:
        v = select_setting(args.input_opacities, idx)
        return max(0.05, min(1.0, float(v))) if v is not None else 0.75

    def select_style(idx: int) -> str:
        return normalize_line_style(select_setting(args.input_styles, idx))

    try: region_filters = parse_region_filters(args.region)
    except ValueError as exc: parser.error(str(exc))
    
    seq_filters = parse_sequence_filters(args.sequence_id)
    gene_filters = parse_gene_filters(args.gene_filter)
    region_filters = region_filters or None
    seq_filters = seq_filters or None
    gene_filters = gene_filters or None

    dataset_meta = [
        DatasetMeta(name=dataset_labels[idx], color=dataset_colors[idx], opacity=select_opacity(idx), style=select_style(idx))
        for idx in range(dataset_count)
    ]
    
    all_segments: List[RegionSegment] = []
    label_mode = "none" if args.color_only else (args.label_mode or "gene+coords")
    if label_mode not in LABEL_MODE_CHOICES: label_mode = "gene+coords"
    
    for idx, path in enumerate(input_paths):
        for segment in iter_file_segments(
            path, idx, dataset_labels[idx], dataset_colors[idx], label_mode != "none",
            label_mode, select_kind(idx), dataset_meta[idx].opacity, dataset_meta[idx].style,
            annotation_field, seq_filters, region_filters, gene_filters
        ):
            all_segments.append(segment)
            
    if not all_segments and not seq_filters: parser.error("No valid regions found and no sequence filters provided.")
    
    classification_mode = dataset_count == 1 and any(segment.category for segment in all_segments)
    if dataset_count == 1 and not classification_mode: 
        if not input_colors: dataset_meta[0].color = DEFAULT_SINGLE_COLOR
    
    grouped = group_segments(all_segments, dataset_count, seq_filters)
    used_categories = sorted({segment.category for segment in all_segments if segment.category})
    
    if classification_mode:
        legend_entries = [(CLASS_DISPLAY.get(c, c), CLASS_COLOR_MAP.get(c, DEFAULT_SINGLE_COLOR)) for c in used_categories]
        legend_title = "Functional class"
    else:
        legend_entries = [(meta.name, meta.color) for meta in dataset_meta]
        legend_title = "Data source"

    if args.orientation == "horizontal":
        assembly = args.assembly
        if assembly == "auto":
            assembly = detect_assembly(grouped.keys())
        render_karyogram(
            grouped, dataset_meta, Path(args.output), args, classification_mode,
            CLASS_COLOR_MAP, legend_entries, legend_title, DEFAULT_SINGLE_COLOR, assembly,
        )
    else:
        render_plot(
            grouped, dataset_meta, Path(args.output), args, classification_mode,
            CLASS_COLOR_MAP, legend_entries, legend_title, DEFAULT_SINGLE_COLOR
        )

    print(f"Wrote visualization to {args.output}")
    return 0


if __name__ == "__main__":
    main()

