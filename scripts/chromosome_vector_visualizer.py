#!/usr/bin/env python3
"""Render chromosome-style SVG tracks for genome scanner outputs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DATASET_COLORS = [
    "#ef476f",
    "#118ab2",
    "#06d6a0",
    "#ffd166",
    "#073b4c",
    "#b5179e",
    "#4895ef",
    "#ffb703",
    "#219ebc",
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
    "lncrna": "lnc_rna",
    "lnc_rna": "lnc_rna",
    "mrna": "mrna",
    "messenger_rna": "mrna",
    "mirna": "mirna",
    "micro_rna": "mirna",
    "pseudogene": "pseudogene",
    "pseudo": "pseudogene",
    "five_prime_utr": "five_prime_utr",
    "utr5": "five_prime_utr",
    "5utr": "five_prime_utr",
    "five_primeutr": "five_prime_utr",
    "three_prime_utr": "three_prime_utr",
    "utr3": "three_prime_utr",
    "3utr": "three_prime_utr",
    "three_primeutr": "three_prime_utr",
}

ANNOTATION_FIELDS = ["region_types", "region_annotations", "region_type"]
DEFAULT_SINGLE_COLOR = "#ff7f50"

SEQUENCE_FIELDS = [
    "sequence_id",
    "chrom",
    "chromosome",
    "chrom_or_seq_id",
    "seq_id",
    "contig",
]

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


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "\t" in line:
                return "\t"
            if "," in line:
                return ","
    return "\t"


def normalize_seq_id(value: str | None) -> str:
    if not value:
        return "unknown"
    return value.strip() or "unknown"


def format_chromosome_label(seq_id: str) -> str:
    token = (seq_id or "").strip()
    if not token:
        return "Chromosome"
    normalized = token.split()[0]

    def map_number(text: str) -> str | None:
        try:
            number = int(text)
        except ValueError:
            return None
        symbol = CHROMOSOME_NUMBER_LABELS.get(number)
        if symbol:
            return f"Chromosome {symbol}"
        if number > 0:
            return f"Chromosome {number}"
        return None

    match = re.match(r"NC_0*(\d+)", normalized, re.IGNORECASE)
    if match:
        mapped = map_number(match.group(1))
        if mapped:
            return mapped
    if normalized.lower().startswith("chr"):
        suffix = normalized[3:].strip()
        if not suffix:
            return "Chromosome"
        if suffix.upper() in ("X", "Y"):
            return f"Chromosome {suffix.upper()}"
        mapped = map_number(suffix)
        if mapped:
            return mapped
    mapped = map_number(normalized)
    if mapped:
        return mapped
    return token


def infer_coordinate_columns(headers: Sequence[str], kind: str) -> Tuple[str, str]:
    lowered = {h.lower(): h for h in headers}
    if kind != "auto":
        start, end = KIND_COLUMN_CHOICES[kind]
        if start in headers and end in headers:
            return start, end
        if start.lower() in lowered and end.lower() in lowered:
            return lowered[start.lower()], lowered[end.lower()]
        raise ValueError(
            f"Input kind '{kind}' expects columns '{start}'/'{end}' but they were not found."
        )
    for start, end in FALLBACK_COLUMN_ORDER:
        if start in headers and end in headers:
            return start, end
        if start.lower() in lowered and end.lower() in lowered:
            return lowered[start.lower()], lowered[end.lower()]
    raise ValueError("Could not determine start/end column names automatically.")


def parse_sequence_filters(values: Sequence[str]) -> set[str]:
    result = set()
    for value in values:
        if not value:
            continue
        result.add(normalize_seq_id(value.strip()))
    return result


def parse_region_filters(values: Sequence[str]) -> Dict[str, List[Tuple[int, int]]]:
    filters: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for spec in values:
        if not spec:
            continue
        try:
            seq_part, coords = spec.split(":", 1)
            start_text, end_text = coords.replace(",", "").split("-")
            start = int(start_text)
            end = int(end_text)
        except Exception:
            raise ValueError(f"Region '{spec}' must look like SEQ:START-END.")
        if start <= 0 or end <= 0:
            raise ValueError(f"Region '{spec}' must use positive coordinates.")
        if end < start:
            start, end = end, start
        seq_id = normalize_seq_id(seq_part)
        filters.setdefault(seq_id, []).append((start, end))
    return filters


def within_region(seq_id: str, start: int, end: int, filters: Dict[str, List[Tuple[int, int]]]) -> bool:
    if not filters:
        return True
    ranges = filters.get(seq_id)
    if not ranges:
        return False
    for region_start, region_end in ranges:
        if end >= region_start and start <= region_end:
            return True
    return False


def parse_gene_filters(values: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for entry in values:
        if not entry:
            continue
        for token in re.split(r"[;,/| ]+", entry):
            token = token.strip().lower()
            if token:
                result.add(token)
    return result


def format_segment_label(
    seq_id: str,
    start: int,
    end: int,
    gene_value: str | None,
    label_mode: str,
) -> str:
    if label_mode == "none":
        return ""
    coords = f"{seq_id}:{start}-{end}"
    gene_clean = (gene_value or "").strip()
    has_gene = bool(gene_clean) and gene_clean.upper() != "NA"
    if label_mode == "gene":
        return gene_clean if has_gene else coords
    if label_mode == "gene+coords":
        return f"{gene_clean} ({coords})" if has_gene else coords
    return coords


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def normalize_line_style(value: str | None) -> str:
    if not value:
        return "solid"
    cleaned = value.strip().lower()
    if cleaned in {"solid", "dashed"}:
        return cleaned
    return "solid"


def select_field(row: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        val = row.get(name)
        if val:
            return val.strip()
    return ""


def normalize_category_token(token: str) -> str | None:
    key = token.strip().lower().replace(" ", "_")
    if not key:
        return None
    return CLASS_ALIASES.get(key)


def detect_category(row: Dict[str, str]) -> str | None:
    for field in ANNOTATION_FIELDS:
        raw = row.get(field)
        if not raw or raw.upper() == "NA":
            continue
        tokens = re.split(r"[;,/| ]+", raw)
        for token in tokens:
            normalized = normalize_category_token(token)
            if normalized:
                return normalized
    region_type = row.get("feature_type")
    if region_type:
        normalized = normalize_category_token(region_type)
        if normalized:
            return normalized
    return None


def build_segment_from_row(
    row: Dict[str, str],
    start_col: str,
    end_col: str,
    show_labels: bool,
    label_mode: str,
    dataset_index: int,
    dataset_name: str,
    dataset_color: str,
    opacity: float,
    line_style: str,
    annotation_field: str | None,
    seq_filters: set[str] | None,
    region_filters: Dict[str, List[Tuple[int, int]]] | None,
    gene_filters: set[str] | None,
) -> RegionSegment | None:
    seq_id = normalize_seq_id(select_field(row, SEQUENCE_FIELDS))
    if seq_filters and seq_id not in seq_filters:
        return None
    start = parse_int(row.get(start_col))
    end = parse_int(row.get(end_col))
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    if region_filters and not within_region(seq_id, start, end, region_filters):
        return None
    gene_value = select_field(row, GENE_FIELDS)
    if show_labels:
        label = format_segment_label(seq_id, start, end, gene_value, label_mode)
    else:
        label = ""
    if gene_filters:
        names = set()
        if gene_value:
            for token in re.split(r"[;,/| ]+", gene_value):
                token = token.strip().lower()
                if token:
                    names.add(token)
        if not names or names.isdisjoint(gene_filters):
            return None
    annotation = None
    if annotation_field:
        raw = row.get(annotation_field)
        if raw and raw.upper() != "NA":
            annotation = raw
    category = detect_category(row)
    return RegionSegment(
        seq_id=seq_id,
        start=start,
        end=end,
        label=label,
        dataset_index=dataset_index,
        dataset_name=dataset_name,
        dataset_color=dataset_color,
        category=category,
        opacity=opacity,
        line_style=line_style,
        annotation=annotation,
    )


def iter_file_segments(
    path: Path,
    dataset_index: int,
    dataset_name: str,
    dataset_color: str,
    show_labels: bool,
    label_mode: str,
    kind_hint: str,
    opacity: float,
    line_style: str,
    annotation_field: str | None,
    seq_filters: set[str] | None,
    region_filters: Dict[str, List[Tuple[int, int]]] | None,
    gene_filters: set[str] | None,
) -> Iterable[RegionSegment]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        headers = reader.fieldnames
        if not headers:
            return
        start_col, end_col = infer_coordinate_columns(headers, kind_hint)
        annotation_col = None
        if annotation_field:
            if annotation_field in headers:
                annotation_col = annotation_field
            else:
                lowered = {header.lower(): header for header in headers}
                annotation_col = lowered.get(annotation_field.lower())
        for row in reader:
            segment = build_segment_from_row(
                row,
                start_col,
                end_col,
                show_labels,
                label_mode,
                dataset_index,
                dataset_name,
                dataset_color,
                opacity,
                line_style,
                annotation_col,
                seq_filters,
                region_filters,
                gene_filters,
            )
            if segment is not None:
                yield segment


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def natural_key(value: str) -> Tuple:
    parts = re.findall(r"\d+|\D+", value)
    key: List[Tuple[int, object]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def group_segments(
    segments: Sequence[RegionSegment],
    dataset_count: int,
) -> Dict[str, Dict[int, List[RegionSegment]]]:
    def empty_bucket() -> Dict[int, List[RegionSegment]]:
        return {idx: [] for idx in range(dataset_count)}

    grouped: Dict[str, Dict[int, List[RegionSegment]]] = {}
    for segment in segments:
        bucket = grouped.setdefault(segment.seq_id, empty_bucket())
        bucket[segment.dataset_index].append(segment)
    for seq_bucket in grouped.values():
        for dataset_segments in seq_bucket.values():
            dataset_segments.sort(key=lambda seg: seg.start)
    return grouped


def render_svg(
    grouped: Dict[str, Dict[int, List[RegionSegment]]],
    dataset_meta: Sequence[DatasetMeta],
    output_path: Path,
    *,
    chrom_width: int,
    chrom_spacing: int,
    padding: int,
    label_space: int,
    max_height: int,
    color_only: bool,
    title: str | None,
    classification_mode: bool,
    category_colors: Dict[str, str],
    legend_entries: Sequence[Tuple[str, str]],
    legend_title: str | None,
    fallback_color: str,
    density_view: bool,
    annotation_mode: bool,
) -> None:
    seq_order = sorted(grouped, key=natural_key)
    length_map = {
        seq: max(
            (segment.end for dataset_segments in seq_data.values() for segment in dataset_segments),
            default=0,
        )
        for seq, seq_data in grouped.items()
    }
    max_length = max(length_map.values()) if length_map else 0
    scale = max_height / max_length if max_length else 1.0
    title_offset = 0
    if title:
        title_offset = 40
    chrom_top = padding + title_offset
    chromosome_heights = {seq: max(length * scale, 4.0) for seq, length in length_map.items()}
    max_chrom_height = max(chromosome_heights.values(), default=0)
    dataset_count = len(dataset_meta)
    base_width = max(6.0, float(chrom_width))
    if dataset_count <= 1:
        dataset_gap = 0.0
        track_width = base_width
        total_chrom_width = base_width
    else:
        dataset_gap = max(2.0, base_width * 0.03)
        total_gap = dataset_gap * (dataset_count - 1)
        available = max(base_width - total_gap, dataset_count * 4.0)
        track_width = available / dataset_count
        total_chrom_width = track_width * dataset_count + total_gap
    extra_label_space = 0 if color_only else label_space
    column_stride = total_chrom_width + chrom_spacing + extra_label_space
    legend_width = 220 if legend_entries else 0
    width = padding * 2 + max(1, len(seq_order)) * column_stride + legend_width
    height = chrom_top + max_chrom_height + padding

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    if title:
        lines.append(
            f'<text x="{width / 2:.1f}" y="{padding + 4}" text-anchor="middle" '
            f'font-size="20" font-family="Arial" fill="#1a1a1a">{escape_xml(title)}</text>'
        )

    def compute_density_map(chrom_length: int, segments: List[RegionSegment], bins: int = 160) -> List[float]:
        if chrom_length <= 0 or not segments:
            return [0.0] * bins
        density = [0.0] * bins
        for segment in segments:
            norm_start = max(0.0, (segment.start - 1) / chrom_length)
            norm_end = min(1.0, segment.end / chrom_length)
            start_bin = int(norm_start * bins)
            end_bin = int(norm_end * bins)
            for bin_idx in range(start_bin, min(end_bin + 1, bins)):
                overlap_start = max(norm_start, bin_idx / bins)
                overlap_end = min(norm_end, (bin_idx + 1) / bins)
                if overlap_end > overlap_start:
                    density[bin_idx] += overlap_end - overlap_start
        max_value = max(density) if density else 0.0
        if max_value == 0:
            return density
        return [value / max_value for value in density]

    for idx, seq in enumerate(seq_order):
        seq_data = grouped[seq]
        chrom_height = chromosome_heights[seq]
        x = padding + idx * column_stride
        y = chrom_top
        seq_label = format_chromosome_label(seq)
        lines.append(
            f'<text x="{x + total_chrom_width / 2:.1f}" y="{y - 6}" text-anchor="middle" '
            f'font-size="12" font-family="Arial" fill="#222">{escape_xml(seq_label)}</text>'
        )
        for dataset_idx in range(dataset_count):
            dx = x + dataset_idx * (track_width + dataset_gap)
            lines.append(
                f'<rect x="{dx:.2f}" y="{y:.2f}" width="{track_width:.2f}" height="{chrom_height:.2f}" '
                f'rx="{track_width / 2:.2f}" fill="#f8f8f8" stroke="#444" stroke-width="1"/>'
            )
            segments = seq_data.get(dataset_idx, [])
            if density_view and segments:
                densities = compute_density_map(length_map.get(seq, 0), segments)
                bin_count = len(densities)
                bin_height = chrom_height / max(bin_count, 1)
                for bin_idx, value in enumerate(densities):
                    if value <= 0:
                        continue
                    y0 = y + bin_idx * bin_height
                    opacity = dataset_meta[dataset_idx].opacity * min(1.0, value)
                    lines.append(
                        f'<rect x="{dx:.2f}" y="{y0:.2f}" width="{track_width:.2f}" height="{bin_height:.2f}" '
                        f'fill="{dataset_meta[dataset_idx].color}" fill-opacity="{opacity:.3f}" stroke="none"/>'
                    )
            for segment in segments:
                if classification_mode and segment.category:
                    color = category_colors.get(segment.category, fallback_color)
                elif classification_mode:
                    color = fallback_color
                else:
                    color = dataset_meta[dataset_idx].color
                start_pos = max(segment.start - 1, 0)
                end_pos = max(segment.end, segment.start)
                rect_y = y + start_pos * scale
                rect_height = max((end_pos - start_pos) * scale, 3.0)
                rect_width = max(track_width - 4, 2.0)
                rect_x = dx + (track_width - rect_width) / 2
                fill_opacity = max(0.05, min(1.0, segment.opacity))
                dash_attr = ' stroke-dasharray="4,4"' if segment.line_style == "dashed" else ""
                lines.append(
                    f'<rect x="{rect_x:.2f}" y="{rect_y:.2f}" width="{rect_width:.2f}" height="{rect_height:.2f}" '
                    f'rx="4" fill="{color}" fill-opacity="{fill_opacity:.3f}" stroke="{color}" stroke-opacity="{fill_opacity:.3f}" stroke-width="1"{dash_attr}/>'
                )
                if annotation_mode and segment.annotation:
                    ann_y = max(y + 10, rect_y - 2)
                    ann_text = escape_xml(str(segment.annotation))
                    lines.append(
                        f'<text x="{rect_x + rect_width / 2:.2f}" y="{ann_y:.2f}" font-size="9" '
                        f'font-family="Arial" fill="#111" text-anchor="middle">{ann_text}</text>'
                    )
                if color_only or not segment.label:
                    continue
                mid_y = rect_y + rect_height / 2
                label_x = x + total_chrom_width + 12
                label_text = escape_xml(segment.label)
                lines.append(
                    f'<line x1="{rect_x + rect_width}" y1="{mid_y}" '
                    f'x2="{label_x - 4}" y2="{mid_y}" stroke="{color}" stroke-width="1"/>'
                )
                lines.append(
                    f'<text x="{label_x}" y="{mid_y + 4}" font-size="11" '
                    f'font-family="Arial" fill="#111">{label_text}</text>'
                )

    if legend_entries:
        legend_x = width - legend_width + 20
        legend_y = padding + (20 if title else 0)
        if legend_title:
            lines.append(
                f'<text x="{legend_x}" y="{legend_y}" font-size="13" '
                f'font-family="Arial" fill="#111">{escape_xml(legend_title)}</text>'
            )
            legend_y += 12
        legend_y += 6
        for label, color in legend_entries:
            lines.append(
                f'<rect x="{legend_x}" y="{legend_y - 10}" width="14" height="14" '
                f'rx="2" fill="{color}" stroke="#333" stroke-width="0.5"/>'
            )
            lines.append(
                f'<text x="{legend_x + 20}" y="{legend_y + 2}" font-size="12" '
                f'font-family="Arial" fill="#111">{escape_xml(label)}</text>'
            )
            legend_y += 20

    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render narrow chromosome tracks with hit regions highlighted."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "One or more TSV/CSV files to visualize. When multiple files are provided, "
            "each is assigned a dedicated chromosome column and dataset color."
        ),
    )
    parser.add_argument(
        "--input-kind",
        dest="input_kinds",
        action="append",
        choices=["auto", *KIND_COLUMN_CHOICES.keys()],
        help=(
            "Specify the hit format for each input (repeat for multiple files). "
            "If omitted, formats are auto-detected."
        ),
    )
    parser.add_argument(
        "--input-label",
        dest="input_labels",
        action="append",
        help="Override the legend label for each input file (repeatable).",
    )
    parser.add_argument(
        "--input-opacity",
        dest="input_opacities",
        action="append",
        type=float,
        help="Opacity (0-1) per input dataset (repeatable).",
    )
    parser.add_argument(
        "--input-style",
        dest="input_styles",
        action="append",
        help="Stroke style per dataset (solid or dashed, repeatable).",
    )
    parser.add_argument(
        "--output",
        default="output_/chromosome_tracks.svg",
        help="Output SVG path (default: output_/chromosome_tracks.svg).",
    )
    parser.add_argument(
        "--chrom-width",
        type=int,
        default=120,
        help="Total chromosome width (pixels) shared proportionally across datasets (default: 120).",
    )
    parser.add_argument(
        "--chrom-spacing",
        type=int,
        default=140,
        help="Horizontal spacing between chromosomes (default: 140).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=40,
        help="Outer padding around the drawing in pixels (default: 40).",
    )
    parser.add_argument(
        "--label-space",
        type=int,
        default=180,
        help="Reserved horizontal space for gene/position labels (default: 180).",
    )
    parser.add_argument(
        "--label-mode",
        choices=LABEL_MODE_CHOICES,
        default="gene+coords",
        help="Label rendering: 'none' hides labels, 'gene' shows only gene names, 'gene+coords' appends coordinates.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1200,
        help="Maximum chromosome height in pixels (default: 1200).",
    )
    parser.add_argument(
        "--color-only",
        action="store_true",
        help="Only color hit regions on the chromosomes (alias for --label-mode none).",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Render chromosome density heatmaps instead of solid bars.",
    )
    parser.add_argument(
        "--annotate-column",
        help="Optional column name whose values will be annotated next to each hit (e.g., longest_palindrome_sequence).",
    )
    parser.add_argument(
        "--sequence-id",
        action="append",
        default=[],
        help="Restrict output to specific sequence IDs (repeatable).",
    )
    parser.add_argument(
        "--region",
        action="append",
        metavar="SEQ:START-END",
        default=[],
        help="Restrict output to genomic ranges (repeat per region, e.g., chr1:100000-200000).",
    )
    parser.add_argument(
        "--gene-filter",
        action="append",
        default=[],
        help="Only plot hits whose gene names match any of the provided values (repeatable).",
    )
    parser.add_argument(
        "--title",
        help="Optional title text displayed above the chromosomes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_paths = [Path(path) for path in args.inputs]
    if not input_paths:
        parser.error("At least one input file must be provided.")
    for path in input_paths:
        if not path.is_file():
            parser.error(f"Input file '{path}' does not exist.")
    dataset_count = len(input_paths)
    dataset_colors = [
        DATASET_COLORS[idx % len(DATASET_COLORS)] for idx in range(dataset_count)
    ]
    dataset_labels: List[str] = []
    input_labels = args.input_labels or []
    for idx, path in enumerate(input_paths):
        if idx < len(input_labels) and input_labels[idx]:
            dataset_labels.append(input_labels[idx])
        else:
            dataset_labels.append(path.stem)
    kind_list = args.input_kinds or []

    def select_kind(idx: int) -> str:
        if not kind_list:
            return "auto"
        if len(kind_list) == 1:
            return kind_list[0]
        if idx < len(kind_list) and kind_list[idx]:
            return kind_list[idx]
        return "auto"

    annotation_field = (args.annotate_column or "").strip() or None

    def select_setting(options: Sequence | None, idx: int):
        if not options:
            return None
        if len(options) == 1:
            return options[0]
        if idx < len(options):
            return options[idx]
        return options[-1]

    def clamp_opacity(value: float | None) -> float:
        if value is None:
            return 0.75
        return max(0.05, min(1.0, float(value)))

    def select_opacity(idx: int) -> float:
        return clamp_opacity(select_setting(args.input_opacities, idx))

    def select_style(idx: int) -> str:
        return normalize_line_style(select_setting(args.input_styles, idx))

    try:
        region_filters = parse_region_filters(args.region)
    except ValueError as exc:
        parser.error(str(exc))
    seq_filters = parse_sequence_filters(args.sequence_id)
    gene_filters = parse_gene_filters(args.gene_filter)
    region_filters = region_filters or None
    seq_filters = seq_filters or None
    gene_filters = gene_filters or None

    dataset_meta = [
        DatasetMeta(
            name=dataset_labels[idx],
            color=dataset_colors[idx],
            opacity=select_opacity(idx),
            style=select_style(idx),
        )
        for idx in range(dataset_count)
    ]
    all_segments: List[RegionSegment] = []
    label_mode = args.label_mode or "gene+coords"
    if args.color_only:
        label_mode = "none"
    if label_mode not in LABEL_MODE_CHOICES:
        label_mode = "gene+coords"
    show_labels = label_mode != "none"
    for idx, path in enumerate(input_paths):
        kind_hint = select_kind(idx)
        for segment in iter_file_segments(
            path,
            dataset_index=idx,
            dataset_name=dataset_labels[idx],
            dataset_color=dataset_colors[idx],
            show_labels=show_labels,
            label_mode=label_mode,
            kind_hint=kind_hint,
            opacity=dataset_meta[idx].opacity,
            line_style=dataset_meta[idx].style,
            annotation_field=annotation_field,
            seq_filters=seq_filters,
            region_filters=region_filters,
            gene_filters=gene_filters,
        ):
            all_segments.append(segment)
    if not all_segments:
        parser.error("No valid regions were found across the provided input files.")
    classification_present = any(segment.category for segment in all_segments)
    classification_mode = dataset_count == 1 and classification_present
    if dataset_count == 1 and not classification_mode:
        dataset_meta[0].color = DEFAULT_SINGLE_COLOR
    grouped = group_segments(all_segments, dataset_count)
    used_categories = sorted(
        {segment.category for segment in all_segments if segment.category}
    )
    if classification_mode:
        legend_entries = [
            (CLASS_DISPLAY.get(category, category), CLASS_COLOR_MAP.get(category, DEFAULT_SINGLE_COLOR))
            for category in used_categories
        ]
        legend_title = "Functional class"
    else:
        legend_entries = [(meta.name, meta.color) for meta in dataset_meta]
        legend_title = "Data source"
    render_svg(
        grouped,
        dataset_meta,
        Path(args.output),
        chrom_width=max(6, args.chrom_width),
        chrom_spacing=max(40, args.chrom_spacing),
        padding=max(10, args.padding),
        label_space=max(40, args.label_space),
        max_height=max(200, args.max_height),
        color_only=label_mode == "none",
        title=args.title,
        classification_mode=classification_mode,
        category_colors=CLASS_COLOR_MAP,
        legend_entries=legend_entries,
        legend_title=legend_title,
        fallback_color=DEFAULT_SINGLE_COLOR,
        density_view=args.density,
        annotation_mode=bool(annotation_field),
    )
    print(
        f"Wrote chromosome visualization for {len(all_segments)} regions "
        f"across {len(grouped)} sequence(s) from {dataset_count} input file(s) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
