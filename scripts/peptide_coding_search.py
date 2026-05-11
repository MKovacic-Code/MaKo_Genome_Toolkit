#!/usr/bin/env python3
"""Scan genomic FASTA files for peptide motifs across selectable reading frames."""

from __future__ import annotations

import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import FrozenSet, Iterable, Iterator, List, Sequence, Tuple


CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

CODON_WEIGHTS = {
    "TTT": 0.58,
    "TTC": 0.92,
    "TTA": 0.14,
    "TTG": 0.13,
    "CTT": 0.13,
    "CTC": 0.4,
    "CTA": 0.14,
    "CTG": 1.0,
    "ATT": 0.46,
    "ATC": 0.91,
    "ATA": 0.17,
    "ATG": 1.0,
    "GTT": 0.18,
    "GTC": 0.47,
    "GTA": 0.11,
    "GTG": 1.0,
    "TCT": 0.19,
    "TCC": 0.56,
    "TCA": 0.15,
    "TCG": 0.13,
    "AGT": 0.15,
    "AGC": 0.68,
    "CCT": 0.31,
    "CCC": 0.34,
    "CCA": 0.3,
    "CCG": 1.0,
    "ACT": 0.28,
    "ACC": 0.89,
    "ACA": 0.31,
    "ACG": 0.4,
    "GCT": 0.27,
    "GCC": 1.0,
    "GCA": 0.34,
    "GCG": 0.54,
    "TAT": 0.45,
    "TAC": 0.55,
    "CAT": 0.39,
    "CAC": 0.61,
    "CAA": 0.29,
    "CAG": 0.71,
    "AAT": 0.47,
    "AAC": 0.53,
    "AAA": 0.42,
    "AAG": 0.58,
    "GAT": 0.46,
    "GAC": 0.54,
    "GAA": 0.43,
    "GAG": 0.57,
    "TGT": 0.38,
    "TGC": 0.62,
    "TGG": 1.0,
    "CGT": 0.4,
    "CGC": 1.0,
    "CGA": 0.11,
    "CGG": 0.21,
    "AGA": 0.2,
    "AGG": 0.2,
    "GGT": 0.33,
    "GGC": 1.0,
    "GGA": 0.5,
    "GGG": 0.55,
    "TAA": 0.0,
    "TAG": 0.0,
    "TGA": 0.0,
}

AA_IUPAC = {
    "A": "A",
    "B": "DN",
    "C": "C",
    "D": "D",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "J": "IL",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "P": "P",
    "Q": "Q",
    "R": "R",
    "S": "S",
    "T": "T",
    "V": "V",
    "W": "W",
    "Y": "Y",
    "Z": "EQ",
    "X": "ACDEFGHIKLMNPQRSTVWY",
    "*": "*",
}

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


@dataclass(frozen=True)
class MotifPattern:
    raw: str
    profile: Tuple[FrozenSet[str], ...]
    length: int

@dataclass(frozen=True)
class ContentLimit:
    bases: frozenset[str]
    min_pct: float
    max_pct: float

@dataclass(frozen=True)
class RepeatLimit:
    pattern_regex: re.Pattern[str]
    max_repeats: int


_WORKER_CONFIG: dict | None = None


def build_motif_pattern(pattern: str) -> MotifPattern:
    normalized = pattern.strip()
    if not normalized:
        raise ValueError("Motif patterns cannot be empty.")
    profile: list[FrozenSet[str]] = []
    for char in normalized.upper():
        allowed = AA_IUPAC.get(char)
        if not allowed:
            raise ValueError(f"Unsupported IUPAC amino acid code '{char}' in '{pattern}'")
        profile.append(frozenset(allowed))
    return MotifPattern(raw=normalized, profile=tuple(profile), length=len(profile))


def build_motif_patterns(patterns: Sequence[str]) -> list[MotifPattern]:
    compiled: list[MotifPattern] = []
    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        compiled.append(build_motif_pattern(pattern))
    return compiled


def match_motif_in_window(window: str, motif: MotifPattern, max_mismatches: int) -> int | None:
    """Return the minimal mismatches for motif inside the window or None if no match."""
    if motif.length == 0 or motif.length > len(window):
        return None
    upper_window = window.upper()
    remaining = len(upper_window) - motif.length + 1
    best: int | None = None
    for start in range(remaining):
        mismatches = 0
        for idx, allowed in enumerate(motif.profile):
            residue = upper_window[start + idx]
            if residue not in allowed:
                mismatches += 1
                if mismatches > max_mismatches:
                    break
        else:
            if best is None or mismatches < best:
                best = mismatches
                if best == 0:
                    return 0
    return best


def calc_codon_efficiency(sequence: str) -> float:
    total = 0.0
    count = 0
    for idx in range(0, len(sequence) - 2, 3):
        codon = sequence[idx : idx + 3].upper()
        weight = CODON_WEIGHTS.get(codon)
        if weight is not None:
            total += weight
            count += 1
    return total / count if count else 0.0


def filter_non_overlapping_hits(hits: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    sorted_hits = sorted(
        hits,
        key=lambda hit: (
            int(hit["nt_start"]),
            int(hit.get("mismatches", 0)),  # type: ignore[arg-type]
            int(hit["nt_end"]),
        ),
    )
    kept: list[dict[str, object]] = []
    last_end = -1
    for hit in sorted_hits:
        start = int(hit["nt_start"])
        if start > last_end:
            kept.append(hit)
            last_end = int(hit["nt_end"])
    return kept


def _init_worker(config: dict) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config

def parse_amino_content(specs: list[str]) -> list[ContentLimit]:
    limits = []
    for spec in specs:
        parts = spec.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid amino-content format '{spec}'. Expected RESIDUES:MIN:MAX.")
        bases_str, min_val, max_val = parts
        allowed = set()
        for char in bases_str.upper():
            if char not in AA_IUPAC:
                raise ValueError(f"Invalid IUPAC code '{char}' in amino-content '{spec}'.")
            allowed.update(AA_IUPAC[char])
        limits.append(ContentLimit(frozenset(allowed), float(min_val), float(max_val)))
    return limits

def parse_max_repeats(specs: list[str]) -> list[RepeatLimit]:
    limits = []
    for spec in specs:
        parts = spec.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid max-repeat format '{spec}'. Expected PATTERN:MAX.")
        pat_str, max_val = parts
        
        # build the base regex string for the pattern
        base_parts = []
        for char in pat_str.upper():
            allowed = AA_IUPAC.get(char)
            if not allowed:
                raise ValueError(f"Unsupported IUPAC amino acid code '{char}' in '{pat_str}'")
            if len(allowed) == 1:
                base_parts.append(allowed)
            else:
                base_parts.append(f"[{allowed}]")
        base_regex_str = "".join(base_parts)
        
        # Compile a regex that matches max_val + 1 repeats
        repeat_regex_str = f"(?:{base_regex_str}){{{int(max_val) + 1},}}"
        compiled_regex = re.compile(repeat_regex_str, re.IGNORECASE)
        
        limits.append(RepeatLimit(compiled_regex, int(max_val)))
    return limits


def _process_sequence_worker(payload: Tuple[str, str]) -> Tuple[str, list[dict[str, object]]]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker configuration not initialized.")
    seq_id, sequence = payload
    hits = find_hits_for_sequence(
        seq_id,
        sequence,
        frames=_WORKER_CONFIG["frames"],
        window=_WORKER_CONFIG["window"],
        step=_WORKER_CONFIG["step"],
        motif_patterns=_WORKER_CONFIG["motif_patterns"],
        exclude_patterns=_WORKER_CONFIG["exclude_patterns"],
        max_mismatches=_WORKER_CONFIG["max_mismatches"],
        allow_overlap=_WORKER_CONFIG["allow_overlap"],
    )
    return seq_id, hits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search peptide motifs across genomic reading frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fasta", help="Genome FASTA file (nucleotide sequences).")
    parser.add_argument("--window", type=int, default=15, help="Peptide window length (amino acids).")
    parser.add_argument("--step", type=int, default=5, help="Sliding window step size in amino acids.")
    parser.add_argument(
        "--motif",
        action="append",
        default=[],
        help="IUPAC amino acid motif to search for (can be repeated).",
    )
    parser.add_argument(
        "--exclude-motif",
        action="append",
        default=[],
        help="Motifs to exclude (windows containing any are skipped).",
    )
    parser.add_argument(
        "--frame",
        action="append",
        default=None,
        help="Reading frame to analyze (choices: +0,+1,+2,-0,-1,-2). Repeat for multiples.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=0,
        help="Maximum amino acid mismatches allowed when checking motifs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 uses all detected CPU cores).",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Include overlapping windows (default behavior filters overlaps).",
    )
    parser.add_argument(
        "--output-prefix",
        default="peptide_coding_hits",
        help="Output filename prefix (TSV).",
    )
    parser.add_argument(
        "--output-name",
        default="default",
        help="Output folder suffix (files go to output_<name>).",
    )
    parser.add_argument(
        "--amino-content",
        action="append",
        default=[],
        help="Limit the percent content of amino acids (RESIDUES:MIN:MAX). Example: ST:20:50 (Ser/Thr 20-50%%).",
    )
    parser.add_argument(
        "--max-repeat",
        action="append",
        default=[],
        help="Maximum allowed number of consecutive occurrences for a pattern. Example: GP:4.",
    )
    parser.add_argument(
        "--strand",
        action="append",
        choices=["forward", "reverse", "combined", "both"],
        help="Strand mode. If 'combined', attempts to combine forward and reverse frames.",
    )
    parser.add_argument(
        "--combined-forward-len",
        type=int,
        help="Forward-frame length to include in combined windows (default: --window).",
    )
    parser.add_argument(
        "--combined-reverse-len",
        type=int,
        help="Reverse-frame length to include in combined windows (default: --window).",
    )
    parser.add_argument(
        "--combined-overlap",
        type=int,
        default=0,
        help="Overlap between forward and reverse pieces in combined windows.",
    )
    return parser.parse_args(argv)


def parse_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    name = None
    seq_chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq_chunks).upper()
                name = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if name is not None:
            yield name, "".join(seq_chunks).upper()


def translate_sequence(seq: str) -> str:
    aa_chars = []
    for idx in range(0, len(seq) - 2, 3):
        codon = seq[idx : idx + 3]
        aa_chars.append(CODON_TABLE.get(codon, "X"))
    return "".join(aa_chars)


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def build_regex(pattern: str) -> re.Pattern[str]:
    parts = []
    for char in pattern.upper():
        allowed = AA_IUPAC.get(char)
        if not allowed:
            raise ValueError(f"Unsupported IUPAC amino acid code '{char}' in '{pattern}'")
        if len(allowed) == 1:
            parts.append(allowed)
        else:
            parts.append(f"[{allowed}]")
    return re.compile("".join(parts), re.IGNORECASE)


def compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    items: list[re.Pattern[str]] = []
    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        items.append(build_regex(pattern))
    return items


def ensure_output_dir(name: str) -> Path:
    suffix = name.strip() or "default"
    path = Path(f"output_{suffix}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_frames(selection: Sequence[str] | None) -> list[Tuple[str, int]]:
    default_frames = ["+0", "+1", "+2"]
    frames = selection or default_frames
    normalized: list[Tuple[str, int]] = []
    valid = {"+0", "+1", "+2", "-0", "-1", "-2"}
    for frame in frames:
        frame = frame.strip()
        if not frame:
            continue
        if frame not in valid:
            raise ValueError(f"Invalid frame identifier '{frame}', expected one of {sorted(valid)}.")
        strand = frame[0]
        offset = int(frame[1])
        normalized.append((strand, offset))
    if not normalized:
        raise ValueError("At least one reading frame must be selected.")
    return normalized


def scan_combined_peptide_windows(
    seq_id: str,
    sequence: str,
    offset: int,
    forward_len: int,
    reverse_len: int,
    overlap: int,
    step: int,
    motif_patterns: Sequence[MotifPattern],
    exclude_patterns: Sequence[re.Pattern[str]],
    content_limits: Sequence[ContentLimit],
    repeat_limits: Sequence[RepeatLimit],
    max_mismatches: int,
) -> list[dict[str, object]]:
    seq_len = len(sequence)
    combined_span = forward_len + reverse_len - overlap
    if combined_span <= 0:
        return []
    
    frame_seq_fwd = sequence[offset:]
    frame_seq_rev = reverse_complement(sequence)[offset:]
    
    aa_seq_fwd = translate_sequence(frame_seq_fwd)
    aa_seq_rev = translate_sequence(frame_seq_rev)
    
    if not aa_seq_fwd or not aa_seq_rev:
        return []
        
    aa_limit = len(aa_seq_fwd) - combined_span + 1
    if aa_limit <= 0:
        return []
        
    hits = []
    f_nt_len = forward_len * 3
    r_nt_len = reverse_len * 3
    
    for aa_start in range(0, aa_limit, max(1, step)):
        fwd_part = aa_seq_fwd[aa_start : aa_start + forward_len]
        
        # Calculate reverse coords
        rev_end = len(aa_seq_rev) - aa_start
        rev_start = rev_end - reverse_len
        
        rev_part = aa_seq_rev[rev_start : rev_end]
        aa_window = fwd_part + rev_part
        
        if exclude_patterns and any(p.search(aa_window) for p in exclude_patterns):
            continue
            
        skip_window = False
        if content_limits:
            for limit in content_limits:
                match_count = sum(1 for aa in aa_window if aa in limit.bases)
                pct = (match_count / len(aa_window)) * 100
                if not (limit.min_pct <= pct <= limit.max_pct):
                    skip_window = True
                    break
        if skip_window: continue
        
        if repeat_limits:
            for limit in repeat_limits:
                if limit.pattern_regex.search(aa_window):
                    skip_window = True
                    break
        if skip_window: continue

        match_records = []
        if motif_patterns:
            for motif in motif_patterns:
                mismatch = match_motif_in_window(aa_window, motif, max_mismatches)
                if mismatch is not None:
                    match_records.append((motif.raw, mismatch))
            if not match_records:
                continue
        else:
            match_records.append(("ANY", 0))
            
        f_aa_offset = aa_start * 3
        nt_start_fwd = offset + f_aa_offset
        nt_end_fwd = nt_start_fwd + f_nt_len - 1
        
        r_aa_offset = rev_start * 3
        rc_pos = offset + r_aa_offset
        nt_end_rev = seq_len - rc_pos - 1
        nt_start_rev = nt_end_rev - r_nt_len + 1
        
        codon_eff = 0.0 # Placeholder for combined
        
        for motif_repr, mismatches in match_records:
            hits.append(
                {
                    "sequence_id": seq_id,
                    "strand": "combined",
                    "frame": f"comb{offset}",
                    "aa_start": aa_start + 1,
                    "aa_end": aa_start + combined_span,
                    "nt_start": nt_start_fwd + 1,
                    "nt_end": nt_start_rev + 1, # using nt_end for the second piece start
                    "matched_motif": motif_repr,
                    "peptide_window": aa_window,
                    "mismatches": mismatches,
                    "codon_efficiency": codon_eff,
                }
            )
    return hits

def find_hits_for_sequence(
    seq_id: str,
    sequence: str,
    frames: list[Tuple[str, int]],
    window: int,
    step: int,
    motif_patterns: Sequence[MotifPattern],
    exclude_patterns: Sequence[re.Pattern[str]],
    content_limits: Sequence[ContentLimit],
    repeat_limits: Sequence[RepeatLimit],
    max_mismatches: int,
    allow_overlap: bool,
    strand_modes: list[str] = None,
    combined_forward_len: int = 0,
    combined_reverse_len: int = 0,
    combined_overlap: int = 0,
) -> list[dict[str, object]]:
    window_nt = window * 3
    seq_len = len(sequence)
    all_hits: list[dict[str, object]] = []
    
    if "combined" in (strand_modes or []):
        offsets = {offset for _, offset in frames}
        for offset in offsets:
            all_hits.extend(scan_combined_peptide_windows(
                seq_id, sequence, offset, combined_forward_len, combined_reverse_len, combined_overlap,
                step, motif_patterns, exclude_patterns, content_limits, repeat_limits, max_mismatches
            ))
            
    for strand, offset in frames:
        if strand_modes and strand not in strand_modes and "combined" not in strand_modes and "both" not in strand_modes:
            continue
        # Skip normal frames if ONLY combined was requested
        if strand_modes and "combined" in strand_modes and strand not in strand_modes and "both" not in strand_modes:
            continue

        frame_hits: list[dict[str, object]] = []
        if strand == "+":
            frame_seq = sequence[offset:]
            strand_label = "+"
        else:
            frame_seq = reverse_complement(sequence)[offset:]
            strand_label = "-"
        aa_seq = translate_sequence(frame_seq)
        if not aa_seq:
            continue
        aa_len = len(aa_seq)
        for aa_start in range(0, aa_len - window + 1, max(1, step)):
            aa_window = aa_seq[aa_start : aa_start + window]
            if exclude_patterns and any(pattern.search(aa_window) for pattern in exclude_patterns):
                continue
            
            skip_window = False
            if content_limits:
                for limit in content_limits:
                    match_count = sum(1 for aa in aa_window if aa in limit.bases)
                    pct = (match_count / window) * 100
                    if not (limit.min_pct <= pct <= limit.max_pct):
                        skip_window = True
                        break
            if skip_window:
                continue

            if repeat_limits:
                for limit in repeat_limits:
                    if limit.pattern_regex.search(aa_window):
                        skip_window = True
                        break
            if skip_window:
                continue
            match_records: list[Tuple[str, int]] = []
            if motif_patterns:
                for motif in motif_patterns:
                    mismatch = match_motif_in_window(aa_window, motif, max_mismatches)
                    if mismatch is not None:
                        match_records.append((motif.raw, mismatch))
                if not match_records:
                    continue
            else:
                match_records.append(("ANY", 0))
            aa_offset = aa_start * 3
            coding_nt = frame_seq[aa_offset : aa_offset + window_nt]
            if len(coding_nt) < window_nt:
                continue
            codon_eff = round(calc_codon_efficiency(coding_nt), 4)
            if strand_label == "+":
                nt_start = offset + aa_offset
                nt_end = nt_start + window_nt - 1
            else:
                rc_pos = offset + aa_offset
                nt_end = seq_len - rc_pos - 1
                nt_start = nt_end - window_nt + 1
            for motif_repr, mismatches in match_records:
                frame_hits.append(
                    {
                        "sequence_id": seq_id,
                        "strand": strand_label,
                        "frame": f"{strand_label}{offset}",
                        "aa_start": aa_start + 1,
                        "aa_end": aa_start + window,
                        "nt_start": nt_start + 1,
                        "nt_end": nt_end + 1,
                        "matched_motif": motif_repr,
                        "peptide_window": aa_window,
                        "mismatches": mismatches,
                        "codon_efficiency": codon_eff,
                    }
                )
        if frame_hits:
            if not allow_overlap:
                frame_hits = filter_non_overlapping_hits(frame_hits)
            all_hits.extend(frame_hits)
    return all_hits


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    fasta_path = Path(args.fasta)
    if not fasta_path.is_file():
        raise SystemExit(f"FASTA file '{fasta_path}' does not exist.")
    try:
        frames = iter_frames(args.frame)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.window <= 0 or args.step <= 0:
        raise SystemExit("Window and step must be positive integers.")
    if args.max_mismatches < 0:
        raise SystemExit("Maximum mismatches must be zero or a positive integer.")
    if args.workers < 0:
        raise SystemExit("Worker count must be zero or a positive integer.")
    worker_count = args.workers if args.workers and args.workers > 0 else os.cpu_count() or 1
    worker_count = max(1, worker_count)
    try:
        motif_patterns = build_motif_patterns(args.motif)
        content_limits = parse_amino_content(args.amino_content)
        repeat_limits = parse_max_repeats(args.max_repeat)
    except ValueError as exc:
        raise SystemExit(str(exc))
    exclude_patterns = compile_patterns(args.exclude_motif)
    output_dir = ensure_output_dir(args.output_name)
    output_path = output_dir / f"{args.output_prefix.strip() or 'peptide_coding_hits'}.tsv"
    total_hits = 0
    config = {
        "frames": frames,
        "window": args.window,
        "step": args.step,
        "motif_patterns": motif_patterns,
        "exclude_patterns": exclude_patterns,
        "content_limits": content_limits,
        "repeat_limits": repeat_limits,
        "max_mismatches": args.max_mismatches,
        "allow_overlap": args.allow_overlap,
        "strand_modes": args.strand or [],
        "combined_forward_len": args.combined_forward_len or args.window,
        "combined_reverse_len": args.combined_reverse_len or args.window,
        "combined_overlap": args.combined_overlap,
    }
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence_id",
                "strand",
                "frame",
                "aa_start",
                "aa_end",
                "nt_start",
                "nt_end",
                "matched_motif",
                "peptide_window",
                "mismatches",
                "codon_efficiency",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        if worker_count == 1:
            for seq_id, seq in parse_fasta(fasta_path):
                hits = find_hits_for_sequence(seq_id, seq, **config)
                if not hits:
                    continue
                for hit in hits:
                    writer.writerow(hit)
                total_hits += len(hits)
        else:
            payloads = parse_fasta(fasta_path)
            with ProcessPoolExecutor(
                max_workers=worker_count, initializer=_init_worker, initargs=(config,)
            ) as executor:
                for seq_id, hits in executor.map(_process_sequence_worker, payloads):
                    if not hits:
                        continue
                    for hit in hits:
                        writer.writerow(hit)
                    total_hits += len(hits)
    print(f"Wrote {total_hits} peptide motif hits to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
