#!/usr/bin/env python3
"""Genome scanner that filters FASTA windows by motif counts, base content, and repeat limits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, as_completed, wait

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Pattern, Sequence, Set, Tuple

try:
    from cython_helpers import count_iupac_motif as cython_count_iupac_motif
    from cython_helpers import longest_run as cython_longest_run
    from cython_helpers import scan_windows_fast as cython_scan_windows_fast
    from cython_helpers import find_longest_intrastrand_complement_cython
    from cython_helpers import find_longest_interstrand_complement_cython
    from cython_helpers import g4hunter_score as cython_g4hunter_score
except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional dependency
    cython_count_iupac_motif = None
    cython_longest_run = None
    cython_scan_windows_fast = None
    find_longest_intrastrand_complement_cython = None
    find_longest_interstrand_complement_cython = None
    cython_g4hunter_score = None
BASES = ("A", "C", "G", "T")
IUPAC_CODES = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "U": "T",
    "R": "[AG]",
    "Y": "[CT]",
    "S": "[GC]",
    "W": "[AT]",
    "K": "[GT]",
    "M": "[AC]",
    "B": "[CGT]",
    "D": "[AGT]",
    "H": "[ACT]",
    "V": "[ACG]",
    "N": "[ACGT]",
}

# U is included so RNA sequences reverse-complement correctly (U pairs with A).
# A maps to T here; RNA output is rendered by converting T->U at the end.
RC_MAP = str.maketrans(
    "ACGTURYKMSWBDHVNacgturykmswbdhvn",
    "TGCAAYRMKSWVHDBNtgcaayrmkswvhdbn",
)
COMPLEMENT = {a: b for a, b in zip("ACGTURYKMSWBDHVN", "TGCAAYRMKSWVHDBN")}
SCRIPT_DIR = Path(__file__).resolve().parent
PROFILE_STORE = SCRIPT_DIR.parent / "chemistry_profiles.json"
CHEMISTRY_PROFILES = {
    "relaxed": {
        "description": "Loose GC constraints and higher repeat tolerance for exploratory scans.",
        "base_content": ["G:20:80", "C:20:80"],
        "max_repeat": ["G:6", "C:6", "A:8", "T:8"],
    },
    "stringent": {
        "description": "High GC enrichment with tight repeat caps for classic G4 searches.",
        "base_content": ["G:40:70", "C:30:70"],
        "max_repeat": ["G:4", "C:4"],
        "motifs": ["GGGN{1,7}GGG:1"],
    },
}

SEQUENCE_CLASS_PREFIXES = {
    "chromosome": (">NC_0",),
    "scaffold": (">NT_", ">NW_"),
}
DEFAULT_SEQUENCE_CLASSES: Tuple[str, ...] = ("chromosome",)
CHROMOSOME_NUMBER_ALIASES = {"X": 23, "Y": 24}
CHROMOSOME_NUMBER_RANGE = tuple(range(1, 25))

MOTIF_REGEX_CACHE: Dict[str, Pattern[str]] = {}
SELF_COMP_TOKEN_CACHE: Dict[str, Tuple["PatternToken", ...]] = {}


@dataclass(frozen=True)
class PatternToken:
    """Represents an IUPAC code repeated between min and max times."""

    allowed: FrozenSet[str]
    min_repeat: int
    max_repeat: int


@dataclass(frozen=True)
class MotifConstraint:
    """Holds compiled motif information."""

    label: str
    required_count: int
    pattern: str
    regex: re.Pattern[str] | None
    tokens: Tuple[PatternToken, ...] = ()


@dataclass(frozen=True)
class ExcludeMotifConstraint:
    """Motifs that disqualify a window when present."""

    label: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class MotifSelfCompConstraint:
    """Require two sections within a motif match to be reverse-complementary.

    Two modes are supported:
    - ``whole_motif=False`` (default): check two explicitly specified sections
      defined by (sec1_start, sec1_len) and (sec2_start, sec2_len).
    - ``whole_motif=True``: scan all non-overlapping pairs of equal-length
      sub-regions (length between ``min_comp_len`` and ``max_comp_len``) within
      each motif match and pass if any pair is sufficiently reverse-complementary.

    ``min_mismatches`` / ``max_mismatches`` define an *inclusive* mismatch
    range.  Setting ``min_mismatches > 0`` means the sections must differ by at
    least that many bases (useful for imperfect / non-exact hairpin searches).
    """

    motif_pattern: str
    # --- explicit-section mode fields (ignored when whole_motif=True) ---
    sec1_start: int = 0
    sec1_len: int = 0
    sec2_start: int = 0
    sec2_len: int = 0
    # --- mismatch range (applies to both modes) ---
    max_mismatches: int = 0
    min_mismatches: int = 0
    # --- whole-motif scan mode ---
    whole_motif: bool = False
    min_comp_len: int = 1   # Now represents number of MATCHING bases
    max_comp_len: int | None = None
    excluded_motifs: Sequence[str] = field(default_factory=list)
    compiled_motif: MotifConstraint | None = None


@dataclass(frozen=True)
class RuntimeOptions:
    """Execution-time toggles for accelerated scans."""

    use_gpu: bool = False
    vectorized_base: bool = False
    gpu_kernels: bool = False


def normalize_seq_name(name: str | None) -> str:
    if not name:
        return "unknown"
    return name.split()[0]


def load_custom_profiles() -> Dict[str, Dict[str, List[str]]]:
    if not PROFILE_STORE.is_file():
        return {}
    try:
        with PROFILE_STORE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        cleaned: Dict[str, Dict[str, List[str]]] = {}
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            profile: Dict[str, List[str]] = {}
            for field in ("motifs", "motifs_forward", "motifs_reverse", "base_content", "max_repeat"):
                raw = value.get(field)
                if isinstance(raw, list):
                    profile[field] = [str(item) for item in raw if str(item).strip()]
            if profile:
                cleaned[key.lower()] = profile
        return cleaned
    except Exception:
        return {}


def available_profiles() -> Dict[str, Dict[str, List[str]]]:
    profiles: Dict[str, Dict[str, List[str]]] = {}
    for key, value in CHEMISTRY_PROFILES.items():
        entries: Dict[str, List[str]] = {}
        for field in ("motifs", "motifs_forward", "motifs_reverse", "base_content", "max_repeat"):
            if field in value:
                entries[field] = list(value[field])
        profiles[key.lower()] = entries
    profiles.update(load_custom_profiles())
    return profiles


def parse_profile_specs(profile_name: str | None, parser: argparse.ArgumentParser) -> Dict[str, List[str]]:
    if not profile_name:
        return {}
    profiles = available_profiles()
    profile = profiles.get(profile_name.lower()) or profiles.get(profile_name)
    if not profile:
        parser.error(f"Unknown chemistry profile '{profile_name}'. Available: {', '.join(sorted(profiles))}")
    return profile


def within_region(seq_id: str, start: int, end: int, region_filters: Dict[str, List[Tuple[int, int]]] | None) -> bool:
    if not region_filters:
        return True
    filters = region_filters.get(seq_id)
    if not filters:
        return False
    for region_start, region_end in filters:
        if start >= region_start and end <= region_end:
            return True
    return False


def parse_region_specs(parser: argparse.ArgumentParser, specs: Sequence[str]) -> Dict[str, List[Tuple[int, int]]]:
    filters: Dict[str, List[Tuple[int, int]]] = {}
    for raw in specs:
        try:
            seq_part, coords = raw.split(":", 1)
            start_text, end_text = coords.replace(",", "").split("-")
            start = int(start_text)
            end = int(end_text)
        except Exception:
            parser.error(f"Region '{raw}' must look like SEQ:START-END (e.g., chr1:1000-2000).")
        if start <= 0 or end <= 0:
            parser.error(f"Region '{raw}' must use positive coordinates.")
        if end < start:
            start, end = end, start
        normalized = normalize_seq_name(seq_part)
        filters.setdefault(normalized, []).append((start, end))
    return filters


def prefixes_for_classes(classes: Sequence[str] | None) -> Tuple[str, ...] | None:
    if not classes:
        return None
    prefixes: List[str] = []
    for class_name in classes:
        prefixes.extend(SEQUENCE_CLASS_PREFIXES.get(class_name, ()))
    # Preserve order but drop duplicates
    ordered_unique = list(dict.fromkeys(prefixes))
    return tuple(ordered_unique or SEQUENCE_CLASS_PREFIXES["chromosome"])


def parse_chromosome_numbers(
    parser: argparse.ArgumentParser, specs: Sequence[str] | None
) -> Set[int]:
    numbers: Set[int] = set()
    if not specs:
        return numbers
    for raw in specs:
        if raw is None:
            continue
        text = raw.strip().upper()
        if not text:
            continue
        if text.startswith("CHR"):
            text = text[3:]
        if text in CHROMOSOME_NUMBER_ALIASES:
            value = CHROMOSOME_NUMBER_ALIASES[text]
        else:
            try:
                value = int(text)
            except ValueError:
                parser.error(
                    f"Chromosome value '{raw}' must be an integer between 1 and 24 or X/Y."
                )
        if value not in CHROMOSOME_NUMBER_RANGE:
            parser.error(
                f"Chromosome value '{raw}' must be between 1 and 24 (23 = X, 24 = Y)."
            )
        numbers.add(value)
    return numbers


def chromosome_number_from_id(seq_id: str) -> int | None:
    base = normalize_seq_name(seq_id).split(".")[0]
    match = re.search(r"NC_(\d+)", base)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    if value not in CHROMOSOME_NUMBER_RANGE:
        return None
    return value


def sequence_matches_chromosome_filter(seq_id: str, allowed_numbers: Set[int] | None) -> bool:
    if not allowed_numbers:
        return True
    chrom_number = chromosome_number_from_id(seq_id)
    if chrom_number is None:
        return True
    return chrom_number in allowed_numbers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan FASTA genomes for selected chromosome or scaffold records in fixed windows and "
            "report those that satisfy motif, base composition, and repeat constraints."
        )
    )
    parser.add_argument("fasta", help=".fna FASTA file to scan")
    parser.add_argument(
        "--window",
        type=int,
        required=True,
        metavar="N",
        help="window length (nucleotides) to evaluate",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        metavar="N",
        help="slide the window by this many nucleotides (default: 1)",
    )
    parser.add_argument(
        "--motif",
        action="append",
        default=[],
        metavar="PATTERN:COUNT",
        help=(
            "IUPAC motif and the minimum number of occurrences per window. "
            "Example: --motif AGTC:3. Quantifiers like N{1,7} are supported for variable repeats. "
            "Provide multiple --motif arguments for multiple patterns."
        ),
    )
    parser.add_argument(
        "--exclude-motif",
        action="append",
        default=[],
        metavar="PATTERN",
        help=(
            "Motif pattern (IUPAC) that must NOT appear in reported windows. "
            "Excluded motifs use the same syntax as --motif but do not allow repeats/quantifiers."
        ),
    )
    parser.add_argument(
        "--base-content",
        action="append",
        default=[],
        metavar="BASE:MIN:MAX",
        help=(
            "Limit the percent content of a base within each window. "
            "Percentages are inclusive. Example: --base-content G:40:60."
        ),
    )
    parser.add_argument(
        "--max-repeat",
        action="append",
        default=[],
        metavar="BASE:MAX",
        help=(
            "Maximum allowed number of consecutive occurrences for a base. "
            "Example: --max-repeat G:4 prevents any run of five or more Gs."
        ),
    )
    parser.add_argument(
        "--motif-forward",
        action="append",
        default=[],
        metavar="PATTERN:COUNT",
        help="Motif constraints applied only to the forward (+) strand windows.",
    )
    parser.add_argument(
        "--motif-reverse",
        action="append",
        default=[],
        metavar="PATTERN:COUNT",
        help="Motif constraints applied only to the reverse (-) strand windows.",
    )
    parser.add_argument(
        "--motif-mismatches",
        type=int,
        default=0,
        metavar="N",
        help="Allow up to N mismatches when counting motif hits (default: 0 = exact matches).",
    )
    parser.add_argument(
        "--sequence-id",
        action="append",
        default=[],
        help="Restrict scanning to specific sequence IDs (repeat per ID, e.g., --sequence-id chr1).",
    )
    parser.add_argument(
        "--sequence-class",
        action="append",
        dest="sequence_classes",
        choices=["chromosome", "scaffold"],
        help=(
            "Select which FASTA record classes to scan: chromosomes (NC_0*) and/or scaffolds (NT_/NW_). "
            "Repeat to include multiple classes (default: chromosome)."
        ),
    )
    parser.add_argument(
        "--chromosome-number",
        action="append",
        dest="chromosome_numbers",
        metavar="N|X|Y",
        help=(
            "Restrict chromosome accessions by number (1-24, 23=X, 24=Y). "
            "Repeat this option to include multiple chromosomes. Default: all chromosomes."
        ),
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="SEQ:START-END",
        help="Restrict scanning to genomic ranges (repeat per region, e.g., chr1:100000-200000).",
    )
    parser.add_argument(
        "--non-overlapping",
        action="store_true",
        help="Report only non-overlapping windows for each strand (skip windows intersecting previous hits).",
    )
    parser.add_argument(
        "--require-palindrome",
        action="store_true",
        help="Require each reported window to contain a palindromic (reverse-complement) sequence.",
    )
    parser.add_argument(
        "--palindrome-min-len",
        type=int,
        default=8,
        metavar="N",
        help="Minimum palindromic sequence length evaluated for filtering/reporting (default: 8).",
    )
    parser.add_argument(
        "--require-intrastrand",
        action="store_true",
        help="Require each reported window to contain an intrastrand complementary sequence (inverted repeat).",
    )
    parser.add_argument(
        "--intrastrand-min-len",
        type=int,
        default=6,
        metavar="N",
        help="Minimum length of intrastrand complementary sequence (default: 6).",
    )
    parser.add_argument(
        "--intrastrand-max-len",
        type=int,
        default=None,
        metavar="N",
        help="Maximum length of intrastrand complementary sequence (default: None/no limit).",
    )
    parser.add_argument(
        "--require-interstrand",
        action="store_true",
        help="Require each reported window to contain an interstrand complementary sequence (direct repeat).",
    )
    parser.add_argument(
        "--interstrand-min-len",
        type=int,
        default=6,
        metavar="N",
        help="Minimum length of interstrand complementary sequence (default: 6).",
    )
    parser.add_argument(
        "--interstrand-max-len",
        type=int,
        default=None,
        metavar="N",
        help="Maximum length of interstrand complementary sequence (default: None/no limit).",
    )
    parser.add_argument(
        "--intrastrand-mismatches",
        type=int,
        default=0,
        metavar="N",
        help="Allowed mismatches for intrastrand complementary sequence (default: 0).",
    )
    parser.add_argument(
        "--intrastrand-gap-min",
        type=int,
        default=0,
        metavar="N",
        help="Minimum gap length in nucleotides between intrastrand complementary strands (default: 0).",
    )
    parser.add_argument(
        "--intrastrand-gap-max",
        type=int,
        default=None,
        metavar="N",
        help="Maximum gap length in nucleotides between intrastrand complementary strands (default: None/no limit).",
    )
    parser.add_argument(
        "--interstrand-mismatches",
        type=int,
        default=0,
        metavar="N",
        help="Allowed mismatches for interstrand complementary sequence (default: 0).",
    )
    parser.add_argument(
        "--interstrand-gap-min",
        type=int,
        default=0,
        metavar="N",
        help="Minimum gap length in nucleotides between interstrand complementary strands (default: 0).",
    )
    parser.add_argument(
        "--interstrand-gap-max",
        type=int,
        default=None,
        metavar="N",
        help="Maximum gap length in nucleotides between interstrand complementary strands (default: None/no limit).",
    )
    parser.add_argument(
        "--output-prefix",
        default="nt_sequence_hits",
        help="Prefix for the tab-delimited output file (default: nt_sequence_hits)",
    )
    parser.add_argument(
        "--output-name",
        default="default",
        help="Name used to create the output_<name> directory for result files (default: default).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of worker processes to use. Set to 0 to use all available CPUs "
            "(default: 1 = no multiprocessing)."
        ),
    )
    parser.add_argument(
        "--vectorized-base",
        action="store_true",
        help="Use NumPy to pre-filter windows by base content when possible.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Attempt to offload base-content prefilters to CuPy (GPU) when available.",
    )
    parser.add_argument(
        "--gpu-kernels",
        action="store_true",
        help=(
            "Use the custom CUDA RawKernels in mako_gpu for the base-content "
            "prefilter (instead of the high-level CuPy path). Opt-in: validate "
            "with scripts/tests/test_cpu_gpu_parity.py on your GPU first. Has no "
            "effect unless the GPU engine is active."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help=(
            "Execution engine. 'auto' uses a CUDA GPU when one is detected and "
            "falls back to CPU otherwise; 'cpu' forces CPU; 'gpu' requests the GPU "
            "and falls back to CPU with a warning if unavailable. Results are "
            "identical across engines. (--use-gpu remains as an alias for 'gpu'.)"
        ),
    )
    parser.add_argument(
        "--chemistry-profile",
        help="Apply a named chemistry profile (built-in or saved) to pre-fill motif/base limits.",
    )
    parser.add_argument(
        "--strand",
        dest="strand_selections",
        action="append",
        choices=["forward", "reverse", "combined"],
        help="Select strands/windows to evaluate; repeat for multiple (default: forward).",
    )
    parser.add_argument(
        "--scan-strands",
        dest="strand_selections",
        action="append",
        choices=["forward", "reverse", "combined", "both"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--combined-forward-len",
        type=int,
        metavar="N",
        default=None,
        help="Forward-strand length to include in combined windows (default: --window value).",
    )
    parser.add_argument(
        "--combined-reverse-len",
        type=int,
        metavar="N",
        default=None,
        help="Reverse-strand length to include in combined windows (default: --window value).",
    )
    parser.add_argument(
        "--combined-overlap",
        type=int,
        default=0,
        metavar="N",
        help="Overlap between forward and reverse pieces in combined windows (0 = adjacent).",
    )
    parser.add_argument(
        "--motif-self-comp",
        action="append",
        default=[],
        metavar="SPEC",
        help=(
            "Require at least one motif match to contain self-complementary regions. "
            "Two formats are accepted:\n"
            "  Explicit sections: PATTERN:START1,LEN1:START2,LEN2:MIN_MM,MAX_MM\n"
            "    Example: GGGN{1,7}GGG:0,3:-3,3:0,0  (first 3 and last 3 bases, 0 mismatches)\n"
            "  Whole-motif scan: PATTERN:whole:MIN_LEN,MAX_LEN:MIN_MM,MAX_MM\n"
            "    Example: GGGN{1,7}GGG:whole:3,5:0,2  (scan all non-overlapping pairs 3-5 bp long, 0-2 mm)\n"
            "Legacy format PATTERN:START1,LEN1:START2,LEN2[:MAX_MM] is still accepted."
        ),
    )
    parser.add_argument(
        "--sequence-type",
        choices=["auto", "dna", "rna"],
        default="auto",
        help=(
            "Nucleic-acid alphabet of the input FASTA. 'rna' (or auto-detected RNA, "
            "e.g. viral RNA genomes) analyses U as T and writes sequence columns "
            "using U; 'dna' uses T. Motifs and base/repeat constraints accept U or T "
            "interchangeably regardless. Default: auto-detect."
        ),
    )
    parser.add_argument(
        "--min-g4hunter",
        type=float,
        default=None,
        metavar="SCORE",
        help=(
            "Optional post-scan filter: keep only hits whose matched motif (or the "
            "window, if no motif is searched) has a canonical G4Hunter score >= SCORE. "
            "Adds a 'best_g4hunter' column. The score is signed (positive = G4/G-rich, "
            "negative = i-motif/C-rich); typical G4 thresholds are ~1.0-1.5."
        ),
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help=(
            "Write hits incrementally to a crash-safe JSONL sidecar and checkpoint "
            "per sequence, then finalize the usual TSV. Sequences are processed in a "
            "deterministic order so an interrupted run can resume without duplicates. "
            "Recommended for very long, genome-scale searches."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        metavar="PATH",
        default=None,
        help="Checkpoint file path (default: <output_dir>/<prefix>.checkpoint.json). Implies --stream-output.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=5000,
        metavar="N",
        help="Stream mode: flush+fsync the JSONL sidecar every N hits (default: 5000).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Stream mode: ignore and overwrite any existing checkpoint/JSONL instead of resuming.",
    )
    return parser


def parse_motif_specs(parser: argparse.ArgumentParser, specs: Sequence[str]) -> List[MotifConstraint]:
    constraints: List[MotifConstraint] = []
    use_cython = cython_count_iupac_motif is not None
    for raw_spec in specs:
        if ":" not in raw_spec:
            parser.error(f"Motif specification '{raw_spec}' must be PATTERN:COUNT.")
        pattern_part, count_part = raw_spec.split(":", 1)
        pattern = pattern_part.strip().upper()
        if not pattern:
            parser.error("Motif pattern cannot be empty.")
        try:
            count = int(count_part)
        except ValueError as exc:
            parser.error(f"Invalid motif count in '{raw_spec}': {exc}")  # pragma: no cover
        if count < 1:
            parser.error(f"Motif count must be >= 1 in '{raw_spec}'.")
        regex = None
        if not use_cython:
            regex = re.compile(f"(?=({iupac_to_regex(parser, pattern)}))", re.IGNORECASE)
        try:
            tokens = build_pattern_tokens(pattern)
        except ValueError as exc:
            parser.error(str(exc))
        constraints.append(MotifConstraint(pattern, count, pattern, regex, tokens))
    return constraints


def parse_exclude_motifs(
    parser: argparse.ArgumentParser, specs: Sequence[str]
) -> List[ExcludeMotifConstraint]:
    constraints: List[ExcludeMotifConstraint] = []
    for raw_spec in specs:
        pattern = raw_spec.strip().upper()
        if not pattern:
            parser.error("Excluded motif pattern cannot be empty.")
        if "{" in pattern or "}" in pattern:
            parser.error(
                f"Excluded motif '{raw_spec}' may not include repeats/quantifiers like '{{n}}' or '{{a,b}}'."
            )
        regex_pattern = iupac_to_regex(parser, pattern)
        regex = re.compile(regex_pattern, re.IGNORECASE)
        constraints.append(ExcludeMotifConstraint(pattern, regex))
    return constraints


def _parse_section_spec(text: str) -> Tuple[int, int]:
    """Parse 'START,LEN' into (start, length)."""
    text = text.strip()
    if "," not in text:
        raise ValueError(f"Section '{text}' must be START,LEN (e.g., 0,3 or -3,3).")
    start_str, len_str = text.split(",", 1)
    start = int(start_str)
    length = int(len_str)
    if length <= 0:
        raise ValueError(f"Section length must be positive (got '{text}').")
    return start, length


def _parse_mm_range(text: str, label: str, parser: argparse.ArgumentParser) -> Tuple[int, int]:
    """Parse 'MIN,MAX' or a single 'N' into (min, max) mismatch counts."""
    text = text.strip()
    if "," in text:
        parts = text.split(",", 1)
        try:
            lo, hi = int(parts[0]), int(parts[1])
        except ValueError:
            parser.error(f"{label} mismatch range '{text}' must be MIN,MAX integers.")
    else:
        try:
            lo = hi = int(text)
        except ValueError:
            parser.error(f"{label} mismatch value '{text}' must be an integer.")
    if lo < 0 or hi < 0:
        parser.error(f"{label} mismatch range must be non-negative.")
    if lo > hi:
        parser.error(f"{label} mismatch min ({lo}) must not exceed max ({hi}).")
    return lo, hi


def _parse_len_range(text: str, label: str, parser: argparse.ArgumentParser) -> Tuple[int, int | None]:
    """Parse 'MIN,MAX' or a single 'N' into (min, max) length bounds."""
    text = text.strip()
    if "," in text:
        parts = text.split(",", 1)
        try:
            lo = int(parts[0])
            hi_str = parts[1].strip()
            hi: int | None = None if hi_str in ("", "*", "any") else int(hi_str)
        except ValueError:
            parser.error(f"{label} length range '{text}' must be MIN,MAX integers.")
    else:
        try:
            lo = int(text)
            hi = None
        except ValueError:
            parser.error(f"{label} length value '{text}' must be an integer.")
    if lo < 1:
        parser.error(f"{label} minimum complementary length must be >= 1.")
    if hi is not None and hi < lo:
        parser.error(f"{label} length max ({hi}) must be >= min ({lo}).")
    return lo, hi


def parse_motif_self_comp_specs(
    parser: argparse.ArgumentParser, specs: Sequence[str]
) -> List[MotifSelfCompConstraint]:
    """Parse --motif-self-comp specs into MotifSelfCompConstraint objects.

    Accepted formats
    ----------------
    Whole-motif scan::
        PATTERN:whole:MIN_LEN,MAX_LEN[:MIN_MM,MAX_MM][:EXCL_MOTIFS]

    Explicit sections::
        PATTERN:START1,LEN1:START2,LEN2[:MIN_MM,MAX_MM][:EXCL_MOTIFS]
    """
    constraints: List[MotifSelfCompConstraint] = []
    for raw_spec in specs:
        parts = raw_spec.split(":")
        if len(parts) < 3:
            parser.error(f"Self-comp spec '{raw_spec}' needs at least PATTERN:MODE_OR_SECTION1:...")
        
        pattern = parts[0].strip().upper()
        mode_or_sec1 = parts[1].strip().lower()
        
        min_mm, max_mm = 0, 0
        excluded_motifs = []
        
        if mode_or_sec1 == "whole":
            # --- whole-motif scan mode ---
            min_len, max_len = _parse_len_range(parts[2], f"'{raw_spec}'", parser)
            if len(parts) >= 4:
                p3 = parts[3].strip()
                if p3 and (p3[0].isdigit() or (len(p3) > 1 and p3[0] == "-" and p3[1].isdigit())):
                    min_mm, max_mm = _parse_mm_range(p3, f"'{raw_spec}'", parser)
                    if len(parts) >= 5:
                        excluded_motifs = [m.strip().upper() for m in parts[4].split(",") if m.strip()]
                else:
                    excluded_motifs = [m.strip().upper() for m in p3.split(",") if m.strip()]
            
            # Build MotifConstraint once for caching
            cached_tokens = build_pattern_tokens(pattern)
            compiled_motif = MotifConstraint(
                label=pattern,
                required_count=1,
                pattern=pattern,
                regex=None,
                tokens=cached_tokens,
            )
            constraints.append(
                MotifSelfCompConstraint(
                    motif_pattern=pattern,
                    whole_motif=True,
                    min_comp_len=min_len,
                    max_comp_len=max_len,
                    min_mismatches=min_mm,
                    max_mismatches=max_mm,
                    excluded_motifs=excluded_motifs,
                    compiled_motif=compiled_motif
                )
            )
        else:
            # --- explicit-section mode ---
            try:
                s1_start, s1_len = _parse_section_spec(parts[1])
                s2_start, s2_len = _parse_section_spec(parts[2])
            except ValueError as exc:
                parser.error(str(exc))
            
            if len(parts) >= 4:
                p3 = parts[3].strip()
                if p3 and (p3[0].isdigit() or (len(p3) > 1 and p3[0] == "-" and p3[1].isdigit())):
                    min_mm, max_mm = _parse_mm_range(p3, f"'{raw_spec}'", parser)
                    if len(parts) >= 5:
                        excluded_motifs = [m.strip().upper() for m in parts[4].split(",") if m.strip()]
                else:
                    excluded_motifs = [m.strip().upper() for m in p3.split(",") if m.strip()]
            
            # Build MotifConstraint once for caching
            cached_tokens = build_pattern_tokens(pattern)
            compiled_motif = MotifConstraint(
                label=pattern,
                required_count=1,
                pattern=pattern,
                regex=None,
                tokens=cached_tokens,
            )
            constraints.append(
                MotifSelfCompConstraint(
                    motif_pattern=pattern,
                    sec1_start=s1_start,
                    sec1_len=s1_len,
                    sec2_start=s2_start,
                    sec2_len=s2_len,
                    min_mismatches=min_mm,
                    max_mismatches=max_mm,
                    excluded_motifs=excluded_motifs,
                    compiled_motif=compiled_motif
                )
            )
    return constraints


def _extract_section(sequence: str, start: int, length: int) -> str:
    """Extract a slice from sequence using Python-style negative start indices."""
    seq_len = len(sequence)
    if start < 0:
        actual_start = seq_len + start
    else:
        actual_start = start
    actual_start = max(0, min(actual_start, seq_len))
    actual_end = min(actual_start + length, seq_len)
    return sequence[actual_start:actual_end]


def sections_are_complementary(
    section1: str,
    section2: str,
    min_matches: int,
    max_matches: int | None,
    min_mismatches: int,
    max_mismatches: int,
) -> bool:
    """Return True if section1 is reverse-complementary to section2 within the match and mismatch ranges.
    Mismatches are NOT allowed at the 5' or 3' ends of the complementary alignment.
    """
    if len(section1) != len(section2):
        return False
    rc = reverse_complement(section2)
    # Check 5' and 3' ends
    if section1[0] != rc[0] or section1[-1] != rc[-1]:
        return False
    
    mismatches = sum(a != b for a, b in zip(section1, rc))
    matches = len(section1) - mismatches
    
    if not (min_mismatches <= mismatches <= max_mismatches):
        return False
    if not (min_matches <= matches <= (max_matches if max_matches is not None else float('inf'))):
        return False
        
    return True


def check_motif_self_comp(
    window_seq: str,
    constraint: MotifSelfCompConstraint,
    max_motif_mismatches: int = 0,
) -> bool:
    """Return True if at least one motif match in window_seq satisfies self-complementarity."""
    seq = window_seq.upper()
    if constraint.compiled_motif:
        temp_constraint = constraint.compiled_motif
    else:
        cached_tokens = SELF_COMP_TOKEN_CACHE.get(constraint.motif_pattern)
        if cached_tokens is None:
            cached_tokens = build_pattern_tokens(constraint.motif_pattern)
            SELF_COMP_TOKEN_CACHE[constraint.motif_pattern] = cached_tokens
        temp_constraint = MotifConstraint(
            label=constraint.motif_pattern,
            required_count=1,
            pattern=constraint.motif_pattern,
            regex=None,
            tokens=cached_tokens,
        )
    matches = motif_matches(seq, temp_constraint, max_motif_mismatches)
    if not matches:
        return False

    # Prepare excluded motif regexes once if needed
    excl_regexes = []
    if constraint.excluded_motifs:
        for pat in constraint.excluded_motifs:
            excl_regexes.append(re.compile(translate_iupac_pattern(pat), re.IGNORECASE))

    def passed_excluded_check(s1: str, s2: str) -> bool:
        if not excl_regexes:
            return True
        for regex in excl_regexes:
            if regex.search(s1) or regex.search(s2):
                return False
        return True

    for m_start, m_end in matches:
        matched_seq = seq[m_start:m_end]
        m_len = len(matched_seq)

        m_lo = max(1, constraint.min_comp_len)
        m_hi = constraint.max_comp_len
        mm_lo = constraint.min_mismatches
        mm_hi = constraint.max_mismatches

        if constraint.whole_motif:
            # Iterate through all possible total lengths that could satisfy the match/mismatch ranges
            # total_len = matches + mismatches
            # min total = m_lo + mm_lo
            # max total = (m_hi or m_len//2) + mm_hi
            t_lo = m_lo + mm_lo
            t_hi = (m_hi if m_hi is not None else m_len // 2) + mm_hi
            t_hi = min(t_hi, m_len // 2)

            for total_len in range(t_lo, t_hi + 1):
                for i in range(m_len - total_len):
                    for j in range(i + total_len, m_len - total_len + 1):
                        sec1 = matched_seq[i:i + total_len]
                        sec2 = matched_seq[j:j + total_len]
                        if sections_are_complementary(
                            sec1, sec2,
                            m_lo, m_hi,
                            mm_lo, mm_hi
                        ):
                            if passed_excluded_check(sec1, sec2):
                                return True
        else:
            # Manual mode: we check all total lengths in the allowed mismatch range for the given match count
            for mm in range(mm_lo, mm_hi + 1):
                total_len = m_lo + mm  # Manual mode treats sec1_len as base match count
                sec1 = _extract_section(matched_seq, constraint.sec1_start, total_len)
                sec2 = _extract_section(matched_seq, constraint.sec2_start, total_len)
                if len(sec1) != total_len or len(sec2) != total_len:
                    continue
                if sections_are_complementary(
                    sec1, sec2,
                    m_lo, m_hi,
                    mm_lo, mm_hi
                ):
                    if passed_excluded_check(sec1, sec2):
                        return True
    return False


def parse_base_content_specs(
    parser: argparse.ArgumentParser, specs: Sequence[str]
) -> Dict[str, Tuple[float, float]]:
    constraints: Dict[str, Tuple[float, float]] = {}
    for raw_spec in specs:
        parts = raw_spec.split(":")
        if len(parts) != 3:
            parser.error(f"Base content specification '{raw_spec}' must be BASE:MIN:MAX.")
        base, min_part, max_part = parts
        base = base.strip().upper()
        if base == "U":  # RNA: treat uracil as thymine
            base = "T"
        if base not in BASES:
            parser.error(f"Base content only supports A, C, G, T, or U (got '{base}').")
        try:
            min_pct = float(min_part)
            max_pct = float(max_part)
        except ValueError as exc:
            parser.error(f"Invalid percentage in '{raw_spec}': {exc}")  # pragma: no cover
        if not 0 <= min_pct <= 100 or not 0 <= max_pct <= 100:
            parser.error(f"Percentages must be between 0 and 100 in '{raw_spec}'.")
        if min_pct > max_pct:
            parser.error(f"Minimum percentage exceeds maximum in '{raw_spec}'.")
        constraints[base] = (min_pct, max_pct)
    return constraints


def parse_max_repeat_specs(parser: argparse.ArgumentParser, specs: Sequence[str]) -> Dict[str, int]:
    constraints: Dict[str, int] = {}
    for raw_spec in specs:
        if ":" not in raw_spec:
            parser.error(f"Repeat specification '{raw_spec}' must be BASE:MAX.")
        base_part, max_part = raw_spec.split(":", 1)
        base = base_part.strip().upper()
        if base == "U":  # RNA: treat uracil as thymine
            base = "T"
        if base not in BASES:
            parser.error(f"Repeat constraints only support bases A, C, G, T, or U (got '{base}').")
        try:
            limit = int(max_part)
        except ValueError as exc:
            parser.error(f"Repeat limit in '{raw_spec}' is invalid: {exc}")  # pragma: no cover
        if limit < 1:
            parser.error(f"Repeat limit must be >= 1 in '{raw_spec}'.")
        constraints[base] = limit
    return constraints


def iupac_to_regex(parser: argparse.ArgumentParser, pattern: str) -> str:
    try:
        return translate_iupac_pattern(pattern)
    except ValueError as exc:
        parser.error(str(exc))


def parse_iupac_token_specs(pattern: str) -> List[Tuple[str, int, int]]:
    tokens: List[Tuple[str, int, int]] = []
    idx = 0
    upper_pattern = pattern.upper()
    while idx < len(upper_pattern):
        char = upper_pattern[idx]
        if char not in IUPAC_CODES:
            raise ValueError(
                f"Unsupported IUPAC nucleotide '{char}' in motif '{pattern}'. "
                "Motifs may also include quantifiers like N{{1,7}}."
            )
        idx += 1
        min_rep = 1
        max_rep = 1
        if idx < len(upper_pattern) and upper_pattern[idx] == "{":
            end = upper_pattern.find("}", idx)
            if end == -1:
                raise ValueError(f"Unterminated quantifier in motif '{pattern}'.")
            quant_body = upper_pattern[idx + 1 : end]
            parts = quant_body.split(",")
            if not 1 <= len(parts) <= 2 or not all(part.isdigit() for part in parts):
                raise ValueError(
                    f"Invalid quantifier '{{{quant_body}}}' in motif '{pattern}'. "
                    "Use formats like {{3}} or {{1,7}}."
                )
            if len(parts) == 1:
                min_rep = max_rep = int(parts[0])
            else:
                min_rep = int(parts[0])
                max_rep = int(parts[1])
                if min_rep > max_rep:
                    raise ValueError(f"Quantifier lower bound exceeds upper bound in motif '{pattern}'.")
            if min_rep <= 0 or max_rep <= 0:
                raise ValueError(f"Quantifier values must be positive in motif '{pattern}'.")
            idx = end + 1
        tokens.append((char, min_rep, max_rep))
    return tokens


def build_pattern_tokens(pattern: str) -> Tuple[PatternToken, ...]:
    specs = parse_iupac_token_specs(pattern)
    tokens: List[PatternToken] = []
    for code, min_rep, max_rep in specs:
        raw = IUPAC_CODES[code]
        if raw.startswith("[") and raw.endswith("]"):
            allowed_chars = frozenset(raw[1:-1])
        else:
            allowed_chars = frozenset(raw)
        tokens.append(PatternToken(allowed_chars, min_rep, max_rep))
    return tuple(tokens)


def translate_iupac_pattern(pattern: str) -> str:
    translated: List[str] = []
    for code, min_rep, max_rep in parse_iupac_token_specs(pattern):
        base_regex = IUPAC_CODES[code]
        token = f"(?:{base_regex})" if len(base_regex) > 1 else base_regex
        if min_rep == max_rep == 1:
            translated.append(token)
        else:
            if min_rep == max_rep:
                quant_body = str(min_rep)
            else:
                quant_body = f"{min_rep},{max_rep}"
            translated.append(f"{token}{{{quant_body}}}")
    return "".join(translated)


def iter_nc_sequences(
    fasta_path: Path, record_prefixes: Sequence[str] | None = None
) -> Iterable[Tuple[str, str]]:
    """Yield (sequence_id, sequence) pairs. If record_prefixes is None, all sequences are yielded."""
    header: str | None = None
    seq_chunks: List[str] = []
    include_current = False
    
    if record_prefixes is None:
        normalized_prefixes = None
    else:
        normalized_prefixes = tuple(prefix.upper() for prefix in record_prefixes)

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if include_current and seq_chunks:
                    yield format_seq_id(header), "".join(seq_chunks)
                header = line
                header_upper = header.upper()
                if normalized_prefixes is None:
                    include_current = True
                else:
                    include_current = any(header_upper.startswith(prefix) for prefix in normalized_prefixes)
                seq_chunks = []
            elif include_current:
                seq_chunks.append(line.upper())
        if include_current and seq_chunks and header:
            yield format_seq_id(header), "".join(seq_chunks)


def format_seq_id(header: str | None) -> str:
    if not header:
        return "unknown"
    return header[1:].split()[0]


def build_prefix_counts(sequence: str) -> Dict[str, List[int]]:
    seq_len = len(sequence)
    a = [0] * (seq_len + 1)
    c = [0] * (seq_len + 1)
    g = [0] * (seq_len + 1)
    t = [0] * (seq_len + 1)
    cur_a = cur_c = cur_g = cur_t = 0
    for idx, nuc in enumerate(sequence):
        if nuc == 'A':
            cur_a += 1
        elif nuc == 'C':
            cur_c += 1
        elif nuc == 'G':
            cur_g += 1
        elif nuc == 'T':
            cur_t += 1
        pos = idx + 1
        a[pos] = cur_a
        c[pos] = cur_c
        g[pos] = cur_g
        t[pos] = cur_t
    return {"A": a, "C": c, "G": g, "T": t}


def _vectorized_candidate_positions(
    sequence: str,
    window: int,
    step: int,
    base_constraints: Dict[str, Tuple[float, float]],
) -> List[int] | None:
    if not base_constraints:
        return None
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - optional dependency
        return None
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    limit = len(sequence) - window + 1
    if limit <= 0:
        return None
    starts = np.arange(0, limit, step, dtype=np.int64)
    if starts.size == 0:
        return []
    mask = np.ones_like(starts, dtype=bool)
    for base, (min_pct, max_pct) in base_constraints.items():
        code = ord(base.upper())
        hits = (seq_bytes == code).astype(np.int32)
        prefix = np.concatenate(([0], np.cumsum(hits)))
        ends = starts + window
        counts = prefix[ends] - prefix[starts]
        pct = (counts.astype(np.float64) * 100.0) / window
        mask &= (pct >= min_pct) & (pct <= max_pct)
        if not mask.any():
            return []
    return starts[mask].astype(int).tolist()


def _gpu_candidate_positions(
    sequence: str,
    window: int,
    step: int,
    base_constraints: Dict[str, Tuple[float, float]],
) -> List[int] | None:
    if not base_constraints:
        return None
    try:
        import cupy as cp  # type: ignore
        import numpy as np
    except ImportError:  # pragma: no cover - optional dependency
        return None
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    seq_gpu = cp.asarray(seq_bytes)
    limit = len(sequence) - window + 1
    if limit <= 0:
        return None
    starts = cp.arange(0, limit, step, dtype=cp.int32)
    if starts.size == 0:
        return []
    mask = cp.ones(starts.shape, dtype=bool)
    for base, (min_pct, max_pct) in base_constraints.items():
        code = ord(base.upper())
        hits = (seq_gpu == code)
        prefix = cp.concatenate((cp.zeros(1, dtype=cp.int32), cp.cumsum(hits, dtype=cp.int32)))
        ends = starts + window
        counts = prefix[ends] - prefix[starts]
        pct = counts.astype(cp.float64) * (100.0 / window)
        mask = mask & (pct >= min_pct) & (pct <= max_pct)
        if not cp.any(mask):
            return []
    selected = cp.asnumpy(starts[mask])
    if selected.size == 0:
        return []
    return selected.astype(int).tolist()


_BASE_NIBBLE = {"A": 1, "C": 2, "G": 4, "T": 8, "U": 8}


def _extract_fixed_motifs(motifs: Sequence[MotifConstraint] | None) -> List[Tuple[List[int], int]]:
    """Per-position IUPAC masks + required count for fixed-length motifs only.

    A motif is "fixed" when every token is a single position (min==max==1), i.e.
    it has no quantifiers. Such motifs admit an exact superset prefilter on the
    GPU (overlapping count >= required); variable-length motifs are skipped here
    and matched entirely on the CPU.
    """
    fixed: List[Tuple[List[int], int]] = []
    for constraint in motifs or []:
        tokens = constraint.tokens or build_pattern_tokens(constraint.pattern)
        if not tokens:
            continue
        masks: List[int] = []
        ok = True
        for token in tokens:
            if token.min_repeat != 1 or token.max_repeat != 1:
                ok = False
                break
            mask = 0
            for base in token.allowed:
                mask |= _BASE_NIBBLE.get(base.upper(), 0)
            masks.append(mask)
        if ok and masks:
            fixed.append((masks, int(constraint.required_count)))
    return fixed


def build_candidate_positions(
    sequence: str,
    window: int,
    step: int,
    base_constraints: Dict[str, Tuple[float, float]],
    runtime: RuntimeOptions | None,
    repeat_constraints: Dict[str, int] | None = None,
    motifs: Sequence[MotifConstraint] | None = None,
) -> List[int] | None:
    if not runtime:
        return None
    if runtime.use_gpu:
        # Every CuPy path — custom RawKernels AND high-level ops like cumsum —
        # is JIT-compiled by NVRTC, which needs the CUDA toolkit headers. Probe
        # once; if compilation is impossible (e.g. headers missing) skip all GPU
        # work and fall back, instead of crashing mid-scan.
        gpu_ok = False
        try:
            import mako_gpu

            gpu_ok = mako_gpu.kernels_available()
        except Exception:  # pragma: no cover
            gpu_ok = False
        if gpu_ok:
            if runtime.gpu_kernels:
                # Custom fused RawKernel (opt-in). Returns an exact superset.
                try:
                    fixed_motifs = _extract_fixed_motifs(motifs)
                    if base_constraints or repeat_constraints or fixed_motifs:
                        fused = mako_gpu.fused_predicate_prefilter(
                            sequence, window, step, base_constraints or {},
                            repeat_constraints or {}, fixed_motifs,
                        )
                        if fused is not None:
                            return fused
                except Exception:  # pragma: no cover - kernel/driver issue → fall through
                    pass
            if base_constraints:
                try:
                    gpu_positions = _gpu_candidate_positions(sequence, window, step, base_constraints)
                    if gpu_positions is not None:
                        return gpu_positions
                except Exception:  # pragma: no cover
                    pass
    if not base_constraints:
        return None
    if runtime.vectorized_base:
        vector_positions = _vectorized_candidate_positions(sequence, window, step, base_constraints)
        if vector_positions is not None:
            return vector_positions
    return None


def detect_gpu_capabilities() -> Dict[str, object]:
    """Probe for a usable CUDA device at startup.

    Returns a dict describing availability and, when present, the device name,
    compute capability and total memory. Never raises: any failure to import the
    GPU backend or query the driver is reported as ``available=False`` with a
    reason so callers can transparently fall back to the CPU engine.
    """
    info: Dict[str, object] = {
        "available": False,
        "backend": None,
        "device": None,
        "compute_capability": None,
        "total_mem_bytes": None,
        "reason": "",
    }
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # ImportError, or a CuPy install with a broken CUDA runtime
        info["reason"] = f"CuPy unavailable ({exc.__class__.__name__})"
        return info
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            info["reason"] = "no CUDA devices detected"
            return info
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        _free, total = cp.cuda.Device(0).mem_info
        info.update(
            {
                "available": True,
                "backend": "cupy",
                "device": name.decode() if isinstance(name, (bytes, bytearray)) else str(name),
                "compute_capability": f"{props['major']}.{props['minor']}",
                "total_mem_bytes": int(total),
            }
        )
    except Exception as exc:  # driver/runtime mismatch, no permission, etc.
        info["reason"] = f"CUDA query failed ({exc.__class__.__name__}: {exc})"
    return info


def resolve_engine(requested: str, use_gpu_flag: bool, caps: Dict[str, object]) -> str:
    """Resolve the concrete engine ('cpu' or 'gpu') from the requested mode.

    'auto' picks the GPU when available, else CPU. An explicit 'gpu' request that
    cannot be satisfied degrades to CPU (the caller is expected to warn). The
    legacy ``--use-gpu`` flag is treated as requesting 'gpu' when the mode is
    left at its 'auto' default.
    """
    mode = (requested or "auto").lower()
    if use_gpu_flag and mode == "auto":
        mode = "gpu"
    if mode == "auto":
        return "gpu" if caps.get("available") else "cpu"
    if mode == "gpu" and not caps.get("available"):
        return "cpu"
    return mode


def _py_longest_run(subseq: str, base: str) -> int:
    longest = 0
    current = 0
    for char in subseq:
        if char == base:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def longest_run(subseq: str, base: str) -> int:
    if cython_longest_run is not None:
        return cython_longest_run(subseq, base)
    return _py_longest_run(subseq, base)


def _suffix_min_lengths(tokens: Sequence[PatternToken]) -> List[int]:
    suffix = [0] * (len(tokens) + 1)
    for idx in range(len(tokens) - 1, -1, -1):
        suffix[idx] = suffix[idx + 1] + tokens[idx].min_repeat
    return suffix


def _match_tokens_with_mismatches(
    sequence: str,
    tokens: Sequence[PatternToken],
    seq_pos: int,
    token_idx: int,
    mismatches_used: int,
    max_mismatches: int,
    suffix_mins: Sequence[int],
) -> int | None:
    if token_idx >= len(tokens):
        return seq_pos
    token = tokens[token_idx]
    remaining_min = suffix_mins[token_idx + 1]
    seq_len = len(sequence)
    max_available = seq_len - remaining_min - seq_pos
    if max_available < token.min_repeat:
        return None
    max_repeat = min(token.max_repeat, max_available)
    if max_repeat < token.min_repeat:
        return None
    allowed = token.allowed
    for repeat in range(max_repeat, token.min_repeat - 1, -1):
        end_pos = seq_pos + repeat
        mismatches = mismatches_used
        exceeded = False
        for idx in range(seq_pos, end_pos):
            if sequence[idx] not in allowed:
                mismatches += 1
                if mismatches > max_mismatches:
                    exceeded = True
                    break
        if exceeded:
            continue
        result = _match_tokens_with_mismatches(
            sequence,
            tokens,
            end_pos,
            token_idx + 1,
            mismatches,
            max_mismatches,
            suffix_mins,
        )
        if result is not None:
            return result
    return None


def motif_matches(window_seq: str, constraint: MotifConstraint, max_mismatches: int) -> List[Tuple[int, int]]:
    if not window_seq:
        return []
    if max_mismatches <= 0:
        regex = get_motif_regex(constraint)
        matches: List[Tuple[int, int]] = []
        for match in regex.finditer(window_seq):
            captured = match.group(1)
            if not captured:
                continue
            matches.append((match.start(1), match.end(1)))
        return matches
    tokens = constraint.tokens or build_pattern_tokens(constraint.pattern)
    if not tokens:
        return []
    suffix_mins = _suffix_min_lengths(tokens)
    min_len = suffix_mins[0]
    seq_len = len(window_seq)
    if seq_len < min_len or min_len <= 0:
        return []
    matches: List[Tuple[int, int]] = []
    limit = seq_len - min_len + 1
    for start in range(limit):
        end = _match_tokens_with_mismatches(
            window_seq,
            tokens,
            start,
            0,
            0,
            max_mismatches,
            suffix_mins,
        )
        if end is not None:
            matches.append((start, end))
    return matches


def motif_hit_count(window_seq: str, constraint: MotifConstraint, max_mismatches: int = 0) -> int:
    if max_mismatches <= 0 and cython_count_iupac_motif is not None:
        return cython_count_iupac_motif(window_seq, constraint.pattern)
    if max_mismatches <= 0:
        regex = get_motif_regex(constraint)
        return sum(1 for _ in regex.finditer(window_seq))
    return len(motif_matches(window_seq, constraint, max_mismatches))


def get_motif_regex(constraint: MotifConstraint) -> re.Pattern[str]:
    if constraint.regex is not None:
        return constraint.regex
    cached = MOTIF_REGEX_CACHE.get(constraint.pattern)
    if cached is not None:
        return cached
    regex_body = translate_iupac_pattern(constraint.pattern)
    compiled = re.compile(f"(?=({regex_body}))", re.IGNORECASE)
    MOTIF_REGEX_CACHE[constraint.pattern] = compiled
    return compiled


def get_matched_motif_sequence(
    sequence: str,
    motif_spans: Sequence[Sequence[Tuple[int, int]]],
) -> str:
    """Returns the uppercase substring of sequence that covers all motif matches based on pre-calculated spans."""
    if not sequence or not motif_spans:
        return ""
    
    min_start = len(sequence)
    max_end = 0
    found = False
    
    for spans in motif_spans:
        for start, end in spans:
            min_start = min(min_start, start)
            max_end = max(max_end, end)
            found = True
            
    if not found:
        return ""
    
    return sequence[max(0, min_start):min(len(sequence), max_end)].upper()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(RC_MAP)[::-1]


def g4hunter_best_score(sequence: str) -> float:
    """Canonical G4Hunter score (Bedrat, Lacroix & Mergny 2016): the signed mean
    of per-base run-length scores (+min(run,4) within G-tracts, -min(run,4)
    within C-tracts, 0 otherwise). Positive => G-quadruplex propensity, negative
    => i-motif. U is treated as T (irrelevant: the score depends only on G/C)."""
    if not sequence:
        return 0.0
    if cython_g4hunter_score is not None:
        return cython_g4hunter_score(sequence, 0)
    seq = sequence.upper()
    n = len(seq)
    scores = [0] * n
    i = 0
    while i < n:
        base = seq[i]
        if base == "G" or base == "C":
            j = i
            while j < n and seq[j] == base:
                j += 1
            run = j - i
            val = run if run < 4 else 4
            if base == "C":
                val = -val
            for k in range(i, j):
                scores[k] = val
            i = j
        else:
            i += 1
    return sum(scores) / n if n else 0.0


def find_longest_palindrome(sequence: str) -> Tuple[int, int]:
    seq = sequence.upper()
    n = len(seq)
    if n == 0:
        return 0, 0
    best_start = 0
    best_end = 0

    def expand(left: int, right: int) -> Tuple[int, int]:
        while left >= 0 and right < n:
            right_base = seq[right]
            left_base = seq[left]
            complement = COMPLEMENT.get(right_base, right_base)
            if left_base != complement:
                break
            left -= 1
            right += 1
        return left + 1, right

    for center in range(n):
        start, end = expand(center, center)
        if end - start > best_end - best_start:
            best_start, best_end = start, end
    for center in range(n - 1):
        start, end = expand(center, center + 1)
        if end - start > best_end - best_start:
            best_start, best_end = start, end
    return best_start, best_end


def find_longest_intrastrand_complement(
    seq: str,
    min_len: int,
    max_len: int | None = None,
    allowed_mismatches: int = 0,
    gap_min_len: int = 0,
    gap_max_len: int | None = None
) -> Tuple[int, int, int]:
    """Finds the longest non-overlapping pair of reverse-complementary substrings in seq.
    Returns (start1, start2, length) where start1 < start2.
    """
    if find_longest_intrastrand_complement_cython is not None:
        return find_longest_intrastrand_complement_cython(
            seq, min_len, max_len, allowed_mismatches, gap_min_len, gap_max_len
        )

    seq = seq.upper()
    n = len(seq)
    best_len = 0
    best_i = -1
    best_j = -1

    for i in range(n - 2 * min_len - gap_min_len + 1):
        j_start = i + 2 * max(min_len, best_len + 1) + gap_min_len - 1
        j_end = n
        if max_len is not None and gap_max_len is not None:
            j_end = min(j_end, i + 2 * max_len + gap_max_len)

        for j in range(j_start, j_end):
            max_possible = (j - i + 1 - gap_min_len) // 2
            if max_len is not None:
                max_possible = min(max_possible, max_len)
            
            if max_possible <= best_len:
                continue

            mismatches = 0
            curr_best_len = 0
            for length in range(1, max_possible + 1):
                left_base = seq[i + length - 1]
                right_base = seq[j - length + 1]
                comp = COMPLEMENT.get(right_base, right_base)
                if left_base != comp:
                    mismatches += 1
                if mismatches > allowed_mismatches:
                    break
                
                # Check gap constraints for this length
                gap = j - i - 2 * length + 1
                if gap >= gap_min_len and (gap_max_len is None or gap <= gap_max_len):
                    curr_best_len = length
            
            if curr_best_len >= min_len and curr_best_len > best_len:
                best_len = curr_best_len
                best_i = i
                best_j = j

    return best_i, best_j, best_len


def find_longest_interstrand_complement(
    seq: str,
    min_len: int,
    max_len: int | None = None,
    allowed_mismatches: int = 0,
    gap_min_len: int = 0,
    gap_max_len: int | None = None
) -> Tuple[int, int, int]:
    """Finds the longest non-overlapping pair of identical substrings in seq.
    Returns (start1, start2, length) where start1 < start2.
    """
    if find_longest_interstrand_complement_cython is not None:
        return find_longest_interstrand_complement_cython(
            seq, min_len, max_len, allowed_mismatches, gap_min_len, gap_max_len
        )

    seq = seq.upper()
    n = len(seq)
    best_len = 0
    best_i = -1
    best_j = -1

    for i in range(n - 2 * min_len - gap_min_len + 1):
        j_start = i + max(min_len, best_len + 1) + gap_min_len
        j_end = n - max(min_len, best_len + 1) + 1
        if gap_max_len is not None:
            j_end = min(j_end, (n + i + gap_max_len) // 2 + 1)
            if max_len is not None:
                j_end = min(j_end, i + max_len + gap_max_len + 1)

        for j in range(j_start, j_end):
            max_possible = min(j - i - gap_min_len, n - j)
            if max_len is not None:
                max_possible = min(max_possible, max_len)
            
            if max_possible <= best_len:
                continue

            mismatches = 0
            curr_best_len = 0
            for length in range(1, max_possible + 1):
                if seq[i + length - 1] != seq[j + length - 1]:
                    mismatches += 1
                if mismatches > allowed_mismatches:
                    break
                
                # Check gap constraints for this length
                gap = j - i - length
                if gap >= gap_min_len and (gap_max_len is None or gap <= gap_max_len):
                    curr_best_len = length
            
            if curr_best_len >= min_len and curr_best_len > best_len:
                best_len = curr_best_len
                best_i = i
                best_j = j

    return best_i, best_j, best_len





def build_hit_record(
    seq_id: str,
    display_sequence: str,
    analysis_sequence: str,
    window_start: int,
    window_end: int,
    motifs: Sequence[MotifConstraint],
    base_constraints: Dict[str, Tuple[float, float]],
    repeat_constraints: Dict[str, int],
    strand_label: str,
    prefix_counts: Dict[str, List[int]] | None = None,
    prefix_range: Tuple[int, int] | None = None,
    exclude_motifs: Sequence[ExcludeMotifConstraint] | None = None,
    palindrome_config: Dict[str, object] | None = None,
    intrastrand_config: Dict[str, object] | None = None,
    interstrand_config: Dict[str, object] | None = None,
    region_filters: Dict[str, List[Tuple[int, int]]] | None = None,
    max_motif_mismatches: int = 0,
    self_comp_constraints: Sequence[MotifSelfCompConstraint] | None = None,
    precalculated_motif_spans: List[List[Tuple[int, int]]] | None = None,
) -> Dict[str, object] | None:
    analysis = analysis_sequence.upper()
    if "U" in analysis:  # RNA: analyse uracil as thymine (covers combined windows)
        analysis = analysis.replace("U", "T")
    if not analysis:
        return None
    window_len = len(analysis)
    motif_hits: List[Tuple[str, int]] = []
    motif_spans: List[List[Tuple[int, int]]] = []
    for idx, constraint in enumerate(motifs):
        if precalculated_motif_spans is not None and idx < len(precalculated_motif_spans):
            spans = precalculated_motif_spans[idx]
        else:
            spans = motif_matches(analysis, constraint, max_motif_mismatches)
        motif_spans.append(spans)
        count = len(spans)
        motif_hits.append((constraint.label, count))
        if count < constraint.required_count:
            return None
    base_percentages: List[Tuple[str, float]] = []
    if base_constraints:
        if prefix_counts is not None and prefix_range is not None:
            start_idx, end_idx = prefix_range
            window_len = end_idx - start_idx
            if window_len <= 0:
                return None
            for base, (min_pct, max_pct) in base_constraints.items():
                count = prefix_counts[base][end_idx] - prefix_counts[base][start_idx]
                pct = (count / window_len) * 100
                base_percentages.append((base, pct))
                if pct < min_pct or pct > max_pct:
                    return None
        else:
            counts = Counter(analysis)
            for base, (min_pct, max_pct) in base_constraints.items():
                pct = (counts.get(base, 0) / window_len) * 100
                base_percentages.append((base, pct))
                if pct < min_pct or pct > max_pct:
                    return None
    max_runs: List[Tuple[str, int]] = []
    for base, limit in repeat_constraints.items():
        run = longest_run(analysis, base)
        max_runs.append((base, run))
        if run > limit:
            return None
    if exclude_motifs:
        for exclude in exclude_motifs:
            if exclude.regex.search(analysis):
                return None
    palindrome_enabled = False
    palindrome_min_len = 0
    pal_start = 0
    pal_end = 0
    pal_len = 0
    pal_seq = ""
    # Only run the (O(n^2)) palindrome search when the feature is explicitly
    # enabled, so a default scan neither pays the cost nor emits palindrome
    # columns. Disabled => no computation, no columns.
    if palindrome_config and palindrome_config.get("enabled"):
        palindrome_enabled = True
        try:
            palindrome_min_len = int(palindrome_config.get("min_len", 0))
        except (TypeError, ValueError):
            palindrome_min_len = 0
        palindrome_min_len = max(0, palindrome_min_len)
        pal_start, pal_end = find_longest_palindrome(analysis)
        pal_len = pal_end - pal_start
        if pal_len > 0:
            pal_seq = analysis[pal_start:pal_end]
        if pal_len < max(1, palindrome_min_len):
            return None
    if pal_len > 0:
        if strand_label == "-":
            palindrome_start_coord = window_end - pal_end + 1
            palindrome_end_coord = window_end - pal_start
        else:
            palindrome_start_coord = window_start + pal_start
            palindrome_end_coord = window_start + pal_end - 1
    else:
        palindrome_start_coord = None
        palindrome_end_coord = None

    # Intrastrand complementarity scan
    intrastrand_enabled = False
    intrastrand_min_len = 6
    intrastrand_max_len = None
    intra_len = 0
    intra_seq = ""
    intra_start1 = "NA"
    intra_end1 = "NA"
    intra_start2 = "NA"
    intra_end2 = "NA"
    if intrastrand_config and intrastrand_config.get("enabled"):
        intrastrand_enabled = True
        try:
            intrastrand_min_len = max(1, int(intrastrand_config.get("min_len", 6) or 6))
        except (TypeError, ValueError):
            intrastrand_min_len = 6
        try:
            val = intrastrand_config.get("max_len")
            intrastrand_max_len = int(val) if val is not None and str(val).strip() else None
        except (TypeError, ValueError):
            intrastrand_max_len = None

        intrastrand_mismatches = int(intrastrand_config.get("mismatches", 0) or 0)
        intrastrand_gap_min = int(intrastrand_config.get("gap_min", 0) or 0)
        try:
            val = intrastrand_config.get("gap_max")
            intrastrand_gap_max = int(val) if val is not None and str(val).strip() else None
        except (TypeError, ValueError):
            intrastrand_gap_max = None

        best_i, best_j, best_len = find_longest_intrastrand_complement(
            analysis, intrastrand_min_len, intrastrand_max_len,
            allowed_mismatches=intrastrand_mismatches,
            gap_min_len=intrastrand_gap_min,
            gap_max_len=intrastrand_gap_max
        )
        if best_len >= intrastrand_min_len:
            intra_len = best_len
            intra_seq = analysis[best_i : best_i + best_len]
            if strand_label == "-":
                intra_start1 = window_end - (best_i + best_len) + 1
                intra_end1 = window_end - best_i
                intra_start2 = window_end - best_j
                intra_end2 = window_end - (best_j - best_len + 1)
            else:
                intra_start1 = window_start + best_i
                intra_end1 = window_start + best_i + best_len - 1
                intra_start2 = window_start + (best_j - best_len + 1)
                intra_end2 = window_start + best_j
        
        if intra_len < intrastrand_min_len or (intrastrand_max_len is not None and intra_len > intrastrand_max_len):
            return None

    # Interstrand complementarity scan
    interstrand_enabled = False
    interstrand_min_len = 6
    interstrand_max_len = None
    inter_len = 0
    inter_seq = ""
    inter_start1 = "NA"
    inter_end1 = "NA"
    inter_start2 = "NA"
    inter_end2 = "NA"
    if interstrand_config and interstrand_config.get("enabled"):
        interstrand_enabled = True
        try:
            interstrand_min_len = max(1, int(interstrand_config.get("min_len", 6) or 6))
        except (TypeError, ValueError):
            interstrand_min_len = 6
        try:
            val = interstrand_config.get("max_len")
            interstrand_max_len = int(val) if val is not None and str(val).strip() else None
        except (TypeError, ValueError):
            interstrand_max_len = None

        interstrand_mismatches = int(interstrand_config.get("mismatches", 0) or 0)
        interstrand_gap_min = int(interstrand_config.get("gap_min", 0) or 0)
        try:
            val = interstrand_config.get("gap_max")
            interstrand_gap_max = int(val) if val is not None and str(val).strip() else None
        except (TypeError, ValueError):
            interstrand_gap_max = None

        best_i, best_j, best_len = find_longest_interstrand_complement(
            analysis, interstrand_min_len, interstrand_max_len,
            allowed_mismatches=interstrand_mismatches,
            gap_min_len=interstrand_gap_min,
            gap_max_len=interstrand_gap_max
        )
        if best_len >= interstrand_min_len:
            inter_len = best_len
            inter_seq = analysis[best_i : best_i + best_len]
            if strand_label == "-":
                inter_start1 = window_end - (best_i + best_len) + 1
                inter_end1 = window_end - best_i
                inter_start2 = window_end - (best_j + best_len) + 1
                inter_end2 = window_end - best_j
            else:
                inter_start1 = window_start + best_i
                inter_end1 = window_start + best_i + best_len - 1
                inter_start2 = window_start + best_j
                inter_end2 = window_start + best_j + best_len - 1

        if inter_len < interstrand_min_len or (interstrand_max_len is not None and inter_len > interstrand_max_len):
            return None

    self_comp_passed = False
    if self_comp_constraints:
        for comp_constraint in self_comp_constraints:
            if check_motif_self_comp(analysis, comp_constraint, max_motif_mismatches):
                self_comp_passed = True
                break
        if not self_comp_passed:
            return None
    if region_filters and not within_region(seq_id, window_start, window_end, region_filters):
        return None
    if motifs:
        matched_seq = get_matched_motif_sequence(analysis, motif_spans)
    else:
        matched_seq = ""
    return {
        "sequence_id": seq_id,
        "strand": strand_label,
        "window_start": window_start,
        "window_end": window_end,
        "window_sequence": display_sequence,  # Preserves case (e.g. combined: fwd=UPPER, rev=lower)
        "motif_sequence": matched_seq,
        "motif_hits": motif_hits,
        "base_percentages": base_percentages,
        "max_runs": max_runs,
        "is_self_complementary": self_comp_passed,
        "palindrome_hairpin_sequence": pal_seq.upper() if pal_len else "NA",
        "palindrome_hairpin_length": pal_len,
        "palindrome_hairpin_start": palindrome_start_coord if palindrome_start_coord is not None else "NA",
        "palindrome_hairpin_end": palindrome_end_coord if palindrome_end_coord is not None else "NA",
        "intrastrand_sequence": intra_seq.upper() if intra_len else "NA",
        "intrastrand_length": intra_len,
        "intrastrand_start1": intra_start1,
        "intrastrand_end1": intra_end1,
        "intrastrand_start2": intra_start2,
        "intrastrand_end2": intra_end2,
        "interstrand_sequence": inter_seq.upper() if inter_len else "NA",
        "interstrand_length": inter_len,
        "interstrand_start1": inter_start1,
        "interstrand_end1": inter_end1,
        "interstrand_start2": inter_start2,
        "interstrand_end2": inter_end2,
    }


def build_hit_record_fast(
    seq_id: str,
    display_sequence: str,
    analysis_sequence: str,
    window_start: int,
    window_end: int,
    motifs: Sequence[MotifConstraint],
    base_items: Sequence[Tuple[str, Tuple[float, float]]],
    repeat_items: Sequence[Tuple[str, int]],
    strand_label: str,
    motif_counts: Sequence[int],
    base_percentages: Sequence[float],
    repeat_runs: Sequence[int],
    pal_start_offset: int,
    pal_end_offset: int,
    palindrome_config: Dict[str, object] | None,
    intrastrand_config: Dict[str, object] | None = None,
    interstrand_config: Dict[str, object] | None = None,
    region_filters: Dict[str, List[Tuple[int, int]]] | None = None,
    self_comp_constraints: Sequence[MotifSelfCompConstraint] | None = None,
    max_motif_mismatches: int = 0,
) -> Dict[str, object] | None:
    self_comp_passed = False
    if self_comp_constraints:
        for comp_constraint in self_comp_constraints:
            if check_motif_self_comp(analysis_sequence, comp_constraint, max_motif_mismatches):
                self_comp_passed = True
                break
        if not self_comp_passed:
            return None
    if region_filters and not within_region(seq_id, window_start, window_end, region_filters):
        return None
    motif_hits: List[Tuple[str, int]] = []
    for idx, constraint in enumerate(motifs):
        # Use the count from Cython directly — no need to re-run motif_matches
        count = motif_counts[idx] if idx < len(motif_counts) else 0
        motif_hits.append((constraint.label, count))
    base_pct_pairs: List[Tuple[str, float]] = []
    for idx, (base, _) in enumerate(base_items):
        pct = float(base_percentages[idx]) if idx < len(base_percentages) else 0.0
        base_pct_pairs.append((base, pct))
    repeat_pairs: List[Tuple[str, int]] = []
    for idx, (base, _) in enumerate(repeat_items):
        run_len = repeat_runs[idx] if idx < len(repeat_runs) else 0
        repeat_pairs.append((base, run_len))
    # Only compute motif spans when there are motifs to report —
    # this avoids repeating the expensive motif_matches() that Cython already performed
    if motifs:
        motif_spans: List[List[Tuple[int, int]]] = []
        for idx, constraint in enumerate(motifs):
            if idx < len(motif_counts) and motif_counts[idx] > 0:
                motif_spans.append(motif_matches(analysis_sequence, constraint, max_motif_mismatches))
            else:
                motif_spans.append([])
        matched_seq = get_matched_motif_sequence(analysis_sequence, motif_spans)
    else:
        matched_seq = ""
    pal_len = pal_end_offset - pal_start_offset if pal_start_offset >= 0 and pal_end_offset > pal_start_offset else 0
    palindrome_enabled = bool(palindrome_config.get("enabled")) if palindrome_config else False
    palindrome_min_len = int(palindrome_config.get("min_len", 0) or 0) if palindrome_config else 0
    if pal_len <= 0:
        palindrome_seq = "NA"
        palindrome_len_value = 0
        palindrome_start_coord: int | str = "NA"
        palindrome_end_coord: int | str = "NA"
    else:
        palindrome_seq = analysis_sequence[pal_start_offset:pal_end_offset]
        palindrome_len_value = pal_len
        if strand_label == "-":
            palindrome_start_coord = window_end - pal_end_offset + 1
            palindrome_end_coord = window_end - pal_start_offset
        else:
            palindrome_start_coord = window_start + pal_start_offset
            palindrome_end_coord = window_start + pal_end_offset - 1
        if palindrome_enabled and palindrome_len_value < max(1, palindrome_min_len):
            return None

    # Intrastrand complementarity scan
    intrastrand_enabled = False
    intrastrand_min_len = 6
    intrastrand_max_len = None
    intra_len = 0
    intra_seq = ""
    intra_start1 = "NA"
    intra_end1 = "NA"
    intra_start2 = "NA"
    intra_end2 = "NA"
    if intrastrand_config and intrastrand_config.get("enabled"):
        intrastrand_enabled = True
        try:
            intrastrand_min_len = max(1, int(intrastrand_config.get("min_len", 6) or 6))
        except (TypeError, ValueError):
            intrastrand_min_len = 6
        try:
            val = intrastrand_config.get("max_len")
            intrastrand_max_len = int(val) if val is not None and str(val).strip() else None
        except (TypeError, ValueError):
            intrastrand_max_len = None

        intrastrand_mismatches = int(intrastrand_config.get("mismatches", 0) or 0)
        intrastrand_gap_min = int(intrastrand_config.get("gap_min", 0) or 0)
        try:
            val = intrastrand_config.get("gap_max")
            intrastrand_gap_max = int(val) if val is not None and str(val).strip() else None
        except (TypeError, ValueError):
            intrastrand_gap_max = None

        best_i, best_j, best_len = find_longest_intrastrand_complement(
            analysis_sequence, intrastrand_min_len, intrastrand_max_len,
            allowed_mismatches=intrastrand_mismatches,
            gap_min_len=intrastrand_gap_min,
            gap_max_len=intrastrand_gap_max
        )
        if best_len >= intrastrand_min_len:
            intra_len = best_len
            intra_seq = analysis_sequence[best_i : best_i + best_len]
            if strand_label == "-":
                intra_start1 = window_end - (best_i + best_len) + 1
                intra_end1 = window_end - best_i
                intra_start2 = window_end - best_j
                intra_end2 = window_end - (best_j - best_len + 1)
            else:
                intra_start1 = window_start + best_i
                intra_end1 = window_start + best_i + best_len - 1
                intra_start2 = window_start + (best_j - best_len + 1)
                intra_end2 = window_start + best_j
        
        if intra_len < intrastrand_min_len or (intrastrand_max_len is not None and intra_len > intrastrand_max_len):
            return None

    # Interstrand complementarity scan
    interstrand_enabled = False
    interstrand_min_len = 6
    interstrand_max_len = None
    inter_len = 0
    inter_seq = ""
    inter_start1 = "NA"
    inter_end1 = "NA"
    inter_start2 = "NA"
    inter_end2 = "NA"
    if interstrand_config and interstrand_config.get("enabled"):
        interstrand_enabled = True
        try:
            interstrand_min_len = max(1, int(interstrand_config.get("min_len", 6) or 6))
        except (TypeError, ValueError):
            interstrand_min_len = 6
        try:
            val = interstrand_config.get("max_len")
            interstrand_max_len = int(val) if val is not None and str(val).strip() else None
        except (TypeError, ValueError):
            interstrand_max_len = None

        interstrand_mismatches = int(interstrand_config.get("mismatches", 0) or 0)
        interstrand_gap_min = int(interstrand_config.get("gap_min", 0) or 0)
        try:
            val = interstrand_config.get("gap_max")
            interstrand_gap_max = int(val) if val is not None and str(val).strip() else None
        except (TypeError, ValueError):
            interstrand_gap_max = None

        best_i, best_j, best_len = find_longest_interstrand_complement(
            analysis_sequence, interstrand_min_len, interstrand_max_len,
            allowed_mismatches=interstrand_mismatches,
            gap_min_len=interstrand_gap_min,
            gap_max_len=interstrand_gap_max
        )
        if best_len >= interstrand_min_len:
            inter_len = best_len
            inter_seq = analysis_sequence[best_i : best_i + best_len]
            if strand_label == "-":
                inter_start1 = window_end - (best_i + best_len) + 1
                inter_end1 = window_end - best_i
                inter_start2 = window_end - (best_j + best_len) + 1
                inter_end2 = window_end - best_j
            else:
                inter_start1 = window_start + best_i
                inter_end1 = window_start + best_i + best_len - 1
                inter_start2 = window_start + best_j
                inter_end2 = window_start + best_j + best_len - 1

        if inter_len < interstrand_min_len or (interstrand_max_len is not None and inter_len > interstrand_max_len):
            return None

    return {
        "sequence_id": seq_id,
        "strand": strand_label,
        "window_start": window_start,
        "window_end": window_end,
        "window_sequence": display_sequence,
        "motif_sequence": matched_seq,
        "motif_hits": motif_hits,
        "base_percentages": base_pct_pairs,
        "max_runs": repeat_pairs,
        "is_self_complementary": self_comp_passed,
        "palindrome_hairpin_sequence": palindrome_seq.upper(),
        "palindrome_hairpin_length": palindrome_len_value,
        "palindrome_hairpin_start": palindrome_start_coord,
        "palindrome_hairpin_end": palindrome_end_coord,
        "intrastrand_sequence": intra_seq.upper() if intra_len else "NA",
        "intrastrand_length": intra_len,
        "intrastrand_start1": intra_start1,
        "intrastrand_end1": intra_end1,
        "intrastrand_start2": intra_start2,
        "intrastrand_end2": intra_end2,
        "interstrand_sequence": inter_seq.upper() if inter_len else "NA",
        "interstrand_length": inter_len,
        "interstrand_start1": inter_start1,
        "interstrand_end1": inter_end1,
        "interstrand_start2": inter_start2,
        "interstrand_end2": inter_end2,
    }


def fast_scan_sequence(
    seq_id: str,
    display_sequence: str,
    analysis_sequence: str,
    window: int,
    step: int,
    motifs: Sequence[MotifConstraint],
    base_constraints: Dict[str, Tuple[float, float]],
    repeat_constraints: Dict[str, int],
    strand_label: str = "+",
    coord_transform: Callable[[int, int], Tuple[int, int]] | None = None,
    exclude_motifs: Sequence[ExcludeMotifConstraint] | None = None,
    palindrome_config: Dict[str, object] | None = None,
    intrastrand_config: Dict[str, object] | None = None,
    interstrand_config: Dict[str, object] | None = None,
    region_filters: Dict[str, List[Tuple[int, int]]] | None = None,
    non_overlapping: bool = False,
    candidate_positions: List[int] | None = None,
    self_comp_constraints: Sequence[MotifSelfCompConstraint] | None = None,
    max_motif_mismatches: int = 0,
) -> List[Dict[str, object]]:
    if cython_scan_windows_fast is None:
        return []
    base_items = list(base_constraints.items())
    repeat_items = list(repeat_constraints.items())
    motif_patterns = tuple(constraint.pattern for constraint in motifs)
    motif_required = tuple(constraint.required_count for constraint in motifs)
    exclude_patterns = tuple(ex.pattern for ex in exclude_motifs) if exclude_motifs else ()
    pal_enabled = bool(palindrome_config.get("enabled")) if palindrome_config else False
    pal_min_len = int(palindrome_config.get("min_len", 0) or 0) if palindrome_config else 0
    fast_rows = cython_scan_windows_fast(
        analysis_sequence,
        window,
        step,
        motif_patterns,
        motif_required,
        exclude_patterns,
        tuple((base, limits[0], limits[1]) for base, limits in base_items),
        tuple((base, limit) for base, limit in repeat_items),
        pal_enabled,
        pal_min_len,
        bool(non_overlapping),
        candidate_positions,
    )
    hits: List[Dict[str, object]] = []
    for (
        start_idx,
        end_idx,
        motif_counts,
        base_pct_values,
        repeat_run_values,
        pal_rel_start,
        pal_rel_end,
    ) in fast_rows:
        window_seq = display_sequence[start_idx:end_idx]
        analysis_window = analysis_sequence[start_idx:end_idx]
        if coord_transform:
            window_start, window_end = coord_transform(start_idx, end_idx)
        else:
            window_start, window_end = start_idx + 1, end_idx
        record = build_hit_record_fast(
            seq_id,
            window_seq,
            analysis_window,
            window_start,
            window_end,
            motifs,
            base_items,
            repeat_items,
            strand_label,
            motif_counts,
            base_pct_values,
            repeat_run_values,
            pal_rel_start,
            pal_rel_end,
            palindrome_config or {},
            intrastrand_config=intrastrand_config,
            interstrand_config=interstrand_config,
            region_filters=region_filters,
            self_comp_constraints=self_comp_constraints,
            max_motif_mismatches=max_motif_mismatches,
        )
        if record:
            hits.append(record)
    return hits


def _gpu_complement_narrow(
    sequence: str,
    window: int,
    step: int,
    candidate_positions: List[int] | None,
    intrastrand_config: Dict[str, object] | None,
    interstrand_config: Dict[str, object] | None,
    runtime_options: RuntimeOptions | None,
) -> List[int] | None:
    """Optionally tighten candidate windows by the require-* complementarity bound
    on the GPU. Exact match to the CPU require gate (the CPU recomputes the
    reported coordinates), so it only ever drops windows that cannot pass. Any
    failure leaves the candidates untouched."""
    if not (runtime_options and runtime_options.use_gpu and runtime_options.gpu_kernels):
        return candidate_positions

    def _opt_int(value):
        try:
            return int(value) if value is not None and str(value).strip() != "" else None
        except (ValueError, TypeError):
            return None

    specs: List[Tuple[str, int, int | None, int, int, int | None]] = []
    for cfg, kind in ((intrastrand_config, "intra"), (interstrand_config, "inter")):
        if cfg and cfg.get("enabled"):
            min_len = int(cfg.get("min_len", 0) or 0)
            if min_len <= 0:
                continue
            specs.append((
                kind, min_len, _opt_int(cfg.get("max_len")),
                int(cfg.get("mismatches", 0) or 0),
                int(cfg.get("gap_min", 0) or 0), _opt_int(cfg.get("gap_max")),
            ))
    if not specs:
        return candidate_positions
    try:
        import mako_gpu

        if not mako_gpu.kernels_available():
            return candidate_positions
        starts = candidate_positions
        if starts is None:
            seq_len = len(sequence)
            if seq_len < window:
                return candidate_positions
            starts = list(range(0, seq_len - window + 1, step))
        for kind, min_len, max_len, mismatches, gap_min, gap_max in specs:
            filtered = mako_gpu.complement_require_prefilter(
                sequence, window, starts, kind, min_len, max_len, mismatches, gap_min, gap_max,
            )
            if filtered is not None:
                starts = filtered
        return starts
    except Exception:  # pragma: no cover - any GPU issue → keep CPU candidates
        return candidate_positions


def scan_sequence(
    seq_id: str,
    sequence: str,
    window: int,
    step: int,
    motifs: Sequence[MotifConstraint],
    base_constraints: Dict[str, Tuple[float, float]],
    repeat_constraints: Dict[str, int],
    strand_label: str = "+",
    coord_transform: Callable[[int, int], Tuple[int, int]] | None = None,
    exclude_motifs: Sequence[ExcludeMotifConstraint] | None = None,
    palindrome_config: Dict[str, object] | None = None,
    intrastrand_config: Dict[str, object] | None = None,
    interstrand_config: Dict[str, object] | None = None,
    region_filters: Dict[str, List[Tuple[int, int]]] | None = None,
    non_overlapping: bool = False,
    runtime_options: RuntimeOptions | None = None,
    max_motif_mismatches: int = 0,
    self_comp_constraints: Sequence[MotifSelfCompConstraint] | None = None,
) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    seq_len = len(sequence)
    if seq_len < window:
        return hits
    display_source = sequence
    analysis_source = sequence.upper()
    if "U" in analysis_source:  # RNA: analyse uracil as thymine across all paths
        analysis_source = analysis_source.replace("U", "T")
    candidate_positions = build_candidate_positions(
        analysis_source,
        window,
        step,
        base_constraints,
        runtime_options,
        repeat_constraints=repeat_constraints,
        # Fixed-length motifs can be exactly superset-prefiltered on the GPU; only
        # pass them when there is no per-motif mismatch tolerance (the regex/CPU
        # path owns the mismatch case).
        motifs=motifs if max_motif_mismatches <= 0 else None,
    )
    candidate_positions = _gpu_complement_narrow(
        analysis_source, window, step, candidate_positions,
        intrastrand_config, interstrand_config, runtime_options,
    )
    if max_motif_mismatches <= 0 and cython_scan_windows_fast is not None:
        return fast_scan_sequence(
            seq_id,
            display_source,
            analysis_source,
            window,
            step,
            motifs,
            base_constraints,
            repeat_constraints,
            strand_label=strand_label,
            coord_transform=coord_transform,
            exclude_motifs=exclude_motifs,
            palindrome_config=palindrome_config,
            intrastrand_config=intrastrand_config,
            interstrand_config=interstrand_config,
            region_filters=region_filters,
            non_overlapping=non_overlapping,
            candidate_positions=candidate_positions,
            self_comp_constraints=self_comp_constraints,
            max_motif_mismatches=max_motif_mismatches,
        )
    prefix_counts = build_prefix_counts(analysis_source) if base_constraints else None
    limit = seq_len - window + 1
    last_end = -1
    if candidate_positions is not None:
        start_iter: Iterable[int] = candidate_positions
    else:
        start_iter = range(0, limit, step)
    for start in start_iter:
        if start < 0 or start + window > seq_len:
            continue
        end = start + window
        window_seq = display_source[start:end]
        analysis_window = analysis_source[start:end]
        window_start, window_end = coord_transform(start, end) if coord_transform else (start + 1, end)
        hit = build_hit_record(
            seq_id,
            window_seq,
            analysis_window,
            window_start,
            window_end,
            motifs,
            base_constraints,
            repeat_constraints,
            strand_label,
            prefix_counts,
            (start, end) if prefix_counts else None,
            exclude_motifs=exclude_motifs,
            palindrome_config=palindrome_config,
            intrastrand_config=intrastrand_config,
            interstrand_config=interstrand_config,
            region_filters=region_filters,
            max_motif_mismatches=max_motif_mismatches,
            self_comp_constraints=self_comp_constraints,
        )
        if hit:
            if non_overlapping and last_end >= 0 and hit["window_start"] <= last_end:
                continue
            if non_overlapping:
                last_end = hit["window_end"]
            hits.append(hit)
    return hits


def _reverse_coord_transform(seq_len: int) -> Callable[[int, int], Tuple[int, int]]:
    def transform(start: int, end: int) -> Tuple[int, int]:
        window_start = seq_len - end + 1
        window_end = seq_len - start
        return window_start, window_end

    return transform


def scan_combined_windows(
    seq_id: str,
    sequence: str,
    forward_len: int,
    reverse_len: int,
    overlap: int,
    step: int,
    motifs: Sequence[MotifConstraint],
    base_constraints: Dict[str, Tuple[float, float]],
    repeat_constraints: Dict[str, int],
    exclude_motifs: Sequence[ExcludeMotifConstraint] | None = None,
    palindrome_config: Dict[str, object] | None = None,
    intrastrand_config: Dict[str, object] | None = None,
    interstrand_config: Dict[str, object] | None = None,
    region_filters: Dict[str, List[Tuple[int, int]]] | None = None,
    non_overlapping: bool = False,
    max_motif_mismatches: int = 0,
    self_comp_constraints: Sequence[MotifSelfCompConstraint] | None = None,
) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    if forward_len <= 0 or reverse_len <= 0:
        return hits
    if overlap < 0 or overlap > min(forward_len, reverse_len):
        return hits
    combined_span = forward_len + reverse_len - overlap
    if combined_span <= 0:
        return hits
    seq_len = len(sequence)
    limit = seq_len - combined_span + 1
    if limit <= 0:
        return hits
    last_end = -1
    for start in range(0, limit, step):
        forward_part = sequence[start : start + forward_len].upper()
        reverse_region_start = start + forward_len - overlap
        reverse_region_end = reverse_region_start + reverse_len
        reverse_source = sequence[reverse_region_start:reverse_region_end]
        reverse_part = reverse_complement(reverse_source).lower()
        display_seq = forward_part + reverse_part
        analysis_seq = (forward_part + reverse_part).upper()

        # Enforce "desired motif in both strands" as requested.
        # This excludes windows that would only match a forward-only or reverse-only scan.
        if motifs:
            match_both = True
            pre_spans: List[List[Tuple[int, int]]] = []
            for constraint in motifs:
                fwd_spans = motif_matches(forward_part, constraint, max_motif_mismatches)
                rev_spans = motif_matches(reverse_part.upper(), constraint, max_motif_mismatches)
                if not fwd_spans or not rev_spans:
                    match_both = False
                    break
                # Shift reverse spans to match analysis_seq (forward_part + reverse_part)
                offset = len(forward_part)
                shifted_rev = [(s + offset, e + offset) for s, e in rev_spans]
                pre_spans.append(fwd_spans + shifted_rev)
            if not match_both:
                continue
        else:
            pre_spans = None

        window_start = start + 1
        window_end = reverse_region_end
        # Strand-specific coordinate ranges for annotation
        fwd_start = start + 1
        fwd_end = start + forward_len
        rev_start = reverse_region_start + 1
        rev_end = reverse_region_end
        hit = build_hit_record(
            seq_id,
            display_seq,
            analysis_seq,
            window_start,
            window_end,
            motifs,
            base_constraints,
            repeat_constraints,
            "combined",
            exclude_motifs=exclude_motifs,
            palindrome_config=palindrome_config,
            intrastrand_config=intrastrand_config,
            interstrand_config=interstrand_config,
            region_filters=region_filters,
            max_motif_mismatches=max_motif_mismatches,
            self_comp_constraints=self_comp_constraints,
            precalculated_motif_spans=pre_spans,
        )
        if hit:
            hit["forward_start"] = fwd_start
            hit["forward_end"] = fwd_end
            hit["reverse_start"] = rev_start
            hit["reverse_end"] = rev_end
            if non_overlapping and last_end >= 0 and hit["window_start"] <= last_end:
                continue
            if non_overlapping:
                last_end = hit["window_end"]
            hits.append(hit)
    return hits


def scan_with_config(
    seq_id: str,
    sequence: str,
    config: Dict[str, object],
    motifs: Sequence[MotifConstraint],
    base_constraints: Dict[str, Tuple[float, float]],
    repeat_constraints: Dict[str, int],
    exclude_motifs: Sequence[ExcludeMotifConstraint] | None = None,
) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    strands = set(config.get("strands", []))
    if not strands:
        strands.add("forward")
    if "both" in strands:
        strands.update({"forward", "reverse"})
        strands.discard("both")
    window = int(config["window"])
    step = int(config["step"])
    motifs_forward = config.get("motifs_forward") or []
    motifs_reverse = config.get("motifs_reverse") or []
    region_filters = config.get("region_filters")
    non_overlapping = bool(config.get("non_overlapping"))
    palindrome_config = {
        "enabled": bool(config.get("palindrome_required")),
        "min_len": int(config.get("palindrome_min_len", 0) or 0),
    }
    intrastrand_config = {
        "enabled": bool(config.get("intrastrand_required")),
        "min_len": int(config.get("intrastrand_min_len", 0) or 0),
        "max_len": config.get("intrastrand_max_len"),
        "mismatches": int(config.get("intrastrand_mismatches", 0) or 0),
        "gap_min": int(config.get("intrastrand_gap_min", 0) or 0),
        "gap_max": config.get("intrastrand_gap_max"),
    }
    if intrastrand_config["max_len"] is not None:
        try:
            intrastrand_config["max_len"] = int(intrastrand_config["max_len"])
        except (ValueError, TypeError):
            intrastrand_config["max_len"] = None
    if intrastrand_config["gap_max"] is not None:
        try:
            intrastrand_config["gap_max"] = int(intrastrand_config["gap_max"])
        except (ValueError, TypeError):
            intrastrand_config["gap_max"] = None

    interstrand_config = {
        "enabled": bool(config.get("interstrand_required")),
        "min_len": int(config.get("interstrand_min_len", 0) or 0),
        "max_len": config.get("interstrand_max_len"),
        "mismatches": int(config.get("interstrand_mismatches", 0) or 0),
        "gap_min": int(config.get("interstrand_gap_min", 0) or 0),
        "gap_max": config.get("interstrand_gap_max"),
    }
    if interstrand_config["max_len"] is not None:
        try:
            interstrand_config["max_len"] = int(interstrand_config["max_len"])
        except (ValueError, TypeError):
            interstrand_config["max_len"] = None
    if interstrand_config["gap_max"] is not None:
        try:
            interstrand_config["gap_max"] = int(interstrand_config["gap_max"])
        except (ValueError, TypeError):
            interstrand_config["gap_max"] = None

    runtime_options: RuntimeOptions | None = config.get("runtime_options")
    max_motif_mismatches = int(config.get("max_motif_mismatches", 0) or 0)
    self_comp_constraints = config.get("self_comp_constraints")
    if "forward" in strands:
        hits.extend(
            scan_sequence(
                seq_id,
                sequence,
                window,
                step,
                motifs_forward or motifs,
                base_constraints,
                repeat_constraints,
                strand_label="+",
                exclude_motifs=exclude_motifs,
                palindrome_config=palindrome_config,
                intrastrand_config=intrastrand_config,
                interstrand_config=interstrand_config,
                region_filters=region_filters,
                non_overlapping=non_overlapping,
                runtime_options=runtime_options,
                max_motif_mismatches=max_motif_mismatches,
                self_comp_constraints=self_comp_constraints,
            )
        )
    if "reverse" in strands:
        rc_seq = reverse_complement(sequence)
        hits.extend(
            scan_sequence(
                seq_id,
                rc_seq,
                window,
                step,
                motifs_reverse or motifs,
                base_constraints,
                repeat_constraints,
                strand_label="-",
                coord_transform=_reverse_coord_transform(len(sequence)),
                exclude_motifs=exclude_motifs,
                palindrome_config=palindrome_config,
                intrastrand_config=intrastrand_config,
                interstrand_config=interstrand_config,
                region_filters=region_filters,
                non_overlapping=non_overlapping,
                runtime_options=runtime_options,
                max_motif_mismatches=max_motif_mismatches,
                self_comp_constraints=self_comp_constraints,
            )
        )
    if "combined" in strands:
        hits.extend(
            scan_combined_windows(
                seq_id,
                sequence,
                int(config["combined_forward_len"]),
                int(config["combined_reverse_len"]),
                int(config["combined_overlap"]),
                step,
                motifs,
                base_constraints,
                repeat_constraints,
                exclude_motifs=exclude_motifs,
                palindrome_config=palindrome_config,
                intrastrand_config=intrastrand_config,
                interstrand_config=interstrand_config,
                region_filters=region_filters,
                non_overlapping=non_overlapping,
                max_motif_mismatches=max_motif_mismatches,
                self_comp_constraints=self_comp_constraints,
            )
        )
    return hits


def format_kv_pairs(pairs: Sequence[Tuple[str, object]], formatter) -> str:
    if not pairs:
        return "NA"
    return ";".join(formatter(key, value) for key, value in pairs)


def index_fasta_records(
    fasta_path: Path, record_prefixes: Sequence[str] | None
) -> List[Tuple[str, int, int]]:
    """Byte-offset index of matching FASTA records: (seq_id, data_start, data_end).

    ``data_start``/``data_end`` bound the record's sequence lines (header excluded).
    Building this lets each worker read its own sequence from disk, so we never
    ship 250 MB chromosomes through multiprocessing pipes (which exhausts Windows
    I/O resources -> WinError 1450). One pass, negligible memory.
    """
    normalized = tuple(p.upper() for p in record_prefixes) if record_prefixes else None
    records: List[Tuple[str, int, int]] = []
    cur_id: str | None = None
    cur_start = 0
    offset = 0
    with open(fasta_path, "rb") as handle:
        for raw in handle:
            n = len(raw)
            stripped = raw.strip()
            if stripped[:1] == b">":
                if cur_id is not None:
                    records.append((cur_id, cur_start, offset))
                header = stripped.decode("utf-8", "replace")
                include = normalized is None or any(header.upper().startswith(p) for p in normalized)
                if include and len(header) > 1:
                    cur_id = header[1:].split()[0]
                    cur_start = offset + n
                else:
                    cur_id = None
            offset += n
        if cur_id is not None:
            records.append((cur_id, cur_start, offset))
    return records


def read_record_sequence(fasta_path: Path | str, data_start: int, data_end: int) -> str:
    """Read and normalise one record's sequence, identically to iter_nc_sequences
    (each line stripped + uppercased, blank/header lines skipped, concatenated)."""
    with open(fasta_path, "rb") as handle:
        handle.seek(data_start)
        raw = handle.read(data_end - data_start)
    chunks: List[str] = []
    for line in raw.split(b"\n"):
        stripped = line.strip()
        if stripped and stripped[:1] != b">":
            chunks.append(stripped.decode("utf-8", "replace").upper())
    return "".join(chunks)


def _scan_sequence_file_worker(payload):
    """Worker that reads its sequence from the FASTA (tiny payload, no big pipe
    transfer), then scans it."""
    (
        fasta_path,
        seq_id,
        data_start,
        data_end,
        scan_config,
        motifs,
        base_constraints,
        repeat_constraints,
        exclude_motifs,
    ) = payload
    sequence = read_record_sequence(fasta_path, data_start, data_end)
    hits = scan_with_config(
        seq_id, sequence, scan_config, motifs, base_constraints,
        repeat_constraints, exclude_motifs=exclude_motifs,
    )
    return seq_id, hits


def determine_is_rna_records(
    sequence_type: str, fasta_path: Path, records: Sequence[Tuple[str, int, int]]
) -> bool:
    """RNA-alphabet decision when sequences live on disk (sample the first records)."""
    if sequence_type == "rna":
        return True
    if sequence_type == "dna":
        return False
    for seq_id, start, end in records[:3]:
        sample = read_record_sequence(fasta_path, start, min(end, start + 100_000))
        if sample.count("U") > 0 and sample.count("U") >= sample.count("T"):
            return True
    return False


_RNA_SEQUENCE_FIELDS = (
    "window_sequence",
    "motif_sequence",
    "palindrome_hairpin_sequence",
    "intrastrand_sequence",
    "interstrand_sequence",
)


def determine_is_rna(sequence_type: str, sequences: Sequence[Tuple[str, str]]) -> bool:
    """Decide whether output sequences should use the RNA (U) alphabet."""
    if sequence_type == "rna":
        return True
    if sequence_type == "dna":
        return False
    # auto: sample the loaded sequences for uracil.
    for _seq_id, seq in sequences[:5]:
        sample = seq[:100000].upper()
        if sample.count("U") > 0 and sample.count("U") >= sample.count("T"):
            return True
    return False


def apply_rna_alphabet(rows: Sequence[Dict[str, object]]) -> None:
    """Render sequence-bearing output columns in the RNA alphabet (T->U).

    Analysis is always done in T-space, so derived sequences (motifs, reverse
    strand, complement regions) come back as T; this converts them to U for the
    report when the input is RNA."""
    for row in rows:
        for field_name in _RNA_SEQUENCE_FIELDS:
            value = row.get(field_name)
            if isinstance(value, str) and value and value != "NA":
                row[field_name] = value.replace("T", "U").replace("t", "u")


class StreamingHitWriter:
    """Append hit rows to a JSONL sidecar, flushing+fsyncing periodically.

    The JSONL is the crash-survival artifact: every line is a complete, valid
    hit record, so an interrupted run leaves a readable file. ``tell()`` returns
    the on-disk byte offset after a flush, which the checkpoint records so a
    resumed run can truncate any partially written tail.
    """

    def __init__(self, path: Path, flush_every: int = 5000, append: bool = False) -> None:
        self.path = path
        self.flush_every = max(1, int(flush_every))
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a" if append else "w", encoding="utf-8", newline="\n")
        self._buffer: List[str] = []
        self.written = 0

    def write(self, row: Dict[str, object]) -> None:
        self._buffer.append(json.dumps(row, separators=(",", ":")))
        self.written += 1
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._buffer:
            self._handle.write("\n".join(self._buffer) + "\n")
            self._buffer.clear()
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def tell(self) -> int:
        self.flush()
        return self._handle.tell()

    def close(self) -> None:
        self.flush()
        self._handle.close()


def scan_params_signature(args: argparse.Namespace) -> str:
    """Stable hash of the search parameters so a resume refuses mismatched runs."""
    import hashlib

    relevant = {
        k: v
        for k, v in sorted(vars(args).items())
        if k not in {"stream_output", "checkpoint", "flush_every", "restart", "workers", "output_name", "output_prefix"}
    }
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def load_checkpoint(path: Path, signature: str) -> Tuple[Set[str], int]:
    """Return (completed_sequence_ids, jsonl_byte_offset) from a matching checkpoint."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set(), 0
    if data.get("signature") != signature:
        print(
            "[stream] existing checkpoint was produced with different search parameters; "
            "ignoring it (use --restart to overwrite).",
            file=sys.stderr,
        )
        return set(), 0
    return set(data.get("completed", [])), int(data.get("jsonl_bytes", 0))


def save_checkpoint(path: Path, signature: str, completed: Set[str], jsonl_bytes: int) -> None:
    """Atomically persist resume state (temp file + replace)."""
    payload = {"signature": signature, "completed": sorted(completed), "jsonl_bytes": jsonl_bytes}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def finalize_stream_to_tsv(jsonl_path: Path, output_dir: Path, output_prefix: str) -> int:
    """Reload the streamed JSONL hits and write the canonical TSV via write_outputs.

    Reusing write_outputs guarantees the finalized TSV is byte-for-byte identical
    to a non-streamed run with the same hits and column rules.
    """
    rows: List[Dict[str, object]] = []
    if jsonl_path.is_file():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    write_outputs(rows, output_dir, output_prefix)
    return len(rows)


def write_outputs(rows: Sequence[Dict[str, object]], output_dir: Path, output_prefix: str) -> None:
    if not rows:
        return

    # Check which columns have useful data to omit unused columns
    def has_useful_data(key: str, check_keys: List[str] = None) -> bool:
        keys_to_check = check_keys if check_keys else [key]
        for row in rows:
            for k in keys_to_check:
                val = row.get(k)
                if val is not None and val not in (0, "0", "NA", "", [], {}):
                    return True
        return False

    headers = ["sequence_id", "strand", "window_start", "window_end", "window_sequence"]

    # Include strand-specific coordinate columns when combined windows are used
    if has_useful_data("forward_start"):
        headers.extend(["forward_start", "forward_end", "reverse_start", "reverse_end"])
    
    show_motif = has_useful_data("motif_sequence") or has_useful_data("motif_hits")
    if show_motif:
        headers.extend(["motif_sequence", "motif_hits"])
        
    if has_useful_data("base_percentages"):
        headers.append("base_content_pct")
        
    if has_useful_data("max_runs"):
        headers.append("max_consecutive_runs")
        
    if has_useful_data("is_self_complementary"):
        headers.append("is_self_complementary")
        
    show_pal = has_useful_data("palindrome_hairpin_length")
    if show_pal:
        headers.extend([
            "palindrome_hairpin_sequence", "palindrome_hairpin_length",
            "palindrome_hairpin_start", "palindrome_hairpin_end"
        ])

    show_intra = has_useful_data("intrastrand_length")
    if show_intra:
        headers.extend([
            "intrastrand_sequence", "intrastrand_length",
            "intrastrand_start1", "intrastrand_end1",
            "intrastrand_start2", "intrastrand_end2"
        ])
        
    show_inter = has_useful_data("interstrand_length")
    if show_inter:
        headers.extend([
            "interstrand_sequence", "interstrand_length",
            "interstrand_start1", "interstrand_end1",
            "interstrand_start2", "interstrand_end2"
        ])

    if any("best_g4hunter" in row for row in rows):
        headers.append("best_g4hunter")

    formatted_rows: List[List[str]] = []
    for row in rows:
        formatted_row = []
        for h in headers:
            if h == "motif_hits":
                formatted_row.append(format_kv_pairs(row["motif_hits"], lambda k, v: f"{k}={v}"))
            elif h == "base_content_pct":
                formatted_row.append(format_kv_pairs(row["base_percentages"], lambda k, v: f"{k}={v:.2f}"))
            elif h == "max_consecutive_runs":
                formatted_row.append(format_kv_pairs(row["max_runs"], lambda k, v: f"{k}={v}"))
            elif h in ("is_self_complementary", "palindrome_hairpin_length", "palindrome_hairpin_start", "palindrome_hairpin_end",
                       "intrastrand_length", "intrastrand_start1", "intrastrand_end1", "intrastrand_start2", "intrastrand_end2",
                       "interstrand_length", "interstrand_start1", "interstrand_end1", "interstrand_start2", "interstrand_end2"):
                val = row.get(h, "NA")
                formatted_row.append(str(val) if val is not None else "NA")
            else:
                formatted_row.append(str(row.get(h, "NA")))
        formatted_rows.append(formatted_row)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_prefix}.tsv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        writer.writerows(formatted_rows)


def _scan_sequence_worker(payload):
    (
        seq_id,
        sequence,
        scan_config,
        motifs,
        base_constraints,
        repeat_constraints,
        exclude_motifs,
    ) = payload
    hits = scan_with_config(
        seq_id,
        sequence,
        scan_config,
        motifs,
        base_constraints,
        repeat_constraints,
        exclude_motifs=exclude_motifs,
    )
    return seq_id, hits


def run_streaming_scan(
    args: argparse.Namespace,
    fasta_path: Path,
    records: Sequence[Tuple[str, int, int]],
    scan_config: Dict[str, object],
    motifs: Sequence[MotifConstraint],
    base_constraints: Dict[str, Tuple[float, float]],
    repeat_constraints: Dict[str, int],
    exclude_motifs: Sequence[ExcludeMotifConstraint] | None,
    output_dir: Path,
    allowed_sequences: Set[str],
    region_filters: Dict[str, List[Tuple[int, int]]],
    chromosome_filter: Set[int] | None,
    worker_count: int,
) -> int:
    """Scan with incremental, crash-safe output and resume support.

    Sequences are processed in a deterministic (FASTA input) order; each one's
    hits are streamed to a JSONL sidecar and a checkpoint is written afterwards,
    so an interrupted run resumes from the next unfinished sequence with no
    duplicate rows. The canonical TSV is produced at the end via write_outputs.
    """
    output_prefix = args.output_prefix
    jsonl_path = output_dir / f"{output_prefix}.partial.jsonl"
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else output_dir / f"{output_prefix}.checkpoint.json"
    )
    signature = scan_params_signature(args)
    is_rna = determine_is_rna_records(args.sequence_type, fasta_path, records)
    min_g4 = args.min_g4hunter
    fasta_str = str(fasta_path)

    # Deterministic work list (same record filters as the standard path). Holds
    # byte offsets, not sequences, so workers read from disk (no big pipe sends).
    work: List[Tuple[str, int, int]] = []
    for seq_id, data_start, data_end in records:
        nid = normalize_seq_name(seq_id)
        if allowed_sequences and nid not in allowed_sequences:
            continue
        if region_filters and args.region and nid not in region_filters:
            continue
        if chromosome_filter and not sequence_matches_chromosome_filter(nid, chromosome_filter):
            continue
        work.append((seq_id, data_start, data_end))

    output_dir.mkdir(parents=True, exist_ok=True)
    completed: Set[str] = set()
    append = False
    if args.restart:
        for stale in (jsonl_path, checkpoint_path):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
    else:
        completed, jsonl_bytes = load_checkpoint(checkpoint_path, signature)
        if completed and jsonl_path.exists():
            with jsonl_path.open("r+b") as handle:
                handle.truncate(jsonl_bytes)  # drop any partially written tail
            append = True
            print(
                f"[stream] resuming: {len(completed)} sequence(s) already done; "
                f"JSONL truncated to {jsonl_bytes} bytes.",
                flush=True,
            )

    pending = [(sid, s, e) for (sid, s, e) in work if sid not in completed]
    writer = StreamingHitWriter(jsonl_path, args.flush_every, append=append)

    def emit(seq_id: str, hits: Sequence[Dict[str, object]]) -> int:
        emitted = 0
        for hit in hits:
            if min_g4 is not None:
                motif_seq = hit.get("motif_sequence") or ""
                score_seq = (
                    motif_seq
                    if isinstance(motif_seq, str) and motif_seq and motif_seq != "NA"
                    else str(hit.get("window_sequence") or "")
                )
                score = g4hunter_best_score(score_seq)
                hit["best_g4hunter"] = round(score, 4)
                if score < min_g4:
                    continue
            if is_rna:
                apply_rna_alphabet([hit])
            writer.write(hit)
            emitted += 1
        return emitted

    total = len(work)
    try:
        if worker_count == 1 or len(pending) <= 1:
            for seq_id, data_start, data_end in pending:
                sequence = read_record_sequence(fasta_path, data_start, data_end)
                hits = scan_with_config(
                    seq_id, sequence, scan_config, motifs, base_constraints,
                    repeat_constraints, exclude_motifs=exclude_motifs,
                )
                n = emit(seq_id, hits)
                completed.add(seq_id)
                save_checkpoint(checkpoint_path, signature, completed, writer.tell())
                print(f"[stream] {len(completed)}/{total} {seq_id}: {n} hits (checkpointed)", flush=True)
        else:
            # Workers read their sequence from disk (tiny payloads -> no Windows
            # WinError 1450), and a FIFO window preserves the deterministic
            # submission/emit order needed for safe resume.
            import collections

            max_in_flight = max(2, worker_count * 2)
            pending_iter = iter(pending)
            fifo: "collections.deque" = collections.deque()
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                def _submit_next_stream() -> bool:
                    try:
                        sid, d_start, d_end = next(pending_iter)
                    except StopIteration:
                        return False
                    fut = executor.submit(
                        _scan_sequence_file_worker,
                        (fasta_str, sid, d_start, d_end, scan_config, motifs,
                         base_constraints, repeat_constraints, exclude_motifs),
                    )
                    fifo.append((sid, fut))
                    return True

                for _ in range(max_in_flight):
                    if not _submit_next_stream():
                        break
                while fifo:
                    seq_id, future = fifo.popleft()  # next in submission order
                    _sid, hits = future.result()
                    n = emit(seq_id, hits)
                    completed.add(seq_id)
                    save_checkpoint(checkpoint_path, signature, completed, writer.tell())
                    print(f"[stream] {len(completed)}/{total} {seq_id}: {n} hits (checkpointed)", flush=True)
                    _submit_next_stream()
    finally:
        writer.close()

    n_rows = finalize_stream_to_tsv(jsonl_path, output_dir, output_prefix)
    print(f"[stream] finalized {n_rows} hits to {output_dir / (output_prefix + '.tsv')}.")
    print(f"[stream] crash-safe sidecar: {jsonl_path} | checkpoint: {checkpoint_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be a positive integer.")
    if args.step <= 0:
        parser.error("--step must be a positive integer.")
    if args.palindrome_min_len < 0:
        parser.error("--palindrome-min-len must be non-negative.")
    if args.motif_mismatches < 0:
        parser.error("--motif-mismatches must be non-negative.")
    fasta_path = Path(args.fasta)
    if not fasta_path.is_file():
        parser.error(f"FASTA file '{args.fasta}' does not exist.")
    strand_values = args.strand_selections or ["forward"]
    strand_modes = set()
    for value in strand_values:
        if value == "both":
            strand_modes.update({"forward", "reverse"})
        else:
            strand_modes.add(value)
    if not strand_modes:
        strand_modes.add("forward")
    combined_forward_len = args.combined_forward_len or args.window
    combined_reverse_len = args.combined_reverse_len or args.window
    combined_overlap = args.combined_overlap
    if "combined" in strand_modes:
        if combined_forward_len <= 0 or combined_reverse_len <= 0:
            parser.error("Combined forward/reverse lengths must be positive when scanning combined windows.")
        max_overlap = min(combined_forward_len, combined_reverse_len)
        if combined_overlap < 0 or combined_overlap > max_overlap:
            parser.error(f"--combined-overlap must be between 0 and {max_overlap}.")
    profile_values = parse_profile_specs(args.chemistry_profile, parser)
    motif_specs = list(profile_values.get("motifs", [])) + (args.motif or [])
    motifs_forward_specs = list(profile_values.get("motifs_forward", [])) + (args.motif_forward or [])
    motifs_reverse_specs = list(profile_values.get("motifs_reverse", [])) + (args.motif_reverse or [])
    base_content_specs = list(profile_values.get("base_content", [])) + (args.base_content or [])
    repeat_specs = list(profile_values.get("max_repeat", [])) + (args.max_repeat or [])
    motifs = parse_motif_specs(parser, motif_specs)
    motifs_forward = parse_motif_specs(parser, motifs_forward_specs) if motifs_forward_specs else []
    motifs_reverse = parse_motif_specs(parser, motifs_reverse_specs) if motifs_reverse_specs else []
    base_constraints = parse_base_content_specs(parser, base_content_specs)
    repeat_constraints = parse_max_repeat_specs(parser, repeat_specs)
    exclude_motifs = parse_exclude_motifs(parser, args.exclude_motif or [])
    self_comp_constraints = parse_motif_self_comp_specs(parser, args.motif_self_comp or [])
    # Use explicitly requested classes, or None (meaning scan all)
    sequence_classes = args.sequence_classes or None
    record_prefixes = prefixes_for_classes(sequence_classes)
    # Index record byte-offsets instead of loading the whole genome into RAM.
    # Workers read their own sequence from disk, so nothing large goes through
    # the multiprocessing pipes (avoids Windows WinError 1450 on big chromosomes).
    records = index_fasta_records(fasta_path, record_prefixes)
    allowed_sequences = {normalize_seq_name(seq) for seq in args.sequence_id if seq}
    allowed_sequences = {seq for seq in allowed_sequences if seq}
    region_filters = parse_region_specs(parser, args.region) if args.region else {}
    chromosome_numbers = parse_chromosome_numbers(parser, args.chromosome_numbers)
    chromosome_filter: Set[int] | None = chromosome_numbers if args.chromosome_numbers else None
    worker_count = args.workers
    if worker_count == 0:
        worker_count = os.cpu_count() or 1
    worker_count = max(1, worker_count)
    all_hits: List[Dict[str, object]] = []
    gpu_caps = detect_gpu_capabilities()
    requested_engine = getattr(args, "engine", "auto")
    if requested_engine == "gpu" and not gpu_caps["available"] and not (args.use_gpu and requested_engine == "auto"):
        print(
            f"[engine] GPU requested but unavailable ({gpu_caps['reason']}); falling back to CPU.",
            file=sys.stderr,
        )
    engine = resolve_engine(requested_engine, bool(args.use_gpu), gpu_caps)
    if gpu_caps["available"]:
        vram_mb = int(gpu_caps["total_mem_bytes"]) // (1024 * 1024)
        print(
            f"[engine] mode={engine} gpu='{gpu_caps['device']}' "
            f"cc={gpu_caps['compute_capability']} vram={vram_mb}MB",
            flush=True,
        )
    else:
        print(f"[engine] mode={engine} (no CUDA device: {gpu_caps['reason']})", flush=True)
    if engine == "gpu":
        try:
            import mako_gpu

            if not mako_gpu.kernels_available():
                print(
                    "[engine] GPU selected but CUDA kernels could not be compiled "
                    f"({mako_gpu.kernels_status().get('kernel_reason')}); the scan "
                    "will run on the CPU (results are identical, just not accelerated).",
                    file=sys.stderr,
                )
                print(f"[engine] Fix: {mako_gpu.KERNEL_FIX_HINT}", file=sys.stderr)
        except Exception:
            pass
        # GPU acceleration here is a candidate *prefilter*, not a full offload, so
        # it only engages when the search has something it can pre-filter:
        # base-content always, or (with --gpu-kernels) repeat caps / fixed-length
        # motifs / require-complement. Tell the user when the GPU will sit idle.
        gpu_kernels_on = bool(getattr(args, "gpu_kernels", False))
        gpu_eligible = bool(base_constraints)
        if gpu_kernels_on:
            gpu_eligible = (
                gpu_eligible
                or bool(repeat_constraints)
                or bool(_extract_fixed_motifs(motifs))
                or bool(getattr(args, "require_intrastrand", False))
                or bool(getattr(args, "require_interstrand", False))
            )
        if not gpu_eligible:
            if not gpu_kernels_on:
                print(
                    "[engine] GPU will be IDLE for this search: without --gpu-kernels the GPU "
                    "only accelerates --base-content filtering, which this search does not use. "
                    "Tick 'Use custom CUDA kernels' (--gpu-kernels) to offload fixed-motif / "
                    "repeat-cap prefiltering. Note: the motif scan itself always runs on the "
                    "multi-core CPU (Cython) path; the GPU only narrows candidate windows.",
                    file=sys.stderr,
                )
            else:
                print(
                    "[engine] GPU has no offloadable filter in this search (no base-content / "
                    "repeat cap / fixed-length motif / require-complement); the scan runs on "
                    "the CPU path.",
                    file=sys.stderr,
                )
        elif gpu_kernels_on and worker_count > 4:
            print(
                f"[engine] Note: --gpu-kernels with {worker_count} workers makes each worker "
                "process build its own CUDA context (extra VRAM + first-use kernel compile) "
                "and serialise on the single GPU. For GPU runs, fewer workers (e.g. "
                "--workers 2-4) is usually faster.",
                file=sys.stderr,
            )
    runtime_options = RuntimeOptions(
        use_gpu=(engine == "gpu"),
        vectorized_base=bool(args.vectorized_base),
        gpu_kernels=bool(getattr(args, "gpu_kernels", False)),
    )
    scan_config = {
        "window": args.window,
        "step": args.step,
        "strands": tuple(sorted(strand_modes)),
        "combined_forward_len": combined_forward_len,
        "combined_reverse_len": combined_reverse_len,
        "combined_overlap": combined_overlap,
        "motifs": motifs,
        "motifs_forward": motifs_forward,
        "motifs_reverse": motifs_reverse,
        "non_overlapping": bool(args.non_overlapping),
        "region_filters": region_filters,
        "palindrome_required": bool(args.require_palindrome),
        "palindrome_min_len": args.palindrome_min_len,
        "intrastrand_required": bool(args.require_intrastrand),
        "intrastrand_min_len": args.intrastrand_min_len,
        "intrastrand_max_len": args.intrastrand_max_len,
        "intrastrand_mismatches": args.intrastrand_mismatches,
        "intrastrand_gap_min": args.intrastrand_gap_min,
        "intrastrand_gap_max": args.intrastrand_gap_max,
        "interstrand_required": bool(args.require_interstrand),
        "interstrand_min_len": args.interstrand_min_len,
        "interstrand_max_len": args.interstrand_max_len,
        "interstrand_mismatches": args.interstrand_mismatches,
        "interstrand_gap_min": args.interstrand_gap_min,
        "interstrand_gap_max": args.interstrand_gap_max,
        "runtime_options": runtime_options,
        "max_motif_mismatches": int(args.motif_mismatches),
        "self_comp_constraints": self_comp_constraints,
    }
    output_dir = Path(f"output_{args.output_name}")
    if not records:
        write_outputs(all_hits, output_dir, args.output_prefix)
        class_text = ", ".join(sequence_classes) if sequence_classes else "all"
        print(
            f"No sequences matched the requested record classes ({class_text}); no output rows written.",
            file=sys.stderr,
        )
        return 0
    if args.stream_output or args.checkpoint:
        return run_streaming_scan(
            args, fasta_path, records, scan_config, motifs, base_constraints, repeat_constraints,
            exclude_motifs, output_dir, allowed_sequences, region_filters,
            chromosome_filter, worker_count,
        )

    # Apply per-record filters once (sequence-id / region / chromosome).
    work: List[Tuple[str, int, int]] = []
    for seq_id, data_start, data_end in records:
        normalized_id = normalize_seq_name(seq_id)
        if allowed_sequences and normalized_id not in allowed_sequences:
            continue
        if region_filters and args.region and normalized_id not in region_filters:
            continue
        if chromosome_filter and not sequence_matches_chromosome_filter(normalized_id, chromosome_filter):
            continue
        work.append((seq_id, data_start, data_end))
    total_records = len(work)
    total_bases = sum(end - start for _, start, end in work) or 1
    completed_bases = 0
    print(
        f"[scanner] Processing {total_records} sequences with {worker_count} worker(s)...",
        flush=True,
    )
    if worker_count == 1 or total_records <= 1:
        for index, (seq_id, data_start, data_end) in enumerate(work, 1):
            sequence = read_record_sequence(fasta_path, data_start, data_end)
            seq_hits = scan_with_config(
                seq_id, sequence, scan_config, motifs, base_constraints,
                repeat_constraints, exclude_motifs=exclude_motifs,
            )
            all_hits.extend(seq_hits)
            completed_bases += data_end - data_start
            percent = (completed_bases / total_bases) * 100
            print(
                f"[scanner] Progress: {percent:.2f}% - Completed {seq_id}: "
                f"{len(seq_hits)} hits ({index}/{total_records})",
                flush=True,
            )
    else:
        # Workers read their own sequence from disk, so payloads are a few bytes
        # and no large chromosome ever crosses a multiprocessing pipe (this is the
        # fix for Windows WinError 1450). In-flight tasks are still bounded.
        max_in_flight = max(2, worker_count * 2)
        work_iter = iter(work)
        done_index = 0
        fasta_str = str(fasta_path)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            in_flight: Dict[object, Tuple[str, int]] = {}

            def _submit_next() -> bool:
                try:
                    sid, d_start, d_end = next(work_iter)
                except StopIteration:
                    return False
                fut = executor.submit(
                    _scan_sequence_file_worker,
                    (fasta_str, sid, d_start, d_end, scan_config, motifs,
                     base_constraints, repeat_constraints, exclude_motifs),
                )
                in_flight[fut] = (sid, d_end - d_start)
                return True

            for _ in range(max_in_flight):
                if not _submit_next():
                    break
            while in_flight:
                finished, _pending = wait(set(in_flight), return_when=FIRST_COMPLETED)
                for future in finished:
                    seq_id, seq_size = in_flight.pop(future)
                    _res_seq_id, seq_hits = future.result()
                    all_hits.extend(seq_hits)
                    done_index += 1
                    completed_bases += seq_size
                    percent = (completed_bases / total_bases) * 100
                    print(
                        f"[scanner] Progress: {percent:.2f}% - Completed {seq_id}: "
                        f"{len(seq_hits)} hits ({done_index}/{total_records})",
                        flush=True,
                    )
                    _submit_next()
    # End-stage: optional G4Hunter filter on the matched motifs, then RNA output.
    if args.min_g4hunter is not None:
        kept: List[Dict[str, object]] = []
        for hit in all_hits:
            motif_seq = hit.get("motif_sequence") or ""
            score_seq = (
                motif_seq
                if isinstance(motif_seq, str) and motif_seq and motif_seq != "NA"
                else str(hit.get("window_sequence") or "")
            )
            score = g4hunter_best_score(score_seq)
            hit["best_g4hunter"] = round(score, 4)
            if score >= args.min_g4hunter:
                kept.append(hit)
        removed = len(all_hits) - len(kept)
        all_hits = kept
        print(
            f"[scanner] G4Hunter filter (best score >= {args.min_g4hunter}): "
            f"kept {len(all_hits)}, removed {removed}.",
            flush=True,
        )

    if determine_is_rna_records(args.sequence_type, fasta_path, records):
        apply_rna_alphabet(all_hits)

    write_outputs(all_hits, output_dir, args.output_prefix)
    if not all_hits:
        print("No windows matched the provided criteria.", file=sys.stderr)
    else:
        print(
            f"Recorded {len(all_hits)} matching windows to "
            f"{output_dir / (args.output_prefix + '.tsv')}."
        )
        # Report total motif counts
        motif_counts = {}
        for hit in all_hits:
            for m_label, m_count in hit.get("motif_hits", []):
                motif_counts[m_label] = motif_counts.get(m_label, 0) + m_count
        if motif_counts:
            print("\nTotal motif occurrences found in matching windows:")
            for label, count in sorted(motif_counts.items()):
                print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
