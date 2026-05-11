#!/usr/bin/env python3
"""
Split a long RNA sequence into overlapping windows for downstream triplex scanning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, Tuple

ALLOWED_BASES = {"A", "C", "G", "U", "T"}


def read_rna_sequence(path: Path) -> Tuple[str, str]:
    """
    Return (label, sequence) from a FASTA or plain-text RNA file.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    if text.startswith(">"):
        lines = text.splitlines()
        label = lines[0][1:].strip() or "RNA"
        seq = "".join(line.strip() for line in lines[1:])
    else:
        label = path.stem or "RNA"
        seq = "".join(line.strip() for line in text.splitlines())
    cleaned = "".join(base for base in seq.upper() if base in ALLOWED_BASES)
    if not cleaned:
        raise ValueError(f"No RNA bases found in {path}")
    return label, cleaned.replace("T", "U")


def generate_windows(
    sequence: str,
    window: int,
    step: int,
    min_length: int,
    include_partial: bool,
) -> Iterator[Tuple[int, str]]:
    seq_len = len(sequence)
    if seq_len < min_length:
        return
    for start in range(0, seq_len - window + 1, step):
        yield start, sequence[start : start + window]
    if include_partial:
        tail_start = ((seq_len - window) // step + 1) * step
        if tail_start < seq_len:
            tail_seq = sequence[tail_start:]
            if len(tail_seq) >= min_length:
                yield tail_start, tail_seq


def write_windows(
    label: str,
    sequence: str,
    window: int,
    step: int,
    min_length: int,
    include_partial: bool,
    output: Path,
) -> None:
    with output.open("w", encoding="utf-8") as handle:
        for start, subseq in generate_windows(sequence, window, step, min_length, include_partial):
            end = start + len(subseq)
            handle.write(f">{label}_win_{start}_{end}\n")
            handle.write(subseq + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split an RNA sequence into overlapping windows for triplex scanning."
    )
    parser.add_argument("rna", help="Path to RNA sequence (FASTA or plain text).")
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Window length for each fragment (default: 60).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=20,
        help="Shift between consecutive windows (default: 20).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum length to keep a fragment (default: 20).",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="Include the final trailing sequence even if shorter than --window.",
    )
    parser.add_argument(
        "--output",
        default="rna_windows.fasta",
        help="Output FASTA file (default: rna_windows.fasta).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.window <= 0:
        parser.error("--window must be positive.")
    if args.step <= 0:
        parser.error("--step must be positive.")
    if args.min_length <= 0:
        parser.error("--min-length must be positive.")

    label, sequence = read_rna_sequence(Path(args.rna))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_windows(
        label,
        sequence,
        args.window,
        args.step,
        args.min_length,
        args.include_partial,
        output_path,
    )
    print(
        f"Generated RNA windows from '{args.rna}' "
        f"with window={args.window}, step={args.step} -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
