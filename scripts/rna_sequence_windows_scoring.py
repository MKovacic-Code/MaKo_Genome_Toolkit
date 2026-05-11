#!/usr/bin/env python3
"""
Score RNA windows for triplex-forming potential based on base composition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

TRIPLEX_BASE_SCORE = {
    "C": 2.5,
    "U": 1.5,
    "T": 1.5,
    "G": -1.5,
    "A": -0.5,
}


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        label: str | None = None
        seq_chunks: List[str] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None and seq_chunks:
                    records.append((label, "".join(seq_chunks)))
                label = line[1:].strip() or f"SEQ_{len(records)+1}"
                seq_chunks = []
            else:
                seq_chunks.append(line.upper())
        if label is not None and seq_chunks:
            records.append((label, "".join(seq_chunks)))
    if not records:
        raise ValueError(f"No sequences found in {path}")
    return records


def _longest_run(seq: str, bases: Sequence[str]) -> int:
    allowed = set(bases)
    longest = 0
    current = 0
    for base in seq:
        if base in allowed:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def score_sequence(seq: str) -> Tuple[float, int, int, float]:
    normalized = seq.upper().replace("T", "U")
    total_len = len(normalized)
    score = 0.0
    matches = 0
    mismatches = 0
    for base in normalized:
        delta = TRIPLEX_BASE_SCORE.get(base)
        if delta is None:
            continue
        score += delta
        if delta > 0:
            matches += 1
        else:
            mismatches += 1

    # Reward long pyrimidine runs (C/U)
    longest_pyrimidine = _longest_run(normalized, ("C", "U"))
    score += 0.5 * longest_pyrimidine

    # Penalize G clusters (disruptive for pyrimidine triplexes)
    longest_g_run = _longest_run(normalized, ("G",))
    score -= 0.5 * longest_g_run

    # Normalize for length (favor sequences with high pyrimidine density)
    if total_len > 0:
        pyrimidine_fraction = matches / total_len
        score += pyrimidine_fraction * 5

    return score, matches, mismatches, pyrimidine_fraction


def write_outputs(
    rows: Sequence[Tuple[str, str, float, int, int, float]],
    filtered_ids: Sequence[int],
    output_dir: Path,
    prefix: str,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"{prefix}.tsv"
    fasta_path = output_dir / f"{prefix}.fasta"
    with tsv_path.open("w", encoding="utf-8") as tsv:
        tsv.write("id\tlength\tscore\tmatches\tmismatches\tpyrimidine_fraction\tsequence\n")
        for seq_id, seq, score, matches, mismatches, pyr_frac in rows:
            tsv.write(
                f"{seq_id}\t{len(seq)}\t{score:.2f}\t{matches}\t{mismatches}\t{pyr_frac:.3f}\t{seq}\n"
            )
    with fasta_path.open("w", encoding="utf-8") as fasta:
        for index in filtered_ids:
            seq_id, seq, *_ = rows[index]
            fasta.write(f">{seq_id}\n{seq}\n")
    return tsv_path, fasta_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score RNA windows for triplex-forming potential and filter by score."
    )
    parser.add_argument("rna_fasta", help="RNA windows in FASTA format.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=20.0,
        help="Minimum cumulative score to retain a sequence (default: 20).",
    )
    parser.add_argument(
        "--min-pyrimidine",
        type=float,
        default=0.5,
        help="Minimum pyrimidine fraction (0-1) required to retain a sequence (default: 0.5).",
    )
    parser.add_argument(
        "--min-spacing",
        type=int,
        default=0,
        help="Minimum distance between retained windows (0 disables spacing filter).",
    )
    parser.add_argument(
        "--output-prefix",
        default="rna_windows_scored",
        help="Base name for output files (default: rna_windows_scored).",
    )
    parser.add_argument(
        "--output-dir",
        default="output_scored",
        help="Directory for output TSV/FASTA (default: output_scored).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    records = read_fasta(Path(args.rna_fasta))
    scored_rows: List[Tuple[str, str, float, int, int, float]] = []
    filtered_indices: List[int] = []
    for idx, (seq_id, seq) in enumerate(records):
        score, matches, mismatches, pyr_frac = score_sequence(seq)
        scored_rows.append((seq_id, seq, score, matches, mismatches, pyr_frac))
        if score >= args.min_score and pyr_frac >= args.min_pyrimidine:
            filtered_indices.append(idx)

    if args.min_spacing > 0 and len(filtered_indices) > 1:
        filtered_indices.sort()
        pruned: List[int] = []
        last_idx = filtered_indices[0]
        pruned.append(last_idx)
        for idx in filtered_indices[1:]:
            prev_seq_id = scored_rows[last_idx][0]
            curr_seq_id = scored_rows[idx][0]
            try:
                prev_start = int(prev_seq_id.rsplit("_", 2)[-2])
                curr_start = int(curr_seq_id.rsplit("_", 2)[-2])
            except (ValueError, IndexError):
                prev_start = last_idx * 1_000
                curr_start = idx * 1_000
            if abs(curr_start - prev_start) >= args.min_spacing:
                pruned.append(idx)
                last_idx = idx
            else:
                if scored_rows[idx][2] > scored_rows[last_idx][2]:
                    pruned[-1] = idx
                    last_idx = idx
        filtered_indices = pruned
    output_dir = Path(args.output_dir)
    tsv_path, fasta_path = write_outputs(scored_rows, filtered_indices, output_dir, args.output_prefix)
    print(
        f"Scored {len(records)} sequences. "
        f"{len(filtered_indices)} met the threshold (>= {args.min_score}).\n"
        f"TSV: {tsv_path}\nFiltered FASTA: {fasta_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
