#!/usr/bin/env python3
"""
Command-line interface for the triplex scanning Cython extension.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from triplex_scan import find_triplex_hits


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")
    return seq.translate(table)[::-1]


def _clean_sequence(text: str, allowed: str) -> str:
    allowed_set = set(allowed)
    return "".join(ch for ch in text.upper() if ch in allowed_set)


def read_rna_records(path: Path) -> List[Tuple[str, bytes]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    records: List[Tuple[str, bytes]] = []
    if text.startswith(">"):
        label = None
        seq_parts: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None and seq_parts:
                    seq = "".join(seq_parts).replace("T", "U").upper()
                    if seq:
                        records.append((label, seq.encode("ascii")))
                label = line[1:].strip() or f"RNA_{len(records)+1}"
                seq_parts = []
            else:
                seq_parts.append("".join(ch for ch in line.upper() if ch in "ACGTU"))
        if label is not None and seq_parts:
            seq = "".join(seq_parts).replace("T", "U").upper()
            if seq:
                records.append((label, seq.encode("ascii")))
    else:
        seq = "".join(ch for ch in text.upper() if ch in "ACGTU").replace("T", "U")
        if seq:
            records.append((path.stem or "RNA_1", seq.encode("ascii")))
    if not records:
        raise ValueError(f"No RNA bases found in {path}")
    return records


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    sequences: List[Tuple[str, str]] = []
    header = "GENOME"
    chunks: List[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if chunks:
                    seq = "".join(chunks)
                    if seq:
                        sequences.append((header, seq))
                    chunks.clear()
                header = line[1:].strip() or f"SEQ_{len(sequences) + 1}"
            else:
                cleaned = _clean_sequence(line, "ACGT")
                if cleaned:
                    chunks.append(cleaned)
        if chunks:
            seq = "".join(chunks)
            if seq:
                sequences.append((header, seq))

    if not sequences:
        raise ValueError(f"No DNA sequences found in {path}")
    return sequences


def _scan_triplex_worker(
    payload: Tuple[
        str,
        int,
        str,
        bytes,
        int,
        int,
        int,
        int,
        Tuple[str, ...],
    ]
) -> Tuple[str, List[Tuple[str, int, int, str, int, int, int, int, int]]]:
    (
        seq_id,
        dna_seq,
        dna_len,
        rna_label,
        rna_seq,
        min_len,
        max_len,
        min_score,
        max_mismatches,
        strands,
    ) = payload
    rows: List[Tuple[str, int, int, str, int, int, int, int, int]] = []

    dna_forward = dna_seq.encode("ascii")

    def format_hits(
        raw_hits: List[Tuple[int, int, int, int, int]],
        strand_symbol: str,
        coord_transform,
    ):
        for start, end, length, score, mismatches in raw_hits:
            adj_start, adj_end = coord_transform(start, end)
            rows.append(
                (
                    rna_label,
                    seq_id,
                    adj_start,
                    adj_end,
                    strand_symbol,
                    score,
                    length,
                    mismatches,
                    0,
                    length,
                )
            )

    if "forward" in strands:
        forward_hits = find_triplex_hits(
            rna_seq,
            dna_forward,
            min_len=min_len,
            max_len=max_len,
            min_score=min_score,
            max_mismatches=max_mismatches,
        )
        format_hits(forward_hits, "+", lambda s, e: (s, e))
    if "reverse" in strands:
        rc_seq = reverse_complement(dna_seq)
        reverse_hits = find_triplex_hits(
            rna_seq,
            rc_seq.encode("ascii"),
            min_len=min_len,
            max_len=max_len,
            min_score=min_score,
            max_mismatches=max_mismatches,
        )
        format_hits(
            reverse_hits,
            "-",
            lambda s, e: (dna_len - e, dna_len - s),
        )
    return seq_id, rows


def scan_sequences(
    rna_label: str,
    rna: bytes,
    fasta_entries: Sequence[Tuple[str, bytes]],
    min_len: int,
    max_len: int,
    min_score: int,
    max_mismatches: int,
    workers: int,
    strands: Tuple[str, ...],
) -> List[Tuple[str, str, int, int, str, int, int, int, int, int]]:
    worker_count = workers
    if worker_count == 0:
        worker_count = os.cpu_count() or 1
    worker_count = max(1, worker_count)

    formatted: List[Tuple[str, str, int, int, str, int, int, int, int, int]] = []
    entries = list(fasta_entries)
    total = len(entries)
    if total:
        print(
            f"[triplex] Scanning {total} DNA sequences for RNA '{rna_label}' with {worker_count} worker(s)...",
            flush=True,
        )
    if worker_count == 1 or total <= 1:
        for index, (seq_id, dna) in enumerate(entries, 1):
            print(f"[triplex] Scanning {seq_id} ({index}/{total})", flush=True)
            payload = (
                seq_id,
                dna,
                len(dna),
                rna_label,
                rna,
                min_len,
                max_len,
                min_score,
                max_mismatches,
                strands,
            )
            _, seq_hits = _scan_triplex_worker(payload)
            formatted.extend(seq_hits)
            print(f"[triplex] {seq_id}: {len(seq_hits)} hits", flush=True)
    else:
        payloads = [
            (
                seq_id,
                dna,
                len(dna),
                rna_label,
                rna,
                min_len,
                max_len,
                min_score,
                max_mismatches,
                strands,
            )
            for seq_id, dna in entries
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for index, (seq_id, seq_hits) in enumerate(executor.map(_scan_triplex_worker, payloads), 1):
                formatted.extend(seq_hits)
                print(
                    f"[triplex] Completed {seq_id}: {len(seq_hits)} hits ({index}/{total})",
                    flush=True,
                )
    return formatted


def write_hits_table(
    output_name: str,
    output_prefix: str,
    rows: Sequence[Tuple[str, str, int, int, str, int, int, int, int, int]],
) -> Path:
    output_dir = Path(f"output_{output_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{output_prefix}.tsv"
    headers = [
        "rna_id",
        "chrom_or_seq_id",
        "start",
        "end",
        "strand",
        "score",
        "length",
        "mismatches",
        "rna_start",
        "rna_end",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict pyrimidine-motif RNA–DNA triplex target sites."
    )
    parser.add_argument("--rna", help="Path to RNA sequence text file.")
    parser.add_argument(
        "--rna-seq",
        help="Inline RNA sequence (ACGU). If provided, overrides --rna file contents.",
    )
    parser.add_argument("--dna", required=True, help="Path to DNA FASTA file.")
    parser.add_argument("--min", type=int, default=15, help="Minimum window length.")
    parser.add_argument("--max", type=int, default=40, help="Maximum window length.")
    parser.add_argument("--min-score", type=int, default=20, help="Minimum score.")
    parser.add_argument(
        "--max-mismatches", type=int, default=3, help="Maximum allowed mismatches."
    )
    parser.add_argument(
        "--output-prefix",
        default="triplex_hits",
        help="Base filename (without extension) for the generated TSV file.",
    )
    parser.add_argument(
        "--output-name",
        default="default",
        help="Name used to create the output_<name> directory for the generated TSV file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = all CPU cores).",
    )
    parser.add_argument(
        "--strand",
        dest="strand",
        action="append",
        choices=["forward", "reverse", "both"],
        help="DNA strands to scan (deprecated; use --scan-forward/--scan-reverse).",
    )
    parser.add_argument(
        "--scan-forward",
        dest="strand",
        action="append_const",
        const="forward",
        help="Include the forward (+) DNA strand.",
    )
    parser.add_argument(
        "--scan-reverse",
        dest="strand",
        action="append_const",
        const="reverse",
        help="Include the reverse (-) DNA strand.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.rna and not args.rna_seq:
        parser.error("Provide either --rna or --rna-seq.")
    rna_records: List[Tuple[str, bytes]]
    if args.rna_seq:
        inline = "".join(ch for ch in args.rna_seq.upper() if ch in "ACGTU").replace("T", "U")
        if not inline:
            parser.error("Inline RNA sequence contains no valid bases.")
        rna_records = [("RNA_inline", inline.encode("ascii"))]
    else:
        rna_records = read_rna_records(Path(args.rna))
    fasta_entries = read_fasta(Path(args.dna))

    strands = args.strand or ["forward", "reverse"]
    strand_set = set()
    for item in strands:
        if item == "both":
            strand_set.update({"forward", "reverse"})
        else:
            strand_set.add(item)
    if not strand_set:
        strand_set.update({"forward", "reverse"})
    strand_tuple = tuple(sorted(strand_set))

    print(
        "rna_id\tchrom_or_seq_id\tstart\tend\tstrand\tscore\tlength\tmismatches\trna_start\trna_end"
    )
    hits: List[Tuple[str, str, int, int, str, int, int, int, int, int]] = []
    for rna_label, rna_seq in rna_records:
        rna_hits = scan_sequences(
            rna_label,
            rna_seq,
            fasta_entries,
            min_len=args.min,
            max_len=args.max,
            min_score=args.min_score,
            max_mismatches=args.max_mismatches,
            workers=args.workers,
            strands=strand_tuple,
        )
        hits.extend(rna_hits)
        for row in rna_hits:
            print("\t".join(str(value) for value in row))
    tsv_path = write_hits_table(args.output_name, args.output_prefix, hits)
    print(f"Wrote {len(hits)} triplex hits to {tsv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
