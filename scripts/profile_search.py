#!/usr/bin/env python3
"""Profile a genome search to see whether it is I/O-bound or compute-bound.

In the normal scanner the three phases are interleaved, so you cannot tell what
actually limits a run. This tool separates and times them:

  1. read + parse the FASTA          -> disk I/O
  2. scan the in-memory sequences     -> compute (per CPU core)
  3. write the results                -> output I/O

and prints a verdict. If a search is *compute-bound* even across all cores, a
GPU full-scan (the "genome-resident GPU scanner") can give a real speedup. If it
is *I/O-bound*, the multi-core CPU scan already keeps up with the disk and a GPU
would mostly wait -so it is not worth building for that workload.

Example (profile ~200 Mbp of a search like the one in your log):
  python scripts/profile_search.py data_human_homo_sapiens_GRCh38/GRCh38_genomic.fna \
      --window 100 --step 30 --motif AGCGA:4 --palindrome-min-len 8 \
      --strand forward --strand reverse --limit-bp 200000000 --workers 0

Run it twice: the first run reads cold from disk, the second reads from the OS
page cache -the difference is the true disk-read cost.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import nt_sequence_search as nss  # noqa: E402

try:
    import psutil  # type: ignore

    _PROC = psutil.Process()
except Exception:  # pragma: no cover - psutil optional
    _PROC = None


def _rss_mb():
    if _PROC is not None:
        try:
            return _PROC.memory_info().rss / (1024 * 1024)
        except Exception:
            return None
    return None


class _Parser:
    """Stand-in so the nss parse_* helpers can call .error()."""

    def error(self, message):  # pragma: no cover
        raise SystemExit(f"error: {message}")


def _build_config(args, motifs):
    """A scan_with_config dict mirroring the real scanner's defaults."""
    return {
        "window": args.window,
        "step": args.step,
        "strands": tuple(sorted(args.strands)),
        "combined_forward_len": args.window,
        "combined_reverse_len": args.window,
        "combined_overlap": 0,
        "motifs": motifs,
        "motifs_forward": [],
        "motifs_reverse": [],
        "non_overlapping": bool(args.non_overlapping),
        "region_filters": {},
        "palindrome_required": bool(args.require_palindrome),
        "palindrome_min_len": args.palindrome_min_len,
        "intrastrand_required": bool(args.require_intrastrand),
        "intrastrand_min_len": args.intrastrand_min_len,
        "intrastrand_max_len": None,
        "intrastrand_mismatches": 0,
        "intrastrand_gap_min": 0,
        "intrastrand_gap_max": None,
        "interstrand_required": bool(args.require_interstrand),
        "interstrand_min_len": args.interstrand_min_len,
        "interstrand_max_len": None,
        "interstrand_mismatches": 0,
        "interstrand_gap_min": 0,
        "interstrand_gap_max": None,
        "runtime_options": nss.RuntimeOptions(),
        "max_motif_mismatches": int(args.motif_mismatches),
        "self_comp_constraints": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile I/O vs compute for a genome search.")
    parser.add_argument("fasta", help="Genome FASTA to profile.")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--motif", action="append", default=[], help="PATTERN:COUNT (repeatable).")
    parser.add_argument("--base-content", action="append", default=[], help="BASE:MIN:MAX (repeatable).")
    parser.add_argument("--max-repeat", action="append", default=[], help="BASE:MAX (repeatable).")
    parser.add_argument("--exclude-motif", action="append", default=[])
    parser.add_argument("--motif-mismatches", type=int, default=0)
    parser.add_argument("--strand", dest="strands", action="append",
                        choices=["forward", "reverse", "combined"], default=[])
    parser.add_argument("--non-overlapping", action="store_true")
    parser.add_argument("--require-palindrome", action="store_true")
    parser.add_argument("--palindrome-min-len", type=int, default=0)
    parser.add_argument("--require-intrastrand", action="store_true")
    parser.add_argument("--intrastrand-min-len", type=int, default=6)
    parser.add_argument("--require-interstrand", action="store_true")
    parser.add_argument("--interstrand-min-len", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU cores assumed for the multi-core estimate (0 = all).")
    parser.add_argument("--limit-bp", type=float, default=200e6,
                        help="Stop after this many bases (0 = whole file). Default 200 Mbp.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    parser_stub = _Parser()
    motifs = nss.parse_motif_specs(parser_stub, args.motif)
    base = nss.parse_base_content_specs(parser_stub, args.base_content)
    repeat = nss.parse_max_repeat_specs(parser_stub, args.max_repeat)
    exclude = nss.parse_exclude_motifs(parser_stub, args.exclude_motif)
    args.strands = set(args.strands) or {"forward"}
    if "both" in args.strands:
        args.strands.update({"forward", "reverse"})
        args.strands.discard("both")
    config = _build_config(args, motifs)

    fasta = Path(args.fasta)
    if not fasta.is_file():
        raise SystemExit(f"No such file: {fasta}")
    limit = int(args.limit_bp) if args.limit_bp else None
    workers = args.workers or (os.cpu_count() or 1)

    # ---- Phase 1: read + parse (disk I/O) ----
    t0 = time.perf_counter()
    sequences = []
    total_bp = 0
    for seq_id, seq in nss.iter_nc_sequences(fasta, None):
        sequences.append((seq_id, seq))
        total_bp += len(seq)
        if limit and total_bp >= limit:
            break
    t_io = time.perf_counter() - t0
    rss_after_read = _rss_mb()
    if not sequences:
        raise SystemExit("No sequences read.")

    # ---- Phase 2: scan (compute, single core for a clean per-core number) ----
    t0 = time.perf_counter()
    all_hits = []
    for seq_id, seq in sequences:
        all_hits.extend(
            nss.scan_with_config(seq_id, seq, config, motifs, base, repeat, exclude_motifs=exclude)
        )
    t_compute = time.perf_counter() - t0

    # ---- Phase 3: write output ----
    out_dir = Path(tempfile.mkdtemp(prefix="mako_prof_"))
    t0 = time.perf_counter()
    nss.write_outputs(all_hits, out_dir, "profile_hits")
    t_write = time.perf_counter() - t0
    try:
        for f in out_dir.glob("*"):
            f.unlink()
        out_dir.rmdir()
    except Exception:
        pass

    # ---- Report ----
    def mbps(n, t):
        return f"{n / t / 1e6:.1f} Mbp/s" if t > 0 else "inf"

    mp_compute = t_compute / workers  # idealised wall time of the scan across all cores

    print(f"\nProfiled {total_bp:,} bp across {len(sequences)} sequence(s); found {len(all_hits):,} hits.")
    print(f"  multi-core estimate uses {workers} core(s)" + (f"; RAM after read ~ {rss_after_read:.0f} MB" if rss_after_read else ""))
    print()
    print(f"  {'phase':<26}{'time (s)':>10}{'throughput':>14}")
    print("  " + "-" * 50)
    print(f"  {'1. read + parse (disk)':<26}{t_io:>10.2f}{mbps(total_bp, t_io):>14}")
    print(f"  {'2. scan (1 core)':<26}{t_compute:>10.2f}{mbps(total_bp, t_compute):>14}")
    print(f"  {f'   scan (~{workers} cores, est.)':<26}{mp_compute:>10.2f}{mbps(total_bp, mp_compute):>14}")
    print(f"  {'3. write output':<26}{t_write:>10.4f}{('  ' + mbps(max(1, len(all_hits)), t_write).replace('Mbp', 'Mrow')) if t_write > 0 else '':>14}")
    print()

    if mp_compute > t_io * 1.3:
        verdict = ("COMPUTE-BOUND -the scan dominates even across all cores. Moving the "
                   "per-window scan onto the GPU can give a real speedup.")
    elif t_io > mp_compute * 1.3:
        verdict = ("I/O-BOUND -reading the genome dominates; the multi-core CPU scan already "
                   "keeps up, so a GPU scan would mostly wait on disk. Re-run to warm the OS "
                   "cache and compare the read time.")
    else:
        verdict = ("BALANCED -disk read and multi-core compute are comparable; the GPU would "
                   "help the compute half, but the disk read remains a floor.")
    print(f"  Verdict: {verdict}")
    print(f"  (disk read {t_io:.2f}s  vs  ~{workers}-core scan {mp_compute:.2f}s; "
          f"lower bound ~ {max(t_io, mp_compute):.2f}s for this slice)")
    print("  Note: the multi-core figure is an upper bound -real multiprocessing also pays "
          "IPC cost to ship each sequence to a worker. Run twice (cold vs OS-cached) for the "
          "true read cost.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
