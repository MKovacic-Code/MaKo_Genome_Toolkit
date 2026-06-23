#!/usr/bin/env python3
"""Benchmark CPU vs GPU nucleotide scanning across dataset sizes and query counts.

Reports, per configuration: wall time, throughput (bases/second), speedup,
peak host RAM, and (when a GPU is active) VRAM usage. Runs the real
``scan_sequence`` in-process so the numbers reflect the production code path.

Usage:
    python scripts/benchmark_search.py --sizes 1e5,1e6,1e7 --engines cpu,gpu
    python scripts/benchmark_search.py --queries 1,2,4 --size 2e6 --csv bench.csv

On a host without CUDA the GPU rows are reported as "n/a" and only CPU timings
are produced; the script never fails for lack of a GPU.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import nt_sequence_search as nss  # noqa: E402

try:
    import mako_gpu  # noqa: E402
except Exception:  # pragma: no cover
    mako_gpu = None

try:
    import psutil  # type: ignore

    _PROC = psutil.Process()
except Exception:  # pragma: no cover - psutil optional
    psutil = None
    _PROC = None


def _peak_rss_mb() -> Optional[float]:
    if _PROC is not None:
        try:
            return _PROC.memory_info().rss / (1024 * 1024)
        except Exception:
            return None
    return None


def _vram_used_mb() -> Optional[float]:
    if mako_gpu is None or not mako_gpu.gpu_available():
        return None
    try:
        import cupy as cp  # type: ignore

        return cp.get_default_memory_pool().used_bytes() / (1024 * 1024)
    except Exception:
        return None


def make_genome(rng: random.Random, length: int) -> str:
    # Slightly GC-biased so base-content filters actually select windows.
    return "".join(rng.choices("ACGT", weights=(22, 28, 28, 22), k=length))


def run_once(seq: str, queries: int, use_gpu: bool, gpu_kernels: bool) -> Tuple[float, int, Optional[float], Optional[float]]:
    """Return (seconds, hit_count, rss_mb, vram_mb) for one scan."""
    motif_specs = [spec for spec in ["GGG:1", "CCC:1", "GGGN{1,5}GGG:1", "TTAGGG:1"][:queries]]
    motifs = nss.parse_motif_specs(_DummyParser(), motif_specs)
    base = {"G": (25.0, 75.0)}
    runtime = nss.RuntimeOptions(use_gpu=use_gpu, gpu_kernels=gpu_kernels, vectorized_base=not use_gpu)
    start = time.perf_counter()
    hits = nss.scan_sequence("bench", seq, 30, 5, motifs, base, {}, strand_label="+", runtime_options=runtime)
    elapsed = time.perf_counter() - start
    return elapsed, len(hits), _peak_rss_mb(), _vram_used_mb()


class _DummyParser:
    def error(self, message):  # pragma: no cover
        raise ValueError(message)


def format_bps(bases: int, seconds: float) -> str:
    if seconds <= 0:
        return "inf"
    bps = bases / seconds
    for unit, scale in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if bps >= scale:
            return f"{bps / scale:.2f} {unit}bp/s"
    return f"{bps:.0f} bp/s"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CPU vs GPU scan benchmark.")
    parser.add_argument("--sizes", default="1e5,1e6", help="Comma-separated genome sizes (bases).")
    parser.add_argument("--size", type=float, default=None, help="Single genome size (overrides --sizes for query scaling).")
    parser.add_argument("--queries", default="1", help="Comma-separated query (motif) counts.")
    parser.add_argument("--engines", default="cpu,gpu", help="Engines to test: cpu, gpu, or both.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each measurement and keep the best time.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--csv", default=None, help="Optional path to also write results as CSV.")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    sizes = [int(float(s)) for s in (str(args.size).split(",") if args.size else args.sizes.split(","))]
    query_counts = [int(q) for q in args.queries.split(",")]
    engines = [e.strip() for e in args.engines.split(",")]

    gpu_ok = mako_gpu is not None and mako_gpu.gpu_available()
    if mako_gpu is not None:
        print(f"GPU backend: {mako_gpu.backend_info()}")
    print(f"psutil RAM tracking: {'on' if _PROC else 'off (pip install psutil)'}\n")

    header = f"{'size(bp)':>12} {'queries':>7} {'engine':>6} {'time(s)':>9} {'throughput':>12} {'hits':>9} {'RAM(MB)':>9} {'VRAM(MB)':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    rows: List[Dict[str, object]] = []
    for size in sizes:
        seq = make_genome(rng, size)
        for q in query_counts:
            cpu_time = None
            for engine in engines:
                if engine == "gpu" and not gpu_ok:
                    print(f"{size:>12} {q:>7} {'gpu':>6} {'n/a':>9} {'n/a':>12} {'n/a':>9} {'n/a':>9} {'n/a':>9} {'n/a':>8}")
                    continue
                use_gpu = engine == "gpu"
                best = None
                for _ in range(max(1, args.repeats)):
                    elapsed, hits, rss, vram = run_once(seq, q, use_gpu, gpu_kernels=use_gpu)
                    if best is None or elapsed < best[0]:
                        best = (elapsed, hits, rss, vram)
                elapsed, hits, rss, vram = best
                if engine == "cpu":
                    cpu_time = elapsed
                speedup = f"{cpu_time / elapsed:.2f}x" if (engine == "gpu" and cpu_time) else "-"
                print(
                    f"{size:>12} {q:>7} {engine:>6} {elapsed:>9.3f} {format_bps(size, elapsed):>12} "
                    f"{hits:>9} {('%.0f' % rss) if rss else 'n/a':>9} {('%.0f' % vram) if vram else 'n/a':>9} {speedup:>8}"
                )
                rows.append({
                    "size": size, "queries": q, "engine": engine, "seconds": elapsed,
                    "throughput_bps": size / elapsed if elapsed else 0, "hits": hits,
                    "ram_mb": rss, "vram_mb": vram, "speedup": speedup,
                })

    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
