#!/usr/bin/env python3
"""CPU/GPU parity and regression tests for the nucleotide scanner.

Runnable two ways:
    pytest scripts/tests/test_cpu_gpu_parity.py
    python  scripts/tests/test_cpu_gpu_parity.py     # plain runner, no pytest needed

What it checks:
  * The 2-bit and 4-bit (IUPAC) encoders round-trip / match their definitions.
  * ``mako_gpu.reference_base_content_prefilter`` (which the CUDA kernel mirrors
    line-for-line) returns *exactly* the window starts the CPU scanner keeps,
    across randomized synthetic genomes and hand-picked edge cases. This
    certifies the kernel's logic even on a machine with no GPU.
  * When a CUDA device is present, the real ``base_content_prefilter`` kernel and
    a full ``scan_sequence`` run with ``gpu_kernels=True`` produce identical
    results to the CPU path (the GPU-certification gate).

On a GPU-less host the GPU-specific assertions are skipped (reported), but the
reference-vs-CPU equivalence — which is what makes the kernel trustworthy — still
runs fully.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mako_gpu  # noqa: E402
import nt_sequence_search as nss  # noqa: E402


# --------------------------------------------------------------------------- #
# Reference CPU prefilter — mirrors build_hit_record's base-content test exactly
# --------------------------------------------------------------------------- #
def cpu_base_content_prefilter(sequence, window, step, base_constraints):
    """The window starts the CPU scanner would keep on base composition alone."""
    s = sequence.upper().replace("U", "T")
    n = len(s)
    limit = n - window + 1
    out = []
    if limit <= 0:
        return out
    for start in range(0, limit, step):
        w = s[start:start + window]
        ok = True
        for base, (lo, hi) in base_constraints.items():
            b = base.upper()
            b = "T" if b == "U" else b
            pct = (w.count(b) / window) * 100
            if pct < lo or pct > hi:
                ok = False
                break
        if ok:
            out.append(start)
    return out


def random_sequence(rng, length, alphabet="ACGT"):
    return "".join(rng.choice(alphabet) for _ in range(length))


# --------------------------------------------------------------------------- #
# Encoder tests
# --------------------------------------------------------------------------- #
def test_two_bit_roundtrip():
    seq = "ACGTACGTTTTGGGCCCAAAA"
    packed, ambiguous = mako_gpu.two_bit_encode(seq)
    assert not ambiguous.any()
    decoded = []
    table = "ACGT"
    for byte in packed:
        for shift in (0, 2, 4, 6):
            decoded.append(table[(int(byte) >> shift) & 0b11])
    assert "".join(decoded[: len(seq)]) == seq
    # U decodes as T; N flagged ambiguous.
    _packed, amb = mako_gpu.two_bit_encode("ACGUN")
    assert amb.tolist() == [False, False, False, False, True]


def test_nibble_matches_iupac_mask():
    masks = mako_gpu.nibble_encode("ACGTUNRY")
    assert masks.tolist() == [1, 2, 4, 8, 8, 15, 5, 10]


# --------------------------------------------------------------------------- #
# Reference == CPU (kernel-logic certification, GPU not required)
# --------------------------------------------------------------------------- #
def _check_reference_matches_cpu(seq, window, step, constraints):
    ref = mako_gpu.reference_base_content_prefilter(seq, window, step, constraints)
    cpu = cpu_base_content_prefilter(seq, window, step, constraints)
    assert ref == cpu, (
        f"reference != CPU for window={window} step={step} constraints={constraints}\n"
        f"  ref={ref[:20]}...\n  cpu={cpu[:20]}..."
    )


def test_reference_matches_cpu_randomized():
    rng = random.Random(20240517)
    constraint_sets = [
        {"G": (40.0, 70.0)},
        {"G": (0.0, 100.0)},                       # trivially passes all
        {"G": (30.0, 60.0), "C": (20.0, 50.0)},    # multi-base
        {"A": (0.0, 10.0)},                        # strict
        {"U": (10.0, 90.0)},                       # RNA constraint (U == T)
    ]
    for _ in range(200):
        alphabet = rng.choice(["ACGT", "ACGTU", "ACGTN", "acgtACGT"])
        length = rng.randint(0, 300)
        seq = random_sequence(rng, length, alphabet)
        window = rng.randint(1, 60)
        step = rng.randint(1, 10)
        constraints = rng.choice(constraint_sets)
        _check_reference_matches_cpu(seq, window, step, constraints)


def test_reference_edge_cases():
    cases = [
        ("", 10, 1, {"G": (0.0, 100.0)}),                # empty
        ("ACGT", 10, 1, {"G": (0.0, 100.0)}),            # window > len
        ("GGGGGGGG", 4, 1, {"G": (100.0, 100.0)}),       # all-G, exact bound
        ("GGGGGGGG", 4, 1, {"C": (0.0, 0.0)}),           # zero-C bound
        ("ACGTACGT", 4, 100, {"A": (0.0, 100.0)}),       # step > len
        ("NNNNNNNN", 4, 1, {"G": (0.0, 0.0)}),           # all-N
        ("ACGUACGU", 4, 1, {"T": (0.0, 100.0)}),         # RNA, constrain T (==U)
    ]
    for seq, window, step, constraints in cases:
        _check_reference_matches_cpu(seq, window, step, constraints)


# --------------------------------------------------------------------------- #
# GPU parity (runs fully only when a CUDA device is present)
# --------------------------------------------------------------------------- #
def _gpu_kernels_ready() -> bool:
    """True only if real RawKernels can compile/run; prints an actionable hint otherwise."""
    if not mako_gpu.gpu_available():
        print("  [skip] no CUDA device")
        return False
    if not mako_gpu.kernels_available():
        status = mako_gpu.kernels_status()
        print(f"  [skip] GPU present but kernels will not compile: {status.get('kernel_reason')}")
        print(f"         Fix: {mako_gpu.KERNEL_FIX_HINT}")
        return False
    return True


def test_gpu_prefilter_matches_cpu():
    if not _gpu_kernels_ready():
        return
    rng = random.Random(7)
    for _ in range(50):
        seq = random_sequence(rng, rng.randint(50, 5000), "ACGT")
        window = rng.randint(5, 200)
        step = rng.randint(1, 7)
        constraints = {"G": (30.0, 70.0), "C": (20.0, 60.0)}
        gpu = mako_gpu.base_content_prefilter(seq, window, step, constraints)
        cpu = cpu_base_content_prefilter(seq, window, step, constraints)
        assert gpu == cpu, "GPU base_content_prefilter diverged from CPU"
    # Chunking path: force tiny tiles so the larger-than-VRAM logic is exercised.
    seq = random_sequence(rng, 4000, "ACGT")
    gpu = mako_gpu.base_content_prefilter(seq, 50, 3, {"G": (30.0, 70.0)}, chunk_starts=128)
    cpu = cpu_base_content_prefilter(seq, 50, 3, {"G": (30.0, 70.0)})
    assert gpu == cpu, "GPU chunked prefilter diverged from CPU"


def test_end_to_end_scan_parity():
    """Full scan_sequence: CPU vs GPU-kernels engine must yield identical hits."""
    rng = random.Random(99)
    seq = random_sequence(rng, 3000, "ACGT")
    motifs = nss.parse_motif_specs(_DummyParser(), ["GGG:1"])
    base = {"G": (25.0, 75.0)}
    repeat = {}
    cpu_hits = nss.scan_sequence(
        "t", seq, 20, 3, motifs, base, repeat, strand_label="+",
        runtime_options=nss.RuntimeOptions(use_gpu=False),
    )
    gpu_hits = nss.scan_sequence(
        "t", seq, 20, 3, motifs, base, repeat, strand_label="+",
        runtime_options=nss.RuntimeOptions(use_gpu=True, gpu_kernels=True),
    )
    key = lambda h: (h["window_start"], h["window_end"])
    assert [key(h) for h in cpu_hits] == [key(h) for h in gpu_hits], (
        "GPU-engine scan produced different hits than CPU "
        + ("(GPU active)" if mako_gpu.gpu_available() else "(GPU absent → fallback path)")
    )


class _DummyParser:
    """Minimal stand-in so parse_motif_specs can call .error()."""

    def error(self, message):  # pragma: no cover
        raise ValueError(message)


# --------------------------------------------------------------------------- #
# Intra/interstrand complementarity prefilter
# --------------------------------------------------------------------------- #
def test_complement_lengths_match_cpu():
    """GPU per-window complementary length must equal the CPU function exactly."""
    if not _gpu_kernels_ready():
        return
    rng = random.Random(5)
    cpu_fn = {
        "intra": nss.find_longest_intrastrand_complement,
        "inter": nss.find_longest_interstrand_complement,
    }
    for kind in ("intra", "inter"):
        for _ in range(60):
            window = rng.randint(12, 60)
            seq = random_sequence(rng, window, "ACGT")
            min_len = rng.randint(2, 5)
            max_len = rng.choice([None, rng.randint(5, 12)])
            mm = rng.choice([0, 0, 1])
            gap_min = rng.choice([0, 0, 1, 2])
            gap_max = rng.choice([None, rng.randint(2, 8)])
            gpu = mako_gpu.complement_lengths(
                seq, window, [0], kind, min_len, max_len, mm, gap_min, gap_max)
            cpu_len = cpu_fn[kind](seq, min_len, max_len, allowed_mismatches=mm,
                                   gap_min_len=gap_min, gap_max_len=gap_max)[2]
            assert int(gpu[0]) == cpu_len, (
                f"{kind} complement length differs (gpu={int(gpu[0])} cpu={cpu_len}) "
                f"seq={seq} min={min_len} max={max_len} mm={mm} gap=({gap_min},{gap_max})"
            )


def test_complement_require_end_to_end():
    """Full scan with --require-intrastrand under fused engine == CPU."""
    rng = random.Random(321)
    seq = random_sequence(rng, 2000, "ACGT")
    cfg = {"enabled": True, "min_len": 4, "max_len": None,
           "mismatches": 0, "gap_min": 0, "gap_max": None}
    common = dict(strand_label="+", intrastrand_config=cfg)
    cpu = nss.scan_sequence("t", seq, 30, 5, [], {}, {},
                            runtime_options=nss.RuntimeOptions(use_gpu=False), **common)
    gpu = nss.scan_sequence("t", seq, 30, 5, [], {}, {},
                            runtime_options=nss.RuntimeOptions(use_gpu=True, gpu_kernels=True), **common)
    key = lambda h: (h["window_start"], h["window_end"])
    assert [key(h) for h in cpu] == [key(h) for h in gpu], (
        "require-intrastrand scan differs from CPU "
        + ("(GPU active)" if mako_gpu.gpu_available() else "(GPU absent → fallback)")
    )


# --------------------------------------------------------------------------- #
# Fused predicate (composition + repeat caps + fixed-motif prefilter)
# --------------------------------------------------------------------------- #
def _motifs(specs):
    return nss.parse_motif_specs(_DummyParser(), specs) if specs else []


def _cpu_hit_starts(seq, window, step, motifs, base, repeat):
    """0-based forward-strand window starts the CPU scanner keeps."""
    hits = nss.scan_sequence(
        "t", seq, window, step, motifs, base, repeat, strand_label="+",
        runtime_options=nss.RuntimeOptions(use_gpu=False),
    )
    return sorted(h["window_start"] - 1 for h in hits)  # window_start is 1-based


def test_extract_fixed_motifs():
    fixed = dict((tuple(m), r) for m, r in nss._extract_fixed_motifs(_motifs(["GGG:2"])))
    assert fixed == {(4, 4, 4): 2}                       # G=4
    assert nss._extract_fixed_motifs(_motifs(["RYR:1"]))[0][0] == [5, 10, 5]  # R=5 Y=10
    # Quantified / variable-length motif is NOT treated as fixed.
    assert nss._extract_fixed_motifs(_motifs(["GGGN{1,5}GGG:1"])) == []


def test_fused_reference_no_false_negatives():
    """Every true CPU hit must survive the fused prefilter (superset property)."""
    rng = random.Random(2024)
    base_opts = [{}, {"G": (20.0, 80.0)}, {"G": (30.0, 70.0), "C": (10.0, 60.0)}]
    repeat_opts = [{}, {"A": 6}, {"G": 4, "C": 4}]
    motif_opts = [[], ["GGG:1"], ["GGGTTAGGG:1"], ["RYR:1"], ["GGGN{1,5}GGG:1"], ["GGG:2"]]
    for _ in range(200):
        seq = random_sequence(rng, rng.randint(30, 400), rng.choice(["ACGT", "ACGTN", "ACGTU"]))
        window = rng.randint(6, 40)
        step = rng.randint(1, 5)
        base = rng.choice(base_opts)
        repeat = rng.choice(repeat_opts)
        specs = rng.choice(motif_opts)
        motifs = _motifs(specs)
        if not (base or repeat or motifs):
            continue
        fixed = nss._extract_fixed_motifs(motifs)
        ref = set(mako_gpu.reference_fused_prefilter(seq, window, step, base, repeat, fixed))
        cpu = _cpu_hit_starts(seq, window, step, motifs, base, repeat)
        missing = [s for s in cpu if s not in ref]
        assert not missing, (
            f"fused prefilter dropped true hit(s) {missing[:5]} "
            f"(window={window} step={step} base={base} repeat={repeat} motif={specs})"
        )


def test_fused_gpu_matches_reference():
    if not _gpu_kernels_ready():
        return
    rng = random.Random(11)
    for _ in range(40):
        seq = random_sequence(rng, rng.randint(50, 4000), "ACGT")
        window = rng.randint(8, 120)
        step = rng.randint(1, 6)
        base = {"G": (25.0, 75.0)}
        repeat = {"G": 6}
        fixed = nss._extract_fixed_motifs(_motifs(["GGG:1"]))
        gpu = mako_gpu.fused_predicate_prefilter(seq, window, step, base, repeat, fixed)
        ref = mako_gpu.reference_fused_prefilter(seq, window, step, base, repeat, fixed)
        assert gpu == ref, "fused GPU kernel diverged from its reference"


def test_fused_end_to_end_scan_parity():
    """Full scan with fused GPU kernels must equal the CPU scan."""
    rng = random.Random(123)
    seq = random_sequence(rng, 4000, "ACGT")
    motifs = _motifs(["GGG:1"])
    base = {"G": (25.0, 75.0)}
    repeat = {"G": 8}
    cpu = nss.scan_sequence("t", seq, 24, 3, motifs, base, repeat, strand_label="+",
                            runtime_options=nss.RuntimeOptions(use_gpu=False))
    gpu = nss.scan_sequence("t", seq, 24, 3, motifs, base, repeat, strand_label="+",
                            runtime_options=nss.RuntimeOptions(use_gpu=True, gpu_kernels=True))
    key = lambda h: (h["window_start"], h["window_end"])
    assert [key(h) for h in cpu] == [key(h) for h in gpu], (
        "fused-engine scan differs from CPU "
        + ("(GPU active)" if mako_gpu.gpu_available() else "(GPU absent → fallback)")
    )


# --------------------------------------------------------------------------- #
# Plain runner (no pytest dependency)
# --------------------------------------------------------------------------- #
def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {test.__name__}: {exc.__class__.__name__}: {exc}")
    gpu = mako_gpu.backend_info()
    print(f"\nGPU backend: {gpu}")
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
