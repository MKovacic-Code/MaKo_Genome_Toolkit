#!/usr/bin/env python3
"""CUDA acceleration for the MaKo nucleotide scanner via CuPy RawKernels.

This module provides custom CUDA-C kernels (compiled at runtime by CuPy's NVRTC,
so no separate ``nvcc``/CMake build step is required) for the computational
hot-spots of ``nt_sequence_search``. It is imported lazily and degrades
gracefully: every public function returns ``None`` (signalling "not handled")
when CuPy or a CUDA device is unavailable, so callers transparently fall back to
the CPU implementation.

Design goals (see docs/GPU.md):
  * **Bit-exact with the CPU path.** Kernels reproduce the exact integer counts
    and ``(count / window) * 100`` float64 comparison used by the CPU scanner,
    so GPU candidate positions are identical to the CPU ones. The CPU re-checks
    every candidate in ``build_hit_record`` regardless, so the GPU stage can
    only ever be an exact accelerator, never a source of divergent results.
  * **Larger-than-VRAM datasets.** ``base_content_prefilter`` tiles the window
    range so peak device memory is bounded by ``chunk_starts`` rather than the
    sequence length.
  * **Coalesced, low-divergence kernels.** One thread per window start, linear
    global-memory reads of the prefix-sum arrays.

The first wired kernel is the base-content candidate prefilter (the dominant
cost when base-composition constraints are active). The 2-bit / 4-bit encoders
and their references are provided for the planned fused per-window predicate
kernel and are validated by ``scripts/tests/test_cpu_gpu_parity.py``.

NOTE: the CUDA-C below is written to mirror the CPU logic exactly but must be
compiled and certified on an actual NVIDIA GPU (run the parity test there before
enabling ``--gpu-kernels`` in production).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Optional CuPy backend
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - availability depends on the host
    import cupy as _cp  # type: ignore

    _CUPY_IMPORT_ERROR: Optional[str] = None
except Exception as _exc:  # ImportError or broken CUDA runtime
    _cp = None
    _CUPY_IMPORT_ERROR = f"{_exc.__class__.__name__}: {_exc}"


def gpu_available() -> bool:
    """Return True if a usable CUDA device is present via CuPy."""
    if _cp is None:
        return False
    try:
        return _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def backend_info() -> Dict[str, object]:
    """Lightweight description of the GPU backend for logging/benchmarks."""
    if _cp is None:
        return {"available": False, "reason": _CUPY_IMPORT_ERROR or "CuPy not installed"}
    try:
        if _cp.cuda.runtime.getDeviceCount() <= 0:
            return {"available": False, "reason": "no CUDA devices"}
        props = _cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        free, total = _cp.cuda.Device(0).mem_info
        return {
            "available": True,
            "device": name.decode() if isinstance(name, (bytes, bytearray)) else str(name),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "vram_total_bytes": int(total),
            "vram_free_bytes": int(free),
        }
    except Exception as exc:
        return {"available": False, "reason": f"{exc.__class__.__name__}: {exc}"}


_kernels_probe: Optional[bool] = None
_kernels_reason: str = ""

# Actionable hint for the most common failure: NVRTC can't find CUDA headers.
KERNEL_FIX_HINT = (
    "Install the CUDA toolkit headers, e.g.  pip install cupy-cuda12x[ctk]  "
    "(or set the CUDA_PATH environment variable to a CUDA Toolkit install). "
    "Blackwell GPUs (compute capability 12.0, e.g. RTX 50-series) require "
    "CUDA 12.8 or newer."
)


def kernels_available() -> bool:
    """True only if a trivial RawKernel actually compiles and runs.

    A GPU can be present (``gpu_available()`` True) yet kernels still fail to
    build when the CUDA toolkit headers are missing (CuPy ships the runtime but
    NVRTC needs headers). Probing once here lets the prefilters fall back to CPU
    cleanly and lets callers emit a single actionable message. Result is cached.
    """
    global _kernels_probe, _kernels_reason
    if _kernels_probe is not None:
        return _kernels_probe
    if _cp is None or not gpu_available():
        _kernels_probe = False
        _kernels_reason = _CUPY_IMPORT_ERROR or "no CUDA device"
        return False
    try:
        probe = _cp.RawKernel(r'extern "C" __global__ void _mako_probe(int* x){ x[0] = 1; }', "_mako_probe")
        out = _cp.zeros(1, dtype=_cp.int32)
        probe((1,), (1,), (out,))
        _cp.cuda.runtime.deviceSynchronize()
        _kernels_probe = bool(int(out[0]) == 1)
        if not _kernels_probe:
            _kernels_reason = "probe kernel returned unexpected result"
    except Exception as exc:
        _kernels_probe = False
        _kernels_reason = f"{exc.__class__.__name__}: {exc}"
    return _kernels_probe


def kernels_status() -> Dict[str, object]:
    """Backend info plus whether RawKernels can compile, with a fix hint."""
    ok = kernels_available()
    info = dict(backend_info())
    info["kernels_compilable"] = ok
    info["kernel_reason"] = _kernels_reason
    if not ok and backend_info().get("available"):
        info["fix"] = KERNEL_FIX_HINT
    return info


# --------------------------------------------------------------------------- #
# Compact nucleotide encoders (numpy reference; used by the planned fused kernel)
# --------------------------------------------------------------------------- #
# 2-bit packing: A=0 C=1 G=2 T/U=3. Non-ACGT bases cannot be represented and are
# flagged in a companion boolean mask so the caller can fall back for those
# windows (preserving exact results on ambiguous genome characters).
_TWO_BIT = np.full(256, 0, dtype=np.uint8)
for _b, _v in ((65, 0), (97, 0), (67, 1), (99, 1), (71, 2), (103, 2),
               (84, 3), (116, 3), (85, 3), (117, 3)):  # A C G T U (upper+lower)
    _TWO_BIT[_b] = _v

# 4-bit IUPAC mask (identical to cython_helpers.mask_from_base / mask_from_code):
# A=1 C=2 G=4 T/U=8, ambiguity codes as unions, N=15. A direct AND of a base mask
# with a pattern mask reproduces IUPAC matching exactly.
_NIBBLE = np.zeros(256, dtype=np.uint8)
for _ch, _mask in (("A", 1), ("C", 2), ("G", 4), ("T", 8), ("U", 8),
                   ("R", 5), ("Y", 10), ("S", 6), ("W", 9), ("K", 12),
                   ("M", 3), ("B", 14), ("D", 13), ("H", 11), ("V", 7), ("N", 15)):
    _NIBBLE[ord(_ch)] = _mask
    _NIBBLE[ord(_ch.lower())] = _mask


def two_bit_encode(sequence: str) -> Tuple[np.ndarray, np.ndarray]:
    """Pack an ACGT(U) sequence into 2 bits/base.

    Returns ``(packed, ambiguous_mask)`` where ``packed`` is a ``uint8`` array of
    ``ceil(len/4)`` bytes (4 bases per byte, low bits first) and
    ``ambiguous_mask`` flags positions whose base is not A/C/G/T/U (which 2-bit
    cannot represent). Round-trips exactly for pure-ACGT input.
    """
    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    codes = _TWO_BIT[raw]
    acgt = np.zeros(256, dtype=bool)
    for _c in (65, 97, 67, 99, 71, 103, 84, 116, 85, 117):
        acgt[_c] = True
    ambiguous = ~acgt[raw]
    n = codes.size
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    grouped = codes.reshape(-1, 4)
    packed = (grouped[:, 0] | (grouped[:, 1] << 2) | (grouped[:, 2] << 4) | (grouped[:, 3] << 6)).astype(np.uint8)
    return packed, ambiguous


def nibble_encode(sequence: str) -> np.ndarray:
    """Encode each base as its 4-bit IUPAC mask (see ``_NIBBLE``)."""
    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    return _NIBBLE[raw]


# --------------------------------------------------------------------------- #
# Base-content candidate prefilter
# --------------------------------------------------------------------------- #
# CUDA kernel: one thread per window start. Reads precomputed per-base prefix
# sums and applies the same min/max %% test as the CPU path. Output is a 0/1 mask.
_BASE_CONTENT_KERNEL_SRC = r"""
extern "C" __global__
void base_content_filter(
        const int* __restrict__ prefix,     // [num_bases * (sub_len + 1)] row-major
        const double* __restrict__ min_pct, // [num_bases]
        const double* __restrict__ max_pct, // [num_bases]
        const int num_bases,
        const long long sub_len,
        const int window,
        const long long num_starts,
        const int* __restrict__ local_starts, // [num_starts] window start (local coords)
        unsigned char* __restrict__ passed)   // [num_starts] output 0/1
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= num_starts) return;
    long long start = local_starts[i];
    long long end = start + window;
    bool ok = true;
    for (int b = 0; b < num_bases; ++b) {
        const int* pb = prefix + (long long)b * (sub_len + 1);
        int count = pb[end] - pb[start];
        double pct = (double)count / (double)window * 100.0;   // matches CPU float64
        if (pct < min_pct[b] || pct > max_pct[b]) { ok = false; break; }
    }
    passed[i] = ok ? (unsigned char)1 : (unsigned char)0;
}
"""

_base_content_kernel = None


def _get_base_content_kernel():
    global _base_content_kernel
    if _base_content_kernel is None:
        _base_content_kernel = _cp.RawKernel(_BASE_CONTENT_KERNEL_SRC, "base_content_filter")
    return _base_content_kernel


def _base_codes(base_constraints: Dict[str, Tuple[float, float]]):
    """Stable ordering of (base_byte, min_pct, max_pct). Treats U as T."""
    items = []
    for base, (lo, hi) in base_constraints.items():
        b = base.upper()
        if b == "U":
            b = "T"
        items.append((ord(b), float(lo), float(hi)))
    return items


def base_content_prefilter(
    sequence: str,
    window: int,
    step: int,
    base_constraints: Dict[str, Tuple[float, float]],
    chunk_starts: int = 64_000_000,
) -> Optional[List[int]]:
    """GPU candidate prefilter by base composition (bit-exact with the CPU path).

    Returns the sorted list of 0-based window start indices that satisfy every
    base-content constraint, or ``None`` to signal "not handled" (no GPU, no
    constraints, or empty range) so the caller falls back to CPU/NumPy.

    Peak VRAM is bounded by ``chunk_starts``: the window range is tiled and each
    tile only materialises prefix sums for its own slice of the sequence.
    """
    if not base_constraints or not kernels_available():
        return None
    seq_upper = sequence.upper().replace("U", "T")
    n = len(seq_upper)
    limit = n - window + 1
    if limit <= 0:
        return []
    codes = _base_codes(base_constraints)
    num_bases = len(codes)
    base_bytes = np.array([c for c, _, _ in codes], dtype=np.uint8)
    min_arr = _cp.asarray(np.array([lo for _, lo, _ in codes], dtype=np.float64))
    max_arr = _cp.asarray(np.array([hi for _, _, hi in codes], dtype=np.float64))

    seq_np = np.frombuffer(seq_upper.encode("ascii", "replace"), dtype=np.uint8)
    all_starts = np.arange(0, limit, step, dtype=np.int64)
    if all_starts.size == 0:
        return []

    kernel = _get_base_content_kernel()
    threads = 256
    selected: List[int] = []

    for off in range(0, all_starts.size, chunk_starts):
        chunk = all_starts[off: off + chunk_starts]
        s_lo = int(chunk[0])
        s_hi = int(chunk[-1])
        seq_lo = s_lo
        seq_hi = min(n, s_hi + window)
        sub = _cp.asarray(seq_np[seq_lo:seq_hi])
        sub_len = int(sub.size)
        # Per-base prefix sums on device (int32, matching CPU counts).
        prefix = _cp.empty((num_bases, sub_len + 1), dtype=_cp.int32)
        for b in range(num_bases):
            hits = (sub == int(base_bytes[b])).astype(_cp.int32)
            prefix[b, 0] = 0
            _cp.cumsum(hits, dtype=_cp.int32, out=prefix[b, 1:])
        local_starts = _cp.asarray((chunk - seq_lo).astype(np.int32))
        num_starts = int(local_starts.size)
        passed = _cp.empty(num_starts, dtype=_cp.uint8)
        blocks = (num_starts + threads - 1) // threads
        kernel(
            (blocks,), (threads,),
            (prefix.ravel(), min_arr, max_arr, np.int32(num_bases),
             np.int64(sub_len), np.int32(window), np.int64(num_starts),
             local_starts, passed),
        )
        keep = chunk[_cp.asnumpy(passed).astype(bool)]
        selected.extend(int(x) for x in keep)
        # Free per-tile device memory promptly.
        del sub, prefix, local_starts, passed
        _cp.get_default_memory_pool().free_all_blocks()

    return selected


# --------------------------------------------------------------------------- #
# Base-mask encoding (matching-semantics: cython_helpers.mask_from_base)
# --------------------------------------------------------------------------- #
# For *matching* the genome, only canonical bases are non-zero: A=1 C=2 G=4
# T/U=8, and everything else (N, ambiguity codes) is 0 (matches nothing). This
# differs from nibble_encode (pattern/code masks); the two are intentionally
# distinct, mirroring mask_from_base vs mask_from_code in the Cython helpers.
_BASE_MASK = np.zeros(256, dtype=np.uint8)
for _ch, _m in (("A", 1), ("C", 2), ("G", 4), ("T", 8), ("U", 8)):
    _BASE_MASK[ord(_ch)] = _m
    _BASE_MASK[ord(_ch.lower())] = _m

_BASE_CODE = {"A": 1, "C": 2, "G": 4, "T": 8, "U": 8}


def base_mask_encode(sequence: str) -> np.ndarray:
    """Encode a sequence as per-base *matching* masks (mask_from_base semantics)."""
    raw = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    return _BASE_MASK[raw]


# --------------------------------------------------------------------------- #
# Fused per-window predicate prefilter (composition + repeat caps + fixed motifs)
# --------------------------------------------------------------------------- #
# One thread per window start. Computes an EXACT superset of the windows that can
# pass: base composition (exact), repeat caps (exact), and for fixed-length IUPAC
# motifs the *overlapping* match count >= required_count. The latter is a
# necessary condition for the CPU's non-overlapping regex count to reach the
# requirement, so no true hit is ever dropped; the CPU re-validates every
# surviving candidate, keeping results bit-identical. Variable-length / quantified
# motifs are not pre-filtered here (handled entirely by the CPU).
_FUSED_KERNEL_SRC = r"""
extern "C" __global__
void fused_predicate(
        const unsigned char* __restrict__ bm,
        const long long sub_len,
        const int window,
        const long long num_starts,
        const int* __restrict__ local_starts,
        const int num_base,
        const unsigned char* __restrict__ base_codes,
        const double* __restrict__ min_pct,
        const double* __restrict__ max_pct,
        const int num_rep,
        const unsigned char* __restrict__ rep_codes,
        const int* __restrict__ rep_limits,
        const int num_motif,
        const int* __restrict__ motif_off,
        const int* __restrict__ motif_masks,
        const int* __restrict__ motif_required,
        unsigned char* __restrict__ passed)
{
    long long w = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (w >= num_starts) return;
    long long s = local_starts[w];
    bool ok = true;

    for (int b = 0; b < num_base && ok; ++b) {
        unsigned char code = base_codes[b];
        int cnt = 0;
        for (int k = 0; k < window; ++k) if (bm[s + k] == code) ++cnt;
        double pct = (double)cnt / (double)window * 100.0;
        if (pct < min_pct[b] || pct > max_pct[b]) ok = false;
    }
    for (int r = 0; r < num_rep && ok; ++r) {
        unsigned char code = rep_codes[r];
        int run = 0, best = 0;
        for (int k = 0; k < window; ++k) {
            if (bm[s + k] == code) { run++; if (run > best) best = run; }
            else run = 0;
        }
        if (best > rep_limits[r]) ok = false;
    }
    for (int m = 0; m < num_motif && ok; ++m) {
        int o0 = motif_off[m], ml = motif_off[m + 1] - o0;
        int cnt = 0;
        for (int o = 0; o + ml <= window; ++o) {
            bool match = true;
            for (int k = 0; k < ml; ++k) {
                unsigned char bmask = bm[s + o + k];
                if (bmask == 0 || (bmask & (unsigned char)motif_masks[o0 + k]) == 0) { match = false; break; }
            }
            if (match) ++cnt;
        }
        if (cnt < motif_required[m]) ok = false;
    }
    passed[w] = ok ? (unsigned char)1 : (unsigned char)0;
}
"""

_fused_kernel = None


def _get_fused_kernel():
    global _fused_kernel
    if _fused_kernel is None:
        _fused_kernel = _cp.RawKernel(_FUSED_KERNEL_SRC, "fused_predicate")
    return _fused_kernel


def _flatten_motifs(fixed_motifs):
    """(masks_list, required) -> flat int arrays for the kernel."""
    offsets = [0]
    flat: List[int] = []
    required: List[int] = []
    for masks, req in fixed_motifs:
        flat.extend(int(m) for m in masks)
        offsets.append(len(flat))
        required.append(int(req))
    return (np.array(offsets, dtype=np.int32),
            np.array(flat or [0], dtype=np.int32),
            np.array(required or [0], dtype=np.int32))


def fused_predicate_prefilter(
    sequence: str,
    window: int,
    step: int,
    base_constraints: Dict[str, Tuple[float, float]],
    repeat_constraints: Dict[str, int],
    fixed_motifs: List[Tuple[List[int], int]],
    chunk_starts: int = 32_000_000,
) -> Optional[List[int]]:
    """GPU candidate prefilter by composition + repeat caps + fixed-motif counts.

    ``fixed_motifs`` is a list of ``(per_position_masks, required_count)`` for
    fixed-length IUPAC motifs only. Returns sorted candidate start indices (an
    exact superset of the CPU hits) or ``None`` when not handled (no GPU, or
    nothing to filter).
    """
    if not kernels_available():
        return None
    if not base_constraints and not repeat_constraints and not fixed_motifs:
        return None
    seq_upper = sequence.upper().replace("U", "T")
    n = len(seq_upper)
    limit = n - window + 1
    if limit <= 0:
        return []
    bm_np = base_mask_encode(seq_upper)

    base_codes = np.array([_BASE_CODE.get(b.upper(), 0) for b in base_constraints], dtype=np.uint8)
    min_pct = _cp.asarray(np.array([lo for lo, _ in base_constraints.values()], dtype=np.float64))
    max_pct = _cp.asarray(np.array([hi for _, hi in base_constraints.values()], dtype=np.float64))
    base_codes_g = _cp.asarray(base_codes)
    rep_codes = np.array([_BASE_CODE.get(b.upper(), 0) for b in repeat_constraints], dtype=np.uint8)
    rep_codes_g = _cp.asarray(rep_codes)
    rep_limits_g = _cp.asarray(np.array(list(repeat_constraints.values()) or [0], dtype=np.int32))
    moff, mflat, mreq = _flatten_motifs(fixed_motifs)
    moff_g, mflat_g, mreq_g = _cp.asarray(moff), _cp.asarray(mflat), _cp.asarray(mreq)

    all_starts = np.arange(0, limit, step, dtype=np.int64)
    if all_starts.size == 0:
        return []
    kernel = _get_fused_kernel()
    threads = 128
    selected: List[int] = []
    for off in range(0, all_starts.size, chunk_starts):
        chunk = all_starts[off: off + chunk_starts]
        seq_lo = int(chunk[0])
        seq_hi = min(n, int(chunk[-1]) + window)
        bm = _cp.asarray(bm_np[seq_lo:seq_hi])
        local = _cp.asarray((chunk - seq_lo).astype(np.int32))
        num_starts = int(local.size)
        passed = _cp.empty(num_starts, dtype=_cp.uint8)
        blocks = (num_starts + threads - 1) // threads
        kernel(
            (blocks,), (threads,),
            (bm, np.int64(int(bm.size)), np.int32(window), np.int64(num_starts), local,
             np.int32(len(base_codes)), base_codes_g, min_pct, max_pct,
             np.int32(len(rep_codes)), rep_codes_g, rep_limits_g,
             np.int32(len(fixed_motifs)), moff_g, mflat_g, mreq_g, passed),
        )
        keep = chunk[_cp.asnumpy(passed).astype(bool)]
        selected.extend(int(x) for x in keep)
        del bm, local, passed
        _cp.get_default_memory_pool().free_all_blocks()
    return selected


def reference_fused_prefilter(
    sequence: str,
    window: int,
    step: int,
    base_constraints: Dict[str, Tuple[float, float]],
    repeat_constraints: Dict[str, int],
    fixed_motifs: List[Tuple[List[int], int]],
) -> List[int]:
    """Pure-Python reference for the fused predicate (the kernel mirrors this)."""
    if not base_constraints and not repeat_constraints and not fixed_motifs:
        return []
    seq_upper = sequence.upper().replace("U", "T")
    n = len(seq_upper)
    limit = n - window + 1
    if limit <= 0:
        return []
    bm = base_mask_encode(seq_upper)
    base_items = [(_BASE_CODE.get(b.upper(), 0), lo, hi) for b, (lo, hi) in base_constraints.items()]
    rep_items = [(_BASE_CODE.get(b.upper(), 0), lim) for b, lim in repeat_constraints.items()]
    out: List[int] = []
    for start in range(0, limit, step):
        win = bm[start:start + window]
        ok = True
        for code, lo, hi in base_items:
            pct = int(np.count_nonzero(win == code)) / window * 100.0
            if pct < lo or pct > hi:
                ok = False
                break
        if ok:
            for code, lim in rep_items:
                run = best = 0
                for v in win:
                    if v == code:
                        run += 1
                        best = max(best, run)
                    else:
                        run = 0
                if best > lim:
                    ok = False
                    break
        if ok:
            for masks, req in fixed_motifs:
                ml = len(masks)
                cnt = 0
                for o in range(0, window - ml + 1):
                    if all((int(win[o + k]) != 0 and (int(win[o + k]) & masks[k]) != 0) for k in range(ml)):
                        cnt += 1
                if cnt < req:
                    ok = False
                    break
        if ok:
            out.append(start)
    return out


def reference_base_content_prefilter(
    sequence: str,
    window: int,
    step: int,
    base_constraints: Dict[str, Tuple[float, float]],
) -> List[int]:
    """Pure-NumPy reference implementing the exact same predicate as the kernel.

    Used by the parity tests to validate the kernel logic even on machines
    without a GPU, and as a CPU cross-check of the GPU output.
    """
    if not base_constraints:
        return []
    seq_upper = sequence.upper().replace("U", "T")
    n = len(seq_upper)
    limit = n - window + 1
    if limit <= 0:
        return []
    seq_np = np.frombuffer(seq_upper.encode("ascii", "replace"), dtype=np.uint8)
    starts = np.arange(0, limit, step, dtype=np.int64)
    mask = np.ones(starts.shape, dtype=bool)
    for b, lo, hi in _base_codes(base_constraints):
        hits = (seq_np == b).astype(np.int32)
        prefix = np.concatenate(([0], np.cumsum(hits)))
        counts = prefix[starts + window] - prefix[starts]
        pct = counts.astype(np.float64) / float(window) * 100.0
        mask &= (pct >= lo) & (pct <= hi)
    return [int(s) for s in starts[mask]]


# --------------------------------------------------------------------------- #
# Intra/interstrand complementarity prefilter (require-* features)
# --------------------------------------------------------------------------- #
# One thread per candidate window computes the longest complementary stretch,
# faithfully mirroring find_longest_{intra,inter}strand_complement_cython. Used
# only to drop windows that cannot satisfy a --require-intrastrand /
# --require-interstrand length bound; the CPU recomputes the reported fields.
# The reference for certification is the existing CPU function itself, so these
# kernels are certified solely by CPU-vs-GPU equality on a real device.

# Complement table matching cython_helpers COMPLEMENT_TABLE (analysis is upper +
# U->T, but the full IUPAC mapping is provided for safety).
_COMPLEMENT_TABLE = np.arange(256, dtype=np.uint8)
for _a, _b in (("A", "T"), ("C", "G"), ("G", "C"), ("T", "A"), ("U", "A"),
               ("R", "Y"), ("Y", "R"), ("K", "M"), ("M", "K"), ("S", "S"),
               ("W", "W"), ("B", "V"), ("D", "H"), ("H", "D"), ("V", "B"), ("N", "N")):
    _COMPLEMENT_TABLE[ord(_a)] = ord(_b)
    _COMPLEMENT_TABLE[ord(_a.lower())] = ord(_b.lower())

_INTRA_KERNEL_SRC = r"""
extern "C" __global__
void intra_complement_len(
        const unsigned char* __restrict__ seq,
        const int* __restrict__ starts,
        const int num, const int window,
        const int min_len, const int max_len, const int allowed_mm,
        const int gap_min, const int gap_max,
        const unsigned char* __restrict__ comp,
        int* __restrict__ out_len)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num) return;
    const unsigned char* s = seq + starts[t];
    int n = window, best = 0;
    int ilimit = n - 2 * min_len - gap_min + 1;
    for (int i = 0; i < ilimit; ++i) {
        int mlb = (min_len > best + 1) ? min_len : best + 1;
        int jstart = i + 2 * mlb + gap_min - 1;
        int jend = n;
        if (max_len != -1 && gap_max != -1) {
            int cap = i + 2 * max_len + gap_max;
            if (cap < jend) jend = cap;
        }
        for (int j = jstart; j < jend; ++j) {
            int max_possible = (j - i + 1 - gap_min) / 2;
            if (max_len != -1 && max_len < max_possible) max_possible = max_len;
            if (max_possible <= best) continue;
            int mm = 0, curr = 0;
            for (int length = 1; length <= max_possible; ++length) {
                unsigned char lb = s[i + length - 1];
                unsigned char rb = s[j - length + 1];
                if (lb != comp[rb]) mm++;
                if (mm > allowed_mm) break;
                int gap = j - i - 2 * length + 1;
                if (gap >= gap_min && (gap_max == -1 || gap <= gap_max)) curr = length;
            }
            if (curr >= min_len && curr > best) best = curr;
        }
    }
    out_len[t] = best;
}
"""

_INTER_KERNEL_SRC = r"""
extern "C" __global__
void inter_complement_len(
        const unsigned char* __restrict__ seq,
        const int* __restrict__ starts,
        const int num, const int window,
        const int min_len, const int max_len, const int allowed_mm,
        const int gap_min, const int gap_max,
        int* __restrict__ out_len)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num) return;
    const unsigned char* s = seq + starts[t];
    int n = window, best = 0;
    int ilimit = n - 2 * min_len - gap_min + 1;
    for (int i = 0; i < ilimit; ++i) {
        int mlb = (min_len > best + 1) ? min_len : best + 1;
        int jstart = i + mlb + gap_min;
        int jend = n - mlb + 1;
        if (gap_max != -1) {
            int cap = (n + i + gap_max) / 2 + 1;
            if (cap < jend) jend = cap;
            if (max_len != -1) { int cap2 = i + max_len + gap_max + 1; if (cap2 < jend) jend = cap2; }
        }
        for (int j = jstart; j < jend; ++j) {
            int a = j - i - gap_min, b = n - j;
            int max_possible = a < b ? a : b;
            if (max_len != -1 && max_len < max_possible) max_possible = max_len;
            if (max_possible <= best) continue;
            int mm = 0, curr = 0;
            for (int length = 1; length <= max_possible; ++length) {
                if (s[i + length - 1] != s[j + length - 1]) mm++;
                if (mm > allowed_mm) break;
                int gap = j - i - length;
                if (gap >= gap_min && (gap_max == -1 || gap <= gap_max)) curr = length;
            }
            if (curr >= min_len && curr > best) best = curr;
        }
    }
    out_len[t] = best;
}
"""

_intra_kernel = None
_inter_kernel = None


def _get_complement_kernel(kind: str):
    global _intra_kernel, _inter_kernel
    if kind == "intra":
        if _intra_kernel is None:
            _intra_kernel = _cp.RawKernel(_INTRA_KERNEL_SRC, "intra_complement_len")
        return _intra_kernel
    if _inter_kernel is None:
        _inter_kernel = _cp.RawKernel(_INTER_KERNEL_SRC, "inter_complement_len")
    return _inter_kernel


def complement_lengths(
    sequence: str,
    window: int,
    candidate_starts: List[int],
    kind: str,
    min_len: int,
    max_len: Optional[int],
    allowed_mismatches: int,
    gap_min: int,
    gap_max: Optional[int],
) -> Optional[np.ndarray]:
    """Longest complementary length per candidate window (GPU). None if unavailable."""
    if not kernels_available() or not candidate_starts:
        return None if not kernels_available() else np.empty(0, dtype=np.int32)
    seq_upper = sequence.upper().replace("U", "T")
    seq_np = np.frombuffer(seq_upper.encode("ascii", "replace"), dtype=np.uint8)
    starts = np.asarray(candidate_starts, dtype=np.int32)
    seq_g = _cp.asarray(seq_np)
    starts_g = _cp.asarray(starts)
    out = _cp.empty(starts.size, dtype=_cp.int32)
    ml = int(max_len) if max_len is not None else -1
    gmax = int(gap_max) if gap_max is not None else -1
    threads = 64
    blocks = (int(starts.size) + threads - 1) // threads
    kernel = _get_complement_kernel(kind)
    if kind == "intra":
        comp_g = _cp.asarray(_COMPLEMENT_TABLE)
        kernel((blocks,), (threads,),
               (seq_g, starts_g, np.int32(starts.size), np.int32(window),
                np.int32(min_len), np.int32(ml), np.int32(allowed_mismatches),
                np.int32(gap_min), np.int32(gmax), comp_g, out))
    else:
        kernel((blocks,), (threads,),
               (seq_g, starts_g, np.int32(starts.size), np.int32(window),
                np.int32(min_len), np.int32(ml), np.int32(allowed_mismatches),
                np.int32(gap_min), np.int32(gmax), out))
    return _cp.asnumpy(out)


def complement_require_prefilter(
    sequence: str,
    window: int,
    candidate_starts: List[int],
    kind: str,
    min_len: int,
    max_len: Optional[int],
    allowed_mismatches: int,
    gap_min: int,
    gap_max: Optional[int],
) -> Optional[List[int]]:
    """Keep only candidate window starts whose longest complementary stretch
    satisfies the require bound (min_len <= best <= max_len). Exact match to the
    CPU require gate; CPU still recomputes the reported coordinates. None when
    the GPU is unavailable so the caller leaves candidates unchanged."""
    lengths = complement_lengths(
        sequence, window, candidate_starts, kind, min_len, max_len,
        allowed_mismatches, gap_min, gap_max,
    )
    if lengths is None:
        return None
    keep: List[int] = []
    for start, best in zip(candidate_starts, lengths.tolist()):
        if best >= min_len and (max_len is None or best <= max_len):
            keep.append(int(start))
    return keep
