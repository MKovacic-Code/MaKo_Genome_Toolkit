# GPU acceleration & large-scale search

This document covers the CUDA acceleration, the crash-safe streaming/resume
pipeline, and how to build, validate, and benchmark them. The toolkit always
runs on CPU; the GPU path is an **optional, bit-exact accelerator** with
automatic fallback.

## Execution engines

`nt_sequence_search.py` selects an engine at startup:

| Flag | Behaviour |
|---|---|
| `--engine auto` (default) | Use a CUDA GPU if one is detected, else CPU. |
| `--engine cpu` | Force CPU. |
| `--engine gpu` | Request the GPU; **warn and fall back to CPU** if unavailable. |
| `--use-gpu` | Legacy alias for `--engine gpu`. |
| `--gpu-kernels` | Use the custom CUDA RawKernels in `mako_gpu` (opt-in; see *Validation*). Without it, the GPU engine uses the high-level CuPy path. |

At startup the scanner prints a line such as:

```
[engine] mode=gpu gpu='NVIDIA RTX A4000' cc=8.6 vram=16376MB
```

`detect_gpu_capabilities()` never raises — a missing driver, missing CuPy, or
zero devices simply degrades to CPU.

## What runs on the GPU

Two GPU candidate-prefilter kernels ship in [`scripts/mako_gpu.py`](../scripts/mako_gpu.py):

1. **`base_content_filter`** — base composition only (prefix sums + the exact
   `(count / window) * 100` float64 test).
2. **`fused_predicate`** — base composition **+ repeat caps + fixed-length motif
   counts**, one thread per window start over the 4-bit base-mask encoding
   (`base_mask_encode`, matching `cython_helpers.mask_from_base`). For motifs it
   uses the *overlapping* match count `>= required_count`, which is a **necessary
   condition** for the CPU's non-overlapping regex count to reach the
   requirement — so no true hit is ever dropped.

Both return an **exact superset** of the CPU hits; the CPU re-validates every
surviving candidate in `build_hit_record` (including the real non-overlapping
regex match, exclude motifs, palindrome/complementarity, etc.), so the GPU stage
can only ever be an exact accelerator. Variable-length / quantified motifs (e.g.
`GGGN{1,7}GGG`) are left entirely to the CPU. The fused kernel is exercised via
`build_candidate_positions` when `--gpu-kernels` is active.

3. **`intra_complement_len` / `inter_complement_len`** — per-candidate-window
   longest-complementary-length, mirroring `find_longest_{intra,inter}strand_
   complement_cython`, used to drop windows that cannot satisfy a
   `--require-intrastrand` / `--require-interstrand` length bound. Its only
   reference is the existing CPU function, so it is certified solely by
   CPU-vs-GPU equality on a device (`test_complement_lengths_match_cpu`).

### Robust fallback

Every CuPy path — custom RawKernels *and* high-level ops like `cumsum` — is
JIT-compiled by NVRTC, which needs the CUDA toolkit headers. `kernels_available()`
probes this once; if compilation is impossible (e.g. headers missing, or a
Blackwell GPU with a pre-12.8 toolkit) the scanner prints a one-line fix hint and
**falls back to CPU instead of crashing**. So `--engine gpu --gpu-kernels` always
produces correct results, accelerated when the toolchain is complete.

### Larger-than-VRAM datasets

`base_content_prefilter(..., chunk_starts=N)` tiles the window range so peak
device memory is bounded by the tile size, not the sequence length. Each tile
materialises prefix sums only for its own slice and frees them before the next
tile. The genome is never duplicated on the host (sequences are processed one at
a time; the analysis copy is created lazily).

## Crash-safe streaming & resume

For genome-scale runs, enable incremental, resumable output:

```bash
python scripts/nt_sequence_search.py genome.fna --window 200 --step 20 \
    --base-content G:40:70 --motif GGGN{1,7}GGG:1 \
    --stream-output --flush-every 5000 --output-name run1
```

* Hits stream to `output_run1/<prefix>.partial.jsonl` (flushed + `fsync`ed every
  `--flush-every` hits). Every line is a complete, valid record, so an
  interrupted run leaves a readable file.
* A checkpoint (`<prefix>.checkpoint.json`) records completed sequences and the
  JSONL byte offset after each.
* On restart with the same command, the run **resumes**: it truncates any
  partially written tail, skips completed sequences (sequences are processed in
  deterministic FASTA order), and appends the rest — **no duplicate rows**.
* The canonical TSV is produced at the end via the same `write_outputs` used by
  the non-streaming path, so the format is identical.
* `--restart` ignores an existing checkpoint and starts fresh.
* `--checkpoint PATH` sets a custom checkpoint location (implies `--stream-output`).

The checkpoint stores a hash of the search parameters and refuses to resume a run
launched with different parameters.

## Validation (required before trusting `--gpu-kernels`)

The completion bar is **bit-identical CPU/GPU output**. Certify it on your GPU:

```bash
# Full parity + regression + property tests. On a GPU host this exercises the
# real kernels; everywhere it certifies the kernel logic against the exact CPU
# semantics via the NumPy reference.
python scripts/tests/test_cpu_gpu_parity.py
# or: pytest scripts/tests/test_cpu_gpu_parity.py
```

The suite checks: encoder correctness; `reference_base_content_prefilter` ==
CPU across 200 randomized synthetic genomes + edge cases (empty, window>len,
all-N, exact bounds, RNA/U, lowercase); the GPU kernel == CPU (incl. the chunked
path); and a full `scan_sequence` CPU-vs-GPU equality. Only enable `--gpu-kernels`
in production after this passes on your device.

## Benchmarking

```bash
pip install psutil   # optional, for host RAM tracking
python scripts/benchmark_search.py --sizes 1e5,1e6,1e7 --queries 1,2,4 --engines cpu,gpu --csv bench.csv
```

Reports wall time, throughput (bases/s), speedup, peak host RAM, and VRAM, and
scales across genome size and query count. GPU rows show `n/a` when no device is
present.

## Building on Windows

1. Install the **NVIDIA CUDA Toolkit** matching your driver (e.g. CUDA 12.x).
2. Install CuPy for that CUDA version into the same environment the toolkit uses:
   ```powershell
   .\.venv\Scripts\python -m pip install cupy-cuda12x
   ```
   (Use `cupy-cuda11x` for CUDA 11.) No `nvcc`/CMake build is needed — the
   kernels are compiled at runtime by CuPy's NVRTC.
3. Verify:
   ```powershell
   .\.venv\Scripts\python -c "import mako_gpu; print(mako_gpu.backend_info())"
   ```
4. Run the parity test (above), then use `--engine gpu --gpu-kernels`.

### Linux notes

Identical, except install the CUDA Toolkit via your distro / NVIDIA's runfile and
the matching `cupy-cudaXYz` wheel. A C++/CUDA build toolchain is only required if
you later replace the CuPy RawKernels with a compiled CPython/PyBind11 extension.

## Dependencies

* Required (CPU): see `requirements.txt`.
* GPU (optional): `cupy-cudaXYz` matching your CUDA Toolkit + an NVIDIA GPU.
* Benchmark (optional): `psutil` for host RAM tracking.

## Trade-offs & future work

* **4-bit vs 2-bit encoding.** The 4-bit IUPAC nibble encoding mirrors the
  existing Cython mask logic exactly (and handles `N`/ambiguity), so it is the
  basis for the fused kernel. 2-bit packing is provided for pure-ACGT fast paths
  but flags ambiguous positions for CPU fallback to preserve exact results.
* **Ordered MP consumption.** Deterministic resume requires consuming worker
  results in submission order, which can hold some completed-but-unconsumed
  results in RAM if early sequences are slow. Fine at chromosome scale; for
  hundreds of thousands of short sequences, batch them.
* **Finalize memory.** The streamed JSONL is the low-RAM artifact; the final TSV
  step reloads hits (not the genome) to reuse `write_outputs`. For extreme hit
  counts, process the JSONL directly or add a two-pass finalizer.
* **Next kernels.** Fused per-window predicate (motif/repeat/composition) and the
  complementarity scans, each gated by the same parity harness.
