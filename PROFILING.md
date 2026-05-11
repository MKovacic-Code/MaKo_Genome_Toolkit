# Profiling `scripts/nt_sequence_search.py`

Use `cProfile` to understand where time is spent before and after enabling the Cython helpers.

## Quick one-off profile

```powershell
python -m cProfile -o profile.stats scripts/nt_sequence_search.py genome.fna `
  --window 200 --step 20 --motif AGTC:3 --workers 0
```

This writes a binary profile to `profile.stats`. Inspect the hottest functions with `pstats`:

```powershell
python - <<'PY'
import pstats
stats = pstats.Stats("profile.stats")
stats.strip_dirs().sort_stats("cumulative").print_stats(20)
PY
```

Key functions to watch are `scan_sequence`, `motif_hit_count`, `longest_run`, and `build_prefix_counts`.

## Re-running after Cython builds

After building the extension (instructions in the README below), re-run the same `cProfile` command. Compare the `pstats` output for the pure-Python run versus the Cython-accelerated run to ensure that the hotspots shifted away from `motif_hit_count` / `longest_run` and towards unavoidable Python overhead such as file parsing.

For more detailed visualization, you can load `profile.stats` into third-party viewers (SnakeViz, gprof2dot) if they are available in your environment.

## Building the Cython helper extension

1. Install build tooling (once per environment):

   ```powershell
   python -m pip install --upgrade pip build wheel cython
   ```

2. Compile the helper module in-place:

   ```powershell
   python setup.py build_ext --inplace
   ```

   Successful builds produce a `cython_helpers.*.pyd` (Windows) or `.so` (Linux/macOS) file inside the `scripts` directory (next to `nt_sequence_search.py`). The main script automatically imports it when present—no flag changes are required.

3. Re-run your workload to confirm the speedup and capture new `cProfile` stats as shown above.
