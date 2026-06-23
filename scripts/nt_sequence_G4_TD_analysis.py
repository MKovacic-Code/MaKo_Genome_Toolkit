#!/usr/bin/env python3
"""Compute G4/G4Hunter/Tm metrics for nucleotide scan hits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from cython_helpers import (
        base_counts as cy_base_counts,
        g4boost_score as cy_g4boost_score,
        g4hunter_score as cy_g4hunter_score,
        longest_run as cy_longest_run,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    cy_base_counts = None
    cy_g4boost_score = None
    cy_g4hunter_score = None
    cy_longest_run = None

try:
    from Bio.SeqUtils import MeltingTemp as BioMelting
except ImportError:  # pragma: no cover - optional dependency
    BioMelting = None

BASES = ("A", "C", "G", "T")
COMPLEMENT_MAP = str.maketrans("ACGTacgt", "TGCAtgca")

# Approximate codon adaptability weights (normalized ~0-1).
CODON_WEIGHTS: Dict[str, float] = {
    "TTT": 0.58,
    "TTC": 0.92,
    "TTA": 0.14,
    "TTG": 0.13,
    "CTT": 0.13,
    "CTC": 0.4,
    "CTA": 0.14,
    "CTG": 1.0,
    "ATT": 0.46,
    "ATC": 0.91,
    "ATA": 0.17,
    "ATG": 1.0,
    "GTT": 0.18,
    "GTC": 0.47,
    "GTA": 0.11,
    "GTG": 1.0,
    "TCT": 0.19,
    "TCC": 0.56,
    "TCA": 0.15,
    "TCG": 0.13,
    "AGT": 0.15,
    "AGC": 0.68,
    "CCT": 0.31,
    "CCC": 0.34,
    "CCA": 0.3,
    "CCG": 1.0,
    "ACT": 0.28,
    "ACC": 0.89,
    "ACA": 0.31,
    "ACG": 0.4,
    "GCT": 0.27,
    "GCC": 1.0,
    "GCA": 0.34,
    "GCG": 0.54,
    "TAT": 0.45,
    "TAC": 0.55,
    "CAT": 0.39,
    "CAC": 0.61,
    "CAA": 0.29,
    "CAG": 0.71,
    "AAT": 0.47,
    "AAC": 0.53,
    "AAA": 0.42,
    "AAG": 0.58,
    "GAT": 0.46,
    "GAC": 0.54,
    "GAA": 0.43,
    "GAG": 0.57,
    "TGT": 0.38,
    "TGC": 0.62,
    "TGG": 1.0,
    "CGT": 0.4,
    "CGC": 1.0,
    "CGA": 0.11,
    "CGG": 0.21,
    "AGA": 0.2,
    "AGG": 0.2,
    "GGT": 0.33,
    "GGC": 1.0,
    "GGA": 0.5,
    "GGG": 0.55,
    "TAA": 0.0,
    "TAG": 0.0,
    "TGA": 0.0,
}

NN_PARAMS = {
    "AA": (-7.9, -22.2),
    "TT": (-7.9, -22.2),
    "AT": (-7.2, -20.4),
    "TA": (-7.2, -21.3),
    "CA": (-8.5, -22.7),
    "TG": (-8.5, -22.7),
    "GT": (-8.4, -22.4),
    "AC": (-8.4, -22.4),
    "CT": (-7.8, -21.0),
    "AG": (-7.8, -21.0),
    "GA": (-8.2, -22.2),
    "TC": (-8.2, -22.2),
    "CG": (-10.6, -27.2),
    "GC": (-9.8, -24.4),
    "GG": (-8.0, -19.9),
    "CC": (-8.0, -19.9),
}

NN_INIT = {"dh": 0.2, "ds": -5.7}
R_GAS = 1.987  # cal/mol*K


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "\t" in line:
                return "\t"
            if "," in line:
                return ","
    return "\t"


def load_hits(path: Path) -> List[Dict[str, str]]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [row for row in reader]


def merge_overlapping_hits(rows: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    by_seq: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        seq_id = row.get("sequence_id") or row.get("chrom") or row.get("seq_id") or "unknown"
        strand = row.get("strand", "+") or "+"
        by_seq.setdefault((seq_id, strand), []).append(row)
    merged: List[Dict[str, object]] = []

    def finalize(
        seq_id: str,
        strand: str,
        start: int,
        end: int,
        sequence: str,
        count: int,
    ) -> None:
        merged.append(
            {
                "sequence_id": seq_id,
                "strand": strand,
                "window_start": start,
                "window_end": end,
                "window_sequence": sequence,
                "merged_windows": count,
            }
        )

    for (seq_id, strand), bucket in by_seq.items():
        bucket.sort(key=lambda row: int(row.get("window_start") or row.get("start") or 0))
        current_start = None
        current_end = None
        current_sequence = ""
        count = 0
        for row in bucket:
            try:
                start = int(row.get("window_start") or row.get("start") or 0)
                end = int(row.get("window_end") or row.get("end") or start)
            except ValueError:
                continue
            sequence = row.get("window_sequence") or row.get("sequence") or ""
            if current_start is None:
                current_start = start
                current_end = end
                current_sequence = sequence
                count = 1
                continue
            if start > current_end:
                finalize(seq_id, strand, current_start, current_end, current_sequence, count)
                current_start = start
                current_end = end
                current_sequence = sequence
                count = 1
                continue
            count += 1
            overlap = current_end - start + 1
            if overlap < len(sequence):
                current_sequence += sequence[overlap:]
            current_end = max(current_end, end)
        if current_start is not None:
            finalize(seq_id, strand, current_start, current_end, current_sequence, count)
    return merged


def base_composition(sequence: str) -> Tuple[Counter, float]:
    if cy_base_counts is not None:
        counts = Counter(dict(zip(BASES, cy_base_counts(sequence))))
    else:
        counts = Counter(sequence)
    gc_count = counts.get("G", 0) + counts.get("C", 0)
    gc_pct = (gc_count / len(sequence)) * 100 if sequence else 0.0
    return counts, gc_pct


def _g4hunter_base_scores(sequence: str) -> List[int]:
    """Per-base G4Hunter scores (Bedrat, Lacroix & Mergny, NAR 2016).

    Every base in a run of *n* consecutive G's scores ``+min(n, 4)``; every base
    in a run of *n* consecutive C's scores ``-min(n, 4)``; A/T and any other
    character score 0. Run length is evaluated in the context of the whole
    sequence, which is what makes the score sensitive to G-tract structure
    rather than mere G content.
    """
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
    return scores


def g4hunter_score(sequence: str, window: int = 0) -> float:
    """Canonical G4Hunter score: the signed mean of per-base run-length scores.

    With ``window`` <= 0 or >= len(sequence) the whole-sequence mean is returned
    (the G4Hunter score of ``sequence``). With a smaller positive ``window`` the
    sliding-window mean of largest magnitude is returned (sign preserved),
    matching the convention used when locating the most G4/i-motif-prone
    sub-region. Positive values indicate G4 (G-rich) propensity, negative values
    indicate i-motif (C-rich) propensity; |score| ~>= 1.0-1.5 is the usual G4
    calling threshold. Range is roughly -4 .. +4.
    """
    if cy_g4hunter_score is not None:
        return cy_g4hunter_score(sequence, window)
    seq_len = len(sequence)
    if seq_len == 0:
        return 0.0
    scores = _g4hunter_base_scores(sequence)
    if window <= 0 or window >= seq_len:
        return sum(scores) / seq_len
    window_sum = sum(scores[:window])
    best = window_sum / window
    for i in range(window, seq_len):
        window_sum += scores[i] - scores[i - window]
        mean_val = window_sum / window
        if abs(mean_val) > abs(best):
            best = mean_val
    return best


def g4boost_score(sequence: str) -> float:
    """Composite G4-propensity heuristic in [0, 1] (higher = more G4-prone).

    NOTE: this is a custom weighted heuristic, not the published G4Boost ML
    model. The pure-Python path mirrors the Cython implementation exactly so
    both return identical values regardless of whether the extension is loaded.
    """
    if cy_g4boost_score is not None:
        return cy_g4boost_score(sequence)
    seq = sequence.upper()
    length = len(seq)
    if length == 0:
        return 0.0
    g_count = seq.count("G")
    c_count = seq.count("C")
    a_count = seq.count("A")
    g_pct = g_count / length
    c_pct = c_count / length
    purine_pct = (g_count + a_count) / length
    run_ratio = longest_run(seq, "G") / length
    balance = 1.0 - abs(g_pct - c_pct)
    score = 0.55 * g_pct + 0.25 * run_ratio + 0.1 * purine_pct + 0.1 * balance
    return min(1.0, max(0.0, score))


def calc_codon_efficiency(sequence: str) -> float:
    total = 0.0
    count = 0
    for i in range(0, len(sequence), 3):
        codon = sequence[i : i + 3]
        if len(codon) == 3 and codon.upper() in CODON_WEIGHTS:
            total += CODON_WEIGHTS[codon.upper()]
            count += 1
    return total / count if count else 0.0


def longest_run(sequence: str, base: str) -> int:
    if cy_longest_run is not None:
        return cy_longest_run(sequence, base)
    longest = 0
    current = 0
    for char in sequence:
        if char == base:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT_MAP)[::-1]


def i_motif_score(sequence: str) -> float:
    length = len(sequence)
    if length == 0:
        return 0.0
    seq = sequence.upper()
    c_ratio = seq.count("C") / length
    longest_c = longest_run(seq, "C")
    c_runs = len(re.findall(r"C{3,}", seq))
    score = c_ratio * (longest_c + c_runs)
    return round(score, 4)


def r_loop_score(sequence: str) -> float:
    length = len(sequence)
    if length == 0:
        return 0.0
    seq = sequence.upper()
    g_count = seq.count("G")
    c_count = seq.count("C")
    gc_total = g_count + c_count
    if gc_total == 0:
        return 0.0
    skew = abs(g_count - c_count) / gc_total
    gc_fraction = gc_total / length
    score = gc_fraction * skew
    return round(score, 4)


def hairpin_score(sequence: str, min_stem: int = 3, max_loop: int = 12) -> float:
    seq = sequence.upper()
    n = len(seq)
    if n < min_stem * 2:
        return 0.0
    best = 0
    comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
    max_stem_len = min(30, n // 2)
    for stem in range(min_stem, max_stem_len + 1):
        max_loop_len = min(max_loop, n - 2 * stem)
        for loop in range(max_loop_len + 1):
            max_start = n - (2 * stem + loop)
            if max_start < 0:
                break
            for start in range(max_start + 1):
                left = seq[start : start + stem]
                right_start = start + stem + loop
                right = seq[right_start : right_start + stem]
                if len(right) < stem:
                    break
                match = True
                for idx in range(stem):
                    if comp.get(left[idx], "") != right[stem - idx - 1]:
                        match = False
                        break
                if match:
                    best = max(best, stem)
    if best == 0:
        return 0.0
    return round(best / n, 4)


def cpg_island_score(sequence: str) -> float:
    seq = sequence.upper()
    length = len(seq)
    if length == 0:
        return 0.0
    g_count = seq.count("G")
    c_count = seq.count("C")
    gc_fraction = (g_count + c_count) / length
    observed = seq.count("CG")
    expected = (c_count * g_count) / length if length else 0.0
    ratio = (observed / expected) if expected > 0 else 0.0
    score = min(gc_fraction * ratio, 1.0)
    return round(score, 4)


def find_top_oligos(sequence: str, min_len: int, max_len: int, g4_window: int, top_n: int) -> List[Dict[str, object]]:
    seq_len = len(sequence)
    if seq_len < min_len:
        return []
    candidates: List[Dict[str, object]] = []
    for length in range(min_len, min(max_len, seq_len) + 1):
        window_size = min(g4_window, length)
        for start in range(0, seq_len - length + 1):
            end = start + length
            subseq = sequence[start:end]
            g4h = g4hunter_score(subseq, window_size)
            g4b = g4boost_score(subseq)
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "sequence": subseq,
                    "g4hunter": g4h,
                    "g4boost": g4b,
                }
            )
    candidates.sort(key=lambda item: (item["g4hunter"], item["g4boost"]), reverse=True)
    return candidates[: max(1, top_n)]


def tm_wallace(sequence: str) -> float:
    if not sequence:
        return float("nan")
    counts = Counter(sequence.upper())
    return 2 * (counts.get("A", 0) + counts.get("T", 0)) + 4 * (counts.get("G", 0) + counts.get("C", 0))


def tm_nearest_neighbor(sequence: str, dna_conc: float, salt: float) -> Tuple[float, float, float]:
    if len(sequence) < 2:
        return float("nan"), float("nan"), float("nan")
    seq = sequence.upper().replace("U", "T")
    delta_h = NN_INIT["dh"]
    delta_s = NN_INIT["ds"]
    for i in range(len(seq) - 1):
        pair = seq[i : i + 2]
        params = NN_PARAMS.get(pair)
        if params is None:
            continue
        delta_h += params[0]
        delta_s += params[1]
    if delta_s == 0:
        return float("nan"), delta_h, delta_s
    tm = (1000 * delta_h) / (delta_s + (1.987 * math.log(dna_conc / 4))) - 273.15 + 16.6 * math.log10(salt)
    return tm, delta_h, delta_s


def _process_hit(
    row: Dict[str, object],
    min_oligo: int,
    max_oligo: int,
    g4_window: int,
    dna_conc: float,
    salt: float,
    reaction_temp: float,
    compute_structures: bool,
    top_oligos: int,
    structure_only: bool,
) -> List[Dict[str, object]]:
    region_sequence_raw = row["window_sequence"]
    region_sequence = str(region_sequence_raw)
    analysis_sequence = region_sequence.upper()
    region_start = int(row["window_start"])
    region_end = int(row["window_end"])
    seq_id = str(row["sequence_id"])
    strand = (row.get("strand") or "+")[:1]
    region_g4hunter = g4hunter_score(analysis_sequence, min(g4_window, len(analysis_sequence)))
    region_g4boost = g4boost_score(analysis_sequence)
    top_candidates = find_top_oligos(analysis_sequence, min_oligo, max_oligo, g4_window, top_oligos)
    rows: List[Dict[str, object]] = []
    for rank, candidate in enumerate(top_candidates, start=1):
        local_start = candidate["start"]
        abs_start = region_start + local_start
        abs_end = abs_start + (candidate["end"] - candidate["start"]) - 1
        seq = candidate["sequence"]
        counts, gc_pct = base_composition(seq)
        if structure_only:
            codon_eff: float | str = "NA"
            tm_wallace_value: float | str = "NA"
            tm_nn_value: float | str = "NA"
            nn_dh: float | str = "NA"
            nn_ds: float | str = "NA"
        else:
            codon_eff = round(calc_codon_efficiency(seq), 4)
            tm_wallace_value = round(tm_wallace(seq), 2)
            tm_nn_value_raw, nn_dh_raw, nn_ds_raw = tm_nearest_neighbor(seq, dna_conc, salt)
            tm_nn_value = round(tm_nn_value_raw, 2) if not math.isnan(tm_nn_value_raw) else "NA"
            nn_dh = round(nn_dh_raw, 4)
            nn_ds = round(nn_ds_raw, 4)
        best_seq = seq
        if compute_structures:
            i_motif = i_motif_score(best_seq)
            r_loop = r_loop_score(best_seq)
            hairpin = hairpin_score(best_seq)
            cpg_score = cpg_island_score(best_seq)
        else:
            i_motif = "NA"
            r_loop = "NA"
            hairpin = "NA"
            cpg_score = "NA"
        rows.append(
            {
                "sequence_id": seq_id,
                "strand": strand,
                "region_start": region_start,
                "region_end": region_end,
                "region_length": region_end - region_start + 1,
                "region_sequence": region_sequence,
                "region_source_windows": row.get("merged_windows", 1),
                "region_g4hunter": round(region_g4hunter, 4),
                "region_g4boost": round(region_g4boost, 4),
                "oligo_rank": rank,
                "best_oligo_sequence": seq,
                "best_oligo_start": abs_start,
                "best_oligo_end": abs_end,
                "best_oligo_length": candidate["end"] - candidate["start"],
                "best_oligo_g4hunter": round(candidate["g4hunter"], 4),
                "best_oligo_g4boost": round(candidate["g4boost"], 4),
                "best_oligo_gc_pct": round(gc_pct, 2),
                "best_oligo_base_counts": ";".join(f"{base}={counts[base]}" for base in BASES),
                "best_oligo_codon_efficiency": codon_eff,
                "best_oligo_tm_wallace": tm_wallace_value,
                "best_oligo_tm_nearest_neighbor": tm_nn_value,
                "best_oligo_nn_delta_h_kcal": nn_dh,
                "best_oligo_nn_delta_s_cal": nn_ds,
                "best_oligo_i_motif_score": i_motif,
                "best_oligo_r_loop_score": r_loop,
                "best_oligo_hairpin_score": hairpin,
                "best_oligo_cpg_score": cpg_score,
                "tm_reference_temperature": round(reaction_temp, 2),
            }
        )
    return rows


def _process_hit_worker(payload):
    return _process_hit(*payload)


def _passes_thresholds(row: Dict[str, object], thresholds: Dict[str, float | None]) -> bool:
    for key, threshold in thresholds.items():
        if threshold is None:
            continue
        value = row.get(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if numeric < threshold:
            return False
    return True


def _update_summary_vectors(summary: Dict[str, List[float]], row: Dict[str, object], structure_only: bool) -> None:
    def try_append(key: str) -> None:
        value = row.get(key)
        if value in ("NA", None, ""):
            return
        try:
            summary[key].append(float(value))
        except (TypeError, ValueError):
            return

    for key in ("best_oligo_g4hunter", "best_oligo_g4boost"):
        try_append(key)
    if not structure_only:
        for key in ("best_oligo_tm_wallace", "best_oligo_tm_nearest_neighbor"):
            try_append(key)
    for key in ("best_oligo_i_motif_score", "best_oligo_r_loop_score", "best_oligo_hairpin_score", "best_oligo_cpg_score"):
        try_append(key)


def build_histogram(values: List[float], bins: int = 12) -> List[Dict[str, float]]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if v_min == v_max:
        return [{"start": v_min, "end": v_max, "count": len(values)}]
    step = (v_max - v_min) / bins
    hist = [{"start": v_min + step * idx, "end": v_min + step * (idx + 1), "count": 0} for idx in range(bins)]
    for value in values:
        if value == v_max:
            hist[-1]["count"] += 1
            continue
        idx = int((value - v_min) / step)
        hist[min(idx, bins - 1)]["count"] += 1
    return hist


def write_summary(summary_vectors: Dict[str, List[float]], path: str) -> None:
    if not path:
        return
    path_obj = Path(path)
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    summary_payload: Dict[str, object] = {
        "fields": {},
    }
    for key, values in summary_vectors.items():
        if not values:
            continue
        summary_payload["fields"][key] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "histogram": build_histogram(values),
        }
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2)
    except Exception:
        pass


def process_hits(
    hits: Sequence[Dict[str, object]],
    min_oligo: int,
    max_oligo: int,
    g4_window: int,
    workers: int,
    dna_conc: float,
    monovalent_salt: float,
    reaction_temp: float,
    min_g4hunter: float | None = None,
    min_g4boost: float | None = None,
    min_tm_wallace: float | None = None,
    min_tm_nearest: float | None = None,
    min_i_motif: float | None = None,
    min_r_loop: float | None = None,
    min_hairpin: float | None = None,
    min_cpg: float | None = None,
    top_oligos: int = 1,
    structure_only: bool = False,
    summary_path: str | None = None,
) -> List[Dict[str, object]]:
    worker_count = workers
    if worker_count == 0:
        worker_count = os.cpu_count() or 1
    worker_count = max(1, worker_count)
    thresholds = {
        "best_oligo_g4hunter": min_g4hunter,
        "best_oligo_g4boost": min_g4boost,
        "best_oligo_tm_wallace": min_tm_wallace,
        "best_oligo_tm_nearest_neighbor": min_tm_nearest,
        "best_oligo_i_motif_score": min_i_motif,
        "best_oligo_r_loop_score": min_r_loop,
        "best_oligo_hairpin_score": min_hairpin,
        "best_oligo_cpg_score": min_cpg,
    }
    structure_thresholds = (
        min_i_motif,
        min_r_loop,
        min_hairpin,
        min_cpg,
    )
    compute_structures = True
    payloads = [
        (
            row,
            min_oligo,
            max_oligo,
            g4_window,
            dna_conc,
            monovalent_salt,
            reaction_temp,
            compute_structures,
            max(1, top_oligos),
            structure_only,
        )
        for row in hits
    ]
    rows: List[Dict[str, object]] = []
    total = len(payloads)
    summary_vectors: Dict[str, List[float]] = defaultdict(list)
    if total:
        print(
            f"[g4] Computing thermo metrics for {total} merged regions using {worker_count} worker(s)...",
            flush=True,
        )
    if worker_count == 1 or total <= 1:
        for index, payload in enumerate(payloads, 1):
            print(f"[g4] Processing region {index}/{total}", flush=True)
            for result in _process_hit(*payload):
                if _passes_thresholds(result, thresholds):
                    rows.append(result)
                    _update_summary_vectors(summary_vectors, result, structure_only)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for index, result in enumerate(executor.map(_process_hit_worker, payloads), 1):
                print(f"[g4] Processed region {index}/{total}", flush=True)
                for record in result:
                    if _passes_thresholds(record, thresholds):
                        rows.append(record)
                        _update_summary_vectors(summary_vectors, record, structure_only)
    if summary_path:
        write_summary(summary_vectors, summary_path)
    return rows


def write_outputs(rows: Sequence[Dict[str, object]], output_dir: Path, output_prefix: str) -> None:
    headers = [
        "sequence_id",
        "strand",
        "region_start",
        "region_end",
        "region_length",
        "region_sequence",
        "region_source_windows",
        "region_g4hunter",
        "region_g4boost",
        "oligo_rank",
        "best_oligo_sequence",
        "best_oligo_start",
        "best_oligo_end",
        "best_oligo_length",
        "best_oligo_g4hunter",
        "best_oligo_g4boost",
        "best_oligo_gc_pct",
        "best_oligo_base_counts",
        "best_oligo_codon_efficiency",
        "best_oligo_tm_wallace",
        "best_oligo_tm_nearest_neighbor",
        "best_oligo_nn_delta_h_kcal",
        "best_oligo_nn_delta_s_cal",
        "best_oligo_i_motif_score",
        "best_oligo_r_loop_score",
        "best_oligo_hairpin_score",
        "best_oligo_cpg_score",
        "tm_reference_temperature",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{output_prefix}.tsv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row[h] for h in headers])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute G4Hunter/G4Boost/Tm metrics for nt_sequence_hits TSV files."
    )
    parser.add_argument("hits", help="Input nt_sequence_hits TSV file.")
    parser.add_argument(
        "--min-oligo",
        type=int,
        default=15,
        help="Minimum oligonucleotide length evaluated within each window (default: 15).",
    )
    parser.add_argument(
        "--max-oligo",
        type=int,
        default=60,
        help="Maximum oligonucleotide length evaluated within each window (default: 60).",
    )
    parser.add_argument(
        "--g4-window",
        type=int,
        default=25,
        help="Window size used inside the G4Hunter approximation (default: 25).",
    )
    parser.add_argument(
        "--output-prefix",
        default="nt_sequence_hits_analyzed",
        help="Base name for the analysis TSV output (default: nt_sequence_hits_analyzed).",
    )
    parser.add_argument(
        "--output-name",
        default="default",
        help="Name used to create the output_<name> directory for analyzed result files (default: default).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 uses all CPUs).",
    )
    parser.add_argument(
        "--dna-conc",
        type=float,
        default=0.001,
        help="DNA concentration in mol/L for nearest-neighbor Tm estimates (default: 0.001).",
    )
    parser.add_argument(
        "--monovalent-salt",
        type=float,
        default=0.15,
        help="Monovalent salt concentration in mol/L for nearest-neighbor Tm (default: 0.15).",
    )
    parser.add_argument(
        "--reaction-temp",
        type=float,
        default=25.0,
        help="Reference reaction temperature in Celsius stored alongside the results (default: 25).",
    )
    parser.add_argument(
        "--top-oligos",
        type=int,
        default=1,
        help="Number of top oligos per region to retain (default: 1).",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Skip codon and Tm calculations for faster structure-only scoring.",
    )
    parser.add_argument(
        "--summary-json",
        help="Optional path to write histogram summary data as JSON.",
    )
    parser.add_argument(
        "--min-g4hunter",
        type=float,
        help="Minimum best-oligo G4Hunter score required to keep a region.",
    )
    parser.add_argument(
        "--min-g4boost",
        type=float,
        help="Minimum best-oligo G4Boost score required to keep a region.",
    )
    parser.add_argument(
        "--min-tm-wallace",
        type=float,
        help="Minimum Wallace TM (°C) required to keep a region.",
    )
    parser.add_argument(
        "--min-tm-nearest",
        type=float,
        help="Minimum nearest-neighbor TM (°C) required to keep a region.",
    )
    parser.add_argument(
        "--min-i-motif",
        type=float,
        help="Minimum i-motif score required to keep a region.",
    )
    parser.add_argument(
        "--min-r-loop",
        type=float,
        help="Minimum R-loop score required to keep a region.",
    )
    parser.add_argument(
        "--min-hairpin",
        type=float,
        help="Minimum hairpin formation score required to keep a region.",
    )
    parser.add_argument(
        "--min-cpg",
        type=float,
        help="Minimum CpG island score required to keep a region.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    hits_path = Path(args.hits)
    if not hits_path.is_file():
        parser.error(f"Hits file '{hits_path}' does not exist.")
    raw_hits = load_hits(hits_path)
    merged_hits = merge_overlapping_hits(raw_hits)
    if len(merged_hits) < len(raw_hits):
        print(
            f"Merged {len(raw_hits)} input windows into {len(merged_hits)} merged regions.",
            file=sys.stderr,
        )
    analyzed = process_hits(
        merged_hits,
        args.min_oligo,
        args.max_oligo,
        args.g4_window,
        args.workers,
        args.dna_conc,
        args.monovalent_salt,
        args.reaction_temp,
        args.min_g4hunter,
        args.min_g4boost,
        args.min_tm_wallace,
        args.min_tm_nearest,
        args.min_i_motif,
        args.min_r_loop,
        args.min_hairpin,
        args.min_cpg,
        max(1, args.top_oligos),
        bool(args.structure_only),
        args.summary_json,
    )
    output_dir = Path(f"output_{args.output_name}")
    write_outputs(analyzed, output_dir, args.output_prefix)
    print(
        f"Wrote {len(analyzed)} analyzed regions to "
        f"{output_dir / (args.output_prefix + '.tsv')}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
