# cython: boundscheck=False, wraparound=False, cdivision=True, infer_types=True
"""
High-performance scanning routines for pyrimidine-motif RNA–DNA triplex discovery.
"""

import csv
from pathlib import Path

from cython cimport Py_ssize_t


cdef inline int _score_pair(unsigned char r_base, unsigned char d_base) nogil:
    """Score a single RNA vs DNA-purine base according to simplified Hoogsteen rules."""
    if r_base == 85:  # U
        if d_base == 65:  # A
            return 2
    elif r_base == 67:  # C
        if d_base == 71:  # G
            return 2
    elif r_base == 71:  # G weak match with G
        if d_base == 71:
            return 1
    elif r_base == 65:  # A weak match with A
        if d_base == 65:
            return 1
    return -2


cdef int _score_window(
    const unsigned char* rna,
    const unsigned char* dna,
    Py_ssize_t length,
    int* mismatch_count
) nogil:
    """Compute the ungapped alignment score over a window and count mismatches."""
    cdef Py_ssize_t i
    cdef int total = 0
    cdef int pair_score
    mismatch_count[0] = 0

    for i in range(length):
        pair_score = _score_pair(rna[i], dna[i])
        total += pair_score
        if pair_score <= 0:
            mismatch_count[0] += 1

    return total


cpdef list find_triplex_hits(
    bytes rna_seq,
    bytes dna_seq,
    int min_len=15,
    int max_len=40,
    int min_score=20,
    int max_mismatches=3
):
    """
    Scan the purine strand of a DNA sequence for RNA triplex windows satisfying score filters.

    Returns list of tuples: (start, end, length, score, mismatch_count).
    """
    cdef Py_ssize_t dna_len = len(dna_seq)
    cdef Py_ssize_t rna_len = len(rna_seq)
    cdef int window_min = min_len
    cdef int window_max = max_len
    cdef Py_ssize_t start, limit
    cdef int window_len
    cdef int score
    cdef int mismatch_tmp

    if dna_len == 0 or rna_len == 0:
        return []

    if window_min < 12:
        window_min = 12
    if window_max > rna_len:
        window_max = rna_len
    if window_max > 40:
        window_max = 40
    if window_min > window_max:
        window_min = window_max

    cdef const unsigned char[:] rna_mv = rna_seq
    cdef const unsigned char[:] dna_mv = dna_seq
    cdef list hits = []

    for window_len in range(window_min, window_max + 1):
        if dna_len < window_len:
            break
        limit = dna_len - window_len + 1
        for start in range(limit):
            mismatch_tmp = 0
            score = _score_window(&rna_mv[0], &dna_mv[start], window_len, &mismatch_tmp)
            if score >= min_score and mismatch_tmp <= max_mismatches:
                hits.append((start, start + window_len, window_len, score, mismatch_tmp))

    return hits


def write_triplex_hits_file(
    bytes rna_seq,
    bytes dna_seq,
    str seq_id,
    str output_name,
    str output_prefix="triplex_hits",
    int min_len=15,
    int max_len=40,
    int min_score=20,
    int max_mismatches=3,
    bint truncate_existing=False,
):
    """
    Run the triplex scan and append results to output_<name>/<prefix>.tsv.

    Returns the list of hits written to disk.
    """
    hits = find_triplex_hits(
        rna_seq,
        dna_seq,
        min_len=min_len,
        max_len=max_len,
        min_score=min_score,
        max_mismatches=max_mismatches,
    )
    output_dir = Path(f"output_{output_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / f"{output_prefix}.tsv"
    if truncate_existing and tsv_path.exists():
        tsv_path.unlink()
    headers = [
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
    file_exists = tsv_path.exists()
    with tsv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if not file_exists:
            writer.writerow(headers)
        for start, end, length, score, mismatches in hits:
            writer.writerow(
                [seq_id, start, end, "+", score, length, mismatches, 0, length]
            )
    return hits
