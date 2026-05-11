# cython: boundscheck=False, wraparound=False, initializedcheck=False, language_level=3
"""Baseline Cython helpers shared by nt_sequence_search."""

from cpython.unicode cimport PyUnicode_AsUTF8AndSize
from libc.math cimport fabs
from libc.stdlib cimport free, malloc


cdef inline unsigned char to_upper(unsigned char value):
    if 97 <= value <= 122:  # a-z
        return value - 32
    return value


cdef inline unsigned int mask_from_code(unsigned char code):
    cdef unsigned char upper = to_upper(code)
    if upper == 65:  # A
        return 1
    elif upper == 67:  # C
        return 2
    elif upper == 71:  # G
        return 4
    elif upper in (84, 85):  # T or U
        return 8
    elif upper == 82:  # R: A or G
        return 5
    elif upper == 89:  # Y: C or T
        return 10
    elif upper == 83:  # S: G or C
        return 6
    elif upper == 87:  # W: A or T
        return 9
    elif upper == 75:  # K: G or T
        return 12
    elif upper == 77:  # M: A or C
        return 3
    elif upper == 66:  # B: C/G/T
        return 14
    elif upper == 68:  # D: A/G/T
        return 13
    elif upper == 72:  # H: A/C/T
        return 11
    elif upper == 86:  # V: A/C/G
        return 7
    elif upper == 78:  # N: any base
        return 15
    else:
        raise ValueError(f"Unsupported IUPAC code: {chr(code)}")


cdef inline unsigned int mask_from_base(unsigned char base):
    cdef unsigned char upper = to_upper(base)
    if upper == 65:
        return 1
    elif upper == 67:
        return 2
    elif upper == 71:
        return 4
    elif upper in (84, 85):
        return 8
    elif upper == 78:  # treat N as matching no canonical base
        return 0
    else:
        return 0


cdef int longest_run_bytes(const char* seq_data, Py_ssize_t seq_len, unsigned char target):
    cdef Py_ssize_t idx
    cdef int longest = 0
    cdef int current = 0
    for idx in range(seq_len):
        if seq_data[idx] == target:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


cpdef int longest_run(str sequence, str base):
    cdef Py_ssize_t seq_len, base_len
    cdef const char* seq_data = PyUnicode_AsUTF8AndSize(sequence, &seq_len)
    cdef const char* base_data = PyUnicode_AsUTF8AndSize(base, &base_len)
    if seq_data is NULL or base_data is NULL:
        raise ValueError("Failed to access sequence data.")
    if base_len == 0:
        return 0
    cdef unsigned char target = to_upper(base_data[0])
    return longest_run_bytes(seq_data, seq_len, target)


cpdef int count_iupac_motif(str sequence, str motif):
    cdef Py_ssize_t seq_len, motif_len
    cdef const char* seq_data = PyUnicode_AsUTF8AndSize(sequence, &seq_len)
    cdef const char* motif_data = PyUnicode_AsUTF8AndSize(motif, &motif_len)
    if seq_data is NULL or motif_data is NULL:
        raise ValueError("Failed to access sequence data.")
    if motif_len == 0 or motif_len > seq_len:
        return 0
    cdef unsigned int* motif_masks = <unsigned int*>malloc(motif_len * sizeof(unsigned int))
    if motif_masks is NULL:
        raise MemoryError()
    cdef Py_ssize_t idx
    cdef int count = 0
    cdef Py_ssize_t offset, inner
    cdef unsigned int base_mask
    cdef bint match
    try:
        for idx in range(motif_len):
            motif_masks[idx] = mask_from_code(<unsigned char>motif_data[idx])
        for offset in range(seq_len - motif_len + 1):
            match = True
            for inner in range(motif_len):
                base_mask = mask_from_base(<unsigned char>seq_data[offset + inner])
                if base_mask == 0 or (motif_masks[inner] & base_mask) == 0:
                    match = False
                    break
            if match:
                count += 1
        return count
    finally:
        free(motif_masks)


cpdef tuple base_counts(str sequence):
    cdef Py_ssize_t seq_len
    cdef const char* seq_data = PyUnicode_AsUTF8AndSize(sequence, &seq_len)
    if seq_data is NULL:
        raise ValueError("Failed to access sequence data.")
    cdef long a_count = 0
    cdef long c_count = 0
    cdef long g_count = 0
    cdef long t_count = 0
    cdef Py_ssize_t idx
    cdef unsigned char upper
    for idx in range(seq_len):
        upper = to_upper(seq_data[idx])
        if upper == 65:
            a_count += 1
        elif upper == 67:
            c_count += 1
        elif upper == 71:
            g_count += 1
        elif upper in (84, 85):
            t_count += 1
    return a_count, c_count, g_count, t_count


cpdef double g4hunter_score(str sequence, int window):
    cdef Py_ssize_t seq_len
    cdef const char* seq_data = PyUnicode_AsUTF8AndSize(sequence, &seq_len)
    if seq_data is NULL:
        raise ValueError("Failed to access sequence data.")
    if seq_len == 0:
        return 0.0
    if window <= 0 or window > seq_len:
        window = seq_len
    cdef Py_ssize_t segments = seq_len - window + 1
    cdef Py_ssize_t offset, inner
    cdef double total
    cdef double accum = 0.0
    cdef unsigned char base
    for offset in range(segments):
        total = 0.0
        for inner in range(window):
            base = to_upper(seq_data[offset + inner])
            if base == 71:
                total += 1.0
            elif base == 67:
                total -= 1.0
        accum += fabs(total / window)
    return accum / segments


cpdef double g4boost_score(str sequence):
    cdef Py_ssize_t seq_len
    cdef const char* seq_data = PyUnicode_AsUTF8AndSize(sequence, &seq_len)
    if seq_data is NULL:
        raise ValueError("Failed to access sequence data.")
    if seq_len == 0:
        return 0.0
    cdef long a_count = 0
    cdef long c_count = 0
    cdef long g_count = 0
    cdef long t_count = 0
    cdef Py_ssize_t idx
    cdef unsigned char upper
    for idx in range(seq_len):
        upper = to_upper(seq_data[idx])
        if upper == 65:
            a_count += 1
        elif upper == 67:
            c_count += 1
        elif upper == 71:
            g_count += 1
        elif upper in (84, 85):
            t_count += 1
    cdef double length = seq_len
    cdef double g_pct = g_count / length
    cdef double c_pct = c_count / length
    cdef double purine_pct = (g_count + a_count) / length
    cdef int g_run = longest_run_bytes(seq_data, seq_len, 71)
    cdef double balance = 1.0 - fabs(g_pct - c_pct)
    cdef double run_ratio = g_run / length
    cdef double score = 0.55 * g_pct + 0.25 * run_ratio + 0.1 * purine_pct + 0.1 * balance
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
