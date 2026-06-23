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
    # Canonical G4Hunter (Bedrat, Lacroix & Mergny, NAR 2016): assign each base a
    # run-length score (+min(run,4) within a G-tract, -min(run,4) within a
    # C-tract, 0 otherwise), then take the signed mean. With window <= 0 or
    # window >= len the whole-sequence mean is returned; with a smaller positive
    # window the sliding-window mean of largest magnitude (sign preserved) is
    # returned. Range is roughly -4 .. +4; positive => G4, negative => i-motif.
    cdef Py_ssize_t seq_len
    cdef const char* seq_data = PyUnicode_AsUTF8AndSize(sequence, &seq_len)
    if seq_data is NULL:
        raise ValueError("Failed to access sequence data.")
    if seq_len == 0:
        return 0.0
    cdef signed char* scores = <signed char*>malloc(seq_len * sizeof(signed char))
    if scores is NULL:
        raise MemoryError()
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j, k
    cdef unsigned char base
    cdef int run, val
    cdef double total = 0.0
    cdef double window_sum = 0.0
    cdef double best, mean_val
    try:
        while i < seq_len:
            base = to_upper(seq_data[i])
            if base == 71 or base == 67:  # G or C
                j = i
                while j < seq_len and to_upper(seq_data[j]) == base:
                    j += 1
                run = <int>(j - i)
                val = run if run < 4 else 4
                if base == 67:  # C-tract is negative
                    val = -val
                for k in range(i, j):
                    scores[k] = <signed char>val
                i = j
            else:
                scores[i] = 0
                i += 1
        if window <= 0 or window >= seq_len:
            for i in range(seq_len):
                total += scores[i]
            return total / seq_len
        for i in range(window):
            window_sum += scores[i]
        best = window_sum / window
        for i in range(window, seq_len):
            window_sum += scores[i] - scores[i - window]
            mean_val = window_sum / window
            if fabs(mean_val) > fabs(best):
                best = mean_val
        return best
    finally:
        free(scores)


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


cdef unsigned char COMPLEMENT_TABLE[256]

cdef void init_table():
    cdef int i
    for i in range(256):
        COMPLEMENT_TABLE[i] = <unsigned char>i
    COMPLEMENT_TABLE[65] = 84   # A -> T
    COMPLEMENT_TABLE[67] = 71   # C -> G
    COMPLEMENT_TABLE[71] = 67   # G -> C
    COMPLEMENT_TABLE[84] = 65   # T -> A
    COMPLEMENT_TABLE[85] = 65   # U -> A
    COMPLEMENT_TABLE[82] = 89   # R -> Y
    COMPLEMENT_TABLE[89] = 82   # Y -> R
    COMPLEMENT_TABLE[75] = 77   # K -> M
    COMPLEMENT_TABLE[77] = 75   # M -> K
    COMPLEMENT_TABLE[83] = 83   # S -> S
    COMPLEMENT_TABLE[87] = 87   # W -> W
    COMPLEMENT_TABLE[66] = 86   # B -> V
    COMPLEMENT_TABLE[68] = 72   # D -> H
    COMPLEMENT_TABLE[72] = 68   # H -> D
    COMPLEMENT_TABLE[86] = 66   # V -> B
    COMPLEMENT_TABLE[78] = 78   # N -> N

    COMPLEMENT_TABLE[97] = 116  # a -> t
    COMPLEMENT_TABLE[99] = 103  # c -> g
    COMPLEMENT_TABLE[103] = 99  # g -> c
    COMPLEMENT_TABLE[116] = 97  # t -> a
    COMPLEMENT_TABLE[117] = 97  # u -> a
    COMPLEMENT_TABLE[114] = 121  # r -> y
    COMPLEMENT_TABLE[121] = 114  # y -> r
    COMPLEMENT_TABLE[107] = 109  # k -> m
    COMPLEMENT_TABLE[109] = 107  # m -> k
    COMPLEMENT_TABLE[115] = 115  # s -> s
    COMPLEMENT_TABLE[119] = 119  # w -> w
    COMPLEMENT_TABLE[98] = 118  # b -> v
    COMPLEMENT_TABLE[100] = 104  # d -> h
    COMPLEMENT_TABLE[104] = 100  # h -> d
    COMPLEMENT_TABLE[118] = 98   # v -> b
    COMPLEMENT_TABLE[110] = 110  # n -> n

init_table()


cdef inline int min_int(int a, int b):
    return a if a < b else b

cdef inline int max_int(int a, int b):
    return a if a > b else b


cpdef tuple find_longest_intrastrand_complement_cython(
    str sequence,
    int min_len,
    object max_len_obj,
    int allowed_mismatches,
    int gap_min_len,
    object gap_max_len_obj
):
    cdef Py_ssize_t n
    cdef const char* seq = PyUnicode_AsUTF8AndSize(sequence, &n)
    if seq is NULL:
        raise ValueError("Failed to access sequence data.")

    cdef int max_len = -1
    if max_len_obj is not None:
        max_len = max_len_obj

    cdef int gap_max_len = -1
    if gap_max_len_obj is not None:
        gap_max_len = gap_max_len_obj

    cdef int best_len = 0
    cdef Py_ssize_t best_i = -1
    cdef Py_ssize_t best_j = -1

    cdef Py_ssize_t i, j
    cdef int length, j_start, j_end, max_possible, mismatches, curr_best_len, gap
    cdef unsigned char left_base, right_base, comp

    for i in range(n - 2 * min_len - gap_min_len + 1):
        j_start = i + 2 * max_int(min_len, best_len + 1) + gap_min_len - 1
        j_end = n
        if max_len != -1 and gap_max_len != -1:
            j_end = min_int(j_end, i + 2 * max_len + gap_max_len)

        for j in range(j_start, j_end):
            max_possible = (j - i + 1 - gap_min_len) // 2
            if max_len != -1 and max_len < max_possible:
                max_possible = max_len

            if max_possible <= best_len:
                continue

            mismatches = 0
            curr_best_len = 0
            for length in range(1, max_possible + 1):
                left_base = <unsigned char>seq[i + length - 1]
                right_base = <unsigned char>seq[j - length + 1]
                comp = COMPLEMENT_TABLE[right_base]
                if left_base != comp:
                    mismatches += 1
                if mismatches > allowed_mismatches:
                    break

                gap = j - i - 2 * length + 1
                if gap >= gap_min_len and (gap_max_len == -1 or gap <= gap_max_len):
                    curr_best_len = length

            if curr_best_len >= min_len and curr_best_len > best_len:
                best_len = curr_best_len
                best_i = i
                best_j = j

    return best_i, best_j, best_len


cpdef tuple find_longest_interstrand_complement_cython(
    str sequence,
    int min_len,
    object max_len_obj,
    int allowed_mismatches,
    int gap_min_len,
    object gap_max_len_obj
):
    cdef Py_ssize_t n
    cdef const char* seq = PyUnicode_AsUTF8AndSize(sequence, &n)
    if seq is NULL:
        raise ValueError("Failed to access sequence data.")

    cdef int max_len = -1
    if max_len_obj is not None:
        max_len = max_len_obj

    cdef int gap_max_len = -1
    if gap_max_len_obj is not None:
        gap_max_len = gap_max_len_obj

    cdef int best_len = 0
    cdef Py_ssize_t best_i = -1
    cdef Py_ssize_t best_j = -1

    cdef Py_ssize_t i, j
    cdef int length, j_start, j_end, max_possible, mismatches, curr_best_len, gap
    cdef unsigned char left_base, right_base

    for i in range(n - 2 * min_len - gap_min_len + 1):
        j_start = i + max_int(min_len, best_len + 1) + gap_min_len
        j_end = n - max_int(min_len, best_len + 1) + 1
        
        if gap_max_len != -1:
            j_end = min_int(j_end, (n + i + gap_max_len) // 2 + 1)
            if max_len != -1:
                j_end = min_int(j_end, i + max_len + gap_max_len + 1)

        for j in range(j_start, j_end):
            max_possible = min_int(j - i - gap_min_len, n - j)
            if max_len != -1 and max_len < max_possible:
                max_possible = max_len

            if max_possible <= best_len:
                continue

            mismatches = 0
            curr_best_len = 0
            for length in range(1, max_possible + 1):
                left_base = <unsigned char>seq[i + length - 1]
                right_base = <unsigned char>seq[j + length - 1]
                if left_base != right_base:
                    mismatches += 1
                if mismatches > allowed_mismatches:
                    break

                gap = j - i - length
                if gap >= gap_min_len and (gap_max_len == -1 or gap <= gap_max_len):
                    curr_best_len = length

            if curr_best_len >= min_len and curr_best_len > best_len:
                best_len = curr_best_len
                best_i = i
                best_j = j

    return best_i, best_j, best_len

