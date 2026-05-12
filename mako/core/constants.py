"""Shared constants for the MaKo Genome Toolkit."""

IUPAC_CODE = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "U",
    "R": "[AG]", "Y": "[CTU]", "S": "[GC]", "W": "[ATU]",
    "K": "[GTU]", "M": "[AC]", "B": "[CGTU]", "D": "[AGTU]",
    "H": "[ACTU]", "V": "[ACG]", "N": "[ACGTU]",
}

DNA_COMPLEMENT = str.maketrans("atgcATGC", "tacgTACG")

TAB_ACCENTS = {
    "inputs": "#0f766e",    # Teal-700
    "scanner": "#1d4ed8",   # Blue-700
    "peptide": "#b45309",   # Amber-700
    "triplex": "#7c3aed",   # Purple-700
    "combined": "#be123c",  # Rose-700
    "viz": "#b45309",       # Amber-700
    "tools": "#0369a1",     # Sky-700
}

VECTOR_CLASS_COLORS = {
    "lnc_rna": "#8e44ad",
    "mrna": "#e63946",
    "mirna": "#457b9d",
    "pseudogene": "#f77f00",
    "five_prime_utr": "#43aa8b",
    "three_prime_utr": "#577590",
}

VECTOR_CLASS_DISPLAY = {
    "lnc_rna": "lncRNA",
    "mrna": "mRNA",
    "mirna": "miRNA",
    "pseudogene": "Pseudogene",
    "five_prime_utr": "5' UTR",
    "three_prime_utr": "3' UTR",
}

VECTOR_DATASET_COLORS = [
    "#ef476f", "#118ab2", "#06d6a0", "#ffd166", "#073b4c",
    "#b5179e", "#4895ef", "#ffb703", "#219ebc",
]

CLASS_LEGEND_ORDER = [
    "lnc_rna", "mrna", "mirna", "pseudogene", "five_prime_utr", "three_prime_utr",
]
