from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"

# Ensure scripts are in path for legacy imports
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Script Definitions
SCAN_SCRIPT = SCRIPTS_DIR / "nt_sequence_search.py"
ANALYSIS_SCRIPT = SCRIPTS_DIR / "nt_sequence_G4_TD_analysis.py"
ANNOTATION_SCRIPT = SCRIPTS_DIR / "nt_sequence_annotation.py"
VISUALIZER_SCRIPT = SCRIPTS_DIR / "chromosome_visualizer.py"
PUBLICATION_VIZ_SCRIPT = SCRIPTS_DIR / "publication_visualizer.py"
GENE_SCRIPT = SCRIPTS_DIR / "exctract_genes_nt_sequence_annotated.py"
TRIPLEX_SCRIPT = SCRIPTS_DIR / "triplex_search.py"
SPLIT_RNA_SCRIPT = SCRIPTS_DIR / "rna_sequence_windows_split.py"
SCORE_RNA_SCRIPT = SCRIPTS_DIR / "rna_sequence_windows_scoring.py"
ANNOTATE_TRIPLEX_SCRIPT = SCRIPTS_DIR / "triplex_annotation.py"
TRIPLEX_GENE_SCRIPT = SCRIPTS_DIR / "exctract_genes_triplex_annotated.py"
PEPTIDE_SCRIPT = SCRIPTS_DIR / "peptide_coding_search.py"
PEPTIDE_ANNOTATION_SCRIPT = SCRIPTS_DIR / "peptide_annotation.py"

