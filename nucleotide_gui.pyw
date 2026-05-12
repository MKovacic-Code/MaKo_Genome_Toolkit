#!/usr/bin/env pythonw
"""Simple Tk GUI wrapper for nt_sequence_search.py."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List
import csv
import configparser
from tkinter import (
    BOTH,
    BOTTOM,
    END,
    LEFT,
    RIGHT,
    TOP,
    BooleanVar,
    Button,
    Canvas,
    Checkbutton,
    Entry,
    Frame,
    Label,
    LabelFrame,
    Listbox,
    OptionMenu,
    Radiobutton,
    Scrollbar,
    StringVar,
    Tk,
    Toplevel,
    X,
    Y,
    filedialog,
    messagebox,
    scrolledtext,
    colorchooser,
)
from tkinter import simpledialog
from tkinter import ttk

ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Tool-tab analysis imports (guarded so GUI still loads if a script is missing)
try:
    from nt_sequence_G4_TD_analysis import (
        g4hunter_score as _g4hunter_score,
        g4boost_score as _g4boost_score,
        base_composition as _base_composition,
        tm_wallace as _tm_wallace,
        tm_nearest_neighbor as _tm_nearest_neighbor,
        calc_codon_efficiency as _calc_codon_efficiency,
        i_motif_score as _i_motif_score,
        r_loop_score as _r_loop_score,
        hairpin_score as _hairpin_score,
        cpg_island_score as _cpg_island_score,
        CODON_WEIGHTS as _CODON_WEIGHTS,
        reverse_complement as _reverse_complement,
    )
    from peptide_coding_search import translate_sequence as _translate_sequence, CODON_TABLE as _CODON_TABLE
    _TOOLS_AVAILABLE = True
except Exception:
    _TOOLS_AVAILABLE = False

DEFAULT_DATA_DIR_NAME = "data_human_homo_sapiens"
PROFILE_STORE_PATH = ROOT_DIR / "chemistry_profiles.json"
BUILTIN_CHEMISTRY_PROFILES = {
    "relaxed": {
        "description": "Loose GC content and repeat limits.",
        "base_content": ["G:20:80", "C:20:80"],
        "max_repeat": ["G:6", "C:6", "A:8", "T:8"],
    },
    "stringent": {
        "description": "High GC enrichment and tight repeats for classic G4 scans.",
        "base_content": ["G:40:70", "C:30:70"],
        "max_repeat": ["G:4", "C:4"],
        "motifs": ["GGGN{1,7}GGG:1"],
    },
}

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

try:
    from chromosome_vector_visualizer import (
        CLASS_COLOR_MAP as VECTOR_CLASS_COLORS,
        CLASS_DISPLAY as VECTOR_CLASS_DISPLAY,
        DATASET_COLORS as VECTOR_DATASET_COLORS,
    )
except Exception:
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
        "#ef476f",
        "#118ab2",
        "#06d6a0",
        "#ffd166",
        "#073b4c",
        "#b5179e",
        "#4895ef",
        "#ffb703",
        "#219ebc",
    ]

CLASS_LEGEND_ORDER = [
    "lnc_rna",
    "mrna",
    "mirna",
    "pseudogene",
    "five_prime_utr",
    "three_prime_utr",
]

TAB_ACCENTS = {
    "inputs": "#0f766e",
    "scanner": "#1d4ed8",
    "peptide": "#b45309",
    "triplex": "#7c3aed",
    "viz": "#f97316",
    "tools": "#059669",
}


class ToolTip:
    def __init__(self, widget, text: str, wraplength: int = 260) -> None:
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self.tipwindow: Toplevel | None = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, _event=None) -> None:
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        Label(
            tw,
            text=self.text,
            justify=LEFT,
            relief="solid",
            borderwidth=1,
            background="#ffffe0",
            wraplength=self.wraplength,
        ).pack(ipadx=6, ipady=4)

    def hide_tip(self, _event=None) -> None:
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


class ScannerGUI:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("MaKo Genome Toolkit")
        self.root.geometry("1700x1024")

        self.genome_choices = self._discover_genome_dirs()
        self._genome_slug_to_path = {choice["slug"]: choice["path"] for choice in self.genome_choices}
        self._genome_slug_to_label = {choice["slug"]: choice["label"] for choice in self.genome_choices}
        self._genome_label_to_slug = {choice["label"]: choice["slug"] for choice in self.genome_choices}
        default_genome = (
            DEFAULT_DATA_DIR_NAME
            if DEFAULT_DATA_DIR_NAME in self._genome_slug_to_path
            else (self.genome_choices[0]["slug"] if self.genome_choices else "")
        )
        self.genome_display_var = StringVar(
            value=self._genome_slug_to_label.get(default_genome, "No genome data folder detected")
        )
        self.selected_genome_var = StringVar(value=default_genome)
        self._current_genome_slug = default_genome
        self._suspend_genome_trace = False
        self.selected_genome_var.trace_add("write", self._handle_genome_selection)
        self.genome_combo = None

        self.fasta_var = StringVar()
        self.window_var = StringVar(value="50")
        self.step_var = StringVar(value="5")
        self.output_name_var = StringVar()
        self.workers_var = StringVar(value="0")
        self.forward_strand_var = BooleanVar(value=True)
        self.reverse_strand_var = BooleanVar(value=False)
        self.combined_strand_var = BooleanVar(value=False)
        self.combined_forward_var = StringVar(value=self.window_var.get())
        self.combined_reverse_var = StringVar(value=self.window_var.get())
        self.combined_overlap_var = StringVar(value="0")
        self.dna_conc_var = StringVar(value="0.001")  # 1 mM
        self.salt_conc_var = StringVar(value="0.15")  # 150 mM
        self.temperature_var = StringVar(value="25")  # Celsius
        self.require_palindrome_var = BooleanVar(value=False)
        self.palindrome_min_len_var = StringVar(value="8")
        self.top_oligos_var = StringVar(value="1")
        self.structure_only_var = BooleanVar(value=False)
        self.analysis_summary_path: str | None = None
        self.min_g4hunter_filter_var = StringVar()
        self.min_g4boost_filter_var = StringVar()
        self.min_tm_wallace_filter_var = StringVar()
        self.min_tm_nearest_filter_var = StringVar()
        self.min_i_motif_filter_var = StringVar()
        self.min_r_loop_filter_var = StringVar()
        self.min_hairpin_filter_var = StringVar()
        self.min_cpg_filter_var = StringVar()
        self.gene_list_var = StringVar()
        self.motif_widget = None
        self.exclude_motif_widget = None
        self.base_widget = None
        self.repeat_widget = None
        self.combined_entry_widgets: list[Entry] = []
        self.sequence_filter_widget = None
        self.region_filter_widget = None
        self.forward_motif_widget = None
        self.reverse_motif_widget = None

        self.process_hits_var = StringVar()
        self.processed_hits_var = StringVar()
        self.gbff_var = StringVar()
        self.gff3_var = StringVar()
        self.gff_var = StringVar()
        self.gtf_var = StringVar()
        self.gpff_var = StringVar()
        self.min_oligo_var = StringVar(value="15")
        self.max_oligo_var = StringVar(value="60")
        self.g4_window_var = StringVar(value="25")
        self.viz_hits_var = StringVar()
        self.viz_default_filename = "genome_hits.svg"
        self.viz_output_var = StringVar()
        self._changing_viz_path = False
        self.viz_output_var.trace_add("write", lambda *_: self._on_viz_var_change())
        self.viz_format_var = StringVar(value="auto")
        self.viz_scope_var = StringVar(value="regions")
        self.viz_label_mode_var = StringVar(value="name")
        self.process_hits_var.trace_add("write", lambda *_: self._sync_process_to_viz())
        self.processed_hits_var.trace_add("write", lambda *_: self._sync_process_to_viz())
        self.combined_strand_var.trace_add("write", lambda *_: self._toggle_combined_fields())
        self.window_var.trace_add("write", lambda *_: self._sync_combined_defaults())

        self.triplex_rna_var = StringVar()
        self.triplex_rna_windows_var = StringVar()
        self.triplex_dna_var = StringVar()
        self.triplex_min_len_var = StringVar(value="15")
        self.triplex_max_len_var = StringVar(value="40")
        self.triplex_min_score_var = StringVar(value="20")
        self.triplex_max_mismatches_var = StringVar(value="3")
        self.triplex_output_var = StringVar()
        self.triplex_workers_var = StringVar(value="0")
        self.triplex_forward_var = BooleanVar(value=True)
        self.triplex_reverse_var = BooleanVar(value=True)
        self.triplex_rna_text_widget = None
        self.triplex_input_mode_var = StringVar(value="windows")
        self.triplex_windows_controls = None
        self.triplex_rna_controls = None
        self.split_window_var = StringVar(value="60")
        self.split_step_var = StringVar(value="20")
        self.split_min_length_var = StringVar(value="20")
        self.split_include_partial_var = BooleanVar(value=True)
        self.split_output_var = StringVar()
        self.score_min_score_var = StringVar(value="20")
        self.score_tsv_var = StringVar()
        self.score_fasta_var = StringVar()
        self.score_min_pyrimidine_var = StringVar(value="0.5")
        self.score_min_spacing_var = StringVar(value="0")
        self.triplex_annotation_var = StringVar()
        self.analyzed_hits_var = StringVar()
        self.annotation_input_var = StringVar()
        self.triplex_gene_prefix_var = StringVar(value="triplex_gene_list")
        self.triplex_gene_list_var = StringVar()
        self.non_overlapping_var = BooleanVar(value=False)
        self.peptide_fasta_var = StringVar()
        self.peptide_window_var = StringVar(value="15")
        self.peptide_step_var = StringVar(value="5")
        self.peptide_comb_fwd_var = StringVar(value="0")
        self.peptide_comb_rev_var = StringVar(value="0")
        self.peptide_comb_overlap_var = StringVar(value="0")
        self.peptide_max_mismatches_var = StringVar(value="0")
        self.peptide_workers_var = StringVar(value="0")
        self.peptide_hits_var = StringVar()
        self.peptide_annotation_var = StringVar()
        self.peptide_motif_widget = None
        self.peptide_exclude_widget = None
        self.peptide_frame_vars = {
            "+0": BooleanVar(value=True),
            "+1": BooleanVar(value=True),
            "+2": BooleanVar(value=True),
            "-0": BooleanVar(value=False),
            "-1": BooleanVar(value=False),
            "-2": BooleanVar(value=False),
        }

        # Combined Search tab variables
        self.combined_pep_mismatches_var = StringVar(value="0")
        self.combined_comb_fwd_var = StringVar(value="0")
        self.combined_comb_rev_var = StringVar(value="0")
        self.combined_comb_overlap_var = StringVar(value="0")
        self.combined_nt_motif_widget = None
        self.combined_exclude_nt_widget = None
        self.combined_pep_motif_widget = None
        self.combined_exclude_pep_widget = None
        self.combined_base_widget = None
        self.combined_repeat_widget = None
        self.combined_self_comp_widget = None
        self.combined_nt_sub_window_var = StringVar(value="")
        self.combined_nt_sub_offset_var = StringVar(value="0")
        self.combined_pep_sub_window_var = StringVar(value="")
        self.combined_pep_sub_offset_var = StringVar(value="0")
        self.combined_frame_vars = {
            "+0": BooleanVar(value=True),
            "+1": BooleanVar(value=True),
            "+2": BooleanVar(value=True),
            "-0": BooleanVar(value=True),
            "-1": BooleanVar(value=True),
            "-2": BooleanVar(value=True),
        }

        self.vector_input_var = StringVar()
        self.vector_output_var = StringVar()
        self.vector_label_mode_var = StringVar(value="gene+coords")
        self.vector_density_var = BooleanVar(value=False)
        self.vector_annotate_var = BooleanVar(value=False)
        self.vector_title_var = StringVar()
        self.vector_annotation_column_var = StringVar(value="longest_palindrome_sequence")
        self.vector_seq_ids_var = StringVar()
        self.vector_region_filter_var = StringVar()
        self.vector_gene_filter_var = StringVar()
        self.vector_output_format_var = StringVar(value="png")
        self.vector_dpi_var = StringVar(value="300")
        self.vector_fig_width_var = StringVar(value="12.0")
        self.vector_fig_height_var = StringVar(value="8.0")
        self.vector_font_family_var = StringVar(value="Arial")
        self.vector_font_size_var = StringVar(value="12")
        self.vector_palette_var = StringVar(value="default")
        self.vector_scale_mode_var = StringVar(value="absolute")
        self.vector_bg_color_var = StringVar(value="#f8f8f8")
        self.vector_bg_alpha_var = StringVar(value="1.0")
        self.vector_legend_pos_var = StringVar(value="right")
        self.vector_source_flags = {
            "nt_sequence_hits": BooleanVar(value=False),
            "nt_sequence_hits_analyzed": BooleanVar(value=False),
            "nt_sequence_hits_annotated": BooleanVar(value=True),
            "triplex_hits": BooleanVar(value=False),
            "triplex_hits_annotated": BooleanVar(value=False),
            "peptide_hits": BooleanVar(value=False),
            "peptide_hits_annotated": BooleanVar(value=False),
        }
        self.vector_dataset_options: Dict[str, Dict[str, StringVar]] = {
            key: {
                "opacity": StringVar(value="0.75"),
                "style": StringVar(value="solid"),
                "color": StringVar(value=""),
            }
            for key in self.vector_source_flags
        }
        self.vector_custom_rows: list[dict[str, object]] = []
        self._file_picker_validators: list[Callable[[], None]] = []
        self._loading_settings = False
        self._saved_motif_text = ""
        self._saved_exclude_text = ""
        self._saved_base_text = ""
        self._saved_repeat_text = ""
        self._saved_forward_motif_text = ""
        self._saved_reverse_motif_text = ""
        self._saved_self_comp_text = ""  # legacy: kept for backward-compat with ini loading
        self._self_comp_rules: list[dict] = []  # new structured rules
        self._saved_profile_name = "custom"
        self.chemistry_profile_var = StringVar(value="Custom (manual)")
        self.custom_profiles: Dict[str, Dict[str, List[str]]] = {}
        self.chemistry_profiles = self._load_chemistry_profiles()
        self._persisted_string_vars: dict[str, StringVar] = {
            "output_name": self.output_name_var,
            "genome_fasta": self.fasta_var,
            "gbff_path": self.gbff_var,
            "gff3_path": self.gff3_var,
            "gff_path": self.gff_var,
            "gtf_path": self.gtf_var,
            "gpff_path": self.gpff_var,
            "triplex_windows": self.triplex_rna_windows_var,
            "triplex_rna": self.triplex_rna_var,
            "triplex_dna": self.triplex_dna_var,
            "triplex_input_mode": self.triplex_input_mode_var,
            "selected_genome": self.selected_genome_var,
            "peptide_fasta": self.peptide_fasta_var,
            "palindrome_min_len": self.palindrome_min_len_var,
            "top_oligos": self.top_oligos_var,
            "vector_annotation_column": self.vector_annotation_column_var,
            "vector_seq_ids": self.vector_seq_ids_var,
            "vector_region_filters": self.vector_region_filter_var,
            "vector_gene_filters": self.vector_gene_filter_var,
            "vector_label_mode": self.vector_label_mode_var,
        }
        self._persisted_option_vars: dict[str, BooleanVar] = {
            "require_palindrome": self.require_palindrome_var,
            "non_overlapping": self.non_overlapping_var,
            "structure_only": self.structure_only_var,
            "vector_density_view": self.vector_density_var,
            "vector_annotate_hits": self.vector_annotate_var,
        }

        refresh_cb = lambda *_: self._refresh_output_paths()
        for var in (
            self.output_name_var,
            self.triplex_gene_prefix_var,
        ):
            var.trace_add("write", refresh_cb)
        persist_cb = lambda *_: self._persist_settings()
        for var in list(self._persisted_string_vars.values()) + list(
            self._persisted_option_vars.values()
        ):
            var.trace_add("write", persist_cb)
        self.triplex_input_mode_var.trace_add("write", lambda *_: self._update_triplex_input_mode())

        # Tools tab variables
        self.tool_g4_input_var = StringVar()
        self.tool_g4_window_var = StringVar(value="25")
        self.tool_codon_input_var = StringVar()
        self.tool_translate_input_var = StringVar()
        self.tool_thermo_input_var = StringVar()
        self.tool_thermo_dna_conc_var = StringVar(value="0.001")
        self.tool_thermo_salt_conc_var = StringVar(value="0.15")
        self.tool_mutator_input_var = StringVar()
        self.tool_mutator_target_var = StringVar(value="1.5")
        self.tool_mutator_max_mut_var = StringVar(value="5")

        self.status_var = StringVar(value="Idle")
        self.status_label = None
        self.status_progress = None
        self._status_reset_job: str | None = None
        self._status_colors = {
            "info": "#1e272e",
            "busy": "#0056a3",
            "success": "#2e7d32",
            "error": "#c0392b",
        }

        self._load_persisted_settings()
        self._load_defaults_async()
        self._build_layout()
        self._refresh_output_paths()

    def _build_layout(self) -> None:
        outer = Frame(self.root)
        outer.pack(fill=BOTH, expand=True)

        log_frame = LabelFrame(outer, text="Run Log")
        log_frame.pack(side=BOTTOM, fill=X, padx=10, pady=(0, 5))
        self.log_box = scrolledtext.ScrolledText(log_frame, wrap="word", height=8)
        self.log_box.pack(fill=BOTH, expand=True)
        self.log_box.configure(state="disabled")
        status_row = Frame(log_frame)
        status_row.pack(fill=X, padx=4, pady=(4, 0))
        self.status_label = Label(status_row, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side=LEFT, fill=BOTH, expand=True)
        self.status_progress = ttk.Progressbar(status_row, mode="indeterminate", length=160)
        self.status_progress.pack(side=RIGHT, padx=(8, 0))
        self.status_progress.stop()

        canvas = Canvas(outer, highlightthickness=0)
        scrollbar = Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")

        content = Frame(canvas)
        canvas.create_window((0, 0), window=content, anchor="nw")

        top_panel = Frame(content)
        top_panel.pack(fill=BOTH, expand=False, padx=10, pady=(10, 0))
        
        left_top = Frame(top_panel)
        left_top.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
        right_top = Frame(top_panel)
        right_top.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5, 0))

        self._build_genome_selector(left_top)
        self._build_output_panel(right_top)

        def _configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        content.bind("<Configure>", _configure_scroll)
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))

        style = ttk.Style()
        style.configure(
            "GenomeNotebook.TNotebook",
            borderwidth=0,
        )
        style.configure(
            "GenomeNotebook.TNotebook.Tab",
            padding=(18, 10),
            font=("Segoe UI", 11, "bold"),
        )
        notebook = ttk.Notebook(content, style="GenomeNotebook.TNotebook")
        notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
        def _on_tab_change(event=None):
            self._run_file_picker_validators()
            current_tab_name = notebook.select()
            for tab_name in notebook.tabs():
                tab_widget = notebook.nametowidget(tab_name)
                if tab_widget.winfo_children():
                    wrapper = tab_widget.winfo_children()[0]
                    if tab_name == current_tab_name:
                        if hasattr(wrapper, "_pack_info"):
                            wrapper.pack(**wrapper._pack_info)
                    else:
                        if not hasattr(wrapper, "_pack_info"):
                            info = wrapper.pack_info()
                            if info:
                                wrapper._pack_info = {k: v for k, v in info.items() if k not in ['in']}
                        wrapper.pack_forget()

        notebook.bind("<<NotebookTabChanged>>", _on_tab_change)

        input_tab = Frame(notebook)
        notebook.add(input_tab, text="Inputs (Outputs)")
        self._build_input_tab(input_tab, "Inputs (Outputs)", TAB_ACCENTS["inputs"])

        scanner_tab = Frame(notebook)
        notebook.add(scanner_tab, text="Nucleotide Sequence Scanner")
        self._build_scanner_tab(
            scanner_tab, "Nucleotide Sequence Scanner", TAB_ACCENTS["scanner"]
        )

        analysis_tab = Frame(notebook)
        notebook.add(analysis_tab, text="Nucleotide Hits Analyses")
        self._build_analysis_tab(
            analysis_tab, "Nucleotide Hits Analyses", TAB_ACCENTS["scanner"]
        )

        peptide_tab = Frame(notebook)
        notebook.add(peptide_tab, text="Peptide Coding Search")
        self._build_peptide_tab(peptide_tab, "Peptide Coding Search", TAB_ACCENTS["peptide"])

        combined_tab = Frame(notebook)
        notebook.add(combined_tab, text="Combined Search")
        self._build_combined_tab(combined_tab, "Combined Search", TAB_ACCENTS.get("combined", "#FF9800"))

        triplex_tab = Frame(notebook)
        notebook.add(triplex_tab, text="RNA/DNA Triplex Finder")
        self._build_triplex_tab(
            triplex_tab, "RNA/DNA Triplex Finder", TAB_ACCENTS["triplex"]
        )

        viz_tab = Frame(notebook)
        notebook.add(viz_tab, text="Visualization")
        self._build_visualization_tab(
            viz_tab, "Visualization", TAB_ACCENTS["viz"]
        )

        tools_tab = Frame(notebook)
        notebook.add(tools_tab, text="Tools")
        self._build_tools_tab(tools_tab, "Sequence Tools", TAB_ACCENTS["tools"])

        self._setup_sync_bindings()
        _on_tab_change()

        # Run Log is now pinned to the bottom of the window (created above)

    def _setup_sync_bindings(self):
        def _sync(source, target):
            if not source or not target:
                return
            src_text = source.get("1.0", "end-1c")
            if target.get("1.0", "end-1c") != src_text:
                target.delete("1.0", "end")
                target.insert("1.0", src_text)

        pairs = [
            (self.motif_widget, self.combined_nt_motif_widget),
            (self.exclude_motif_widget, self.combined_exclude_nt_widget),
            (self.base_widget, self.combined_base_widget),
            (self.repeat_widget, self.combined_repeat_widget),
            (self.forward_motif_widget, self.combined_forward_motif_widget),
            (self.reverse_motif_widget, self.combined_reverse_motif_widget),
            (self.peptide_motif_widget, self.combined_pep_motif_widget),
            (self.peptide_exclude_widget, self.combined_exclude_pep_widget),
            (self.amino_content_widget, self.combined_amino_content_widget),
            (self.peptide_repeat_widget, self.combined_peptide_repeat_widget),
        ]

        for w1, w2 in pairs:
            if w1 and w2:
                w1.bind("<KeyRelease>", lambda e, s=w1, t=w2: _sync(s, t))
                w2.bind("<KeyRelease>", lambda e, s=w2, t=w1: _sync(s, t))

    def _add_tab_banner(self, parent: Frame, title: str, color: str) -> None:
        spacer = Frame(parent, height=6, bg=color)
        spacer.pack(fill="x", padx=10, pady=(4, 6))

    def _build_genome_selector(self, parent: Frame) -> None:
        frame = LabelFrame(parent, text="Genome Dataset")
        frame.pack(fill=BOTH, expand=True, pady=(0, 0))
        if not self.genome_choices:
            Label(
                frame,
                text="No genome data folders were found (expected directories named 'data_*').",
                justify=LEFT,
                wraplength=420,
            ).pack(fill="x", padx=8, pady=6)
            return
        hint_row = Frame(frame)
        hint_row.pack(fill="x", padx=8, pady=(4, 6))
        Label(
            hint_row,
            text="Select which genome reference folder to use for defaults and auto-filled annotation inputs.",
            justify=LEFT,
            anchor="w",
        ).pack(anchor="w")
        values = [choice["label"] for choice in self.genome_choices]
        if not self.selected_genome_var.get() and self.genome_choices:
            self.selected_genome_var.set(self.genome_choices[0]["slug"])
        if not self.genome_display_var.get().strip():
            self.genome_display_var.set(self._genome_slug_to_label.get(self.selected_genome_var.get(), values[0]))
        row = Frame(frame)
        row.pack(fill="x", padx=8, pady=(0, 6))
        Label(row, text="Genome folder:", width=18, anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 8))
        combo = ttk.Combobox(
            row,
            state="readonly",
            values=values,
            textvariable=self.genome_display_var,
            width=40,
        )
        combo.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        row.columnconfigure(1, weight=1)
        combo.bind("<<ComboboxSelected>>", self._on_genome_combo_selected)
        self.genome_combo = combo
        Button(row, text="Open folder", command=self._open_selected_genome_dir).grid(row=0, column=2, sticky="e")

    def _open_selected_genome_dir(self) -> None:
        directory = self._get_genome_data_dir()
        if directory == ROOT_DIR:
            messagebox.showinfo("Genome folder", "No genome data directory is currently selected.")
            return
        self._open_path_default(str(directory))

    def _on_genome_combo_selected(self, _event=None) -> None:
        label = self.genome_display_var.get().strip()
        slug = self._genome_label_to_slug.get(label)
        if slug:
            self.selected_genome_var.set(slug)

    def _handle_genome_selection(self, *_args) -> None:
        if self._suspend_genome_trace:
            return
        slug = self.selected_genome_var.get().strip()
        label = self._genome_slug_to_label.get(slug, "No genome data folder detected")
        if self.genome_display_var.get() != label:
            self.genome_display_var.set(label)
        if slug == self._current_genome_slug:
            return
        previous_slug = self._current_genome_slug
        self._current_genome_slug = slug
        self._update_genome_defaults(prev_slug=previous_slug)

    def _discover_genome_dirs(self) -> list[dict[str, object]]:
        choices: list[dict[str, object]] = []
        for path in sorted(ROOT_DIR.glob(f"{DATA_DIR_PREFIX}*")):
            if not path.is_dir():
                continue
            slug = path.name
            choices.append({"slug": slug, "label": self._format_genome_label(slug), "path": path})
        fallback = ROOT_DIR / DEFAULT_DATA_DIR_NAME
        if not choices and fallback.is_dir():
            slug = fallback.name
            choices.append({"slug": slug, "label": self._format_genome_label(slug), "path": fallback})
        return choices

    @staticmethod
    def _format_genome_label(slug: str) -> str:
        trimmed = slug[len(DATA_DIR_PREFIX) :] if slug.startswith(DATA_DIR_PREFIX) else slug
        parts = [part.capitalize() for part in trimmed.split("_") if part]
        if len(parts) >= 2:
            return f"{parts[0]} - {' '.join(parts[1:])}"
        return " ".join(parts) if parts else slug

    def _get_genome_data_dir(self) -> Path:
        slug = self.selected_genome_var.get().strip()
        candidate = self._genome_slug_to_path.get(slug)
        if candidate and candidate.is_dir():
            try:
                return candidate.relative_to(ROOT_DIR)
            except ValueError:
                return candidate
        fallback = self._genome_slug_to_path.get(DEFAULT_DATA_DIR_NAME)
        if fallback and fallback.is_dir():
            try:
                return fallback.relative_to(ROOT_DIR)
            except ValueError:
                return fallback
        if self.genome_choices:
            first = self.genome_choices[0]["path"]
            if isinstance(first, Path) and first.is_dir():
                try:
                    return first.relative_to(ROOT_DIR)
                except ValueError:
                    return first
        legacy = ROOT_DIR / DEFAULT_DATA_DIR_NAME
        if legacy.is_dir():
            try:
                return legacy.relative_to(ROOT_DIR)
            except ValueError:
                return legacy
        return Path(".")

    def _update_genome_defaults(self, prev_slug: str | None = None, force: bool = False) -> None:
        dependent_vars = {
            "fasta_var",
            "triplex_dna_var",
            "peptide_fasta_var",
            "gbff_var",
            "gff3_var",
            "gff_var",
            "gtf_var",
            "gpff_var",
        }
        defaults = self._discover_default_paths()
        prev_path = self._genome_slug_to_path.get(prev_slug) if prev_slug else None
        for attr in dependent_vars:
            candidate = defaults.get(attr)
            if not candidate:
                continue
            var = getattr(self, attr, None)
            if not isinstance(var, StringVar):
                continue
            current = var.get().strip()
            replace = force or not current or not Path(current).is_file()
            if not replace and prev_path and current:
                try:
                    current_path = Path(current).resolve()
                    if current_path.is_relative_to(prev_path.resolve()):
                        replace = True
                except Exception:
                    if str(prev_path) and str(prev_path) in current:
                        replace = True
            if replace:
                var.set(candidate)

    def _build_output_panel(self, parent: Frame) -> None:
        panel = LabelFrame(parent, text="Output Directory")
        panel.pack(fill=BOTH, expand=True, pady=(0, 0))
        Label(
            panel,
            text="All generated files are written to output_<name>. Set your preferred suffix below.",
            wraplength=220,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        self._add_labeled_entry(panel, "Folder suffix:", self.output_name_var, 18)
        Button(
            panel,
            text="Open output folder",
            command=self._open_output_dir,
        ).pack(fill=BOTH, pady=(6, 0))

    def _build_input_tab(self, parent: Frame, heading: str, accent_color: str) -> None:
        wrapper = Frame(parent)
        wrapper.pack(fill="x", expand=False, padx=10, pady=10)

        self._add_tab_banner(wrapper, heading, accent_color)

        paned = ttk.Panedwindow(wrapper, orient="horizontal")
        paned.pack(fill=BOTH, expand=True)
        left_col = Frame(paned)
        right_col = Frame(paned)
        paned.add(left_col, weight=2)
        paned.add(right_col, weight=3)

        genome_section = LabelFrame(left_col, text="Genome Inputs")
        genome_section.pack(fill=BOTH, expand=True, pady=(0, 10))
        genome_hint = "Provide genome FASTA plus annotation files (GTF/GFF/GBFF/GPFF). Defaults follow the selected genome folder."
        Label(
            genome_section,
            text=genome_hint,
            wraplength=260,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        self._add_file_picker(
            genome_section,
            "Genome FASTA:",
            self.fasta_var,
            [("FASTA", "*.fa *.fna *.fasta"), ("All", "*.*")],
            dialog_title="Select genome FASTA",
            base_dir="genome",
        )
        self._add_file_picker(
            genome_section,
            "Triplex DNA FASTA:",
            self.triplex_dna_var,
            [("FASTA", "*.fa *.fna *.fasta"), ("All", "*.*")],
            dialog_title="Select DNA FASTA for triplex search",
        )

        annotation_inputs = LabelFrame(genome_section, text="Advanced: Annotation Overrides (Auto-detected)")
        annotation_inputs.pack(fill=BOTH, pady=(6, 0))
        self._add_file_picker(
            annotation_inputs,
            "GTF file:",
            self.gtf_var,
            [("GTF", "*.gtf"), ("All", "*.*")],
            dialog_title="Select GTF file",
        )
        self._add_file_picker(
            annotation_inputs,
            "GFF3 file:",
            self.gff3_var,
            [("GFF3", "*.gff3"), ("All", "*.*")],
            dialog_title="Select GFF3 file",
        )
        self._add_file_picker(
            annotation_inputs,
            "GFF file:",
            self.gff_var,
            [("GFF", "*.gff"), ("All", "*.*")],
            dialog_title="Select GFF file",
        )
        self._add_file_picker(
            annotation_inputs,
            "GBFF file:",
            self.gbff_var,
            [("GBFF", "*.gbff"), ("All", "*.*")],
            dialog_title="Select GenBank flat file",
        )
        self._add_file_picker(
            annotation_inputs,
            "GPFF file:",
            self.gpff_var,
            [("GPFF", "*.gpff"), ("All", "*.*")],
            dialog_title="Select GenBank protein flat file",
        )

        output_dir_section = LabelFrame(right_col, text="Output Directory")
        output_dir_section.pack(fill=BOTH, pady=(0, 10))
        Label(
            output_dir_section,
            text="All generated files are written to output_<name>. Set your preferred suffix below.",
            wraplength=260,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        self._add_labeled_entry(output_dir_section, "Output folder name:", self.output_name_var, 18)
        Button(
            output_dir_section,
            text="Open output folder",
            command=self._open_output_dir,
        ).pack(fill=BOTH, pady=(6, 0))

        outputs_section = LabelFrame(right_col, text="Workflow Output Files")
        outputs_section.pack(fill="x", expand=False, pady=(0, 10))
        Label(
            outputs_section,
            text="Review or override the auto-filled output paths for each workflow stage.",
            wraplength=240,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        self._add_file_picker(
            outputs_section,
            "nt_sequence_hits.tsv:",
            self.process_hits_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.process_hits_var.get(), "NT sequence hits"),
            base_dir="output",
            require_output_dir=True,
        )
        self._add_file_picker(
            outputs_section,
            "nt_sequence_hits_analyzed.tsv:",
            self.analyzed_hits_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.analyzed_hits_var.get(), "Analyzed hits"),
            base_dir="output",
            require_output_dir=True,
        )
        self._add_file_picker(
            outputs_section,
            "nt_sequence_hits_annotated.tsv:",
            self.processed_hits_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.processed_hits_var.get(), "Annotated hits"),
            base_dir="output",
            require_output_dir=True,
        )
        self._add_file_picker(
            outputs_section,
            "gene_list.tsv:",
            self.gene_list_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.gene_list_var.get(), "Gene list"),
            base_dir="output",
            require_output_dir=True,
        )
        self._add_file_picker(
            outputs_section,
            "triplex_hits.tsv:",
            self.triplex_output_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.triplex_output_var.get(), "Triplex hits"),
            base_dir="output",
            require_output_dir=True,
        )
        self._add_file_picker(
            outputs_section,
            "triplex_hits_annotated.tsv:",
            self.triplex_annotation_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.triplex_annotation_var.get(), "Annotated triplex"),
            base_dir="output",
            require_output_dir=True,
        )
        annotation_row = LabelFrame(outputs_section, text="NT annotation source")
        annotation_row.pack(fill=BOTH, pady=(6, 4))
        Entry(annotation_row, textvariable=self.annotation_input_var).pack(side=LEFT, fill=BOTH, expand=True, padx=(4, 4))
        Button(
            annotation_row,
            text="Use raw",
            command=lambda: self.annotation_input_var.set(self.process_hits_var.get()),
        ).pack(side=LEFT, padx=2)
        Button(
            annotation_row,
            text="Use analyzed",
            command=lambda: self.annotation_input_var.set(self.analyzed_hits_var.get()),
        ).pack(side=LEFT, padx=2)
        Button(
            annotation_row,
            text="Open",
            command=lambda: self._open_table_window(self.annotation_input_var.get(), "Annotation input"),
        ).pack(side=LEFT, padx=2)
        self._add_file_picker(
            outputs_section,
            "triplex_gene_list.tsv:",
            self.triplex_gene_list_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.triplex_gene_list_var.get(), "Triplex gene list"),
        )
        self._add_file_picker(
            outputs_section,
            "peptide_coding_hits.tsv:",
            self.peptide_hits_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.peptide_hits_var.get(), "Peptide hits"),
        )
        self._add_file_picker(
            outputs_section,
            "peptide_coding_hits_annotated.tsv:",
            self.peptide_annotation_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(
                self.peptide_annotation_var.get(), "Annotated peptide hits"
            ),
        )
        self._add_file_picker(
            outputs_section,
            "Visualization source (TSV/CSV):",
            self.viz_hits_var,
            [("TSV/CSV", "*.tsv *.csv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.viz_hits_var.get(), "Viz source"),
        )
        self._add_file_picker(
            outputs_section,
            "Visualization output (SVG):",
            self.viz_output_var,
            save_dialog=True,
        )
    def _build_scanner_tab(self, parent: Frame, heading: str, accent_color: str) -> None:
        wrapper = Frame(parent)
        wrapper.pack(fill=BOTH, expand=True, padx=6, pady=6)
        self._add_tab_banner(wrapper, heading, accent_color)

        paned = ttk.Panedwindow(wrapper, orient="horizontal")
        paned.pack(fill=BOTH, expand=False)
        left_col = Frame(paned)
        right_col = Frame(paned)
        paned.add(left_col, weight=1)
        paned.add(right_col, weight=1)

        # Scan column
        scan_section = LabelFrame(left_col, text="Nucleotide Scanning Parameters")
        scan_section.pack(fill=BOTH, pady=(0, 4))
        Label(
            scan_section,
            text=(
                "Set the window, strands, and step size for the genome scanner. "
                "Use advanced filters to define motif constraints via IUPAC codes."
            ),
            wraplength=240,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        self._add_labeled_entry(scan_section, "Window length:", self.window_var, 12)
        self._add_labeled_entry(scan_section, "Workers (0 = all CPUs):", self.workers_var, 15)

        strand_label = Label(scan_section, text="Strands / windows to scan:")
        strand_label.pack(anchor="w", pady=(6, 0))
        strand_row = Frame(scan_section)
        strand_row.pack(anchor="w", pady=2)
        Checkbutton(strand_row, text="Forward (+)", variable=self.forward_strand_var).pack(side=LEFT, padx=(0, 6))
        Checkbutton(strand_row, text="Reverse (-)", variable=self.reverse_strand_var).pack(side=LEFT, padx=(0, 6))
        Checkbutton(strand_row, text="Combined", variable=self.combined_strand_var).pack(side=LEFT)

        step_row = Frame(scan_section)
        step_row.pack(fill=BOTH, pady=(6, 2))
        Label(step_row, text="Step size (applies to all strands):", width=28, anchor="w").pack(side=LEFT)
        Entry(step_row, textvariable=self.step_var, width=10).pack(side=LEFT, fill=BOTH, expand=True)

        combined_box = LabelFrame(scan_section, text="Combined window parameters")
        combined_box.pack(fill=BOTH, pady=(6, 4))
        self.combined_entry_widgets = [
            self._add_labeled_entry(combined_box, "Forward length:", self.combined_forward_var, 10),
            self._add_labeled_entry(combined_box, "Reverse length:", self.combined_reverse_var, 10),
            self._add_labeled_entry(combined_box, "Overlap:", self.combined_overlap_var, 10),
        ]

        # Reactive Oligonucleotide View
        demo_frame = LabelFrame(right_col, text="Reactive Oligonucleotide Sequence Scheme")
        demo_frame.pack(fill="x", pady=(0, 4))
        Label(
            demo_frame,
            text="Representative motif sequence based on your first Motif IUPAC rule:",
            justify=LEFT,
        ).pack(anchor="w", pady=(1, 2))
        self.oligonucleotide_demo_var = StringVar(value="[No Motif Entered]")
        Label(
            demo_frame,
            textvariable=self.oligonucleotide_demo_var,
            font=("Consolas", 11, "bold"),
            fg="#0f766e",
            justify=LEFT,
            wraplength=400,
        ).pack(anchor="w", pady=(0, 2))

        container = LabelFrame(right_col, text="Advanced Filters & Constraints")
        container.pack(fill="x", expand=False, pady=(0, 4))
        
        motif_help = (
            "Motifs (PATTERN:COUNT per line). PATTERN may use IUPAC nucleotide codes "
            "such as R=[AG], Y=[CT], N=[ACGT], or explicit repeats like N{1,7}. "
            "Example entry: GGGN{1,7}GGG:1 meaning at least one occurrence.\n\n"
            "IUPAC Codes:\n"
            "G = Guanine\n"
            "A = Adenine\n"
            "T = Thymine\n"
            "C = Cytosine\n"
            "R = G or A (puRine)\n"
            "Y = T or C (pYrimidine)\n"
            "M = A or C (aMino)\n"
            "K = G or T (Keto)\n"
            "S = G or C (Strong interaction - 3 H bonds)\n"
            "W = A or T (Weak interaction - 2 H bonds)\n"
            "H = A or C or T (not-G)\n"
            "B = G or T or C (not-A)\n"
            "V = G or C or A (not-T)\n"
            "D = G or A or T (not-C)\n"
            "N = G or A or T or C (aNy)"
        )
        self.motif_widget = self._add_multiline_section(
            container,
            "Motifs (PATTERN:COUNT per line, IUPAC allowed):",
            help_text=motif_help,
        )
        if self._saved_motif_text:
            self.motif_widget.insert("1.0", self._saved_motif_text)
            
        self.motif_widget.bind("<KeyRelease>", self._update_reactive_oligo)
            
        exclude_help = (
            "Enter one PATTERN per line to disqualify windows containing that motif. "
            "Use the same IUPAC codes as above, but exclude motifs may not use repeats/quantifiers."
        )
        self.exclude_motif_widget = self._add_multiline_section(
            container,
            "Excluded motifs (PATTERN per line, no repeats):",
            help_text=exclude_help,
        )
        if self._saved_exclude_text:
            self.exclude_motif_widget.insert("1.0", self._saved_exclude_text)
            
        self.base_widget = self._add_multiline_section(
            container, "Base content limits (BASE:MIN:MAX per line, e.g. G:40:60):"
        )
        if self._saved_base_text:
            self.base_widget.insert("1.0", self._saved_base_text)
            
        self.repeat_widget = self._add_multiline_section(
            container, "Max repeats (BASE:MAX per line, e.g. G:4):"
        )
        if self._saved_repeat_text:
            self.repeat_widget.insert("1.0", self._saved_repeat_text)
            
        self.forward_motif_widget = self._add_multiline_section(
            container, "Forward-only motifs (optional, PATTERN:COUNT per line):"
        )
        if self._saved_forward_motif_text:
            self.forward_motif_widget.insert("1.0", self._saved_forward_motif_text)
            
        self.reverse_motif_widget = self._add_multiline_section(
            container, "Reverse-only motifs (optional, PATTERN:COUNT per line):"
        )
        if self._saved_reverse_motif_text:
            self.reverse_motif_widget.insert("1.0", self._saved_reverse_motif_text)
        
        # --- Self-Complementarity Constraint Panel ---
        sc_lf = LabelFrame(container, text="Self-Complementarity Constraints")
        sc_lf.pack(fill=BOTH, pady=(3, 0))

        # Mode toggle
        sc_mode_row = Frame(sc_lf)
        sc_mode_row.pack(anchor="w", pady=(3, 2))
        self._sc_mode_var = StringVar(value="whole")
        Radiobutton(sc_mode_row, text="Whole-motif scan", variable=self._sc_mode_var,
                    value="whole", command=self._sc_toggle_mode).pack(side=LEFT)
        Radiobutton(sc_mode_row, text="Specify regions manually", variable=self._sc_mode_var,
                    value="manual", command=self._sc_toggle_mode).pack(side=LEFT, padx=(10, 0))

        # Pattern row (shared)
        sc_pat_row = Frame(sc_lf)
        sc_pat_row.pack(fill=X, pady=2)
        Label(sc_pat_row, text="Motif pattern (IUPAC):", width=22, anchor="w").pack(side=LEFT)
        self._sc_pattern_var = StringVar()
        Entry(sc_pat_row, textvariable=self._sc_pattern_var, width=22).pack(side=LEFT, fill=X, expand=True)

        # Whole-motif mode fields
        self._sc_whole_frame = Frame(sc_lf)
        self._sc_whole_frame.pack(fill=X, pady=1)
        wf = self._sc_whole_frame
        Label(wf, text="Complementary pairs min:", width=22, anchor="w").pack(side=LEFT)
        self._sc_len_min_var = StringVar(value="3")
        Entry(wf, textvariable=self._sc_len_min_var, width=5).pack(side=LEFT)
        Label(wf, text=" max:").pack(side=LEFT)
        self._sc_len_max_var = StringVar(value="")
        Entry(wf, textvariable=self._sc_len_max_var, width=5).pack(side=LEFT)
        Label(wf, text=" (matching bases)", fg="gray").pack(side=LEFT)

        # Manual region fields
        self._sc_manual_frame = Frame(sc_lf)
        # (not packed initially)
        mf = self._sc_manual_frame
        reg1_row = Frame(mf)
        reg1_row.pack(fill=X, pady=1)
        Label(reg1_row, text="Region 1 — Start:", width=18, anchor="w").pack(side=LEFT)
        self._sc_r1_start_var = StringVar(value="0")
        Entry(reg1_row, textvariable=self._sc_r1_start_var, width=5).pack(side=LEFT)
        Label(reg1_row, text=" Pairs:").pack(side=LEFT)
        self._sc_r1_len_var = StringVar(value="3")
        Entry(reg1_row, textvariable=self._sc_r1_len_var, width=5).pack(side=LEFT)
        reg2_row = Frame(mf)
        reg2_row.pack(fill=X, pady=1)
        Label(reg2_row, text="Region 2 — Start:", width=18, anchor="w").pack(side=LEFT)
        self._sc_r2_start_var = StringVar(value="-3")
        Entry(reg2_row, textvariable=self._sc_r2_start_var, width=5).pack(side=LEFT)
        Label(reg2_row, text=" Pairs:").pack(side=LEFT)
        self._sc_r2_len_var = StringVar(value="3")
        Entry(reg2_row, textvariable=self._sc_r2_len_var, width=5).pack(side=LEFT)

        # Mismatch range (shared)
        sc_mm_row = Frame(sc_lf)
        sc_mm_row.pack(fill=X, pady=2)
        Label(sc_mm_row, text="Mismatches min:", width=18, anchor="w").pack(side=LEFT)
        self._sc_mm_min_var = StringVar(value="0")
        Entry(sc_mm_row, textvariable=self._sc_mm_min_var, width=4).pack(side=LEFT)
        Label(sc_mm_row, text=" max:").pack(side=LEFT)
        self._sc_mm_max_var = StringVar(value="0")
        Entry(sc_mm_row, textvariable=self._sc_mm_max_var, width=4).pack(side=LEFT)

        # Excluded motifs (shared)
        sc_excl_row = Frame(sc_lf)
        sc_excl_row.pack(fill=X, pady=2)
        Label(sc_excl_row, text="Excluded motifs:", width=18, anchor="w").pack(side=LEFT)
        self._sc_excl_var = StringVar()
        Entry(sc_excl_row, textvariable=self._sc_excl_var).pack(side=LEFT, fill=X, expand=True)
        Label(sc_excl_row, text=" (comma-separated)", fg="gray", font=("Segoe UI", 8)).pack(side=LEFT, padx=5)

        # Add/Remove buttons
        sc_btn_row = Frame(sc_lf)
        sc_btn_row.pack(fill=X, pady=(2, 1))
        Button(sc_btn_row, text="＋ Add Rule", command=self._sc_add_rule).pack(side=LEFT, padx=(0, 4))
        Button(sc_btn_row, text="✕ Remove Selected", command=self._sc_remove_rule).pack(side=LEFT)

        # Rules listbox
        sc_list_frame = Frame(sc_lf)
        sc_list_frame.pack(fill=BOTH, expand=True, pady=(2, 4))
        sc_list_vsb = Scrollbar(sc_list_frame, orient="vertical")
        self._sc_rules_listbox = Listbox(
            sc_list_frame, height=4, yscrollcommand=sc_list_vsb.set,
            font=("Consolas", 9), activestyle="dotbox",
        )
        sc_list_vsb.configure(command=self._sc_rules_listbox.yview)
        self._sc_rules_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        sc_list_vsb.pack(side=RIGHT, fill=Y)

        # Populate from existing rules
        for rule in self._self_comp_rules:
            self._sc_rules_listbox.insert(END, self._sc_rule_label(rule))

        # Kept for legacy persistence lookup, but now managed via _self_comp_rules
        self.self_comp_widget = None

        filter_section = LabelFrame(scan_section, text="Sequence & Region Filters")
        filter_section.pack(fill=BOTH, pady=(6, 0))
        Label(
            filter_section,
            text="Limit scanning to sequence IDs (one per line) and/or genomic ranges (e.g., chr1:100000-200000).",
            wraplength=260,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        Label(filter_section, text="Sequence IDs:").pack(anchor="w")
        self.sequence_filter_widget = scrolledtext.ScrolledText(filter_section, height=3, wrap="word")
        self.sequence_filter_widget.pack(fill=BOTH, pady=(0, 4))
        Label(filter_section, text="Regions (SEQ:START-END):").pack(anchor="w")
        self.region_filter_widget = scrolledtext.ScrolledText(filter_section, height=3, wrap="word")
        self.region_filter_widget.pack(fill=BOTH, pady=(0, 4))
        Checkbutton(
            filter_section,
            text="Report only non-overlapping hits",
            variable=self.non_overlapping_var,
        ).pack(anchor="w", pady=(2, 2))

        profile_frame = Frame(scan_section)
        profile_frame.pack(fill=BOTH, pady=(6, 0))
        Label(profile_frame, text="Chemistry profile:").pack(side=LEFT, padx=(0, 4))
        self.profile_selector = ttk.Combobox(
            profile_frame,
            textvariable=self.chemistry_profile_var,
            state="readonly",
            width=22,
        )
        self.profile_selector.pack(side=LEFT, padx=(0, 4))
        ttk.Button(profile_frame, text="Apply", command=self._apply_selected_profile).pack(side=LEFT, padx=2)
        ttk.Button(profile_frame, text="Save current…", command=self._prompt_save_profile).pack(side=LEFT, padx=2)
        ttk.Button(profile_frame, text="Delete", command=self._delete_selected_profile).pack(side=LEFT, padx=2)
        self._profile_display_to_key: Dict[str, str] = {}
        self._refresh_profile_options()

        palindrome_frame = LabelFrame(scan_section, text="Palindromic Filter")
        palindrome_frame.pack(fill=BOTH, pady=(6, 0))
        Checkbutton(
            palindrome_frame,
            text="Require palindromic sequence",
            variable=self.require_palindrome_var,
        ).pack(anchor="w")
        palindrome_entry = self._add_labeled_entry(
            palindrome_frame, "Min palindrome length:", self.palindrome_min_len_var, 10
        )
        ToolTip(
            palindrome_entry,
            "Optional: windows must contain a palindrome at least this long. Leave blank to skip filtering.",
        )

        profile_btn_frame = Frame(scan_section)
        profile_btn_frame.pack(fill=X, pady=(10, 0))
        Button(profile_btn_frame, text="Load Profile", command=lambda: self.load_search_profile("nucleotide")).pack(side=LEFT, expand=True, fill=X, padx=(0, 2))
        Button(profile_btn_frame, text="Save Profile", command=lambda: self.save_search_profile("nucleotide")).pack(side=RIGHT, expand=True, fill=X, padx=(2, 0))

        self.run_button = Button(scan_section, text="Run Genome Scan", command=self.run_scan, height=2)
        self.run_button.pack(fill=BOTH, pady=(8, 0))

        self._toggle_combined_fields()

    def _build_analysis_tab(self, parent: Frame, heading: str, accent_color: str) -> None:
        wrapper = Frame(parent)
        wrapper.pack(fill="x", expand=False, padx=10, pady=10)
        self._add_tab_banner(wrapper, heading, accent_color)

        paned = ttk.Panedwindow(wrapper, orient="horizontal")
        paned.pack(fill=BOTH, expand=False)
        left_col = Frame(paned)
        right_col = Frame(paned)
        paned.add(left_col, weight=1)
        paned.add(right_col, weight=1)

        # Processing column
        process_section = LabelFrame(left_col, text="Hits Analysis and Annotation")
        process_section.pack(fill=BOTH, pady=(0, 10))
        Label(
            process_section,
            text=(
                "Configure oligo metrics and thermodynamic settings for nt_sequence_G4_TD_analysis.py "
                "(DNA concentration, salt, and reaction temperature affect Tm calculations)."
            ),
            wraplength=340,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        self.process_workers_var = StringVar(value="0")
        self._add_labeled_entry(process_section, "Workers (0 = all CPUs):", self.process_workers_var, 15)
        self._add_labeled_entry(process_section, "Min oligo length:", self.min_oligo_var, 8)
        self._add_labeled_entry(process_section, "Max oligo length:", self.max_oligo_var, 8)
        self._add_labeled_entry(process_section, "G4 window:", self.g4_window_var, 8)
        self._add_labeled_entry(process_section, "DNA conc (M):", self.dna_conc_var, 12)
        self._add_labeled_entry(process_section, "Monovalent salt (M):", self.salt_conc_var, 12)
        self._add_labeled_entry(process_section, "Reaction temp (°C):", self.temperature_var, 12)
        self._add_labeled_entry(process_section, "Top oligos per region:", self.top_oligos_var, 6)
        Checkbutton(
            process_section,
            text="Structure-only mode (skip codon/Tm calculations)",
            variable=self.structure_only_var,
        ).pack(anchor="w", pady=(2, 4))
        
        filter_section = LabelFrame(right_col, text="Score Filters")
        filter_section.pack(fill=BOTH, pady=(0, 10))
        filter_entries = [
            (self.min_g4hunter_filter_var, "Min G4Hunter:", "Keep oligos with G4Hunter >= this value."),
            (self.min_g4boost_filter_var, "Min G4Boost:", "Keep oligos with G4Boost >= this value."),
            (self.min_tm_wallace_filter_var, "Min TM (Wallace):", "Keep oligos whose Wallace TM (°C) is above this value."),
            (self.min_tm_nearest_filter_var, "Min TM (NN):", "Keep oligos whose nearest-neighbor TM (°C) is above this value."),
            (self.min_i_motif_filter_var, "Min i-motif score:", "Drop oligos with i-motif scores below this threshold."),
            (self.min_r_loop_filter_var, "Min R-loop score:", "Drop oligos with R-loop scores below this threshold."),
            (self.min_hairpin_filter_var, "Min hairpin score:", "Filter out oligos with hairpin propensity below this value."),
            (self.min_cpg_filter_var, "Min CpG score:", "Keep oligos with CpG island scores >= this value."),
        ]
        for var, label, tip in filter_entries:
            entry = self._add_labeled_entry(filter_section, label, var, 12)
            ToolTip(entry, tip + " Leave empty to disable filtering.")
        self.analysis_button = Button(
            filter_section,
            text="Run G4/Thermo Analysis",
            command=self.run_nt_analysis,
            height=2,
        )
        self.analysis_button.pack(fill=BOTH, pady=(6, 0))
        self.summary_button = Button(
            filter_section,
            text="Show Score Summary",
            command=self._show_analysis_summary,
            state="disabled",
        )
        self.summary_button.pack(fill=BOTH, pady=(4, 0))
        self.process_button = Button(
            filter_section,
            text="Run NT Annotation",
            command=self.run_nt_annotation,
            height=2,
        )
        self.process_button.pack(fill=BOTH, pady=(6, 0))

    def _build_peptide_tab(self, parent: Frame, heading: str, accent_color: str) -> None:
        wrapper = Frame(parent)
        wrapper.pack(fill="x", expand=False, padx=10, pady=10)
        self._add_tab_banner(wrapper, heading, accent_color)

        paned = ttk.Panedwindow(wrapper, orient="horizontal")
        paned.pack(fill=BOTH, expand=False)
        left_col = Frame(paned)
        right_col = Frame(paned)
        paned.add(left_col, weight=3)
        paned.add(right_col, weight=2)

        input_section = LabelFrame(left_col, text="Peptide Search Inputs")
        input_section.pack(fill=BOTH, pady=(0, 10))
        Label(
            input_section,
            text="Use the selected genome FASTA or override below to scan translated ORFs.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        self._add_file_picker(
            input_section,
            "Genome FASTA:",
            self.peptide_fasta_var,
            [("FASTA", "*.fa *.fna *.fasta"), ("All", "*.*")],
            dialog_title="Select genome FASTA for peptide scanning",
        )

        param_section = LabelFrame(left_col, text="Window Parameters")
        param_section.pack(fill=BOTH, pady=(0, 10))
        self._add_labeled_entry(param_section, "Peptide window (aa):", self.peptide_window_var, 12)
        self._add_labeled_entry(param_section, "Step size (aa):", self.peptide_step_var, 12)
        
        comb_frame = Frame(param_section)
        comb_frame.pack(fill=X, pady=(4,0))
        Label(comb_frame, text="Combined Window (0=disable):", font=("Segoe UI", 8, "italic"), anchor="w").pack(fill=X, padx=5)
        self._add_labeled_entry(param_section, "  Forward len:", self.peptide_comb_fwd_var, 8)
        self._add_labeled_entry(param_section, "  Reverse len:", self.peptide_comb_rev_var, 8)
        self._add_labeled_entry(param_section, "  Overlap:", self.peptide_comb_overlap_var, 8)
        
        self._add_labeled_entry(
            param_section, "Max motif mismatches:", self.peptide_max_mismatches_var, 12
        )
        self._add_labeled_entry(
            param_section, "Worker processes (0=auto):", self.peptide_workers_var, 12
        )

        frame_section = LabelFrame(left_col, text="Reading Frames")
        frame_section.pack(fill=BOTH, pady=(0, 10))
        Label(
            frame_section,
            text="Choose which ORFs to analyze (plus/minus strands with offsets).",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        frame_rows = Frame(frame_section)
        frame_rows.pack(anchor="w")
        for idx, frame_id in enumerate(("+0", "+1", "+2", "-0", "-1", "-2")):
            Checkbutton(
                frame_rows,
                text=frame_id,
                variable=self.peptide_frame_vars[frame_id],
            ).grid(row=idx // 3, column=idx % 3, sticky="w", padx=6, pady=2)

        motif_section = LabelFrame(right_col, text="Motif Filters")
        motif_section.pack(fill="x", expand=False, pady=(0, 10))
        Label(
            motif_section,
            text="Specify IUPAC amino acid motifs (one per line). All matching windows are reported.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        aa_help_text = (
            "IUPAC Amino Acid Codes:\n\n"
            "Standard single-letter codes: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y\n\n"
            "Ambiguity codes:\n"
            "B = D or N (Aspartate or Asparagine)\n"
            "Z = E or Q (Glutamate or Glutamine)\n"
            "J = I or L (Isoleucine or Leucine)\n"
            "X = any amino acid\n"
            "* = stop codon / translation termination\n\n"
            "All other single letters match themselves exactly."
        )
        self.peptide_motif_widget = self._add_multiline_section(
            motif_section, "Motifs (IUPAC allowed):", help_text=aa_help_text
        )
        self.peptide_exclude_widget = self._add_multiline_section(
            motif_section, "Exclude motifs (skip windows containing any):"
        )
        self.amino_content_widget = self._add_multiline_section(
            motif_section, "Amino acid content limits (RESIDUE:MIN:MAX):"
        )
        self.peptide_repeat_widget = self._add_multiline_section(
            motif_section, "Max repeats (PATTERN:MAX):"
        )

        output_section = LabelFrame(right_col, text="Peptide Output Files")
        output_section.pack(fill=BOTH, pady=(0, 10))
        self._add_file_picker(
            output_section,
            "peptide_coding_hits.tsv:",
            self.peptide_hits_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.peptide_hits_var.get(), "Peptide hits"),
        )
        self._add_file_picker(
            output_section,
            "peptide_coding_hits_annotated.tsv:",
            self.peptide_annotation_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            open_handler=lambda: self._open_table_window(self.peptide_annotation_var.get(), "Annotated peptide hits"),
        )

        actions = Frame(right_col)
        actions.pack(fill=BOTH, pady=(10, 0))
        profile_btn_frame = Frame(actions)
        profile_btn_frame.pack(fill=X, pady=(10, 0))
        Button(profile_btn_frame, text="Load Profile", command=lambda: self.load_search_profile("peptide")).pack(side=LEFT, expand=True, fill=X, padx=(0, 2))
        Button(profile_btn_frame, text="Save Profile", command=lambda: self.save_search_profile("peptide")).pack(side=RIGHT, expand=True, fill=X, padx=(2, 0))

        self.peptide_button = Button(
            actions, text="Run Peptide Coding Search", command=self.run_peptide_search, height=2
        )
        self.peptide_button.pack(fill=BOTH, pady=(8, 0))
        self.peptide_annotate_button = Button(
            actions,
            text="Annotate Peptide Hits",
            command=self.run_peptide_annotation,
            height=2,
        )
        self.peptide_annotate_button.pack(fill=BOTH, pady=(6, 0))


    def _build_combined_tab(self, parent: Frame, heading: str, accent_color: str) -> None:
        wrapper = Frame(parent)
        wrapper.pack(fill="x", expand=False, padx=10, pady=10)
        self._add_tab_banner(wrapper, heading, accent_color)

        Label(
            wrapper,
            text=(
                "Search for regions that satisfy both nucleotide and peptide search criteria simultaneously. "
                "The genome is scanned for nucleotide motifs, and the resulting windows are translated into "
                "amino acids based on the selected reading frames to check for peptide motifs and properties. "
                "You can define an overall search window (e.g., 5000 nt) and then specify smaller subwindows "
                "within it for the nucleotide and peptide parts to find motifs that are spaced further apart."
            ),
            wraplength=800,
            justify=LEFT,
            foreground="#555555"
        ).pack(anchor="w", pady=(5, 10))

        paned = ttk.Panedwindow(wrapper, orient="horizontal")
        paned.pack(fill=BOTH, expand=False)
        left_col = Frame(paned)
        right_col = Frame(paned)
        paned.add(left_col, weight=1)
        paned.add(right_col, weight=1)

        # Left Column: NT Parameters
        nt_section = LabelFrame(left_col, text="Nucleotide Search Parameters")
        nt_section.pack(fill="x", pady=(0, 10), padx=(0, 5), expand=False)
        
        nt_help_text = (
            "IUPAC Nucleotide Codes:\n\n"
            "A, C, G, T, U: Standard bases\n"
            "R: A/G (purine), Y: C/T (pyrimidine)\n"
            "S: G/C (strong), W: A/T (weak)\n"
            "K: G/T (keto), M: A/C (amino)\n"
            "B: C/G/T, D: A/G/T, H: A/C/T, V: A/C/G\n"
            "N: Any base"
        )
        self.combined_nt_motif_widget = self._add_multiline_section(nt_section, "NT Motifs (PATTERN:COUNT):", help_text=nt_help_text)
        self.combined_exclude_nt_widget = self._add_multiline_section(nt_section, "Exclude NT Motifs (PATTERN):")
        self.combined_base_widget = self._add_multiline_section(nt_section, "Base Constraints (BASE:MIN:MAX):")
        self.combined_repeat_widget = self._add_multiline_section(nt_section, "Repeat Constraints (BASE:MAX):")
        self.combined_forward_motif_widget = self._add_multiline_section(nt_section, "Forward-only motifs:")
        self.combined_reverse_motif_widget = self._add_multiline_section(nt_section, "Reverse-only motifs:")
        self.combined_self_comp_widget = self._add_multiline_section(nt_section, "Self-Comp Constraints (PATTERN:MODE:LEN:MM[:EXCL]):")

        # Right Column: Peptide Parameters
        pep_section = LabelFrame(right_col, text="Peptide Search Parameters")
        pep_section.pack(fill="x", pady=(0, 10), padx=(5, 0), expand=False)
        
        pep_help_text = (
            "IUPAC Amino Acid Codes:\n\n"
            "Standard: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y\n"
            "Ambiguity:\n"
            "B: D or N (Aspartate or Asparagine)\n"
            "Z: E or Q (Glutamate or Glutamine)\n"
            "J: I or L (Isoleucine or Leucine)\n"
            "X: Any amino acid\n"
            "*: Stop codon / translation termination"
        )
        self.combined_pep_motif_widget = self._add_multiline_section(pep_section, "Peptide Motifs (IUPAC):", help_text=pep_help_text)
        self.combined_exclude_pep_widget = self._add_multiline_section(pep_section, "Exclude Peptide Motifs:")
        self.combined_amino_content_widget = self._add_multiline_section(pep_section, "Amino acid content limits:")
        self.combined_peptide_repeat_widget = self._add_multiline_section(pep_section, "Max repeats (PATTERN:MAX):")
        
        frame_section = Frame(pep_section)
        frame_section.pack(fill=BOTH, pady=5)
        Label(frame_section, text="Reading Frames:").pack(anchor="w")
        frame_rows = Frame(frame_section)
        frame_rows.pack(anchor="w")
        for idx, frame_id in enumerate(("+0", "+1", "+2", "-0", "-1", "-2")):
            Checkbutton(
                frame_rows,
                text=frame_id,
                variable=self.combined_frame_vars[frame_id],
            ).grid(row=idx // 3, column=idx % 3, sticky="w", padx=6, pady=2)
            
        pep_misc = Frame(pep_section)
        pep_misc.pack(fill=X, pady=5)
        self._add_labeled_entry(pep_misc, "Max Peptide Mismatches:", self.combined_pep_mismatches_var, 12).pack(side=LEFT)

        # Bottom section for global inputs
        bottom_frame = Frame(wrapper)
        bottom_frame.pack(fill=X, pady=(10, 0))
        
        param_section = LabelFrame(bottom_frame, text="Global Run Parameters")
        param_section.pack(fill=BOTH)
        
        row_f = Frame(param_section)
        row_f.pack(fill=X, padx=5, pady=5)
        
        self._add_labeled_entry(row_f, "Window (nt):", self.window_var, 10)
        self._add_labeled_entry(row_f, "Step (nt):", self.step_var, 10)
        self._add_labeled_entry(row_f, "Workers (0=all):", self.workers_var, 10)
        Checkbutton(
            row_f,
            text="Non-overlapping",
            variable=self.non_overlapping_var,
        ).pack(side=LEFT, padx=10)
        
        row_c = Frame(param_section)
        row_c.pack(fill=X, padx=5, pady=(0, 5))
        Label(row_c, text="Combined Strand Window (nt):", font=("Segoe UI", 9, "italic")).pack(side=LEFT, padx=(0, 4))
        self._add_labeled_entry(row_c, "Fwd:", self.combined_comb_fwd_var, 6)
        self._add_labeled_entry(row_c, "Rev:", self.combined_comb_rev_var, 6)
        self._add_labeled_entry(row_c, "Overlap:", self.combined_comb_overlap_var, 6)
        
        sub_row = Frame(param_section)
        sub_row.pack(fill=X, padx=5, pady=(0, 5))
        Label(sub_row, text="Subwindow (Length:Offset) - Optional:", font=("Segoe UI", 9, "italic")).pack(side=LEFT, padx=(0, 4))
        self._add_labeled_entry(sub_row, "NT Len:", self.combined_nt_sub_window_var, 6)
        self._add_labeled_entry(sub_row, "NT Off:", self.combined_nt_sub_offset_var, 6)
        self._add_labeled_entry(sub_row, "Pep Len:", self.combined_pep_sub_window_var, 6)
        self._add_labeled_entry(sub_row, "Pep Off:", self.combined_pep_sub_offset_var, 6)

        profile_btn_frame = Frame(param_section)
        profile_btn_frame.pack(fill=X, padx=5, pady=(5, 0))
        Button(profile_btn_frame, text="Load Profile", command=lambda: self.load_search_profile("combined")).pack(side=LEFT, expand=True, fill=X, padx=(0, 2))
        Button(profile_btn_frame, text="Save Profile", command=lambda: self.save_search_profile("combined")).pack(side=RIGHT, expand=True, fill=X, padx=(2, 0))

        Button(
            param_section,
            text="Run Combined Search",
            bg="#d97706",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            command=self.run_combined_scan,
            height=2,
        ).pack(fill=BOTH, padx=5, pady=10)

    def _build_triplex_tab(self, parent: Frame, heading: str, accent_color: str) -> None:
        wrapper = Frame(parent)
        wrapper.pack(fill="x", expand=False, padx=10, pady=10)

        self._add_tab_banner(wrapper, heading, accent_color)

        paned = ttk.Panedwindow(wrapper, orient="horizontal")
        paned.pack(fill=BOTH, expand=False)
        left_col = Frame(paned)
        right_col = Frame(paned)
        paned.add(left_col, weight=3)
        paned.add(right_col, weight=2)

        inputs = LabelFrame(left_col, text="Triplex Input Files")
        inputs.pack(fill=BOTH, pady=(0, 10))
        Label(
            inputs,
            text="Provide either RNA windows (recommended) or raw RNA to be windowed on the fly, plus a DNA FASTA to scan.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        mode_frame = LabelFrame(inputs, text="RNA Input Mode")
        mode_frame.pack(fill=BOTH, pady=(0, 8))
        Label(
            mode_frame,
            text="Toggle the active RNA input controls.",
            anchor="w",
            justify=LEFT,
        ).pack(anchor="w")
        Radiobutton(
            mode_frame,
            text="Use RNA windows FASTA (default)",
            variable=self.triplex_input_mode_var,
            value="windows",
            command=self._update_triplex_input_mode,
        ).pack(anchor="w", pady=(2, 0))
        Radiobutton(
            mode_frame,
            text="Use RNA FASTA/text input",
            variable=self.triplex_input_mode_var,
            value="rna",
            command=self._update_triplex_input_mode,
        ).pack(anchor="w", pady=(0, 2))
        self.triplex_windows_controls = self._add_file_picker(
            inputs,
            "RNA windows FASTA (preferred):",
            self.triplex_rna_windows_var,
            [("FASTA", "*.fa *.fasta"), ("All", "*.*")],
            dialog_title="Select RNA windows FASTA",
            collect_controls=True,
        )
        self.triplex_rna_controls = self._add_file_picker(
            inputs,
            "RNA FASTA sequence file:",
            self.triplex_rna_var,
            [("RNA/Seq", "*.txt *.fa *.fasta *.rna"), ("All", "*.*")],
            dialog_title="Select RNA sequence file",
            collect_controls=True,
        )
        self._add_file_picker(
            inputs,
            "DNA FASTA sequence file:",
            self.triplex_dna_var,
            [("FASTA", "*.fa *.fna *.fasta"), ("All", "*.*")],
            dialog_title="Select DNA FASTA file",
        )

        sequence_section = LabelFrame(left_col, text="RNA Sequence (Optional)")
        sequence_section.pack(fill=BOTH, pady=(0, 10))
        Label(
            sequence_section,
            text="Paste an RNA sequence (ACGU). If provided, it overrides the RNA file.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        self.triplex_rna_text_widget = scrolledtext.ScrolledText(sequence_section, height=4, wrap="word")
        self.triplex_rna_text_widget.pack(fill=BOTH, expand=True)

        params = LabelFrame(left_col, text="Triplex Searching Parameters & Output")
        params.pack(fill=BOTH, pady=(0, 10))
        Label(
            params,
            text="Set scanning thresholds, strand options, and output locations.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        self._add_labeled_entry(params, "Min window length:", self.triplex_min_len_var, 10)
        self._add_labeled_entry(params, "Max window length:", self.triplex_max_len_var, 10)
        self._add_labeled_entry(params, "Min score:", self.triplex_min_score_var, 10)
        self._add_labeled_entry(params, "Max mismatches:", self.triplex_max_mismatches_var, 10)
        self._add_labeled_entry(params, "Triplex workers (0 = all CPUs):", self.triplex_workers_var, 12)
        strand_frame = Frame(params)
        strand_frame.pack(fill=BOTH, pady=(4, 2))
        Label(strand_frame, text="DNA strands to scan:", anchor="w").pack(anchor="w")
        strand_opts = Frame(strand_frame)
        strand_opts.pack(anchor="w")
        Checkbutton(
            strand_opts, text="Forward (+)", variable=self.triplex_forward_var
        ).pack(side=LEFT, padx=(0, 10))
        Checkbutton(
            strand_opts, text="Reverse (-)", variable=self.triplex_reverse_var
        ).pack(side=LEFT)
        self._add_file_picker(
            params,
            "Triplex hits output (.tsv):",
            self.triplex_output_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
            enable_browse=True,
            open_handler=lambda: self._open_table_window(self.triplex_output_var.get(), "Triplex Hits"),
        )

        Label(
            left_col,
            text="Triplex searches respect the shared output folder name from the Genome Scanner tab.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))

        self.triplex_button = Button(left_col, text="Run Triplex Finder", command=self.run_triplex_search, height=2)
        self.triplex_button.pack(fill=BOTH, pady=(4, 0))
        self._update_triplex_input_mode()

        windowing = LabelFrame(right_col, text="RNA Windowing Tool")
        windowing.pack(fill=BOTH, pady=(0, 10))
        Label(
            windowing,
            text="Split long RNA sequences into overlapping windows before scanning.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        self._add_labeled_entry(windowing, "Window length:", self.split_window_var, 10)
        self._add_labeled_entry(windowing, "Step size:", self.split_step_var, 10)
        self._add_labeled_entry(windowing, "Min length:", self.split_min_length_var, 10)
        Checkbutton(
            windowing,
            text="Include partial final window",
            variable=self.split_include_partial_var,
        ).pack(anchor="w", pady=(4, 2))
        self._add_file_picker(
            windowing,
            "Window output FASTA:",
            self.split_output_var,
            [("FASTA", "*.fa *.fasta"), ("All", "*.*")],
        )
        self.split_button = Button(
            windowing, text="Generate RNA Windows", command=self.run_rna_windowing, height=2
        )
        self.split_button.pack(fill=BOTH, pady=(6, 0))

        scoring = LabelFrame(right_col, text="RNA Scoring, Annotation & Genes")
        scoring.pack(fill="x", expand=False, pady=(0, 10))
        Label(
            scoring,
            text="Score RNA windows, annotate triplex hits, and extract overlapping genes.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        Label(
            scoring,
            text=(
                "Recommended settings: Min score ≥ 20, pyrimidine fraction ≥ 0.5. "
                "Use spacing (e.g., 20–40 nt) to keep only one representative window per region."
            ),
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        self._add_labeled_entry(scoring, "Min score:", self.score_min_score_var, 10)
        self._add_labeled_entry(scoring, "Min pyrimidine frac:", self.score_min_pyrimidine_var, 8)
        self._add_labeled_entry(scoring, "Min spacing (nt):", self.score_min_spacing_var, 10)
        self._add_file_picker(
            scoring,
            "Score summary TSV:",
            self.score_tsv_var,
            [("TSV", "*.tsv"), ("All", "*.*")],
        )
        self._add_file_picker(
            scoring,
            "Filtered FASTA:",
            self.score_fasta_var,
            [("FASTA", "*.fa *.fasta"), ("All", "*.*")],
        )
        self.score_button = Button(
            scoring, text="Score & Filter Windows", command=self.run_rna_scoring, height=2
        )
        self.score_button.pack(fill=BOTH, pady=(6, 0))
        self.annotate_triplex_button = Button(
            scoring,
            text="Annotate Triplex Hits",
            command=self.run_triplex_annotation,
            height=2,
        )
        self.annotate_triplex_button.pack(fill=BOTH, pady=(6, 0))
        self.triplex_gene_button = Button(
            scoring,
            text="Extract Triplex Genes",
            command=self.run_triplex_gene_extract,
            height=2,
        )
        self.triplex_gene_button.pack(fill=BOTH, pady=(6, 0))

    def _build_visualization_tab(self, parent: Frame, heading: str, accent_color: str) -> None:
        wrapper = Frame(parent)
        wrapper.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self._add_tab_banner(wrapper, heading, accent_color)

        paned = ttk.Panedwindow(wrapper, orient="horizontal")
        paned.pack(fill=BOTH, expand=True)
        left_col = Frame(paned)
        right_col = Frame(paned)
        paned.add(left_col, weight=3)
        paned.add(right_col, weight=2)

        selection = LabelFrame(left_col, text="Select Results")
        selection.pack(fill=BOTH, pady=(0, 10))
        Label(
            selection,
            text="Enable one or more datasets to overlay their hits. Each dataset receives proportional chromosome width, its own color, and customizable opacity/line style for better overlap handling.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        source_rows = [
            ("nt_sequence_hits", "NT sequence hits", self.process_hits_var, "nt_sequence_hits"),
            ("nt_sequence_hits_analyzed", "NT sequence hits (analyzed)", self.analyzed_hits_var, "nt_sequence_hits_analyzed"),
            (
                "nt_sequence_hits_annotated",
                "NT sequence hits (annotated)",
                self.processed_hits_var,
                "nt_sequence_hits_annotated",
            ),
            ("triplex_hits", "Triplex hits", self.triplex_output_var, "triplex_hits"),
            ("triplex_hits_annotated", "Triplex hits (annotated)", self.triplex_annotation_var, "triplex_hits_annotated"),
            ("peptide_hits", "Peptide coding hits", self.peptide_hits_var, "peptide_hits"),
            (
                "peptide_hits_annotated",
                "Peptide hits (annotated)",
                self.peptide_annotation_var,
                "peptide_hits_annotated",
            ),
        ]
        for key, label, var_ref, _kind in source_rows:
            card = Frame(selection, relief="groove", borderwidth=1, padx=6, pady=4)
            card.pack(fill=BOTH, pady=3)
            top_row = Frame(card)
            top_row.pack(fill=BOTH)
            Checkbutton(top_row, text=label, variable=self.vector_source_flags[key]).pack(side=LEFT)
            Button(
                top_row,
                text="Open",
                command=lambda ref=var_ref: self._open_path_default(ref.get()),
            ).pack(side=RIGHT, padx=4)
            Button(
                top_row,
                text="View",
                command=lambda ref=var_ref, title=label: self._open_table_window(ref.get(), title),
            ).pack(side=RIGHT, padx=4)
            controls = self.vector_dataset_options[key]
            style_row = Frame(card)
            style_row.pack(anchor="w", pady=(4, 0))
            Label(style_row, text="Opacity (0-1):").pack(side=LEFT)
            Entry(style_row, textvariable=controls["opacity"], width=6).pack(side=LEFT, padx=(4, 12))
            ttk.Combobox(
                style_row,
                textvariable=controls["style"],
                state="readonly",
                values=("solid", "dashed"),
                width=8,
            ).pack(side=LEFT)
            
            Label(style_row, text="Color:").pack(side=LEFT, padx=(12, 4))
            c_btn = Button(style_row, text="   ", width=2, relief="sunken")
            c_btn.pack(side=LEFT)
            if controls["color"].get():
                c_btn.configure(bg=controls["color"].get())
            c_btn.configure(command=lambda v=controls["color"], b=c_btn: self.choose_color(v, b))

        gene_section = LabelFrame(left_col, text="Gene List Extraction")
        gene_section.pack(fill=BOTH, pady=(0, 10))
        self.genes_button = Button(
            gene_section, text="Extract Gene List", command=self.run_gene_extract, height=2
        )
        self.genes_button.pack(fill=BOTH, pady=(6, 0))

        custom_section = LabelFrame(left_col, text="Custom Overlays")
        custom_section.pack(fill=BOTH, expand=True, pady=(4, 0))
        Label(
            custom_section,
            text="Add extra TSV files (e.g., alternative filtering outputs) and choose their data type.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        self._add_vector_custom_row(custom_section, self.vector_input_var)
        Button(
            custom_section,
            text="Add another file",
            command=lambda: self._add_vector_custom_row(custom_section),
        ).pack(anchor="w", pady=(6, 0))

        options = LabelFrame(right_col, text="Display Options")
        options.pack(fill=BOTH, pady=(0, 10))
        label_box = LabelFrame(options, text="Chromosome labels")
        label_box.pack(fill=BOTH, pady=(0, 6))
        Radiobutton(
            label_box,
            text="Hide labels",
            variable=self.vector_label_mode_var,
            value="none",
        ).pack(anchor="w", pady=1)
        Radiobutton(
            label_box,
            text="Gene names only",
            variable=self.vector_label_mode_var,
            value="gene",
        ).pack(anchor="w", pady=1)
        Radiobutton(
            label_box,
            text="Gene + coordinates",
            variable=self.vector_label_mode_var,
            value="gene+coords",
        ).pack(anchor="w", pady=1)
        Label(options, text="Plot title (optional):").pack(anchor="w", pady=(6, 0))
        Entry(options, textvariable=self.vector_title_var).pack(fill=BOTH, pady=(0, 4))
        Checkbutton(
            options,
            text="Overlay density (heatmap) view",
            variable=self.vector_density_var,
        ).pack(anchor="w", pady=2)
        Checkbutton(
            options,
            text="Annotate per-hit column values (palindrome/PQS scores, etc.)",
            variable=self.vector_annotate_var,
        ).pack(anchor="w", pady=2)
        annotate_row = Frame(options)
        annotate_row.pack(fill=BOTH, padx=(18, 0), pady=(0, 6))
        Label(annotate_row, text="Annotation column:").pack(side=LEFT)
        Entry(annotate_row, textvariable=self.vector_annotation_column_var).pack(
            side=LEFT, fill=BOTH, expand=True, padx=(6, 0)
        )
        Label(
            options,
            text="Examples: longest_palindrome_sequence, pqsfinder_score, region_annotations",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))

        Label(options, text="Scaling Mode:").pack(anchor="w")
        ttk.Combobox(options, textvariable=self.vector_scale_mode_var, state="readonly", values=("absolute", "relative")).pack(fill=BOTH, pady=(0, 4))

        pub_section = LabelFrame(right_col, text="Publication Settings")
        pub_section.pack(fill=BOTH, pady=(0, 10))
        
        row1 = Frame(pub_section)
        row1.pack(fill=X, padx=5, pady=2)
        Label(row1, text="Format:").pack(side=LEFT)
        ttk.Combobox(row1, textvariable=self.vector_output_format_var, state="readonly", values=("png", "pdf", "svg", "tiff", "eps"), width=5).pack(side=LEFT, padx=(4, 10))
        Label(row1, text="DPI:").pack(side=LEFT)
        Entry(row1, textvariable=self.vector_dpi_var, width=5).pack(side=LEFT, padx=(4, 10))
        Label(row1, text="W (in):").pack(side=LEFT)
        Entry(row1, textvariable=self.vector_fig_width_var, width=4).pack(side=LEFT, padx=(4, 10))
        Label(row1, text="H:").pack(side=LEFT)
        Entry(row1, textvariable=self.vector_fig_height_var, width=4).pack(side=LEFT, padx=(4, 0))
        
        row2 = Frame(pub_section)
        row2.pack(fill=X, padx=5, pady=2)
        Label(row2, text="Font:").pack(side=LEFT)
        ttk.Combobox(row2, textvariable=self.vector_font_family_var, values=("Arial", "Times New Roman", "Helvetica", "Courier New"), width=12).pack(side=LEFT, padx=(4, 10))
        Label(row2, text="Size:").pack(side=LEFT)
        Entry(row2, textvariable=self.vector_font_size_var, width=4).pack(side=LEFT, padx=(4, 10))
        Label(row2, text="Palette:").pack(side=LEFT)
        ttk.Combobox(row2, textvariable=self.vector_palette_var, values=("default", "tab10", "Set2", "Dark2", "Paired", "viridis"), width=8).pack(side=LEFT, padx=(4, 0))
        
        row3 = Frame(pub_section)
        row3.pack(fill=X, padx=5, pady=2)
        Label(row3, text="BG Color:").pack(side=LEFT)
        bg_btn = Button(row3, text="   ", width=2, relief="sunken", bg=self.vector_bg_color_var.get())
        bg_btn.pack(side=LEFT, padx=(4, 10))
        bg_btn.configure(command=lambda v=self.vector_bg_color_var, b=bg_btn: self.choose_color(v, b))
        Label(row3, text="BG Alpha:").pack(side=LEFT)
        Entry(row3, textvariable=self.vector_bg_alpha_var, width=5).pack(side=LEFT, padx=(4, 10))
        Label(row3, text="Legend:").pack(side=LEFT)
        ttk.Combobox(row3, textvariable=self.vector_legend_pos_var, state="readonly", values=("right", "bottom"), width=8).pack(side=LEFT, padx=(4, 0))

        filter_box = LabelFrame(right_col, text="Input Filters")
        filter_box.pack(fill=BOTH, pady=(0, 10))
        Label(
            filter_box,
            text="Limit the rendering to specific chromosomes, genomic ranges, or gene names before plotting.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        seq_entry = self._add_labeled_entry(
            filter_box, "Sequence IDs:", self.vector_seq_ids_var, width=28
        )
        seq_entry.config(width=28)
        Label(
            filter_box,
            text="Separate IDs with spaces or commas (e.g., chr1 chrX contig5).",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        self._add_labeled_entry(
            filter_box, "Regions (chr:start-end):", self.vector_region_filter_var, width=28
        )
        Label(
            filter_box,
            text="Example: chr1:150000-220000; add multiple ranges separated by commas or spaces.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))
        self._add_labeled_entry(
            filter_box, "Gene filters:", self.vector_gene_filter_var, width=28
        )
        Label(
            filter_box,
            text="Provide gene names separated by commas/space; matching hits only will be drawn.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 4))

        legend_frame = LabelFrame(right_col, text="Legend Preview")
        legend_frame.pack(fill=BOTH, pady=(0, 10))
        self._populate_color_legend(legend_frame)

        output_frame = LabelFrame(right_col, text="Output & Actions")
        output_frame.pack(fill=BOTH, expand=True)
        Label(
            output_frame,
            text="SVG output is placed in the selected output_<name> directory.",
            wraplength=320,
            justify=LEFT,
        ).pack(anchor="w", pady=(0, 6))
        self._add_file_picker(
            output_frame,
            "SVG output path:",
            self.vector_output_var,
            save_dialog=True,
        )
        self.vector_viz_button = Button(
            output_frame,
            text="Render Publication Plot",
            command=self.run_vector_visualization,
            height=2,
        )
        self.vector_viz_button.pack(fill=BOTH, pady=(12, 0))
        Button(
            output_frame,
            text="View Result File",
            command=lambda: self._open_path_default(self.vector_output_var.get()),
        ).pack(fill=BOTH, pady=(6, 0))

    def _build_tools_tab(self, parent: Frame, heading: str, accent_color: str) -> None:
        canvas = Canvas(parent, highlightthickness=0)
        v_scroll = Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = Frame(canvas)
        
        scroll_frame.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scroll.set)
        
        v_scroll.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        wrapper = Frame(scroll_frame)
        wrapper.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        self._add_tab_banner(wrapper, heading, accent_color)
        
        if not _TOOLS_AVAILABLE:
            Label(wrapper, text="Analysis scripts (nt_sequence_G4_TD_analysis.py) not found. Tools tab disabled.", fg="red").pack(pady=20)
            return

        # 1. G4Hunter Scorer
        g4_frame, g4_content = self._make_collapsible_tool(wrapper, "G4Hunter Scorer")
        Label(g4_content, text="NT Sequence:").pack(anchor="w")
        g4_text = scrolledtext.ScrolledText(g4_content, height=4, width=60)
        g4_text.pack(fill=X, pady=(0, 6))
        row = Frame(g4_content)
        row.pack(fill=X)
        self._add_labeled_entry(row, "Window Size:", self.tool_g4_window_var, 6)
        Button(row, text="Run Analysis", command=lambda: self._run_g4hunter_tool(g4_text, g4_results)).pack(side=LEFT, padx=10)
        g4_results = scrolledtext.ScrolledText(g4_content, height=8, width=60, state="disabled", bg="#f8f9fa")
        g4_results.pack(fill=X, pady=6)
        Button(g4_content, text="Save Results", command=lambda: self._save_tool_results(g4_results)).pack(side=RIGHT, padx=2)
        Button(g4_content, text="Copy Results", command=lambda: self._copy_tool_results(g4_results)).pack(side=RIGHT, padx=2)

        # 2. Codon Optimizer
        codon_frame, codon_content = self._make_collapsible_tool(wrapper, "Codon Optimizer")
        Label(codon_content, text="Input NT or Peptide Sequence:").pack(anchor="w")
        codon_text = scrolledtext.ScrolledText(codon_content, height=4, width=60)
        codon_text.pack(fill=X, pady=(0, 6))
        Button(codon_content, text="Optimize Sequence", command=lambda: self._run_codon_optimizer_tool(codon_text, codon_results)).pack(anchor="w")
        codon_results = scrolledtext.ScrolledText(codon_content, height=8, width=60, state="disabled", bg="#f8f9fa")
        codon_results.pack(fill=X, pady=6)
        Button(codon_content, text="Save Results", command=lambda: self._save_tool_results(codon_results)).pack(side=RIGHT, padx=2)
        Button(codon_content, text="Copy Results", command=lambda: self._copy_tool_results(codon_results)).pack(side=RIGHT, padx=2)

        # 3. 6-Frame Translator
        trans_frame, trans_content = self._make_collapsible_tool(wrapper, "6-Frame Translator")
        Label(trans_content, text="NT Sequence:").pack(anchor="w")
        trans_text = scrolledtext.ScrolledText(trans_content, height=4, width=60)
        trans_text.pack(fill=X, pady=(0, 6))
        Button(trans_content, text="Translate All Frames", command=lambda: self._run_translator_tool(trans_text, trans_results)).pack(anchor="w")
        trans_results = scrolledtext.ScrolledText(trans_content, height=12, width=60, state="disabled", bg="#f8f9fa")
        trans_results.pack(fill=X, pady=6)
        Button(trans_content, text="Save Results", command=lambda: self._save_tool_results(trans_results)).pack(side=RIGHT, padx=2)
        Button(trans_content, text="Copy Results", command=lambda: self._copy_tool_results(trans_results)).pack(side=RIGHT, padx=2)

        # 4. Thermodynamic Analyzer
        thermo_frame, thermo_content = self._make_collapsible_tool(wrapper, "Thermodynamic Analyzer")
        Label(thermo_content, text="NT Sequence:").pack(anchor="w")
        thermo_text = scrolledtext.ScrolledText(thermo_content, height=4, width=60)
        thermo_text.pack(fill=X, pady=(0, 6))
        row = Frame(thermo_content)
        row.pack(fill=X)
        self._add_labeled_entry(row, "DNA Conc (M):", self.tool_thermo_dna_conc_var, 8)
        self._add_labeled_entry(row, "Salt Conc (M):", self.tool_thermo_salt_conc_var, 8)
        Button(row, text="Analyze Properties", command=lambda: self._run_thermo_tool(thermo_text, thermo_results)).pack(side=LEFT, padx=10)
        thermo_results = scrolledtext.ScrolledText(thermo_content, height=10, width=60, state="disabled", bg="#f8f9fa")
        thermo_results.pack(fill=X, pady=6)
        Button(thermo_content, text="Save Results", command=lambda: self._save_tool_results(thermo_results)).pack(side=RIGHT, padx=2)
        Button(thermo_content, text="Copy Results", command=lambda: self._copy_tool_results(thermo_results)).pack(side=RIGHT, padx=2)

        # 5. G4 Mutation Optimizer
        mut_frame, mut_content = self._make_collapsible_tool(wrapper, "G4 Mutation Optimizer")
        Label(mut_content, text="NT Sequence:").pack(anchor="w")
        mut_text = scrolledtext.ScrolledText(mut_content, height=4, width=60)
        mut_text.pack(fill=X, pady=(0, 6))
        row = Frame(mut_content)
        row.pack(fill=X)
        self._add_labeled_entry(row, "Target G4 Score:", self.tool_mutator_target_var, 6)
        self._add_labeled_entry(row, "Max Mutations:", self.tool_mutator_max_mut_var, 6)
        Button(row, text="Optimize G4", command=lambda: self._run_g4_mutator_tool(mut_text, mut_results)).pack(side=LEFT, padx=10)
        mut_results = scrolledtext.ScrolledText(mut_content, height=8, width=60, state="disabled", bg="#f8f9fa")
        mut_results.pack(fill=X, pady=6)
        Button(mut_content, text="Save Results", command=lambda: self._save_tool_results(mut_results)).pack(side=RIGHT, padx=2)
        Button(mut_content, text="Copy Results", command=lambda: self._copy_tool_results(mut_results)).pack(side=RIGHT, padx=2)

    def _make_collapsible_tool(self, parent: Frame, title: str) -> Tuple[LabelFrame, Frame]:
        lf = LabelFrame(parent, text=f" ▶ {title}", font=("Segoe UI", 10, "bold"), labelanchor="nw")
        lf.pack(fill=X, pady=5)
        
        content = Frame(lf, padx=10, pady=10)
        content.pack(fill=X)
        
        is_expanded = [True]
        
        def toggle():
            if is_expanded[0]:
                content.pack_forget()
                lf.configure(text=f" ◀ {title}")
                is_expanded[0] = False
            else:
                content.pack(fill=X)
                lf.configure(text=f" ▶ {title}")
                is_expanded[0] = True
        
        # Click on the LabelFrame title? Use a small button for reliability
        btn_row = Frame(lf)
        # We can't easily put it in the label bar of a LabelFrame in pure Tkinter without hacks
        # So we'll just put a tiny button at the top of content or better, as a separate header button
        
        # Actually, let's use a trick: bind click to the LabelFrame label if possible.
        # But simpler: just add a toggle button inside the frame at the top.
        Button(content, text="Collapse", command=toggle, font=("Segoe UI", 8)).pack(anchor="ne")
        
        return lf, content

    def _copy_tool_results(self, text_widget: scrolledtext.ScrolledText) -> None:
        content = text_widget.get("1.0", END).strip()
        if not content:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.root.update()
        # Optionally show a status message if you want
        
    def _save_tool_results(self, text_widget: scrolledtext.ScrolledText) -> None:
        content = text_widget.get("1.0", END).strip()
        if not content:
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".tsv",
            filetypes=[("TSV files", "*.tsv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
            
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Success", f"Results saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")

    def _run_g4hunter_tool(self, input_widget, output_widget) -> None:
        seq = input_widget.get("1.0", END).strip().upper()
        if not seq: return
        
        try:
            w_size = int(self.tool_g4_window_var.get())
        except: w_size = 25
        
        g4h = _g4hunter_score(seq, w_size)
        g4b = _g4boost_score(seq)
        counts, gc_pct = _base_composition(seq)
        
        res = []
        res.append(f"G4Hunter Score (window={w_size}): {g4h:.4f}")
        res.append(f"G4Boost Score: {g4b:.4f}")
        res.append(f"GC Content: {gc_pct:.2f}%")
        res.append(f"Base Counts: {', '.join(f'{b}={c}' for b, c in counts.items())}")
        res.append("\nSequence:")
        res.append(seq)
        
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert(END, "\n".join(res))
        output_widget.configure(state="disabled")

    def _run_codon_optimizer_tool(self, input_widget, output_widget) -> None:
        raw_input = input_widget.get("1.0", END).strip()
        if not raw_input: return
        
        # Check if peptide or NT
        is_peptide = any(c not in "ACGTUNacgtun \n\r\t" for c in raw_input)
        
        optimized = []
        if is_peptide:
            # Reverse translate using best codons
            # We need a reverse codon map
            rev_map = {}
            for codon, aa in _CODON_TABLE.items():
                weight = _CODON_WEIGHTS.get(codon, 0.0)
                if aa not in rev_map or weight > rev_map[aa][1]:
                    rev_map[aa] = (codon, weight)
            
            pep = "".join(raw_input.split()).upper()
            for aa in pep:
                if aa in rev_map:
                    optimized.append(rev_map[aa][0])
                else:
                    optimized.append("???")
            opt_seq = "".join(optimized)
        else:
            # Re-encode NT sequence
            nt = "".join(raw_input.split()).upper().replace("U", "T")
            # Best synonyms for each AA
            best_synonyms = {}
            for codon, aa in _CODON_TABLE.items():
                weight = _CODON_WEIGHTS.get(codon, 0.0)
                if aa not in best_synonyms or weight > best_synonyms[aa][1]:
                    best_synonyms[aa] = (codon, weight)
            
            for i in range(0, len(nt) - 2, 3):
                codon = nt[i:i+3]
                aa = _CODON_TABLE.get(codon)
                if aa and aa in best_synonyms:
                    optimized.append(best_synonyms[aa][0])
                else:
                    optimized.append(codon)
            opt_seq = "".join(optimized)
            
        cai_before = _calc_codon_efficiency(raw_input) if not is_peptide else 0.0
        cai_after = _calc_codon_efficiency(opt_seq)
        
        res = []
        res.append(f"Input Type: {'Peptide' if is_peptide else 'Nucleotide'}")
        if not is_peptide:
            res.append(f"Codon Efficiency (CAI) Before: {cai_before:.4f}")
        res.append(f"Codon Efficiency (CAI) After: {cai_after:.4f}")
        res.append("\nOptimized Sequence:")
        res.append(opt_seq)
        
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert(END, "\n".join(res))
        output_widget.configure(state="disabled")

    def _run_translator_tool(self, input_widget, output_widget) -> None:
        nt = "".join(input_widget.get("1.0", END).strip().split()).upper().replace("U", "T")
        if not nt: return
        
        rc = _reverse_complement(nt)
        
        res = []
        res.append("--- Forward Frames ---")
        for f in range(3):
            pep = _translate_sequence(nt[f:])
            res.append(f"Frame +{f}: {pep}")
            
        res.append("\n--- Reverse Frames ---")
        for f in range(3):
            pep = _translate_sequence(rc[f:])
            res.append(f"Frame -{f}: {pep}")
            
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert(END, "\n".join(res))
        output_widget.configure(state="disabled")

    def _run_thermo_tool(self, input_widget, output_widget) -> None:
        seq = "".join(input_widget.get("1.0", END).strip().split()).upper().replace("U", "T")
        if not seq: return
        
        try:
            conc = float(self.tool_thermo_dna_conc_var.get())
            salt = float(self.tool_thermo_salt_conc_var.get())
        except:
            conc, salt = 0.001, 0.15
            
        tm_w = _tm_wallace(seq)
        tm_nn, dh, ds = _tm_nearest_neighbor(seq, conc, salt)
        counts, gc_pct = _base_composition(seq)
        
        res = []
        res.append(f"Length: {len(seq)} bp")
        res.append(f"GC Content: {gc_pct:.2f}%")
        res.append(f"Tm (Wallace): {tm_w:.1f} °C")
        res.append(f"Tm (Nearest-Neighbor): {tm_nn:.2f} °C")
        res.append(f"Delta H: {dh:.2f} kcal/mol")
        res.append(f"Delta S: {ds:.2f} cal/mol*K")
        res.append("\n--- Structural Scores ---")
        res.append(f"i-Motif Score: {_i_motif_score(seq):.4f}")
        res.append(f"R-Loop Score: {_r_loop_score(seq):.4f}")
        res.append(f"Hairpin Score: {_hairpin_score(seq):.4f}")
        res.append(f"CpG Island Score: {_cpg_island_score(seq):.4f}")
        
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert(END, "\n".join(res))
        output_widget.configure(state="disabled")

    def _run_g4_mutator_tool(self, input_widget, output_widget) -> None:
        seq = "".join(input_widget.get("1.0", END).strip().split()).upper().replace("U", "T")
        if not seq: return
        
        try:
            target = float(self.tool_mutator_target_var.get())
            max_mut = int(self.tool_mutator_max_mut_var.get())
            w_size = int(self.tool_g4_window_var.get())
        except:
            target, max_mut, w_size = 1.5, 5, 25
            
        current_seq = list(seq)
        current_score = _g4hunter_score("".join(current_seq), w_size)
        mutations = []
        
        # Greedy optimization
        for _ in range(max_mut):
            if current_score >= target:
                break
                
            best_mut = None
            best_new_score = current_score
            
            # Try mutation at each position to a G (since we want to increase G4 score usually)
            # Actually, G4Hunter score is boosted by G runs and decreased by C runs.
            for i in range(len(current_seq)):
                orig = current_seq[i]
                if orig == 'G': continue
                
                for alt in ['G', 'A', 'T']: # Swapping to G is best, but others might help if we are swapping away from C
                    if alt == orig: continue
                    current_seq[i] = alt
                    new_score = _g4hunter_score("".join(current_seq), w_size)
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_mut = (i, orig, alt)
                    current_seq[i] = orig
            
            if best_mut:
                idx, o, a = best_mut
                current_seq[idx] = a
                current_score = best_new_score
                mutations.append(f"Pos {idx+1}: {o} -> {a} (New Score: {current_score:.4f})")
            else:
                break
                
        res = []
        res.append(f"Original Score: {_g4hunter_score(seq, w_size):.4f}")
        res.append(f"Final Score: {current_score:.4f}")
        res.append(f"Mutations Applied: {len(mutations)}")
        res.append("\n".join(mutations))
        res.append("\nOptimized Sequence:")
        res.append("".join(current_seq))
        
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert(END, "\n".join(res))
        output_widget.configure(state="disabled")



    def _add_vector_custom_row(self, parent: Frame, preset_var: StringVar | None = None) -> None:
        include_var = BooleanVar(value=not self.vector_custom_rows)
        path_var = preset_var or StringVar()
        kind_var = StringVar(value="auto")
        label_var = StringVar(value=f"Custom {len(self.vector_custom_rows) + 1}")
        opacity_var = StringVar(value="0.75")
        style_var = StringVar(value="solid")
        color_var = StringVar(value="")

        frame = Frame(parent, relief="groove", borderwidth=1)
        frame.pack(fill=BOTH, pady=4)
        Checkbutton(frame, text="Include this file", variable=include_var).pack(anchor="w", pady=(4, 0))
        row = Frame(frame)
        row.pack(fill=BOTH, pady=2)
        Entry(row, textvariable=path_var).pack(side=LEFT, fill=BOTH, expand=True)
        Button(
            row,
            text="Browse",
            command=lambda: self.browse_file(
                path_var, file_types=[("TSV/CSV", "*.tsv *.csv"), ("All", "*.*")]
            ),
        ).pack(side=LEFT, padx=4)
        Button(
            row,
            text="View",
            command=lambda var=path_var, lbl=label_var: self._open_table_window(
                var.get(), lbl.get() or "Custom visualization input"
            ),
        ).pack(side=LEFT, padx=4)
        Button(
            row,
            text="Open",
            command=lambda: self._open_path_default(path_var.get()),
        ).pack(side=LEFT, padx=4)

        kind_row = Frame(frame)
        kind_row.pack(fill=BOTH, pady=2)
        Label(kind_row, text="Interpret as:").pack(side=LEFT)
        OptionMenu(
            kind_row,
            kind_var,
            "auto",
            "nucleotide_hits",
            "processed_hits",
            "triplex_hits",
            "annotated_triplex",
        ).pack(side=LEFT, padx=4)

        label_row = Frame(frame)
        label_row.pack(fill=BOTH, pady=(0, 4))
        Label(label_row, text="Legend label:").pack(side=LEFT)
        Entry(label_row, textvariable=label_var).pack(side=LEFT, fill=BOTH, expand=True, padx=4)
        style_row = Frame(frame)
        style_row.pack(fill=BOTH, pady=(0, 4))
        Label(style_row, text="Opacity (0-1):").pack(side=LEFT)
        Entry(style_row, textvariable=opacity_var, width=6).pack(side=LEFT, padx=(4, 12))
        Label(style_row, text="Style:").pack(side=LEFT)
        ttk.Combobox(style_row, textvariable=style_var, state="readonly", values=("solid", "dashed"), width=8).pack(side=LEFT, padx=(4, 12))
        Label(style_row, text="Color:").pack(side=LEFT, padx=4)
        c_btn = Button(style_row, text="   ", width=2, relief="sunken")
        c_btn.pack(side=LEFT)
        c_btn.configure(command=lambda v=color_var, b=c_btn: self.choose_color(v, b))

        def remove_row() -> None:
            frame.destroy()
            self.vector_custom_rows[:] = [row for row in self.vector_custom_rows if row["frame"] is not frame]

        Button(frame, text="Remove", command=remove_row).pack(anchor="e", padx=4, pady=(0, 4))
        self.vector_custom_rows.append(
            {
                "frame": frame,
                "include": include_var,
                "path": path_var,
                "kind": kind_var,
                "label": label_var,
                "opacity": opacity_var,
                "style": style_var,
                "color": color_var,
            }
        )

    def _normalize_opacity_input(self, value: str | None) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.75
        numeric = max(0.05, min(1.0, numeric))
        return f"{numeric:.3f}"

    def _normalize_line_style_input(self, value: str | None) -> str:
        if not value:
            return "solid"
        cleaned = value.strip().lower()
        return cleaned if cleaned == "dashed" else "solid"

    def _split_filter_values(self, text: str | None) -> List[str]:
        tokens: List[str] = []
        if not text:
            return tokens
        for token in re.split(r"[\s,;]+", text.strip()):
            token = token.strip()
            if token:
                tokens.append(token)
        return tokens

    def _populate_color_legend(self, parent: Frame) -> None:
        functional_frame = Frame(parent)
        functional_frame.pack(fill=BOTH, pady=(0, 8))
        Label(functional_frame, text="Functional classes").pack(anchor="w")
        for key in CLASS_LEGEND_ORDER:
            color = VECTOR_CLASS_COLORS.get(key)
            if not color:
                continue
            label = VECTOR_CLASS_DISPLAY.get(key, key.replace("_", " ").title())
            self._add_color_chip(functional_frame, label, color)
        dataset_frame = Frame(parent)
        dataset_frame.pack(fill=BOTH)
        Label(dataset_frame, text="Dataset palette order").pack(anchor="w")
        for idx, color in enumerate(VECTOR_DATASET_COLORS, 1):
            self._add_color_chip(dataset_frame, f"Input #{idx}", color)

    def _add_color_chip(self, parent: Frame, label_text: str, color: str) -> None:
        row = Frame(parent)
        row.pack(anchor="w", pady=1, fill=BOTH)
        swatch = Canvas(row, width=26, height=14, highlightthickness=1, highlightbackground="#555555")
        swatch.create_rectangle(0, 0, 26, 14, outline="#222222", fill=color)
        swatch.pack(side=LEFT, padx=(0, 6))
        Label(row, text=label_text).pack(side=LEFT)

    def _add_labeled_entry(self, parent: Frame, label: str, variable: StringVar, width: int) -> Entry:
        frame = Frame(parent)
        frame.pack(fill=BOTH, pady=2)
        Label(frame, text=label, width=24, anchor="w").pack(side=LEFT)
        entry = Entry(frame, textvariable=variable, width=width)
        entry.pack(side=LEFT, fill=BOTH, expand=True)
        return entry

    def _add_file_picker(
        self,
        parent: Frame,
        label_text: str,
        variable: StringVar,
        file_types=None,
        save_dialog: bool = False,
        pad_top: bool = True,
        enable_browse: bool = True,
        open_handler=None,
        dialog_title: str | None = None,
        collect_controls: bool = False,
        *,
        base_dir: str | None = None,
        require_output_dir: bool = False,
    ) -> dict[str, object] | None:
        if open_handler is None:
            # Default view handler
            open_handler = lambda p=variable, l=label_text: self._open_table_window(p.get(), l.strip(":"))

        top = 6 if pad_top else 0
        Label(parent, text=label_text).pack(anchor="w", pady=(top, 0))
        row = Frame(parent)
        row.pack(fill=BOTH, pady=2)
        entry = Entry(row, textvariable=variable)
        entry.pack(side=LEFT, fill=BOTH, expand=True)
        controls = {"entry": entry} if collect_controls else None
        open_btn = Button(
            row,
            text="Open",
            command=lambda var=variable: self._open_path_default(var.get()),
        )
        open_btn.pack(side=RIGHT, padx=5)
        if collect_controls:
            controls["open"] = open_btn
        view_btn = None
        if open_handler:
            view_btn = Button(row, text="View", command=open_handler)
            view_btn.pack(side=RIGHT, padx=5)
            if collect_controls:
                controls["view"] = view_btn
        if enable_browse:
            target_base = base_dir or ("output" if require_output_dir else "genome")
            if save_dialog:
                browse_btn = Button(row, text="Save As", command=lambda: self.save_file_dialog(variable))
            else:
                browse_btn = Button(
                    row,
                    text="Browse",
                    command=lambda: self.browse_file(
                        variable, file_types=file_types, dialog_title=dialog_title, base_dir=target_base
                    ),
                )
            browse_btn.pack(side=RIGHT, padx=5)
            if collect_controls:
                controls["browse"] = browse_btn

        def update_state(*_ignored):
            path_text = variable.get().strip()
            allowed = False
            if path_text:
                file_path = Path(path_text)
                if file_path.is_file():
                    allowed = True
                elif not file_path.is_absolute():
                    # Fallback: check relative to script directory
                    script_root = Path(__file__).parent.resolve()
                    if (script_root / file_path).is_file():
                        allowed = True
            state = "normal" if allowed else "disabled"
            try:
                open_btn.configure(state=state)
            except Exception:
                pass
            if view_btn is not None:
                try:
                    view_btn.configure(state=state)
                except Exception:
                    pass

        variable.trace_add("write", update_state)
        update_state()
        self._register_file_picker_validator(update_state)
        return controls

    def _register_file_picker_validator(self, callback: Callable[[], None]) -> None:
        self._file_picker_validators.append(callback)

    def _run_file_picker_validators(self) -> None:
        for validator in list(self._file_picker_validators):
            try:
                validator()
            except Exception:
                continue

    def _open_table_window(self, path: str, title: str) -> None:
        path = path.strip()
        if not path:
            messagebox.showerror("Missing file", f"Please select a file before opening {title}.")
            return
        file_path = Path(path)
        if not file_path.is_absolute() and not file_path.exists():
            root = Path(__file__).parent.resolve()
            if (root / file_path).exists():
                file_path = root / file_path

        if not file_path.is_file():
            messagebox.showerror("Invalid file", f"No file found at '{file_path}'.")
            return
        try:
            delimiter = self._detect_table_delimiter(file_path)
            with file_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                rows = list(reader)
        except Exception as exc:
            messagebox.showerror("Error reading file", str(exc))
            return
        if not rows:
            messagebox.showinfo("Empty file", f"'{file_path.name}' contains no rows to display.")
            return
        headers = rows[0]
        data_rows = rows[1:]
        top = Toplevel(self.root)
        top.title(f"{title} - {file_path.name}")
        top.geometry("1100x700")

        # Top control frame
        top_controls = Frame(top)
        top_controls.pack(fill=X, padx=10, pady=(10, 0))

        # Search box
        search_frame = Frame(top_controls)
        search_frame.pack(side=LEFT)
        Label(search_frame, text="Search:").pack(side=LEFT)
        search_var = StringVar()
        search_entry = Entry(search_frame, textvariable=search_var, width=30)
        search_entry.pack(side=LEFT, padx=(4, 8))

        # Column visibility toggles
        toggle_lf = LabelFrame(top_controls, text="Show/Hide Columns")
        toggle_lf.pack(side=LEFT, fill=X, expand=True, padx=(10, 0))
        
        toggle_canvas = Canvas(toggle_lf, height=30)
        toggle_hsb = Scrollbar(toggle_lf, orient="horizontal", command=toggle_canvas.xview)
        toggle_inner = Frame(toggle_canvas)
        
        toggle_canvas.create_window((0, 0), window=toggle_inner, anchor="nw")
        toggle_canvas.configure(xscrollcommand=toggle_hsb.set)
        
        toggle_canvas.pack(side=TOP, fill=X, expand=True)
        toggle_hsb.pack(side=BOTTOM, fill=X)

        col_vars = {}
        
        table_frame = Frame(top)
        table_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        columns = ["#"] + headers
        tree = ttk.Treeview(table_frame, columns=columns, show="headings")

        def update_columns():
            display = ["#"] + [c for c in headers if col_vars[c].get()]
            tree["displaycolumns"] = display

        for col in headers:
            v = BooleanVar(value=True)
            col_vars[col] = v
            cb = Checkbutton(toggle_inner, text=col, variable=v, command=update_columns)
            cb.pack(side=LEFT, padx=5)
            
        def _on_toggle_configure(event):
            toggle_canvas.configure(scrollregion=toggle_canvas.bbox("all"))
        toggle_inner.bind("<Configure>", _on_toggle_configure)

        display_rows = list(data_rows)
        sort_states = {col: False for col in columns}
        column_widths = {}
        if data_rows:
            column_widths["#"] = max(60, len(str(len(data_rows))) * 9 + 20)
            for idx, col in enumerate(headers):
                values = [str(col)] + [str(row[idx]) for row in data_rows]
                max_len = max(len(value) for value in values)
                column_widths[col] = min(max(90, max_len * 9 + 28), 600)
        else:
            column_widths["#"] = 60

        def sort_column(column: str) -> None:
            nonlocal display_rows
            if column == "#":
                populate(display_rows)
                col_values = [str(idx) for idx, _ in enumerate(display_rows, 1)]
            else:
                column_index = headers.index(column)
                reverse = sort_states.get(column, False)

                def sort_key(row):
                    value = row[column_index]
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return str(value)

                display_rows = sorted(display_rows, key=sort_key, reverse=reverse)
                sort_states[column] = not reverse
                populate(display_rows)
                col_values = [row[column_index] for row in display_rows]
            
            items = tree.get_children()
            if items:
                tree.selection_set(items)
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append("\n".join(str(v) for v in col_values))
                self.root.update()
            except Exception:
                pass

        for col in columns:
            tree.heading(col, text=col, command=lambda c=col: sort_column(c))
            tree.column(col, width=column_widths.get(col, 120), anchor="w", stretch=True)
        vsb = Scrollbar(table_frame, orient="vertical", command=tree.yview)
        hsb = Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        status_label = Label(top, text="", fg="#666666")
        status_label.pack(side=BOTTOM, fill=X, padx=10, pady=(0, 5), anchor="w")

        def populate(filtered_rows: list[list[str]]) -> None:
            tree.delete(*tree.get_children())
            LIMIT = 5000
            for idx, row in enumerate(filtered_rows[:LIMIT], 1):
                values = [idx] + [str(cell) for cell in row]
                tree.insert("", END, values=values)
            
            if len(filtered_rows) > LIMIT:
                status_label.config(text=f"Showing first {LIMIT} of {len(filtered_rows)} rows. Use Search to filter findings.")
            else:
                status_label.config(text=f"Showing {len(filtered_rows)} rows.")
            current_selection = tree.selection()
            if current_selection:
                tree.selection_remove(current_selection)

        def apply_search(*_args) -> None:
            nonlocal display_rows
            query = search_var.get().lower().strip()
            if not query:
                display_rows = list(data_rows)
            else:
                display_rows = [
                    row
                    for row in data_rows
                    if any(query in str(cell).lower() for cell in row)
                ]
            populate(display_rows)

        def copy_selected() -> None:
            items = tree.selection()
            if not items:
                return
            lines = []
            for item in items:
                values = tree.item(item)["values"]
                lines.append("\t".join(str(value) for value in values))
            self.root.clipboard_clear()
            self.root.clipboard_append("\n".join(lines))
            self.root.update()

        populate(display_rows)
        search_entry.bind("<Return>", apply_search)
        Button(search_frame, text="Apply", command=apply_search).pack(side=LEFT, padx=(0, 4))
        Button(search_frame, text="Copy Selected", command=copy_selected).pack(side=LEFT)

        # Selection and Clipboard Extensions
        last_cell = {"val": None}

        def on_tree_click(event):
            region = tree.identify_region(event.x, event.y)
            if region == "cell":
                row_id = tree.identify_row(event.y)
                col_id = tree.identify_column(event.x)
                try:
                    col_idx = int(col_id.replace("#", "")) - 1
                    item_data = tree.item(row_id)
                    values = item_data.get("values", [])
                    if 0 <= col_idx < len(values):
                        val = values[col_idx]
                        last_cell["val"] = val
                        status_label.config(text=f"Cell: {val}")
                except (ValueError, IndexError):
                    pass

        def copy_cell():
            if last_cell["val"] is not None:
                self.root.clipboard_clear()
                self.root.clipboard_append(str(last_cell["val"]))
                self.root.update()
                status_label.config(text=f"Copied cell: {last_cell['val']}")

        def copy_all():
            lines = ["\t".join(headers)]
            for row in display_rows:
                lines.append("\t".join(str(c) for c in row))
            self.root.clipboard_clear()
            self.root.clipboard_append("\n".join(lines))
            self.root.update()
            status_label.config(text=f"Copied entire table ({len(display_rows)} rows) to clipboard.")

        def show_context_menu(event):
            # Update cell under cursor before showing
            row_id = tree.identify_row(event.y)
            col_id = tree.identify_column(event.x)
            if row_id and col_id:
                try:
                    col_idx = int(col_id.replace("#", "")) - 1
                    values = tree.item(row_id).get("values", [])
                    if 0 <= col_idx < len(values):
                        last_cell["val"] = values[col_idx]
                except: pass

            menu = Menu(top, tearoff=0)
            if last_cell["val"] is not None:
                menu.add_command(label=f"Copy Cell '{last_cell['val']}'", command=copy_cell)
            
            sel = tree.selection()
            if sel:
                menu.add_command(label=f"Copy Selected Rows ({len(sel)})", command=copy_selected)
            
            menu.add_separator()
            menu.add_command(label="Copy Whole Table", command=copy_all)
            menu.post(event.x_root, event.y_root)

        def ctrl_c_handler(event=None):
            if len(tree.selection()) > 1:
                copy_selected()
            elif last_cell["val"] is not None:
                copy_cell()
            else:
                copy_selected()

        tree.bind("<Button-1>", on_tree_click, add="+")
        tree.bind("<Control-c>", ctrl_c_handler)
        tree.bind("<Button-3>", show_context_menu) # Win/Linux
        tree.bind("<Button-2>", show_context_menu) # macOS

    def _open_path_default(self, path: str) -> None:
        path = (path or "").strip()
        if not path:
            messagebox.showerror("Missing path", "Please select a file path first.")
            return
        target = Path(path)
        if not target.is_absolute() and not target.exists():
            root = Path(__file__).parent.resolve()
            if (root / target).exists():
                target = root / target

        if not target.exists():
            messagebox.showerror("Missing file", f"No file found at '{target}'.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            messagebox.showerror("Open file", f"Could not open '{target}': {exc}")

    def _open_output_dir(self) -> None:
        target = self._get_output_dir()
        if not target.is_absolute():
            target = ROOT_DIR / target
        target.mkdir(parents=True, exist_ok=True)
        self._open_path_default(str(target))

    def choose_color(self, target_var: StringVar, button: Button | None = None) -> None:
        current = target_var.get() or "#ff0000"
        _, hex_color = colorchooser.askcolor(initialcolor=current, title="Select Color")
        if hex_color:
            target_var.set(hex_color)
            if button:
                try:
                    button.configure(bg=hex_color)
                except:
                    pass

    @staticmethod
    def _detect_table_delimiter(path: Path) -> str:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if "\t" in line:
                    return "\t"
                if "," in line:
                    return ","
        return "\t"

    def _update_reactive_oligo(self, _event=None) -> None:
        if not self.motif_widget:
            return
        text = self.motif_widget.get("1.0", END).strip()
        if not text:
            self.oligonucleotide_demo_var.set("[No Motif Entered]")
            return
        
        first_line = text.split("\n")[0].split(":")[0].strip()
        if not first_line:
            self.oligonucleotide_demo_var.set("[No Motif Entered]")
            return
            
        iupac_map = {
            "G": "G", "A": "A", "T": "T", "C": "C",
            "R": "A", "Y": "C", "M": "A", "K": "G",
            "S": "G", "W": "A", "H": "A", "B": "C",
            "V": "G", "D": "G", "N": "A"
        }
        
        result = ""
        i = 0
        while i < len(first_line):
            c = first_line[i].upper()
            if c in iupac_map:
                base = iupac_map[c]
                i += 1
                if i < len(first_line) and first_line[i] == "{":
                    end = first_line.find("}", i)
                    if end != -1:
                        range_str = first_line[i+1:end]
                        parts = range_str.split(",")
                        count = int(parts[0]) if parts[0].isdigit() else 1
                        result += base * count
                        i = end + 1
                    else:
                        result += base
                else:
                    result += base
            else:
                i += 1
        
        if result:
            self.oligonucleotide_demo_var.set(result)
        else:
            self.oligonucleotide_demo_var.set("[Invalid Motif]")

    # ── Self-complementarity constraint panel helpers ──────────────────────

    def _sc_toggle_mode(self) -> None:
        """Show/hide whole-motif or manual-region sub-frames."""
        mode = getattr(self, "_sc_mode_var", None)
        if mode is None:
            return
        whole = getattr(self, "_sc_whole_frame", None)
        manual = getattr(self, "_sc_manual_frame", None)
        if mode.get() == "whole":
            if manual is not None:
                manual.pack_forget()
            if whole is not None:
                whole.pack(fill=X, pady=1)
        else:
            if whole is not None:
                whole.pack_forget()
            if manual is not None:
                manual.pack(fill=X, pady=1)

    def _sc_rule_label(self, rule: dict) -> str:
        """Return a compact human-readable label for the rules listbox."""
        pattern = rule.get("pattern", "?")
        excl = rule.get("excluded_motifs", [])
        excl_str = f" | excl={','.join(excl)}" if excl else ""
        if rule.get("whole_motif"):
            lo = rule.get("min_comp_len", 1)
            hi = rule.get("max_comp_len", "") or "*"
            mm_lo = rule.get("min_mismatches", 0)
            mm_hi = rule.get("max_mismatches", 0)
            return f"{pattern} | whole | pairs={lo}–{hi} | mismatches={mm_lo}–{mm_hi}{excl_str}"
        else:
            r1s = rule.get("r1_start", 0)
            r1l = rule.get("r1_len", 0)
            r2s = rule.get("r2_start", 0)
            r2l = rule.get("r2_len", 0)
            mm_lo = rule.get("min_mismatches", 0)
            mm_hi = rule.get("max_mismatches", 0)
            return f"{pattern} | sec1={r1s}, pairs={r1l} | sec2={r2s}, pairs={r2l} | mm={mm_lo}–{mm_hi}{excl_str}"

    def _sc_add_rule(self) -> None:
        """Validate form fields and add a new rule to the listbox."""
        pattern = getattr(self, "_sc_pattern_var", None)
        if pattern is None:
            return
        pat = pattern.get().strip().upper()
        if not pat:
            messagebox.showwarning("Missing Pattern", "Please enter a motif pattern (IUPAC).")
            return
        mode = self._sc_mode_var.get()
        try:
            mm_min = int(self._sc_mm_min_var.get() or "0")
            mm_max = int(self._sc_mm_max_var.get() or "0")
            if mm_min < 0 or mm_max < 0:
                raise ValueError("negative")
            if mm_min > mm_max:
                raise ValueError("min > max")
        except ValueError as exc:
            messagebox.showerror("Invalid Mismatches", f"Mismatch values must be non-negative integers with min ≤ max.\n{exc}")
            return

        rule: dict = {
            "pattern": pat,
            "min_mismatches": mm_min,
            "max_mismatches": mm_max,
        }
        if mode == "whole":
            try:
                lo = int(self._sc_len_min_var.get() or "1")
                hi_raw = self._sc_len_max_var.get().strip()
                hi = int(hi_raw) if hi_raw else None
                if lo < 1:
                    raise ValueError("min length < 1")
                if hi is not None and hi < lo:
                    raise ValueError("max length < min length")
            except ValueError as exc:
                messagebox.showerror("Invalid Length", f"Complementary length values must be positive integers.\n{exc}")
                return
            rule["whole_motif"] = True
            rule["min_comp_len"] = lo
            rule["max_comp_len"] = hi
        else:
            try:
                r1s = int(self._sc_r1_start_var.get() or "0")
                r1l = int(self._sc_r1_len_var.get() or "1")
                r2s = int(self._sc_r2_start_var.get() or "0")
                r2l = int(self._sc_r2_len_var.get() or "1")
                if r1l < 1 or r2l < 1:
                    raise ValueError("length must be >= 1")
            except ValueError as exc:
                messagebox.showerror("Invalid Regions", f"Region start/length values must be integers.\n{exc}")
                return
            rule["whole_motif"] = False
            rule["r1_start"] = r1s
            rule["r1_len"] = r1l
            rule["r2_start"] = r2s
            rule["r2_len"] = r2l

        # Excluded motifs
        excl_raw = self._sc_excl_var.get().strip().upper()
        rule["excluded_motifs"] = [m.strip() for m in excl_raw.split(",") if m.strip()]

        self._self_comp_rules.append(rule)
        lb = getattr(self, "_sc_rules_listbox", None)
        if lb is not None:
            lb.insert(END, self._sc_rule_label(rule))

    def _sc_remove_rule(self) -> None:
        """Remove the selected rule from the listbox."""
        lb = getattr(self, "_sc_rules_listbox", None)
        if lb is None:
            return
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        lb.delete(idx)
        if 0 <= idx < len(self._self_comp_rules):
            self._self_comp_rules.pop(idx)

    def _sc_rules_to_cli_args(self) -> list[str]:
        """Serialize the self-comp rules list to a list of CLI spec strings."""
        specs: list[str] = []
        for rule in self._self_comp_rules:
            pat = rule.get("pattern", "")
            if not pat:
                continue
            mm_lo = rule.get("min_mismatches", 0)
            mm_hi = rule.get("max_mismatches", 0)
            excl = ",".join(rule.get("excluded_motifs", []))
            excl_suffix = f":{excl}" if excl else ""
            
            if rule.get("whole_motif"):
                lo = rule.get("min_comp_len", 1)
                hi = rule.get("max_comp_len", None)
                hi_str = str(hi) if hi is not None else "*"
                specs.append(f"{pat}:whole:{lo},{hi_str}:{mm_lo},{mm_hi}{excl_suffix}")
            else:
                r1s = rule.get("r1_start", 0)
                r1l = rule.get("r1_len", 0)
                r2s = rule.get("r2_start", 0)
                r2l = rule.get("r2_len", 0)
                specs.append(f"{pat}:{r1s},{r1l}:{r2s},{r2l}:{mm_lo},{mm_hi}{excl_suffix}")
        return specs

    def _sc_cli_spec_to_rule(self, spec: str) -> dict | None:
        """Convert a CLI self-comp spec string back to a rule dict for display."""
        try:
            parts = spec.split(":")
            if len(parts) < 3:
                return None
            pattern = parts[0].strip().upper()
            mode_token = parts[1].strip().lower()
            
            mm_lo, mm_hi = 0, 0
            excluded_motifs = []
            
            def parse_excl(p_idx):
                if len(parts) > p_idx:
                    p = parts[p_idx].strip()
                    if p:
                        return [m.strip().upper() for m in p.split(",") if m.strip()]
                return []

            if mode_token == "whole":
                len_parts = parts[2].split(",")
                lo = int(len_parts[0])
                hi_str = len_parts[1].strip() if len(len_parts) > 1 else ""
                hi = None if hi_str in ("", "*", "any") else int(hi_str)
                
                if len(parts) >= 4:
                    p3 = parts[3].strip()
                    if p3 and (p3[0].isdigit() or (len(p3) > 1 and p3[0] == "-" and p3[1].isdigit())):
                        mm_parts = p3.split(",")
                        mm_lo = int(mm_parts[0])
                        mm_hi = int(mm_parts[1]) if len(mm_parts) > 1 else mm_lo
                        excluded_motifs = parse_excl(4)
                    else:
                        excluded_motifs = [m.strip().upper() for m in p3.split(",") if m.strip()]

                return {
                    "pattern": pattern, "whole_motif": True,
                    "min_comp_len": lo, "max_comp_len": hi,
                    "min_mismatches": mm_lo, "max_mismatches": mm_hi,
                    "excluded_motifs": excluded_motifs
                }
            else:
                # explicit section
                s1_parts = parts[1].split(",")
                r1s, r1l = int(s1_parts[0]), int(s1_parts[1])
                s2_parts = parts[2].split(",")
                r2s, r2l = int(s2_parts[0]), int(s2_parts[1])
                
                if len(parts) >= 4:
                    p3 = parts[3].strip()
                    if p3 and (p3[0].isdigit() or (len(p3) > 1 and p3[0] == "-" and p3[1].isdigit())):
                        mm_parts = p3.split(",")
                        mm_lo = int(mm_parts[0])
                        mm_hi = int(mm_parts[1]) if len(mm_parts) > 1 else mm_lo
                        excluded_motifs = parse_excl(4)
                    else:
                        excluded_motifs = [m.strip().upper() for m in p3.split(",") if m.strip()]

                return {
                    "pattern": pattern, "whole_motif": False,
                    "r1_start": r1s, "r1_len": r1l,
                    "r2_start": r2s, "r2_len": r2l,
                    "min_mismatches": mm_lo, "max_mismatches": mm_hi,
                    "excluded_motifs": excluded_motifs
                }
        except Exception:
            return None

    def _add_multiline_section(
        self, parent: Frame, title: str, sample_text: str | None = None, help_text: str | None = None
    ):
        frame = Frame(parent)
        frame.pack(fill=BOTH, pady=3)
        title_row = Frame(frame)
        title_row.pack(anchor="w", fill=BOTH)
        Label(title_row, text=title).pack(side=LEFT, anchor="w")
        if help_text:
            help_label = Label(title_row, text="?", relief="solid", borderwidth=1, width=2)
            help_label.pack(side=LEFT, padx=(6, 0))
            ToolTip(help_label, help_text)
        text_widget = scrolledtext.ScrolledText(frame, height=3, wrap="word")
        text_widget.pack(fill=BOTH, expand=True)
        if sample_text:
            text_widget.insert("1.0", sample_text)
        return text_widget

    def browse_file(
        self,
        target_var: StringVar,
        file_types=None,
        dialog_title: str | None = None,
        base_dir: str | None = None,
    ) -> None:
        file_types = file_types or [
            ("FASTA Files", "*.fna *.fa *.fasta"),
            ("All Files", "*.*"),
        ]
        title = dialog_title or "Select a file"
        initial_dir: str | None = None
        current_value = target_var.get().strip()
        if current_value:
            current_path = Path(current_value)
            if current_path.exists():
                try:
                    initial_dir = str(current_path.parent)
                except Exception:
                    initial_dir = None
        if not initial_dir:
            if base_dir == "output":
                output_dir = self._get_output_dir()
                initial_dir = str(output_dir if output_dir.is_absolute() else ROOT_DIR / output_dir)
            else:
                data_dir = self._get_genome_data_dir()
                initial_dir = str(data_dir if data_dir.is_absolute() else ROOT_DIR / data_dir)
        filename = filedialog.askopenfilename(
            title=title,
            filetypes=file_types,
            initialdir=initial_dir,
        )
        if filename:
            try:
                rel_path = Path(filename).relative_to(ROOT_DIR)
                target_var.set(str(rel_path.as_posix()))
            except ValueError:
                target_var.set(filename)

    def save_file_dialog(self, target_var: StringVar) -> None:
        output_dir = self._get_output_dir()
        initial_dir = str(output_dir if output_dir.is_absolute() else ROOT_DIR / output_dir)
        filename = filedialog.asksaveasfilename(
            title="Save SVG output",
            defaultextension=".svg",
            filetypes=[("SVG Files", "*.svg"), ("All Files", "*.*")],
            initialdir=initial_dir,
        )
        if filename:
            try:
                rel_path = Path(filename).relative_to(ROOT_DIR)
                target_var.set(str(rel_path.as_posix()))
            except ValueError:
                target_var.set(filename)
            self._refresh_output_paths()

    def append_log(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert(END, message + "\n")
        self.log_box.see(END)
        self.log_box.configure(state="disabled")

    def _sync_process_to_viz(self) -> None:
        if self.viz_hits_var.get().strip():
            return
        candidate = self.processed_hits_var.get().strip() or self.process_hits_var.get().strip()
        if candidate:
            self.viz_hits_var.set(candidate)


    def run_combined_scan(self) -> None:
        fasta_path = self.fasta_var.get().strip()
        if not fasta_path:
            messagebox.showerror("Missing input", "Please select a FASTA (.fna) file.")
            return
        fasta_path_obj = Path(fasta_path)
        if not fasta_path_obj.is_file():
            messagebox.showerror("Invalid file", f"No file found at '{fasta_path}'.")
            return
            
        script_path = SCRIPT_DIR / "combined_sequence_search.py"
        if not script_path.is_file():
            messagebox.showerror("Missing script", f"Could not find {script_path.name} next to this GUI.")
            return
            
        try:
            window = int(self.window_var.get())
            step = int(self.step_var.get())
            workers = int(self.workers_var.get())
            comb_fwd = int(self.combined_comb_fwd_var.get().strip() or "0")
            comb_rev = int(self.combined_comb_rev_var.get().strip() or "0")
            comb_overlap = int(self.combined_comb_overlap_var.get().strip() or "0")
        except ValueError as exc:
            messagebox.showerror("Invalid numeric value", str(exc))
            return
            
        output_prefix = "combined_search_hits"
        output_name = self.output_name_var.get().strip() or "default"
        
        # Get NT params
        nt_motifs = self._get_advanced_lines(self.combined_nt_motif_widget, "_saved_comb_nt_motif") if self.combined_nt_motif_widget else []
        exclude_nt_motifs = self._get_advanced_lines(self.combined_exclude_nt_widget, "_saved_comb_excl_nt") if self.combined_exclude_nt_widget else []
        base_limits = self._get_advanced_lines(self.combined_base_widget, "_saved_comb_base") if self.combined_base_widget else []
        repeat_limits = self._get_advanced_lines(self.combined_repeat_widget, "_saved_comb_rep") if self.combined_repeat_widget else []
        self_comp_motifs = self._get_advanced_lines(self.combined_self_comp_widget, "_saved_comb_sc") if self.combined_self_comp_widget else []
        fwd_nt_motifs = self._get_advanced_lines(self.combined_forward_motif_widget, "_saved_comb_fwd") if self.combined_forward_motif_widget else []
        rev_nt_motifs = self._get_advanced_lines(self.combined_reverse_motif_widget, "_saved_comb_rev") if self.combined_reverse_motif_widget else []
        
        # Get Peptide params
        pep_motifs = self._get_advanced_lines(self.combined_pep_motif_widget, "_saved_comb_pep_motif") if self.combined_pep_motif_widget else []
        exclude_pep_motifs = self._get_advanced_lines(self.combined_exclude_pep_widget, "_saved_comb_excl_pep") if self.combined_exclude_pep_widget else []
        pep_amino_content = self._get_advanced_lines(self.combined_amino_content_widget, "_saved_comb_amino") if self.combined_amino_content_widget else []
        pep_repeats = self._get_advanced_lines(self.combined_peptide_repeat_widget, "_saved_comb_pep_rep") if self.combined_peptide_repeat_widget else []
        pep_mismatches = self.combined_pep_mismatches_var.get().strip() or "0"
        
        frames = [frame for frame, var in self.combined_frame_vars.items() if var.get()]
        if not frames:
            messagebox.showerror("Invalid input", "Please select at least one reading frame.")
            return

        cmd = [
            sys.executable,
            str(script_path),
            fasta_path,
            "--window", str(window),
            "--step", str(step),
            "--output-prefix", output_prefix,
            "--output-name", output_name,
            "--workers", str(workers),
        ]
        
        for m in nt_motifs: cmd.extend(["--nt-motif", m])
        for m in exclude_nt_motifs: cmd.extend(["--exclude-nt-motif", m])
        for b in base_limits: cmd.extend(["--base-content", b])
        for r in repeat_limits: cmd.extend(["--max-repeat", r])
        for sc in self_comp_motifs: cmd.extend(["--motif-self-comp", sc])
        
        # Combined script doesn't support fwd/rev-only natively without prefixes if it uses nt_sequence_search's parsing.
        # Wait, nt_sequence_search uses `+:` and `-:` prefixes for forward-only and reverse-only motifs!
        for m in fwd_nt_motifs: cmd.extend(["--nt-motif", f"+:{m}"])
        for m in rev_nt_motifs: cmd.extend(["--nt-motif", f"-:{m}"])
        
        for m in pep_motifs: cmd.extend(["--pep-motif", m])
        for m in exclude_pep_motifs: cmd.extend(["--exclude-pep-motif", m])
        for a in pep_amino_content: cmd.extend(["--amino-content", a])
        for r in pep_repeats: cmd.extend(["--pep-repeat", r])
        for f in frames: cmd.extend(["--frame", f])
        
        nt_sub_window = self.combined_nt_sub_window_var.get().strip()
        nt_sub_offset = self.combined_nt_sub_offset_var.get().strip()
        pep_sub_window = self.combined_pep_sub_window_var.get().strip()
        pep_sub_offset = self.combined_pep_sub_offset_var.get().strip()
        
        if nt_sub_window: cmd.extend(["--nt-sub-window", nt_sub_window])
        if nt_sub_offset: cmd.extend(["--nt-sub-offset", nt_sub_offset])
        if pep_sub_window: cmd.extend(["--pep-sub-window", pep_sub_window])
        if pep_sub_offset: cmd.extend(["--pep-sub-offset", pep_sub_offset])
        
        cmd.extend(["--pep-mismatches", pep_mismatches])
        
        if comb_fwd > 0 and comb_rev > 0:
            cmd.extend([
                "--strand", "combined",
                "--combined-forward-len", str(comb_fwd),
                "--combined-reverse-len", str(comb_rev),
                "--combined-overlap", str(comb_overlap)
            ])
        
        # We can add strands based on global checkboxes if needed, or default to all
        if self.forward_strand_var.get(): cmd.extend(["--strand", "forward"])
        if self.reverse_strand_var.get(): cmd.extend(["--strand", "reverse"])
        
        if self.require_palindrome_var.get():
            cmd.append("--require-palindrome")
            pal_min = self.palindrome_min_len_var.get().strip()
            if pal_min:
                cmd.extend(["--palindrome-min-len", pal_min])
                
        if self.non_overlapping_var.get():
            cmd.append("--non-overlapping")

        # Include sequence/region filters from main tab
        seq_filters = self._get_advanced_lines(self.sequence_filter_widget, "_saved_seq_filter")
        reg_filters = self._get_advanced_lines(self.region_filter_widget, "_saved_reg_filter")
        for sf in seq_filters: cmd.extend(["--sequence-id", sf])
        for rf in reg_filters: cmd.extend(["--region", rf])
        
        classes = self._get_sequence_classes()
        if classes:
            for c in classes:
                cmd.extend(["--sequence-class", c])

        self._run_subprocess(cmd, "Combined Search Scanner")

    def save_search_profile(self, mode: str) -> None:
        file_path = filedialog.asksaveasfilename(
            title=f"Save {mode.capitalize()} Profile",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not file_path:
            return

        data = {
            "mode": mode,
            "window": self.window_var.get(),
            "step": self.step_var.get(),
            "workers": self.workers_var.get(),
            "output_name": self.output_name_var.get(),
            "forward_strand": self.forward_strand_var.get(),
            "reverse_strand": self.reverse_strand_var.get(),
            "sequence_filter": self._get_advanced_lines(self.sequence_filter_widget, "_saved_seq_filter"),
            "region_filter": self._get_advanced_lines(self.region_filter_widget, "_saved_reg_filter")
        }

        if mode in ("nucleotide", "combined"):
            data.update({
                "require_palindrome": self.require_palindrome_var.get(),
                "palindrome_min_len": self.palindrome_min_len_var.get(),
                "non_overlapping": self.non_overlapping_var.get(),
                "nt_motifs": self._get_advanced_lines(self.motif_widget, "_saved_motif_text"),
                "exclude_nt_motifs": self._get_advanced_lines(self.exclude_motif_widget, "_saved_exclude_text"),
                "base_content": self._get_advanced_lines(self.base_widget, "_saved_base_text"),
                "repeat_limits": self._get_advanced_lines(self.repeat_widget, "_saved_repeat_text"),
                "forward_only_motifs": self._get_advanced_lines(self.forward_only_motif_widget, "_saved_forward_only_motif_text"),
                "reverse_only_motifs": self._get_advanced_lines(self.reverse_only_motif_widget, "_saved_reverse_only_motif_text"),
                "self_comp": self._get_advanced_lines(self.sc_widget, "_saved_sc_text"),
            })

        if mode in ("peptide", "combined"):
            data.update({
                "pep_motifs": self._get_advanced_lines(self.pep_motif_widget, "_saved_pep_motif"),
                "exclude_pep_motifs": self._get_advanced_lines(self.pep_exclude_motif_widget, "_saved_pep_exclude"),
                "amino_content": self._get_advanced_lines(self.pep_amino_widget, "_saved_pep_amino"),
                "pep_repeats": self._get_advanced_lines(self.pep_repeat_widget, "_saved_pep_repeat"),
                "frames": self._get_advanced_lines(self.pep_frame_widget, "_saved_pep_frame"),
                "pep_mismatches": self.peptide_mismatches_var.get(),
            })

        if mode == "combined":
            data.update({
                "combined_strand": self.combined_strand_var.get(),
                "combined_forward_len": self.combined_forward_var.get(),
                "combined_reverse_len": self.combined_reverse_var.get(),
                "combined_overlap": self.combined_overlap_var.get(),
                "nt_sub_window": self.combined_nt_sub_window_var.get(),
                "nt_sub_offset": self.combined_nt_sub_offset_var.get(),
                "pep_sub_window": self.combined_pep_sub_window_var.get(),
                "pep_sub_offset": self.combined_pep_sub_offset_var.get(),
            })

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                import json
                json.dump(data, f, indent=4)
            self.append_log(f"Saved {mode} search profile to {Path(file_path).name}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save profile:\n{e}")

    def load_search_profile(self, mode: str) -> None:
        file_path = filedialog.askopenfilename(
            title=f"Load {mode.capitalize()} Profile",
            filetypes=[("JSON files", "*.json")],
        )
        if not file_path:
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load profile:\n{e}")
            return

        if data.get("mode") != mode:
            if not messagebox.askyesno(
                "Profile Mismatch", 
                f"This profile was saved from a '{data.get('mode')}' search. Are you sure you want to load it into the '{mode}' tab?"
            ):
                return

        def set_var(var, key, default=""):
            if key in data:
                var.set(str(data[key]))

        set_var(self.window_var, "window")
        set_var(self.step_var, "step")
        set_var(self.workers_var, "workers")
        set_var(self.output_name_var, "output_name")
        if "forward_strand" in data: self.forward_strand_var.set(bool(data["forward_strand"]))
        if "reverse_strand" in data: self.reverse_strand_var.set(bool(data["reverse_strand"]))
        
        if "sequence_filter" in data: self._set_multiline_widget(self.sequence_filter_widget, "_saved_seq_filter", data["sequence_filter"])
        if "region_filter" in data: self._set_multiline_widget(self.region_filter_widget, "_saved_reg_filter", data["region_filter"])

        if mode in ("nucleotide", "combined"):
            if "require_palindrome" in data: self.require_palindrome_var.set(bool(data["require_palindrome"]))
            if "non_overlapping" in data: self.non_overlapping_var.set(bool(data["non_overlapping"]))
            set_var(self.palindrome_min_len_var, "palindrome_min_len")
            
            if "nt_motifs" in data: self._set_multiline_widget(self.motif_widget, "_saved_motif_text", data["nt_motifs"])
            if "exclude_nt_motifs" in data: self._set_multiline_widget(self.exclude_motif_widget, "_saved_exclude_text", data["exclude_nt_motifs"])
            if "base_content" in data: self._set_multiline_widget(self.base_widget, "_saved_base_text", data["base_content"])
            if "repeat_limits" in data: self._set_multiline_widget(self.repeat_widget, "_saved_repeat_text", data["repeat_limits"])
            if "forward_only_motifs" in data: self._set_multiline_widget(self.forward_only_motif_widget, "_saved_forward_only_motif_text", data["forward_only_motifs"])
            if "reverse_only_motifs" in data: self._set_multiline_widget(self.reverse_only_motif_widget, "_saved_reverse_only_motif_text", data["reverse_only_motifs"])
            if "self_comp" in data: self._set_multiline_widget(self.sc_widget, "_saved_sc_text", data["self_comp"])

        if mode in ("peptide", "combined"):
            set_var(self.peptide_mismatches_var, "pep_mismatches")
            if "pep_motifs" in data: self._set_multiline_widget(self.pep_motif_widget, "_saved_pep_motif", data["pep_motifs"])
            if "exclude_pep_motifs" in data: self._set_multiline_widget(self.pep_exclude_motif_widget, "_saved_pep_exclude", data["exclude_pep_motifs"])
            if "amino_content" in data: self._set_multiline_widget(self.pep_amino_widget, "_saved_pep_amino", data["amino_content"])
            if "pep_repeats" in data: self._set_multiline_widget(self.pep_repeat_widget, "_saved_pep_repeat", data["pep_repeats"])
            if "frames" in data: self._set_multiline_widget(self.pep_frame_widget, "_saved_pep_frame", data["frames"])

        if mode == "combined":
            if "combined_strand" in data: self.combined_strand_var.set(bool(data["combined_strand"]))
            set_var(self.combined_forward_var, "combined_forward_len")
            set_var(self.combined_reverse_var, "combined_reverse_len")
            set_var(self.combined_overlap_var, "combined_overlap")
            set_var(self.combined_nt_sub_window_var, "nt_sub_window")
            set_var(self.combined_nt_sub_offset_var, "nt_sub_offset")
            set_var(self.combined_pep_sub_window_var, "pep_sub_window")
            set_var(self.combined_pep_sub_offset_var, "pep_sub_offset")
            
        self.append_log(f"Loaded {mode} search profile from {Path(file_path).name}")

    def run_scan(self) -> None:
        fasta_path = self.fasta_var.get().strip()
        if not fasta_path:
            messagebox.showerror("Missing input", "Please select a FASTA (.fna) file.")
            return
        fasta_path_obj = Path(fasta_path)
        if not fasta_path_obj.is_file():
            messagebox.showerror("Invalid file", f"No file found at '{fasta_path}'.")
            return
        if not SCAN_SCRIPT.is_file():
            messagebox.showerror("Missing script", f"Could not find {SCAN_SCRIPT.name} next to this GUI.")
            return
        try:
            window = int(self.window_var.get())
            step = int(self.step_var.get())
            workers = int(self.workers_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid numeric value", str(exc))
            return
        output_prefix = "nt_sequence_hits"
        motifs = self._get_advanced_lines(self.motif_widget, "_saved_motif_text")
        exclude_motifs = self._get_advanced_lines(self.exclude_motif_widget, "_saved_exclude_text")
        base_limits = self._get_advanced_lines(self.base_widget, "_saved_base_text")
        repeats = self._get_advanced_lines(self.repeat_widget, "_saved_repeat_text")
        forward_only_motifs = self._get_advanced_lines(
            self.forward_motif_widget, "_saved_forward_motif_text"
        )
        reverse_only_motifs = self._get_advanced_lines(
            self.reverse_motif_widget, "_saved_reverse_motif_text"
        )
        self_comp_motifs = self._sc_rules_to_cli_args()
        pal_min_text = self.palindrome_min_len_var.get().strip()
        pal_min_value = None
        if pal_min_text:
            try:
                pal_min_value = int(pal_min_text)
                if pal_min_value < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid palindrome length", "Min palindrome length must be a non-negative integer."
                )
                return

        cmd = [
            sys.executable,
            str(SCAN_SCRIPT),
            fasta_path,
            "--window",
            str(window),
            "--step",
            str(step),
            "--output-prefix",
            output_prefix,
            "--workers",
            str(workers),
        ]
        output_name = self._get_output_name()
        cmd.extend(["--output-name", output_name])
        for motif in motifs:
            cmd.extend(["--motif", motif])
        for motif in exclude_motifs:
            cmd.extend(["--exclude-motif", motif])
        for spec in base_limits:
            cmd.extend(["--base-content", spec])
        for spec in repeats:
            cmd.extend(["--max-repeat", spec])
        for motif in forward_only_motifs:
            cmd.extend(["--motif-forward", motif])
        for motif in reverse_only_motifs:
            cmd.extend(["--motif-reverse", motif])
        for motif in self_comp_motifs:
            cmd.extend(["--motif-self-comp", motif])
        for seq_filter in self._read_text_lines(self.sequence_filter_widget):
            cmd.extend(["--sequence-id", seq_filter])
        for region_filter in self._read_text_lines(self.region_filter_widget):
            cmd.extend(["--region", region_filter])
        if self.non_overlapping_var.get():
            cmd.append("--non-overlapping")
        if self.require_palindrome_var.get():
            cmd.append("--require-palindrome")
        if pal_min_value is not None:
            cmd.extend(["--palindrome-min-len", str(pal_min_value)])
        strands = self._selected_strands()
        if not strands:
            messagebox.showerror(
                "Missing strand selection", "Select at least one strand or combined window to scan."
            )
            return
        for strand in strands:
            cmd.extend(["--strand", strand])
        if "combined" in strands:
            try:
                combined_forward = int(self.combined_forward_var.get())
                combined_reverse = int(self.combined_reverse_var.get())
                combined_overlap = int(self.combined_overlap_var.get())
            except ValueError:
                messagebox.showerror(
                    "Invalid combined inputs", "Combined window lengths and overlap must be integers."
                )
                return
            if combined_forward <= 0 or combined_reverse <= 0:
                messagebox.showerror(
                    "Invalid combined inputs", "Combined forward and reverse lengths must be positive integers."
                )
                return
            max_overlap = min(combined_forward, combined_reverse)
            if combined_overlap < 0 or combined_overlap > max_overlap:
                messagebox.showerror(
                    "Invalid overlap",
                    f"Overlap must be between 0 and {max_overlap} (min of the two lengths).",
                )
                return
            cmd.extend(
                [
                    "--combined-forward-len",
                    str(combined_forward),
                    "--combined-reverse-len",
                    str(combined_reverse),
                    "--combined-overlap",
                    str(combined_overlap),
                ]
            )

        output_dir = self._get_output_dir()
        hits_tsv = output_dir / f"{output_prefix}.tsv"
        self.process_hits_var.set(str(hits_tsv))
        self.append_log("Running: " + " ".join(cmd))
        self.run_button.configure(state="disabled")

        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.run_button, "Scanning nucleotide windows..."),
            daemon=True,
        )
        thread.start()

    def run_nt_analysis(self) -> None:
        hits_path = self.process_hits_var.get().strip()
        if not hits_path:
            messagebox.showerror("Missing input", "Please select an nt_sequence_hits TSV file to analyze.")
            return
        path_obj = Path(hits_path)
        if not path_obj.is_file():
            messagebox.showerror("Invalid file", f"No file found at '{hits_path}'.")
            return
        if not ANALYSIS_SCRIPT.is_file():
            messagebox.showerror(
                "Missing script", f"Could not find {ANALYSIS_SCRIPT.name} next to this GUI."
            )
            return
        try:
            min_oligo = int(self.min_oligo_var.get())
            max_oligo = int(self.max_oligo_var.get())
            g4_window = int(self.g4_window_var.get())
            dna_conc = float(self.dna_conc_var.get())
            salt_conc = float(self.salt_conc_var.get())
            reaction_temp = float(self.temperature_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid numeric value", str(exc))
            return
        cmd = [
            sys.executable,
            str(ANALYSIS_SCRIPT),
            hits_path,
            "--min-oligo",
            str(min_oligo),
            "--max-oligo",
            str(max_oligo),
            "--g4-window",
            str(g4_window),
        ]
        workers = self.process_workers_var.get().strip() or "0"
        cmd.extend(["--workers", workers])
        output_prefix = "nt_sequence_hits_analyzed"
        output_name = self._get_output_name()
        cmd.extend(["--output-prefix", output_prefix, "--output-name", output_name])
        cmd.extend(
            [
                "--dna-conc",
                str(dna_conc),
                "--monovalent-salt",
                str(salt_conc),
                "--reaction-temp",
                str(reaction_temp),
            ]
        )
        try:
            top_oligos = int(self.top_oligos_var.get().strip() or "1")
            if top_oligos <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid value", "Top oligos per region must be a positive integer.")
            return
        cmd.extend(["--top-oligos", str(top_oligos)])
        if self.structure_only_var.get():
            cmd.append("--structure-only")
        filter_args = [
            (self.min_g4hunter_filter_var, "--min-g4hunter", "Min G4Hunter"),
            (self.min_g4boost_filter_var, "--min-g4boost", "Min G4Boost"),
            (self.min_tm_wallace_filter_var, "--min-tm-wallace", "Min TM (Wallace)"),
            (self.min_tm_nearest_filter_var, "--min-tm-nearest", "Min TM (NN)"),
            (self.min_i_motif_filter_var, "--min-i-motif", "Min i-motif score"),
            (self.min_r_loop_filter_var, "--min-r-loop", "Min R-loop score"),
            (self.min_hairpin_filter_var, "--min-hairpin", "Min hairpin score"),
            (self.min_cpg_filter_var, "--min-cpg", "Min CpG score"),
        ]
        for var, flag, label in filter_args:
            value = var.get().strip()
            if not value:
                continue
            try:
                float(value)
            except ValueError:
                messagebox.showerror("Invalid filter", f"{label} must be a numeric value.")
                return
            cmd.extend([flag, value])
        analysis_path = self._get_output_dir() / f"{output_prefix}.tsv"
        self.analyzed_hits_var.set(str(analysis_path))
        self.viz_hits_var.set(str(analysis_path))
        if not self.annotation_input_var.get().strip():
            self.annotation_input_var.set(str(analysis_path))
        summary_path = analysis_path.with_name(analysis_path.stem + "_summary.json")
        if summary_path.exists():
            try:
                summary_path.unlink()
            except Exception:
                pass
        cmd.extend(["--summary-json", str(summary_path)])
        self.analysis_summary_path = str(summary_path)
        self.summary_button.configure(state="disabled")
        self.append_log("Running: " + " ".join(cmd))
        self.analysis_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.analysis_button, "Running G4/Thermo analysis...", self._on_analysis_complete),
            daemon=True,
        )
        thread.start()

    def _on_analysis_complete(self) -> None:
        if self.analysis_summary_path and Path(self.analysis_summary_path).is_file():
            self.summary_button.configure(state="normal")
        else:
            self.summary_button.configure(state="disabled")

    def _show_analysis_summary(self) -> None:
        path = self.analysis_summary_path
        if not path:
            messagebox.showinfo("Summary unavailable", "Run the G4/Thermo analysis first.")
            return
        summary_file = Path(path)
        if not summary_file.is_file():
            messagebox.showerror("Summary unavailable", f"No summary file found at {summary_file}.")
            self.summary_button.configure(state="disabled")
            return
        try:
            with summary_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            messagebox.showerror("Summary error", str(exc))
            return
        fields = data.get("fields", {})
        if not fields:
            messagebox.showinfo("Summary", "No histogram data available yet.")
            return
        top = Toplevel(self.root)
        top.title("G4/Thermo Score Summary")
        top.geometry("540x600")
        container = Frame(top)
        container.pack(fill=BOTH, expand=True)
        canvas = Canvas(container)
        scrollbar = Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=RIGHT, fill="y")
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        inner = Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def _update_scroll(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _update_scroll)
        friendly = {
            "best_oligo_g4hunter": "G4Hunter",
            "best_oligo_g4boost": "G4Boost",
            "best_oligo_tm_wallace": "Tm (Wallace)",
            "best_oligo_tm_nearest_neighbor": "Tm (Nearest Neighbor)",
            "best_oligo_i_motif_score": "i-motif score",
            "best_oligo_r_loop_score": "R-loop score",
            "best_oligo_hairpin_score": "Hairpin score",
            "best_oligo_cpg_score": "CpG score",
        }
        for key, stats in fields.items():
            hist = stats.get("histogram")
            if not hist:
                continue
            frame = Frame(inner)
            frame.pack(fill=BOTH, padx=10, pady=8)
            label = friendly.get(key, key)
            info = (
                f"{label} (n={stats.get('count')}, "
                f"min={stats.get('min'):.2f}, max={stats.get('max'):.2f}, mean={stats.get('mean'):.2f})"
            )
            Label(frame, text=info, anchor="w").pack(fill=BOTH)
            hist_canvas = Canvas(frame, width=500, height=160, bg="#ffffff", highlightthickness=1, highlightbackground="#cccccc")
            hist_canvas.pack(fill=BOTH, expand=True, pady=(4, 0))
            max_count = max((bin_entry.get("count", 0) for bin_entry in hist), default=0) or 1
            width = 420
            height = 100
            x_origin = 40
            y_origin = 125
            bar_width = width / max(len(hist), 1)
            for idx, bin_entry in enumerate(hist):
                count = bin_entry.get("count", 0)
                bar_height = (count / max_count) * height if max_count else 0
                x0 = x_origin + idx * bar_width
                y0 = y_origin - bar_height
                x1 = x0 + bar_width * 0.9
                y1 = y_origin
                hist_canvas.create_rectangle(x0, y0, x1, y1, fill="#2563eb", outline="")
            hist_canvas.create_line(x_origin, y_origin, x_origin + width, y_origin, fill="#555555")
            # Y-axis: count labels
            hist_canvas.create_text(x_origin - 4, y_origin, text="0", anchor="e", font=("Segoe UI", 8), fill="#555555")
            hist_canvas.create_text(x_origin - 4, y_origin - height, text=str(max_count), anchor="e", font=("Segoe UI", 8), fill="#555555")
            hist_canvas.create_text(x_origin - 4, y_origin - height // 2, text=str(max_count // 2), anchor="e", font=("Segoe UI", 8), fill="#555555")
            # X-axis: value range labels
            if hist:
                x_min = hist[0].get("start", 0)
                x_max = hist[-1].get("end", 0)
                hist_canvas.create_text(x_origin, y_origin + 12, text=f"{x_min:.2f}", anchor="w", font=("Segoe UI", 8), fill="#555555")
                hist_canvas.create_text(x_origin + width, y_origin + 12, text=f"{x_max:.2f}", anchor="e", font=("Segoe UI", 8), fill="#555555")
                mid_val = (x_min + x_max) / 2
                hist_canvas.create_text(x_origin + width // 2, y_origin + 12, text=f"{mid_val:.2f}", anchor="center", font=("Segoe UI", 8), fill="#555555")
        if not inner.winfo_children():
            Label(inner, text="No histogram data to display.", anchor="w").pack(fill=BOTH, padx=10, pady=10)

    def run_nt_annotation(self) -> None:
        input_path = self.annotation_input_var.get().strip() or self.analyzed_hits_var.get().strip()
        if not input_path:
            input_path = self.process_hits_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing input", "Select a TSV file to annotate first.")
            return
        path_obj = Path(input_path)
        if not path_obj.is_file():
            messagebox.showerror("Invalid file", f"No file found at '{input_path}'.")
            return
        self.annotation_input_var.set(str(path_obj))
        if not ANNOTATION_SCRIPT.is_file():
            messagebox.showerror(
                "Missing script", f"Could not find {ANNOTATION_SCRIPT.name} next to this GUI."
            )
            return
        cmd = [
            sys.executable,
            str(ANNOTATION_SCRIPT),
            input_path,
        ]
        gbff_path = self.gbff_var.get().strip()
        if gbff_path:
            if not Path(gbff_path).is_file():
                messagebox.showerror("Invalid GBFF file", f"No file found at '{gbff_path}'.")
                return
            cmd.extend(["--gbff", gbff_path])
        gff3_path = self.gff3_var.get().strip()
        if gff3_path:
            if not Path(gff3_path).is_file():
                messagebox.showerror("Invalid GFF3 file", f"No file found at '{gff3_path}'.")
                return
            cmd.extend(["--gff3", gff3_path])
        gff_path = self.gff_var.get().strip()
        if gff_path:
            if not Path(gff_path).is_file():
                messagebox.showerror("Invalid GFF file", f"No file found at '{gff_path}'.")
                return
            cmd.extend(["--gff", gff_path])
        gtf_path = self.gtf_var.get().strip()
        if gtf_path:
            if not Path(gtf_path).is_file():
                messagebox.showerror("Invalid GTF file", f"No file found at '{gtf_path}'.")
                return
            cmd.extend(["--gtf", gtf_path])
        gpff_path = self.gpff_var.get().strip()
        if gpff_path:
            if not Path(gpff_path).is_file():
                messagebox.showerror("Invalid GPFF file", f"No file found at '{gpff_path}'.")
                return
            cmd.extend(["--gpff", gpff_path])
        default_output = self._default_annotated_path(Path(input_path))
        requested_output = Path(self.processed_hits_var.get().strip()) if self.processed_hits_var.get().strip() else default_output
        if requested_output != default_output:
            cmd.extend(["--output", str(requested_output)])
        else:
            self.processed_hits_var.set(str(default_output))
        output_path = requested_output if requested_output != default_output else default_output
        self.processed_hits_var.set(str(output_path))
        self.viz_hits_var.set(str(output_path))
        self.append_log("Running: " + " ".join(cmd))
        self.process_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.process_button, "Running NT annotation..."),
            daemon=True,
        )
        thread.start()

    def run_peptide_search(self) -> None:
        fasta_path = self.peptide_fasta_var.get().strip() or self.fasta_var.get().strip()
        if not fasta_path:
            messagebox.showerror("Missing FASTA", "Select a genome FASTA file for the peptide search.")
            return
        fasta_obj = Path(fasta_path)
        if not fasta_obj.is_file():
            messagebox.showerror("Invalid FASTA", f"No FASTA file found at '{fasta_path}'.")
            return
        if not PEPTIDE_SCRIPT.is_file():
            messagebox.showerror(
                "Missing script", f"Could not find {PEPTIDE_SCRIPT.name} next to this GUI."
            )
            return
        try:
            window = int(self.peptide_window_var.get())
            step = int(self.peptide_step_var.get())
            max_mismatches = int(self.peptide_max_mismatches_var.get().strip() or "0")
            workers = int(self.peptide_workers_var.get().strip() or "0")
            comb_fwd = int(self.peptide_comb_fwd_var.get().strip() or "0")
            comb_rev = int(self.peptide_comb_rev_var.get().strip() or "0")
            comb_overlap = int(self.peptide_comb_overlap_var.get().strip() or "0")
        except ValueError:
            messagebox.showerror(
                "Invalid numeric input", "Window, step, mismatches, and worker counts must be integers."
            )
            return
        if max_mismatches < 0:
            messagebox.showerror("Invalid mismatches", "Maximum mismatches cannot be negative.")
            return
        if workers < 0:
            messagebox.showerror("Invalid worker count", "Worker processes must be zero or a positive integer.")
            return
        frames = [frame for frame, var in self.peptide_frame_vars.items() if var.get()]
        if not frames:
            messagebox.showerror("Missing frames", "Select at least one reading frame to scan.")
            return
        motifs = self._read_lines(self.peptide_motif_widget)
        excludes = self._read_lines(self.peptide_exclude_widget)
        output_prefix = "peptide_coding_hits"
        output_dir = self._get_output_dir()
        hits_path = output_dir / f"{output_prefix}.tsv"
        self.peptide_hits_var.set(str(hits_path))
        cmd = [
            sys.executable,
            str(PEPTIDE_SCRIPT),
            fasta_path,
            "--window",
            str(window),
            "--step",
            str(step),
            "--output-prefix",
            output_prefix,
            "--output-name",
            self._get_output_name(),
        ]
        if max_mismatches:
            cmd.extend(["--max-mismatches", str(max_mismatches)])
        if workers:
            cmd.extend(["--workers", str(workers)])
        for motif in motifs:
            cmd.extend(["--motif", motif])
        for motif in excludes:
            cmd.extend(["--exclude-motif", motif])
        for frame in frames:
            cmd.extend(["--frame", frame])
            
        amino_content = self._read_lines(self.amino_content_widget)
        pep_repeats = self._read_lines(self.peptide_repeat_widget)
        for content in amino_content:
            cmd.extend(["--amino-content", content])
        for repeat in pep_repeats:
            cmd.extend(["--max-repeat", repeat])
            
        if comb_fwd > 0 and comb_rev > 0:
            cmd.extend([
                "--strand", "combined",
                "--combined-forward-len", str(comb_fwd),
                "--combined-reverse-len", str(comb_rev),
                "--combined-overlap", str(comb_overlap)
            ])
            
        if not self.non_overlapping_var.get():
            cmd.append("--allow-overlap")
        self.append_log("Running: " + " ".join(cmd))
        button = getattr(self, "peptide_button", None) or self.run_button
        button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, button, "Scanning peptide motifs..."),
            daemon=True,
        )
        thread.start()

    def run_peptide_annotation(self) -> None:
        hits_path = self.peptide_hits_var.get().strip()
        if not hits_path:
            messagebox.showerror("Missing peptide hits", "Run the peptide coding search first.")
            return
        hits_file = Path(hits_path)
        if not hits_file.is_file():
            messagebox.showerror("Invalid hits file", f"No file found at '{hits_file}'.")
            return
        if not PEPTIDE_ANNOTATION_SCRIPT.is_file():
            messagebox.showerror(
                "Missing script", f"Could not find {PEPTIDE_ANNOTATION_SCRIPT.name} next to this GUI."
            )
            return
        cmd = [
            sys.executable,
            str(PEPTIDE_ANNOTATION_SCRIPT),
            hits_path,
        ]
        gff3_path = self.gff3_var.get().strip()
        if gff3_path:
            cmd.extend(["--gff3", gff3_path])
        for gff_extra in filter(None, [self.gff_var.get().strip()]):
            cmd.extend(["--gff", gff_extra])
        gtf_path = self.gtf_var.get().strip()
        if gtf_path:
            cmd.extend(["--gtf", gtf_path])
        gbff_path = self.gbff_var.get().strip()
        if gbff_path:
            cmd.extend(["--gbff", gbff_path])
        gpff_path = self.gpff_var.get().strip()
        if gpff_path:
            cmd.extend(["--gpff", gpff_path])
        default_output = self._default_annotated_path(hits_file)
        requested = Path(self.peptide_annotation_var.get().strip()) if self.peptide_annotation_var.get().strip() else default_output
        if requested != default_output:
            cmd.extend(["--output", str(requested)])
        else:
            self.peptide_annotation_var.set(str(default_output))
        final_path = requested if requested != default_output else default_output
        self.peptide_annotation_var.set(str(final_path))
        self.append_log("Running: " + " ".join(cmd))
        self.peptide_annotate_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.peptide_annotate_button, "Annotating peptide hits..."),
            daemon=True,
        )
        thread.start()

    def run_triplex_search(self) -> None:
        window_path = self.triplex_rna_windows_var.get().strip()
        rna_path = self.triplex_rna_var.get().strip()
        dna_path = self.triplex_dna_var.get().strip()
        rna_sequence = ""
        if self.triplex_rna_text_widget is not None:
            rna_sequence = "".join(self.triplex_rna_text_widget.get("1.0", END).split())
        using_windows = self._get_triplex_input_mode() == "windows"
        if using_windows:
            if not window_path:
                messagebox.showerror(
                    "Missing RNA windows",
                    "Select or generate an RNA windows FASTA, or switch to the RNA FASTA mode.",
                )
                return
            window_file = Path(window_path)
            if not window_file.is_file():
                messagebox.showerror("Invalid RNA windows file", f"No file found at '{window_path}'.")
                return
            rna_path = window_path
            rna_sequence = ""
        else:
            if not rna_path and not rna_sequence:
                messagebox.showerror(
                    "Missing RNA",
                    "Provide an RNA FASTA/text file or paste an RNA sequence when not using windows.",
                )
                return
            if rna_path and not Path(rna_path).is_file():
                messagebox.showerror("Invalid RNA file", f"No file found at '{rna_path}'.")
                return
        if not dna_path:
            messagebox.showerror("Missing input", "Select a DNA FASTA file to scan.")
            return
        if not Path(dna_path).is_file():
            messagebox.showerror("Invalid DNA file", f"No file found at '{dna_path}'.")
            return
        if not TRIPLEX_SCRIPT.is_file():
            messagebox.showerror("Missing script", f"Could not find {TRIPLEX_SCRIPT.name} next to this GUI.")
            return
        try:
            min_len = int(self.triplex_min_len_var.get())
            max_len = int(self.triplex_max_len_var.get())
            min_score = int(self.triplex_min_score_var.get())
            max_mismatches = int(self.triplex_max_mismatches_var.get())
            workers = int(self.triplex_workers_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid numeric value", str(exc))
            return
        output_prefix = "triplex_hits"
        output_name = self._get_output_name()
        cmd = [
            sys.executable,
            str(TRIPLEX_SCRIPT),
            "--dna",
            dna_path,
            "--min",
            str(min_len),
            "--max",
            str(max_len),
            "--min-score",
            str(min_score),
            "--max-mismatches",
            str(max_mismatches),
            "--output-prefix",
            output_prefix,
            "--output-name",
            output_name,
            "--workers",
            str(workers),
        ]
        strands = []
        if self.triplex_forward_var.get():
            strands.append("forward")
            cmd.append("--scan-forward")
        if self.triplex_reverse_var.get():
            strands.append("reverse")
            cmd.append("--scan-reverse")
        if not strands:
            messagebox.showerror("Missing strand selection", "Select at least one DNA strand to scan.")
            return
        if rna_sequence:
            cmd.extend(["--rna-seq", rna_sequence])
        else:
            cmd.extend(["--rna", rna_path])
        triplex_path = self._get_output_dir() / f"{output_prefix}.tsv"
        self.triplex_output_var.set(str(triplex_path))
        self.append_log("Running: " + " ".join(cmd))
        self.triplex_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.triplex_button, "Running triplex finder..."),
            daemon=True,
        )
        thread.start()

    def run_rna_windowing(self) -> None:
        rna_path_input = self.triplex_rna_var.get().strip()
        inline_sequence = ""
        if self.triplex_rna_text_widget is not None:
            inline_sequence = "".join(self.triplex_rna_text_widget.get("1.0", END).split())
        if not rna_path_input and not inline_sequence:
            messagebox.showerror(
                "Missing RNA", "Provide an RNA sequence file or paste an RNA sequence first."
            )
            return
        input_path: Path
        if rna_path_input and Path(rna_path_input).is_file():
            input_path = Path(rna_path_input)
        elif inline_sequence:
            temp_path = self._get_output_dir() / "inline_rna_input.fa"
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8") as handle:
                handle.write(">RNA_INLINE\n")
                handle.write(inline_sequence.upper() + "\n")
            input_path = temp_path
            self.triplex_rna_var.set(str(temp_path))
        else:
            messagebox.showerror("Invalid RNA file", f"No file found at '{rna_path_input}'.")
            return
        output_path = Path(self.split_output_var.get().strip() or "")
        if not output_path:
            output_path = self._get_output_dir() / "rna_windows.fasta"
            self.split_output_var.set(str(output_path))
        self.triplex_rna_windows_var.set(str(output_path))
        if self._get_triplex_input_mode() != "windows":
            self.triplex_input_mode_var.set("windows")
            self._update_triplex_input_mode()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            window = int(self.split_window_var.get())
            step = int(self.split_step_var.get())
            min_length = int(self.split_min_length_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid numeric value", str(exc))
            return
        cmd = [
            sys.executable,
            str(SPLIT_RNA_SCRIPT),
            str(input_path),
            "--window",
            str(window),
            "--step",
            str(step),
            "--min-length",
            str(min_length),
            "--output",
            str(output_path),
        ]
        if self.split_include_partial_var.get():
            cmd.append("--include-partial")
        self.append_log("Running: " + " ".join(cmd))
        self.split_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.split_button, "Generating RNA windows..."),
            daemon=True,
        )
        thread.start()

    def run_rna_scoring(self) -> None:
        fasta_path = self.split_output_var.get().strip()
        if not fasta_path:
            messagebox.showerror("Missing input", "Generate RNA windows first.")
            return
        fasta_file = Path(fasta_path)
        if not fasta_file.is_file():
            messagebox.showerror("Invalid input", f"No file found at '{fasta_file}'.")
            return
        try:
            min_score = float(self.score_min_score_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid score", str(exc))
            return
        output_dir = self._get_output_dir()
        prefix = "rna_windows_scored"
        score_tsv = output_dir / f"{prefix}.tsv"
        score_fasta = output_dir / f"{prefix}.fasta"
        self.score_tsv_var.set(str(score_tsv))
        self.score_fasta_var.set(str(score_fasta))
        self.triplex_rna_var.set(str(score_fasta))
        self.triplex_rna_windows_var.set(str(score_fasta))
        cmd = [
            sys.executable,
            str(SCORE_RNA_SCRIPT),
            str(fasta_file),
            "--min-score",
            str(min_score),
            "--min-pyrimidine",
            self.score_min_pyrimidine_var.get().strip() or "0.5",
            "--min-spacing",
            self.score_min_spacing_var.get().strip() or "0",
            "--output-prefix",
            prefix,
            "--output-dir",
            str(output_dir),
        ]
        self.append_log("Running: " + " ".join(cmd))
        self.score_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.score_button, "Scoring RNA windows..."),
            daemon=True,
        )
        thread.start()

    def run_triplex_annotation(self) -> None:
        hits_path = self.triplex_output_var.get().strip()
        if not hits_path:
            messagebox.showerror("Missing triplex hits", "Run the triplex finder first.")
            return
        if not Path(hits_path).is_file():
            messagebox.showerror("Invalid hits file", f"No file found at '{hits_path}'.")
            return
        derived_path = self._default_annotated_path(Path(hits_path))
        requested = Path(self.triplex_annotation_var.get().strip()) if self.triplex_annotation_var.get().strip() else derived_path
        requested.parent.mkdir(parents=True, exist_ok=True)
        if requested != derived_path:
            output_arg = str(requested)
        else:
            output_arg = None
            self.triplex_annotation_var.set(str(derived_path))
        cmd = [
            sys.executable,
            str(ANNOTATE_TRIPLEX_SCRIPT),
            hits_path,
        ]
        if output_arg:
            cmd.extend(["--output", output_arg])
        gff_path = self.gff3_var.get().strip()
        if gff_path:
            cmd.extend(["--gff3", gff_path])
        extra_gff = self.gff_var.get().strip()
        if extra_gff:
            cmd.extend(["--gff", extra_gff])
        gtf_path = self.gtf_var.get().strip()
        if gtf_path:
            cmd.extend(["--gtf", gtf_path])
        gbff_path = self.gbff_var.get().strip()
        if gbff_path:
            cmd.extend(["--gbff", gbff_path])
        gpff_path = self.gpff_var.get().strip()
        if gpff_path:
            cmd.extend(["--gpff", gpff_path])
        output_path = requested if output_arg else derived_path
        self.triplex_annotation_var.set(str(output_path))
        self.viz_hits_var.set(str(output_path))
        self.append_log("Running: " + " ".join(cmd))
        self.annotate_triplex_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.annotate_triplex_button, "Annotating triplex hits..."),
            daemon=True,
        )
        thread.start()

    def run_triplex_gene_extract(self) -> None:
        annotation_path = self.triplex_annotation_var.get().strip()
        if not annotation_path:
            messagebox.showerror("Missing annotation", "Run triplex annotation first.")
            return
        annot_file = Path(annotation_path)
        if not annot_file.is_file():
            messagebox.showerror("Invalid annotation", f"No file found at '{annot_file}'.")
            return
        prefix = self.triplex_gene_prefix_var.get().strip() or "triplex_gene_list"
        self.triplex_gene_prefix_var.set(prefix)
        gene_txt = self._get_output_dir() / f"{prefix}.tsv"
        self.triplex_gene_list_var.set(str(gene_txt))
        cmd = [
            sys.executable,
            str(TRIPLEX_GENE_SCRIPT),
            str(annot_file),
            "--output-prefix",
            prefix,
            "--output-name",
            self._get_output_name(),
        ]
        self.append_log("Running: " + " ".join(cmd))
        self.triplex_gene_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.triplex_gene_button, "Collecting triplex gene list..."),
            daemon=True,
        )
        thread.start()

    def run_vector_visualization(self) -> None:
        if not PUBLICATION_VIZ_SCRIPT.is_file():
            messagebox.showerror(
                "Missing script", f"Could not find {PUBLICATION_VIZ_SCRIPT.name} next to this GUI."
            )
            return
        dataset_paths: List[str] = []
        dataset_kinds: List[str] = []
        dataset_labels: List[str] = []
        dataset_opacities: List[str] = []
        dataset_styles: List[str] = []
        dataset_colors: List[str] = []

        source_map = {
            "nt_sequence_hits": (self.process_hits_var, "nt_sequence_hits", "NT sequence hits"),
            "nt_sequence_hits_analyzed": (
                self.analyzed_hits_var,
                "nt_sequence_hits_analyzed",
                "NT sequence hits (analyzed)",
            ),
            "nt_sequence_hits_annotated": (
                self.processed_hits_var,
                "nt_sequence_hits_annotated",
                "NT sequence hits (annotated)",
            ),
            "triplex_hits": (self.triplex_output_var, "triplex_hits", "Triplex hits"),
            "triplex_hits_annotated": (
                self.triplex_annotation_var,
                "triplex_hits_annotated",
                "Triplex hits (annotated)",
            ),
            "peptide_hits": (self.peptide_hits_var, "peptide_hits", "Peptide coding hits"),
            "peptide_hits_annotated": (
                self.peptide_annotation_var,
                "peptide_hits_annotated",
                "Peptide hits (annotated)",
            ),
        }
        for key, flag in self.vector_source_flags.items():
            if not flag.get():
                continue
            var_ref, kind, label = source_map[key]
            path = var_ref.get().strip()
            if not path:
                messagebox.showerror("Missing file", f"Set a path for '{label}' before rendering.")
                return
            dataset_paths.append(path)
            dataset_kinds.append(kind)
            dataset_labels.append(label)
            controls = self.vector_dataset_options[key]
            dataset_opacities.append(self._normalize_opacity_input(controls["opacity"].get()))
            dataset_styles.append(self._normalize_line_style_input(controls["style"].get()))
            dataset_colors.append(controls["color"].get().strip())

        for row in self.vector_custom_rows:
            if not row["include"].get():
                continue
            path = row["path"].get().strip()
            if not path:
                messagebox.showerror("Missing file", "Provide a path for each selected custom overlay.")
                return
            dataset_paths.append(path)
            dataset_kinds.append(row["kind"].get() or "auto")
            label_value = row["label"].get().strip() or Path(path).stem
            dataset_labels.append(label_value)
            dataset_opacities.append(self._normalize_opacity_input(row["opacity"].get()))
            dataset_styles.append(self._normalize_line_style_input(row["style"].get()))
            dataset_colors.append(row["color"].get().strip())

        if not dataset_paths:
            messagebox.showerror("No datasets", "Select at least one dataset or custom file to visualize.")
            return

        for path in dataset_paths:
            if not Path(path).is_file():
                messagebox.showerror("Invalid file", f"No file found at '{path}'.")
                return

        fmt = self.vector_output_format_var.get().strip() or "png"
        output_value = (self.vector_output_var.get() or "").strip()
        if not output_value:
            output_value = str(self._get_output_dir() / f"chromosome_tracks.{fmt}")
            self.vector_output_var.set(output_value)
        else:
            p = Path(output_value)
            if p.suffix.lower() != f".{fmt}":
                output_value = str(p.with_suffix(f".{fmt}"))
                self.vector_output_var.set(output_value)
        output_path = Path(output_value)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PUBLICATION_VIZ_SCRIPT),
        ]
        cmd.extend(dataset_paths)
        for kind in dataset_kinds:
            cmd.extend(["--input-kind", kind or "auto"])
        for label in dataset_labels:
            cmd.extend(["--input-label", label])
        for opacity in dataset_opacities:
            cmd.extend(["--input-opacity", opacity])
        for style in dataset_styles:
            cmd.extend(["--input-style", style])
        for color in dataset_colors:
            cmd.extend(["--input-color", color if color else ""])
            
        cmd.extend(["--output", str(output_path)])
        cmd.extend(["--output-format", fmt])
        cmd.extend(["--dpi", self.vector_dpi_var.get().strip() or "300"])
        cmd.extend(["--fig-width", self.vector_fig_width_var.get().strip() or "12.0"])
        cmd.extend(["--fig-height", self.vector_fig_height_var.get().strip() or "8.0"])
        cmd.extend(["--font-family", self.vector_font_family_var.get().strip() or "Arial"])
        cmd.extend(["--font-size", self.vector_font_size_var.get().strip() or "12"])
        cmd.extend(["--palette", self.vector_palette_var.get().strip() or "default"])
        cmd.extend(["--scale-mode", self.vector_scale_mode_var.get().strip() or "absolute"])
        cmd.extend(["--bg-color", self.vector_bg_color_var.get().strip() or "#f8f8f8"])
        cmd.extend(["--bg-alpha", self.vector_bg_alpha_var.get().strip() or "1.0"])
        cmd.extend(["--legend-pos", self.vector_legend_pos_var.get().strip() or "right"])

        label_mode = (self.vector_label_mode_var.get() or "gene+coords").strip()
        if label_mode not in {"none", "gene", "gene+coords"}:
            label_mode = "gene+coords"
        cmd.extend(["--label-mode", label_mode])
        if self.vector_density_var.get():
            cmd.append("--density")
        if self.vector_annotate_var.get():
            column_name = self.vector_annotation_column_var.get().strip() or "longest_palindrome_sequence"
            cmd.extend(["--annotate-column", column_name])
        for seq_value in self._split_filter_values(self.vector_seq_ids_var.get()):
            cmd.extend(["--sequence-id", seq_value])
        for region_value in self._split_filter_values(self.vector_region_filter_var.get()):
            cmd.extend(["--region", region_value])
        for gene_value in self._split_filter_values(self.vector_gene_filter_var.get()):
            cmd.extend(["--gene-filter", gene_value])
        title = self.vector_title_var.get().strip()
        if title:
            cmd.extend(["--title", title])
        self.append_log("Running: " + " ".join(cmd))
        self.vector_viz_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.vector_viz_button, "Rendering chromosome tracks..."),
            daemon=True,
        )
        thread.start()

    def run_visualization(self) -> None:
        hits_path = (
            self.viz_hits_var.get().strip()
            or self.processed_hits_var.get().strip()
            or self.process_hits_var.get().strip()
        )
        if not hits_path:
            messagebox.showerror("Missing input", "Select a processed TSV (or CSV) file to visualize.")
            return
        input_file = Path(hits_path)
        if not input_file.is_file():
            messagebox.showerror("Invalid file", f"No file found at '{hits_path}'.")
            return
        if not VISUALIZER_SCRIPT.is_file():
            messagebox.showerror(
                "Missing script", f"Could not find {VISUALIZER_SCRIPT.name} next to this GUI."
            )
            return
        scope = self.viz_scope_var.get()
        if scope == "genes":
            gene_list = self.gene_list_var.get().strip()
            if not gene_list:
                messagebox.showerror("Missing gene list", "Run the gene extraction step first.")
                return
            gene_path = Path(gene_list)
            if not gene_path.is_file():
                messagebox.showerror("Invalid gene list", f"No file found at '{gene_path}'.")
                return
            input_file = gene_path
        output_path = self._resolve_viz_output_path(self.viz_output_var.get())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._set_viz_output_var(output_path)
        viz_mode = self.viz_format_var.get()
        cli_viz_mode = "csv" if viz_mode == "custom" else viz_mode
        cmd = [
            "cmd",
            "/c",
            "start",
            "",
            sys.executable,
            "chromosome_visualizer.py",
            str(input_file),
            "--input-format",
            cli_viz_mode,
            "--mode",
            scope,
            "--output",
            str(output_path),
            "--label-mode",
            self.viz_label_mode_var.get(),
        ]
        self.append_log(
            "Launching visualization command in new terminal: "
            + " ".join(cmd[4:])
        )
        try:
            subprocess.Popen(cmd, cwd=str(Path(__file__).parent))
        except Exception as exc:
            messagebox.showerror("Visualization error", str(exc))
            return

    def run_gene_extract(self) -> None:
        hits_path = (
            self.processed_hits_var.get().strip()
            or self.viz_hits_var.get().strip()
            or self.process_hits_var.get().strip()
        )
        if not hits_path:
            messagebox.showerror("Missing input", "Please select the processed TSV file first.")
            return
        path_obj = Path(hits_path)
        if not path_obj.is_file():
            messagebox.showerror("Invalid file", f"No file found at '{hits_path}'.")
            return
        if not GENE_SCRIPT.is_file():
            messagebox.showerror("Missing script", f"Could not find {GENE_SCRIPT.name} next to this GUI.")
            return
        cmd = [
            sys.executable,
            str(GENE_SCRIPT),
            hits_path,
        ]
        output_prefix = "gene_list"
        output_name = self._get_output_name()
        cmd.extend(["--output-prefix", output_prefix, "--output-name", output_name])
        gene_tsv = self._get_output_dir() / f"{output_prefix}.tsv"
        self.gene_list_var.set(str(gene_tsv))
        self.append_log("Running: " + " ".join(cmd))
        self.genes_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._execute_command,
            args=(cmd, self.genes_button, "Extracting gene list..."),
            daemon=True,
        )
        thread.start()

    def _set_status(self, message: str, level: str = "info") -> None:
        self.status_var.set(message)
        color = self._status_colors.get(level, self._status_colors["info"])
        if self.status_label is not None:
            try:
                self.status_label.configure(fg=color)
            except Exception:
                pass
        if level == "busy" and self._status_reset_job is not None:
            try:
                self.root.after_cancel(self._status_reset_job)
            except Exception:
                pass
            self._status_reset_job = None

    def _execute_command(
        self, cmd, button_widget: Button, status_message: str | None = None, post_callback=None
    ) -> None:
        self.root.after(0, lambda: self._begin_command(status_message or "Working..."))
        success = False
        t_start = time.time()
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT_DIR))
            stdout = completed.stdout.strip()
            stderr = completed.stderr.strip()
            if stdout:
                self.root.after(0, self.append_log, stdout)
            if stderr:
                self.root.after(0, self.append_log, f"[stderr] {stderr}")
            success = completed.returncode == 0
            elapsed = time.time() - t_start
            elapsed_str = self._format_elapsed(elapsed)
            if success:
                self.root.after(0, self.append_log, f"Command completed successfully in {elapsed_str}.")
            else:
                self.root.after(
                    0, self.append_log, f"Command failed with exit code {completed.returncode} after {elapsed_str}."
                )
        except Exception as exc:
            elapsed = time.time() - t_start
            elapsed_str = self._format_elapsed(elapsed)
            self.root.after(0, self.append_log, f"Error running scanner after {elapsed_str}: {exc}")
            success = False
        finally:
            self.root.after(0, button_widget.configure, {"state": "normal"})
            self.root.after(0, lambda: self._end_command(success))
            if post_callback:
                self.root.after(0, post_callback)
            self.root.after(0, self._refresh_output_paths)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remaining = seconds % 60
        if minutes < 60:
            return f"{minutes}m {remaining:.0f}s"
        hours = int(minutes // 60)
        minutes = minutes % 60
        return f"{hours}h {minutes}m {remaining:.0f}s"

    def _begin_command(self, message: str) -> None:
        if self.status_progress is not None:
            try:
                self.status_progress.start(12)
            except Exception:
                pass
        if self._status_reset_job is not None:
            try:
                self.root.after_cancel(self._status_reset_job)
            except Exception:
                pass
            self._status_reset_job = None
        self._set_status(message, level="busy")

    def _end_command(self, success: bool) -> None:
        if self.status_progress is not None:
            try:
                self.status_progress.stop()
            except Exception:
                pass
        msg = "Completed successfully" if success else "Check log for errors"
        level = "success" if success else "error"
        self._set_status(msg, level=level)
        try:
            self._status_reset_job = self.root.after(
                4000, lambda: self._set_status("Idle")
            )
        except Exception:
            self._status_reset_job = None

    @staticmethod
    def _read_lines(widget) -> list[str]:
        if widget is None:
            return []
        value = widget.get("1.0", END)
        result = []
        for line in value.splitlines():
            line = line.strip()
            if line:
                result.append(line)
        return result

    def _read_text_lines(self, widget) -> list[str]:
        if widget is None:
            return []
        text = widget.get("1.0", END)
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _capture_advanced_state(self) -> None:
        if self.motif_widget is not None:
            self._saved_motif_text = self.motif_widget.get("1.0", END).strip()
        if self.exclude_motif_widget is not None:
            self._saved_exclude_text = self.exclude_motif_widget.get("1.0", END).strip()
        if self.base_widget is not None:
            self._saved_base_text = self.base_widget.get("1.0", END).strip()
        if self.repeat_widget is not None:
            self._saved_repeat_text = self.repeat_widget.get("1.0", END).strip()
        if self.forward_motif_widget is not None:
            self._saved_forward_motif_text = self.forward_motif_widget.get("1.0", END).strip()
        if self.reverse_motif_widget is not None:
            self._saved_reverse_motif_text = self.reverse_motif_widget.get("1.0", END).strip()
        if getattr(self, "self_comp_widget", None) is not None:
            # legacy text widget no longer used – self_comp_rules list is live
            pass

    def _ensure_advanced_panel(self) -> None:
        pass

    def _get_advanced_lines(self, widget, saved_attr: str) -> list[str]:
        if widget is not None:
            text = widget.get("1.0", END)
        else:
            text = getattr(self, saved_attr, "")
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _set_multiline_widget(self, widget, saved_attr: str, lines: list[str]) -> None:
        content = "\n".join(lines)
        if widget is not None:
            widget.delete("1.0", END)
            widget.insert("1.0", content)
        setattr(self, saved_attr, content)

    def _load_chemistry_profiles(self) -> Dict[str, Dict[str, List[str]]]:
        profiles: Dict[str, Dict[str, List[str]]] = {}
        for key, value in BUILTIN_CHEMISTRY_PROFILES.items():
            entry: Dict[str, List[str]] = {}
            for field in ("motifs", "motifs_forward", "motifs_reverse", "base_content", "max_repeat", "motif_self_comp"):
                if field in value:
                    entry[field] = list(value[field])
            profiles[key] = entry
        custom: Dict[str, Dict[str, List[str]]] = {}
        if PROFILE_STORE_PATH.is_file():
            try:
                with PROFILE_STORE_PATH.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                for name, payload in data.items():
                    if not isinstance(payload, dict):
                        continue
                    entry: Dict[str, List[str]] = {}
                    for field in ("motifs", "motifs_forward", "motifs_reverse", "base_content", "max_repeat", "motif_self_comp"):
                        values = payload.get(field)
                        if isinstance(values, list):
                            entry[field] = [str(item) for item in values if str(item).strip()]
                    if entry:
                        normalized = name.lower()
                        custom[normalized] = entry
                        profiles[normalized] = entry
            except Exception:
                custom = {}
        self.custom_profiles = custom
        return profiles

    def _write_profile_store(self) -> None:
        if not self.custom_profiles:
            if PROFILE_STORE_PATH.is_file():
                try:
                    PROFILE_STORE_PATH.unlink()
                except Exception:
                    pass
            return
        try:
            with PROFILE_STORE_PATH.open("w", encoding="utf-8") as handle:
                json.dump(self.custom_profiles, handle, indent=2)
        except Exception as exc:
            messagebox.showerror("Profile save error", str(exc))

    def _refresh_profile_options(self) -> None:
        options = ["Custom (manual)"]
        mapping = {"Custom (manual)": "custom"}
        for key in sorted(self.chemistry_profiles):
            display = key.title()
            options.append(display)
            mapping[display] = key
        self._profile_display_to_key = mapping
        self.profile_selector["values"] = options
        current = self.chemistry_profile_var.get()
        if current not in options:
            self.chemistry_profile_var.set("Custom (manual)")

    def _apply_selected_profile(self) -> None:
        display = self.chemistry_profile_var.get()
        key = self._profile_display_to_key.get(display)
        if not key or key == "custom":
            return
        profile = self.chemistry_profiles.get(key)
        if not profile:
            messagebox.showerror("Profile error", f"No data found for profile '{display}'.")
            return
        pass
        if "motifs" in profile:
            self._set_multiline_widget(self.motif_widget, "_saved_motif_text", profile["motifs"])
        if "motifs_forward" in profile:
            self._set_multiline_widget(self.forward_motif_widget, "_saved_forward_motif_text", profile["motifs_forward"])
        if "motifs_reverse" in profile:
            self._set_multiline_widget(self.reverse_motif_widget, "_saved_reverse_motif_text", profile["motifs_reverse"])
        if "base_content" in profile:
            self._set_multiline_widget(self.base_widget, "_saved_base_text", profile["base_content"])
        if "max_repeat" in profile:
            self._set_multiline_widget(self.repeat_widget, "_saved_repeat_text", profile["max_repeat"])
        if "motif_self_comp" in profile:
            # Legacy: load CLI spec strings back into the rules list
            lb = getattr(self, "_sc_rules_listbox", None)
            if lb is not None:
                lb.delete(0, END)
            self._self_comp_rules.clear()
            for spec in profile["motif_self_comp"]:
                rule = self._sc_cli_spec_to_rule(spec)
                if rule:
                    self._self_comp_rules.append(rule)
                    if lb is not None:
                        lb.insert(END, self._sc_rule_label(rule))
        self._update_reactive_oligo()

    def _prompt_save_profile(self) -> None:
        pass
        name = simpledialog.askstring("Save chemistry profile", "Profile name:")
        if not name:
            return
        key = name.strip().lower()
        if not key:
            return
        profile = {
            "motifs": self._get_advanced_lines(self.motif_widget, "_saved_motif_text"),
            "motifs_forward": self._get_advanced_lines(self.forward_motif_widget, "_saved_forward_motif_text"),
            "motifs_reverse": self._get_advanced_lines(self.reverse_motif_widget, "_saved_reverse_motif_text"),
            "base_content": self._get_advanced_lines(self.base_widget, "_saved_base_text"),
            "max_repeat": self._get_advanced_lines(self.repeat_widget, "_saved_repeat_text"),
            "motif_self_comp": self._sc_rules_to_cli_args(),
        }
        self.custom_profiles[key] = profile
        self._write_profile_store()
        self.chemistry_profiles = self._load_chemistry_profiles()
        self._refresh_profile_options()

    def _delete_selected_profile(self) -> None:
        display = self.chemistry_profile_var.get()
        key = self._profile_display_to_key.get(display)
        if not key or key == "custom":
            return
        if key not in self.custom_profiles:
            messagebox.showinfo("Delete profile", "Built-in profiles cannot be deleted.")
            return
        if not messagebox.askyesno("Delete profile", f"Delete profile '{display}'?"):
            return
        self.custom_profiles.pop(key, None)
        self._write_profile_store()
        self.chemistry_profiles = self._load_chemistry_profiles()
        self.chemistry_profile_var.set("Custom (manual)")
        self._refresh_profile_options()

    def _load_defaults_async(self) -> None:
        def worker():
            defaults = self._discover_default_paths()
            self.root.after(0, lambda: self._apply_default_paths(defaults))

        threading.Thread(target=worker, daemon=True).start()

    def _discover_default_paths(self, data_dir: Path | None = None) -> Dict[str, str]:
        defaults: Dict[str, str] = {}
        data_dir = data_dir or self._get_genome_data_dir()
        if not data_dir.is_dir():
            return defaults

        def pick_first(patterns: list[str]) -> Path | None:
            data_dir_abs = data_dir if data_dir.is_absolute() else ROOT_DIR / data_dir
            for pattern in patterns:
                for candidate in sorted(data_dir_abs.glob(pattern)):
                    if candidate.is_file():
                        try:
                            return candidate.relative_to(ROOT_DIR)
                        except ValueError:
                            return candidate
            return None

        fasta = pick_first(["*.fna", "*.fa", "*.fasta"])
        if fasta:
            defaults["fasta_var"] = str(fasta)
            defaults["triplex_dna_var"] = str(fasta)
            defaults["peptide_fasta_var"] = str(fasta)

        hits = pick_first(["*nt_sequence_hits*.tsv"])
        if hits:
            defaults["process_hits_var"] = str(hits)

        viz_only = pick_first(["*.csv"])
        if viz_only:
            defaults["viz_hits_var"] = str(viz_only)

        gbff = pick_first(["*.gbff"])
        if gbff:
            defaults["gbff_var"] = str(gbff)
        gff3 = pick_first(["*.gff3"])
        if gff3:
            defaults["gff3_var"] = str(gff3)
        gff = pick_first(["*.gff"])
        if gff:
            defaults.setdefault("gff_var", str(gff))
        gtf = pick_first(["*.gtf"])
        if gtf:
            defaults["gtf_var"] = str(gtf)
        gpff = pick_first(["*.gpff"])
        if gpff:
            defaults["gpff_var"] = str(gpff)

        processed = pick_first(["*nt_sequence_hits_annotated*.tsv"])
        if processed:
            defaults["processed_hits_var"] = str(processed)

        gene_list = pick_first(["gene_list*.tsv"])
        if gene_list:
            defaults["gene_list_var"] = str(gene_list)

        triplex_hits = pick_first(["*triplex_hits*.tsv"])
        if triplex_hits:
            defaults["triplex_output_var"] = str(triplex_hits)

        return defaults

    def _apply_default_paths(self, defaults: Dict[str, str]) -> None:
        for attr, value in defaults.items():
            var = getattr(self, attr, None)
            if isinstance(var, StringVar) and not var.get().strip():
                var.set(value)

    def _selected_strands(self) -> list[str]:
        strands: list[str] = []
        if self.forward_strand_var.get():
            strands.append("forward")
        if self.reverse_strand_var.get():
            strands.append("reverse")
        if self.combined_strand_var.get():
            strands.append("combined")
        return strands

    def _toggle_combined_fields(self) -> None:
        enable = bool(self.combined_strand_var.get())
        state = "normal" if enable else "disabled"
        for widget in self.combined_entry_widgets:
            widget.configure(state=state)

    def _get_triplex_input_mode(self) -> str:
        mode = (self.triplex_input_mode_var.get() or "windows").strip().lower()
        if mode not in {"windows", "rna"}:
            mode = "windows"
            self.triplex_input_mode_var.set(mode)
        return mode

    def _update_triplex_input_mode(self, *_args) -> None:
        using_windows = self._get_triplex_input_mode() == "windows"
        self._set_widget_group_state(self.triplex_windows_controls, using_windows)
        self._set_widget_group_state(self.triplex_rna_controls, not using_windows)
        if self.triplex_rna_text_widget is not None:
            state = "disabled" if using_windows else "normal"
            self.triplex_rna_text_widget.configure(state=state)

    @staticmethod
    def _set_widget_group_state(group: dict[str, object] | None, enabled: bool) -> None:
        if not group:
            return
        state = "normal" if enabled else "disabled"
        for widget in group.values():
            try:
                widget.configure(state=state)
            except Exception:
                continue

    def _sync_combined_defaults(self) -> None:
        window_value = self.window_var.get()
        if not self.combined_forward_var.get().strip():
            self.combined_forward_var.set(window_value)
        if not self.combined_reverse_var.get().strip():
            self.combined_reverse_var.set(window_value)

    def _get_output_name(self) -> str:
        return (self.output_name_var.get() or "").strip()

    def _get_output_dir(self) -> Path:
        suffix = self._get_output_name()
        base = Path(f"output_{suffix}" if suffix else "output_")
        if not base.is_absolute():
            # Anchor to script location
            return (Path(__file__).parent / base).resolve()
        return base

    def _resolve_viz_output_path(self, raw_value: str | None = None) -> Path:
        raw = (raw_value or "").strip()
        base_name = Path(raw).name if raw else self.viz_default_filename
        if not base_name:
            base_name = self.viz_default_filename
        return self._get_output_dir() / base_name

    @staticmethod
    def _default_annotated_path(input_path: Path) -> Path:
        suffix = input_path.suffix or ".tsv"
        base = input_path.with_suffix("")
        return base.with_name(base.name + "_annotated").with_suffix(suffix)

    def _set_viz_output_var(self, path: Path) -> None:
        value = str(path)
        if self.viz_output_var.get().strip() == value:
            return
        self._changing_viz_path = True
        self.viz_output_var.set(value)
        self._changing_viz_path = False

    def _on_viz_var_change(self, *_args) -> None:
        if self._changing_viz_path:
            return
        resolved = self._resolve_viz_output_path(self.viz_output_var.get())
        self._set_viz_output_var(resolved)

    def _refresh_output_paths(self, *_args) -> None:
        base_dir = self._get_output_dir()

        def ensure(var: StringVar, path: Path) -> None:
            value = str(path)
            if var.get() != value:
                var.set(value)

        raw_prefix = "nt_sequence_hits"
        processed_prefix = "nt_sequence_hits_annotated"
        gene_prefix = "gene_list"
        triplex_prefix = "triplex_hits"

        ensure(self.process_hits_var, base_dir / f"{raw_prefix}.tsv")
        analysis_prefix = "nt_sequence_hits_analyzed"
        analysis_path = base_dir / f"{analysis_prefix}.tsv"
        ensure(self.analyzed_hits_var, analysis_path)
        processed_path = base_dir / f"{processed_prefix}.tsv"
        ensure(self.processed_hits_var, processed_path)
        ensure(self.viz_hits_var, processed_path)
        if not self.annotation_input_var.get().strip():
            self.annotation_input_var.set(str(analysis_path))
        ensure(self.gene_list_var, base_dir / f"{gene_prefix}.tsv")
        ensure(self.triplex_output_var, base_dir / f"{triplex_prefix}.tsv")
        score_prefix = "rna_windows_scored"
        ensure(self.score_tsv_var, base_dir / f"{score_prefix}.tsv")
        ensure(self.score_fasta_var, base_dir / f"{score_prefix}.fasta")
        ensure(self.triplex_annotation_var, base_dir / "triplex_hits_annotated.tsv")
        gene_prefix_triplex = self.triplex_gene_prefix_var.get().strip() or "triplex_gene_list"
        ensure(self.triplex_gene_list_var, base_dir / f"{gene_prefix_triplex}.tsv")
        ensure(self.vector_input_var, processed_path)
        ensure(self.vector_output_var, base_dir / "chromosome_tracks.svg")
        peptide_prefix = "peptide_coding_hits"
        ensure(self.peptide_hits_var, base_dir / f"{peptide_prefix}.tsv")
        ensure(self.peptide_annotation_var, base_dir / f"{peptide_prefix}_annotated.tsv")

        resolved_viz = self._resolve_viz_output_path(self.viz_output_var.get())
        self._set_viz_output_var(resolved_viz)
        ensure(self.split_output_var, base_dir / "rna_windows.fasta")
        if not self.triplex_rna_windows_var.get().strip():
            self.triplex_rna_windows_var.set(self.split_output_var.get())
        self._persist_settings()
        self._run_file_picker_validators()

    def _persist_settings(self) -> None:
        if getattr(self, "_loading_settings", False):
            return
        cfg_path = ROOT_DIR / "gui_settings.ini"
        try:
            cfg = configparser.ConfigParser()
            cfg["paths"] = {}
            _NON_PATH_KEYS = {
                "output_name", "selected_genome", "triplex_input_mode",
                "vector_annotation_column", "vector_label_mode",
                "vector_seq_ids", "vector_region_filters", "vector_gene_filters",
                "palindrome_min_len", "top_oligos",
            }
            for key, var in self._persisted_string_vars.items():
                value = var.get().strip()
                if value:
                    if key not in _NON_PATH_KEYS:
                        # Convert to relative path if inside ROOT_DIR
                        try:
                            p = Path(value)
                            if not p.is_absolute():
                                p = ROOT_DIR / p
                            resolved = p.resolve()
                            value = str(resolved.relative_to(ROOT_DIR.resolve()).as_posix())
                        except (ValueError, OSError):
                            # Path is outside our toolkit root — keep as absolute
                            pass
                    cfg["paths"][key] = value
            cfg["options"] = {}
            for key, var in self._persisted_option_vars.items():
                cfg["options"][key] = "1" if var.get() else "0"
            with cfg_path.open("w", encoding="utf-8") as handle:
                cfg.write(handle)
        except Exception as exc:
            print(f"Warning: Unable to save GUI settings: {exc}", file=sys.stderr)

    def _load_persisted_settings(self) -> None:
        cfg_path = ROOT_DIR / "gui_settings.ini"
        if not cfg_path.is_file():
            return
        parser = configparser.ConfigParser()
        try:
            parser.read(cfg_path, encoding="utf-8")
        except Exception as exc:
            print(f"Warning: Unable to load GUI settings: {exc}", file=sys.stderr)
            return
        self._loading_settings = True
        try:
            for key, var in self._persisted_string_vars.items():
                value = parser.get("paths", key, fallback="").strip()
                if key == "selected_genome" and value and value not in self._genome_slug_to_path:
                    continue
                # Drop paths that no longer exist on disk OR that point outside
                # our toolkit root (e.g. a stale path from an old installation).
                NON_PATH_KEYS = {
                    "output_name", "selected_genome", "triplex_input_mode",
                    "vector_annotation_column", "vector_label_mode",
                    "vector_seq_ids", "vector_region_filters", "vector_gene_filters",
                    "palindrome_min_len", "top_oligos",
                }
                if value and key not in NON_PATH_KEYS:
                    p = Path(value)
                    if not p.is_absolute():
                        p = ROOT_DIR / p
                    # Must exist
                    if not p.exists():
                        continue
                    # Keep it as relative if it's within ROOT_DIR
                    try:
                        value = str(p.resolve().relative_to(ROOT_DIR.resolve()).as_posix())
                    except ValueError:
                        pass
                if value:
                    var.set(value)
            for key, var in self._persisted_option_vars.items():
                try:
                    setting = parser.getboolean("options", key, fallback=var.get())
                except ValueError:
                    setting = var.get()
                var.set(setting)
        finally:
            self._loading_settings = False


def main() -> None:
    root = Tk()
    ScannerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
