import sys
from pathlib import Path
from typing import Tuple, List, Callable
from tkinter import (
    Frame, Label, LabelFrame, Button, Entry, StringVar, 
    Canvas, Scrollbar, LEFT, RIGHT, TOP, BOTH, X, Y, END,
    filedialog, messagebox, scrolledtext
)
from mako.gui.components.utils import add_tab_banner, add_labeled_entry

# Try to import analysis logic (Legacy imports for now)
try:
    from nt_sequence_G4_TD_analysis import (
        g4hunter_score, g4boost_score, base_composition, 
        tm_wallace, tm_nearest_neighbor, calc_codon_efficiency,
        i_motif_score, r_loop_score, hairpin_score, cpg_island_score,
        reverse_complement
    )
    from peptide_coding_search import translate_sequence, CODON_TABLE
    _TOOLS_AVAILABLE = True
except ImportError:
    _TOOLS_AVAILABLE = False

class ToolsTab:
    def __init__(self, parent: Frame, root: Tk, heading: str, accent_color: str):
        self.parent = parent
        self.root = root
        
        # Variables
        self.tool_g4_window_var = StringVar(value="25")
        self.tool_thermo_dna_conc_var = StringVar(value="0.001")
        self.tool_thermo_salt_conc_var = StringVar(value="0.15")
        self.tool_mutator_target_var = StringVar(value="1.5")
        self.tool_mutator_max_mut_var = StringVar(value="5")
        
        self._build_ui(heading, accent_color)

    def _build_ui(self, heading: str, accent_color: str):
        canvas = Canvas(self.parent, highlightthickness=0)
        v_scroll = Scrollbar(self.parent, orient="vertical", command=canvas.yview)
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
        
        add_tab_banner(wrapper, heading, accent_color)
        
        if not _TOOLS_AVAILABLE:
            Label(wrapper, text="Analysis scripts not found. Tools tab disabled.", fg="red").pack(pady=20)
            return

        # 1. G4Hunter Scorer
        g4_frame, g4_content = self._make_collapsible_tool(wrapper, "G4Hunter Scorer")
        Label(g4_content, text="NT Sequence:").pack(anchor="w")
        g4_text = scrolledtext.ScrolledText(g4_content, height=4, width=60)
        g4_text.pack(fill=X, pady=(0, 6))
        row = Frame(g4_content)
        row.pack(fill=X)
        add_labeled_entry(row, "Window Size:", self.tool_g4_window_var, 6)
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
        add_labeled_entry(row, "DNA Conc (M):", self.tool_thermo_dna_conc_var, 8)
        add_labeled_entry(row, "Salt Conc (M):", self.tool_thermo_salt_conc_var, 8)
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
        add_labeled_entry(row, "Target G4 Score:", self.tool_mutator_target_var, 6)
        add_labeled_entry(row, "Max Mutations:", self.tool_mutator_max_mut_var, 6)
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
                lf.configure(text=f" ◀ {title}"); is_expanded[0] = False
            else:
                content.pack(fill=X)
                lf.configure(text=f" ▶ {title}"); is_expanded[0] = True
        Button(content, text="Collapse", command=toggle, font=("Segoe UI", 8)).pack(anchor="ne")
        return lf, content

    def _copy_tool_results(self, text_widget: scrolledtext.ScrolledText) -> None:
        content = text_widget.get("1.0", END).strip()
        if not content: return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.root.update()

    def _save_tool_results(self, text_widget: scrolledtext.ScrolledText) -> None:
        content = text_widget.get("1.0", END).strip()
        if not content: return
        path = filedialog.asksaveasfilename(
            defaultextension=".tsv",
            filetypes=[("TSV files", "*.tsv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path: return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Success", f"Results saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {e}")

    # Tool Execution Methods (Copied from GUI and adapted)
    def _run_g4hunter_tool(self, input_widget, output_widget) -> None:
        seq = input_widget.get("1.0", END).strip().upper()
        if not seq: return
        try: w_size = int(self.tool_g4_window_var.get())
        except: w_size = 25
        
        g4h = g4hunter_score(seq, w_size)
        g4b = g4boost_score(seq)
        counts, gc_pct = base_composition(seq)
        
        res = [f"G4Hunter Score (window={w_size}): {g4h:.4f}", f"G4Boost Score: {g4b:.4f}", 
               f"GC Content: {gc_pct:.2f}%", f"Base Counts: {', '.join(f'{b}={c}' for b, c in counts.items())}",
               "\nSequence:", seq]
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert("1.0", "\n".join(res))
        output_widget.configure(state="disabled")

    def _run_codon_optimizer_tool(self, input_widget, output_widget) -> None:
        raw = input_widget.get("1.0", END).strip()
        if not raw: return
        is_peptide = any(c in "DEFHIKLMNPQRSTVWYZ" for c in raw.upper())
        if is_peptide:
            seq = translate_sequence(raw.upper(), to_nt=True)
            label = "Optimized NT sequence (from peptide):"
        else:
            pep = translate_sequence(raw.upper())
            seq = translate_sequence(pep, to_nt=True)
            label = "Optimized NT sequence (codon-re-encoded):"
        
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert("1.0", f"{label}\n\n{seq}")
        output_widget.configure(state="disabled")

    def _run_translator_tool(self, input_widget, output_widget) -> None:
        nt = input_widget.get("1.0", END).strip().upper()
        if not nt: return
        res = []
        for frame in [1, 2, 3]:
            res.append(f"Forward Frame {frame}: {translate_sequence(nt[frame-1:])}")
        rev = reverse_complement(nt)
        for frame in [1, 2, 3]:
            res.append(f"Reverse Frame {frame}: {translate_sequence(rev[frame-1:])}")
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert("1.0", "\n".join(res))
        output_widget.configure(state="disabled")

    def _run_thermo_tool(self, input_widget, output_widget) -> None:
        seq = input_widget.get("1.0", END).strip().upper()
        if not seq: return
        try:
            conc = float(self.tool_thermo_dna_conc_var.get())
            salt = float(self.tool_thermo_salt_conc_var.get())
        except: conc, salt = 0.001, 0.15
        
        tw = tm_wallace(seq)
        tnn = tm_nearest_neighbor(seq, conc, salt)
        im = i_motif_score(seq)
        rl = r_loop_score(seq)
        hp = hairpin_score(seq)
        cpg = cpg_island_score(seq)
        
        res = [f"Melting Temp (Wallace): {tw:.2f} C", f"Melting Temp (NN): {tnn:.2f} C",
               f"i-Motif Propensity: {im:.2f}", f"R-Loop Propensity: {rl:.2f}",
               f"Hairpin Score: {hp:.2f}", f"CpG Island Score: {cpg:.2f}"]
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert("1.0", "\n".join(res))
        output_widget.configure(state="disabled")

    def _run_g4_mutator_tool(self, input_widget, output_widget) -> None:
        seq = input_widget.get("1.0", END).strip().upper()
        if not seq: return
        try:
            target = float(self.tool_mutator_target_var.get())
            max_mut = int(self.tool_mutator_max_mut_var.get())
            w_size = int(self.tool_g4_window_var.get())
        except: target, max_mut, w_size = 1.5, 5, 25

        def greedy_mutate(s, tgt, limit, window):
            curr = s
            curr_score = g4hunter_score(curr, window)
            for _ in range(limit):
                if abs(curr_score - tgt) < 0.05: break
                best_mut = curr; best_score = curr_score
                for i in range(len(curr)):
                    for b in "ACGT":
                        if curr[i] == b: continue
                        trial = curr[:i] + b + curr[i+1:]
                        trial_score = g4hunter_score(trial, window)
                        if abs(trial_score - tgt) < abs(best_score - tgt):
                            best_mut = trial; best_score = trial_score
                if best_mut == curr: break
                curr = best_mut; curr_score = best_score
            return curr, curr_score

        mutated, final_score = greedy_mutate(seq, target, max_mut, w_size)
        res = [f"Initial G4 Score: {g4hunter_score(seq, w_size):.4f}",
               f"Final G4 Score: {final_score:.4f}", f"Target: {target}",
               f"Mutations used: {sum(1 for i in range(len(seq)) if seq[i] != mutated[i])} / {max_mut}",
               "\nOptimized Sequence:", mutated]
        output_widget.configure(state="normal")
        output_widget.delete("1.0", END)
        output_widget.insert("1.0", "\n".join(res))
        output_widget.configure(state="disabled")
