#!/usr/bin/env python3
"""Scan genomic FASTA files for regions matching BOTH nucleotide and peptide constraints."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set

# Import from nt_sequence_search
try:
    from nt_sequence_search import (
        build_parser as build_nt_parser,
        parse_motif_specs,
        parse_exclude_motifs,
        parse_base_content_specs,
        parse_max_repeat_specs,
        parse_motif_self_comp_specs,
        parse_region_specs,
        parse_chromosome_numbers,
        sequence_matches_chromosome_filter,
        normalize_seq_name,
        iter_nc_sequences,
        format_seq_id,
        scan_sequence,
        scan_combined_windows,
        MotifConstraint,
        ExcludeMotifConstraint,
        MotifSelfCompConstraint,
    )
except ImportError as e:
    raise SystemExit(f"Error importing from nt_sequence_search.py: {e}")

# Import from peptide_coding_search
try:
    from peptide_coding_search import (
        build_motif_patterns as build_pep_motif_patterns,
        compile_patterns as compile_pep_patterns,
        iter_frames,
        translate_sequence,
        match_motif_in_window,
        calc_codon_efficiency,
        MotifPattern,
        parse_amino_content,
        parse_max_repeats as parse_pep_max_repeats,
    )
except ImportError as e:
    raise SystemExit(f"Error importing from peptide_coding_search.py: {e}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for regions matching BOTH nucleotide and peptide constraints."
    )
    # Basic
    parser.add_argument("fasta", help="Genome FASTA file to scan")
    parser.add_argument("--window", type=int, required=True, help="Window length (nucleotides)")
    parser.add_argument("--step", type=int, default=1, help="Slide step size (nucleotides)")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0=all)")
    parser.add_argument("--output-prefix", default="combined_search_hits")
    parser.add_argument("--output-name", default="combined")
    
    # NT params
    parser.add_argument("--nt-motif", action="append", default=[], help="NT Motif (PATTERN:COUNT)")
    parser.add_argument("--exclude-nt-motif", action="append", default=[], help="NT Exclude Motif")
    parser.add_argument("--base-content", action="append", default=[])
    parser.add_argument("--max-repeat", action="append", default=[])
    parser.add_argument("--nt-motif-mismatches", type=int, default=0)
    parser.add_argument("--motif-self-comp", action="append", default=[])
    parser.add_argument("--require-palindrome", action="store_true")
    parser.add_argument("--palindrome-min-len", type=int, default=8)
    parser.add_argument("--non-overlapping", action="store_true")
    parser.add_argument("--strand", dest="strand_selections", action="append", choices=["forward", "reverse", "both", "combined"], default=[])
    parser.add_argument("--combined-forward-len", type=int, default=0)
    parser.add_argument("--combined-reverse-len", type=int, default=0)
    parser.add_argument("--combined-overlap", type=int, default=0)

    # Filters
    parser.add_argument("--sequence-id", action="append", default=[])
    parser.add_argument("--sequence-class", action="append", dest="sequence_classes", choices=["chromosome", "scaffold"])
    parser.add_argument("--chromosome-number", action="append", dest="chromosome_numbers")
    parser.add_argument("--region", action="append", default=[])

    # Peptide params
    parser.add_argument("--pep-motif", action="append", default=[], help="Peptide Motif")
    parser.add_argument("--exclude-pep-motif", action="append", default=[])
    parser.add_argument("--amino-content", action="append", default=[])
    parser.add_argument("--pep-repeat", action="append", default=[])
    parser.add_argument("--frame", action="append", default=None, help="Reading frame (+0, -1, etc.)")
    parser.add_argument("--pep-mismatches", type=int, default=0)
    
    # Subwindow params
    parser.add_argument("--nt-sub-window", type=int, help="NT subwindow length")
    parser.add_argument("--nt-sub-offset", type=int, default=0, help="NT subwindow offset")
    parser.add_argument("--pep-sub-window", type=int, help="Peptide subwindow length")
    parser.add_argument("--pep-sub-offset", type=int, default=0, help="Peptide subwindow offset")

    return parser.parse_args()

def reverse_complement(seq: str) -> str:
    # Local rc
    return seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]

_WORKER_CONFIG: dict | None = None

def _init_worker(config: dict) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config

def _scan_sequence_worker(payload: Tuple[str, str]) -> Tuple[str, List[Dict[str, object]]]:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Worker config missing")
    seq_id, sequence = payload
    
    config = _WORKER_CONFIG
    overall_window = config["window"]
    step = config["step"]
    
    nt_sub_window = config["nt_sub_window"]
    nt_sub_offset = config["nt_sub_offset"]
    pep_sub_window = config["pep_sub_window"]
    pep_sub_offset = config["pep_sub_offset"]
    
    nt_hits = []
    strands = config.get("strand_selections") or ["forward"]
    if "both" in strands:
        strands = ["forward", "reverse"]
        
    if "forward" in strands:
        hits = scan_sequence(
            seq_id, sequence, nt_sub_window, step,
            motifs=config["nt_motifs"],
            base_constraints=config["base_constraints"],
            repeat_constraints=config["repeat_constraints"],
            strand_label="+",
            coord_transform=None,
            exclude_motifs=config["exclude_nt_motifs"],
            palindrome_config=config["palindrome_config"],
            region_filters=config["region_filters"],
            non_overlapping=False, # We filter non-overlapping at the end of combined search
            candidate_positions=None,
            max_motif_mismatches=config["nt_motif_mismatches"],
            self_comp_constraints=config["self_comp_constraints"]
        )
        nt_hits.extend(hits)
        
    if "reverse" in strands:
        rc_seq = reverse_complement(sequence)
        seq_len = len(sequence)
        
        def rev_transform(start: int, end: int) -> Tuple[int, int]:
            return seq_len - end + 1, seq_len - start
            
        hits = scan_sequence(
            seq_id, rc_seq, nt_sub_window, step,
            motifs=config["nt_motifs"],
            base_constraints=config["base_constraints"],
            repeat_constraints=config["repeat_constraints"],
            strand_label="-",
            coord_transform=rev_transform,
            exclude_motifs=config["exclude_nt_motifs"],
            palindrome_config=config["palindrome_config"],
            region_filters=config["region_filters"],
            non_overlapping=False, # We filter non-overlapping at the end of combined search
            candidate_positions=None,
            max_motif_mismatches=config["nt_motif_mismatches"],
            self_comp_constraints=config["self_comp_constraints"]
        )
        nt_hits.extend(hits)
        
    if "combined" in strands:
        hits = scan_combined_windows(
            seq_id, sequence,
            config["combined_forward_len"],
            config["combined_reverse_len"],
            config["combined_overlap"],
            step,
            motifs=config["nt_motifs"],
            base_constraints=config["base_constraints"],
            repeat_constraints=config["repeat_constraints"],
            exclude_motifs=config["exclude_nt_motifs"],
            palindrome_config=config["palindrome_config"],
            region_filters=config["region_filters"],
            non_overlapping=False, # We filter non-overlapping at the end of combined search
            candidate_positions=None,
            max_motif_mismatches=config["nt_motif_mismatches"],
            self_comp_constraints=config["self_comp_constraints"]
        )
        nt_hits.extend(hits)
        
    # Now check peptide constraints on these hits
    combined_hits = []
    pep_motifs = config["pep_motifs"]
    pep_excludes = config["pep_excludes"]
    pep_mismatches = config["pep_mismatches"]
    pep_content = config["pep_content"]
    pep_repeats = config["pep_repeats"]
    frames = config["frames"]
    
    for hit in nt_hits:
        strand = hit["strand"]
        
        # We found NT hit in some window. 
        # But we want to re-anchor it to an OVERALL window.
        # scan_sequence returns window_start relative to the sequence provided.
        # If it's reverse strand, it was RC'd.
        
        nt_start_in_scan = hit["window_start"]
        seq_len = len(sequence)
        
        # Candidate overall window start relative to the scanned sequence (RC'd if strand is -)
        can_overall_start = nt_start_in_scan - nt_sub_offset
        can_overall_end = can_overall_start + overall_window - 1
        
        # Current window sequence in the scan (RC'd if strand is -)
        scanned_seq = sequence if strand == "+" else reverse_complement(sequence)
        
        if can_overall_start < 1 or can_overall_end > len(scanned_seq):
            continue
            
        overall_subseq = scanned_seq[can_overall_start - 1 : can_overall_end]
        pep_subseq_local = overall_subseq[pep_sub_offset : pep_sub_offset + pep_sub_window]
        
        for f_strand, f_offset in frames:
            if f_strand == "+":
                subseq = pep_subseq_local
            else:
                subseq = reverse_complement(pep_subseq_local)
                    
            subseq = subseq[f_offset:]
            aa_seq = translate_sequence(subseq)
            
            if not aa_seq: continue
            
            if pep_excludes and any(pattern.search(aa_seq) for pattern in pep_excludes):
                continue
                
            skip_pep = False
            if pep_content:
                for limit in pep_content:
                    match_count = sum(1 for aa in aa_seq if aa in limit.bases)
                    pct = (match_count / len(aa_seq)) * 100
                    if not (limit.min_pct <= pct <= limit.max_pct):
                        skip_pep = True
                        break
            if skip_pep: continue
            
            if pep_repeats:
                for limit in pep_repeats:
                    if limit.pattern_regex.search(aa_seq):
                        skip_pep = True
                        break
            if skip_pep: continue
                
            match_records = []
            if pep_motifs:
                for motif in pep_motifs:
                    mismatch = match_motif_in_window(aa_seq, motif, pep_mismatches)
                    if mismatch is not None:
                        match_records.append((motif.raw, mismatch))
                if not match_records:
                    continue
            else:
                match_records.append(("ANY", 0))
                
            eff = calc_codon_efficiency(subseq)
            
            # Map overall window back to genome coords
            if strand == "+":
                overall_start_gen = can_overall_start
                overall_end_gen = can_overall_end
            else:
                overall_start_gen = seq_len - can_overall_end + 1
                overall_end_gen = seq_len - can_overall_start + 1
                
            comb_hit = hit.copy()
            comb_hit["window_start"] = overall_start_gen
            comb_hit["window_end"] = overall_end_gen
            comb_hit["window_sequence"] = overall_subseq
            comb_hit["pep_subwindow_sequence"] = pep_subseq_local
            comb_hit["pep_frame"] = f"{f_strand}{f_offset}"
            comb_hit["peptide_sequence"] = aa_seq
            comb_hit["matched_pep_motifs"] = [(m, mm) for m, mm in match_records]
            comb_hit["codon_efficiency"] = round(eff, 4)
            
            combined_hits.append(comb_hit)
            
    return seq_id, combined_hits

def filter_non_overlapping(hits: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Greedily filter hits to ensure no genomic overlap within each sequence."""
    if not hits: return []
    # Sort by sequence_id, then start
    hits.sort(key=lambda x: (str(x["sequence_id"]), int(x["window_start"])))
    
    final_hits = []
    last_end = {} # sequence_id -> last_end
    
    for hit in hits:
        sid = hit["sequence_id"]
        start = int(hit["window_start"])
        end = int(hit["window_end"])
        
        if sid not in last_end or start > last_end[sid]:
            final_hits.append(hit)
            last_end[sid] = end
            
    return final_hits

def main():
    args = parse_args()
    parser = argparse.ArgumentParser()
    
    nt_motifs = parse_motif_specs(parser, args.nt_motif)
    exclude_nt_motifs = parse_exclude_motifs(parser, args.exclude_nt_motif)
    base_constraints = parse_base_content_specs(parser, args.base_content)
    repeat_constraints = parse_max_repeat_specs(parser, args.max_repeat)
    self_comp_constraints = parse_motif_self_comp_specs(parser, args.motif_self_comp)
    region_filters = parse_region_specs(parser, args.region) if args.region else None
    chromosome_filter = parse_chromosome_numbers(parser, args.chromosome_numbers) if args.chromosome_numbers else None
    
    palindrome_config = {
        "enabled": args.require_palindrome,
        "min_len": args.palindrome_min_len,
    } if args.require_palindrome else None
    
    frames = iter_frames(args.frame)
    pep_motifs = build_pep_motif_patterns(args.pep_motif)
    pep_excludes = compile_pep_patterns(args.exclude_pep_motif)
    pep_content = parse_amino_content(args.amino_content)
    pep_repeats = parse_pep_max_repeats(args.pep_repeat)
    
    config = {
        "window": args.window,
        "step": args.step,
        "strand_selections": args.strand_selections,
        "non_overlapping": args.non_overlapping,
        
        "nt_motifs": nt_motifs,
        "exclude_nt_motifs": exclude_nt_motifs,
        "base_constraints": base_constraints,
        "repeat_constraints": repeat_constraints,
        "nt_motif_mismatches": args.nt_motif_mismatches,
        "self_comp_constraints": self_comp_constraints,
        "region_filters": region_filters,
        "palindrome_config": palindrome_config,
        
        "combined_forward_len": args.combined_forward_len or args.window,
        "combined_reverse_len": args.combined_reverse_len or args.window,
        "combined_overlap": args.combined_overlap,
        
        "frames": frames,
        "pep_motifs": pep_motifs,
        "pep_excludes": pep_excludes,
        "pep_content": pep_content,
        "pep_repeats": pep_repeats,
        "pep_mismatches": args.pep_mismatches,
        
        "nt_sub_window": args.nt_sub_window or args.window,
        "nt_sub_offset": args.nt_sub_offset,
        "pep_sub_window": args.pep_sub_window or args.window,
        "pep_sub_offset": args.pep_sub_offset,
    }
    
    worker_count = args.workers if args.workers > 0 else os.cpu_count() or 1
    
    out_dir = Path(f"output_{args.output_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.output_prefix}.tsv"
    
    all_hits = []
    
    with open(args.fasta, "r") as f:
        lines = f.readlines()
        
    sequences = []
    allowed_ids = set(args.sequence_id) if args.sequence_id else None
    
    for seq_id, seq in iter_nc_sequences(lines, args.sequence_classes, allowed_ids):
        norm_id = normalize_seq_name(seq_id)
        if chromosome_filter and not sequence_matches_chromosome_filter(norm_id, chromosome_filter):
            continue
        sequences.append((format_seq_id(seq_id), seq))
        
    if worker_count == 1:
        _init_worker(config)
        for index, (seq_id, seq) in enumerate(sequences, 1):
            print(f"[scanner] Processing {seq_id} ({index}/{len(sequences)})")
            _, hits = _scan_sequence_worker((seq_id, seq))
            all_hits.extend(hits)
    else:
        print(f"[scanner] Processing {len(sequences)} sequences with {worker_count} workers...")
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_worker, initargs=(config,)) as pool:
            for index, (seq_id, hits) in enumerate(pool.map(_scan_sequence_worker, sequences), 1):
                all_hits.extend(hits)
                print(f"[scanner] Completed sequence {index}/{len(sequences)}")
                
    if args.non_overlapping:
        print(f"[scanner] Filtering {len(all_hits)} hits for non-overlapping regions...")
        all_hits = filter_non_overlapping(all_hits)
                
    if not all_hits:
        print("No hits found.")
        return 0
        
    headers = [
        "sequence_id", "nt_strand", "window_start", "window_end", 
        "nt_subwindow_start", "nt_subwindow_end", "nt_motif_hits", "nt_base_content", "nt_max_repeats",
        "pep_subwindow_start", "pep_subwindow_end", "pep_frame", "peptide_sequence", 
        "matched_pep_motifs", "codon_efficiency", "is_self_complementary",
        "palindrome_hairpin_sequence", "palindrome_hairpin_length", "pep_subwindow_sequence", "window_sequence"
    ]
    
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for hit in all_hits:
            row = hit.copy()
            row["nt_strand"] = hit.get("strand", "")
            
            # Subwindow relative positions
            w_start = int(hit["window_start"])
            w_strand = hit["strand"]
            
            # Map subwindow coords relative to overall window
            if w_strand == "+":
                row["nt_subwindow_start"] = w_start + config["nt_sub_offset"]
                row["nt_subwindow_end"] = row["nt_subwindow_start"] + config["nt_sub_window"] - 1
                row["pep_subwindow_start"] = w_start + config["pep_sub_offset"]
                row["pep_subwindow_end"] = row["pep_subwindow_start"] + config["pep_sub_window"] - 1
            else:
                # In reverse strand, overall window is [w_start, w_end].
                # The sub-sequence was extracted from the RC sequence.
                # So the start of RC subwindow is mapped from the end of the genome subwindow.
                row["nt_subwindow_end"] = int(hit["window_end"]) - config["nt_sub_offset"]
                row["nt_subwindow_start"] = row["nt_subwindow_end"] - config["nt_sub_window"] + 1
                row["pep_subwindow_end"] = int(hit["window_end"]) - config["pep_sub_offset"]
                row["pep_subwindow_start"] = row["pep_subwindow_end"] - config["pep_sub_window"] + 1
            
            if "motif_hits" in row and row["motif_hits"]:
                row["nt_motif_hits"] = ", ".join(f"{k}({v})" for k, v in row["motif_hits"])
            else:
                row["nt_motif_hits"] = ""
                
            if "matched_pep_motifs" in row and row["matched_pep_motifs"]:
                row["matched_pep_motifs"] = ", ".join(f"{k}({mm}mm)" for k, mm in row["matched_pep_motifs"])
            else:
                row["matched_pep_motifs"] = ""
                
            if "base_percentages" in row and row["base_percentages"]:
                row["nt_base_content"] = ", ".join(f"{b}={p:.1f}%" for b, p in row["base_percentages"])
            if "max_runs" in row and row["max_runs"]:
                row["nt_max_repeats"] = ", ".join(f"{b}={r}" for b, r in row["max_runs"])
                
            writer.writerow(row)
            
    print(f"Done. Wrote {len(all_hits)} hits to {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
