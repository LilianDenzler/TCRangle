from __future__ import annotations

import os
import sys
import argparse
from typing import Dict, Tuple, List, Optional, Callable

import numpy as np
from Bio.PDB import (
    PDBParser, PDBIO, Select, is_aa, Structure, Model, Chain, Residue
)
from Bio.SeqUtils import seq1

# Local modules you already have (from previous messages)
#from TCR_NUMBERING.test.anarci_numbering import anarcii_number_pdb  # returns per-chain mapping + ctype
from tcr_pairing import pair_tcrs_by_interface   # interface-based pairing
import pymol2

# -----------------------------
# IMGT region definitions (IMGT positions)
# -----------------------------
CDR_FR_RANGES = {
    # Alpha-like (A)
    "A_FR1":   (3, 26),
    "A_CDR1":  (27, 38),
    "A_FR2":   (39, 55),
    "A_CDR2":  (56, 65),
    "A_FR3":   (66, 104),
    "A_CDR3":  (105, 117),
    "A_FR4":   (118, 125),
    # Beta-like (B)
    "B_FR1":   (3, 26),
    "B_CDR1":  (27, 38),
    "B_FR2":   (39, 55),
    "B_CDR2":  (56, 65),
    "B_FR3":   (66, 104),
    "B_CDR3":  (105, 117),
    "B_FR4":   (118, 125),
}
VARIABLE_RANGE = (1, 128)

# -----------------------------
# Helpers
# -----------------------------
def ensure_outdir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

class ChainResidueSelect(Select):
    """
    Select filter for writing PDB subsets. Provide:
      - allowed_chains: set[str]
      - residue_predicate(chain_id, residue) -> bool
    """
    def __init__(self, allowed_chains: Optional[set] = None,
                 residue_predicate: Optional[Callable[[str, Residue], bool]] = None):
        super().__init__()
        self.allowed_chains = allowed_chains
        self.residue_predicate = residue_predicate

    def accept_model(self, model):
        return 1 if model.id == 0 else 0  # first model only

    def accept_chain(self, chain):
        if self.allowed_chains is None:
            return 1
        return 1 if chain.id in self.allowed_chains else 0

    def accept_residue(self, residue):
        chain = residue.get_parent()
        cid = chain.id
        if self.residue_predicate is None:
            return 1
        return 1 if self.residue_predicate(cid, residue) else 0

def shift_residue_numbers(structure: Structure.Structure, offset: int = 1000) -> None:
    """Shift all residue sequence numbers by offset to avoid collisions during renumbering."""
    model0 = next(structure.get_models())
    for chain in model0:
        for residue in chain:
            hetflag, resseq, icode = residue.id
            residue.id = (hetflag, resseq + offset, icode)



def apply_imgt_renumbering(
    structure: Structure.Structure,
    per_chain_map: Dict[str, Dict[Tuple[int, str], Tuple[Tuple[int, str], str]]]
) -> None:
    """
    Apply IMGT renumbering to structure in place.
    Strategy:
      - First shift all residue IDs by +1000 (avoid collisions).
      - For each chain, for residues with a mapping -> set IMGT (new) id.
      - For non-mapped residues (e.g., constant domain), either keep shifted id
        or (if fill_nonmapped_sequentially=True) assign sequential numbers
        after the max IMGT number observed in that chain.
    """
    shift_residue_numbers(structure, offset=1000)
    model0 = next(structure.get_models())

    for chain in model0:
        cid = chain.id
        cmap = per_chain_map.get(cid, {})
        for enum,residue in enumerate(chain):
            hetflag, resseq, icode = residue.id
            old_key = (resseq - 1000, icode)  # undo shift to look up original
            if old_key in cmap:
                (new_resid, _aa) = cmap[old_key]
                assert _aa == seq1(residue.get_resname()), f"Residue name mismatch for chain {cid} res {old_key}: anarci {_aa} vs pdb {seq1(residue.get_resname())}"
                residue.id = (hetflag, int(new_resid[0]), str(new_resid[1]))
            else:
                raise ValueError(f"No IMGT mapping for chain {cid} residue {old_key} ({seq1(residue.get_resname())})")


def rename_two_chains_exact(structure: Structure.Structure,
                            old_a: str, old_b: str,
                            new_a: str, new_b: str) -> None:
    """
    Rename exactly two chains (old_a -> new_a, old_b -> new_b) in-place.
    Leaves all other chains untouched.
    """
    model0 = next(structure.get_models())
    for chain in model0:
        if chain.id == old_a:
            chain.id = new_a
        elif chain.id == old_b:
            chain.id = new_b

def rename_tcr_chains_for_pairs(structure: Structure.Structure,
                                pair_chain_ids: List[Tuple[str, str]],
                                other_chain_prefix: str = "X") -> Dict[str, str]:
    """
    Rename chain ids in-place to A,B (pair 1), C,D (pair 2), etc.
    Returns a dict old_id -> new_id for all chains touched.
    Non-paired chains keep their original id (or get prefixed with `other_chain_prefix` if rename needed).
    """
    model0 = next(structure.get_models())
    # Build renaming map
    remap: Dict[str, str] = {}
    next_letter_code = ord("A")
    for k, (aid, bid) in enumerate(pair_chain_ids, start=1):
        a_new = chr(next_letter_code); next_letter_code += 1
        b_new = chr(next_letter_code); next_letter_code += 1
        remap[aid] = a_new
        remap[bid] = b_new

    # Ensure uniqueness; if a new ID conflicts with an existing chain that’s not being remapped,
    # give that other chain a prefixed name.
    used_new = set(remap.values())
    for chain in model0:
        if chain.id in remap:
            continue
        if chain.id in used_new:
            # rename this non-paired chain to avoid collision
            new_id = f"{other_chain_prefix}{chain.id}"
            # ensure unique
            suffix = 1
            while new_id in used_new:
                new_id = f"{other_chain_prefix}{chain.id}{suffix}"
                suffix += 1
            remap[chain.id] = new_id
            used_new.add(new_id)

    # Apply remapping
    for chain in model0:
        if chain.id in remap:
            chain.id = remap[chain.id]
    return remap

def write_pdb_subset(
    structure: Structure.Structure,
    out_path: str,
    allowed_chains: Optional[set] = None,
    residue_predicate: Optional[Callable[[str, Residue], bool]] = None
) -> None:
    """Write a subset of the structure (first model) to out_path."""
    ensure_outdir(out_path)
    io = PDBIO()
    io.set_structure(structure)
    #remove Nones from allowed_chains
    if allowed_chains is not None:
        allowed_chains = {cid for cid in allowed_chains if cid is not None}
    io.save(out_path, select=ChainResidueSelect(allowed_chains, residue_predicate))

def extract_region_sequence(structure: Structure.Structure,
                            chain_id: str,
                            start: int,
                            end: int) -> str:
    model0 = next(structure.get_models())
    seq_chars = []
    chain = model0[chain_id]

    if start is None and end is None:
        for res in chain:
            seq_chars.append(seq1(res.get_resname()))
        return "".join(seq_chars)
    else:
        for res in chain:
            resnum = res.id[1]
            if start <= resnum <= end:
                seq_chars.append(seq1(res.get_resname()))
    return "".join(seq_chars)

def collect_cdr_fr_sequences(structure: Structure.Structure,
                             chain_map: Dict[str, str]) -> Dict[str, str]:
    """
    chain_map: mapping "alpha"->chain_id, "beta"->chain_id (actual IDs after renaming)
    Returns dict region_name -> sequence
    """
    seqs: Dict[str, str] = {}
    for region, (s, e) in CDR_FR_RANGES.items():
        if region.startswith("A_") and "alpha" in chain_map:
            cid = chain_map["alpha"]
        elif region.startswith("B_") and "beta" in chain_map:
            cid = chain_map["beta"]
        else:
            continue
        seqs[region] = extract_region_sequence(structure, cid, s, e)
        #add variable domain
    seqs["A_fv"] = extract_region_sequence(structure, chain_map["alpha"], 1, 128)
    seqs["B_fv"] = extract_region_sequence(structure, chain_map["beta"], 1, 128)
    seqs["A_full"] = extract_region_sequence(structure, chain_map["alpha"], None, None)
    seqs["B_full"] = extract_region_sequence(structure, chain_map["beta"], None, None)

    return seqs

def region_predicate_factory(regions: List[Tuple[str, Tuple[int, int]]],
                             chain_map: Dict[str, str]) -> Callable[[str, Residue], bool]:
    """
    regions: list of tuples like ("A_CDR1", (start, end))
    chain_map: {"alpha": "A", "beta": "B"}
    """
    intervals_by_chain: Dict[str, List[Tuple[int, int]]] = {}
    for rname, (s, e) in regions:
        if rname.startswith("A_") and "alpha" in chain_map:
            cid = chain_map["alpha"]
        elif rname.startswith("B_") and "beta" in chain_map:
            cid = chain_map["beta"]
        else:
            continue
        intervals_by_chain.setdefault(cid, []).append((s, e))

    def pred(chain_id: str, residue: Residue) -> bool:
        if chain_id not in intervals_by_chain:
            return False
        resnum = residue.id[1]
        for (s, e) in intervals_by_chain[chain_id]:
            if s <= resnum <= e:
                return True
        return False

    return pred
def _blend_with_white(rgb, factor=0.35):
    """Lighten an RGB tuple by blending toward white."""
    r, g, b = rgb
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)

def write_pymol_vis_script(
    per_tcr_imgt_pdb: str,
    alpha_chain_id: str,
    alpha_chain_type: str,
    beta_chain_id: str,
    beta_chain_type: str,
    out_pml: Optional[str] = None,
    object_name: Optional[str] = None,
) -> str:
    """
    Create a PyMOL .pml for an IMGT-renumbered per-TCR PDB.
    - Colors alpha/beta chains differently.
    - Variable domains (resi VARIABLE_RANGE) are lighter shades.
    - CDR1/2/3 are colored distinctly and labeled.
    - Chains are labeled "alpha"/"beta".
    Relies on global CDR_FR_RANGES and VARIABLE_RANGE.
    """
    import os

    base = os.path.splitext(os.path.basename(per_tcr_imgt_pdb))[0]
    if object_name is None:
        object_name = base
    if out_pml is None:
        out_pml = os.path.join(os.path.dirname(per_tcr_imgt_pdb), base + "_vis.pml")

    # Colors (RGB 0..1)
    alpha_rgb = (0.20, 0.45, 0.85)       # blue-ish
    beta_rgb  = (0.85, 0.35, 0.35)       # red-ish
    alpha_var_rgb = _blend_with_white(alpha_rgb, 0.35)
    beta_var_rgb  = _blend_with_white(beta_rgb, 0.35)
    cdr1_rgb = (1.00, 0.85, 0.10)        # golden
    cdr2_rgb = (0.10, 0.90, 0.70)        # teal
    cdr3_rgb = (0.85, 0.10, 0.85)        # magenta

    # Pull ranges from your globals
    a_cdr1 = CDR_FR_RANGES["A_CDR1"]; a_cdr2 = CDR_FR_RANGES["A_CDR2"]; a_cdr3 = CDR_FR_RANGES["A_CDR3"]
    b_cdr1 = CDR_FR_RANGES["B_CDR1"]; b_cdr2 = CDR_FR_RANGES["B_CDR2"]; b_cdr3 = CDR_FR_RANGES["B_CDR3"]
    var_lo, var_hi = VARIABLE_RANGE

    # Label midpoints
    var_mid = (var_lo + var_hi) // 2
    a1m = (a_cdr1[0] + a_cdr1[1]) // 2
    a2m = (a_cdr2[0] + a_cdr2[1]) // 2
    a3m = (a_cdr3[0] + a_cdr3[1]) // 2
    b1m = (b_cdr1[0] + b_cdr1[1]) // 2
    b2m = (b_cdr2[0] + b_cdr2[1]) // 2
    b3m = (b_cdr3[0] + b_cdr3[1]) // 2

    def rgb(v): return f"{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}"

    lines = []
    ap = lines.append

    # Load & view
    ap("reinitialize")
    ap(f'load "{per_tcr_imgt_pdb}", {object_name}')
    ap("hide everything")
    ap(f"show cartoon, {object_name}")
    ap("set cartoon_smooth_loops, on")
    ap("set cartoon_fancy_helices, on")
    ap("bg white")
    ap("set ray_opaque_background, off")

    # Define colors
    ap(f"set_color alpha_col, [{rgb(alpha_rgb)}]")
    ap(f"set_color beta_col,  [{rgb(beta_rgb)}]")
    ap(f"set_color alpha_var_col, [{rgb(alpha_var_rgb)}]")
    ap(f"set_color beta_var_col,  [{rgb(beta_var_rgb)}]")
    ap(f"set_color cdr1_col, [{rgb(cdr1_rgb)}]")
    ap(f"set_color cdr2_col, [{rgb(cdr2_rgb)}]")
    ap(f"set_color cdr3_col, [{rgb(cdr3_rgb)}]")

    # Selections with known chain IDs
    ap(f"select alpha_chain, {object_name} and chain {alpha_chain_id}")
    ap(f"select beta_chain,  {object_name} and chain {beta_chain_id}")

    # Base chain colors
    ap("color alpha_col, alpha_chain")
    ap("color beta_col,  beta_chain")

    # Variable (lighter)
    ap(f"select alpha_var, alpha_chain and resi {var_lo}-{var_hi}")
    ap(f"select beta_var,  beta_chain and resi {var_lo}-{var_hi}")
    ap("color alpha_var_col, alpha_var")
    ap("color beta_var_col,  beta_var")

    # CDRs (override colors)
    ap(f"select alpha_cdr1, alpha_chain and resi {a_cdr1[0]}-{a_cdr1[1]}")
    ap(f"select alpha_cdr2, alpha_chain and resi {a_cdr2[0]}-{a_cdr2[1]}")
    ap(f"select alpha_cdr3, alpha_chain and resi {a_cdr3[0]}-{a_cdr3[1]}")
    ap("color cdr1_col, alpha_cdr1")
    ap("color cdr2_col, alpha_cdr2")
    ap("color cdr3_col, alpha_cdr3")

    ap(f"select beta_cdr1, beta_chain and resi {b_cdr1[0]}-{b_cdr1[1]}")
    ap(f"select beta_cdr2, beta_chain and resi {b_cdr2[0]}-{b_cdr2[1]}")
    ap(f"select beta_cdr3, beta_chain and resi {b_cdr3[0]}-{b_cdr3[1]}")
    ap("color cdr1_col, beta_cdr1")
    ap("color cdr2_col, beta_cdr2")
    ap("color cdr3_col, beta_cdr3")

    # --- Labels (create them first) ---
    ap(f"label (alpha_chain and name CA and resi {var_mid}), '{alpha_chain_type}'")
    ap(f"label (beta_chain  and name CA and resi {var_mid}), '{beta_chain_type}'")

    ap("label first (alpha_cdr1 and name CA), 'CDR1'")
    ap("label first (alpha_cdr2 and name CA), 'CDR2'")
    ap("label first (alpha_cdr3 and name CA), 'CDR3'")
    ap("label first (beta_cdr1  and name CA), 'CDR1'")
    ap("label first (beta_cdr2  and name CA), 'CDR2'")
    ap("label first (beta_cdr3  and name CA), 'CDR3'")

    # --- Define label colors as named colors (so set label_color can use them) ---
    ap(f"set_color lbl_alpha_col, [{rgb(alpha_rgb)}]")
    ap(f"set_color lbl_beta_col,  [{rgb(beta_rgb)}]")
    ap(f"set_color lbl_cdr1_col,  [{rgb(cdr1_rgb)}]")
    ap(f"set_color lbl_cdr2_col,  [{rgb(cdr2_rgb)}]")
    ap(f"set_color lbl_cdr3_col,  [{rgb(cdr3_rgb)}]")

    # --- Make explicit selections for the label atoms we want to style ---
    ap(f"select sel_chain_alpha_label, (alpha_chain and name CA and resi {var_mid})")
    ap(f"select sel_chain_beta_label,  (beta_chain  and name CA and resi {var_mid})")
    ap("select sel_alpha_cdr1_label, first (alpha_cdr1 and name CA)")
    ap("select sel_alpha_cdr2_label, first (alpha_cdr2 and name CA)")
    ap("select sel_alpha_cdr3_label, first (alpha_cdr3 and name CA)")
    ap("select sel_beta_cdr1_label,  first (beta_cdr1  and name CA)")
    ap("select sel_beta_cdr2_label,  first (beta_cdr2  and name CA)")
    ap("select sel_beta_cdr3_label,  first (beta_cdr3  and name CA)")

    # --- Global defaults for labels (affects everything) ---
    ap("set label_outline_color, black")
    ap("set label_font_id, 7")

    # --- Per-selection overrides (size & color) ---
    # Bigger chain labels
    ap("set label_size, 25, sel_chain_alpha_label")
    ap("set label_size, 25, sel_chain_beta_label")
    ap("set label_color, lbl_alpha_col, sel_chain_alpha_label")
    ap("set label_color, lbl_beta_col,  sel_chain_beta_label")

    # Colored CDR labels (keep default size 14)
    ap("set label_color, lbl_cdr1_col, sel_alpha_cdr1_label")
    ap("set label_color, lbl_cdr2_col, sel_alpha_cdr2_label")
    ap("set label_color, lbl_cdr3_col, sel_alpha_cdr3_label")
    ap("set label_color, lbl_cdr1_col, sel_beta_cdr1_label")
    ap("set label_color, lbl_cdr2_col, sel_beta_cdr2_label")
    ap("set label_color, lbl_cdr3_col, sel_beta_cdr3_label")
    ap("set label_size, 14, sel_alpha_cdr1_label")
    ap("set label_size, 14, sel_alpha_cdr2_label")
    ap("set label_size, 14, sel_alpha_cdr3_label")
    ap("set label_size, 14, sel_beta_cdr1_label")
    ap("set label_size, 14, sel_beta_cdr2_label")
    ap("set label_size, 14, sel_beta_cdr3_label")

    # Clean side panel & zoom
    ap("disable alpha_cdr1; disable alpha_cdr2; disable alpha_cdr3")
    ap("disable beta_cdr1;  disable beta_cdr2;  disable beta_cdr3")
    ap("disable alpha_var;   disable beta_var")
    ap(f"zoom {object_name}, 2.0")
    with open(out_pml, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with pymol2.PyMOL() as pm:
        cmd = pm.cmd
        cmd.reinitialize()
        cmd.do(f'@{out_pml}')
        cmd.save(out_pml.replace(".pml", ".pse"))
    os.system(f"pymol -cq {out_pml.replace('.pml', '.pse')}")
    return out_pml

def renumber_all(input_pdb, per_chain_map, imgt_all_path):
    # Copy structure for renumbered-all
    parser = PDBParser(QUIET=True)
    structure_imgt_all = parser.get_structure("all_imgt", input_pdb)
    apply_imgt_renumbering(structure_imgt_all, per_chain_map)

    # IMPORTANT: Do NOT rename chains in the full IMGT file
    write_pdb_subset(structure_imgt_all, imgt_all_path, allowed_chains=None)  # entire structure
    print(f"[imgt] wrote full IMGT-numbered structure (original chain IDs preserved): {imgt_all_path}")
    #get sequence from pdb
    model_imgt=PDBParser(QUIET=True).get_structure("imgt",imgt_all_path)[0]
    sequences_imgt=[]
    sequences=[]
    for chain in model_imgt:
        seq_chars = []
        for res in chain:
            seq_chars.append(seq1(res.get_resname()))
        seq="".join(seq_chars)
        sequences_imgt.append((chain.id,seq))
    parser = PDBParser(QUIET=True)
    structure_all = parser.get_structure("all", input_pdb)
    for chain in structure_all[0]:
        seq_chars = []
        for res in chain:
            seq_chars.append(seq1(res.get_resname()))
        seq="".join(seq_chars)
        sequences.append((chain.id,seq))
    assert sequences==sequences_imgt
# -----------------------------
# Main pipeline
# -----------------------------
def process_pdb(
    input_pdb: str,
    out_prefix: Optional[str],
    contact_cutoff: 5.0,
    min_contacts: 50,
    write_cdr_pdb: bool=False,
    write_fr_pdb: bool=False,
    write_fv: bool = False,
    write_splits: bool = False,
    legacy_anarci: bool = True,
    visualize: bool = False,
    write_fasta: bool = False,
    write_germlines: bool = False,
) -> Dict[str, object]:
    """
    Orchestrates:
      1) Pairing (if multiple TCRs) → split into *_1, *_2... per TCR pair (unpaired skipped but logged).
      2) IMGT renumber full original file (chains renamed by pair index if >1 TCR).
      3) For each TCR: write per-TCR full (IMGT), variable-only, optional CDR-only/FR-only; export sequences.

    Returns:
      dict with all generated paths, e.g. {
        "imgt_all": "X_imgt.pdb",
        "germlines": "X_germlines.txt" or None,
        "splits": ["X_1.pdb", ...],
        "pairs": [
           {
             "index": 1,
             "alpha_chain": "H",
             "beta_chain": "L",
             "alpha_label": "A",
             "beta_label": "B",
             "files": {
               "per_tcr_full": "X_pair1_imgt.pdb",
               "variable": "X_pair1_imgt_fv.pdb" or None,
               "cdrs": "X_pair1_imgt_CDRs.pdb" or None,
               "frs": "X_pair1_imgt_FRs.pdb" or None,
               "pml": "X_pair1_imgt_vis.pml" or None
             }
           },
           ...
        ],
        "fasta": "X_cdr_fr_seqs.fasta" or None
      }
    """
    base = out_prefix or os.path.splitext(input_pdb)[0]
    parser = PDBParser(QUIET=True)
    structure_all = parser.get_structure("all", input_pdb)

    outputs: Dict[str, object] = {
        "imgt_all": None,
        "germlines": None,
        "splits": [],
        "pairs": [],
        "fasta": None,
    }

    # Pairing on the original file (interface-based)
    pairs, per_chain_map, germline_info = pair_tcrs_by_interface(
        input_pdb, contact_cutoff=contact_cutoff, min_contacts=min_contacts, legacy_anarci=legacy_anarci
    )

    # Optional germlines file
    if write_germlines:
        if not germline_info:
            print("[germline] No germline information available, running legacy ANARCI. This will not change the numbering")
            _, _, germline_info = pair_tcrs_by_interface(
                input_pdb, contact_cutoff=contact_cutoff, min_contacts=min_contacts, legacy_anarci=True
            )
        germlines_path = f"{base}_germlines.txt"
        with open(germlines_path, "w") as f:
            for cid, g in germline_info.items():
                f.write(f"{cid}\t{g}\n")
        outputs["germlines"] = germlines_path

    # Renumber full file with IMGT (do NOT rename chains here)
    imgt_all_path = f"{base}_imgt.pdb"
    renumber_all(input_pdb, per_chain_map, imgt_all_path)
    print(f"[imgt] wrote full IMGT-numbered structure (original chain IDs preserved): {imgt_all_path}")
    outputs["imgt_all"] = imgt_all_path

    # Optionally split multi-TCR into separate files *_1, *_2, ...
    if write_splits:
        if len(pairs) > 1:
            for idx, pair in enumerate(pairs, start=1):
                aid = pair["alpha_chain"]
                bid = pair["beta_chain"]
                out_path = f"{base}_{idx}.pdb"
                write_pdb_subset(structure_all, out_path, allowed_chains={aid, bid})
                print(f"[split] wrote {out_path} with chains {aid}-{bid}")
                outputs["splits"].append(out_path)
        elif len(pairs) == 1:
            print("[split] single TCR detected; no split file copies needed.")
        else:
            print("[split] No TCR pairs detected by interface (will still renumber full file).")

    # Per-TCR outputs
    seq_fasta_lines: List[str] = []
    if pairs:
        for idx, pair in enumerate(pairs, start=1):
            # Fresh structure and renumbering for per-TCR file
            s = parser.get_structure(f"pair{idx}", input_pdb)
            apply_imgt_renumbering(s, per_chain_map)

            ctype_a = pair["alpha_chain_type"]
            ctype_b = pair["beta_chain_type"]
            aid = pair["alpha_chain"]
            bid = pair["beta_chain"]

            if {ctype_a, ctype_b} == {"A", "B"}:
                new_a, new_b = "A", "B"
            elif {ctype_a, ctype_b} == {"G", "D"}:
                new_a, new_b = "G", "D"
            else:
                new_a, new_b = "A", "B"

            # Rename only for per-TCR/variable files
            rename_two_chains_exact(s, aid, bid, new_a, new_b)

            # Per-TCR full
            if len(pairs) == 1:
                per_tcr_full = imgt_all_path
            else:
                per_tcr_full = f"{base}_pair{idx}_imgt.pdb"
                write_pdb_subset(s, per_tcr_full, allowed_chains={new_a, new_b})
                print(f"[pair] wrote IMGT-numbered TCR {idx} ({new_a}/{new_b}): {per_tcr_full}")

            # Visualization (optional)
            pml_path = None
            if visualize:
                pml_path = write_pymol_vis_script(
                    per_tcr_imgt_pdb=per_tcr_full,
                    alpha_chain_id=new_a,
                    alpha_chain_type=ctype_a,
                    beta_chain_id=new_b,
                    beta_chain_type=ctype_b,
                )

            # Variable-only (1..128)
            def var_pred(chain_id: str, residue: Residue) -> bool:
                resnum = residue.id[1]
                return 1 <= resnum <= 128

            if len(pairs) == 1:
                per_tcr_var = f"{base}_imgt_fv.pdb"
            else:
                per_tcr_var = f"{base}_pair{idx}_imgt_fv.pdb"

            variable_path = None
            if write_fv:
                write_pdb_subset(s, per_tcr_var, allowed_chains={new_a, new_b}, residue_predicate=var_pred)
                print(f"[pair] wrote variable-only TCR {idx} ({new_a}/{new_b}): {per_tcr_var}")
                variable_path = per_tcr_var

            # Sequences (CDR/FR)
            chain_map = {"alpha": new_a, "beta": new_b}
            seqs = collect_cdr_fr_sequences(s, chain_map)
            for region, seq in seqs.items():
                seq_fasta_lines.append(f">{os.path.basename(per_tcr_full)}|{region}")
                seq_fasta_lines.append(seq if seq else "-")

            # CDR-only (optional)
            cdrs_path = None
            if write_cdr_pdb:
                cdr_regions = [(k, v) for k, v in CDR_FR_RANGES.items() if k.endswith("CDR1") or k.endswith("CDR2") or k.endswith("CDR3")]
                pred = region_predicate_factory(cdr_regions, chain_map)
                if len(pairs) == 1:
                    per_tcr_cdr = f"{base}_imgt_CDRs.pdb"
                else:
                    per_tcr_cdr = f"{base}_pair{idx}_imgt_CDRs.pdb"
                write_pdb_subset(s, per_tcr_cdr, allowed_chains={new_a, new_b}, residue_predicate=pred)
                print(f"[pair] wrote CDR-only TCR {idx} ({new_a}/{new_b}): {per_tcr_cdr}")
                cdrs_path = per_tcr_cdr

            # FR-only (optional)
            frs_path = None
            if write_fr_pdb:
                fr_regions = [(k, v) for k, v in CDR_FR_RANGES.items() if "FR" in k]
                pred = region_predicate_factory(fr_regions, chain_map)
                if len(pairs) == 1:
                    per_tcr_fr = f"{base}_imgt_FRs.pdb"
                else:
                    per_tcr_fr = f"{base}_pair{idx}_imgt_FRs.pdb"
                write_pdb_subset(s, per_tcr_fr, allowed_chains={new_a, new_b}, residue_predicate=pred)
                print(f"[pair] wrote FR-only TCR {idx} ({new_a}/{new_b}): {per_tcr_fr}")
                frs_path = per_tcr_fr

            # Collect per-pair outputs
            outputs["pairs"].append({
                "index": idx,
                "alpha_chain": aid,
                "beta_chain": bid,
                "alpha_label": new_a,
                "beta_label": new_b,
                "files": {
                    "full": per_tcr_full,
                    "variable": variable_path,
                    "cdrs": cdrs_path,
                    "frs": frs_path,
                    "pml": pml_path,
                }
            })
    else:
        print("[pair] No TCR pairs were emitted; skipping per-TCR outputs.")

    # FASTA (optional)
    if write_fasta and seq_fasta_lines:
        seq_out = f"{base}_cdr_fr_seqs.fasta"
        ensure_outdir(seq_out)
        with open(seq_out, "w") as fh:
            fh.write("\n".join(seq_fasta_lines) + "\n")
        print(f"[seq] wrote CDR/FR sequences: {seq_out}")
        outputs["fasta"] = seq_out

    return outputs
# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Split, IMGT-renumber, and export TCR structures and CDR/FRs.")
    ap.add_argument("input_pdb", help="Input PDB path (may contain multiple TCRs)")
    ap.add_argument("--out-folder", default=None, help="Output Folder (default: input filename without .pdb)")
    ap.add_argument("--contact-cutoff", type=float, default=5.0, help="Å cutoff for heavy-atom contacts")
    ap.add_argument("--min-contacts", type=int, default=50, help="Minimum interface contacts to consider a valid pair")
    ap.add_argument("--write-cdr-pdb", action="store_true", help="Also write per-TCR CDR-only PDB files")
    ap.add_argument("--write-fr-pdb", action="store_true", help="Also write per-TCR FR-only PDB files")
    ap.add_argument("--write-fv", action="store_true", help="write variable-only PDB files (default: always)")
    ap.add_argument("--legacy_anarci", action="store_true", help="use ANARCI version 1, i.e. legacu version for numbering")
    ap.add_argument("--vis", action="store_true", help="visualise with pymol")
    ap.add_argument("--write_fasta", action="store_true", help="write fasta file with cdr/fr sequences and full sequences")
    ap.add_argument("--write_germlines", action="store_true", help="write most probable germline for each chain to a text file")
    args = ap.parse_args()
    if args.out_folder:
        pdb_name=os.path.basename(args.input_pdb)
        pdb_name=pdb_name.replace('.pdb','')
        out_prefix = os.path.join(args.out_folder, pdb_name)

    process_pdb(
        input_pdb=args.input_pdb,
        out_prefix=out_prefix,
        contact_cutoff=args.contact_cutoff,
        min_contacts=args.min_contacts,
        write_cdr_pdb=args.write_cdr_pdb,
        write_fr_pdb=args.write_fr_pdb,
        write_fv=args.write_fv,
        legacy_anarci=args.legacy_anarci,
        write_splits=False,
        visualize=args.vis,
        write_fasta=args.write_fasta,
        write_germlines=args.write_germlines,
    )

if __name__ == "__main__":
    main()
