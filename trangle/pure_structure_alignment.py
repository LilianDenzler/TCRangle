import biotite.structure as struc
import biotite.structure

import biotite.structure.io as strucio
import warnings
import numpy as np

def align_and_score_internal(static_pdb_file, mobile_pdb_file, output_pdb_file):
    """
    Performs structural alignment and calculates the TM-score using
    biotite's internal functions.

    Args:
        static_pdb_file (str): Path to the reference PDB file (will not move).
        mobile_pdb_file (str): Path to the PDB file to be aligned.
        output_pdb_file (str): Path to save the aligned mobile structure.
    """
    print(f"Loading structures: '{static_pdb_file}' and '{mobile_pdb_file}'...")
    # A warning is raised if the file has no model records, which we can ignore.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        static_structure = strucio.load_structure(static_pdb_file, model=1)
        mobile_structure = strucio.load_structure(mobile_pdb_file, model=1)

    # We will align the C-alpha atoms for a backbone-based alignment
    static_ca = static_structure[static_structure.atom_name == "CA"]
    mobile_ca = mobile_structure[mobile_structure.atom_name == "CA"]

    # --- Step 1: Structural Homology Alignment ---
    # This function finds homologous regions and superimposes them.
    # It returns the transformed mobile structure and the indices of aligned atoms.
    # '3di' uses dihedral angles, which is great for finding structural similarity.
    print("Performing structural homology alignment...")
    superimposed, _, ref_indices, sub_indices = struc.superimpose_structural_homologs(fixed=static_ca, mobile=mobile_ca, max_iterations=1)
    score=struc.tm_score(static_ca, superimposed, ref_indices, sub_indices)
    print(score)

    print(f"✅ Alignment complete.")
    print(f"   Internal Alignment Score: {score}")

    # --- Step 3: Apply transformation to the full mobile structure and save ---
    # The transformation was calculated on the C-alpha atoms, but we can apply
    # it to the entire structure for visualization.

    mobile_structure_aligned = superimposed

    strucio.save_structure(output_pdb_file, mobile_structure_aligned)
    print(f"💾 Aligned full-atom structure saved to '{output_pdb_file}'")


if __name__ == '__main__':
    # --- Define your input and output files ---
    STATIC_PDB = "/workspaces/Graphormer/TRangle/data/consensus_A.pdb"
    MOBILE_PDB = "/workspaces/Graphormer/TRangle/imgt_variable/3vxr.pdb"
    OUTPUT_PDB = 'aligned_internal_biotite.pdb'

    try:
        align_and_score_internal(STATIC_PDB, MOBILE_PDB, OUTPUT_PDB)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
