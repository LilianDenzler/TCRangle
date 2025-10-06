import os
import numpy as np
import pandas as pd
import biotite.structure as bts
import biotite.structure.io as btsio
import matplotlib.pyplot as plt
import warnings
from anarci import number
from Bio.SeqUtils import seq1
from pathlib import Path
from trangle.numbering import process_pdb
# Suppress PDB parsing warnings for cleaner output
warnings.filterwarnings("ignore", ".*is discontinuous.*")


#package home path
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data","consensus_output")

ranges_imgt = {
    "AFR1": [3,26],
    "ACDR1": [27,38],
    "AFR2": [39,55],
    "ACDR2": [56,65],
    "AFR3": [66,104],
    "ACDR3": [105,117],
    "AFR4": [118,125],
    "BFR1": [3,26],
    "BCDR1": [27,38],
    "BFR2": [39,55],
    "BCDR2": [56,65],
    "BFR3": [66,104],
    "BFR4": [118,125],
    "BCDR3": [105,117]
    }

def imgt_numbering(pdb_file_path, output_dir="."):
    outputs=process_pdb(
        input_pdb=pdb_file_path,
        out_prefix=output_dir,
        write_fv= True
        )
    full_imgt=outputs["pairs"]["files"]["full"]
    variable_pdb_imgt=outputs["pairs"]["files"]["variable"]
    return full_imgt, variable_pdb_imgt

def determine_consensus_atoms(stats_df, ranges_imgt):
    #get residue numbers with mean rmsd < 2.0 and stddev < 2.0
    consensus_alignment_residues = stats_df[(stats_df['Mean_RMSD'] < 2.0) & (stats_df['Std_Dev'] < 2.0)].index.tolist()
    cdr_ranges_imgt = {k: v for k, v in ranges_imgt.items() if "CDR" in k}
    consensus_alignment_residues_non_cdr=[]
    for res in consensus_alignment_residues:
        in_cdr=False
        for cdr, (start, end) in cdr_ranges_imgt.items():
            if start <= res <= end:
                in_cdr=True
                break
        if not in_cdr:
            consensus_alignment_residues_non_cdr.append(res)
    return consensus_alignment_residues_non_cdr

# --- Helper function for PyMOL CGO Arrow ---
def add_cgo_arrow(start, end, color, radius=0.3):
    """Generates a CGO string for a PyMOL arrow object."""
    return f"""[
        cgo.CYLINDER,{start[0]:.3f},{start[1]:.3f},{start[2]:.3f},
                     {end[0]:.3f},{end[1]:.3f},{end[2]:.3f},
                     {radius},
                     {color[0]},{color[1]},{color[2]},
                     {color[0]},{color[1]},{color[2]},
        cgo.CONE,{end[0]:.3f},{end[1]:.3f},{end[2]:.3f},
                 {start[0]:.3f},{start[1]:.3f},{start[2]:.3f},
                 {radius*1.5},0.0,
                 {color[0]},{color[1]},{color[2]},
                 {color[0]},{color[1]},{color[2]},1.0
    ]"""

# --- Corrected PCA Calculation Function ---
def generate_pca_axes(consensus_pdb, consensus_alignment_residues):
    """
    Calculates the principal component axes and centroid for a set of residues.
    """
    print(f"Running PCA on {os.path.basename(consensus_pdb)}...")
    structure = btsio.load_structure(consensus_pdb)

    # Correctly filter for C-alpha atoms within the consensus residue list
    ca_mask = (structure.atom_name == "CA") & np.isin(structure.res_id, consensus_alignment_residues)
    ca_atoms = structure[ca_mask]

    if ca_atoms.array_length() == 0:
        print("Warning: No atoms found for PCA calculation. Returning None.")
        return None, None

    coords = ca_atoms.coord

    # Centroid is the mean of the coordinates
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # Calculate PCA
    covariance_matrix = np.cov(centered_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort axes by importance (largest eigenvalue first)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_axes = eigenvectors[:, sorted_indices]

    print(f"  Centroid calculated at: {centroid.tolist()}")
    print(f"  PC1 direction: {principal_axes[:, 0].tolist()}")
    print(f"  PC2 direction: {principal_axes[:, 1].tolist()}")

    return principal_axes, centroid

def save_consensus_with_pseudoatoms(consensus_pdb_path, principal_axes, centroid, output_dir="."):
    """
    Loads the consensus structure and saves a new PDB file containing
    pseudoatoms for the centroid and PCA vector endpoints.
    """
    consensus_structure = btsio.load_structure(consensus_pdb_path)
    scale = 20.0 # Same scale as the PyMOL script for consistency

    # Define coordinates for the new pseudoatoms
    pc1_end = centroid + scale * principal_axes[:, 0]
    pc2_end = centroid + scale * principal_axes[:, 1]

    # Create biotite.Atom objects for the pseudoatoms
    # They are created as HETATM by default if res_name is not standard
    centroid_atom = bts.Atom(
        coord=centroid, atom_name="CEN", res_id=900, res_name="CEN",
        chain_id="Z", element="X"
    )
    pc1_atom = bts.Atom(
        coord=pc1_end, atom_name="PC1", res_id=901, res_name="PC1",
        chain_id="Z", element="X"
    )
    pc2_atom = bts.Atom(
        coord=pc2_end, atom_name="PC2", res_id=902, res_name="PC2",
        chain_id="Z", element="X"
    )

    # Combine the original structure with the new pseudoatoms
    # The '+' operator concatenates AtomArray objects
    combined_structure = consensus_structure + bts.array([centroid_atom, pc1_atom, pc2_atom])

    # Define the output path and save the new PDB file
    output_pdb_path = os.path.join(output_dir, "average_structure_with_pca.pdb")
    btsio.save_structure(output_pdb_path, combined_structure)
    print(f"Saved consensus structure with PCA pseudoatoms to: {output_pdb_path}")
    return output_pdb_path

# --- New PyMOL Visualization Function ---
def pymol_visualise_consensus_pca(consensus_pdb, principal_axes, centroid, output_dir="."):
    """
    Generates a PyMOL script to visualize the consensus structure,
    its centroid, and the first two principal component axes.
    """
    pdb_name = Path(consensus_pdb).stem
    out_prefix = pdb_name
    vis_folder = output_dir
    scale = 20.0  # Scale factor to control the length of the PCA arrows

    # Define end points for the first two PCA vectors
    pc1_end = centroid + scale * principal_axes[:, 0]
    pc2_end = centroid + scale * principal_axes[:, 1]

    script = f"""
import numpy as np
from pymol import cmd, cgo
cmd.load("{consensus_pdb}","{pdb_name}")
cmd.bg_color("white")
cmd.hide("everything","all")
cmd.show("cartoon","{pdb_name}")
cmd.color("gray70","{pdb_name}")
cmd.set("cartoon_transparency", 0.3, "{pdb_name}")

# Show centroid
cmd.pseudoatom("centroid_{pdb_name}", pos={list(centroid)}, color="red")
cmd.show("spheres","centroid_{pdb_name}")
cmd.set("sphere_scale", 0.6, "centroid_{pdb_name}")

# Create CGO arrows for PCA axes
cmd.load_cgo({add_cgo_arrow(centroid, pc1_end, (0.2, 0.5, 1.0))}, "PC1_{pdb_name}")
cmd.load_cgo({add_cgo_arrow(centroid, pc2_end, (0.1, 0.8, 0.1))}, "PC2_{pdb_name}")

cmd.orient()
cmd.zoom("all", 1.2)
cmd.png("{os.path.join(vis_folder, out_prefix + "_pca_vis.png")}", dpi=300, ray=1)
cmd.save("{os.path.join(vis_folder, out_prefix + "_pca_vis.pse")}")
cmd.quit()
"""
    vis_script_path = os.path.join(vis_folder, out_prefix + "_pca_vis.py")
    with open(vis_script_path, "w") as f:
        f.write(script)
    print(f"Saved PyMOL PCA visualization script to: {vis_script_path}")
    os.system(f"pymol -cq {vis_script_path}")
    return vis_script_path


def align_structures_biotite(pdb_files, ref_pdb_file, chain_name, output_dir="."):
    """
    Performs structural alignment, calculates per-residue scores,
    and generates an average structure.
    """
    #1. renumber reference structure as imgt
    _,ref_pdb_file =imgt_numbering(ref_pdb_file, output_dir=output_dir)
    # 1. Load reference structure and filter for C-alpha atoms of the specified chain
    ref_structure_full = btsio.load_structure(ref_pdb_file)
    ref_structure = ref_structure_full[ref_structure_full.chain_id == chain_name]
    ref_ca = ref_structure[ref_structure.atom_name == "CA"]

    # Initialize DataFrames to store results
    pdb_basenames = [os.path.basename(f) for f in pdb_files]
    aligned_coords_df = pd.DataFrame(index=pdb_basenames, columns=ref_ca.res_id)
    distance_df = pd.DataFrame(index=pdb_basenames, columns=ref_ca.res_id)

    print(f"Aligning {len(pdb_files)} structures to {os.path.basename(ref_pdb_file)} on chain {chain_name}...")

    for pdb_file in pdb_files:
        current_pdb_name = os.path.basename(pdb_file)
        imgt_renum_pdb, pdb_file=imgt_numbering(pdb_file, output_dir=output_dir)
        mobile_structure_full = btsio.load_structure(pdb_file)
        mobile_structure = mobile_structure_full[mobile_structure_full.chain_id == chain_name]
        mobile_ca = mobile_structure[mobile_structure.atom_name == "CA"]

        if mobile_ca.array_length() == 0:
            print(f"Warning: No C-alpha atoms found for chain '{chain_name}' in {current_pdb_name}. Skipping.")
            continue

        # 2. Superimpose mobile C-alphas onto reference C-alphas
        fitted_mobile, transform, ref_indices, mobile_indices = bts.superimpose_structural_homologs(
            fixed=ref_ca, mobile=mobile_ca, max_iterations=1
        )
        score = bts.tm_score(ref_ca, fitted_mobile, ref_indices, mobile_indices)
        print(f"  TM-score ({chain_name}): {score:.4f}")

        # 3. Store aligned coordinates and calculate per-residue distances
        aligned_ref_res_ids = ref_ca[ref_indices].res_id
        aligned_mobile_res_ids = fitted_mobile[mobile_indices].res_id
        assert len(aligned_ref_res_ids) == len(aligned_mobile_res_ids), "Mismatched aligned residues"

        for res_id_mobile, res_id_ref in zip(aligned_mobile_res_ids, aligned_ref_res_ids):
            coord_mobile = fitted_mobile[fitted_mobile.res_id == res_id_mobile].coord[0]
            coord_ref = ref_ca[ref_ca.res_id == res_id_ref].coord[0]

            aligned_coords_df.loc[current_pdb_name, res_id_ref] = tuple(coord_mobile)
            distance = bts.distance(coord_ref, coord_mobile)
            distance_df.loc[current_pdb_name, res_id_ref] = distance
        # Clean up temporary files
        if os.path.exists(imgt_renum_pdb):
            os.remove(imgt_renum_pdb)
        if os.path.exists(pdb_file):
            os.remove(pdb_file)

    # 4. Calculate statistics for per-residue distances
    stats_df = pd.DataFrame({
        'Mean_RMSD': distance_df.mean(axis=0),
        'Std_Dev': distance_df.std(axis=0)
    }).sort_index()
    stats_file = os.path.join(output_dir, "per_residue_alignment_scores.csv")
    stats_df.to_csv(stats_file)
    print(f"\nSaved per-residue alignment scores to: {stats_file}")
    consensus_alignment_residues=determine_consensus_atoms(stats_df, ranges_imgt)
    print(f"Consensus atoms determined: {consensus_alignment_residues}")
    #write to file
    consensus_file = os.path.join(output_dir, "consensus_alignment_residues.txt")
    with open(consensus_file, "w") as f:
        f.write(",".join(map(str, consensus_alignment_residues)))
    print(f"Saved consensus alignment residues to: {consensus_file}")

    # 5. Plot the per-residue alignment scores
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    # --- NEW CODE BLOCK FOR PLOT BACKGROUNDS ---
    # Annotate secondary structure for the entire reference structure
    sse = bts.annotate_sse(ref_structure)
    sheet_res_list = []
    start_sheet=None
    end_sheet=None
    print(sse)
    assert len(ref_ca.res_id) == len(sse), "SSE annotation length mismatch"
    for res_id, s in zip(ref_ca.res_id, sse):
        if s == 'b' and start_sheet is None:
            start_sheet=res_id
            print(f"  Start of beta-sheet at residue {res_id}")
        if s != 'b' and start_sheet is not None and end_sheet is None:
            print(f"  End of beta-sheet at residue {res_id-1}")
            end_sheet=res_id-1
            sheet_res_list.append((start_sheet,end_sheet))
            start_sheet=None
            end_sheet=None

    for start, end in sheet_res_list:
        ax.axvspan(start, end, color='red', alpha=0.15, zorder=0,
                   label='Beta-Sheet' if 'Beta-Sheet' not in ax.get_legend_handles_labels()[1] else "")


    # Highlight CDRs using ANARCI
    cdr_colors = {"ACDR1": "gold", "ACDR2": "lightgreen", "ACDR3": "cyan","BCDR1": "gold", "BCDR2": "lightgreen", "BCDR3": "cyan"}
    for cdr_name, (start, end) in ranges_imgt.items():
        if cdr_name in cdr_colors:
            if chain_name in cdr_name:
                ax.axvspan(start, end, color=cdr_colors[cdr_name], alpha=0.3, zorder=0,
                            label=cdr_name if cdr_name not in ax.get_legend_handles_labels()[1] else "")
    # --- END PLOT BACKGROUNDS ---

    ax.errorbar(
        stats_df.index, stats_df['Mean_RMSD'], yerr=stats_df['Std_Dev'],
        fmt='-o', markersize=4, capsize=3, color='royalblue', ecolor='lightskyblue',
        label='Mean RMSD'
    )
    ax.set_xlabel("Residue ID (IMGT Numbering)")
    ax.set_ylabel("Average Distance (Å)")
    ax.set_title(f"Per-Residue Alignment Score vs. Reference (Chain {chain_name})")
    ax.legend(loc='upper left', fontsize='medium')
    ax.set_xlim(stats_df.index.min() - 1, stats_df.index.max() + 1)
    plot_file = os.path.join(output_dir, "per_residue_rmsd_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved alignment score plot to: {plot_file}")

    # 6. Calculate and save the average structure
    new_coords = []
    for res_id in ref_ca.res_id:
        if res_id in aligned_coords_df.columns:
            coords_array = np.array(aligned_coords_df[res_id].dropna().tolist())
            if len(coords_array) > 0:
                avg_coord = np.mean(coords_array, axis=0)
            else:
                avg_coord = np.array([np.nan, np.nan, np.nan])
        else:
            avg_coord = np.array([np.nan, np.nan, np.nan])
        new_coords.append(avg_coord)

    avg_structure = ref_ca.copy()
    avg_structure.coord = np.array(new_coords)
    avg_structure = avg_structure[~np.isnan(avg_structure.coord).any(axis=1)]

    avg_pdb_file = os.path.join(output_dir, "average_structure.pdb")
    btsio.save_structure(avg_pdb_file, avg_structure)
    print(f"Saved average C-alpha structure to: {avg_pdb_file}")
    principal_axes, centroid = generate_pca_axes(avg_pdb_file, consensus_alignment_residues)
    pymol_visualise_consensus_pca(avg_pdb_file, principal_axes, centroid, output_dir)
    save_consensus_with_pseudoatoms(avg_pdb_file, principal_axes, centroid, output_dir)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    folder = "/workspaces/Graphormer/TRangle/imgt_variable"
    REFERENCE_PDB = "/workspaces/Graphormer/TRangle/imgt_variable/2cdf.pdb"
    output_dir = data_path

    # --- SCRIPT EXECUTION ---
    if not os.path.exists(folder):
        print(f"Error: PDB folder not found at {folder}")
    else:
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        pdb_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdb")]
        pdb_files = [f for f in pdb_files if os.path.basename(f) != os.path.basename(REFERENCE_PDB)]

        alpha_out=os.path.join(output_dir, "chain_A")
        if not os.path.exists(alpha_out): os.makedirs(alpha_out)

        beta_out=os.path.join(output_dir, "chain_B")
        if not os.path.exists(beta_out): os.makedirs(beta_out)

        print("\n--- Processing Chain A ---")
        align_structures_biotite(pdb_files, REFERENCE_PDB, chain_name="A",output_dir=alpha_out)
        print("\n--- Processing Chain B ---")
        align_structures_biotite(pdb_files, REFERENCE_PDB, chain_name="B",output_dir=beta_out)
        print("\nAll done!")


