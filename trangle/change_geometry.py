import os
import numpy as np
import warnings
from pathlib import Path
import math
import json
import tempfile
import biotite.structure as bts
import biotite.structure.io as btsio
from trangle.anarci_numbering import variable_renumber
# Suppress PDB parsing warnings for cleaner output
warnings.filterwarnings("ignore", ".*is discontinuous.*")
#get folder of current file
Trangle_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data","consensus_output")
out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"output")


def write_renumbered_fv(out_path, in_path):
    imgt_pdb = os.path.join(out_path)
    variable_pdb_imgt = os.path.join(out_path.replace(".pdb", "_fv.pdb"))
    A_chain, B_chain, imgt_path=variable_renumber(in_path, imgt_pdb, variable_pdb_imgt)
    return out_path

# ========================
# Geometry and Math Helpers
# ========================

def normalize(v):
    """Normalizes a vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def build_geometry_from_angles(BA, BC1, BC2, AC1, AC2, dc):
    """
    Constructs a 3D coordinate system for two domains (A and B)
    based on 6 geometric parameters.
    """
    # Convert degrees to radians
    ba_rad, bc1_rad, bc2_rad = np.radians([BA, BC1, BC2])
    ac1_rad, ac2_rad = np.radians([AC1, AC2])

    # --- Define coordinates in a local frame for Domain B ---
    B_C = np.array([0.0, 0.0, 0.0])
    B_V1_dir = np.array([np.cos(bc1_rad), np.sin(bc1_rad), 0.0])

    b2_x = np.cos(bc2_rad)
    b2_y = -(b2_x * np.cos(bc1_rad)) / np.sin(bc1_rad)
    b2_z_sq = 1 - b2_x**2 - b2_y**2
    if b2_z_sq < -1e-6: raise ValueError("BC1/BC2 angles are not geometrically compatible.")
    B_V2_dir = normalize(np.array([b2_x, b2_y, np.sqrt(max(0, b2_z_sq))]))

    # --- Define coordinates for Domain A ---
    A_C = np.array([dc, 0.0, 0.0])
    A_V1_dir = np.array([
        -np.cos(ac1_rad),
        np.sin(ac1_rad) * np.cos(ba_rad),
        np.sin(ac1_rad) * np.sin(ba_rad)
    ])

    a2_x = -np.cos(ac2_rad)
    c1 = -A_V1_dir[0] * a2_x
    r_sq = 1.0 - a2_x**2
    y1, z1 = A_V1_dir[1], A_V1_dir[2]

    # Handle the case where y1 is close to zero to avoid division errors
    if np.isclose(y1, 0):
        if np.isclose(z1, 0): raise ValueError("A_V1_dir cannot be parallel to the center axis.")
        a2_z = c1 / z1
        rad_y_sq = r_sq - a2_z**2
        if rad_y_sq < -1e-6: raise ValueError("AC1/AC2 angles are not geometrically compatible.")
        a2_y = np.sqrt(max(0, rad_y_sq))
    else:
        qa = z1**2 + y1**2
        qb = -2 * c1 * z1
        qc = c1**2 - r_sq * y1**2
        rad_quad = qb**2 - 4 * qa * qc
        if rad_quad < -1e-6: raise ValueError("AC1/AC2 angles are not geometrically compatible.")

        a2_z = (-qb + np.sqrt(max(0, rad_quad))) / (2 * qa)
        a2_y = (c1 - z1 * a2_z) / y1

    A_V2_dir = normalize(np.array([a2_x, a2_y, a2_z]))

    return (A_C, A_C + A_V1_dir, A_C + A_V2_dir, B_C, B_C + B_V1_dir, B_C + B_V2_dir)


def apply_transformation(coords, R, t):
    """Applies rotation (R) and translation (t) to a set of coordinates."""
    return (coords @ R.T) + t

def change_geometry(cons_pdb_with_pca, chain_id, target_centroid, target_v1, target_v2):
    """
    Moves a consensus chain structure to a new target geometry by reading its
    source geometry from pseudoatoms.
    """
    structure = btsio.load_structure(cons_pdb_with_pca, model=1)
    chain = structure[structure.chain_id == chain_id]

    # 1. Determine the source geometry from pseudoatoms in the same file
    try:
        source_centroid = structure[(structure.res_name == 'CEN') & (structure.chain_id == 'Z')].coord[0]
        source_v1_end = structure[(structure.res_name == 'PC1') & (structure.chain_id == 'Z')].coord[0]
        source_v2_end = structure[(structure.res_name == 'PC2') & (structure.chain_id == 'Z')].coord[0]
        source_v1_dir = normalize(source_v1_end - source_centroid)
        source_v2_dir = normalize(source_v2_end - source_centroid)
    except IndexError:
        raise ValueError(f"Could not find CEN, PC1, PC2 pseudoatoms in {cons_pdb_with_pca}")

    # 2. Define the target geometry vectors relative to the centroid
    target_v1_dir = normalize(target_v1 - target_centroid)
    target_v2_dir = normalize(target_v2 - target_centroid)

    # 3. Calculate transformation
    R_source = np.stack([source_v1_dir, source_v2_dir, np.cross(source_v1_dir, source_v2_dir)], axis=1)
    R_target = np.stack([target_v1_dir, target_v2_dir, np.cross(target_v1_dir, target_v2_dir)], axis=1)
    rotation = R_target @ np.linalg.inv(R_source)

    # 4. Apply transformation to the entire chain
    chain.coord = apply_transformation(chain.coord, np.identity(3), -source_centroid) # Move to origin
    chain.coord = apply_transformation(chain.coord, rotation, np.zeros(3))           # Rotate
    chain.coord = apply_transformation(chain.coord, np.identity(3), target_centroid)  # Move to target

    return chain

def move_chains_to_geometry(new_consensus_pdb, input_pdb, output_pdb,A_consenus_res, B_consenus_res):
    """
    Aligns the chains of an input PDB to the newly generated consensus geometry.
    """
    aligned_chain_A = align_chain_to_consensus(input_pdb, new_consensus_pdb, "A", static_consenus_res=A_consenus_res, mobile_consenus_res=A_consenus_res)
    aligned_chain_B = align_chain_to_consensus(input_pdb, new_consensus_pdb, "B", static_consenus_res=B_consenus_res, mobile_consenus_res=B_consenus_res)

    final_aligned_structure = aligned_chain_A + aligned_chain_B
    btsio.save_structure(output_pdb, final_aligned_structure)
    print(f"Saved final aligned structure to: {output_pdb}")

def align_chain_to_consensus(mobile_pdb_path, static_pdb_path, chain_id, mobile_consenus_res=None, static_consenus_res=None):
    """Helper to align a single chain and return the transformed AtomArray."""
    static_struct = btsio.load_structure(static_pdb_path, model=1)
    mobile_struct = btsio.load_structure(mobile_pdb_path, model=1)
    static_ca = static_struct[(static_struct.atom_name == "CA") & (static_struct.chain_id == chain_id)]
    mobile_ca = mobile_struct[(mobile_struct.atom_name == "CA") & (mobile_struct.chain_id == chain_id)]
    if static_consenus_res:
        static_ca = static_ca[np.isin(static_ca.res_id, static_consenus_res)]
    if mobile_consenus_res:
        mobile_ca = mobile_ca[np.isin(mobile_ca.res_id, mobile_consenus_res)]
    if static_ca.array_length() < 4 or mobile_ca.array_length() < 4:
        raise ValueError(f"Not enough C-alpha atoms for alignment on chain {chain_id}")

    _, transform, _, _ = bts.superimpose_structural_homologs(fixed=static_ca, mobile=mobile_ca)

    mobile_chain_full = mobile_struct[mobile_struct.chain_id == chain_id]

    # Apply transformation using a robust matrix operation
    M = np.asarray(transform.as_matrix(), dtype=np.float64)[0]
    R, t = M[:3, :3], M[:3, 3]
    mobile_chain_full.coord = (mobile_chain_full.coord @ R.T) + t

    return mobile_chain_full

def add_cgo_arrow(start, end, color, radius=0.3):
    """Generates a CGO string for a PyMOL arrow object."""
    # This function is unchanged but included for completeness
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

def generate_pymol_script(aligned_pdb, new_consensus_pdb, A_C, A_V1, A_V2, B_C, B_V1, B_V2, out_prefix, vis_folder):
    """
    Generates a PyMOL script to visualize the final alignment with longer PCA
    axes and visible centroids.
    """
    pdb_name = Path(aligned_pdb).stem

    # --- FIX: Increased scale factor for longer PCA axes ---
    scale = 10.0

    # Calculate endpoints for the CGO arrows
    a1_end = A_C + scale * (A_V1 - A_C)
    a2_end = A_C + scale * (A_V2 - A_C)
    b1_end = B_C + scale * (B_V1 - B_C)
    b2_end = B_C + scale * (B_V2 - B_C)

    script = f"""
import numpy as np
from pymol import cmd, cgo
cmd.load("{aligned_pdb}","aligned_{pdb_name}")
cmd.load("{new_consensus_pdb}","consensus_geom")
cmd.bg_color("white")
cmd.hide("everything","all")

cmd.show("cartoon","aligned_{pdb_name}")
# --- FIX: Using standard PyMOL color names ---
cmd.color("marine","aligned_{pdb_name} and chain A")
cmd.color("teal","aligned_{pdb_name} and chain B")

cmd.show("cartoon","consensus_geom and polymer")
# --- FIX: Using a more standard gray color ---
cmd.color("gray","consensus_geom")
cmd.set("cartoon_transparency", 0.5, "consensus_geom and polymer")

# --- FIX: Explicitly create and show spheres for the centroids ---
cmd.pseudoatom("centroid_A", pos={list(A_C)}, color="red")
cmd.pseudoatom("centroid_B", pos={list(B_C)}, color="orange")
cmd.pseudoatom("PCA_A1", pos={list(a1_end)}, color="white")
cmd.pseudoatom("PCA_A2", pos={list(a2_end)}, color="white")
cmd.pseudoatom("PCA_B1", pos={list(b1_end)}, color="white")
cmd.pseudoatom("PCA_B2", pos={list(b2_end)}, color="white")
cmd.show("spheres", "centroid_A or centroid_B or PCA_A1 or PCA_A2 or PCA_B1 or PCA_B2")
cmd.set("sphere_scale", 0.5, "centroid_A or centroid_B or PCA_A1 or PCA_A2 or PCA_B1 or PCA_B2")

# Load CGO arrows representing the scaled PCA axes
cmd.load_cgo({add_cgo_arrow(A_C, a1_end, (0.2, 0.5, 1.0))}, "PC1_A")
cmd.load_cgo({add_cgo_arrow(A_C, a2_end, (0.1, 0.8, 0.1))}, "PC2_A")
cmd.load_cgo({add_cgo_arrow(B_C, b1_end, (1.0, 0.5, 0.2))}, "PC1_B")
cmd.load_cgo({add_cgo_arrow(B_C, b2_end, (0.8, 0.8, 0.1))}, "PC2_B")
cmd.load_cgo({add_cgo_arrow(A_C, B_C, (0.5,0.0,0.5))},"dc")


# --- measurements (wizard-equivalent) ---
cmd.distance("dc_len", "centroid_B", "centroid_A")                 # distance dc
cmd.angle("BC1_ang", "PCA_B1", "centroid_B", "centroid_A")         # angle BC1
cmd.angle("BC2_ang", "PCA_B2", "centroid_B", "centroid_A")         # angle BC2
cmd.angle("AC1_ang", "PCA_A1", "centroid_A", "centroid_B")         # angle AC1
cmd.angle("AC2_ang", "PCA_A2", "centroid_A", "centroid_B")         # angle AC2
cmd.dihedral("BA_dih", "PCA_B1", "centroid_B", "centroid_A", "PCA_A1")  # dihedral BA

# Ensure measurement objects are visible
cmd.enable("dc_len")
cmd.enable("BC1_ang")
cmd.enable("BC2_ang")
cmd.enable("AC1_ang")
cmd.enable("AC2_ang")
cmd.enable("BA_dih")

# Global styling for measurement dashes & labels (applies to all three)
cmd.set("dash_width", 3.0)
cmd.set("dash_gap", 0.0)
cmd.set("label_size", 18)
cmd.set("label_color", "black")
cmd.set("label_distance_digits", 2)  # for distances
cmd.set("label_angle_digits", 1)     # for angles/dihedrals

cmd.orient()
cmd.zoom("all", 1.2)
cmd.png("{os.path.join(vis_folder, out_prefix + "_final_vis.png")}", dpi=300, ray=1)
cmd.save("{os.path.join(vis_folder, out_prefix + "_final_vis.pse")}")
cmd.quit()
"""
    vis_script_path = os.path.join(vis_folder, out_prefix + "_final_vis.py")
    with open(vis_script_path, "w") as f:
        f.write(script)
    return vis_script_path

def run(input_pdb, out_path, BA, BC1, BC2, AC1, AC2, dc, data_path=data_path):
    # --- Define paths to input data ---
    consA_pca_path = os.path.join(data_path, "chain_A/average_structure_with_pca.pdb")
    consB_pca_path = os.path.join(data_path, "chain_B/average_structure_with_pca.pdb")
    #read file with consensus alignment residues as list of integers
    with open(os.path.join(data_path, "chain_A/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    A_consenus_res = [int(x) for x in content.split(",") if x.strip()]
    with open(os.path.join(data_path, "chain_B/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    B_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    # --- Setup output directories ---
    pdb_name = Path(input_pdb).stem
    out_dir = Path(out_path); out_dir.mkdir(exist_ok=True)
    tmp_out = out_dir / pdb_name; tmp_out.mkdir(exist_ok=True)
    vis_folder = tmp_out / "vis"; vis_folder.mkdir(exist_ok=True)

    renumbered_pdb = str(tmp_out / f"{pdb_name}_imgt.pdb")
    renumbered_pdb_fv =str(tmp_out / f"{pdb_name}_imgt_fv.pdb")
    write_renumbered_fv(renumbered_pdb, input_pdb)

    # 1. Build the target geometry from the input angles
    A_C, A_V1, A_V2, B_C, B_V1, B_V2 = build_geometry_from_angles(BA, BC1, BC2, AC1, AC2, dc)

    # 2. Move the consensus chains to this new target geometry
    new_chain_A = change_geometry(consA_pca_path, "A", A_C, A_V1, A_V2)
    new_chain_B = change_geometry(consB_pca_path, "B", B_C, B_V1, B_V2)

    # 3. Combine the moved chains and add pseudoatoms for the new geometry
    new_consensus_structure = new_chain_A + new_chain_B

    # Create pseudoatoms for the new target geometry for validation
    target_pseudoatoms_A = bts.array([
        bts.Atom(coord=A_C, atom_name="CA", res_id=900, res_name="GEA", chain_id="X", element="X"),
        bts.Atom(coord=A_V1, atom_name="V1", res_id=900, res_name="GEA", chain_id="X", element="X"),
        bts.Atom(coord=A_V2, atom_name="V2", res_id=900, res_name="GEA", chain_id="X", element="X")
    ])
    target_pseudoatoms_B = bts.array([
        bts.Atom(coord=B_C, atom_name="CB", res_id=901, res_name="GEB", chain_id="Y", element="X"),
        bts.Atom(coord=B_V1, atom_name="V1", res_id=901, res_name="GEB", chain_id="Y", element="X"),
        bts.Atom(coord=B_V2, atom_name="V2", res_id=901, res_name="GEB", chain_id="Y", element="X")
    ])

    structure_with_geom = new_consensus_structure + target_pseudoatoms_A + target_pseudoatoms_B
    new_consensus_pdb = str(tmp_out / "consensus_oriented.pdb")
    btsio.save_structure(new_consensus_pdb, structure_with_geom)
    print(f"Saved new target geometry with pseudoatoms to: {new_consensus_pdb}")

    # 4. Align the input TCR chains to the new consensus geometry
    final_aligned_pdb = str(tmp_out / f"{pdb_name}_oriented.pdb")
    move_chains_to_geometry(new_consensus_pdb, renumbered_pdb_fv, final_aligned_pdb, A_consenus_res, B_consenus_res)

    # 5. Generate visualization script
    vis_script = generate_pymol_script(
        final_aligned_pdb, new_consensus_pdb,
        A_C, A_V1, A_V2, B_C, B_V1, B_V2, pdb_name, str(vis_folder)
    )

    print(f"\n✅ PyMOL script saved. Run with:\n   pymol -cq {vis_script}")
    os.system(f"pymol -cq {vis_script}")
    print(f"Output files saved in: {tmp_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reorient a TCR structure based on 6 geometric parameters.")
    parser.add_argument('--input_pdb', type=str, required=True, help='Path to input PDB file.')
    parser.add_argument('--data_path', type=str, required=False, help='Path to data directory containing consensus files.', default=data_path)
    parser.add_argument('--out_path', type=str, required=False, help='Output directory for results.', default=out_path)
    parser.add_argument('--BA', type=float, required=True, help='Torsion angle between PC1_A and PC1_B.')
    parser.add_argument('--BC1', type=float, required=True, help='Bend angle between PC1_B and center axis.')
    parser.add_argument('--BC2', type=float, required=True, help='Bend angle between PC2_B and center axis.')
    parser.add_argument('--AC1', type=float, required=True, help='Bend angle between PC1_A and center axis.')
    parser.add_argument('--AC2', type=float, required=True, help='Bend angle between PC2_A and center axis.')
    parser.add_argument('--dc', type=float, required=True, help='Distance between centroids.')

    args = parser.parse_args()

    run(
        input_pdb=args.input_pdb,
        out_path=args.out_path,
        BA=args.BA, BC1=args.BC1, BC2=args.BC2,
        AC1=args.AC1, AC2=args.AC2, dc=args.dc,
        data_path=args.data_path
    )
    """
    BA = 113.229304
    BC1 = 98.75194
    BC2 = 9.350585
    AC1 = 71.58541
    AC2 = 154.623844
    dc = 23.978255
    input_pdb = "/workspaces/Graphormer/TRangle/examples/A6prmtop_first_frame_GT.pdb"
    out_path = "output_try_change_geometry"
    data_path = "/workspaces/Graphormer/TRangle/data/consensus_output"
    run(
        input_pdb=input_pdb,
        out_path=out_path,
        BA=BA, BC1=BC1, BC2=BC2,
        AC1=AC1, AC2=AC2, dc=dc,
        data_path=data_path
    )
    """

