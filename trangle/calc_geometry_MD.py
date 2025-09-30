from trangle.calc_geometry import *
import MDAnalysis as mda
import tempfile
from tqdm import tqdm
import sys, os
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore", message=".*formalcharges.*")

def run(input_pdb, input_md, out_path, data_path=data_path, vis=False):
    """
    data_path should contain:
      - chain_A/average_structure_with_pca.pdb   (has CEN/PC1/PC2 in chain Z)
      - chain_B/average_structure_with_pca.pdb   (has CEN/PC1/PC2 in chain Z)
    """
    consA_pca_path = os.path.join(data_path, "chain_A/average_structure_with_pca.pdb")
    consB_pca_path = os.path.join(data_path, "chain_B/average_structure_with_pca.pdb")
     #read file with consensus alignment residues as list of integers
    with open(os.path.join(data_path, "chain_A/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    A_consenus_res = [int(x) for x in content.split(",") if x.strip()]
    with open(os.path.join(data_path, "chain_B/consensus_alignment_residues.txt"), "r") as f:
        content = f.read().strip()
    B_consenus_res = [int(x) for x in content.split(",") if x.strip()]

    pdb_name = Path(input_pdb).stem
    out_dir = Path(out_path); out_dir.mkdir(exist_ok=True)
    tmp_out = out_dir / pdb_name; tmp_out.mkdir(exist_ok=True)
    vis_folder = tmp_out / "vis"
    if vis:
        vis_folder.mkdir(exist_ok=True)
    imgt_path = str(tmp_out / f"{pdb_name}_imgt.pdb")
    imgt_path, fv_input=write_renumbered_fv(imgt_path, input_pdb)
    input_pdb=imgt_path
    # MDAnalysis Universe
    u = mda.Universe(str(input_pdb), str(input_md))
    # Prepare arrays for results; list-of-dicts is fine but arrays are faster
    frames = []
    times  = []
    BA_arr  = []
    BC1_arr = []
    AC1_arr = []
    BC2_arr = []
    AC2_arr = []
    dc_arr  = []

    # Stream frames
    for ts in tqdm(u.trajectory, total=len(u.trajectory), desc="Processing frames"):
        # Write out current frame to PDB
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull):
            with tempfile.TemporaryDirectory() as td:
                tmp_pdb = Path(td) / "frame.pdb"
                u.atoms.write(tmp_pdb.as_posix())  # writes PDB

                # Process (align + compute angles + visualize)
                result_frame = process(
                    input_pdb=tmp_pdb,
                    consA_with_pca=consA_pca_path,
                    consB_with_pca=consB_pca_path,
                    out_dir=str(tmp_out),
                    vis_folder=str(vis_folder) if vis else None,
                    A_consenus_res=A_consenus_res,
                    B_consenus_res=B_consenus_res
                )
            BA, BC1, AC1, BC2, AC2, dc = (
                result_frame["BA"],
                result_frame["BC1"],
                result_frame["AC1"],
                result_frame["BC2"],
                result_frame["AC2"],
                result_frame["dc"],
            )
            frames.append(ts.frame)
            times.append(getattr(ts, "time", np.nan))  # ps if present
            BA_arr.append(BA); BC1_arr.append(BC1); AC1_arr.append(AC1)
            BC2_arr.append(BC2); AC2_arr.append(AC2); dc_arr.append(dc)
    os.remove(imgt_path)
    os.remove(fv_input)
    df = pd.DataFrame({
        "frame": frames,
        "time_ps": times,
        "BA": BA_arr,
        "BC1": BC1_arr,
        "AC1": AC1_arr,
        "BC2": BC2_arr,
        "AC2": AC2_arr,
        "dc": dc_arr,
    })
    df.to_csv(tmp_out / "angles_results.csv", index=False)
    print(f"📄 Saved: {tmp_out/'angles_results.csv'}")
    if vis:
        print(f"🖼️  Figures/PSE in: {vis_folder}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate TCR geometric parameters over an MD trajectory.")
    parser.add_argument('--input_pdb', type=str, required=True, help='Path to input PDB file.')
    parser.add_argument('--input_md', type=str, required=True, help='Path to input MD trajectory file (e.g., DCD).')
    parser.add_argument('--data_path', type=str, required=False, help='Path to data directory containing consensus files.', default=data_path)
    parser.add_argument('--out_path', type=str, required=False, help='Output directory for results.', default=out_path)
    parser.add_argument('--vis', action='store_true', help='Disable visualization output.')
    args = parser.parse_args()
    run(
        input_pdb=args.input_pdb,
        input_md=args.input_md,
        out_path=args.out_path,
        data_path=args.data_path,
        vis=args.vis
    )