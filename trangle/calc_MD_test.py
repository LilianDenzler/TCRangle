#!/usr/bin/env python3
from pathlib import Path
import os
import math
import warnings
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import MDAnalysis as mda
from tqdm import tqdm

# --- suppress noisy warnings (e.g., formalcharges) ---
warnings.filterwarnings("ignore", message=".*formalcharges.*")

# ---------- math helpers ----------
def as_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)

def angle_between(v1, v2):
    v1 = as_unit(v1); v2 = as_unit(v2)
    dot = np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def kabsch(mobile_xyz, target_xyz):
    """
    R, t such that: R @ mobile + t ~= target
    mobile_xyz, target_xyz: (N,3)
    """
    cm_m = mobile_xyz.mean(axis=0)
    cm_t = target_xyz.mean(axis=0)
    X = mobile_xyz - cm_m
    Y = target_xyz - cm_t
    C = X.T @ Y
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = cm_t - R @ cm_m
    return R, t

def apply_rt(R, t, xyz):
    xyz = np.asarray(xyz, dtype=float)
    return (R @ xyz.T).T + t

# ---------- consensus loaders ----------
def read_pseudo_points_chainZ(pdb_path):
    """
    Read CEN/PC1/PC2 from chain Z in consensus PDB.
    Returns dict with np.array(3,) for C, V1, V2.
    """
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("cons", pdb_path)
    try:
        z = s[0]['Z']
    except KeyError:
        raise ValueError(f"Chain 'Z' not found in {pdb_path}")

    def _get(resn):
        for r in z:
            if r.get_resname() == resn:
                for a in r:
                    return np.asarray(a.get_coord(), dtype=float)
        raise ValueError(f"Pseudoatom '{resn}' not found in chain Z of {pdb_path}")

    return {"C": _get('CEN'), "V1": _get('PC1'), "V2": _get('PC2')}

def load_consensus_coreset_CA(pdb_path, chain_id, resid_list):
    """
    Return (N,3) CA coordinates from consensus PDB for given chain/resids (ordered as resid_list).
    """
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("ref", pdb_path)
    try:
        ch = s[0][chain_id]
    except KeyError:
        raise ValueError(f"Chain '{chain_id}' not found in {pdb_path}")

    ca_map = {}
    for r in ch:
        rid = r.id[1]
        if rid in resid_list:
            for a in r:
                if a.get_name() == 'CA':
                    ca_map[rid] = np.asarray(a.get_coord(), dtype=float)
                    break

    coords = []
    missing = []
    for rid in resid_list:
        v = ca_map.get(rid)
        if v is None:
            missing.append(rid)
        else:
            coords.append(v)
    if missing:
        raise ValueError(f"Missing CA for resids {missing} in {pdb_path} chain {chain_id}")
    return np.vstack(coords)

# ---------- MDAnalysis helpers ----------
def build_ca_indices(u, chain_id, resid_list):
    """
    Return atom indices of CA atoms in MDAnalysis Universe matching chain_id and resid_list (ordered).
    """
    ca = u.select_atoms("name CA and protein")
    ca = ca.select_atoms(f"chainID {chain_id}")
    idx_map = {}
    for a in ca:
        rid = int(a.resid)
        if rid not in idx_map:
            idx_map[rid] = a.index

    idxs, missing = [], []
    for rid in resid_list:
        i = idx_map.get(rid)
        if i is None:
            missing.append(rid)
        else:
            idxs.append(i)
    if missing:
        raise ValueError(f"Trajectory missing CA for resids {missing} in chain {chain_id}")
    return np.asarray(idxs, dtype=int)

# ---------- angle computation ----------
def compute_angles(A_axes, B_axes_trans):
    """
    A_axes: dict with 'C','V1','V2' (consensus A frame, fixed)
    B_axes_trans: dict with 'C','V1','V2' transformed into same frame (per frame)
    """
    Cvec = as_unit(B_axes_trans["C"] - A_axes["C"])
    A1   = as_unit(A_axes["V1"] - A_axes["C"])
    A2   = as_unit(A_axes["V2"] - A_axes["C"])
    B1   = as_unit(B_axes_trans["V1"] - B_axes_trans["C"])
    B2   = as_unit(B_axes_trans["V2"] - B_axes_trans["C"])

    nx = np.cross(A1, Cvec)
    ny = np.cross(Cvec, nx)
    Lp = as_unit(np.array([0.0, np.dot(A1, nx), np.dot(A1, ny)]))
    Hp = as_unit(np.array([0.0, np.dot(B1, nx), np.dot(B1, ny)]))
    BA = angle_between(Lp, Hp)
    if np.cross(Lp, Hp)[0] < 0:
        BA = -BA

    BC1 = angle_between(B1, -Cvec)
    AC1 = angle_between(A1,  Cvec)
    BC2 = angle_between(B2, -Cvec)
    AC2 = angle_between(A2,  Cvec)
    dc  = float(np.linalg.norm(B_axes_trans["C"] - A_axes["C"]))
    return BA, BC1, AC1, BC2, AC2, dc

# ---------- main ----------
def run(input_pdb, input_md, out_path, data_path, stride=1, start=None, stop=None):
    """
    Fast per-frame TCR angle calculation.
    - No per-frame I/O
    - Uses coreset CA atoms and consensus pseudoatoms
    - Outputs CSV: frame,time_ps,BA,BC1,AC1,BC2,AC2,dc
    """
    data_path = Path(data_path)
    consA_pdb = data_path / "chain_A/average_structure_with_pca.pdb"
    consB_pdb = data_path / "chain_B/average_structure_with_pca.pdb"

    A_resids = [int(x) for x in (data_path / "chain_A/consensus_alignment_residues.txt").read_text().strip().split(",") if x.strip()]
    B_resids = [int(x) for x in (data_path / "chain_B/consensus_alignment_residues.txt").read_text().strip().split(",") if x.strip()]

    # Consensus references (loaded once)
    A_ref_CA = load_consensus_coreset_CA(str(consA_pdb), "A", A_resids)  # (Na,3)
    B_ref_CA = load_consensus_coreset_CA(str(consB_pdb), "B", B_resids)  # (Nb,3)
    A_axes   = read_pseudo_points_chainZ(str(consA_pdb))                 # fixed per run
    B_axes   = read_pseudo_points_chainZ(str(consB_pdb))                 # transformed per frame

    # MDAnalysis universe
    u = mda.Universe(str(input_pdb), str(input_md))

    # Optional cropping/stride
    traj = u.trajectory
    if start is not None or stop is not None or stride != 1:
        start_idx = 0 if start is None else start
        stop_idx  = len(traj) if stop is None else stop
        traj = traj[start_idx:stop_idx:stride]
        total = (stop_idx - start_idx + (stride - 1)) // stride
    else:
        total = len(traj)

    # Precompute CA indices for the topology
    A_idx = build_ca_indices(u, "A", A_resids)
    B_idx = build_ca_indices(u, "B", B_resids)

    # Output containers
    frames, times = [], []
    BA_arr, BC1_arr, AC1_arr, BC2_arr, AC2_arr, dc_arr = ([] for _ in range(6))

    # Loop
    for ts in tqdm(traj, total=total, desc=f"Processing {Path(input_pdb).stem}"):
        A_CA = u.atoms.positions[A_idx]  # (Na,3)
        B_CA = u.atoms.positions[B_idx]  # (Nb,3)

        # Step 1: align frame A -> consensus A
        R_A, t_A = kabsch(A_CA, A_ref_CA)

        # Bring frame-B CA into consensus-A frame
        B_CA_in_Aframe = apply_rt(R_A, t_A, B_CA)

        # Step 2: align consensus B -> (frame-B in A frame)
        R_B, t_B = kabsch(B_ref_CA, B_CA_in_Aframe)

        # Transform B pseudoatom axes with step 2
        B_trans = {
            "C":  apply_rt(R_B, t_B, B_axes["C"]),
            "V1": apply_rt(R_B, t_B, B_axes["V1"]),
            "V2": apply_rt(R_B, t_B, B_axes["V2"]),
        }

        # Angles in consensus-A frame (A_axes fixed)
        BA, BC1, AC1, BC2, AC2, dc = compute_angles(A_axes, B_trans)

        frames.append(ts.frame)
        times.append(getattr(ts, "time", np.nan))
        BA_arr.append(BA); BC1_arr.append(BC1); AC1_arr.append(AC1)
        BC2_arr.append(BC2); AC2_arr.append(AC2); dc_arr.append(dc)

    # Write CSV
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{Path(input_pdb).stem}_angles.csv"
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
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote per-frame angles: {out_csv}")

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fast TCR angle calculation over MD trajectories (no per-frame I/O).")
    p.add_argument("--input_pdb", required=True, help="Topology PDB with chainIDs A/B and resids matching coreset files.")
    p.add_argument("--input_md",  required=True, help="Trajectory file (e.g., XTC/TRR/DCD).")
    p.add_argument("--data_path", required=True, help="Folder with consensus files and coreset lists.")
    p.add_argument("--out_path",  required=True, help="Output folder for CSV.")
    p.add_argument("--stride", type=int, default=1, help="Stride for frames (default: 1).")
    p.add_argument("--start",  type=int, default=None, help="Start frame index (default: None).")
    p.add_argument("--stop",   type=int, default=None, help="Stop frame index (exclusive, default: None).")
    args = p.parse_args()

    run(
        input_pdb=args.input_pdb,
        input_md=args.input_md,
        out_path=args.out_path,
        data_path=args.data_path,
        stride=args.stride,
        start=args.start,
        stop=args.stop,
    )
