# tcr_pairing.py
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional
import numpy as np

from Bio.PDB import PDBParser, is_aa
from scipy.optimize import linear_sum_assignment

from anarci_numbering import (
    anarcii_number_pdb,   # from the module I provided earlier
    legacy_anarci_number_pdb
)

# Helper: get all heavy-atom coords for residues by old-id set
def _coords_for_residues(chain, allowed_old_ids: set[Tuple[int, str]]) -> np.ndarray:
    pts = []
    for res in chain:
        if not is_aa(res, standard=False):
            continue
        key = (res.id[1], res.id[2])  # (resseq, icode)
        if key not in allowed_old_ids:
            continue
        for atom in res.get_atoms():
            pts.append(atom.get_coord())
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(pts, dtype=float)

def _centroid(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    return X.mean(axis=0)

def _contact_count(A: np.ndarray, B: np.ndarray, cutoff: float = 5.0) -> int:
    """Heavy-atom contact count within cutoff (Å), O(N*M) but fine for domains."""
    if A.size == 0 or B.size == 0:
        return 0
    # pairwise distances without huge memory: chunked
    cnt = 0
    chunk = 1024
    for i in range(0, A.shape[0], chunk):
        a = A[i:i+chunk]
        # (na,1,3) - (1,nb,3) -> (na,nb,3)
        d2 = np.sum((a[:, None, :] - B[None, :, :]) ** 2, axis=2)
        cnt += int(np.sum(d2 <= cutoff * cutoff))
    return cnt

def structure_matching_pairs(results, model0, alpha_ids, beta_ids, contact_cutoff=5.0, min_contacts=50):
    # 2) For each candidate, collect old residue IDs whose new IMGT number is in 1..128
    def variable_old_ids(chain_id: str) -> set[Tuple[int, str]]:
        mapping = results[chain_id]["mapping"]  # list of (aa, (old_num, old_icode), (new_num, new_icode))
        return {(old_num, old_icode) for (_, (old_num, old_icode), (new_num, _)) in mapping
                if 1 <= int(new_num) <= 128}

    # Chain objects by id
    chain_by_id = {ch.id: ch for ch in model0}

    alpha_coords, beta_coords = {}, {}
    alpha_cent, beta_cent = {}, {}

    for aid in alpha_ids:
        if aid not in chain_by_id:
            continue
        old_ids = variable_old_ids(aid)
        A = _coords_for_residues(chain_by_id[aid], old_ids)
        alpha_coords[aid] = A
        alpha_cent[aid] = _centroid(A)

    for bid in beta_ids:
        if bid not in chain_by_id:
            continue
        old_ids = variable_old_ids(bid)
        B = _coords_for_residues(chain_by_id[bid], old_ids)
        beta_coords[bid] = B
        beta_cent[bid] = _centroid(B)

    # 3) Build cost matrix with epsilon centroid tiebreaker
    m, n = len(alpha_ids), len(beta_ids)
    if m == 0 or n == 0:
        return []

    cost = np.zeros((m, n), dtype=float)
    meta: Dict[Tuple[int, int], Dict[str, float]] = {}
    epsilon = 1e-3  # keeps contact count dominant

    for i, aid in enumerate(alpha_ids):
        for j, bid in enumerate(beta_ids):
            A = alpha_coords.get(aid, np.zeros((0, 3)))
            B = beta_coords.get(bid, np.zeros((0, 3)))
            c = _contact_count(A, B, cutoff=contact_cutoff)

            ca = np.nan_to_num(alpha_cent.get(aid, np.array([np.nan]*3)), nan=0.0)
            cb = np.nan_to_num(beta_cent.get(bid,  np.array([np.nan]*3)), nan=0.0)
            d = float(np.linalg.norm(ca - cb))

            cost[i, j] = -float(c) + epsilon * d
            meta[(i, j)] = {
                "contacts": int(c),
                "centroid_dist": d,
                "alpha_n_atoms": int(A.shape[0]),
                "beta_n_atoms": int(B.shape[0]),
            }
    # 4) Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    # 5) Build matches + filter weak pairs
    matches: List[Tuple[str, str, Dict[str, float]]] = []
    for i, j in zip(row_ind, col_ind):
        aid = alpha_ids[i]
        bid = beta_ids[j]
        info = meta[(i, j)]
        #chack that only alpha/beta or gamma/delta pairs are allowed
        if results[aid]["ctype"] in ["A"] and results[bid]["ctype"] in ["B"]:
            if info["contacts"] >= min_contacts:
                matches.append((aid, bid, info))
        elif results[aid]["ctype"] in ["G"] and results[bid]["ctype"] in ["D"]:
            if info["contacts"] >= min_contacts:
                matches.append((aid, bid, info))

    # Optional: if nothing passed the threshold, return the best one anyway
    if not matches:
        # pick the (i,j) with minimal cost, only allow alpha/beta or gamma/delta pairs
        best = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = int(best[0]), int(best[1])
        info = meta[(i, j)]
        aid = alpha_ids[i]
        bid = beta_ids[j]
        if results[aid]["ctype"] in ["A"] and results[bid]["ctype"] in ["B"]:
            if  info["contacts"] > 0:
                matches.append((aid, bid, info))
        elif results[aid]["ctype"] in ["G"] and results[bid]["ctype"] in ["D"]:
            if  info["contacts"] > 0:
                matches.append((aid, bid, info))

    if not matches: #if still no matches, allow any pair
        # pick the (i,j) with minimal cost
        best = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = int(best[0]), int(best[1])
        info = meta[(i, j)]
        if info["contacts"] > 0:
            matches.append((alpha_ids[i], beta_ids[j], info))
    return matches


def pair_tcrs_by_interface(pdb_path: str,
                           contact_cutoff: float = 5.0,
                           min_contacts: int = 50,
                           centroid_tiebreaker: bool = True,
                           legacy_anarci: bool = False
                           ) -> List[Tuple[str, str, Dict[str, float]]]:
    """

    Strategy:
      - Number chains via Anarcii
      - Gather heavy-atom coords for IMGT 1..128 (variable domain) per chain
      - Score alpha-beta pairs by:
          score = -contact_count (more contacts is better)
        Tiebreak with centroid distance if requested
      - Solve assignment with Hungarian algorithm
      - Drop matches with too few contacts

    diagnostics includes:
      {
        "contacts": int,
        "centroid_dist": float,
        "alpha_n_atoms": int,
        "beta_n_atoms": int
      }
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("tcrs", pdb_path)
    model0 = next(structure.get_models())

    # 1) Number chains with Anarcii and get mappings
    if legacy_anarci:
        results=legacy_anarci_number_pdb(pdb_path)
        germline_info={cid: r.get("germlines") for cid, r in results.items()}

    else:
        results = anarcii_number_pdb(pdb_path)
        germline_info=None

    per_chain_map={cid: r["mapping"] for cid, r in results.items()}
    alpha_ids = [cid for cid, r in results.items() if r["ctype"] in ["A","G"]]
    beta_ids  = [cid for cid, r in results.items() if r["ctype"] in ["B","D"]]
    if len(alpha_ids) == 1 and len(beta_ids) == 1:
        matches=[(alpha_ids[0], beta_ids[0], None)]

    elif not alpha_ids or not beta_ids:
        return []
    else:
        matches=structure_matching_pairs(results, model0, alpha_ids, beta_ids, contact_cutoff=contact_cutoff, min_contacts=min_contacts)
    unpaired=set(alpha_ids + beta_ids) - set([x for p in matches for x in p[:2]])
    match_info=[]
    for a,b,info in matches:
        print(f"Matched TCR pair: {a}-{b} | {info}")
        a_chain_type = results[a]["ctype"]
        b_chain_type = results[b]["ctype"]
        match_dict={
            "alpha_chain": a,
            "alpha_chain_type": a_chain_type,
            "beta_chain": b,
            "beta_chain_type": b_chain_type
        }
        match_info.append((match_dict))
    for unpair_chain in sorted(unpaired):
        chain_type = results[unpair_chain]["ctype"]
        print(f"Unpaired chain: {unpair_chain} ({chain_type})")
        if chain_type in ["A","G"]:
            unpaired_dict={
            "alpha_chain": unpair_chain,
            "chain_type": chain_type,
            "beta_chain": None,
            "beta_chain_type": None,
            }
        if chain_type in ["B","D"]:
            unpaired_dict={
            "alpha_chain": None,
            "chain_type": None,
            "beta_chain": unpair_chain,
            "beta_chain_type": chain_type,
            }
        match_info.append((unpaired_dict))
    if germline_info:
        return match_info, per_chain_map, germline_info
    return match_info, per_chain_map, None



if __name__ == "__main__":

    pdb = "/path/to/multi_tcr.pdb"
    pairs = pair_tcrs_by_interface(pdb, contact_cutoff=5.0, min_contacts=50)
    for a, b, info in pairs:
        print(f"Pair: {a}-{b} | contacts={info['contacts']} | d_centroid={info['centroid_dist']:.2f} Å")
