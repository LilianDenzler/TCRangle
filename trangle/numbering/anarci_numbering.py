# anarci_numbering.py
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional
from Bio.PDB import PDBParser, is_aa, Chain
from Bio.SeqUtils import seq1
from anarcii import Anarcii
from anarci import anarci
# ------------------------------
# Env / resource helpers
# ------------------------------
def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return max(1, int(v))
    except ValueError:
        return default

def _pick_ncpu(default_cap: int = 16) -> int:
    env_n = _env_int("ANARCI_NCPU", -1)
    if env_n > 0:
        return env_n
    try:
        n = os.cpu_count() or 1
    except Exception:
        n = 1
    return max(1, min(n, default_cap))

def _pick_cpu_flag() -> bool:
    """
    Default to GPU when available unless ANARCI_CPU is set.
    Set ANARCI_CPU=1 to force CPU, ANARCI_CPU=0 to force GPU.
    """
    #Auto-detect GPU if torch is present
    try:
        import torch
        return not torch.cuda.is_available()  # cpu=True when no CUDA
    except Exception:
        return True  # fall back to CPU if torch isn't around

# ------------------------------
# Global model (singleton)
# ------------------------------
_ANARCI_MODEL: Optional[Anarcii] = None

def get_anarci_model() -> Anarcii:
    global _ANARCI_MODEL
    if _ANARCI_MODEL is None:
        _ANARCI_MODEL = Anarcii(
            seq_type="tcr",
            batch_size=_env_int("ANARCI_BATCH", 128),
            cpu=_pick_cpu_flag(),
            ncpu=_pick_ncpu(),
            mode="accuracy",
            verbose=False,
        )
    return _ANARCI_MODEL

# ------------------------------
# Relaxed residue mapping
# ------------------------------
RELAXED_3to1 = {
    # histidine protonation states
    "HIE": "H", "HID": "H", "HIP": "H",
    # disulfide cysteine
    "CYX": "C", "CYS": "C",
    # selenomethionine
    "MSE": "M",
    # uncommon canonical
    "SEC": "U", "PYL": "O",
}

def three_to_one_standardized(resname: str) -> str:
    r = (resname or "").upper().strip()
    if r in RELAXED_3to1:
        return RELAXED_3to1[r]
    try:
        return seq1(r)
    except Exception:
        return "X"

# ------------------------------
# Chain numbering
# ------------------------------
MappingRow = Tuple[str, Tuple[int, str], Tuple[int, str]]
#            (aa_imgt, (old_num, old_icode), (new_num, new_icode))


def anarcii_number_chain(
    chain: Chain,
    model: Optional[Anarcii] = None
) -> Tuple[Optional[str], List[MappingRow], Dict[str, int]]:
    """
    Return (ctype, mapping, meta) for a chain.

    mapping rows are: (aa_imgt, (old_num, old_icode), (new_num, new_icode))
    meta: {"query_start": int, "query_end": int}
    """
    if model is None:
        model = get_anarci_model()

    residues = [res for res in chain]
    if not residues:
        return None, [], {"query_start": 0, "query_end": -1}

    seq_one = "".join(seq1(res.get_resname()) for res in residues)

    # Call once and reuse outputs (avoid duplicate calls downstream)
    numbered = model.number([seq_one])
    data = numbered["Sequence 1"]
    chain_type = data.get("chain_type")
    numbering  = data.get("numbering", [])
    qstart     = data.get("query_start")
    qend       = data.get("query_end")
    numbering_map=mapping(chain, numbering, qstart, qend)
    return chain_type, numbering_map




# ------------------------------
# PDB-level numbering (SINGLE definition)
# ------------------------------
def anarcii_number_pdb(
    pdb_path: str,
    model: Optional[Anarcii] = None
) -> Dict[str, Dict[str, object]]:
    """
    Number all AA chains in a PDB.

    Returns a dict per chain:
      {
        chain_id: {
           "ctype": 'A'|'B'|'G'|'D'|None,
           "mapping": [ (aa_imgt, (old_num, old_icode), (new_num, new_icode)), ... ],
           "query_start": int,
           "query_end": int,
           "score": float,
        }, ...
      }
    """
    if model is None:
        model = get_anarci_model()


    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    model0 = next(structure.get_models())

    results: Dict[str, Dict[str, object]] = {}

    for chain in model0:
        residues = [res for res in chain]
        if not residues:
            continue

        seq_one = "".join(seq1(res.get_resname()) for res in residues)
        print(f"[info] Numbering chain {chain.id} with length {len(seq_one)}")
        ana = model.number([seq_one]).get("Sequence 1", {})
        assert ana["scheme"] == "imgt", f"Unexpected scheme: {ana.get('scheme')}"
        ctype, mapping = anarcii_number_chain(chain, model=model)

        results[chain.id] = {
            "ctype": ctype,
            "mapping": mapping,
            "score": float(ana.get("score", 0.0)),
        }

    return results

def mapping(chain, numbering, query_start, query_end):
    numbering_map={}
    for res_new in numbering:
        ((new_num, new_icode), aa_imgt_new)=res_new
        if aa_imgt_new=="-":
            continue
        else:
            first_imgt_number=new_num
            break

    for enum,res in enumerate(chain):
        resname=res.get_resname()
        old_resid=res.id[1]
        old_icode=res.id[2]
        if enum<query_start:
            new_num=int(first_imgt_number-query_start+enum)
            numbering_map[(old_resid, old_icode)]=((new_num, ' '), seq1(resname))
    chain_sliced=[res for res in chain]
    chain_sliced=chain_sliced[query_start:query_end+1]
    numbering_sliced=[res_new for res_new in numbering if res_new[1]!="-"]
    for enum, res_old, res_new in zip(range(query_start,query_end+1),chain_sliced, numbering_sliced):
        if enum<query_start:
            continue
        resname=res_old.get_resname()
        old_resid=res_old.id[1]
        old_icode=res_old.id[2]
        ((new_num, new_icode), aa_imgt_new)=res_new
        if aa_imgt_new=="-":
            continue
        else:
            assert aa_imgt_new==three_to_one_standardized(resname) or aa_imgt_new==seq1(resname), f"Residue name mismatch for chain {chain.id} res {old_resid}{old_icode}: anarci {aa_imgt_new} vs pdb {seq1(resname)}"
        numbering_map[(old_resid, old_icode)]=((new_num, new_icode), seq1(resname))
        last_imgt=new_num
    chain_sliced=[res for res in chain]
    chain_sliced=chain_sliced[query_end:]
    for enum,res in enumerate(chain_sliced):
        resname=res.get_resname()
        old_resid=res.id[1]
        old_icode=res.id[2]
        new_num=last_imgt+1+enum
        numbering_map[(old_resid, old_icode)] = ((new_num, ' '), seq1(resname))
    return numbering_map


def legacy_anarci_number_chain(chain):
    chain_id=chain.id
    standardised_seq=[three_to_one_standardized(res.get_resname()) for res in chain]
    seq_one = "".join(standardised_seq)
    print(seq_one)
    print(f"[info] Numbering chain {chain.id} with length {len(seq_one)}")
    results = anarci([("seq",seq_one)], scheme="imgt", output=False, assign_germline=True)
    # Unpack the results. We get three lists
    numbering, alignment_details, hit_tables = results
    numbering = numbering[0][0][0]
    alignment_details = alignment_details[0][0]
    chain_id=alignment_details["id"]
    chain_type=chain_id.split("_")[1]
    species=chain_id.split("_")[0]
    query_start=int(alignment_details["query_start"])
    query_end=int(alignment_details["query_end"])
    germlines=alignment_details["germlines"]
    numbering_map=mapping(chain, numbering, query_start, query_end)
    chain_result={
            "ctype": chain_type,
            "mapping": numbering_map,
            "score": float(alignment_details.get("score", 0.0)),
            "germlines": germlines,
        }
    return chain_result


# ------------------------------
# legacy anarci numbering
def legacy_anarci_number_pdb(
    pdb_path: str
) -> Dict[str, Dict[str, object]]:
    """
    Number all AA chains in a PDB.

    Returns a dict per chain:
      {
        chain_id: {
           "ctype": 'A'|'B'|'G'|'D'|None,
           "mapping": [ (aa_imgt, (old_num, old_icode), (new_num, new_icode)), ... ],
           "query_start": int,
           "query_end": int,
           "score": float,
        }, ...
      }
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    model0 = next(structure.get_models())
    results: Dict[str, Dict[str, object]] = {}
    for chain in model0:
        chain_result=legacy_anarci_number_chain(chain)
        results[chain.id] = chain_result
    return results





if __name__ == "__main__":
    pdb_file="/workspaces/Graphormer/TCR_NUMBERING/7S8I.pdb"
    anarcii_number_pdb(pdb_file)
    #result=legacy_anarci_number_pdb(pdb_file)