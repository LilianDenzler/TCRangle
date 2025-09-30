#!/usr/bin/env python3
import pathlib
import argparse
from typing import *
from anarci import anarci
from fastcore.xtras import dict2obj
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO, Select
from Bio.PDB.Entity import Entity
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from pathlib import Path
import os


def get_structure(path: pathlib.Path) -> Structure:
    """Parses PDB file, returns a structure object.

    Args:
        data_path (pathlib.Path): Path to data
    """
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    return parser.get_structure(path.stem, path)

def get_structure_sequences(structure: Structure) -> Dict[str, SeqRecord]:
    """Gets the sequences of each chain as a list of SeqRecord objects
    """
    return {
        seq.id[-1]: seq # 1U8L:A[-1] -> A
        for seq in SeqIO.PdbIO.AtomIterator(
            structure.id, structure=structure)
        }

def parse_chain_type(details: Dict[str, Any]) -> str:
    """Takes a one letter chain identifier and converts it to either A, B or raises error

    Args:
        details (Dict): A dictionary containing details of the TCR alignment, chain
        type will be either A (alpha)), B (beta), G (gamma) or D (delta).

    Returns:
        str: A chain identifier
    """
    chain_type = details['chain_type']
    if chain_type in ('A', 'G'):
        return 'A'
    elif chain_type in ('B', 'D'):
        return 'B'
    else:
        raise ValueError(f'chain type {chain_type}')

def get_gap_position_ids(numbering: Dict[str, Dict[str, Any]]) -> List:
    return [res_id[:3] for res_id in list(numbering['A'].numbering) if res_id[-1] == '-']

def detach_children(entity: Entity, children: Sequence):
    for child in children: entity.detach_child(child)

def number_sequences(sequences: Dict[str, SeqRecord], scheme: str = 'imgt') -> Dict[str, Dict[str, Any]]:
    """Takes a list of chain: sequence mappings and aligns the sequences to get their chain identifer and numbering
    The numbering is then returned as a dictionary mapping query chain name to Fv chain, span and

    Args:
        sequences (Dict[str, SeqRecord]):
        scheme (str, optional): Numbering scheme to use. Can be one of: imgt, chothia, kabat or martin. Defaults to 'chothia'.

    Returns:
        Dict[str, Dict[str, Any]]: e.g. {'A': {'Chain': 'H', 'span': slice(0, 107, None), numbering: (' ', 1', ' ', 'T)}}
    """
    formatted_input = [
        (chain, str(seq_record.seq))
        for chain, seq_record in sequences.items()
    ]
    numbering, details, _ = anarci(formatted_input, scheme = scheme)
    new_numbering=[]
    new_details=[]
    for num, det in zip(numbering, details):
        if not num or not det:
            print(f'No numbering found, skipping')
            continue
        if len(new_numbering)==2:
            continue
        new_numbering.append(num)
        new_details.append(det)

    numbering = new_numbering
    details = new_details

    numbering = [
        [(' ', res_id[0][0], res_id[0][1]) for res_id in num[0][0] #(hetcode, seqid, icode)
        if res_id[1] != '-'] # remove gapped elements
        for num in numbering
        if num # drop numbering where no Fv could be found
    ]

    details = dict2obj([det[0] for num, det in zip(numbering, details) if num]) # extract details from list wrapper

    return dict2obj({
        det.query_name:
        {'chain': parse_chain_type(det),'span': slice(det.query_start, det.query_end), 'numbering': num}
        for num, det in zip(numbering, details)
    })

def contains_single_model(structure: Structure) -> bool:
    return len(structure.get_list()) == 1

def new_numbering_ends_on_higher_reseqid(old2new: Tuple) -> bool:
    return old2new[-1][0][1] < old2new[-1][1][1]

def renumber_structure(structure: Structure, numbering: Dict) -> None:
    assert contains_single_model(structure), 'Structure contains more than one model'
    model = structure[0]

    # Remove chains not identified by ANARCI
    keep_chains = set(numbering.keys())
    remove_chains = [chain.id for chain in model if chain.id not in keep_chains]
    for chain_id in remove_chains:
        model.detach_child(chain_id)

    # Clean renumbering
    for name, num in numbering.items():
        res_ids = [res.id for res in model[name].get_residues()]
        fv_res_ids = res_ids[num.span]
        non_fv_res_ids = set(res_ids) - set(fv_res_ids)
        detach_children(model[name], non_fv_res_ids)

        # Temporary IDs to avoid conflicts
        temp_ids = [(' ', 10000 + i, ' ') for i in range(len(fv_res_ids))]
        for old, temp in zip(fv_res_ids, temp_ids):
            model[name][old].id = temp

        # Final IMGT IDs from ANARCI
        for temp, new in zip(temp_ids, num.numbering):
            if model[name].has_id(temp):
                model[name][temp].id = new

        # Rename chain to 'A' or 'B'
        model[name].id = num.chain




class FvSelect(Select):
    """Sublassess select so that only residues in the Fv will be written to disk
    """
    A= range(129)
    B = range(129)

    def accept_residue(self, residue):

        hetcode, seqid, icode = residue.id
        chain = residue.parent.id

        if chain == 'A' and seqid in self.A:
            return True
        elif chain == 'B' and seqid in self.B:
            return True
        else:
            return False

class ChothiaSelect(FvSelect):
    """Sublassess select so that only residues in the Chothia numbered Fv
    will be written to disk
    """
    A = range(120)
    B = range(120)

def write_pdb(path: str, structure: Structure, selector: Select) -> None:
    """Writes the Fv portion of a pdb to disk
    """
    io = PDBIO()
    io.set_structure(structure)
    io.save(path, select=FvSelect())


def write_renumbered_fv(output_path, inputpdb, fv_only=True):
    structure = get_structure(inputpdb)
    sequences = get_structure_sequences(structure)
    full_numbering = number_sequences(sequences, scheme='imgt')

    # Extract ordered chain IDs from structure
    ordered_chain_ids = [chain.id for chain in structure[0].get_chains()]

    # Find first pair of consecutive chains that are α/β (A+B or G+D)
    valid_pair = []
    t1=None
    t2=None
    for i in range(len(ordered_chain_ids)):
        c = ordered_chain_ids[i]
        if c in full_numbering.keys():
            t = full_numbering[c]['chain']
            if t1 is None:
                t1 = t
                c1 = c
            elif t2 is None:
                t2 = t
                c2 = c
            if ['A', 'B'] == [t1, t2]:
                valid_pair.append((c1, c2))
                t1, c1, t2, c2 = None, None, None, None
            if ['B', 'A'] == [t1, t2]:
                valid_pair.append((c2, c1))
                t1, c1, t2, c2 = None, None, None, None

    if not valid_pair:
        print(f"Could not find valid α/β pair in {inputpdb}, skipping.")
        return None
    valid_pair = valid_pair[0]  # Take the first valid pair
    # Filter everything else
    for chain in list(structure[0].child_dict):
        if chain not in valid_pair:
            structure[0].detach_child(chain)

    # Renumber and save
    numbering = {k: full_numbering[k] for k in valid_pair}
    renumber_structure(structure, numbering)
    if fv_only:
        write_pdb(output_path, structure, FvSelect())
    else:
        io = PDBIO()
        io.set_structure(structure)
        io.save(output_path)
    return valid_pair, full_numbering

def get_renumbered_fv_from_saved(structure, valid_pair, full_numbering):
    for chain in list(structure[0].child_dict):
        if chain not in valid_pair:
            structure[0].detach_child(chain)

    # Renumber and save
    numbering = {k: full_numbering[k] for k in valid_pair}
    renumber_structure(structure, numbering)
    return structure

if __name__ == "__main__":
    raw_dir = "/workspaces/Graphormer/TRangle/imgt"
    out_dir = "/workspaces/Graphormer/TRangle/imgt_variable"
    os.makedirs(out_dir, exist_ok=True)

    for pdb_file in os.listdir(raw_dir):
        test_pdb = os.path.join(raw_dir, pdb_file)
        print(f"Processing {test_pdb}")

        structure = get_structure(test_pdb)
        sequences = get_structure_sequences(structure)
        full_numbering = number_sequences(sequences, scheme='imgt')

        # Extract ordered chain IDs from structure
        ordered_chain_ids = [chain.id for chain in structure[0].get_chains()]

        # Find first pair of consecutive chains that are α/β (A+B or G+D)
        valid_pair = None
        for i in range(len(ordered_chain_ids) - 1):
            c1, c2 = ordered_chain_ids[i], ordered_chain_ids[i + 1]
            if c1 in full_numbering and c2 in full_numbering:
                t1 = full_numbering[c1]['chain']
                t2 = full_numbering[c2]['chain']
                if {'A', 'B'} == {t1, t2}:  # one alpha, one beta
                    valid_pair = (c1, c2)
                    break

        if not valid_pair:
            print(f"Could not find valid α/β pair in {pdb_file}, skipping.")
            continue

        # Filter everything else
        for chain in list(structure[0].child_dict):
            if chain not in valid_pair:
                structure[0].detach_child(chain)

        # Renumber and save
        numbering = {k: full_numbering[k] for k in valid_pair}
        renumber_structure(structure, numbering)

        output_path = os.path.join(out_dir, f"{Path(test_pdb).stem}.pdb")
        write_pdb(output_path, structure, FvSelect())