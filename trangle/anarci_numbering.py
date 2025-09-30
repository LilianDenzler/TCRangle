from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np
import os
import sys
import argparse
import numpy as np
import time
import numpy as np
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser, is_aa
from anarci import number
from anarci import anarci
from Bio.PDB import PDBParser, PDBIO, Chain, Residue, Atom, Model, Structure
from Bio.PDB import PDBParser, PDBIO, Select
import copy
import biotite.structure as bts
import biotite.structure.io as btsio

def shift_residue_numbers(structure, offset=1000):
    """Shift all residue numbers by a large offset to avoid collisions."""
    for model in structure:
        for chain in model:
            for residue in chain:
                resid = residue.id
                new_resseq = resid[1] + offset
                residue.id = (resid[0], new_resseq, resid[2])

def renumber_pdb(input_pdb, imgt_pdb, variable_imgt_pdb, A_chain_original_id, B_chain_original_id, renumbering_mapping_A, renumbering_mapping_B):
    """
    Renumbers residues of two specified chains to the IMGT scheme,
    renames the chains to 'A' and 'B', and saves the full and variable-only structures.
    """
    io = PDBIO()
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("TCR", input_pdb)

    # It's safer to work with a copy if you plan multiple modifications
    # and saves. Let's assume the current structure object is fine.

    # This is a good strategy to avoid residue number collisions during renumbering.
    # We assume this function is defined elsewhere in your code.
    shift_residue_numbers(structure, offset=1000)

    # --- Helper function to process a single chain ---
    def _process_chain(chain, mapping):
        """Helper to renumber residues in a single chain based on a mapping."""
        # FIX: Initialize last_resid for each chain to prevent errors.
        # We start with a placeholder residue number.
        last_resid = (" ", 0, " ")

        for residue in chain:
            # The original residue number before the +1000 shift
            original_resnum = residue.id[1] - 1000
            original_icode = residue.id[2]

            # Check if the original residue (number, insertion_code) is in our mapping
            if (original_resnum, original_icode) in mapping:
                (new_num_tuple, expected_res_name) = mapping[(original_resnum, original_icode)]

                # Sanity check
                actual_res_name = seq1(residue.get_resname())
                if actual_res_name != expected_res_name:
                    print(f"WARNING: Mismatch for residue {residue.id[1]-1000}! PDB has {actual_res_name}, mapping expects {expected_res_name}.")

                # Re-assign the residue ID
                residue.id = (residue.id[0], new_num_tuple[0], new_num_tuple[1])
                last_resid = residue.id
            else:
                # If not in mapping, renumber by incrementing the last known number
                # This handles residues in the constant region that aren't in the IMGT mapping.
                new_resnum = last_resid[1] + 1
                residue.id = (" ", new_resnum, " ")
                last_resid = residue.id

    # --- Main Logic ---
    model = structure[0]
    # FIX: Rename the chain IDs directly
    if A_chain_original_id in model and B_chain_original_id in model:
        #print(f"Processing Chain {A_chain_original_id} -> A")
        _process_chain(model[A_chain_original_id], renumbering_mapping_A)
        model[A_chain_original_id].id = 'A'

        #print(f"Processing Chain {B_chain_original_id} -> B")
        _process_chain(model[B_chain_original_id], renumbering_mapping_B)
        model[B_chain_original_id].id = 'B'
    else:
        print(f"ERROR: Could not find original chains '{A_chain_original_id}' and '{B_chain_original_id}' in the PDB.")
        return

    # --- Save the full renumbered structure ---
    #print(f"Saving fully renumbered structure (Chains A, B) to {imgt_pdb}")
    io.set_structure(structure)
    io.save(imgt_pdb) # No need for a Select class if you're saving everything

    # --- Create and save the variable-only structure ---
    if variable_imgt_pdb is None:
        print("No variable_imgt_pdb provided, skipping variable domain extraction.")
        return

    # To avoid modifying the structure we just saved, it's good practice
    # to work on a copy, but detaching children works too.
    for chain_id in ['A', 'B']:
        if chain_id in model:
            chain = model[chain_id]
            residues_to_remove = []
            for residue in chain:
                # IMGT variable domains are typically numbered 1-128
                if not (1 <= residue.id[1] <= 128):
                    residues_to_remove.append(residue.id)

            if residues_to_remove:
                print(f"Removing {len(residues_to_remove)} non-variable residues from Chain {chain_id}")
                for res_id in residues_to_remove:
                    chain.detach_child(res_id)

    #print(f"Saving variable-only structure to {variable_imgt_pdb}")
    io.set_structure(structure)
    io.save(variable_imgt_pdb)

def identify_tcr_chains_with_anarci(pdb_file):
    """Identify TCR α and β chains and renumber using ANARCI."""
    parser = PDBParser(QUIET=True)
    #print(f"Parsing PDB file: {pdb_file}")
    parser = PDBParser(QUIET=True)
    for i, structure in enumerate(parser.get_structure("name", pdb_file).get_models()):
        if i == 0:
            # do stuff with first model only
            break
    A_chain, B_chain = None, None
    renumbering_mapping_A, renumbering_mapping_B = [], []

    for chain in structure:
        residues = [res for res in chain if is_aa(res, standard=True)]
        if not residues:
            continue

        resnames_old = [seq1(res.get_resname()) for res in residues]
        seq = "".join(resnames_old)

        # Use ANARCI for numbering
        numbering, chain_type = number(seq, scheme='IMGT')
        if not numbering:
            continue  # Skip if ANARCI fails

        if chain_type == 'A' or chain_type == 'L':
            A_chain = chain.id
            renumbering_mapping = renumbering_mapping_A
        elif chain_type == 'B' or chain_type == 'H':
            B_chain = chain.id
            renumbering_mapping = renumbering_mapping_B
        else:
            continue

        # Map original residue numbers to ANARCI numbering
        newnums, resnames = [], []
        for out in numbering:
            (newnum, insertioncode) = out[0]
            residue_name = out[1]
            if residue_name == "-":
                continue
            newnums.append((newnum, insertioncode))
            resnames.append(residue_name)

        # Create renumbering mapping
        resids_old =[(res.id[1], res.id[2]) for res in residues]
        #sometimes anarci will truncate starting residue
        if resnames_old[0] != resnames[0]:
            #print("Truncating first residue")
            resnames=[resnames_old[0]]+resnames
            newnums=[(0," ")]+newnums

        for res_anarci, num_anarci, res, num in zip(resnames, newnums, resnames_old, resids_old):
            #print(f"Renumbering {res} to {num_anarci} ({res_anarci})")
            #print(res_anarci, res)
            assert res_anarci == res  # Ensure correctness
            renumbering_mapping.append([res_anarci, num, num_anarci])
    return A_chain, B_chain, renumbering_mapping_A, renumbering_mapping_B

def variable_renumber(pdb_path, imgt_path, variable_pdb_imgt):
    variable_domains = {
    "A": [1, 128],
    "B": [1, 128]}
    #these definitions were taken from https://plueckthun.bioc.uzh.ch/antibody/Numbering/Numbering.html


    A_chain, B_chain, renumbering_mapping_A, renumbering_mapping_B=identify_tcr_chains_with_anarci(pdb_path)

    renumbering_mapping_A_dict = {old: (new,res) for res, old, new in renumbering_mapping_A}
    renumbering_mapping_B_dict = {old: (new,res) for res, old, new in renumbering_mapping_B}
    renumber_pdb(
        input_pdb=pdb_path,
        imgt_pdb=imgt_path,
        variable_imgt_pdb=variable_pdb_imgt,
        A_chain_original_id=A_chain,
        B_chain_original_id=B_chain,
        renumbering_mapping_A=renumbering_mapping_A_dict,
        renumbering_mapping_B=renumbering_mapping_B_dict
    )


    #test that the sequences are the same
    structure_imgt = btsio.load_structure(imgt_path, model=1)
    structure_org = btsio.load_structure(pdb_path, model=1)
    sequences_imgt = {}
    sequences_org = {}
    for structure, sequences in [(structure_imgt, sequences_imgt), (structure_org, sequences_org)]:
    # Iterate through each chain found in the structure
        for chain_id in bts.get_chains(structure):
            chain = structure[structure.chain_id == chain_id]
            x, sequence = bts.get_residues(chain)
            sequence = [seq1(res) for res in sequence]
            # Store the sequence in the dictionary
            sequences[chain_id] = "".join(sequence)
    #print("Sequences from IMGT-renumbered PDB:", sequences_imgt)
    #print("Sequences from original PDB:", sequences_org)
    assert sequences_imgt["A"]==sequences_org[A_chain], f"Mismatch in sequence for chain A: {sequences_imgt['A']} vs {sequences_org[A_chain]}"
    assert sequences_imgt["B"]==sequences_org[B_chain], f"Mismatch in sequence for chain B: {sequences_imgt['B']} vs {sequences_org[B_chain]}"


    return A_chain, B_chain, imgt_path

def get_cdr_atoms(imgt_pdb, A_chain, B_chain):
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
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("tcr", imgt_pdb)
    # Create mapping from chain index in MDTraj to PDB chain name
    chain_map = {}
    for i, chain in enumerate(structure[0]):
        chain_map[chain.id] = i  # e.g. {0: "A", 1: "B"}

    all_cdr_indexes = {cdr: [] for cdr in ranges_imgt.keys()}
    all_cdr_indexes_allatoms = {cdr: [] for cdr in ranges_imgt.keys()}
    cdr_sequences = {}
    for cdr_name, (start_resnum, end_resnum) in ranges_imgt.items():
        if "A" in cdr_name:
            chain_label = A_chain
        elif "B" in cdr_name:
            chain_label = B_chain
        else:
            continue

        chain_index = chain_map[chain_label]
        atom_indices = []
        all_atom_indices=[]
        aa_seq= ""
        for model in structure:
            for chain in model:
                if chain.id != chain_label:
                    continue
                for residue in chain:
                    resnum = residue.id[1]
                    if start_resnum <= resnum <= end_resnum:
                        aa_seq += seq1(residue.get_resname())
                        for atom in residue:
                            all_atom_indices.append(atom.serial_number-1)
                            if atom.name == "CA":
                                atom_indices.append(atom.serial_number-1)
        if cdr_name in cdr_sequences:
            cdr_sequences[cdr_name]+=aa_seq
        else:
            cdr_sequences[cdr_name]=aa_seq

        if len(atom_indices) == 0:
            raise ValueError(f"[Warning] No atoms found for {cdr_name}")
        all_cdr_indexes[cdr_name]=atom_indices
        all_cdr_indexes_allatoms[cdr_name]=all_atom_indices
    return all_cdr_indexes, all_cdr_indexes_allatoms, cdr_sequences

def get_old_res_ids_fv_only(renumbering_mapping_A, renumbering_mapping_B, A_chain, B_chain):
    variable_domains = {
    "A": [1, 128],
    "B": [1, 128]}
    renumbering_mapping_A_dict = {new: (old,res) for res, old, new in renumbering_mapping_A}
    renumbering_mapping_B_dict = {new: (old,res) for res, old, new in renumbering_mapping_B}
    #get list of old residue ids in variable domain
    old_res_ids_A = [renumbering_mapping_A_dict[new][0] for new in range(variable_domains["A"][0], variable_domains["A"][1]+1) if new in renumbering_mapping_A_dict]
    old_res_ids_B = [renumbering_mapping_B_dict[new][0] for new in range(variable_domains["B"][0], variable_domains["B"][1]+1) if new in renumbering_mapping_B_dict]
    return old_res_ids_A, old_res_ids_B, A_chain, B_chain

def get_old_res_ids_of_new_list(new_residue_list, renumbering_mapping_A, renumbering_mapping_B, A_chain, B_chain):
    renumbering_mapping_A_dict = {new: (old,res) for res, old, new in renumbering_mapping_A}
    renumbering_mapping_B_dict = {new: (old,res) for res, old, new in renumbering_mapping_B}
    old_res_ids_A = [renumbering_mapping_A_dict[new][0] for new in new_residue_list if new in renumbering_mapping_A_dict]
    old_res_ids_B = [renumbering_mapping_B_dict[new][0] for new in new_residue_list if new in renumbering_mapping_B_dict]
    return old_res_ids_A, old_res_ids_B, A_chain, B_chain

def run(input_pdb, imgt_pdb, variable_pdb_imgt):
    A_chain, B_chain, imgt_path=variable_renumber(input_pdb, imgt_pdb, variable_pdb_imgt)
    all_cdr_indexes, all_cdr_indexes_allatoms, cdr_sequences=get_cdr_atoms(imgt_pdb, A_chain, B_chain)
    return imgt_path, all_cdr_indexes, all_cdr_indexes_allatoms, cdr_sequences

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Prepare PDB file for imgt numbering')
    parser.add_argument('input_pdb', type=str, help='Input PDB file')
    parser.add_argument('imgt_pdb', type=str, help='imgt renumbered PDB file')
    parser.add_argument('variable_pdb_imgt', type=str, help='Variable domains output with imgt numbering')

    args = parser.parse_args()
    run(args.input_pdb, args.imgt_pdb, args.variable_pdb_imgt)
