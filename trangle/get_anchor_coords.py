import os
from Bio.PDB import PDBParser, Residue
from trangle.numbering import process_pdb
# --- Configuration: IMGT Anchor Residue Numbers ---
# These are the conserved residues that flank the CDR loops in an IMGT-numbered structure.
IMGT_ANCHORS = {
    'CDR1': {'start': 23, 'end': 41},
    'CDR2': {'start': 55, 'end': 66},
    'CDR3': {'start': 104, 'end': 118}
}
def write_renumbered_fv(out_folder, in_path):
    """
    Uses your ANARCII renumbering to produce an IMGT-numbered FV PDB.
    Mirrors your existing helper signature/behavior.
    """
    outputs=process_pdb(
        input_pdb=in_path,
        out_prefix=out_folder,
        write_fv= True
        )
    full_imgt=outputs["pairs"]["files"]["full"]
    variable_pdb_imgt=outputs["pairs"]["files"]["variable"]
    return full_imgt,variable_pdb_imgt

def get_tcr_cdr_anchor_coords(pdb_file: str) -> dict:
    """
    Loads an IMGT-numbered TCR structure and returns the 3D coordinates
    of the alpha-carbon (CA) for each CDR anchor residue.

    Args:
        pdb_file: The file path to the input PDB file.

    Returns:
        A dictionary where keys are chain IDs and values are another dictionary
        containing the [x, y, z] coordinates for each CDR anchor.

        Example structure:
        {
          'A': {
            'CDR1': {'start': [27.8, 16.1, -4.3], 'end': [22.8, 13.1, -5.8]},
            ...
          },
          'B': { ... }
        }
    """
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure('TCR', pdb_file)
    except FileNotFoundError:
        print(f"❌ Error: PDB file not found at '{pdb_file}'")
        return {}

    anchor_coords = {}

    for chain in structure.get_chains():
        chain_id = chain.get_id()
        chain_coords = {
            'CDR1': {'start': None, 'end': None},
            'CDR2': {'start': None, 'end': None},
            'CDR3': {'start': None, 'end': None}
        }

        for res in chain.get_residues():
            # Check if the residue number matches any of our anchor points
            res_num = res.get_id()[1]

            for cdr, anchors in IMGT_ANCHORS.items():
                is_start = res_num == anchors['start']
                is_end = res_num == anchors['end']

                if is_start or is_end:
                    try:
                        # Get the coordinates of the Alpha Carbon atom
                        ca_coords = res['CA'].get_coord().tolist()
                        key = 'start' if is_start else 'end'
                        chain_coords[cdr][key] = [round(c, 3) for c in ca_coords]
                    except KeyError:
                        # This residue is missing an alpha-carbon, so we can't get coords
                        pass

        # Only add the chain to results if at least one coordinate was found
        if any(coords for cdr_vals in chain_coords.values() for coords in cdr_vals.values()):
            anchor_coords[chain_id] = chain_coords

    return anchor_coords

def format_residue(residue: Residue) -> str:
    """Helper function to format residue information for printing."""
    if residue is None:
        return "Not Found"
    # Returns ID in the format "ALA 23A" where A is an insertion code, if present
    res_id = residue.get_id()
    res_name = residue.get_resname()
    return f"{res_name} {res_id[1]}{res_id[2].strip()}"

def run(pdb_file):
    out_folder = os.path.dirname(pdb_file)
    renumbered_pdb, renumbered_pdb_fv=write_renumbered_fv(out_folder, pdb_file)
    tcr_anchors = get_tcr_cdr_anchor_coords(renumbered_pdb)
    os.remove(renumbered_pdb)
    os.remove(renumbered_pdb_fv)  # Clean up the temporary file
    return tcr_anchors

if __name__ == '__main__':
    # --- Create a dummy PDB file for demonstration ---
    # --- Run the function and print the results ---
    import argparse
    parser = argparse.ArgumentParser(description="Extract TCR CDR anchor coordinates from a PDB file.")
    parser.add_argument('pdb_file', type=str, help='Path to the input PDB file.')
    args = parser.parse_args()
    pdb_file= args.pdb_file
    tcr_anchors=run(pdb_file)
    print(tcr_anchors)