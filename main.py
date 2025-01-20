from Bio.PDB import PDBList, PDBParser, PPBuilder
from Bio.PDB.DSSP import DSSP
import os

def download_pdb_file(pdb_id, save_dir="."):
    """Download a PDB file from the Protein Data Bank and ensure it is saved in .pdb format."""
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=save_dir)
    
    # Rename the file to ensure it has a .pdb extension if necessary
    if pdb_file.endswith(".ent"):
        proper_pdb_file = os.path.join(save_dir, f"{pdb_id}.pdb")
        os.rename(pdb_file, proper_pdb_file)
        pdb_file = proper_pdb_file
    
    print(f"Downloaded and saved PDB file for {pdb_id} to {pdb_file}")
    return pdb_file


def extract_sequence_and_check_chain_breaks(pdb_file):
    """Extract the sequence and check for chain breaks using PPBuilder."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    ppb = PPBuilder()
    sequences = []
    chain_breaks = []

    for pp in ppb.build_peptides(structure):
        sequence = pp.get_sequence()
        sequences.append(str(sequence))

        # Check for chain breaks
        residues = pp.get_ca_list()
        for i in range(len(residues) - 1):
            current_res = residues[i]
            next_res = residues[i + 1]

            current_res_num = current_res.get_id()[1]  # Residue sequence number
            next_res_num = next_res.get_id()[1]

            # Handle possible insertion codes or non-numeric identifiers
            if not isinstance(current_res_num, int) or not isinstance(next_res_num, int):
                continue

            if current_res_num + 1 != next_res_num:
                chain_breaks.append((current_res.get_id(), next_res.get_id()))

    return sequences, chain_breaks


def analyze_secondary_structure(pdb_file):
    """Analyze secondary structure and return alpha-helices and beta-sheets with residue spans."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    model = structure[0]  # Assuming a single model

    # Use DSSP to analyze secondary structure
    dssp = DSSP(model, pdb_file)

    helices = []
    sheets = []

    current_helix_start = None
    current_sheet_start = None

    for key in dssp.keys():
        residue = dssp[key]
        ss = residue[2]  # Secondary structure type

        if ss in ['H', 'G', 'I']:  # Helix types
            if current_helix_start is None:
                current_helix_start = key
            current_sheet_start = None
        elif ss in ['B', 'E']:  # Sheet types
            if current_sheet_start is None:
                current_sheet_start = key
            current_helix_start = None
        else:  # Coil or other structure
            if current_helix_start:
                helices.append((current_helix_start, key))
                current_helix_start = None
            if current_sheet_start:
                sheets.append((current_sheet_start, key))
                current_sheet_start = None

    # Handle any unclosed spans
    if current_helix_start:
        helices.append((current_helix_start, list(dssp.keys())[-1]))
    if current_sheet_start:
        sheets.append((current_sheet_start, list(dssp.keys())[-1]))

    return helices, sheets



if __name__ == "__main__":
    pdb_id = "1IQZ"
    save_directory = "./pdb_files"

    # Download the PDB file
    pdb_file_path = download_pdb_file(pdb_id, save_dir=save_directory)

    # Extract sequence and check for chain breaks
    sequences, chain_breaks = extract_sequence_and_check_chain_breaks(pdb_file_path)

    print("Sequences:")
    for i, sequence in enumerate(sequences):
        print(f"Peptide {i + 1}: {sequence}")

    if chain_breaks:
        print("\nChain Breaks Detected:")
        for break_pair in chain_breaks:
            print(f"Break between residues {break_pair[0]} and {break_pair[1]}")
    else:
        print("\nNo chain breaks detected.")
        print("Length of peptide 1 is", len(sequences[0]))

    # Analyze secondary structure
        helices, sheets = analyze_secondary_structure(pdb_file_path)

        print("\nSecondary Structure:")
        print("Alpha-Helices:")
        for helix in helices:
            print(f"Helix from {helix[0]} to {helix[1]}")

        print("Beta-Sheets:")
        for sheet in sheets:
            print(f"Sheet from {sheet[0]} to {sheet[1]}")
