from Bio.PDB import PDBList, PDBParser, NeighborSearch, PPBuilder
from Bio.PDB.DSSP import DSSP
import os
import warnings
from math import acos, degrees
import numpy as np

# Configuration
LIGAND_NAME = "SF4"  # Name of the ligand to analyze
INTERACTION_DISTANCE = 5.0  # Distance threshold for interaction detection (in Å)
ANGLE_THRESHOLD = 120  # Angle threshold (in degrees) for valid interactions
DISTANCE_THRESHOLD = 4.5  # Maximum distance to consider residue-ligand interaction
TARGET_RESIDUES = ['CYS', 'HIS', 'SER', 'THR', 'TYR', 'PRO', 'ILE']  # Residues to prioritize for mutation


# Part 1: Protein Structure Analysis

def download_pdb_file(pdb_id, save_dir="."):
    """
    Download a PDB file from the Protein Data Bank.

    Parameters:
    pdb_id (str): The PDB ID of the protein structure to download.
    save_dir (str): Directory where the PDB file will be saved. Defaults to the current directory.

    Returns:
    str: Path to the downloaded PDB file.
    """
    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=save_dir)
    if pdb_file.endswith(".ent"):
        proper_pdb_file = os.path.join(save_dir, f"{pdb_id}.pdb")
        os.rename(pdb_file, proper_pdb_file)
        pdb_file = proper_pdb_file

    print(f"Downloaded and saved PDB file for {pdb_id} to {pdb_file}")

    return pdb_file


def extract_sequence_and_check_chain_breaks(pdb_file):
    """
    Extract the sequence from the PDB file and identify chain breaks.

    Parameters:
    pdb_file (str): Path to the PDB file.

    Returns:
    tuple: A tuple containing:
        - List of sequences (list of str).
        - List of chain breaks (list of tuples).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    ppb = PPBuilder()
    sequences = []
    chain_breaks = []

    for pp in ppb.build_peptides(structure):
        sequence = pp.get_sequence()
        sequences.append(str(sequence))

        residues = pp.get_ca_list()
        for i in range(len(residues) - 1):
            current_res = residues[i]
            next_res = residues[i + 1]
            current_res_num = current_res.get_id()[1]
            next_res_num = next_res.get_id()[1]

            if not isinstance(current_res_num, int) or not isinstance(next_res_num, int):
                continue

            if current_res_num + 1 != next_res_num:
                chain_breaks.append((current_res.get_id(), next_res.get_id()))

    return sequences, chain_breaks


def analyze_secondary_structure(pdb_file):
    """
    Analyze the secondary structure of a protein and identify alpha-helices and beta-sheets.

    Parameters:
    pdb_file (str): Path to the PDB file.

    Returns:
    tuple: A tuple containing:
        - List of alpha-helices (list of tuples with start and end residues).
        - List of beta-sheets (list of tuples with start and end residues).
    """
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


def display_secondary_structure(helices, sheets):
    """
    Display secondary structure information in a human-readable format.

    Parameters:
    helices (list): List of alpha-helices with start and end residues.
    sheets (list): List of beta-sheets with start and end residues.
    """
    print("\nSecondary Structure:")
    
    print("Alpha-Helices:")
    for helix in helices:
        print(f"Helix from {helix[0]} to {helix[1]}")
    
    print("\nBeta-Sheets:")
    for sheet in sheets:
        print(f"Sheet from {sheet[0]} to {sheet[1]}")


def calculate_angle(v1, v2):
    """
    Calculate the angle between two vectors.

    Parameters:
    v1 (numpy.ndarray): First vector.
    v2 (numpy.ndarray): Second vector.

    Returns:
    float: Angle in degrees between the two vectors.
    """
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = acos(dot_product / magnitude)

    return degrees(angle)


def get_interacting_residues(pdb_file, ligand_name=LIGAND_NAME, interaction_distance=INTERACTION_DISTANCE, angle_cutoff=ANGLE_THRESHOLD):
    """
    Identify residues interacting with the specified ligand.

    Parameters:
    pdb_file (str): Path to the PDB file.
    ligand_name (str): Name of the ligand to analyze.
    interaction_distance (float): Distance threshold for identifying interactions (in Å).
    angle_cutoff (float): Angle threshold for valid interactions (in degrees).

    Returns:
    list: List of tuples, each containing a residue, its angle, and distance to the ligand.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]

    ligand_atoms = []
    interaction_data = []

    for chain in model:
        for residue in chain:
            if residue.get_resname() == ligand_name:
                ligand_atoms.extend(residue.get_atoms())

    if not ligand_atoms:
        print(f"No atoms found for ligand {ligand_name} in {pdb_file}.")
        return []

    ns = NeighborSearch([atom for atom in model.get_atoms()])

    for ligand_atom in ligand_atoms:
        close_atoms = ns.search(ligand_atom.coord, interaction_distance)
        for close_atom in close_atoms:
            residue = close_atom.get_parent()
            if residue.get_resname() != ligand_name:
                vector1 = ligand_atom.coord - close_atom.coord
                for other_atom in residue.get_atoms():
                    if other_atom != close_atom:
                        vector2 = other_atom.coord - close_atom.coord
                        angle = calculate_angle(vector1, vector2)
                        if angle <= angle_cutoff:
                            interaction_data.append((residue, angle, ligand_atom - close_atom))
                            break

    sorted_interactions = sorted(interaction_data, key=lambda x: x[2])

    return [(residue, angle, distance) for residue, angle, distance in sorted_interactions]


def select_residues_to_mutate(interactions, distance_threshold=DISTANCE_THRESHOLD, angle_threshold=ANGLE_THRESHOLD, target_residues=TARGET_RESIDUES):
    """
    Select residues for mutation based on interaction data.

    Parameters:
    interactions (list): List of tuples containing residue interaction data (residue, angle, distance).
    distance_threshold (float): Maximum allowable distance for selection.
    angle_threshold (float): Maximum allowable angle for selection.
    target_residues (list): Specific residue types to prioritize for mutation.

    Returns:
    list: List of tuples representing residues selected for mutation.
    """
    selected_residues = []
    for residue, angle, distance in interactions:
        res_name = residue.get_resname()
        res_number = residue.get_id()[1]
        if distance <= distance_threshold and angle <= angle_threshold:
            if not target_residues or res_name in target_residues:
                selected_residues.append((res_name, res_number, distance))

    return selected_residues


def mutate_sequence_full_coverage(sequence, selected_residues, mutation_residues=("G", "A"), residue_offset=1):
    """
    Mutate residues in the sequence with full coverage and handle edge cases.

    Parameters:
    sequence (str): The original protein sequence.
    selected_residues (list): List of tuples representing residues selected for mutation.
    mutation_residues (tuple): Residues to use for mutation, default is ("G", "A").
    residue_offset (int): Offset for residue numbering, default is 1.

    Returns:
    str: The mutated sequence.
    """
    mutated_sequence = list(sequence)
    for idx, (_, res_number, _) in enumerate(selected_residues):
        position = res_number - residue_offset
        if 0 <= position < len(mutated_sequence):
            mutated_sequence[position] = mutation_residues[idx % len(mutation_residues)]
        else:
            print(f"Warning: Residue number {res_number} is out of sequence range.")

    return "".join(mutated_sequence)


def mutate_and_save_sequence(pdb_file, start_residue, end_residue, chain_id, target_residue="P", fasta_file="mutated_sequence.fasta"):
    """
    Mutate residues in a specified range to a target residue and save the sequence in FASTA format.

    Parameters:
    pdb_file (str): Path to the input PDB file.
    start_residue (int): Starting residue number of the secondary structure.
    end_residue (int): Ending residue number of the secondary structure.
    chain_id (str): Chain identifier containing the secondary structure.
    target_residue (str): Single-letter code of the residue to mutate to, default is 'P'.
    fasta_file (str): Path to save the mutated sequence in FASTA format.

    Returns:
    str: Message indicating the mutated sequence has been saved.
    """
    from Bio.PDB import PDBParser, PPBuilder

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Extract the sequence of the specified chain
    ppb = PPBuilder()
    sequence = None
    for pp in ppb.build_peptides(structure):
        if pp[0].get_full_id()[2] == chain_id:  # Ensure this peptide belongs to the correct chain
            sequence = list(str(pp.get_sequence()))
            break

    if sequence is None:
        raise ValueError(f"Chain {chain_id} not found or has no sequence in {pdb_file}.")

    # Mutate residues in the range
    for i in range(start_residue - 1, end_residue):  # Convert to 0-based index
        if 0 <= i < len(sequence):
            sequence[i] = target_residue
        else:
            print(f"Warning: Residue number {i + 1} is out of sequence bounds. Skipping.")

    # Save mutated sequence to FASTA format
    with open(fasta_file, "w") as fasta_out:
        fasta_out.write(">mutated_sequence\n")
        fasta_out.write("".join(sequence))

    return {"mutated_sequence": "".join(sequence), "file_path": fasta_file}


def save_to_fasta(sequence, file_path="mutated_sequence.fasta"):
    """
    Save a sequence in FASTA format.

    Parameters:
    sequence (str): The sequence to save.
    file_path (str): Path to the output FASTA file, default is "mutated_sequence.fasta".

    Returns:
    str: Path to the saved FASTA file.
    """
    with open(file_path, "w") as fasta_file:
        fasta_file.write(">mutated_sequence\n")
        fasta_file.write(sequence)

    print(f"Mutated sequence saved to {file_path}")


def refined_get_ligand_interactions(pdb_file, ligand_name="SF4", interaction_distance=5.0):
    """
    Refine residue-ligand interaction measurements by including specific interaction types.

    Parameters:
    pdb_file (str): Path to the input PDB file.
    ligand_name (str): Name of the ligand to analyze, default is "SF4".
    interaction_distance (float): Distance threshold for identifying interactions, default is 5.0 Å.

    Returns:
    dict: A dictionary with residue IDs as keys and tuples of residue objects and their minimum distances to the ligand.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]

    ligand_atoms = []
    residue_distances = {}

    # Locate ligand atoms
    for chain in model:
        for residue in chain:
            if residue.get_resname() == ligand_name:
                ligand_atoms.extend(residue.get_atoms())

    if not ligand_atoms:
        print(f"No atoms found for ligand {ligand_name} in {pdb_file}.")
        return {}

    ns = NeighborSearch([atom for atom in model.get_atoms()])

    for atom in ligand_atoms:
        close_atoms = ns.search(atom.coord, interaction_distance)
        for close_atom in close_atoms:
            residue = close_atom.get_parent()

            if residue.get_resname() != ligand_name:  # Exclude ligand self-interaction
                residue_id = residue.get_id()[1]  # Use residue sequence number
                if residue_id not in residue_distances:
                    residue_distances[residue_id] = (residue, float("inf"))

                # Update minimum distance for the residue
                distance = atom - close_atom
                if distance < residue_distances[residue_id][1]:
                    residue_distances[residue_id] = (residue, distance)

    return residue_distances


def compare_residue_distances(wild_type_pdb, mutant_pdb, ligand_name="SF4", interaction_distance=5.0):
    """
    Track residue-ligand distance shifts between wild-type and mutant structures.

    Parameters:
    wild_type_pdb (str): Path to the wild-type PDB file.
    mutant_pdb (str): Path to the mutant PDB file.
    ligand_name (str): Name of the ligand to analyze, default is "SF4".
    interaction_distance (float): Distance threshold for identifying interactions, default is 5.0 Å.

    Returns:
    dict: A dictionary comparing residue-ligand distances between wild-type and mutant structures.
    """
    # Get interactions for wild-type and mutant
    wild_type_distances = refined_get_ligand_interactions(wild_type_pdb, ligand_name, interaction_distance)
    mutant_distances = refined_get_ligand_interactions(mutant_pdb, ligand_name, interaction_distance)

    # Compare distances for the same residues
    comparison = {}
    for residue_id, (wild_res, wild_dist) in wild_type_distances.items():
        mutant_entry = mutant_distances.get(residue_id)
        if mutant_entry:
            mutant_res, mutant_dist = mutant_entry
            comparison[residue_id] = {
                "wild_type": (wild_res.get_resname(), wild_dist),
                "mutant": (mutant_res.get_resname(), mutant_dist),
            }
        else:
            comparison[residue_id] = {
                "wild_type": (wild_res.get_resname(), wild_dist),
                "mutant": None,
            }

    # Include mutant-only residues
    for residue_id, (mutant_res, mutant_dist) in mutant_distances.items():
        if residue_id not in comparison:
            comparison[residue_id] = {
                "wild_type": None,
                "mutant": (mutant_res.get_resname(), mutant_dist),
            }

    return comparison


if __name__ == "__main__":
    # Download a PDB file and extract sequence information
    pdb_id = "1IQZ"
    save_directory = "./pdb_files"
    pdb_file_path = download_pdb_file(pdb_id, save_dir=save_directory)

    sequences, chain_breaks = extract_sequence_and_check_chain_breaks(pdb_file_path)
    if not sequences:
        print("No sequences found in the PDB file.")
        exit()

    main_sequence = sequences[0]

    # Display sequence information and check for chain breaks
    print(f"Extracted Sequence: {main_sequence}")
    print(f"Sequence Length: {len(main_sequence)}")
    
    if chain_breaks:
        print(f"Chain breaks found: {chain_breaks}")
    else:
        print("No chain breaks found.")
    
    # Analyze secondary structure and display results
    helices, sheets = analyze_secondary_structure(pdb_file_path)
    display_secondary_structure(helices, sheets)
    
    # Mutate a specific range of residues and save the sequence to a FASTA file
    mutate_and_save_sequence(
    pdb_file=pdb_file_path,
    start_residue=48,
    end_residue=61,
    chain_id="A",
    target_residue="P",  # Mutate to Alanine
    fasta_file="helix_mutation_pro.fasta"
    )

    # Analyze interactions between residues and a ligand
    interactions = get_interacting_residues(pdb_file_path)
    print("\nInteracting residues:")
    for res, angle, dist in interactions:
        print(f"{res.get_resname()} {res.get_id()[1]}: Distance {dist:.2f} Å, Angle {angle:.2f}°")

    # Employ algorithm to select residues for mutation based on interactions
    selected_residues = select_residues_to_mutate(interactions)

    print("\nSelected residues for mutation:")
    for res_name, res_number, distance in selected_residues:
        print(f"{res_name} {res_number}: {distance:.2f} Å")

    mutated_sequence = mutate_sequence_full_coverage(main_sequence, selected_residues)
    print(f"\nOriginal Sequence: {main_sequence}")
    print(f"Mutated Sequence: {mutated_sequence}")

    # save mutated sequence to FASTA file
    save_to_fasta(mutated_sequence)
    
    
    # Compare residue-ligand distances between wild-type and mutant structures
    # uncomment the following lines to run the comparison
    wild_type_pdb = "./1IQZ.pdb"
    mutant_pdb = "./pdb_files/mutant_with_ligands.pdb"

    # Analyze residue-ligand distance shifts
    distance_comparison = compare_residue_distances(wild_type_pdb, mutant_pdb, ligand_name="SF4", interaction_distance=5.0)

    # Display shifts
    print("\nResidue-Ligand Distance Shifts:")
    for residue_id, shift_data in distance_comparison.items():
        wild_info = shift_data["wild_type"]
        mutant_info = shift_data["mutant"]
        print(f"Residue ID {residue_id}:")
        if wild_info:
            print(f"  Wild-Type: {wild_info[0]} - Distance: {wild_info[1]:.2f} Å")
        else:
            print("  Wild-Type: None")
        if mutant_info:
            print(f"  Mutant: {mutant_info[0]} - Distance: {mutant_info[1]:.2f} Å")
        else:
            print("  Mutant: None")