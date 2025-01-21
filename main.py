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


def download_pdb_file(pdb_id, save_dir="."):
    """Download a PDB file from the Protein Data Bank."""

    pdbl = PDBList()
    pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=save_dir)
    if pdb_file.endswith(".ent"):
        proper_pdb_file = os.path.join(save_dir, f"{pdb_id}.pdb")
        os.rename(pdb_file, proper_pdb_file)
        pdb_file = proper_pdb_file

    print(f"Downloaded and saved PDB file for {pdb_id} to {pdb_file}")

    return pdb_file


def extract_sequence_and_check_chain_breaks(pdb_file):
    """Extract the sequence and check for chain breaks."""

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


def calculate_angle(v1, v2):
    """Calculate the angle between two vectors in degrees."""

    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = acos(dot_product / magnitude)

    return degrees(angle)


def get_interacting_residues(pdb_file, ligand_name=LIGAND_NAME, interaction_distance=INTERACTION_DISTANCE, angle_cutoff=ANGLE_THRESHOLD):
    """
    Identify residues interacting with the specified ligand based on distance and angle.

    :param pdb_file: str, path to the PDB file.
    :param ligand_name: str, name of the ligand to analyze.
    :param interaction_distance: float, distance threshold for identifying interactions.
    :param angle_cutoff: float, minimum angle to consider valid interactions.
    :return: list of tuples, each containing a residue, its angle, and distance to the ligand.
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

    :param interactions: list of tuples, residue interaction data (residue, angle, distance).
    :param distance_threshold: float, maximum allowable distance for selection.
    :param angle_threshold: float, maximum allowable angle for selection.
    :param target_residues: list of str, specific residue types to prioritize.
    :return: list of tuples, residues selected for mutation.
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

    :param sequence: str, the original protein sequence.
    :param selected_residues: list of tuples, residues selected for mutation.
    :param mutation_residues: tuple of str, residues to use for mutation.
    :param residue_offset: int, offset for residue numbering.
    :return: str, the mutated sequence.
    """

    mutated_sequence = list(sequence)
    for idx, (_, res_number, _) in enumerate(selected_residues):
        position = res_number - residue_offset
        if 0 <= position < len(mutated_sequence):
            mutated_sequence[position] = mutation_residues[idx % len(mutation_residues)]
        else:
            print(f"Warning: Residue number {res_number} is out of sequence range.")

    return "".join(mutated_sequence)


def save_to_fasta(sequence, file_path="mutated_sequence.fasta"):
    """Save sequence in FASTA format."""

    with open(file_path, "w") as fasta_file:
        fasta_file.write(">mutated_sequence\n")
        fasta_file.write(sequence)

    print(f"Mutated sequence saved to {file_path}")


if __name__ == "__main__":
    pdb_id = "1IQZ"
    save_directory = "./pdb_files"
    pdb_file_path = download_pdb_file(pdb_id, save_dir=save_directory)

    sequences, chain_breaks = extract_sequence_and_check_chain_breaks(pdb_file_path)
    if not sequences:
        print("No sequences found in the PDB file.")
        exit()

    main_sequence = sequences[0]

    print(f"Extracted Sequence: {main_sequence}")
    print(f"Sequence Length: {len(main_sequence)}")

    interactions = get_interacting_residues(pdb_file_path)
    print("\nInteracting residues:")
    for res, angle, dist in interactions:
        print(f"{res.get_resname()} {res.get_id()[1]}: Distance {dist:.2f} Å, Angle {angle:.2f}°")

    selected_residues = select_residues_to_mutate(interactions)

    print("\nSelected residues for mutation:")
    for res_name, res_number, distance in selected_residues:
        print(f"{res_name} {res_number}: {distance:.2f} Å")

    mutated_sequence = mutate_sequence_full_coverage(main_sequence, selected_residues)
    print(f"\nOriginal Sequence: {main_sequence}")
    print(f"Mutated Sequence: {mutated_sequence}")

    save_to_fasta(mutated_sequence)
