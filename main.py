from Bio.PDB import PDBList, PDBParser, PPBuilder, NeighborSearch
from Bio.PDB.DSSP import DSSP
import os
import warnings
from math import acos, degrees
import numpy as np

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


def analyze_secondary_structure_and_ligand_interactions(pdb_file, ligand_name="SF4", interaction_distance=5.0):
    """Analyze secondary structure and identify residues interacting with the specified ligand."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    model = structure[0]  # Assuming a single model
    
    # Suppress specific warning related to DSSP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        dssp = DSSP(model, pdb_file)

    helices = []
    sheets = []

    current_helix_start = None
    current_sheet_start = None

    for key in dssp.keys():
        residue = key[1]
        ss = dssp[key][2]  # Secondary structure type

        if ss in ['H', 'G', 'I']:  # Helix types
            if current_helix_start is None:
                current_helix_start = key
            current_sheet_start = None
        elif ss in ['B', 'E']:  # Sheet types
            if current_sheet_start is None:
                current_sheet_start = key
            current_helix_start = None
        else:
            if current_helix_start:
                helices.append((current_helix_start, key))
                current_helix_start = None
            if current_sheet_start:
                sheets.append((current_sheet_start, key))
                current_sheet_start = None

    if current_helix_start:
        helices.append((current_helix_start, list(dssp.keys())[-1][1]))
    if current_sheet_start:
        sheets.append((current_sheet_start, list(dssp.keys())[-1][1]))

    ligand_residues = []
    ligand_atoms = []

    for chain in model:
        for residue in chain:
            if residue.get_resname() == ligand_name:
                ligand_atoms.extend(residue.get_atoms())

    ns = NeighborSearch([atom for atom in model.get_atoms()])

    residue_distances = []
    
    for atom in ligand_atoms:
        close_atoms = ns.search(atom.coord, interaction_distance)
        for close_atom in close_atoms:
            residue = close_atom.get_parent()
            if residue not in ligand_residues:
                ligand_residues.append(residue)
                
    # Calculate distances and sort residues by proximity
    for residue in ligand_residues:
        min_distance = float("inf")
        for atom in residue.get_atoms():
            for ligand_atom in ligand_atoms:
                distance = atom - ligand_atom
                if distance < min_distance:
                    min_distance = distance
        residue_distances.append((residue, min_distance))

    sorted_residues = sorted(residue_distances, key=lambda x: x[1])

    return helices, sheets, sorted_residues


def select_residues_to_mutate(residue_info, distance_threshold=4.5, target_residues=None):
    """
    Select residues to mutate based on proximity and type.

    :param residue_info: list of tuples, [(residue_name, residue_number, distance), ...].
    :param distance_threshold: float, maximum distance from ligand to include residues.
    :param target_residues: list, optional, specific residue types to include (e.g., ['CYS', 'THR']).
    :return: list of tuples, residues selected for mutation.
    """
    selected_residues = []

    for res_name, res_number, distance in residue_info:
        if distance <= distance_threshold and (not target_residues or res_name in target_residues):
            selected_residues.append((res_name, res_number, distance))

    return selected_residues


def induce_mutations(sequence, selected_residues, mutation_residues=("G", "A")):
    """
    Induce mutations in the sequence based on selected residues.

    :param sequence: str, original protein sequence.
    :param selected_residues: list of tuples, residues selected for mutation [(res_name, res_number, distance), ...].
    :param mutation_residues: tuple, possible mutation residues (e.g., ("G", "A")).
    :return: str, mutated sequence.
    """
    mutated_sequence = list(sequence)

    for idx, (res_name, res_number, _) in enumerate(selected_residues):
        position = res_number - 1  # Convert to 0-based index
        mutated_sequence[position] = mutation_residues[idx % len(mutation_residues)]

    return "".join(mutated_sequence)


def save_sequence_for_alphafold(sequence, file_path="mutated_sequence.fasta"):
    """
    Save the mutated sequence in FASTA format for AlphaFold or similar tools.

    :param sequence: str, mutated protein sequence.
    :param file_path: str, output file path.
    """
    with open(file_path, "w") as fasta_file:
        fasta_file.write(">mutated_sequence\n")
        fasta_file.write(sequence)
    print(f"Mutated sequence saved to {file_path}")


def calculate_angle(v1, v2):
    """
    Calculate the angle between two vectors in degrees.
    """
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = acos(dot_product / magnitude)
    return degrees(angle)

def refined_get_ligand_interactions(pdb_file, ligand_name="SF4", interaction_distance=5.0):
    """
    Refine residue-ligand interaction measure by including specific interaction types.
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
    pdb_id = "1IQZ"
    save_directory = "./pdb_files"

    pdb_file_path = download_pdb_file(pdb_id, save_dir=save_directory)

    sequences, chain_breaks = extract_sequence_and_check_chain_breaks(pdb_file_path)

    print("Sequences:")
    for i, sequence in enumerate(sequences):
        # Naming of the protein sequence
        print(f"Peptide {i + 1}: {sequence}")

    if chain_breaks:
        print("\nChain Breaks Detected:")
        for break_pair in chain_breaks:
            print(f"Break between residues {break_pair[0]} and {break_pair[1]}")
    else:
        print("\nNo chain breaks detected.")

    helices, sheets, ligand_residues = analyze_secondary_structure_and_ligand_interactions(pdb_file_path)

    print("\nSecondary Structure:")
    print("Alpha-Helices:")
    for helix in helices:
        print(f"Helix from {helix[0]} to {helix[1]}")

    print("Beta-Sheets:")
    for sheet in sheets:
        print(f"Sheet from {sheet[0]} to {sheet[1]}")

    # print("\nResidues interacting with the ligand:")
    # for ligand_residue in ligand_residues:
    #     print(ligand_residue)
    
    print("\nResidues interacting with the ligand:")
    for residue, distance in ligand_residues:
        print(f"{residue.get_full_id()} with minimum distance: {distance:.2f} Å")


    print("\nResidues interacting with the ligand (sorted by proximity):")
    residue_info = []
    for residue, distance in ligand_residues:
        resname = residue.get_resname()
        resnum = residue.get_id()[1]
        chain_id = residue.get_full_id()[2]
        residue_info.append((resname, resnum, distance))
        #print(f"Residue {resname} {resnum} in chain {chain_id} is interacting with minimum distance {distance:.2f}Å")
        print(f"{resname} {resnum} interacts with ligand at a distance of {distance:.2f} Å")
        
    # Select residues to mutate based on proximity to the ligand
    selected_residues = select_residues_to_mutate(residue_info, distance_threshold=4.5, target_residues=['CYS', 'HIS', 'SER', 'THR', 'TYR', 'PRO', 'ILE'])
    
    print("\nSelected residues for mutation:")
    # for res_name, res_number, distance in selected_residues:
    #     print(f"Residue: {res_name} {res_number}, Distance to ligand: {distance:.2f} Å")
    for idx, (res_name, res_number, distance) in enumerate(selected_residues):
        mutation = "G" if idx % 2 == 0 else "A"
        print(f"Residue: {res_name} {res_number}, Distance: {distance:.2f} Å --> Mutation: {mutation}")
        
    # Mutate the selected residues
    if sequences:
        print('\nMutating the selected residues...')
        main_sequence = sequences[0]
        mutated_sequence = induce_mutations(main_sequence, selected_residues)
        sequences.append(mutated_sequence)
        
        print(f"Original Sequence: \n{main_sequence}")
        print('')
        print(f"Mutated Sequence: \n{mutated_sequence}")
        print('')
        print(f"They are equal: {sequences[0] == sequences[1]} \n")
        
        # save mutated sequence to fasta file
        save_sequence_for_alphafold(mutated_sequence, file_path="mutated_sequence.fasta")
        
    # Compare residue distances in wild-type and mutant structures
    wild_type_pdb_path = pdb_file_path  # Replace with actual path
    mutant_pdb_path = "./pdb_files/mutant_with_ligands.pdb"  # Replace with actual path

    comparison = compare_residue_distances(pdb_file_path, mutant_pdb_path)

    # Print the comparison
    print("\nResidue-Ligand Distance Comparison:")
    for residue_id, data in comparison.items():
        wild_type_data = data.get("wild_type", ("None", "N/A"))
        mutant_data = data.get("mutant", ("None", "N/A"))

        print(f"Residue {residue_id}:")
        if wild_type_data and wild_type_data[1] != "N/A":
            print(f"  Wild-Type: {wild_type_data[0]} at {wild_type_data[1]:.2f} Å")
        else:
            print("  Wild-Type: None")

        if mutant_data and mutant_data[1] != "N/A":
            print(f"  Mutant: {mutant_data[0]} at {mutant_data[1]:.2f} Å")
        else:
            print("  Mutant: None")
