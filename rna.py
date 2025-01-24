import numpy as np
import pandas as pd

def is_complementary(a, b):
    """
    Check if two RNA bases can form a complementary pair.
    Args:
        a (str): First RNA base.
        b (str): Second RNA base.
    Returns:
        bool: True if the bases are complementary, False otherwise.
    """
    valid = {"AU", "UA", "CG", "GC", "GU", "UG"}
    return f"{a}{b}" in valid

def initialize_matrix(seq_length):
    """
    Initialize a dynamic programming matrix for RNA folding.
    Args:
        seq_length (int): Length of the RNA sequence.
    Returns:
        np.ndarray: A 2D matrix initialized to zeros.
    """
    return np.zeros((seq_length, seq_length), dtype=int)

def fill_matrix(seq):
    """
    Fill the Nussinov dynamic programming matrix for RNA secondary structure prediction.
    Args:
        seq (str): RNA sequence.
    Returns:
        np.ndarray: Filled Nussinov matrix.
    """
    l = len(seq)
    matrix = initialize_matrix(l)

    # Fill matrix
    for n in range(1, l):
        for i in range(l - n):
            j = i + n

            # Case 1: Do not pair i
            max_val = matrix[i + 1, j]

            # Case 2: Do not pair j
            max_val = max(max_val, matrix[i, j - 1])

            # Case 3: Pair i and j (if allowed)
            if is_complementary(seq[i], seq[j]):
                max_val = max(max_val, matrix[i + 1, j - 1] + 1)

            # Case 4: Bifurcation
            for k in range(i + 1, j):
                max_val = max(max_val, matrix[i, k] + matrix[k + 1, j])

            # Save maximum score
            matrix[i, j] = max_val

    return matrix

def traceback(matrix, seq):
    """
    Generate RNA secondary structure from the Nussinov matrix.
    Args:
        matrix (np.ndarray): Nussinov matrix.
        seq (str): RNA sequence.
    Returns:
        str: Dot-bracket notation of the predicted secondary structure.
    """
    structure = ["."] * len(seq)
    stack = [(0, len(seq) - 1)]

    while stack:
        i, j = stack.pop()
        if i >= j:
            continue

        if matrix[i, j] == matrix[i + 1, j]:
            stack.append((i + 1, j))
        elif matrix[i, j] == matrix[i, j - 1]:
            stack.append((i, j - 1))
        elif matrix[i, j] == matrix[i + 1, j - 1] + 1 and is_complementary(seq[i], seq[j]):
            structure[i] = "("
            structure[j] = ")"
            stack.append((i + 1, j - 1))
        else:
            for k in range(i + 1, j):
                if matrix[i, j] == matrix[i, k] + matrix[k + 1, j]:
                    stack.append((i, k))
                    stack.append((k + 1, j))
                    break

    return "".join(structure)

def fill_matrix_with_constraints(seq, constraints):
    """
    Fill the Nussinov matrix while considering pseudoknot constraints.
    Args:
        seq (str): RNA sequence.
        constraints (set): Set of index pairs representing forbidden base pairs.
    Returns:
        np.ndarray: Filled Nussinov matrix with constraints applied.
    """
    l = len(seq)
    matrix = initialize_matrix(l)

    # Fill matrix with constraints
    for n in range(1, l):
        for i in range(l - n):
            j = i + n

            # Case 1: Do not pair i
            max_val = matrix[i + 1, j]

            # Case 2: Do not pair j
            max_val = max(max_val, matrix[i, j - 1])

            # Case 3: Pair i and j (if allowed)
            if is_complementary(seq[i], seq[j]) and (i, j) not in constraints:
                max_val = max(max_val, matrix[i + 1, j - 1] + 1)

            # Case 4: Bifurcation
            for k in range(i + 1, j):
                max_val = max(max_val, matrix[i, k] + matrix[k + 1, j])

            # Save maximum score
            matrix[i, j] = max_val

    return matrix

def traceback_with_constraints(matrix, seq, constraints):
    """
    Generate RNA secondary structure from the Nussinov matrix with constraints.
    Args:
        matrix (np.ndarray): Nussinov matrix.
        seq (str): RNA sequence.
        constraints (set): Set of index pairs representing forbidden base pairs.
    Returns:
        str: Dot-bracket notation of the predicted secondary structure with constraints applied.
    """
    structure = ["."] * len(seq)
    stack = [(0, len(seq) - 1)]

    while stack:
        i, j = stack.pop()
        if i >= j:
            continue

        if matrix[i, j] == matrix[i + 1, j]:
            stack.append((i + 1, j))
        elif matrix[i, j] == matrix[i, j - 1]:
            stack.append((i, j - 1))
        elif matrix[i, j] == matrix[i + 1, j - 1] + 1 and is_complementary(seq[i], seq[j]) and (i, j) not in constraints:
            structure[i] = "("
            structure[j] = ")"
            stack.append((i + 1, j - 1))
        else:
            for k in range(i + 1, j):
                if matrix[i, j] == matrix[i, k] + matrix[k + 1, j]:
                    stack.append((i, k))
                    stack.append((k + 1, j))
                    break

    return "".join(structure)

def parse_sequences_and_constraints(file_path):
    """
    Parse RNA sequences and pseudoknot constraints from a file.
    Args:
        file_path (str): Path to the input file.
    Returns:
        list: List of tuples containing sequences and constraints.
    """
    sequences = []
    constraints_list = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Process each sequence and its corresponding dot-bracket notation
        for i in range(1, len(lines), 2):  # Step over sequences and dot-bracket notations
            sequence = lines[i].strip()
            dot_bracket = lines[i + 1].strip()

            # Extract pseudoknot constraints from dot-bracket notation
            stack = {}
            constraints = set()

            for idx, char in enumerate(dot_bracket):
                if char in "[{":
                    stack[char] = stack.get(char, []) + [idx]
                elif char in "]}":
                    matching_char = "[" if char == "]" else "{"
                    if stack.get(matching_char):
                        start_idx = stack[matching_char].pop()
                        constraints.add((start_idx, idx))

            sequences.append(sequence)
            constraints_list.append(constraints)

    return list(zip(sequences, constraints_list))

def calculate_hamming_distance(struct1, struct2):
    """
    Calculate the Hamming distance between two RNA structures.
    Args:
        struct1 (str): First dot-bracket notation.
        struct2 (str): Second dot-bracket notation.
    Returns:
        int: Hamming distance between the two structures.
    """
    return sum(1 for a, b in zip(struct1, struct2) if a != b)

def calculate_base_pair_distance(struct1, struct2):
    """
    Calculate the base pair distance between two RNA structures.
    Args:
        struct1 (str): First dot-bracket notation.
        struct2 (str): Second dot-bracket notation.
    Returns:
        int: Base pair distance between the two structures.
    """
    def get_base_pairs(struct):
        stack = []
        pairs = set()
        for i, char in enumerate(struct):
            if char in "([{":
                stack.append(i)
            elif char in ")]}":
                if stack:
                    start = stack.pop()
                    pairs.add((start, i))
        return pairs

    pairs1 = get_base_pairs(struct1)
    pairs2 = get_base_pairs(struct2)

    # Symmetric difference of the base pairs sets
    return len(pairs1.symmetric_difference(pairs2))

def process_sequences_with_models(file_path):
    """
    Process RNA sequences, predicting structures with and without constraints.
    Args:
        file_path (str): Path to the input file.
    Returns:
        list: List of dictionaries containing results for each sequence.
    """
    parsed_data = parse_sequences_and_constraints(file_path)
    results = []

    for idx, (seq, constraints) in enumerate(parsed_data, start=1):
        # Unconstrained folding
        matrix_unconstrained = fill_matrix(seq)
        structure_unconstrained = traceback(matrix_unconstrained, seq)

        # Constrained folding
        matrix_constrained = fill_matrix_with_constraints(seq, constraints)
        structure_constrained = traceback_with_constraints(matrix_constrained, seq, constraints)

        # Calculate metrics
        hamming_dist = calculate_hamming_distance(structure_unconstrained, structure_constrained)
        base_pair_dist = calculate_base_pair_distance(structure_unconstrained, structure_constrained)

        # Collect results
        results.append({
            "sequence": seq,
            "constraints": constraints,
            "unconstrained": structure_unconstrained,
            "constrained": structure_constrained,
            "hamming": hamming_dist,
            "base_pair": base_pair_dist
        })

        # Print consolidated output
        print(f"Sequence {idx}: {seq}\nConstraints: {constraints}\nUnconstrained: {structure_unconstrained}\nConstrained: {structure_constrained}\nHamming Distance: {hamming_dist}\nBase Pair Distance: {base_pair_dist}\n")

    # Save results to a DataFrame for easier analysis
    df = pd.DataFrame(results)
    df.to_csv("rna_results.csv", index=False)
    print("Results saved to rna_results.csv")

    return results

if __name__ == "__main__":
    # File path to the sequences file
    file_path = "./wbn215_sequences.txt"

    # Process sequences
    folded_structures = process_sequences_with_models(file_path)
