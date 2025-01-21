import numpy as np

def is_complementary(a, b):
    """Check for base-pairing. Returns boolean."""
    valid = {"AU", "UA", "CG", "GC", "GU", "UG"}
    return f"{a}{b}" in valid

def initialize_matrix(seq_length):
    """Initialize the dynamic programming matrix."""
    return np.zeros((seq_length, seq_length), dtype=int)

def fill_matrix(seq):
    """Fill the Nussinov matrix without constraints."""
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
    """Trace back through the matrix to generate the structure without constraints."""
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
    """Fill the Nussinov matrix with pseudoknot constraints."""
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
    """Trace back through the matrix to generate the structure with pseudoknots."""
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
    Parse RNA sequences and their pseudoknot constraints from a sequences file.

    Args:
        file_path (str): Path to the sequences file.

    Returns:
        List of tuples: Each tuple contains (sequence, constraints) where constraints
                        is a set of index pairs for pseudoknots.
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

def process_sequences_with_models(file_path):
    """Process all sequences, folding with both unconstrained and constrained models."""
    parsed_data = parse_sequences_and_constraints(file_path)
    results = []

    for seq, constraints in parsed_data:
        # Unconstrained folding
        matrix_unconstrained = fill_matrix(seq)
        structure_unconstrained = traceback(matrix_unconstrained, seq)

        # Constrained folding
        matrix_constrained = fill_matrix_with_constraints(seq, constraints)
        structure_constrained = traceback_with_constraints(matrix_constrained, seq, constraints)

        results.append((seq, structure_unconstrained, structure_constrained))

    return results

# File path to the sequences file
file_path = "./wbn215_sequences.txt"

# Process sequences
folded_structures = process_sequences_with_models(file_path)

for seq, unconstrained, constrained in folded_structures:
    print(f"Sequence: {seq}\nUnconstrained: {unconstrained}\nConstrained: {constrained}\n")
