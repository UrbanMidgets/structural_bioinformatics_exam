import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_complementary(a, b):
    """
    Check if two RNA bases can pair.
    Args:
        a (str): First base.
        b (str): Second base.
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
        np.ndarray: A 2D numpy array initialized to zeros.
    """
    return np.zeros((seq_length, seq_length), dtype=int)

def fill_matrix(seq):
    """
    Fill the Nussinov dynamic programming matrix.
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
    Traceback through the matrix to generate the RNA secondary structure.
    Args:
        matrix (np.ndarray): Nussinov matrix.
        seq (str): RNA sequence.
    Returns:
        str: Dot-bracket notation of the RNA structure.
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

def apply_pseudoknot_constraints(structure, constraints):
    """
    Apply pseudoknot constraints to an RNA secondary structure.
    Args:
        structure (str): Dot-bracket notation of the RNA structure.
        constraints (list of tuple): List of (i, j) constraints for pseudoknots.
    Returns:
        str: Modified dot-bracket notation with pseudoknots applied.
    """    
    structure = list(structure)  # Convert to mutable list

    for i, j in constraints:
        # Debugging: Print before applying constraints
        print(f"Applying constraint ({i}, {j}) to structure: {''.join(structure)}")

        # Remove existing pairs at positions i and j if necessary
        if structure[i] in "()[]{}":
            for k, char in enumerate(structure):
                if char in ")]}" and structure[k] in "([{":
                    if structure[k] == "(" and char == ")":
                        if i == k or j == k:
                            structure[k] = "."
                            structure[structure.index(char)] = "."
                            break

        if structure[j] in "()[]{}":
            for k, char in enumerate(structure):
                if char in ")]}" and structure[k] in "([{":
                    if structure[k] == "(" and char == ")":
                        if i == k or j == k:
                            structure[k] = "."
                            structure[structure.index(char)] = "."
                            break

        # Explanation: Force apply pseudoknot constraints
        # At this step, any existing pairs involving i and j are removed to prioritize pseudoknot placement.
        # This ensures that pseudoknots are accurately represented in the constrained structure.
        structure[i] = "["
        structure[j] = "]"

        # Debugging: Print after applying constraints
        print(f"Structure after applying constraint ({i}, {j}): {''.join(structure)}")

    return "".join(structure)

def calculate_hamming_distance(struct1, struct2):
    """
    Calculate the Hamming distance between two RNA structures.
    Args:
        struct1 (str): First dot-bracket notation.
        struct2 (str): Second dot-bracket notation.
    Returns:
        int: Hamming distance.
    """
    return sum(1 for a, b in zip(struct1, struct2) if a != b)

def calculate_base_pair_distance(struct1, struct2):
    """
    Calculate the base pair distance between two RNA structures.
    Args:
        struct1 (str): First dot-bracket notation.
        struct2 (str): Second dot-bracket notation.
    Returns:
        int: Base pair distance.
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

def two_step_folding(seq, constraints):
    """Perform two-step folding: unconstrained folding followed by pseudoknot application.

    The two-step approach first generates a secondary structure without considering pseudoknot constraints.
    In the second step, pseudoknot constraints are explicitly applied to the structure, potentially overriding
    existing base pairs. This differs from the single-step approach, where constraints are integrated directly
    into the folding algorithm, making it less flexible but potentially more biologically stable.
    """
    # Step 1: Fold without constraints
    matrix = fill_matrix(seq)
    unconstrained_structure = traceback(matrix, seq)

    # Debugging: Print the unconstrained structure
    print(f"Unconstrained structure: {unconstrained_structure}")

    # Step 2: Apply pseudoknot constraints
    constrained_structure = apply_pseudoknot_constraints(unconstrained_structure, constraints)

    # Debugging: Print the constrained structure
    print(f"Constrained structure: {constrained_structure}")

    return unconstrained_structure, constrained_structure

def parse_sequences_and_constraints(file_path):
    """
    Parse RNA sequences and their pseudoknot constraints from a sequences file.
    Args:
        file_path (str): Path to the sequences file.
    Returns:
        List of tuples: Each tuple contains (sequence, constraints).
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

def process_sequences_with_two_step_folding(file_path):
    """Perform two-step folding on all sequences in the file."""
    parsed_data = parse_sequences_and_constraints(file_path)
    results = []

    for seq, constraints in parsed_data:
        print(f"Processing sequence: {seq}")  # Debugging: Print sequence being processed
        print(f"Constraints: {constraints}")  # Debugging: Print parsed constraints

        unconstrained, constrained = two_step_folding(seq, constraints)

        # Calculate Hamming and Base Pair distances
        hamming_dist = calculate_hamming_distance(unconstrained, constrained)
        base_pair_dist = calculate_base_pair_distance(unconstrained, constrained)

        results.append({
            "sequence": seq,
            "unconstrained": unconstrained,
            "constrained": constrained,
            "hamming": hamming_dist,
            "base_pair": base_pair_dist
        })

    # Save results to a CSV file for further analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_two_step.csv", index=False)
    print("Results saved to results_two_step.csv")

    return results

def summarize_statistics(df):
    """Generate summary statistics for Hamming and Base Pair distances."""
    stats = {
        "Hamming Distance": {
            "Mean": df["hamming"].mean(),
            "Median": df["hamming"].median(),
            "Std Dev": df["hamming"].std(),
            "Min": df["hamming"].min(),
            "Max": df["hamming"].max(),
        },
        "Base Pair Distance": {
            "Mean": df["base_pair"].mean(),
            "Median": df["base_pair"].median(),
            "Std Dev": df["base_pair"].std(),
            "Min": df["base_pair"].min(),
            "Max": df["base_pair"].max(),
        },
    }
    return stats

def plot_distributions(single_df, two_step_df):
    """Plot distributions of Hamming and Base Pair distances for both single-step and two-step approaches."""
    plt.figure(figsize=(8, 8))

    # Hamming Distance Comparison
    plt.subplot(2, 1, 1)
    plt.hist(single_df["hamming"], bins=20, color="skyblue", alpha=0.6, edgecolor="black", label="Single-Step")
    plt.hist(two_step_df["hamming"], bins=20, color="orange", alpha=0.6, edgecolor="black", label="Two-Step")
    plt.title("Hamming Distance Distribution")
    plt.xlabel("Hamming Distance")
    plt.ylabel("Frequency")
    plt.legend()

    # Base Pair Distance Comparison
    plt.subplot(2, 1, 2)
    plt.hist(single_df["base_pair"], bins=20, color="salmon", alpha=0.6, edgecolor="black", label="Single-Step")
    plt.hist(two_step_df["base_pair"], bins=20, color="green", alpha=0.6, edgecolor="black", label="Two-Step")
    plt.title("Base Pair Distance Distribution")
    plt.xlabel("Base Pair Distance")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # File path to the sequences file
    sequences_file_path = "./wbn215_sequences.txt"

    # Perform two-step folding and save results
    process_sequences_with_two_step_folding(sequences_file_path)

    # Load the single-step and two-step results for analysis
    single_results_file_path = "rna_results.csv"
    two_step_results_file_path = "results_two_step.csv"
    single_df = pd.read_csv(single_results_file_path)
    two_step_df = pd.read_csv(two_step_results_file_path)

    # Generate and print summary statistics
    print("Single-Step Results Summary:")
    single_stats = summarize_statistics(single_df)
    for metric, values in single_stats.items():
        print(f"\n{metric} (Single-Step)")
        for stat, value in values.items():
            print(f"  {stat}: {value:.2f}")

    print("\nTwo-Step Results Summary:")
    two_step_stats = summarize_statistics(two_step_df)
    for metric, values in two_step_stats.items():
        print(f"\n{metric} (Two-Step)")
        for stat, value in values.items():
            print(f"  {stat}: {value:.2f}")

    # Plot distributions
    plot_distributions(single_df, two_step_df)

if __name__ == "__main__":
    main()
