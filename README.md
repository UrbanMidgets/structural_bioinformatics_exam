# structural_bioinformatics_exam
Repository containing all code related to exam project in structural bioinformatics (UCPH). Course code: NBIA05014U.

# Structural Bioinformatics Exam Project

This repository contains all the code related to the Structural Bioinformatics exam project at the University of Copenhagen (UCPH), course code: NBIA05014U.

## Repository Structure

- **pdb_files/**: Directory containing Protein Data Bank (PDB) files used in the project.
  - `1iqz.pdb`: Example PDB file utilized for structural analysis.

- **helix_mutation.fasta**: FASTA file containing sequences related to alanine-helix mutations studied in the project.

- **helix_mutation_pro.fasta**: FASTA file with proline mutations in helix sequences.

- **protein.py**: Module containing functions and classes for protein structure analysis.
  - **Details**: Includes methods for parsing PDB files, analyzing secondary structures, and performing mutations.

- **requirements.txt**: List of Python dependencies required to run the project's scripts.
  - **Installation**: Install the dependencies using `pip install -r requirements.txt`.

- **results_two_step.csv**: CSV file containing results from a two-step analysis process.

- **rna.py**: Module dedicated to RNA structure prediction and analysis.
  - **Details**: Implements algorithms for RNA secondary structure prediction using dynamic programming approaches.

- **rna_results.csv**: CSV file containing results from RNA structure predictions.

- **rna_two_step.py**: Script for performing a two-step RNA analysis.
  - **Usage**: Execute this script to run the two-step RNA analysis pipeline.

- **wbn215_sequences.txt**: Text file containing RNA sequences and their corresponding dot-bracket notations used for structure prediction.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/UrbanMidgets/structural_bioinformatics_exam.git
   cd structural_bioinformatics_exam
