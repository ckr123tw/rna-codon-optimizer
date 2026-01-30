"""
Codon table and utilities for translating amino acid sequences to RNA.
Uses the standard genetic code for mammalian systems.
"""

import random
from typing import Dict, List

# Standard genetic code: amino acid -> list of codons (RNA)
CODON_TABLE: Dict[str, List[str]] = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],  # Alanine
    'C': ['UGU', 'UGC'],  # Cysteine
    'D': ['GAU', 'GAC'],  # Aspartic acid
    'E': ['GAA', 'GAG'],  # Glutamic acid
    'F': ['UUU', 'UUC'],  # Phenylalanine
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],  # Glycine
    'H': ['CAU', 'CAC'],  # Histidine
    'I': ['AUU', 'AUC', 'AUA'],  # Isoleucine
    'K': ['AAA', 'AAG'],  # Lysine
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],  # Leucine
    'M': ['AUG'],  # Methionine (start codon)
    'N': ['AAU', 'AAC'],  # Asparagine
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],  # Proline
    'Q': ['CAA', 'CAG'],  # Glutamine
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],  # Arginine
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],  # Serine
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],  # Threonine
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],  # Valine
    'W': ['UGG'],  # Tryptophan
    'Y': ['UAU', 'UAC'],  # Tyrosine
    '*': ['UAA', 'UAG', 'UGA'],  # Stop codons
}

# Codon usage frequencies for human genes (based on codon usage database)
# Higher weights indicate more commonly used codons
HUMAN_CODON_WEIGHTS: Dict[str, List[float]] = {
    'A': [0.26, 0.40, 0.23, 0.11],  # GCU, GCC, GCA, GCG
    'C': [0.45, 0.55],  # UGU, UGC
    'D': [0.46, 0.54],  # GAU, GAC
    'E': [0.42, 0.58],  # GAA, GAG
    'F': [0.45, 0.55],  # UUU, UUC
    'G': [0.16, 0.34, 0.25, 0.25],  # GGU, GGC, GGA, GGG
    'H': [0.41, 0.59],  # CAU, CAC
    'I': [0.36, 0.48, 0.16],  # AUU, AUC, AUA
    'K': [0.42, 0.58],  # AAA, AAG
    'L': [0.07, 0.13, 0.13, 0.20, 0.07, 0.40],  # UUA, UUG, CUU, CUC, CUA, CUG
    'M': [1.0],  # AUG
    'N': [0.46, 0.54],  # AAU, AAC
    'P': [0.28, 0.33, 0.27, 0.11],  # CCU, CCC, CCA, CCG
    'Q': [0.25, 0.75],  # CAA, CAG
    'R': [0.08, 0.19, 0.11, 0.21, 0.20, 0.21],  # CGU, CGC, CGA, CGG, AGA, AGG
    'S': [0.15, 0.22, 0.12, 0.15, 0.15, 0.24],  # UCU, UCC, UCA, UCG, AGU, AGC
    'T': [0.24, 0.36, 0.28, 0.12],  # ACU, ACC, ACA, ACG
    'V': [0.18, 0.24, 0.11, 0.47],  # GUU, GUC, GUA, GUG
    'W': [1.0],  # UGG
    'Y': [0.43, 0.57],  # UAU, UAC
    '*': [0.33, 0.33, 0.34],  # UAA, UAG, UGA
}


def get_random_codon(amino_acid: str, use_weights: bool = False) -> str:
    """
    Get a random codon for the given amino acid.
    
    Args:
        amino_acid: Single letter amino acid code
        use_weights: If True, use human codon usage frequencies for selection
        
    Returns:
        RNA codon (3 nucleotides)
        
    Raises:
        ValueError: If amino acid is not recognized
    """
    if amino_acid not in CODON_TABLE:
        raise ValueError(f"Unknown amino acid: {amino_acid}")
    
    codons = CODON_TABLE[amino_acid]
    
    if use_weights and amino_acid in HUMAN_CODON_WEIGHTS:
        weights = HUMAN_CODON_WEIGHTS[amino_acid]
        return random.choices(codons, weights=weights, k=1)[0]
    else:
        return random.choice(codons)


def translate_amino_acid_to_cds(
    amino_acid_sequence: str,
    use_weights: bool = False,
    include_stop: bool = True
) -> str:
    """
    Translate an amino acid sequence to a CDS (coding sequence) by randomly
    selecting codons.
    
    Args:
        amino_acid_sequence: Amino acid sequence (single letter codes)
        use_weights: If True, use human codon usage frequencies
        include_stop: If True, add a stop codon at the end
        
    Returns:
        RNA CDS sequence
        
    Raises:
        ValueError: If the sequence contains invalid amino acids
    """
    # Ensure first amino acid is Methionine (start codon)
    if amino_acid_sequence and amino_acid_sequence[0] != 'M':
        raise ValueError("Amino acid sequence must start with Methionine (M)")
    
    cds = ""
    for aa in amino_acid_sequence:
        cds += get_random_codon(aa, use_weights)
    
    if include_stop:
        cds += get_random_codon('*', use_weights)
    
    return cds


def validate_rna_sequence(sequence: str) -> bool:
    """
    Validate that a sequence contains only valid RNA nucleotides.
    
    Args:
        sequence: RNA sequence to validate
        
    Returns:
        True if valid, False otherwise
    """
    valid_nucleotides = set('AUGC')
    return all(nuc in valid_nucleotides for nuc in sequence.upper())


def reverse_translate_codon(codon: str) -> str:
    """
    Get the amino acid encoded by a codon.
    
    Args:
        codon: RNA codon (3 nucleotides)
        
    Returns:
        Single letter amino acid code
        
    Raises:
        ValueError: If codon is invalid
    """
    codon = codon.upper()
    if len(codon) != 3:
        raise ValueError(f"Codon must be 3 nucleotides, got {len(codon)}")
    
    for aa, codons in CODON_TABLE.items():
        if codon in codons:
            return aa
    
    raise ValueError(f"Unknown codon: {codon}")


def reverse_translate_cds(cds: str) -> str:
    """
    Translate a CDS back to amino acid sequence.
    
    Args:
        cds: RNA coding sequence
        
    Returns:
        Amino acid sequence
        
    Raises:
        ValueError: If CDS length is not a multiple of 3
    """
    if len(cds) % 3 != 0:
        raise ValueError(f"CDS length must be a multiple of 3, got {len(cds)}")
    
    amino_acids = []
    for i in range(0, len(cds), 3):
        codon = cds[i:i+3]
        aa = reverse_translate_codon(codon)
        amino_acids.append(aa)
    
    return ''.join(amino_acids)


if __name__ == "__main__":
    # Example usage
    test_aa_sequence = "MYPFIRTARMFGA"
    
    print(f"Amino acid sequence: {test_aa_sequence}")
    print("\nRandom codon selection (uniform):")
    cds1 = translate_amino_acid_to_cds(test_aa_sequence, use_weights=False)
    print(f"CDS: {cds1}")
    print(f"Length: {len(cds1)} nucleotides")
    
    print("\nWeighted codon selection (human codon usage):")
    cds2 = translate_amino_acid_to_cds(test_aa_sequence, use_weights=True)
    print(f"CDS: {cds2}")
    
    print("\nReverse translation:")
    aa_back = reverse_translate_cds(cds1[:-3])  # Exclude stop codon
    print(f"Recovered AA: {aa_back}")
    print(f"Match: {aa_back == test_aa_sequence}")
