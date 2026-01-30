"""
RNA sequence generator that assembles 5'UTR, CDS, and 3'UTR into complete sequences.
"""

from typing import Optional
from .codon_table import (
    translate_amino_acid_to_cds,
    validate_rna_sequence,
    reverse_translate_cds
)


class RNASequenceGenerator:
    """Generate complete RNA sequences from UTRs and amino acid sequences."""
    
    def __init__(self, use_codon_weights: bool = True):
        """
        Initialize the RNA sequence generator.
        
        Args:
            use_codon_weights: Whether to use human codon usage frequencies
        """
        self.use_codon_weights = use_codon_weights
    
    def generate_sequence(
        self,
        utr5: str,
        utr3: str,
        amino_acid_sequence: str,
        include_stop: bool = True
    ) -> dict:
        """
        Generate a complete RNA sequence from 5'UTR, amino acids, and 3'UTR.
        
        Args:
            utr5: 5' untranslated region sequence
            utr3: 3' untranslated region sequence
            amino_acid_sequence: Amino acid sequence to encode
            include_stop: Whether to include stop codon in CDS
            
        Returns:
            Dictionary containing:
                - 'full_sequence': Complete RNA sequence
                - 'utr5': 5'UTR sequence
                - 'cds': Coding sequence
                - 'utr3': 3'UTR sequence
                - 'amino_acids': Input amino acid sequence
                - 'length': Total sequence length
                
        Raises:
            ValueError: If sequences are invalid
        """
        # Validate UTR sequences
        if not validate_rna_sequence(utr5):
            raise ValueError("5'UTR contains invalid nucleotides")
        if not validate_rna_sequence(utr3):
            raise ValueError("3'UTR contains invalid nucleotides")
        
        # Ensure amino acid sequence starts with Methionine
        if not amino_acid_sequence.startswith('M'):
            raise ValueError("Amino acid sequence must start with Methionine (M)")
        
        # Generate CDS from amino acid sequence
        cds = translate_amino_acid_to_cds(
            amino_acid_sequence,
            use_weights=self.use_codon_weights,
            include_stop=include_stop
        )
        
        # Assemble full sequence
        full_sequence = utr5 + cds + utr3
        
        return {
            'full_sequence': full_sequence,
            'utr5': utr5,
            'cds': cds,
            'utr3': utr3,
            'amino_acids': amino_acid_sequence,
            'length': len(full_sequence),
            'utr5_length': len(utr5),
            'cds_length': len(cds),
            'utr3_length': len(utr3)
        }
    
    def validate_sequence(self, sequence_dict: dict) -> bool:
        """
        Validate that a generated sequence is correct.
        
        Args:
            sequence_dict: Output from generate_sequence()
            
        Returns:
            True if valid, False otherwise
        """
        # Check that assembly is correct
        expected = sequence_dict['utr5'] + sequence_dict['cds'] + sequence_dict['utr3']
        if sequence_dict['full_sequence'] != expected:
            return False
        
        # Check that CDS translates back correctly
        cds = sequence_dict['cds']
        # Remove stop codon if present
        if cds.endswith(('UAA', 'UAG', 'UGA')):
            cds = cds[:-3]
        
        try:
            recovered_aa = reverse_translate_cds(cds)
            return recovered_aa == sequence_dict['amino_acids']
        except ValueError:
            return False
    
    def generate_multiple_variants(
        self,
        utr5: str,
        utr3: str,
        amino_acid_sequence: str,
        n_variants: int = 10,
        include_stop: bool = True
    ) -> list:
        """
        Generate multiple sequence variants with different codon choices.
        
        Args:
            utr5: 5' untranslated region sequence
            utr3: 3' untranslated region sequence
            amino_acid_sequence: Amino acid sequence to encode
            n_variants: Number of variants to generate
            include_stop: Whether to include stop codon in CDS
            
        Returns:
            List of sequence dictionaries
        """
        variants = []
        for _ in range(n_variants):
            variant = self.generate_sequence(
                utr5=utr5,
                utr3=utr3,
                amino_acid_sequence=amino_acid_sequence,
                include_stop=include_stop
            )
            variants.append(variant)
        
        return variants


def create_template_sequence(
    amino_acid_sequence: str,
    utr5_length: int = 50,
    utr3_length: int = 100
) -> dict:
    """
    Create a template sequence with random UTRs for testing.
    
    Args:
        amino_acid_sequence: Amino acid sequence to encode
        utr5_length: Desired length of 5'UTR
        utr3_length: Desired length of 3'UTR
        
    Returns:
        Dictionary with template UTRs
    """
    import random
    
    nucleotides = ['A', 'U', 'G', 'C']
    
    # Generate random UTRs
    utr5 = ''.join(random.choices(nucleotides, k=utr5_length))
    utr3 = ''.join(random.choices(nucleotides, k=utr3_length))
    
    return {
        'utr5': utr5,
        'utr3': utr3,
        'amino_acids': amino_acid_sequence
    }


if __name__ == "__main__":
    # Example usage
    generator = RNASequenceGenerator(use_codon_weights=True)
    
    # Test amino acid sequence
    aa_seq = "MYPFIRTARMFGA"
    
    # Create template with random UTRs
    template = create_template_sequence(aa_seq, utr5_length=30, utr3_length=50)
    
    print("=" * 60)
    print("RNA Sequence Generator Test")
    print("=" * 60)
    print(f"Amino acid sequence: {aa_seq}")
    print(f"5'UTR ({len(template['utr5'])} nt): {template['utr5']}")
    print(f"3'UTR ({len(template['utr3'])} nt): {template['utr3']}")
    print()
    
    # Generate a single sequence
    seq = generator.generate_sequence(
        utr5=template['utr5'],
        utr3=template['utr3'],
        amino_acid_sequence=aa_seq
    )
    
    print("Generated Sequence:")
    print(f"  Full length: {seq['length']} nt")
    print(f"  5'UTR: {seq['utr5_length']} nt")
    print(f"  CDS: {seq['cds_length']} nt")
    print(f"  3'UTR: {seq['utr3_length']} nt")
    print(f"  CDS: {seq['cds']}")
    print(f"  Valid: {generator.validate_sequence(seq)}")
    print()
    
    # Generate multiple variants
    print("Generating 5 variants with different codon usage:")
    variants = generator.generate_multiple_variants(
        utr5=template['utr5'],
        utr3=template['utr3'],
        amino_acid_sequence=aa_seq,
        n_variants=5
    )
    
    for i, variant in enumerate(variants, 1):
        print(f"  Variant {i} CDS: {variant['cds']}")
        # Check if CDS is unique
    unique_cds = len(set(v['cds'] for v in variants))
    print(f"  Unique CDS sequences: {unique_cds}/5")
