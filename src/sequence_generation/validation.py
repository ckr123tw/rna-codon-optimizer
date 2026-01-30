"""
RNA sequence validation utilities including amino acid sequence verification.
"""

from typing import Optional, Tuple, Dict
from .codon_table import reverse_translate_cds, validate_rna_sequence


def validate_cds_amino_acids(
    cds_sequence: str,
    expected_amino_acids: str,
    allow_stop: bool = True
) -> Tuple[bool, str]:
    """
    Validate that a CDS sequence translates to the expected amino acid sequence.
    
    Args:
        cds_sequence: RNA CDS sequence to validate
        expected_amino_acids: Expected amino acid sequence
        allow_stop: Whether to allow stop codon at the end
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if valid RNA sequence
    if not validate_rna_sequence(cds_sequence):
        return False, "Invalid RNA sequence (contains non-AUGC characters)"
    
    # Check length is multiple of 3
    if len(cds_sequence) % 3 != 0:
        return False, f"CDS length ({len(cds_sequence)}) is not a multiple of 3"
    
    # Translate CDS to amino acids
    try:
        translated_aa = reverse_translate_cds(cds_sequence)
    except Exception as e:
        return False, f"Translation failed: {str(e)}"
    
    # Remove stop codon if present and allowed
    if allow_stop and translated_aa.endswith('*'):
        translated_aa = translated_aa[:-1]
    
    # Compare with expected sequence
    if translated_aa == expected_amino_acids:
        return True, "Valid"
    else:
        # Find mismatch position
        mismatch_positions = []
        min_len = min(len(translated_aa), len(expected_amino_acids))
        
        for i in range(min_len):
            if translated_aa[i] != expected_amino_acids[i]:
                mismatch_positions.append(i)
        
        if len(translated_aa) != len(expected_amino_acids):
            error_msg = (
                f"Length mismatch: translated={len(translated_aa)}, "
                f"expected={len(expected_amino_acids)}"
            )
        else:
            error_msg = (
                f"Amino acid mismatch at positions {mismatch_positions[:5]}"
                f" (showing first 5)"
            )
        
        error_msg += f"\nTranslated: {translated_aa[:50]}..."
        error_msg += f"\nExpected:   {expected_amino_acids[:50]}..."
        
        return False, error_msg


def validate_full_rna_sequence(
    full_sequence: str,
    utr5: str,
    utr3: str,
    expected_amino_acids: str,
    allow_stop: bool = True
) -> Dict[str, any]:
    """
    Validate a complete RNA sequence (5'UTR + CDS + 3'UTR).
    
    Args:
        full_sequence: Complete RNA sequence to validate
        utr5: Expected 5'UTR sequence
        utr3: Expected 3'UTR sequence
        expected_amino_acids: Expected amino acid sequence
        allow_stop: Whether to allow stop codon
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': []
    }
    
    # Check if valid RNA
    if not validate_rna_sequence(full_sequence):
        results['is_valid'] = False
        results['errors'].append("Invalid RNA sequence")
        return results
    
    # Check 5'UTR
    if not full_sequence.startswith(utr5):
        results['is_valid'] = False
        results['errors'].append(f"5'UTR mismatch")
    
    # Check 3'UTR
    if not full_sequence.endswith(utr3):
        results['is_valid'] = False
        results['errors'].append(f"3'UTR mismatch")
    
    # Extract CDS
    utr5_len = len(utr5)
    utr3_len = len(utr3)
    cds = full_sequence[utr5_len:-utr3_len if utr3_len > 0 else None]
    
    # Validate CDS
    cds_valid, cds_error = validate_cds_amino_acids(
        cds, expected_amino_acids, allow_stop
    )
    
    if not cds_valid:
        results['is_valid'] = False
        results['errors'].append(f"CDS validation failed: {cds_error}")
    
    # Add details
    results['utr5_length'] = len(utr5)
    results['cds_length'] = len(cds)
    results['utr3_length'] = len(utr3)
    results['total_length'] = len(full_sequence)
    
    return results


def compute_validation_reward(
    generated_sequence: str,
    utr5: str,
    utr3: str,
    expected_amino_acids: str,
    base_reward: float = 0.0
) -> float:
    """
    Compute reward bonus/penalty for sequence validation.
    
    Args:
        generated_sequence: Generated RNA sequence
        utr5: Expected 5'UTR
        utr3: Expected 3'UTR
        expected_amino_acids: Expected amino acid sequence
        base_reward: Base reward from metrics
        
    Returns:
        Adjusted reward value
    """
    validation = validate_full_rna_sequence(
        generated_sequence,
        utr5,
        utr3,
        expected_amino_acids
    )
    
    if validation['is_valid']:
        # Bonus for valid sequence
        return base_reward + 1.0
    else:
        # Large penalty for invalid sequence
        return base_reward - 10.0


if __name__ == "__main__":
    # Test validation
    print("=" * 60)
    print("Amino Acid Sequence Validation Test")
    print("=" * 60)
    
    # Test case 1: Valid sequence
    test_aa = "MYPFIRTARM"
    test_cds = "AUGUAUCCAUUCAUAAGAACAGCAAGAAUAUG"  # Valid CDS
    
    is_valid, msg = validate_cds_amino_acids(test_cds, test_aa, allow_stop=False)
    print(f"\nTest 1 - Valid CDS:")
    print(f"  CDS: {test_cds}")
    print(f"  Expected AA: {test_aa}")
    print(f"  Result: {is_valid}")
    print(f"  Message: {msg}")
    
    # Test case 2: Invalid sequence (length not multiple of 3)
    test_cds_invalid = "AUGUA"  # Only 5 nucleotides
    is_valid, msg = validate_cds_amino_acids(test_cds_invalid, test_aa)
    print(f"\nTest 2 - Invalid CDS (wrong length):")
    print(f"  CDS: {test_cds_invalid}")
    print(f"  Result: {is_valid}")
    print(f"  Message: {msg}")
    
    # Test case 3: Full sequence validation
    from .sequence_generator import RNASequenceGenerator
    
    generator = RNASequenceGenerator()
    result = generator.generate_sequence(
        utr5="AUGCUGACUGACU",
        utr3="UGACUGACUAGCU",
        amino_acid_sequence=test_aa
    )
    
    validation = validate_full_rna_sequence(
        result['full_sequence'],
        result['utr5'],
        result['utr3'],
        test_aa
    )
    
    print(f"\nTest 3 - Full Sequence Validation:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Errors: {validation['errors']}")
    print(f"  Lengths: 5'UTR={validation['utr5_length']}, "
          f"CDS={validation['cds_length']}, 3'UTR={validation['utr3_length']}")
