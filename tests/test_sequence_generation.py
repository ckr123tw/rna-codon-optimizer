"""
Unit tests for sequence generation components.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sequence_generation import (
    CODON_TABLE,
    get_random_codon,
    translate_amino_acid_to_cds,
    validate_rna_sequence,
    reverse_translate_codon,
    reverse_translate_cds,
    RNASequenceGenerator,
    create_template_sequence
)


class TestCodonTable(unittest.TestCase):
    """Test codon table functions."""
    
    def test_codon_table_completeness(self):
        """Test that all amino acids are in the codon table."""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY*'
        for aa in amino_acids:
            self.assertIn(aa, CODON_TABLE)
            self.assertGreater(len(CODON_TABLE[aa]), 0)
    
    def test_get_random_codon(self):
        """Test random codon selection."""
        codon = get_random_codon('M')
        self.assertEqual(codon, 'AUG')  # Only one codon for Met
        
        # Test with multiple codons
        codon = get_random_codon('A')
        self.assertIn(codon, CODON_TABLE['A'])
    
    def test_validate_rna_sequence(self):
        """Test RNA sequence validation."""
        self.assertTrue(validate_rna_sequence('AUGCUA'))
        self.assertFalse(validate_rna_sequence('ATGCTA'))  # DNA
        self.assertFalse(validate_rna_sequence('AUGCXA'))  # Invalid char
    
    def test_reverse_translate_codon(self):
        """Test codon to amino acid translation."""
        self.assertEqual(reverse_translate_codon('AUG'), 'M')
        self.assertEqual(reverse_translate_codon('UGG'), 'W')
        self.assertEqual(reverse_translate_codon('UAA'), '*')
    
    def test_translate_amino_acid_to_cds(self):
        """Test amino acid to CDS translation."""
        aa_seq = "MYPFIR"
        cds = translate_amino_acid_to_cds(aa_seq, use_weights=False, include_stop=False)
        
        # Should be 3 * 6 = 18 nucleotides
        self.assertEqual(len(cds), 18)
        self.assertTrue(validate_rna_sequence(cds))
        
        # Should start with AUG (Met)
        self.assertTrue(cds.startswith('AUG'))
        
        # Reverse translation should match
        recovered = reverse_translate_cds(cds)
        self.assertEqual(recovered, aa_seq)


class TestRNASequenceGenerator(unittest.TestCase):
    """Test RNA sequence generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = RNASequenceGenerator(use_codon_weights=True)
        self.test_aa = "MYPFIRTARM"
        self.test_utr5 = "AUGCUGACUGACUAGCU"
        self.test_utr3 = "UGACUGACUAGCUAGCU"
    
    def test_generate_sequence(self):
        """Test complete sequence generation."""
        result = self.generator.generate_sequence(
            utr5=self.test_utr5,
            utr3=self.test_utr3,
            amino_acid_sequence=self.test_aa
        )
        
        # Check structure
        self.assertIn('full_sequence', result)
        self.assertIn('utr5', result)
        self.assertIn('cds', result)
        self.assertIn('utr3', result)
        
        # Check lengths
        expected_length = len(self.test_utr5) + (len(self.test_aa) * 3) + 3 + len(self.test_utr3)  # +3 for stop
        self.assertEqual(result['length'], expected_length)
        
        # Check assembly
        self.assertTrue(result['full_sequence'].startswith(self.test_utr5))
        self.assertTrue(result['full_sequence'].endswith(self.test_utr3))
    
    def test_validate_sequence(self):
        """Test sequence validation."""
        result = self.generator.generate_sequence(
            utr5=self.test_utr5,
            utr3=self.test_utr3,
            amino_acid_sequence=self.test_aa
        )
        
        self.assertTrue(self.generator.validate_sequence(result))
    
    def test_generate_multiple_variants(self):
        """Test generating multiple sequence variants."""
        n_variants = 5
        variants = self.generator.generate_multiple_variants(
            utr5=self.test_utr5,
            utr3=self.test_utr3,
            amino_acid_sequence=self.test_aa,
            n_variants=n_variants
        )
        
        self.assertEqual(len(variants), n_variants)
        
        # All variants should be valid
        for variant in variants:
            self.assertTrue(self.generator.validate_sequence(variant))
        
        # CDS sequences should potentially differ due to codon choice
        cds_sequences = [v['cds'] for v in variants]
        # Note: May not all be unique if amino acid sequence is short


class TestTemplateSequence(unittest.TestCase):
    """Test template sequence creation."""
    
    def test_create_template_sequence(self):
        """Test creating template sequences."""
        aa_seq = "MYPFIRTARM"
        template = create_template_sequence(aa_seq, utr5_length=30, utr3_length=50)
        
        self.assertIn('utr5', template)
        self.assertIn('utr3', template)
        self.assertIn('amino_acids', template)
        
        self.assertEqual(len(template['utr5']), 30)
        self.assertEqual(len(template['utr3']), 50)
        self.assertEqual(template['amino_acids'], aa_seq)
        
        # Should be valid RNA
        self.assertTrue(validate_rna_sequence(template['utr5']))
        self.assertTrue(validate_rna_sequence(template['utr3']))


if __name__ == '__main__':
    unittest.main()
