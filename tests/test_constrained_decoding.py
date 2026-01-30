import unittest
import torch
from transformers import AutoTokenizer
from src.sequence_generation.constrained_decoding import StatefulCodonConstraintLogitsProcessor
from src.sequence_generation.codon_table import CODON_TABLE

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.vocab = {
            'A': 1, 'U': 2, 'G': 3, 'C': 4,
            'T': 5, # Some models use T
            'M': 10, # Dummy
        }
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
    def encode(self, text, add_special_tokens=False):
        # deeply simplified mock
        return [self.vocab.get(c, 0) for c in text]
        
    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]
            
        return "".join([self.ids_to_tokens.get(i, "") for i in token_ids])

class TestConstrainedDecoding(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.processor = StatefulCodonConstraintLogitsProcessor(
            tokenizer=self.tokenizer,
            target_aa_sequence="M", # Just Methionine (AUG)
            prompt_length=0
        )
        
    def test_initial_valid_nucleotides(self):
        # Start of sequence (position 0) for 'M' -> AUG
        # Valid next: 'A'
        valid = self.processor._get_valid_next_nucleotides_for_seq("")
        self.assertEqual(valid, {'A'})
        
    def test_second_position_validity(self):
        # After 'A', for 'M' -> AUG
        # Valid next: 'U'
        valid = self.processor._get_valid_next_nucleotides_for_seq("A")
        self.assertEqual(valid, {'U'})
        
    def test_third_position_validity(self):
        # After 'AU', for 'M' -> AUG
        # Valid next: 'G'
        valid = self.processor._get_valid_next_nucleotides_for_seq("AU")
        self.assertEqual(valid, {'G'})
        
    def test_multiple_codons(self):
        # Test for Leucine (L) which has many codons: UUA, UUG, CUU, CUC, CUA, CUG
        processor = StatefulCodonConstraintLogitsProcessor(
            tokenizer=self.tokenizer,
            target_aa_sequence="L",
            prompt_length=0
        )
        
        # Pos 0: U or C
        valid = processor._get_valid_next_nucleotides_for_seq("")
        self.assertEqual(valid, {'U', 'C'})
        
        # If started with U: next is U (for UUA, UUG)
        valid = processor._get_valid_next_nucleotides_for_seq("U")
        self.assertEqual(valid, {'U'})
        
        # If started with C: next is U (for CU*)
        valid_c = processor._get_valid_next_nucleotides_for_seq("C")
        self.assertEqual(valid_c, {'U'})
        
        # If UU: next is A or G (UUA, UUG)
        valid_uu = processor._get_valid_next_nucleotides_for_seq("UU")
        self.assertEqual(valid_uu, {'A', 'G'})
        
        # If CU: next is U, C, A, G (CU*)
        valid_cu = processor._get_valid_next_nucleotides_for_seq("CU")
        self.assertEqual(valid_cu, {'U', 'C', 'A', 'G'})

    def test_logit_masking(self):
        # Test actual masking
        # prompt_length=2 (dummy prompt)
        processor = StatefulCodonConstraintLogitsProcessor(
            tokenizer=self.tokenizer,
            target_aa_sequence="M", 
            prompt_length=2 
        )
        
        # Input ids: [10, 10, 1] -> Prompt (10,10) + Generated 'A' (1)
        # So we are at position 1 (0-indexed) of generated sequence. 
        # Seq so far: "A". Target: "M" -> AUG. Next should be 'U' (id 2).
        
        input_ids = torch.tensor([[10, 10, 1]]) 
        scores = torch.zeros((1, 20)) # simple vocab size
        
        # Call processor
        new_scores = processor(input_ids, scores)
        
        # U (id 2) should NOT be masked (-inf)
        self.assertTrue(new_scores[0, 2] > -100)
        
        # A, G, C should be masked
        self.assertTrue(new_scores[0, 1] == -float('inf'))
        self.assertTrue(new_scores[0, 3] == -float('inf'))
        self.assertTrue(new_scores[0, 4] == -float('inf'))

if __name__ == '__main__':
    unittest.main()
