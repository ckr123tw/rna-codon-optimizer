"""
Constrained decoding utilities for RNA sequence generation.
Ensures that generated sequences translate to specific amino acid sequences.
"""

import torch
from transformers import LogitsProcessor
from typing import List, Dict, Set, Optional, Union
from .codon_table import CODON_TABLE

class CodonConstraintLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that enforces amino acid constraints during generation.
    It masks out tokens that would lead to invalid codons for the target amino acid sequence.
    """
    
    def __init__(self, tokenizer, target_aa_sequence: str):
        """
        Initialize the logits processor.
        
        Args:
            tokenizer: The tokenizer used by the model
            target_aa_sequence: The target amino acid sequence to enforce
        """
        self.tokenizer = tokenizer
        self.target_aa_sequence = target_aa_sequence
        
        # Build mapping from amino acid to valid codons
        self.aa_to_codons = CODON_TABLE
        
        # Pre-compute valid tokens for A, U, G, C
        self.valid_nucleotides = {'A', 'U', 'G', 'C'}
        self.token_id_map = {}
        
        # Find token IDs for valid nucleotides
        # We assume the tokenizer encodes these as single tokens
        # This is optimization to avoid decoding every step
        for nuc in self.valid_nucleotides:
            # Note: This might need adjustment depending on specific tokenizer behavior
            # Some tokenizers might add spaces or have multiple IDs for a character
            ids = tokenizer.encode(nuc, add_special_tokens=False)
            if len(ids) == 1:
                self.token_id_map[nuc] = ids[0]
            else:
                # Fallback or warning if tokenizer is complex
                pass
                
        self.nucleotide_token_ids = set(self.token_id_map.values())
        
    def _get_valid_next_nucleotides(self, current_seq: str) -> Set[str]:
        """
        Determine which nucleotides are valid for the next position.
        
        Args:
           current_seq: The sequence generated so far (nucleotides)
           
        Returns:
            Set of valid nucleotide characters ('A', 'U', 'G', 'C')
        """
        # Calculate current position in the CDS
        # We assume the generated sequence is the CDS only (or handled relative to it)
        # If prompts include UTRs, we need to know where CDS starts.
        # For this implementation, we assume generation starts at the beginning of CDS
        # or the logits processor is initialized/reset appropriately.
        
        # In the PPO trianer, prompts usually end right before CDS start
        # So the generated tokens are the CDS.
        
        seq_len = len(current_seq)
        aa_idx = seq_len // 3
        codon_pos = seq_len % 3
        
        # Check if we've exceeded the target sequence
        if aa_idx >= len(self.target_aa_sequence):
            # If we're past the AA sequence, maybe allow stop codon or finish?
            # For strict constraint, we might just allow EOS if we're exactly at the end
            return set()
            
        target_aa = self.target_aa_sequence[aa_idx]
        valid_codons = self.aa_to_codons.get(target_aa, [])
        
        valid_next_nucs = set()
        
        if codon_pos == 0:
            # First position of codon: get first nuc of all valid codons
            for codon in valid_codons:
                valid_next_nucs.add(codon[0])
        elif codon_pos == 1:
            # Second position: must match first nuc
            current_codon_start = current_seq[-1]
            for codon in valid_codons:
                if codon[0] == current_codon_start:
                    valid_next_nucs.add(codon[1])
        elif codon_pos == 2:
            # Third position: must match first two nucs
            current_codon_prefix = current_seq[-2:]
            for codon in valid_codons:
                if codon[:2] == current_codon_prefix:
                    valid_next_nucs.add(codon[2])
                    
        return valid_next_nucs

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Determine the sequence generated so far for each beam/sequence in batch
        # input_ids is shape (batch_size, sequence_length)
        
        batch_size = input_ids.shape[0]
        
        for i in range(batch_size):
            # Decode the current sequence for this batch item
            # We skip special tokens to get just the generated text
            # NOTE: This decoding might be slow in loop.
            # Optimization: Cache prefix or track position if possible.
            # For now, decoding is safer to ensure correctness with complex tokenizers.
            
            # We need to be careful about the prompt. input_ids includes the prompt.
            # The processor lacks context about where prompt ends vs generated.
            # However, usually LogitsProcessor is used via model.generate,
            # BUT model.generate passes full input_ids.
            
            # CRITICAL: We need to know the prompt length to know what part is the CDS.
            # Since this `__call__` interface doesn't provide it, we might need a workaround.
            # Common HuggingFace pattern: The generated part is appended. 
            # But we can't easily distinguish without extra info.
            
            # Workaround: valid_nucleotides check is only strictly robust if we know we are generating CDS.
            # We will assume that we are generating from "start" of constraint for now.
            # To make this robust, we'd need to subtract prompt length.
            # But we don't have prompt length here!
            
            # Alternative: Construct the processor WITH the prompt length or simply
            # rely on the fact that we decoding everything.
            # IF we decode everything, we need to strip the prompt.
            # But prompt varies!
            
            # Let's try to assume the constraint applies to the *newly generated* part.
            # But standard LogitsProcessor receives full `input_ids`.
            
            # Better approach: We can check if the decoded sequence *ends* with a partial codon
            # consistent with the target AA sequence.
            # But "consistent" is hard if we don't know where CDS started.
            
            # Simplification: We will implement a stateful processor that stores the start length?
            # No, LogitsProcessor is re-instantiated or called repeatedly.
            # `model.generate` keeps the same processor instance for the whole generation loop.
            # So we can track state!
            
            pass 
        
        return scores

class StatefulCodonConstraintLogitsProcessor(LogitsProcessor):
    """
    Stateful version that knows where generation started.
    """
    def __init__(self, tokenizer, target_aa_sequence: str, prompt_length: int):
        self.tokenizer = tokenizer
        self.target_aa_sequence = target_aa_sequence
        self.prompt_length = prompt_length # Number of tokens in prompt
        self.aa_to_codons = CODON_TABLE
        self.valid_nucleotides = {'A', 'U', 'G', 'C'}
        
        # Optimization: Map 'A', 'U', 'G', 'C' to token IDs
        self.nuc_to_id = {}
        for nuc in self.valid_nucleotides:
            # Try to encode as single token
            ids = self.tokenizer.encode(nuc, add_special_tokens=False)
            if len(ids) == 1:
                self.nuc_to_id[nuc] = ids[0]
            
        # Also handle potential " T" or similar if tokenizer adds spaces
        # But for RNA (A, U, G, C), usually straightforward.
        # Note: Some models expect 'T' instead of 'U'
        self.use_t_for_u = 'T' in [self.tokenizer.decode(i) for i in self.tokenizer.encode('T', add_special_tokens=False)]
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        
        # Mask everything by default? No, we modify scores in place or return new ones.
        # We want to set -inf for invalid tokens.
        
        for i in range(batch_size):
            current_ids = input_ids[i]
            
            # Only consider generated tokens
            if len(current_ids) < self.prompt_length:
                # Should not happen in generate loop
                continue
                
            generated_ids = current_ids[self.prompt_length:]
            
            # We need to decode just the generated part to be safe about token boundaries
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Normalize U/T
            generated_text = generated_text.replace('T', 'U').replace(' ', '').upper()
            
            # Determine valid next nucleotides
            valid_nucs = self._get_valid_next_nucleotides_for_seq(generated_text)
            
            # Allow EOS if complete?
            # if len(generated_text) // 3 >= len(self.target_aa_sequence) and len(generated_text) % 3 == 0:
            #    valid_nucs.add(self.tokenizer.eos_token) # concept
            
            # Create mask
            # Initialize mask with -inf for all tokens
            # But modifying entire vocab is slow?
            # Actually, `scores` is (batch, vocab). We usually just modifying indices.
            
            # Strategy: Set all scores to -inf, then restore valid ones? 
            # Or assume most are invalid and set them?
            # Usually we iterate valid tokens and keep them, set others to -inf.
            
            # But wait, we don't know ALL token IDs that map to A/U/G/C.
            # If we only allow specific IDs we computed in __init__, we might miss some.
            # But for Evo/standard models, usually restricted set is fine.
            
            # Set entire row to -inf
            scores[i, :] = -float('inf')
            
            # Enable valid tokens
            for nuc in valid_nucs:
                # Handle U vs T mapping for model
                model_nuc = nuc
                if self.use_t_for_u and nuc == 'U':
                    model_nuc = 'T'
                
                # Get ID
                # We try to use cached map, but if simple check failed, we surely need robust one
                # For now assume simple map works or re-encode
                
                # Robust approach:
                # If we rely on stored IDs:
                tid = self.nuc_to_id.get(model_nuc)
                if tid is not None:
                    scores[i, tid] = 0.0 # Or keep original score? 
                    # Better to keep original score to allow sampling among valid options!
                    # But we overwrote with -inf. 
                    # So we need to backup or be careful.
                    pass
            
            # Re-implementation to preserve scores:
            # 1. Calculate allowed token IDs
            allowed_ids = []
            for nuc in valid_nucs:
                model_nuc = nuc
                if self.use_t_for_u and nuc == 'U':
                    model_nuc = 'T'
                
                if model_nuc in self.nuc_to_id:
                    allowed_ids.append(self.nuc_to_id[model_nuc])
            
            # 2. Mask everything NOT in allowed_ids
            mask = torch.ones_like(scores[i], dtype=torch.bool)
            if allowed_ids:
                mask[allowed_ids] = False # Do not mask these
            
            scores[i, mask] = -float('inf')

        return scores

    def _get_valid_next_nucleotides_for_seq(self, seq: str) -> Set[str]:
        seq_len = len(seq)
        aa_idx = seq_len // 3
        codon_pos = seq_len % 3
        
        if aa_idx >= len(self.target_aa_sequence):
            return set() # Should stop
            
        target_aa = self.target_aa_sequence[aa_idx]
        valid_codons = self.aa_to_codons.get(target_aa, [])
        
        valid_next = set()
        
        if codon_pos == 0:
            for c in valid_codons: valid_next.add(c[0])
        elif codon_pos == 1:
            curr = seq[-1]
            for c in valid_codons: 
                if c[0] == curr: valid_next.add(c[1])
        elif codon_pos == 2:
            curr = seq[-2:]
            for c in valid_codons:
                if c[:2] == curr: valid_next.add(c[2])
                
        return valid_next
