"""
RNA Sequence Generation Module

This module handles codon selection, sequence assembly, and RNA embedding.
"""

from .codon_table import (
    CODON_TABLE,
    HUMAN_CODON_WEIGHTS,
    get_random_codon,
    translate_amino_acid_to_cds,
    reverse_translate_codon,
    reverse_translate_cds,
    validate_rna_sequence
)

from .sequence_generator import RNASequenceGenerator, create_template_sequence

from .evo_embedder import EvoEmbedder

from .validation import (
    validate_cds_amino_acids,
    validate_full_rna_sequence,
    compute_validation_reward
)

__all__ = [
    'CODON_TABLE',
    'HUMAN_CODON_WEIGHTS',
    'get_random_codon',
    'translate_amino_acid_to_cds',
    'validate_rna_sequence',
    'reverse_translate_codon',
    'reverse_translate_cds',
    'RNASequenceGenerator',
    'create_template_sequence',
]
