"""Sequence generation module for RNA codon optimization."""

from .codon_table import (
    CODON_TABLE,
    HUMAN_CODON_WEIGHTS,
    get_random_codon,
    translate_amino_acid_to_cds,
    validate_rna_sequence,
    reverse_translate_codon,
    reverse_translate_cds,
)

from .sequence_generator import (
    RNASequenceGenerator,
    create_template_sequence,
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
