"""
Enhanced LoRA adapter with cell line/tissue conditioning and multi-metric prompting.
"""

from .lora_adapter import EvoLoRAAdapter
from typing import Optional, Dict


def format_multi_metric_prompt(
    utr5: str,
    utr3: str,
    amino_acid_sequence: str,
    targets: Dict[str, float],
    cell_line: Optional[str] = None,
    tissue: Optional[str] = None,
    include_instructions: bool = True
) -> str:
    """
    Format a conditioning prompt with multiple metrics and biological context.
    
    Args:
        utr5: 5' UTR sequence
        utr3: 3' UTR sequence
        amino_acid_sequence: Amino acid sequence to encode
        targets: Dictionary of target metrics, e.g.:
            {'translation_efficiency': 5.0, 'half_life': 4.5}
        cell_line: Cell line (e.g., 'HEK293', 'HeLa')
        tissue: Tissue type (e.g., 'kidney', 'liver')
        include_instructions: Whether to include natural language instructions
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    if include_instructions:
        # Build target description
        target_desc = []
        if 'translation_efficiency' in targets:
            target_desc.append(f"translation efficiency {targets['translation_efficiency']:.2f}")
        if 'half_life' in targets:
            target_desc.append(f"half-life {targets['half_life']:.2f} hours")
        
        prompt += f"Generate an RNA sequence with {' and '.join(target_desc)}.\n"
        
        # Add biological context
        if cell_line:
            prompt += f"Cell line: {cell_line}\n"
        if tissue:
            prompt += f"Tissue: {tissue}\n"
        
        prompt += f"Amino acid sequence: {amino_acid_sequence}\n\n"
    
    # Convert to DNA for Evo model
    utr5_dna = utr5.replace('U', 'T')
    utr3_dna = utr3.replace('U', 'T')
    
    # Structured format
    prompt += f"5'UTR: {utr5_dna}\n"
    
    if cell_line:
        prompt += f"CELL_LINE: {cell_line}\n"
    if tissue:
        prompt += f"TISSUE: {tissue}\n"
    
    prompt += f"CDS encoding: {amino_acid_sequence}\n"
    
    # Add target metrics
    for metric, value in targets.items():
        metric_key = metric.upper().replace('_', '_')
        prompt += f"TARGET_{metric_key}: {value:.2f}\n"
    
    prompt += f"3'UTR: {utr3_dna}\n"
    prompt += "CDS: "  # Model should complete this
    
    return prompt


# Example usage
if __name__ == "__main__":
    test_prompt = format_multi_metric_prompt(
        utr5="AUGCUGACUGACUAGCUAGCU",
        utr3="UGACUGACUGACUAGCUAGCU",
        amino_acid_sequence="MYPFIRTARM",
        targets={
            'translation_efficiency': 5.5,
            'half_life': 4.2
        },
        cell_line="HEK293",
        tissue="kidney"
    )
    
    print("=" * 60)
    print("Multi-Metric Prompt Example")
    print("=" * 60)
    print(test_prompt)
