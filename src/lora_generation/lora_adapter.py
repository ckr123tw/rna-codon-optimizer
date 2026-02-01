"""
LoRA (Low-Rank Adaptation) adapter for Evo model.
Enables efficient fine-tuning of the foundation model for conditional generation.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, List, Dict
import random

from ..sequence_generation.constrained_decoding import StatefulCodonConstraintLogitsProcessor

class EvoLoRAAdapter:
    """
    LoRA adapter for Evo model to enable efficient fine-tuning.
    """
    
    def __init__(
        self,
        model_name: str = "togethercomputer/evo-1-8k-base",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize LoRA adapter for Evo model.
        
        Args:
            model_name: Hugging Face model identifier
            lora_r: LoRA rank (lower = fewer parameters)
            lora_alpha: LoRA alpha parameter (scaling factor)
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of modules to apply LoRA to. If None, uses default for Evo.
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        self.lora_alpha = lora_alpha
        
        # Mock mode check
        if model_name == "mock":
            print("Mock model requested. Using mock LoRA adapter.")
            self.base_model = None
            self.model = None
            self.tokenizer = None
            return

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Load base model
        # Note: Evo is a large model, ensure sufficient memory
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None
            )
            if self.device != 'cuda':
                self.base_model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running in mock mode (no model loaded)")
            self.base_model = None
            self.model = None
            return

        # Configure LoRA
        if target_modules is None:
            # Default target modules for Evo (StripedHyena architecture)
            # Adjust based on specific model architecture inspection
            target_modules = ["q_proj", "v_proj", "out_proj"] 
            
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, peft_config)
        self.model.print_trainable_parameters()
        
    def generate_sequences(
        self,
        prompt: str,
        num_sequences: int = 5,
        max_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        amino_acid_constraint: Optional[str] = None
    ) -> List[str]:
        """
        Generate RNA sequences using the LoRA-adapted model.
        
        Args:
            prompt: Conditioning prompt (e.g., includes UTRs and amino acid info)
            num_sequences: Number of sequences to generate
            max_length: Maximum sequence length
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            amino_acid_constraint: Optional amino acid sequence to constrain generation
            
        Returns:
            List of generated sequences
        """
        if self.model is None:
            print("Warning: Using mock generation (model not loaded)")
            # Return mock sequences for testing
            nucleotides = ['A', 'U', 'G', 'C']
            return [''.join(random.choices(nucleotides, k=100)) for _ in range(num_sequences)]
        
        self.model.eval()
        
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        prompt_length = inputs['input_ids'].shape[1]
        
        # Setup logits processor for constraints if needed
        logits_processor = LogitsProcessorList()
        if amino_acid_constraint:
            constraint_processor = StatefulCodonConstraintLogitsProcessor(
                tokenizer=self.tokenizer,
                target_aa_sequence=amino_acid_constraint,
                prompt_length=prompt_length
            )
            logits_processor.append(constraint_processor)
        
        # Generate sequences
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=logits_processor
            )
        
        # Decode sequences
        generated_sequences = []
        for output in outputs:
            # Decode only the generated part
            generated_ids = output[prompt_length:]
            seq = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Convert T back to U for RNA
            seq = seq.replace('T', 'U').replace(' ', '')
            generated_sequences.append(seq)
        
        return generated_sequences
    
    def save_adapter(self, path: str):
        """Save only the LoRA adapter weights (much smaller than full model)."""
        if self.model is None:
            print("Warning: No model to save")
            return
        
        self.model.save_pretrained(path)
        print(f"LoRA adapter saved to {path}")
    
    def load_adapter(self, path: str):
        """Load LoRA adapter weights."""
        if self.model is None:
            print("Warning: No model to load into")
            return
        
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.base_model, path)
        self.model = self.model.to(self.device)
        print(f"LoRA adapter loaded from {path}")


def format_conditional_prompt(
    utr5: str,
    utr3: str,
    amino_acid_sequence: str,
    target_metrics: Optional[Dict[str, float]] = None,
    cell_line: Optional[str] = None,
    include_instructions: bool = True,
    # Backward compatibility
    target_efficiency: Optional[float] = None
) -> str:
    """
    Format a conditioning prompt for the LoRA model with multi-metric targets.
    
    The prompt includes:
        - Cell Line (optional)
        - 5'UTR (fixed)
        - Target metrics (translation efficiency, half-life, etc.)
        - Amino acid sequence constraint
        - 3'UTR (fixed)
    
    Args:
        utr5: 5' UTR sequence
        utr3: 3' UTR sequence
        amino_acid_sequence: Amino acid sequence to encode
        target_metrics: Dictionary of target metric values
            Example: {'translation_efficiency': 0.85, 'half_life': 12.0}
            Supported metrics:
                - 'translation_efficiency': Target TE value (0.0-1.0 typically)
                - 'half_life': Target mRNA half-life in hours
                - 'stability': Target stability score
                - Custom metrics as defined in your critic model
        cell_line: Target cell line name (e.g., 'HEK293')
        include_instructions: Whether to include natural language instructions
        target_efficiency: DEPRECATED - Use target_metrics instead.
            Kept for backward compatibility.
        
    Returns:
        Formatted prompt string
        
    Examples:
        # Multi-metric prompt
        >>> prompt = format_conditional_prompt(
        ...     utr5="AUGCAUGC",
        ...     utr3="GCUAGCUA",
        ...     amino_acid_sequence="MVKL",
        ...     target_metrics={'translation_efficiency': 0.85, 'half_life': 12.0}
        ... )
        
        # Single metric (backward compatible)
        >>> prompt = format_conditional_prompt(
        ...     utr5="AUGCAUGC",
        ...     utr3="GCUAGCUA", 
        ...     amino_acid_sequence="MVKL",
        ...     target_efficiency=0.85
        ... )
    """
    # Backward compatibility: convert target_efficiency to target_metrics
    if target_metrics is None:
        if target_efficiency is not None:
            target_metrics = {'translation_efficiency': target_efficiency}
        else:
            target_metrics = {'translation_efficiency': 0.5}  # Default
    
    # Metric display names for readable prompts
    METRIC_DISPLAY_NAMES = {
        'translation_efficiency': 'TE',
        'half_life': 'Half-Life (h)',
        'stability': 'Stability',
        'expression_level': 'Expression'
    }
    
    prompt = ""
    
    if include_instructions:
        # Build natural language instruction with all target metrics
        metric_descriptions = []
        for metric, value in target_metrics.items():
            display_name = METRIC_DISPLAY_NAMES.get(metric, metric.replace('_', ' ').title())
            if metric == 'translation_efficiency':
                metric_descriptions.append(f"translation efficiency {value:.2f}")
            elif metric == 'half_life':
                metric_descriptions.append(f"half-life of {value:.1f}h")
            else:
                metric_descriptions.append(f"{display_name.lower()} of {value:.2f}")
        
        prompt += f"Generate an RNA sequence with {', '.join(metric_descriptions)}"
        if cell_line:
            prompt += f" in {cell_line} cells"
        prompt += ".\n"
        prompt += f"Amino acid sequence: {amino_acid_sequence}\n"
    
    # Convert to DNA for Evo model
    utr5_dna = utr5.replace('U', 'T')
    utr3_dna = utr3.replace('U', 'T')
    
    if cell_line:
        prompt += f"Cell Line: {cell_line}\n"
    prompt += f"5'UTR: {utr5_dna}\n"
    prompt += f"CDS encoding: {amino_acid_sequence}\n"
    
    # Add all target metrics to prompt
    for metric, value in target_metrics.items():
        display_name = METRIC_DISPLAY_NAMES.get(metric, metric.replace('_', ' ').title())
        prompt += f"Target {display_name}: {value:.2f}\n"
    
    prompt += "CDS: "  # Model should complete this
    
    return prompt


if __name__ == "__main__":
    # Test LoRA adapter
    print("=" * 60)
    print("LoRA Adapter Test")
    print("=" * 60)
    
    # Initialize adapter
    lora_adapter = EvoLoRAAdapter(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    # Test prompt formatting
    test_prompt = format_conditional_prompt(
        utr5="AUGCUGACUGACUAGCUAGCU",
        utr3="UGACUGACUGACUAGCUAGCU",
        amino_acid_sequence="MYPFIRTARM",
        target_efficiency=3.5
    )
    
    print("\nTest prompt:")
    print(test_prompt)
    
    # Test generation
    print("\nGenerating sequences...")
    sequences = lora_adapter.generate_sequences(
        prompt=test_prompt,
        num_sequences=3,
        max_length=200,
        temperature=1.0
    )
    
    print(f"\nGenerated {len(sequences)} sequences:")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {seq[:80]}...")
