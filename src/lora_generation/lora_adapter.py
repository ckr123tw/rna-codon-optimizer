"""
LoRA (Low-Rank Adaptation) adapter for Evo model.
Enables efficient fine-tuning of the foundation model for conditional generation.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, List
import random


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
            lora_dropout: Dropout for LoRA layers
            target_modules: Which modules to apply LoRA to (default: attention)
            device: Device to use
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Initializing Evo model with LoRA: {model_name}")
        print(f"  LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"  Device: {self.device}")
        
        # Load base model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load as causal LM for generation
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32  # Use float32 for stability
            )
            
            # Configure LoRA
            if target_modules is None:
                # Target attention layers by default
                target_modules = ["q_proj", "v_proj"]  # Common attention module names
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.base_model, lora_config)
            self.model = self.model.to(self.device)
            
            # Print trainable parameters
            self.model.print_trainable_parameters()
            
            print("LoRA model initialized successfully!")
            
        except Exception as e:
            print(f"Error loading Evo model: {e}")
            print("Using mock model for demonstration")
            self.model = None
            self.tokenizer = None
    
    def generate_sequences(
        self,
        prompt: str,
        num_sequences: int = 5,
        max_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
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
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode sequences
        generated_sequences = []
        for output in outputs:
            seq = self.tokenizer.decode(output, skip_special_tokens=True)
            # Convert T back to U for RNA
            seq = seq.replace('T', 'U')
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
    target_efficiency: float,
    include_instructions: bool = True
) -> str:
    """
    Format a conditioning prompt for the LoRA model.
    
    The prompt includes:
        - 5'UTR (fixed)
        - Target translation efficiency
        - Amino acid sequence constraint
        - 3'UTR (fixed)
    
    Args:
        utr5: 5' UTR sequence
        utr3: 3' UTR sequence
        amino_acid_sequence: Amino acid sequence to encode
        target_efficiency: Desired translation efficiency
        include_instructions: Whether to include natural language instructions
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    if include_instructions:
        prompt += f"Generate an RNA sequence with translation efficiency {target_efficiency:.2f}.\n"
        prompt += f"Amino acid sequence: {amino_acid_sequence}\n"
    
    # Convert to DNA for Evo model
    utr5_dna = utr5.replace('U', 'T')
    utr3_dna = utr3.replace('U', 'T')
    
    prompt += f"5'UTR: {utr5_dna}\n"
    prompt += f"CDS encoding: {amino_acid_sequence}\n"
    prompt += f"Target TE: {target_efficiency:.2f}\n"
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
