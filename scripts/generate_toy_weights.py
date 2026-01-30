"""
Generate toy example weights for testing the RNA Codon Optimization Pipeline.

This script creates small, random weights that can be used to verify the
pipeline runs correctly without needing to download large foundation models.

Usage:
    python scripts/generate_toy_weights.py

This will create:
    - models/toy_critic.pt - Toy critic model weights
    - models/toy_evo_embedder.pt - Toy embedder weights (mock)
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.critic.multi_metric_critic import MultiMetricCritic


def generate_toy_critic_weights(output_path: str = "models/toy_critic.pt"):
    """Generate toy critic model weights."""
    print("Generating toy critic weights...")
    
    # Create a small critic model with default parameters
    critic = MultiMetricCritic(
        input_dim=1024,  # Evo-1-8k embedding size
        shared_dims=[512, 256],
        metrics=['translation_efficiency', 'half_life'],
        head_dims=[128],
        dropout=0.1,
        num_cell_lines=10,
        cell_embedding_dim=32
    )
    
    # Save the randomly initialized weights
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state_dict': critic.state_dict(),
        'metrics': ['translation_efficiency', 'half_life'],
        'num_cell_lines': 10,
        'input_dim': 1024,
        'shared_dims': [512, 256]
    }, output_path)
    
    print(f"Saved toy critic weights to: {output_path}")
    print(f"  - Model size: {sum(p.numel() for p in critic.parameters()):,} parameters")
    return output_path


def generate_toy_embedder_config(output_path: str = "models/toy_embedder_config.json"):
    """Generate toy embedder configuration (not actual weights - those would be huge)."""
    import json
    
    print("Generating toy embedder configuration...")
    
    config = {
        "model_type": "toy",
        "embedding_dim": 1024,
        "description": "Toy embedder returns random 1024-dim embeddings for testing",
        "note": "For real training, replace with actual Evo-1-8k model from HuggingFace"
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved toy embedder config to: {output_path}")
    return output_path


def generate_toy_dataset(output_path: str = "data/toy_dataset.csv"):
    """Generate a minimal toy dataset if the full one doesn't exist."""
    import pandas as pd
    
    if os.path.exists(output_path):
        print(f"Toy dataset already exists: {output_path}")
        return output_path
    
    print("Generating minimal toy dataset...")
    
    # Generate 100 random sequences with TE and HalfLife
    np.random.seed(42)
    n_samples = 100
    
    # Create random RNA-like sequences (using T instead of U for simplicity)
    nucleotides = ['A', 'T', 'G', 'C']
    sequences = [''.join(np.random.choice(nucleotides, size=300)) for _ in range(n_samples)]
    
    cell_lines = ['HEK293', 'HeLa', 'HepG2', 'muscle_tissue', 'neurons']
    
    df = pd.DataFrame({
        'sequence': sequences,
        'TE': np.random.rand(n_samples) * 5,  # TE between 0-5
        'HalfLife': np.random.rand(n_samples) * 10,  # Half-life between 0-10
        'cell_line': np.random.choice(cell_lines, n_samples)
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Saved toy dataset to: {output_path}")
    print(f"  - {n_samples} samples")
    print(f"  - Cell lines/tissues: {cell_lines}")
    return output_path


def main():
    print("=" * 60)
    print("Generating Toy Weights for RNA Codon Optimization Pipeline")
    print("=" * 60)
    print()
    
    # Generate all toy artifacts
    critic_path = generate_toy_critic_weights()
    embedder_config = generate_toy_embedder_config()
    dataset_path = generate_toy_dataset()
    
    print()
    print("=" * 60)
    print("TOY ARTIFACTS GENERATED SUCCESSFULLY")
    print("=" * 60)
    print()
    print("To run a test training with toy data and weights:")
    print()
    print("  ./run_training.sh --data_path data/toy_dataset.csv --model_name mock")
    print()
    print("For real training, replace:")
    print("  1. data/toy_dataset.csv -> Your actual TE/Half-life dataset")
    print("  2. --model_name mock -> Remove flag to use real Evo-1-8k model")
    print("  3. models/toy_critic.pt -> Will be overwritten during training")
    print()


if __name__ == "__main__":
    main()
